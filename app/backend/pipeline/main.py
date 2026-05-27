"""
main.py
강의 영상 분석 통합 파이프라인

실행 흐름:
  [병렬] P1A extract_slides            — 슬라이드 프레임 추출
         P1B analyze_audio_quality     — 오디오 품질 분석
  [병렬] P2A textualize_slides         — 슬라이드 텍스트 + 강조 추출
         P2B transcribe_audio          — 전체 전사 (scene 매핑용)
  [병렬] P3A analyze_annotation        — 필기 강조 분석
         P3B process_audio             — 오디오 후처리
                     text_processor    — 2-pass 교정 + 침묵 구간 저장
                     emphasis          — 오디오 강조 감지
  [병렬] P4A classify_slides           — 슬라이드 역할 분류
         P4B save_scene_structure      — scene 구조 저장 (3B 결과 기반)
  [직렬] P5  fusion                    — 최종 통합
  [직렬] G1  graph_triples             — 그래프 Parquet 생성
  [직렬] G2  lance_index               — fused → Parquet + LanceDB (Gemini 임베딩, stem 필터)
  [직렬] G3  graphrag_index            — fused → GraphRAG parquet workspace

Usage:
    python main.py --input lecture.mp4
    python main.py --input lecture.mp4 --output output/ --slides output/slides/
    python main.py --input lecture.mp4 --skip-extract
    python main.py --input lecture.mp4 --debug --masks
    python main.py --input lecture.mp4 --force
"""

try:
    import lzma
except ImportError:
    from backports import lzma  # lzma가 없으면 backports.lzma를 사용

import json
import os
import shutil
import sys
import time
import argparse
import logging
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Optional

import librosa
from .utils import resolve_backend_root, resolve_pipeline_package_root

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


REPO_ROOT = resolve_backend_root()
DEFAULT_RECOMMENDER_METADATA_DIR = os.getenv(
    "GRAPHLEC_METADATA_DIR",
    "/app/metadata" if Path("/app/metadata").exists() else str(REPO_ROOT / "app" / "backend" / "metadata"),
)
DEFAULT_RECOMMENDER_DB_DIR = os.getenv(
    "RECOMMENDER_DB_DIR",
    "/lance/lancedb" if Path("/lance").exists() else str(REPO_ROOT / "data" / "lancedb"),
)
# 외부 라이브러리 노이즈 로그 억제
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("google").setLevel(logging.WARNING)
logging.getLogger("google.ai.generativelanguage").setLevel(logging.WARNING)
logging.getLogger("google.genai").setLevel(logging.WARNING)

# ──────────────────────────────────────────────────────────────
# 출력 헬퍼
# ──────────────────────────────────────────────────────────────

def _pipeline_root_dir() -> Path:
    env_root = os.getenv("PIPELINE_ROOT")
    if env_root:
        return Path(env_root).resolve()
    return Path(__file__).resolve().parents[3]


def _analyzer_run_module() -> str:
    package_name = (__package__ or "").strip()
    if package_name:
        return f"{package_name}.analyzer.run_all"
    return "app.backend.pipeline.analyzer.run_all"

def _banner(title: str):
    print("\n" + "═" * 70)
    print(f"  {title}")
    print("═" * 70)


def _done(label: str, elapsed: float):
    print(f"\n  ✓ {label}  ({elapsed:.1f}초)")
    print("─" * 70)


def _save_json(path: Path, data: Any):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _fmt_ts(sec: float) -> str:
    m, s = int(sec) // 60, int(sec) % 60
    return f"{m:02d}:{s:02d}"


def _is_done(path: Path, label: str, force: bool) -> bool:
    """출력 파일이 이미 존재하면 True 반환 (force=True면 항상 False)."""
    if not force and path.exists() and path.stat().st_size > 0:
        print(f"\n  ⏭  {label} — 출력 파일 존재, 스킵")
        print(f"     {path}")
        print("─" * 70)
        return True
    return False


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _format_emphasis_reason(ann: dict) -> dict:
    detail = ann.get("emphasis_detail")
    if isinstance(detail, dict):
        return detail
    audio_emphasis = ann.get("audio_emphasis")
    if isinstance(audio_emphasis, dict):
        return {
            **audio_emphasis,
            "methods": ann.get("emphasis_methods", []),
            "keywords": {
                "all_keywords": ann.get("emphasis_keywords", []),
                "by_method": ann.get("emphasis_keywords_by_method", {}),
            },
            "detection_count": ann.get("detection_count", 0),
        }
    return {
        "score": ann.get("emphasis_score"),
        "methods": ann.get("emphasis_methods", []),
        "keywords": ann.get("emphasis_keywords", []),
        "keywords_by_method": ann.get("emphasis_keywords_by_method", {}),
        "detection_count": ann.get("detection_count", 0),
    }


# ──────────────────────────────────────────────────────────────
# 전사 헬퍼
# ──────────────────────────────────────────────────────────────

def _transcribe_by_scene(
    video_path: str,
    duration: float,
    meta_path: Optional[str],
    slide_ranges: list[dict],
    output_dir: Path,
) -> dict:
    """
    전역 전사 후 scene 시간축에 매핑하기 위한 세그먼트를 생성한다.

    Returns:
        {
            "segments": [{"start","end","text","words","scene_index"?}, ...],
            "silences": [{"start","end","duration"}, ...]   # 영상 절대 시간
        }
    """
    from .transcriber import transcribe_video

    if not meta_path or not slide_ranges:
        print("  ℹ️ metadata 없음 → 전체 전사 방식 사용")
        return transcribe_video(video_path, duration, output_dir=output_dir)

    unique_scenes = len(slide_ranges)
    print(f"  ℹ️ scene {unique_scenes}개 기준 시간 매핑용 전체 전사 사용")
    return transcribe_video(video_path, duration, output_dir=output_dir)


# ──────────────────────────────────────────────────────────────
# 파이프라인 스테이지
# ──────────────────────────────────────────────────────────────

def extract_slides(args, slides_dir: Path, output_dir: Path) -> dict:
    from .slide_extractor import (
        build_canonical_slide_annotations,
        build_scene_slide_map,
        extract_slides,
    )

    stem = Path(args.input).stem
    meta_path = output_dir / f"{stem}_metadata.json"
    scene_slide_map_path = output_dir / f"{stem}_scene_slide_map.json"
    canonical_slide_annotations_path = output_dir / f"{stem}_canonical_slide_annotations.json"

    if _is_done(meta_path, "P1A extract_slides — 슬라이드 추출", args.force):
        return {
            "meta_path": str(meta_path),
            "scene_slide_map_path": str(scene_slide_map_path),
            "canonical_slide_annotations_path": str(canonical_slide_annotations_path),
            "elapsed": 0.0,
        }

    _banner("P1A extract_slides — 슬라이드 추출  (slide_extractor)")
    t0 = time.time()
    metadata = extract_slides(
        input_path=args.input,
        output_dir=str(slides_dir),
        debug=args.debug,
        decode_backend=getattr(args, "slide_decode_backend", None),
        extract_workers=getattr(args, "slide_extract_workers", None),
    )
    elapsed = time.time() - t0

    _save_json(meta_path, metadata)
    _save_json(scene_slide_map_path, build_scene_slide_map(metadata))
    _save_json(canonical_slide_annotations_path, build_canonical_slide_annotations(metadata))

    scene_count = len({
        idx for idx in (
            m.get("scene_index") if m.get("scene_index") is not None else m.get("slide_index")
            for m in metadata
        )
        if idx is not None
    })
    _done(f"scene {scene_count}개, 프레임 {len(metadata)}개 추출", elapsed)
    return {
        "meta_path": str(meta_path),
        "scene_slide_map_path": str(scene_slide_map_path),
        "canonical_slide_annotations_path": str(canonical_slide_annotations_path),
        "elapsed": elapsed,
    }


def analyze_audio_quality(args, output_dir: Path) -> dict:
    """P1B analyze_audio_quality: 오디오 품질 분석 (slide_extractor와 병렬)."""
    from .audio_analyzer import extract_audio_from_video, analyze_audio_features, evaluate_audio_quality
    from .utils import get_video_duration

    stem = Path(args.input).stem
    audio_quality_path = output_dir / f"{stem}_audio_quality.json"

    if _is_done(audio_quality_path, "P1B analyze_audio_quality — 오디오 품질 분석", args.force):
        # duration은 파일에서 복원
        duration = 0.0
        try:
            with open(audio_quality_path) as f:
                q = json.load(f)
            duration = _safe_float(q.get("duration_sec"), 0.0)
        except Exception:
            pass
        if duration == 0.0:
            duration = get_video_duration(args.input)
        return {"duration": duration, "elapsed": 0.0}

    video_path = args.input

    _banner("P1B analyze_audio_quality — 오디오 품질 분석  (audio_analyzer)")
    t0 = time.time()
    duration = get_video_duration(video_path)
    audio_path = str(output_dir / "temp_full_audio.wav")
    extract_audio_from_video(video_path, audio_path)
    try:
        audio_features = analyze_audio_features(audio_path)
        audio_quality = evaluate_audio_quality(audio_features)
        _save_json(output_dir / f"{stem}_audio_features.json", audio_features)
        _save_json(audio_quality_path, audio_quality)
        print(f"  ✓ 품질: {audio_quality['overall_score']}/100 ({audio_quality['overall_grade']})")
    finally:
        Path(audio_path).unlink(missing_ok=True)
    elapsed = time.time() - t0
    _done("오디오 품질 분석", elapsed)
    return {"duration": duration, "elapsed": elapsed}


def textualize_slides(args, slides_dir: Path, output_dir: Path) -> dict:
    from .slide_textualizer import TextualizationPipeline, Config as TextConfig

    stem = Path(args.input).stem
    textualized_path = output_dir / f"{stem}_slide_textualized.json"

    if _is_done(textualized_path, "P2A textualize_slides — 슬라이드 텍스트화", args.force):
        return {"textualized_path": str(textualized_path), "elapsed": 0.0}

    _banner("P2A textualize_slides — 슬라이드 텍스트화  (slide_textualizer)")
    t0 = time.time()
    text_config = TextConfig(
        slides_dir=slides_dir,
        output_dir=output_dir,
        output_filename=textualized_path.name,
        max_retries=args.retries,
    )
    text_result = TextualizationPipeline(text_config).run()
    elapsed = time.time() - t0

    meta = text_result["metadata"]
    _done(f"scene {meta['total_scenes']}개 / slide {meta['total_slides']}개 텍스트화", elapsed)
    return {"textualized_path": str(textualized_path), "elapsed": elapsed}


def transcribe_audio(args, meta_path: str, duration: float, output_dir: Path) -> dict:
    from .segment_grouper import load_slide_ranges

    stem = Path(args.input).stem
    transcript_raw_path = output_dir / f"{stem}_transcript_raw.json"

    if _is_done(transcript_raw_path, "P2B transcribe_audio — 전체 전사", args.force):
        return {"transcript_raw_path": str(transcript_raw_path), "elapsed": 0.0}

    _banner("P2B transcribe_audio — 전체 전사  (Groq Whisper)")
    t0 = time.time()
    slide_ranges = load_slide_ranges(meta_path, duration) if meta_path and Path(meta_path).is_file() else []
    transcribe_result = _transcribe_by_scene(args.input, duration, meta_path, slide_ranges, output_dir)
    payload = {
        "video_path": args.input,
        "segment_count": len(transcribe_result.get("segments", [])),
        "silence_count": len(transcribe_result.get("silences", [])),
        "segments": transcribe_result.get("segments", []),
        "silences": transcribe_result.get("silences", []),
    }
    _save_json(transcript_raw_path, payload)
    elapsed = time.time() - t0
    _done(
        f"전체 전사 {payload['segment_count']}개 세그먼트, 무음 {payload['silence_count']}개",
        elapsed,
    )
    return {"transcript_raw_path": str(transcript_raw_path), "elapsed": elapsed}


def analyze_slide_annotations(args, slides_dir: Path, output_dir: Path) -> dict:
    from .annotation_analyzer import analyze_all

    stem = Path(args.input).stem
    annotation_path = output_dir / f"{stem}_annotation.json"

    if _is_done(annotation_path, "P3A analyze_annotation — 필기 강조 분석", args.force):
        return {"annotation_path": str(annotation_path), "elapsed": 0.0}

    _banner("P3A analyze_annotation — 필기 강조 분석  (annotation_analyzer)")
    t0 = time.time()
    annot_results = analyze_all(
        slides_dir=str(slides_dir),
        output_path=str(annotation_path),
        save_masks=args.masks,
        per_annot_mode=getattr(args, "per_annot_mode", False),
    )
    elapsed = time.time() - t0

    n_annots = sum(len(r.get("annotations", [])) for r in annot_results)
    _done(f"{len(annot_results)}개 분석, 총 {n_annots}개 강조 추출", elapsed)
    return {"annotation_path": str(annotation_path), "elapsed": elapsed}


def process_audio(
    args,
    meta_path: str,
    textualized_path: str,
    duration: float,
    output_dir: Path,
    transcript_raw_path: Optional[str] = None,
    on_contexts_ready: Optional[Callable[[dict], None]] = None,
) -> dict:
    from .text_processor import correct_segments_two_pass
    from .segment_grouper import (
        load_slide_ranges,
        group_segments_by_context,
        group_segments_by_scene_and_context,
        expand_group_annotations_to_segments,
    )
    from .audio_analyzer import extract_audio_from_video
    from .emphasis_audio import detect_emphasis_by_std
    from .emphasis_keyword import (
        detect_emphasis_by_keywords_weighted,
        detect_emphasis_by_topic_keyword_repetition,
        get_topic_keywords_filtered_v2,
        get_topic_keyword_count_map,
        get_topic_keyword_score_map,
        topic_keyword_count_items,
    )
    from .emphasis_combiner import combine_emphasis_simple

    stem = Path(args.input).stem
    segments_path = output_dir / f"{stem}_segments.json"
    silences_path = output_dir / f"{stem}_silences.json"
    emphasis_path = output_dir / f"{stem}_emphasis.json"
    by_scene_path = output_dir / f"{stem}_by_scene.json"

    def _by_scene_has_context_schema(path: Path) -> bool:
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            scenes = payload.get("scenes", [])
            if not scenes:
                return False
            has_contexts = False
            for scene in scenes:
                for ctx in scene.get("contexts", []) or []:
                    has_contexts = True
                    if "audio_emphasis" not in ctx:
                        return False
            return has_contexts
        except Exception:
            return False

    # 세그먼트만 있고 by_scene이 없으면 context-first verifier 입력을 복원할 수 없으므로 재실행
    seg_ok = (
        not args.force
        and segments_path.exists()
        and segments_path.stat().st_size > 0
    )
    silences_ok = silences_path.exists() and silences_path.stat().st_size > 0
    emphasis_ok = emphasis_path.exists() and emphasis_path.stat().st_size > 0
    by_scene_ok = (
        by_scene_path.exists()
        and by_scene_path.stat().st_size > 0
        and _by_scene_has_context_schema(by_scene_path)
    )
    if seg_ok and silences_ok and emphasis_ok and by_scene_ok:
        print(f"\n  ⏭  P3B process_audio — 오디오 파이프라인 출력 파일 존재, 스킵")
        print(f"     {segments_path}")
        print(f"     {by_scene_path}")
        print("─" * 70)
        # in-memory 데이터를 저장된 파일에서 복원
        slides_structure = None
        if by_scene_path.exists():
            try:
                with open(by_scene_path) as f:
                    slides_structure = json.load(f).get("scenes")
            except Exception:
                pass

        annotated_segments: list[dict] = []
        try:
            with open(segments_path) as f:
                annotated_segments = json.load(f).get("segments", [])
        except Exception:
            pass

        slide_ranges = load_slide_ranges(meta_path, duration) if meta_path and Path(meta_path).is_file() else []
        if on_contexts_ready and slides_structure:
            try:
                on_contexts_ready({
                    "segments_path": str(segments_path),
                    "slides_structure": slides_structure,
                    "slide_ranges": slide_ranges,
                    "duration": duration,
                })
            except Exception as exc:
                log.warning(f"analyzer 조기 시작 실패(기존 context 사용): {exc}")

        return {
            "segments_path": str(segments_path),
            "silences_path": str(silences_path),
            "emphasis_path": str(emphasis_path),
            "annotated_segments": annotated_segments,
            "annotated_groups": [],
            "scenes_structure": slides_structure,
            "slide_ranges": slide_ranges,
            "duration": duration,
        }
    elif not args.force and segments_path.exists():
        log.warning(
            "P3B process_audio 캐시가 불완전하여 재실행합니다 "
            f"(segments={segments_path.exists()}, silences={silences_path.exists()}, "
            f"emphasis={emphasis_path.exists()}, by_scene={by_scene_path.exists()})"
        )

    if seg_ok and not by_scene_ok:
        print(
            f"\n  ⚠️  {by_scene_path.name} 없음 — 세그먼트만 있는 불완전 상태입니다. "
            "P3B process_audio 전체를 다시 실행합니다."
        )
        print("─" * 70)

    video_path = args.input

    _banner("P3B process_audio — 오디오 파이프라인")

    # 슬라이드 텍스트화 데이터 로드
    textualized_data: dict = {"scenes": []}
    if textualized_path and Path(textualized_path).is_file():
        with open(textualized_path, "r", encoding="utf-8") as f:
            textualized_data = json.load(f)

    # metadata 로드 (슬라이드 occurrence 정보)
    metadata: list[dict] = []
    if meta_path and Path(meta_path).is_file():
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    slide_ranges = load_slide_ranges(meta_path, duration) if meta_path and Path(meta_path).is_file() else []
    if transcript_raw_path and Path(transcript_raw_path).is_file():
        print("  [3B-1] 사전 전사 로드...")
        t0 = time.time()
        with open(transcript_raw_path, "r", encoding="utf-8") as f:
            transcribe_payload = json.load(f)
        segments_raw = transcribe_payload.get("segments", [])
        detected_silences = transcribe_payload.get("silences", [])
        print(f"    ✓ {len(segments_raw)}개 세그먼트, 무음 {len(detected_silences)}개 로드  ({time.time()-t0:.1f}초)")
    else:
        print("  [3B-1] 전체 전사 폴백 실행...")
        t0 = time.time()
        transcribe_result = _transcribe_by_scene(video_path, duration, meta_path, slide_ranges, output_dir)
        segments_raw = transcribe_result.get("segments", [])
        detected_silences = transcribe_result.get("silences", [])
        print(f"    ✓ {len(segments_raw)}개 세그먼트, 무음 {len(detected_silences)}개  ({time.time()-t0:.1f}초)")

    # [3B-2] 2-pass 텍스트 교정 (text_processor 내부 엔진)
    print("  [3B-2] 텍스트 교정 (2-pass)...")
    t0 = time.time()
    textualized_dir = Path(textualized_path).parent if textualized_path else output_dir
    segments = correct_segments_two_pass(
        segments=segments_raw,
        metadata=metadata,
        textualized_data=textualized_data,
        textualized_dir=textualized_dir,
    )
    segments_clean = [{k: v for k, v in s.items() if k != "words"} for s in segments]
    _save_json(segments_path, {
        "video_path": video_path,
        "segment_count": len(segments_clean),
        "segments": segments_clean,
    })
    print(f"    ✓ 교정 완료  ({time.time()-t0:.1f}초)")
    # [3B-3] 침묵 구간 저장 (transcriber가 ffmpeg silencedetect로 사전 감지)
    print("  [3B-3] 침묵 구간 저장...")
    MIN_SILENCE_SEC = 0.6
    silences: list[dict] = [
        {
            "index": i,
            "start": float(s["start"]),
            "end": float(s["end"]),
            "duration": float(s.get("duration", s["end"] - s["start"])),
        }
        for i, s in enumerate(detected_silences)
    ]
    _save_json(silences_path, {
        "video_path": video_path,
        "total_duration_sec": duration,
        "segment_count": len(segments_clean),
        "min_silence_sec": MIN_SILENCE_SEC,
        "silence_count": len(silences),
        "total_silence_duration_sec": sum(s["duration"] for s in silences),
        "silences": silences,
    })
    print(f"    ✓ 침묵 {len(silences)}개")

    # [3B-4] 오디오 강조 감지
    print("  [3B-4] 오디오 강조 감지...")
    t0 = time.time()
    audio_path_temp = str(output_dir / "temp_analysis_audio.wav")
    extract_audio_from_video(video_path, audio_path_temp)
    annotated_segments: list[dict] = []
    annotated_groups: list[dict] = []
    scenes_structure = None
    emphasis_sections: list[dict] = []
    topic_kw_set: set[str] = set()
    topic_keyword_counts: dict[str, int] = {}
    topic_keyword_scores: dict[str, int] = {}
    try:
        y, sr = librosa.load(audio_path_temp, sr=16000)
        if slide_ranges:
            groups, scenes_structure = group_segments_by_scene_and_context(
                segments_clean, slide_ranges, duration, use_pause_sentence=False, use_llm_merge=True
            )
        else:
            groups = group_segments_by_context(segments_clean)
            scenes_structure = None

        if on_contexts_ready and scenes_structure:
            try:
                on_contexts_ready({
                    "segments_path": str(segments_path),
                    "slides_structure": scenes_structure,
                    "slide_ranges": slide_ranges,
                    "duration": duration,
                })
            except Exception as exc:
                log.warning(f"analyzer 조기 시작 실패(context 사용): {exc}")

        topic_kw_set = get_topic_keywords_filtered_v2(
            groups, min_freq=5, max_keywords=20, max_segment_ratio=1.0,
            min_keyword_len=2, candidate_pool_size=80, use_llm_filter=True,
        )
        topic_keyword_counts = get_topic_keyword_count_map(
            groups,
            min_freq=5,
            max_keywords=20,
            max_segment_ratio=1.0,
            min_keyword_len=2,
            candidate_pool_size=80,
            use_llm_filter=False,
            _topic_keywords_override=topic_kw_set,
        )
        topic_keyword_scores = get_topic_keyword_score_map(topic_keyword_counts)
        audio_emphasis = detect_emphasis_by_std(y, sr, groups)
        keyword_emphasis = detect_emphasis_by_keywords_weighted(groups)
        topic_emphasis = detect_emphasis_by_topic_keyword_repetition(
            groups, window=2, min_keyword_len=2, max_segment_ratio=1.0,
            min_freq=5, max_keywords=20, use_llm_filter=False, min_keyword_count=1,
            _topic_keywords_override=topic_kw_set,
            _topic_keyword_count_map=topic_keyword_counts,
            _topic_keyword_score_map=topic_keyword_scores,
        )
        annotated_groups, emphasis_sections = combine_emphasis_simple(
            audio_emphasis, keyword_emphasis + topic_emphasis, groups,
        )
        annotated_segments = expand_group_annotations_to_segments(
            annotated_groups, segments_clean, groups
        )
        _save_json(emphasis_path, {
            "method": "std_topic_v2",
            "description": "표준편차 + 가중치 키워드 + 주제 키워드 반복",
            "keyword_report": {
                "description": "강조 감지에 사용된 주제 키워드",
                "topic_keywords": {
                    "keywords": sorted(topic_kw_set),
                    "count": len(topic_kw_set),
                },
                "audio_topic_keywords": topic_keyword_count_items(topic_keyword_counts),
            },
            "statistics": {
                "total_count": len(emphasis_sections),
                "ratio": round(len(emphasis_sections) / len(groups), 3) if groups else 0,
                "duration": round(sum(s["end"] - s["start"] for s in emphasis_sections), 2),
            },
            "topic_keywords": sorted(topic_kw_set),
            "emphasis_sections": emphasis_sections,
        })
        print(f"    ✓ 강조 {len(emphasis_sections)}개  ({time.time()-t0:.1f}초)")
    finally:
        Path(audio_path_temp).unlink(missing_ok=True)

    _done("오디오 파이프라인", 0.0)
    return {
        "segments_path": str(segments_path),
        "silences_path": str(silences_path),
        "emphasis_path": str(emphasis_path),
        "annotated_segments": annotated_segments,
        "annotated_groups": annotated_groups,
        "scenes_structure": scenes_structure,
        "slide_ranges": slide_ranges,
        "duration": duration,
    }


def classify_slides(
    args, textualized_path: str, meta_path: str, silences_path: str, output_dir: Path
) -> dict:
    from .slide_classifier import classify_slides

    stem = Path(args.input).stem
    classified_path = output_dir / f"{stem}_slide_classified.json"

    if _is_done(classified_path, "P4A classify_slides — 슬라이드 분류", args.force):
        return {"classified_path": str(classified_path), "elapsed": 0.0}

    _banner("P4A classify_slides — 슬라이드 분류  (slide_classifier)")
    t0 = time.time()
    classified = classify_slides(
        textualized_path=textualized_path,
        metadata_path=meta_path,
        silences_path=silences_path,
        output_path=str(classified_path),
    )
    elapsed = time.time() - t0
    _done(f"슬라이드 {len(classified)}개 분류", elapsed)
    return {"classified_path": str(classified_path), "elapsed": elapsed}


def save_scene_structure(args, audio_result: dict, output_dir: Path) -> dict:
    from .segment_grouper import group_segments_by_scene_and_context

    stem = Path(args.input).stem
    by_scene_path = output_dir / f"{stem}_by_scene.json"

    scenes_structure = audio_result.get("scenes_structure")
    annotated_segments = audio_result.get("annotated_segments", [])
    annotated_groups = audio_result.get("annotated_groups", [])
    slide_ranges = audio_result.get("slide_ranges", [])
    duration = audio_result.get("duration", 0.0)

    def _has_emphasis_schema(path: Path) -> bool:
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            for scene in payload.get("scenes", []):
                for ctx in scene.get("contexts", []):
                    if "audio_emphasis" not in ctx:
                        return False
            return True
        except Exception:
            return False

    def _to_public_scene_entry(scene: dict) -> dict:
        clean = json.loads(json.dumps(scene))
        if clean.get("scene_id") is None and clean.get("scene_index") is not None:
            clean["scene_id"] = f"scene/{int(clean['scene_index']):04d}"
        return clean

    def _scene_payload(scenes: list[dict]) -> dict:
        public_scenes = [_to_public_scene_entry(scene) for scene in scenes]
        slide_numbers = {
            scene.get("slide_number")
            for scene in public_scenes
            if scene.get("slide_number") is not None
        }
        return {
            "unit": "scene",
            "scene_count": len(public_scenes),
            "slide_count": len(slide_numbers),
            "scenes": public_scenes,
        }

    if not slide_ranges:
        if by_scene_path.exists() and by_scene_path.stat().st_size > 0 and _has_emphasis_schema(by_scene_path):
            return {
                "by_scene_path": str(by_scene_path),
                "elapsed": 0.0,
            }
        raise RuntimeError("P4B save_scene_structure — scene 구조 저장 실패: slide_ranges가 비어 있습니다.")

    if not scenes_structure and annotated_segments:
        log.warning("P4B save_scene_structure — scene 구조가 비어 있어 annotated_segments 기반으로 재구성합니다.")
        try:
            _, scenes_structure = group_segments_by_scene_and_context(
                annotated_segments,
                slide_ranges,
                duration,
                use_pause_sentence=False,
                use_llm_merge=False,
            )
        except Exception as exc:
            raise RuntimeError(f"P4B save_scene_structure — scene 구조 재구성 실패: {exc}") from exc

    if not scenes_structure:
        raise RuntimeError(
            "P4B save_scene_structure — scene 구조 저장 실패: scenes_structure가 비어 있습니다. "
            "P3B process_audio — 오디오 후처리 결과를 확인해주세요."
        )

    if _is_done(by_scene_path, "P4B save_scene_structure — scene 구조 저장", args.force):
        return {
            "by_scene_path": str(by_scene_path),
            "elapsed": 0.0,
        }

    _banner("P4B save_scene_structure — scene 구조 저장")
    t0 = time.time()

    scenes_with_emphasis = json.loads(json.dumps(scenes_structure))
    annot_by_start = {g.get("start"): g for g in annotated_groups}
    empty_audio_emphasis = {
        "audio": {
            "std_based_score": 0.0,
            "audio_signal_score": 0,
            "volume_score": 0.0,
            "pitch_score": 0.0,
            "volume_metric": 0.0,
            "pitch_metric": 0.0,
            "volume_rank": None,
            "pitch_rank": None,
            "volume_ratio": 0.0,
            "pitch_variation": 0.0,
        },
        "importance_keywords": {
            "score": 0,
            "category_scores": {"exam": 0, "strong": 0, "summary": 0},
            "matched_categories": [],
            "matched_keywords": {"exam": [], "strong": [], "summary": []},
        },
        "topic": {
            "keywords": [],
            "keyword_scores": {},
            "topic_keyword_score": 0,
            "audio_topic_total_count_sum": 0,
        },
    }
    for scene in scenes_with_emphasis:
        new_contexts = []
        for ctx in scene.get("contexts", []):
            ann = annot_by_start.get(ctx.get("start"))
            ordered_ctx = {
                "context_index": ctx.get("context_index"),
                "start": ctx.get("start"),
                "end": ctx.get("end"),
                "text": ctx.get("text"),
            }
            audio_emphasis = ann.get("audio_emphasis") if ann else None
            ordered_ctx["audio_emphasis"] = audio_emphasis or empty_audio_emphasis

            if ann:
                ordered_ctx["emphasis_methods"] = ann.get("emphasis_methods", [])
                ordered_ctx["detection_count"] = int(ann.get("detection_count", 0) or 0)
                ordered_ctx["emphasis_detail"] = _format_emphasis_reason(ann)
            else:
                ordered_ctx["emphasis_methods"] = []
                ordered_ctx["detection_count"] = 0
                ordered_ctx["emphasis_detail"] = {}
            ordered_ctx["segment_indices"] = ctx.get("segment_indices", [])
            ordered_ctx["segments"] = ctx.get("segments", [])
            new_contexts.append(ordered_ctx)
        scene["contexts"] = new_contexts
    _save_json(by_scene_path, _scene_payload(scenes_with_emphasis))

    elapsed = time.time() - t0
    _done("by_scene 구조 저장", elapsed)
    return {
        "by_scene_path": str(by_scene_path),
        "elapsed": elapsed,
    }


def fuse_preprocessed_data(
    args,
    textualized_path: str,
    annotation_path: str,
    audio_result: dict,
    output_dir: Path,
) -> dict:
    from .fusion import Config as FusionConfig, run_fusion

    stem = Path(args.input).stem
    fused_path = output_dir / f"{stem}_fused.json"
    by_scene_path = output_dir / f"{stem}_by_scene.json"

    if _is_done(fused_path, "P5 fusion — 데이터 퓨전", args.force):
        return {"fused_path": str(fused_path), "elapsed": 0.0}

    _banner("P5 fusion — 데이터 퓨전  (fusion)")
    t0 = time.time()

    cfg = FusionConfig(
        stem=stem,
        output_dir=output_dir,
        slides_dir=Path(args.slides),
        audio_path=by_scene_path,
        classified_path=output_dir / f"{stem}_slide_classified.json",
        annotation_path=Path(annotation_path),
        output_path=fused_path,
    )
    fused_output = run_fusion(cfg)
    fused_scenes = fused_output.get("scenes", [])
    fused_slides = fused_output.get("slides", [])
    logical_slide_count = len(fused_slides)

    # run_fusion 반환 스키마를 메인 파이프라인 출력 형식에 맞게 감싼다.
    _save_json(
        fused_path,
        {
            "video_path": args.input,
            "description": "영상(scene+slide+annotation) + 오디오 퓨전 결과",
            "scene_count": len(fused_scenes),
            "slide_count": logical_slide_count,
            "fusion_metadata": fused_output.get("metadata", {}),
            "slides": fused_slides,
            "scenes": fused_scenes,
        },
    )
    elapsed = time.time() - t0
    _done(f"scene {len(fused_scenes)}개 / slide {logical_slide_count}개 퓨전", elapsed)
    return {"fused_path": str(fused_path), "elapsed": elapsed}


def build_analyzer_input(
    args,
    meta_path: str,
    textualized_path: str,
    segments_path: str,
    output_dir: Path,
    duration: float,
    slides_structure: Optional[list[dict]] = None,
) -> dict:
    from .segment_grouper import load_slide_ranges
    from .text_processor import classify_lecture_domain

    stem = Path(args.input).stem
    analyzer_dir = output_dir / f"{stem}_analyzer"
    analyzer_dir.mkdir(parents=True, exist_ok=True)
    merged_clean_path = analyzer_dir / f"{stem}_merged_clean.json"

    if not args.force and merged_clean_path.exists() and merged_clean_path.stat().st_size > 0:
        try:
            with open(merged_clean_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            if any(slide.get("contexts") for slide in existing.get("slides", [])):
                print(f"\n  ⏭  V1 build_analyzer_input — verifier 입력 context 파일 존재, 스킵")
                print(f"     {merged_clean_path}")
                print("─" * 70)
                return {"merged_clean_path": str(merged_clean_path), "elapsed": 0.0}
        except Exception:
            pass

    _banner("V1 build_analyzer_input — verifier 입력용 merged_clean 생성")
    t0 = time.time()

    with open(textualized_path, "r", encoding="utf-8") as f:
        textualized = json.load(f)
    with open(segments_path, "r", encoding="utf-8") as f:
        segment_payload = json.load(f)
    segments = segment_payload.get("segments", [])
    slide_ranges = load_slide_ranges(meta_path, duration) if meta_path and Path(meta_path).is_file() else []
    metadata = []
    if meta_path and Path(meta_path).is_file():
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    scene_meta_by_index: dict[int, dict] = {}
    for entry in metadata:
        if entry.get("capture_type") != "base" and int(entry.get("annot_index", 0) or 0) != 0:
            continue
        scene_idx = entry.get("scene_index")
        if not isinstance(scene_idx, int) or scene_idx in scene_meta_by_index:
            continue
        scene_meta_by_index[scene_idx] = {
            "slide_number": entry.get("slide_number", entry.get("slide_canonical_index", scene_idx)),
            "slide_is_revisit": bool(
                entry.get("slide_is_revisit", entry.get("same_slide_is_revisit", False))
            ),
            "slide_visit_order": int(
                entry.get("slide_visit_order", entry.get("same_slide_visit_order", 1)) or 1
            ),
        }

    slide_meta_by_no = {}
    for slide in textualized.get("scenes", []):
        scene_no = slide.get("scene_number", slide.get("slide_number"))
        scene_info = scene_meta_by_index.get(scene_no if isinstance(scene_no, int) else -1, {})
        slide_no = scene_info.get(
            "slide_number",
            slide.get("slide_number"),
        )
        if isinstance(slide_no, int):
            text_parts = []
            if slide.get("t1"):
                text_parts.append(str(slide.get("t1")))
            if slide.get("t1_structure"):
                text_parts.append(str(slide.get("t1_structure")))
            candidate_meta = {
                "title": str(slide.get("title", "") or ""),
                "slide_text": "\n".join(part for part in text_parts if part),
                "slide_id": slide.get("slide_id", ""),
                "slide_type": slide.get("slide_type", ""),
                "text_source": slide.get("text_source", ""),
                "t1": slide.get("t1", ""),
                "t1_structure": slide.get("t1_structure", ""),
                "image_path": slide.get("image_path", ""),
            }
            current_meta = slide_meta_by_no.get(slide_no)
            if current_meta is None or len(candidate_meta["slide_text"]) > len(current_meta.get("slide_text", "")):
                slide_meta_by_no[slide_no] = candidate_meta

    segs_by_logical_slide: dict[int, list[dict]] = {}
    for seg in segments:
        slide_no = seg.get("slide_number")
        if isinstance(slide_no, int):
            seg_copy = {
                "start": float(seg.get("start", 0.0) or 0.0),
                "end": float(seg.get("end", seg.get("start", 0.0)) or 0.0),
                "text": str(seg.get("text", "") or "").strip(),
            }
            segs_by_logical_slide.setdefault(slide_no, []).append(seg_copy)

    contexts_by_slide: dict[int, list[dict]] = {}
    context_count = 0
    for slide_ctx in slides_structure or []:
        scene_idx = slide_ctx.get("slide_index", slide_ctx.get("scene_index"))
        scene_info = scene_meta_by_index.get(scene_idx if isinstance(scene_idx, int) else -1, {})
        slide_no = scene_info.get("slide_number", scene_idx)
        if not isinstance(slide_no, int):
            continue
        visit_order = int(scene_info.get("slide_visit_order", 1) or 1)
        for raw_ctx in slide_ctx.get("contexts", []) or []:
            text = str(raw_ctx.get("text", "") or "").strip()
            if not text:
                continue
            context_index = int(raw_ctx.get("context_index", 0) or 0)
            context_count += 1
            scene_token = int(scene_idx) if isinstance(scene_idx, int) else context_count
            context_id = f"S{slide_no:03d}-SC{scene_token:04d}-C{context_index + 1:03d}"
            context_payload = {
                "context_id": context_id,
                "slide_number": slide_no,
                "scene_index": scene_idx,
                "visit_order": visit_order,
                "context_index": context_index,
                "start_time": float(raw_ctx.get("start", 0.0) or 0.0),
                "end_time": float(raw_ctx.get("end", raw_ctx.get("start", 0.0)) or 0.0),
                "text": text,
                "source_segment_indices": raw_ctx.get("segment_indices", []),
            }
            contexts_by_slide.setdefault(slide_no, []).append(context_payload)

    if not contexts_by_slide:
        log.warning("V1 build_analyzer_input context 입력이 비어 있어 segment를 context 단위로 폴백합니다.")
        for slide_no, slide_segments in segs_by_logical_slide.items():
            for idx, seg in enumerate(sorted(slide_segments, key=lambda item: item.get("start", 0.0))):
                text = str(seg.get("text", "") or "").strip()
                if not text:
                    continue
                context_count += 1
                contexts_by_slide.setdefault(slide_no, []).append({
                    "context_id": f"S{slide_no:03d}-V01-C{idx + 1:03d}",
                    "slide_number": slide_no,
                    "scene_index": None,
                    "visit_order": 1,
                    "context_index": idx,
                    "start_time": float(seg.get("start", 0.0) or 0.0),
                    "end_time": float(seg.get("end", seg.get("start", 0.0)) or 0.0),
                    "text": text,
                    "source_segment_indices": [],
                })

    slide_titles = [slide_meta_by_no.get(slide_no, {}).get("title", "") for slide_no in sorted(slide_meta_by_no)]
    transcript_sample = " ".join(str(seg.get("text", "") or "") for seg in segments[:30])
    domain_info = classify_lecture_domain(slide_titles, transcript_sample)

    occurrences_by_logical_slide: dict[int, list[dict]] = {}
    for slide_range in slide_ranges:
        scene_idx = int(slide_range["scene_index"])
        scene_info = scene_meta_by_index.get(scene_idx, {})
        slide_no = scene_info.get("slide_number", scene_idx)
        if not isinstance(slide_no, int):
            continue
        start_sec = float(slide_range["start_sec"])
        end_sec = float(slide_range["end_sec"])
        slide_duration = round(end_sec - start_sec, 1)
        occurrences_by_logical_slide.setdefault(slide_no, []).append({
            "scene_index": scene_idx,
            "start_sec": start_sec,
            "end_sec": end_sec,
            "duration": slide_duration,
            "is_dup": bool(scene_info.get("slide_is_revisit", False)),
            "visit_order": int(scene_info.get("slide_visit_order", 1) or 1),
        })

    all_slide_numbers = sorted(
        set(slide_meta_by_no)
        | set(segs_by_logical_slide)
        | set(occurrences_by_logical_slide)
        | set(contexts_by_slide)
    )
    slides = []
    for slide_no in all_slide_numbers:
        slide_meta = slide_meta_by_no.get(slide_no, {})
        transcript_segments = sorted(segs_by_logical_slide.get(slide_no, []), key=lambda item: item.get("start", 0.0))
        occurrences = sorted(
            occurrences_by_logical_slide.get(slide_no, []),
            key=lambda item: (item["start_sec"], item["scene_index"]),
        )
        if occurrences:
            start_sec = min(item["start_sec"] for item in occurrences)
            end_sec = max(item["end_sec"] for item in occurrences)
            total_duration_sec = sum(item["duration"] for item in occurrences)
        elif transcript_segments:
            start_sec = min(item["start"] for item in transcript_segments)
            end_sec = max(item["end"] for item in transcript_segments)
            total_duration_sec = round(end_sec - start_sec, 1)
        else:
            start_sec = 0.0
            end_sec = 0.0
            total_duration_sec = 0.0
        slide_payload = {
            "slide_number": slide_no,
            "title": slide_meta.get("title", ""),
            "time_range": f"{_fmt_ts(start_sec)} ~ {_fmt_ts(end_sec)}",
            "time_range_seconds": [start_sec, end_sec],
            "total_duration": round(total_duration_sec, 1),
            "occurrences": occurrences,
            "slide_text": slide_meta.get("slide_text", ""),
            "contexts": sorted(contexts_by_slide.get(slide_no, []), key=lambda item: item.get("start_time", 0.0)),
            "context_count": len(contexts_by_slide.get(slide_no, [])),
        }
        for key in (
            "slide_id",
            "slide_type",
            "text_source",
            "t1",
            "t1_structure",
            "image_path",
        ):
            value = slide_meta.get(key)
            if value not in (None, "", []):
                slide_payload[key] = value
        slides.append(slide_payload)

    total_duration = 0.0
    if slides:
        total_duration = max(float(slide["time_range_seconds"][1]) for slide in slides)
    elif segments:
        total_duration = max(float(seg.get("end", 0.0) or 0.0) for seg in segments)
    else:
        total_duration = duration

    result = {
        "description": "슬라이드+전사 통합 JSON (교정 완료, 검증용)",
        "domain": domain_info.get("domain", ""),
        "subdomain": domain_info.get("subdomain", ""),
        "total_slides": len(slides),
        "total_contexts": context_count,
        "total_duration_formatted": _fmt_ts(total_duration),
        "slides": slides,
    }
    _save_json(merged_clean_path, result)

    elapsed = time.time() - t0
    _done("analyzer 입력용 merged_clean 생성", elapsed)
    return {"merged_clean_path": str(merged_clean_path), "elapsed": elapsed}


def _claim_output_is_final_verification(claim_output_path: Path) -> bool:
    if not claim_output_path.exists():
        return False
    try:
        with open(claim_output_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return False
    return payload.get("mode") == "classified_issue_verifier"


def extract_claims(args, merged_clean_path: str, output_dir: Path) -> dict:
    from .analyzer.claim_extractor import (
        _claim_extract_batch_mode,
        _claim_extract_context_window,
        extract_claims_only,
    )
    from .analyzer.claim_pipeline import prepare_verification
    from .analyzer.verifier_utils import _write_claims_jsonl

    def _claim_cache_matches(path: Path, batch_mode: str, context_window: tuple[int, int]) -> bool:
        if not path.exists() or path.stat().st_size <= 0:
            return False
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return False
        return (
            payload.get("claim_batch_mode") == batch_mode
            and tuple(payload.get("claim_context_window", [])) == context_window
        )

    def _claim_output_payload(claim: dict) -> dict:
        context_id = str(claim.get("context_id") or "").strip()
        payload = {
            "claim_id": claim.get("claim_id", ""),
            "context_id": context_id,
            "claim_text": claim.get("claim_text", ""),
            "source_slice": claim.get("source_slice", ""),
            "resolved_claim": claim.get("resolved_claim", ""),
            "claim_type": claim.get("claim_type", ""),
            "needs_context": bool(claim.get("needs_context", False)),
        }
        return {key: value for key, value in payload.items() if value not in ("", [], None)}

    stem = Path(args.input).stem
    analyzer_dir = output_dir / f"{stem}_analyzer"
    analyzer_dir.mkdir(parents=True, exist_ok=True)
    claim_stub_path = analyzer_dir / f"{stem}_verification_final.json"
    claims_jsonl_path = analyzer_dir / f"{stem}_claims.jsonl"
    claims_json_path = analyzer_dir / f"{stem}_claims.json"
    merged_file = Path(merged_clean_path)
    claim_batch_mode = _claim_extract_batch_mode()
    claim_context_window = _claim_extract_context_window()

    if (
        not args.force
        and claims_jsonl_path.exists()
        and _claim_cache_matches(claims_json_path, claim_batch_mode, claim_context_window)
        and claims_jsonl_path.stat().st_mtime >= merged_file.stat().st_mtime
    ):
        print(f"\n  ⏭  V2A extract_claims — claim 추출 출력 파일 존재, 스킵")
        print(f"     {claims_jsonl_path}")
        print("─" * 70)
        return {
            "claims_jsonl": str(claims_jsonl_path),
            "claims_json": str(claims_json_path),
            "claim_count": 0,
            "elapsed": 0.0,
            "skipped": True,
        }

    _banner("V2A extract_claims — claim 추출")
    t0 = time.time()
    ctx = prepare_verification(str(merged_file))
    claims_by_batch, api_calls, token_usage = extract_claims_only(
        ctx["contexts"],
        ctx["current_date"],
        ctx["hint"],
        ctx["slide_ctx"],
    )
    claims: list[dict] = []
    for _, batch_claims in claims_by_batch:
        claims.extend(_claim_output_payload(claim) for claim in batch_claims)

    claims_log_path = _write_claims_jsonl(claims, claim_stub_path)
    claims_json_path.write_text(
        json.dumps(
            {
                "mode": "claim_extraction",
                "claim_batch_mode": claim_batch_mode,
                "claim_context_window": list(claim_context_window),
                "merged_path": str(merged_file),
                "claims_log_path": claims_log_path,
                "claim_count": len(claims),
                "api_calls": api_calls,
                "token_usage": token_usage,
                "claims": claims,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    elapsed = time.time() - t0
    _done(f"claim {len(claims)}개 추출", elapsed)
    return {
        "claims_jsonl": str(claims_jsonl_path),
        "claims_json": str(claims_json_path),
        "claim_count": len(claims),
        "api_calls": api_calls,
        "elapsed": elapsed,
    }


def judge_issues(args, merged_clean_path: str, output_dir: Path, claims_jsonl: str) -> dict:
    from .analyzer.run_all import run_issue_judge_only

    stem = Path(args.input).stem
    analyzer_dir = output_dir / f"{stem}_analyzer"
    analyzer_dir.mkdir(parents=True, exist_ok=True)

    _banner("V2B judge_issues — 1차 issue 판단")
    t0 = time.time()
    result = run_issue_judge_only(
        merged_clean_path,
        output_dir=str(analyzer_dir),
        claims_jsonl=claims_jsonl,
        issue_judge_min_confidence=getattr(args, "issue_judge_min_confidence", None),
    )
    elapsed = time.time() - t0
    _done("1차 issue judge", elapsed)
    return {"elapsed": elapsed, **result}


def start_verifier_background(args, merged_clean_path: str, output_dir: Path) -> dict:
    stem = Path(args.input).stem
    analyzer_dir = output_dir / f"{stem}_analyzer"
    analyzer_dir.mkdir(parents=True, exist_ok=True)
    claim_output_path = analyzer_dir / f"{stem}_verification_final.json"
    claim_report_path = analyzer_dir / f"{stem}_report.txt"
    analyzer_log_path = analyzer_dir / f"{stem}_analyzer.log"

    if (
        not args.force
        and _claim_output_is_final_verification(claim_output_path)
        and claim_output_path.stat().st_mtime >= Path(merged_clean_path).stat().st_mtime
    ):
        print(f"\n  ⏭  V2 start_verifier_background — verifier 출력 파일 존재, 스킵")
        print(f"     {claim_output_path}")
        print("─" * 70)
        return {
            "claim_output": str(claim_output_path),
            "claim_report": "",
            "log_path": str(analyzer_log_path),
            "elapsed": 0.0,
            "pid": None,
            "spawned": False,
        }

    _banner("V2 start_verifier_background — verifier 백그라운드 실행")
    t0 = time.time()
    pkg_root = resolve_pipeline_package_root()
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "pipeline.analyzer.run_all",
        merged_clean_path,
        "--output-dir",
        str(analyzer_dir),
    ]

    with open(analyzer_log_path, "a", encoding="utf-8") as log_fp:
        log_fp.write(
            f"\n=== verifier launch {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n"
        )
        log_fp.write(f"cwd       : {pkg_root}\n")
        log_fp.write(f"cmd       : {' '.join(cmd)}\n")
        log_fp.write("mode      : classified_issue_pipeline\n")
        log_fp.flush()
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        proc = subprocess.Popen(
            cmd,
            cwd=str(pkg_root),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log_fp,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    elapsed = time.time() - t0
    print(f"\n  ✓ verifier 백그라운드 시작  ({elapsed:.1f}초)")
    print(f"     PID : {proc.pid}")
    print("     mode: classified_issue_pipeline")
    print(f"     로그: {analyzer_log_path}")
    print("─" * 70)
    return {
        "claim_output": str(claim_output_path),
        "claim_report": "",
        "log_path": str(analyzer_log_path),
        "elapsed": elapsed,
        "pid": proc.pid,
        "spawned": True,
    }


def run_verifier(args, merged_clean_path: str, output_dir: Path) -> dict:
    """Run the verifier synchronously for approval-gated workflows."""
    from .analyzer.run_all import run_classified_issue_pipeline

    stem = Path(args.input).stem
    analyzer_dir = output_dir / f"{stem}_analyzer"
    analyzer_dir.mkdir(parents=True, exist_ok=True)
    claim_output_path = analyzer_dir / f"{stem}_verification_final.json"
    claim_report_path = analyzer_dir / f"{stem}_report.txt"

    if (
        not args.force
        and _claim_output_is_final_verification(claim_output_path)
        and claim_output_path.stat().st_mtime >= Path(merged_clean_path).stat().st_mtime
    ):
        print(f"\n  ⏭  V2 run_verifier — verifier 출력 파일 존재, 스킵")
        print(f"     {claim_output_path}")
        print("─" * 70)
        return {
            "claim_output": str(claim_output_path),
            "claim_report": str(claim_report_path) if claim_report_path.exists() else "",
            "elapsed": 0.0,
            "pid": None,
            "spawned": False,
            "skipped": True,
        }

    _banner("V2 run_verifier — verifier 실행")
    t0 = time.time()
    result = run_classified_issue_pipeline(
        merged_clean_path,
        output_dir=str(analyzer_dir),
        issue_judge_min_confidence=getattr(args, "issue_judge_min_confidence", None),
    )
    elapsed = time.time() - t0
    _done("verifier 실행", elapsed)
    return {
        "claim_output": result.get("claim_output", str(claim_output_path)),
        "claim_report": result.get("claim_report", str(claim_report_path)),
        "elapsed": elapsed,
        "pid": None,
        "spawned": False,
        **result,
    }


def generate_graph_triples(args, output_dir: Path, slides_dir: Path) -> dict:
    from .json_to_graph_triples import Config as TripleConfig, GraphPipeline

    stem = Path(args.input).stem
    triples_parquet = output_dir / f"{stem}_graph_triples.parquet"
    nodes_parquet = output_dir / f"{stem}_nodes.parquet"
    edges_parquet = output_dir / f"{stem}_edges.parquet"

    if _is_done(triples_parquet, "G1 graph_triples — 그래프 트리플 생성", args.force):
        return {
            "triples_parquet": str(triples_parquet),
            "nodes_parquet": str(nodes_parquet),
            "edges_parquet": str(edges_parquet),
            "elapsed": 0.0,
        }

    _banner("G1 graph_triples — 그래프 트리플 생성  (json_to_graph_triples)")
    t0 = time.time()

    cfg = TripleConfig(
        stem=stem,
        output_dir=output_dir,
        slides_dir=slides_dir,
        lecture_title=(getattr(args, "title", "") or "").strip() or stem,
    )
    GraphPipeline(cfg).run()

    elapsed = time.time() - t0
    _done("그래프 트리플 생성", elapsed)
    return {
        "triples_parquet": str(triples_parquet),
        "nodes_parquet": str(nodes_parquet),
        "edges_parquet": str(edges_parquet),
        "elapsed": elapsed,
    }


def build_lance_index(args, output_dir: Path, slides_dir: Path) -> dict:
    """fused.json → 청크 임베딩 → Parquet + LanceDB (단일 테이블, stem 필터)."""
    from .lance_ingest import default_lance_root, ingest_stem_to_lance

    stem = Path(args.input).stem
    from .config import output_paths

    paths = output_paths(stem, output_dir, slides_dir)
    fused_path = paths["fused"]
    lance_root = Path(args.lance_root) if getattr(args, "lance_root", None) else default_lance_root()
    parquet_path = output_dir / f"{stem}_chunks_lance.parquet"

    if _is_done(parquet_path, "G2 lance_index — Lance 인덱스 생성", args.force):
        return {"elapsed": 0.0, "parquet_path": str(parquet_path), "skipped": True}

    if not fused_path.exists():
        raise FileNotFoundError(f"G2 lance_index — fused 파일 없음. P5 fusion — 데이터 퓨전이 필요합니다: {fused_path}")

    _banner("G2 lance_index — LanceDB 인덱스  (Gemini 임베딩 + lance_ingest)")
    t0 = time.time()
    result = ingest_stem_to_lance(
        stem=stem,
        fused_path=fused_path,
        output_dir=output_dir,
        lance_root=lance_root,
    )
    elapsed = time.time() - t0
    cnt = result.get("count", 0)
    _done(f"Lance 인덱스 ({cnt}청크)", elapsed)
    return {"elapsed": elapsed, **result}


def _find_graphrag_executable() -> Optional[str]:
    configured = os.getenv("GRAPHRAG_BIN")
    if configured and Path(configured).exists():
        return configured
    return shutil.which("graphrag")


def _write_graphrag_env(workspace_dir: Path, api_key: str) -> None:
    env_path = workspace_dir / ".env"
    existing: list[str] = []
    if env_path.exists():
        existing = [
            line
            for line in env_path.read_text(encoding="utf-8").splitlines()
            if not line.startswith("GRAPHRAG_API_KEY=")
        ]
    existing.append(f"GRAPHRAG_API_KEY={api_key}")
    env_path.write_text("\n".join(existing).strip() + "\n", encoding="utf-8")


def _patch_graphrag_extract_prompt(workspace_dir: Path) -> None:
    """Keep bilingual lecture terms on one canonical GraphRAG entity."""
    prompt_path = workspace_dir / "prompts" / "extract_graph.txt"
    if not prompt_path.exists():
        return

    text = prompt_path.read_text(encoding="utf-8")
    original = text
    marker = "-GraphLec Entity Canonicalization Rules-"
    rules = f"""

{marker}
- The lecture content is primarily Korean and may include English terms in parentheses.
- Use the dominant Korean lecture term as the canonical entity name when Korean and English refer to the same concept.
- Treat parenthesized English terms, acronyms, capitalization variants, and translations as aliases, not separate entities.
- Do not emit separate entities for bilingual variants, parenthesized aliases, acronyms, casing variants, or direct translations of the same concept; emit one canonical entity and mention aliases in the description.
- If a concept appears only in English and no Korean equivalent is present in the text, keep the English name.
- Apply the same canonical entity name consistently in relationships.
"""

    anchor = "Format each entity as (\"entity\"<|><entity_name><|><entity_type><|><entity_description>)"
    if marker not in text and anchor in text:
        text = text.replace(anchor, anchor + rules, 1)

    text = text.replace(
        "3. Return output in English as a single list of all the entities and relationships identified in steps 1 and 2. Use **##** as the list delimiter.",
        "3. Return output in the dominant lecture language as a single list of all the entities and relationships identified in steps 1 and 2. For Korean lectures, use Korean canonical entity names and descriptions. Use **##** as the list delimiter.",
    )

    if text != original:
        prompt_path.write_text(text, encoding="utf-8")


def _graphrag_workspace_dir(args, output_dir: Path, stem: str) -> Path:
    root = getattr(args, "graphrag_root", None)
    if root:
        return Path(root) / stem
    return output_dir / "graphrag"


def build_graphrag_index(args, output_dir: Path) -> dict:
    """fused.json → GraphRAG workspace parquet."""
    from .config import output_paths
    from .fused_to_graphrag_text import fused_to_graphrag_text

    stem = Path(args.input).stem
    paths = output_paths(stem, output_dir, Path(args.slides))
    fused_path = paths["fused"]
    workspace_dir = _graphrag_workspace_dir(args, output_dir, stem)
    input_dir = workspace_dir / "input"
    output_graph_dir = workspace_dir / "output"
    entities_path = output_graph_dir / "entities.parquet"
    relationships_path = output_graph_dir / "relationships.parquet"
    input_path = input_dir / f"{stem}.txt"
    metrics_path = workspace_dir / "graphrag_index_metrics.json"
    model_name = os.getenv("GRAPHLEC_GRAPHRAG_MODEL", "gpt-5.4")
    embedding_model = os.getenv("GRAPHLEC_GRAPHRAG_EMBEDDING_MODEL", "text-embedding-3-small")
    method_name = getattr(args, "graphrag_method", "standard")

    if (
        not args.force
        and entities_path.exists()
        and entities_path.stat().st_size > 0
        and relationships_path.exists()
        and relationships_path.stat().st_size > 0
    ):
        print("\n  ⏭  G3 graphrag_index — GraphRAG parquet 출력 파일 존재, 스킵")
        print(f"     {output_graph_dir}")
        print("─" * 70)
        return {
            "elapsed": 0.0,
            "skipped": True,
            "workspace_dir": str(workspace_dir),
            "input_path": str(input_path),
            "entities_parquet": str(entities_path),
            "relationships_parquet": str(relationships_path),
        }

    if not fused_path.exists():
        raise FileNotFoundError(f"G3 graphrag_index — fused 파일 없음. P5 fusion — 데이터 퓨전이 필요합니다: {fused_path}")

    graphrag_bin = _find_graphrag_executable()
    if not graphrag_bin:
        raise RuntimeError("G3 graphrag_index — graphrag CLI를 찾을 수 없습니다. requirements 설치 후 다시 실행하세요.")

    api_key = os.getenv("GRAPHRAG_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("G3 graphrag_index — GRAPHRAG_API_KEY 또는 OPENAI_API_KEY 환경변수가 필요합니다.")

    _banner("G3 graphrag_index — GraphRAG 인덱스  (fused → parquet workspace)")
    t0 = time.time()

    workspace_dir.mkdir(parents=True, exist_ok=True)
    input_dir.mkdir(parents=True, exist_ok=True)
    with open(fused_path, "r", encoding="utf-8") as f:
        fused = json.load(f)
    input_path.write_text(fused_to_graphrag_text(fused, stem=stem), encoding="utf-8")

    env = os.environ.copy()
    env["GRAPHRAG_API_KEY"] = api_key
    env["PYTHONUNBUFFERED"] = "1"
    _write_graphrag_env(workspace_dir, api_key)

    metrics_path.write_text(
        json.dumps(
            {
                "stem": stem,
                "status": "running",
                "model": model_name,
                "embedding_model": embedding_model,
                "method": method_name,
                "started_at_epoch": t0,
                "workspace_dir": str(workspace_dir),
                "input_path": str(input_path),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    settings_path = workspace_dir / "settings.yaml"
    if not settings_path.exists():
        subprocess.run(
            [
                graphrag_bin,
                "init",
                "--root",
                str(workspace_dir),
                "--model",
                model_name,
                "--embedding",
                embedding_model,
            ],
            check=True,
            env=env,
        )
        _write_graphrag_env(workspace_dir, api_key)

    _patch_graphrag_extract_prompt(workspace_dir)

    if args.force and output_graph_dir.exists():
        shutil.rmtree(output_graph_dir)

    subprocess.run(
        [
            graphrag_bin,
            "index",
            "--root",
            str(workspace_dir),
            "--method",
            method_name,
        ],
        check=True,
        env=env,
    )

    elapsed = time.time() - t0
    metrics_path.write_text(
        json.dumps(
            {
                "stem": stem,
                "status": "done",
                "model": model_name,
                "embedding_model": embedding_model,
                "method": method_name,
                "started_at_epoch": t0,
                "finished_at_epoch": time.time(),
                "elapsed": elapsed,
                "elapsed_sec": elapsed,
                "workspace_dir": str(workspace_dir),
                "input_path": str(input_path),
                "entities_parquet": str(entities_path),
                "relationships_parquet": str(relationships_path),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    _done("GraphRAG 인덱스 생성", elapsed)
    return {
        "elapsed": elapsed,
        "workspace_dir": str(workspace_dir),
        "input_path": str(input_path),
        "entities_parquet": str(entities_path),
        "relationships_parquet": str(relationships_path),
        "metrics_path": str(metrics_path),
    }


def generate_metadata(args, output_dir: Path, slides_dir: Path) -> dict:
    """G4 metadata: 강의 메타데이터 생성."""
    from .generate_metadata import generate_metadata

    stem         = Path(args.input).stem
    metadata_dir = Path(getattr(args, "metadata_dir", DEFAULT_RECOMMENDER_METADATA_DIR))
    output_path  = metadata_dir / f"{stem}_metadata.json"

    if _is_done(output_path, "G4 metadata — 메타데이터 생성", args.force):
        return {"metadata_path": str(output_path), "elapsed": 0.0}

    _banner("G4 metadata — 메타데이터 생성  (generate_metadata)")
    t0 = time.time()

    generate_metadata(
        stem          = stem,
        title         = getattr(args, "title", ""),
        instructor_id = getattr(args, "instructor", ""),
        output_dir    = output_dir,
        metadata_dir  = metadata_dir,
        uploaded_at   = getattr(args, "uploaded_at", None),
    )

    elapsed = time.time() - t0
    _done("메타데이터 생성", elapsed)
    return {"metadata_path": str(output_path), "elapsed": elapsed}


def build_recommender_index(args) -> dict:
    """G5 recommender_index: 추천용 metadata 임베딩 인덱스 생성 (build_index.py)."""
    recommender_dir = Path(__file__).resolve().parents[1] / "recommender"
    script_path = recommender_dir / "build_index.py"
    metadata_dir = Path(getattr(args, "metadata_dir", DEFAULT_RECOMMENDER_METADATA_DIR))
    db_dir = Path(getattr(args, "recommender_db_dir", DEFAULT_RECOMMENDER_DB_DIR))

    _banner("G5 recommender_index — 추천 인덱스 생성  (build_index)")
    t0 = time.time()

    cmd = [
        sys.executable,
        str(script_path),
        "--metadata_dir",
        str(metadata_dir),
        "--db_dir",
        str(db_dir),
    ]
    subprocess.run(cmd, check=True)

    elapsed = time.time() - t0
    _done("추천 인덱스 생성", elapsed)
    return {"elapsed": elapsed, "db_dir": str(db_dir)}


# ──────────────────────────────────────────────────────────────
# 파이프라인 오케스트레이션
# ──────────────────────────────────────────────────────────────

class PipelineRuntime:
    """Shared state for composing the pipeline into workflow-sized units."""

    def __init__(self, args, progress_callback=None):
        from .config import output_paths

        self.args = args
        self.progress_callback = progress_callback
        self.total_start = time.time()
        self.timings: dict[str, float] = {}
        self.stage_status: dict[str, str] = {}
        self.stem = Path(args.input).stem
        self.slides_dir = Path(args.slides)
        self.output_dir = Path(args.output)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.slides_dir.mkdir(parents=True, exist_ok=True)
        self.timing_path = self.output_dir / "pipeline_timings.json"
        self.paths = output_paths(self.stem, self.output_dir, self.slides_dir)

    def notify_stage(self, stage_key: str, status: str) -> None:
        if not self.progress_callback:
            return
        try:
            self.progress_callback(stage_key, status)
        except Exception as e:
            log.warning(f"progress_callback failed for {stage_key}: {e}")

    def write_timings(self, current_stage: str | None = None, status: str = "running") -> None:
        payload = {
            "stem": self.stem,
            "status": status,
            "current_stage": current_stage,
            "started_at_epoch": self.total_start,
            "updated_at_epoch": time.time(),
            "elapsed_total_sec": time.time() - self.total_start,
            "timings": self.timings,
            "stage_status": self.stage_status,
        }
        tmp_path = self.timing_path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp_path.replace(self.timing_path)

    def record_timing(self, stage: str, elapsed: float, status: str = "done") -> None:
        self.timings[stage] = elapsed
        self.stage_status[stage] = status
        self.write_timings(stage)


def _create_pipeline_runtime(args, progress_callback=None, title: str = "강의 영상 분석 통합 파이프라인") -> PipelineRuntime:
    runtime = PipelineRuntime(args, progress_callback)
    runtime.write_timings("pipeline_start")
    try:
        from .cost_report import configure as configure_cost_report, reset as reset_cost_report

        reset_cost_report()
        configure_cost_report(stem=runtime.stem, output_dir=runtime.output_dir)
    except Exception as e:
        log.warning(f"cost_report 초기화 실패: {e}")

    print("\n" + "═" * 70)
    print(f"  {title}")
    print("═" * 70)
    print(f"  입력 영상 : {args.input}")
    print(f"  슬라이드  : {runtime.slides_dir}")
    print(f"  출력      : {runtime.output_dir}")
    if args.force:
        print("  ⚠️  --force: 모든 단계 강제 재실행")
    return runtime


def run_preprocess_pipeline(args, runtime: PipelineRuntime, *, should_build_analyzer_input: bool = True) -> dict:
    """Build the shared artifacts consumed by graph and verifier workflows."""
    stem = runtime.stem
    paths = runtime.paths
    output_dir = runtime.output_dir
    slides_dir = runtime.slides_dir
    timings = runtime.timings

    r9: dict = {}

    _banner("P1 extract_media — 슬라이드 추출 + 오디오 품질 분석")
    t_parallel = time.time()
    audio_analyze_result: dict = {}
    runtime.notify_stage("preprocess_extract_media", "run")

    if args.skip_extract:
        log.info("P1A extract_slides — 슬라이드 추출 건너뜀 (--skip-extract)")
        meta_path = str(paths["metadata"])
        timings["P1A extract_slides — 슬라이드 추출"] = 0.0
        audio_analyze_result = analyze_audio_quality(args, output_dir)
        timings["P1B analyze_audio_quality — 오디오 품질 분석"] = audio_analyze_result["elapsed"]
    else:
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_1a = executor.submit(extract_slides, args, slides_dir, output_dir)
            future_1b = executor.submit(analyze_audio_quality, args, output_dir)
            for future in as_completed([future_1a, future_1b]):
                if future is future_1a:
                    r1 = future.result()
                    meta_path = r1["meta_path"]
                    timings["P1A extract_slides — 슬라이드 추출"] = r1["elapsed"]
                else:
                    audio_analyze_result = future.result()
                    timings["P1B analyze_audio_quality — 오디오 품질 분석"] = audio_analyze_result["elapsed"]

    duration = audio_analyze_result.get("duration", 0.0)
    timings["P1 extract_media total — 슬라이드 추출 + 오디오 품질 분석 총합"] = time.time() - t_parallel
    runtime.notify_stage("preprocess_extract_media", "done")
    runtime.write_timings("P1 extract_media total — 슬라이드 추출 + 오디오 품질 분석 총합")
    print(f"\n  ✓ P1 extract_media 완료 — 슬라이드 추출 + 오디오 품질 분석  ({timings['P1 extract_media total — 슬라이드 추출 + 오디오 품질 분석 총합']:.1f}초)")
    print("─" * 70)

    _banner("P2 textualize_transcribe — 슬라이드 텍스트화 + 전체 전사")
    t_parallel = time.time()
    transcript_result: dict = {}
    runtime.notify_stage("preprocess_textualize_transcribe", "run")

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_2a = executor.submit(textualize_slides, args, slides_dir, output_dir)
        future_2b = executor.submit(transcribe_audio, args, meta_path, duration, output_dir)
        for future in as_completed([future_2a, future_2b]):
            if future is future_2a:
                r2 = future.result()
                textualized_path = r2["textualized_path"]
                timings["P2A textualize_slides — 슬라이드 텍스트화"] = r2["elapsed"]
            else:
                transcript_result = future.result()
                timings["P2B transcribe_audio — 전체 전사"] = transcript_result["elapsed"]

    timings["P2 textualize_transcribe total — 텍스트화 + 전사 총합"] = time.time() - t_parallel
    runtime.notify_stage("preprocess_textualize_transcribe", "done")
    runtime.write_timings("P2 textualize_transcribe total — 텍스트화 + 전사 총합")
    print(f"\n  ✓ P2 textualize_transcribe 완료 — 슬라이드 텍스트화 + 전체 전사  ({timings['P2 textualize_transcribe total — 텍스트화 + 전사 총합']:.1f}초)")
    print("─" * 70)

    _banner("P3 enrich_audio_annotation — 필기 강조 분석 + 오디오 후처리")
    t_parallel = time.time()
    audio_result: dict = {}
    annotation_result: dict = {}
    analyzer_lock = Lock()
    analyzer_input_built = {"done": False}
    runtime.notify_stage("preprocess_enrich_audio_annotation", "run")

    transcript_raw_path = transcript_result.get(
        "transcript_raw_path",
        str(output_dir / f"{stem}_transcript_raw.json"),
    )

    def _build_analyzer_input_once(audio_payload: dict) -> None:
        with analyzer_lock:
            if analyzer_input_built["done"]:
                return
            runtime.notify_stage("verifier_build_analyzer_input", "run")
            local_r9 = build_analyzer_input(
                args,
                meta_path=meta_path,
                textualized_path=textualized_path,
                segments_path=audio_payload.get("segments_path", str(paths["segments"])),
                output_dir=output_dir,
                duration=audio_payload.get("duration", duration),
                slides_structure=audio_payload.get("slides_structure"),
            )
            timings["V1 build_analyzer_input — verifier 입력 생성"] = local_r9["elapsed"]
            r9.update(local_r9)
            analyzer_input_built["done"] = True
            runtime.notify_stage("verifier_build_analyzer_input", "done")

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_a = executor.submit(analyze_slide_annotations, args, slides_dir, output_dir)
        future_b = executor.submit(
            process_audio,
            args,
            meta_path,
            textualized_path,
            duration,
            output_dir,
            transcript_raw_path,
            _build_analyzer_input_once if should_build_analyzer_input else None,
        )
        for future in as_completed([future_a, future_b]):
            if future is future_a:
                annotation_result = future.result()
                timings["P3A analyze_annotation — 필기 강조 분석"] = annotation_result["elapsed"]
            else:
                audio_result = future.result()
                timings["P3B process_audio — 오디오 후처리"] = audio_result.get("elapsed", 0.0)
                if should_build_analyzer_input and not analyzer_input_built["done"]:
                    _build_analyzer_input_once(audio_result)

    timings["P3 enrich_audio_annotation total — 보강 분석 총합"] = time.time() - t_parallel
    runtime.notify_stage("preprocess_enrich_audio_annotation", "done")
    runtime.write_timings("P3 enrich_audio_annotation total — 보강 분석 총합")
    print(f"\n  ✓ P3 enrich_audio_annotation 완료 — 필기 강조 분석 + 오디오 후처리  ({timings['P3 enrich_audio_annotation total — 보강 분석 총합']:.1f}초)")
    print("─" * 70)

    _banner("P4 classify_scene — 슬라이드 분류 + scene 구조 저장")
    t_parallel = time.time()
    classified_result: dict = {}
    by_scene_result: dict = {}
    silences_path = audio_result.get("silences_path", str(paths["silences"]))
    annotation_path = annotation_result.get("annotation_path", str(paths["annotation"]))
    runtime.notify_stage("preprocess_classify_scene", "run")

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_c = executor.submit(
            classify_slides, args, textualized_path, meta_path, silences_path, output_dir
        )
        future_d = executor.submit(save_scene_structure, args, audio_result, output_dir)
        for future in as_completed([future_c, future_d]):
            if future is future_c:
                classified_result = future.result()
                timings["P4A classify_slides — 슬라이드 분류"] = classified_result.get("elapsed", 0.0)
            else:
                by_scene_result = future.result()
                timings["P4B save_scene_structure — scene 구조 저장"] = by_scene_result.get("elapsed", 0.0)

    timings["P4 classify_scene total — 구조화 총합"] = time.time() - t_parallel
    runtime.notify_stage("preprocess_classify_scene", "done")
    runtime.write_timings("P4 classify_scene total — 구조화 총합")
    print(f"\n  ✓ P4 classify_scene 완료 — 슬라이드 분류 + scene 구조 저장  ({timings['P4 classify_scene total — 구조화 총합']:.1f}초)")
    print("─" * 70)

    runtime.notify_stage("preprocess_fusion", "run")
    r5 = fuse_preprocessed_data(
        args,
        textualized_path=textualized_path,
        annotation_path=annotation_path,
        audio_result=audio_result,
        output_dir=output_dir,
    )
    timings["P5 fusion — 데이터 퓨전"] = r5["elapsed"]
    runtime.notify_stage("preprocess_fusion", "done")
    runtime.write_timings("P5 fusion — 데이터 퓨전")

    if should_build_analyzer_input and not timings.get("V1 build_analyzer_input — verifier 입력 생성"):
        timings["V1 build_analyzer_input — verifier 입력 생성"] = 0.0

    return {
        "meta_path": meta_path,
        "textualized_path": textualized_path,
        "transcript_result": transcript_result,
        "audio_result": audio_result,
        "annotation_path": annotation_path,
        "annotation_result": annotation_result,
        "classified_result": classified_result,
        "by_scene_result": by_scene_result,
        "fusion_result": r5,
        "analyzer_input_result": r9,
    }


def run_graph_pipeline(args, runtime: PipelineRuntime, preprocess_result: Optional[dict] = None) -> dict:
    """Build graph/search/recommendation artifacts from shared preprocess output."""
    output_dir = runtime.output_dir
    slides_dir = runtime.slides_dir
    timings = runtime.timings
    r6: dict = {}
    r7: dict = {}
    r7b: dict = {}
    r8: dict = {}
    r11: dict = {}

    runtime.notify_stage("graph_triples", "run")
    if args.skip_graph_triples:
        print("\n  ⏭  G1 graph_triples — 그래프 트리플 생성 스킵")
        print("─" * 70)
        timings["G1 graph_triples — 그래프 트리플 생성"] = 0.0
        timings["G1 neo4j_load — Neo4j 적재"] = 0.0
    else:
        r6 = generate_graph_triples(args, output_dir, slides_dir)
        timings["G1 graph_triples — 그래프 트리플 생성"] = r6["elapsed"]
        print("\n  ⏭  Neo4j 적재 — 강의 시청 화면 진입 시 자동 적재")
        print("─" * 70)
        timings["G1 neo4j_load — Neo4j 적재"] = 0.0
    runtime.notify_stage("graph_triples", "done")

    if args.skip_lance_index:
        print("\n  ⏭  G2 lance_index — Lance 인덱스 생성 스킵")
        print("─" * 70)
        timings["G2 lance_index — Lance 인덱스 생성"] = 0.0
    else:
        runtime.notify_stage("graph_lance_index", "run")
        r7 = build_lance_index(args, output_dir, slides_dir)
        timings["G2 lance_index — Lance 인덱스 생성"] = r7.get("elapsed", 0.0)
        runtime.notify_stage("graph_lance_index", "done")

    if getattr(args, "skip_graphrag_index", False):
        print("\n  ⏭  G3 graphrag_index — GraphRAG 인덱스 생성 스킵")
        print("─" * 70)
        runtime.record_timing("G3 graphrag_index — GraphRAG 인덱스 생성", 0.0, "skipped")
    else:
        runtime.notify_stage("graph_graphrag_index", "run")
        runtime.stage_status["G3 graphrag_index — GraphRAG 인덱스 생성"] = "run"
        runtime.write_timings("G3 graphrag_index — GraphRAG 인덱스 생성")
        r7b = build_graphrag_index(args, output_dir)
        runtime.notify_stage("graph_graphrag_index", "done")
        runtime.record_timing("G3 graphrag_index — GraphRAG 인덱스 생성", r7b.get("elapsed", 0.0), "done")

    if getattr(args, "skip_metadata", False):
        print("\n  ⏭  G4 metadata — 메타데이터 생성 스킵")
        print("─" * 70)
        timings["G4 metadata — 메타데이터 생성"] = 0.0
    else:
        runtime.notify_stage("graph_metadata", "run")
        r8 = generate_metadata(args, output_dir, slides_dir)
        timings["G4 metadata — 메타데이터 생성"] = r8["elapsed"]
        runtime.notify_stage("graph_metadata", "done")

    if getattr(args, "skip_recommender_index", False):
        print("\n  ⏭  G5 recommender_index — 추천 인덱스 생성 스킵")
        print("─" * 70)
        timings["G5 recommender_index — 추천 인덱스 생성"] = 0.0
    else:
        runtime.notify_stage("graph_recommender_index", "run")
        r11 = build_recommender_index(args)
        timings["G5 recommender_index — 추천 인덱스 생성"] = r11["elapsed"]
        runtime.notify_stage("graph_recommender_index", "done")

    runtime.write_timings("graph_pipeline_done")
    return {
        "graph_triples_result": r6,
        "lance_result": r7,
        "graphrag_result": r7b,
        "metadata_result": r8,
        "recommender_result": r11,
    }


def run_verifier_pipeline(
    args,
    runtime: PipelineRuntime,
    preprocess_result: Optional[dict] = None,
    *,
    background: bool = False,
) -> dict:
    """Run verifier stages from the shared analyzer input."""
    timings = runtime.timings
    if getattr(args, "skip_analyzer", False):
        timings["V2 start_verifier_background — verifier 백그라운드 시작"] = 0.0
        return {}

    analyzer_result = (preprocess_result or {}).get("analyzer_input_result", {})
    merged_clean_path = analyzer_result.get(
        "merged_clean_path",
        str(runtime.output_dir / f"{runtime.stem}_analyzer" / f"{runtime.stem}_merged_clean.json"),
    )
    if not Path(merged_clean_path).exists():
        raise FileNotFoundError(f"verifier 입력 파일 없음: {merged_clean_path}")

    r10: dict = {}
    r10a: dict = {}
    r10b: dict = {}

    if getattr(args, "stop_after_claim_extract", False) or getattr(args, "stop_after_issue_judge", False):
        runtime.notify_stage("verifier_run", "run")
        r10a = extract_claims(args, merged_clean_path=merged_clean_path, output_dir=runtime.output_dir)
        timings["V2A extract_claims — claim 추출"] = r10a["elapsed"]
        if getattr(args, "stop_after_issue_judge", False):
            r10b = judge_issues(
                args,
                merged_clean_path=merged_clean_path,
                output_dir=runtime.output_dir,
                claims_jsonl=r10a["claims_jsonl"],
            )
            timings["V2B judge_issues — 1차 issue 판단"] = r10b["elapsed"]
        timings["V2 start_verifier_background — verifier 백그라운드 시작"] = 0.0
        runtime.notify_stage("verifier_run", "done")
    elif background:
        runtime.notify_stage("verifier_run", "run")
        r10 = start_verifier_background(args, merged_clean_path, runtime.output_dir)
        timings["V2 start_verifier_background — verifier 백그라운드 시작"] = r10["elapsed"]
        runtime.notify_stage("verifier_run", "done")
    else:
        runtime.notify_stage("verifier_run", "run")
        r10 = run_verifier(args, merged_clean_path, runtime.output_dir)
        timings["V2 run_verifier — verifier 실행"] = r10["elapsed"]
        runtime.notify_stage("verifier_run", "done")

    runtime.write_timings("verifier_pipeline_done")
    return {
        "verifier_result": r10,
        "claims_result": r10a,
        "issue_judge_result": r10b,
    }


def _print_stop_after_summary(args, runtime: PipelineRuntime, verifier_result: dict) -> None:
    if getattr(args, "stop_after_issue_judge", False):
        option_name = "--stop-after-issue-judge"
    elif getattr(args, "stop_after_claim_extract", False):
        option_name = "--stop-after-claim-extract"
    else:
        option_name = "--stop-after-verifier-start"

    r10 = verifier_result.get("verifier_result", {})
    r10a = verifier_result.get("claims_result", {})
    r10b = verifier_result.get("issue_judge_result", {})
    print(f"\n  ⏹  {option_name}: 요청한 analyzer 단계 후 파이프라인을 종료합니다.")
    print("  생성된 analyzer 관련 파일:")
    for path_str in (
        str(runtime.output_dir / f"{runtime.stem}_analyzer" / f"{runtime.stem}_merged_clean.json"),
        r10a.get("claims_jsonl", ""),
        r10a.get("claims_json", ""),
        r10b.get("issue_judge_summary", ""),
        r10b.get("issue_judge_comparison", ""),
        *list((r10b.get("issue_judge_paths") or {}).values()),
        r10.get("claim_output", ""),
        r10.get("log_path", ""),
    ):
        if path_str:
            p = Path(path_str)
            print(f"    {'✓' if p.exists() else '…'}  {p}")
    if r10.get("spawned"):
        print(f"\n  verifier는 백그라운드에서 계속 실행 중입니다. (PID {r10.get('pid')})")


def _print_generated_files(runtime: PipelineRuntime, preprocess_result: dict, graph_result: dict, verifier_result: dict) -> None:
    output_dir = runtime.output_dir
    stem = runtime.stem
    audio_result = preprocess_result.get("audio_result", {})
    graph_triples = graph_result.get("graph_triples_result", {})
    graphrag = graph_result.get("graphrag_result", {})
    metadata = graph_result.get("metadata_result", {})
    analyzer = preprocess_result.get("analyzer_input_result", {})
    verifier = verifier_result.get("verifier_result", {})
    claims = verifier_result.get("claims_result", {})
    issue_judge = verifier_result.get("issue_judge_result", {})

    print("\n  생성된 파일:")
    output_files = [
        audio_result.get("segments_path", ""),
        audio_result.get("silences_path", ""),
        audio_result.get("emphasis_path", ""),
        preprocess_result.get("annotation_path", ""),
        preprocess_result.get("textualized_path", ""),
        preprocess_result.get("classified_result", {}).get("classified_path", ""),
        preprocess_result.get("by_scene_result", {}).get("by_scene_path", ""),
        preprocess_result.get("fusion_result", {}).get("fused_path", ""),
        graph_triples.get("triples_parquet", ""),
        graph_triples.get("nodes_parquet", ""),
        graph_triples.get("edges_parquet", ""),
        str(output_dir / f"{stem}_chunks_lance.parquet"),
        graphrag.get("input_path", ""),
        graphrag.get("entities_parquet", ""),
        graphrag.get("relationships_parquet", ""),
        metadata.get("metadata_path", ""),
        str(Path(getattr(runtime.args, "recommender_db_dir", DEFAULT_RECOMMENDER_DB_DIR))),
        analyzer.get("merged_clean_path", str(output_dir / f"{stem}_analyzer" / f"{stem}_merged_clean.json")),
        claims.get("claims_jsonl", ""),
        claims.get("claims_json", ""),
        issue_judge.get("issue_judge_summary", ""),
        issue_judge.get("issue_judge_comparison", ""),
        *list((issue_judge.get("issue_judge_paths") or {}).values()),
        verifier.get("log_path", ""),
    ]
    for analyzer_path in (verifier.get("claim_output", ""), verifier.get("claim_report", "")):
        if analyzer_path and Path(analyzer_path).exists():
            output_files.append(analyzer_path)
    for path_str in output_files:
        if not path_str:
            continue
        p = Path(path_str)
        print(f"    {'✓' if p.exists() else '✗'}  {p}")
    if verifier.get("spawned"):
        print(f"\n  verifier는 백그라운드에서 계속 실행 중입니다. (PID {verifier.get('pid')})")
        print(f"  로그 파일: {verifier.get('log_path')}")
        print()


def _finish_pipeline_run(runtime: PipelineRuntime) -> None:
    try:
        from .cost_report import write_report

        cost_report_path = write_report(
            stem=runtime.stem,
            output_dir=runtime.output_dir,
            timings=runtime.timings,
            analyzer_output_path=runtime.output_dir / f"{runtime.stem}_analyzer" / f"{runtime.stem}_verification_final.json",
        )
        print(f"\n  ✓ 비용 리포트 저장: {cost_report_path}")
    except Exception as e:
        print(f"\n  ⚠️ 비용 리포트 저장 실패: {e}")

    total_elapsed = time.time() - runtime.total_start
    try:
        runtime.write_timings("finished", status="finished")
    except Exception as e:
        print(f"\n  ⚠️ 타이밍 파일 저장 실패: {e}")
    print("\n" + "═" * 70)
    print("  단계별 소요 시간 (현재까지)")
    print("═" * 70)
    for stage, t in runtime.timings.items():
        label = "  (스킵)" if t == 0.0 else f"  {t:>7.1f}초"
        print(f"    {stage:<30} {label}")
    print(f"\n    {'총 소요 시간':<30}  {total_elapsed:>7.1f}초")


def run_direct_upload_workflow(args, progress_callback=None) -> dict:
    """Direct workflow: shared preprocess, then graph/upload artifacts."""
    runtime = _create_pipeline_runtime(args, progress_callback, "Direct upload pipeline")
    preprocess_result: dict = {}
    graph_result: dict = {}
    try:
        preprocess_result = run_preprocess_pipeline(args, runtime, should_build_analyzer_input=False)
        runtime.timings["V2 start_verifier_background — verifier 백그라운드 시작"] = 0.0
        graph_result = run_graph_pipeline(args, runtime, preprocess_result)
        _print_generated_files(runtime, preprocess_result, graph_result, {})
        return {"preprocess": preprocess_result, "graph": graph_result}
    except Exception as e:
        print(f"\n❌ 파이프라인 오류: {e}")
        raise
    finally:
        _finish_pipeline_run(runtime)


def run_verified_upload_workflow(args, progress_callback=None) -> dict:
    """Verified workflow: shared preprocess, then synchronous verifier; graph waits for approval."""
    runtime = _create_pipeline_runtime(args, progress_callback, "Verified upload pipeline")
    preprocess_result: dict = {}
    verifier_result: dict = {}
    try:
        preprocess_result = run_preprocess_pipeline(args, runtime, should_build_analyzer_input=True)
        verifier_result = run_verifier_pipeline(args, runtime, preprocess_result, background=False)
        _print_generated_files(runtime, preprocess_result, {}, verifier_result)
        return {"preprocess": preprocess_result, "verifier": verifier_result}
    except Exception as e:
        print(f"\n❌ 파이프라인 오류: {e}")
        raise
    finally:
        _finish_pipeline_run(runtime)


def run_pipeline(args, progress_callback=None):
    runtime = _create_pipeline_runtime(args, progress_callback)
    preprocess_result: dict = {}
    graph_result: dict = {}
    verifier_result: dict = {}
    try:
        should_stop_after_verifier = (
            getattr(args, "stop_after_claim_extract", False)
            or getattr(args, "stop_after_issue_judge", False)
            or getattr(args, "stop_after_verifier_start", False)
        )
        should_run_verifier = should_stop_after_verifier or not getattr(args, "skip_analyzer", False)
        preprocess_result = run_preprocess_pipeline(
            args,
            runtime,
            should_build_analyzer_input=should_run_verifier,
        )

        if should_run_verifier:
            verifier_result = run_verifier_pipeline(
                args,
                runtime,
                preprocess_result,
                background=not (
                    getattr(args, "stop_after_claim_extract", False)
                    or getattr(args, "stop_after_issue_judge", False)
                ),
            )
        else:
            runtime.timings["V2 start_verifier_background — verifier 백그라운드 시작"] = 0.0

        if should_stop_after_verifier:
            _print_stop_after_summary(args, runtime, verifier_result)
            return

        graph_result = run_graph_pipeline(args, runtime, preprocess_result)
        _print_generated_files(runtime, preprocess_result, graph_result, verifier_result)

    except Exception as e:
        print(f"\n❌ 파이프라인 오류: {e}")
        raise
    finally:
        _finish_pipeline_run(runtime)


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────

def get_parser():
    
    from .config import DEFAULT_SLIDES_DIR, DEFAULT_OUTPUT_DIR

    parser = argparse.ArgumentParser(
        description="강의 영상 분석 통합 파이프라인",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python main.py --input input/lecture.mp4
  python main.py --input input/lecture.mp4 --output output/ --slides output_slides/
  python main.py --input input/lecture.mp4 --skip-extract
  python main.py --input input/lecture.mp4 --debug --masks
  python main.py --input input/lecture.mp4 --force
  python main.py --input input/lecture.mp4 --skip-lance-index
  python main.py --input input/lecture.mp4 --skip-graphrag-index
  python main.py --input input/lecture.mp4 --skip-neo4j
        """,
    )
    parser.add_argument("--input",  "-i", default="input/lecture.mp4", help="입력 강의 영상 경로 (.mp4)")
    parser.add_argument("--slides", "-s", default=str(DEFAULT_SLIDES_DIR),
                        help=f"슬라이드 프레임 저장 디렉토리 (default: {DEFAULT_SLIDES_DIR})")
    parser.add_argument("--output", "-o", default=str(DEFAULT_OUTPUT_DIR),
                        help=f"분석 결과 저장 디렉토리 (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--skip-extract", action="store_true",
                        help="P1A extract_slides 건너뜀 (이미 슬라이드가 추출된 경우)")
    parser.add_argument("--force", action="store_true",
                        help="출력 파일이 있어도 모든 단계 강제 재실행")
    parser.add_argument("--retries", type=int, default=3,
                        help="Gemini API 재시도 횟수 (default: 3)")
    parser.add_argument("--debug", action="store_true", help="P1 extract_media 디버그 로그 출력")
    parser.add_argument(
        "--slide-decode-backend",
        choices=["opencv", "ffmpeg-cuda", "ffmpeg-videotoolbox", "auto"],
        default=os.getenv("GRAPHLEC_SLIDE_DECODE_BACKEND", "auto"),
        help="P1A extract_slides 프레임 디코드 백엔드 (default: auto)",
    )
    parser.add_argument(
        "--slide-extract-workers",
        type=int,
        default=int(os.getenv("GRAPHLEC_SLIDE_EXTRACT_WORKERS", "0")),
        help="P1A extract_slides 시간 청크 병렬 추출 worker 수 (기본: 0, chunk 개수만큼 자동)",
    )
    parser.add_argument("--masks", action="store_true", help="P3A analyze_annotation diff 마스크 이미지 저장")
    parser.add_argument(
        "--per-annot-mode",
        dest="per_annot_mode",
        action="store_true",
        help="P3A analyze_annotation을 annot별 개별 호출 방식으로 실행",
    )
    parser.add_argument("--legacy-per-annot", dest="per_annot_mode", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--no-batch", dest="per_annot_mode", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--skip-graph-triples", action="store_true",
                        help="G1 graph_triples 그래프 Parquet(triples/nodes/edges) 생성 스킵")
    parser.add_argument(
        "--skip-neo4j",
        action="store_true",
        help="호환성 유지용 옵션입니다. Neo4j 적재는 강의 시청 화면 진입 시 수행됩니다.",
    )
    parser.add_argument("--skip-lance-index", action="store_true",
                        help="G2 lance_index LanceDB+Parquet 인덱스 스킵")
    parser.add_argument(
        "--lance-root",
        default=None,
        help="LanceDB 저장 경로 (기본: 환경변수 GRAPHLEC_LANCE_ROOT 또는 data/lancedb)",
    )
    parser.add_argument("--skip-graphrag-index", action="store_true",
                        help="G3 graphrag_index GraphRAG parquet 인덱스 생성 스킵")
    parser.add_argument(
        "--graphrag-root",
        default=None,
        help="GraphRAG workspace root override. 기본값은 output_dir/graphrag",
    )
    parser.add_argument(
        "--graphrag-method",
        default=os.getenv("GRAPHLEC_GRAPHRAG_METHOD", "standard"),
        choices=["standard", "fast", "standard-update", "fast-update"],
        help="GraphRAG index method (default: standard)",
    )
    parser.add_argument("--skip-metadata", action="store_true",
                        help="G4 metadata 메타데이터 생성 스킵")
    parser.add_argument("--skip-analyzer", action="store_true",
                        help="V2 run_verifier verifier 실행 스킵")
    parser.add_argument(
        "--stop-after-verifier-start",
        action="store_true",
        help="P3B process_audio context 기반 analyzer 입력 생성 및 verifier 시작 후 종료",
    )
    parser.add_argument(
        "--stop-after-claim-extract",
        action="store_true",
        help="P3B process_audio context 기반 analyzer 입력 생성 및 claim 추출 후 종료",
    )
    parser.add_argument(
        "--stop-after-issue-judge",
        action="store_true",
        help="P3B process_audio context 기반 analyzer 입력 생성, claim 추출, 1차 issue judge 후 종료",
    )
    parser.add_argument(
        "--issue-judge-min-confidence",
        type=float,
        default=None,
        help="1차 issue judge 후보 저장 confidence 기준. 기본값은 환경변수 또는 0.8",
    )
    parser.add_argument("--skip-recommender-index", action="store_true",
                        help="G5 recommender_index 추천 인덱스 생성(build_index) 스킵")
    parser.add_argument("--metadata-dir", dest="metadata_dir", default=DEFAULT_RECOMMENDER_METADATA_DIR,
                        help=f"메타데이터 저장 디렉토리 (default: {DEFAULT_RECOMMENDER_METADATA_DIR})")
    parser.add_argument("--recommender-db-dir", dest="recommender_db_dir", default=DEFAULT_RECOMMENDER_DB_DIR,
                        help=f"추천 인덱스 LanceDB 경로 (default: {DEFAULT_RECOMMENDER_DB_DIR})")
    parser.add_argument("--title",      default="", help="강의명 (미입력 시 Gemini 자동 생성)")
    parser.add_argument("--instructor", default="", help="교수자명")
    parser.add_argument("--domain",     default="", help="도메인 (미입력 시 Gemini 자동 추론)")
    parser.add_argument("--uploaded-at", dest="uploaded_at", default=None,
                        help="강의 업로드 시각 ISO 문자열")
    
    return parser

def main():
    args = get_parser().parse_args()

    if not args.skip_extract and not Path(args.input).exists():
        print(f"❌ 입력 영상 없음: {args.input}")
        sys.exit(1)

    run_pipeline(args)


if __name__ == "__main__":
    main()
