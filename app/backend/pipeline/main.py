"""
main.py
=======
강의 영상 분석 통합 파이프라인

실행 흐름:
  [병렬] Stage 1A: slide_extractor     — 슬라이드 프레임 추출
         Stage 1B: audio_analyzer      — 오디오 품질 분석
  [직렬] Stage 2 : slide_textualizer   — 슬라이드 텍스트 + 강조 추출
  [병렬] Stage 3A: annotation_analyzer — 필기 강조 분석
         Stage 3B: 오디오 파이프라인
                     transcriber       — 슬라이드별 전사 (metadata 필요)
                     text_processor    — 2-pass 교정 (make_merged 기반) + 침묵/지시어 추출
                     emphasis          — 오디오 강조 감지
  [병렬] Stage 4A: slide_classifier    — 슬라이드 역할 분류
         Stage 4B: by_slide 구조 저장  — (3B 결과 기반)
  [직렬] Stage 5 : fusion              — 최종 통합
  [직렬] Stage 6 : 그래프 Parquet       — json_to_graph_triples
  [직렬] Stage 7 : lance_ingest          — fused → Parquet + LanceDB (Gemini 임베딩, stem 필터)
  [직렬] Stage 7B: GraphRAG index        — fused → GraphRAG parquet workspace

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
from typing import Any, Optional

import librosa
from .deictics import (
    classify_ambiguous_deictics_with_llm,
    extract_deictics_from_segments,
)
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
    return {
        "score": ann.get("emphasis_score"),
        "methods": ann.get("emphasis_methods", []),
        "keywords": ann.get("emphasis_keywords", []),
        "keywords_by_method": ann.get("emphasis_keywords_by_method", {}),
        "detection_count": ann.get("detection_count", 0),
    }


def _auto_register_lecture(stem: str) -> None:
    """
    파이프라인 성공 후 web Lecture 테이블에 stem을 자동 등록한다.
    - 이미 있으면 유지
    - 없으면 title=stem 으로 생성
    """
    try:
        import os
        from dotenv import load_dotenv

        web_dir = REPO_ROOT / "web"
        if not web_dir.exists():
            print("\n  ⚠️ Lecture 자동 등록 스킵: web 디렉터리를 찾을 수 없습니다.")
            return

        load_dotenv(override=False)
        # 로컬 테스트는 SQLite 단일 DB로 통일한다.
        os.environ["USE_SQLITE"] = "1"

        if str(web_dir) not in sys.path:
            sys.path.insert(0, str(web_dir))

        os.environ.setdefault("DJANGO_SETTINGS_MODULE", "graphlec_site.settings")

        import django  # noqa: PLC0415

        django.setup()

        from lectures.models import Lecture  # noqa: PLC0415

        lec, created = Lecture.objects.get_or_create(
            stem=stem,
            defaults={"title": stem},
        )
        status = "생성" if created else "기존 유지"
        print(f"\n  ✓ Lecture 자동 등록: {lec.stem} ({status}, db=sqlite)")
        print("─" * 70)
    except Exception as e:
        print(f"\n  ⚠️ Lecture 자동 등록 실패(분석 결과는 정상 생성): {e}")
        print("─" * 70)


# ──────────────────────────────────────────────────────────────
# 전사 헬퍼
# ──────────────────────────────────────────────────────────────

def _transcribe_by_slide(
    video_path: str,
    duration: float,
    meta_path: Optional[str],
    slide_ranges: list[dict],
    output_dir: Path,
) -> dict:
    """
    슬라이드별 전사 (metadata 있을 때) 또는 전체 전사 (fallback).

    Returns:
        {
            "segments": [{"start","end","text","words","slide_index"?}, ...],
            "silences": [{"start","end","duration"}, ...]   # 영상 절대 시간
        }
    """
    from .transcriber import transcribe_video, transcribe_range

    if not meta_path or not slide_ranges:
        print("  ℹ️ metadata 없음 → 전체 전사 방식 사용")
        return transcribe_video(video_path, duration, output_dir=output_dir)

    all_segments: list[dict] = []
    all_silences: list[dict] = []
    for r in slide_ranges:
        sidx = r["slide_index"]
        start_sec = float(r["start_sec"])
        end_sec = float(r["end_sec"])
        print(f"    ▶ 슬라이드 {sidx}: {start_sec:.1f}s ~ {end_sec:.1f}s 전사...")
        result = transcribe_range(video_path, start_sec, end_sec, output_dir)
        for seg in result.get("segments", []):
            s = seg.copy()
            s["slide_index"] = sidx
            all_segments.append(s)
        all_silences.extend(result.get("silences", []))

    all_segments.sort(key=lambda s: (s.get("start", 0.0), s.get("end", 0.0)))
    # 슬라이드 범위가 겹치는 경우 silences가 중복될 수 있음 → dedup (start, end 기준)
    seen_sil: set = set()
    deduped_silences: list[dict] = []
    for sil in sorted(all_silences, key=lambda x: (x["start"], x["end"])):
        key = (round(sil["start"], 3), round(sil["end"], 3))
        if key in seen_sil:
            continue
        seen_sil.add(key)
        deduped_silences.append(sil)

    return {"segments": all_segments, "silences": deduped_silences}


# ──────────────────────────────────────────────────────────────
# 파이프라인 스테이지
# ──────────────────────────────────────────────────────────────

def stage1_extract(args, slides_dir: Path, output_dir: Path) -> dict:
    from .slide_extractor import extract_slides

    stem = Path(args.input).stem
    meta_path = output_dir / f"{stem}_metadata.json"

    if _is_done(meta_path, "Stage 1A 슬라이드 추출", args.force):
        return {"meta_path": str(meta_path), "elapsed": 0.0}

    _banner("Stage 1  —  슬라이드 추출  (slide_extractor)")
    t0 = time.time()
    metadata = extract_slides(
        input_path=args.input,
        output_dir=str(slides_dir),
        debug=args.debug,
    )
    elapsed = time.time() - t0

    _save_json(meta_path, metadata)

    slide_count = len({m["slide_index"] for m in metadata})
    _done(f"슬라이드 {slide_count}개, 프레임 {len(metadata)}개 추출", elapsed)
    return {"meta_path": str(meta_path), "elapsed": elapsed}


def stage1b_audio_analyze(args, output_dir: Path) -> dict:
    """Stage 1B: 오디오 품질 분석 (slide_extractor와 병렬)"""
    from .audio_analyzer import extract_audio_from_video, analyze_audio_features, evaluate_audio_quality
    from .utils import get_video_duration

    stem = Path(args.input).stem
    audio_quality_path = output_dir / f"{stem}_audio_quality.json"

    if _is_done(audio_quality_path, "Stage 1B 오디오 품질 분석", args.force):
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

    _banner("Stage 1B  —  오디오 품질 분석  (audio_analyzer)")
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


def stage2_textualize(args, slides_dir: Path, output_dir: Path) -> dict:
    from .slide_textualizer import TextualizationPipeline, Config as TextConfig

    stem = Path(args.input).stem
    textualized_path = output_dir / f"{stem}_slide_textualized.json"

    if _is_done(textualized_path, "Stage 2 슬라이드 텍스트화", args.force):
        return {"textualized_path": str(textualized_path), "elapsed": 0.0}

    _banner("Stage 2  —  슬라이드 텍스트화  (slide_textualizer)")
    t0 = time.time()
    text_config = TextConfig(
        slides_dir=slides_dir,
        output_dir=output_dir,
        output_filename=textualized_path.name,
        max_retries=args.retries,
    )
    text_result = TextualizationPipeline(text_config).run()
    elapsed = time.time() - t0

    _done(f"슬라이드 {text_result['metadata']['total_slides']}개 텍스트화", elapsed)
    return {"textualized_path": str(textualized_path), "elapsed": elapsed}


def stage3a_annotation(args, slides_dir: Path, output_dir: Path) -> dict:
    from .annotation_analyzer import analyze_all

    stem = Path(args.input).stem
    annotation_path = output_dir / f"{stem}_annotation.json"

    if _is_done(annotation_path, "Stage 3A annotation", args.force):
        return {"annotation_path": str(annotation_path), "elapsed": 0.0}

    _banner("Stage 3A  —  필기 강조 분석  (annotation_analyzer)")
    t0 = time.time()
    annot_results = analyze_all(
        slides_dir=str(slides_dir),
        output_path=str(annotation_path),
        save_masks=args.masks,
    )
    elapsed = time.time() - t0

    n_annots = sum(len(r.get("annotations", [])) for r in annot_results)
    _done(f"{len(annot_results)}개 분석, 총 {n_annots}개 강조 추출", elapsed)
    return {"annotation_path": str(annotation_path), "elapsed": elapsed}


def stage3b_audio(args, meta_path: str, textualized_path: str, duration: float, output_dir: Path) -> dict:
    from .text_processor import correct_segments_two_pass
    from .segment_grouper import (
        load_slide_ranges,
        group_segments_by_context,
        group_segments_by_slide_and_context,
        expand_group_annotations_to_segments,
    )
    from .audio_analyzer import extract_audio_from_video
    from .emphasis_audio import detect_emphasis_by_std
    from .emphasis_keyword import (
        detect_emphasis_by_keywords_weighted,
        detect_emphasis_by_topic_keyword_repetition,
        get_topic_keywords_filtered_v2,
    )
    from .emphasis_combiner import combine_emphasis_simple

    stem = Path(args.input).stem
    segments_path = output_dir / f"{stem}_segments.json"
    silences_path = output_dir / f"{stem}_silences.json"
    emphasis_path = output_dir / f"{stem}_emphasis.json"
    by_slide_path = output_dir / f"{stem}_by_slide.json"

    # 세그먼트만 있고 by_slide가 없으면(파일 삭제·불완전 실행) 스킵하면 Stage 4B·5가 깨짐 → 3B 전체 재실행
    seg_ok = (
        not args.force
        and segments_path.exists()
        and segments_path.stat().st_size > 0
    )
    by_slide_ok = by_slide_path.exists() and by_slide_path.stat().st_size > 0
    if seg_ok and by_slide_ok:
        print(f"\n  ⏭  Stage 3B 오디오 파이프라인 — 출력 파일 존재, 스킵")
        print(f"     {segments_path}")
        print(f"     {by_slide_path}")
        print("─" * 70)
        # in-memory 데이터를 저장된 파일에서 복원
        slides_structure = None
        if by_slide_path.exists():
            try:
                with open(by_slide_path) as f:
                    slides_structure = json.load(f).get("slides")
            except Exception:
                pass

        annotated_segments: list[dict] = []
        try:
            with open(segments_path) as f:
                annotated_segments = json.load(f).get("segments", [])
        except Exception:
            pass

        slide_ranges = load_slide_ranges(meta_path, duration) if meta_path and Path(meta_path).is_file() else []

        return {
            "segments_path": str(segments_path),
            "silences_path": str(silences_path),
            "deictics_path": str(output_dir / f"{stem}_deictics.json"),
            "emphasis_path": str(emphasis_path),
            "annotated_segments": annotated_segments,
            "annotated_groups": [],
            "slides_structure": slides_structure,
            "slide_ranges": slide_ranges,
            "duration": duration,
        }

    if seg_ok and not by_slide_ok:
        print(
            f"\n  ⚠️  {by_slide_path.name} 없음 — 세그먼트만 있는 불완전 상태입니다. "
            "Stage 3B 전체를 다시 실행합니다."
        )
        print("─" * 70)

    video_path = args.input

    _banner("Stage 3B  —  오디오 파이프라인")

    # 슬라이드 텍스트화 데이터 로드
    textualized_data: dict = {"slides": []}
    if textualized_path and Path(textualized_path).is_file():
        with open(textualized_path, "r", encoding="utf-8") as f:
            textualized_data = json.load(f)

    # metadata 로드 (슬라이드 occurrence 정보)
    metadata: list[dict] = []
    if meta_path and Path(meta_path).is_file():
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    # [3B-1] 슬라이드별 전사
    print("  [3B-1] 슬라이드별 전사...")
    t0 = time.time()
    slide_ranges = load_slide_ranges(meta_path, duration) if meta_path and Path(meta_path).is_file() else []
    transcribe_result = _transcribe_by_slide(video_path, duration, meta_path, slide_ranges, output_dir)
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

    # [3B-4] 지시어 추출
    print("  [3B-4] 지시어 추출...")
    deictics_report = extract_deictics_from_segments(segments, text_field="text_corrected")
    deictics_report["video_path"] = video_path
    _save_json(output_dir / f"{stem}_deictics.json", deictics_report)
    ambiguous_report = classify_ambiguous_deictics_with_llm(
        segments_clean, deictics_report, threshold=0.6, context_window=2
    )
    ambiguous_report["video_path"] = video_path
    _save_json(output_dir / f"{stem}_deictics_ambiguous.json", ambiguous_report)
    print(f"    ✓ 지시어 {deictics_report['deictic_total_count']}개, 애매 {ambiguous_report['ambiguous_count']}개")

    # [3B-5] 오디오 강조 감지
    print("  [3B-5] 오디오 강조 감지...")
    t0 = time.time()
    audio_path_temp = str(output_dir / "temp_analysis_audio.wav")
    extract_audio_from_video(video_path, audio_path_temp)
    annotated_segments: list[dict] = []
    annotated_groups: list[dict] = []
    slides_structure = None
    emphasis_sections: list[dict] = []
    topic_kw_set: set[str] = set()
    try:
        y, sr = librosa.load(audio_path_temp, sr=16000)
        if slide_ranges:
            groups, slides_structure = group_segments_by_slide_and_context(
                segments_clean, slide_ranges, duration, use_pause_sentence=False, use_llm_merge=True
            )
        else:
            groups = group_segments_by_context(segments_clean)
            slides_structure = None

        topic_kw_set = get_topic_keywords_filtered_v2(
            groups, min_freq=5, max_keywords=20, max_segment_ratio=1.0,
            min_keyword_len=2, candidate_pool_size=80, use_llm_filter=True,
        )
        audio_emphasis = detect_emphasis_by_std(y, sr, groups)
        keyword_emphasis = detect_emphasis_by_keywords_weighted(groups)
        topic_emphasis = detect_emphasis_by_topic_keyword_repetition(
            groups, window=2, min_keyword_len=2, max_segment_ratio=1.0,
            min_freq=5, max_keywords=20, use_llm_filter=False, min_keyword_count=1,
            _topic_keywords_override=topic_kw_set,
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
        "deictics_path": str(output_dir / f"{stem}_deictics.json"),
        "emphasis_path": str(emphasis_path),
        "annotated_segments": annotated_segments,
        "annotated_groups": annotated_groups,
        "slides_structure": slides_structure,
        "slide_ranges": slide_ranges,
        "duration": duration,
    }


def stage4a_classify(
    args, textualized_path: str, meta_path: str, silences_path: str, output_dir: Path
) -> dict:
    from .slide_classifier import classify_slides

    stem = Path(args.input).stem
    classified_path = output_dir / f"{stem}_slide_classified.json"

    if _is_done(classified_path, "Stage 4A 슬라이드 분류", args.force):
        return {"classified_path": str(classified_path), "elapsed": 0.0}

    _banner("Stage 4A  —  슬라이드 분류  (slide_classifier)")
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


def stage4b_save_by_slide(args, audio_result: dict, output_dir: Path) -> dict:
    stem = Path(args.input).stem
    by_slide_path = output_dir / f"{stem}_by_slide.json"

    slides_structure = audio_result.get("slides_structure")
    annotated_segments = audio_result.get("annotated_segments", [])
    annotated_groups = audio_result.get("annotated_groups", [])
    slide_ranges = audio_result.get("slide_ranges", [])
    duration = audio_result.get("duration", 0.0)

    if not slides_structure or not slide_ranges:
        if by_slide_path.exists() and by_slide_path.stat().st_size > 0:
            return {
                "by_slide_path": str(by_slide_path),
                "elapsed": 0.0,
            }
        raise FileNotFoundError(
            f"{by_slide_path} 을(를) 만들 수 없습니다(slides_structure 또는 slide_ranges 없음). "
            "세그먼트 파일만 있고 by_slide가 비어 있거나 삭제된 경우 "
            "`python -m pipeline.main --input ... --force` 로 Stage 3B 이후를 다시 실행하세요."
        )

    if _is_done(by_slide_path, "Stage 4B by_slide 저장", args.force):
        return {
            "by_slide_path": str(by_slide_path),
            "elapsed": 0.0,
        }

    _banner("Stage 4B  —  by_slide 구조 저장")
    t0 = time.time()

    slides_with_emphasis = json.loads(json.dumps(slides_structure))
    if annotated_groups:
        annot_by_start = {g.get("start"): g for g in annotated_groups}
        for slide in slides_with_emphasis:
            new_contexts = []
            for ctx in slide.get("contexts", []):
                ann = annot_by_start.get(ctx.get("start"))
                ordered_ctx = {
                    "context_index": ctx.get("context_index"),
                    "start": ctx.get("start"),
                    "end": ctx.get("end"),
                    "text": ctx.get("text"),
                }
                if ann and ann.get("emphasis") == "강조":
                    ordered_ctx["emphasis"] = {
                        "state": "강조",
                        "detected": True,
                        "detail": _format_emphasis_reason(ann),
                    }
                else:
                    ordered_ctx["emphasis"] = {"state": None, "detected": False}
                ordered_ctx["segment_indices"] = ctx.get("segment_indices", [])
                ordered_ctx["segments"] = ctx.get("segments", [])
                new_contexts.append(ordered_ctx)
            slide["contexts"] = new_contexts
    _save_json(by_slide_path, {"slides": slides_with_emphasis})

    elapsed = time.time() - t0
    _done("by_slide 구조 저장", elapsed)
    return {
        "by_slide_path": str(by_slide_path),
        "elapsed": elapsed,
    }


def stage5_fusion(
    args,
    textualized_path: str,
    annotation_path: str,
    audio_result: dict,
    output_dir: Path,
) -> dict:
    from .fusion import Config as FusionConfig, run_fusion

    stem = Path(args.input).stem
    fused_path = output_dir / f"{stem}_fused.json"

    if _is_done(fused_path, "Stage 5 퓨전", args.force):
        return {"fused_path": str(fused_path), "elapsed": 0.0}

    _banner("Stage 5  —  퓨전  (fusion)")
    t0 = time.time()

    cfg = FusionConfig(
        stem=stem,
        output_dir=output_dir,
        slides_dir=Path(args.slides),
        audio_path=output_dir / f"{stem}_by_slide.json",
        classified_path=output_dir / f"{stem}_slide_classified.json",
        annotation_path=Path(annotation_path),
        output_path=fused_path,
    )
    fused_output = run_fusion(cfg)

    # run_fusion 반환 스키마를 메인 파이프라인 출력 형식에 맞게 감싼다.
    _save_json(
        fused_path,
        {
            "video_path": args.input,
            "description": "영상(slide+annotation) + 오디오 퓨전 결과",
            "slide_count": len(fused_output.get("slides", [])),
            "fusion_metadata": fused_output.get("metadata", {}),
            "slides": fused_output.get("slides", []),
        },
    )
    elapsed = time.time() - t0
    _done(f"슬라이드 {len(fused_output.get('slides', []))}개 퓨전", elapsed)
    return {"fused_path": str(fused_path), "elapsed": elapsed}


def stage9_build_analyzer_merged_clean(
    args,
    meta_path: str,
    textualized_path: str,
    segments_path: str,
    output_dir: Path,
    duration: float,
) -> dict:
    from .segment_grouper import load_slide_ranges
    from .text_processor import classify_lecture_domain

    stem = Path(args.input).stem
    merged_clean_path = output_dir / f"{stem}_merged_clean.json"

    if _is_done(merged_clean_path, "Stage 9A analyzer 입력 생성", args.force):
        return {"merged_clean_path": str(merged_clean_path), "elapsed": 0.0}

    _banner("Stage 9A  —  analyzer 입력용 merged_clean 생성")
    t0 = time.time()

    with open(textualized_path, "r", encoding="utf-8") as f:
        textualized = json.load(f)
    with open(segments_path, "r", encoding="utf-8") as f:
        segment_payload = json.load(f)
    segments = segment_payload.get("segments", [])
    slide_ranges = load_slide_ranges(meta_path, duration) if meta_path and Path(meta_path).is_file() else []

    slide_meta_by_no = {}
    for slide in textualized.get("slides", []):
        slide_no = slide.get("slide_number")
        if isinstance(slide_no, int):
            text_parts = []
            if slide.get("t1"):
                text_parts.append(str(slide.get("t1")))
            if slide.get("t1_structure"):
                text_parts.append(str(slide.get("t1_structure")))
            slide_meta_by_no[slide_no] = {
                "title": str(slide.get("title", "") or ""),
                "slide_text": "\n".join(part for part in text_parts if part),
            }

    segs_by_slide: dict[int, list[dict]] = {}
    for seg in segments:
        slide_no = seg.get("slide_index")
        if isinstance(slide_no, int):
            seg_copy = {
                "start": float(seg.get("start", 0.0) or 0.0),
                "end": float(seg.get("end", seg.get("start", 0.0)) or 0.0),
                "text": str(seg.get("text", "") or "").strip(),
            }
            segs_by_slide.setdefault(slide_no, []).append(seg_copy)

    slide_titles = [slide_meta_by_no.get(slide_no, {}).get("title", "") for slide_no in sorted(slide_meta_by_no)]
    transcript_sample = " ".join(str(seg.get("text", "") or "") for seg in segments[:30])
    domain_info = classify_lecture_domain(slide_titles, transcript_sample)

    slides = []
    for slide_range in slide_ranges:
        slide_no = int(slide_range["slide_index"])
        start_sec = float(slide_range["start_sec"])
        end_sec = float(slide_range["end_sec"])
        slide_meta = slide_meta_by_no.get(slide_no, {})
        transcript_segments = sorted(segs_by_slide.get(slide_no, []), key=lambda item: item.get("start", 0.0))
        slide_duration = round(end_sec - start_sec, 1)
        slides.append({
            "slide_number": slide_no,
            "title": slide_meta.get("title", ""),
            "time_range": f"{_fmt_ts(start_sec)} ~ {_fmt_ts(end_sec)}",
            "time_range_seconds": [start_sec, end_sec],
            "total_duration": slide_duration,
            "occurrences": [{
                "start_sec": start_sec,
                "end_sec": end_sec,
                "duration": slide_duration,
                "is_dup": False,
            }],
            "slide_text": slide_meta.get("slide_text", ""),
            "transcript_segments": transcript_segments,
            "transcript": " ".join(str(seg.get("text", "") or "") for seg in transcript_segments).strip(),
            "segment_count": len(transcript_segments),
        })

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
        "total_transcript_segments": len(segments),
        "total_duration_formatted": _fmt_ts(total_duration),
        "slides": slides,
    }
    _save_json(merged_clean_path, result)

    elapsed = time.time() - t0
    _done("analyzer 입력용 merged_clean 생성", elapsed)
    return {"merged_clean_path": str(merged_clean_path), "elapsed": elapsed}


def stage10_run_analyzers(args, merged_clean_path: str, output_dir: Path) -> dict:
    from .analyzer.run_all import run_all_analyzers

    stem = Path(args.input).stem
    analyzer_dir = output_dir / f"{stem}_analyzer"
    claim_output_path = analyzer_dir / f"{stem}_content_verification.json"
    claim_report_path = analyzer_dir / f"{stem}_content_verification_report.txt"
    cross_model = os.getenv("CROSS_VERIFY_MODEL", "").strip()
    use_cross = bool(cross_model and os.getenv("OPENAI_API_KEY"))

    if not args.force and claim_output_path.exists() and (use_cross or claim_report_path.exists()):
        print(f"\n  ⏭  Stage 10 verifier 실행 — 출력 파일 존재, 스킵")
        print(f"     {claim_output_path}")
        print("─" * 70)
        return {
            "claim_output": str(claim_output_path),
            "claim_report": str(claim_report_path) if claim_report_path.exists() else "",
            "elapsed": 0.0,
        }

    _banner("Stage 10  —  verifier 실행")
    t0 = time.time()
    result = run_all_analyzers(
        merged_clean_path,
        output_dir=str(analyzer_dir),
    )
    elapsed = time.time() - t0
    _done("verifier 실행", elapsed)
    return {"claim_output": str(claim_output_path), "elapsed": elapsed, **result}


def stage10_spawn_analyzers_subprocess(args, merged_clean_path: str, output_dir: Path) -> dict:
    stem = Path(args.input).stem
    analyzer_dir = output_dir / f"{stem}_analyzer"
    analyzer_dir.mkdir(parents=True, exist_ok=True)
    claim_output_path = analyzer_dir / f"{stem}_content_verification.json"
    claim_report_path = analyzer_dir / f"{stem}_content_verification_report.txt"
    analyzer_log_path = analyzer_dir / f"{stem}_analyzer.log"
    cross_model = os.getenv("CROSS_VERIFY_MODEL", "").strip()
    use_cross = bool(cross_model and os.getenv("OPENAI_API_KEY"))

    if not args.force and claim_output_path.exists() and (use_cross or claim_report_path.exists()):
        print(f"\n  ⏭  Stage 10 verifier 실행 — 출력 파일 존재, 스킵")
        print(f"     {claim_output_path}")
        print("─" * 70)
        return {
            "claim_output": str(claim_output_path),
            "claim_report": str(claim_report_path) if claim_report_path.exists() else "",
            "log_path": str(analyzer_log_path),
            "elapsed": 0.0,
            "pid": None,
            "spawned": False,
        }

    _banner("Stage 10  —  verifier 백그라운드 실행")
    t0 = time.time()
    pkg_root = resolve_pipeline_package_root()
    cmd = [
        sys.executable,
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
        log_fp.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(pkg_root),
            stdin=subprocess.DEVNULL,
            stdout=log_fp,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    elapsed = time.time() - t0
    print(f"\n  ✓ verifier 백그라운드 시작  ({elapsed:.1f}초)")
    print(f"     PID : {proc.pid}")
    print(f"     로그: {analyzer_log_path}")
    print("─" * 70)
    return {
        "claim_output": str(claim_output_path),
        "claim_report": str(claim_report_path),
        "log_path": str(analyzer_log_path),
        "elapsed": elapsed,
        "pid": proc.pid,
        "spawned": True,
    }


def stage6_graph_triples(args, output_dir: Path, slides_dir: Path) -> dict:
    from .json_to_graph_triples import Config as TripleConfig, GraphPipeline

    stem = Path(args.input).stem
    triples_parquet = output_dir / f"{stem}_graph_triples.parquet"
    nodes_parquet = output_dir / f"{stem}_nodes.parquet"
    edges_parquet = output_dir / f"{stem}_edges.parquet"

    if _is_done(triples_parquet, "Stage 6 그래프 트리플 생성", args.force):
        return {
            "triples_parquet": str(triples_parquet),
            "nodes_parquet": str(nodes_parquet),
            "edges_parquet": str(edges_parquet),
            "elapsed": 0.0,
        }

    _banner("Stage 6  —  그래프 트리플 생성  (json_to_graph_triples)")
    t0 = time.time()

    cfg = TripleConfig(
        stem=stem,
        output_dir=output_dir,
        slides_dir=slides_dir,
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


def stage7_lance_index(args, output_dir: Path, slides_dir: Path) -> dict:
    """fused.json → 청크 임베딩 → Parquet + LanceDB (단일 테이블, stem 필터)."""
    from .lance_ingest import default_lance_root, ingest_stem_to_lance

    stem = Path(args.input).stem
    from .config import output_paths

    paths = output_paths(stem, output_dir, slides_dir)
    fused_path = paths["fused"]
    lance_root = Path(args.lance_root) if getattr(args, "lance_root", None) else default_lance_root()
    parquet_path = output_dir / f"{stem}_chunks_lance.parquet"

    if _is_done(parquet_path, "Stage 7 Lance 인덱스", args.force):
        return {"elapsed": 0.0, "parquet_path": str(parquet_path), "skipped": True}

    if not fused_path.exists():
        raise FileNotFoundError(f"Stage 7: fused 파일 없음 — Stage 5 퓨전이 필요합니다: {fused_path}")

    _banner("Stage 7  —  LanceDB 인덱스  (Gemini 임베딩 + lance_ingest)")
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


def stage7b_graphrag_index(args, output_dir: Path) -> dict:
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

    if (
        not args.force
        and entities_path.exists()
        and entities_path.stat().st_size > 0
        and relationships_path.exists()
        and relationships_path.stat().st_size > 0
    ):
        print("\n  ⏭  Stage 7B GraphRAG 인덱스 — parquet 출력 파일 존재, 스킵")
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
        raise FileNotFoundError(f"Stage 7B: fused 파일 없음 — Stage 5 퓨전이 필요합니다: {fused_path}")

    graphrag_bin = _find_graphrag_executable()
    if not graphrag_bin:
        raise RuntimeError("Stage 7B: graphrag CLI를 찾을 수 없습니다. requirements 설치 후 다시 실행하세요.")

    api_key = os.getenv("GRAPHRAG_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Stage 7B: GRAPHRAG_API_KEY 또는 OPENAI_API_KEY 환경변수가 필요합니다.")

    _banner("Stage 7B  —  GraphRAG 인덱스  (fused → parquet workspace)")
    t0 = time.time()

    workspace_dir.mkdir(parents=True, exist_ok=True)
    input_dir.mkdir(parents=True, exist_ok=True)
    with open(fused_path, "r", encoding="utf-8") as f:
        fused = json.load(f)
    input_path.write_text(fused_to_graphrag_text(fused, stem=stem), encoding="utf-8")

    env = os.environ.copy()
    env["GRAPHRAG_API_KEY"] = api_key
    _write_graphrag_env(workspace_dir, api_key)

    settings_path = workspace_dir / "settings.yaml"
    if not settings_path.exists():
        subprocess.run(
            [
                graphrag_bin,
                "init",
                "--root",
                str(workspace_dir),
                "--model",
                os.getenv("GRAPHLEC_GRAPHRAG_MODEL", "gpt-4.1-mini"),
                "--embedding",
                os.getenv("GRAPHLEC_GRAPHRAG_EMBEDDING_MODEL", "text-embedding-3-small"),
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
            getattr(args, "graphrag_method", "standard"),
        ],
        check=True,
        env=env,
    )

    elapsed = time.time() - t0
    _done("GraphRAG 인덱스 생성", elapsed)
    return {
        "elapsed": elapsed,
        "workspace_dir": str(workspace_dir),
        "input_path": str(input_path),
        "entities_parquet": str(entities_path),
        "relationships_parquet": str(relationships_path),
    }


def stage8_generate_metadata(args, output_dir: Path, slides_dir: Path) -> dict:
    """Stage 8: 강의 메타데이터 생성."""
    from .generate_metadata import generate_metadata

    stem         = Path(args.input).stem
    metadata_dir = Path(getattr(args, "metadata_dir", DEFAULT_RECOMMENDER_METADATA_DIR))
    output_path  = metadata_dir / f"{stem}_metadata.json"

    if _is_done(output_path, "Stage 8 메타데이터 생성", args.force):
        return {"metadata_path": str(output_path), "elapsed": 0.0}

    _banner("Stage 8  —  메타데이터 생성  (generate_metadata)")
    t0 = time.time()

    generate_metadata(
        stem          = stem,
        title         = getattr(args, "title", ""),
        instructor_id = getattr(args, "instructor", ""),
        output_dir    = output_dir,
        metadata_dir  = metadata_dir,
    )

    elapsed = time.time() - t0
    _done("메타데이터 생성", elapsed)
    return {"metadata_path": str(output_path), "elapsed": elapsed}


def stage11_build_recommender_index(args) -> dict:
    """Stage 11: 추천용 metadata 임베딩 인덱스 생성 (build_index.py)."""
    recommender_dir = Path(__file__).resolve().parents[1] / "recommender"
    script_path = recommender_dir / "build_index.py"
    metadata_dir = Path(getattr(args, "metadata_dir", DEFAULT_RECOMMENDER_METADATA_DIR))
    db_dir = Path(getattr(args, "recommender_db_dir", DEFAULT_RECOMMENDER_DB_DIR))

    _banner("Stage 11  —  추천 인덱스 생성  (build_index)")
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
# 메인 파이프라인
# ──────────────────────────────────────────────────────────────

def run_pipeline(args, progress_callback=None):
    total_start = time.time()
    timings: dict[str, float] = {}
    
    def notify_stage(stage_key, status):
        if progress_callback:
            try:
                progress_callback(stage_key, status)
            except Exception as e:
                log.warning(f"progress_callback failed for {stage_key}: {e}")

    from .config import output_paths, DEFAULT_SLIDES_DIR, DEFAULT_OUTPUT_DIR

    stem = Path(args.input).stem
    slides_dir = Path(args.slides)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    slides_dir.mkdir(parents=True, exist_ok=True)

    paths = output_paths(stem, output_dir, slides_dir)

    print("\n" + "═" * 70)
    print("  강의 영상 분석 통합 파이프라인")
    print("═" * 70)
    print(f"  입력 영상 : {args.input}")
    print(f"  슬라이드  : {slides_dir}")
    print(f"  출력      : {output_dir}")
    if args.force:
        print("  ⚠️  --force: 모든 단계 강제 재실행")

    try:
        r9: dict = {}
        r10: dict = {}
        r7b: dict = {}

        # ── Stage 1 (병렬 A/B) ──
        _banner("Stage 1  —  병렬 실행 (슬라이드 추출 + 오디오 품질 분석)")
        t_parallel = time.time()
        audio_analyze_result: dict = {}
        
        notify_stage("scene", "run")
        notify_stage("voice", "run")

        if args.skip_extract:
            log.info("Stage 1A 건너뜀 (--skip-extract)")
            meta_path = str(paths["metadata"])
            timings["Stage 1A 슬라이드 추출"] = 0.0
            audio_analyze_result = stage1b_audio_analyze(args, output_dir)
            timings["Stage 1B 오디오 품질 분석"] = audio_analyze_result["elapsed"]
        else:
            with ThreadPoolExecutor(max_workers=2) as executor:
                future_1a = executor.submit(stage1_extract, args, slides_dir, output_dir)
                future_1b = executor.submit(stage1b_audio_analyze, args, output_dir)
                for future in as_completed([future_1a, future_1b]):
                    if future is future_1a:
                        r1 = future.result()
                        meta_path = r1["meta_path"]
                        timings["Stage 1A 슬라이드 추출"] = r1["elapsed"]
                    else:
                        audio_analyze_result = future.result()
                        timings["Stage 1B 오디오 품질 분석"] = audio_analyze_result["elapsed"]

        duration = audio_analyze_result.get("duration", 0.0)
        timings["Stage 1 병렬 총"] = time.time() - t_parallel
        
        notify_stage("scene", "done")
        notify_stage("voice", "done")
        
        print(f"\n  ✓ Stage 1 완료  ({timings['Stage 1 병렬 총']:.1f}초)")
        print("─" * 70)

        # ── Stage 2 (직렬) ──
        r2 = stage2_textualize(args, slides_dir, output_dir)
        textualized_path = r2["textualized_path"]
        timings["Stage 2 슬라이드 텍스트화"] = r2["elapsed"]

        # ── Stage 3 (병렬 A/B) ──
        _banner("Stage 3  —  병렬 실행 (annotation + 오디오)")
        t_parallel = time.time()
        audio_result: dict = {}
        annotation_result: dict = {}
        
        notify_stage("stt", "run")

        with ThreadPoolExecutor(max_workers=2) as executor:
            future_a = executor.submit(stage3a_annotation, args, slides_dir, output_dir)
            future_b = executor.submit(stage3b_audio, args, meta_path, textualized_path, duration, output_dir)
            for future in as_completed([future_a, future_b]):
                if future is future_a:
                    annotation_result = future.result()
                    timings["Stage 3A annotation"] = annotation_result["elapsed"]
                else:
                    audio_result = future.result()
                    segments_path = audio_result.get("segments_path", str(paths["segments"]))
                    r9 = stage9_build_analyzer_merged_clean(
                        args,
                        meta_path=meta_path,
                        textualized_path=textualized_path,
                        segments_path=segments_path,
                        output_dir=output_dir,
                        duration=duration,
                    )
                    timings["Stage 9 analyzer 입력 생성"] = r9["elapsed"]
                    if getattr(args, "skip_analyzer", False):
                        timings["Stage 10 verifier 백그라운드 시작"] = 0.0
                    else:
                        r10 = stage10_spawn_analyzers_subprocess(
                            args,
                            merged_clean_path=r9["merged_clean_path"],
                            output_dir=output_dir,
                        )
                        timings["Stage 10 verifier 백그라운드 시작"] = r10["elapsed"]

        timings["Stage 3 병렬 총"] = time.time() - t_parallel
        
        notify_stage("stt", "done")
        
        print(f"\n  ✓ Stage 3 완료  ({timings['Stage 3 병렬 총']:.1f}초)")
        print("─" * 70)

        # ── Stage 4 (병렬 C/D) ──
        _banner("Stage 4  —  병렬 실행 (classifier + by_slide 저장)")
        t_parallel = time.time()
        classified_result: dict = {}
        by_slide_result: dict = {}

        silences_path = audio_result.get("silences_path", str(paths["silences"]))
        annotation_path = annotation_result.get("annotation_path", str(paths["annotation"]))

        with ThreadPoolExecutor(max_workers=2) as executor:
            future_c = executor.submit(
                stage4a_classify, args, textualized_path, meta_path, silences_path, output_dir
            )
            future_d = executor.submit(stage4b_save_by_slide, args, audio_result, output_dir)
            for future in as_completed([future_c, future_d]):
                if future is future_c:
                    classified_result = future.result()
                    timings["Stage 4A 분류"] = classified_result.get("elapsed", 0.0)
                else:
                    by_slide_result = future.result()
                    timings["Stage 4B by_slide 저장"] = by_slide_result.get("elapsed", 0.0)

        timings["Stage 4 병렬 총"] = time.time() - t_parallel
        print(f"\n  ✓ Stage 4 완료  ({timings['Stage 4 병렬 총']:.1f}초)")
        print("─" * 70)

        # ── Stage 5 (직렬) ──
        notify_stage("integrate", "run")
        
        r5 = stage5_fusion(
            args,
            textualized_path=textualized_path,
            annotation_path=annotation_path,
            audio_result=audio_result,
            output_dir=output_dir,
        )
        timings["Stage 5 퓨전"] = r5["elapsed"]

        r6: dict = {}
        if args.skip_graph_triples:
            print("\n  ⏭  Stage 6 그래프 트리플 생성 — 사용자 옵션으로 스킵")
            print("─" * 70)
            timings["Stage 6 그래프 트리플"] = 0.0
            timings["Neo4j 적재"] = 0.0
        else:
            r6 = stage6_graph_triples(args, output_dir, slides_dir)
            timings["Stage 6 그래프 트리플"] = r6["elapsed"]

            print("\n  ⏭  Neo4j 적재 — 강의 시청 화면 진입 시 자동 적재")
            print("─" * 70)
            timings["Neo4j 적재"] = 0.0
                
        notify_stage("integrate", "done")

        if args.skip_lance_index:
            print("\n  ⏭  Stage 7 Lance 인덱스 — 사용자 옵션으로 스킵")
            print("─" * 70)
            timings["Stage 7 Lance 인덱스"] = 0.0
        else:
            notify_stage("summarize", "run")
            r7 = stage7_lance_index(args, output_dir, slides_dir)
            timings["Stage 7 Lance 인덱스"] = r7.get("elapsed", 0.0)
            notify_stage("summarize", "done")

        if getattr(args, "skip_graphrag_index", False):
            print("\n  ⏭  Stage 7B GraphRAG 인덱스 — 사용자 옵션으로 스킵")
            print("─" * 70)
            timings["Stage 7B GraphRAG 인덱스"] = 0.0
        else:
            r7b = stage7b_graphrag_index(args, output_dir)
            timings["Stage 7B GraphRAG 인덱스"] = r7b.get("elapsed", 0.0)

        # ── Stage 8 (직렬): 메타데이터 생성 ──  ← 여기 추가
        r8: dict = {}
        if getattr(args, "skip_metadata", False):
            print("\n  ⏭  Stage 8 메타데이터 생성 — 사용자 옵션으로 스킵")
            print("─" * 70)
            timings["Stage 8 메타데이터 생성"] = 0.0
        else:
            r8 = stage8_generate_metadata(args, output_dir, slides_dir)
            timings["Stage 8 메타데이터 생성"] = r8["elapsed"]

        if not timings.get("Stage 9 analyzer 입력 생성"):
            timings["Stage 9 analyzer 입력 생성"] = 0.0
        if "Stage 10 verifier 백그라운드 시작" not in timings:
            timings["Stage 10 verifier 백그라운드 시작"] = 0.0

        # ── Stage 11 (직렬): 추천 인덱스 생성 ──
        if getattr(args, "skip_recommender_index", False):
            print("\n  ⏭  Stage 11 추천 인덱스 생성 — 사용자 옵션으로 스킵")
            print("─" * 70)
            timings["Stage 11 추천 인덱스 생성"] = 0.0
        else:
            r11 = stage11_build_recommender_index(args)
            timings["Stage 11 추천 인덱스 생성"] = r11["elapsed"]

        # ── Lecture 자동 등록 ──
        _auto_register_lecture(stem)

        # ── 생성된 파일 목록 ──
        print("\n  생성된 파일:")
        output_files = [
            audio_result.get("segments_path", ""),
            audio_result.get("silences_path", ""),
            audio_result.get("deictics_path", ""),
            audio_result.get("emphasis_path", ""),
            annotation_path,
            textualized_path,
            classified_result.get("classified_path", ""),
            by_slide_result.get("by_slide_path", ""),
            r5.get("fused_path", ""),
            r6.get("triples_parquet", ""),
            r6.get("nodes_parquet", ""),
            r6.get("edges_parquet", ""),
            str(output_dir / f"{stem}_chunks_lance.parquet"),
            r7b.get("input_path", ""),
            r7b.get("entities_parquet", ""),
            r7b.get("relationships_parquet", ""),
            r8.get("metadata_path", ""),
            str(Path(getattr(args, "recommender_db_dir", DEFAULT_RECOMMENDER_DB_DIR))),
            r9.get("merged_clean_path", str(output_dir / f"{stem}_merged_clean.json")),
            r10.get("log_path", ""),
        ]
        for analyzer_path in (
            r10.get("claim_output", ""),
            r10.get("claim_report", ""),
        ):
            if analyzer_path and Path(analyzer_path).exists():
                output_files.append(analyzer_path)
        for path_str in output_files:
            if not path_str:
                continue
            p = Path(path_str)
            print(f"    {'✓' if p.exists() else '✗'}  {p}")
        if r10.get("spawned"):
            print(f"\n  verifier는 백그라운드에서 계속 실행 중입니다. (PID {r10.get('pid')})")
            print(f"  로그 파일: {r10.get('log_path')}")
            print()

    except Exception as e:
        print(f"\n❌ 파이프라인 오류: {e}")
        raise

    finally:
        # 성공/실패 무관하게 항상 타이밍 출력
        total_elapsed = time.time() - total_start
        print("\n" + "═" * 70)
        print("  단계별 소요 시간 (현재까지)")
        print("═" * 70)
        for stage, t in timings.items():
            label = "  (스킵)" if t == 0.0 else f"  {t:>7.1f}초"
            print(f"    {stage:<30} {label}")
        print(f"\n    {'총 소요 시간':<30}  {total_elapsed:>7.1f}초")


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
                        help="Stage 1A 건너뜀 (이미 슬라이드가 추출된 경우)")
    parser.add_argument("--force", action="store_true",
                        help="출력 파일이 있어도 모든 단계 강제 재실행")
    parser.add_argument("--retries", type=int, default=3,
                        help="Gemini API 재시도 횟수 (default: 3)")
    parser.add_argument("--debug", action="store_true", help="Stage 1 디버그 로그 출력")
    parser.add_argument("--masks", action="store_true", help="Stage 3A diff 마스크 이미지 저장")
    parser.add_argument("--skip-graph-triples", action="store_true",
                        help="Stage 6 그래프 Parquet(triples/nodes/edges) 생성 스킵")
    parser.add_argument(
        "--skip-neo4j",
        action="store_true",
        help="호환성 유지용 옵션입니다. Neo4j 적재는 강의 시청 화면 진입 시 수행됩니다.",
    )
    parser.add_argument("--skip-lance-index", action="store_true",
                        help="Stage 7 LanceDB+Parquet 인덱스 스킵")
    parser.add_argument(
        "--lance-root",
        default=None,
        help="LanceDB 저장 경로 (기본: 환경변수 GRAPHLEC_LANCE_ROOT 또는 data/lancedb)",
    )
    parser.add_argument("--skip-graphrag-index", action="store_true",
                        help="Stage 7B GraphRAG parquet 인덱스 생성 스킵")
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
                        help="Stage 8 메타데이터 생성 스킵")
    parser.add_argument("--skip-analyzer", action="store_true",
                        help="Stage 10 verifier 실행 스킵")
    parser.add_argument("--skip-recommender-index", action="store_true",
                        help="Stage 11 추천 인덱스 생성(build_index) 스킵")
    parser.add_argument("--metadata-dir", dest="metadata_dir", default=DEFAULT_RECOMMENDER_METADATA_DIR,
                        help=f"메타데이터 저장 디렉토리 (default: {DEFAULT_RECOMMENDER_METADATA_DIR})")
    parser.add_argument("--recommender-db-dir", dest="recommender_db_dir", default=DEFAULT_RECOMMENDER_DB_DIR,
                        help=f"추천 인덱스 LanceDB 경로 (default: {DEFAULT_RECOMMENDER_DB_DIR})")
    parser.add_argument("--title",      default="", help="강의명 (미입력 시 Gemini 자동 생성)")
    parser.add_argument("--instructor", default="", help="교수자명")
    parser.add_argument("--domain",     default="", help="도메인 (미입력 시 Gemini 자동 추론)")
    
    return parser

def main():
    args = get_parser().parse_args()

    if not args.skip_extract and not Path(args.input).exists():
        print(f"❌ 입력 영상 없음: {args.input}")
        sys.exit(1)

    run_pipeline(args)


if __name__ == "__main__":
    main()
