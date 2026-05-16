"""
fusion.py — 멀티모달 강의 데이터 통합

입력:
  - by_scene.json       : 오디오 전사 + 강조 감지 (scene occurrence 단위)
  - slide_classified.json : 슬라이드 텍스트 + 시각 강조 + role 분류
  - annotation.json       : 강사 필기 annotation 이벤트

출력:
  - fused.json : scene 단위 통합 텍스트 + 강조 점수

사용법:
  python fusion.py
  python fusion.py --audio by_scene.json --classified slide_classified.json
                   --annotation annotation.json --output fused.json
"""

import json
import re
import argparse
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


# ============================================================================
#  설정
# ============================================================================

@dataclass
class Config:
    stem:            str  = "lecture"
    output_dir:      Path = Path("output")
    slides_dir:      Path = Path("output_slides")

    # 경로 — 직접 지정 시 우선, 없으면 __post_init__에서 config.output_paths()로 채움
    audio_path:          Path = field(default=None)
    classified_path:     Path = field(default=None)
    annotation_path:     Path = field(default=None)
    output_path:         Path = field(default=None)
    slide_metadata_path: Path = field(default=None)  # output_slides/metadata.json

    def __post_init__(self):
        try:
            from .config import output_paths
            paths = output_paths(self.stem, self.output_dir, self.slides_dir)
            if self.audio_path is None:
                self.audio_path = paths["by_scene"]
            if self.classified_path is None: self.classified_path = paths["classified"]
            if self.annotation_path is None: self.annotation_path = paths["annotation"]
            if self.output_path     is None: self.output_path     = paths["fused"]
        except ImportError:
            d, s = self.output_dir, self.stem
            if self.audio_path is None:
                self.audio_path = d / f"{s}_by_scene.json"
            if self.classified_path is None: self.classified_path = d / f"{s}_slide_classified.json"
            if self.annotation_path is None: self.annotation_path = d / f"{s}_annotation.json"
            if self.output_path     is None: self.output_path     = d / f"{s}_fused.json"
        if self.slide_metadata_path is None:
            self.slide_metadata_path = self.slides_dir / "metadata.json"

    # 강조 가중치
    W_AUDIO:    float = 0.8
    W_VISUAL:   float = 1.5
    W_ANNOT:    float = 1.0   # annotation 점수는 자체 배율이 크므로 1.0
    BOTH_BONUS: float = 1.3   # 오디오 + annotation 동시 감지 보너스
    AUDIO_SIGNAL_MAX: float = 10.0      # volume_score(5) + pitch_score(5)
    AUDIO_TOPIC_MAX: float = 60.0       # 20 keywords, 4개씩 5~1점
    AUDIO_IMPORTANCE_MAX: float = 9.0   # exam(5) + strong(3) + summary(1)

    # 시간 윈도우
    TIME_WINDOW_SEC:          float = 20.0  # 강조 합산용 annotation ↔ audio 매칭
    DEICTIC_WINDOW_BEFORE_SEC: float = 5.0  # 지시어 발화 기준 이전
    DEICTIC_WINDOW_AFTER_SEC:  float = 2.0  # 지시어 발화 기준 이후

    # 키워드 필터
    MIN_KEYWORD_LEN: int = 2
    MIN_SLIDE_TEXT_LINE_LEN: int = 4  # 슬라이드 본문 라인 최소 길이 (불릿·번호 제외)
    SLIDE_TEXT_SCORE: float = 0.3     # 슬라이드 본문 등장 1회당 점수
    STOPWORDS: frozenset = frozenset({
        # 조사·접속사
        "이", "그", "저", "은", "는", "가", "을", "를", "의", "에", "도",
        "와", "과", "하고", "이고", "이며", "그리고", "그래서", "하지만",
        "또한", "즉", "따라서", "그러나", "또는", "및",
        # 서술어
        "있습니다", "있어요", "합니다", "해요", "됩니다", "돼요",
        "입니다", "이에요", "이다", "한다", "된다",
        # 일반 명사·대명사
        "것", "거", "수", "때", "더", "많이", "같은", "이런", "그런",
        "시절", "이후", "전과", "앎",
        # 추상 메타 단어 (강의 구조어)
        "개념", "정의", "목표", "목적", "기능", "시작", "발전", "차이",
        "종류", "특징", "핵심", "단어", "강의", "내용", "설명", "이해",
        "개요", "소개", "정리", "비교", "분석", "예시", "문제",
        # 일반 동사·행위어
        "실행", "요청", "종료", "생각", "과정", "사용", "제공",
        "처리", "수행", "동작", "발생", "설치", "구현", "관련",
        # 영어 불용어
        "the", "a", "an", "is", "are", "was", "were", "to", "of", "in",
        "and", "or", "for", "with", "that", "this", "be", "by",
        "chapter",
    })


# ============================================================================
#  가중치 테이블
# ============================================================================

# 슬라이드 디자인 강조 타입 → 점수
SLIDE_EMPHASIS_WEIGHTS: dict[str, float] = {
    "color":     3.0,
    "highlight": 3.0,
    "bold":      2.0,
    "underline": 2.0,
    "box":       1.0,
    "callout":   1.0,
    "italic":    0.5,
    "other":     0.5,
}

# annotation 타입 → 점수
ANNOT_TYPE_WEIGHTS: dict[str, float] = {
    "circle":           3.0,
    "underline":        3.0,
    "bracket":          3.0,
    "arrow":            2.0,
    "handwritten_text": 2.0,  # handwritten_content != null 인 경우만
    "cross":            1.0,
    "other":            1.0,
}

# annotation confidence → 배율
ANNOT_CONFIDENCE_MULT: dict[str, float] = {
    "high":   1.0,
    "medium": 0.7,
    "low":    0.4,
}

# 지시어 패턴
DEICTIC_PATTERNS = re.compile(
    r'(?<!\w)(이것|이거|이것들|이게|저것|저거|저게|여기|저기|이쪽|저쪽|이 부분|저 부분|이 내용|이 개념)(?!\w)'
)

# ============================================================================
#  점수 계산 함수
# ============================================================================

def score_slide_emphasis(slide_emphasis_items: list[dict]) -> float:
    """slide_classified의 slide_emphasis 배열 → 시각 강조 점수."""
    total = 0.0
    for item in slide_emphasis_items:
        raw_type = item.get("type") or "other"
        t = (raw_type[0] if isinstance(raw_type, list) else raw_type).lower()
        total += SLIDE_EMPHASIS_WEIGHTS.get(t, SLIDE_EMPHASIS_WEIGHTS["other"])
    return round(total, 3)


def score_annotation_emphasis(annotations: list[dict]) -> float:
    """annotation 배열 → annotation 강조 점수."""
    total = 0.0
    for ann in annotations:
        ann_type = (ann.get("type") or "other").lower()
        # handwritten_text는 내용이 인식된 경우만
        if ann_type == "handwritten_text" and not ann.get("handwritten_content"):
            continue
        type_w = ANNOT_TYPE_WEIGHTS.get(ann_type, ANNOT_TYPE_WEIGHTS["other"])
        conf   = (ann.get("confidence") or "low").lower()
        conf_m = ANNOT_CONFIDENCE_MULT.get(conf, ANNOT_CONFIDENCE_MULT["low"])
        total += type_w * conf_m
    return round(total, 3)


def score_audio_emphasis(contexts: list[dict], cfg: Config) -> dict:
    """by_scene contexts 배열 → 오디오 강조 점수와 raw breakdown."""
    scored_contexts = []
    for ctx in contexts:
        detail = get_context_emphasis_detail(ctx)
        audio  = detail.get("audio")
        topic = detail.get("topic") or {}
        importance = detail.get("importance_keywords") or {}
        if not audio:
            continue
        signal_score = float(
            audio.get(
                "audio_signal_score",
                audio.get("std_based_score", audio.get("score", 0.0)),
            )
            or 0.0
        )
        topic_score = float(topic.get("topic_keyword_score", 0.0) or 0.0)
        importance_score = float(importance.get("score", 0.0) or 0.0)
        raw_score = signal_score + topic_score + importance_score
        norm = (
            min(signal_score, cfg.AUDIO_SIGNAL_MAX)
            + min(topic_score, cfg.AUDIO_TOPIC_MAX)
            + min(importance_score, cfg.AUDIO_IMPORTANCE_MAX)
        ) / (cfg.AUDIO_SIGNAL_MAX + cfg.AUDIO_TOPIC_MAX + cfg.AUDIO_IMPORTANCE_MAX)
        scored_contexts.append({
            "raw": raw_score,
            "norm": norm,
            "audio_signal_raw": signal_score,
            "topic_keyword_raw": topic_score,
            "importance_keyword_raw": importance_score,
        })
    if not scored_contexts:
        return {
            "raw": 0.0,
            "norm": 0.0,
            "audio_signal_raw": 0.0,
            "topic_keyword_raw": 0.0,
            "importance_keyword_raw": 0.0,
        }
    best = max(scored_contexts, key=lambda item: item["raw"])
    return {
        "raw": round(best["raw"], 3),
        "norm": round(best["norm"], 3),
        "audio_signal_raw": round(best["audio_signal_raw"], 3),
        "topic_keyword_raw": round(best["topic_keyword_raw"], 3),
        "importance_keyword_raw": round(best["importance_keyword_raw"], 3),
    }


def get_context_emphasis_detail(ctx: dict) -> dict:
    """라벨 필드 없이 feature/detail 위치를 호환 조회한다."""
    return ctx.get("audio_emphasis") or ctx.get("emphasis_detail") or ctx.get("emphasis", {}).get("detail") or {}


def _merge_keyword_entries(existing: list[dict], incoming: list[dict]) -> list[dict]:
    """여러 scene에서 나온 emphasized_keywords를 slide 단위로 합치되 audio 신호는 제외한다."""
    merged: dict[str, dict] = {}
    for item in existing + incoming:
        kw = item.get("keyword")
        if not kw:
            continue
        sources = [s for s in item.get("sources", []) if s != "audio"]
        visual_score = float(item.get("visual_score", 0.0) or 0.0)
        annotation_score = float(item.get("annotation_score", 0.0) or 0.0)
        slide_text_score = float(item.get("slide_text_score", 0.0) or 0.0)
        if not sources and visual_score == 0.0 and annotation_score == 0.0 and slide_text_score == 0.0:
            continue
        if kw not in merged:
            merged[kw] = {
                "keyword": kw,
                "sources": [],
                "visual_score": 0.0,
                "annotation_score": 0.0,
                "slide_text_score": 0.0,
            }
        entry = merged[kw]
        entry["sources"] = sorted(set(entry["sources"]) | set(sources))
        entry["visual_score"] = round(entry["visual_score"] + visual_score, 3)
        entry["annotation_score"] = round(entry["annotation_score"] + annotation_score, 3)
        entry["slide_text_score"] = round(entry["slide_text_score"] + slide_text_score, 3)
    result = list(merged.values())
    result.sort(key=lambda x: (-len(x["sources"]), x["keyword"]))
    return result


def _append_unique(seq: list, value):
    if value is not None and value not in seq:
        seq.append(value)


# ============================================================================
#  키워드 추출 및 소스 태깅
# ============================================================================

def normalize_keyword(word: str) -> str:
    word = word.lower().strip()
    word = re.sub(r'[^\w가-힣a-z0-9]', '', word)
    word = re.sub(
        r'(은|는|이|가|을|를|의|에서|에게|에|로|으로|와|과|도|만|까지|부터|처럼|이란|란)$',
        '', word
    )
    return word.strip()


def _extract_content_words(text: str, min_len: int) -> list[str]:
    try:
        from .emphasis_keyword import extract_contiguous_content_words
    except ImportError:
        from emphasis_keyword import extract_contiguous_content_words
    return extract_contiguous_content_words(text, min_length=min_len)


def extract_keywords_from_text(text: str, stopwords: frozenset, min_len: int) -> set[str]:
    result = set()
    for word in _extract_content_words(text, min_len):
        kw = normalize_keyword(word)
        if len(kw) >= min_len and kw not in stopwords:
            result.add(kw)
    return result


def build_emphasized_keywords(
    audio_keywords: list[str],        # by_scene emphasis.keywords.all_keywords
    visual_keywords: list[str],        # slide_classified slide_emphasis[].text
    annotation_keywords: list[str],    # annotation target_content 토큰
    slide_text_keywords: list[str],    # slide_text 본문 라인 스캔 (4번째 소스)
    cfg: Config,
) -> list[dict]:
    """
    네 소스의 키워드를 통합, 소스 태깅.
      - audio / visual / annotation: 강조 신호 → sources 배열에 포함
      - slide_text: 본문 빈도 신호 → sources에 미포함, slide_text_score에만 누적
    같은 키워드가 여러 강조 소스에서 나오면 sources 배열에 모두 포함.
    """
    kw_map: dict[str, dict] = {}

    def add(kw: str, source: str, score_key: str, score_val: float):
        kw = normalize_keyword(kw)
        if not kw or len(kw) < cfg.MIN_KEYWORD_LEN or kw in cfg.STOPWORDS:
            return
        if kw not in kw_map:
            kw_map[kw] = {"keyword": kw, "sources": [], "audio_score": 0.0,
                          "visual_score": 0.0, "annotation_score": 0.0,
                          "slide_text_score": 0.0}
        entry = kw_map[kw]
        # slide_text는 sources에 포함하지 않음 (강조 신호가 아닌 본문 신호)
        if source != "slide_text" and source not in entry["sources"]:
            entry["sources"].append(source)
        entry[score_key] = round(entry[score_key] + score_val, 3)

    for kw in audio_keywords:
        add(kw, "audio", "audio_score", 1.0)

    for kw in visual_keywords:
        add(kw, "visual", "visual_score", 1.0)

    for kw in annotation_keywords:
        add(kw, "annotation", "annotation_score", 1.0)

    for kw in slide_text_keywords:
        add(kw, "slide_text", "slide_text_score", cfg.SLIDE_TEXT_SCORE)

    # sources 리스트 정렬 (재현성)
    result = []
    for entry in kw_map.values():
        entry["sources"] = sorted(entry["sources"])
        result.append(entry)

    # 소스 수 내림차순, 알파벳 오름차순 정렬
    result.sort(key=lambda x: (-len(x["sources"]), x["keyword"]))
    return result


# ============================================================================
#  annotation 인덱싱
# ============================================================================

def build_annotation_index(annotation_data: list[dict]) -> dict[int, list[dict]]:
    """
    logical slide_number → annotation 이벤트 리스트 인덱스.
    annotation.json은 annotation 이벤트(annot_index별) 배열이므로
    같은 논리 슬라이드에 여러 scene 이벤트가 있을 수 있음.
    """
    index: dict[int, list[dict]] = {}
    for event in annotation_data:
        sid = event.get("slide_number")
        if sid is None:
            continue
        if sid not in index:
            index[sid] = []
        index[sid].append(event)
    return index


def _annotation_timestamp(event: dict, ann: Optional[dict] = None) -> Optional[float]:
    if ann:
        for key in ("first_seen_timestamp_sec", "timestamp_sec"):
            if ann.get(key) is not None:
                return float(ann[key])
    if event.get("timestamp_sec") is not None:
        return float(event["timestamp_sec"])
    return None


def filter_annotation_events_for_scene(
    events: list[dict],
    start_sec: Optional[float],
    end_sec: Optional[float],
) -> list[dict]:
    """논리 슬라이드에 묶인 annotation 중 현재 scene 시간대의 이벤트만 남긴다."""
    if start_sec is None or end_sec is None:
        return copy_annotation_events(events)

    start = float(start_sec)
    end = float(end_sec)
    filtered: list[dict] = []
    for event in events:
        event_ts = _annotation_timestamp(event)
        anns = event.get("annotations")
        if isinstance(anns, list):
            kept_annotations = []
            for ann in anns:
                ann_ts = _annotation_timestamp(event, ann)
                if ann_ts is None or start <= ann_ts <= end:
                    kept_annotations.append(ann)
            if kept_annotations:
                event_copy = json.loads(json.dumps(event))
                event_copy["annotations"] = kept_annotations
                filtered.append(event_copy)
                continue
        if event_ts is None or start <= event_ts <= end:
            filtered.append(json.loads(json.dumps(event)))
    return filtered


def flatten_annotations_for_slide(events: list[dict]) -> list[dict]:
    """
    슬라이드의 annotation 이벤트 리스트 → 개별 annotation 목록 (timestamp 포함).
    """
    result = []
    for event in events:
        event_ts = event.get("timestamp_sec")
        for ann in event.get("annotations", []):
            ann_ts = (
                ann.get("first_seen_timestamp_sec")
                if ann.get("first_seen_timestamp_sec") is not None
                else ann.get("timestamp_sec")
            )
            result.append({
                **ann,
                "timestamp_sec": ann_ts if ann_ts is not None else event_ts,
            })
    return result


def copy_annotation_events(events: list[dict]) -> list[dict]:
    """fusion 결과에 실을 수 있도록 annotation 이벤트를 그대로 복사."""
    return json.loads(json.dumps(events))


def attach_scene_to_annotation_events(
    events: list[dict],
    scene_id: str,
    scene_number,
    scene_index,
) -> list[dict]:
    """scene에 실리는 annotation 이벤트에 scene 출처를 명시한다."""
    result = copy_annotation_events(events)
    for event in result:
        if "slide_number" in event:
            event["source_slide_number"] = event.pop("slide_number")
        event["scene_id"] = scene_id
        event["scene_number"] = scene_number
        event["scene_index"] = scene_index
    return result


def attach_scene_to_flat_annotations(
    annotations: list[dict],
    scene_id: str,
    scene_number,
    scene_index,
    slide_number,
) -> list[dict]:
    """flatten annotation에도 scene/slide 출처를 붙인다."""
    result = copy_annotation_events(annotations)
    for ann in result:
        ann["scene_id"] = scene_id
        ann["scene_number"] = scene_number
        ann["scene_index"] = scene_index
        ann["source_slide_number"] = slide_number
    return result


def extract_slide_summary(events: list[dict]) -> str:
    """annotation 이벤트들에서 첫 번째 유효한 slide_summary를 반환."""
    for event in events:
        summary = str(event.get("slide_summary") or "").strip()
        if summary:
            return summary
    return ""


def build_annotation_highlights_summary(annotations: list[dict]) -> str:
    """
    개별 annotation 목록을 사람이 읽기 쉬운 짧은 요약 문장으로 변환.
    annotation.json의 slide_summary를 보완하는 구조화 요약이다.
    """
    if not annotations:
        return ""

    targets: list[str] = []
    target_seen: set[str] = set()
    type_seen: list[str] = []
    for ann in annotations:
        ann_type = str(ann.get("type") or "other").strip()
        if ann_type and ann_type not in type_seen:
            type_seen.append(ann_type)

        target = str(
            ann.get("target_content")
            or ann.get("handwritten_content")
            or ""
        ).strip()
        if target and target not in target_seen:
            target_seen.add(target)
            targets.append(target)

    if targets:
        preview = ", ".join(targets[:5])
        if len(targets) > 5:
            preview += f" 외 {len(targets) - 5}개"
    else:
        preview = "명시적 텍스트 대상 없음"

    type_preview = ", ".join(type_seen[:5]) if type_seen else "other"
    return (
        f"강조 표시는 총 {len(annotations)}개이며, "
        f"주요 대상은 {preview}이다. "
        f"표시 유형은 {type_preview}가 포함된다."
    )


# ============================================================================
#  지시어 매칭
# ============================================================================

def find_deictic_target(
    seg_start: float,
    seg_text: str,
    slide_annotations: list[dict],
    cfg: Config,
) -> Optional[dict]:
    """
    segment 텍스트에 지시어가 있으면, 발화 시점 기준 윈도우 내 annotation 중
    가장 가까운 것을 지시 대상으로 반환.
    """
    if not DEICTIC_PATTERNS.search(seg_text):
        return None
    if not slide_annotations:
        return None

    t_low = seg_start - cfg.DEICTIC_WINDOW_BEFORE_SEC
    t_high = seg_start + cfg.DEICTIC_WINDOW_AFTER_SEC

    candidates = [
        ann for ann in slide_annotations
        if ann.get("timestamp_sec") is not None
        and t_low <= ann["timestamp_sec"] <= t_high
    ]
    if not candidates:
        return None

    closest = min(candidates, key=lambda a: abs(a["timestamp_sec"] - seg_start))

    return {
        "target_content": closest.get("target_content"),
        "annotation_type": closest.get("type"),
        "bbox": closest.get("target_bbox") or closest.get("annotation_bbox"),
        "timestamp_sec": closest.get("timestamp_sec"),
        "confidence": closest.get("confidence"),
    }


# ============================================================================
#  both_bonus 계산
# ============================================================================

def calc_both_bonus(
    audio_score: float,
    annot_score: float,
    annotation_timestamps: list[float],
    audio_contexts: list[dict],
    cfg: Config,
) -> float:
    """
    오디오 강조 구간과 annotation 타임스탬프가 TIME_WINDOW_SEC 내에 겹치면 BOTH_BONUS 부여.
    """
    if audio_score == 0.0 or annot_score == 0.0:
        return 0.0

    for ctx in audio_contexts:
        detail = get_context_emphasis_detail(ctx)
        audio = detail.get("audio") or {}
        score = audio.get("std_based_score", audio.get("score", 0))
        if not score:
            continue
        ctx_start = ctx["start"]
        for ts in annotation_timestamps:
            if abs(ctx_start - ts) <= cfg.TIME_WINDOW_SEC:
                return cfg.BOTH_BONUS

    return 0.0


# ============================================================================
#  메인 퓨전 로직
# ============================================================================

def run_fusion(cfg: Config) -> dict:
    # ── 데이터 로드 ──────────────────────────────────────────────────────────
    with open(cfg.audio_path,      encoding="utf-8") as f:
        audio_data = json.load(f)
    with open(cfg.classified_path, encoding="utf-8") as f:
        classified_data = json.load(f)
    with open(cfg.annotation_path, encoding="utf-8") as f:
        annotation_data = json.load(f)

    audio_slides      = audio_data["scenes"]
    classified_slides = classified_data["scenes"]
    annot_index       = build_annotation_index(annotation_data)

    # audio를 scene_index 기준으로 인덱싱
    audio_index: dict[int, dict] = {
        s.get("scene_index"): s
        for s in audio_slides
        if s.get("scene_index") is not None
    }

    # classified를 slide_id 기준으로 인덱싱
    classified_index: dict[str, dict] = {s["slide_id"]: s for s in classified_slides}

    print(f"로드 완료: audio {len(audio_slides)}개, "
          f"classified {len(classified_slides)}개, "
          f"annotation events {len(annotation_data)}개")

    fused_scenes = []
    slide_records: dict[str, dict] = {}
    _seg_idx = 0

    for cl_slide in classified_slides:
        slide_id  = cl_slide["slide_id"]
        slide_num = cl_slide["slide_number"]
        scene_num = cl_slide.get("scene_number", cl_slide.get("scene_index", slide_num))
        scene_label = int(scene_num) if isinstance(scene_num, int) else int(slide_num)

        # audio scene 매핑
        au_slide = audio_index.get(scene_num)

        # annotation 이벤트 (논리 슬라이드로 찾고, 현재 scene 시간대로 좁힘)
        annot_events_all = annot_index.get(slide_num, [])
        annot_events = filter_annotation_events_for_scene(
            annot_events_all,
            au_slide.get("start_sec") if au_slide else None,
            au_slide.get("end_sec") if au_slide else None,
        )
        flat_annots    = flatten_annotations_for_slide(annot_events)
        annot_ts_list  = [a["timestamp_sec"] for a in flat_annots if a.get("timestamp_sec")]
        slide_summary  = extract_slide_summary(annot_events)

        # ── 강조 점수 계산 ───────────────────────────────────────────────────
        visual_score = score_slide_emphasis(cl_slide.get("slide_emphasis", []))
        annot_score  = score_annotation_emphasis(flat_annots)
        audio_score  = score_audio_emphasis(au_slide["contexts"] if au_slide else [], cfg)

        both_bonus = calc_both_bonus(
            audio_score["norm"], annot_score, annot_ts_list,
            au_slide["contexts"] if au_slide else [],
            cfg,
        )

        total_score = round(
            cfg.W_VISUAL * visual_score
            + cfg.W_ANNOT  * annot_score
            + cfg.W_AUDIO  * audio_score["norm"]
            + both_bonus,
            3,
        )

        # ── 강조 키워드 수집 ─────────────────────────────────────────────────
        audio_kws: list[str] = []
        if au_slide:
            for ctx in au_slide["contexts"]:
                detail = get_context_emphasis_detail(ctx)
                kws    = detail.get("keywords", {}).get("all_keywords", [])
                if not kws:
                    kws = (detail.get("topic") or {}).get("keywords", [])
                audio_kws.extend(kws)

        visual_kws: list[str] = []
        for item in cl_slide.get("slide_emphasis", []):
            txt = item.get("text", "")
            if txt:
                visual_kws.extend(extract_keywords_from_text(txt, cfg.STOPWORDS, cfg.MIN_KEYWORD_LEN))

        annot_kws: list[str] = []
        for ann in flat_annots:
            tc = ann.get("target_content", "")
            if tc:
                annot_kws.extend(extract_keywords_from_text(tc, cfg.STOPWORDS, cfg.MIN_KEYWORD_LEN))

        # 4번째 소스: 슬라이드 본문 라인 단위 스캔
        # 강조 신호 없이도 슬라이드에 반복 등장하는 핵심 개념 포착
        # Kiwi 기반 내용어 추출: 공백/기호 없이 붙어 있던 명사열만 복합어로 수집
        slide_text_kws: list[str] = []
        for line in cl_slide.get("t1", "").splitlines():
            line = line.strip()
            if len(line) < cfg.MIN_SLIDE_TEXT_LINE_LEN:
                continue  # 불릿(•, □), 번호(1.), 짧은 기호 제외

            slide_text_kws.extend(
                extract_keywords_from_text(line, cfg.STOPWORDS, cfg.MIN_KEYWORD_LEN)
            )

        emphasized_keywords = build_emphasized_keywords(
            audio_kws, visual_kws, annot_kws, slide_text_kws, cfg
        )

        # ── contexts + segments (지시어 매칭 포함) ──────────────────────────
        fused_contexts = []
        if au_slide:
            for ctx in au_slide["contexts"]:
                detail   = get_context_emphasis_detail(ctx)
                audio_emphasis = ctx.get("audio_emphasis") or {
                    key: detail.get(key)
                    for key in ("audio", "importance_keywords", "topic")
                    if detail.get(key) is not None
                }
                stressed = int(ctx.get("detection_count", 0) or 0) > 0

                fused_segs = []
                for seg in ctx.get("segments", []):
                    seg_audio_emphasis = seg.get("audio_emphasis") or audio_emphasis
                    deictic_target = find_deictic_target(
                        seg["start"], seg["text"], flat_annots, cfg
                    )
                    fused_segs.append({
                        "segment_id":     f"segment/{_seg_idx:04d}",
                        "start":          seg["start"],
                        "end":            seg["end"],
                        "text":           seg["text"],
                        "stressed":       stressed,  # context 단위 플래그를 segment에 상속
                        "audio_emphasis": seg_audio_emphasis,
                        "deictic_target": deictic_target,
                    })
                    _seg_idx += 1

                fused_contexts.append({
                    "context_index": ctx["context_index"],
                    "start":         ctx["start"],
                    "end":           ctx["end"],
                    "text":          ctx["text"],
                    "stressed":      stressed,
                    "audio_emphasis": audio_emphasis,
                    "segments":      fused_segs,
                })

        # ── annotation 요약 ──────────────────────────────────────────────────
        annotations_summary = []
        for ann in flat_annots:
            ann_type = (ann.get("type") or "other").lower()
            if ann_type == "handwritten_text" and not ann.get("handwritten_content"):
                continue
            type_w = ANNOT_TYPE_WEIGHTS.get(ann_type, ANNOT_TYPE_WEIGHTS["other"])
            conf   = (ann.get("confidence") or "low").lower()
            conf_m = ANNOT_CONFIDENCE_MULT.get(conf, ANNOT_CONFIDENCE_MULT["low"])
            annotations_summary.append({
                "type":            ann.get("type"),
                "target_content":  ann.get("target_content"),
                "handwritten_content": ann.get("handwritten_content"),
                "score":           round(type_w * conf_m, 3),
                "confidence":      ann.get("confidence"),
                "timestamp_sec":   ann.get("timestamp_sec"),
                "bbox":            ann.get("target_bbox") or ann.get("annotation_bbox"),
            })
        annotation_highlights_summary = build_annotation_highlights_summary(annotations_summary)

        scene_id = cl_slide.get("scene_id", f"scene/{scene_label:04d}")
        scene_index = cl_slide.get("scene_index", scene_num)
        start_sec = au_slide["start_sec"] if au_slide else None
        end_sec = au_slide["end_sec"] if au_slide else None
        annotation_events = attach_scene_to_annotation_events(
            annot_events,
            scene_id,
            scene_num,
            scene_index,
        )
        scene_flat_annots = attach_scene_to_flat_annotations(
            flat_annots,
            scene_id,
            scene_num,
            scene_index,
            slide_num,
        )
        slide_emphasis_score = {
            "visual":     round(cfg.W_VISUAL * visual_score, 3),
        }
        slide_emphasis_score["total"] = slide_emphasis_score["visual"]
        scene_annotation_score = round(cfg.W_ANNOT * annot_score, 3)

        # ── slide 저장소 구성 ────────────────────────────────────────────────
        if slide_id not in slide_records:
            slide_records[slide_id] = {
                "slide_id":     slide_id,
                "slide_number": slide_num,
                "slide_canonical_number": cl_slide.get("slide_canonical_number", slide_num),
                "representative_scene_number": cl_slide.get("representative_scene_number", slide_num),
                "title":        cl_slide.get("title", ""),
                "slide_text":   cl_slide.get("t1", ""),
                "t1_structure": cl_slide.get("t1_structure"),
                "slide_type":   cl_slide.get("slide_type", "text"),
                "slide_topic_keywords": cl_slide.get("slide_topic_keywords", []),
                "slide_topic_keyword_scores": cl_slide.get("slide_topic_keyword_scores", {}),
                "slide_topic_keyword_score": cl_slide.get("slide_topic_keyword_score", 0),
                "slide_topic_total_count_sum": cl_slide.get("slide_topic_total_count_sum", 0),
                "scene_ids": [],
                "scene_numbers": [],
                "scene_indexes": [],
                "emphasis_score": {
                    "visual": 0.0,
                    "total": 0.0,
                },
                "emphasized_keywords": [],
            }

        slide_record = slide_records[slide_id]
        _append_unique(slide_record["scene_ids"], scene_id)
        _append_unique(slide_record["scene_numbers"], scene_num)
        _append_unique(slide_record["scene_indexes"], scene_index)
        if slide_emphasis_score["total"] >= slide_record["emphasis_score"].get("total", 0.0):
            slide_record["emphasis_score"] = {
                **slide_emphasis_score,
            }
        slide_record["emphasized_keywords"] = _merge_keyword_entries(
            slide_record["emphasized_keywords"],
            emphasized_keywords,
        )

        # ── scene 통합 ───────────────────────────────────────────────────────
        fused_scene = {
            "scene_id":     scene_id,
            "scene_number": scene_num,
            "scene_index": scene_index,
            "start_sec":    start_sec,
            "end_sec":      end_sec,
            "role":         cl_slide.get("role"),
            "slide_id":     slide_id,
            "slide_number": slide_num,
            "slide_canonical_number": cl_slide.get("slide_canonical_number", slide_num),
            "slide_visit_order": cl_slide.get("slide_visit_order", 1),
            "slide_is_revisit": cl_slide.get("slide_is_revisit", False),
            "representative_scene_number": cl_slide.get("representative_scene_number", slide_num),
            "title":        cl_slide.get("title", ""),
            "slide_text":   cl_slide.get("t1", ""),
            "t1_structure": cl_slide.get("t1_structure", ""),
            "slide_type":   cl_slide.get("slide_type", "text"),
            "emphasis_score": {
                "annotation": scene_annotation_score,
                "total": scene_annotation_score,
            },
            "slide_summary": slide_summary,
            "annotation_highlights_summary": annotation_highlights_summary,
            "annotation": annotation_events,
            "annotation_events": annotation_events,
            "annotation_flat": scene_flat_annots,
            "annotations": scene_flat_annots,
            "annotations_summary": annotations_summary,
            "contexts":            fused_contexts,
        }

        fused_scenes.append(fused_scene)
        print(f"  scene_{scene_label:03d} / {slide_id} | role={cl_slide.get('role'):12s} | "
              f"total={total_score:.2f} "
              f"(vis={visual_score:.1f} ann={annot_score:.1f} aud={audio_score['norm']:.2f} bonus={both_bonus:.1f})")

    # ── 출력 ─────────────────────────────────────────────────────────────────
    fused_slides = sorted(
        slide_records.values(),
        key=lambda slide: (
            slide.get("slide_number") is None,
            slide.get("slide_number") or 0,
            slide.get("slide_id") or "",
        ),
    )
    logical_slide_count = len(fused_slides)

    output = {
        "metadata": {
            "total_scenes": len(fused_scenes),
            "total_slides": logical_slide_count,
            "source_files": {
                "audio":      str(cfg.audio_path),
                "classified": str(cfg.classified_path),
                "annotation": str(cfg.annotation_path),
            },
            "fusion_weights": {
                "W_AUDIO":                    cfg.W_AUDIO,
                "W_VISUAL":                   cfg.W_VISUAL,
                "W_ANNOT":                    cfg.W_ANNOT,
                "BOTH_BONUS":                 cfg.BOTH_BONUS,
                "AUDIO_SIGNAL_MAX":            cfg.AUDIO_SIGNAL_MAX,
                "AUDIO_TOPIC_MAX":             cfg.AUDIO_TOPIC_MAX,
                "AUDIO_IMPORTANCE_MAX":        cfg.AUDIO_IMPORTANCE_MAX,
                "TIME_WINDOW_SEC":            cfg.TIME_WINDOW_SEC,
                "DEICTIC_WINDOW_BEFORE_SEC":  cfg.DEICTIC_WINDOW_BEFORE_SEC,
                "DEICTIC_WINDOW_AFTER_SEC":   cfg.DEICTIC_WINDOW_AFTER_SEC,
            },
        },
        "slides": fused_slides,
        "scenes": fused_scenes,
    }

    return output


# ============================================================================
#  엔트리포인트
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="멀티모달 강의 데이터 퓨전",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--stem", required=True,
                        help="강의 파일 stem (예: os1-1). 경로는 config.output_paths()로 자동 완성됨")
    parser.add_argument("--output_dir", default="output",      help="출력 디렉토리 (기본: output)")
    parser.add_argument("--slides_dir", default="output_slides",help="슬라이드 디렉토리 (기본: output_slides)")
    args = parser.parse_args()

    cfg = Config(
        stem       = args.stem,
        output_dir = Path(args.output_dir),
        slides_dir = Path(args.slides_dir),
    )

    for p, label in [
        (cfg.audio_path,      "audio"),
        (cfg.classified_path, "classified"),
        (cfg.annotation_path, "annotation"),
    ]:
        if not p.exists():
            print(f"❌ {label} 파일 없음: {p}")
            return

    print("\n" + "="*60)
    print("🔀 멀티모달 퓨전 시작")
    print("="*60)

    start = time.time()
    output = run_fusion(cfg)

    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cfg.output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    elapsed = time.time() - start
    total_kw = sum(len(s.get("emphasized_keywords", [])) for s in output.get("slides", []))
    total_deictic = sum(
        1
        for s in output["scenes"]
        for ctx in s["contexts"]
        for seg in ctx["segments"]
        if seg.get("deictic_target")
    )

    print("\n" + "="*60)
    print("✅ 완료")
    print("="*60)
    print(f"  scene          : {output['metadata']['total_scenes']}개")
    print(f"  slide          : {output['metadata']['total_slides']}개")
    print(f"  강조 키워드    : {total_kw}개")
    print(f"  지시어 매칭    : {total_deictic}개")
    print(f"  처리 시간      : {elapsed:.2f}초")
    print(f"  출력 파일      : {cfg.output_path}")


if __name__ == "__main__":
    main()
