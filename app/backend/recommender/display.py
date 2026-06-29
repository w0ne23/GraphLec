"""
recommender/display.py
──────────────────────
추천 결과 표시 점수 및 reason 문자열 생성.
"""

from __future__ import annotations

from typing import Any, Optional


DISPLAY_SCORE_FULL_RATIO    = 0.8
CONDITION_DISPLAY_SHARE     = 20
CONTENT_MEANING_DISPLAY_SHARE = 100 - CONDITION_DISPLAY_SHARE


def _ratio(value: Any) -> float:
    try:
        numeric = float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return numeric / 100.0 if numeric > 1.0 else numeric


def _display_component_score(value: Any) -> float:
    return max(0.0, min(_ratio(value) / DISPLAY_SCORE_FULL_RATIO, 1.0)) * 100.0


def _condition_score_ratio(detail: dict) -> float:
    condition_scores: list[float] = []
    has_duration_condition = bool(
        _ratio(detail.get("duration_score")) > 0.0
        or detail.get("duration_mismatch") is True
    )

    if has_duration_condition:
        condition_scores.append(_ratio(detail.get("duration_score")))
    if detail.get("visual_preference"):
        condition_scores.append(max(
            _ratio(detail.get("visual_score")),
            _ratio(detail.get("visual_density_score")),
            _ratio(detail.get("visual_concept_score")),
        ))
    if detail.get("application_preference"):
        condition_scores.append(_ratio(detail.get("application_score")))
    if detail.get("listenability_preference"):
        condition_scores.append(_ratio(detail.get("listenability_score")))
    if detail.get("slow_speech_preference"):
        condition_scores.append(_ratio(detail.get("speech_rate_score")))
    if detail.get("recency_preference"):
        condition_scores.append(_ratio(detail.get("recency_score")))

    if condition_scores:
        combined = _ratio(detail.get("combined_boost"))
        if combined > 0.0:
            return combined
        return sum(condition_scores) / len(condition_scores)

    warnings = detail.get("condition_warnings") if isinstance(detail.get("condition_warnings"), list) else []
    return _ratio(detail.get("combined_boost")) if warnings else 1.0


def _content_display_share(content_ratio: float, meaning_ratio: float) -> int:
    total = max(content_ratio, 0.0) + max(meaning_ratio, 0.0)
    if total <= 0.0:
        return CONTENT_MEANING_DISPLAY_SHARE // 2
    return int(round((max(content_ratio, 0.0) / total) * CONTENT_MEANING_DISPLAY_SHARE))


def _display_score(internal_score: float, _tier: str, detail: Optional[dict] = None) -> int:
    """내부 랭킹 점수를 사용자 표시용 추천 적합도로 변환."""
    if not detail:
        return int(round(max(0.0, min(internal_score, 1.0)) * 100))

    content_ratio = _ratio(detail.get("content_score", _ratio(detail.get("content_pct"))))
    meaning_ratio = max(
        _ratio(detail.get("vec_score")),
        _ratio(detail.get("graph_score")),
        _ratio(detail.get("sim_keyword")),
    )
    condition_ratio = _condition_score_ratio(detail)

    content_share = _content_display_share(content_ratio, meaning_ratio)
    meaning_share = CONTENT_MEANING_DISPLAY_SHARE - content_share

    score = (
        _display_component_score(content_ratio) * (content_share / 100.0)
        + _display_component_score(meaning_ratio) * (meaning_share / 100.0)
        + _display_component_score(condition_ratio) * (CONDITION_DISPLAY_SHARE / 100.0)
    )
    return int(round(score))


def _build_reason(detail: dict, tier: str = "direct") -> str:
    if tier == "related":
        parts = []
        if detail.get("dm_keyword", 0) > 0.1:
            parts.append(f"키워드 일부 포함 {detail['dm_keyword']:.0%}")
        if detail.get("sim_keyword", 0) >= 0.6:
            parts.append(f"주제 근접 {detail['sim_keyword']:.0%}")
        if detail.get("domain_score", 0) == 1.0:
            parts.append("도메인 일치")
        note = " · ".join(parts) if parts else "주변 개념 포함"
        return f"질의 주제와 직접 일치하지 않지만 관련 개념을 포함합니다 ({note})"

    parts = []
    if detail.get("domain_score", 0) == 1.0:
        parts.append("도메인 일치")
    if detail.get("depth_score", 0) > 0.3:
        parts.append(f"개념 깊이 {detail['depth_score']:.0%}")
    if detail.get("contrast_bonus", 0) > 0:
        parts.append("비교 분석형")
    if detail.get("application_preference") and detail.get("application_score", 0) > 0:
        parts.append(f"예제/시연 지향 {detail['application_score']:.0%}")
    if detail.get("listenability_preference") and detail.get("listenability_score", 0) > 0:
        parts.append(f"청취 품질 {detail['listenability_score']:.0%}")
    if detail.get("slow_speech_preference") and detail.get("speech_rate_score", 0) > 0:
        parts.append(f"발화 속도 적합 {detail['speech_rate_score']:.0%}")
    if detail.get("recency_preference") and detail.get("recency_score", 0) >= 0.6:
        parts.append("최근 업로드")
    if detail.get("sim_keyword", 0) >= 0.6:
        parts.append(f"키워드 유사도 {detail['sim_keyword']:.0%}")
    if detail.get("dm_keyword", 0) > 0.1:
        parts.append(f"키워드 직접 매칭 {detail['dm_keyword']:.0%}")
    return " · ".join(parts) if parts else "관련 강의"
