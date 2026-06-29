"""
recommender/scoring.py
──────────────────────
강의별 점수 계산 함수.
"""

from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from typing import Optional

from recommender.types import LectureMetadata, LexicalStats, QueryContext
from recommender.utils import (
    _append_terms,
    _compact_term,
    _concept_match,
    _expanded_lookup_terms,
    _normalize_term,
    _term_lookup_keys,
    _tokenize_text,
    _expanded_topic_terms,
)


# ── 대조·비교형 signals ───────────────────────────────────────────────────────

_CONTRAST_TYPES = frozenset({
    "contrasts", "lacks", "differs", "vs", "versus",
    "compared_to", "unlike", "opposes", "excludes",
})

_COMPARISON_SIGNALS = frozenset({
    "차이", "비교", "vs", "versus", "차이점", "대비",
    "구분", "다른점", "차이를", "비교해", "비교한",
})


# ── 조건 감지 terms ───────────────────────────────────────────────────────────

_VISUAL_CONDITION_TERMS = frozenset({
    "그림", "도식", "도표", "시각", "시각자료", "시각 자료",
    "이미지", "다이어그램", "위주",
})

_APPLICATION_CONDITION_TERMS = frozenset({
    "예제", "예시", "사례", "실습", "시연", "데모", "적용", "활용",
    "문제풀이", "풀이", "이론만 말고",
})

_DELIVERY_CONDITION_TERMS = frozenset({
    "음질", "녹음", "청취", "듣기", "발화", "말", "빠르지", "느리",
    "천천히", "천천", "여유", "명료",
})

_RECENCY_CONDITION_TERMS = frozenset({
    "최근", "최신", "새로운", "새로", "새로 올라온", "업로드", "업데이트",
    "요즘", "근래", "최근 업로드", "최신 강의",
})

_NON_CONTENT_QUERY_TERMS = frozenset({
    "강의", "추천", "내용", "설명", "요약", "개념", "주제", "관련",
    "기초", "입문", "초급", "쉬운", "쉽게", "쉬움", "설명이 쉬운",
    "알려줘", "찾아줘", "보여줘", "내외", "이내", "이하", "정도",
})


# ── 조건 필터링 ───────────────────────────────────────────────────────────────

def _strip_non_content_modifiers(term: str, blocked: set[str]) -> str:
    cleaned = _normalize_term(term)
    for blocked_term in sorted(blocked, key=len, reverse=True):
        cleaned = cleaned.replace(blocked_term, " ")
    return re.sub(r"\s+", " ", cleaned).strip()


def _content_terms_only(terms: list[str], conditions: dict) -> list[str]:
    blocked: set[str] = set()
    if conditions.get("prefers_visual"):
        blocked.update(_VISUAL_CONDITION_TERMS)
    if conditions.get("prefers_application"):
        blocked.update(_APPLICATION_CONDITION_TERMS)
    if conditions.get("prefers_slow_speech") or conditions.get("prefers_listenability"):
        blocked.update(_DELIVERY_CONDITION_TERMS)
    if conditions.get("prefers_recency"):
        blocked.update(_RECENCY_CONDITION_TERMS)
    blocked.update(_NON_CONTENT_QUERY_TERMS)

    normalized_blocked = {_normalize_term(term) for term in blocked}
    cleaned = []
    seen = set()
    for term in terms or []:
        normalized = _normalize_term(term)
        if not normalized or normalized in normalized_blocked:
            continue
        stripped = _strip_non_content_modifiers(normalized, normalized_blocked)
        if stripped and stripped not in normalized_blocked and stripped not in seen:
            seen.add(stripped)
            cleaned.append(stripped)
    return cleaned


# ── 직접 매칭 점수 ────────────────────────────────────────────────────────────

def _direct_match_score(
    query_keywords:    list[str],
    inferred_keywords: list[str],
    lec:               LectureMetadata,
    inferred_weight:   float = 0.5,
) -> dict:
    q_tokens = _expanded_lookup_terms(query_keywords)
    i_tokens = _expanded_lookup_terms(inferred_keywords)
    all_tokens = q_tokens | i_tokens
    if not all_tokens:
        return {"title": 0.0, "keyword": 0.0, "summary": 0.0}

    title_words = set(_tokenize_text(lec.title))

    def title_hit(tokens: set) -> float:
        hits = sum(1 for t in tokens if any(_concept_match(w, t) for w in title_words))
        return hits / len(tokens) if tokens else 0.0

    title_match = (
        title_hit(q_tokens) * 1.0 +
        title_hit(i_tokens) * inferred_weight
    ) / (1.0 + inferred_weight) if (q_tokens or i_tokens) else 0.0

    total_kw_score = sum(k["score"] for k in lec.keywords) or 1.0

    def kw_matched_score(tokens: set) -> float:
        return sum(
            k["score"] for k in lec.keywords
            if any(
                _concept_match(str(k.get("keyword", "")), t)
                for t in tokens
                if len(t) > 1
            )
        )

    q_kw = kw_matched_score(q_tokens)
    i_kw = kw_matched_score(i_tokens)
    kw_match = (q_kw * 1.0 + i_kw * inferred_weight) / (total_kw_score * (1.0 + inferred_weight))

    def summary_hit(tokens: set) -> float:
        summary_terms = _tokenize_text(lec.summary)
        hits = sum(1 for t in tokens if any(_concept_match(term, t) for term in summary_terms))
        return hits / len(tokens) if tokens else 0.0

    sum_match = (
        summary_hit(q_tokens) * 1.0 +
        summary_hit(i_tokens) * inferred_weight
    ) / (1.0 + inferred_weight) if (q_tokens or i_tokens) else 0.0

    return {
        "title":        title_match,
        "keyword":      kw_match,
        "summary":      sum_match,
        "q_kw_matched": (not q_tokens) or q_kw > 0,
    }


def _direct_match_score_tfirf(
    query_keywords:    list[str],
    inferred_keywords: list[str],
    lec:               LectureMetadata,
    stats:             LexicalStats,
    idf_floor:         float = 0.05,
    inferred_weight:   float = 0.5,
) -> dict:
    base = _direct_match_score(
        query_keywords,
        inferred_keywords,
        lec,
        inferred_weight=inferred_weight,
    )

    q_tokens = _expanded_lookup_terms(query_keywords)
    i_tokens = _expanded_lookup_terms(inferred_keywords)
    if not (q_tokens or i_tokens):
        base["keyword_legacy"] = base["keyword"]
        return base

    keyword_weights = []
    for item in lec.keywords or []:
        keyword = _normalize_term(item.get("keyword", ""))
        if not keyword:
            continue
        try:
            score = float(item.get("score", 1.0))
        except (TypeError, ValueError):
            score = 1.0
        irf = max(stats.irf.get(keyword, 0.0), idf_floor)
        keyword_weights.append((keyword, max(score, 0.0) * irf))

    total_weight = sum(weight for _, weight in keyword_weights) or 1.0

    def matched_weight(tokens: set[str]) -> float:
        return sum(
            weight
            for keyword, weight in keyword_weights
            if any(
                _concept_match(keyword, token)
                for token in tokens
                if len(token) > 1
            )
        )

    q_kw = matched_weight(q_tokens)
    i_kw = matched_weight(i_tokens)
    tfirf_keyword = (
        q_kw * 1.0 + i_kw * inferred_weight
    ) / (total_weight * (1.0 + inferred_weight))

    base["keyword_legacy"] = base["keyword"]
    base["keyword"] = tfirf_keyword
    base["q_kw_matched"] = (not q_tokens) or q_kw > 0
    return base


# ── 개념 역할 매칭 ────────────────────────────────────────────────────────────

def _role_weight_for_concept(
    lec: LectureMetadata,
    concept: str,
    core_weight: float = 1.0,
    intro_weight: float = 0.35,
) -> float:
    if not concept:
        return 0.0

    role_weights = {
        "core": core_weight,
        "introduced": intro_weight,
        "prerequisite": 0.60,
    }
    concept_roles = lec.concept_roles
    best = 0.0

    if isinstance(concept_roles, dict):
        for role, concepts in concept_roles.items():
            weight = role_weights.get(role, 0.0)
            for role_concept in (concepts or []):
                if _concept_match(str(role_concept), concept):
                    best = max(best, weight)
    elif isinstance(concept_roles, list):
        for cr in (concept_roles or []):
            if not isinstance(cr, dict):
                continue
            if _concept_match(str(cr.get("concept", "")), concept):
                best = max(best, role_weights.get(cr.get("role", ""), 0.0))

    return best


def _query_concept_in_role(
    lec: LectureMetadata,
    query_concepts: set[str],
    target_role: str,
) -> bool:
    concepts = {
        _normalize_term(concept)
        for concept in query_concepts
        if _normalize_term(concept)
    }
    if not concepts:
        return False

    role_terms: list[str] = []
    concept_roles = lec.concept_roles
    if isinstance(concept_roles, dict):
        _append_terms(role_terms, concept_roles.get(target_role, []))
    elif isinstance(concept_roles, list):
        for item in concept_roles:
            if not isinstance(item, dict) or item.get("role") != target_role:
                continue
            _append_terms(role_terms, item.get("concept"))

    normalized_roles = [
        _normalize_term(term)
        for term in role_terms
        if _normalize_term(term)
    ]
    return any(
        _concept_match(role_term, query_term)
        for role_term in normalized_roles
        for query_term in concepts
    )


# ── 그래프 점수 ───────────────────────────────────────────────────────────────

def _compute_graph_score(
    lec: LectureMetadata,
    query_concepts: set[str],
    core_weight: float = 1.0,
    intro_weight: float = 0.35,
) -> float:
    query_concepts = {c for c in query_concepts if c}
    if not query_concepts:
        return 0.0

    role_hits = [
        _role_weight_for_concept(lec, concept, core_weight, intro_weight)
        for concept in query_concepts
    ]
    role_score = sum(role_hits) / len(query_concepts)

    graph: dict[str, list[tuple[str, float]]] = {}
    for rel in (lec.concept_relations or []):
        src = rel.get("from", rel.get("source", ""))
        dst = rel.get("to", rel.get("target", ""))
        weight = float(rel.get("weight") or 1.0)
        if src and dst:
            graph.setdefault(src, []).append((dst, weight))
            graph.setdefault(dst, []).append((src, weight))

    relation_scores = []
    for node, neighbor_weights in graph.items():
        if not any(_concept_match(node, qc) for qc in query_concepts):
            continue
        if not neighbor_weights:
            continue
        total_weight = sum(w for _, w in neighbor_weights) or 1.0
        related_weight = sum(
            w
            for neighbor, w in neighbor_weights
            if any(_concept_match(neighbor, qc) for qc in query_concepts)
        )
        relation_scores.append(related_weight / total_weight)

    relation_score = (
        sum(relation_scores) / len(relation_scores)
        if relation_scores else 0.0
    )

    return round(min(role_score * 0.65 + relation_score * 0.35, 1.0), 4)


def _compute_depth_score(focus_concept: str, target: LectureMetadata) -> float:
    """
    focus_concept이 강의에서 얼마나 깊이 다뤄지는지 추정.

    v2: concept_relations로 로컬 그래프를 구성하고 BFS 홉 거리 기반 점수 계산.
      - focus_concept이 그래프 내 노드면 BFS로 인접 개념 분포를 측정
      - 그래프에 없으면 keyword 기반 fallback
      가중합: role×0.5 + hop×0.5
    """
    if not focus_concept:
        return 0.0

    # ── 1. role 점수 (concept_roles 기반) ──────────────────────────
    role_score    = 0.0
    concept_roles = target.concept_roles
    if isinstance(concept_roles, dict):
        ROLE_WEIGHTS = {"core": 1.0, "introduced": 0.1}
        for role, concepts in concept_roles.items():
            weight = ROLE_WEIGHTS.get(role, 0.0)
            for concept in (concepts or []):
                if _concept_match(concept, focus_concept):
                    role_score = max(role_score, weight)
    else:
        for cr in (concept_roles or []):
            if not isinstance(cr, dict):
                continue
            if _concept_match(cr.get("concept", ""), focus_concept):
                role = cr.get("role", "")
                if role == "core":
                    role_score = 1.0
                elif role == "introduced":
                    role_score = max(role_score, 0.1)

    # ── 2. 로컬 그래프 구성 (weighted) ─────────────────────────────
    graph: dict[str, list[tuple[str, float]]] = {}
    for rel in (target.concept_relations or []):
        src = rel.get("from", rel.get("source", ""))
        dst = rel.get("to",   rel.get("target", ""))
        w_raw = float(rel.get("weight") or 1.0)
        weight = w_raw / (1.0 + w_raw)  # co-occurrence count → (0, 1) 정규화
        if src and dst:
            graph.setdefault(src, []).append((dst, weight))
            graph.setdefault(dst, []).append((src, weight))

    focus_node = next(
        (n for n in graph if _concept_match(n, focus_concept)),
        None
    )

    # ── 3. BFS 홉 거리 점수 (weighted) ─────────────────────────────
    if focus_node:
        dist_map: dict[str, int] = {focus_node: 0}
        weight_map: dict[str, float] = {focus_node: 1.0}
        frontier = [focus_node]
        for _ in range(3):
            next_frontier = []
            for node in frontier:
                for neighbor, w in graph.get(node, []):
                    if neighbor and neighbor not in dist_map:
                        dist_map[neighbor] = dist_map[node] + 1
                        weight_map[neighbor] = min(weight_map[node], w)
                        next_frontier.append(neighbor)
            frontier = next_frontier
            if not frontier:
                break

        # 거리 d, 경로 weight w인 노드 기여: w / (d+1)
        # 정규화 기준: 5개 1홉 노드가 모두 weight=1 → hop_score=1.0
        hop_score = sum(
            weight_map.get(n, 1.0) / (d + 1)
            for n, d in dist_map.items()
            if n != focus_node and d > 0
        )
        hop_score = min(hop_score / 5.0, 1.0)
    else:
        sub_kw_count = sum(
            1 for k in target.keywords
            if focus_concept in k["keyword"] and k["keyword"] != focus_concept
        )
        hop_score = min(sub_kw_count / 3.0, 1.0)

    depth = role_score * 0.5 + hop_score * 0.5
    return round(depth, 4)


# ── 대조 / 재현도 / 파편화 ────────────────────────────────────────────────────

def _compute_contrast_signal(lec: LectureMetadata) -> float:
    """
    강의 내 대조·비교형 concept_relations 비율.
    반환: 0.0(비교 요소 없음) ~ 1.0(전체가 대조 관계)
    """
    relations = lec.concept_relations or []
    if not relations:
        return 0.0
    total_weight = sum(float(rel.get("weight") or 1.0) for rel in relations) or 1.0
    contrast_weight = sum(
        float(rel.get("weight") or 1.0)
        for rel in relations
        if rel.get("type", "").lower() in _CONTRAST_TYPES
    )
    return round(min(contrast_weight / total_weight, 1.0), 4)


def _detect_comparison_intent(query_keywords: list[str], query: str) -> bool:
    q = query.lower()
    return (
        any(sig in q for sig in _COMPARISON_SIGNALS) or
        any(kw in _COMPARISON_SIGNALS for kw in (query_keywords or []))
    )


def _compute_fragmentation_penalty(concept_roles) -> float:
    """
    core 비율 낮고 introduced 비율 높을수록 패널티.
    반환: 0.0(응집) ~ 1.0(파편화)
    """
    if isinstance(concept_roles, dict):
        n_core  = len(concept_roles.get("core", []))
        n_intro = len(concept_roles.get("introduced", []))
    elif isinstance(concept_roles, list):
        n_core  = sum(1 for cr in concept_roles if isinstance(cr, dict) and cr.get("role") == "core")
        n_intro = sum(1 for cr in concept_roles if isinstance(cr, dict) and cr.get("role") == "introduced")
    else:
        return 0.0

    total = n_core + n_intro
    if total == 0:
        return 0.0

    core_ratio  = n_core  / total
    intro_ratio = n_intro / total
    return float(min(intro_ratio * (1.0 - core_ratio), 1.0))


# ── subject 매칭 ─────────────────────────────────────────────────────────────

def _required_subject_terms(ctx: QueryContext) -> set[str]:
    if ctx.intent != "recommend":
        return set()
    return {
        _normalize_term(term)
        for term in _expanded_topic_terms(ctx.query_keywords)
        if _normalize_term(term)
    }


def _lecture_subject_terms(lec: LectureMetadata) -> list[str]:
    terms: list[str] = []
    _append_terms(terms, lec.title)
    _append_terms(terms, lec.summary)
    _append_terms(terms, [item.get("keyword", "") for item in (lec.keywords or []) if isinstance(item, dict)])
    _append_terms(terms, lec.concept_roles)
    for relation in lec.concept_relations or []:
        if not isinstance(relation, dict):
            continue
        _append_terms(terms, relation.get("from"))
        _append_terms(terms, relation.get("to"))
        _append_terms(terms, relation.get("source"))
        _append_terms(terms, relation.get("target"))
    for community in lec.communities or []:
        if not isinstance(community, dict):
            continue
        _append_terms(terms, community.get("title"))
        _append_terms(terms, community.get("summary"))
        _append_terms(terms, community.get("nodes"))
    return [_normalize_term(term) for term in terms if _normalize_term(term)]


def _required_subject_match_type(lec: LectureMetadata, required_terms: set[str]) -> str:
    if not required_terms:
        return "not_required"
    lecture_terms = _lecture_subject_terms(lec)
    if not lecture_terms:
        return "none"

    lecture_keys = set()
    for term in lecture_terms:
        lecture_keys.update(_term_lookup_keys(term))
    required_keys = set()
    for term in required_terms:
        required_keys.update(_term_lookup_keys(term))

    if lecture_keys & required_keys:
        return "canonical"
    if any(
        _concept_match(lecture_term, required_term)
        for lecture_term in lecture_terms
        for required_term in required_terms
    ):
        return "partial"
    return "none"


# ── 재현도 (recency) ──────────────────────────────────────────────────────────

def _parse_uploaded_at(uploaded_at: Optional[str]) -> Optional[datetime]:
    if not uploaded_at:
        return None
    text = str(uploaded_at).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _compute_recency_score(
    uploaded_at: Optional[str],
    half_life_days: float,
    now: Optional[datetime] = None,
) -> float:
    uploaded_dt = _parse_uploaded_at(uploaded_at)
    if uploaded_dt is None:
        return 0.0
    now_dt = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    age_days = max((now_dt - uploaded_dt).total_seconds() / 86400.0, 0.0)
    half_life = max(float(half_life_days or 1.0), 1.0)
    return round(math.exp(-age_days * math.log(2) / half_life), 4)
