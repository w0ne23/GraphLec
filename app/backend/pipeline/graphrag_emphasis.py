"""
graphrag_emphasis.py
GraphLec 멀티모달 강조 신호를 GraphRAGEntity에 boost 가중치로 저장.

구현 순서:
  Step 1: keyword_match — entity title vs slide_text_score 기반 키워드 (순수 텍스트 신호)
  Step 2: visual_match + annotation_match
    2a. visual_match     — visual_score 기반 키워드 매칭 (시각 채널 독립 신호)
    2b. annotation_match — AnnotationEmphasis target/handwritten 직접 매칭
  Step 3: audio_segment_match — stressed segment 비율 기반 (stressed_count / total)
  Step 4: local boost 정규화 및 final_weight 저장 (breakdown + raw 포함)
  Step 5: relation_boost — GRAPHRAG_RELATES_TO 엣지에 endpoint boost 기반 가중치 부여
  [미구현] Slide/Scene/Context 단위 분리 — 스키마 확정 후 별도 처리
  [보류]   global boost — 추후 별도 배치
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

_EMPHASIS_WEIGHTS = {
    "keyword":    1.0,
    "visual":     1.0,
    "annotation": 1.5,
    "audio":      1.0,
}
_TOTAL_WEIGHT = sum(_EMPHASIS_WEIGHTS.values())  # 4.5

# audio_match: total_segments가 이 값보다 작으면 분모를 고정해 소수 세그먼트 비율 팽창 방지
_MIN_SEGMENT_DENOM = 3


def _kw_text_score(entry: dict[str, Any]) -> float:
    """
    텍스트 채널 신호만 반환 — audio/visual/annotation 중복 집계 방지.
    우선순위:
      1. slide_text_score 있으면 그 값 사용 (sub-score와 무관한 독립 텍스트 신호)
      2. slide_text_score 없고 sub-score(audio/visual/annotation)도 없으면 raw score 사용
      3. slide_text_score 없고 sub-score 있으면 0 반환 (해당 채널이 Step 2/3에서 처리)
    """
    sts = entry.get("slide_text_score")
    if sts is not None:
        try:
            return float(sts)
        except (TypeError, ValueError):
            return 0.0
    has_sub = any(
        entry.get(f) for f in ("audio_score", "visual_score", "annotation_score")
    )
    if has_sub:
        return 0.0
    try:
        return float(entry.get("score") or 0)
    except (TypeError, ValueError):
        return 0.0


def _norm(text: Any) -> str:
    return re.sub(r"\s+", "", str(text or "").lower())


def _parse_json_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(x) for x in value]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
        except json.JSONDecodeError:
            pass
    return []


# ── Step 1 ───────────────────────────────────────────────────────────────────

def _build_slide_keyword_map(fused: dict[str, Any]) -> dict[str, list[tuple[str, float]]]:
    """slide_id -> [(keyword_norm, slide_text_score)]"""
    result: dict[str, list[tuple[str, float]]] = {}
    for slide in fused.get("slides") or []:
        sid = str(slide.get("slide_id") or "").strip()
        if not sid:
            continue
        kws: list[tuple[str, float]] = []
        for kw in slide.get("emphasized_keywords") or []:
            text = (kw.get("keyword") or kw.get("text") or kw.get("name") or "").strip()
            if not text:
                continue
            score = _kw_text_score(kw)
            if score <= 0:
                continue
            kws.append((_norm(text), score))
        if kws:
            result[sid] = kws
    return result


def compute_keyword_match(
    session,
    stem: str,
    fused_path: Path,
) -> dict[str, Any]:
    """
    Step 1: entity title/description vs slide_text_score 기반 키워드.
    순수 텍스트 신호만 사용해 audio/visual/annotation 채널과 중복 방지.
    Writes emphasis_keyword_match to GraphRAGEntity nodes.
    """
    with fused_path.open(encoding="utf-8") as f:
        fused = json.load(f)

    slide_kw_map = _build_slide_keyword_map(fused)

    records = session.run(
        """
        MATCH (e:GraphRAGEntity {stem: $stem})
        RETURN e.id AS id, e.title AS title, e.description AS description, e.slide_ids AS slide_ids
        """,
        stem=stem,
    ).data()

    updates: list[dict[str, Any]] = []
    for rec in records:
        title_norm = _norm(rec.get("title") or "")
        desc_norm = _norm(rec.get("description") or "")
        slide_ids = _parse_json_list(rec.get("slide_ids"))

        kw_total = 0.0
        matched_kws: set[str] = set()
        for sid in slide_ids:
            for kw_norm, ks in slide_kw_map.get(sid, []):
                if not kw_norm:
                    continue
                if kw_norm in title_norm or kw_norm in desc_norm or title_norm in kw_norm:
                    kw_total += ks
                    matched_kws.add(kw_norm)

        updates.append({
            "id":               rec["id"],
            "keyword_match":    round(kw_total, 4),
            "matched_keywords": sorted(matched_kws),
        })

    session.run(
        """
        UNWIND $rows AS row
        MATCH (e:GraphRAGEntity {stem: $stem, id: row.id})
        SET e.emphasis_keyword_match    = row.keyword_match,
            e.emphasis_matched_keywords = row.matched_keywords
        """,
        stem=stem,
        rows=updates,
    )

    nonzero = sum(1 for u in updates if u["keyword_match"] > 0)
    return {
        "keyword_match_entities": len(updates),
        "keyword_match_nonzero":  nonzero,
    }


# ── Step 2a ──────────────────────────────────────────────────────────────────

def _build_slide_visual_kw_map(fused: dict[str, Any]) -> dict[str, list[tuple[str, float]]]:
    """slide_id -> [(keyword_norm, visual_score)]  — visual_score > 0 인 항목만"""
    result: dict[str, list[tuple[str, float]]] = {}
    for slide in fused.get("slides") or []:
        sid = str(slide.get("slide_id") or "").strip()
        if not sid:
            continue
        kws: list[tuple[str, float]] = []
        for kw in slide.get("emphasized_keywords") or []:
            text = (kw.get("keyword") or kw.get("text") or kw.get("name") or "").strip()
            if not text:
                continue
            try:
                vs = float(kw.get("visual_score") or 0)
            except (TypeError, ValueError):
                vs = 0.0
            if vs <= 0:
                continue
            kws.append((_norm(text), vs))
        if kws:
            result[sid] = kws
    return result


def compute_visual_match(
    session,
    stem: str,
    fused_path: Path,
) -> dict[str, Any]:
    """
    Step 2a: entity title/description vs visual_score keywords.
    시각 채널(슬라이드 텍스트·필기·하이라이트)에서 강조된 키워드만 사용.
    Writes emphasis_visual_match to GraphRAGEntity nodes.
    """
    with fused_path.open(encoding="utf-8") as f:
        fused = json.load(f)

    slide_kw_map = _build_slide_visual_kw_map(fused)

    records = session.run(
        """
        MATCH (e:GraphRAGEntity {stem: $stem})
        RETURN e.id AS id, e.title AS title, e.description AS description, e.slide_ids AS slide_ids
        """,
        stem=stem,
    ).data()

    updates: list[dict[str, Any]] = []
    for rec in records:
        title_norm = _norm(rec.get("title") or "")
        desc_norm = _norm(rec.get("description") or "")
        slide_ids = _parse_json_list(rec.get("slide_ids"))

        visual_total = 0.0
        matched_kws: set[str] = set()
        for sid in slide_ids:
            for kw_norm, vs in slide_kw_map.get(sid, []):
                if kw_norm in title_norm or kw_norm in desc_norm or title_norm in kw_norm:
                    visual_total += vs
                    matched_kws.add(kw_norm)

        updates.append({
            "id":               rec["id"],
            "visual_match":     round(visual_total, 4),
            "matched_keywords": sorted(matched_kws),
        })

    session.run(
        """
        UNWIND $rows AS row
        MATCH (e:GraphRAGEntity {stem: $stem, id: row.id})
        SET e.emphasis_visual_match    = row.visual_match,
            e.emphasis_visual_keywords = row.matched_keywords
        """,
        stem=stem,
        rows=updates,
    )

    nonzero = sum(1 for u in updates if u["visual_match"] > 0)
    return {
        "visual_match_entities": len(updates),
        "visual_match_nonzero":  nonzero,
    }


# ── Step 2b ──────────────────────────────────────────────────────────────────

def compute_annotation_match(
    session,
    stem: str,
) -> dict[str, Any]:
    """
    Step 2b: entity title/description vs AnnotationEmphasis target_content / handwritten_content.
    Slide 노드의 HAS_ANNOTATION 엣지를 통해 해당 강의의 모든 주석을 일괄 조회.
    Writes emphasis_annotation_match to GraphRAGEntity nodes.
    """
    ann_records = session.run(
        """
        MATCH (s:Slide {stem: $stem})-[:HAS_ANNOTATION]->(a:AnnotationEmphasis)
        RETURN s.id AS slide_id,
               a.target_content      AS target,
               a.handwritten_content AS handwritten,
               a.score               AS score
        """,
        stem=stem,
    ).data()

    # slide_id → [(norm_text, score)]
    slide_ann_map: dict[str, list[tuple[str, float]]] = {}
    for rec in ann_records:
        sid = str(rec.get("slide_id") or "").strip()
        if not sid:
            continue
        try:
            score = float(rec.get("score") or 0)
        except (TypeError, ValueError):
            score = 0.0
        for raw in (rec.get("target"), rec.get("handwritten")):
            if not raw:
                continue
            norm = _norm(raw)
            if norm:
                slide_ann_map.setdefault(sid, []).append((norm, score))

    entity_records = session.run(
        """
        MATCH (e:GraphRAGEntity {stem: $stem})
        RETURN e.id AS id, e.title AS title, e.description AS description, e.slide_ids AS slide_ids
        """,
        stem=stem,
    ).data()

    updates: list[dict[str, Any]] = []
    for rec in entity_records:
        title_norm = _norm(rec.get("title") or "")
        desc_norm = _norm(rec.get("description") or "")
        slide_ids = _parse_json_list(rec.get("slide_ids"))

        ann_total = 0.0
        for sid in slide_ids:
            for ann_norm, score in slide_ann_map.get(sid, []):
                if ann_norm in title_norm or title_norm in ann_norm or ann_norm in desc_norm:
                    ann_total += score

        updates.append({
            "id":               rec["id"],
            "annotation_match": round(ann_total, 4),
        })

    session.run(
        """
        UNWIND $rows AS row
        MATCH (e:GraphRAGEntity {stem: $stem, id: row.id})
        SET e.emphasis_annotation_match = row.annotation_match
        """,
        stem=stem,
        rows=updates,
    )

    nonzero = sum(1 for u in updates if u["annotation_match"] > 0)
    return {
        "annotation_match_entities": len(updates),
        "annotation_match_nonzero":  nonzero,
    }


# ── Step 3 ───────────────────────────────────────────────────────────────────

def compute_audio_segment_match(
    session,
    stem: str,
) -> dict[str, Any]:
    """
    Step 3: entity와 GRAPHRAG_MENTIONED_IN_SEGMENT로 연결된 Segment 중
    segment.stressed=true 이거나 그 부모 Context.stressed=true 인 비율을 집계.

    audio_match = stressed_count / max(total_segments, _MIN_SEGMENT_DENOM)
      - _MIN_SEGMENT_DENOM: 세그먼트 수 적은 entity의 비율 팽창 방지
    Writes emphasis_audio_match, emphasis_audio_stressed_count, emphasis_audio_total_segments.
    """
    records = session.run(
        """
        MATCH (e:GraphRAGEntity {stem: $stem})
        OPTIONAL MATCH (e)-[:GRAPHRAG_MENTIONED_IN_SEGMENT]->(seg:Segment {stem: $stem})
        OPTIONAL MATCH (ctx:Context {stem: $stem})-[:HAS_SEGMENT]->(seg)
        WITH e.id AS id,
             count(
               CASE WHEN seg IS NOT NULL AND (seg.stressed = true OR ctx.stressed = true)
                    THEN 1 ELSE null END
             ) AS stressed_count,
             count(seg) AS total_segments
        RETURN id, stressed_count, total_segments
        """,
        stem=stem,
    ).data()

    updates: list[dict[str, Any]] = []
    for rec in records:
        stressed = rec["stressed_count"]
        total = rec["total_segments"]
        ratio = round(stressed / max(total, _MIN_SEGMENT_DENOM), 4)
        updates.append({
            "id":             rec["id"],
            "audio_match":    ratio,
            "stressed_count": stressed,
            "total_segments": total,
        })

    session.run(
        """
        UNWIND $rows AS row
        MATCH (e:GraphRAGEntity {stem: $stem, id: row.id})
        SET e.emphasis_audio_match          = row.audio_match,
            e.emphasis_audio_stressed_count = row.stressed_count,
            e.emphasis_audio_total_segments = row.total_segments
        """,
        stem=stem,
        rows=updates,
    )

    nonzero = sum(1 for u in updates if u["audio_match"] > 0)
    return {
        "audio_match_entities": len(updates),
        "audio_match_nonzero":  nonzero,
    }


# ── Step 4 ───────────────────────────────────────────────────────────────────

def compute_final_weight(
    session,
    stem: str,
) -> dict[str, Any]:
    """
    Step 4: 강의 내 로컬 정규화 → emphasis_boost_local → final_weight 저장.

    각 신호를 강의 최댓값으로 나눠 [0, 1] 정규화 후 가중 평균:
      emphasis_boost_local = Σ(weight_i * norm_i) / total_weight  →  [0, 1]

    graphrag_importance = degree  (GraphRAG 구조적 중요도 base)
    final_weight = graphrag_importance * (1 + emphasis_boost_local)

    저장 프로퍼티:
      graphrag_importance       — degree 기반 GraphRAG 구조 중요도
      emphasis_raw              — 정규화 전 신호값 JSON string {kw, vis, ann, aud}
      emphasis_boost_breakdown  — 채널별 가중 기여도 JSON string {keyword, visual, annotation, audio}
      emphasis_sources          — 비-zero 기여 채널 목록 (Neo4j list property, Cypher IN 조회 가능)
      emphasis_boost_local      — 최종 combined boost [0, 1]
      emphasis_boost_global     — null 초기화 (추후 별도 배치로 주입; 배치 구현 시
                                  이 SET을 제거하고 COALESCE 또는 외부 주입 방식으로 교체)
      final_weight              — graphrag_importance * (1 + boost)
    """
    records = session.run(
        """
        MATCH (e:GraphRAGEntity {stem: $stem})
        RETURN e.id AS id,
               COALESCE(e.emphasis_keyword_match,    0.0) AS kw,
               COALESCE(e.emphasis_visual_match,     0.0) AS vis,
               COALESCE(e.emphasis_annotation_match, 0.0) AS ann,
               COALESCE(e.emphasis_audio_match,      0.0) AS aud,
               COALESCE(e.degree, 1)                      AS degree
        """,
        stem=stem,
    ).data()

    if not records:
        return {"final_weight_entities": 0, "final_weight_nonzero": 0}

    max_kw  = max((r["kw"]  for r in records), default=0.0) or 1.0
    max_vis = max((r["vis"] for r in records), default=0.0) or 1.0
    max_ann = max((r["ann"] for r in records), default=0.0) or 1.0
    # audio는 이미 [0, 1] 비율 — 재정규화 없이 절대 강조 강도 보존
    max_aud = 1.0

    w = _EMPHASIS_WEIGHTS
    updates: list[dict[str, Any]] = []
    for rec in records:
        breakdown = {
            "keyword":    round(w["keyword"]    * (rec["kw"]  / max_kw)  / _TOTAL_WEIGHT, 4),
            "visual":     round(w["visual"]     * (rec["vis"] / max_vis) / _TOTAL_WEIGHT, 4),
            "annotation": round(w["annotation"] * (rec["ann"] / max_ann) / _TOTAL_WEIGHT, 4),
            "audio":      round(w["audio"]      * (rec["aud"] / max_aud) / _TOTAL_WEIGHT, 4),
        }
        boost = round(sum(breakdown.values()), 4)
        graphrag_importance = int(rec["degree"])

        updates.append({
            "id":                  rec["id"],
            "graphrag_importance": graphrag_importance,
            "raw":                 json.dumps(
                                       {"kw":  round(rec["kw"],  4),
                                        "vis": round(rec["vis"], 4),
                                        "ann": round(rec["ann"], 4),
                                        "aud": round(rec["aud"], 4)},
                                       ensure_ascii=False,
                                   ),
            "breakdown":           json.dumps(breakdown, ensure_ascii=False),
            "sources":             [ch for ch, v in breakdown.items() if v > 0],
            "boost":               boost,
            "final_weight":        round(graphrag_importance * (1.0 + boost), 4),
        })

    session.run(
        """
        UNWIND $rows AS row
        MATCH (e:GraphRAGEntity {stem: $stem, id: row.id})
        SET e.graphrag_importance      = row.graphrag_importance,
            e.emphasis_raw             = row.raw,
            e.emphasis_boost_breakdown = row.breakdown,
            e.emphasis_sources         = row.sources,
            e.emphasis_boost_local     = row.boost,
            e.emphasis_boost_global    = null,
            e.final_weight             = row.final_weight
        """,
        stem=stem,
        rows=updates,
    )

    nonzero = sum(1 for u in updates if u["boost"] > 0)
    return {
        "final_weight_entities": len(updates),
        "final_weight_nonzero":  nonzero,
    }


# ── Step 5 ───────────────────────────────────────────────────────────────────

def compute_relation_boost(
    session,
    stem: str,
) -> dict[str, Any]:
    """
    Step 5: GRAPHRAG_RELATES_TO 엣지에 endpoint_boost 저장 (마지막 단계).

    endpoint_boost = 0.7 * max(src_boost, tgt_boost) + 0.3 * min(src_boost, tgt_boost)
      - 한쪽 핵심 개념에 연결된 관계도 중요하다고 보는 강의 맥락 반영
    emphasis_edge_weight = r.weight * (1 + endpoint_boost)
      - 원본 weight 보존, emphasis_edge_weight를 별도 프로퍼티로 추가
    """
    result = session.run(
        """
        MATCH (src:GraphRAGEntity {stem: $stem})-[r:GRAPHRAG_RELATES_TO]->(tgt:GraphRAGEntity {stem: $stem})
        WHERE src.emphasis_boost_local IS NOT NULL AND tgt.emphasis_boost_local IS NOT NULL
        WITH r,
             CASE WHEN src.emphasis_boost_local >= tgt.emphasis_boost_local
                  THEN src.emphasis_boost_local ELSE tgt.emphasis_boost_local END AS max_b,
             CASE WHEN src.emphasis_boost_local <= tgt.emphasis_boost_local
                  THEN src.emphasis_boost_local ELSE tgt.emphasis_boost_local END AS min_b,
             COALESCE(r.weight, 1.0) AS base_w
        SET r.emphasis_edge_weight = round(base_w * (1 + 0.7 * max_b + 0.3 * min_b), 4)
        RETURN count(r) AS updated
        """,
        stem=stem,
    ).single()

    updated = result["updated"] if result else 0
    return {"relation_boost_edges": updated}
