"""
graphrag_emphasis.py
GraphLec 멀티모달 강조 신호를 GraphRAGEntity에 boost 가중치로 저장.

구현 순서:
  Step 1: keyword_match — entity title vs emphasized_keywords (전체 score)
  Step 2: visual_match + annotation_match
    2a. visual_match     — visual_score 기반 키워드 매칭 (시각 강조 신호)
    2b. annotation_match — AnnotationEmphasis target/handwritten 매칭
  Step 3: audio_segment_match — stressed segment 연결
  Step 4: local boost 정규화 및 final_weight 저장
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

_SCORE_FIELDS = (
    "score",
    "audio_score",
    "visual_score",
    "annotation_score",
    "slide_text_score",
)


def _kw_score(entry: dict[str, Any]) -> float:
    if "score" in entry:
        try:
            return float(entry["score"])
        except (TypeError, ValueError):
            return 0.0
    return sum(float(entry.get(f) or 0) for f in _SCORE_FIELDS if f != "score")


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


def _build_slide_keyword_map(fused: dict[str, Any]) -> dict[str, list[tuple[str, float]]]:
    """slide_id -> [(keyword_norm, score)]"""
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
            kws.append((_norm(text), _kw_score(kw)))
        if kws:
            result[sid] = kws
    return result


def compute_keyword_match(
    session,
    stem: str,
    fused_path: Path,
) -> dict[str, Any]:
    """
    Step 1: entity title/description vs slide emphasized_keywords.
    Writes emphasis_keyword_match property to GraphRAGEntity nodes.
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
            "id": rec["id"],
            "keyword_match": round(kw_total, 4),
            "matched_keywords": sorted(matched_kws),
        })

    session.run(
        """
        UNWIND $rows AS row
        MATCH (e:GraphRAGEntity {stem: $stem, id: row.id})
        SET e.emphasis_keyword_match = row.keyword_match,
            e.emphasis_matched_keywords = row.matched_keywords
        """,
        stem=stem,
        rows=updates,
    )

    nonzero = sum(1 for u in updates if u["keyword_match"] > 0)
    return {
        "keyword_match_entities": len(updates),
        "keyword_match_nonzero": nonzero,
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
    슬라이드 텍스트·필기 등 시각 채널에서 강조된 키워드만 사용.
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
            "id": rec["id"],
            "visual_match": round(visual_total, 4),
            "matched_keywords": sorted(matched_kws),
        })

    session.run(
        """
        UNWIND $rows AS row
        MATCH (e:GraphRAGEntity {stem: $stem, id: row.id})
        SET e.emphasis_visual_match = row.visual_match,
            e.emphasis_visual_keywords = row.matched_keywords
        """,
        stem=stem,
        rows=updates,
    )

    nonzero = sum(1 for u in updates if u["visual_match"] > 0)
    return {
        "visual_match_entities": len(updates),
        "visual_match_nonzero": nonzero,
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
               a.target_content AS target,
               a.handwritten_content AS handwritten,
               a.score AS score
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
            "id": rec["id"],
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
        "annotation_match_nonzero": nonzero,
    }
