"""
graphrag_emphasis.py
GraphLec 멀티모달 강조 신호를 GraphRAGEntity에 boost 가중치로 저장.

구현 순서:
  Step 1 (현재): keyword_match — entity title vs emphasized_keywords
  Step 2: slide_propagation — 슬라이드 emphasis_score 전파
  Step 3: annotation_match — AnnotationEmphasis target/handwritten 매칭
  Step 4: audio_segment_match — stressed segment 연결
  Step 5: local boost 정규화 및 final_weight 저장
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
