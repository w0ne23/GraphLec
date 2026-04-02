"""고정 Cypher 템플릿으로 내용형 Neo4j 조회 (키워드 + stem)."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

from .graph_constants import CONCEPT_SEMANTIC_REL_TYPES


def _run_cypher_dicts(session, cypher: str, params: dict[str, Any]) -> list[dict[str, Any]]:
    result = session.run(cypher, **params)
    return [dict(record) for record in result]


def _load_neo4j_limits() -> dict[str, int]:
    p = Path(__file__).resolve().parent / "intent_config.json"
    if p.is_file():
        with open(p, encoding="utf-8") as f:
            cfg = json.load(f)
        neo = cfg.get("neo4j_limits") or {}
        return {
            "seg_per_kw": int(neo.get("seg_per_kw", 22)),
            "slide_per_kw": int(neo.get("slide_per_kw", 10)),
            "sub_per_kw": int(neo.get("sub_per_kw", 28)),
        }
    return {"seg_per_kw": 22, "slide_per_kw": 10, "sub_per_kw": 28}


_NEO_LIM = _load_neo4j_limits()
SEG_LIM = int(os.getenv("GRAPHLEC_CONTENT_SEG_LIMIT", str(_NEO_LIM["seg_per_kw"])))
SLIDE_LIM = int(os.getenv("GRAPHLEC_CONTENT_SLIDE_LIMIT", str(_NEO_LIM["slide_per_kw"])))
SUB_LIM = int(os.getenv("GRAPHLEC_CONTENT_SUB_LIMIT", str(_NEO_LIM["sub_per_kw"])))


def _relevance_score(text: str, keywords: list[str]) -> int:
    return sum(1 for kw in keywords if kw in text)


def run_content_queries(
    session, stem: str, keywords: list[str]
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    results: dict[str, list[dict[str, Any]]] = defaultdict(list)
    raw_all: list[dict[str, Any]] = []
    seen_segs: set[tuple[Any, Any]] = set()
    seen_slides: set[Any] = set()

    q_sub = f"""
    MATCH (sub:Concept {{stem: $stem}})-[r]->(c:Concept {{stem: $stem}})
    WHERE type(r) IN $concept_rel_types
      AND toLower(coalesce(c.name,'')) CONTAINS toLower($kw)
    RETURN coalesce(sub.id,'') AS sub_id, sub.name AS sub_concept,
           coalesce(c.id,'') AS concept_id, c.name AS parent_concept
    LIMIT {SUB_LIM}
    """
    q_seg = f"""
    MATCH (slide:Slide {{stem: $stem}})-[:HAS_SCENE]->(scene:Scene {{stem: $stem}})
          -[:HAS_SEGMENT]->(seg:Segment {{stem: $stem}})-[:MENTIONS]->(c:Concept {{stem: $stem}})
    WHERE toLower(coalesce(c.name,'')) CONTAINS toLower($kw)
    RETURN coalesce(seg.text,'') AS segment_text, seg.start AS start, seg.end AS end,
           slide.slide_number AS slide_number,
           coalesce(slide.id,'') AS slide_id, coalesce(seg.id,'') AS segment_id, coalesce(c.id,'') AS concept_id,
           c.name AS concept
    ORDER BY seg.start LIMIT {SEG_LIM}
    """
    q_slide_concept = f"""
    MATCH (slide:Slide {{stem: $stem}})-[:APPEARS_IN]->(c:Concept {{stem: $stem}})
    WHERE toLower(coalesce(c.name,'')) CONTAINS toLower($kw)
    RETURN slide.slide_number AS slide_number, coalesce(slide.id,'') AS slide_id,
           slide.title AS title, slide.slide_text AS slide_text, coalesce(c.id,'') AS concept_id
    ORDER BY slide.slide_number LIMIT {SLIDE_LIM}
    """
    q_slide_text = f"""
    MATCH (slide:Slide {{stem: $stem}})
    WHERE toLower(coalesce(slide.slide_text,'')) CONTAINS toLower($kw)
    RETURN slide.slide_number AS slide_number, coalesce(slide.id,'') AS slide_id,
           slide.title AS title, slide.slide_text AS slide_text
    ORDER BY slide.slide_number LIMIT {SLIDE_LIM}
    """
    q_seg_text = f"""
    MATCH (slide:Slide {{stem: $stem}})-[:HAS_SCENE]->(scene:Scene {{stem: $stem}})
          -[:HAS_SEGMENT]->(seg:Segment {{stem: $stem}})
    WHERE toLower(coalesce(seg.text,'')) CONTAINS toLower($kw)
    RETURN coalesce(seg.text,'') AS segment_text, seg.start AS start, seg.end AS end,
           slide.slide_number AS slide_number,
           coalesce(slide.id,'') AS slide_id, coalesce(seg.id,'') AS segment_id
    ORDER BY seg.start LIMIT {SEG_LIM}
    """

    for kw in keywords:
        for key, q, needs_rel in (
            ("sub_concepts", q_sub, True),
            ("segments", q_seg, False),
            ("slides", q_slide_concept, False),
            ("slides", q_slide_text, False),
            ("segments", q_seg_text, False),
        ):
            params: dict[str, Any] = {"stem": stem, "kw": kw}
            if needs_rel:
                params["concept_rel_types"] = list(CONCEPT_SEMANTIC_REL_TYPES)
            for row in _run_cypher_dicts(session, q, params):
                raw_all.append(row)
                if key == "segments":
                    k2 = (row.get("slide_number"), row.get("start"))
                    if k2 not in seen_segs:
                        seen_segs.add(k2)
                        results["segments"].append(row)
                elif key == "slides":
                    sn = row.get("slide_number")
                    if sn not in seen_slides:
                        seen_slides.add(sn)
                        results["slides"].append(row)
                else:
                    results["sub_concepts"].append(row)

    if results.get("segments"):
        results["segments"].sort(
            key=lambda r: _relevance_score(str(r.get("segment_text", "")), keywords),
            reverse=True,
        )
    if results.get("slides"):
        results["slides"].sort(
            key=lambda r: _relevance_score(
                str(r.get("slide_text", "")) + str(r.get("title", "")), keywords
            ),
            reverse=True,
        )

    return dict(results), raw_all
