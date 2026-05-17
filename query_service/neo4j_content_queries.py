"""고정 Cypher 템플릿으로 내용형 Neo4j 조회 (키워드 + stem)."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

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
            "graphrag_entity_per_kw": int(neo.get("graphrag_entity_per_kw", 12)),
            "graphrag_rel_per_kw": int(neo.get("graphrag_rel_per_kw", 12)),
        }
    return {
        "seg_per_kw": 22,
        "slide_per_kw": 10,
        "sub_per_kw": 28,
        "graphrag_entity_per_kw": 12,
        "graphrag_rel_per_kw": 12,
    }


_NEO_LIM = _load_neo4j_limits()
SEG_LIM = int(os.getenv("GRAPHLEC_CONTENT_SEG_LIMIT", str(_NEO_LIM["seg_per_kw"])))
SLIDE_LIM = int(os.getenv("GRAPHLEC_CONTENT_SLIDE_LIMIT", str(_NEO_LIM["slide_per_kw"])))
SUB_LIM = int(os.getenv("GRAPHLEC_CONTENT_SUB_LIMIT", str(_NEO_LIM["sub_per_kw"])))
GR_ENTITY_LIM = int(os.getenv("GRAPHLEC_GRAPHRAG_ENTITY_LIMIT", str(_NEO_LIM["graphrag_entity_per_kw"])))
GR_REL_LIM = int(os.getenv("GRAPHLEC_GRAPHRAG_REL_LIMIT", str(_NEO_LIM["graphrag_rel_per_kw"])))


def _relevance_score(text: str, keywords: list[str]) -> int:
    t = text.lower()
    return sum(1 for kw in keywords if kw.lower() in t)


def run_content_queries(
    session, stem: str, keywords: list[str]
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    results: dict[str, list[dict[str, Any]]] = defaultdict(list)
    raw_all: list[dict[str, Any]] = []
    seen_segs: set[tuple[Any, Any]] = set()
    seen_slides: set[Any] = set()
    seen_gr_entities: set[Any] = set()
    seen_gr_rels: set[tuple[Any, Any, Any]] = set()

    q_slide_text = f"""
    MATCH (slide:Slide {{stem: $stem}})
    WITH slide, toLower(
        coalesce(slide.title,'') + ' ' +
        coalesce(slide.slide_text,'') + ' ' +
        coalesce(slide.emphasis_keywords_text,'') + ' ' +
        (CASE WHEN coalesce(slide.emphasis_total, 0) > 0 OR coalesce(slide.emphasis_keywords_text, '') <> ''
              THEN '강조 emphasized highlight 핵심 중요' ELSE '' END)
    ) AS haystack
    WITH slide, haystack, [k IN $keywords WHERE haystack CONTAINS toLower(k)] AS hits
    WHERE haystack CONTAINS toLower($kw)
    OPTIONAL MATCH (scene:Scene {{stem: $stem}})-[:USES_SLIDE]->(slide)
    RETURN slide.slide_number AS slide_number, coalesce(slide.id,'') AS slide_id,
           slide.title AS title, slide.slide_text AS slide_text,
           min(scene.start_sec) AS start_sec, max(scene.end_sec) AS end_sec,
           size(hits) AS relevance
    ORDER BY relevance DESC, slide.slide_number LIMIT {SLIDE_LIM}
    """
    q_seg_text = f"""
    MATCH (scene:Scene {{stem: $stem}})-[:USES_SLIDE]->(slide:Slide {{stem: $stem}})
    MATCH (scene)-[:HAS_CONTEXT]->(ctx:Context {{stem: $stem}})-[:HAS_SEGMENT]->(seg:Segment {{stem: $stem}})
    WITH slide, scene, ctx, seg,
         toLower(
            coalesce(slide.title,'') + ' ' +
            coalesce(slide.emphasis_keywords_text,'') + ' ' +
            (CASE WHEN coalesce(slide.emphasis_total, 0) > 0 OR coalesce(slide.emphasis_keywords_text, '') <> ''
                  THEN '강조 emphasized highlight 핵심 중요' ELSE '' END) + ' ' +
            coalesce(ctx.audio_emphasis, '') + ' ' +
            coalesce(ctx.emphasis_score, '') + ' ' +
            (CASE WHEN coalesce(scene.emphasis_total, 0) > 0
                  THEN '강조 emphasized highlight 핵심 중요' ELSE '' END) + ' ' +
            coalesce(ctx.text,'') + ' ' +
            coalesce(seg.text,'')
         ) AS haystack
    WITH slide, scene, ctx, seg, haystack, [k IN $keywords WHERE haystack CONTAINS toLower(k)] AS hits
    WHERE haystack CONTAINS toLower($kw)
    RETURN coalesce(seg.text,'') AS segment_text, seg.start AS start, seg.end AS end,
           slide.slide_number AS slide_number,
           coalesce(slide.id,'') AS slide_id, coalesce(seg.id,'') AS segment_id,
           coalesce(scene.id,'') AS scene_id, coalesce(ctx.id,'') AS context_id,
           scene.start_sec AS scene_start_sec, scene.end_sec AS scene_end_sec,
           size(hits) AS relevance
    ORDER BY relevance DESC, seg.start LIMIT {SEG_LIM}
    """
    q_graphrag_entity = f"""
    MATCH (ge:GraphRAGEntity {{stem: $stem}})
    WITH ge, toLower(
        coalesce(ge.title, '') + ' ' +
        coalesce(ge.description, '') + ' ' +
        coalesce(ge.type, '') + ' ' +
        reduce(s = '', x IN coalesce(ge.emphasis_matched_keywords, []) | s + ' ' + x) + ' ' +
        reduce(s = '', x IN coalesce(ge.emphasis_visual_keywords, []) | s + ' ' + x) + ' ' +
        reduce(s = '', x IN coalesce(ge.emphasis_sources, []) | s + ' ' + x) + ' ' +
        (CASE WHEN coalesce(ge.emphasis_boost_local, 0) > 0
              THEN '강조 emphasized highlight 핵심 중요' ELSE '' END)
    ) AS haystack
    WITH ge, haystack, [k IN $keywords WHERE haystack CONTAINS toLower(k)] AS hits
    WHERE haystack CONTAINS toLower($kw)
    OPTIONAL MATCH (ge)-[:GRAPHRAG_APPEARS_IN]->(slide:Slide {{stem: $stem}})
    OPTIONAL MATCH (ge)-[:GRAPHRAG_APPEARS_IN_SCENE]->(scene:Scene {{stem: $stem}})
    RETURN coalesce(ge.id, '') AS graphrag_entity_id,
           ge.title AS graphrag_title,
           ge.type AS graphrag_type,
           ge.description AS graphrag_description,
           ge.degree AS degree,
           ge.frequency AS frequency,
           ge.emphasis_boost_local AS emphasis_boost_local,
           ge.final_weight AS final_weight,
           ge.emphasis_sources AS emphasis_sources,
           ge.emphasis_matched_keywords AS emphasis_matched_keywords,
           ge.emphasis_visual_keywords AS emphasis_visual_keywords,
           [] AS concept_ids,
           [] AS concept_names,
           collect(DISTINCT slide.slide_number) AS slide_numbers,
           collect(DISTINCT coalesce(slide.id, '')) AS slide_ids,
           collect(DISTINCT coalesce(scene.id, '')) AS scene_ids,
           collect(DISTINCT scene.start_sec) AS scene_start_secs,
           collect(DISTINCT scene.end_sec) AS scene_end_secs,
           size(hits) AS relevance
    ORDER BY relevance DESC, coalesce(ge.final_weight, ge.degree, 0) DESC, coalesce(ge.frequency, 0) DESC
    LIMIT {GR_ENTITY_LIM}
    """
    q_graphrag_rel = f"""
    MATCH (src:GraphRAGEntity {{stem: $stem}})-[r:GRAPHRAG_RELATES_TO]->(tgt:GraphRAGEntity {{stem: $stem}})
    WITH src, r, tgt, toLower(
        coalesce(src.title, '') + ' ' +
        coalesce(tgt.title, '') + ' ' +
        coalesce(r.description, '') + ' ' +
        reduce(s = '', x IN coalesce(src.emphasis_matched_keywords, []) | s + ' ' + x) + ' ' +
        reduce(s = '', x IN coalesce(tgt.emphasis_matched_keywords, []) | s + ' ' + x) + ' ' +
        reduce(s = '', x IN coalesce(src.emphasis_visual_keywords, []) | s + ' ' + x) + ' ' +
        reduce(s = '', x IN coalesce(tgt.emphasis_visual_keywords, []) | s + ' ' + x) + ' ' +
        (CASE WHEN coalesce(r.emphasis_edge_weight, 0) > coalesce(r.weight, 0)
              THEN '강조 emphasized highlight 핵심 중요' ELSE '' END)
    ) AS haystack
    WITH src, r, tgt, haystack, [k IN $keywords WHERE haystack CONTAINS toLower(k)] AS hits
    WHERE haystack CONTAINS toLower($kw)
    RETURN coalesce(src.id, '') AS src_id,
           src.title AS src_title,
           coalesce(tgt.id, '') AS tgt_id,
           tgt.title AS tgt_title,
           r.description AS rel_description,
           r.weight AS weight,
           r.emphasis_edge_weight AS emphasis_edge_weight,
           r.combined_degree AS combined_degree,
           [] AS src_concept_ids,
           [] AS tgt_concept_ids,
           size(hits) AS relevance
    ORDER BY relevance DESC, coalesce(r.emphasis_edge_weight, r.weight, 0) DESC, coalesce(r.combined_degree, 0) DESC
    LIMIT {GR_REL_LIM}
    """

    for kw in keywords:
        for key, q, needs_rel in (
            ("slides", q_slide_text, False),
            ("segments", q_seg_text, False),
            ("graphrag_entities", q_graphrag_entity, False),
            ("graphrag_relationships", q_graphrag_rel, False),
        ):
            params: dict[str, Any] = {"stem": stem, "kw": kw, "keywords": keywords}
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
                elif key == "graphrag_entities":
                    eid = row.get("graphrag_entity_id")
                    if eid and eid not in seen_gr_entities:
                        seen_gr_entities.add(eid)
                        results["graphrag_entities"].append(row)
                elif key == "graphrag_relationships":
                    k2 = (row.get("src_id"), row.get("tgt_id"), row.get("rel_description"))
                    if k2 not in seen_gr_rels:
                        seen_gr_rels.add(k2)
                        results["graphrag_relationships"].append(row)
                else:
                    results["sub_concepts"].append(row)

    if results.get("segments"):
        results["segments"].sort(
            key=lambda r: (
                int(r.get("relevance") or 0),
                _relevance_score(str(r.get("segment_text", "")), keywords),
            ),
            reverse=True,
        )
    if results.get("slides"):
        results["slides"].sort(
            key=lambda r: (
                int(r.get("relevance") or 0),
                _relevance_score(str(r.get("slide_text", "")) + str(r.get("title", "")), keywords),
            ),
            reverse=True,
        )

    return dict(results), raw_all
