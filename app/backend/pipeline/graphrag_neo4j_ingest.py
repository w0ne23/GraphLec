"""
Microsoft GraphRAG parquet output -> Neo4j concept layer.

This loader keeps GraphRAG output in the same lecture stem namespace as the
GraphLec structural graph and adds bridge edges back to Slide nodes.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd

_SLIDE_ID_RE = re.compile(r"slide_\d{3,}")
_SEGMENT_ID_RE = re.compile(r"graphlec_seg:(segment/\d+)")
_STRUCTURAL_ENTITY_RE = re.compile(r"^(slide[_ ]?\d+|chapter\s*\d+)$", re.IGNORECASE)


def _repo_root() -> Path:
    return Path("/pipeline") if Path("/pipeline").exists() else Path(__file__).resolve().parents[3]


def default_graphrag_root() -> Path:
    return Path(os.getenv("GRAPHLEC_GRAPHRAG_ROOT", str(_repo_root() / "graphrag_workspaces")))


def find_graphrag_output_dir(
    stem: str,
    output_dir: Path | None = None,
    *,
    aliases: list[str] | None = None,
) -> Path | None:
    """Find a GraphRAG output directory for a lecture stem."""
    names: list[str] = []
    for name in [stem, *(aliases or [])]:
        clean = str(name or "").strip()
        if clean and clean not in names:
            names.append(clean)

    candidates: list[Path] = []
    if output_dir:
        candidates.extend(
            [
                output_dir / "graphrag" / "output",
                output_dir / "graphrag_output",
                output_dir / "output",
            ]
        )
        for name in names:
            candidates.append(output_dir / "graphrag_workspaces" / name / "output")

    root = default_graphrag_root()
    for name in names:
        candidates.append(root / name / "output")
        candidates.append(_repo_root() / "graphrag_workspaces" / name / "output")

    seen: set[Path] = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        if (path / "entities.parquet").is_file() and (path / "relationships.parquet").is_file():
            return path
    return None


def _str_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value)


def _int_cell(value: Any, default: int = 0) -> int:
    try:
        if value is None or pd.isna(value):
            return default
    except Exception:
        pass
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _float_cell(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return default
    except Exception:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _list_cell(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw = value.strip()
        if not raw or raw == "[]":
            return []
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [str(x) for x in parsed if str(x)]
        except json.JSONDecodeError:
            return [x for x in re.findall(r"[0-9A-Za-z][0-9A-Za-z_-]{7,}", raw) if x]
        return [raw]
    try:
        if pd.isna(value):
            return []
    except Exception:
        pass
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple, set)):
        return [str(x) for x in value if str(x)]
    return [str(value)]


def _norm_title(value: Any) -> str:
    return re.sub(r"\s+", "", _str_cell(value).lower())


def _json_props(props: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in props.items():
        if value is None:
            continue
        if isinstance(value, (dict, list)):
            out[key] = json.dumps(value, ensure_ascii=False)
        else:
            out[key] = value
    return out


def _extract_slide_ids(text: Any) -> list[str]:
    return sorted(set(_SLIDE_ID_RE.findall(_str_cell(text))))


def _extract_segment_ids(text: Any) -> list[str]:
    return sorted(set(_SEGMENT_ID_RE.findall(_str_cell(text))))


def delete_graphrag_layer_tx(tx, stem: str) -> None:
    """Remove only GraphRAG nodes/relationships for a stem, preserving GraphLec graph."""
    tx.run(
        """
        MATCH (n {stem: $stem})
        WHERE n:GraphRAGEntity OR n:GraphRAGTextUnit OR n:GraphRAGCommunity
        DETACH DELETE n
        """,
        stem=stem,
    )


def delete_custom_concept_layer_tx(tx, stem: str) -> None:
    """Remove GraphLec's LLM-generated Concept layer while keeping structure nodes."""
    tx.run("MATCH (n:Concept {stem: $stem}) DETACH DELETE n", stem=stem)


def load_graphrag_layer_tx(tx, stem: str, graphrag_dir: Path) -> dict[str, int]:
    """Load GraphRAG parquet files into the current Neo4j transaction."""
    entities_path = graphrag_dir / "entities.parquet"
    relationships_path = graphrag_dir / "relationships.parquet"
    text_units_path = graphrag_dir / "text_units.parquet"
    communities_path = graphrag_dir / "communities.parquet"
    reports_path = graphrag_dir / "community_reports.parquet"

    entities = pd.read_parquet(entities_path)
    relationships = pd.read_parquet(relationships_path)
    text_units = pd.read_parquet(text_units_path) if text_units_path.is_file() else pd.DataFrame()
    communities = pd.read_parquet(communities_path) if communities_path.is_file() else pd.DataFrame()
    reports = pd.read_parquet(reports_path) if reports_path.is_file() else pd.DataFrame()

    entity_rows: list[dict[str, Any]] = []
    entity_by_title: dict[str, str] = {}
    entity_text_units: list[dict[str, str]] = []
    entity_slides: set[tuple[str, str]] = set()
    entity_segments: set[tuple[str, str]] = set()

    text_unit_slide_map: dict[str, list[str]] = {}
    text_unit_segment_map: dict[str, list[str]] = {}
    text_unit_rows: list[dict[str, Any]] = []
    for _, row in text_units.iterrows():
        gr_id = _str_cell(row.get("id")).strip()
        if not gr_id:
            continue
        node_id = f"graphrag/text_unit/{gr_id}"
        raw_text = row.get("text")
        slide_ids = _extract_slide_ids(raw_text)
        segment_ids = _extract_segment_ids(raw_text)
        text_unit_slide_map[gr_id] = slide_ids
        text_unit_segment_map[gr_id] = segment_ids
        text_unit_rows.append(
            _json_props(
                {
                    "id": node_id,
                    "stem": stem,
                    "graphrag_id": gr_id,
                    "human_readable_id": _str_cell(row.get("human_readable_id")),
                    "text": _str_cell(raw_text)[:12000],
                    "n_tokens": _int_cell(row.get("n_tokens")),
                    "document_id": _str_cell(row.get("document_id")),
                    "entity_ids": _list_cell(row.get("entity_ids")),
                    "relationship_ids": _list_cell(row.get("relationship_ids")),
                    "slide_ids": slide_ids,
                    "segment_ids": segment_ids,
                    "source": "microsoft_graphrag",
                }
            )
        )

    for _, row in entities.iterrows():
        gr_id = _str_cell(row.get("id")).strip()
        title = _str_cell(row.get("title")).strip()
        if not gr_id or not title:
            continue
        if _STRUCTURAL_ENTITY_RE.match(title):
            continue
        node_id = f"graphrag/entity/{gr_id}"
        title_norm = _norm_title(title)
        entity_by_title.setdefault(title_norm, node_id)
        tu_ids = _list_cell(row.get("text_unit_ids"))
        entity_slide_ids: list[str] = []
        entity_segment_ids: list[str] = []
        for tu_id in tu_ids:
            entity_text_units.append({"entity_id": node_id, "text_unit_id": f"graphrag/text_unit/{tu_id}"})
            for slide_id in text_unit_slide_map.get(tu_id, []):
                entity_slides.add((node_id, slide_id))
                if slide_id not in entity_slide_ids:
                    entity_slide_ids.append(slide_id)
            for seg_id in text_unit_segment_map.get(tu_id, []):
                entity_segments.add((node_id, seg_id))
                if seg_id not in entity_segment_ids:
                    entity_segment_ids.append(seg_id)
        entity_rows.append(
            _json_props(
                {
                    "id": node_id,
                    "stem": stem,
                    "graphrag_id": gr_id,
                    "human_readable_id": _str_cell(row.get("human_readable_id")),
                    "title": title,
                    "name": title,
                    "title_norm": title_norm,
                    "type": _str_cell(row.get("type")),
                    "description": _str_cell(row.get("description"))[:12000],
                    "text_unit_ids": tu_ids,
                    "slide_ids": sorted(entity_slide_ids),
                    "segment_ids": sorted(entity_segment_ids),
                    "frequency": _int_cell(row.get("frequency")),
                    "degree": _int_cell(row.get("degree")),
                    "source": "microsoft_graphrag",
                }
            )
        )

    tx.run(
        """
        UNWIND $rows AS props
        MERGE (n:GraphRAGTextUnit:ConceptGraph {stem: props.stem, id: props.id})
        SET n += props
        """,
        rows=text_unit_rows,
    )
    tx.run(
        """
        UNWIND $rows AS props
        MERGE (n:GraphRAGEntity:ConceptGraph {stem: props.stem, id: props.id})
        SET n += props
        """,
        rows=entity_rows,
    )
    tx.run(
        """
        UNWIND $rows AS row
        MATCH (e:GraphRAGEntity {stem: $stem, id: row.entity_id})
        MATCH (tu:GraphRAGTextUnit {stem: $stem, id: row.text_unit_id})
        MERGE (e)-[:GRAPHRAG_SUPPORTED_BY]->(tu)
        """,
        stem=stem,
        rows=entity_text_units,
    )

    text_unit_slides = [
        {"text_unit_id": f"graphrag/text_unit/{tu_id}", "slide_id": slide_id}
        for tu_id, slide_ids in text_unit_slide_map.items()
        for slide_id in slide_ids
    ]
    tx.run(
        """
        UNWIND $rows AS row
        MATCH (tu:GraphRAGTextUnit {stem: $stem, id: row.text_unit_id})
        MATCH (s:Slide {stem: $stem, id: row.slide_id})
        MERGE (tu)-[:GRAPHRAG_MENTIONS_SLIDE]->(s)
        """,
        stem=stem,
        rows=text_unit_slides,
    )
    tx.run(
        """
        UNWIND $rows AS row
        MATCH (e:GraphRAGEntity {stem: $stem, id: row.entity_id})
        MATCH (s:Slide {stem: $stem, id: row.slide_id})
        MERGE (e)-[:GRAPHRAG_APPEARS_IN]->(s)
        """,
        stem=stem,
        rows=[{"entity_id": e, "slide_id": s} for e, s in sorted(entity_slides)],
    )

    text_unit_segments = [
        {"text_unit_id": f"graphrag/text_unit/{tu_id}", "segment_id": seg_id}
        for tu_id, seg_ids in text_unit_segment_map.items()
        for seg_id in seg_ids
    ]
    if text_unit_segments:
        tx.run(
            """
            UNWIND $rows AS row
            MATCH (tu:GraphRAGTextUnit {stem: $stem, id: row.text_unit_id})
            MATCH (seg:Segment {stem: $stem, id: row.segment_id})
            MERGE (tu)-[:GRAPHRAG_MENTIONS_SEGMENT]->(seg)
            """,
            stem=stem,
            rows=text_unit_segments,
        )

    if entity_segments:
        tx.run(
            """
            UNWIND $rows AS row
            MATCH (e:GraphRAGEntity {stem: $stem, id: row.entity_id})
            MATCH (seg:Segment {stem: $stem, id: row.segment_id})
            MERGE (e)-[:GRAPHRAG_MENTIONED_IN_SEGMENT]->(seg)
            """,
            stem=stem,
            rows=[{"entity_id": e, "segment_id": s} for e, s in sorted(entity_segments)],
        )

    tx.run(
        """
        MATCH (e:GraphRAGEntity {stem: $stem})
        MATCH (a:AnnotationEmphasis {stem: $stem})
        WHERE (a.target_content IS NOT NULL AND toLower(a.target_content) CONTAINS toLower(e.title))
           OR (a.handwritten_content IS NOT NULL AND toLower(a.handwritten_content) CONTAINS toLower(e.title))
        MERGE (e)-[:GRAPHRAG_HIGHLIGHTED_IN]->(a)
        """,
        stem=stem,
    )

    rel_rows: list[dict[str, Any]] = []
    for _, row in relationships.iterrows():
        src_id = entity_by_title.get(_norm_title(row.get("source")))
        tgt_id = entity_by_title.get(_norm_title(row.get("target")))
        if not src_id or not tgt_id:
            continue
        rel_rows.append(
            _json_props(
                {
                    "id": _str_cell(row.get("id")),
                    "human_readable_id": _str_cell(row.get("human_readable_id")),
                    "src": src_id,
                    "tgt": tgt_id,
                    "source_title": _str_cell(row.get("source")),
                    "target_title": _str_cell(row.get("target")),
                    "description": _str_cell(row.get("description"))[:4000],
                    "weight": _float_cell(row.get("weight")),
                    "combined_degree": _int_cell(row.get("combined_degree")),
                    "text_unit_ids": _list_cell(row.get("text_unit_ids")),
                    "source": "microsoft_graphrag",
                }
            )
        )
    tx.run(
        """
        UNWIND $rows AS row
        MATCH (a:GraphRAGEntity {stem: $stem, id: row.src})
        MATCH (b:GraphRAGEntity {stem: $stem, id: row.tgt})
        MERGE (a)-[r:GRAPHRAG_RELATES_TO {graphrag_id: row.id}]->(b)
        SET r += row
        """,
        stem=stem,
        rows=rel_rows,
    )

    reports_by_community = {
        _str_cell(row.get("community")): row
        for _, row in reports.iterrows()
        if _str_cell(row.get("community"))
    }
    community_rows: list[dict[str, Any]] = []
    community_entities: list[dict[str, str]] = []
    for _, row in communities.iterrows():
        community = _str_cell(row.get("community")).strip()
        if not community:
            continue
        report = reports_by_community.get(community)
        title = _str_cell(report.get("title")) if report is not None else _str_cell(row.get("title"))
        summary = _str_cell(report.get("summary")) if report is not None else ""
        community_id = f"graphrag/community/{community}"
        community_rows.append(
            _json_props(
                {
                    "id": community_id,
                    "stem": stem,
                    "graphrag_id": _str_cell(row.get("id")),
                    "community": community,
                    "human_readable_id": _str_cell(row.get("human_readable_id")),
                    "title": title,
                    "summary": summary[:8000],
                    "level": _int_cell(row.get("level")),
                    "parent": _str_cell(row.get("parent")),
                    "children": _list_cell(row.get("children")),
                    "rank": _float_cell(report.get("rank")) if report is not None else 0.0,
                    "size": _int_cell(row.get("size")),
                    "source": "microsoft_graphrag",
                }
            )
        )
        for entity_id in _list_cell(row.get("entity_ids")):
            community_entities.append(
                {"community_id": community_id, "entity_id": f"graphrag/entity/{entity_id}"}
            )
    tx.run(
        """
        UNWIND $rows AS props
        MERGE (n:GraphRAGCommunity:ConceptGraph {stem: props.stem, id: props.id})
        SET n += props
        """,
        rows=community_rows,
    )
    tx.run(
        """
        UNWIND $rows AS row
        MATCH (c:GraphRAGCommunity {stem: $stem, id: row.community_id})
        MATCH (e:GraphRAGEntity {stem: $stem, id: row.entity_id})
        MERGE (c)-[:GRAPHRAG_HAS_ENTITY]->(e)
        """,
        stem=stem,
        rows=community_entities,
    )

    return {
        "entities": len(entity_rows),
        "relationships": len(rel_rows),
        "text_units": len(text_unit_rows),
        "communities": len(community_rows),
        "entity_text_unit_links": len(entity_text_units),
        "text_unit_slide_links": len(text_unit_slides),
        "entity_slide_links": len(entity_slides),
        "text_unit_segment_links": len(text_unit_segments),
        "entity_segment_links": len(entity_segments),
        "community_entity_links": len(community_entities),
    }
