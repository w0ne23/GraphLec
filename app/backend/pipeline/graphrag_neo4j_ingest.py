"""
Microsoft GraphRAG parquet output -> Neo4j concept layer.

This loader keeps GraphRAG output in the same lecture stem namespace as the
GraphLec structural graph and adds bridge edges back to Slide, Scene, and
Segment nodes.
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
_STRUCTURAL_ENTITY_RE = re.compile(
    r"^(slide[_ ]?\d+|chapter\s*\d+|segment[/_ ]?\d+|context[/_ ]?\d+|scene[/_ ]?\d+|graphlec_seg:segment/\d+)$",
    re.IGNORECASE,
)
_TRAILING_ALIAS_RE = re.compile(r"^\s*(?P<head>.+?)\s*[\(\[]\s*(?P<alias>[^)\]]+)\s*[\)\]]\s*$")
_HANGUL_RE = re.compile(r"[가-힣]")


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


def _alias_key(value: Any) -> str:
    text = _str_cell(value).lower()
    return re.sub(r"[\W_]+", "", text)


def _concept_key(value: Any) -> str:
    key = _alias_key(value)
    for suffix in ("들의", "으로", "에서", "에게", "에는", "은", "는", "이", "가", "을", "를", "의", "들", "s"):
        if len(key) > len(suffix) + 1 and key.endswith(suffix):
            key = key[: -len(suffix)]
            break
    return key


def _has_hangul(value: Any) -> bool:
    return bool(_HANGUL_RE.search(_str_cell(value)))


def _split_title_alias(title: Any) -> tuple[str, str | None]:
    text = _str_cell(title).strip()
    match = _TRAILING_ALIAS_RE.match(text)
    if not match:
        return text, None
    return match.group("head").strip(), match.group("alias").strip()


def _preferred_canonical_title(title: Any) -> str:
    head, alias = _split_title_alias(title)
    if alias and _has_hangul(head):
        return head
    if alias and not _has_hangul(head) and _has_hangul(alias):
        return alias
    return _str_cell(title).strip()


def _best_entity_row(rows: list[dict[str, Any]], canonical_title: str) -> dict[str, Any]:
    canonical_key = _alias_key(canonical_title)

    def score(row: dict[str, Any]) -> tuple[int, int, int, int, int]:
        title = _str_cell(row.get("title")).strip()
        head, alias = _split_title_alias(title)
        plain_title = head if alias else title
        return (
            1 if _alias_key(plain_title) == canonical_key else 0,
            1 if _has_hangul(title) else 0,
            1 if alias is None else 0,
            _int_cell(row.get("frequency")) + _int_cell(row.get("degree")),
            len(_str_cell(row.get("description"))),
        )

    return max(rows, key=score)


def _entity_names(row: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for value in [_str_cell(row.get("title")), *_list_cell(row.get("aliases"))]:
        value = value.strip()
        if value and value not in names:
            names.append(value)
    return names


def _char_ngram_counts(text: str, n: int = 3) -> dict[str, int]:
    clean = _alias_key(text)
    if len(clean) < n:
        return {clean: 1} if clean else {}
    counts: dict[str, int] = {}
    for i in range(len(clean) - n + 1):
        gram = clean[i : i + n]
        counts[gram] = counts.get(gram, 0) + 1
    return counts


def _cosine_counts(a: dict[str, int], b: dict[str, int]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(count * b.get(key, 0) for key, count in a.items())
    if dot <= 0:
        return 0.0
    norm_a = sum(count * count for count in a.values()) ** 0.5
    norm_b = sum(count * count for count in b.values()) ** 0.5
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


def _semantic_merge_reason(a: dict[str, Any], b: dict[str, Any]) -> str | None:
    a_names = _entity_names(a)
    b_names = _entity_names(b)
    a_keys = {_concept_key(name) for name in a_names if _concept_key(name)}
    b_keys = {_concept_key(name) for name in b_names if _concept_key(name)}

    for key in sorted(a_keys & b_keys):
        if len(key) >= 3:
            return f"same_normalized_name:{key}"

    a_tu = set(_list_cell(a.get("text_unit_ids")))
    b_tu = set(_list_cell(b.get("text_unit_ids")))
    if not (a_tu & b_tu):
        return None

    a_text = " ".join([*a_names, _str_cell(a.get("description"))])
    b_text = " ".join([*b_names, _str_cell(b.get("description"))])
    desc_sim = _cosine_counts(_char_ngram_counts(a_text), _char_ngram_counts(b_text))
    if desc_sim >= 0.96:
        return f"shared_text_unit_description_similarity:{desc_sim:.3f}"
    return None


def _merge_entity_group(rows: list[dict[str, Any]], canonical_title: str | None = None) -> dict[str, Any]:
    if canonical_title is None:
        title_votes: dict[str, int] = {}
        for row in rows:
            title = _preferred_canonical_title(row.get("title"))
            title_votes[title] = title_votes.get(title, 0) + _int_cell(row.get("frequency"), 1)
        canonical_title = max(title_votes, key=title_votes.get)

    primary = _best_entity_row(rows, canonical_title)
    aliases: list[str] = []
    graphrag_ids: list[str] = []
    human_ids: list[str] = []
    text_unit_ids: set[str] = set()
    descriptions: list[str] = []
    merge_reasons: set[str] = set()

    for row in rows:
        graphrag_ids.extend(_list_cell(row.get("graphrag_ids")) or [_str_cell(row.get("id"))])
        human_ids.extend(_list_cell(row.get("human_readable_ids")) or [_str_cell(row.get("human_readable_id"))])
        text_unit_ids.update(_list_cell(row.get("text_unit_ids")))
        descriptions.extend(desc for desc in [_str_cell(row.get("description")).strip()] if desc and desc not in descriptions)
        merge_reasons.update(_list_cell(row.get("merge_reasons")))
        for name in _entity_names(row):
            if _alias_key(name) != _alias_key(canonical_title) and name not in aliases:
                aliases.append(name)

    return {
        **primary,
        "id": _str_cell(primary.get("id")).strip(),
        "title": canonical_title,
        "description": "\n\n".join(descriptions)[:12000],
        "text_unit_ids": sorted(text_unit_ids),
        "frequency": sum(_int_cell(row.get("frequency")) for row in rows),
        "degree": sum(_int_cell(row.get("degree")) for row in rows),
        "aliases": sorted(alias for alias in aliases if alias),
        "graphrag_ids": sorted(set(graphrag_id for graphrag_id in graphrag_ids if graphrag_id)),
        "human_readable_ids": sorted(set(human_id for human_id in human_ids if human_id)),
        "merged_entity_count": sum(max(_int_cell(row.get("merged_entity_count"), 1), 1) for row in rows),
        "merge_reasons": sorted(merge_reasons),
    }


def _semantic_merge_entity_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    parent = list(range(len(rows)))
    reasons: dict[int, set[str]] = {i: set(_list_cell(row.get("merge_reasons"))) for i, row in enumerate(rows)}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int, reason: str) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            reasons[ra].add(reason)
            return
        parent[rb] = ra
        reasons[ra].update(reasons.get(rb, set()))
        reasons[ra].add(reason)

    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            reason = _semantic_merge_reason(rows[i], rows[j])
            if reason:
                union(i, j, reason)

    groups: dict[int, list[dict[str, Any]]] = {}
    for idx, row in enumerate(rows):
        root = find(idx)
        groups.setdefault(root, []).append(row)

    merged: list[dict[str, Any]] = []
    for root, group_rows in groups.items():
        for row in group_rows:
            row["merge_reasons"] = sorted(reasons.get(root, set()))
        merged.append(_merge_entity_group(group_rows))
    return merged


def _merge_entity_rows(entities: pd.DataFrame) -> tuple[list[dict[str, Any]], dict[str, str], dict[str, str], int]:
    """Canonicalize bilingual GraphRAG entity variants before Neo4j ingest."""
    raw_rows: list[dict[str, Any]] = []
    direct_title_map: dict[str, str] = {}
    alias_candidates: dict[str, set[str]] = {}

    for _, row in entities.iterrows():
        gr_id = _str_cell(row.get("id")).strip()
        title = _str_cell(row.get("title")).strip()
        if not gr_id or not title or _STRUCTURAL_ENTITY_RE.match(title):
            continue

        row_dict = row.to_dict()
        raw_rows.append(row_dict)

        canonical = _preferred_canonical_title(title)
        head, alias = _split_title_alias(title)
        for candidate in [title, canonical, head]:
            key = _alias_key(candidate)
            if key:
                direct_title_map[key] = canonical
        alias_key = _alias_key(alias)
        if alias_key:
            alias_candidates.setdefault(alias_key, set()).add(canonical)

    alias_to_canonical = dict(direct_title_map)
    for key, candidates in alias_candidates.items():
        if len(candidates) != 1:
            continue
        canonical = next(iter(candidates))
        existing = alias_to_canonical.get(key)
        if not existing or _has_hangul(canonical):
            alias_to_canonical[key] = canonical

    grouped: dict[str, list[dict[str, Any]]] = {}
    canonical_titles: dict[str, str] = {}
    for row in raw_rows:
        title = _str_cell(row.get("title")).strip()
        canonical = alias_to_canonical.get(_alias_key(title), _preferred_canonical_title(title))
        key = _alias_key(canonical)
        grouped.setdefault(key, []).append(row)
        canonical_titles.setdefault(key, canonical)

    initial_rows: list[dict[str, Any]] = []
    for key, rows in grouped.items():
        canonical_title = canonical_titles[key]
        merged = _merge_entity_group(rows, canonical_title)
        if len(rows) > 1:
            merged["merge_reasons"] = sorted(
                set(_list_cell(merged.get("merge_reasons"))) | {"alias_or_translation"}
            )
        initial_rows.append(merged)

    merged_rows = _semantic_merge_entity_rows(initial_rows)
    entity_id_map: dict[str, str] = {}
    title_to_node_id: dict[str, str] = {}

    for row in merged_rows:
        node_id = f"graphrag/entity/{_str_cell(row.get('id')).strip()}"
        for graphrag_id in _list_cell(row.get("graphrag_ids")):
            entity_id_map[graphrag_id] = node_id
        for title in _entity_names(row):
            title_to_node_id[_norm_title(title)] = node_id
            title_to_node_id[_alias_key(title)] = node_id

    return merged_rows, entity_id_map, title_to_node_id, max(0, len(raw_rows) - len(merged_rows))


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

    merged_entities, entity_id_map, entity_by_title, merged_entity_count = _merge_entity_rows(entities)

    entity_rows: list[dict[str, Any]] = []
    entity_text_units: list[dict[str, str]] = []
    entity_slides: set[tuple[str, str]] = set()
    entity_segments: set[tuple[str, str]] = set()

    for row in merged_entities:
        gr_id = _str_cell(row.get("id")).strip()
        title = _str_cell(row.get("title")).strip()
        if not gr_id or not title:
            continue
        node_id = f"graphrag/entity/{gr_id}"
        title_norm = _norm_title(title)
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
                    "aliases": _list_cell(row.get("aliases")),
                    "graphrag_ids": _list_cell(row.get("graphrag_ids")),
                    "human_readable_ids": _list_cell(row.get("human_readable_ids")),
                    "merged_entity_count": _int_cell(row.get("merged_entity_count"), 1),
                    "merge_reasons": _list_cell(row.get("merge_reasons")),
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
        MATCH (tu:GraphRAGTextUnit {stem: $stem, id: row.text_unit_id})
        MATCH (sc:Scene {stem: $stem, source_slide_id: row.slide_id})
        MERGE (tu)-[:GRAPHRAG_MENTIONS_SCENE]->(sc)
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
    tx.run(
        """
        UNWIND $rows AS row
        MATCH (e:GraphRAGEntity {stem: $stem, id: row.entity_id})
        MATCH (sc:Scene {stem: $stem, source_slide_id: row.slide_id})
        MERGE (e)-[:GRAPHRAG_APPEARS_IN_SCENE]->(sc)
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
        tx.run(
            """
            UNWIND $rows AS row
            MATCH (tu:GraphRAGTextUnit {stem: $stem, id: row.text_unit_id})
            MATCH (sc:Scene {stem: $stem})-[:HAS_CONTEXT]->(ctx:Context {stem: $stem})
            MATCH (ctx)-[:HAS_SEGMENT]->(seg:Segment {stem: $stem, id: row.segment_id})
            MERGE (tu)-[:GRAPHRAG_MENTIONS_SCENE]->(sc)
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
            UNWIND $rows AS row
            MATCH (e:GraphRAGEntity {stem: $stem, id: row.entity_id})
            MATCH (sc:Scene {stem: $stem})-[:HAS_CONTEXT]->(ctx:Context {stem: $stem})
            MATCH (ctx)-[:HAS_SEGMENT]->(seg:Segment {stem: $stem, id: row.segment_id})
            MERGE (e)-[:GRAPHRAG_APPEARS_IN_SCENE]->(sc)
            """,
            stem=stem,
            rows=[{"entity_id": e, "segment_id": s} for e, s in sorted(entity_segments)],
        )

    text_unit_scene_links = tx.run(
        """
        MATCH (:GraphRAGTextUnit {stem: $stem})-[r:GRAPHRAG_MENTIONS_SCENE]->(:Scene {stem: $stem})
        RETURN count(r) AS count
        """,
        stem=stem,
    ).single()
    entity_scene_links = tx.run(
        """
        MATCH (:GraphRAGEntity {stem: $stem})-[r:GRAPHRAG_APPEARS_IN_SCENE]->(:Scene {stem: $stem})
        RETURN count(r) AS count
        """,
        stem=stem,
    ).single()

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
        src_id = entity_by_title.get(_norm_title(row.get("source"))) or entity_by_title.get(_alias_key(row.get("source")))
        tgt_id = entity_by_title.get(_norm_title(row.get("target"))) or entity_by_title.get(_alias_key(row.get("target")))
        if not src_id or not tgt_id or src_id == tgt_id:
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
    community_entity_keys: set[tuple[str, str]] = set()
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
            canonical_entity_id = entity_id_map.get(entity_id)
            if not canonical_entity_id:
                continue
            link_key = (community_id, canonical_entity_id)
            if link_key in community_entity_keys:
                continue
            community_entity_keys.add(link_key)
            community_entities.append(
                {"community_id": community_id, "entity_id": canonical_entity_id}
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
        "entities_before_canonical_merge": len(entities),
        "entities": len(entity_rows),
        "entities_merged": merged_entity_count,
        "relationships": len(rel_rows),
        "text_units": len(text_unit_rows),
        "communities": len(community_rows),
        "entity_text_unit_links": len(entity_text_units),
        "text_unit_slide_links": len(text_unit_slides),
        "entity_slide_links": len(entity_slides),
        "text_unit_scene_links": text_unit_scene_links["count"] if text_unit_scene_links else 0,
        "entity_scene_links": entity_scene_links["count"] if entity_scene_links else 0,
        "text_unit_segment_links": len(text_unit_segments),
        "entity_segment_links": len(entity_segments),
        "community_entity_links": len(community_entities),
    }
