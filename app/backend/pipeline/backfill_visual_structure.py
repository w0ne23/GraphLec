"""Backfill structured visual fields from existing t1_structure text.

This is a no-vision, no-LLM migration helper for already generated lectures.
It enriches visual_assets with:
  - visual_elements / visual_elements_text
  - visual_relations / visual_relations_text
  - layout / layout_text

The extractor is intentionally conservative: it only derives what is already
described in t1_structure, visual asset descriptions, or raw visual text.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any


VISUAL_ASSET_TYPES = {"table", "diagram", "figure", "list", "chart", "image", "other"}


def _clean(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _sentences(text: str) -> list[str]:
    text = str(text or "").replace("\r", "\n")
    parts: list[str] = []
    for block in text.split("\n"):
        block = block.strip()
        if not block:
            continue
        start = 0
        for match in re.finditer(r"다\.|[.!?。]", block):
            parts.append(block[start : match.end()])
            start = match.end()
        if start < len(block):
            parts.append(block[start:])
    return [_clean(p) for p in parts if _clean(p)]


def _quoted_terms(text: str) -> list[str]:
    terms = re.findall(r"[\"'“”‘’「」『』]([^\"'“”‘’「」『』]+)[\"'“”‘’「」『』]", text)
    out: list[str] = []
    for term in terms:
        term = _clean(term)
        if term and term not in out:
            out.append(term)
    return out


def _normalize_asset_type(slide: dict[str, Any], asset: dict[str, Any] | None = None) -> str:
    asset = asset or {}
    asset_type = _clean(asset.get("asset_type") or asset.get("type")).lower()

    text = " ".join(
        [
            _clean(asset.get("title")),
            _clean(asset.get("description")),
            _clean(asset.get("raw_text") or asset.get("text")),
            _clean(slide.get("t1_structure")),
            _clean(slide.get("slide_type")),
        ]
    ).lower()
    if _has_table_term(text):
        return "table"
    if any(k in text for k in ("다이어그램", "구조도", "화살표", "diagram")):
        return "diagram"
    if any(k in text for k in ("목록", "리스트", "불릿", "list")):
        return "list"
    if any(k in text for k in ("차트", "그래프", "chart", "graph")):
        return "chart"
    if any(k in text for k in ("계층", "연결")):
        return "diagram"
    if asset_type in VISUAL_ASSET_TYPES:
        return asset_type
    if slide.get("slide_type") in {"mixed", "image_only"}:
        return "figure"
    return "other"


def _ensure_visual_assets(slide: dict[str, Any]) -> list[dict[str, Any]]:
    assets = slide.get("visual_assets")
    if not isinstance(assets, list):
        assets = []

    normalized: list[dict[str, Any]] = []
    for idx, asset in enumerate(assets, start=1):
        if isinstance(asset, str):
            asset = {"description": asset}
        if not isinstance(asset, dict):
            continue
        desc = _clean(asset.get("description") or asset.get("summary"))
        raw_text = _clean(asset.get("raw_text") or asset.get("text"))
        title = _clean(asset.get("title") or asset.get("caption"))
        if not (title or desc or raw_text):
            continue
        item = dict(asset)
        item["asset_index"] = int(item.get("asset_index") or idx)
        item["asset_type"] = _normalize_asset_type(slide, item)
        item["title"] = title
        item["description"] = desc
        item["raw_text"] = raw_text
        if not isinstance(item.get("bbox"), dict):
            item["bbox"] = None
        normalized.append(item)

    if normalized:
        return normalized

    structure = _clean(slide.get("t1_structure"))
    if not structure:
        return []
    return [
        {
            "asset_index": 1,
            "asset_type": _normalize_asset_type(slide, {"description": structure}),
            "title": _clean(slide.get("title")),
            "description": structure,
            "raw_text": "",
            "bbox": None,
        }
    ]


def _add_element(elements: list[dict[str, Any]], type_: str, label: str = "", role: str = "", meaning: str = "") -> None:
    entry = {
        "type": _clean(type_).lower() or "other",
        "label": _clean(label),
        "role": _clean(role),
        "meaning": _clean(meaning),
        "bbox": None,
    }
    key = (entry["type"], entry["label"], entry["role"], entry["meaning"])
    if any((e.get("type"), e.get("label"), e.get("role"), e.get("meaning")) == key for e in elements):
        return
    if entry["label"] or entry["role"] or entry["meaning"]:
        elements.append(entry)


def _add_relation(
    relations: list[dict[str, Any]],
    source: str = "",
    target: str = "",
    relation: str = "",
    visual_cue: str = "",
    direction: str = "",
    meaning: str = "",
) -> None:
    entry = {
        "source": _clean(source),
        "target": _clean(target),
        "relation": _clean(relation),
        "visual_cue": _clean(visual_cue).lower(),
        "direction": _clean(direction).lower(),
        "meaning": _clean(meaning),
    }
    key = tuple(entry.values())
    if any(tuple(r.values()) == key for r in relations):
        return
    if any(entry.values()):
        relations.append(entry)


def _append_layout(layout: dict[str, list[str]], key: str, values: list[str]) -> None:
    if not values:
        return
    bucket = layout.setdefault(key, [])
    for value in values:
        value = _clean(value)
        if value and value not in bucket:
            bucket.append(value)


def _has_table_term(text: str) -> bool:
    text = _clean(text)
    if not text:
        return False
    table_terms = (
        "비교표",
        "테이블",
        "table",
        "표 형태",
        "표 형식",
        "표로 구성",
        "표로 정리",
        "표에서",
        "표에는",
        "표의",
        "표 안",
        "셀",
        "행과 열",
        "행/열",
    )
    if any(term in text for term in table_terms):
        return True
    return bool(re.search(r"(^|[\s\"'“”‘’「」『』])표($|[\s\"'“”‘’「」『』])", text))


def _clean_relation_term(value: str) -> str:
    value = _clean(value)
    value = re.sub(r"^(?:화살표는|이는|이것은|그림은|다이어그램은)\s*", "", value)
    value = re.sub(r"^(?:그리고|또는|및)\s*", "", value)
    return _clean(value)


def _relation_from_sentence(sentence: str) -> str:
    if "의존" in sentence:
        return "depends_on"
    if "요청" in sentence:
        return "requests"
    if "접근" in sentence:
        return "accesses"
    return ""


def _extract_layout(sentences: list[str]) -> dict[str, list[str]]:
    layout: dict[str, list[str]] = {}
    mapping = [
        ("top", ("최상단", "상단", "가장 상위", "상위")),
        ("middle", ("중앙", "중간", "중간 계층", "중앙에는")),
        ("bottom", ("하단", "아래", "가장 하위", "하위")),
        ("left", ("왼쪽", "좌측")),
        ("right", ("오른쪽", "우측")),
    ]
    for sent in sentences:
        terms = _quoted_terms(sent)
        for key, words in mapping:
            if any(word in sent for word in words):
                if terms:
                    _append_layout(layout, key, terms)
                else:
                    # Fallback: keep a short phrase when no quoted labels exist.
                    _append_layout(layout, key, [sent[:80]])
    return layout


def _extract_elements_and_relations(text: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, list[str]]]:
    sentences = _sentences(text)
    elements: list[dict[str, Any]] = []
    relations: list[dict[str, Any]] = []
    layout = _extract_layout(sentences)

    for sent in sentences:
        terms = _quoted_terms(sent)
        if any(k in sent for k in ("화살표", "양방향", "단방향", "방향")):
            direction = "bidirectional" if "양방향" in sent else ("source_to_target" if "화살표" in sent else "")
            _add_element(elements, "arrow", label="화살표", role="관계 표시", meaning=sent)
            _add_relation(relations, visual_cue="arrow", direction=direction, meaning=sent)

        if any(k in sent for k in ("박스", "상자", "테두리", "블록")):
            for term in terms or [sent[:80]]:
                _add_element(elements, "box", label=term, role="묶음/계층 표시", meaning=sent)

        if "아이콘" in sent:
            for term in terms or [sent[:80]]:
                _add_element(elements, "icon_group", label=term, role="예시/구성 요소 표시", meaning=sent)

        if any(k in sent for k in ("계층", "레이어")):
            for term in terms:
                _add_element(elements, "layer", label=term, role="계층", meaning=sent)

        if any(k in sent for k in ("말풍선", "콜아웃", "callout")):
            for term in terms or [sent[:80]]:
                _add_element(elements, "callout", label=term, role="보조 설명", meaning=sent)

        if _has_table_term(sent):
            for term in terms or [sent[:80]]:
                _add_element(elements, "table_cell", label=term, role="표 항목", meaning=sent)

        # Korean dependency/request/manage relation patterns.
        for src, tgt, rel_word in re.findall(
            r"([가-힣A-Za-z0-9#/+()\s]+?)가\s+([가-힣A-Za-z0-9#/+()\s]+?)에\s*(의존|요청|접근)",
            sent,
        ):
            relation = {"의존": "depends_on", "요청": "requests", "접근": "accesses"}.get(rel_word, rel_word)
            _add_relation(
                relations,
                source=_clean_relation_term(src),
                target=_clean_relation_term(tgt),
                relation=relation,
                visual_cue="arrow" if "화살표" in sent else "",
                meaning=sent,
            )

        shared_relation = _relation_from_sentence(sent)
        if shared_relation:
            for src, tgt in re.findall(r"([가-힣A-Za-z0-9#/+()\s]+?)가\s+([가-힣A-Za-z0-9#/+()\s]+?)에(?:,|\s|$)", sent):
                _add_relation(
                    relations,
                    source=_clean_relation_term(src),
                    target=_clean_relation_term(tgt),
                    relation=shared_relation,
                    visual_cue="arrow" if "화살표" in sent else "",
                    meaning=sent,
                )

        for src, tgt, rel_word in re.findall(
            r"([가-힣A-Za-z0-9#/+()\s]+?)는\s+([가-힣A-Za-z0-9#/+()\s]+?)를\s*(관리|제어|보호|제공)",
            sent,
        ):
            relation = {"관리": "manages", "제어": "controls", "보호": "protects", "제공": "provides"}.get(rel_word, rel_word)
            _add_relation(relations, source=src, target=tgt, relation=relation, meaning=sent)

        if any(k in sent for k in ("연결", "이어", "상호작용")):
            _add_relation(relations, relation="connected_to", visual_cue="line/arrow", meaning=sent)

    return elements, relations, layout


def _elements_text(elements: list[dict[str, Any]]) -> str:
    lines = []
    for el in elements:
        parts = [_clean(el.get("type")), _clean(el.get("label")), _clean(el.get("role")), _clean(el.get("meaning"))]
        line = " | ".join(part for part in parts if part)
        if line:
            lines.append(line)
    return "\n".join(lines)


def _relations_text(relations: list[dict[str, Any]]) -> str:
    lines = []
    for rel in relations:
        endpoints = " -> ".join(part for part in [_clean(rel.get("source")), _clean(rel.get("target"))] if part)
        parts = [endpoints, _clean(rel.get("relation")), _clean(rel.get("visual_cue")), _clean(rel.get("direction")), _clean(rel.get("meaning"))]
        line = " | ".join(part for part in parts if part)
        if line:
            lines.append(line)
    return "\n".join(lines)


def _layout_text(layout: dict[str, Any]) -> str:
    lines = []
    for key, value in layout.items():
        if isinstance(value, list):
            value_s = ", ".join(_clean(v) for v in value if _clean(v))
        else:
            value_s = _clean(value)
        if value_s:
            lines.append(f"{key}: {value_s}")
    return "\n".join(lines)


def _slide_id(slide: dict[str, Any]) -> str:
    raw_id = _clean(slide.get("slide_id"))
    if raw_id:
        return raw_id
    slide_number = slide.get("slide_number")
    if slide_number is None:
        slide_number = slide.get("page") or slide.get("scene_number")
    try:
        return f"slide_{int(slide_number):03d}"
    except Exception:
        return ""


def _asset_text(asset: dict[str, Any]) -> str:
    parts = [
        _clean(asset.get("title")),
        _clean(asset.get("description")),
        _clean(asset.get("raw_text") or asset.get("text")),
        _clean(asset.get("visual_elements_text")),
        _clean(asset.get("visual_relations_text")),
        _clean(asset.get("layout_text")),
    ]
    return "\n".join(part for part in parts if part)


def enrich_slide(slide: dict[str, Any]) -> bool:
    assets = _ensure_visual_assets(slide)
    changed = assets != slide.get("visual_assets")
    for asset in assets:
        source_text = "\n".join(
            part
            for part in [
                _clean(asset.get("description")),
                _clean(asset.get("raw_text") or asset.get("text")),
                _clean(slide.get("t1_structure")),
            ]
            if part
        )
        elements, relations, layout = _extract_elements_and_relations(source_text)

        if elements and asset.get("visual_elements") != elements:
            asset["visual_elements"] = elements
            changed = True
        if relations and asset.get("visual_relations") != relations:
            asset["visual_relations"] = relations
            changed = True
        if layout and asset.get("layout") != layout:
            asset["layout"] = layout
            changed = True

        fields = {
            "visual_elements_text": _elements_text(asset.get("visual_elements") or []),
            "visual_relations_text": _relations_text(asset.get("visual_relations") or []),
            "layout_text": _layout_text(asset.get("layout") or {}),
        }
        for key, value in fields.items():
            if value and asset.get(key) != value:
                asset[key] = value
                changed = True

    if changed:
        slide["visual_assets"] = assets
    return changed


def _iter_slide_objects(data: Any) -> list[dict[str, Any]]:
    slides: list[dict[str, Any]] = []
    if isinstance(data, list):
        slides.extend(item for item in data if isinstance(item, dict))
    elif isinstance(data, dict):
        for key in ("slides", "scenes"):
            value = data.get(key)
            if isinstance(value, list):
                slides.extend(item for item in value if isinstance(item, dict))
    return slides


def enrich_file(path: Path, backup: bool = True) -> int:
    data = json.loads(path.read_text(encoding="utf-8"))
    count = 0
    for slide in _iter_slide_objects(data):
        if enrich_slide(slide):
            count += 1

    if count:
        if backup:
            backup_path = path.with_suffix(path.suffix + ".visual_backfill_bak")
            if not backup_path.exists():
                shutil.copy2(path, backup_path)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return count


def _json_dumps(value: Any) -> str:
    return json.dumps(value or {}, ensure_ascii=False)


def _load_enriched_fused(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    changed = 0
    for slide in _iter_slide_objects(data):
        if enrich_slide(slide):
            changed += 1
    if changed:
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return data


def backfill_graph_parquet(fused_path: Path, graph_dir: Path, backup: bool = True) -> dict[str, Any]:
    """Upsert VisualAsset node properties in existing graph parquet without rebuilding concepts."""
    import pandas as pd

    data = _load_enriched_fused(fused_path)
    stem = _clean(data.get("stem")) or fused_path.name.replace("_fused.json", "")
    nodes_path = graph_dir / f"{stem}_nodes.parquet"
    edges_path = graph_dir / f"{stem}_edges.parquet"
    triples_path = graph_dir / f"{stem}_graph_triples.parquet"

    if not nodes_path.is_file():
        raise FileNotFoundError(f"nodes parquet not found: {nodes_path}")
    if not edges_path.is_file():
        raise FileNotFoundError(f"edges parquet not found: {edges_path}")
    if not triples_path.is_file():
        raise FileNotFoundError(f"triples parquet not found: {triples_path}")

    slides_by_id: dict[str, dict[str, Any]] = {}
    visual_nodes: list[dict[str, str]] = []
    visual_edges: list[dict[str, str]] = []
    visual_triples: list[dict[str, str]] = []

    seen_slides: set[str] = set()
    for slide in _iter_slide_objects(data):
        sid = _slide_id(slide)
        if not sid or sid in seen_slides:
            continue
        seen_slides.add(sid)
        assets = _ensure_visual_assets(slide)
        if not assets:
            continue

        visual_asset_text = "\n\n".join(_asset_text(asset) for asset in assets if _asset_text(asset))
        slides_by_id[sid] = {"visual_asset_text": visual_asset_text}

        for idx, asset in enumerate(assets, start=1):
            vid = f"{sid}/visual/{idx:02d}"
            props = {
                "asset_index": asset.get("asset_index", idx),
                "asset_type": asset.get("asset_type") or _normalize_asset_type(slide, asset),
                "description": asset.get("description", ""),
                "raw_text": asset.get("raw_text", ""),
                "visual_elements": asset.get("visual_elements", []),
                "visual_relations": asset.get("visual_relations", []),
                "layout": asset.get("layout", {}),
                "visual_elements_text": asset.get("visual_elements_text", ""),
                "visual_relations_text": asset.get("visual_relations_text", ""),
                "layout_text": asset.get("layout_text", ""),
                "bbox": asset.get("bbox"),
                "slide_id": sid,
                "slide_number": slide.get("slide_number"),
                "scene_number": slide.get("representative_scene_number", slide.get("scene_number")),
                "title": asset.get("title") or slide.get("title", ""),
                "image_path": slide.get("image_path", ""),
            }
            visual_nodes.append({
                "stem": stem,
                "node_id": vid,
                "label": "VisualAsset",
                "properties_json": _json_dumps(props),
            })
            visual_edges.append({
                "stem": stem,
                "src_id": sid,
                "rel_type": "HAS_VISUAL_ASSET",
                "tgt_id": vid,
                "properties_json": "{}",
            })
            visual_triples.extend([
                {"stem": stem, "subject": sid, "predicate": "HAS_VISUAL_ASSET", "object": vid, "properties": ""},
                {"stem": stem, "subject": vid, "predicate": "type", "object": "VisualAsset", "properties": _json_dumps(props)},
            ])

    def backup_file(path: Path) -> None:
        if backup:
            backup_path = path.with_suffix(path.suffix + ".visual_backfill_bak")
            if not backup_path.exists():
                shutil.copy2(path, backup_path)

    nodes = pd.read_parquet(nodes_path)
    edges = pd.read_parquet(edges_path)
    triples = pd.read_parquet(triples_path)

    slide_ids = set(slides_by_id)
    old_visual_ids = set(
        nodes.loc[
            (nodes["label"] == "VisualAsset")
            & nodes["node_id"].astype(str).str.extract(r"^(slide_\d{3})/visual/", expand=False).isin(slide_ids),
            "node_id",
        ].astype(str)
    )
    old_visual_ids.update(f"{sid}/visual/structure" for sid in slide_ids)
    new_visual_ids = {row["node_id"] for row in visual_nodes}
    visual_ids = old_visual_ids | new_visual_ids

    nodes = nodes[~nodes["node_id"].astype(str).isin(visual_ids)].copy()
    if visual_nodes:
        nodes = pd.concat([nodes, pd.DataFrame(visual_nodes)], ignore_index=True)

    def update_slide_props(row: Any) -> str:
        sid = _clean(row.get("node_id"))
        if row.get("label") != "Slide" or sid not in slides_by_id:
            return row.get("properties_json")
        try:
            props = json.loads(row.get("properties_json") or "{}")
        except Exception:
            props = {}
        props["visual_asset_text"] = slides_by_id[sid]["visual_asset_text"]
        return _json_dumps(props)

    nodes["properties_json"] = nodes.apply(update_slide_props, axis=1)

    edges = edges[
        ~(
            edges["src_id"].astype(str).isin(slide_ids)
            & (edges["rel_type"].astype(str) == "HAS_VISUAL_ASSET")
        )
        & ~edges["src_id"].astype(str).isin(visual_ids)
        & ~edges["tgt_id"].astype(str).isin(visual_ids)
    ].copy()
    if visual_edges:
        edges = pd.concat([edges, pd.DataFrame(visual_edges)], ignore_index=True)

    triples = triples[
        ~(
            triples["subject"].astype(str).isin(slide_ids)
            & (triples["predicate"].astype(str) == "HAS_VISUAL_ASSET")
        )
        & ~triples["subject"].astype(str).isin(visual_ids)
        & ~triples["object"].astype(str).isin(visual_ids)
    ].copy()

    def update_slide_triple_props(row: Any) -> str:
        sid = _clean(row.get("subject"))
        if row.get("predicate") != "type" or row.get("object") != "Slide" or sid not in slides_by_id:
            return row.get("properties")
        try:
            props = json.loads(row.get("properties") or "{}")
        except Exception:
            props = {}
        props["visual_asset_text"] = slides_by_id[sid]["visual_asset_text"]
        return _json_dumps(props)

    triples["properties"] = triples.apply(update_slide_triple_props, axis=1)
    if visual_triples:
        triples = pd.concat([triples, pd.DataFrame(visual_triples)], ignore_index=True)

    for path in (nodes_path, edges_path, triples_path):
        backup_file(path)
    nodes.to_parquet(nodes_path, index=False)
    edges.to_parquet(edges_path, index=False)
    triples.to_parquet(triples_path, index=False)

    return {
        "stem": stem,
        "visual_asset_count": len(visual_nodes),
        "slide_count": len(slides_by_id),
        "nodes_path": str(nodes_path),
        "edges_path": str(edges_path),
        "triples_path": str(triples_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill structured visual fields from t1_structure.")
    parser.add_argument("paths", nargs="+", type=Path, help="JSON files to enrich in place")
    parser.add_argument("--graph-dir", type=Path, help="Also upsert VisualAsset fields into existing graph parquet directory")
    parser.add_argument("--fused", type=Path, help="Fused JSON to use for --graph-dir (defaults to first *_fused.json path)")
    parser.add_argument("--no-backup", action="store_true", help="Do not create .visual_backfill_bak files")
    args = parser.parse_args()

    for path in args.paths:
        changed = enrich_file(path, backup=not args.no_backup)
        print(f"{path}: enriched {changed} slide entries")

    if args.graph_dir:
        fused_path = args.fused
        if fused_path is None:
            fused_path = next((path for path in args.paths if path.name.endswith("_fused.json")), None)
        if fused_path is None:
            raise SystemExit("--graph-dir requires --fused or a *_fused.json positional path")
        result = backfill_graph_parquet(fused_path, args.graph_dir, backup=not args.no_backup)
        print(f"{args.graph_dir}: upserted {result['visual_asset_count']} VisualAsset nodes for {result['slide_count']} slides")


if __name__ == "__main__":
    main()
