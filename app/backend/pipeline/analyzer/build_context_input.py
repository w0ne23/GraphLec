from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


_CANDIDATE_KEYS = {
    "text_corrected_candidate",
    "candidate_only",
    "correction_candidate",
}


def _strip_candidate_fields(value):
    if isinstance(value, dict):
        return {
            key: _strip_candidate_fields(item)
            for key, item in value.items()
            if key not in _CANDIDATE_KEYS
        }
    if isinstance(value, list):
        return [_strip_candidate_fields(item) for item in value]
    return value


def _segment_text(seg: dict) -> str:
    # 가짜/수정 전사 실험에서는 사용자가 바꾼 `text`가 정답 입력이다.
    # text_corrected나 candidate 계열 필드가 남아 있어도 analyzer에는 섞지 않는다.
    return str(
        seg.get("text")
        or seg.get("text_corrected")
        or seg.get("text_original")
        or ""
    ).strip()


def _clean_segment(seg: dict) -> dict | None:
    text = _segment_text(seg)
    if not text:
        return None
    start = float(seg.get("start", seg.get("start_time", 0.0)) or 0.0)
    end = float(seg.get("end", seg.get("end_time", start)) or start)
    return {
        "start": start,
        "end": end,
        "text": text,
        "text_corrected": text,
        "text_original": str(seg.get("text_original") or seg.get("text") or text).strip(),
        "correction_status": "locked_input",
    }


def _default_output_path(input_path: Path) -> Path:
    stem = input_path.stem
    if stem.endswith("_merged_clean"):
        job_stem = stem[: -len("_merged_clean")]
    else:
        job_stem = stem
    return input_path.parent / f"{job_stem}_analyzer" / input_path.name


def _load_contexts_by_slide(by_scene_path: Path | None) -> dict[int, list[dict]]:
    if not by_scene_path or not by_scene_path.exists() or by_scene_path.stat().st_size <= 0:
        return {}
    payload = json.loads(by_scene_path.read_text(encoding="utf-8"))
    contexts_by_slide: dict[int, list[dict]] = {}
    for scene in payload.get("scenes") or payload.get("slides") or []:
        if not isinstance(scene, dict):
            continue
        slide_no = scene.get("slide_number", scene.get("slide_index"))
        if not isinstance(slide_no, int):
            continue
        for ctx in scene.get("contexts", []) or []:
            if isinstance(ctx, dict) and str(ctx.get("text", "") or "").strip():
                contexts_by_slide.setdefault(slide_no, []).append(dict(ctx))
    return contexts_by_slide


def _normalize_existing_contexts(
    raw_contexts: list[dict],
    slide_no: int,
    clean_segments: list[dict],
) -> list[dict]:
    contexts = []
    for idx, raw_ctx in enumerate(raw_contexts or []):
        if not isinstance(raw_ctx, dict):
            continue
        text = str(raw_ctx.get("text", "") or "").strip()
        if not text:
            continue
        segment_indices = list(
            raw_ctx.get("source_segment_indices")
            or raw_ctx.get("segment_indices")
            or []
        )
        source_segments = []
        for segment_index in segment_indices:
            try:
                segment_index = int(segment_index)
            except (TypeError, ValueError):
                continue
            if 0 <= segment_index < len(clean_segments):
                source_segments.append(clean_segments[segment_index])
        contexts.append({
            "context_id": str(raw_ctx.get("context_id") or f"S{slide_no:03d}-C{idx + 1:03d}"),
            "slide_number": slide_no,
            "scene_index": raw_ctx.get("scene_index"),
            "visit_order": int(raw_ctx.get("visit_order", 1) or 1),
            "context_index": int(raw_ctx.get("context_index", idx) or 0),
            "start_time": float(raw_ctx.get("start_time", raw_ctx.get("start", 0.0)) or 0.0),
            "end_time": float(raw_ctx.get("end_time", raw_ctx.get("end", raw_ctx.get("start", 0.0))) or 0.0),
            "text": text,
            "text_source": "locked_context_text",
            "analyzer_text_locked": True,
            "source_segment_indices": segment_indices,
            "source_segments": raw_ctx.get("source_segments") or raw_ctx.get("segments") or source_segments,
        })
    return contexts


def build_context_input(
    input_path: Path,
    output_path: Path,
    *,
    use_llm_merge: bool,
    by_scene_path: Path | None = None,
) -> dict:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    result = _strip_candidate_fields(copy.deepcopy(payload))
    slides = result.get("slides") or []
    by_scene_contexts = _load_contexts_by_slide(by_scene_path)

    total_segments = 0
    total_contexts = 0
    for slide in slides:
        slide_no = int(slide.get("slide_number", 0) or 0)
        raw_segments = slide.get("transcript_segments") or []
        clean_segments = []
        for seg in raw_segments:
            if not isinstance(seg, dict):
                continue
            cleaned = _clean_segment(seg)
            if cleaned:
                clean_segments.append(cleaned)

        raw_contexts = by_scene_contexts.get(slide_no) or slide.get("contexts") or []
        contexts = _normalize_existing_contexts(raw_contexts, slide_no, clean_segments)

        if not by_scene_contexts and not contexts and clean_segments:
            for idx, seg in enumerate(clean_segments):
                text = str(seg.get("text", "") or "").strip()
                if not text:
                    continue
                contexts.append({
                    "context_id": f"S{slide_no:03d}-C{idx + 1:03d}",
                    "slide_number": slide_no,
                    "scene_index": slide.get("scene_index"),
                    "visit_order": int(slide.get("visit_order", 1) or 1),
                    "context_index": idx,
                    "start_time": float(seg.get("start", 0.0) or 0.0),
                    "end_time": float(seg.get("end", seg.get("start", 0.0)) or 0.0),
                    "text": text,
                    "text_source": "locked_context_text",
                    "analyzer_text_locked": True,
                    "source_segment_indices": [],
                    "source_segments": [seg],
                })

        slide["transcript_segments"] = clean_segments
        slide["transcript"] = " ".join(seg["text"] for seg in clean_segments).strip()
        slide["segment_count"] = len(clean_segments)
        slide["contexts"] = contexts
        slide["context_count"] = len(contexts)
        total_segments += len(clean_segments)
        total_contexts += len(contexts)

    result["analyzer_transcript_policy"] = "locked_context_text_from_input"
    result["analyzer_context_source"] = str(input_path)
    if by_scene_contexts:
        result["analyzer_context_source"] = str(by_scene_path)
    result["total_transcript_segments"] = total_segments
    result["total_contexts"] = total_contexts

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "total_slides": len(slides),
        "total_transcript_segments": total_segments,
        "total_contexts": total_contexts,
        "use_llm_merge": use_llm_merge,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build analyzer merged_clean with locked context units from an input merged_clean JSON.",
    )
    parser.add_argument("input_json")
    parser.add_argument("--output", default="")
    parser.add_argument(
        "--no-llm-merge",
        action="store_true",
        help="Compatibility flag. Contexts are loaded from *_by_scene.json or existing slide contexts.",
    )
    parser.add_argument(
        "--by-scene-json",
        default="",
        help="Use existing *_by_scene.json contexts instead of regenerating contexts.",
    )
    args = parser.parse_args()

    input_path = Path(args.input_json).resolve()
    output_path = Path(args.output).resolve() if args.output else _default_output_path(input_path)
    summary = build_context_input(
        input_path,
        output_path,
        use_llm_merge=not args.no_llm_merge,
        by_scene_path=Path(args.by_scene_json).resolve() if args.by_scene_json else None,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
