"""
fused_to_graphrag_text.py
=========================

Convert GraphLec's `{stem}_fused.json` into one natural-language `.txt`
document suitable for Microsoft GraphRAG indexing.

The output keeps slide-level blocks in a single lecture document so GraphRAG can
build concepts over the whole lecture while preserving slide ids for later
linking back to GraphLec's structural graph.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable


SCORE_FIELDS = (
    "score",
    "audio_score",
    "visual_score",
    "annotation_score",
    "slide_text_score",
)


def _clean_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _clean_inline(value: Any) -> str:
    return re.sub(r"\s+", " ", _clean_text(value)).strip()


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _format_seconds(sec: Any) -> str:
    if sec is None:
        return ""
    value = _as_float(sec, -1.0)
    if value < 0:
        return ""
    minutes = int(value) // 60
    seconds = value - minutes * 60
    return f"{minutes:02d}:{seconds:05.2f}"


def _keyword_score(entry: dict[str, Any]) -> float:
    if "score" in entry:
        return _as_float(entry.get("score"))
    return sum(_as_float(entry.get(field)) for field in SCORE_FIELDS if field != "score")


def _iter_context_texts(slide: dict[str, Any]) -> Iterable[str]:
    for ctx in slide.get("contexts") or []:
        segments = [s for s in (ctx.get("segments") or []) if _clean_inline(s.get("text"))]
        if segments and any(s.get("segment_id") for s in segments):
            # embed segment IDs for structural graph linking at ingest time
            parts = []
            for seg in segments:
                seg_id = _clean_inline(seg.get("segment_id"))
                text = _clean_inline(seg.get("text"))
                if seg_id:
                    parts.append(f"[graphlec_seg:{seg_id}] {text}")
                else:
                    parts.append(text)
            if parts:
                yield " ".join(parts)
        else:
            text = _clean_text(ctx.get("text"))
            if text:
                yield text
            elif segments:
                yield " ".join(_clean_inline(s.get("text")) for s in segments)


def _format_keywords(
    slide: dict[str, Any],
    *,
    min_score: float,
    include_scores: bool,
) -> str:
    items: list[tuple[str, float, str]] = []
    for entry in slide.get("emphasized_keywords") or []:
        keyword = _clean_inline(entry.get("keyword") or entry.get("text") or entry.get("name"))
        if not keyword:
            continue
        score = _keyword_score(entry)
        if score <= min_score:
            continue
        sources = ", ".join(str(s) for s in (entry.get("sources") or []) if s)
        items.append((keyword, score, sources))

    if not items:
        return ""

    items.sort(key=lambda item: (-item[1], item[0]))
    if include_scores:
        rendered = [
            f"{keyword}(score {score:.2f}{', sources ' + sources if sources else ''})"
            for keyword, score, sources in items
        ]
    else:
        rendered = [keyword for keyword, _, _ in items]
    return "강조 키워드: " + ", ".join(rendered) + "."


def _format_annotations(slide: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for ann in slide.get("annotations_summary") or []:
        parts: list[str] = []
        ann_type = _clean_inline(ann.get("type"))
        target = _clean_inline(ann.get("target_content"))
        handwritten = _clean_inline(ann.get("handwritten_content"))
        confidence = _clean_inline(ann.get("confidence"))
        timestamp = _format_seconds(ann.get("timestamp_sec"))
        score = ann.get("score")

        if ann_type:
            parts.append(f"유형은 {ann_type}")
        if target:
            parts.append(f"대상 내용은 '{target}'")
        if handwritten:
            parts.append(f"필기 내용은 '{handwritten}'")
        if confidence:
            parts.append(f"신뢰도는 {confidence}")
        if score is not None:
            parts.append(f"점수는 {_as_float(score):.2f}")
        if timestamp:
            parts.append(f"시점은 {timestamp}")

        if parts:
            lines.append("주석 요약: " + ", ".join(parts) + ".")
    return lines


def slide_to_block(
    slide: dict[str, Any],
    *,
    include_slide_text: bool = True,
    include_metadata: bool = True,
    keyword_min_score: float = 0.0,
    include_keyword_scores: bool = True,
) -> str:
    slide_id = _clean_inline(slide.get("slide_id"))
    slide_number = slide.get("slide_number")
    title = _clean_inline(slide.get("title"))
    role = _clean_inline(slide.get("role"))
    start = _format_seconds(slide.get("start_sec"))
    end = _format_seconds(slide.get("end_sec"))

    heading_label = f"슬라이드 {slide_number}" if slide_number is not None else "슬라이드"
    if title:
        heading_label += f": {title}"

    lines = [f"## {heading_label}"]

    if include_metadata:
        tag_parts = []
        if slide_id:
            tag_parts.append(f"id={slide_id}")
        if start and end:
            tag_parts.append(f"time={start}~{end}")
        elif start:
            tag_parts.append(f"time={start}")
        if role:
            tag_parts.append(f"role={role}")
        if tag_parts:
            lines.append("[graphlec:" + " | ".join(tag_parts) + "]")

    if title:
        lines.append(f"제목: {title}.")

    if include_slide_text:
        slide_text = _clean_text(slide.get("slide_text"))
        if slide_text:
            lines.append("슬라이드 원문:\n" + slide_text)

    context_texts = list(_iter_context_texts(slide))
    if context_texts:
        lines.append("강의 전사:")
        lines.extend(context_texts)

    keyword_line = _format_keywords(
        slide,
        min_score=keyword_min_score,
        include_scores=include_keyword_scores,
    )
    if keyword_line:
        lines.append(keyword_line)

    annotation_lines = _format_annotations(slide)
    if annotation_lines:
        lines.extend(annotation_lines)

    return "\n\n".join(line for line in lines if line).strip()


def fused_to_graphrag_text(
    fused: dict[str, Any],
    *,
    stem: str | None = None,
    include_slide_text: bool = True,
    include_metadata: bool = True,
    keyword_min_score: float = 0.0,
    include_keyword_scores: bool = True,
) -> str:
    slides = fused.get("slides") or []
    metadata = fused.get("metadata") or {}
    lecture_name = stem or _clean_inline(metadata.get("stem")) or "lecture"

    blocks = [
        f"# 강의 문서: {lecture_name}",
        (
            "이 문서는 GraphLec fused.json에서 Microsoft GraphRAG 인덱싱을 위해 "
            "슬라이드 단위 블록으로 변환한 단일 강의 텍스트입니다."
        ),
    ]

    for slide in slides:
        block = slide_to_block(
            slide,
            include_slide_text=include_slide_text,
            include_metadata=include_metadata,
            keyword_min_score=keyword_min_score,
            include_keyword_scores=include_keyword_scores,
        )
        if block:
            blocks.append(block)

    return "\n\n---\n\n".join(blocks).strip() + "\n"


def resolve_fused_path(args: argparse.Namespace) -> Path:
    if args.fused_path:
        return args.fused_path
    if not args.stem:
        raise SystemExit("--fused-path 또는 --stem 중 하나는 필요합니다.")
    return args.output_dir / f"{args.stem}_fused.json"


def resolve_output_path(fused_path: Path, args: argparse.Namespace) -> Path:
    if args.output_path:
        return args.output_path
    name = fused_path.name
    if name.endswith("_fused.json"):
        return fused_path.with_name(name[: -len("_fused.json")] + "_graphrag.txt")
    if name.endswith(".json"):
        return fused_path.with_suffix(".graphrag.txt")
    return fused_path.with_name(fused_path.name + "_graphrag.txt")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="GraphLec fused.json을 Microsoft GraphRAG 입력용 단일 txt로 변환합니다."
    )
    parser.add_argument("--fused-path", type=Path, default=None, help="입력 fused.json 경로")
    parser.add_argument("--stem", default=None, help="강의 stem. --fused-path가 없으면 output_dir/{stem}_fused.json 사용")
    parser.add_argument("--output-dir", type=Path, default=Path("output"), help="fused.json 기본 디렉터리")
    parser.add_argument("--output-path", type=Path, default=None, help="출력 txt 경로")
    parser.add_argument(
        "--keyword-min-score",
        type=float,
        default=0.0,
        help="이 값보다 큰 emphasized_keywords만 포함합니다. 기본값 0.0",
    )
    parser.add_argument("--no-slide-text", action="store_true", help="slide_text를 출력에서 제외")
    parser.add_argument("--no-metadata", action="store_true", help="slide_id/time/role 출처 문장을 제외")
    parser.add_argument("--no-keyword-scores", action="store_true", help="강조 키워드 점수와 sources를 숨김")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    fused_path = resolve_fused_path(args)
    output_path = resolve_output_path(fused_path, args)

    with fused_path.open(encoding="utf-8") as f:
        fused = json.load(f)

    text = fused_to_graphrag_text(
        fused,
        stem=args.stem or fused_path.name.removesuffix("_fused.json"),
        include_slide_text=not args.no_slide_text,
        include_metadata=not args.no_metadata,
        keyword_min_score=args.keyword_min_score,
        include_keyword_scores=not args.no_keyword_scores,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")

    slide_count = len(fused.get("slides") or [])
    print(f"wrote {output_path} ({slide_count} slides, {len(text):,} chars)")


if __name__ == "__main__":
    main()
