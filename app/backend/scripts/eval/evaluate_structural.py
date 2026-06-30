#!/usr/bin/env python3
"""Evaluate structural QnA results with slide and time overlap metrics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def collect_pred_slides(row: dict[str, Any]) -> set[int]:
    slides: set[int] = set()
    for key in ("related_slides", "retrieved_chunks", "timestamps"):
        for item in row.get(key) or []:
            value = item.get("slide_number")
            if value is None:
                value = item.get("slide")
            try:
                if value is not None:
                    slides.add(int(value))
            except Exception:
                continue
    return slides


def collect_pred_ranges(row: dict[str, Any]) -> list[tuple[float, float]]:
    ranges: list[tuple[float, float]] = []
    for key in ("timestamps", "retrieved_chunks", "related_slides"):
        for item in row.get(key) or []:
            start = item.get("start_sec")
            end = item.get("end_sec")
            if start is None:
                start = item.get("start")
            if end is None:
                end = item.get("end")
            try:
                if start is not None and end is not None:
                    s = float(start)
                    e = float(end)
                    if e >= s:
                        ranges.append((s, e))
            except Exception:
                continue
    return ranges


def max_time_iou(pred: list[tuple[float, float]], ref: list[dict[str, Any]]) -> float:
    best = 0.0
    for ps, pe in pred:
        for rr in ref:
            rs = float(rr.get("start_sec", 0.0))
            re = float(rr.get("end_sec", 0.0))
            inter = max(0.0, min(pe, re) - max(ps, rs))
            union = max(pe, re) - min(ps, rs)
            if union > 0:
                best = max(best, inter / union)
    return best


def evaluate_row(row: dict[str, Any]) -> dict[str, Any]:
    ref_slides = {int(x) for x in row.get("reference_slides") or []}
    pred_slides = collect_pred_slides(row)
    pred_ranges = collect_pred_ranges(row)
    time_iou = max_time_iou(pred_ranges, row.get("reference_time_ranges") or [])
    slide_hit = bool(ref_slides and pred_slides and ref_slides.intersection(pred_slides))
    return {
        "id": row.get("id"),
        "question": row.get("question"),
        "http_status": row.get("http_status"),
        "source_mode": row.get("source_mode"),
        "slide_hit": int(slide_hit),
        "pred_slides": ",".join(str(x) for x in sorted(pred_slides)),
        "ref_slides": ",".join(str(x) for x in sorted(ref_slides)),
        "time_iou": round(time_iou, 4),
        "has_response": int(bool(row.get("response"))),
        "response": row.get("response", ""),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "id",
        "question",
        "http_status",
        "source_mode",
        "slide_hit",
        "pred_slides",
        "ref_slides",
        "time_iou",
        "has_response",
        "response",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    rows = [row for row in read_jsonl(args.input) if row.get("type") == "structural"]
    scores = [evaluate_row(row) for row in rows]
    write_csv(args.out, scores)
    if scores:
        slide_acc = sum(row["slide_hit"] for row in scores) / len(scores)
        avg_iou = sum(float(row["time_iou"]) for row in scores) / len(scores)
        print(f"structural_count={len(scores)} slide_accuracy={slide_acc:.3f} avg_time_iou={avg_iou:.3f}")
    print(f"saved: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
