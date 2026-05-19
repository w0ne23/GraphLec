#!/usr/bin/env python3
"""Run the analyzer repeatedly and summarize output stability.

This is an evaluation harness, not part of the production analyzer path.
Use it to check whether the same input/prompt produces the same issues across
multiple runs.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any

from evaluate_gold_set import evaluate as evaluate_gold


VISIBLE_STATUSES = {"confirmed", "professor_check"}


def _compact(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _base_stem(merged_path: Path) -> str:
    return merged_path.stem.replace("_merged_clean", "").replace("_merged", "")


def _result_stem(base_stem: str, suffix: str) -> str:
    suffix = re.sub(r"[^A-Za-z0-9가-힣_.-]+", "_", suffix).strip("_.-")
    return f"{base_stem}_{suffix}" if suffix else base_stem


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _issue_key(item: dict[str, Any]) -> str:
    claim = _compact(item.get("claim_text") or item.get("source_text") or "")
    if not claim:
        problem = item.get("problem") if isinstance(item.get("problem"), dict) else {}
        claim = _compact(problem.get("source_text") or problem.get("summary") or "")
    return claim.lower()[:180]


def _issue_rows(path: Path) -> list[dict[str, Any]]:
    payload = _load_json(path)
    rows: list[dict[str, Any]] = []
    for item in payload.get("feedback_items") or []:
        if not isinstance(item, dict):
            continue
        problem = item.get("problem") if isinstance(item.get("problem"), dict) else {}
        rows.append(
            {
                "key": _issue_key(item),
                "feedback_id": item.get("feedback_id"),
                "status": item.get("status"),
                "score": item.get("score", item.get("confidence")),
                "feedback_label": item.get("feedback_label"),
                "feedback_type": item.get("feedback_type"),
                "claim": _compact(item.get("claim_text") or item.get("source_text") or ""),
                "summary": _compact(problem.get("summary") or ""),
            }
        )
    return rows


def _summary(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    summary = payload.get("summary") or {}
    return {
        "extracted_claim_count": summary.get("extracted_claim_count", 0),
        "issue_union_raw_count": summary.get("issue_union_raw_count", 0),
        "issue_clustered_count": summary.get("issue_clustered_count", 0),
        "confirmed": summary.get("confirmed_feedback_count", 0),
        "professor_check": summary.get("professor_check_feedback_count", 0),
        "rejected": summary.get("rejected_feedback_count", 0),
    }


def _format_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        cells = []
        for value in row:
            text = str(value).replace("\n", " ").replace("|", "\\|")
            cells.append(text)
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _status_rank(status: str) -> int:
    return {"confirmed": 3, "professor_check": 2, "rejected": 1}.get(str(status or ""), 0)


def _build_report(
    *,
    merged_path: Path,
    output_dir: Path,
    run_records: list[dict[str, Any]],
    report_path: Path,
    gold_path: Path | None,
) -> str:
    successful = [r for r in run_records if r["exit_code"] == 0 and r["result_path"].exists()]
    lines = [
        "# Analyzer Stability Report",
        "",
        f"- input: `{merged_path}`",
        f"- output_dir: `{output_dir}`",
        f"- runs: `{len(run_records)}`",
        f"- successful: `{len(successful)}/{len(run_records)}`",
        f"- generated_at: `{datetime.now().isoformat(timespec='seconds')}`",
        "",
    ]

    summary_rows = []
    for record in run_records:
        s = _summary(record["result_path"]) if record["result_path"].exists() else {}
        summary_rows.append(
            [
                record["run"],
                record["exit_code"],
                round(record["elapsed_sec"], 1),
                s.get("extracted_claim_count", 0),
                s.get("issue_union_raw_count", 0),
                s.get("issue_clustered_count", 0),
                s.get("confirmed", 0),
                s.get("professor_check", 0),
                s.get("rejected", 0),
                record["result_path"].name,
            ]
        )
    lines += [
        "## Run Summary",
        "",
        _format_table(
            ["run", "exit", "sec", "claims", "raw", "merged", "confirmed", "check", "rejected", "file"],
            summary_rows,
        ),
        "",
    ]

    all_rows: list[tuple[int, dict[str, Any]]] = []
    for record in successful:
        for row in _issue_rows(record["result_path"]):
            all_rows.append((record["run"], row))

    issue_runs: dict[str, set[int]] = defaultdict(set)
    issue_statuses: dict[str, Counter] = defaultdict(Counter)
    issue_scores: dict[str, list[float]] = defaultdict(list)
    examples: dict[str, dict[str, Any]] = {}
    for run_no, row in all_rows:
        key = row["key"]
        issue_runs[key].add(run_no)
        issue_statuses[key][str(row.get("status") or "")] += 1
        examples.setdefault(key, row)
        try:
            issue_scores[key].append(float(row.get("score")))
        except (TypeError, ValueError):
            pass

    stability_rows = []
    for key, runs in sorted(issue_runs.items(), key=lambda item: (-len(item[1]), item[0])):
        ex = examples[key]
        statuses = issue_statuses[key]
        status = max(statuses, key=lambda s: (_status_rank(s), statuses[s])) if statuses else ""
        scores = issue_scores.get(key) or []
        stability_rows.append(
            [
                f"{len(runs)}/{len(successful)}",
                status,
                round(sum(scores) / len(scores), 3) if scores else "",
                ex.get("feedback_label") or "",
                (ex.get("claim") or ex.get("summary") or "")[:140],
            ]
        )

    lines += [
        "## Issue Stability",
        "",
        "같은 claim 텍스트 기준으로 반복 출현 횟수를 계산합니다. `3/3`은 안정 출력, `1/3`은 흔들리는 출력입니다.",
        "",
        _format_table(["seen", "status", "avg_score", "label", "claim"], stability_rows),
        "",
    ]

    if gold_path and gold_path.exists():
        gold_rows = []
        visible_counts = []
        for record in successful:
            gold_report = evaluate_gold(gold_path, record["result_path"])
            visible_counts.append(gold_report["summary"]["visible_confirmed_or_professor_check"])
            for row in gold_report["rows"]:
                gold_rows.append(
                    [
                        record["run"],
                        row.get("gold_id"),
                        row.get("status"),
                        row.get("matched_feedback_id") or "",
                        row.get("actual_label") or "",
                        _compact(row.get("claim") or "")[:100],
                    ]
                )
        lines += [
            "## Gold Set Stability",
            "",
            f"- visible counts by run: `{visible_counts}`",
            "",
            _format_table(["run", "gold", "status", "feedback", "actual", "claim"], gold_rows),
            "",
        ]

    failed = [r for r in run_records if r["exit_code"] != 0]
    if failed:
        lines += ["## Failed Runs", ""]
        for record in failed:
            lines.append(f"- run {record['run']}: exit={record['exit_code']}, log=`{record['log_path']}`")
        lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return str(report_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run analyzer repeatedly and report output stability")
    parser.add_argument("merged_path", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--claims-jsonl", type=Path, default=None)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--prefix", default="stability")
    parser.add_argument(
        "--gold",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "app"
        / "backend"
        / "pipeline"
        / "analyzer"
        / "gold_sets"
        / "os_concepts_gold_set.json",
    )
    parser.add_argument("--cross-models", nargs="+", default=["gpt-5.4", "claude-sonnet-4.5"])
    parser.add_argument("--crosscheck-models", nargs="+", default=["gpt-5.4", "claude-sonnet-4.5", "grok-4.3"])
    parser.add_argument("--judge-max-workers", type=int, default=6)
    parser.add_argument("--crosscheck-group-max-workers", type=int, default=6)
    parser.add_argument("--issue-cluster-max-workers", type=int, default=6)
    parser.add_argument("--cross-batch-size", type=int, default=5)
    args = parser.parse_args()

    merged_path = args.merged_path.resolve()
    output_dir = (args.output_dir or merged_path.parent).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    base_stem = _base_stem(merged_path)
    report_dir = output_dir / f"{args.prefix}_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    report_dir.mkdir(parents=True, exist_ok=True)

    run_records: list[dict[str, Any]] = []
    for run_no in range(1, max(1, args.runs) + 1):
        suffix = f"{args.prefix}_run{run_no}"
        result_path = output_dir / f"{_result_stem(base_stem, suffix)}_content_verification.json"
        log_path = report_dir / f"run_{run_no}.log"
        cmd = [
            sys.executable,
            "-m",
            "pipeline.analyzer.run_all",
            str(merged_path),
            "--output-dir",
            str(output_dir),
            "--result-suffix",
            suffix,
            "--judge-max-workers",
            str(args.judge_max_workers),
            "--crosscheck-group-max-workers",
            str(args.crosscheck_group_max_workers),
            "--issue-cluster-max-workers",
            str(args.issue_cluster_max_workers),
            "--cross-batch-size",
            str(args.cross_batch_size),
            "--cross-models",
            *args.cross_models,
            "--crosscheck-models",
            *args.crosscheck_models,
        ]
        if args.claims_jsonl:
            cmd += ["--claims-jsonl", str(args.claims_jsonl.resolve())]

        print(f"\n=== stability run {run_no}/{args.runs} ===")
        print(" ".join(cmd))
        started = time.time()
        with log_path.open("w", encoding="utf-8") as fp:
            fp.write("$ " + " ".join(cmd) + "\n\n")
            fp.flush()
            proc = subprocess.run(cmd, cwd=Path.cwd(), stdout=fp, stderr=subprocess.STDOUT, text=True)
        elapsed = time.time() - started
        print(f"run {run_no} exit={proc.returncode} elapsed={elapsed:.1f}s log={log_path}")
        run_records.append(
            {
                "run": run_no,
                "exit_code": proc.returncode,
                "elapsed_sec": elapsed,
                "result_path": result_path,
                "log_path": log_path,
            }
        )

    report_path = report_dir / "STABILITY_REPORT.md"
    _build_report(
        merged_path=merged_path,
        output_dir=output_dir,
        run_records=run_records,
        report_path=report_path,
        gold_path=args.gold.resolve() if args.gold else None,
    )
    print(f"\nreport: {report_path}")


if __name__ == "__main__":
    main()
