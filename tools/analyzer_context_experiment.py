#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_JOB_ID = "3158f481-8604-401e-8468-c322edddcd37"


def _run(cmd: list[str], cwd: Path, *, log_path: Path | None = None) -> tuple[int, float]:
    started = time.time()
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as fp:
            fp.write("$ " + " ".join(cmd) + "\n\n")
            fp.flush()
            proc = subprocess.run(cmd, cwd=cwd, stdout=fp, stderr=subprocess.STDOUT, text=True)
    else:
        proc = subprocess.run(cmd, cwd=cwd, text=True)
    return proc.returncode, time.time() - started


def _capture(cmd: list[str], cwd: Path) -> str:
    proc = subprocess.run(cmd, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return proc.stdout.strip()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _jsonl_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as fp:
        return sum(1 for line in fp if line.strip())


def _result_summary(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    summary = payload.get("summary") or {}
    score_report = payload.get("crosscheck_score_report") or {}
    model_reports = score_report.get("model_reports") or []
    return {
        "path": str(path),
        "exists": path.exists(),
        "extracted_claim_count": summary.get("extracted_claim_count", 0),
        "issue_union_raw_count": summary.get("issue_union_raw_count", summary.get("feedback_candidate_count", 0)),
        "issue_clustered_count": summary.get("issue_clustered_count", summary.get("feedback_candidate_count", 0)),
        "confirmed": summary.get("confirmed_feedback_count", 0),
        "professor_check": summary.get("professor_check_feedback_count", 0),
        "rejected": summary.get("rejected_feedback_count", 0),
        "slide_typos": len(payload.get("slide_typos") or []),
        "models": {
            str(row.get("model")): {
                "mean_vote_score": row.get("mean_vote_score"),
                "agree_rate": row.get("agree_rate"),
                "disagree_rate": row.get("disagree_rate"),
                "inconclusive_rate": row.get("inconclusive_rate"),
                "failed_rate": row.get("failed_rate"),
            }
            for row in model_reports
            if row.get("model")
        },
    }


def _compact(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _issue_key(item: dict[str, Any]) -> str:
    ids = item.get("utterance_ids") or [item.get("utterance_id") or item.get("source_claim_id") or ""]
    ids_text = ",".join(str(x) for x in ids if x)
    claim = _compact(item.get("claim_text") or item.get("resolved_claim") or item.get("problem", {}).get("problematic_content") or "")
    return f"{ids_text}|{claim[:100]}"


def _issue_rows(path: Path) -> list[dict[str, Any]]:
    payload = _load_json(path)
    rows = []
    for item in payload.get("feedback_items") or []:
        problem = item.get("problem") or {}
        evidence = item.get("evidence") or {}
        rows.append({
            "key": _issue_key(item),
            "status": item.get("status", ""),
            "score": item.get("confidence", item.get("score")),
            "utterance_id": item.get("utterance_id", ""),
            "utterance_ids": item.get("utterance_ids") or [],
            "claim_text": _compact(item.get("claim_text") or item.get("resolved_claim") or ""),
            "feedback_label": item.get("feedback_label", ""),
            "summary": _compact(problem.get("summary") or ""),
            "slide_number": evidence.get("slide_number") or item.get("location", {}).get("slide_number"),
        })
    return rows


def _status_rank(status: str) -> int:
    return {"confirmed": 3, "professor_check": 2, "rejected": 1}.get(status, 0)


def _format_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        values = [str(v).replace("\n", " ") for v in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _build_report(
    *,
    repo: Path,
    report_dir: Path,
    baseline_path: Path,
    run_records: list[dict[str, Any]],
    started_at: str,
    finished_at: str,
) -> str:
    git_status = _capture(["git", "status", "--short", "--branch"], repo)
    git_stat = _capture(["git", "diff", "--stat"], repo)
    baseline = _result_summary(baseline_path) if baseline_path.exists() else {}

    summaries = []
    all_issue_rows: list[tuple[int, dict[str, Any]]] = []
    for record in run_records:
        result_path = Path(record.get("result_path", ""))
        summary = _result_summary(result_path) if result_path.exists() else {}
        summary.update({
            "run": record["run"],
            "exit_code": record["exit_code"],
            "elapsed_sec": round(record["elapsed_sec"], 1),
            "claims_jsonl_count": _jsonl_count(Path(record.get("claims_path", ""))),
        })
        summaries.append(summary)
        if result_path.exists():
            for row in _issue_rows(result_path):
                all_issue_rows.append((record["run"], row))

    issue_runs: dict[str, set[int]] = defaultdict(set)
    issue_statuses: dict[str, Counter] = defaultdict(Counter)
    issue_scores: dict[str, list[float]] = defaultdict(list)
    issue_examples: dict[str, dict[str, Any]] = {}
    for run_no, row in all_issue_rows:
        key = row["key"]
        issue_runs[key].add(run_no)
        issue_statuses[key][row["status"]] += 1
        try:
            issue_scores[key].append(float(row["score"]))
        except (TypeError, ValueError):
            pass
        issue_examples.setdefault(key, row)

    stable_rows = []
    for key, runs in sorted(issue_runs.items(), key=lambda item: (-len(item[1]), item[0])):
        ex = issue_examples[key]
        scores = issue_scores.get(key) or []
        status_counts = issue_statuses[key]
        main_status = max(status_counts.items(), key=lambda item: (_status_rank(item[0]), item[1]))[0]
        stable_rows.append([
            f"{len(runs)}/{len(run_records)}",
            main_status,
            round(sum(scores) / len(scores), 3) if scores else "",
            ex.get("slide_number") or "",
            ",".join(ex.get("utterance_ids") or [ex.get("utterance_id", "")]),
            ex.get("feedback_label") or "",
            ex.get("claim_text")[:120],
        ])

    summary_rows = []
    for s in summaries:
        summary_rows.append([
            s.get("run"),
            s.get("exit_code"),
            s.get("elapsed_sec"),
            s.get("extracted_claim_count", 0),
            s.get("issue_union_raw_count", 0),
            s.get("issue_clustered_count", 0),
            s.get("confirmed", 0),
            s.get("professor_check", 0),
            s.get("rejected", 0),
            s.get("slide_typos", 0),
        ])

    success = [s for s in summaries if s.get("exit_code") == 0 and s.get("extracted_claim_count", 0)]
    avg_claims = round(sum(s.get("extracted_claim_count", 0) for s in success) / len(success), 2) if success else 0
    avg_issues = round(sum(s.get("issue_clustered_count", 0) for s in success) / len(success), 2) if success else 0

    lines = [
        "# Analyzer Context-First 반복 실험 리포트",
        "",
        f"- 시작: `{started_at}`",
        f"- 종료: `{finished_at}`",
        f"- 리포트 디렉토리: `{report_dir}`",
        f"- 성공 run: `{len(success)}/{len(run_records)}`",
        f"- 성공 run 평균 claim 수: `{avg_claims}`",
        f"- 성공 run 평균 merge 후 issue 수: `{avg_issues}`",
        "",
        "## 현재 로컬 변경 상태",
        "",
        "```text",
        git_status,
        "```",
        "",
        "## 변경 파일 규모",
        "",
        "```text",
        git_stat,
        "```",
        "",
        "## 기준 결과",
        "",
    ]
    if baseline:
        lines.append(_format_table(
            ["baseline", "claim", "raw issue", "merged issue", "confirmed", "professor", "rejected", "typo"],
            [[
                baseline_path.name,
                baseline.get("extracted_claim_count", 0),
                baseline.get("issue_union_raw_count", 0),
                baseline.get("issue_clustered_count", 0),
                baseline.get("confirmed", 0),
                baseline.get("professor_check", 0),
                baseline.get("rejected", 0),
                baseline.get("slide_typos", 0),
            ]],
        ))
    else:
        lines.append("- baseline 파일을 찾지 못했습니다.")

    lines += [
        "",
        "## 10회 반복 요약",
        "",
        _format_table(
            ["run", "exit", "sec", "claim", "raw", "merged", "confirmed", "professor", "rejected", "typo"],
            summary_rows,
        ),
        "",
        "## 반복 안정성 기준 이슈",
        "",
    ]
    if stable_rows:
        lines.append(_format_table(
            ["등장", "주상태", "평균점수", "slide", "ids", "type", "claim"],
            stable_rows[:30],
        ))
    else:
        lines.append("- 성공한 issue 결과가 없어 안정성 표를 만들 수 없습니다.")

    lines += [
        "",
        "## 모델별 평균 반응",
        "",
    ]
    model_acc: dict[str, list[float]] = defaultdict(list)
    for s in summaries:
        for model, row in (s.get("models") or {}).items():
            value = row.get("mean_vote_score")
            if value is not None:
                model_acc[model].append(float(value))
    if model_acc:
        lines.append(_format_table(
            ["model", "runs", "mean_vote_score_avg"],
            [[model, len(vals), round(sum(vals) / len(vals), 4)] for model, vals in sorted(model_acc.items())],
        ))
    else:
        lines.append("- 모델별 score report가 없습니다.")

    lines += [
        "",
        "## 1차 판단",
        "",
        "- context-first 입력은 `slides[].contexts[]`를 analyzer utterance로 사용하므로 지시어/문장 조각 문제를 줄이는 방향입니다.",
        "- 운영 기본 batch는 context 1개당 1회 호출이 아니라 slide 단위 batch로 둬서 API 호출 수를 낮췄습니다.",
        "- confirmed/professor_check로 반복 등장하는 이슈를 우선 검토하고, 1회성 rejected/low-score 이슈는 프롬프트나 merge 기준 개선 대상으로 분리하는 것이 좋습니다.",
        "- 이 리포트는 실험 산출물이며, 최종 코드 채택/커밋/푸시는 별도로 판단해야 합니다.",
    ]
    return "\n".join(lines) + "\n"


def _find_result(output_dir: Path, input_stem: str, suffix: str) -> Path:
    expected = output_dir / f"{input_stem}_{suffix}_content_verification.json"
    if expected.exists():
        return expected
    matches = sorted(output_dir.glob(f"*{suffix}*_content_verification.json"), key=lambda p: p.stat().st_mtime)
    return matches[-1] if matches else expected


def main() -> int:
    parser = argparse.ArgumentParser(description="Run repeated analyzer context-first experiments and create a report.")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--job-id", default=DEFAULT_JOB_ID)
    parser.add_argument("--repo", default=".")
    parser.add_argument("--input-name", default=None)
    parser.add_argument("--baseline-name", default=None)
    parser.add_argument("--claim-workers", type=int, default=2)
    parser.add_argument("--judge-workers", type=int, default=2)
    parser.add_argument("--cross-workers", type=int, default=4)
    parser.add_argument("--cluster-workers", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    result_dir = repo / "local_storage" / "results" / args.job_id
    stem = args.job_id
    input_name = args.input_name or f"{stem}_anothor_context_input.json"
    baseline_name = args.baseline_name or f"{stem}_parallel_w4_b5_content_verification.json"
    input_path = result_dir / input_name
    baseline_path = result_dir / baseline_name

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = result_dir / f"context_repeat_report_{timestamp}"
    report_dir.mkdir(parents=True, exist_ok=True)
    started_at = dt.datetime.now().isoformat(timespec="seconds")

    preflight = {
        "repo": str(repo),
        "job_id": args.job_id,
        "input_path": str(input_path),
        "input_exists": input_path.exists(),
        "baseline_path": str(baseline_path),
        "baseline_exists": baseline_path.exists(),
        "docker_services": _capture(["docker", "compose", "ps", "--services", "--filter", "status=running"], repo),
        "git_status": _capture(["git", "status", "--short", "--branch"], repo),
    }
    (report_dir / "preflight.json").write_text(json.dumps(preflight, ensure_ascii=False, indent=2), encoding="utf-8")

    run_records: list[dict[str, Any]] = []
    input_stem = Path(input_name).stem
    if not input_path.exists():
        finished_at = dt.datetime.now().isoformat(timespec="seconds")
        report = _build_report(
            repo=repo,
            report_dir=report_dir,
            baseline_path=baseline_path,
            run_records=run_records,
            started_at=started_at,
            finished_at=finished_at,
        )
        (report_dir / "REPORT.md").write_text(report, encoding="utf-8")
        print(report_dir / "REPORT.md")
        return 2

    for run_no in range(1, args.runs + 1):
        suffix = f"context_repeat_{timestamp}_r{run_no:02d}"
        log_path = report_dir / "logs" / f"run_{run_no:02d}.log"
        container_input = f"/pipeline/local_storage/results/{args.job_id}/{input_name}"
        container_output = f"/pipeline/local_storage/results/{args.job_id}"
        cmd = [
            "docker", "compose", "exec", "-T",
            "-e", "VERIFIER_CLAIM_EXTRACT_CONTEXT_MODE=card_lite",
            "-e", "VERIFIER_CLAIM_EXTRACT_BATCH_MODE=context",
            "-e", "VERIFIER_CLAIM_EXTRACT_CONTEXT_PREV=2",
            "-e", "VERIFIER_CLAIM_EXTRACT_CONTEXT_NEXT=1",
            "-e", "VERIFIER_CLAIM_EXTRACT_CONTEXT_GROUP_SIZE=4",
            "-e", f"VERIFIER_CLAIM_EXTRACT_MAX_WORKERS={args.claim_workers}",
            "-e", f"VERIFIER_JUDGE_BATCH_MAX_WORKERS={args.judge_workers}",
            "-e", "VERIFIER_CROSSCHECK_MAX_ISSUES_PER_BATCH=5",
            "-e", f"VERIFIER_CROSSCHECK_GROUP_MAX_WORKERS={args.cross_workers}",
            "-e", f"VERIFIER_ISSUE_CLUSTER_MAX_WORKERS={args.cluster_workers}",
            "backend", "python", "-m", "pipeline.analyzer.run_all",
            container_input,
            "--output-dir", container_output,
            "--result-suffix", suffix,
            "--claim-max-workers", str(args.claim_workers),
            "--judge-max-workers", str(args.judge_workers),
            "--crosscheck-group-max-workers", str(args.cross_workers),
            "--issue-cluster-max-workers", str(args.cluster_workers),
            "--cross-models", "gpt-5.4", "claude-sonnet-4.5",
            "--crosscheck-models", "gpt-5.4", "claude-sonnet-4.5", "grok-4.3",
        ]
        if args.dry_run:
            exit_code, elapsed = 0, 0.0
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text("$ " + " ".join(cmd) + "\n", encoding="utf-8")
        else:
            exit_code, elapsed = _run(cmd, repo, log_path=log_path)
        result_path = _find_result(result_dir, input_stem, suffix)
        claims_path = result_path.with_name(result_path.name.replace("_content_verification.json", "_claims_extracted.jsonl"))
        record = {
            "run": run_no,
            "suffix": suffix,
            "exit_code": exit_code,
            "elapsed_sec": elapsed,
            "log_path": str(log_path),
            "result_path": str(result_path),
            "claims_path": str(claims_path),
        }
        run_records.append(record)
        (report_dir / "run_records.json").write_text(json.dumps(run_records, ensure_ascii=False, indent=2), encoding="utf-8")

    finished_at = dt.datetime.now().isoformat(timespec="seconds")
    report = _build_report(
        repo=repo,
        report_dir=report_dir,
        baseline_path=baseline_path,
        run_records=run_records,
        started_at=started_at,
        finished_at=finished_at,
    )
    report_path = report_dir / "REPORT.md"
    report_path.write_text(report, encoding="utf-8")
    print(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
