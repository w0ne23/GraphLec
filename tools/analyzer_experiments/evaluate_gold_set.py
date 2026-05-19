#!/usr/bin/env python3
"""Evaluate analyzer output against a manually curated gold set.

This module is intentionally outside the analyzer decision path. It reads a
completed content_verification JSON and reports whether known injected issues
surfaced in the final output.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


VISIBLE_STATUSES = {"confirmed", "professor_check"}


def _norm(text: Any) -> str:
    return " ".join(str(text or "").replace("\u200b", " ").split()).lower()


def _item_text(item: dict[str, Any]) -> str:
    """Match only the judged issue text, not broad context evidence."""
    parts: list[str] = []
    for key in ("claim_text", "source_text", "feedback_label", "feedback_type"):
        parts.append(str(item.get(key) or ""))

    problem = item.get("problem") if isinstance(item.get("problem"), dict) else {}
    for key in (
        "source_text",
        "summary",
        "context_issue_summary",
        "context_resolution",
        "context_resolution_reason",
        "correction_hint",
    ):
        parts.append(str(problem.get(key) or ""))

    return _norm("\n".join(parts))


def _matches(gold: dict[str, Any], item_text: str) -> bool:
    any_terms = [_norm(term) for term in gold.get("must_include_any") or [] if _norm(term)]
    all_terms = [_norm(term) for term in gold.get("must_include_all") or [] if _norm(term)]
    if any_terms and not any(term in item_text for term in any_terms):
        return False
    return all(term in item_text for term in all_terms)


def evaluate(gold_path: Path, result_path: Path) -> dict[str, Any]:
    gold = json.loads(gold_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    items = result.get("feedback_items") or []
    searchable = [(item, _item_text(item)) for item in items if isinstance(item, dict)]

    rows: list[dict[str, Any]] = []
    found_visible = 0
    found_any = 0
    for target in gold.get("items") or []:
        matches = [(item, text) for item, text in searchable if _matches(target, text)]
        visible = [item for item, _ in matches if str(item.get("status") or "") in VISIBLE_STATUSES]
        best = visible[0] if visible else (matches[0][0] if matches else None)
        status = str(best.get("status") or "missing") if best else "missing"
        if matches:
            found_any += 1
        if visible:
            found_visible += 1
        rows.append(
            {
                "gold_id": target.get("gold_id"),
                "label": target.get("label"),
                "expected_issue_type": target.get("expected_issue_type"),
                "matched": bool(matches),
                "visible": bool(visible),
                "matched_feedback_id": best.get("feedback_id") if best else None,
                "status": status,
                "score": best.get("score", best.get("confidence")) if best else None,
                "actual_issue_type": best.get("feedback_type") if best else None,
                "actual_label": best.get("feedback_label") if best else None,
                "claim": (best.get("claim_text") or best.get("source_text")) if best else None,
            }
        )

    return {
        "gold_set_id": gold.get("gold_set_id"),
        "result_file": str(result_path),
        "summary": {
            "gold_count": len(gold.get("items") or []),
            "matched_any_status": found_any,
            "visible_confirmed_or_professor_check": found_visible,
            "missing": len(gold.get("items") or []) - found_any,
            "hidden_by_rejection": found_any - found_visible,
        },
        "rows": rows,
    }


def print_markdown(report: dict[str, Any]) -> None:
    summary = report["summary"]
    print(f"# Gold Set Evaluation: {report['gold_set_id']}")
    print()
    print(
        f"- gold: {summary['gold_count']}, matched: {summary['matched_any_status']}, "
        f"visible: {summary['visible_confirmed_or_professor_check']}, "
        f"missing: {summary['missing']}, rejected_only: {summary['hidden_by_rejection']}"
    )
    print()
    print("| ID | expected | status | feedback | actual | claim |")
    print("|---|---|---|---|---|---|")
    for row in report["rows"]:
        claim = _norm(row.get("claim") or "")
        if len(claim) > 90:
            claim = claim[:89] + "…"
        print(
            "| {gold_id} | {expected_issue_type} | {status} | {matched_feedback_id} | "
            "{actual_label} | {claim} |".format(
                gold_id=row.get("gold_id") or "",
                expected_issue_type=row.get("expected_issue_type") or "",
                status=row.get("status") or "",
                matched_feedback_id=row.get("matched_feedback_id") or "",
                actual_label=row.get("actual_label") or "",
                claim=claim.replace("|", "\\|"),
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_json", type=Path)
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
    parser.add_argument("--json", action="store_true", help="print raw JSON report")
    args = parser.parse_args()

    report = evaluate(args.gold, args.result_json)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print_markdown(report)


if __name__ == "__main__":
    main()
