"""
교차 검증 (Cross-Model Verification)

1단계 claim 추출 → raw claim inventory 추출 (필요 시 반복 합의)
2단계 claim 판정 → GPT/Claude 다중 모델 후보 합집합
3단계 텍스트+문맥 교차검증 → 각 모델이 합집합 이슈를 재판정
4단계 grounding → 통과한 이슈만 primary 모델로 재검증

사용법:
    python -m analyzer.cross_pipeline <merged_clean.json>
    python -m analyzer.cross_pipeline <merged_clean.json> --models gpt-5.4 claude-sonnet-4.5
    python -m analyzer.cross_pipeline <merged_clean.json> --mode independent
"""

import argparse
import json
import math
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from . import claim_common as cv
from .cross_merge import (
    _issue_match_key,
    canonicalize_issues_with_llm,
    rebuild_claim_batches,
    union_claims,
    union_issues,
)
from .cross_utils import (
    CLAIM_EXTRACT_MODEL,
    _ROOT,
    _collect_env_vars,
    _empty_token_usage,
    _format_token_summary,
    _load_claims_jsonl,
    _merge_token_usage,
    _write_claims_jsonl,
)
from .claim_crosscheck import _CRITERIA_WEIGHTS as CROSSCHECK_CRITERIA_WEIGHTS
from .cross_workers import (
    cross_recheck_worker,
    extract_worker,
    independent_worker,
    judge_worker,
    slide_typo_worker,
    stages_3_4_worker,
)


# ── 교차 검증 메인 ────────────────────────────────────────


_CROSSCHECK_FEEDBACK_FIELDS = (
    "issue",
    "correct_info",
    "why_wrong",
    "counterexample",
    "issue_basis",
    "student_error",
    "counterexample_or_condition",
    "context_resolution",
    "evidence_in_context",
    "student_misunderstanding",
    "why_it_matters",
    "suggested_rephrase",
    "teaching_note",
    "recommendation",
)


def _claim_consensus_key(claim: dict) -> str:
    if claim.get("claim_fingerprint"):
        return str(claim.get("claim_fingerprint"))
    if claim.get("claim_id"):
        return str(claim.get("claim_id"))
    uid = str(claim.get("utterance_id", "") or "")
    text = cv._compact_text(claim.get("claim_text", ""))
    return f"{uid}::{text}"


def _merge_claim_extraction_runs(extract_results: list[dict], min_rate: float) -> list[dict]:
    if not extract_results:
        return []
    if len(extract_results) == 1:
        return union_claims(extract_results)

    threshold = max(1, math.ceil(len(extract_results) * max(0.0, min(1.0, float(min_rate)))))
    buckets: dict[str, dict] = {}
    order: list[str] = []

    for run_idx, result in enumerate(extract_results):
        seen_in_run = set()
        for item in result.get("claims_by_batch", []):
            for claim in item.get("claims", []):
                key = _claim_consensus_key(claim)
                if not key or key in seen_in_run:
                    continue
                seen_in_run.add(key)
                if key not in buckets:
                    buckets[key] = {"claim": claim, "runs": set()}
                    order.append(key)
                buckets[key]["runs"].add(run_idx)

    return [
        buckets[key]["claim"]
        for key in order
        if len(buckets[key]["runs"]) >= threshold
    ]


def _combine_extract_results(extract_results: list[dict], merged_claims: list[dict]) -> dict:
    if not extract_results:
        return {
            "model": CLAIM_EXTRACT_MODEL,
            "claims_by_batch": [],
            "api_calls": 0,
            "token_usage": _empty_token_usage(),
            "claim_run_counts": [],
        }

    first = extract_results[0]
    token_usage = _empty_token_usage()
    api_calls = 0
    run_counts = []
    for result in extract_results:
        api_calls += int(result.get("api_calls", 0) or 0)
        token_usage = _merge_token_usage(token_usage, result.get("token_usage"))
        run_counts.append(
            sum(len(item.get("claims", [])) for item in result.get("claims_by_batch", []))
        )

    return {
        "model": first.get("model", CLAIM_EXTRACT_MODEL),
        "claims_by_batch": first.get("claims_by_batch", []),
        "api_calls": api_calls,
        "token_usage": token_usage,
        "claim_run_counts": run_counts,
        "consensus_claim_count": len(merged_claims),
    }

DEFAULT_CROSSCHECK_GEMINI_MODEL = "gemini-2.5-flash"


_VERDICT_SCORE = {
    "agree": 1.0,
    "inconclusive": 0.5,
    "disagree": 0.0,
}
_DEFAULT_CONFIRM_THRESHOLD = 0.8
_DEFAULT_PROFESSOR_CHECK_THRESHOLD = 0.45


def _split_model_specs(value: str | None) -> list[str]:
    if not value:
        return []
    return [part for part in re.split(r"[\s,]+", str(value).strip()) if part]


def _default_verifier_models() -> list[str]:
    configured = _split_model_specs(os.getenv("CROSS_VERIFY_MODELS"))
    if configured:
        return configured
    legacy = _split_model_specs(os.getenv("CROSS_VERIFY_MODEL"))
    if legacy:
        return legacy
    return ["gpt-5.4", "claude-sonnet-4.5"]


def _default_crosscheck_source_models(judge_models: list[str]) -> list[str]:
    configured = _split_model_specs(os.getenv("CROSS_CHECK_MODELS"))
    return configured or judge_models


def _is_strong_feedback_model(model: str) -> bool:
    lowered = str(model or "").lower()
    return (
        lowered.startswith(("gpt", "o1", "o3", "grok", "deepseek"))
        or "claude" in lowered
        or "sonnet" in lowered
        or "opus" in lowered
    )


def _is_gemini_model(model: str) -> bool:
    return "gemini" in str(model or "").lower()


def _crosscheck_model_for(model: str) -> str:
    if not _is_gemini_model(model):
        return model
    configured = str(os.getenv("VERIFIER_CROSSCHECK_GEMINI_MODEL", "") or "").strip()
    if configured and str(model or "").strip().lower() in {"gemini", "gemini-default"}:
        return configured
    return model or configured or DEFAULT_CROSSCHECK_GEMINI_MODEL


def _env_float(name: str, default: float, *, minimum: float | None = None, maximum: float | None = None) -> float:
    try:
        value = float(os.getenv(name, str(default)) or default)
    except (TypeError, ValueError):
        value = default
    if minimum is not None:
        value = max(minimum, value)
    if maximum is not None:
        value = min(maximum, value)
    return value


def _crosscheck_confirm_threshold() -> float:
    return _env_float("CROSS_VERIFY_SCORE_CONFIRM_THRESHOLD", _DEFAULT_CONFIRM_THRESHOLD, minimum=0.0, maximum=1.0)


def _crosscheck_professor_check_threshold() -> float:
    return _env_float(
        "CROSS_VERIFY_SCORE_PROFESSOR_CHECK_THRESHOLD",
        _DEFAULT_PROFESSOR_CHECK_THRESHOLD,
        minimum=0.0,
        maximum=1.0,
    )


def _default_crosscheck_weight(model: str) -> float:
    lowered = str(model or "").lower()
    if lowered.startswith(("gpt-5.4", "gpt-5.3", "gpt-5.2", "o3")):
        return 1.0
    if lowered.startswith(("gpt", "o1")):
        return 0.9
    if "opus" in lowered:
        return 1.0
    if "claude" in lowered or "sonnet" in lowered:
        return 0.95
    if lowered.startswith("grok"):
        return 0.85
    if lowered.startswith("deepseek-v4-pro") or lowered.startswith("deepseek-reasoner"):
        return 0.75
    if lowered.startswith("deepseek-v4-flash") or lowered.startswith("deepseek-chat"):
        return 0.6
    if lowered.startswith("deepseek"):
        return 0.55
    if "haiku" in lowered:
        return 0.55
    if "gemini" in lowered and "pro" in lowered:
        return 0.8
    if "gemini" in lowered and "flash" in lowered:
        return 0.6
    if "qwen" in lowered:
        return 0.55 if "32" in lowered else 0.45
    if "llama" in lowered or "mistral" in lowered:
        return 0.45
    return 0.6


def _parse_crosscheck_weight_overrides() -> dict[str, float]:
    raw = str(os.getenv("CROSS_VERIFY_MODEL_WEIGHTS", "") or "").strip()
    if not raw:
        return {}
    overrides: dict[str, float] = {}
    for part in re.split(r"[,\n]+", raw):
        item = part.strip()
        if not item:
            continue
        if "=" in item:
            key, value = item.split("=", 1)
        elif ":" in item:
            key, value = item.rsplit(":", 1)
        else:
            continue
        key = key.strip()
        try:
            parsed = max(0.0, float(value.strip()))
        except ValueError:
            continue
        if key:
            overrides[key] = parsed
            overrides[key.lower()] = parsed
    return overrides


def _crosscheck_weight_map(models: list[str], crosscheck_model_map: dict[str, str]) -> dict[str, float]:
    overrides = _parse_crosscheck_weight_overrides()
    weights: dict[str, float] = {}
    for source_model in models:
        resolved_model = crosscheck_model_map.get(source_model, source_model)
        if not resolved_model:
            continue
        value = (
            overrides.get(source_model)
            if source_model in overrides
            else overrides.get(
                str(source_model).lower(),
                overrides.get(
                    resolved_model,
                    overrides.get(str(resolved_model).lower(), _default_crosscheck_weight(resolved_model)),
                ),
            )
        )
        weights[str(resolved_model)] = round(float(value), 4)
    return weights


def _normalize_verdict(verdict: str) -> str:
    value = str(verdict or "").lower().strip()
    return value if value in _VERDICT_SCORE else "inconclusive"


def _model_confidence(row: dict) -> float:
    value = row.get("confidence", row.get("score", row.get("issue_score")))
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        confidence = float(_VERDICT_SCORE[_normalize_verdict(row.get("verdict", "inconclusive"))])
    if confidence > 1.0 and confidence <= 100.0:
        confidence = confidence / 100.0
    return max(0.0, min(1.0, confidence))


def _score_status(score: float) -> str:
    confirm_threshold = _crosscheck_confirm_threshold()
    professor_threshold = min(_crosscheck_professor_check_threshold(), confirm_threshold)
    if score >= confirm_threshold:
        return "confirmed"
    if score >= professor_threshold:
        return "professor_check"
    return "rejected"


def _score_verdict(score: float) -> str:
    status = _score_status(score)
    if status == "confirmed":
        return "agree"
    if status == "professor_check":
        return "inconclusive"
    return "disagree"


def _bucket_from_confidence(confidence: float) -> str:
    if confidence >= _crosscheck_confirm_threshold():
        return "agree"
    if confidence >= _crosscheck_professor_check_threshold():
        return "inconclusive"
    return "disagree"


def _score_details(details: list[dict], weight_map: dict[str, float], *, excluded_model: str | None = None) -> dict:
    weighted_sum = 0.0
    total_weight = 0.0
    contributing_models: list[str] = []
    vote_counts = {"agree": 0, "inconclusive": 0, "disagree": 0}
    for row in details:
        if not isinstance(row, dict):
            continue
        model = str(row.get("model") or row.get("resolved_model") or row.get("source_model") or "").strip()
        if not model or model == excluded_model:
            continue
        if row.get("model_failed") or row.get("excluded_from_score"):
            row["excluded_from_score"] = True
            row["model_weight"] = 0.0
            row["weighted_score"] = 0.0
            row.setdefault("verdict", "inconclusive")
            row.setdefault("decision", row.get("verdict", "inconclusive"))
            continue
        vote_score = _model_confidence(row)
        verdict = _bucket_from_confidence(vote_score)
        weight = float(weight_map.get(model, _default_crosscheck_weight(model)))
        row["verdict"] = verdict
        row["decision"] = verdict
        row["confidence"] = round(vote_score, 4)
        row["vote_score"] = round(vote_score, 4)
        row["model_weight"] = round(weight, 4)
        row["weighted_score"] = round(vote_score * weight, 4)
        if weight <= 0:
            continue
        vote_counts[verdict] += 1
        contributing_models.append(model)
        weighted_sum += vote_score * weight
        total_weight += weight
    score = weighted_sum / total_weight if total_weight > 0 else 0.5
    score = max(0.0, min(1.0, score))
    return {
        "score": round(score, 4),
        "score_percent": round(score * 100, 1),
        "score_verdict": _score_verdict(score),
        "status": _score_status(score),
        "total_weight": round(total_weight, 4),
        "contributing_models": contributing_models,
        "vote_counts": vote_counts,
        "thresholds": {
            "confirmed": _crosscheck_confirm_threshold(),
            "professor_check": _crosscheck_professor_check_threshold(),
        },
    }


def _build_crosscheck_score_report(scored_issues: list[dict], weight_map: dict[str, float]) -> dict:
    model_stats: dict[str, dict] = {
        model: {
            "model": model,
            "weight": weight,
            "total": 0,
            "agree": 0,
            "disagree": 0,
            "inconclusive": 0,
            "failed": 0,
            "mean_vote_score": 0.0,
            "mean_weighted_score": 0.0,
        }
        for model, weight in weight_map.items()
    }
    issue_status_counts = {"confirmed": 0, "professor_check": 0, "rejected": 0}
    for issue in scored_issues:
        scoring = issue.get("crosscheck_scoring", {}) or {}
        status = str(scoring.get("status") or "professor_check")
        if status in issue_status_counts:
            issue_status_counts[status] += 1
        for row in issue.get("crosscheck_details", []) or []:
            if not isinstance(row, dict):
                continue
            model = str(row.get("model") or "").strip()
            if not model:
                continue
            stat = model_stats.setdefault(
                model,
                {
                    "model": model,
                    "weight": float(weight_map.get(model, _default_crosscheck_weight(model))),
                    "total": 0,
                    "agree": 0,
                    "disagree": 0,
                    "inconclusive": 0,
                    "failed": 0,
                    "mean_vote_score": 0.0,
                    "mean_weighted_score": 0.0,
                },
            )
            if row.get("model_failed") or row.get("excluded_from_score"):
                stat["failed"] += 1
                continue
            vote_score = _model_confidence(row)
            verdict = _bucket_from_confidence(vote_score)
            row["verdict"] = verdict
            row["decision"] = verdict
            row["confidence"] = round(vote_score, 4)
            stat["total"] += 1
            stat[verdict] += 1
            stat["mean_vote_score"] += vote_score
            stat["mean_weighted_score"] += float(row.get("weighted_score", 0) or 0)

    model_reports = []
    for model, stat in model_stats.items():
        total = int(stat.get("total", 0) or 0)
        failed = int(stat.get("failed", 0) or 0)
        attempted = total + failed
        if total:
            stat["agree_rate"] = round(stat["agree"] / total, 4)
            stat["disagree_rate"] = round(stat["disagree"] / total, 4)
            stat["inconclusive_rate"] = round(stat["inconclusive"] / total, 4)
            stat["mean_vote_score"] = round(stat["mean_vote_score"] / total, 4)
            stat["mean_weighted_score"] = round(stat["mean_weighted_score"] / total, 4)
        else:
            stat["agree_rate"] = 0.0
            stat["disagree_rate"] = 0.0
            stat["inconclusive_rate"] = 0.0
            stat["mean_vote_score"] = 0.0
            stat["mean_weighted_score"] = 0.0
        stat["attempted_total"] = attempted
        stat["failed_rate"] = round(failed / attempted, 4) if attempted else 0.0
        flags = []
        if attempted >= 3 and stat["failed_rate"] >= 0.5:
            flags.append("api_failure_possible")
        if total >= 5 and stat["agree_rate"] >= 0.9:
            flags.append("agree_bias_possible")
        if total >= 5 and stat["inconclusive_rate"] >= 0.75:
            flags.append("low_decisiveness_possible")
        stat["flags"] = flags
        model_reports.append(stat)

    leave_one_out = []
    for excluded_model in weight_map:
        counts = {"confirmed": 0, "professor_check": 0, "rejected": 0}
        changed = 0
        score_delta_sum = 0.0
        for issue in scored_issues:
            base_scoring = issue.get("crosscheck_scoring", {}) or {}
            base_score = float(base_scoring.get("score", 0.5) or 0.5)
            base_status = str(base_scoring.get("status") or "professor_check")
            rescored = _score_details(
                [dict(row) for row in issue.get("crosscheck_details", []) or []],
                weight_map,
                excluded_model=excluded_model,
            )
            counts[rescored["status"]] += 1
            if rescored["status"] != base_status:
                changed += 1
            score_delta_sum += abs(base_score - float(rescored["score"]))
        total_issues = len(scored_issues)
        leave_one_out.append({
            "excluded_model": excluded_model,
            "counts": counts,
            "changed_issue_count": changed,
            "changed_issue_rate": round(changed / total_issues, 4) if total_issues else 0.0,
            "mean_abs_score_delta": round(score_delta_sum / total_issues, 4) if total_issues else 0.0,
        })

    return {
        "algorithm": "weighted_criteria_confidence_average",
        "confidence_definition": "이 이슈를 교수에게 보여줄 만큼 문제가 실제로 남아 있는 정도",
        "criteria_weights": {k: round(v, 4) for k, v in CROSSCHECK_CRITERIA_WEIGHTS.items()},
        "thresholds": {
            "confirmed": _crosscheck_confirm_threshold(),
            "professor_check": _crosscheck_professor_check_threshold(),
        },
        "model_weights": weight_map,
        "status_counts": issue_status_counts,
        "model_reports": sorted(model_reports, key=lambda row: str(row.get("model", ""))),
        "leave_one_out": sorted(leave_one_out, key=lambda row: str(row.get("excluded_model", ""))),
    }


def _best_crosscheck_detail(details: list[dict], verdict: str = "agree") -> dict:
    candidates = [row for row in details if row.get("verdict") == verdict]
    if not candidates:
        return {}
    strong = [row for row in candidates if _is_strong_feedback_model(row.get("model", ""))]
    return (strong or candidates)[0]


def _ordered_crosscheck_details(details: list[dict]) -> list[dict]:
    ordered = []
    for verdict in ("agree", "inconclusive", "disagree"):
        candidates = [row for row in details if row.get("verdict") == verdict]
        strong = [row for row in candidates if _is_strong_feedback_model(row.get("model", ""))]
        weak = [row for row in candidates if row not in strong]
        ordered.extend(strong + weak)
    return ordered


def _combined_crosscheck_reasons(details: list[dict], *, include_disagree: bool = False) -> str:
    allowed = {"agree", "inconclusive"}
    if include_disagree:
        allowed.add("disagree")
    parts = []
    for row in details:
        verdict = str(row.get("verdict", "") or "")
        reason = str(row.get("reason", "") or "").strip()
        if verdict in allowed and reason:
            parts.append(f"[{row.get('model')}] {verdict}: {reason}")
    return " / ".join(parts)


def _apply_crosscheck_feedback_fields(issue: dict, details: list[dict], *, professor_check: bool = False) -> None:
    sources = _ordered_crosscheck_details(details)
    if not sources:
        return
    for field in _CROSSCHECK_FEEDBACK_FIELDS:
        for source in sources:
            value = str(source.get(field, "") or "").strip()
            if value:
                issue[field] = value
                break

    if not issue.get("why_wrong"):
        combined_reason = _combined_crosscheck_reasons(details, include_disagree=professor_check)
        if combined_reason:
            issue["why_wrong"] = combined_reason
    if not issue.get("evidence_in_context"):
        combined_evidence = " / ".join(
            f"[{row.get('model')}] {row.get('evidence_in_context')}"
            for row in details
            if row.get("verdict") in {"agree", "inconclusive"} and row.get("evidence_in_context")
        )
        if combined_evidence:
            issue["evidence_in_context"] = combined_evidence
        elif issue.get("why_wrong"):
            issue["evidence_in_context"] = issue["why_wrong"]
    if not issue.get("context_resolution") and all(row.get("verdict") == "agree" for row in details):
        issue["context_resolution"] = "해소 안 됨"

    if professor_check:
        reason_parts = [
            f"[{row.get('model')}] {row.get('verdict')}: {row.get('reason', '')}"
            for row in details
        ]
        issue["issue_basis"] = "교수 확인 필요"
        issue["context_resolution"] = "모델 간 판단 불일치"
        issue["why_wrong"] = (
            "확정 오류로 단정하지 않았습니다. "
            + " / ".join(reason_parts)
        )
        if not issue.get("issue"):
            issue["issue"] = "모델 간 판단이 갈린 교수 확인 후보입니다."
        if not issue.get("recommendation"):
            issue["recommendation"] = "교수자가 실제 의도와 강의 문맥을 확인해 표시 여부를 결정하세요."


def cross_verify(
    merged_path: str,
    models: list[str],
    num_runs: int,
    min_rate: float,
    batch_size: int,
    env_vars: dict,
    judge_batch_size: int | None = None,
    claims_jsonl: str | None = None,
    crosscheck_models: list[str] | None = None,
    claim_runs: int = 1,
    claim_min_rate: float = 0.5,
) -> dict:
    root = str(_ROOT)

    # ── Phase 1: claim 추출 (단일 모델) ──
    print(f"\n{'='*60}")
    phase1_title = f"claim 재사용 — {claims_jsonl}" if claims_jsonl else f"claim 추출 (단일) — {CLAIM_EXTRACT_MODEL}"
    print(f"  Phase 1: {phase1_title}")
    print(f"{'='*60}")

    if claims_jsonl:
        from .claim_pipeline import prepare_verification

        loaded_claims = _load_claims_jsonl(claims_jsonl)
        ctx = prepare_verification(merged_path)
        unique_utts = ctx["utterances"]
        extract_result = {
            "model": f"claims_jsonl:{claims_jsonl}",
            "claims_by_batch": [{"batch": unique_utts, "claims": loaded_claims}],
            "api_calls": 0,
            "token_usage": _empty_token_usage(),
        }
        merged_claims = union_claims([extract_result])
        extract_claim_count = len(loaded_claims)
        print(f"  기존 claim 파일 사용: {claims_jsonl}")
    else:
        extract_args = (merged_path, CLAIM_EXTRACT_MODEL, batch_size, root, env_vars)
        claim_runs = max(1, int(claim_runs or 1))
        extract_results = []

        for run_idx in range(claim_runs):
            if claim_runs > 1:
                print(f"\n  [{CLAIM_EXTRACT_MODEL}] claim 추출 반복 {run_idx + 1}/{claim_runs}")
            try:
                extract_results.append(extract_worker(extract_args))
            except Exception as e:
                print(f"  ❌ [{CLAIM_EXTRACT_MODEL}] 추출 실패: {e}")
                extract_results.append({
                    "model": CLAIM_EXTRACT_MODEL,
                    "claims_by_batch": [],
                    "api_calls": 0,
                    "token_usage": _empty_token_usage(),
                })

        merged_claims = _merge_claim_extraction_runs(extract_results, claim_min_rate)
        extract_result = _combine_extract_results(extract_results, merged_claims)
        extract_claim_count = len(merged_claims)
        if claim_runs > 1:
            print(
                "  claim 추출 합의: "
                f"runs={claim_runs}, min_rate={claim_min_rate:g}, "
                f"run_counts={extract_result.get('claim_run_counts', [])}, "
                f"kept={extract_claim_count}개"
            )

        # utterances 복원 (첫 모델의 배치에서)
        utterances = []
        for item in extract_result["claims_by_batch"]:
            utterances.extend(item["batch"])
        # dedupe utterances by id
        seen_uids = set()
        unique_utts = []
        for u in utterances:
            uid = u.get("utterance_id")
            if uid not in seen_uids:
                seen_uids.add(uid)
                unique_utts.append(u)

    print(f"\n  ── claim 입력 결과 ──")
    print(f"    [{extract_result['model']}]: {extract_claim_count}개")
    print(f"    판정 입력 claim 수: {len(merged_claims)}개")

    merged_batches = rebuild_claim_batches(merged_claims, unique_utts, judge_batch_size or batch_size)

    # ── Phase 2: claim 판정 (병렬) ──
    print(f"\n{'='*60}")
    print(f"  Phase 2: claim 판정 (병렬) — 합집합 {len(merged_claims)}개")
    print(f"{'='*60}")

    judge_args = [
        (merged_path, model, merged_batches, num_runs, min_rate, root, env_vars)
        for model in models
    ]

    judge_results = {}
    with ProcessPoolExecutor(max_workers=len(models)) as executor:
        futures = {executor.submit(judge_worker, a): a[1] for a in judge_args}
        for future in as_completed(futures):
            model = futures[future]
            try:
                judge_results[model] = future.result()
            except Exception as e:
                print(f"  ❌ [{model}] 판정 실패: {e}")
                judge_results[model] = {
                    "model": model,
                    "issues": [],
                    "api_calls": 0,
                    "token_usage": _empty_token_usage(),
                }

    # 합집합 + 공통/단독 탐지 분류
    judge_list = [{"model": m, "issues": r["issues"]} for m, r in judge_results.items()]
    raw_unioned, raw_intersected, raw_exclusive = union_issues(judge_list)
    unioned, intersected, exclusive, issue_cluster_token_usage = canonicalize_issues_with_llm(
        raw_unioned,
        models,
    )

    total_union = len(unioned)
    total_exclusive = sum(len(v) for v in exclusive.values())

    print(f"\n  ── 이슈 분류 (합집합 {total_union}건) ──")
    for model, result in judge_results.items():
        print(f"    [{model}]: {len(result['issues'])}건 탐지")
    if len(raw_unioned) != len(unioned) or len(raw_intersected) != len(intersected):
        print(
            f"    LLM issue 묶음: {len(raw_unioned)}건 → {len(unioned)}건 "
            f"(공통 {len(raw_intersected)}건 → {len(intersected)}건)"
        )

    agreement_label = "양쪽 모두 탐지" if len(models) == 2 else "모든 모델 1차 탐지"
    print(f"    {agreement_label}: {len(intersected)}건 → 공통 탐지 후보")
    for model, issues in exclusive.items():
        if issues:
            print(f"    [{model}] 단독 {len(issues)}건")
            for issue in issues:
                print(f"      • {issue.get('claim_text','')[:80]}")

    # ── Phase 3: 합집합 전체 → 각 모델이 독립 crosscheck ──
    cross_recheck_verified = []
    cross_recheck_rejected = []
    cross_recheck_inconclusive = []
    crosscheck_source_models = list(dict.fromkeys(crosscheck_models or _default_crosscheck_source_models(models)))
    cross_recheck_usage_per_model = {m: _empty_token_usage() for m in crosscheck_source_models}
    crosscheck_mode = "weighted_model_score"
    crosscheck_model_map = {model: _crosscheck_model_for(model) for model in crosscheck_source_models}
    resolved_crosscheck_models = list(dict.fromkeys(crosscheck_model_map.values()))
    crosscheck_weights = _crosscheck_weight_map(crosscheck_source_models, crosscheck_model_map)
    crosscheck_score_report = _build_crosscheck_score_report([], crosscheck_weights)
    if total_union > 0 and crosscheck_source_models:
        print(f"\n{'='*60}")
        print(f"  Phase 3: 텍스트+문맥 교차검증 — 모델별 독립 판정 + 가중 점수")
        print(f"{'='*60}")
        print("  crosscheck 가중치:")
        for model, weight in crosscheck_weights.items():
            if model in resolved_crosscheck_models:
                print(f"    [{model}] weight={weight:g}")
        if any(source != target for source, target in crosscheck_model_map.items()):
            print("  crosscheck 실행 모델:")
            for source, target in crosscheck_model_map.items():
                if source == target:
                    print(f"    [{source}]")
                else:
                    print(f"    [{source}] → [{target}]")

        cross_results = {}
        cross_args_by_source = {
            source_model: (unioned, merged_path, cross_model, root, env_vars)
            for source_model, cross_model in crosscheck_model_map.items()
        }
        with ProcessPoolExecutor(max_workers=len(crosscheck_source_models)) as executor:
            futures = {
                executor.submit(cross_recheck_worker, args): source_model
                for source_model, args in cross_args_by_source.items()
            }
            for future in as_completed(futures):
                model = futures[future]
                cross_model = crosscheck_model_map.get(model, model)
                try:
                    result_for_model = future.result()
                    cross_results[model] = result_for_model
                    cross_recheck_usage_per_model[model] = _merge_token_usage(
                        cross_recheck_usage_per_model[model],
                        result_for_model.get("token_usage"),
                    )
                except Exception as e:
                    print(f"  ❌ [{model}] 텍스트+문맥 교차검증 실패: {e}")
                    cross_results[model] = {
                        "model": cross_model,
                        "verdicts": {},
                        "token_usage": _empty_token_usage(),
                    }

        scored_issues = []
        for issue in unioned:
            cv.normalize_issue_metadata(issue)
            key = _issue_match_key(issue)
            details = []
            agree_models = []
            disagree_models = []
            inconclusive_models = []

            for model in crosscheck_source_models:
                cross_model = crosscheck_model_map.get(model, model)
                verdict_row = cross_results.get(model, {}).get("verdicts", {}).get(key, {
                    "verdict": "inconclusive",
                    "reason": "crosscheck 결과 없음",
                    "resolved_model": cross_model,
                    "model_failed": True,
                    "excluded_from_score": True,
                })
                verdict = str(verdict_row.get("verdict", "inconclusive") or "inconclusive")
                resolved_model = str(
                    verdict_row.get("resolved_model")
                    or cross_results.get(model, {}).get("model")
                    or cross_model
                )
                detail = {
                    "model": resolved_model,
                    "source_model": model,
                    "stage": "independent_crosscheck",
                    **verdict_row,
                }
                detail["model"] = resolved_model
                detail["resolved_model"] = resolved_model
                details.append(detail)

                if verdict == "agree":
                    agree_models.append(resolved_model)
                elif verdict == "disagree":
                    disagree_models.append(resolved_model)
                else:
                    inconclusive_models.append(resolved_model)

            scoring = _score_details(details, crosscheck_weights)
            score = float(scoring["score"])
            score_percent = float(scoring["score_percent"])
            weighted_status = str(scoring["status"])
            score_verdict = str(scoring["score_verdict"])

            issue["crosscheck_details"] = details
            crosscheck_context_text = next(
                (
                    str(row.get("crosscheck_context_text", "") or "")
                    for row in details
                    if row.get("crosscheck_context_text")
                ),
                "",
            )
            if crosscheck_context_text:
                issue["crosscheck_context_text"] = crosscheck_context_text
            issue["crosscheck_verdicts"] = {
                row["model"]: row.get("verdict", "inconclusive")
                for row in details
            }
            issue["crosscheck_scoring"] = scoring
            issue["crosscheck_score"] = score
            issue["crosscheck_score_percent"] = score_percent
            issue["crosscheck_score_verdict"] = score_verdict
            issue["crosscheck_weighted_status"] = weighted_status
            issue["crosscheck_model_weights"] = {
                row["model"]: row.get("model_weight", crosscheck_weights.get(row["model"], 0))
                for row in details
            }
            issue["cross_model_agreement"] = len(agree_models)
            issue["cross_recheck_model"] = ", ".join(agree_models)

            reasons = [
                f"[{row['model']}] {row.get('verdict', 'inconclusive')}: {row.get('reason', '')}"
                for row in details
            ]
            combined_reason = " / ".join(reasons)
            score_reason = (
                f"가중 점수={score:.3f}({score_percent:.1f}점), "
                f"판정={weighted_status}, "
                f"기준 confirmed>={_crosscheck_confirm_threshold():.2f}, "
                f"professor_check>={_crosscheck_professor_check_threshold():.2f}"
            )
            if combined_reason:
                combined_reason = f"{score_reason} / {combined_reason}"
            else:
                combined_reason = score_reason

            if weighted_status == "confirmed":
                if cv.should_route_issue_to_professor_check(issue):
                    _apply_crosscheck_feedback_fields(issue, details, professor_check=True)
                    issue["cross_recheck"] = None
                    issue["rejection_stage"] = "텍스트+문맥 교차검증"
                    issue["rejection_reason"] = f"{cv.metadata_review_reason(issue)} / {combined_reason}"
                    issue["professor_check_reason"] = issue["rejection_reason"]
                    cross_recheck_inconclusive.append(issue)
                else:
                    _apply_crosscheck_feedback_fields(issue, details)
                    issue["cross_recheck"] = True
                    issue["cross_recheck_reason"] = combined_reason
                    cross_recheck_verified.append(issue)
            elif weighted_status == "professor_check":
                _apply_crosscheck_feedback_fields(issue, details, professor_check=True)
                issue["cross_recheck"] = None
                issue["rejection_stage"] = "텍스트+문맥 교차검증"
                issue["rejection_reason"] = (
                    f"가중 점수가 교수 확인 구간입니다. "
                    f"동의={', '.join(agree_models)}; "
                    f"비동의={', '.join(disagree_models) or '없음'}; "
                    f"불확실={', '.join(inconclusive_models) or '없음'}"
                    f" / {combined_reason}"
                )
                issue["professor_check_reason"] = issue["rejection_reason"]
                cross_recheck_inconclusive.append(issue)
            else:
                issue["cross_recheck"] = False
                issue["rejection_stage"] = "텍스트+문맥 교차검증"
                issue["rejection_reason"] = (
                    f"가중 점수가 기각 구간입니다. "
                    f"비동의={', '.join(disagree_models) or '없음'}; "
                    f"불확실={', '.join(inconclusive_models) or '없음'}"
                    f" / {combined_reason}"
                )
                cross_recheck_rejected.append(issue)
            scored_issues.append(issue)

        crosscheck_score_report = _build_crosscheck_score_report(scored_issues, crosscheck_weights)

        if cross_recheck_verified:
            print(f"\n    ✅ 텍스트+문맥 교차검증 통과: {len(cross_recheck_verified)}건")
        if cross_recheck_rejected:
            print(f"    ❌ 텍스트+문맥 교차검증 거부: {len(cross_recheck_rejected)}건")
        if cross_recheck_inconclusive:
            print(f"    ⚠️ 텍스트+문맥 교차검증 불확실: {len(cross_recheck_inconclusive)}건")

    all_confirmed = cross_recheck_verified

    # ── Phase 4: primary 모델 grounding ──
    primary = models[0]
    if all_confirmed:
        print(f"\n{'='*60}")
        print(f"  Phase 4: grounding — [{primary}]")
        print(f"{'='*60}")

        with ProcessPoolExecutor(max_workers=1) as executor:
            future = executor.submit(stages_3_4_worker,
                                     (merged_path, primary, all_confirmed, root, env_vars))
            final = future.result()
    else:
        final = {
            "issues": [],
            "grounding_rejected": [],
            "token_usage": _empty_token_usage(),
        }

    # ── 별도: 슬라이드 오타 검사 ──
    slide_typo_result = {
        "model": primary,
        "slide_typos": [],
        "api_calls": 0,
        "failures": 0,
        "token_usage": _empty_token_usage(),
    }
    try:
        slide_typo_result = slide_typo_worker((merged_path, primary, root, env_vars))
    except Exception as e:
        print(f"  ❌ [{primary}] 슬라이드 오타 검사 실패: {e}")

    # ── 결과 조합 ──
    all_rejected = final["grounding_rejected"] + cross_recheck_rejected
    usage_model_keys = list(dict.fromkeys(models + crosscheck_source_models))
    token_usage_per_model = {m: _empty_token_usage() for m in usage_model_keys}
    extract_token_usage = extract_result.get("token_usage")
    for m in models:
        token_usage_per_model[m] = _merge_token_usage(
            token_usage_per_model[m],
            judge_results.get(m, {}).get("token_usage"),
        )
    for m in crosscheck_source_models:
        token_usage_per_model[m] = _merge_token_usage(
            token_usage_per_model.get(m),
            cross_recheck_usage_per_model.get(m),
        )
    token_usage_per_model[primary] = _merge_token_usage(
        token_usage_per_model.get(primary),
        final.get("token_usage"),
        slide_typo_result.get("token_usage"),
    )
    total_token_usage = _merge_token_usage(
        extract_token_usage,
        issue_cluster_token_usage,
        *(token_usage_per_model.values()),
    )
    result = {
        "mode": "cross_verification",
        "crosscheck_mode": crosscheck_mode,
        "crosscheck_context_mode": os.getenv("VERIFIER_CROSSCHECK_CONTEXT_MODE", "expanded") or "expanded",
        "crosscheck_focus_window": os.getenv("VERIFIER_CROSSCHECK_FOCUS_WINDOW", "5") or "5",
        "models": models,
        "crosscheck_models": resolved_crosscheck_models,
        "crosscheck_source_models": crosscheck_source_models,
        "crosscheck_model_map": crosscheck_model_map,
        "crosscheck_model_weights": crosscheck_weights,
        "crosscheck_score_report": crosscheck_score_report,
        "primary_model": primary,
        "claim_extract_model": CLAIM_EXTRACT_MODEL,
        "claim_extract_runs": extract_result.get("claim_run_counts", []),
        "claim_extract_min_rate": claim_min_rate,
        "claims_source_path": claims_jsonl or "",
        "claims_per_model": {CLAIM_EXTRACT_MODEL: extract_claim_count},
        "merged_claims_count": len(merged_claims),
        "merged_claims": merged_claims,
        "issues_per_model": {m: len(r["issues"]) for m, r in judge_results.items()},
        "issue_detection_total": sum(len(r["issues"]) for r in judge_results.values()),
        "issue_union_count": total_union,
        "issue_union_raw_count": len(raw_unioned),
        "issue_clustered_count": total_union,
        "issue_cluster_reduced_count": max(0, len(raw_unioned) - total_union),
        "exclusive_count": total_exclusive,
        "intersected_count": len(intersected),
        "intersected_raw_count": len(raw_intersected),
        "cross_recheck_verified_count": len(cross_recheck_verified),
        "cross_recheck_inconclusive_count": len(cross_recheck_inconclusive),
        "confirmed_count": len(all_confirmed),
        "issues": final["issues"],
        "slide_typos": slide_typo_result.get("slide_typos", []),
        "crosscheck_rejected_issues": cross_recheck_rejected,
        "crosscheck_inconclusive_issues": cross_recheck_inconclusive,
        "grounding_rejected_issues": final["grounding_rejected"],
        "rejected_issues": all_rejected,
        "claim_extract_token_usage": extract_token_usage,
        "issue_cluster_token_usage": issue_cluster_token_usage,
        "token_usage_per_model": token_usage_per_model,
        "token_usage": total_token_usage,
        "overall_assessment": {
            "has_issues": len(final["issues"]) > 0,
            "total_issues": len(final["issues"]),
            "severity_breakdown": _count_severity(final["issues"]),
        },
        "crosscheck_filtered": len(cross_recheck_rejected),
        "crosscheck_inconclusive_filtered": len(cross_recheck_inconclusive),
        "grounding_filtered": len(final["grounding_rejected"]),
        "grounding_failures": len(final["grounding_rejected"]),
        "slide_typo_failures": int(slide_typo_result.get("failures", 0) or 0),
    }

    # ── 로그 기록 ──
    try:
        from .verification_logger import log_cross_result
        video_name = Path(merged_path).stem.replace("_merged_clean", "").replace("_merged", "")
        log_cross_result(video_name, result, judge_results)
    except ImportError:
        pass
    except Exception as e:
        print(f"  ⚠️ 로그 기록 실패: {e}")

    return result


def _count_severity(issues):
    bd = {}
    for i in issues:
        s = i.get("severity", "minor")
        bd[s] = bd.get(s, 0) + 1
    return bd


# ── 기존 독립 실행 ──────────────────────────────────────

# ── 출력 ────────────────────────────────────────────────

def print_cross_result(result: dict):
    print(f"\n\n{'='*60}")
    print(f"  📊 교차 검증 결과")
    print(f"{'='*60}\n")

    models = result["models"]
    ipm = result.get("issues_per_model", {})
    inter = result.get("intersected_count", 0)
    xr = result.get("cross_recheck_verified_count", 0)
    xi = result.get("cross_recheck_inconclusive_count", 0)
    confirmed = result.get("confirmed_count", 0)

    print(f"  모델: {', '.join(models)}")
    if result.get("crosscheck_mode"):
        print(f"  crosscheck: {result.get('crosscheck_mode')}")
    crosscheck_models = result.get("crosscheck_models") or models
    if crosscheck_models != models:
        print(f"  crosscheck 모델: {', '.join(crosscheck_models)}")
    crosscheck_source_models = result.get("crosscheck_source_models") or []
    if crosscheck_source_models and crosscheck_source_models != crosscheck_models:
        print(f"  crosscheck source 모델: {', '.join(crosscheck_source_models)}")
    score_report = result.get("crosscheck_score_report", {}) or {}
    if score_report:
        thresholds = score_report.get("thresholds", {}) or {}
        print(
            "  crosscheck 점수: "
            f"confirmed>={float(thresholds.get('confirmed', _DEFAULT_CONFIRM_THRESHOLD) or 0):.2f}, "
            f"professor_check>={float(thresholds.get('professor_check', _DEFAULT_PROFESSOR_CHECK_THRESHOLD) or 0):.2f}"
        )
        weights = score_report.get("model_weights", {}) or {}
        if weights:
            print("  crosscheck 가중치: " + ", ".join(f"{m}={w:g}" for m, w in weights.items()))
    extract_model = result.get("claim_extract_model", CLAIM_EXTRACT_MODEL)
    extract_counts = result.get("claims_per_model", {})
    print(f"  claim 추출: [{extract_model}] {extract_counts.get(extract_model, result['merged_claims_count'])}개")
    extract_runs = result.get("claim_extract_runs") or []
    if len(extract_runs) > 1:
        print(
            "  claim 추출 반복: "
            f"runs={len(extract_runs)}, min_rate={result.get('claim_extract_min_rate', 0.5)}, "
            f"run_counts={extract_runs}"
        )
    print(f"  판정 입력 claim 수: {result['merged_claims_count']}개")
    print()
    print(f"  ── 판정 흐름 ──")
    for m, cnt in ipm.items():
        print(f"    [{m}] {cnt}건 탐지")
    detection_total = result.get("issue_detection_total", sum(ipm.values()))
    issue_union = result.get("issue_union_count", 0)
    exclusive = result.get("exclusive_count", max(issue_union - inter, 0))
    agreement_label = "양쪽 1차 탐지" if len(models) == 2 else "모든 모델 1차 탐지"
    print(f"    총 탐지 수: {detection_total}건")
    raw_union = result.get("issue_union_raw_count", issue_union)
    reduced = result.get("issue_cluster_reduced_count", 0)
    if reduced:
        print(f"    이슈 합집합: {raw_union}건 → LLM 묶음 후 {issue_union}건")
    else:
        print(f"    이슈 합집합: {issue_union}건")
    print(f"    ├─ {agreement_label}: {inter}건")
    print(f"    ├─ 단독 1차 탐지: {exclusive}건")
    print(f"    ├─ 텍스트+문맥 교차검증 통과: {xr}건")
    print(f"    ├─ 텍스트+문맥 교차검증 불확실: {xi}건")
    print(f"    └─ 확정 이슈 (grounding 진입): {confirmed}건")

    gf = result.get("grounding_failures", 0)
    if gf:
        print(f"    → grounding 기각: {gf}건")
    print()

    token_usage_per_model = result.get("token_usage_per_model", {})
    extract_usage = result.get("claim_extract_token_usage", {})
    cluster_usage = result.get("issue_cluster_token_usage", {})
    if token_usage_per_model:
        print(f"  ── 토큰 사용량 ──")
        if extract_usage:
            print(f"    [claim 추출:{extract_model}] {_format_token_summary({'total': extract_usage.get('total', {})}) if 'total' in extract_usage else _format_token_summary(extract_usage)}")
        if cluster_usage and cluster_usage.get("total", {}).get("total_tokens", 0):
            print(f"    [issue 묶음] {_format_token_summary(cluster_usage)}")
        usage_models = list(dict.fromkeys(models + list((result.get("crosscheck_source_models") or []))))
        for model in usage_models:
            usage = token_usage_per_model.get(model, {})
            print(f"    [{model}] {_format_token_summary(usage)}")
        print(f"    [전체] {_format_token_summary(result.get('token_usage', {}))}")
        print()

    score_report = result.get("crosscheck_score_report", {}) or {}
    if score_report.get("model_reports"):
        print(f"  ── crosscheck 모델 리포트 ──")
        for row in score_report.get("model_reports", []):
            flags = ", ".join(row.get("flags", []) or [])
            print(
                f"    [{row.get('model')}] weight={float(row.get('weight', 0) or 0):.2f} "
                f"agree={float(row.get('agree_rate', 0) or 0):.0%} "
                f"inconclusive={float(row.get('inconclusive_rate', 0) or 0):.0%} "
                f"disagree={float(row.get('disagree_rate', 0) or 0):.0%}"
                + (f" flags={flags}" if flags else "")
            )
        leave_one_out = score_report.get("leave_one_out", []) or []
        if leave_one_out:
            print("    제외 실험:")
            for row in leave_one_out:
                counts = row.get("counts", {}) or {}
                print(
                    f"      - {row.get('excluded_model')}: "
                    f"confirmed={counts.get('confirmed', 0)}, "
                    f"professor_check={counts.get('professor_check', 0)}, "
                    f"rejected={counts.get('rejected', 0)}, "
                    f"변경={row.get('changed_issue_count', 0)}건"
                )
        print()

    issues = result.get("issues", [])
    if issues:
        print(f"  ✅ 최종 확정 이슈: {len(issues)}건")
        for i, issue in enumerate(issues):
            src = "양쪽" if issue.get("cross_model_agreement", 0) >= 2 else f"교차검증({issue.get('cross_recheck_model','?')} 동의)"
            score = float(issue.get("crosscheck_score", issue.get("confidence", 0)) or 0)
            print(f"    [{i+1}] {issue['type']} sev={issue['severity']} score={score:.3f} ({src})")
            print(f"        {issue.get('claim_text','')[:100]}")
            print(f"        → {issue.get('issue','')[:100]}")
            confirmation_reason = str(issue.get("cross_recheck_reason", "") or issue.get("why_wrong", "") or "").strip()
            if confirmation_reason:
                print(f"        확정 사유: {confirmation_reason[:260]}")
            evidence_in_context = str(issue.get("evidence_in_context", "") or "").strip()
            if evidence_in_context:
                print(f"        문맥 근거: {evidence_in_context[:260]}")
    else:
        print(f"  최종 이슈: 0건")

    crosscheck_rejected = result.get("crosscheck_rejected_issues", [])
    crosscheck_inconclusive = result.get("crosscheck_inconclusive_issues", [])
    grounding_rejected = result.get("grounding_rejected_issues", [])

    if crosscheck_rejected:
        print(f"\n  ❌ 텍스트+문맥 교차검증 기각: {len(crosscheck_rejected)}건")
        for i, issue in enumerate(crosscheck_rejected):
            reason = issue.get("rejection_reason", "텍스트+문맥 교차검증에서 유지되지 않음")
            print(f"    [{i+1}] {issue['type']} sev={issue['severity']}")
            print(f"        claim: {issue.get('claim_text','')[:100]}")
            print(f"        issue: {issue.get('issue','')[:100]}")
            print(f"        사유: {reason[:120]}")

    if crosscheck_inconclusive:
        print(f"\n  ⚠️ 텍스트+문맥 교차검증 불확실: {len(crosscheck_inconclusive)}건")
        for i, issue in enumerate(crosscheck_inconclusive):
            reason = issue.get("rejection_reason", "텍스트+문맥 교차검증에서 확정 판단 실패")
            print(f"    [{i+1}] {issue['type']} sev={issue['severity']}")
            print(f"        claim: {issue.get('claim_text','')[:100]}")
            print(f"        issue: {issue.get('issue','')[:100]}")
            print(f"        사유: {reason[:120]}")

    if grounding_rejected:
        print(f"\n  ❌ grounding 기각: {len(grounding_rejected)}건")
        for i, issue in enumerate(grounding_rejected):
            reason = issue.get("grounding_reason", "")
            print(f"    [{i+1}] {issue['type']} sev={issue['severity']}")
            print(f"        claim: {issue.get('claim_text','')[:100]}")
            print(f"        issue: {issue.get('issue','')[:100]}")
            if reason:
                print(f"        사유: {reason[:120]}")

    slide_typos = result.get("slide_typos", [])
    if slide_typos:
        print(f"\n  ✏️ 슬라이드 오타: {len(slide_typos)}건")
        for i, typo in enumerate(slide_typos[:10], 1):
            print(
                f"    [{i}] 슬라이드 {typo.get('slide_number', '?')} | "
                f"{typo.get('problematic_text', '')} -> {typo.get('corrected_text', '')}"
            )
    if result.get("slide_typo_failures", 0):
        print(f"  ⚠️ 슬라이드 오타 검사 실패: {result['slide_typo_failures']}건")


# ── main ────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="교차 검증 / 모델 비교")
    parser.add_argument("merged_path", help="merged_clean.json 경로")
    parser.add_argument("--models", nargs="+",
                        default=None,
                        help="판정/검증에 사용할 모델 (기본: CROSS_VERIFY_MODELS)")
    parser.add_argument("--mode", choices=["cross", "independent"], default="cross",
                        help="cross=교차검증(기본), independent=독립비교")
    parser.add_argument("--num-runs", type=int, default=1, help="1차 judge 반복 횟수 (기본 1)")
    parser.add_argument("--min-rate", type=float, default=0.5)
    parser.add_argument("--claim-runs", type=int, default=1, help="claim 추출 반복 횟수 (기본 1)")
    parser.add_argument("--claim-min-rate", type=float, default=0.5, help="claim 반복 추출 시 유지할 최소 탐지 비율")
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--claims-jsonl", default=None, help="이미 추출된 claims_extracted.jsonl 경로. 지정하면 claim 추출을 건너뜀")
    parser.add_argument(
        "--crosscheck-models",
        nargs="+",
        default=None,
        help="3단계 crosscheck 전용 모델 목록. 지정하지 않으면 CROSS_CHECK_MODELS 또는 --models를 사용",
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if not Path(args.merged_path).exists():
        print(f"❌ 파일 없음: {args.merged_path}")
        sys.exit(1)

    env_vars = _collect_env_vars()
    models = args.models or _default_verifier_models()
    if len(models) < 2:
        raise RuntimeError("cross verifier는 최소 2개 모델이 필요합니다. CROSS_VERIFY_MODELS 또는 --models를 확인하세요.")

    if args.mode == "cross":
        result = cross_verify(
            args.merged_path, models, args.num_runs,
            args.min_rate, args.batch_size, env_vars,
            claims_jsonl=args.claims_jsonl,
            crosscheck_models=args.crosscheck_models,
            claim_runs=args.claim_runs,
            claim_min_rate=args.claim_min_rate,
        )
        print_cross_result(result)
    else:
        # 기존 독립 비교 모드
        worker_args = [
            (args.merged_path, m, args.num_runs, args.min_rate, str(_ROOT), env_vars)
            for m in models
        ]
        results = {}
        with ProcessPoolExecutor(max_workers=len(models)) as executor:
            futures = {executor.submit(independent_worker, a): a[1] for a in worker_args}
            for future in as_completed(futures):
                model = futures[future]
                try:
                    results[model] = future.result()
                except Exception as e:
                    print(f"  ❌ [{model}] 실패: {e}")
        result = {m: results[m] for m in models if m in results}

    if args.output:
        claims_log_path = None
        if args.mode == "cross":
            claims_log_path = _write_claims_jsonl(result.get("merged_claims", []), args.output)
        if claims_log_path:
            result["claims_log_path"] = claims_log_path
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\n💾 결과 저장: {args.output}")
        if claims_log_path:
            print(f"💾 claim 저장: {claims_log_path}")


if __name__ == "__main__":
    main()
