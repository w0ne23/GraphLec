"""merged_clean.json 입력 기준 verifier 실행기."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

from .claim_pipeline import (
    BATCH_SIZE as CLAIM_BATCH_SIZE,
    format_verification_report,
    prepare_verification,
    verify_lecture_content,
)
from .cross_utils import _collect_env_vars, _write_claims_jsonl


def _base_stem(merged_path: Path) -> str:
    return merged_path.stem.replace("_merged_clean", "").replace("_merged", "")


def _load_verifier():
    return "verifier4", verify_lecture_content, format_verification_report


def _claim_key(payload: dict) -> str:
    uid = str(payload.get("utterance_id", "") or "")
    text = str(payload.get("claim_text", "") or "")[:60]
    return f"{uid}::{text}"


def _normalize_claim_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip()).lower()


def _token_overlap(a: str, b: str) -> float:
    a_tokens = {tok for tok in _normalize_claim_text(a).split() if tok}
    b_tokens = {tok for tok in _normalize_claim_text(b).split() if tok}
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / max(1, min(len(a_tokens), len(b_tokens)))


def _classify_pedagogical_issue(issue: dict) -> dict:
    """Professor-facing category plus a conservative issue pattern hint."""
    text = " ".join(
        str(issue.get(key, "") or "")
        for key in ("claim_text", "issue", "correct_info", "explanation", "grounding_reason")
    )
    normalized = _normalize_claim_text(text)
    issue_desc = str(issue.get("issue", "") or "")

    category = "incorrect"
    label = "틀린 설명"
    rationale = "학생이 그대로 받아들이면 객관적으로 틀린 지식이 될 수 있습니다."

    ambiguity_markers = (
        "오해",
        "혼동",
        "애매",
        "동일",
        "동일시",
        "주어",
        "말실수",
        "표현",
        "들릴",
        "소지",
        "가능성",
    )
    clarification_markers = (
        "구분",
        "보충",
        "명확",
        "상주",
        "실행 상태",
        "적재 상태",
        "설명",
        "단순화",
    )
    needs_clarification = any(marker in normalized for marker in clarification_markers)
    is_ambiguous = any(marker in normalized for marker in ambiguity_markers)
    # Reserved for a future LLM/structure-based classifier. Keep null rather
    # than polluting analysis counts with low-confidence keyword guesses.
    issue_pattern = None
    issue_pattern_reason = None

    if is_ambiguous:
        category = "ambiguous"
        label = "애매한 표현"
        rationale = "표현 자체가 학생에게 다른 의미로 들리거나 개념 혼동을 만들 수 있습니다."
    if needs_clarification:
        category = "needs_clarification"
        label = "보충 설명 필요"
        rationale = "핵심 설명은 대체로 맞지만, 구분이나 보충 설명이 없으면 학생이 애매하게 받아들일 수 있습니다."
    if issue.get("type") == "outdated":
        category = "incorrect"
        label = "현행성 오류"
        rationale = "현재 기준으로 유효하지 않은 정보를 현재 사실처럼 전달할 수 있습니다."
    if "직접 모순" in issue_desc or "반대" in issue_desc:
        category = "incorrect"
        label = "틀린 설명"
        rationale = "슬라이드나 객관적 사실과 직접 충돌할 가능성이 큽니다."

    return {
        "pedagogical_type": category,
        "pedagogical_label": label,
        "pedagogical_rationale": rationale,
        "issue_category": category,
        "issue_category_label": label,
        "issue_category_reason": rationale,
        "issue_pattern": issue_pattern,
        "issue_pattern_reason": issue_pattern_reason,
    }


def _build_utterance_context(utterances: list[dict], uid: str, radius: int = 2) -> dict:
    idx = next((i for i, u in enumerate(utterances) if str(u.get("utterance_id", "") or "") == uid), -1)
    if idx < 0:
        return {"target": None, "before": [], "after": [], "window": []}

    def pack(u: dict) -> dict:
        return {
            "utterance_id": str(u.get("utterance_id", "") or ""),
            "slide_number": u.get("slide_number"),
            "start_time": u.get("start_time"),
            "end_time": u.get("end_time"),
            "text": u.get("text", ""),
        }

    before = [pack(u) for u in utterances[max(0, idx - radius):idx]]
    target = pack(utterances[idx])
    after = [pack(u) for u in utterances[idx + 1:idx + radius + 1]]
    return {"target": target, "before": before, "after": after, "window": before + [target] + after}


def _build_utterance_lookup(merged_path: Path) -> dict[str, dict]:
    ctx = prepare_verification(str(merged_path))
    utterances = ctx.get("utterances", [])
    lookup = {
        str(u.get("utterance_id", "") or ""): {
            "slide_number": u.get("slide_number"),
            "start_time": u.get("start_time"),
            "end_time": u.get("end_time"),
            "text": u.get("text", ""),
        }
        for u in utterances
    }
    for uid in list(lookup):
        lookup[uid]["utterance_context"] = _build_utterance_context(utterances, uid)
    return lookup


def _build_claim_candidates(claims: list[dict]) -> dict[str, list[dict]]:
    candidates: dict[str, list[dict]] = {}
    for claim in claims:
        uid = str(claim.get("utterance_id", "") or "")
        candidates.setdefault(uid, []).append(claim)
    return candidates


def _resolve_claim_match(issue: dict, claim_candidates: dict[str, list[dict]]) -> dict:
    uid = str(issue.get("utterance_id", "") or "")
    candidates = claim_candidates.get(uid, [])
    if not candidates:
        return {}
    if len(candidates) == 1:
        return candidates[0]

    issue_type = str(issue.get("type", "") or "")
    issue_text = issue.get("claim_text", "") or issue.get("problematic_content", "")
    narrowed = [c for c in candidates if str(c.get("claim_type", "") or "") == issue_type]
    pool = narrowed or candidates

    normalized_issue_text = _normalize_claim_text(issue_text)
    for candidate in pool:
        if _normalize_claim_text(candidate.get("claim_text", "")) == normalized_issue_text:
            return candidate

    best = None
    best_score = -1.0
    for candidate in pool:
        claim_text = candidate.get("claim_text", "")
        resolved_claim = candidate.get("resolved_claim", "")
        score = max(
            _token_overlap(issue_text, claim_text),
            _token_overlap(issue_text, resolved_claim),
        )
        if score > best_score:
            best = candidate
            best_score = score

    return best or pool[0]


def _claim_record_from_claim(claim: dict, utterance_lookup: dict[str, dict], stage: str) -> dict:
    uid = str(claim.get("utterance_id", "") or "")
    utt = utterance_lookup.get(uid, {})
    record = {
        "utterance_id": uid,
        "slide_number": utt.get("slide_number"),
        "start_time": utt.get("start_time"),
        "end_time": utt.get("end_time"),
        "claim_type": claim.get("claim_type", ""),
        "claim_text": claim.get("claim_text", ""),
        "resolved_claim": claim.get("resolved_claim", ""),
        "verification_question": claim.get("verification_question", ""),
        "is_approximate": bool(claim.get("is_approximate")),
        "source_claim_key": _claim_key(claim),
        "matched_to_extracted_claim": True,
        "stage": stage,
        "source": {
            "slide_number": utt.get("slide_number"),
            "utterance_id": uid,
            "start_time": utt.get("start_time"),
            "end_time": utt.get("end_time"),
        },
    }
    if stage == "first_stage_rejected":
        record["rejection_reason_code"] = "first_stage_not_flagged"
    return record


def _model_verdicts_from_issue(issue: dict) -> dict:
    verdicts = {}
    for row in issue.get("crosscheck_details", []) or []:
        if not isinstance(row, dict):
            continue
        model = str(row.get("model") or row.get("resolved_model") or "").strip()
        if not model:
            continue
        verdicts[model] = {
            "verdict": row.get("verdict", ""),
            "reason": row.get("reason", ""),
            "resolved_model": row.get("resolved_model", model),
        }
    return verdicts


def _grounding_from_issue(issue: dict) -> dict:
    status = str(issue.get("grounding_status") or "").strip()
    if not status:
        verified = issue.get("grounding_verified")
        if verified is True:
            status = "verified_error"
        elif verified is False:
            status = "rejected_by_evidence"
        elif issue.get("grounding_api_failed"):
            status = "grounding_unavailable"
        else:
            status = "not_applicable"
    sources = issue.get("evidence_sources", [])
    if not isinstance(sources, list):
        sources = []
    return {
        "status": status,
        "reason": issue.get("grounding_reason", ""),
        "evidence_sources": sources,
    }


def _slide_recheck_from_issue(issue: dict) -> dict:
    return {
        "status": issue.get("slide_recheck_status", "not_applicable"),
        "valid": issue.get("slide_recheck_valid"),
        "reason": issue.get("slide_recheck_reason", ""),
        "supporting_slide_status": issue.get("supporting_slide_status", ""),
        "error_origin": issue.get("error_origin", ""),
    }


def _default_rejection_reason_code(stage: str, issue: dict) -> str:
    if issue.get("rejection_reason_code"):
        return str(issue.get("rejection_reason_code"))
    if stage == "crosscheck_rejected":
        return "model_disagreement"
    if stage == "crosscheck_inconclusive":
        return "crosscheck_inconclusive"
    if stage == "slide_recheck_rejected":
        return "slide_context_rejected"
    if stage == "grounding_rejected":
        return "grounding_rejected"
    return ""


def _default_review_reason_code(stage: str, issue: dict) -> str:
    if issue.get("review_reason_code"):
        return str(issue.get("review_reason_code"))
    if stage == "needs_review":
        status = str(issue.get("grounding_status") or "").strip()
        if status:
            return status
        if issue.get("grounding_verified") is None:
            return "grounding_unavailable"
        return "needs_review"
    return ""


def _claim_record_from_issue(
    issue: dict,
    claim_lookup: dict[str, dict],
    claim_candidates: dict[str, list[dict]],
    utterance_lookup: dict[str, dict],
    stage: str,
) -> dict:
    claim = claim_lookup.get(_claim_key(issue), {})
    if not claim:
        claim = _resolve_claim_match(issue, claim_candidates)
    matched = bool(claim)
    record = _claim_record_from_claim(
        {
            "utterance_id": issue.get("utterance_id") or claim.get("utterance_id", ""),
            "claim_type": claim.get("claim_type", ""),
            "claim_text": issue.get("claim_text") or claim.get("claim_text", ""),
            "resolved_claim": claim.get("resolved_claim", ""),
            "verification_question": claim.get("verification_question", ""),
            "is_approximate": claim.get("is_approximate", False),
        },
        utterance_lookup,
        stage,
    )
    record["matched_to_extracted_claim"] = matched
    if matched:
        record["source_claim_key"] = _claim_key(claim)
    else:
        record["source_claim_key"] = _claim_key(
            {
                "utterance_id": issue.get("utterance_id", ""),
                "claim_text": issue.get("claim_text", ""),
            }
        )
    record.update(
        {
            "issue_type": issue.get("type", ""),
            "issue": issue.get("issue", ""),
            "correct_info": issue.get("correct_info", ""),
            "severity": issue.get("severity", ""),
            "confidence": issue.get("confidence", 0),
            "model_verdicts": _model_verdicts_from_issue(issue),
            "slide_recheck": _slide_recheck_from_issue(issue),
            "grounding": _grounding_from_issue(issue),
            "merged_claim_texts": issue.get("merged_claim_texts", []),
            "merged_problematic_contents": issue.get("merged_problematic_contents", []),
            "merged_issue_count": issue.get("merged_issue_count", 1),
            **_classify_pedagogical_issue(issue),
        }
    )
    rejection_reason_code = _default_rejection_reason_code(stage, issue)
    if rejection_reason_code:
        record["rejection_reason_code"] = rejection_reason_code
    review_reason_code = _default_review_reason_code(stage, issue)
    if review_reason_code:
        record["review_reason_code"] = review_reason_code
    if utt := utterance_lookup.get(record["utterance_id"], {}):
        if utt.get("utterance_context"):
            record["utterance_context"] = utt["utterance_context"]
            record["context_text"] = "\n".join(
                f"{item.get('utterance_id')} [{float(item.get('start_time') or 0):.1f}s] {item.get('text', '')}"
                for item in utt["utterance_context"].get("window", [])
            )
    rejection_reason = (
        issue.get("rejection_reason")
        or issue.get("grounding_reason")
        or issue.get("slide_recheck_reason")
        or ""
    )
    if rejection_reason:
        record["rejection_reason"] = rejection_reason
    if stage == "needs_review" and rejection_reason:
        record["review_reason"] = rejection_reason
    return record


def _record_key(record: dict) -> str:
    return _claim_key(
        {
            "utterance_id": record.get("utterance_id", ""),
            "claim_text": record.get("claim_text", ""),
        }
    )


def _dedupe_records(records: list[dict]) -> list[dict]:
    deduped: dict[str, dict] = {}
    for record in records:
        deduped[_record_key(record)] = record
    return list(deduped.values())


def _breakdown(records: list[dict], field: str, *, include_none: bool = False) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        value = record.get(field)
        if value in (None, ""):
            if not include_none:
                continue
            key = "none"
        else:
            key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _augment_decision_flow(result: dict, merged_path: Path) -> dict:
    claims = result.get("merged_claims")
    if claims is None:
        claims = result.get("extracted_claims") or []

    try:
        utterance_lookup = _build_utterance_lookup(merged_path)
    except Exception:
        utterance_lookup = {}

    claim_lookup = {_claim_key(c): c for c in claims}
    claim_candidates = _build_claim_candidates(claims)

    for issue_list_key in (
        "issues",
        "crosscheck_rejected_issues",
        "crosscheck_inconclusive_issues",
        "slide_rejected_issues",
        "grounding_rejected_issues",
        "needs_review_issues",
        "rejected_issues",
    ):
        for issue in result.get(issue_list_key, []) or []:
            issue.update(_classify_pedagogical_issue(issue))
            uid = str(issue.get("utterance_id", "") or "")
            utt = utterance_lookup.get(uid, {})
            if utt.get("utterance_context"):
                issue["utterance_context"] = utt["utterance_context"]
                issue["context_text"] = "\n".join(
                    f"{item.get('utterance_id')} [{float(item.get('start_time') or 0):.1f}s] {item.get('text', '')}"
                    for item in utt["utterance_context"].get("window", [])
                )

    raw_final_confirmed = _dedupe_records([
        _claim_record_from_issue(issue, claim_lookup, claim_candidates, utterance_lookup, "final_confirmed")
        for issue in result.get("issues", [])
    ])
    raw_crosscheck_rejected = _dedupe_records([
        _claim_record_from_issue(issue, claim_lookup, claim_candidates, utterance_lookup, "crosscheck_rejected")
        for issue in result.get("crosscheck_rejected_issues", [])
    ])
    raw_crosscheck_inconclusive = _dedupe_records([
        _claim_record_from_issue(issue, claim_lookup, claim_candidates, utterance_lookup, "crosscheck_inconclusive")
        for issue in result.get("crosscheck_inconclusive_issues", [])
    ])
    raw_slide_rejected = _dedupe_records([
        _claim_record_from_issue(issue, claim_lookup, claim_candidates, utterance_lookup, "slide_recheck_rejected")
        for issue in result.get("slide_rejected_issues", [])
    ])
    raw_grounding_rejected = _dedupe_records([
        _claim_record_from_issue(issue, claim_lookup, claim_candidates, utterance_lookup, "grounding_rejected")
        for issue in result.get("grounding_rejected_issues", [])
    ])
    raw_needs_review = _dedupe_records([
        _claim_record_from_issue(issue, claim_lookup, claim_candidates, utterance_lookup, "needs_review")
        for issue in result.get("needs_review_issues", [])
    ])
    unmatched_issue_records = [
        record
        for record in (
            raw_final_confirmed
            + raw_crosscheck_rejected
            + raw_crosscheck_inconclusive
            + raw_slide_rejected
            + raw_grounding_rejected
            + raw_needs_review
        )
        if not record.get("matched_to_extracted_claim", True)
    ]
    final_confirmed = [record for record in raw_final_confirmed if record.get("matched_to_extracted_claim", True)]
    crosscheck_rejected = [record for record in raw_crosscheck_rejected if record.get("matched_to_extracted_claim", True)]
    crosscheck_inconclusive = [record for record in raw_crosscheck_inconclusive if record.get("matched_to_extracted_claim", True)]
    slide_rejected = [record for record in raw_slide_rejected if record.get("matched_to_extracted_claim", True)]
    grounding_rejected = [record for record in raw_grounding_rejected if record.get("matched_to_extracted_claim", True)]
    needs_review = [record for record in raw_needs_review if record.get("matched_to_extracted_claim", True)]
    issue_keys = {
        record.get("source_claim_key") or _record_key(record)
        for records in (
            final_confirmed,
            crosscheck_rejected,
            crosscheck_inconclusive,
            slide_rejected,
            grounding_rejected,
            needs_review,
        )
        for record in records
    }
    first_stage_rejected = _dedupe_records([
        _claim_record_from_claim(claim, utterance_lookup, "first_stage_rejected")
        for claim in claims
        if _claim_key(claim) not in issue_keys
    ])

    final_rejected_lists = (
        crosscheck_rejected,
        crosscheck_inconclusive,
        slide_rejected,
        grounding_rejected,
        first_stage_rejected,
    )
    final_rejected_keys = {
        _record_key(record)
        for records in final_rejected_lists
        for record in records
    }
    category_breakdown = _breakdown(final_confirmed, "issue_category", include_none=True)
    pattern_breakdown = _breakdown(final_confirmed, "issue_pattern", include_none=True)
    needs_review_pattern_breakdown = _breakdown(needs_review, "issue_pattern", include_none=True)

    result["claim_decision_flow_summary"] = {
        "extracted_claim_count": len(claims),
        "final_confirmed_claim_count": len(final_confirmed),
        "final_confirmed_issue_category_breakdown": category_breakdown,
        "final_confirmed_issue_pattern_breakdown": pattern_breakdown,
        "crosscheck_rejected_claim_count": len(crosscheck_rejected),
        "crosscheck_inconclusive_claim_count": len(crosscheck_inconclusive),
        "slide_rejected_claim_count": len(slide_rejected),
        "slide_recheck_status": result.get("slide_recheck_status", ""),
        "slide_recheck_reason": result.get("slide_recheck_reason", ""),
        "slide_recheck_failure_count": int(result.get("slide_recheck_failures", 0) or 0),
        "grounding_rejected_claim_count": len(grounding_rejected),
        "needs_review_claim_count": len(needs_review),
        "needs_review_issue_pattern_breakdown": needs_review_pattern_breakdown,
        "first_stage_rejected_claim_count": len(first_stage_rejected),
        "final_rejected_claim_count": len(final_rejected_keys),
        "unmatched_issue_record_count": len(unmatched_issue_records),
    }
    result["claim_decision_overview"] = [
        {
            "label": "최종 기각된 claim 수",
            "key": "final_rejected_claim_count",
            "count": len(final_rejected_keys),
        },
        {
            "label": "crosscheck로 최종 확정된 claim",
            "key": "final_confirmed_claims",
            "count": len(final_confirmed),
        },
        {
            "label": "최종 확정 이슈 유형별 수",
            "key": "final_confirmed_issue_category_breakdown",
            "count": len(final_confirmed),
            "breakdown": category_breakdown,
        },
        {
            "label": "최종 확정 이슈 패턴별 수",
            "key": "final_confirmed_issue_pattern_breakdown",
            "count": len(final_confirmed),
            "breakdown": pattern_breakdown,
        },
        {
            "label": "crosscheck에서 기각된 claim",
            "key": "crosscheck_rejected_claims",
            "count": len(crosscheck_rejected),
        },
        {
            "label": "1차에서 기각된 claim",
            "key": "first_stage_rejected_claims",
            "count": len(first_stage_rejected),
        },
    ]
    if crosscheck_inconclusive:
        result["claim_decision_overview"].append(
            {
                "label": "crosscheck 불확실 claim",
                "key": "crosscheck_inconclusive_claims",
                "count": len(crosscheck_inconclusive),
            }
        )
    if slide_rejected:
        result["claim_decision_overview"].append(
            {
                "label": "슬라이드 재검증 기각 claim",
                "key": "slide_rejected_claims",
                "count": len(slide_rejected),
            }
        )
    if grounding_rejected:
        result["claim_decision_overview"].append(
            {
                "label": "grounding 기각 claim",
                "key": "grounding_rejected_claims",
                "count": len(grounding_rejected),
            }
        )
    if needs_review:
        result["claim_decision_overview"].append(
            {
                "label": "리뷰 필요 claim",
                "key": "needs_review_claims",
                "count": len(needs_review),
            }
        )
    if unmatched_issue_records:
        result["claim_decision_overview"].append(
            {
                "label": "추출 claim과 직접 매칭되지 않은 issue 기록",
                "key": "unmatched_issue_records",
                "count": len(unmatched_issue_records),
            }
        )
    result["claim_decision_flow"] = {
        "final_confirmed_claims": final_confirmed,
        "crosscheck_rejected_claims": crosscheck_rejected,
        "first_stage_rejected_claims": first_stage_rejected,
        "crosscheck_inconclusive_claims": crosscheck_inconclusive,
        "slide_rejected_claims": slide_rejected,
        "grounding_rejected_claims": grounding_rejected,
        "needs_review_claims": needs_review,
        "unmatched_issue_records": unmatched_issue_records,
    }
    return result


def _reorder_result_for_output(result: dict) -> dict:
    preferred_order = [
        "claim_decision_overview",
        "claim_decision_flow_summary",
        "claim_decision_flow",
        "overall_assessment",
        "mode",
        "verification_date",
        "models",
        "primary_model",
        "resume_enabled",
        "resume_cache_dir",
        "claim_extract_model",
        "claims_per_model",
        "merged_claims_count",
        "issues_per_model",
        "issue_detection_total",
        "issue_union_count",
        "exclusive_count",
        "intersected_count",
        "cross_recheck_verified_count",
        "cross_recheck_inconclusive_count",
        "confirmed_count",
        "issues",
        "crosscheck_rejected_issues",
        "crosscheck_inconclusive_issues",
        "slide_recheck_status",
        "slide_recheck_reason",
        "slide_rejected_issues",
        "grounding_rejected_issues",
        "needs_review_issues",
        "rejected_issues",
        "slide_typos",
        "slide_typo_status",
        "slide_typo_skip_reason",
        "claims_log_path",
        "merged_claims",
        "claim_extract_token_usage",
        "token_usage_per_model",
        "token_usage",
        "crosscheck_filtered",
        "crosscheck_inconclusive_filtered",
        "slide_recheck_filtered",
        "grounding_filtered",
        "needs_review_count",
        "slide_recheck_failures",
        "grounding_failures",
        "slide_typo_failures",
    ]
    reordered: dict = {}
    for key in preferred_order:
        if key in result:
            reordered[key] = result[key]
    for key, value in result.items():
        if key not in reordered:
            reordered[key] = value
    return reordered


def run_all_analyzers(
    merged_path: str,
    *,
    output_dir: str | None = None,
    claim_runs: int = 1,
    claim_min_rate: float = 0.5,
    claim_batch_size: int = CLAIM_BATCH_SIZE,
    claim_max_workers: int = 4,
    cross_models: list[str] | None = None,
    cross_runs: int = 1,
    cross_min_rate: float = 0.5,
    cross_batch_size: int = 20,
    current_date: str | None = None,
    skip_slide_typo: bool = False,
    resume: bool = False,
) -> dict:
    merged_file = Path(merged_path).resolve()
    if not merged_file.exists():
        raise FileNotFoundError(f"merged_clean 파일 없음: {merged_file}")

    base_stem = _base_stem(merged_file)
    out_dir = Path(output_dir).resolve() if output_dir else merged_file.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    result_json_path = out_dir / f"{base_stem}_content_verification.json"
    report_path = out_dir / f"{base_stem}_content_verification_report.txt"

    from .cross_pipeline import cross_verify

    primary_model = os.getenv("VERIFIER_MODEL", "gemini-2.5-flash").strip() or "gemini-2.5-flash"
    cross_model = os.getenv("CROSS_VERIFY_MODEL", "gpt-5.4").strip() or "gpt-5.4"
    models = cross_models or [primary_model, cross_model]
    if len(models) < 2:
        raise RuntimeError("cross verifier는 최소 2개 모델이 필요합니다.")
    if any(str(model).startswith(("gpt", "o1", "o3")) for model in models) and not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("cross verifier는 OPENAI_API_KEY가 필요합니다.")

    verification_result = cross_verify(
        merged_path=str(merged_file),
        models=models,
        num_runs=cross_runs,
        min_rate=cross_min_rate,
        batch_size=claim_batch_size,
        judge_batch_size=cross_batch_size,
        env_vars=_collect_env_vars(),
        current_date=current_date,
        skip_slide_typo=skip_slide_typo,
        resume=resume,
        cache_dir=str(out_dir / "_verifier_cache"),
    )

    claims_for_log = verification_result.get("merged_claims")
    if claims_for_log is None:
        claims_for_log = verification_result.get("extracted_claims")
    claims_log_path = _write_claims_jsonl(claims_for_log, result_json_path)
    if claims_log_path:
        verification_result["claims_log_path"] = claims_log_path

    verification_result = _augment_decision_flow(verification_result, merged_file)
    verification_result = _reorder_result_for_output(verification_result)

    result_json_path.write_text(
        json.dumps(verification_result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "merged_path": str(merged_file),
        "output_dir": str(out_dir),
        "claim_output": str(result_json_path),
        "claim_report": "",
        "claim_issue_count": verification_result.get("overall_assessment", {}).get("total_issues", 0),
        "used_cross": True,
    }


def main():
    parser = argparse.ArgumentParser(description="merged_clean 입력 기준 verifier 실행")
    parser.add_argument("merged_path", help="merged_clean.json 경로")
    parser.add_argument("--output-dir", default=None, help="결과 저장 디렉토리 (기본: merged 파일 폴더)")
    parser.add_argument("--claim-runs", type=int, default=1)
    parser.add_argument("--claim-min-rate", type=float, default=0.5)
    parser.add_argument("--claim-batch-size", type=int, default=CLAIM_BATCH_SIZE)
    parser.add_argument("--claim-max-workers", type=int, default=4)
    parser.add_argument(
        "--cross-models",
        nargs="+",
        default=None,
        help="cross verifier 모델 목록 (기본: VERIFIER_MODEL + CROSS_VERIFY_MODEL)",
    )
    parser.add_argument(
        "--cross-runs",
        "--judge-runs",
        dest="cross_runs",
        type=int,
        default=1,
        help="1차 judge 반복 횟수 (기본 1). crosscheck의 2는 두 모델 검증을 의미함",
    )
    parser.add_argument("--cross-min-rate", type=float, default=0.5)
    parser.add_argument("--cross-batch-size", type=int, default=20)
    parser.add_argument("--date", default=None, help="검증 기준 날짜 (YYYY-MM-DD)")
    parser.add_argument("--skip-slide-typo", action="store_true", help="슬라이드 오타 검사를 건너뛰고 claim verifier만 실행")
    parser.add_argument("--resume", action="store_true", help="output-dir의 중간 캐시를 재사용해 실패 지점부터 재개")
    args = parser.parse_args()

    result = run_all_analyzers(
        args.merged_path,
        output_dir=args.output_dir,
        claim_runs=args.claim_runs,
        claim_min_rate=args.claim_min_rate,
        claim_batch_size=args.claim_batch_size,
        claim_max_workers=args.claim_max_workers,
        cross_models=args.cross_models,
        cross_runs=args.cross_runs,
        cross_min_rate=args.cross_min_rate,
        cross_batch_size=args.cross_batch_size,
        current_date=args.date,
        skip_slide_typo=args.skip_slide_typo,
        resume=args.resume,
    )

    print("\n=== Verifier 완료 ===")
    print(f"merged    : {result['merged_path']}")
    print(f"claim     : {result['claim_output']}")
    if result["claim_report"]:
        print(f"claim txt : {result['claim_report']}")
    print(f"claim 이슈: {result['claim_issue_count']}건")


if __name__ == "__main__":
    main()
