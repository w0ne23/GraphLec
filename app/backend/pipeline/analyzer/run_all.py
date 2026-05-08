"""merged_clean.json 입력 기준 verifier 실행기."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import json
import os
import re
import shutil
import sys
from pathlib import Path

from . import claim_common as cv
from .claim_pipeline import (
    BATCH_SIZE as CLAIM_BATCH_SIZE,
    format_verification_report,
    prepare_verification,
    verify_lecture_content,
)
from .cross_utils import _collect_env_vars, _write_claims_jsonl


CLAIM_TYPE_LABELS = {
    "definition": "정의/의미 주장",
    "numeric": "수치/통계 주장",
    "causal": "인과/메커니즘 주장",
    "relationship": "관계/비교 주장",
    "currentness": "현행성 주장",
}

STATUS_CONFIRMED = "confirmed"
STATUS_PROFESSOR_CHECK = "professor_check"
STATUS_REJECTED = "rejected"
STATUS_SORT_ORDER = {
    STATUS_CONFIRMED: 0,
    STATUS_PROFESSOR_CHECK: 1,
    STATUS_REJECTED: 2,
}
ISSUE_SORT_ORDER = {
    "factual_error": 0,
    "temporal_error": 1,
    "scope_overclaim": 2,
    "confusing_explanation": 3,
}
ISSUE_BASIS_SORT_ORDER = {
    "명확한 반례 있음": 0,
    "조건/범위 누락": 1,
    "핵심 개념 동일시": 2,
    "주체/과정 혼동": 3,
    "교수 확인 필요": 4,
    "근거 부족": 9,
}
SEVERITY_SORT_ORDER = {
    "critical": 0,
    "major": 1,
    "minor": 2,
}
_DOCKER_LOG_TEE_ENABLED = False


class _DockerLogTee:
    def __init__(self, primary, docker_stream):
        self.primary = primary
        self.docker_stream = docker_stream
        self.encoding = getattr(primary, "encoding", None) or "utf-8"
        self.errors = getattr(primary, "errors", None) or "replace"

    def write(self, data):
        written = self.primary.write(data)
        self.primary.flush()
        self.docker_stream.write(data)
        self.docker_stream.flush()
        return written

    def flush(self):
        self.primary.flush()
        self.docker_stream.flush()

    def fileno(self):
        return self.primary.fileno()

    def isatty(self):
        return self.primary.isatty()

    def __getattr__(self, name):
        return getattr(self.primary, name)


def _same_stream_target(stream, target_path: str) -> bool:
    try:
        return os.fstat(stream.fileno()) == os.stat(target_path)
    except Exception:
        return False


def _enable_docker_log_tee() -> None:
    """
    Docker 백그라운드 verifier는 stdout이 파일로 리다이렉트된다.
    파일 로그는 유지하면서 Docker Desktop/backend logs에도 같은 내용을 흘려보낸다.
    """
    global _DOCKER_LOG_TEE_ENABLED
    if _DOCKER_LOG_TEE_ENABLED:
        return

    flag = os.getenv("VERIFIER_DOCKER_LOGS", "1").strip().lower()
    if flag in {"0", "false", "no", "off"}:
        return

    target_path = os.getenv("VERIFIER_DOCKER_LOG_TARGET", "/proc/1/fd/1")
    if not Path(target_path).exists():
        return
    if not (Path("/.dockerenv").exists() or Path("/pipeline").exists()):
        return
    if _same_stream_target(sys.stdout, target_path) and _same_stream_target(sys.stderr, target_path):
        return

    try:
        docker_stream = open(target_path, "a", encoding="utf-8", errors="replace", buffering=1)
    except OSError:
        return

    if not _same_stream_target(sys.stdout, target_path):
        sys.stdout = _DockerLogTee(sys.stdout, docker_stream)
    if not _same_stream_target(sys.stderr, target_path):
        sys.stderr = _DockerLogTee(sys.stderr, docker_stream)
    _DOCKER_LOG_TEE_ENABLED = True


def _include_debug_fields() -> bool:
    return os.getenv("VERIFIER_INCLUDE_DEBUG_FIELDS", "0").strip() == "1"


def _feedback_sort_key(item: dict, status: str | None = None) -> tuple:
    item_status = status or str(item.get("status", "") or "")
    issue_type = str(item.get("issue_type") or item.get("feedback_type") or item.get("type") or "")
    issue_basis = str(item.get("issue_basis", "") or "")
    score = item.get("crosscheck_score", item.get("score", item.get("confidence", 0)))
    return (
        STATUS_SORT_ORDER.get(item_status, 9),
        ISSUE_SORT_ORDER.get(issue_type, 9),
        ISSUE_BASIS_SORT_ORDER.get(issue_basis, 8),
        SEVERITY_SORT_ORDER.get(str(item.get("severity", "") or ""), 9),
        -float(score or 0),
        float(item.get("start_time", item.get("location", {}).get("start_time", 0)) or 0),
    )


def _base_stem(merged_path: Path) -> str:
    return merged_path.stem.replace("_merged_clean", "").replace("_merged", "")


def _split_model_specs(value: str | None) -> list[str]:
    if not value:
        return []
    return [part for part in re.split(r"[\s,]+", str(value).strip()) if part]


def _default_cross_models() -> list[str]:
    configured = _split_model_specs(os.getenv("CROSS_VERIFY_MODELS"))
    if configured:
        return configured

    legacy = _split_model_specs(os.getenv("CROSS_VERIFY_MODEL"))
    if legacy:
        return legacy

    return ["gpt-5.4", "claude-sonnet-4.5"]


def _default_crosscheck_models() -> list[str]:
    return _split_model_specs(os.getenv("CROSS_CHECK_MODELS"))


def _is_openai_model(model: str) -> bool:
    return str(model or "").lower().startswith(("gpt", "o1", "o3"))


def _is_xai_model(model: str) -> bool:
    return cv._is_xai_model(model)


def _is_deepseek_model(model: str) -> bool:
    return cv._is_deepseek_model(model)


def _is_anthropic_model(model: str) -> bool:
    lowered = str(model or "").lower()
    return lowered.startswith("claude") or "sonnet" in lowered or "opus" in lowered or "haiku" in lowered


def _missing_provider_key(model: str) -> str | None:
    if _is_openai_model(model) and not os.getenv("OPENAI_API_KEY"):
        return "OPENAI_API_KEY"
    if _is_xai_model(model) and not os.getenv("XAI_API_KEY"):
        return "XAI_API_KEY"
    if _is_deepseek_model(model) and not os.getenv("DEEPSEEK_API_KEY"):
        return "DEEPSEEK_API_KEY"
    if _is_anthropic_model(model) and not os.getenv("ANTHROPIC_API_KEY"):
        return "ANTHROPIC_API_KEY"
    return None


def _filter_available_crosscheck_models(models: list[str]) -> list[str]:
    available = []
    skipped = []
    for model in models:
        missing = _missing_provider_key(model)
        if missing:
            skipped.append(f"{model}({missing} 없음)")
            continue
        available.append(model)
    if skipped:
        print(f"  crosscheck 모델 제외: {', '.join(skipped)}")
    return available


def _load_verifier():
    return "verifier4", verify_lecture_content, format_verification_report


def _claims_jsonl_path(output_json_path: str | Path) -> Path:
    output_json_path = Path(output_json_path)
    stem = output_json_path.name
    if stem.endswith("_content_verification.json"):
        prefix = stem[: -len("_content_verification.json")]
    else:
        prefix = output_json_path.stem
    return output_json_path.with_name(f"{prefix}_claims_extracted.jsonl")


def _claim_history_dir(claims_path: Path) -> Path:
    return claims_path.with_name(f"{claims_path.stem}_history")


def _archive_existing_claims_log(claims_path: Path) -> str | None:
    if not claims_path.exists():
        return None
    history_dir = _claim_history_dir(claims_path)
    history_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archived = history_dir / f"{claims_path.stem}_{timestamp}.jsonl"
    shutil.copy2(claims_path, archived)
    return str(archived)


def _load_claims_jsonl(path: str | Path | None) -> list[dict]:
    if not path:
        return []
    jsonl_path = Path(path)
    if not jsonl_path.exists():
        return []

    claims = []
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            claims.append(payload)
    return claims


def _claim_raw_key(claim: dict) -> str:
    parts = [
        str(claim.get("utterance_id", "") or ""),
        str(claim.get("claim_type", "") or ""),
        _normalize_claim_text(claim.get("claim_text", "")),
        _normalize_claim_text(claim.get("resolved_claim", "")),
    ]
    return "||".join(parts)


def _compact_claim_for_diff(claim: dict) -> dict:
    return {
        "utterance_id": str(claim.get("utterance_id", "") or ""),
        "claim_type": str(claim.get("claim_type", "") or ""),
        "claim_text": str(claim.get("claim_text", "") or ""),
        "resolved_claim": str(claim.get("resolved_claim", "") or ""),
        "is_approximate": bool(claim.get("is_approximate")),
    }


def _claims_by_utterance(claims: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for claim in claims:
        uid = str(claim.get("utterance_id", "") or "")
        grouped.setdefault(uid, []).append(_compact_claim_for_diff(claim))
    for items in grouped.values():
        items.sort(key=lambda item: (
            item.get("claim_type", ""),
            item.get("claim_text", ""),
            item.get("resolved_claim", ""),
        ))
    return grouped


def _build_claims_raw_diff(previous_claims: list[dict], current_claims: list[dict]) -> dict:
    previous_keys = [_claim_raw_key(claim) for claim in previous_claims]
    current_keys = [_claim_raw_key(claim) for claim in current_claims]
    previous_counter = Counter(previous_keys)
    current_counter = Counter(current_keys)

    added_remaining = current_counter - previous_counter
    removed_remaining = previous_counter - current_counter
    added = []
    removed = []
    for claim in current_claims:
        key = _claim_raw_key(claim)
        if added_remaining[key] > 0:
            added.append(_compact_claim_for_diff(claim))
            added_remaining[key] -= 1
    for claim in previous_claims:
        key = _claim_raw_key(claim)
        if removed_remaining[key] > 0:
            removed.append(_compact_claim_for_diff(claim))
            removed_remaining[key] -= 1

    previous_by_uid = _claims_by_utterance(previous_claims)
    current_by_uid = _claims_by_utterance(current_claims)
    changed_utterances = []
    for uid in sorted(set(previous_by_uid) | set(current_by_uid)):
        if previous_by_uid.get(uid, []) == current_by_uid.get(uid, []):
            continue
        changed_utterances.append({
            "utterance_id": uid,
            "previous": previous_by_uid.get(uid, []),
            "current": current_by_uid.get(uid, []),
        })

    return {
        "schema_version": "claims_raw_diff.v1",
        "summary": {
            "previous_claim_count": len(previous_claims),
            "current_claim_count": len(current_claims),
            "common_claim_count": sum((previous_counter & current_counter).values()),
            "added_claim_count": len(added),
            "removed_claim_count": len(removed),
            "changed_utterance_count": len(changed_utterances),
        },
        "added_claims": added,
        "removed_claims": removed,
        "changed_utterances": changed_utterances,
    }


def _write_claims_raw_diff(previous_path: str | None, current_path: str | None) -> tuple[str | None, dict | None]:
    if not previous_path or not current_path:
        return None, None
    current = Path(current_path)
    previous_claims = _load_claims_jsonl(previous_path)
    current_claims = _load_claims_jsonl(current)
    diff = _build_claims_raw_diff(previous_claims, current_claims)
    diff["previous_claims_log_path"] = str(previous_path)
    diff["current_claims_log_path"] = str(current)

    if current.name.endswith("_claims_extracted.jsonl"):
        prefix = current.name[: -len("_claims_extracted.jsonl")]
    else:
        prefix = current.stem
    diff_path = current.with_name(f"{prefix}_claims_raw_diff.json")
    diff_path.write_text(json.dumps(diff, ensure_ascii=False, indent=2), encoding="utf-8")

    history_dir = _claim_history_dir(current)
    history_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    history_diff_path = history_dir / f"{prefix}_claims_raw_diff_{timestamp}.json"
    shutil.copy2(diff_path, history_diff_path)
    diff["history_diff_path"] = str(history_diff_path)
    diff_path.write_text(json.dumps(diff, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(diff_path), diff


def _claim_key(payload: dict) -> str:
    uid = str(payload.get("utterance_id", "") or "")
    text = str(payload.get("claim_text", "") or "")[:60]
    return f"{uid}::{text}"


def _claim_type_label(claim_type: str) -> str:
    return CLAIM_TYPE_LABELS.get(str(claim_type or ""), str(claim_type or "unknown"))


def _normalize_claim_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip()).lower()


def _token_overlap(a: str, b: str) -> float:
    a_tokens = {tok for tok in _normalize_claim_text(a).split() if tok}
    b_tokens = {tok for tok in _normalize_claim_text(b).split() if tok}
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / max(1, min(len(a_tokens), len(b_tokens)))


def _classify_pedagogical_issue(issue: dict) -> dict:
    """Professor-facing category derived from the verifier issue type."""
    from . import claim_common as cv

    issue_type = cv.normalize_issue_type(issue.get("type", ""))
    category_map = {
        "factual_error": (
            "incorrect",
            "발언 자체 오류",
            "학생이 그대로 받아들이면 객관적으로 틀린 지식이 될 수 있습니다.",
        ),
        "temporal_error": (
            "incorrect",
            "시대적 오류",
            "현재 기준으로 유효하지 않은 정보를 현재 사실처럼 전달할 수 있습니다.",
        ),
        "confusing_explanation": (
            "confusing",
            "혼동 가능 설명",
            "학생이 핵심 개념을 다른 개념과 혼동할 가능성이 있습니다.",
        ),
        "scope_overclaim": (
            "scope_overclaim",
            "범위 과잉 단정",
            "조건이나 범위 밖의 가능성을 닫아 과도하게 일반화할 수 있습니다.",
        ),
    }
    category, label, rationale = category_map.get(
        issue_type,
        ("professor_check", cv.issue_type_label(issue_type), "교수 확인이 필요한 설명입니다."),
    )

    feedback = issue.get("professor_feedback")
    if not isinstance(feedback, dict):
        feedback = {
            "summary": issue.get("issue", ""),
            "student_misunderstanding": issue.get("student_misunderstanding", ""),
            "why_it_matters": issue.get("why_it_matters", issue.get("explanation", "")),
            "suggested_rephrase": issue.get("suggested_rephrase", issue.get("recommendation", "")),
            "teaching_note": issue.get("teaching_note", issue.get("recommendation", "")),
            "evidence_in_context": issue.get("evidence_in_context", issue.get("problematic_content", "")),
            "why_wrong": issue.get("why_wrong", ""),
            "counterexample": issue.get("counterexample", ""),
            "issue_basis": issue.get("issue_basis", ""),
            "student_error": issue.get("student_error", ""),
            "counterexample_or_condition": issue.get("counterexample_or_condition", ""),
            "context_resolution": issue.get("context_resolution", ""),
        }

    return {
        "issue_type_label": cv.issue_type_label(issue_type),
        "pedagogical_type": category,
        "pedagogical_label": label,
        "pedagogical_rationale": rationale,
        "issue_category": category,
        "issue_category_label": label,
        "issue_category_reason": rationale,
        "professor_feedback": feedback,
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

    issue_text = issue.get("claim_text", "") or issue.get("problematic_content", "")
    claim_type = str(issue.get("claim_type", "") or "")
    pool = candidates
    if claim_type:
        narrowed = [c for c in candidates if str(c.get("claim_type", "") or "") == claim_type]
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
        "is_approximate": bool(claim.get("is_approximate")),
        "source_claim_key": _claim_key(claim),
        "matched_to_extracted_claim": True,
        "stage": stage,
    }
    if claim.get("verification_question"):
        record["verification_question"] = claim.get("verification_question", "")
    return record


def _claim_record_from_issue(
    issue: dict,
    claim_lookup: dict[str, dict],
    claim_candidates: dict[str, list[dict]],
    utterance_lookup: dict[str, dict],
    stage: str,
) -> dict:
    issue = dict(issue)
    issue_type = cv.normalize_issue_type(issue.get("type", ""))
    cv.normalize_issue_metadata(issue, issue_type)
    claim = claim_lookup.get(_claim_key(issue), {})
    if not claim:
        claim = _resolve_claim_match(issue, claim_candidates)
    matched = bool(claim)
    record = _claim_record_from_claim(
        {
            "utterance_id": issue.get("utterance_id") or claim.get("utterance_id", ""),
            "claim_type": issue.get("claim_type") or claim.get("claim_type", ""),
            "claim_text": issue.get("claim_text") or claim.get("claim_text", ""),
            "resolved_claim": issue.get("resolved_claim") or claim.get("resolved_claim", ""),
            "verification_question": issue.get("verification_question") or claim.get("verification_question", ""),
            "is_approximate": issue.get("is_approximate", claim.get("is_approximate", False)),
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
    source_claim_keys = []
    source_issues = issue.get("source_issues") if isinstance(issue.get("source_issues"), list) else []
    for source_issue in source_issues:
        source_claim = _resolve_claim_match(source_issue, claim_candidates)
        if source_claim:
            source_claim_keys.append(_claim_key(source_claim))
    if source_claim_keys:
        record["source_claim_keys"] = list(dict.fromkeys(source_claim_keys))
    record.update(
        {
            "issue_unit_id": issue.get("issue_unit_id", ""),
            "is_issue_unit": bool(issue.get("is_issue_unit")),
            "source_issues": source_issues,
            "utterance_ids": issue.get("utterance_ids", []),
            "issue_type": issue_type,
            "type": issue_type,
            "issue_type_label": issue.get("issue_type_label") or cv.issue_type_label(issue_type),
            "issue": issue.get("issue", ""),
            "correct_info": issue.get("correct_info", ""),
            "explanation": issue.get("explanation", ""),
            "why_wrong": issue.get("why_wrong", ""),
            "counterexample": issue.get("counterexample", ""),
            "issue_basis": issue.get("issue_basis", ""),
            "student_error": issue.get("student_error", ""),
            "counterexample_or_condition": issue.get("counterexample_or_condition", ""),
            "context_resolution": issue.get("context_resolution", ""),
            "recommendation": issue.get("recommendation", ""),
            "student_misunderstanding": issue.get("student_misunderstanding", ""),
            "why_it_matters": issue.get("why_it_matters", ""),
            "suggested_rephrase": issue.get("suggested_rephrase", ""),
            "teaching_note": issue.get("teaching_note", ""),
            "evidence_in_context": issue.get("evidence_in_context", ""),
            "crosscheck_context_text": issue.get("crosscheck_context_text", ""),
            "evidence_sources": issue.get("evidence_sources", []),
            "crosscheck_score": issue.get("crosscheck_score"),
            "crosscheck_score_percent": issue.get("crosscheck_score_percent"),
            "crosscheck_score_verdict": issue.get("crosscheck_score_verdict", ""),
            "crosscheck_weighted_status": issue.get("crosscheck_weighted_status", ""),
            "crosscheck_scoring": issue.get("crosscheck_scoring", {}),
            "canonical_issue_id": issue.get("canonical_issue_id", ""),
            "canonical_issue_count": issue.get("canonical_issue_count", 1),
            "canonical_member_utterance_ids": issue.get("canonical_member_utterance_ids", []),
            "canonical_wrong_proposition": issue.get("canonical_wrong_proposition", ""),
            "canonical_merge_rationale": issue.get("canonical_merge_rationale", ""),
            "verification_basis": issue.get("verification_basis", ""),
            "evidence_need": issue.get("evidence_need", ""),
            "claim_scope": issue.get("claim_scope", ""),
            "review_priority": issue.get("review_priority", ""),
            "crosscheck_details": issue.get("crosscheck_details", []),
            "grounding_verified": issue.get("grounding_verified"),
            "grounding_skipped": issue.get("grounding_skipped", False),
            "grounding_reason": issue.get("grounding_reason", ""),
            "grounding_api_failed": issue.get("grounding_api_failed", False),
            "severity": issue.get("severity", ""),
            "confidence": issue.get("confidence", 0),
            **_classify_pedagogical_issue(issue),
        }
    )
    if utt := utterance_lookup.get(record["utterance_id"], {}):
        if utt.get("utterance_context"):
            record["utterance_context"] = utt["utterance_context"]
            record["context_text"] = "\n".join(
                f"{item.get('utterance_id')} [{float(item.get('start_time') or 0):.1f}s] {item.get('text', '')}"
                for item in utt["utterance_context"].get("window", [])
            )
    rejection_reason = issue.get("rejection_reason") or ""
    if stage != "final_confirmed":
        rejection_reason = rejection_reason or issue.get("grounding_reason") or ""
    if rejection_reason:
        record["rejection_reason"] = rejection_reason
    return record


def _record_key(record: dict) -> str:
    base = _claim_key(
        {
            "utterance_id": record.get("utterance_id", ""),
            "claim_text": record.get("claim_text", ""),
        }
    )
    feedback_type = str(record.get("issue_type") or record.get("feedback_type") or "")
    return f"{base}::{feedback_type}" if feedback_type else base


def _dedupe_records(records: list[dict]) -> list[dict]:
    deduped: dict[str, dict] = {}
    for record in records:
        deduped[_record_key(record)] = record
    return list(deduped.values())


def _source_dedupe_record_payload(record: dict, record_id: str) -> dict:
    return {
        "record_id": record_id,
        "issue_type": record.get("issue_type", ""),
        "issue_type_label": record.get("issue_type_label", ""),
        "claim_text": record.get("claim_text", ""),
        "resolved_claim": record.get("resolved_claim", ""),
        "problematic_content": record.get("problematic_content") or record.get("claim_text", ""),
        "issue": record.get("issue", ""),
        "correct_info": record.get("correct_info", ""),
        "explanation": record.get("explanation", ""),
        "why_wrong": record.get("why_wrong", ""),
        "counterexample": record.get("counterexample", ""),
        "issue_basis": record.get("issue_basis", ""),
        "student_error": record.get("student_error", ""),
        "counterexample_or_condition": record.get("counterexample_or_condition", ""),
        "context_resolution": record.get("context_resolution", ""),
        "recommendation": record.get("recommendation", ""),
        "student_misunderstanding": record.get("student_misunderstanding", ""),
        "why_it_matters": record.get("why_it_matters", ""),
        "suggested_rephrase": record.get("suggested_rephrase", ""),
        "teaching_note": record.get("teaching_note", ""),
        "evidence_in_context": record.get("evidence_in_context", ""),
        "context_text": record.get("context_text", ""),
        "confidence": record.get("confidence", 0),
        "classification": {
            "verification_basis": record.get("verification_basis", ""),
            "evidence_need": record.get("evidence_need", ""),
            "claim_scope": record.get("claim_scope", ""),
            "review_priority": record.get("review_priority", ""),
        },
    }


def _select_source_claim_representative_with_llm(
    source_key: str,
    status: str,
    records: list[dict],
) -> tuple[str | None, dict, dict]:
    """Ask the model to choose one feedback record for duplicate candidates on the same source claim."""
    token_usage = cv._empty_token_usage()
    payload_records = [
        _source_dedupe_record_payload(record, f"r{idx + 1}")
        for idx, record in enumerate(records)
    ]
    prompt = f"""동일한 source claim에서 여러 피드백 후보가 나왔습니다.
최종 UI에서는 같은 source claim에 대해 대표 피드백 하나만 보여주려고 합니다.

source_claim_key: {source_key}
status_bucket: {status}

후보 목록:
{json.dumps(payload_records, ensure_ascii=False, indent=2)}

판단 기준:
- 서로 issue_type이 달라도 같은 source claim에 대한 중복 피드백이면 하나만 남기세요.
- 교수에게 가장 유용한 대표 피드백을 고르세요.
- 학생 오해가 가장 구체적이고, correct_info/suggested_rephrase가 가장 실행 가능한 후보를 우선하세요.
- 단순히 confidence가 높다는 이유만으로 고르지 마세요.
- 여러 후보가 서로 보완적이어도 최종 keep_record_id는 반드시 하나만 선택하세요.
- 새 피드백을 작성하지 말고, 기존 record_id 중 하나만 선택하세요.

응답은 JSON object 하나만 출력하세요.
{{
  "keep_record_id": "r1",
  "related_record_ids": ["r2"],
  "reason": "대표로 선택한 이유",
  "merged_note": "숨겨진 중복 후보에서 참고할 만한 보충점이 있으면 한 문장, 없으면 빈 문자열"
}}"""

    try:
        model = cv._resolve_stage_model("cross_recheck")
        response_format = {"type": "json_object"} if cv._supports_json_object_response_format(model) else None
        text, call_usage = cv._call_llm(
            prompt,
            max_tokens=1024,
            temperature=0.0,
            thinking_budget=512,
            response_format=response_format,
            stage="cross_recheck",
        )
        cv._add_call_usage(token_usage, call_usage)
        cleaned = cv._strip_json_fence(text.strip())
        obj = cv._extract_first_json_object(cleaned)
        payload = json.loads(obj or cleaned)
        keep_id = str(payload.get("keep_record_id", "") or "").strip()
        valid_ids = {item["record_id"] for item in payload_records}
        if keep_id not in valid_ids:
            return None, {
                "status": "failed",
                "reason": f"LLM returned invalid keep_record_id: {keep_id}",
            }, token_usage
        return keep_id, {
            "status": "selected",
            "reason": str(payload.get("reason", "") or ""),
            "merged_note": str(payload.get("merged_note", "") or ""),
            "related_record_ids": [
                str(rid)
                for rid in payload.get("related_record_ids", [])
                if str(rid) in valid_ids and str(rid) != keep_id
            ] if isinstance(payload.get("related_record_ids"), list) else [],
        }, token_usage
    except Exception as e:
        return None, {"status": "failed", "reason": f"{type(e).__name__}: {e}"}, token_usage


def _dedupe_records_by_source_claim_with_llm(
    records: list[dict],
    status: str,
) -> tuple[list[dict], dict]:
    """Use an LLM only for same-source-claim duplicate resolution."""
    token_usage = cv._empty_token_usage()
    grouped: dict[str, list[dict]] = {}
    passthrough: list[dict] = []
    for record in records:
        source_key = str(record.get("source_claim_key") or "")
        if not source_key:
            passthrough.append(record)
            continue
        grouped.setdefault(source_key, []).append(record)

    selected_records = list(passthrough)
    for source_key, group in grouped.items():
        if len(group) == 1:
            selected_records.append(group[0])
            continue

        keep_id, decision, usage = _select_source_claim_representative_with_llm(source_key, status, group)
        token_usage = cv._merge_token_usage(token_usage, usage)
        record_by_id = {f"r{idx + 1}": record for idx, record in enumerate(group)}
        if keep_id and keep_id in record_by_id:
            selected = record_by_id[keep_id]
            related = []
            for rid, record in record_by_id.items():
                if rid == keep_id:
                    continue
                related.append(
                    {
                        "record_id": rid,
                        "issue_type": record.get("issue_type", ""),
                        "issue": record.get("issue", ""),
                        "confidence": record.get("confidence", 0),
                    }
                )
            selected["source_claim_dedupe"] = decision
            selected["deduped_related_feedback"] = related
            selected_records.append(selected)
        else:
            for record in group:
                record["source_claim_dedupe"] = decision
                selected_records.append(record)

    return sorted(selected_records, key=lambda x: float(x.get("start_time", 0) or 0)), token_usage


def _count_by(items: list[dict], key: str, *, fallback: str = "unknown") -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        value = str(item.get(key) or fallback)
        counts[value] = counts.get(value, 0) + 1
    return counts


def _professor_check_reason(record: dict) -> str:
    if record.get("grounding_api_failed"):
        return "외부 근거 검증 응답을 파싱하지 못해 확정 대신 교수 확인 대상으로 분류했습니다."
    check_record = {**record, "type": record.get("issue_type") or record.get("type", "")}
    if record.get("grounding_verified") is None and cv.is_fact_grounded_issue(check_record):
        return "외부 근거 검증 결과가 없어 확정 대신 교수 확인 대상으로 분류했습니다."
    if cv.should_route_issue_to_professor_check(check_record):
        return cv.metadata_review_reason(check_record)
    return str(record.get("rejection_reason") or "교수 확인이 필요한 후보입니다.")


def _should_send_to_professor_check(record: dict) -> bool:
    check_record = {**record, "type": record.get("issue_type") or record.get("type", "")}
    if record.get("grounding_api_failed"):
        return True
    if record.get("grounding_verified") is None and cv.is_fact_grounded_issue(check_record):
        return True
    return cv.should_route_issue_to_professor_check(check_record)


def _split_confirmed_and_professor_check(records: list[dict]) -> tuple[list[dict], list[dict]]:
    confirmed = []
    professor_check = []
    for record in records:
        if _should_send_to_professor_check(record):
            updated = record.copy()
            updated["professor_check_reason"] = _professor_check_reason(updated)
            updated["rejection_reason"] = updated["professor_check_reason"]
            professor_check.append(updated)
        else:
            confirmed.append(record)
    return confirmed, professor_check


_GROUP_STOPWORDS = {
    "그리고",
    "그러나",
    "하지만",
    "또는",
    "혹은",
    "대한",
    "대해",
    "경우",
    "학생",
    "설명",
    "오해",
    "가능성",
    "있습니다",
    "있다",
    "학생이",
    "학생은",
    "이해",
    "위험",
    "역할",
    "핵심",
    "실제",
    "수",
    "때",
    "것",
    "등",
    "및",
    "the",
    "and",
    "or",
    "to",
    "of",
    "in",
    "for",
    "with",
    "that",
    "this",
}


def _feedback_group_text(item: dict) -> str:
    problem = item.get("problem") if isinstance(item.get("problem"), dict) else {}
    professor_feedback = (
        item.get("professor_feedback")
        if isinstance(item.get("professor_feedback"), dict)
        else {}
    )
    return " ".join(
        str(value or "")
        for value in (
            problem.get("summary"),
            problem.get("correct_info"),
            professor_feedback.get("student_misunderstanding"),
            professor_feedback.get("suggested_rephrase"),
        )
    )


def _group_tokens(text: str) -> set[str]:
    tokens = set()
    for token in re.findall(r"[A-Za-z0-9가-힣]+", str(text or "").lower()):
        if len(token) < 2 or token in _GROUP_STOPWORDS:
            continue
        tokens.add(token)
    return tokens


def _token_similarity(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / max(1, min(len(a), len(b)))


def _item_slide_number(item: dict) -> int:
    evidence = item.get("evidence") if isinstance(item.get("evidence"), dict) else {}
    try:
        return int(evidence.get("slide_number") or item.get("slide_number") or 0)
    except Exception:
        return 0


def _same_group_candidate(item: dict, group: dict, similarity: float, shared_count: int) -> bool:
    same_type = item.get("feedback_type") in set(group.get("feedback_types", []))
    item_slide = _item_slide_number(item)
    group_slides = set(group.get("slide_numbers", []))
    near_slide = bool(item_slide and any(abs(item_slide - slide) <= 1 for slide in group_slides))

    if same_type and shared_count >= 4 and similarity >= 0.48:
        return True
    if same_type and near_slide and shared_count >= 3 and similarity >= 0.40:
        return True
    if shared_count >= 5 and similarity >= 0.68:
        return True
    return False


def _dominant_value(items: list[dict], key: str, fallback: str = "") -> str:
    counts: dict[str, int] = {}
    for item in items:
        value = str(item.get(key) or fallback)
        if value:
            counts[value] = counts.get(value, 0) + 1
    if not counts:
        return fallback
    return sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))[0][0]


def _canonical_feedback_group_key(item: dict) -> str:
    evidence = item.get("evidence") if isinstance(item.get("evidence"), dict) else {}
    status = str(item.get("status", "") or "")
    canonical_issue_id = str(
        item.get("canonical_issue_id")
        or evidence.get("canonical_issue_id")
        or ""
    ).strip()
    if canonical_issue_id:
        return f"canonical:{status}:{canonical_issue_id}"

    canonical_wrong = _normalize_claim_text(evidence.get("canonical_wrong_proposition", ""))
    related_ids = evidence.get("related_utterance_ids", [])
    if canonical_wrong and isinstance(related_ids, list) and related_ids:
        related_key = ",".join(sorted(str(uid) for uid in related_ids if uid))
        return f"canonical:{status}:{related_key}:{canonical_wrong}"

    return f"single:{item.get('feedback_id', '')}"


def _build_feedback_groups(feedback_items: list[dict]) -> list[dict]:
    groups_by_key: dict[str, dict] = {}
    for item in feedback_items:
        group_key = _canonical_feedback_group_key(item)
        if group_key not in groups_by_key:
            group_id = f"fg_{len(groups_by_key) + 1:04d}"
            groups_by_key[group_key] = {
                "feedback_group_id": group_id,
                "status": item.get("status", ""),
                "feedback_item_ids": [],
                "feedback_types": [],
                "feedback_labels": [],
                "claim_types": [],
                "slide_numbers": [],
                "representative_feedback_id": item.get("feedback_id", ""),
                "_items": [],
            }
        group = groups_by_key[group_key]
        item["feedback_group_id"] = group["feedback_group_id"]
        if item.get("feedback_id"):
            group["feedback_item_ids"].append(item.get("feedback_id", ""))
        if item.get("feedback_type"):
            group["feedback_types"] = sorted(set(group["feedback_types"] + [item.get("feedback_type", "")]))
        if item.get("feedback_label"):
            group["feedback_labels"] = sorted(set(group["feedback_labels"] + [item.get("feedback_label", "")]))
        if item.get("claim_type"):
            group["claim_types"] = sorted(set(group["claim_types"] + [item.get("claim_type", "")]))
        item_slide = _item_slide_number(item)
        if item_slide:
            group["slide_numbers"] = sorted(set(group.get("slide_numbers", []) + [item_slide]))
        group["_items"].append(item)

    finalized: list[dict] = []
    for group in groups_by_key.values():
        items = group.pop("_items")
        representative = max(items, key=lambda it: float(it.get("confidence", 0) or 0))
        representative_evidence = (
            representative.get("evidence")
            if isinstance(representative.get("evidence"), dict)
            else {}
        )
        location_values = [
            (
                item.get("evidence", {}).get("slide_number")
                if isinstance(item.get("evidence"), dict)
                else None
            )
            for item in items
        ]
        slide_numbers = sorted({v for v in location_values if v is not None})
        group["representative_feedback_id"] = representative.get("feedback_id", group["representative_feedback_id"])
        group["feedback_type"] = _dominant_value(items, "feedback_type")
        group["feedback_label"] = _dominant_value(items, "feedback_label")
        group["claim_type"] = _dominant_value(items, "claim_type")
        group["claim_type_label"] = _dominant_value(items, "claim_type_label")
        group["item_count"] = len(items)
        group["slide_numbers"] = slide_numbers
        group["summary"] = representative.get("problem", {}).get("summary", "")
        group["professor_feedback"] = representative.get("professor_feedback", {})
        if representative.get("canonical_issue_id") or representative_evidence.get("canonical_issue_id"):
            group["canonical_issue_id"] = (
                representative.get("canonical_issue_id")
                or representative_evidence.get("canonical_issue_id")
                or ""
            )
        if representative_evidence.get("related_utterance_ids"):
            group["related_utterance_ids"] = representative_evidence.get("related_utterance_ids", [])
        if representative_evidence.get("canonical_wrong_proposition"):
            group["canonical_wrong_proposition"] = representative_evidence.get("canonical_wrong_proposition", "")
        if representative_evidence.get("merge_rationale"):
            group["merge_rationale"] = representative_evidence.get("merge_rationale", "")
        group["rationale"] = (
            "Phase 2 이후 LLM canonical issue 묶음 결과를 기준으로 그룹화했습니다."
            if group.get("canonical_issue_id") or group.get("canonical_wrong_proposition")
            else "LLM canonical issue가 없는 단일 피드백 항목입니다."
        )
        finalized.append(group)
    finalized.sort(
        key=lambda group: (
            STATUS_SORT_ORDER.get(str(group.get("status", "") or ""), 9),
            ISSUE_SORT_ORDER.get(str(group.get("feedback_type", "") or ""), 9),
            min(group.get("slide_numbers") or [9999]),
        )
    )
    for index, group in enumerate(finalized, start=1):
        old_group_id = group.get("feedback_group_id", "")
        new_group_id = f"fg_{index:04d}"
        group["feedback_group_id"] = new_group_id
        for item in feedback_items:
            if item.get("feedback_group_id") == old_group_id:
                item["feedback_group_id"] = new_group_id
    return finalized


def _claim_payload_v2(claim: dict, utterance_lookup: dict[str, dict]) -> dict:
    uid = str(claim.get("utterance_id", "") or "")
    claim_type = str(claim.get("claim_type", "") or "")
    utt = utterance_lookup.get(uid, {})
    payload = {
        "claim_id": _claim_key(claim),
        "utterance_id": uid,
        "utterance_ids": claim.get("utterance_ids") or [uid],
        "claim_type": claim_type,
        "claim_type_label": _claim_type_label(claim_type),
        "claim_text": claim.get("claim_text", ""),
        "resolved_claim": claim.get("resolved_claim", ""),
        "is_approximate": bool(claim.get("is_approximate")),
        "needs_context": bool(claim.get("needs_context")),
        "resolution_status": claim.get("resolution_status", ""),
        "context_note": claim.get("context_note", ""),
        "location": {
            "start_time": utt.get("start_time"),
            "end_time": utt.get("end_time"),
            "slide_number": utt.get("slide_number"),
        },
    }
    if claim.get("verification_question"):
        payload["verification_question"] = claim.get("verification_question", "")
    return payload


def _crosscheck_verdict(details: list[dict], status: str) -> str:
    scoring = {}
    if isinstance(details, dict):
        scoring = details.get("scoring") or {}
        details = details.get("model_results") or []
    if isinstance(scoring, dict) and scoring.get("score_verdict"):
        return str(scoring.get("score_verdict") or "")
    verdicts = [str(row.get("verdict", "") or "") for row in details if isinstance(row, dict)]
    if not verdicts:
        if status == STATUS_CONFIRMED:
            return "agree"
        if status == STATUS_PROFESSOR_CHECK:
            return "inconclusive"
        if status == STATUS_REJECTED:
            return "disagree"
        return "not_run"
    if all(v == "agree" for v in verdicts):
        return "agree"
    if any(v == "inconclusive" for v in verdicts):
        return "inconclusive"
    if all(v == "disagree" for v in verdicts):
        return "disagree"
    return "mixed"


def _grounding_payload(record: dict) -> dict:
    if record.get("grounding_skipped"):
        return {
            "status": "skipped",
            "reason": record.get("grounding_reason", ""),
        }
    if record.get("grounding_verified") is True:
        return {
            "status": "passed",
            "reason": record.get("grounding_reason", ""),
            "evidence_sources": record.get("evidence_sources", []),
        }
    if record.get("grounding_verified") is False:
        return {
            "status": "rejected",
            "reason": record.get("grounding_reason", ""),
            "evidence_sources": record.get("evidence_sources", []),
        }
    if record.get("grounding_api_failed"):
        return {
            "status": "failed_kept",
            "reason": record.get("grounding_reason", ""),
        }
    return {
        "status": "not_run",
        "reason": record.get("grounding_reason", ""),
    }


def _compact_crosscheck_details(details: list[dict]) -> list[dict]:
    visible_fields = (
        "issue",
        "correct_info",
        "why_wrong",
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
    compact = []
    for row in details:
        if not isinstance(row, dict):
            continue
        item = {
            "model": row.get("model", ""),
            "verdict": row.get("verdict", ""),
            "reason": row.get("reason", ""),
        }
        for numeric_field in ("vote_score", "model_weight", "weighted_score", "confidence"):
            if row.get(numeric_field) is not None:
                item[numeric_field] = row.get(numeric_field)
        for scoring_field in ("criteria_scores", "criteria_evidence", "score_breakdown"):
            if isinstance(row.get(scoring_field), dict):
                item[scoring_field] = row.get(scoring_field)
        for field in visible_fields:
            value = str(row.get(field, "") or "").strip()
            if value:
                item[field] = value
        compact.append(item)
    return compact


def _confirmation_reason(record: dict, crosscheck_details: list[dict]) -> str:
    existing = str(record.get("cross_recheck_reason", "") or "").strip()
    if existing:
        return existing

    parts = []
    for row in crosscheck_details:
        if not isinstance(row, dict):
            continue
        if row.get("verdict") != "agree":
            continue
        reason = str(row.get("reason", "") or "").strip()
        if reason:
            parts.append(f"[{row.get('model', '')}] agree: {reason}")
    return " / ".join(parts)


def _ordered_unique_text(values: list[str]) -> list[str]:
    return [value for value in dict.fromkeys(str(v or "").strip() for v in values) if value]


def _sort_utterance_ids(values: list[str]) -> list[str]:
    unique = _ordered_unique_text(values)
    return sorted(
        unique,
        key=lambda value: int(value[1:]) if re.fullmatch(r"U\d+", value) else 10**12,
    )


def _extract_utterance_ids_from_text(text: str) -> list[str]:
    if not text:
        return []
    return _ordered_unique_text(re.findall(r"\bU\d{4,}\b", str(text)))


def _record_related_utterance_ids(record: dict, crosscheck_details: list[dict]) -> list[str]:
    ids: list[str] = []
    for key in ("utterance_ids", "canonical_member_utterance_ids", "related_utterance_ids"):
        value = record.get(key)
        if isinstance(value, list):
            ids.extend(str(uid or "").strip() for uid in value)
    if record.get("utterance_id"):
        ids.append(str(record.get("utterance_id", "") or "").strip())

    for source_issue in record.get("source_issues", []) or []:
        if isinstance(source_issue, dict) and source_issue.get("utterance_id"):
            ids.append(str(source_issue.get("utterance_id", "") or "").strip())

    text_fields = (
        "claim_text",
        "problematic_content",
        "issue",
        "why_wrong",
        "evidence_in_context",
        "cross_recheck_reason",
        "canonical_merge_rationale",
        "student_error",
    )
    for field in text_fields:
        ids.extend(_extract_utterance_ids_from_text(str(record.get(field, "") or "")))

    detail_fields = (
        "reason",
        "issue",
        "why_wrong",
        "evidence_in_context",
        "teaching_note",
        "recommendation",
    )
    for detail in crosscheck_details:
        if not isinstance(detail, dict):
            continue
        for field in detail_fields:
            ids.extend(_extract_utterance_ids_from_text(str(detail.get(field, "") or "")))

    return _sort_utterance_ids(ids)


def _feedback_payload_v2(record: dict, status: str, index: int) -> dict:
    feedback = record.get("professor_feedback") if isinstance(record.get("professor_feedback"), dict) else {}
    include_debug = _include_debug_fields()
    feedback_type = str(record.get("issue_type") or record.get("type") or "")
    feedback_label = (
        record.get("issue_type_label")
        or record.get("issue_category_label")
        or record.get("pedagogical_label")
        or feedback_type
    )
    source_claim_id = record.get("source_claim_key") if record.get("matched_to_extracted_claim", True) else None
    crosscheck_details = record.get("crosscheck_details") or []
    crosscheck_scoring = record.get("crosscheck_scoring") or {}
    crosscheck_score = record.get("crosscheck_score")
    if crosscheck_score is None and isinstance(crosscheck_scoring, dict):
        crosscheck_score = crosscheck_scoring.get("score")
    confirmation_reason = _confirmation_reason(record, crosscheck_details)
    related_utterance_ids = _record_related_utterance_ids(record, crosscheck_details)
    feedback_id = f"fb_{index + 1:04d}"
    payload = {
        "feedback_id": feedback_id,
        "source_claim_id": source_claim_id,
        "utterance_id": record.get("utterance_id", ""),
        "utterance_ids": related_utterance_ids,
        "related_utterance_ids": related_utterance_ids,
        "claim_text": record.get("claim_text", ""),
        "resolved_claim": record.get("resolved_claim", ""),
        "claim_type": record.get("claim_type", ""),
        "feedback_type": feedback_type,
        "feedback_label": feedback_label,
        "severity": record.get("severity", ""),
        "status": status,
        "score": crosscheck_score,
        "crosscheck_score": crosscheck_score,
        "crosscheck_score_percent": record.get("crosscheck_score_percent"),
        "crosscheck_score_verdict": record.get("crosscheck_score_verdict"),
        "crosscheck_weighted_status": record.get("crosscheck_weighted_status"),
        "location": {
            "start_time": record.get("start_time"),
            "end_time": record.get("end_time"),
            "slide_number": record.get("slide_number"),
        },
        "problem": {
            "problematic_content": record.get("problematic_content") or record.get("claim_text", ""),
            "summary": record.get("issue", "") or feedback.get("summary", ""),
            "correct_info": record.get("correct_info", ""),
            "why_wrong": record.get("why_wrong", ""),
            "issue_basis": record.get("issue_basis", ""),
            "student_error": record.get("student_error", ""),
            "counterexample_or_condition": record.get("counterexample_or_condition", ""),
            "context_resolution": record.get("context_resolution", ""),
            "recommendation": record.get("recommendation", ""),
        },
        "professor_feedback": {
            "student_misunderstanding": (
                record.get("student_misunderstanding")
                or feedback.get("student_misunderstanding", "")
            ),
            "why_it_matters": record.get("why_it_matters") or feedback.get("why_it_matters", ""),
            "suggested_rephrase": (
                record.get("suggested_rephrase")
                or feedback.get("suggested_rephrase", "")
            ),
            "teaching_note": record.get("teaching_note") or feedback.get("teaching_note", ""),
        },
        "evidence": {
            "context_text": record.get("context_text", ""),
            "crosscheck_context_text": record.get("crosscheck_context_text", ""),
            "slide_number": record.get("slide_number"),
            "evidence_in_context": (
                record.get("evidence_in_context")
                or feedback.get("evidence_in_context", "")
            ),
            "evidence_sources": record.get("evidence_sources", []),
            "related_utterance_ids": related_utterance_ids,
        },
    }
    if record.get("issue_unit_id"):
        payload["issue_unit_id"] = record.get("issue_unit_id", "")
    if status == STATUS_CONFIRMED and confirmation_reason:
        payload["confirmation_reason"] = confirmation_reason
        payload["evidence"]["confirmation_reason"] = confirmation_reason
    if record.get("source_claim_keys"):
        payload["source_claim_ids"] = record.get("source_claim_keys", [])
    if record.get("source_issues"):
        payload["evidence"]["source_issues"] = record.get("source_issues", [])
    if crosscheck_details:
        payload["checks"] = {
            "crosscheck": {
                "verdict": record.get("crosscheck_score_verdict") or _crosscheck_verdict(crosscheck_details, status),
                "score": crosscheck_score,
                "score_percent": record.get("crosscheck_score_percent"),
                "status_by_score": record.get("crosscheck_weighted_status"),
                "scoring": crosscheck_scoring,
                "model_results": _compact_crosscheck_details(crosscheck_details),
            },
            "grounding": _grounding_payload(record),
        }
    if int(record.get("canonical_issue_count", 1) or 1) > 1:
        payload["canonical_issue_id"] = record.get("canonical_issue_id", "")
        payload["evidence"]["canonical_issue_id"] = record.get("canonical_issue_id", "")
        payload["evidence"]["canonical_wrong_proposition"] = record.get("canonical_wrong_proposition", "")
        payload["evidence"]["merge_rationale"] = record.get("canonical_merge_rationale", "")
    if include_debug:
        payload["confidence"] = record.get("confidence", 0)
        payload["claim_type_label"] = _claim_type_label(record.get("claim_type", ""))
        payload["problem"]["explanation"] = record.get("explanation", "")
        payload["problem"]["counterexample"] = record.get("counterexample", "")
        payload["classification"] = {
            "verification_basis": record.get("verification_basis", ""),
            "evidence_need": record.get("evidence_need", ""),
            "claim_scope": record.get("claim_scope", ""),
            "review_priority": record.get("review_priority", ""),
        }
        payload["checks"] = {
            "crosscheck": {
                "verdict": record.get("crosscheck_score_verdict") or _crosscheck_verdict(crosscheck_details, status),
                "score": crosscheck_score,
                "score_percent": record.get("crosscheck_score_percent"),
                "status_by_score": record.get("crosscheck_weighted_status"),
                "scoring": crosscheck_scoring,
                "model_results": crosscheck_details,
            },
            "grounding": _grounding_payload(record),
        }
    if record.get("rejection_reason"):
        if status == STATUS_PROFESSOR_CHECK:
            payload["professor_check_reason"] = record["rejection_reason"]
        elif status == STATUS_REJECTED:
            payload["rejection_reason"] = record["rejection_reason"]
    if record.get("professor_check_reason"):
        payload["professor_check_reason"] = record["professor_check_reason"]
    if include_debug and record.get("verification_question"):
        payload["verification_question"] = record["verification_question"]
    if include_debug and record.get("deduped_related_feedback"):
        payload["deduped_related_feedback"] = record["deduped_related_feedback"]
    if include_debug and record.get("source_claim_dedupe"):
        payload["source_claim_dedupe"] = record["source_claim_dedupe"]
    return payload


def _build_v2_output(
    result: dict,
    claims: list[dict],
    utterance_lookup: dict[str, dict],
    confirmed: list[dict],
    professor_check: list[dict],
    rejected: list[dict],
) -> dict:
    raw_models = result.get("pipeline_models") or result.get("models", []) or []
    if isinstance(raw_models, dict):
        raw_models = raw_models.get("judge") or raw_models.get("crosscheck") or []
    legacy_models = list(raw_models)
    crosscheck_models = list(result.get("crosscheck_models") or legacy_models)
    crosscheck_source_models = list(result.get("crosscheck_source_models") or crosscheck_models)
    feedback_items: list[dict] = []
    for status, records in (
        (STATUS_CONFIRMED, confirmed),
        (STATUS_PROFESSOR_CHECK, professor_check),
        (STATUS_REJECTED, rejected),
    ):
        for record in sorted(records, key=lambda item: _feedback_sort_key(item, status)):
            feedback_items.append(_feedback_payload_v2(record, status, len(feedback_items)))
    feedback_items.sort(key=_feedback_sort_key)
    for index, item in enumerate(feedback_items, start=1):
        item["feedback_id"] = f"fb_{index:04d}"

    confirmed_ids = [item["feedback_id"] for item in feedback_items if item["status"] == STATUS_CONFIRMED]
    professor_check_ids = [
        item["feedback_id"] for item in feedback_items if item["status"] == STATUS_PROFESSOR_CHECK
    ]
    rejected_ids = [item["feedback_id"] for item in feedback_items if item["status"] == STATUS_REJECTED]
    confirmed_feedbacks = [item for item in feedback_items if item["status"] == STATUS_CONFIRMED]
    feedback_groups = _build_feedback_groups(feedback_items)
    confirmed_group_ids = [
        group["feedback_group_id"] for group in feedback_groups if group["status"] == STATUS_CONFIRMED
    ]
    professor_check_group_ids = [
        group["feedback_group_id"] for group in feedback_groups if group["status"] == STATUS_PROFESSOR_CHECK
    ]
    rejected_group_ids = [
        group["feedback_group_id"] for group in feedback_groups if group["status"] == STATUS_REJECTED
    ]
    confirmed_groups = [group for group in feedback_groups if group["status"] == STATUS_CONFIRMED]

    breakdown_by_feedback_type: dict[str, int] = {}
    for item in confirmed_feedbacks:
        label = str(item.get("feedback_label") or item.get("feedback_type") or "검토 필요")
        breakdown_by_feedback_type[label] = breakdown_by_feedback_type.get(label, 0) + 1
    breakdown_by_feedback_group_type: dict[str, int] = {}
    for group in confirmed_groups:
        label = str(group.get("feedback_label") or group.get("feedback_type") or "검토 필요")
        breakdown_by_feedback_group_type[label] = breakdown_by_feedback_group_type.get(label, 0) + 1
    breakdown_by_verification_basis: dict[str, int] = {}
    breakdown_by_evidence_need: dict[str, int] = {}
    for item in feedback_items:
        classification = item.get("classification") if isinstance(item.get("classification"), dict) else {}
        basis = str(classification.get("verification_basis") or "unclassified")
        evidence_need = str(classification.get("evidence_need") or "unclassified")
        breakdown_by_verification_basis[basis] = breakdown_by_verification_basis.get(basis, 0) + 1
        breakdown_by_evidence_need[evidence_need] = breakdown_by_evidence_need.get(evidence_need, 0) + 1

    return {
        "schema_version": "content_verification.v2",
        "mode": result.get("mode", "cross_verification"),
        "models": {
            "claim_extract": result.get("claim_extract_model", ""),
            "judge": legacy_models,
            "crosscheck": crosscheck_models,
            "grounding": result.get("primary_model", ""),
        },
        "pipeline_models": legacy_models,
        "crosscheck_models": crosscheck_models,
        "crosscheck_source_models": crosscheck_source_models,
        "crosscheck_model_map": result.get("crosscheck_model_map", {}),
        "crosscheck_model_weights": result.get("crosscheck_model_weights", {}),
        "crosscheck_score_report": result.get("crosscheck_score_report", {}),
        "summary": {
            "extracted_claim_count": len(claims),
            "issue_union_raw_count": result.get("issue_union_raw_count", result.get("issue_union_count", 0)),
            "issue_clustered_count": result.get("issue_clustered_count", result.get("issue_union_count", 0)),
            "issue_cluster_reduced_count": result.get("issue_cluster_reduced_count", 0),
            "feedback_candidate_count": len(feedback_items),
            "confirmed_feedback_count": len(confirmed_ids),
            "professor_check_feedback_count": len(professor_check_ids),
            "rejected_feedback_count": len(rejected_ids),
            "feedback_group_count": len(feedback_groups),
            "confirmed_feedback_group_count": len(confirmed_group_ids),
            "professor_check_feedback_group_count": len(professor_check_group_ids),
            "rejected_feedback_group_count": len(rejected_group_ids),
            "breakdown_by_feedback_type": breakdown_by_feedback_type,
            "breakdown_by_feedback_group_type": breakdown_by_feedback_group_type,
            "breakdown_by_verification_basis": breakdown_by_verification_basis,
            "breakdown_by_evidence_need": breakdown_by_evidence_need,
            "breakdown_by_claim_type": _count_by(claims, "claim_type"),
            "crosscheck_score": {
                "algorithm": (result.get("crosscheck_score_report", {}) or {}).get("algorithm", ""),
                "thresholds": (result.get("crosscheck_score_report", {}) or {}).get("thresholds", {}),
                "status_counts": (result.get("crosscheck_score_report", {}) or {}).get("status_counts", {}),
            },
        },
        "claims": [_claim_payload_v2(claim, utterance_lookup) for claim in claims],
        "feedback_groups": feedback_groups,
        "feedback_items": feedback_items,
        "views": {
            "confirmed_feedback_group_ids": confirmed_group_ids,
            "professor_check_feedback_group_ids": professor_check_group_ids,
            "rejected_feedback_group_ids": rejected_group_ids,
            "confirmed_feedback_ids": confirmed_ids,
            "professor_check_feedback_ids": professor_check_ids,
            "rejected_feedback_ids": rejected_ids,
        },
    }


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
        "grounding_rejected_issues",
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
    raw_grounding_rejected = _dedupe_records([
        _claim_record_from_issue(issue, claim_lookup, claim_candidates, utterance_lookup, "grounding_rejected")
        for issue in result.get("grounding_rejected_issues", [])
    ])
    unmatched_issue_records = [
        record
        for record in (
            raw_final_confirmed
            + raw_crosscheck_rejected
            + raw_crosscheck_inconclusive
            + raw_grounding_rejected
        )
        if not record.get("matched_to_extracted_claim", True)
    ]
    final_confirmed = [record for record in raw_final_confirmed if record.get("matched_to_extracted_claim", True)]
    crosscheck_rejected = [record for record in raw_crosscheck_rejected if record.get("matched_to_extracted_claim", True)]
    crosscheck_inconclusive = [record for record in raw_crosscheck_inconclusive if record.get("matched_to_extracted_claim", True)]
    grounding_rejected = [record for record in raw_grounding_rejected if record.get("matched_to_extracted_claim", True)]
    final_confirmed, professor_check = _split_confirmed_and_professor_check(final_confirmed)
    for record in crosscheck_inconclusive:
        record["professor_check_reason"] = record.get("rejection_reason") or "crosscheck 결과가 불확실합니다."
    professor_check.extend(crosscheck_inconclusive)

    source_claim_dedupe_token_usage = cv._empty_token_usage()
    final_confirmed, usage = _dedupe_records_by_source_claim_with_llm(final_confirmed, STATUS_CONFIRMED)
    source_claim_dedupe_token_usage = cv._merge_token_usage(source_claim_dedupe_token_usage, usage)
    professor_check, usage = _dedupe_records_by_source_claim_with_llm(
        professor_check,
        STATUS_PROFESSOR_CHECK,
    )
    source_claim_dedupe_token_usage = cv._merge_token_usage(source_claim_dedupe_token_usage, usage)
    crosscheck_rejected, usage = _dedupe_records_by_source_claim_with_llm(crosscheck_rejected, STATUS_REJECTED)
    source_claim_dedupe_token_usage = cv._merge_token_usage(source_claim_dedupe_token_usage, usage)
    grounding_rejected, usage = _dedupe_records_by_source_claim_with_llm(grounding_rejected, STATUS_REJECTED)
    source_claim_dedupe_token_usage = cv._merge_token_usage(source_claim_dedupe_token_usage, usage)
    if source_claim_dedupe_token_usage.get("total", {}).get("total_tokens", 0):
        result["source_claim_dedupe_token_usage"] = source_claim_dedupe_token_usage
        result["token_usage"] = cv._merge_token_usage(
            result.get("token_usage", {}),
            source_claim_dedupe_token_usage,
        )
    issue_keys = {
        record.get("source_claim_key") or _record_key(record)
        for records in (
            final_confirmed,
            professor_check,
            crosscheck_rejected,
            grounding_rejected,
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
        grounding_rejected,
        first_stage_rejected,
    )
    final_rejected_keys = {
        _record_key(record)
        for records in final_rejected_lists
        for record in records
    }
    category_breakdown: dict[str, int] = {}
    issue_type_breakdown: dict[str, int] = {}
    verification_basis_breakdown: dict[str, int] = {}
    evidence_need_breakdown: dict[str, int] = {}
    for record in final_confirmed:
        key = str(record.get("issue_category") or record.get("pedagogical_type") or "uncategorized")
        category_breakdown[key] = category_breakdown.get(key, 0) + 1
        type_label = str(
            record.get("issue_type_label")
            or record.get("issue_category_label")
            or record.get("issue_type")
            or "검토 필요"
        )
        issue_type_breakdown[type_label] = issue_type_breakdown.get(type_label, 0) + 1
        basis = str(record.get("verification_basis") or "unclassified")
        evidence_need = str(record.get("evidence_need") or "unclassified")
        verification_basis_breakdown[basis] = verification_basis_breakdown.get(basis, 0) + 1
        evidence_need_breakdown[evidence_need] = evidence_need_breakdown.get(evidence_need, 0) + 1

    result["claim_decision_flow_summary"] = {
        "extracted_claim_count": len(claims),
        "final_confirmed_claim_count": len(final_confirmed),
        "final_confirmed_issue_category_breakdown": category_breakdown,
        "final_confirmed_issue_type_breakdown": issue_type_breakdown,
        "final_confirmed_verification_basis_breakdown": verification_basis_breakdown,
        "final_confirmed_evidence_need_breakdown": evidence_need_breakdown,
        "professor_check_claim_count": len(professor_check),
        "crosscheck_rejected_claim_count": len(crosscheck_rejected),
        "crosscheck_inconclusive_claim_count": len(crosscheck_inconclusive),
        "grounding_rejected_claim_count": len(grounding_rejected),
        "first_stage_rejected_claim_count": len(first_stage_rejected),
        "final_rejected_claim_count": len(final_rejected_keys),
        "unmatched_issue_record_count": len(unmatched_issue_records),
    }
    result["confirmed_count"] = len(final_confirmed)
    result["professor_check_count"] = len(professor_check)
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
            "label": "교수 확인 대상 claim",
            "key": "professor_check_claims",
            "count": len(professor_check),
        },
        {
            "label": "최종 확정 이슈 유형별 수",
            "key": "final_confirmed_issue_category_breakdown",
            "count": len(final_confirmed),
            "breakdown": issue_type_breakdown,
            "category_breakdown": category_breakdown,
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
    if grounding_rejected:
        result["claim_decision_overview"].append(
            {
                "label": "grounding 기각 claim",
                "key": "grounding_rejected_claims",
                "count": len(grounding_rejected),
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
        "professor_check_claims": professor_check,
        "crosscheck_rejected_claims": crosscheck_rejected,
        "first_stage_rejected_claims": first_stage_rejected,
        "crosscheck_inconclusive_claims": crosscheck_inconclusive,
        "grounding_rejected_claims": grounding_rejected,
        "unmatched_issue_records": unmatched_issue_records,
    }
    unmatched_confirmed = [
        record for record in unmatched_issue_records
        if record.get("stage") == "final_confirmed"
    ]
    unmatched_professor_check = [
        record for record in unmatched_issue_records
        if record.get("stage") == "crosscheck_inconclusive"
    ]
    unmatched_confirmed, routed_unmatched_professor_check = _split_confirmed_and_professor_check(unmatched_confirmed)
    unmatched_professor_check.extend(routed_unmatched_professor_check)
    unmatched_rejected = [
        record for record in unmatched_issue_records
        if record.get("stage") in {"crosscheck_rejected", "grounding_rejected"}
    ]
    result.update(
        _build_v2_output(
            result,
            claims,
            utterance_lookup,
            final_confirmed + unmatched_confirmed,
            professor_check + unmatched_professor_check,
            crosscheck_rejected + grounding_rejected + unmatched_rejected,
        )
    )
    return result


def _reorder_result_for_output(result: dict) -> dict:
    preferred_order = [
        "schema_version",
        "mode",
        "crosscheck_mode",
        "crosscheck_context_mode",
        "crosscheck_focus_window",
        "models",
        "crosscheck_models",
        "crosscheck_source_models",
        "crosscheck_model_map",
        "crosscheck_model_weights",
        "crosscheck_score_report",
        "claims_source_path",
        "summary",
        "claims",
        "feedback_groups",
        "feedback_items",
        "views",
        "pipeline_models",
        "claim_decision_overview",
        "claim_decision_flow_summary",
        "claim_decision_flow",
        "overall_assessment",
        "primary_model",
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
        "professor_check_count",
        "issues",
        "crosscheck_rejected_issues",
        "crosscheck_inconclusive_issues",
        "grounding_rejected_issues",
        "rejected_issues",
        "slide_typos",
        "claims_log_path",
        "previous_claims_log_path",
        "claims_raw_diff_path",
        "claims_raw_diff_summary",
        "merged_claims",
        "claim_extract_token_usage",
        "token_usage_per_model",
        "source_claim_dedupe_token_usage",
        "token_usage",
        "crosscheck_filtered",
        "crosscheck_inconclusive_filtered",
        "grounding_filtered",
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


def _compact_result_for_output(result: dict) -> dict:
    if _include_debug_fields():
        return result

    compact_keys = [
        "schema_version",
        "mode",
        "crosscheck_mode",
        "crosscheck_context_mode",
        "crosscheck_focus_window",
        "models",
        "crosscheck_models",
        "crosscheck_source_models",
        "crosscheck_model_map",
        "crosscheck_model_weights",
        "crosscheck_score_report",
        "claims_source_path",
        "summary",
        "claims",
        "feedback_groups",
        "feedback_items",
        "views",
        "slide_typos",
    ]
    return {key: result[key] for key in compact_keys if key in result}


def run_all_analyzers(
    merged_path: str,
    *,
    output_dir: str | None = None,
    claims_jsonl: str | None = None,
    reuse_claims: bool = False,
    claim_runs: int = 1,
    claim_min_rate: float = 0.5,
    claim_batch_size: int = CLAIM_BATCH_SIZE,
    claim_max_workers: int = 4,
    cross_models: list[str] | None = None,
    crosscheck_models: list[str] | None = None,
    cross_runs: int = 1,
    cross_min_rate: float = 0.5,
    cross_batch_size: int = 20,
    current_date: str | None = None,
) -> dict:
    merged_file = Path(merged_path).resolve()
    if not merged_file.exists():
        raise FileNotFoundError(f"merged_clean 파일 없음: {merged_file}")

    base_stem = _base_stem(merged_file)
    out_dir = Path(output_dir).resolve() if output_dir else merged_file.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    result_json_path = out_dir / f"{base_stem}_content_verification.json"
    report_path = out_dir / f"{base_stem}_content_verification_report.txt"
    effective_claims_jsonl = claims_jsonl
    if reuse_claims and not effective_claims_jsonl:
        previous_claims_path = _claims_jsonl_path(result_json_path)
        if previous_claims_path.exists():
            effective_claims_jsonl = str(previous_claims_path)
            print(f"  기존 claim 추출 결과 재사용: {effective_claims_jsonl}")
        else:
            print(f"  재사용할 claim 파일 없음. 새로 추출합니다: {previous_claims_path}")

    from .cross_pipeline import cross_verify

    models = cross_models or _default_cross_models()
    effective_crosscheck_models = crosscheck_models or _default_crosscheck_models()
    if len(models) < 2:
        raise RuntimeError("cross verifier는 최소 2개 모델이 필요합니다. CROSS_VERIFY_MODELS 또는 --cross-models를 확인하세요.")
    missing_judge_keys = [
        f"{model}({missing} 없음)"
        for model in models
        for missing in [_missing_provider_key(model)]
        if missing
    ]
    if missing_judge_keys:
        raise RuntimeError(f"cross verifier judge 모델 키가 필요합니다: {', '.join(missing_judge_keys)}")
    if effective_crosscheck_models:
        effective_crosscheck_models = _filter_available_crosscheck_models(effective_crosscheck_models)
    print(f"  claim 추출 모델: {os.getenv('VERIFIER_CLAIM_EXTRACT_MODEL') or os.getenv('VERIFIER_MODEL') or 'gemini-2.5-flash'}")
    print(f"  verifier judge 모델: {', '.join(models)}")
    if effective_crosscheck_models:
        print(f"  crosscheck 전용 모델: {', '.join(effective_crosscheck_models)}")

    verification_result = cross_verify(
        merged_path=str(merged_file),
        models=models,
        num_runs=cross_runs,
        min_rate=cross_min_rate,
        batch_size=claim_batch_size,
        judge_batch_size=cross_batch_size,
        claims_jsonl=effective_claims_jsonl,
        crosscheck_models=effective_crosscheck_models or None,
        claim_runs=claim_runs,
        claim_min_rate=claim_min_rate,
        env_vars=_collect_env_vars(),
    )

    claims_for_log = verification_result.get("merged_claims")
    if claims_for_log is None:
        claims_for_log = verification_result.get("extracted_claims")
    previous_claims_log_path = _archive_existing_claims_log(_claims_jsonl_path(result_json_path))
    claims_log_path = _write_claims_jsonl(claims_for_log, result_json_path)
    if claims_log_path:
        verification_result["claims_log_path"] = claims_log_path
        if previous_claims_log_path:
            verification_result["previous_claims_log_path"] = previous_claims_log_path
            claims_raw_diff_path, claims_raw_diff = _write_claims_raw_diff(
                previous_claims_log_path,
                claims_log_path,
            )
            if claims_raw_diff_path:
                verification_result["claims_raw_diff_path"] = claims_raw_diff_path
            if claims_raw_diff:
                verification_result["claims_raw_diff_summary"] = claims_raw_diff.get("summary", {})

    verification_result = _augment_decision_flow(verification_result, merged_file)
    verification_result = _reorder_result_for_output(verification_result)
    claim_issue_count = verification_result.get(
        "confirmed_count",
        verification_result.get("overall_assessment", {}).get("total_issues", 0),
    )
    verification_result = _compact_result_for_output(verification_result)

    result_json_path.write_text(
        json.dumps(verification_result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "merged_path": str(merged_file),
        "output_dir": str(out_dir),
        "claim_output": str(result_json_path),
        "claim_report": "",
        "claim_issue_count": claim_issue_count,
        "used_cross": True,
    }


def main():
    _enable_docker_log_tee()

    parser = argparse.ArgumentParser(description="merged_clean 입력 기준 verifier 실행")
    parser.add_argument("merged_path", help="merged_clean.json 경로")
    parser.add_argument("--output-dir", default=None, help="결과 저장 디렉토리 (기본: merged 파일 폴더)")
    parser.add_argument("--claims-jsonl", default=None, help="이미 추출된 claims_extracted.jsonl 경로. 지정하면 claim 추출을 건너뜀")
    parser.add_argument(
        "--reuse-claims",
        action="store_true",
        help="결과 폴더의 기존 *_claims_extracted.jsonl이 있으면 claim 추출을 건너뜀",
    )
    parser.add_argument("--claim-runs", type=int, default=1)
    parser.add_argument("--claim-min-rate", type=float, default=0.5)
    parser.add_argument("--claim-batch-size", type=int, default=CLAIM_BATCH_SIZE)
    parser.add_argument("--claim-max-workers", type=int, default=4)
    parser.add_argument(
        "--cross-models",
        nargs="+",
        default=None,
        help="cross verifier 모델 목록 (기본: CROSS_VERIFY_MODELS, 없으면 CROSS_VERIFY_MODEL)",
    )
    parser.add_argument(
        "--crosscheck-models",
        nargs="+",
        default=None,
        help="3단계 crosscheck 전용 모델 목록. 지정하지 않으면 CROSS_CHECK_MODELS 또는 --cross-models를 사용",
    )
    parser.add_argument(
        "--cross-runs",
        "--judge-runs",
        dest="cross_runs",
        type=int,
        default=1,
        help="1차 judge 반복 횟수 (기본 1). crosscheck 모델 수는 --crosscheck-models 또는 CROSS_CHECK_MODELS로 지정",
    )
    parser.add_argument("--cross-min-rate", type=float, default=0.5)
    parser.add_argument("--cross-batch-size", type=int, default=20)
    parser.add_argument("--date", default=None, help="검증 기준 날짜 (YYYY-MM-DD)")
    args = parser.parse_args()

    result = run_all_analyzers(
        args.merged_path,
        output_dir=args.output_dir,
        claims_jsonl=args.claims_jsonl,
        reuse_claims=args.reuse_claims,
        claim_runs=args.claim_runs,
        claim_min_rate=args.claim_min_rate,
        claim_batch_size=args.claim_batch_size,
        claim_max_workers=args.claim_max_workers,
        cross_models=args.cross_models,
        crosscheck_models=args.crosscheck_models,
        cross_runs=args.cross_runs,
        cross_min_rate=args.cross_min_rate,
        cross_batch_size=args.cross_batch_size,
        current_date=args.date,
    )

    print("\n=== Verifier 완료 ===")
    print(f"merged    : {result['merged_path']}")
    print(f"claim     : {result['claim_output']}")
    if result["claim_report"]:
        print(f"claim txt : {result['claim_report']}")
    print(f"claim 이슈: {result['claim_issue_count']}건")


if __name__ == "__main__":
    main()
