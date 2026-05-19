"""Claim/issue merge helpers for cross-model verification."""

from __future__ import annotations

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed


def _normalize_issue_contract(issue: dict) -> dict:
    from . import claim_common as cv

    return cv.normalize_issue_metadata(issue)


def _compact_issue_text(value: str) -> str:
    return " ".join(str(value or "").split()).strip().lower()


def _issue_source_text(issue: dict) -> str:
    return " ".join(
        str(
            issue.get("source_text")
            or issue.get("claim_context_text")
            or issue.get("claim_text")
            or ""
        ).split()
    ).strip()


def _clamp_score(value, default: float = 0.0) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        score = default
    if 1.0 < score <= 100.0:
        score /= 100.0
    return max(0.0, min(1.0, score))


def _detector_vote_row(model: str, issue: dict) -> dict:
    """Compact per-model Issue_detection vote for later analysis/UI."""
    return {
        "model": str(model or "").strip(),
        "claim_id": str(issue.get("claim_id") or issue.get("source_claim_id") or "").strip(),
        "utterance_id": str(issue.get("utterance_id") or "").strip(),
        "context_id": str(issue.get("context_id") or "").strip(),
        "slide_number": issue.get("slide_number"),
        "start_time": issue.get("start_time"),
        "confidence": _clamp_score(
            issue.get("candidate_confidence", issue.get("confidence", 0.0))
        ),
        "claim_text": str(issue.get("claim_text", "") or "").strip(),
        "source_text": _issue_source_text(issue),
    }


def _detector_votes_map(raw) -> dict[str, list[dict]]:
    if not raw:
        return {}
    votes: dict[str, list[dict]] = {}
    if isinstance(raw, dict):
        iterator = raw.items()
    elif isinstance(raw, list):
        iterator = []
        for row in raw:
            if isinstance(row, dict):
                iterator.append((row.get("model"), row))
    else:
        return {}

    for model, value in iterator:
        model_key = str(model or "").strip()
        if not model_key:
            continue
        rows = value if isinstance(value, list) else [value]
        for row in rows:
            if not isinstance(row, dict):
                continue
            compact = {
                "model": model_key,
                "claim_id": str(row.get("claim_id") or "").strip(),
                "utterance_id": str(row.get("utterance_id") or "").strip(),
                "context_id": str(row.get("context_id") or "").strip(),
                "slide_number": row.get("slide_number"),
                "start_time": row.get("start_time"),
                "confidence": _clamp_score(row.get("confidence", 0.0)),
                "claim_text": str(row.get("claim_text") or "").strip(),
                "source_text": str(row.get("source_text") or "").strip(),
            }
            votes.setdefault(model_key, []).append(compact)
    return votes


def _merge_detector_votes(*vote_maps) -> dict[str, list[dict]]:
    merged: dict[str, list[dict]] = {}
    seen: set[tuple[str, str, str, str]] = set()
    for vote_map in vote_maps:
        for model, rows in _detector_votes_map(vote_map).items():
            for row in rows:
                key = (
                    model,
                    str(row.get("claim_id") or ""),
                    str(row.get("utterance_id") or row.get("context_id") or ""),
                    str(row.get("source_text") or row.get("claim_text") or "")[:120],
                )
                if key in seen:
                    continue
                seen.add(key)
                merged.setdefault(model, []).append(row)
    return merged


def _attach_detector_rollup(issue: dict, models: list[str]) -> dict:
    model_total = max(1, len(models or []))
    detected = sorted(set(str(model or "").strip() for model in issue.get("detected_by_models", []) if str(model or "").strip()))
    issue["detected_by_models"] = detected
    issue["detector_model_count"] = len(detected)
    issue["detector_model_total"] = model_total
    issue["detector_model_agreement_ratio"] = len(detected) / model_total
    if len(detected) == model_total:
        label = "all_models"
    elif len(detected) > 1:
        label = "partial_overlap"
    elif len(detected) == 1:
        label = "single_model"
    else:
        label = "none"
    issue["detector_agreement_label"] = label
    return issue


def _candidate_type_scores(issue: dict) -> dict[str, float]:
    from . import claim_common as cv

    raw_scores = (
        issue.get("candidate_type_scores")
        or issue.get("candidate_issue_type_scores")
        or issue.get("issue_type_scores")
        or {}
    )
    scores = {code: 0.0 for code in ("A", "B", "C", "D")}
    if isinstance(raw_scores, dict):
        for raw_key, raw_value in raw_scores.items():
            key = str(raw_key or "").strip()
            code = key.upper()
            if code not in scores:
                code = cv.issue_type_code(cv.normalize_issue_type(key))
            if code in scores:
                if isinstance(raw_value, dict):
                    raw_value = raw_value.get("score", raw_value.get("confidence", 0.0))
                scores[code] = _clamp_score(raw_value)
    return scores


def _candidate_primary_type_code(issue: dict) -> str:
    from . import claim_common as cv

    raw = str(
        issue.get("candidate_primary_type_code")
        or issue.get("primary_candidate_type_code")
        or issue.get("candidate_issue_type_code")
        or ""
    ).strip()
    code = raw.upper()
    if code in {"A", "B", "C", "D"}:
        return code
    raw_type = str(
        issue.get("candidate_primary_issue_type")
        or issue.get("primary_candidate_issue_type")
        or ""
    ).strip()
    code = cv.issue_type_code(raw_type)
    if code:
        return code
    scores = _candidate_type_scores(issue)
    best = max(scores, key=lambda item: scores[item])
    return best if scores[best] > 0 else ""


def _candidate_primary_issue_type(issue: dict) -> str:
    from . import claim_common as cv

    return cv.ISSUE_CODE_TO_TYPE.get(_candidate_primary_type_code(issue), "")


def _candidate_types_compatible(a: dict, b: dict) -> bool:
    a_code = _candidate_primary_type_code(a)
    b_code = _candidate_primary_type_code(b)
    if not a_code or not b_code or a_code == b_code:
        return True
    # D is often the pedagogical surface of the same A/C issue, so keep it merge-compatible.
    if "D" in {a_code, b_code}:
        return True
    a_scores = _candidate_type_scores(a)
    b_scores = _candidate_type_scores(b)
    return a_scores.get(b_code, 0.0) >= 0.45 or b_scores.get(a_code, 0.0) >= 0.45


def _merge_candidate_type_metadata(dst: dict, src: dict) -> dict:
    dst_scores = _candidate_type_scores(dst)
    src_scores = _candidate_type_scores(src)
    merged_scores = {
        code: max(dst_scores.get(code, 0.0), src_scores.get(code, 0.0))
        for code in ("A", "B", "C", "D")
    }
    primary_code = max(merged_scores, key=lambda item: merged_scores[item])
    if merged_scores[primary_code] <= 0:
        primary_code = ""
    dst["candidate_type_scores"] = merged_scores
    dst["candidate_primary_type_code"] = primary_code
    dst["candidate_primary_issue_type"] = _candidate_primary_issue_type(dst) if primary_code else ""
    return dst


def _issue_match_text(issue: dict) -> str:
    """중복 판단에 쓸 핵심 발화 조각."""
    source_text = _compact_issue_text(_issue_source_text(issue))
    if source_text:
        return source_text
    claim_text = _compact_issue_text(issue.get("claim_text", ""))
    if claim_text:
        return claim_text
    issue_hint = _compact_issue_text(issue.get("issue") or "")
    return issue_hint[:160]


def _token_overlap_ratio(a: str, b: str) -> float:
    a_tokens = {tok for tok in a.split() if tok}
    b_tokens = {tok for tok in b.split() if tok}
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / max(1, min(len(a_tokens), len(b_tokens)))


def _primary_subject_text(issue: dict) -> str:
    """Return the leading subject/topic of the source context when explicit.

    This is a generic guard for merge safety: nearby issues should not be merged
    just because they share a slide if their verified proposition is about a
    different primary subject.
    """
    text = _compact_issue_text(_issue_source_text(issue))
    if not text:
        return ""
    text = re.sub(r"^[a-z]\d+(?:-c\d+)?[:：]\s*", "", text)
    match = re.match(r"(.{2,60}?)(?:은|는|이|가|을|를)\s", text)
    if match:
        subject = match.group(1)
    else:
        subject = " ".join(text.split()[:4])
    subject = re.sub(r"\b의\b", " ", subject)
    subject = re.sub(r"([가-힣a-z0-9]+)의\b", r"\1", subject)
    return " ".join(subject.split()).strip()


def _subjects_compatible(a: dict, b: dict) -> bool:
    a_subject = _primary_subject_text(a)
    b_subject = _primary_subject_text(b)
    if not a_subject or not b_subject:
        return True
    if a_subject == b_subject:
        return True
    if a_subject in b_subject or b_subject in a_subject:
        return True
    return _token_overlap_ratio(a_subject, b_subject) >= 0.67


def _time_distance_sec(a: dict, b: dict) -> float:
    try:
        return abs(float(a.get("start_time", 0) or 0) - float(b.get("start_time", 0) or 0))
    except Exception:
        return 999999.0


def _issue_context_keys(issue: dict) -> set[str]:
    keys: set[str] = set()
    for key in ("context_id", "utterance_id"):
        value = str(issue.get(key, "") or "").strip()
        if value:
            keys.add(value)
    for key in ("context_ids", "utterance_ids", "canonical_member_utterance_ids"):
        values = issue.get(key)
        if isinstance(values, list):
            keys.update(str(value or "").strip() for value in values if str(value or "").strip())
    return keys


def _same_or_adjacent_context(a: dict, b: dict) -> bool:
    a_keys = _issue_context_keys(a)
    b_keys = _issue_context_keys(b)
    if a_keys & b_keys:
        return True
    return (
        int(a.get("slide_number", 0) or 0) == int(b.get("slide_number", 0) or 0)
        and _time_distance_sec(a, b) <= 18.0
    )


def _same_contextual_issue(a: dict, b: dict) -> bool:
    """같은 맥락에서 wording만 조금 다른 동일 이슈인지 판단."""
    same_utterance = (a.get("utterance_id") or "") == (b.get("utterance_id") or "")
    same_context = _same_or_adjacent_context(a, b)
    same_local_context = (
        not same_utterance
        and int(a.get("slide_number", 0) or 0) == int(b.get("slide_number", 0) or 0)
        and _time_distance_sec(a, b) <= 18.0
    )
    if not same_utterance and not same_context and not same_local_context:
        return False
    if not same_utterance and not _subjects_compatible(a, b):
        return False
    if not _candidate_types_compatible(a, b):
        return False
    a_claim = _compact_issue_text(a.get("claim_text", ""))
    b_claim = _compact_issue_text(b.get("claim_text", ""))
    if (same_utterance or same_context) and a_claim and b_claim:
        if a_claim == b_claim:
            return True
        shorter, longer = sorted((a_claim, b_claim), key=len)
        if len(shorter) >= 12 and shorter in longer:
            return True
        if _token_overlap_ratio(a_claim, b_claim) >= 0.72:
            return True

    a_match_text = _issue_match_text(a)
    b_match_text = _issue_match_text(b)
    if a_match_text and b_match_text:
        if a_match_text == b_match_text:
            return True
        shorter, longer = sorted((a_match_text, b_match_text), key=len)
        if len(shorter) >= 8 and shorter in longer:
            return True
        match_threshold = 0.62 if same_context else 0.8
        if _token_overlap_ratio(a_match_text, b_match_text) >= match_threshold:
            return True

    a_issue = _compact_issue_text(a.get("issue", ""))
    b_issue = _compact_issue_text(b.get("issue", ""))
    if a_issue and b_issue:
        shorter, longer = sorted((a_issue, b_issue), key=len)
        if len(shorter) >= 16 and shorter in longer:
            return True
        local_threshold = 0.4 if _time_distance_sec(a, b) <= 5.0 else 0.55
        if _token_overlap_ratio(a_issue, b_issue) >= (local_threshold if (same_local_context or same_context) else 0.8):
            return True

    return False


def _merge_issue_payload(dst: dict, src: dict) -> dict:
    from . import claim_common as cv

    _normalize_issue_contract(dst)
    _normalize_issue_contract(src)
    dst_before_selection = dict(dst)
    merged = dict(dst)
    merged_models = set(merged.get("detected_by_models", []))
    merged_models.update(src.get("detected_by_models", []))
    merged["detected_by_models"] = sorted(merged_models)
    merged["detector_model_votes"] = _merge_detector_votes(
        merged.get("detector_model_votes", {}),
        src.get("detector_model_votes", {}),
    )

    merged_claims = list(
        dict.fromkeys(
            [
                *(merged.get("merged_claim_texts", []) or [merged.get("claim_text", "")]),
                *(src.get("merged_claim_texts", []) or [src.get("claim_text", "")]),
            ]
        )
    )
    merged["merged_claim_texts"] = [x for x in merged_claims if x]

    merged_source_texts = list(
        dict.fromkeys(
            [
                *(merged.get("merged_source_texts", []) or [_issue_source_text(merged)]),
                *(src.get("merged_source_texts", []) or [_issue_source_text(src)]),
            ]
        )
    )
    merged["merged_source_texts"] = [x for x in merged_source_texts if x]

    for list_key in ("source_span_ids", "anchor_utterance_ids"):
        merged[list_key] = list(
            dict.fromkeys(
                [
                    *(merged.get(list_key, []) or []),
                    *(src.get(list_key, []) or []),
                ]
            )
        )
    if not merged.get("source_text") and src.get("source_text"):
        merged["source_text"] = src.get("source_text", "")
    if not merged.get("source_block_id") and src.get("source_block_id"):
        merged["source_block_id"] = src.get("source_block_id", "")
    if not merged.get("claim_context_text") and src.get("claim_context_text"):
        merged["claim_context_text"] = src.get("claim_context_text", "")
    if not merged.get("claim_context_block_id") and src.get("claim_context_block_id"):
        merged["claim_context_block_id"] = src.get("claim_context_block_id", "")
    if not merged.get("claim_context_ids") and src.get("claim_context_ids"):
        merged["claim_context_ids"] = src.get("claim_context_ids", [])

    try:
        dst_conf = float(merged.get("confidence", 0) or 0)
    except Exception:
        dst_conf = 0.0
    try:
        src_conf = float(src.get("confidence", 0) or 0)
    except Exception:
        src_conf = 0.0
    if src_conf > dst_conf:
        keep_lists = {
            "detected_by_models": merged["detected_by_models"],
            "detector_model_votes": merged.get("detector_model_votes", {}),
            "merged_claim_texts": merged["merged_claim_texts"],
            "merged_source_texts": merged["merged_source_texts"],
            "source_span_ids": merged.get("source_span_ids", []),
            "anchor_utterance_ids": merged.get("anchor_utterance_ids", []),
        }
        merged = dict(src)
        merged.update(keep_lists)
    _merge_candidate_type_metadata(merged, dst_before_selection)
    _merge_candidate_type_metadata(merged, src)
    return merged


def _cluster_contextual_issues(issues: list[dict]) -> list[dict]:
    clustered: list[dict] = []
    for issue in issues:
        _normalize_issue_contract(issue)
        merged = False
        for idx, existing in enumerate(clustered):
            if _same_contextual_issue(existing, issue):
                clustered[idx] = _merge_issue_payload(existing, issue)
                merged = True
                break
        if not merged:
            seeded = dict(issue)
            seeded["merged_claim_texts"] = [issue.get("claim_text", "")] if issue.get("claim_text") else []
            seeded["merged_source_texts"] = [_issue_source_text(issue)] if _issue_source_text(issue) else []
            clustered.append(seeded)
    return clustered


def _issue_cluster_model(models: list[str]) -> str:
    configured = str(os.getenv("VERIFIER_ISSUE_CLUSTER_MODEL", "") or "").strip()
    if configured:
        return configured
    for model in models or []:
        lowered = str(model or "").lower()
        if lowered.startswith(("gpt", "o1", "o3")) or "claude" in lowered or "sonnet" in lowered or "opus" in lowered:
            return str(model)
    return str((models or [""])[0] or "")


def _issue_cluster_enabled() -> bool:
    return str(os.getenv("VERIFIER_LLM_ISSUE_CLUSTERING", "1") or "1").strip().lower() not in {
        "0",
        "false",
        "off",
        "none",
    }


def _issue_cluster_max_workers() -> int:
    try:
        value = int(os.getenv("VERIFIER_ISSUE_CLUSTER_MAX_WORKERS", "2") or "2")
    except ValueError:
        value = 2
    return max(1, value)


def _issue_start_time(issue: dict) -> float:
    try:
        return float(issue.get("start_time", 0) or 0)
    except Exception:
        return 0.0


def _issue_cluster_context_gap_sec() -> float:
    try:
        value = float(os.getenv("VERIFIER_ISSUE_CLUSTER_CONTEXT_GAP_SEC", "60") or "60")
    except ValueError:
        value = 60.0
    return max(10.0, value)


def _split_issues_by_context(issues: list[dict]) -> list[list[dict]]:
    """LLM merge가 강의 전체의 같은 키워드를 과하게 묶지 않도록 시간 문맥을 끊는다."""
    if not issues:
        return []
    gap_sec = _issue_cluster_context_gap_sec()
    sorted_issues = sorted(
        issues,
        key=lambda issue: (
            _issue_start_time(issue),
            int(issue.get("slide_number", 0) or 0),
            str(issue.get("utterance_id", "") or ""),
        ),
    )
    chunks: list[list[dict]] = []
    current: list[dict] = []
    prev_time: float | None = None
    for issue in sorted_issues:
        current_time = _issue_start_time(issue)
        if current and prev_time is not None and current_time - prev_time > gap_sec:
            chunks.append(current)
            current = []
        current.append(issue)
        prev_time = current_time
    if current:
        chunks.append(current)
    return chunks


def _issue_cluster_row(issue_id: str, issue: dict) -> str:
    detected = ", ".join(issue.get("detected_by_models", []) or [])
    merged_claims = [x for x in issue.get("merged_claim_texts", []) or [] if x]
    merged_source_texts = [x for x in issue.get("merged_source_texts", []) or [] if x]
    extra_claims = ""
    if merged_claims:
        extra_claims = "\n  - related_claims: " + " | ".join(merged_claims[:4])
    if merged_source_texts:
        extra_claims += "\n  - related_source_texts: " + " | ".join(merged_source_texts[:4])
    return (
        f"### {issue_id}\n"
        f"- utterance_id: {issue.get('utterance_id', '')}\n"
        f"- slide: {issue.get('slide_number', '')}\n"
        f"- start_time: {float(issue.get('start_time', 0) or 0):.1f}\n"
        f"- detected_by: {detected}\n"
        f"- claim_text: {issue.get('claim_text', '')}\n"
        f"- source_text: {_issue_source_text(issue)}\n"
        f"{extra_claims}"
    )


def _parse_issue_cluster_payload(text: str, issue_ids: set[str]) -> list[dict]:
    from . import claim_common as cv

    cleaned = cv._strip_json_fence((text or "").strip())
    candidates = [cleaned]
    obj = cv._extract_first_json_object(cleaned)
    if obj and obj not in candidates:
        candidates.append(obj)

    for candidate in candidates:
        if not candidate:
            continue
        fixed = re.sub(r",\s*([}\]])", r"\1", candidate)
        for payload_text in (candidate, fixed):
            try:
                payload = json.loads(payload_text)
            except json.JSONDecodeError:
                continue
            groups = payload.get("groups") if isinstance(payload, dict) else payload
            if not isinstance(groups, list):
                continue

            parsed = []
            seen: set[str] = set()
            for row in groups:
                if not isinstance(row, dict):
                    continue
                raw_members = row.get("member_issue_ids") or row.get("members") or []
                if not isinstance(raw_members, list):
                    continue
                members = []
                for member in raw_members:
                    member_id = str(member or "").strip()
                    if member_id in issue_ids and member_id not in seen and member_id not in members:
                        members.append(member_id)
                if not members:
                    continue
                seen.update(members)
                representative = str(row.get("representative_issue_id", "") or "").strip()
                if representative not in members:
                    representative = members[0]
                parsed.append(
                    {
                        "canonical_issue_id": str(row.get("canonical_issue_id", "") or "").strip(),
                        "member_issue_ids": members,
                        "representative_issue_id": representative,
                        "canonical_wrong_proposition": str(
                            row.get("canonical_wrong_proposition", "") or ""
                        ).strip(),
                        "merge_rationale": str(row.get("merge_rationale", "") or "").strip(),
                    }
                )
            for issue_id in issue_ids:
                if issue_id not in seen:
                    parsed.append(
                        {
                            "canonical_issue_id": "",
                            "member_issue_ids": [issue_id],
                            "representative_issue_id": issue_id,
                            "canonical_wrong_proposition": "",
                            "merge_rationale": "",
                        }
                    )
            if parsed:
                return parsed
    raise ValueError("issue_cluster_response_parse_failed")


def _individual_issue_key(issue: dict) -> str:
    uid = str(issue.get("utterance_id", "") or "")
    if uid:
        return uid
    claim = _compact_issue_text(_issue_source_text(issue))
    return claim


def _compact_source_issue(issue: dict) -> dict:
    uid = str(issue.get("utterance_id", "") or "").strip()
    utterance_ids = [
        str(value).strip()
        for value in issue.get("utterance_ids", []) or []
        if str(value).strip()
    ]
    antecedent_ids = [
        str(value).strip()
        for value in issue.get("antecedent_context_ids", []) or []
        if str(value).strip()
    ]
    anchor_ids = [
        str(value).strip()
        for value in issue.get("anchor_utterance_ids", []) or []
        if str(value).strip()
    ]
    source_span_ids = [
        str(value).strip()
        for value in issue.get("source_span_ids", []) or []
        if str(value).strip()
    ]
    claim_text = str(issue.get("claim_text", "") or "").strip()
    source_text = _issue_source_text(issue)
    display_claim_text = f"{uid}: {claim_text}" if uid and claim_text else claim_text
    display_source_text = f"{uid}: {source_text}" if uid and source_text else source_text
    return {
        "utterance_id": uid,
        "utterance_ids": list(dict.fromkeys(utterance_ids or ([uid] if uid else []))),
        "anchor_utterance_ids": list(dict.fromkeys(anchor_ids)),
        "source_span_ids": list(dict.fromkeys(source_span_ids)),
        "source_text": source_text,
        "display_source_text": display_source_text,
        "source_block_id": str(issue.get("source_block_id", "") or "").strip(),
        "claim_context_ids": [
            str(value).strip()
            for value in issue.get("claim_context_ids", issue.get("source_span_ids", [])) or []
            if str(value).strip()
        ],
        "claim_context_text": str(
            issue.get("claim_context_text") or issue.get("source_text", "") or ""
        ).strip(),
        "claim_context_block_id": str(
            issue.get("claim_context_block_id") or issue.get("source_block_id", "") or ""
        ).strip(),
        "antecedent_context_ids": list(dict.fromkeys(antecedent_ids)),
        "slide_number": issue.get("slide_number"),
        "start_time": issue.get("start_time"),
        "end_time": issue.get("end_time"),
        "claim_type": issue.get("claim_type") or issue.get("type", ""),
        "claim_text": claim_text,
        "display_claim_text": display_claim_text,
        "detected_by_models": sorted(set(issue.get("detected_by_models", []) or [])),
        "detector_model_votes": _merge_detector_votes(issue.get("detector_model_votes", {})),
        "detector_model_count": issue.get("detector_model_count"),
        "detector_model_total": issue.get("detector_model_total"),
        "detector_model_agreement_ratio": issue.get("detector_model_agreement_ratio"),
        "detector_agreement_label": issue.get("detector_agreement_label", ""),
        "detector_confidence": _clamp_score(
            issue.get("candidate_confidence", issue.get("confidence", 0.0))
        ),
    }


def _ensure_source_issues(issue: dict) -> dict:
    """Ensure every issue carries at least its own source payload.

    LLM clustering creates source_issues for grouped items, but singleton chunks
    and fallback union paths used to pass through without it. Downstream
    crosscheck/frontend code expects this field for stable evidence formatting.
    """
    if not isinstance(issue, dict):
        issue = {}
    source_issues = issue.get("source_issues")
    if isinstance(source_issues, list) and source_issues:
        return issue
    issue["source_issues"] = [_compact_source_issue(issue)]
    issue["canonical_issue_count"] = max(1, int(issue.get("canonical_issue_count", 1) or 1))
    return issue


def _split_group_by_primary_subject(group: dict, issues_by_id: dict[str, dict]) -> list[dict]:
    members = [mid for mid in group.get("member_issue_ids", []) if mid in issues_by_id]
    if len(members) <= 1:
        return [group]

    buckets: list[list[str]] = []
    for member_id in members:
        issue = issues_by_id[member_id]
        for bucket in buckets:
            representative = issues_by_id[bucket[0]]
            if _subjects_compatible(representative, issue):
                bucket.append(member_id)
                break
        else:
            buckets.append([member_id])

    if len(buckets) <= 1:
        return [group]

    split_groups = []
    representative = str(group.get("representative_issue_id", "") or "")
    for bucket in buckets:
        split_representative = representative if representative in bucket else bucket[0]
        split_groups.append(
            {
                **group,
                "member_issue_ids": bucket,
                "representative_issue_id": split_representative,
                "canonical_wrong_proposition": "",
                "merge_rationale": "",
            }
        )
    return split_groups


def _build_issue_units(issues_by_id: dict[str, dict], groups: list[dict]) -> list[dict]:
    """canonical issue를 crosscheck가 검증할 실제 문맥 단위 issue로 만든다.

    최종 강의자 피드백의 단위는 claim 하나가 아니라 같은 오해를 만드는 인접 발화 묶음이다.
    같은 utterance에서 모델별 wording만 다른 후보는 합치고, 서로 다른 utterance가 같은
    설명 오류를 구성하면 하나의 issue unit으로 승격한다.
    """
    guarded_groups: list[dict] = []
    for group in groups:
        guarded_groups.extend(_split_group_by_primary_subject(group, issues_by_id))

    issue_units: list[dict] = []
    for index, group in enumerate(guarded_groups, start=1):
        members = [mid for mid in group.get("member_issue_ids", []) if mid in issues_by_id]
        if not members:
            continue
        canonical_id = group.get("canonical_issue_id") or f"ci_{index:04d}"
        member_utterance_ids = list(
            dict.fromkeys(
                value
                for mid in members
                for value in (
                    issues_by_id[mid].get("utterance_ids")
                    if isinstance(issues_by_id[mid].get("utterance_ids"), list)
                    else [issues_by_id[mid].get("utterance_id", "")]
                )
                if value
            )
        )

        merged_by_source: dict[str, dict] = {}
        ordered_source_keys: list[str] = []
        for member_id in members:
            source_issue = dict(issues_by_id[member_id])
            source_key = _individual_issue_key(source_issue)
            if source_key in merged_by_source:
                merged_by_source[source_key] = _merge_issue_payload(merged_by_source[source_key], source_issue)
            else:
                merged_by_source[source_key] = source_issue
                ordered_source_keys.append(source_key)

        source_issues = [merged_by_source[source_key] for source_key in ordered_source_keys]
        source_issues.sort(key=lambda issue: float(issue.get("start_time", 0) or 0))
        representative = max(source_issues, key=lambda issue: float(issue.get("confidence", 0) or 0))
        issue_unit = dict(representative)

        detected_models: set[str] = set()
        detector_votes: dict[str, list[dict]] = {}
        claim_lines = []
        problem_lines = []
        for source_issue in source_issues:
            detected_models.update(source_issue.get("detected_by_models", []) or [])
            detector_votes = _merge_detector_votes(
                detector_votes,
                source_issue.get("detector_model_votes", {}),
            )
            uid = source_issue.get("utterance_id", "")
            claim = source_issue.get("claim_text", "") or _issue_source_text(source_issue)
            problem = source_issue.get("issue", "")
            if claim:
                claim_lines.append(f"{uid}: {claim}" if uid else claim)
            if problem:
                problem_lines.append(f"{uid}: {problem}" if uid else problem)

        start_times = [float(issue.get("start_time", 0) or 0) for issue in source_issues]
        end_times = [
            float(issue.get("end_time", 0) or 0)
            for issue in source_issues
            if issue.get("end_time") is not None
        ]
        issue_unit["issue_unit_id"] = canonical_id
        issue_unit["is_issue_unit"] = True
        issue_unit["canonical_issue_id"] = canonical_id
        issue_unit["canonical_member_issue_ids"] = members
        issue_unit["canonical_member_utterance_ids"] = member_utterance_ids
        issue_unit["canonical_issue_count"] = len(source_issues)
        issue_unit["source_issues"] = [_compact_source_issue(issue) for issue in source_issues]
        issue_unit["detected_by_models"] = sorted(detected_models)
        issue_unit["detector_model_votes"] = detector_votes
        issue_unit["utterance_id"] = source_issues[0].get("utterance_id", "")
        issue_unit["utterance_ids"] = member_utterance_ids
        issue_unit["source_span_ids"] = list(
            dict.fromkeys(
                value
                for issue in source_issues
                for value in (issue.get("source_span_ids") or issue.get("utterance_ids") or [])
                if value
            )
        )
        issue_unit["anchor_utterance_ids"] = list(
            dict.fromkeys(
                value
                for issue in source_issues
                for value in (issue.get("anchor_utterance_ids") or [])
                if value
            )
        )
        source_texts = [
            str(issue.get("source_text", "") or "").strip()
            for issue in source_issues
            if str(issue.get("source_text", "") or "").strip()
        ]
        if source_texts:
            issue_unit["source_text"] = "\n".join(dict.fromkeys(source_texts))
        claim_context_texts = [
            str(issue.get("claim_context_text") or issue.get("source_text", "") or "").strip()
            for issue in source_issues
            if str(issue.get("claim_context_text") or issue.get("source_text", "") or "").strip()
        ]
        if claim_context_texts:
            issue_unit["claim_context_text"] = "\n".join(dict.fromkeys(claim_context_texts))
        claim_context_ids = [
            value
            for issue in source_issues
            for value in (issue.get("claim_context_ids") or issue.get("source_span_ids") or [])
            if value
        ]
        if claim_context_ids:
            issue_unit["claim_context_ids"] = list(dict.fromkeys(claim_context_ids))
        issue_unit["start_time"] = min(start_times) if start_times else issue_unit.get("start_time")
        if end_times:
            issue_unit["end_time"] = max(end_times)
        candidate_issue_text = "\n".join(dict.fromkeys(claim_lines))
        issue_unit["candidate_issue_text"] = candidate_issue_text
        if problem_lines:
            issue_unit["issue"] = "\n".join(dict.fromkeys(problem_lines))
        elif group.get("canonical_wrong_proposition"):
            issue_unit["issue"] = group["canonical_wrong_proposition"]
        if group.get("canonical_wrong_proposition"):
            issue_unit["canonical_wrong_proposition"] = group["canonical_wrong_proposition"]
            if not issue_unit.get("student_error"):
                issue_unit["student_error"] = group["canonical_wrong_proposition"]
        if group.get("merge_rationale"):
            issue_unit["canonical_merge_rationale"] = group["merge_rationale"]
        for source_issue in source_issues:
            _merge_candidate_type_metadata(issue_unit, source_issue)
        issue_units.append(issue_unit)
    return issue_units


def _rebuild_union_sets(issues: list[dict], models: list[str]) -> tuple[list[dict], list[dict], dict]:
    unioned = []
    intersected = []
    exclusive = {model: [] for model in models}

    for issue in issues:
        issue = _ensure_source_issues(issue)
        _attach_detector_rollup(issue, models)
        issue["cross_model_agreement"] = len(issue["detected_by_models"])
        unioned.append(issue)
        if len(issue["detected_by_models"]) == len(models):
            intersected.append(issue)
        elif len(issue["detected_by_models"]) == 1 and issue["detected_by_models"][0] in exclusive:
            exclusive[issue["detected_by_models"][0]].append(issue)

    unioned.sort(key=lambda x: float(x.get("start_time", 0) or 0))
    intersected.sort(key=lambda x: float(x.get("start_time", 0) or 0))
    for model in models:
        exclusive[model].sort(key=lambda x: float(x.get("start_time", 0) or 0))
    return unioned, intersected, exclusive


def _run_issue_cluster_chunk(
    issues: list[dict],
    models: list[str],
    *,
    id_prefix: str = "i",
    cluster_model: str = "",
) -> tuple[list[dict], dict]:
    from . import claim_common as cv

    if len(issues) <= 1:
        return [_ensure_source_issues(dict(issue)) for issue in issues], cv._empty_token_usage()

    issue_ids = [f"{id_prefix}{i:04d}" for i in range(1, len(issues) + 1)]
    issues_by_id = dict(zip(issue_ids, issues))
    issue_block = "\n\n".join(_issue_cluster_row(issue_id, issue) for issue_id, issue in issues_by_id.items())
    prompt = f"""아래는 서로 다른 judge 모델이 강의에서 발견한 문제 후보 목록입니다.
당신의 작업은 최종 판정이 아니라, 같은 문제 후보를 canonical issue 단위로 묶는 것입니다.

중요: 이 입력은 강의 전체가 아니라 하나의 가까운 설명 문맥 안에서만 잘라낸 후보입니다.
현재 입력 안에서도 학생 오답 명제와 판단 근거가 같을 때 묶고,
단지 같은 키워드나 같은 대주제가 반복된다는 이유로 묶지 마세요.

canonical issue는 최종 항목을 삭제하기 위한 것이 아니라, 강의자 화면에서
같은 근본 문제를 하나의 묶음으로 보여주기 위한 그룹입니다.

묶는 기준:
- 학생이 잘못 외울 수 있는 명제가 사실상 같음
- 문제 삼는 표현/개념이 같음
- source_text/claim_text가 가리키는 핵심 주어/대상이 같음
- 판단 근거와 문맥 확인 대상이 같음
- 같은 발화, 인접 발화, 같은 슬라이드 흐름에서 반복/보강된 문제임
- 서로 다른 문장이라도 하나의 crosscheck 문맥으로 함께 검증되는 결합된 오류임
- 대비되는 두 개념, 단계, 범주, 예시가 서로 뒤바뀐 것처럼 한 쌍의 설명 오류가
  여러 발화에 걸쳐 나타나면 같은 canonical issue로 묶으세요.
- 같은 context/인접 context에서 원문 wording만 조금 다르거나 같은 발화를 다른 모델이 서로 다른 claim으로 잡은 경우는
  가능한 한 하나의 canonical issue로 묶으세요.

묶으면 안 되는 기준:
- 키워드만 같고 학생 오답 명제나 수정 방향이 다르면 분리하세요.
- source_text/claim_text의 핵심 주어/대상이 다르면 분리하세요.
- 한 후보는 A에 대한 설명이고 다른 후보는 B에 대한 설명이면, 같은 슬라이드 흐름이어도 분리하세요.
- canonical_wrong_proposition을 만들 때 서로 다른 후보의 주어를 하나로 바꾸거나 더 넓은 주어로 재작성하지 마세요.
- 같은 utterance라도 서로 다른 문제이면 분리하세요.
- 같은 주제라도 학생 오답 명제와 판단 근거가 다르면 분리하세요.
- 확신이 없으면 분리하세요.

입력:
{issue_block}

응답은 JSON object 하나만 출력하세요.
모든 issue_id는 정확히 한 번만 포함하세요.

형식:
{{
  "groups": [
    {{
      "canonical_issue_id": "ci_0001",
      "member_issue_ids": ["{id_prefix}0001", "{id_prefix}0003"],
      "representative_issue_id": "{id_prefix}0001",
      "canonical_wrong_proposition": "학생이 잘못 외울 수 있는 명제",
      "merge_rationale": "같은 문제로 묶은 짧은 이유"
    }}
  ]
}}"""

    token_usage = cv._empty_token_usage()
    text, call_usage = cv._call_llm(
        prompt,
        max_tokens=min(8192, max(2048, 350 * len(issues))),
        temperature=0.0,
        response_format=(
            {"type": "json_object"}
            if cv._supports_json_object_response_format(cluster_model)
            else None
        ),
        stage="cross_recheck",
    )
    cv._add_call_usage(token_usage, call_usage)
    groups = _parse_issue_cluster_payload(text, set(issue_ids))
    issue_units = _build_issue_units(issues_by_id, groups)
    return issue_units, token_usage


def canonicalize_issues_with_llm(
    issues: list[dict],
    models: list[str],
) -> tuple[list[dict], list[dict], dict, dict]:
    """LLM으로 judge 후보를 canonical issue 단위로 묶는다.

    judge의 tentative type은 참고만 하고, 학생이 잘못 외울 명제와 수정 방향이
    같은지 기준으로 묶는다. 실패 시 원본 이슈를 그대로 반환한다.
    """
    from . import claim_common as cv

    preclustered_issues = _cluster_contextual_issues([dict(issue) for issue in issues])

    if not _issue_cluster_enabled() or len(preclustered_issues) <= 1:
        unioned, intersected, exclusive = _rebuild_union_sets(preclustered_issues, models)
        return unioned, intersected, exclusive, cv._empty_token_usage()

    cluster_model = _issue_cluster_model(models)
    old_model = os.environ.get("VERIFIER_CROSS_RECHECK_MODEL")
    if cluster_model:
        os.environ["VERIFIER_CROSS_RECHECK_MODEL"] = cluster_model
    token_usage = cv._empty_token_usage()
    try:
        chunks = _split_issues_by_context(preclustered_issues)
        max_issues = int(os.getenv("VERIFIER_ISSUE_CLUSTER_MAX", "40") or "40")
        if max_issues > 0:
            bounded_chunks = []
            for chunk in chunks:
                for start in range(0, len(chunk), max_issues):
                    bounded_chunks.append(chunk[start:start + max_issues])
            chunks = [chunk for chunk in bounded_chunks if chunk]
        if len(chunks) > 1:
            print(
                f"    LLM issue 묶음: 후보 {len(preclustered_issues)}건을 "
                f"{len(chunks)}개 문맥 chunk로 나누어 처리합니다."
            )
        def _run_chunk(chunk_index: int, chunk: list[dict]):
            chunk_units, call_usage = _run_issue_cluster_chunk(
                chunk,
                models,
                id_prefix=f"c{chunk_index}_i",
                cluster_model=cluster_model,
            )
            return chunk_index, chunk_units, call_usage

        issue_units = []
        worker_count = min(_issue_cluster_max_workers(), max(1, len(chunks)))
        if worker_count > 1 and len(chunks) > 1:
            print(f"    LLM issue 묶음 병렬 처리: max_workers={worker_count}")
            chunk_results = {}
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                futures = {
                    executor.submit(_run_chunk, chunk_index, chunk): chunk_index
                    for chunk_index, chunk in enumerate(chunks, start=1)
                }
                for future in as_completed(futures):
                    chunk_index = futures[future]
                    try:
                        chunk_results[chunk_index] = future.result()
                    except Exception as e:
                        raise RuntimeError(f"issue clustering chunk {chunk_index} failed: {e}") from e
            ordered_results = [chunk_results[i] for i in range(1, len(chunks) + 1)]
        else:
            ordered_results = [
                _run_chunk(chunk_index, chunk)
                for chunk_index, chunk in enumerate(chunks, start=1)
            ]
        for _chunk_index, chunk_units, call_usage in ordered_results:
            token_usage = cv._merge_token_usage(token_usage, call_usage)
            issue_units.extend(chunk_units)
        unioned, intersected, exclusive = _rebuild_union_sets(issue_units, models)
        return unioned, intersected, exclusive, token_usage
    except Exception as e:
        print(f"  ⚠️ LLM issue clustering 실패, 기존 union 결과를 사용합니다: {e}")
        unioned, intersected, exclusive = _rebuild_union_sets(preclustered_issues, models)
        return unioned, intersected, exclusive, token_usage
    finally:
        if old_model is None:
            os.environ.pop("VERIFIER_CROSS_RECHECK_MODEL", None)
        else:
            os.environ["VERIFIER_CROSS_RECHECK_MODEL"] = old_model


def _claim_key(claim: dict) -> str:
    explicit = str(claim.get("claim_fingerprint") or claim.get("claim_id") or "").strip()
    if explicit:
        return explicit
    uid = claim.get("utterance_id", "")
    text = claim.get("claim_text", "")[:60]
    return f"{uid}::{text}"


def union_claims(results: list[dict]) -> list[dict]:
    """여러 모델의 claim을 합집합 (claim_text + utterance_id 기준 dedupe)."""
    seen = {}
    for r in results:
        for item in r["claims_by_batch"]:
            for claim in item["claims"]:
                key = _claim_key(claim)
                if key not in seen:
                    seen[key] = claim
    return list(seen.values())


def build_issue_detection_stats(
    judge_results: dict,
    unioned: list[dict],
    models: list[str],
) -> dict:
    """Build union-stage detector agreement statistics for reports/UI."""
    model_order = [str(model or "").strip() for model in models or [] if str(model or "").strip()]
    if not model_order and isinstance(judge_results, dict):
        model_order = list(judge_results.keys())
    total_models = max(1, len(model_order))
    per_model_raw = {
        model: len((judge_results.get(model, {}) or {}).get("issues", []) or [])
        for model in model_order
    }
    per_model_union = {model: 0 for model in model_order}
    exclusive_by_model = {model: 0 for model in model_order}
    agreement_distribution: dict[str, int] = {}
    common_all = 0
    partial_overlap = 0
    single_model = 0

    issue_rows = []
    for issue in unioned or []:
        detected = sorted(
            set(str(model or "").strip() for model in issue.get("detected_by_models", []) if str(model or "").strip())
        )
        count = len(detected)
        agreement_distribution[str(count)] = agreement_distribution.get(str(count), 0) + 1
        if count == total_models:
            common_all += 1
        elif count > 1:
            partial_overlap += 1
        elif count == 1:
            single_model += 1
            if detected[0] in exclusive_by_model:
                exclusive_by_model[detected[0]] += 1
        for model in detected:
            if model in per_model_union:
                per_model_union[model] += 1
        issue_rows.append(
            {
                "issue_unit_id": issue.get("issue_unit_id") or issue.get("canonical_issue_id") or "",
                "utterance_id": issue.get("utterance_id", ""),
                "slide_number": issue.get("slide_number"),
                "start_time": issue.get("start_time"),
                "claim_text": issue.get("claim_text", ""),
                "source_text": _issue_source_text(issue),
                "detected_by_models": detected,
                "detector_model_count": count,
                "detector_model_total": total_models,
                "detector_model_agreement_ratio": count / total_models,
            }
        )

    pairwise = []
    for idx, left in enumerate(model_order):
        for right in model_order[idx + 1:]:
            both = 0
            either = 0
            for issue in unioned or []:
                detected = set(issue.get("detected_by_models", []) or [])
                has_left = left in detected
                has_right = right in detected
                if has_left or has_right:
                    either += 1
                if has_left and has_right:
                    both += 1
            pairwise.append(
                {
                    "models": [left, right],
                    "overlap_count": both,
                    "either_count": either,
                    "jaccard": both / either if either else 0.0,
                }
            )

    union_count = len(unioned or [])
    return {
        "models": model_order,
        "model_count": len(model_order),
        "raw_detections_per_model": per_model_raw,
        "union_detections_per_model": per_model_union,
        "union_issue_count": union_count,
        "common_all_model_count": common_all,
        "partial_overlap_count": partial_overlap,
        "single_model_count": single_model,
        "exclusive_by_model": exclusive_by_model,
        "agreement_distribution": agreement_distribution,
        "all_model_agreement_ratio": common_all / union_count if union_count else 0.0,
        "pairwise_overlap": pairwise,
        "issues": issue_rows,
    }


def _judge_context_overlap(default: int = 2) -> int:
    try:
        value = int(os.getenv("VERIFIER_JUDGE_CONTEXT_OVERLAP", str(default)) or str(default))
    except ValueError:
        value = default
    return max(0, value)


def rebuild_claim_batches(
    merged_claims: list[dict],
    utterances: list[dict],
    batch_size: int,
    context_overlap: int | None = None,
) -> list[dict]:
    """합집합 claim을 판정 batch로 재구성.

    Claim은 core context/발화 기준으로 한 번만 판정하고, prompt 문맥에는 core 앞뒤
    N개 context/발화를 추가한다. 이렇게 하면 batch 경계의 첫/마지막 claim도 인접 문맥을 본다.
    """
    safe_batch_size = max(1, int(batch_size or 1))
    overlap = _judge_context_overlap() if context_overlap is None else max(0, int(context_overlap or 0))
    if safe_batch_size > 1:
        overlap = min(overlap, safe_batch_size - 1)

    core_batches = [
        (start, min(start + safe_batch_size, len(utterances)))
        for start in range(0, len(utterances), safe_batch_size)
    ]
    batch_claims: list[dict] = []
    uid_to_core_index = {}
    for core_idx, (start, end) in enumerate(core_batches):
        for u in utterances[start:end]:
            uid_to_core_index[u.get("utterance_id", "")] = core_idx
        ctx_start = max(0, start - overlap)
        ctx_end = min(len(utterances), end + overlap)
        batch_claims.append({
            "batch": utterances[ctx_start:ctx_end],
            "claims": [],
            "core_start": start,
            "core_end": end,
            "context_overlap": overlap,
        })

    for claim in merged_claims:
        uid = claim.get("utterance_id", "")
        core_idx = uid_to_core_index.get(uid)
        if core_idx is None:
            for key in ("source_span_ids", "anchor_utterance_ids", "utterance_ids", "context_ids"):
                for candidate_uid in claim.get(key, []) or []:
                    core_idx = uid_to_core_index.get(str(candidate_uid or "").strip())
                    if core_idx is not None:
                        break
                if core_idx is not None:
                    break
        if core_idx is not None:
            batch_claims[core_idx]["claims"].append(claim)

    uid_to_utterance = {u.get("utterance_id", ""): u for u in utterances}
    uid_to_index = {u.get("utterance_id", ""): idx for idx, u in enumerate(utterances)}
    for item in batch_claims:
        if not item["claims"]:
            continue
        required_ids = {
            str(u.get("utterance_id", "") or "")
            for u in item["batch"]
            if str(u.get("utterance_id", "") or "")
        }
        for claim in item["claims"]:
            for key in (
                "source_span_ids",
                "anchor_utterance_ids",
                "utterance_ids",
                "context_ids",
                "antecedent_context_ids",
            ):
                for value in claim.get(key, []) or []:
                    value = str(value or "").strip()
                    if value in uid_to_utterance:
                        required_ids.add(value)
            for key in ("utterance_id", "context_id"):
                value = str(claim.get(key, "") or "").strip()
                if value in uid_to_utterance:
                    required_ids.add(value)
        item["batch"] = [
            uid_to_utterance[uid]
            for uid in sorted(required_ids, key=lambda value: uid_to_index.get(value, 10**9))
            if uid in uid_to_utterance
        ]

    return [
        item
        for item in batch_claims
        if item["claims"]
    ]


def _issue_match_key(issue: dict) -> str:
    """claim-level 매칭 키."""
    issue_candidate_key = str(issue.get("issue_candidate_key") or "").strip()
    if issue_candidate_key:
        return f"candidate::{issue_candidate_key}"
    claim_id = str(issue.get("claim_id") or issue.get("source_claim_id") or "").strip()
    if claim_id:
        return f"claim::{claim_id}"
    uid = issue.get("utterance_id", "")
    match_text = _issue_match_text(issue)
    return f"{uid}::{match_text}"


def _dedupe_model_issues(issues: list[dict]) -> dict[str, dict]:
    """같은 모델 안에서 같은 이슈 키는 confidence가 더 높은 것을 남긴다."""
    deduped: dict[str, dict] = {}
    for issue in issues or []:
        key = _issue_match_key(issue)
        current = deduped.get(key)
        if current is None:
            deduped[key] = issue
            continue
        try:
            new_conf = float(issue.get("confidence", 0) or 0)
        except Exception:
            new_conf = 0.0
        try:
            old_conf = float(current.get("confidence", 0) or 0)
        except Exception:
            old_conf = 0.0
        if new_conf >= old_conf:
            deduped[key] = issue
    return deduped


def union_issues(results: list[dict]) -> tuple[list[dict], list[dict], dict]:
    """여러 모델의 이슈를 claim-level 합집합으로 모은다."""
    if not results:
        return [], [], {}

    model_issues = {}
    for r in results:
        model = r["model"]
        normalized = [_normalize_issue_contract(issue) for issue in (r["issues"] or [])]
        model_issues[model] = _dedupe_model_issues(normalized)

    models = list(model_issues.keys())
    union_map = {}
    for model in models:
        for key, issue in model_issues[model].items():
            if key not in union_map:
                merged = dict(issue)
                merged["detected_by_models"] = [model]
                merged["detector_model_votes"] = {
                    model: [_detector_vote_row(model, issue)]
                }
                _normalize_issue_contract(merged)
                union_map[key] = merged
            else:
                union_map[key]["detected_by_models"].append(model)
                union_map[key]["detector_model_votes"] = _merge_detector_votes(
                    union_map[key].get("detector_model_votes", {}),
                    {model: [_detector_vote_row(model, issue)]},
                )
                _merge_candidate_type_metadata(union_map[key], issue)
                if float(issue.get("confidence", 0) or 0) > float(union_map[key].get("confidence", 0) or 0):
                    existing_type_scores = _candidate_type_scores(union_map[key])
                    existing_primary_code = _candidate_primary_type_code(union_map[key])
                    existing_primary_type = _candidate_primary_issue_type(union_map[key])
                    existing_detector_votes = union_map[key].get("detector_model_votes", {})
                    for k, v in issue.items():
                        if k not in {"detected_by_models", "detector_model_votes"}:
                            union_map[key][k] = v
                    union_map[key]["detector_model_votes"] = existing_detector_votes
                    union_map[key]["candidate_type_scores"] = {
                        code: max(existing_type_scores.get(code, 0.0), _candidate_type_scores(issue).get(code, 0.0))
                        for code in ("A", "B", "C", "D")
                    }
                    primary_code = max(
                        union_map[key]["candidate_type_scores"],
                        key=lambda code: union_map[key]["candidate_type_scores"][code],
                    )
                    if union_map[key]["candidate_type_scores"].get(primary_code, 0.0) <= 0:
                        primary_code = ""
                    union_map[key]["candidate_primary_type_code"] = primary_code or existing_primary_code
                    union_map[key]["candidate_primary_issue_type"] = (
                        _candidate_primary_issue_type(union_map[key]) or existing_primary_type
                    )

    exclusive = {model: [] for model in models}
    unioned = []
    intersected = []
    for issue in union_map.values():
        issue = _ensure_source_issues(issue)
        _attach_detector_rollup(issue, models)
        issue["cross_model_agreement"] = len(issue["detected_by_models"])
        unioned.append(issue)
        if len(issue["detected_by_models"]) == len(models):
            intersected.append(issue)
        elif len(issue["detected_by_models"]) == 1:
            exclusive[issue["detected_by_models"][0]].append(issue)

    unioned.sort(key=lambda x: float(x.get("start_time", 0) or 0))
    intersected.sort(key=lambda x: float(x.get("start_time", 0) or 0))
    for model in models:
        exclusive[model].sort(key=lambda x: float(x.get("start_time", 0) or 0))

    return unioned, intersected, exclusive
