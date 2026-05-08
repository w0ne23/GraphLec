from __future__ import annotations

import json
import os
import re
from collections import Counter

_CROSSCHECK_VERDICTS = {"agree", "disagree", "inconclusive"}
_CROSSCHECK_PARSE_RETRIES = 1
_CROSSCHECK_EXTRA_FIELDS = (
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

_API_FAILURE_MARKERS = (
    "timeout",
    "timed out",
    "apitimeout",
    "readtimeout",
    "connection error",
    "apiconnectionerror",
    "connecttimeout",
    "connect timeout",
    "handshake",
    "429",
    "500",
    "503",
    "resource_exhausted",
    "unavailable",
    "overloaded",
)

_CRITERIA_WEIGHTS = {
    "misinformation_risk": 0.15,
    "nearby_context_unresolved": 0.25,
    "slide_context_unresolved": 0.25,
    "concrete_basis": 0.20,
    "student_impact": 0.15,
}

_TERM_RE = re.compile(r"[가-힣A-Za-z0-9][가-힣A-Za-z0-9_+-]{1,}")
_KOREAN_SUFFIXES = (
    "에게서는",
    "에게는",
    "에서는",
    "으로는",
    "이라는",
    "라는",
    "이고",
    "이며",
    "에서",
    "으로",
    "에게",
    "부터",
    "까지",
    "처럼",
    "보다",
    "라고",
    "이나",
    "거나",
    "하고",
    "은",
    "는",
    "이",
    "가",
    "을",
    "를",
    "의",
    "도",
    "만",
    "로",
    "과",
    "와",
)
_SEMANTIC_PREFIXES = ("비", "무", "반", "탈")


def _strip_korean_suffix(term: str) -> str:
    text = str(term or "").strip()
    if len(text) <= 2:
        return text
    for suffix in _KOREAN_SUFFIXES:
        if text.endswith(suffix) and len(text) - len(suffix) >= 2:
            return text[: -len(suffix)]
    return text


def _normalize_term(term: str) -> str:
    text = _strip_korean_suffix(term)
    if re.fullmatch(r"[A-Za-z0-9_+-]+", text or ""):
        return text.lower()
    return text


def _extract_terms(text: str) -> list[str]:
    terms = []
    for raw in _TERM_RE.findall(str(text or "")):
        term = _normalize_term(raw)
        if len(term) >= 3:
            terms.append(term)
    return terms


def _levenshtein_distance(a: str, b: str, *, max_distance: int = 2) -> int:
    if a == b:
        return 0
    if abs(len(a) - len(b)) > max_distance:
        return max_distance + 1
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        row_min = i
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            value = min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost)
            current.append(value)
            row_min = min(row_min, value)
        if row_min > max_distance:
            return max_distance + 1
        previous = current
    return previous[-1]


def _looks_like_semantic_prefix_pair(a: str, b: str) -> bool:
    for prefix in _SEMANTIC_PREFIXES:
        if a == f"{prefix}{b}" or b == f"{prefix}{a}":
            return True
    return False


def _terms_are_asr_neighbors(a: str, b: str) -> bool:
    if not a or not b or a == b:
        return False
    if _looks_like_semantic_prefix_pair(a, b):
        return False
    max_len = max(len(a), len(b))
    if max_len < 3:
        return False
    distance = _levenshtein_distance(a, b, max_distance=2)
    if distance <= 1:
        return True
    return max_len >= 6 and distance <= 2 and (distance / max_len) <= 0.25


def _compact_reason(text: str, limit: int = 90) -> str:
    value = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(value) <= limit:
        return value
    return value[: limit - 1].rstrip() + "..."


def _format_utterance_line(u: dict, *, marker: str = "  ") -> str:
    uid = str(u.get("utterance_id", "") or "").strip()
    text = str(u.get("text", "") or "").strip()
    start = float(u.get("start_time", 0) or 0)
    slide_number = u.get("slide_number", "")
    status = str(u.get("correction_status", "") or "").strip()
    risk = str(u.get("correction_risk", "") or "").strip()
    reason = str(u.get("correction_reason", "") or "").strip()
    suffix = ""
    if status == "candidate_only":
        detail = f", risk={risk}" if risk else ""
        if reason:
            detail += f", reason={_compact_reason(reason, 60)}"
        suffix = f" [전사 교정 후보 미적용{detail}]"
    return f"{marker}{uid} [{start:.1f}s, slide {slide_number}] {text}{suffix}"


def _issue_text_for_artifact_check(issue: dict) -> str:
    parts = [
        issue.get("claim_text", ""),
        issue.get("issue", ""),
        issue.get("why_wrong", ""),
        issue.get("student_error", ""),
        issue.get("correct_info", ""),
    ]
    for source in issue.get("source_issues") or []:
        if isinstance(source, dict):
            parts.extend([
                source.get("claim_text", ""),
                source.get("problematic_content", ""),
                source.get("issue", ""),
            ])
    return "\n".join(str(part or "") for part in parts if part)


def _find_transcript_artifact_hint(
    issue: dict,
    utterances: list[dict],
    slides: list[dict],
    target_indices: list[int],
    slide_number: int,
    radius: int = 5,
) -> dict | None:
    target_indices = [idx for idx in target_indices if 0 <= idx < len(utterances)]
    if not target_indices:
        return None

    target_index_set = set(target_indices)
    window_indices: set[int] = set()
    for idx in target_indices:
        window_indices.update(range(max(0, idx - radius), min(len(utterances), idx + radius + 1)))

    target_text = "\n".join(str(utterances[idx].get("text", "") or "") for idx in target_indices)
    issue_text = _issue_text_for_artifact_check(issue)
    target_terms = Counter(_extract_terms(f"{target_text}\n{issue.get('claim_text', '')}"))
    if not target_terms:
        return None

    context_parts = []
    for idx in sorted(window_indices - target_index_set):
        context_parts.append(str(utterances[idx].get("text", "") or ""))
    for slide in slides:
        if int(slide.get("slide_number", 0) or 0) == int(slide_number or 0):
            context_parts.append(str(slide.get("slide_text", "") or ""))
            break
    context_terms = Counter(_extract_terms("\n".join(context_parts)))
    if not context_terms:
        return None

    best: tuple[str, str, int, int] | None = None
    for target_term, target_count in target_terms.items():
        if target_count <= 0 or context_terms.get(target_term, 0) > 0:
            continue
        if target_term not in issue_text and target_term not in target_text:
            continue
        for context_term, context_count in context_terms.items():
            if context_count < 3:
                continue
            if not _terms_are_asr_neighbors(target_term, context_term):
                continue
            candidate = (target_term, context_term, target_count, context_count)
            if best is None or candidate[3] > best[3]:
                best = candidate

    if best is None:
        return None

    bad_term, context_term, bad_count, context_count = best
    note = (
        f"전사 오류 가능성: 대상 발화/claim의 '{bad_term}'는 주변 문맥에서는 거의 보이지 않고, "
        f"슬라이드와 주변 발화에서는 형태가 매우 가까운 '{context_term}'가 {context_count}회 반복됩니다. "
        f"이 이슈가 '{bad_term}' 한 단어에만 의존하면 강의 내용 오류로 확정하지 마세요."
    )
    return {
        "likely": True,
        "suspect_term": bad_term,
        "context_term": context_term,
        "suspect_count": bad_count,
        "context_count": context_count,
        "reason": note,
    }


def _coerce_confidence(value, default: float | None = None) -> float | None:
    if value is None or value == "":
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number > 1.0 and number <= 100.0:
        number = number / 100.0
    return max(0.0, min(1.0, number))


def _normalized_score_map(value, allowed_keys: dict[str, float]) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    scores: dict[str, float] = {}
    for key in allowed_keys:
        raw = value.get(key)
        if isinstance(raw, bool):
            scores[key] = 1.0 if raw else 0.0
            continue
        scores[key] = _coerce_confidence(raw, 0.0) or 0.0
    return scores


def _confidence_from_criteria(payload: dict) -> tuple[float | None, dict]:
    criteria = _normalized_score_map(
        payload.get("criteria_scores") or payload.get("criteria") or {},
        _CRITERIA_WEIGHTS,
    )
    if not criteria:
        return None, {}

    score = sum(criteria.get(key, 0.0) * weight for key, weight in _CRITERIA_WEIGHTS.items())
    score = max(0.0, min(1.0, score))

    gates = []
    if criteria.get("misinformation_risk", 0.0) < 0.5:
        score = min(score, 0.44)
        gates.append("misinformation_risk_not_verified")
    if criteria.get("nearby_context_unresolved", 0.0) < 0.5 and criteria.get("slide_context_unresolved", 0.0) < 0.5:
        score = min(score, 0.44)
        gates.append("resolved_by_nearby_context_and_slide")
    if criteria.get("concrete_basis", 0.0) < 0.5 and criteria.get("student_impact", 0.0) < 0.5:
        score = min(score, 0.79)
        gates.append("weak_concrete_basis_and_student_impact")

    breakdown = {
        "criteria_weights": _CRITERIA_WEIGHTS,
        "raw_score": round(sum(criteria.get(key, 0.0) * weight for key, weight in _CRITERIA_WEIGHTS.items()), 4),
        "applied_gates": gates,
        "computed_confidence": round(score, 4),
    }
    evidence = payload.get("criteria_evidence") if isinstance(payload.get("criteria_evidence"), dict) else {}
    return round(score, 4), {
        "criteria_scores": criteria,
        "criteria_evidence": evidence,
        "score_breakdown": breakdown,
    }


def _confidence_from_verdict(verdict: str) -> float:
    verdict = str(verdict or "").lower().strip()
    if verdict == "agree":
        return 1.0
    if verdict == "disagree":
        return 0.0
    return 0.5


def _verdict_from_confidence(confidence: float | None) -> str:
    if confidence is None:
        return "inconclusive"
    if confidence >= 0.75:
        return "agree"
    if confidence < 0.45:
        return "disagree"
    return "inconclusive"


def _is_crosscheck_api_failure(error: Exception | None) -> bool:
    if error is None:
        return False
    text = f"{type(error).__name__}: {error}".lower()
    return any(marker in text for marker in _API_FAILURE_MARKERS)


def _crosscheck_retry_message(error: Exception) -> str:
    if _is_crosscheck_api_failure(error):
        return "교차 재검증 API 호출 재시도"
    return "교차 재검증 batch JSON 파싱 재시도"


def _pack_crosscheck_payload(payload: dict) -> dict:
    raw_verdict = str(payload.get("verdict", "") or "").lower().strip()
    criteria_confidence, criteria_payload = _confidence_from_criteria(payload)
    confidence = _coerce_confidence(
        payload.get("confidence", payload.get("score", payload.get("issue_score"))),
        None,
    )
    if criteria_confidence is not None:
        confidence = criteria_confidence
    if confidence is None and raw_verdict in _CROSSCHECK_VERDICTS:
        confidence = _confidence_from_verdict(raw_verdict)
    if confidence is None:
        confidence = 0.5
    verdict = raw_verdict if raw_verdict in _CROSSCHECK_VERDICTS else _verdict_from_confidence(confidence)
    result = {
        "verdict": verdict,
        "confidence": confidence,
        "reason": str(payload.get("reason", "") or "").strip(),
    }
    result.update(criteria_payload)
    for field in _CROSSCHECK_EXTRA_FIELDS:
        value = str(payload.get(field, "") or "").strip()
        if value:
            result[field] = value
    return result


def _apply_transcript_artifact_cap(payload: dict, hint: dict | None) -> None:
    if not isinstance(payload, dict) or not isinstance(hint, dict) or not hint.get("likely"):
        return
    try:
        cap = float(os.getenv("VERIFIER_TRANSCRIPT_ARTIFACT_SCORE_CAP", "0.44") or "0.44")
    except ValueError:
        cap = 0.44
    cap = max(0.0, min(1.0, cap))
    original_confidence = _coerce_confidence(payload.get("confidence"), 0.5)
    payload["transcript_artifact_likely"] = True
    payload["transcript_artifact_reason"] = str(hint.get("reason", "") or "").strip()
    payload["transcript_artifact_terms"] = {
        "suspect": hint.get("suspect_term", ""),
        "context": hint.get("context_term", ""),
        "context_count": hint.get("context_count", 0),
    }
    if original_confidence is None or original_confidence <= cap:
        return

    payload["confidence_before_transcript_artifact_cap"] = round(original_confidence, 4)
    payload["confidence"] = round(cap, 4)
    payload["verdict"] = _verdict_from_confidence(cap)
    reason = str(payload.get("reason", "") or "").strip()
    artifact_reason = str(hint.get("reason", "") or "").strip()
    cap_reason = "강한 전사 오류 가능성이 있어 content issue 점수를 상한 처리했습니다."
    payload["reason"] = " / ".join(part for part in [artifact_reason, cap_reason, reason] if part)

    breakdown = payload.get("score_breakdown")
    if isinstance(breakdown, dict):
        gates = breakdown.setdefault("applied_gates", [])
        if isinstance(gates, list) and "transcript_artifact_likely" not in gates:
            gates.append("transcript_artifact_likely")
        breakdown["computed_confidence_before_transcript_artifact_cap"] = round(original_confidence, 4)
        breakdown["computed_confidence"] = round(cap, 4)


def _canonical_issue_id(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    lowered = text.lower().strip()
    m = re.fullmatch(r"(?:issue[_\s-]*)?i?0*(\d+)", lowered)
    if m:
        return f"i{int(m.group(1)):04d}"
    return text


def _issue_id_sort_key(issue_id: str) -> tuple[int, str]:
    canonical = _canonical_issue_id(issue_id)
    m = re.fullmatch(r"i(\d+)", canonical)
    if m:
        return int(m.group(1)), canonical
    return 999999, canonical


def _candidate_crosscheck_rows(payload, issue_ids: set[str]) -> list[dict]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if not isinstance(payload, dict):
        return []

    rows = (
        payload.get("results")
        or payload.get("verdicts")
        or payload.get("items")
        or payload.get("issues")
        or payload.get("judgments")
        or payload.get("scores")
    )
    if isinstance(rows, list):
        return [row for row in rows if isinstance(row, dict)]

    if len(issue_ids) == 1 and (
        "criteria_scores" in payload
        or "criteria" in payload
        or "verdict" in payload
        or "reason" in payload
    ):
        only_issue_id = next(iter(issue_ids))
        return [{**payload, "issue_id": payload.get("issue_id") or only_issue_id}]

    keyed_rows = []
    for key, value in payload.items():
        if not isinstance(value, dict):
            continue
        canonical = _canonical_issue_id(key)
        if key in issue_ids or canonical in {_canonical_issue_id(issue_id) for issue_id in issue_ids}:
            keyed_rows.append({**value, "issue_id": value.get("issue_id") or key})
    return keyed_rows


def _parse_crosscheck_batch_payload(text: str, issue_ids: set[str]) -> dict[str, dict]:
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

            rows = _candidate_crosscheck_rows(payload, issue_ids)
            if not rows:
                continue

            parsed: dict[str, dict] = {}
            ordered_issue_ids = sorted(issue_ids, key=_issue_id_sort_key)
            canonical_map = {_canonical_issue_id(issue_id): issue_id for issue_id in issue_ids}
            for row_index, row in enumerate(rows):
                issue_id = str(row.get("issue_id", "") or row.get("id", "") or "").strip()
                if issue_id not in issue_ids:
                    issue_id = canonical_map.get(_canonical_issue_id(issue_id), "")
                if not issue_id and len(rows) == len(ordered_issue_ids):
                    issue_id = ordered_issue_ids[row_index]
                if not issue_id and len(ordered_issue_ids) == 1:
                    issue_id = ordered_issue_ids[0]
                if issue_id not in issue_ids:
                    continue
                packed = _pack_crosscheck_payload(row)
                if packed["verdict"] in _CROSSCHECK_VERDICTS:
                    parsed[issue_id] = packed
            if parsed:
                return parsed

    raise ValueError("crosscheck_batch_response_parse_failed")


def _build_slide_transcript_block(
    slides: list[dict],
    slide_number: int,
    utterances: list[dict] | None = None,
) -> str:
    slide_text = ""
    transcript_lines = []
    for slide in slides:
        if int(slide.get("slide_number", 0) or 0) != slide_number:
            continue
        # 슬라이드 자체 텍스트 (슬라이드에 적힌 내용)
        slide_text = str(slide.get("slide_text", "") or "").strip()
        break

    if utterances is not None:
        for u in utterances:
            if int(u.get("slide_number", 0) or 0) != slide_number:
                continue
            text = str(u.get("text", "") or "").strip()
            if text:
                transcript_lines.append(_format_utterance_line(u, marker="  "))
    else:
        for slide in slides:
            if int(slide.get("slide_number", 0) or 0) != slide_number:
                continue
            for seg in slide.get("transcript_segments", []) or []:
                start = float(seg.get("start", 0) or 0)
                corr = str(seg.get("text", "") or "").strip()
                orig = str(seg.get("text_original", "") or "").strip()
                status = str(seg.get("correction_status", "") or "").strip()
                text = corr or orig
                if not text:
                    continue
                if status == "candidate_only":
                    transcript_lines.append(f"  [{start:.1f}s] {orig or text}")
                elif corr and orig and corr != orig:
                    transcript_lines.append(f"  [{start:.1f}s] 교정: {corr} | 원문: {orig}")
                else:
                    transcript_lines.append(f"  [{start:.1f}s] {text}")
            break

    parts = []
    if slide_text:
        parts.append(f"[슬라이드 텍스트]\n{slide_text}")
    parts.append("[강의자 발화]\n" + ("\n".join(transcript_lines) if transcript_lines else "(없음)"))
    return "\n".join(parts)


def _build_multi_slide_transcript_block(
    slides: list[dict],
    slide_numbers: list[int],
    utterances: list[dict] | None = None,
) -> str:
    blocks = []
    for slide_number in slide_numbers:
        if slide_number <= 0:
            continue
        block = _build_slide_transcript_block(slides, slide_number, utterances)
        blocks.append(f"[슬라이드 {slide_number}]\n{block}")
    return "\n\n".join(blocks) if blocks else "(없음)"


def _build_utterance_window_block(
    utterances: list[dict],
    target_indices: list[int],
    radius: int = 5,
    target_labels: dict[int, list[str]] | None = None,
) -> str:
    """대상 발화별 ±radius 문맥을 중복 없이 병합해 보여준다."""
    if not utterances or not target_indices:
        return "(없음)"

    labels = target_labels or {}
    index_set: set[int] = set()
    for idx in target_indices:
        if idx < 0 or idx >= len(utterances):
            continue
        start = max(0, idx - radius)
        end = min(len(utterances), idx + radius + 1)
        index_set.update(range(start, end))

    if not index_set:
        return "(없음)"

    lines = []
    target_set = set(target_indices)
    for idx in sorted(index_set):
        u = utterances[idx]
        text = str(u.get("text", "") or "").strip()
        label = ",".join(labels.get(idx, []))
        marker = f">> {label} " if idx in target_set and label else (">> " if idx in target_set else "   ")
        if text:
            lines.append(_format_utterance_line(u, marker=marker))
    return "\n".join(lines)


def _crosscheck_slide_context(ctx: dict, slide_num: int) -> tuple[str, str, str]:
    slide_ctx = ctx["slide_ctx"]
    slides = ctx["slides"]
    utterances = ctx.get("utterances", [])
    prev_slide_num = max(0, int(slide_num or 0) - 1)
    slide_info = slide_ctx.get(slide_num, {})
    title = slide_info.get("title", f"슬라이드 {slide_num}")
    time_range = slide_info.get("time_range", "")
    prev_slide_info = slide_ctx.get(prev_slide_num, {}) if prev_slide_num > 0 else {}
    prev_title = prev_slide_info.get("title", f"슬라이드 {prev_slide_num}") if prev_slide_num > 0 else ""
    prev_time_range = prev_slide_info.get("time_range", "") if prev_slide_num > 0 else ""
    slide_numbers = [n for n in [prev_slide_num, int(slide_num or 0)] if n > 0]
    transcript_block = _build_multi_slide_transcript_block(slides, slide_numbers, utterances)
    target_label = f"{title} ({time_range})"
    prev_label = (prev_title + f" ({prev_time_range})") if prev_slide_num > 0 else "없음"
    return target_label, prev_label, transcript_block


def _crosscheck_context_text(target_label: str, prev_label: str, transcript_block: str) -> str:
    return (
        f"대상 슬라이드: {target_label}\n"
        f"이전 슬라이드: {prev_label}\n\n"
        f"이전+현재 슬라이드 내용 (슬라이드 텍스트 + 강의자 발화)\n"
        f"{transcript_block}"
    )


def _issue_line_for_batch(issue_id: str, issue: dict) -> str:
    from . import claim_common as cv

    issue_type = cv.normalize_issue_type(issue.get("type", ""))
    issue_label = issue.get("issue_type_label") or cv.issue_type_label(issue_type)
    source_issues = issue.get("source_issues") if isinstance(issue.get("source_issues"), list) else []
    related = issue.get("utterance_ids") or issue.get("canonical_member_utterance_ids") or []
    source_block = ""
    if source_issues:
        lines = []
        for source in source_issues:
            uid = source.get("utterance_id", "")
            claim = source.get("claim_text", "") or source.get("problematic_content", "")
            problem = source.get("issue", "")
            if claim:
                lines.append(f"  - {uid}: {claim}")
            if problem:
                lines.append(f"    문제 후보: {problem}")
        source_block = "\n- 묶인 발화/claim:\n" + "\n".join(lines)
    elif related:
        source_block = "\n- 관련 utterance_ids: " + ", ".join(str(uid) for uid in related)
    artifact = issue.get("_transcript_artifact_hint")
    artifact_block = ""
    if isinstance(artifact, dict) and artifact.get("likely"):
        artifact_block = f"\n- 전사 오류 가능성 힌트: {artifact.get('reason', '')}"
    return (
        f"### {issue_id}\n"
        f"- utterance_id: {issue.get('utterance_id', '')}\n"
        f"- issue_unit_id: {issue.get('issue_unit_id') or issue.get('canonical_issue_id') or ''}\n"
        f"- 유형: {issue_label} ({issue_type})\n"
        f"- claim: {issue.get('claim_text', '')}\n"
        f"- 문제: {issue.get('issue', '')}"
        f"{source_block}"
        f"{artifact_block}"
    )


def _split_crosscheck_prompt(prompt: str) -> tuple[str, str | None]:
    dynamic_marker = "## 도메인"
    instruction_marker = "## 판정 절차"
    dynamic_start = prompt.find(dynamic_marker)
    instruction_start = prompt.find(instruction_marker)
    if 0 <= dynamic_start < instruction_start:
        system_prompt = prompt[:dynamic_start].rstrip() + "\n\n" + prompt[instruction_start:].lstrip()
        return prompt[dynamic_start:instruction_start].strip(), system_prompt
    return prompt, None


def _build_crosscheck_batch_prompt(
    valid: list[tuple[str, dict]],
    ctx: dict,
) -> tuple[str, str | None, str, set[str], dict[str, dict]]:
    utterances = ctx["utterances"]
    hint = ctx["hint"]
    utt_map = {u["utterance_id"]: (i, u) for i, u in enumerate(utterances)}

    _, first_utt = utt_map[valid[0][1].get("utterance_id", "")]
    slide_num = int(first_utt.get("slide_number", 0) or 0)
    target_label, prev_label, transcript_block = _crosscheck_slide_context(ctx, slide_num)
    context_text = _crosscheck_context_text(target_label, prev_label, transcript_block)

    target_indices = []
    target_labels: dict[int, list[str]] = {}
    artifact_hints: dict[str, dict] = {}
    for issue_id, issue in valid:
        issue_indices = []
        related_ids = [issue.get("utterance_id", "")]
        related_ids.extend(issue.get("utterance_ids") or [])
        related_ids.extend(issue.get("canonical_member_utterance_ids") or [])
        for uid in dict.fromkeys(str(item or "") for item in related_ids):
            if uid not in utt_map:
                continue
            idx, _ = utt_map[uid]
            issue_indices.append(idx)
            target_indices.append(idx)
            target_labels.setdefault(idx, []).append(issue_id)
        artifact_hint = _find_transcript_artifact_hint(issue, utterances, ctx["slides"], issue_indices, slide_num)
        if artifact_hint:
            issue["_transcript_artifact_hint"] = artifact_hint
            artifact_hints[issue_id] = artifact_hint
    nearby_block = _build_utterance_window_block(utterances, target_indices, target_labels=target_labels)
    context_text = f"{context_text}\n\n대상 발화 전후 ±5개 병합 문맥\n{nearby_block}"
    artifact_block = "\n".join(
        f"- {issue_id}: {hint.get('reason', '')}"
        for issue_id, hint in artifact_hints.items()
        if hint.get("reason")
    )
    if artifact_block:
        context_text = f"{context_text}\n\n전사 오류 가능성 힌트\n{artifact_block}"
    issue_block = "\n\n".join(_issue_line_for_batch(issue_id, issue) for issue_id, issue in valid)
    issue_ids = {issue_id for issue_id, _ in valid}

    prompt = f"""다른 검증 모델이 아래 발화들에서 문제를 발견했습니다.
당신은 각 지적이 타당한지 원문과 강의 문맥만 기준으로 독립 판단해야 합니다.

## 도메인
{hint.get('label', '')}

## 대상 슬라이드
{target_label}

## 이전 슬라이드
{prev_label}

## 이전+현재 슬라이드 내용 (슬라이드 텍스트 + 강의자 발화)
{transcript_block}

## 대상 발화 전후 ±5개 병합 문맥
{nearby_block}

## 전사 오류 가능성 힌트
{artifact_block or "(없음)"}

## 지적 목록
{issue_block}

## 판정 절차
각 issue_id는 하나의 claim이 아니라 같은 오해를 만들 수 있는 문맥 단위 issue일 수 있습니다.
issue 안에 묶인 발화/claim이 여러 개 있으면, 개별 문장 하나가 아니라 그 발화 흐름 전체가 학생에게 남기는
잘못된 명제 또는 오해를 판단하세요.
서로 다른 issue_id끼리는 독립적으로 판단하세요. 새로운 이슈를 만들지 말고, 제공된 issue_id 각각에 대해서만 criteria_scores와 criteria_evidence를 작성하세요.

중요: 실재 대상의 구체 수치/비율/연도/규모를 다루는 이슈에서는 "핵심 설명용 예시라서 학생이 암기하지 않을 것"만으로
가산 기준을 자동으로 낮추거나 감점하지 마세요. 강의의 핵심이 다른 개념이어도, 현실 대상에 붙은 수치가 틀리거나 오래되었으면 교수에게 확인 대상으로
올릴 수 있습니다. 이 경우 문맥이 해결했다는 판단은 "해당 숫자가 임의값/가상값/변수값이라고 명시됨" 또는
"같은 문맥에서 정확한 값이나 최신 값으로 바로 정정됨"일 때만 가능합니다.

## 채점 기준
아래 5개 항목만 채우세요. 5개 항목의 가중치 합은 1.0입니다.
각 항목은 0.0 / 0.5 / 1.0 중 하나를 기본으로 쓰되, 꼭 필요하면 0.25나 0.75를 사용할 수 있습니다.

1. **misinformation_risk (0.15)**
   이 claim 또는 issue가 학생에게 잘못된 정보를 주게 되는 이유를 작성하고 검증하세요.
   잘못 외울 명제가 구체적이고, 원문에 의해 실제로 유도될 수 있으면 높게 줍니다.
   잘못된 명제를 재구성할 수 없거나 특정 단어/지시어만 과해석해야 성립하면 낮게 줍니다.

2. **nearby_context_unresolved (0.25)**
   대상 발화 전후 ±5개 발화 안에서 해당 claim을 해소하는 발화가 있는지 확인하고 검증하세요.
   주변 발화가 대상, 관계, 순서, 주체, 조건, 범위를 충분히 보완하면 낮게 줍니다.
   주변 발화가 해소하지 못하거나 같은 오해를 반복/강화하면 높게 줍니다.

3. **slide_context_unresolved (0.25)**
   해당 슬라이드의 텍스트, 그림 설명, 구조가 해당 claim을 해소하는지 확인하고 검증하세요.
   슬라이드가 문제를 명확히 보완하면 낮게 줍니다.
   슬라이드가 보완하지 못하거나 발화와 충돌하거나 같은 혼동을 강화하면 높게 줍니다.

4. **concrete_basis (0.20)**
   반례, 정답 충돌, 현재성 오류, 범위 과잉, 주체/과정 혼동처럼 구체적인 판단 근거가 있는지 검증하세요.
   현실 대상의 구체 수치/비율/연도/규모가 틀리거나 오래된 경우도 근거가 될 수 있습니다.
   범위 과잉은 단순히 더 많은 예외나 더 넓은 설명이 가능하다는 뜻이 아닙니다.
   강의 문맥 안에서 학생이 실제로 다른 가능성/주체/조건을 배제하는 명제를 외우게 될 때만 높게 줍니다.
   단순히 더 엄밀히 말할 수 있다는 정도, 표현 취향, 세부 생략뿐이면 낮게 줍니다.

5. **student_impact (0.15)**
   학생이 실제로 잘못 외우거나 후속 개념을 혼동할 위험이 구체적인지 검증하세요.
   어떤 오개념으로 이어지는지 설명 가능하면 높게 줍니다.
   오해 가능성이 추상적이거나 교수 표현 취향 수준이면 낮게 줍니다.

추가 판단 원칙:
- 슬라이드와 전사문은 서로 보완 근거입니다. 둘을 함께 봤을 때 학생이 자연스럽게 이해할 최종 의미를 판단하세요.
- 전사 오류 가능성 힌트가 있는 경우, 해당 이슈가 고립된 단어 하나에만 의존하는지 먼저 확인하세요.
- 주변 발화와 슬라이드가 일관되게 유사한 대체어를 지지하고, 문제 제기가 그 고립된 단어 없이는 성립하지 않으면 강의 내용 오류로 확정하지 마세요.
- 단, 같은 오류 표현이 여러 발화에서 반복되거나 슬라이드도 같은 오류를 쓰거나, 고립된 단어 외에도 독립적인 개념 충돌이 있으면 이슈를 유지할 수 있습니다.
- 완전한 정정 문장이 없더라도, 앞뒤 발화나 슬라이드 구조가 생략된 주어/대상/관계/범위를 자연스럽게 지지하면 그 해석을 우선하세요.
- 앞에서 올바른 설명이 한 번 나왔거나 슬라이드에 관련 키워드가 있다는 이유만으로 자동 해소하지 마세요. 뒤따르는 발화 흐름이 다시 다른 오해를 만들면 그 흐름 기준으로 판단하세요.
- 실재 대상의 구체 수치/비율/연도/규모가 틀리거나 오래되었고 학생이 예시 수치로 받아들일 수 있으면 issue를 유지할 수 있습니다. 명시적으로 가상의 대상/임의값/변수값이라고 밝힌 경우에만 이 이유로 낮게 채점할 수 있습니다.
- 반박 근거는 학생이 이 강의 구간에서 배우는 개념 수준과 맞아야 합니다. 강의가 설명하지 않는 세부 구현, 특수 상황, 예외적 전제만으로는 issue를 유지하지 마세요.
- "모든", "오직", "~만", "독점" 같은 닫힌 표현이 있어도, 문맥상 역할/책임/관리 주체/대표 경로를 강조한 표준적 설명이면 그 단어만으로 범위 과잉 점수를 올리지 마세요.
- 범위 과잉 issue를 유지하려면, 원문과 문맥을 함께 본 뒤에도 "다른 가능성은 불가능하다", "다른 주체는 관여하지 않는다", "이 조건에서만 성립한다"처럼 학생이 잘못 외울 닫힌 명제가 구체적으로 남아야 합니다.
- 반례가 강의 범위 밖의 더 상위/하위 계층, 예외적 구현, 고급 세부사항에만 의존하면 concrete_basis와 student_impact를 낮게 주세요.
- 더 자세히 말할 수 있다는 정도, 더 엄밀한 표현 가능성, 표현 취향만이면 모든 항목을 낮게 채점하세요.

최종 점수 기준:
- 0.00~0.44: 기각
- 0.45~0.79: 교수 확인
- 0.80~1.00: 확정

교수 확인 또는 확정 구간이 될 만한 채점이면 교수에게 보여줄 수 있는 설명 필드도 작성하세요.
기각 구간이 될 만한 채점이면 reason에 기각 이유를 쓰고 나머지 설명 필드는 비워도 됩니다.

응답 규칙:
- 반드시 JSON object 하나만 출력
- markdown/code fence 금지
- verdict는 출력하지 마세요
- confidence는 출력하지 마세요
- criteria_scores와 criteria_evidence는 반드시 출력
- 입력된 모든 issue_id에 대해 정확히 하나의 criteria_scores와 criteria_evidence를 출력
- 단일 issue를 받더라도 반드시 results 배열로 출력
- criteria_evidence의 각 값은 1문장 이내로 짧게 출력
- reason은 2문장 이내로 출력
- 추가 설명 필드는 교수에게 보여줄 필요가 있는 경우만 쓰고, 각 필드는 1문장 이내로 출력

응답 예시:
{{
  "results": [
    {{
      "issue_id": "i0001",
      "criteria_scores": {{
        "misinformation_risk": 0.0,
        "nearby_context_unresolved": 0.0,
        "slide_context_unresolved": 0.0,
        "concrete_basis": 0.0,
        "student_impact": 0.0
      }},
      "criteria_evidence": {{
        "misinformation_risk": "잘못된 명제가 구체적으로 남지 않음",
        "nearby_context_unresolved": "±5개 발화가 의미를 보완함",
        "slide_context_unresolved": "슬라이드가 의미를 보완함",
        "concrete_basis": "반례/충돌/조건 근거가 부족함",
        "student_impact": "학생 오개념 위험이 낮음"
      }},
      "reason": "원문과 문맥 기준의 판단 이유"
    }},
    {{
      "issue_id": "i0002",
      "criteria_scores": {{
        "misinformation_risk": 1.0,
        "nearby_context_unresolved": 1.0,
        "slide_context_unresolved": 0.5,
        "concrete_basis": 0.75,
        "student_impact": 0.75
      }},
      "criteria_evidence": {{
        "misinformation_risk": "잘못된 명제와 그 이유",
        "nearby_context_unresolved": "±5개 발화 확인 결과",
        "slide_context_unresolved": "슬라이드 확인 결과",
        "concrete_basis": "반례/충돌/조건 근거",
        "student_impact": "학생 오개념 가능성"
      }},
      "reason": "원문과 문맥 기준의 판단 이유",
      "issue": "문제점 또는 교수 확인 후보 요약",
      "correct_info": "올바른 정보 또는 필요한 조건/범위",
      "why_wrong": "왜 틀렸거나 오해를 부를 수 있는지",
      "counterexample": "반례 또는 예외 조건. 없으면 빈 문자열",
      "issue_basis": "명확한 반례 있음 | 조건/범위 누락 | 핵심 개념 동일시 | 주체/과정 혼동 | 교수 확인 필요",
      "student_error": "학생이 잘못 외울 수 있는 구체적 명제",
      "counterexample_or_condition": "반례 또는 조건",
      "context_resolution": "문맥에서 해소됨 | 일부 해소됨 | 해소 안 됨 | 모델 간 판단 불일치",
      "evidence_in_context": "제공된 문맥에서 판단을 뒷받침하는 근거",
      "student_misunderstanding": "학생 오해 가능성",
      "why_it_matters": "왜 중요한지",
      "suggested_rephrase": "대체 표현",
      "teaching_note": "교수에게 전달할 짧은 메모",
      "recommendation": "수정 또는 보충 방향"
    }}
  ]
}}"""
    prompt, system_prompt = _split_crosscheck_prompt(prompt)
    return prompt, system_prompt, context_text, issue_ids, artifact_hints


def _run_crosscheck_batch_prompt(
    valid: list[tuple[str, dict]],
    ctx: dict,
) -> tuple[dict[str, dict], dict, str, Exception | None]:
    from . import claim_common as cv

    if not valid:
        return {}, cv._empty_token_usage(), "", None

    prompt, system_prompt, context_text, issue_ids, artifact_hints = _build_crosscheck_batch_prompt(valid, ctx)

    model = str(cv._resolve_stage_model("cross_recheck") or "").strip()
    response_format = {"type": "json_object"} if cv._supports_json_object_response_format(model) else None
    max_tokens = min(8192, max(4096, 1600 * len(valid)))
    if cv._is_deepseek_model(model):
        try:
            deepseek_max_tokens = int(os.getenv("VERIFIER_DEEPSEEK_CROSSCHECK_MAX_TOKENS", "2048"))
        except ValueError:
            deepseek_max_tokens = 2048
        max_tokens = max(1024, min(max_tokens, deepseek_max_tokens))
    debug_raw = str(os.getenv("VERIFIER_DEBUG_CROSSCHECK_RAW", "") or "").strip().lower() in {"1", "true", "yes"}

    token_usage = cv._empty_token_usage()
    last_error = None
    for attempt in range(_CROSSCHECK_PARSE_RETRIES + 1):
        text = ""
        try:
            text, call_usage = cv._call_llm(
                prompt,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                thinking_budget=1024,
                response_format=response_format,
                stage="cross_recheck",
            )
            cv._add_call_usage(token_usage, call_usage)
            parsed = _parse_crosscheck_batch_payload(text, issue_ids)
            for issue_id, payload in parsed.items():
                payload.setdefault("crosscheck_context_text", context_text)
                _apply_transcript_artifact_cap(payload, artifact_hints.get(issue_id))
            return parsed, token_usage, context_text, None
        except Exception as e:
            last_error = e
            if cv._is_deepseek_model(model) and _is_crosscheck_api_failure(e):
                break
            if attempt < _CROSSCHECK_PARSE_RETRIES:
                print(
                    f"    ↺ {_crosscheck_retry_message(e)} "
                    f"({attempt+1}/{_CROSSCHECK_PARSE_RETRIES}) [{model}] {type(e).__name__}: {str(e)[:160]}",
                    flush=True,
                )
                if debug_raw and text:
                    preview = re.sub(r"\s+", " ", text).strip()[:800]
                    print(f"      raw preview: {preview}", flush=True)

    return {}, token_usage, context_text, last_error


def judge_single_claim(issue: dict, ctx: dict) -> tuple[dict, dict]:
    """단일 이슈도 batch crosscheck 프롬프트를 1건짜리로 재사용한다."""
    from . import claim_common as cv

    uid = issue.get("utterance_id", "")
    utt_map = {u["utterance_id"]: (i, u) for i, u in enumerate(ctx["utterances"])}
    if uid not in utt_map:
        return {"verdict": "inconclusive", "confidence": 0.5, "reason": "utterance_id를 찾을 수 없음"}, cv._empty_token_usage()

    issue_id = "i0001"
    parsed, token_usage, context_text, last_error = _run_crosscheck_batch_prompt([(issue_id, issue)], ctx)
    payload = parsed.get(issue_id)
    if payload:
        return payload, token_usage

    reason = f"교차 재검증 실패: {last_error}" if last_error else "교차 재검증 응답에 issue_id가 없음"
    return {
        "verdict": "inconclusive",
        "confidence": 0.5,
        "reason": reason,
        "crosscheck_context_text": context_text,
        "model_failed": bool(_is_crosscheck_api_failure(last_error)),
        "excluded_from_score": bool(_is_crosscheck_api_failure(last_error)),
    }, token_usage


def judge_claim_batch(issues: list[dict], ctx: dict) -> tuple[dict[str, dict], dict]:
    """같은 슬라이드 문맥의 여러 이슈를 한 번에 crosscheck한다."""
    from . import claim_common as cv

    if not issues:
        return {}, cv._empty_token_usage()

    utterances = ctx["utterances"]
    utt_map = {u["utterance_id"]: (i, u) for i, u in enumerate(utterances)}

    valid: list[tuple[str, dict]] = []
    payloads: dict[str, dict] = {}
    for idx, issue in enumerate(issues, 1):
        issue_id = f"i{idx:04d}"
        if issue.get("utterance_id", "") not in utt_map:
            payloads[issue_id] = {"verdict": "inconclusive", "confidence": 0.5, "reason": "utterance_id를 찾을 수 없음"}
            continue
        valid.append((issue_id, issue))

    if not valid:
        return payloads, cv._empty_token_usage()

    parsed, token_usage, context_text, last_error = _run_crosscheck_batch_prompt(valid, ctx)
    missing = [pair for pair in valid if pair[0] not in parsed]
    if missing and _is_crosscheck_api_failure(last_error):
        for issue_id, _issue in missing:
            parsed[issue_id] = {
                "verdict": "inconclusive",
                "confidence": 0.5,
                "reason": f"교차 재검증 batch 실패: {last_error}",
                "crosscheck_context_text": context_text,
                "model_failed": True,
                "excluded_from_score": True,
            }
        payloads.update(parsed)
        return payloads, token_usage

    for issue_id, issue in missing:
        single_parsed, call_usage, single_context, single_error = _run_crosscheck_batch_prompt([(issue_id, issue)], ctx)
        cv._add_call_usage(token_usage, call_usage)
        payload = single_parsed.get(issue_id)
        if payload:
            parsed[issue_id] = payload
            continue
        reason_error = single_error or last_error
        parsed[issue_id] = {
            "verdict": "inconclusive",
            "confidence": 0.5,
            "reason": f"교차 재검증 batch 실패: {reason_error}" if reason_error else "교차 재검증 응답에 issue_id가 없음",
            "crosscheck_context_text": single_context or context_text,
            "model_failed": bool(_is_crosscheck_api_failure(reason_error)),
            "excluded_from_score": bool(_is_crosscheck_api_failure(reason_error)),
        }

    payloads.update(parsed)
    return payloads, token_usage
