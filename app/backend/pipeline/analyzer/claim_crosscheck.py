from __future__ import annotations

import json
import os
import re

_CROSSCHECK_VERDICTS = {"agree", "disagree", "inconclusive"}
_CROSSCHECK_PARSE_RETRIES = 1
_CROSSCHECK_EXTRA_FIELDS = (
    "status",
    "checked_type_code",
    "type_gate_passed",
    "context_issue_summary",
    "context_resolution",
    "context_resolution_reason",
    "correction_hint",
    "context_issue_id",
    "merged_into_issue_id",
    "issue_type_rationale",
    "routing_decision",
    "classification_confidence",
    "secondary_issue_type_code",
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
    "credit balance",
    "balance is too low",
    "insufficient_quota",
    "billing",
    "resource_exhausted",
    "unavailable",
    "overloaded",
)

def _issue_type_from_code(code: str) -> str:
    from . import claim_common as cv

    return cv.ISSUE_CODE_TO_TYPE.get(str(code or "").strip().upper(), "")


def _issue_type_code_from_scores(raw_scores) -> str:
    from . import claim_common as cv

    scores = _normalized_issue_type_scores(raw_scores)
    if not scores:
        return ""
    issue_type = max(
        cv.ISSUE_TYPE_ORDER,
        key=lambda item: float(scores.get(item, 0.0) or 0.0),
    )
    return cv.issue_type_code(issue_type) if scores.get(issue_type, 0.0) > 0 else ""


def _checked_type_code_for_issue(issue: dict) -> str:
    from . import claim_common as cv

    raw_code = str(
        issue.get("candidate_primary_type_code")
        or issue.get("primary_candidate_type_code")
        or issue.get("candidate_issue_type_code")
        or issue.get("issue_type_code")
        or ""
    ).strip().upper()
    if raw_code in cv.ISSUE_CODE_TO_TYPE:
        return raw_code

    raw_type = str(
        issue.get("candidate_primary_issue_type")
        or issue.get("primary_candidate_issue_type")
        or issue.get("issue_type")
        or issue.get("type")
        or ""
    ).strip()
    code = cv.issue_type_code(raw_type)
    if code:
        return code

    return _issue_type_code_from_scores(
        issue.get("candidate_type_scores")
        or issue.get("candidate_issue_type_scores")
        or issue.get("issue_type_scores")
        or issue.get("type_scores")
        or {}
    ) or "A"


def _env_float(name: str, default: float, *, minimum: float = 0.0, maximum: float = 1.0) -> float:
    try:
        value = float(os.getenv(name, str(default)) or str(default))
    except ValueError:
        value = default
    return max(minimum, min(maximum, value))


def _env_int(name: str, default: int, *, minimum: int = 1, maximum: int = 4) -> int:
    try:
        value = int(os.getenv(name, str(default)) or str(default))
    except ValueError:
        value = default
    return max(minimum, min(maximum, value))


def _candidate_type_scores_by_code(issue: dict) -> dict[str, float]:
    from . import claim_common as cv

    raw_scores = (
        issue.get("candidate_type_scores")
        or issue.get("candidate_issue_type_scores")
        or issue.get("issue_classification_scores")
        or issue.get("issue_type_scores")
        or issue.get("type_scores")
        or {}
    )
    normalized = _normalized_issue_type_scores(raw_scores)
    return {
        cv.issue_type_code(issue_type): float(score or 0.0)
        for issue_type, score in normalized.items()
        if cv.issue_type_code(issue_type)
    }


def _checked_type_codes_for_issue(issue: dict) -> list[str]:
    from . import claim_common as cv

    primary_code = _checked_type_code_for_issue(issue)
    scores = _candidate_type_scores_by_code(issue)
    if not scores:
        return [primary_code]

    order = {"B": 0, "C": 1, "A": 2, "D": 3}
    if primary_code not in scores:
        scores[primary_code] = max(scores.values() or [0.0])
    top_score = max(scores.get(primary_code, 0.0), max(scores.values() or [0.0]))
    margin = _env_float("VERIFIER_CROSSCHECK_TYPE_ROUTE_MARGIN", 0.08)
    min_score = _env_float(
        "VERIFIER_CROSSCHECK_TYPE_ROUTE_MIN_SCORE",
        0.55,
    )
    max_routes = _env_int("VERIFIER_CROSSCHECK_MAX_TYPE_ROUTES", 2, minimum=1, maximum=4)

    ranked = sorted(
        (
            (code, float(scores.get(code, 0.0) or 0.0))
            for code in cv.ISSUE_CODE_TO_TYPE
        ),
        key=lambda item: (-item[1], order.get(item[0], 99)),
    )
    selected: list[str] = []
    for code, score in ranked:
        if code == primary_code:
            selected.append(code)
            continue
        if len(selected) >= max_routes:
            continue
        if score >= min_score and (top_score - score) <= margin:
            selected.append(code)

    if primary_code not in selected:
        selected.insert(0, primary_code)
    return list(dict.fromkeys(selected[:max_routes]))


def _type_specific_prompt_block(checked_type_code: str) -> str:
    code = str(checked_type_code or "").strip().upper()
    if code == "A":
        return """## A형 factual_error 전용 판정
이 checker는 A형 factual_error만 판단합니다. C형 범위 과잉 단정이나 D형 혼동 가능 설명으로 판단을 확장하지 마세요.

A형은 원문 발화가 정의, 분류, 포함 관계, 주체, 과정, 원인-결과, 작동 방식, 귀속 관계를 직접 잘못 연결한 경우입니다.

kept 조건:
1. 원문 발화에 잘못된 개념 관계가 직접 표현되어 있음
2. 단순 표현 취향이나 용어 엄밀성 문제가 아님
3. 강의 수준에서 중요한 개념 관계임
4. 같은 문맥에서 바로 정정되거나 올바른 뜻으로 좁혀지지 않음
5. 학생이 잘못 외울 구체 명제가 한 문장으로 명확히 남음

rejected 조건:
- 더 정확한 용어를 쓸 수 있다는 수준
- 표현은 거칠지만 핵심 관계가 맞는 경우
- 바로 뒤 문장이나 슬라이드가 같은 관계를 올바르게 보완한 경우
- 외부의 엄밀한 taxonomy를 적용해야만 문제가 되는 경우
- 학생이 잘못 외울 구체 명제가 남지 않는 경우"""
    if code == "B":
        return """## B형 temporal_error 전용 판정
이 checker는 B형 temporal_error만 판단합니다. 정의 오류, 범위 과잉, 혼동 가능 설명으로 판단을 확장하지 마세요.

B형은 현재성, 최신성, 지원 여부, 사용 여부, 버전, 표준, 정책, 통계처럼 시점에 따라 참거짓이 달라질 수 있는 경우입니다.
B형은 확정 오류라기보다 강의 제공 시점 또는 오늘 날짜 기준 확인 후보로 다루세요.

kept 조건:
1. 발화가 현재 또는 특정 시점의 상태를 말함
2. 해당 정보가 강의 제공 시점 또는 오늘 날짜 기준으로 달라질 수 있음
3. 문맥에서 과거 사례, 역사 설명, 특정 시점 설명으로 제한되지 않음
4. 학생이 오래된 정보 또는 현재 확인이 필요한 정보를 일반 사실처럼 외울 가능성이 있음

rejected 조건:
- 과거 사례 소개임이 명확함
- 역사적 설명임이 명확함
- 특정 시점 기준 설명임이 문맥에 있음
- 최신성 문제가 강의 주제와 무관함
- 확인 필요성이 약하거나 학생 오개념과 연결되지 않음"""
    if code == "C":
        return """## C형 scope_overclaim 전용 판정
이 checker는 C형 scope_overclaim만 판단합니다. A형 factual_error나 D형 confusing_explanation으로 판단을 확장하지 마세요.

C형은 전체/일부, 항상/가끔, 오직/복수, 가능/불가능, 일시/영구, 조건부/필연 같은 범위가 실제로 뒤바뀐 경우입니다.
단정어는 주의 신호일 뿐 그 자체로 issue 근거가 아닙니다.

kept 조건:
1. 실제로 무엇이 배제되거나 일반화되었는지 명확함
2. 그 배제/일반화가 강의 범위 제한, 예시 제한, 대표 설명, 대비, 강조가 아니라 일반 사실처럼 전달됨
3. 가능성, 일부 상황, 일시적 결과를 필연적 결과, 전체 상황, 영구적 결과처럼 전달함
4. 강의 수준에서 중요한 반례, 예외, 누락 대상, 조건 차이가 구체적으로 있음
5. 같은 문맥에서 범위가 좁혀지거나 보완되지 않음
6. 학생이 잘못 외울 일반 규칙 또는 배제 명제가 한 문장으로 남음

rejected 조건:
- 단정어만 있고 실제 배제 대상이 불명확함
- 강의 대비, 대표 설명, 강조 표현임이 문맥에서 명확하고 일반 규칙으로 남지 않음
- 입문 수준 단순화로 허용 가능한 설명임
- 더 엄밀하게는 예외가 있다는 수준임
- 슬라이드나 바로 뒤 설명이 범위를 좁힘
- 학생이 잘못 외울 배제 명제를 한 문장으로 쓰기 어려움"""
    return """## D형 confusing_explanation 전용 판정
이 checker는 D형 confusing_explanation만 판단합니다. 명확한 factual error, temporal error, scope overclaim로 판단을 확장하지 마세요.

D형은 명시적 사실 오류로 단정하기 어렵더라도, 설명 방식 때문에 학생이 구체적 오개념을 외울 가능성이 남는 경우입니다.

kept 조건:
1. 학생이 잘못 외울 구체 오개념을 한 문장으로 쓸 수 있음
2. 그 오개념이 단순 표현 어색함이 아니라 개념 관계, 역할, 과정, 원인, 조건 이해에 영향을 줌
3. 같은 문맥의 재표현, 슬라이드, 예시가 그 오해를 해소하지 않음
4. 더 친절한 설명이 가능하다는 수준을 넘어 실제 강의 이슈로 남음

rejected 조건:
- 구체 오개념 문장을 쓸 수 없음
- 막연히 헷갈릴 수 있다는 수준임
- 표현이 어색하지만 핵심 의미는 맞음
- 바로 뒤 문장이나 슬라이드가 의미를 올바르게 좁힘
- 더 자세히 설명하면 좋겠다는 수준임
- 강의 수준 밖의 엄밀성 문제에 가까움"""

def _format_utterance_line(u: dict, *, marker: str = "  ") -> str:
    uid = str(u.get("utterance_id", "") or "").strip()
    text = str(u.get("text", "") or "").strip()
    start = float(u.get("start_time", 0) or 0)
    slide_number = u.get("slide_number", "")
    return f"{marker}{uid} [{start:.1f}s, slide {slide_number}] {text}"


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


def _coerce_bool(value, default: bool | None = None) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None or value == "":
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "kept", "pass", "passed"}:
        return True
    if text in {"0", "false", "no", "n", "rejected", "fail", "failed"}:
        return False
    return default


def _score_from_payload(value: dict, key: str, default: float = 0.0) -> float:
    raw = value.get(key)
    if isinstance(raw, bool):
        return 1.0 if raw else 0.0
    score = _coerce_confidence(raw, default)
    if score is None:
        score = default
    return float(score)


def _normalized_issue_type_scores(value) -> dict[str, float]:
    from . import claim_common as cv

    if not isinstance(value, dict):
        return {}

    normalized: dict[str, float] = {}
    for issue_type in cv.ISSUE_TYPE_ORDER:
        code = cv.issue_type_code(issue_type)
        label = cv.issue_type_label(issue_type)
        candidate_keys = (
            issue_type,
            code,
            code.lower(),
            f"{code}_{issue_type}",
            f"{code.lower()}_{issue_type}",
            f"{code}. {label}",
            f"{code}.{label}",
            label,
        )
        scores = [
            _score_from_payload(value, key)
            for key in candidate_keys
            if key in value
        ]
        if scores:
            normalized[issue_type] = max(scores)

    return normalized


def _normalized_type_judgments(value) -> tuple[dict[str, float], dict[str, dict]]:
    from . import claim_common as cv

    if not isinstance(value, dict):
        return {}, {}

    scores: dict[str, float] = {}
    judgments: dict[str, dict] = {}
    for issue_type in cv.ISSUE_TYPE_ORDER:
        code = cv.issue_type_code(issue_type)
        label = cv.issue_type_label(issue_type)
        candidate_keys = (
            issue_type,
            code,
            code.lower(),
            f"{code}_{issue_type}",
            f"{code.lower()}_{issue_type}",
            f"{code}. {label}",
            f"{code}.{label}",
            label,
        )
        raw = None
        for key in candidate_keys:
            if key in value:
                raw = value.get(key)
                break
        if raw is None:
            continue
        reason = ""
        if isinstance(raw, dict):
            score = _score_from_payload(raw, "score")
            reason = str(raw.get("reason", "") or "").strip()
        else:
            score = _coerce_confidence(raw, 0.0) or 0.0
        scores[issue_type] = score
        judgments[code] = {
            "score": round(float(score), 4),
            "reason": reason,
        }

    return scores, judgments


def _primary_issue_type_from_scores(type_scores: dict[str, float]) -> str:
    from . import claim_common as cv

    if not type_scores:
        return ""
    return max(
        cv.ISSUE_TYPE_ORDER,
        key=lambda issue_type: float(type_scores.get(issue_type, 0.0) or 0.0),
    )


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
    try:
        confirm_threshold = float(os.getenv("CROSS_VERIFY_CONFIRMED_THRESHOLD", "0.80") or "0.80")
    except ValueError:
        confirm_threshold = 0.80
    try:
        professor_threshold = float(os.getenv("CROSS_VERIFY_PROFESSOR_CHECK_THRESHOLD", "0.40") or "0.40")
    except ValueError:
        professor_threshold = 0.40
    if confidence >= confirm_threshold:
        return "agree"
    if confidence >= min(professor_threshold, confirm_threshold):
        return "inconclusive"
    return "disagree"


def _is_crosscheck_api_failure(error: Exception | None) -> bool:
    if error is None:
        return False
    text = f"{type(error).__name__}: {error}".lower()
    return any(marker in text for marker in _API_FAILURE_MARKERS)


def _is_permanent_crosscheck_api_failure(error: Exception | None) -> bool:
    if error is None:
        return False
    text = f"{type(error).__name__}: {error}".lower()
    return any(
        marker in text
        for marker in (
            "credit balance",
            "balance is too low",
            "insufficient_quota",
            "billing",
        )
    )


def _crosscheck_retry_message(error: Exception) -> str:
    if _is_crosscheck_api_failure(error):
        return "교차 재검증 API 호출 재시도"
    return "교차 재검증 batch JSON 파싱 재시도"


def _pack_crosscheck_payload(payload: dict) -> dict:
    from . import claim_common as cv

    raw_status = str(payload.get("status", "") or "").lower().strip()
    raw_verdict = str(payload.get("verdict", "") or "").lower().strip()
    checked_type_code = str(payload.get("checked_type_code", "") or "").strip().upper()
    if checked_type_code not in cv.ISSUE_CODE_TO_TYPE:
        checked_type_code = str(payload.get("routing_decision", "") or "").strip().upper()
    if checked_type_code not in cv.ISSUE_CODE_TO_TYPE:
        checked_type_code = str(payload.get("issue_type_code", "") or "").strip().upper()
    checked_issue_type = _issue_type_from_code(checked_type_code)
    confidence = _coerce_confidence(
        payload.get("confidence", payload.get("score", payload.get("issue_score"))),
        None,
    )
    if confidence is None and raw_verdict in _CROSSCHECK_VERDICTS:
        confidence = _confidence_from_verdict(raw_verdict)
    if confidence is None:
        if raw_status == "kept":
            confidence = 0.65
        elif raw_status in {"rejected", "merged"}:
            confidence = 0.0
        else:
            confidence = 0.5
    verdict = raw_verdict if raw_verdict in _CROSSCHECK_VERDICTS else _verdict_from_confidence(confidence)
    if raw_status == "rejected":
        confidence = min(confidence, 0.39)
        verdict = "disagree"
    elif raw_status == "merged":
        confidence = min(confidence, 0.39)
        verdict = "disagree"
    result = {
        "verdict": verdict,
        "confidence": confidence,
        "reason": str(payload.get("reason", "") or "").strip(),
    }
    if checked_type_code:
        result["checked_type_code"] = checked_type_code
    gate_passed = _coerce_bool(payload.get("type_gate_passed"), None)
    if gate_passed is not None:
        result["type_gate_passed"] = gate_passed
        if not gate_passed:
            result["confidence"] = min(float(result.get("confidence", confidence) or 0.0), 0.39)
            result["verdict"] = "disagree"
            confidence = float(result["confidence"])
    judgment_scores, type_judgments = _normalized_type_judgments(
        payload.get("type_judgments")
        or payload.get("type_judgement")
        or payload.get("issue_type_judgments")
        or {}
    )
    type_scores = judgment_scores or _normalized_issue_type_scores(
        payload.get("issue_type_scores")
        or payload.get("type_scores")
        or payload.get("type_confidence")
        or payload.get("issue_type_confidence")
        or {}
    )
    issue_type = _primary_issue_type_from_scores(type_scores)
    if issue_type not in cv.ALLOWED_ISSUE_TYPES:
        issue_type = checked_issue_type or cv.normalize_issue_type(payload.get("issue_type") or payload.get("type") or "")
    if issue_type in cv.ALLOWED_ISSUE_TYPES:
        result["issue_type"] = issue_type
        result["issue_type_label"] = cv.issue_type_label(issue_type)
        result["issue_type_code"] = cv.issue_type_code(issue_type)
        result["issue_type_code_label"] = cv.issue_type_code_label(issue_type)
    if type_scores:
        result["issue_type_scores"] = {
            issue_type: round(float(type_scores.get(issue_type, 0.0) or 0.0), 4)
            for issue_type in cv.ISSUE_TYPE_ORDER
            if issue_type in type_scores
        }
    if type_judgments:
        result["type_judgments"] = type_judgments
        if not result.get("issue_type_rationale") and issue_type in cv.ALLOWED_ISSUE_TYPES:
            code = cv.issue_type_code(issue_type)
            reason = str((type_judgments.get(code) or {}).get("reason", "") or "").strip()
            if reason:
                result["issue_type_rationale"] = reason
    for field in _CROSSCHECK_EXTRA_FIELDS:
        if field in {"checked_type_code", "type_gate_passed"} and field in result:
            continue
        value = str(payload.get(field, "") or "").strip()
        if value:
            result[field] = value
    _apply_type_specific_score_gates(result)
    if not result.get("context_resolution"):
        result["context_resolution"] = "해소 안 됨" if result.get("verdict") == "agree" else "문맥에서 해소됨"
    return result


def _apply_type_specific_score_gates(result: dict) -> dict:
    from . import claim_common as cv

    issue_type = cv.normalize_issue_type(result.get("issue_type") or result.get("type") or "")
    if issue_type not in cv.ALLOWED_ISSUE_TYPES:
        return result

    confidence = _coerce_confidence(result.get("confidence"), 0.5)
    if confidence is None:
        confidence = 0.5

    gates = []

    if issue_type == "temporal_error":
        confidence = min(confidence, 0.79)
        gates.append("temporal_error_requires_followup_check")
    elif issue_type == "scope_overclaim":
        confidence = min(confidence, 0.79)
        gates.append("scope_overclaim_no_auto_confirm")
    elif issue_type == "confusing_explanation":
        confidence = min(confidence, 0.79)
        gates.append("confusing_explanation_no_auto_confirm")

    result["confidence"] = round(float(confidence), 4)
    result["verdict"] = _verdict_from_confidence(float(confidence))
    if gates:
        result["applied_score_gates"] = list(dict.fromkeys(gates))
    return result


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
    if isinstance(payload, str):
        try:
            return _candidate_crosscheck_rows(json.loads(payload), issue_ids)
        except json.JSONDecodeError:
            return []
    if not isinstance(payload, dict):
        return []

    for wrapper_key in ("json", "json_response", "response", "data", "output", "payload"):
        wrapped = payload.get(wrapper_key)
        if isinstance(wrapped, (dict, list)):
            wrapped_rows = _candidate_crosscheck_rows(wrapped, issue_ids)
            if wrapped_rows:
                return wrapped_rows
        if isinstance(wrapped, str):
            wrapped_rows = _candidate_crosscheck_rows(wrapped, issue_ids)
            if wrapped_rows:
                return wrapped_rows

    rows = (
        payload.get("results")
        or payload.get("result")
        or payload.get("verdicts")
        or payload.get("items")
        or payload.get("issues")
        or payload.get("judgments")
        or payload.get("scores")
    )
    if isinstance(rows, list):
        return [row for row in rows if isinstance(row, dict)]
    if isinstance(rows, str):
        return _candidate_crosscheck_rows(rows, issue_ids)
    if isinstance(rows, dict):
        nested_rows = []
        for key, value in rows.items():
            if not isinstance(value, dict):
                continue
            nested_rows.append({**value, "issue_id": value.get("issue_id") or key})
        if nested_rows:
            return nested_rows

    if len(issue_ids) == 1 and (
        "issue_score" in payload
        or "verdict" in payload
        or "status" in payload
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


def _omitted_crosscheck_payload(issue_id: str) -> dict:
    return _pack_crosscheck_payload(
        {
            "issue_id": issue_id,
            "status": "rejected",
            "issue_type_scores": {"A": 0.0, "B": 0.0, "C": 0.0, "D": 0.0},
            "issue_score": 0.0,
            "reason": "모델이 이 issue 후보를 유지할 이슈로 반환하지 않았습니다.",
            "context_issue_summary": "",
            "context_resolution": "문맥에서 해소됨",
            "context_resolution_reason": "모델 응답에서 해당 issue 후보가 유지 대상에서 제외되었습니다.",
            "correction_hint": "",
            "context_issue_id": "none",
            "merged_into_issue_id": "",
        }
    )


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
            if not rows and isinstance(payload, dict) and any(
                key in payload for key in ("results", "result", "items", "issues", "judgments", "scores")
            ):
                return {
                    issue_id: _omitted_crosscheck_payload(issue_id)
                    for issue_id in sorted(issue_ids, key=_issue_id_sort_key)
                }
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
                for issue_id in sorted(issue_ids - set(parsed), key=_issue_id_sort_key):
                    parsed[issue_id] = _omitted_crosscheck_payload(issue_id)
                return parsed
            if rows:
                return {
                    issue_id: _omitted_crosscheck_payload(issue_id)
                    for issue_id in sorted(issue_ids, key=_issue_id_sort_key)
                }

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
                text = corr or orig
                if not text:
                    continue
                if corr and orig and corr != orig:
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


def _adjacent_slide_numbers(slide_ctx: dict, slide_num: int) -> tuple[int, int]:
    slide_numbers = sorted(n for n in slide_ctx if isinstance(n, int) and n > 0)
    prev_numbers = [n for n in slide_numbers if n < slide_num]
    next_numbers = [n for n in slide_numbers if n > slide_num]
    return (
        prev_numbers[-1] if prev_numbers else 0,
        next_numbers[0] if next_numbers else 0,
    )




def _boundary_context_lines(
    utterances: list[dict],
    slide_num: int,
    *,
    first: bool,
    count: int = 2,
) -> str:
    if slide_num <= 0:
        return "(없음)"
    candidates = [u for u in utterances if int(u.get("slide_number", 0) or 0) == slide_num]
    if not candidates:
        return "(없음)"
    safe_count = max(1, int(count or 1))
    targets = candidates[:safe_count] if first else candidates[-safe_count:]
    return "\n".join(_format_utterance_line(target, marker="  ") for target in targets)


def _current_slide_context_lines(
    utterances: list[dict],
    slide_num: int,
    target_labels: dict[int, list[str]] | None = None,
) -> str:
    labels = target_labels or {}
    lines = []
    for idx, u in enumerate(utterances):
        if int(u.get("slide_number", 0) or 0) != slide_num:
            continue
        text = str(u.get("text", "") or "").strip()
        if not text:
            continue
        label = ",".join(labels.get(idx, []))
        marker = f">> {label} " if label else "  "
        lines.append(_format_utterance_line(u, marker=marker))
    return "\n".join(lines) if lines else "(없음)"


def _crosscheck_slide_context(
    ctx: dict,
    slide_num: int,
    target_labels: dict[int, list[str]] | None = None,
) -> tuple[str, str, str]:
    slide_ctx = ctx["slide_ctx"]
    slides = ctx["slides"]
    utterances = ctx.get("utterances", [])
    slide_info = slide_ctx.get(slide_num, {})
    title = slide_info.get("title", f"슬라이드 {slide_num}")
    time_range = slide_info.get("time_range", "")
    slide_text = ""
    for slide in slides:
        if int(slide.get("slide_number", 0) or 0) == slide_num:
            slide_text = str(slide.get("slide_text", "") or "").strip()
            break

    prev_slide_num, next_slide_num = _adjacent_slide_numbers(slide_ctx, int(slide_num or 0))
    prev_label = f"슬라이드 {prev_slide_num}" if prev_slide_num > 0 else "없음"
    next_label = f"슬라이드 {next_slide_num}" if next_slide_num > 0 else "없음"
    prev_boundary = _boundary_context_lines(utterances, prev_slide_num, first=False, count=2)
    next_boundary = _boundary_context_lines(utterances, next_slide_num, first=True, count=2)
    current_contexts = _current_slide_context_lines(utterances, slide_num, target_labels)
    transcript_block = (
        "[현재 슬라이드 정보 텍스트]\n"
        + (slide_text if slide_text else "(없음)")
        + "\n\n[이전 슬라이드 마지막 문맥 2개"
        + (f": {prev_label}" if prev_slide_num > 0 else "")
        + f"]\n{prev_boundary}\n\n"
        + "[현재 슬라이드 전체 문맥]\n"
        + current_contexts
        + "\n\n[다음 슬라이드 첫 문맥 2개"
        + (f": {next_label}" if next_slide_num > 0 else "")
        + f"]\n{next_boundary}"
    )
    target_label = f"{title} ({time_range})"
    boundary_label = f"prev={prev_label}, next={next_label}"
    return target_label, boundary_label, transcript_block


def _crosscheck_context_text(target_label: str, boundary_label: str, transcript_block: str) -> str:
    return (
        f"대상 슬라이드: {target_label}\n"
        f"경계 문맥: {boundary_label}\n\n"
        f"현재 슬라이드 중심 문맥\n"
        f"{transcript_block}"
    )


def _issue_line_for_batch(issue_id: str, issue: dict, checked_type_code: str | None = None) -> str:
    from . import claim_common as cv

    context_id = issue.get("context_id") or issue.get("utterance_id", "")
    claim_text = str(issue.get("claim_text") or "").strip()
    source_text = str(
        issue.get("source_text")
        or issue.get("claim_context_text")
        or claim_text
        or ""
    ).strip()
    source_span_ids = ", ".join(
        str(x)
        for x in issue.get("source_span_ids", []) or []
    )
    anchor_ids = ", ".join(str(x) for x in issue.get("anchor_utterance_ids", []) or [])
    type_code = str(checked_type_code or _checked_type_code_for_issue(issue) or "").strip().upper()
    type_label = cv.issue_type_code_label(_issue_type_from_code(type_code)) if type_code else "unknown"
    lines = [
        f"### {issue_id}\n"
        f"- context_id: {context_id}\n"
        f"- checked_type: {type_code} {type_label}\n"
        f"- 원문 context text: {source_text}\n"
        f"- extractor claim_text(위치 표식, 원문보다 강하면 무시): {claim_text}"
    ]
    if anchor_ids:
        lines.append(f"- anchor_utterance_ids: {anchor_ids}")
    if source_span_ids:
        lines.append(f"- source_span_ids: {source_span_ids}")
    return "\n".join(lines)


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
    checked_type_code: str | None = None,
) -> tuple[str, str | None, str, set[str]]:
    from . import claim_common as cv

    utterances = ctx["utterances"]
    hint = ctx["hint"]
    utt_map = {u["utterance_id"]: (i, u) for i, u in enumerate(utterances)}

    target_labels: dict[int, list[str]] = {}
    for issue_id, issue in valid:
        related_ids = [issue.get("utterance_id", "")]
        related_ids.extend(issue.get("source_span_ids") or [])
        related_ids.extend(issue.get("anchor_utterance_ids") or [])
        related_ids.extend(issue.get("utterance_ids") or [])
        related_ids.extend(issue.get("canonical_member_utterance_ids") or [])
        for uid in dict.fromkeys(str(item or "") for item in related_ids):
            if uid not in utt_map:
                continue
            idx, _ = utt_map[uid]
            target_labels.setdefault(idx, []).append(issue_id)

    _, first_utt = utt_map[valid[0][1].get("utterance_id", "")]
    slide_num = int(first_utt.get("slide_number", 0) or 0)
    target_label, boundary_label, transcript_block = _crosscheck_slide_context(
        ctx,
        slide_num,
        target_labels,
    )
    context_text = _crosscheck_context_text(target_label, boundary_label, transcript_block)
    checked_type_code = str(checked_type_code or _checked_type_code_for_issue(valid[0][1]) or "A").strip().upper()
    if checked_type_code not in cv.ISSUE_CODE_TO_TYPE:
        checked_type_code = "A"
    checked_issue_type = cv.ISSUE_CODE_TO_TYPE[checked_type_code]
    checked_type_label = cv.issue_type_code_label(checked_issue_type)
    type_prompt_block = _type_specific_prompt_block(checked_type_code)
    issue_block = "\n\n".join(_issue_line_for_batch(issue_id, issue, checked_type_code) for issue_id, issue in valid)
    issue_ids = {issue_id for issue_id, _ in valid}

    prompt = f"""당신은 {checked_type_label} 전용 crosscheck checker입니다.
이전 단계가 표시한 issue 후보가 제공 문맥 안에서 **{checked_type_label} 기준으로만** 남는지 판단하세요.
issue 후보는 검토 위치를 찾기 위한 표식일 뿐이며, 맞는 지적이라고 가정하지 마세요.

하지 말아야 할 일:
- 새 issue 만들기
- 다른 유형으로 재분류하거나 확정하기
- extractor claim_text 또는 issue 후보 문장을 증거처럼 사용하기
- 표현 취향, 더 엄밀한 보충 가능성, 강의 수준 밖 예외만으로 issue 유지하기

## 도메인
{hint.get('label', '')}

## 대상 슬라이드
{target_label}

## 제공 문맥 범위
현재 슬라이드 정보 텍스트 + 현재 슬라이드 전체 문맥 + 이전 슬라이드 마지막 문맥 2개 + 다음 슬라이드 첫 문맥 2개

## 제공 문맥 사용 원칙
- 현재 슬라이드의 발화와 슬라이드 정보 텍스트를 가장 강한 근거로 보세요.
- 이전/다음 슬라이드 문맥은 같은 주제 흐름을 직접 이어받거나 같은 개념을 명시적으로 보완할 때만 해소 근거로 사용하세요.
- 주제가 바뀌었거나 단순히 주변에 등장한 설명이면 issue 후보를 해소한 것으로 보지 마세요.
- 슬라이드 정보 텍스트와 원문 발화가 충돌하면, 학생에게 동시에 노출된 전체 전달 효과를 기준으로 판단하세요.
- 슬라이드가 명확히 맞는 내용을 제시하고 발화의 부정확성을 바로잡는다면 일부 또는 전체 해소로 볼 수 있습니다.
- 반대로 슬라이드나 발화 중 하나가 학생에게 잘못된 명제를 독립적으로 남기면 issue를 유지할 수 있습니다.

## 제공 문맥
{transcript_block}

## 검토 issue 후보 목록
{issue_block}

## 판정 절차
각 issue_id마다 아래만 판단하세요.

1. `원문 context text`와 제공 문맥을 기준으로 실제 전달 내용을 먼저 재구성합니다.
2. extractor claim_text 또는 issue 후보가 원문 context보다 넓거나 강하면 그 강해진 문장은 버리고 원문 context 기준으로만 판단합니다.
3. 원문 context 안에 남는 실제 문제만 아래 {checked_type_label} gate로 평가해 type_gate_passed를 결정합니다.
4. 같은 슬라이드/같은 context의 설명, 표, 그림, 바로 이어지는 재표현이 의미·대상·관계·조건·범위를 실제로 바로잡으면 rejected로 둡니다.
5. 문맥 후에도 학생이 잘못 외울 구체 명제가 {checked_type_label} 기준으로 남으면 kept로 둡니다.
6. 같은 잘못된 명제를 가리키는 중복 후보만 merged로 둡니다. 같은 슬라이드라는 이유만으로 병합하지 마세요.

## Type-specific Crosscheck Gate
{type_prompt_block}

## Final issue_score
issue_score는 {checked_type_label} gate를 적용한 뒤, 이 issue 후보를 해당 유형의 강의자 검토 대상으로 남길 가치입니다.
서버는 모델별 issue_score에 모델 가중치를 곱해 최종 상태를 계산합니다.

점수 구간 기준:
- 0.00~0.39: rejected. 문맥상 해소되었거나 검토 가치가 낮음.
- 0.40~0.79: review/professor_check. 실제 문제가 남을 수 있으나 자동 확정하기는 어려움.
- 0.80~1.00: confirmed 후보. A형처럼 원문 오류가 명확하고 문맥에서도 해소되지 않음.

유형별 점수 원칙:
- A형은 명확한 원문 오류가 남으면 0.80 이상을 줄 수 있습니다.
- B형은 시점/외부 기준 확인이 필요하므로 보통 0.40~0.79에 둡니다.
- C형은 실제 배제 명제, 일반화 과잉, 조건 차이, 결과 과장이 구체적으로 남을 때 0.40 이상을 주세요. 자동 확정처럼 0.80 이상은 주지 마세요.
- D형은 구체적인 학생 오개념 문장이 남을 때만 0.40 이상을 주세요. 자동 확정처럼 0.80 이상은 주지 마세요.

type_gate_passed는 이 issue 후보가 {checked_type_label} 기준을 통과했는지 여부입니다.
다른 유형으로는 가능성이 있어 보여도 {checked_type_label} 기준을 통과하지 못하면 type_gate_passed=false, status=rejected, issue_score 0.39 이하로 두세요.

## context_resolution 기준
- "문맥에서 해소됨": 같은 문맥 안에서 의미, 대상, 관계, 조건, 범위가 명확히 바로잡혀 학생에게 잘못된 명제가 거의 남지 않는 경우
- "일부 해소됨": issue 후보가 원문보다 과장되었거나 문맥이 일부 보완했지만, 더 좁은 오개념이나 혼동 가능성이 남는 경우
- "해소 안 됨": 문맥이 문제를 바로잡지 못했거나 오히려 반복/강화하는 경우

## 응답 규칙
- JSON object 하나만 출력하고 markdown/code fence는 쓰지 마세요.
- 입력된 모든 issue_id에 대해 results 배열에 정확히 하나씩, 입력 순서대로 출력하세요.
- 새로운 issue_id를 만들지 마세요.
- issue_score는 반드시 출력하세요.
- checked_type_code는 "{checked_type_code}"로만 출력하세요.
- type_gate_passed는 반드시 true 또는 false로 출력하세요.
- status는 "kept", "rejected", "merged" 중 하나입니다.
- issue_type은 "{checked_issue_type}"로만 출력하세요.
- issue_type_code는 "{checked_type_code}"로만 출력하세요.
- reason은 {checked_type_label} gate를 통과하거나 통과하지 못한 이유와 문맥 해소 여부를 1~2문장으로 작성하세요.
- context_issue_summary는 kept인 경우 제공 문맥 안에 실제로 남는 문제를 1문장으로 작성하세요.
- rejected인 경우 context_issue_summary는 빈 문자열로 두세요.
- context_resolution은 "문맥에서 해소됨", "일부 해소됨", "해소 안 됨" 중 하나만 쓰세요.
- context_resolution_reason은 문맥이 왜 해소했거나 못 했는지 1문장입니다.
- correction_hint는 필요한 경우만 1문장으로 작성하고, 필요 없으면 빈 문자열로 두세요.
- context_issue_id는 문맥 전체에서 재구성한 실제 문제 단위의 짧은 ID입니다.
- kept인 경우 context_issue_id는 "ci001"처럼 쓰세요.
- rejected인 경우 context_issue_id는 "none"으로 쓰세요.
- merged인 경우 context_issue_id는 대표 issue와 같은 값을 쓰고, merged_into_issue_id에 대표 issue_id를 쓰세요.
- 대표이거나 독립 issue이면 merged_into_issue_id는 빈 문자열로 두세요.

응답 형식:
{{
  "results": [
    {{
      "issue_id": "i0001",
      "status": "kept | rejected | merged",
      "checked_type_code": "{checked_type_code}",
      "type_gate_passed": true,
      "issue_type": "{checked_issue_type}",
      "issue_type_code": "{checked_type_code}",
      "issue_score": 0.0,
      "reason": "",
      "context_issue_summary": "",
      "context_resolution": "문맥에서 해소됨 | 일부 해소됨 | 해소 안 됨",
      "context_resolution_reason": "",
      "correction_hint": "",
      "context_issue_id": "ci001",
      "merged_into_issue_id": ""
    }}
  ]
}}"""
    prompt, system_prompt = _split_crosscheck_prompt(prompt)
    return prompt, system_prompt, context_text, issue_ids


def _group_crosscheck_items_by_type(valid: list[tuple[str, dict]]) -> list[tuple[str, list[tuple[str, dict]]]]:
    groups: dict[str, list[tuple[str, dict]]] = {}
    for issue_id, issue in valid:
        for code in _checked_type_codes_for_issue(issue):
            groups.setdefault(code, []).append((issue_id, issue))
    return [(code, groups[code]) for code in ("A", "B", "C", "D") if groups.get(code)]


def _crosscheck_payload_score(payload: dict) -> float:
    score = _coerce_confidence(
        payload.get("confidence", payload.get("score", payload.get("issue_score"))),
        0.0,
    )
    return float(score or 0.0)


def _prefer_crosscheck_payload(current: dict | None, candidate: dict, primary_code: str) -> bool:
    if not current:
        return True
    candidate_score = _crosscheck_payload_score(candidate)
    current_score = _crosscheck_payload_score(current)
    if candidate_score > current_score + 0.0001:
        return True
    if current_score > candidate_score + 0.0001:
        return False
    candidate_code = str(candidate.get("checked_type_code") or candidate.get("issue_type_code") or "").strip().upper()
    current_code = str(current.get("checked_type_code") or current.get("issue_type_code") or "").strip().upper()
    return candidate_code == primary_code and current_code != primary_code


def _run_crosscheck_batch_prompt(
    valid: list[tuple[str, dict]],
    ctx: dict,
) -> tuple[dict[str, dict], dict, str, Exception | None]:
    from . import claim_common as cv

    if not valid:
        return {}, cv._empty_token_usage(), "", None

    groups = _group_crosscheck_items_by_type(valid)
    if len(groups) <= 1:
        checked_type_code = groups[0][0] if groups else _checked_type_code_for_issue(valid[0][1])
        return _run_crosscheck_batch_prompt_for_type(valid, ctx, checked_type_code)

    issues_by_id = {issue_id: issue for issue_id, issue in valid}
    parsed_all: dict[str, dict] = {}
    token_usage = cv._empty_token_usage()
    context_text = ""
    last_error = None
    for checked_type_code, group in groups:
        parsed, call_usage, call_context, call_error = _run_crosscheck_batch_prompt_for_type(
            group,
            ctx,
            checked_type_code,
        )
        cv._add_call_usage(token_usage, call_usage)
        for issue_id, payload in parsed.items():
            issue = issues_by_id.get(issue_id, {})
            primary_code = _checked_type_code_for_issue(issue)
            routed_codes = _checked_type_codes_for_issue(issue)
            payload["checked_type_candidates"] = routed_codes
            payload["selected_from_multi_type_route"] = len(routed_codes) > 1
            if _prefer_crosscheck_payload(parsed_all.get(issue_id), payload, primary_code):
                parsed_all[issue_id] = payload
        if call_context and not context_text:
            context_text = call_context
        if call_error is not None:
            last_error = call_error
    return parsed_all, token_usage, context_text, last_error


def _run_crosscheck_batch_prompt_for_type(
    valid: list[tuple[str, dict]],
    ctx: dict,
    checked_type_code: str,
) -> tuple[dict[str, dict], dict, str, Exception | None]:
    from . import claim_common as cv

    if not valid:
        return {}, cv._empty_token_usage(), "", None

    prompt, system_prompt, context_text, issue_ids = _build_crosscheck_batch_prompt(valid, ctx, checked_type_code)

    model = str(cv._resolve_stage_model("cross_recheck") or "").strip()
    response_format = {"type": "json_object"} if cv._supports_json_object_response_format(model) else None
    try:
        base_output_tokens = int(os.getenv("VERIFIER_CROSSCHECK_BASE_MAX_TOKENS", "1024"))
        tokens_per_issue = int(os.getenv("VERIFIER_CROSSCHECK_MAX_TOKENS_PER_ISSUE", "420"))
        output_token_cap = int(os.getenv("VERIFIER_CROSSCHECK_MAX_TOKENS", "3072"))
    except ValueError:
        base_output_tokens = 1024
        tokens_per_issue = 420
        output_token_cap = 3072
    max_tokens = min(output_token_cap, max(base_output_tokens, tokens_per_issue * len(valid)))
    if cv._is_anthropic_model(model):
        try:
            anthropic_tokens_per_issue = int(os.getenv("VERIFIER_ANTHROPIC_CROSSCHECK_MAX_TOKENS_PER_ISSUE", "1200"))
            anthropic_output_cap = int(os.getenv("VERIFIER_ANTHROPIC_CROSSCHECK_MAX_TOKENS", "12000"))
        except ValueError:
            anthropic_tokens_per_issue = 1200
            anthropic_output_cap = 12000
        max_tokens = min(
            anthropic_output_cap,
            max(max_tokens, base_output_tokens, anthropic_tokens_per_issue * len(valid)),
        )
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
            return parsed, token_usage, context_text, None
        except Exception as e:
            last_error = e
            if _is_permanent_crosscheck_api_failure(e):
                break
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
