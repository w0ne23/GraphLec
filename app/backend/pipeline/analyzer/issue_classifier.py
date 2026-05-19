"""Issue candidate classification stage.

This stage intentionally runs after issue detection/merge and before
type-specific crosscheck. It does not decide whether an issue is valid; it only
chooses which A-D verifier prompt should inspect the candidate next.
"""

from __future__ import annotations

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from . import claim_common as cv


# Tie-break/order follows specificity: temporal and scope issues should be
# considered before broad factual/confusing labels.
_TYPE_CODES = ("B", "C", "A", "D")


def _clamp_score(value, default: float = 0.0) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        score = default
    if 1.0 < score <= 100.0:
        score /= 100.0
    return max(0.0, min(1.0, score))


def _env_int(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(name, str(default)) or default))
    except ValueError:
        return default


def _split_models(value: str | None) -> list[str]:
    return [item for item in re.split(r"[\s,]+", str(value or "").strip()) if item]


def _primary_from_scores(scores: dict[str, float]) -> tuple[str, str]:
    primary_code = max(_TYPE_CODES, key=lambda code: scores.get(code, 0.0))
    if scores.get(primary_code, 0.0) <= 0:
        return "", ""
    return primary_code, cv.ISSUE_CODE_TO_TYPE.get(primary_code, "")


def _normalize_type_scores(raw_scores) -> tuple[dict[str, float], str, str]:
    scores = {code: 0.0 for code in _TYPE_CODES}
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

    primary_code, primary_issue_type = _primary_from_scores(scores)
    return scores, primary_code, primary_issue_type


def _issue_text(issue: dict, key: str, limit: int = 360) -> str:
    text = " ".join(str(issue.get(key, "") or "").split()).strip()
    return text[:limit]


def _source_issue_lines(issue: dict, limit: int = 4) -> str:
    source_issues = issue.get("source_issues")
    if not isinstance(source_issues, list) or not source_issues:
        return ""
    lines = []
    for source in source_issues[:limit]:
        if not isinstance(source, dict):
            continue
        uid = str(source.get("utterance_id", "") or "").strip()
        source_text = " ".join(
            str(
                source.get("source_text")
                or source.get("claim_context_text")
                or source.get("claim_text")
                or ""
            ).split()
        ).strip()
        claim = " ".join(str(source.get("claim_text", "") or "").split()).strip()
        body = source_text or claim
        if not body:
            continue
        prefix = f"{uid}: " if uid else ""
        lines.append(f"   - {prefix}{body[:220]}")
    if not lines:
        return ""
    return "\n".join(lines)


def _build_prompt(batch: list[tuple[str, dict]], hint: dict) -> str:
    issue_lines = []
    for issue_id, issue in batch:
        source_lines = _source_issue_lines(issue)
        parts = [
            f"{issue_id}.",
            f"   context_id: {issue.get('context_id') or issue.get('utterance_id') or ''}",
            f"   slide: {issue.get('slide_number', '')}",
            f"   source_text: {_issue_text(issue, 'source_text') or _issue_text(issue, 'claim_context_text')}",
            f"   claim_text: {_issue_text(issue, 'claim_text')}",
        ]
        if source_lines:
            parts.append("   source_issues:\n" + source_lines)
        issue_lines.append("\n".join(parts))

    return f"""당신은 강의 검증 파이프라인의 Issue_classification 단계입니다.
이전 Issue_detection 단계가 이미 crosscheck 후보를 골랐습니다.
이 단계에서는 후보를 유지하거나 기각하지 말고, 각 후보의 원문 문맥(source_text)이 A-D 중 어느 검증 유형에 해당하는지만 채점하세요.

강의 도메인: {hint.get('label', '')}

판정 대상 issue 후보:
{chr(10).join(issue_lines)}

분류 기준:
- B temporal_error: 현재성, 최신성, 지원 여부, 사용 여부, 버전, 통계, 시장 상태처럼 강의 제공 시점이나 오늘 기준 확인이 필요한 가능성.
  특히 상대 시점 표현으로 기술 동작, 개발 관행, 지원 여부, 사용 여부를 일반 사실처럼 말하면 B를 높게 채점하세요.
- C scope_overclaim: 조건, 예외, 범위, 배제, 일반화, 전체/일부, 가능/불가능, 일시/영구, 직접/간접이 과하게 닫혀 말해졌을 가능성.
- A factual_error: 정의, 분류, 포함 관계, 수치, 순서, 주체, 과정, 원인-결과, 작동 방식, 귀속 관계 자체가 틀렸을 가능성.
  단, 핵심 정의는 맞지만 설명 중 전환 조건, 선후 단계, 중간 과정이 생략되어 오해가 생기는 경우는 A보다 D에 가깝습니다.
- D confusing_explanation: 사실 오류로 바로 단정하기보다 설명 흐름 때문에 학생이 구체적 오개념을 만들 가능성.
  특히 하나의 정의나 과정 설명에서 중간 단계, 전환 조건, 상태 변화를 건너뛰어 학생이 두 개념을 같은 것으로 외울 수 있으면 D로 분류하세요.

채점 원칙:
1. A-D 점수는 각각 0.0~1.0으로 독립 채점하세요. 합을 1로 맞추지 마세요.
2. primary_type_code는 가장 높은 점수의 코드입니다. 동점이면 B, C, A, D 순서로 고르세요.
3. 상대 시점 표현이 핵심 근거이면, 단순 사실 오류처럼 보여도 B를 우선 높게 주세요.
4. 전체/일부, 가능/불가능, 일시/영구, 직접/간접의 범위가 뒤바뀐 것이 핵심이면 C를 우선 높게 주세요.
5. A와 D가 모두 가능해 보이면, 원문이 틀린 정의를 직접 확정하는지 먼저 보세요.
   직접 확정하면 A, 핵심 결론은 맞을 수 있으나 생략/압축/순서 흐름 때문에 오해가 생기는 정도라면 D를 더 높게 주세요.
6. claim_text가 source_text보다 넓거나 강하면 source_text를 우선하세요.
7. 이 단계는 최종 판정이 아닙니다. 반례 설명, 수정안, 기각 사유는 쓰지 마세요.
8. primary_type_reason은 primary_type_code를 고른 핵심 이유만 1문장으로 쓰세요.
   다른 유형에 대한 이유는 쓰지 말고, 다른 유형 판단은 type_scores confidence로만 표현하세요.
9. issue_id는 입력 값을 그대로 복사하세요.

응답은 JSON object 하나만 출력하세요.
{{
  "classifications": [
    {{
      "issue_id": "ic0001",
      "type_scores": {{"B": 0.0, "C": 0.0, "A": 0.0, "D": 0.0}},
      "primary_type_code": "B",
      "primary_type_reason": "상대 시점 표현이 개발 관행의 현재성 판단을 요구하므로 B가 가장 가깝습니다."
    }}
  ]
}}
"""


def _parse_payload(text: str, issue_ids: set[str]) -> dict[str, tuple[dict[str, float], str, str, str]]:
    cleaned = cv._strip_json_fence((text or "").strip())
    candidates = [cleaned]
    first_obj = cv._extract_first_json_object(cleaned)
    if first_obj and first_obj not in candidates:
        candidates.append(first_obj)

    for candidate in candidates:
        if not candidate:
            continue
        fixed = candidate.replace("```json", "").replace("```", "")
        fixed = re.sub(r",\s*([}\]])", r"\1", fixed)
        try:
            payload = json.loads(fixed)
        except json.JSONDecodeError:
            continue
        rows = payload.get("classifications") if isinstance(payload, dict) else payload
        if not isinstance(rows, list):
            continue
        parsed: dict[str, tuple[dict[str, float], str, str]] = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            issue_id = str(row.get("issue_id", "") or "").strip()
            if issue_id not in issue_ids:
                continue
            raw_scores = (
                row.get("type_scores")
                or row.get("candidate_type_scores")
                or row.get("issue_type_scores")
                or {}
            )
            scores, primary_code, primary_type = _normalize_type_scores(raw_scores)
            explicit = str(row.get("primary_type_code") or row.get("issue_type_code") or "").strip().upper()
            if explicit in _TYPE_CODES:
                primary_code = explicit
                primary_type = cv.ISSUE_CODE_TO_TYPE.get(primary_code, "")
            reason = " ".join(
                str(
                    row.get("primary_type_reason")
                    or row.get("type_reason")
                    or row.get("rationale")
                    or ""
                ).split()
            ).strip()
            parsed[issue_id] = (scores, primary_code, primary_type, reason)
        if parsed:
            return parsed
    raise ValueError("issue_classification_response_parse_failed")


def _classify_batch(
    batch: list[tuple[str, dict]],
    hint: dict,
    model: str | None = None,
) -> tuple[dict[str, tuple[dict[str, float], str, str, str]], dict, bool]:
    model = str(model or cv._resolve_stage_model("issue_classification") or "").strip()
    response_format = {"type": "json_object"} if cv._supports_json_object_response_format(model) else None
    prompt = _build_prompt(batch, hint)
    token_usage = cv._empty_token_usage()
    issue_ids = {issue_id for issue_id, _ in batch}
    for attempt in range(cv.VERIFIER_PARSE_RETRIES + 1):
        text, call_usage = cv._call_llm(
            prompt,
            max_tokens=4096,
            temperature=0.0,
            thinking_budget=1024,
            response_format=response_format,
            stage="issue_classification",
            model_spec_override=model,
        )
        cv._add_call_usage(token_usage, call_usage)
        try:
            return _parse_payload(text, issue_ids), token_usage, False
        except Exception:
            if attempt < cv.VERIFIER_PARSE_RETRIES:
                print(f"    ↺ issue 유형 분류 JSON 파싱 재시도 ({attempt+1}/{cv.VERIFIER_PARSE_RETRIES})")
                continue
            return {}, token_usage, True
    return {}, token_usage, True


def _fallback_classification(issue: dict) -> tuple[dict[str, float], str, str, str]:
    code = str(
        issue.get("candidate_primary_type_code")
        or issue.get("issue_type_code")
        or cv.issue_type_code(str(issue.get("issue_type") or issue.get("type") or ""))
        or ""
    ).strip().upper()
    if code not in _TYPE_CODES:
        code = "A"
    scores = {item: 0.0 for item in _TYPE_CODES}
    scores[code] = 0.5
    return scores, code, cv.ISSUE_CODE_TO_TYPE.get(code, "factual_error"), ""


def classify_issue_candidates(
    issues: list[dict],
    hint: dict,
    *,
    models: list[str] | None = None,
    batch_size: int | None = None,
    max_workers: int | None = None,
) -> tuple[list[dict], dict]:
    """Attach A-D classification metadata to merged issue candidates."""
    if not issues:
        return [], cv._empty_token_usage()

    env_models = _split_models(os.getenv("VERIFIER_ISSUE_CLASSIFIER_MODELS", ""))
    classifier_models = [
        str(model or "").strip()
        for model in (models or env_models or [cv._resolve_stage_model("issue_classification")])
        if str(model or "").strip()
    ]
    classifier_models = list(dict.fromkeys(classifier_models))
    if not classifier_models:
        classifier_models = [cv._resolve_stage_model("issue_classification")]

    configured_batch_size = batch_size or _env_int("VERIFIER_ISSUE_CLASSIFIER_BATCH_SIZE", 20)
    configured_workers = max_workers or _env_int("VERIFIER_ISSUE_CLASSIFIER_MAX_WORKERS", 2)
    indexed = [(f"ic{i:04d}", issue) for i, issue in enumerate(issues, 1)]
    batches = [
        indexed[i : i + configured_batch_size]
        for i in range(0, len(indexed), configured_batch_size)
    ]
    token_usage = cv._empty_token_usage()
    parsed_by_model: dict[str, dict[str, tuple[dict[str, float], str, str, str]]] = {
        model: {} for model in classifier_models
    }
    failed_batches = 0

    jobs = [
        (model, batch_idx, batch)
        for model in classifier_models
        for batch_idx, batch in enumerate(batches, 1)
    ]

    def _run_job(job):
        model, batch_idx, batch = job
        parsed, usage, failed = _classify_batch(batch, hint, model=model)
        return model, batch_idx, batch, parsed, usage, failed

    with ThreadPoolExecutor(max_workers=max(1, min(configured_workers, len(jobs)))) as executor:
        futures = {executor.submit(_run_job, job): job for job in jobs}
        for idx, future in enumerate(as_completed(futures), 1):
            model, batch_idx, batch = futures[future]
            print(
                f"  [{model}] issue 유형 분류 ({batch_idx}/{len(batches)}): {len(batch)}건",
                flush=True,
            )
            try:
                model, _batch_idx, _batch, parsed, usage, failed = future.result()
            except Exception as exc:
                print(f"    ⚠️ [{model}] issue 유형 분류 실패: {exc}")
                parsed, usage, failed = {}, cv._empty_token_usage(), True
            token_usage = cv._merge_token_usage(token_usage, usage)
            parsed_by_model.setdefault(model, {}).update(parsed)
            if failed:
                failed_batches += 1

    output: list[dict] = []
    for issue_id, issue in indexed:
        row = dict(issue)
        model_rows = {
            model: parsed[issue_id]
            for model, parsed in parsed_by_model.items()
            if issue_id in parsed
        }
        if model_rows:
            combined_scores = {}
            for code in _TYPE_CODES:
                combined_scores[code] = sum(
                    scores.get(code, 0.0)
                    for scores, _primary_code, _primary_type, _reason in model_rows.values()
                ) / len(model_rows)
            primary_code, primary_type = _primary_from_scores(combined_scores)
            scores = combined_scores
            reason = ""
            for _model, (_scores, model_primary_code, _model_primary_type, model_reason) in model_rows.items():
                if model_primary_code == primary_code and model_reason:
                    reason = model_reason
                    break
            if not reason:
                for _model, (_scores, _model_primary_code, _model_primary_type, model_reason) in model_rows.items():
                    if model_reason:
                        reason = model_reason
                        break
            row["issue_classification_by_model"] = {
                model: {
                    "type_scores": model_scores,
                    "primary_type_code": model_primary_code,
                    "primary_type": model_primary_type,
                    "primary_type_reason": model_reason,
                }
                for model, (model_scores, model_primary_code, model_primary_type, model_reason) in model_rows.items()
            }
        else:
            scores, primary_code, primary_type, reason = _fallback_classification(row)
            row["issue_classification_by_model"] = {}
        row["issue_classification_scores"] = scores
        row["issue_classification_primary_code"] = primary_code
        row["issue_classification_primary_type"] = primary_type
        row["issue_classification_rationale"] = reason
        if reason and not row.get("issue_type_rationale"):
            row["issue_type_rationale"] = reason
        row["candidate_type_scores"] = scores
        row["candidate_primary_type_code"] = primary_code
        row["candidate_primary_issue_type"] = primary_type
        output.append(row)

    if failed_batches:
        print(f"  ⚠️ issue 유형 분류 fallback 적용: {failed_batches}개 batch")
    return output, token_usage
