from __future__ import annotations

import json
import os


JUDGE_MIN_CONFIDENCE = 0.80


def issue_judge_min_confidence() -> float:
    raw = str(os.getenv("VERIFIER_ISSUE_JUDGE_MIN_CONFIDENCE", str(JUDGE_MIN_CONFIDENCE)) or "").strip()
    try:
        value = float(raw)
    except ValueError:
        return JUDGE_MIN_CONFIDENCE
    return max(0.0, min(1.0, value))


def _claim_id(claim: dict) -> str:
    return str(
        claim.get("claim_id")
        or claim.get("claim_fingerprint")
        or claim.get("context_id")
        or ""
    ).strip()


def _context_id(claim: dict) -> str:
    return str(claim.get("context_id") or "").strip()


def _context_ids(claim: dict) -> list[str]:
    values = claim.get("context_ids")
    if isinstance(values, list):
        ids = [str(value).strip() for value in values if str(value).strip()]
        if ids:
            return ids
    cid = _context_id(claim)
    return [cid] if cid else []


def _build_shared_judge_context(contexts: list[dict], slide_ctx: dict) -> str:
    from . import claim_common as cv

    slide_numbers = []
    seen_slides = set()
    for u in contexts:
        slide_number = int(u.get("slide_number", 0) or 0)
        if slide_number and slide_number not in seen_slides:
            seen_slides.add(slide_number)
            slide_numbers.append(slide_number)

    slide_lines = []
    for slide_number in slide_numbers:
        slide = slide_ctx.get(slide_number, {})
        title = slide.get("title", f"슬라이드 {slide_number}")
        time_range = slide.get("time_range", "")
        slide_lines.append(f"- 슬라이드 {slide_number}: {title} ({time_range})")

    context_lines = [
        f"- {cv._format_context_for_prompt(u)}"
        for u in contexts
    ]

    return (
        "[배치 공통 슬라이드]\n"
        + ("\n".join(slide_lines) if slide_lines else "(슬라이드 정보 없음)")
        + "\n\n[배치 공통 context]\n"
        + ("\n".join(context_lines) if context_lines else "(context 없음)")
    )


def _build_issue_candidate_prompt(
    claims: list[dict],
    contexts: list[dict],
    current_date: str,
    hint: dict,
    slide_ctx: dict,
) -> str:
    shared_context = _build_shared_judge_context(contexts, slide_ctx)
    min_confidence = issue_judge_min_confidence()
    claim_lines = []
    for i, c in enumerate(claims, 1):
        approx = " [근사치]" if c.get("is_approximate") else ""
        claim_id = _claim_id(c) or f"claim_{i}"
        context_id = _context_id(c)
        resolved = str(c.get("resolved_claim") or c.get("claim_text") or "").strip()
        lines = [
            f"{i}. claim_id: {claim_id}",
            f"   context_id: {context_id}",
            f"   claim_type: {c.get('claim_type', '?')}{approx}",
            f"   판정 대상 resolved_claim: {resolved}",
            f"   원문 claim_text: {c.get('claim_text', '')}",
        ]
        if c.get("context_ids"):
            lines.append(f"   context_ids: {', '.join(str(x) for x in c.get('context_ids') or [])}")
        if c.get("antecedent_context_ids"):
            lines.append(
                f"   antecedent_context_ids: {', '.join(str(x) for x in c.get('antecedent_context_ids') or [])}"
            )
        if c.get("needs_context") or str(c.get("resolution_status") or "") == "unresolved":
            lines.append(f"   해소상태: {c.get('resolution_status') or 'unresolved'}, 문맥필요: true")
            if c.get("context_note"):
                lines.append(f"   문맥비고: {c.get('context_note', '')}")
        claim_lines.append("\n".join(lines))

    return f"""당신은 강의 claim 목록에서 1차 issue 후보만 선별하는 판정자입니다.
업로드/검증 기준일: {current_date}
강의 도메인: {hint['label']}

아래는 claim이 나온 강의 문맥과 claim 목록입니다.
이 단계는 후속 classified issue type/severity 단계 전의 1차 후보 추출 단계입니다. 최종 확정, 교수 피드백 작성, 정밀한 issue type 확정은 후속 단계에서 수행합니다.

배치 공통 문맥:
{shared_context}

판정 대상 claim 목록:
{chr(10).join(claim_lines)}

### 판정 기준

- 각 claim의 실제 판정 대상은 반드시 `resolved_claim`입니다.
- `claim_text`는 원문 확인용입니다. issue 후보를 만들 때는 `resolved_claim`이 학생에게 남기는 명제를 기준으로 판단하세요.
- 단, `resolved_claim`이 원문/문맥보다 과도하게 넓어진 것처럼 보이면 issue를 만들지 말고 보수적으로 제외하세요.
- 제공된 주변 문맥이 이미 정정하거나 조건/예외를 충분히 보완했다면 issue로 출력하지 마세요.
- 애매한 지시어를 특정 대상으로 강제 해석해야만 문제가 생기는 경우는 issue로 출력하지 마세요.
- temporal/currentness/outdated 판단은 강의 녹화 시점이 아니라 업로드/검증 기준일({current_date})을 기준으로 하세요.
- 강의자가 "현재", "요즘", "최신", "지원된다", "더 이상 사용하지 않는다"처럼 말한 경우, 학생이 업로드/검증 기준일에 그 정보를 현재 사실로 받아들일 수 있는지를 기준으로 판단하세요.

다음 질문 중 1개 이상에 "예"라고 답할 수 있으면 issue 후보로 출력하세요.
단, 제공된 문맥이 그 문제를 명확히 정정하거나 충분히 해소한 경우는 출력하지 마세요.

1. 학생이 resolved_claim을 그대로 외웠을 때 생길 수 있는 구체적인 잘못된 명제가 있는가?
2. 그 문제가 명확한 반례, 조건/범위 누락, 정의 차이, 현행성 확인 지점, 핵심 개념 혼동으로 설명될 수 있는가?
3. 학생이 업로드/검증 기준일 현재 이 claim을 그대로 받아들였을 때 outdated/currentness 문제가 생길 수 있는가?

### issue 후보로 볼 수 있는 보조 분류 기준

아래 분류는 판단 보조 기준입니다. 아래 분류에 하나라도 포함되는 claim은 issue 후보로 포함시키세요.
응답 JSON에는 type이나 issue_type을 저장하지 마세요.

- factual_error: 객관 사실, 정의, 순서, 메커니즘이 틀린 경우
- temporal_error: 업로드/검증 기준일({current_date}) 기준으로 현재성, 최신성, 지원 여부가 틀린 경우
- confusing_explanation: 학생이 핵심 개념, 주체, 과정, 권한, 대상을 혼동할 구체적 위험이 있는 경우
- scope_overclaim: 문맥을 함께 봐도 다른 가능성/조건/예외를 닫아버리는 과도한 단정이 남는 경우


### 출력하지 않는 경우

- 앞뒤 문맥이나 슬라이드가 자연스럽게 조건/대상을 보완하는 경우
- claim이 unresolved이고 선행사를 확정할 수 없어 문제 명제를 구체적으로 쓸 수 없는 경우

### 응답 (JSON만)

```json
{{
  "issues": [
    {{
      "claim_id": "CL0001",
      "resolved_claim": "판정에 사용한 resolved_claim",
      "claim_text": "원문 claim_text",
      "issue": "학생이 잘못 외울 수 있는 문제 요약",
      "candidate_reason": "왜 1차 issue 후보인지",
      "confidence": 0.0
    }}
  ]
}}
```

지침:
1. confidence {min_confidence:.2f} 미만은 출력하지 마세요.
2. 입력 claim_id에 없는 새 claim을 만들지 마세요.
3. type, issue_type, context_id는 출력하지 마세요.
4. 같은 claim에서 같은 문제는 한 건만 출력하세요.
5. 문제가 없으면 {{"issues": []}}만 출력하세요.
6. JSON 외 텍스트를 출력하지 마세요.
"""


def _order_issue_candidate_fields(issue: dict) -> dict:
    preferred = (
        "issue_id",
        "claim_id",
        "resolved_claim",
        "claim_text",
        "issue",
        "candidate_reason",
        "confidence",
        "context_id",
        "context_ids",
        "slide_number",
        "start_time",
        "end_time",
        "claim_type",
        "needs_context",
        "resolution_status",
    )
    ordered = {key: issue[key] for key in preferred if key in issue}
    ordered.update({key: value for key, value in issue.items() if key not in ordered})
    return ordered


def _judge_issue_candidates(
    claims: list[dict],
    contexts: list[dict],
    current_date: str,
    hint: dict,
    context_map: dict,
    slide_ctx: dict,
) -> tuple[list[dict], bool, int, dict]:
    from . import claim_common as cv

    if not claims:
        return [], False, 0, cv._empty_token_usage()

    claim_by_id = {_claim_id(claim): claim for claim in claims if _claim_id(claim)}
    full_prompt = _build_issue_candidate_prompt(claims, contexts, current_date, hint, slide_ctx)
    system_prompt, prompt = _split_judge_prompt_for_cache(full_prompt)
    response_format = (
        {"type": "json_object"}
        if cv._supports_json_object_response_format(cv._resolve_stage_model("judge"))
        else None
    )
    api_calls = 0
    token_usage = cv._empty_token_usage()

    for attempt in range(cv.VERIFIER_PARSE_RETRIES + 1):
        text, call_usage = cv._call_llm(
            prompt,
            system_prompt=system_prompt,
            max_tokens=8192,
            temperature=0.0,
            thinking_budget=2048,
            response_format=response_format,
            stage="judge",
        )
        api_calls += 1
        cv._add_call_usage(token_usage, call_usage)
        try:
            payload = json.loads(cv._strip_json_fence(text.strip()))
        except json.JSONDecodeError:
            if attempt < cv.VERIFIER_PARSE_RETRIES:
                print(f"    ↺ 1차 issue judge JSON 파싱 재시도 ({attempt+1}/{cv.VERIFIER_PARSE_RETRIES})")
                continue
            return [], True, api_calls, token_usage

        raw = payload.get("issues", [])
        if not isinstance(raw, list):
            return [], False, api_calls, token_usage

        issues = []
        seen = set()
        for raw_issue in raw:
            if not isinstance(raw_issue, dict):
                continue
            claim_id = str(raw_issue.get("claim_id", "") or "").strip()
            source_claim = claim_by_id.get(claim_id)
            if not source_claim:
                continue
            try:
                confidence = float(raw_issue.get("confidence", 0) or 0)
            except Exception:
                confidence = 0.0
            if confidence < issue_judge_min_confidence():
                continue

            context_id = _context_id(source_claim)
            ref = context_map.get(context_id, {})
            issue_key = (
                claim_id,
                cv._compact_text(str(raw_issue.get("issue", "") or ""))[:120],
            )
            if issue_key in seen:
                continue
            seen.add(issue_key)

            resolved_claim = str(
                raw_issue.get("resolved_claim")
                or source_claim.get("resolved_claim")
                or source_claim.get("claim_text")
                or ""
            ).strip()
            claim_text = str(raw_issue.get("claim_text") or source_claim.get("claim_text") or "").strip()
            issue = {
                "claim_id": claim_id,
                "resolved_claim": resolved_claim,
                "claim_text": claim_text,
                "issue": str(raw_issue.get("issue", "") or raw_issue.get("candidate_reason", "") or "").strip(),
                "candidate_reason": str(
                    raw_issue.get("candidate_reason")
                    or raw_issue.get("issue")
                    or raw_issue.get("explanation")
                    or ""
                ).strip(),
                "confidence": confidence,
                "context_id": context_id,
                "context_ids": _context_ids(source_claim),
                "slide_number": ref.get("slide_number", source_claim.get("slide_number")),
                "start_time": ref.get("start_time", source_claim.get("start_time")),
                "end_time": ref.get("end_time", source_claim.get("end_time")),
                "claim_type": source_claim.get("claim_type", ""),
                "needs_context": bool(source_claim.get("needs_context")),
                "resolution_status": source_claim.get("resolution_status", ""),
            }
            issues.append(_order_issue_candidate_fields(issue))

        return issues, False, api_calls, token_usage

    return [], True, api_calls, token_usage


def recover_issue_candidate_judgement(
    claims: list[dict],
    contexts: list[dict],
    current_date: str,
    hint: dict,
    context_map: dict,
    slide_ctx: dict,
    label: str,
) -> tuple[list[dict], bool, int, dict, bool]:
    from . import claim_common as cv

    total_api_calls = 0
    total_token_usage = cv._empty_token_usage()
    last_issues: list[dict] = []
    parse_failed = False
    last_exc = None

    for attempt in range(cv.VERIFIER_BATCH_RECOVERY_RETRIES + 1):
        if attempt > 0:
            print(f"    ↺ {label} 1차 issue judge 재처리 ({attempt}/{cv.VERIFIER_BATCH_RECOVERY_RETRIES})")
        try:
            issues, parse_failed, api_calls, token_usage = _judge_issue_candidates(
                claims, contexts, current_date, hint, context_map, slide_ctx
            )
            total_api_calls += api_calls
            total_token_usage = cv._merge_token_usage(total_token_usage, token_usage)
            last_issues = issues
            if not parse_failed:
                return issues, False, total_api_calls, total_token_usage, True
        except Exception as e:
            last_exc = e

    if last_exc and not last_issues and total_api_calls == 0:
        raise last_exc
    return last_issues, True, total_api_calls, total_token_usage, False


def judge_issue_candidates_only(
    all_claims_by_batch: list[tuple],
    current_date: str,
    hint: dict,
    slide_ctx: dict,
    log_prefix: str = "",
) -> tuple[list[dict], int, dict]:
    """Run only the first issue judge and return issue candidates."""
    from . import claim_common as cv

    total_api = 0
    total_token_usage = cv._empty_token_usage()
    all_issues = []
    prefix = f"[{log_prefix}] " if log_prefix else ""
    total_batches = sum(1 for _, claims in all_claims_by_batch if claims)
    active_batch_idx = 0

    for batch_idx, (batch, claims) in enumerate(all_claims_by_batch, start=1):
        if not claims:
            continue
        active_batch_idx += 1
        batch_map = {u["context_id"]: u for u in batch}
        ids = f"{batch[0]['context_id']}..{batch[-1]['context_id']}"
        print(f"  {prefix}1차 issue judge ({active_batch_idx}/{total_batches}) {ids}", flush=True)
        issues, parse_failed, api_calls, token_usage, ok = recover_issue_candidate_judgement(
            claims,
            batch,
            current_date,
            hint,
            batch_map,
            slide_ctx,
            f"1차 issue judge 배치 {batch_idx} {ids}",
        )
        if not ok:
            print(f"    ⚠️ 1차 issue judge 실패: {ids} — 이 batch는 빈 결과로 기록됩니다.")
        all_issues.extend(issues)
        total_api += api_calls
        total_token_usage = cv._merge_token_usage(total_token_usage, token_usage)

    for index, issue in enumerate(all_issues, start=1):
        issue["issue_id"] = f"I{index:04d}"
        ordered = _order_issue_candidate_fields(issue)
        issue.clear()
        issue.update(ordered)

    return all_issues, total_api, total_token_usage


def _split_judge_prompt_for_cache(prompt: str) -> tuple[str | None, str]:
    dynamic_marker = "배치 공통 문맥:"
    instruction_marker = "### 판정 기준"
    dynamic_start = prompt.find(dynamic_marker)
    instruction_start = prompt.find(instruction_marker)
    if dynamic_start < 0 or instruction_start < 0 or dynamic_start >= instruction_start:
        return None, prompt
    system_prompt = prompt[:dynamic_start].rstrip() + "\n\n" + prompt[instruction_start:].lstrip()
    user_prompt = prompt[dynamic_start:instruction_start].strip()
    return system_prompt, user_prompt
