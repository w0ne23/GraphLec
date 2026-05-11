from __future__ import annotations

import json

JUDGE_MIN_CONFIDENCE = 0.80


def _build_verification_question(claim: dict) -> str:
    """후속 판정에서 살아남은 claim에만 짧은 검증 질문을 생성."""
    text = str(
        claim.get("resolved_claim")
        or claim.get("claim_text")
        or claim.get("problematic_content")
        or ""
    ).strip()
    if not text:
        return ""

    text = " ".join(text.split()).strip(" \t\r\n.。?？!！")
    if not text:
        return ""

    claim_type = str(claim.get("claim_type", "") or "").strip()
    if claim_type == "numeric":
        return f"'{text}'라는 수치나 기준이 정확한가?"
    if claim_type == "causal":
        return f"'{text}'라는 인과 또는 작동 방식 설명이 타당한가?"
    if claim_type == "relationship":
        return f"'{text}'라는 개념 간 관계 설명이 타당한가?"
    if claim_type == "currentness":
        return f"'{text}'라는 현행성 설명이 현재 기준으로 타당한가?"
    return f"'{text}'라는 설명이 타당한가?"


def _issue_type_definitions_for_prompt() -> str:
    return """- A. factual_error (발언 자체 오류): 발화 자체의 객관 사실, 정의, 분류, 수치, 순서, 원인-결과, 작동 방식이 강의 문맥을 함께 봐도 틀린 경우
  포함: 잘못된 명제, 반례, 정답과의 직접 충돌, 주체/과정/대상의 명확한 혼동
  제외: 표현이 조금 부정확하지만 학생이 최종적으로 맞는 개념을 가져가는 경우
- B. temporal_error (시간적 오류): 현재성, 최신성, 지원 여부, 사용 여부, 시점 의존 수치나 상태를 현재 사실처럼 말했지만 기준 시점에서 틀리거나 확인이 필요한 경우
  포함: 현재/요즘/최근/최신/지원 종료/시장 상태/현행 제도/시점 의존 통계
  제외: 녹화 시점이나 역사적 관점 설명으로 자연스럽게 해석되는 경우
- C. scope_overclaim (범위 과잉 단정): 특정 조건에서는 맞지만 모든 경우에 맞는 것처럼 범위, 조건, 예외, 다른 가능성을 닫아 말한 경우
  포함: 항상/모든/반드시/오직/~만/유일 같은 닫힌 명제가 문맥 후에도 남고, 강의 수준에서 의미 있는 반례나 조건이 있는 경우
  제외: 역할, 책임, 대표 경로, 일반적 관례를 강조한 표준적 설명일 뿐 다른 가능성을 실제로 배제하지 않는 경우
- D. confusing_explanation (혼동 가능 설명): 명백한 사실 오류라고 단정되지는 않더라도 학생이 핵심 개념, 주체, 과정, 원인, 조건을 잘못 연결해 외울 가능성이 큰 경우
  포함: 서로 다른 개념 동일시, 주체/과정 혼동, 순서 혼동, 설명 흐름 때문에 남는 구체적 오답 명제
  제외: 단순 비유 취향, 더 자세히 설명 가능함, 막연한 오해 가능성, 지시어를 과하게 확정해야만 생기는 문제"""


def _format_utterance_for_prompt(u: dict) -> str:
    uid = u["utterance_id"]
    ts = f"{u['start_time']:.1f}s"
    corr = str(u.get("text_corrected", "") or "").strip()
    orig = str(u.get("text_original", "") or "").strip()

    if str(u.get("correction_status", "") or "").strip() == "candidate_only":
        return f"{uid} | {ts} | {orig or u.get('text', '')}"

    if corr and orig and corr != orig:
        return f"{uid} | {ts} | 교정: {corr} | 원문: {orig}"

    return f"{uid} | {ts} | {u['text']}"


def _normalize_judge_context_mode(value: str | None = None) -> str:
    from . import claim_common as cv

    return cv.normalize_judge_context_mode(value)


def _build_shared_judge_context(utterances: list[dict], context_mode: str = "batch") -> str:
    utterance_lines = [
        f"- {_format_utterance_for_prompt(u)}"
        for u in utterances
    ]

    return (
        "[배치 공통 발화]\n"
        + ("\n".join(utterance_lines) if utterance_lines else "(발화 없음)")
    )


def _build_judge_prompt(
    claims: list[dict],
    utterances: list[dict],
    current_date: str,
    hint: dict,
    context_mode: str | None = None,
) -> str:
    from . import claim_common as cv

    context_mode = _normalize_judge_context_mode(context_mode)
    shared_context = _build_shared_judge_context(utterances, context_mode)
    issue_type_definitions = _issue_type_definitions_for_prompt()
    context_intro = (
        "아래는 강의 발화에서 추출된 사실 주장(claim) 목록입니다.\n"
        "이 단계에는 빠른 1차 판정을 위한 배치 공통 발화 문맥만 제공됩니다.\n"
        "후속 crosscheck 단계에서 슬라이드 텍스트, 이전/현재 슬라이드 발화, 대상 발화 전후 문맥을 더 넓게 다시 확인합니다.\n"
        "claim은 `utterance_id`로 공통 발화 문맥의 해당 발화를 참조하세요."
    )
    criterion_3 = "3. 배치 발화 문맥만으로 문제가 명확히 해소되지 않아, 후속 crosscheck에서 다시 확인할 가치가 있는가?"
    context_principle = """0. **후보 선별 역할**
   이 단계는 최종 판정자가 아니라 crosscheck 후보 선별자입니다.
   제공된 배치 발화 문맥만으로 명확히 정정되거나 조건/예외가 충분히 보완된 후보만 제외하세요.
   지시어/생략 표현을 특정 대상으로 과하게 확정해야만 생기는 문제는 제외하세요.
   반대로 원문 발화 자체에 구체적인 잘못된 명제가 남고 문맥 해소 여부가 애매하면 후보로 올리세요.
   슬라이드 문맥이나 더 넓은 흐름이 필요한 판단은 후속 crosscheck에서 다시 검증합니다."""
    claim_lines = []
    for i, c in enumerate(claims, 1):
        approx = " [근사치]" if c.get("is_approximate") else ""
        resolved = c.get("resolved_claim", c.get("claim_text", ""))
        needs_context = bool(c.get("needs_context"))
        resolution_status = str(c.get("resolution_status") or ("unresolved" if needs_context else "resolved"))
        context_note = str(c.get("context_note") or "").strip()
        lines = [
            f"{i}. [{c['utterance_id']}] ({c.get('claim_type', '?')}){approx}",
            f"   원문: {c.get('claim_text', '')}",
            f"   해소: {resolved}",
        ]
        if c.get("utterance_ids"):
            lines.append(f"   관련발화: {', '.join(str(x) for x in c.get('utterance_ids') or [])}")
        if needs_context or resolution_status == "unresolved":
            lines.append(f"   해소상태: {resolution_status}, 문맥필요: true")
            if context_note:
                lines.append(f"   문맥비고: {context_note}")
        claim_lines.append("\n".join(lines))

    return f"""당신은 강의 발화에서 crosscheck에 올릴 문제 후보만 선별하는 판정자입니다.
오늘 날짜: {current_date}
강의 도메인: {hint['label']}

{context_intro}

배치 공통 문맥:
{shared_context}

판정 대상 claim 목록:
{chr(10).join(claim_lines)}

### 판정 기준

이 단계의 목적은 최종 교수 피드백 작성이 아니라, **검증할 가치가 있는 후보만 좁히는 것**입니다.
각 claim에 대해 아래 3개 질문에 모두 "예"라고 답할 수 있을 때만 이슈 후보로 출력하세요.

1. 학생이 이 발화를 그대로 외웠을 때 생길 수 있는 **구체적인 잘못된 명제**를 한 문장으로 쓸 수 있는가?
2. 그 명제가 틀리거나 위험한 이유가 단순 취향이 아니라 **명확한 반례, 조건, 범위, 정의 차이, 현행성 확인 대상**으로 설명되는가?
{criterion_3}

하나라도 아니오이면 출력하지 마세요.
막연히 "오해할 수도 있다", "더 엄밀하게 말할 수 있다", "보충하면 좋다"는 이유만으로는 후보가 아닙니다.
문맥이 애매하거나 같은 문제 표현을 반복/강화한다면 후보로 올리고 crosscheck에 맡기세요.

### 공통 issue type 정의

아래 4개 유형만 사용하세요. 같은 정의와 A-D 코드는 crosscheck에서도 그대로 사용됩니다.
{issue_type_definitions}

### 판단 원칙

{context_principle}

1. **명시 오류 우선**
   방향/극성 반전, 부정어 누락/추가, 범주 오귀속, 수치/비율/연도 오류, 서로 다른 개념 동일시는 후보로 올리세요.

2. **현행성**
   "최근/요즘/현재/최신/지원 여부"처럼 시점 의존 표현이 현재 사실처럼 쓰였고 확인이 필요하면 후보로 올리세요.
   단순한 역사 설명이나 녹화 시점 기준 설명으로 자연스럽게 읽히면 제외하세요.

3. **현실 대상의 예시 수치**
   실재 대상의 구체 수치/비율/연도/규모는 예시 문맥이어도 후보가 될 수 있습니다.
   명시적으로 가상값, 임의값, 변수 예시라고 밝힌 경우만 제외하세요.

4. **범위 과잉 단정**
   "모든/항상/반드시/유일/오직/~만/독점" 같은 표현은 단어 하나만으로 후보가 아닙니다.
   배치 문맥 후에도 다른 가능성, 다른 주체, 조건 차이를 실제로 닫는 명제가 남을 때만 후보로 올리세요.

5. **혼동 가능 설명**
   막연한 오해 가능성은 후보가 아닙니다.
   학생이 서로 다른 개념, 주체, 과정, 원인, 조건을 잘못 연결해 외울 구체적인 오답 명제가 있어야 합니다.

6. **제외**
   표현 취향, 더 엄밀한 보충 가능성, 강의 수준 밖 고급 예외, 반올림 차이, 표기 차이,
   대표 예시/개론 설명, 문맥에서 명확히 정정된 claim은 출력하지 마세요.

### 응답 (JSON만)

```json
{{
	"issues": [
    {{
	      "utterance_id": "U0001",
	      "type": "factual_error" | "temporal_error" | "scope_overclaim" | "confusing_explanation",
	      "issue_type_code": "A" | "B" | "C" | "D",
	      "claim_text": "판정한 claim 원문",
	      "problematic_content": "문제 발화 원문 (80자 이내)",
	      "issue": "학생이 잘못 외울 수 있는 명제와 후보 사유를 한 문장으로 작성",
	      "candidate_reason": "명확한 반례/조건/범위/정의 차이/현행성 확인 포인트를 짧게 작성",
	      "candidate_strength": 0.0-1.0
	    }}
	  ]
	}}
```

지침:
1. candidate_strength는 최종 확률이 아니라 crosscheck에 넘길 후보 강도입니다.
2. candidate_strength {JUDGE_MIN_CONFIDENCE:.2f} 미만은 출력하지 마세요.
3. 동일 개념 문제는 한 건만.
4. 단순히 "더 자세히 설명하면 좋겠다" 수준이면 출력하지 마세요.
5. 문제가 없으면 {{"issues": []}}만.
6. JSON 외 텍스트 금지.
7. 이 단계에서 교수 피드백 문장, 대체 표현, 반례 상세 설명, 최종 문맥 해소 여부를 작성하지 마세요.
   그런 필드는 crosscheck 이후 최종 상태가 정해진 뒤에 생성합니다.
8. "표현이 조금 아쉬움", "더 엄밀하게 말할 수 있음", "이론적으로 오해 가능" 수준은 출력하지 마세요.
9. 최종 확정이 필요한 경우에도 여기서는 후보로만 올리고, 확정 판단은 crosscheck에 맡기세요.
10. 출력 순서는 입력 claim 순서를 따르세요.
	"""


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


def _judge_claims(
    claims: list[dict],
    utterances: list[dict],
    current_date: str,
    hint: dict,
    utt_map: dict,
    context_mode: str | None = None,
) -> tuple[list[dict], bool, int, dict]:
    from . import claim_common as cv

    if not claims:
        return [], False, 0, cv._empty_token_usage()

    context_mode = _normalize_judge_context_mode(context_mode)
    full_prompt = _build_judge_prompt(claims, utterances, current_date, hint, context_mode)
    system_prompt, prompt = _split_judge_prompt_for_cache(full_prompt)
    api_calls = 0
    token_usage = cv._empty_token_usage()
    claim_candidates: dict[str, list[dict]] = {}
    for claim in claims:
        claim_candidates.setdefault(str(claim.get("utterance_id", "") or ""), []).append(claim)

    def find_source_claim(uid: str, issue_payload: dict) -> dict:
        candidates = claim_candidates.get(uid, [])
        if not candidates:
            return {}
        if len(candidates) == 1:
            return candidates[0]
        issue_text = cv._compact_text(str(issue_payload.get("claim_text", "") or ""))
        if issue_text:
            for candidate in candidates:
                if cv._compact_text(str(candidate.get("claim_text", "") or "")) == issue_text:
                    return candidate
                if cv._compact_text(str(candidate.get("resolved_claim", "") or "")) == issue_text:
                    return candidate
        return {}

    for attempt in range(cv.VERIFIER_PARSE_RETRIES + 1):
        text, call_usage = cv._call_llm(
            prompt,
            system_prompt=system_prompt,
            max_tokens=8192,
            temperature=0.0,
            thinking_budget=2048,
            stage="judge",
        )
        api_calls += 1
        cv._add_call_usage(token_usage, call_usage)
        try:
            payload = json.loads(cv._strip_json_fence(text.strip()))
        except json.JSONDecodeError:
            if attempt < cv.VERIFIER_PARSE_RETRIES:
                print(f"    ↺ claim 판정 JSON 파싱 재시도 ({attempt+1}/{cv.VERIFIER_PARSE_RETRIES})")
                continue
            return [], True, api_calls, token_usage

        raw = payload.get("issues", [])
        if not isinstance(raw, list):
            return [], False, api_calls, token_usage

        issues = []
        for raw_issue in raw:
            if not isinstance(raw_issue, dict):
                continue
            uid = str(raw_issue.get("utterance_id", "") or "").strip()
            if uid not in utt_map:
                continue
            issue_type = cv.normalize_issue_type(raw_issue.get("type", ""))
            if issue_type not in cv.ALLOWED_ISSUE_TYPES:
                continue
            try:
                conf = float(raw_issue.get("candidate_strength", raw_issue.get("confidence", 0)) or 0)
            except Exception:
                conf = 0.0
            if conf < JUDGE_MIN_CONFIDENCE:
                continue

            ref = utt_map[uid]
            source_claim = find_source_claim(uid, raw_issue)
            issue = {
                "utterance_id": uid,
                "type": issue_type,
                "claim_text": str(raw_issue.get("claim_text", "") or "").strip(),
                "problematic_content": str(raw_issue.get("problematic_content", "") or "").strip(),
                "issue": str(raw_issue.get("issue", "") or raw_issue.get("candidate_reason", "") or "").strip(),
                "candidate_reason": str(
                    raw_issue.get("candidate_reason", "")
                    or raw_issue.get("issue", "")
                    or raw_issue.get("explanation", "")
                    or ""
                ).strip(),
                "candidate_strength": conf,
                "confidence": conf,
            }
            if source_claim:
                issue["claim_type"] = source_claim.get("claim_type", "")
                issue["resolved_claim"] = source_claim.get("resolved_claim", "")
                issue["is_approximate"] = bool(source_claim.get("is_approximate"))
                if not issue["claim_text"]:
                    issue["claim_text"] = source_claim.get("claim_text", "")
            if not issue.get("verification_question"):
                issue["verification_question"] = _build_verification_question(issue or source_claim)
            issue["start_time"] = ref["start_time"]
            issue["timestamp"] = f"[{ref['start_time']:.1f}s]"
            issue["slide_number"] = ref["slide_number"]
            if not issue.get("problematic_content"):
                issue["problematic_content"] = ref["text"][:80]
            if not isinstance(issue.get("evidence_sources"), list):
                issue["evidence_sources"] = []
            if "severity" not in issue:
                issue["severity"] = "major"
            issue["issue_type_label"] = cv.issue_type_label(issue_type)
            issue["issue_type_code"] = cv.issue_type_code(issue_type)
            issue["issue_type_code_label"] = cv.issue_type_code_label(issue_type)

            cv._normalize_severity(issue)
            if cv._is_asr_artifact(issue, utt_map):
                continue
            issues.append(issue)

        return cv._dedupe_issues(issues), False, api_calls, token_usage

    return [], True, api_calls, token_usage


def recover_claim_judgement(
    claims: list[dict],
    utterances: list[dict],
    current_date: str,
    hint: dict,
    utt_map: dict,
    label: str,
    context_mode: str | None = None,
) -> tuple[list[dict], bool, int, dict, bool]:
    from . import claim_common as cv

    total_api_calls = 0
    total_token_usage = cv._empty_token_usage()
    last_issues: list[dict] = []
    parse_failed = False
    last_exc = None

    for attempt in range(cv.VERIFIER_BATCH_RECOVERY_RETRIES + 1):
        if attempt > 0:
            print(f"    ↺ {label} claim 판정 재처리 ({attempt}/{cv.VERIFIER_BATCH_RECOVERY_RETRIES})")
        try:
            issues, parse_failed, api_calls, token_usage = _judge_claims(
                claims, utterances, current_date, hint, utt_map, context_mode
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


def judge_claims_only(
    all_claims_by_batch: list[tuple],
    current_date: str,
    hint: dict,
    num_runs: int = 1,
    min_detection_rate: float = 0.5,
    log_prefix: str = "",
    context_mode: str | None = None,
) -> tuple[list[dict], int, dict]:
    """2단계만 실행: claim 판정. (이슈 리스트, api_calls) 반환."""
    from . import claim_common as cv

    run_results = []
    total_api = 0
    total_token_usage = cv._empty_token_usage()

    context_mode = _normalize_judge_context_mode(context_mode)
    prefix = f"[{log_prefix}] " if log_prefix else ""
    total_batches = sum(1 for _, claims in all_claims_by_batch if claims)
    print(f"  {prefix}claim 판정 문맥 모드: {context_mode}", flush=True)

    for run in range(num_runs):
        if num_runs > 1:
            print(f"\n  {prefix}판정 run {run+1}/{num_runs}")
        run_issues = []
        run_token_usage = cv._empty_token_usage()
        active_batch_idx = 0
        for batch_idx, (batch, claims) in enumerate(all_claims_by_batch, start=1):
            if not claims:
                continue
            active_batch_idx += 1
            batch_map = {u["utterance_id"]: u for u in batch}
            ids = f"{batch[0]['utterance_id']}..{batch[-1]['utterance_id']}"
            print(f"  {prefix}({active_batch_idx}/{total_batches}) {ids}", flush=True)
            issues, parse_failed, api_calls, token_usage, ok = recover_claim_judgement(
                claims, batch, current_date, hint, batch_map,
                f"판정 {run+1} 배치 {batch_idx} {ids}",
                context_mode=context_mode,
            )
            run_issues.extend(issues)
            total_api += api_calls
            run_token_usage = cv._merge_token_usage(run_token_usage, token_usage)
        run_results.append(cv._make_result(cv._dedupe_issues(run_issues), 0, token_usage=run_token_usage))

    print(f"\n  📊 통합 중 (합의 기준 {min_detection_rate:.0%})...")
    merged = cv.merge_multiple_runs(run_results, num_runs, min_detection_rate)
    total_token_usage = cv._merge_token_usage(*(r.get("token_usage") for r in run_results))
    return merged.get("issues", []), total_api, total_token_usage
