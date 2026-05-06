from __future__ import annotations

import json


JUDGE_MIN_CONFIDENCE = 0.80


def _build_shared_judge_context(utterances: list[dict], slide_ctx: dict) -> str:
    from . import claim_common as cv

    slide_numbers = []
    seen_slides = set()
    for u in utterances:
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

    utterance_lines = [
        f"- {cv._format_utterance_for_prompt(u)}"
        for u in utterances
    ]

    return (
        "[배치 공통 슬라이드]\n"
        + ("\n".join(slide_lines) if slide_lines else "(슬라이드 정보 없음)")
        + "\n\n[배치 공통 발화]\n"
        + ("\n".join(utterance_lines) if utterance_lines else "(발화 없음)")
    )


def _build_judge_prompt(
    claims: list[dict],
    utterances: list[dict],
    current_date: str,
    hint: dict,
    slide_ctx: dict,
) -> str:
    shared_context = _build_shared_judge_context(utterances, slide_ctx)
    claim_lines = []
    for i, c in enumerate(claims, 1):
        approx = " [근사치]" if c.get("is_approximate") else ""
        resolved = c.get("resolved_claim", c.get("claim_text", ""))
        claim_lines.append(
            f"{i}. [{c['utterance_id']}] ({c.get('claim_type', '?')}){approx}\n"
            f"   원문: {c.get('claim_text', '')}\n"
            f"   해소: {resolved}"
        )

    return f"""당신은 강의 발화에서 crosscheck에 올릴 문제 후보만 선별하는 판정자입니다.
오늘 날짜: {current_date}
강의 도메인: {hint['label']}

아래는 강의 발화에서 추출된 사실 주장(claim) 목록입니다.
빠른 1차 판정을 위한 배치 공통 문맥이 함께 제공됩니다.
claim은 `utterance_id`로 공통 발화 문맥의 해당 발화를 참조하세요.

배치 공통 문맥:
{shared_context}

판정 대상 claim 목록:
{chr(10).join(claim_lines)}

### 판정 기준

이 단계의 목적은 최종 교수 피드백 작성이 아니라, **검증할 가치가 있는 후보만 좁히는 것**입니다.
각 claim에 대해 아래 3개 질문에 모두 "예"라고 답할 수 있을 때만 이슈 후보로 출력하세요.

1. 학생이 이 발화를 그대로 외웠을 때 생길 수 있는 **구체적인 잘못된 명제**를 한 문장으로 쓸 수 있는가?
2. 그 명제가 틀리거나 위험한 이유가 단순 취향이 아니라 **명확한 반례, 조건, 범위, 정의 차이, 현행성 확인 대상**으로 설명되는가?
3. 제공된 주변 발화가 그 문제를 명확히 정정하거나 조건/예외를 분명히 보완하지 않았는가?

하나라도 아니오이면 출력하지 마세요.
막연히 "오해할 수도 있다", "더 엄밀하게 말할 수 있다", "보충하면 좋다"는 이유만으로는 후보가 아닙니다.
문맥이 애매하거나 같은 문제 표현을 반복/강화한다면 후보로 올리고 crosscheck에 맡기세요.
단, 슬라이드 요약 문구와 발화가 같은 축약 설명을 반복하는 것만으로는 독립적인 강화 근거가 아닙니다.

### 해석 원칙

다음 경우는 매우 신중하게 판단하세요:

0. **강의 문맥 우선**
   판정은 반드시 제공된 주변 발화 안에서 학생이 실제로 이해할 명제를 기준으로 하세요.
   claim의 해소 문장이 과도하게 일반화되어 보이면, 원문 발화와 주변 문맥의 범위로 다시 좁혀 해석하세요.
   주변 발화가 문제를 명확히 정정하거나 조건/예외를 분명히 보완한 경우만 후보에서 제외하세요.
   완전한 정정 문장이 없더라도, 앞뒤 발화나 슬라이드 구조가 생략된 주어/대상/관계/범위를 자연스럽게 지지하면
   그 문맥상 지지되는 해석을 우선하세요. 문제가 되는 해석이 더 약하거나 과도한 추측이면 후보로 올리지 마세요.
   슬라이드나 더 넓은 문맥에서 확인이 필요하지만 구체적인 잘못된 명제가 남는 경우는 후보로 올리세요.

1. **복잡한 현상의 배경 설명**
   여러 원인 중 하나를 대표 축으로 설명하는 경우, "유일한 원인", "전적으로", "오직"처럼
   배타적으로 단정하지 않는 한 후보로 올리지 마세요.

1-1. **대표 효과/대표 구현의 개론 설명**
   개론, 목차, 기능 나열, 슬라이드 괄호 라벨, "나중에 자세히 설명한다"는 예고 문맥에서는
   대표 효과나 구현 예시를 전체 정의로 단정했다고 보지 마세요.
   "X의 대표적 효과/사용 상황을 설명했다"와 "X는 오직 그것뿐이다"를 구분하세요.
   원문 발화나 슬라이드가 배타적 정의를 명시하지 않으면 후보로 올리지 마세요.

2. **상대 시점 표현**
   "최근", "요즘", "현재" 같은 표현은 강의 발화/녹화 시점 기준입니다.
   절대 연도/월/버전/지원 여부처럼 확인 가능한 현재성 주장이 명시될 때만 후보로 올리세요.

2-1. **현실 대상에 붙은 예시 수치**
   실재하는 대상, 집단, 기관, 지역, 제품, 시장, 인구, 통계값 등에 구체적인 수치/비율/연도/규모가 붙으면,
   그 수치가 예시 설명 안에서 쓰였더라도 검증 가능한 numeric/currentness claim으로 다루세요.
   "예를 들어", "한", "이런 식으로" 같은 표현이 있어도, 현실 대상의 수치가 실제와 다르거나 최신성 확인이 필요하면 후보로 올리세요.
   단, 명시적으로 가상의 대상/임의의 숫자/변수 예시라고 밝힌 경우에는 현실 사실 claim으로 보지 마세요.

3. **애매한 지시어/생략 표현**
   둘 이상의 합리적 해석이 가능하면 후보로 올리지 마세요.
   특히 "얘/이거/그거/이 값/해당 항목" 같은 지시어가 슬라이드의 기호/행/도형/목록 항목을 가리키는지,
   직전 발화의 용어를 가리키는지 둘 다 가능하면 특정 선행사 하나로 확정해 문제를 만들지 마세요.
   원문과 슬라이드를 함께 보면 맞는 설명인데, resolved_claim에서만 더 강한 명제로 바뀐 경우는 후보가 아닙니다.
   생략된 주어가 있을 때는 가장 가까운 단어 하나가 아니라, 슬라이드의 배치, 직전 정의, 이어지는 설명까지 포함해
   학생이 자연스럽게 따라갈 대상을 먼저 찾으세요.

4. **더 일반적인 표현 vs 더 구체적인 표현**
   상위 범주 수준으로 말한 것만으로 후보로 올리지 마세요.
   다만 잘못된 범주를 말하거나 직접 모순되면 오류입니다.

5. **혼동 가능 설명**
   학생이 헷갈릴 수 있다는 말만으로는 부족합니다.
   서로 다른 두 개념을 같은 것으로 외우게 하거나, 주체/과정/원인/조건을 잘못 연결하게 만드는
   구체적인 오답 명제가 있어야만 후보로 올리세요.
   단, 그 오답 명제가 애매한 지시어를 한쪽으로만 강제 해석해야 생기는 경우에는 confusing_explanation으로 올리지 마세요.

### 반드시 보고

1. **방향/극성 반전**: 증가↔감소, 촉진↔억제, 양성↔음성, 원인↔결과가 뒤바뀐 경우
2. **부정어 누락/추가**: "~않다"가 빠지거나 추가되어 의미가 반전된 경우
3. **범주 오귀속**: A에 속하는 것을 B에 속한다고 하는 경우
4. **수량/범위 왜곡**: "모든/항상/반드시/유일/오직/~만"으로 단정했는데 실제로는 일부/조건부인 경우
5. **현행성 후보**: 현재/요즘/최신/지원 여부를 현재 사실처럼 강하게 말했고, 최신성 검증이 필요한 경우
6. **핵심 개념 동일시**: 서로 다른 개념, 주체, 과정, 권한, 대상을 같은 것으로 외우게 만드는 경우
7. **범위 과잉 단정**: 특정 조건/도메인에서는 맞지만, A만 맞는 것처럼 말하거나 다른 조건의 가능성을 닫아버린 경우
8. **현실 대상의 틀린 예시 수치**: 예시 문맥이어도 실재 대상에 붙은 구체 수치/비율/연도/규모가 실제와 다르거나 오래된 경우

### 보고 안 하는 경우

- 표현이 비표준적이지만 학생이 결과적으로 맞는 이해를 갖게 되는 경우
- 앞뒤 맥락에서 강의자가 즉시 정정하여 학생이 올바른 정보를 받는 경우
- 교육적 단순화로 세부사항을 생략했지만 핵심 결론은 맞는 경우
- 개론/목차/기능 소개에서 대표 효과, 대표 구현, 대표 사용 상황만 짧게 소개한 경우
- 슬라이드와 발화가 같은 축약 문구를 공유하지만, 그 문구가 정식 정의가 아니라 요약 라벨인 경우
- 조건/예외를 이미 직전/직후 발화에서 충분히 설명해 문제 명제가 사라지는 경우
- 강의 수준에 맞춘 단순화이며 학생이 핵심 결론을 올바르게 가져갈 수 있는 경우
- 같은 개념 문제가 이미 더 대표적인 claim에서 한 번 보고된 경우
- [근사치] claim의 반올림/소수점 차이
- 반박 근거가 강의 범위 밖의 고급 개념/예외뿐인 경우
- 강의 수준 밖의 세부 예외나 하위 구현만으로 핵심 설명을 반박하는 경우
- 관례·컨벤션을 소개하는 발화 ("반드시 ~합니다"라도 해당 도메인 표준 관례면 오류 아님)
- 도메인의 실제 동작, 표준 관례, 강의 자료의 조건과 일치하는 발화
- 해당 분야의 표준 표현이 역할·책임·범위를 설명할 뿐 다른 가능성을 배제하지 않는 경우.
  단, 발화가 실제로 배타적 사용, 다른 주체의 불가능성, 서로 다른 범주의 동일시를 주장할 때만 보고하세요.
- 강의용 표현과 자료상의 표기 차이만 있는 경우
- 지시어가 포함된 발화가 슬라이드와 함께 보면 맞는 설명이고, 문제가 되는 해석이 resolved_claim의 과도한 선행사 확정에서만 생기는 경우
- "가장 유명하다", "대표적이다" 같은 주관적 평가
- 반례가 아니라 더 깊은 수준의 보충 설명만 가능한 경우
- "학생이 혹시 다르게 받아들일 수도 있다" 정도의 낮은 가능성

### 유형 판정

- factual_error: 발언 자체의 객관 사실, 정의, 순서, 메커니즘이 틀림
- temporal_error: 시대적/현행성 오류. 현재 날짜({current_date}) 기준으로 현재성, 최신성, 지원 여부가 틀림
- confusing_explanation: 학생이 핵심 개념을 헷갈릴 만한 설명. 단순 비유 취향이나 더 자세한 설명 요구는 제외
- scope_overclaim: A만 맞다고 설명했지만 조건/범위/도메인에 따라 B도 맞거나 예외가 있어 과도하게 단정함

### 응답 (JSON만)

```json
{{
  "issues": [
    {{
	      "utterance_id": "U0001",
	      "type": "factual_error" | "temporal_error" | "confusing_explanation" | "scope_overclaim",
	      "claim_text": "판정한 claim 원문",
	      "problematic_content": "문제 발화 원문 (80자 이내)",
	      "issue": "학생이 잘못 외울 수 있는 명제와 후보 사유를 한 문장으로 작성",
	      "candidate_reason": "명확한 반례/조건/범위/정의 차이/현행성 확인 포인트를 짧게 작성",
	      "confidence": 0.0-1.0
	    }}
	  ]
	}}
```

지침:
1. confidence {JUDGE_MIN_CONFIDENCE:.2f} 미만은 출력하지 마세요.
2. 동일 개념 문제는 한 건만.
3. 단순히 "더 자세히 설명하면 좋겠다" 수준이면 출력하지 마세요.
4. 문제가 없으면 {{"issues": []}}만.
5. JSON 외 텍스트 금지.
6. 이 단계에서 교수 피드백 문장, 대체 표현, 반례 상세 설명, 최종 문맥 해소 여부를 작성하지 마세요.
   그런 필드는 crosscheck 이후 최종 상태가 정해진 뒤에 생성합니다.
7. "표현이 조금 아쉬움", "더 엄밀하게 말할 수 있음", "이론적으로 오해 가능" 수준은 출력하지 마세요.
8. 최종 확정이 필요한 경우에도 여기서는 후보로만 올리고, 확정 판단은 crosscheck에 맡기세요.
9. 출력 순서는 입력 claim 순서를 따르세요.
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
    slide_ctx: dict,
) -> tuple[list[dict], bool, int, dict]:
    from . import claim_common as cv

    if not claims:
        return [], False, 0, cv._empty_token_usage()

    full_prompt = _build_judge_prompt(claims, utterances, current_date, hint, slide_ctx)
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
                conf = float(raw_issue.get("confidence", 0) or 0)
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
                "confidence": conf,
            }
            if source_claim:
                issue["claim_type"] = source_claim.get("claim_type", "")
                issue["resolved_claim"] = source_claim.get("resolved_claim", "")
                issue["is_approximate"] = bool(source_claim.get("is_approximate"))
                if not issue["claim_text"]:
                    issue["claim_text"] = source_claim.get("claim_text", "")
            if not issue.get("verification_question"):
                issue["verification_question"] = cv.build_verification_question(issue or source_claim)
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
            print(f"    ↺ {label} claim 판정 재처리 ({attempt}/{cv.VERIFIER_BATCH_RECOVERY_RETRIES})")
        try:
            issues, parse_failed, api_calls, token_usage = _judge_claims(
                claims, utterances, current_date, hint, utt_map, slide_ctx
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
    slide_ctx: dict,
    num_runs: int = 1,
    min_detection_rate: float = 0.5,
    log_prefix: str = "",
) -> tuple[list[dict], int, dict]:
    """2단계만 실행: claim 판정. (이슈 리스트, api_calls) 반환."""
    from . import claim_common as cv

    run_results = []
    total_api = 0
    total_token_usage = cv._empty_token_usage()

    prefix = f"[{log_prefix}] " if log_prefix else ""
    total_batches = sum(1 for _, claims in all_claims_by_batch if claims)

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
                claims, batch, current_date, hint, batch_map, slide_ctx,
                f"판정 {run+1} 배치 {batch_idx} {ids}",
            )
            run_issues.extend(issues)
            total_api += api_calls
            run_token_usage = cv._merge_token_usage(run_token_usage, token_usage)
        run_results.append(cv._make_result(cv._dedupe_issues(run_issues), 0, token_usage=run_token_usage))

    print(f"\n  📊 통합 중 (합의 기준 {min_detection_rate:.0%})...")
    merged = cv.merge_multiple_runs(run_results, num_runs, min_detection_rate)
    total_token_usage = cv._merge_token_usage(*(r.get("token_usage") for r in run_results))
    return merged.get("issues", []), total_api, total_token_usage
