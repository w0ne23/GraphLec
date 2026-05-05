from __future__ import annotations

import json


def _build_claim_context_block(claim: dict, utterances: list[dict], slide_ctx: dict) -> str:
    from . import claim_common as cv

    uid = str(claim.get("utterance_id", "") or "")
    idx = next((i for i, u in enumerate(utterances) if str(u.get("utterance_id", "") or "") == uid), -1)
    target = utterances[idx] if idx >= 0 else {}
    slide_number = int(target.get("slide_number", 0) or 0)
    slide = slide_ctx.get(slide_number, {})
    slide_title = slide.get("title", f"슬라이드 {slide_number}") if slide_number else "슬라이드 알 수 없음"
    time_range = slide.get("time_range", "")

    if idx >= 0:
        local = utterances[max(0, idx - 3):idx + 4]
    else:
        local = []
    local_lines = []
    for u in local:
        prefix = "현재" if str(u.get("utterance_id", "") or "") == uid else "문맥"
        local_lines.append(f"   - {prefix}: {cv._format_utterance_for_prompt(u)}")
    local_text = "\n".join(local_lines) if local_lines else "   (발화 문맥 없음)"

    return (
        f"[강의 문맥]\n"
        f"   슬라이드: {slide_title} ({time_range})\n"
        f"   주변 발화:\n{local_text}"
    )


def _build_judge_prompt(
    claims: list[dict],
    utterances: list[dict],
    current_date: str,
    hint: dict,
    slide_ctx: dict,
) -> str:
    claim_lines = []
    for i, c in enumerate(claims, 1):
        approx = " [근사치]" if c.get("is_approximate") else ""
        resolved = c.get("resolved_claim", c.get("claim_text", ""))
        claim_lines.append(
            f"{i}. [{c['utterance_id']}] ({c.get('claim_type', '?')}){approx}\n"
            f"   원문: {c.get('claim_text', '')}\n"
            f"   해소: {resolved}\n"
            f"   검증 질문: {c.get('verification_question', '')}\n"
            f"{_build_claim_context_block(c, utterances, slide_ctx)}"
        )

    return f"""당신은 강의 발화의 정오를 판정하는 전문가입니다.
오늘 날짜: {current_date}
강의 도메인: {hint['label']}
도메인 참고: {hint.get('outdated_guidance', '')}

아래는 강의 발화에서 추출된 사실 주장(claim) 목록입니다.
각 claim에는 빠른 1차 판정과 검토 후보 선별을 위한 주변 발화 문맥이 함께 제공됩니다.

판정 대상 claim 목록:
{chr(10).join(claim_lines)}

### 판정 기준

각 claim에 대해 다음 셋 중 하나로 판단하세요:

1. confirmed_error
   **"학생이 이 발화를 그대로 믿고 실무/시험에 적용했을 때, 객관적으로 틀린 결과가 나오는가?"**

2. needs_review
   객관적 오류로 확정하기는 이르지만, 교수자/검수자가 확인할 만한 오해 가능성이 있는가?

3. no_issue
   문제가 없거나 검토할 가치가 낮은가?

"틀린 결과"란: 코드 에러, 오답, 사실과 반대되는 이해를 말합니다.

### 해석 원칙

다음 경우는 매우 신중하게 판단하세요:

0. **강의 문맥 우선**
   판정은 반드시 제공된 주변 발화 안에서 학생이 실제로 이해할 명제를 기준으로 하세요.
   claim의 해소 문장이 과도하게 일반화되어 보이면, 원문 발화와 주변 문맥의 범위로 다시 좁혀 해석하세요.
   슬라이드 세부 내용까지 확인해야만 판단 가능한 경우는 이 단계에서 확정 오류로 보고하지 마세요.

1. **복잡한 현상의 배경 설명**
   여러 원인 중 하나를 대표 축으로 설명하는 경우, "유일한 원인", "전적으로", "오직"처럼
   배타적으로 단정하지 않는 한 factual_error로 보지 마세요.

2. **상대 시점 표현**
   "최근", "요즘", "현재" 같은 표현은 강의 발화/녹화 시점 기준입니다.
   절대 연도/월/버전이 명시될 때만 엄격히 판단하세요.

3. **애매한 지시어/생략 표현**
   둘 이상의 합리적 해석이 가능하면 오류로 보고하지 마세요.

4. **더 일반적인 표현 vs 더 구체적인 표현**
   상위 범주 수준으로 말한 것만으로 factual_error로 보지 마세요.
   다만 잘못된 범주를 말하거나 직접 모순되면 오류입니다.

### 반드시 보고

1. **방향/극성 반전**: 증가↔감소, 촉진↔억제, 양성↔음성, 원인↔결과가 뒤바뀐 경우
2. **부정어 누락/추가**: "~않다"가 빠지거나 추가되어 의미가 반전된 경우
3. **범주 오귀속**: A에 속하는 것을 B에 속한다고 하는 경우
4. **수량/범위 왜곡**: "모든/항상/반드시"로 단정했는데 실제로는 일부/조건부인 경우. 단, 교육적 관례 소개와 구분할 것.

### 검토 필요 후보로 보고

아래 경우는 확정 오류가 아니더라도 `candidate_status: "needs_review"`로 보고하세요:

- 표현이 과장되어 학생이 범위/예외/조건을 놓칠 가능성이 있는 경우
- 교육적 단순화인지 실제 오류인지 문맥만으로 확정하기 어려운 경우
- 용어가 비슷하지만 다른 개념으로 오해될 가능성이 있는 경우
- 슬라이드나 주변 발화와 함께 보면 확인이 필요한 생략/지시/범위 문제가 있는 경우
- outdated 의심은 있으나 현재 날짜 기준 근거 확인이 필요한 경우

### 보고 안 하는 경우

- 표현이 비표준적이지만 학생이 결과적으로 맞는 이해를 갖게 되는 경우
- 앞뒤 맥락에서 강의자가 즉시 정정하여 학생이 올바른 정보를 받는 경우
- 교육적 단순화로 세부사항을 생략했지만 핵심 결론은 맞는 경우
- [근사치] claim의 반올림/소수점 차이
- 반박 근거가 강의 범위 밖의 고급 개념/예외뿐인 경우
- 입문 강의의 계층 구조 설명을 펌웨어, DMA, 하이퍼바이저, 장치 내부 컨트롤러 같은 고급 예외만으로 반박하는 경우
- 관례·컨벤션을 소개하는 발화 ("반드시 ~합니다"라도 해당 도메인 표준 관례면 오류 아님)
- 라이브러리/프레임워크의 실제 동작과 일치하는 발화
- 강의용 표현과 코드 변수명 차이 (풀네임으로 부르는 것은 오류 아님)
- "가장 유명하다", "대표적이다" 같은 주관적 평가

### 유형 판정

- factual_error: 사실이 틀림
- outdated: {current_date} 기준 deprecated/EOL/폐기된 것을 현재 유효한 것처럼 설명

### 응답 (JSON만)

```json
{{
  "issues": [
    {{
      "utterance_id": "U0001",
      "type": "factual_error" | "outdated",
      "candidate_status": "confirmed_error" | "needs_review",
      "claim_text": "판정한 claim 원문",
      "problematic_content": "문제 발화 원문 (80자 이내)",
      "issue": "학생이 어떤 틀린 지식을 갖게 되는지",
      "correct_info": "올바른 정보",
      "explanation": "왜 틀린지",
      "recommendation": "수정 권고",
      "confidence": 0.0-1.0
    }}
  ]
}}
```

지침:
1. confirmed_error는 confidence 0.70 이상일 때만 출력하세요.
2. needs_review는 confidence 0.55 이상일 때만 출력하세요.
3. 동일 개념 문제는 한 건만.
4. 문제가 없으면 {{"issues": []}}만.
5. JSON 외 텍스트 금지.
"""


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

    prompt = _build_judge_prompt(claims, utterances, current_date, hint, slide_ctx)
    api_calls = 0
    token_usage = cv._empty_token_usage()

    for attempt in range(cv.VERIFIER_PARSE_RETRIES + 1):
        text, call_usage = cv._call_llm(prompt, max_tokens=8192, thinking_budget=2048, stage="judge")
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
        for issue in raw:
            if not isinstance(issue, dict):
                continue
            uid = str(issue.get("utterance_id", "") or "").strip()
            if uid not in utt_map:
                continue
            issue_type = str(issue.get("type", "")).lower().strip()
            if issue_type not in cv.ALLOWED_ISSUE_TYPES:
                continue
            try:
                conf = float(issue.get("confidence", 0) or 0)
            except Exception:
                conf = 0.0
            candidate_status = str(issue.get("candidate_status", "") or "").lower().strip()
            if candidate_status not in {"confirmed_error", "needs_review"}:
                candidate_status = "confirmed_error" if conf >= 0.70 else "needs_review"
            min_confidence = 0.55 if candidate_status == "needs_review" else 0.70
            if conf < min_confidence:
                continue

            ref = utt_map[uid]
            issue["utterance_id"] = uid
            issue["type"] = issue_type
            issue["candidate_status"] = candidate_status
            issue["start_time"] = ref["start_time"]
            issue["timestamp"] = f"[{ref['start_time']:.1f}s]"
            issue["slide_number"] = ref["slide_number"]
            if not issue.get("problematic_content"):
                issue["problematic_content"] = ref["text"][:80]
            if not isinstance(issue.get("evidence_sources"), list):
                issue["evidence_sources"] = []
            if candidate_status == "needs_review":
                issue.setdefault("review_stage", "claim_judge")
                issue.setdefault("review_reason_code", "judge_review_candidate")

            issue.pop("severity", None)
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
) -> tuple[list[dict], int, dict]:
    """2단계만 실행: claim 판정. (이슈 리스트, api_calls) 반환."""
    from . import claim_common as cv

    run_results = []
    total_api = 0
    total_token_usage = cv._empty_token_usage()

    for run in range(num_runs):
        print(f"\n  판정 [{run+1}/{num_runs}]")
        run_issues = []
        run_token_usage = cv._empty_token_usage()
        for batch_idx, (batch, claims) in enumerate(all_claims_by_batch, start=1):
            if not claims:
                continue
            batch_map = {u["utterance_id"]: u for u in batch}
            ids = f"{batch[0]['utterance_id']}..{batch[-1]['utterance_id']}"
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
