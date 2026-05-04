from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

_CROSSCHECK_VERDICTS = {"agree", "disagree", "inconclusive"}
_CROSSCHECK_PARSE_RETRIES = 1
_SLIDE_SUPPORT_STATUSES = {"supports", "contradicts", "neutral", "insufficient"}
_ERROR_ORIGINS = {"speech_error", "slide_error", "slide_speech_mismatch", "unclear"}


def _parse_crosscheck_payload(text: str) -> dict:
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
        compact = re.sub(r"\s+", " ", fixed).strip()
        for payload_text in (candidate, fixed, compact):
            try:
                payload = json.loads(payload_text)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                verdict = str(payload.get("verdict", "") or "").lower().strip()
                reason = str(payload.get("reason", "") or "").strip()
                if verdict in _CROSSCHECK_VERDICTS:
                    return {"verdict": verdict, "reason": reason}

        # 흔한 비표준 응답 형태 복구:
        # {verdict: agree, reason: "..."} / {'verdict':'agree', ...}
        relaxed = fixed
        relaxed = re.sub(r"([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*:)", r'\1"\2"\3', relaxed)
        relaxed = re.sub(
            r'("verdict"\s*:\s*)(agree|disagree|inconclusive)(\s*[,}])',
            lambda m: f'{m.group(1)}"{m.group(2)}"{m.group(3)}',
            relaxed,
            flags=re.IGNORECASE,
        )
        relaxed = relaxed.replace("'", '"')
        try:
            payload = json.loads(relaxed)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            verdict = str(payload.get("verdict", "") or "").lower().strip()
            reason = str(payload.get("reason", "") or "").strip()
            if verdict in _CROSSCHECK_VERDICTS:
                return {"verdict": verdict, "reason": reason}

    verdict = cv._extract_json_like_string_field(cleaned, "verdict").lower().strip()
    if verdict not in _CROSSCHECK_VERDICTS:
        m = re.search(
            r"['\"]?verdict['\"]?\s*:\s*['\"]?(agree|disagree|inconclusive)['\"]?",
            cleaned,
            flags=re.IGNORECASE,
        )
        if m:
            verdict = m.group(1).lower()
    if verdict not in _CROSSCHECK_VERDICTS:
        m = re.search(
            r"['\"]?verdict['\"]?\s*:\s*['\"]?(agr|dis|incon)",
            cleaned,
            flags=re.IGNORECASE,
        )
        if m:
            prefix = m.group(1).lower()
            if prefix.startswith("agr"):
                verdict = "agree"
            elif prefix.startswith("dis"):
                verdict = "disagree"
            elif prefix.startswith("incon"):
                verdict = "inconclusive"

    reason = cv._extract_json_like_string_field(cleaned, "reason")
    if not reason:
        m = re.search(
            r"['\"]?reason['\"]?\s*:\s*['\"]?(.+?)(?:['\"]?\s*[,}]|\n|$)",
            cleaned,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if m:
            reason = re.sub(r"\s+", " ", m.group(1)).strip()

    if verdict in _CROSSCHECK_VERDICTS:
        return {"verdict": verdict, "reason": reason}

    raise ValueError("crosscheck_response_parse_failed")


def _build_slide_transcript_block(slides: list[dict], slide_number: int) -> str:
    slide_text = ""
    structure_text = ""
    meta_lines = []
    transcript_lines = []
    for slide in slides:
        if int(slide.get("slide_number", 0) or 0) != slide_number:
            continue
        # 슬라이드 자체 텍스트 (슬라이드에 적힌 내용)
        slide_text = str(slide.get("slide_text", "") or "").strip()
        structure_text = str(slide.get("t1_structure", "") or "").strip()
        slide_type = str(slide.get("slide_type", "") or "").strip()
        if slide_type:
            meta_lines.append(f"  slide_type: {slide_type}")
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
    if meta_lines:
        parts.append("[슬라이드 텍스트화 메타]\n" + "\n".join(meta_lines))
    if slide_text:
        parts.append(f"[슬라이드 텍스트]\n{slide_text}")
    if structure_text and structure_text not in slide_text:
        parts.append(f"[슬라이드 시각 구조 설명]\n{structure_text}")
    parts.append("[강의자 발화]\n" + ("\n".join(transcript_lines) if transcript_lines else "(없음)"))
    return "\n".join(parts)


def _build_multi_slide_transcript_block(slides: list[dict], slide_numbers: list[int]) -> str:
    blocks = []
    for slide_number in slide_numbers:
        if slide_number <= 0:
            continue
        block = _build_slide_transcript_block(slides, slide_number)
        blocks.append(f"[슬라이드 {slide_number}]\n{block}")
    return "\n\n".join(blocks) if blocks else "(없음)"


def judge_single_claim(issue: dict, ctx: dict) -> tuple[str, str, dict]:
    """단일 이슈를 재검증하여 agree/disagree/inconclusive 반환."""
    from . import claim_common as cv

    uid = issue.get("utterance_id", "")
    claim_text = issue.get("claim_text", "")
    issue_desc = issue.get("issue", "")

    utterances = ctx["utterances"]
    slides = ctx["slides"]
    slide_ctx = ctx["slide_ctx"]
    hint = ctx["hint"]
    current_date = ctx["current_date"]

    utt_map = {u["utterance_id"]: (i, u) for i, u in enumerate(utterances)}
    if uid not in utt_map:
        return "inconclusive", "utterance_id를 찾을 수 없음", cv._empty_token_usage()

    idx, target_utt = utt_map[uid]

    slide_num = target_utt.get("slide_number", 0)
    prev_slide_num = max(0, int(slide_num or 0) - 1)
    slide_info = slide_ctx.get(slide_num, {})
    title = slide_info.get("title", f"슬라이드 {slide_num}")
    time_range = slide_info.get("time_range", "")
    prev_slide_info = slide_ctx.get(prev_slide_num, {}) if prev_slide_num > 0 else {}
    prev_title = prev_slide_info.get("title", f"슬라이드 {prev_slide_num}") if prev_slide_num > 0 else ""
    prev_time_range = prev_slide_info.get("time_range", "") if prev_slide_num > 0 else ""
    slide_numbers = [n for n in [prev_slide_num, int(slide_num or 0)] if n > 0]
    transcript_block = _build_multi_slide_transcript_block(slides, slide_numbers)

    prompt = f"""다른 검증 모델이 아래 발화에서 문제를 발견했습니다.
당신은 이 지적이 타당한지 독립적으로 판단해야 합니다.

## 도메인
{hint.get('label', '')}

## 대상 슬라이드
{title} ({time_range})

## 이전 슬라이드
{(prev_title + f" ({prev_time_range})") if prev_slide_num > 0 else "없음"}

## 이전+현재 슬라이드 내용 (슬라이드 텍스트 + 강의자 발화)
{transcript_block}

## 지적 내용
- claim: {claim_text}
- 문제: {issue_desc}

## 판정 기준
- 학생이 이 발화를 따라했을 때 실제로 틀린 지식을 갖게 되는가?
- 슬라이드 텍스트와 발화가 직접적으로 모순되는가?
- claim이나 문제 제기가 발화 원문보다 넓게 일반화되었다면, 발화 원문과 주변 문맥의 범위로 다시 좁혀 판단하세요.
- 슬라이드가 같은 교육적 단순화를 명시하고 있고 발화가 이를 설명하는 경우, 강의 범위 밖의 고급 예외만으로 오류로 보지 마세요.
- 입문 강의의 운영체제 계층 구조 설명을 펌웨어, DMA, 하이퍼바이저, 장치 내부 컨트롤러 같은 예외만으로 반박하지 마세요.
- 대상 발화가 예시 문장("예를 들어", "이에 해당합니다")이면, 반드시 바로 앞뒤 정의 문장과 연결해서 해석하세요.
- 슬라이드 텍스트에 표, 수식, 데이터가 포함되어 있으면 직접 계산하여 발화와 비교하세요.

위 기준 중 하나라도 YES면 "agree", 모두 NO면 "disagree"로 답하세요.
확신이 부족하거나 응답 형식을 지키기 어렵다면 "inconclusive"를 선택하세요.

응답 규칙:
- 반드시 JSON object 하나만 출력
- markdown/code fence 금지
- 키는 verdict, reason 두 개만 사용
- verdict 값은 agree / disagree / inconclusive 중 하나만 사용

응답 예시:
{{"verdict":"agree","reason":"슬라이드와 발화가 직접 모순됨"}}"""

    model = str(cv._resolve_stage_model("cross_recheck") or "").strip()
    response_format = {"type": "json_object"} if (
        model.startswith("gpt") or model.startswith("o1") or model.startswith("o3")
    ) else None

    token_usage = cv._empty_token_usage()
    last_error = None
    for attempt in range(_CROSSCHECK_PARSE_RETRIES + 1):
        try:
            text, call_usage = cv._call_llm(
                prompt,
                max_tokens=1024,
                thinking_budget=1024,
                response_format=response_format,
                stage="cross_recheck",
            )
            cv._add_call_usage(token_usage, call_usage)
            payload = _parse_crosscheck_payload(text)
            verdict = str(payload.get("verdict", "") or "").lower().strip()
            reason = str(payload.get("reason", "") or "").strip()
            if verdict not in _CROSSCHECK_VERDICTS:
                return "inconclusive", f"알 수 없는 verdict: {verdict}", token_usage
            return verdict, reason, token_usage
        except Exception as e:
            last_error = e
            if attempt < _CROSSCHECK_PARSE_RETRIES:
                print(f"    ↺ 교차 재검증 JSON 파싱 재시도 ({attempt+1}/{_CROSSCHECK_PARSE_RETRIES})")

    return "inconclusive", f"교차 재검증 실패: {last_error}", token_usage


def _build_slide_recheck_prompt(
    issue: dict,
    slide_ctx: dict,
    slides: list[dict],
    hint: dict,
) -> str:
    sn = int(issue.get("slide_number", 0) or 0)
    prev_sn = max(0, sn - 1)
    ctx = slide_ctx.get(sn, {})
    title = ctx.get("title", f"슬라이드 {sn}")
    time_range = ctx.get("time_range", "")
    prev_ctx = slide_ctx.get(prev_sn, {}) if prev_sn > 0 else {}
    prev_title = prev_ctx.get("title", f"슬라이드 {prev_sn}") if prev_sn > 0 else ""
    prev_time_range = prev_ctx.get("time_range", "") if prev_sn > 0 else ""
    transcript_block = _build_multi_slide_transcript_block(slides, [prev_sn, sn] if prev_sn > 0 else [sn])

    claim = issue.get("claim_text", issue.get("problematic_content", ""))
    issue_desc = issue.get("issue", "")
    correct_info = issue.get("correct_info", "")
    explanation = issue.get("explanation", "")

    return f"""당신은 강의 내용 검증의 재검증 단계입니다.
강의 도메인: {hint.get('label', '일반')}

이전 단계에서 아래 이슈가 발견되었습니다.
이제 해당 슬라이드의 **전체 맥락**(이전+현재 슬라이드 텍스트 + 강의자 발화)을 보고,
이 이슈가 정말 타당한지 재확인해주세요.

━━━ {title} ({time_range}) ━━━

[이전 슬라이드]
{(prev_title + f" ({prev_time_range})") if prev_sn > 0 else "없음"}

[이전+현재 슬라이드 내용 (슬라이드 텍스트 + 강의자 발화)]
{transcript_block}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
이전 단계에서 발견된 이슈
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

발화 원문: {claim}
지적 내용: {issue_desc}
제안된 정보: {correct_info}
이전 설명: {explanation}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
재검증 기준
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

슬라이드 전체 맥락을 보고 아래를 확인하세요:

1. **슬라이드의 정확한 조건 확인**
   슬라이드에 있는 코드/수식/다이어그램의 구체적 조건을 정확히 확인하세요.
   발화가 슬라이드의 특정 상황을 설명하는 것이면, 그 상황에서 발화가 맞는지 판단하세요.
   claim 또는 이전 단계의 지적이 발화 원문보다 넓게 일반화되었다면, 발화 원문과 주변 문맥의 범위로 다시 좁혀 판단하세요.
   예: 슬라이드의 코드에 특정 키워드가 없는 상태라면, "이 기능을 사용할 수 없다"는 맞는 설명일 수 있음
   예: 슬라이드의 수식에 특정 전제 조건이 있다면, 발화가 그 전제 하에서 설명하는 것일 수 있음
   **슬라이드에 수치 테이블/데이터가 있으면 직접 계산하여 발화와 비교하세요.**
   예: 발화가 "A 집단이 B 집단보다 크다"고 할 때, 슬라이드 데이터의 A/B 평균을 직접 계산하여 발화가 맞는지 확인하세요.

2. **전사 전문의 전후 맥락 확인**
   강의자가 이후 발화에서 정정하거나 보충 설명했는지 확인하세요.

3. **슬라이드-발화 일치 여부**
   슬라이드 텍스트와 발화가 같은 내용을 말하고 있으면 → 발화는 맞는 설명
   슬라이드 텍스트와 발화가 직접 모순되면 → 이슈 유지
   슬라이드 텍스트 기준으로 약어차이, 영어-한글 혼동되어 잡히는 문제점이라면 이슈에서 탈락.

4. **배경 설명/상대 시점/애매한 지시어 재판단**
   - 복잡한 역사적·사회적·경제적 현상을 설명하면서 여러 요인 중 하나를 대표 축으로 든 경우,
     배타적 단정이 없다면 오류로 보지 마세요.
   - "최근/요즘/현재/당시/주류/추세" 같은 상대 시점 표현은 강의 발화 시점 기준으로 해석하세요.
     녹화 시점을 알 수 없으면 그것만으로 오류를 유지하지 마세요.
   - "하나는/이것/그것/마찬가지고"처럼 선행사가 애매한 표현은
     슬라이드 전체 맥락을 봐도 단일 해석이 확실할 때만 이슈를 유지하세요.

5. **설명적 요약과 배타적 단정 구분**
   - 강의자가 복잡한 현상, 역사, 이론, 제도, 학파의 입장을 설명하면서 하나의 설명 축이나 대표 요인을 말하는 경우,
     그것이 더 완전한 설명이 아니라고 해서 곧바로 오류가 되지는 않습니다.
   - 여러 요인 중 하나를 설명한 것과, 그 요인이 유일한 원인이라고 단정한 것은 다릅니다.
     "오직", "전적으로", "유일한 원인", "~만으로", "반드시 이것 때문이다" 같은 배타적 단정이 없다면
     기본적으로 설명적 요약으로 해석하세요.
   - 단지 더 포괄적이고 더 정확한 보충 설명이 가능하다는 이유만으로 이슈를 유지하지 마세요.
   - 슬라이드가 같은 교육적 단순화를 명시하고 발화가 그 슬라이드를 설명하는 경우, 고급 예외만으로 이슈를 유지하지 마세요.

6. **학파/이론/관점 소개 구간 처리**
   - 슬라이드나 직전 발화가 특정 학파, 이론, 관점, 설명틀을 소개하고 있다면,
     바로 뒤의 발화는 그 관점 안에서 해석하세요.
   - 매 문장마다 "라고 본다", "라고 주장한다", "~의 관점에서는" 같은 한정 표현이 반복되지 않았더라도,
     주변 맥락이 이미 그 관점을 세워주고 있으면 이를 사실 오류로 보지 마세요.
   - 발화의 유일한 문제점이 "관점/주장임을 더 명시적으로 말하지 않았다"는 것뿐이라면,
     학생이 전체 맥락상 올바르게 이해할 수 있는지 먼저 판단하고, 그렇다면 이슈를 기각하세요.

7. **일반화된 표현과 구체적 표기의 관계**
   - 슬라이드가 종(species), 서브타입, 하위 분류, 구체 예시를 제시하고, 발화는 그보다 상위 범주나 더 일반적인 명칭으로 설명할 수 있습니다.
   - 이 경우 발화의 일반적 표현이 슬라이드의 구체 정보와 양립 가능하면 이슈를 유지하지 마세요.
   - "더 구체적으로 말할 수 있었다", "세부 종명을 다 말하지 않았다", "예시를 하나만 말하지 않았다"는 이유만으로는 factual_error가 아닙니다.
   - 일반적 표현이 잘못된 상위 범주이거나, 슬라이드의 구체 대상을 배제하거나, 직접 모순될 때만 이슈를 유지하세요.

결론:
- 슬라이드 전체 맥락을 봐도 여전히 틀리면 → "valid": true
- 슬라이드 맥락을 보면 실제로는 맞는 설명이었으면 → "valid": false
- supporting_slide_status:
  - "supports": 슬라이드가 발화/claim을 뒷받침함
  - "contradicts": 슬라이드가 발화/claim과 직접 충돌함
  - "neutral": 슬라이드가 직접 뒷받침하거나 반박하지 않음
  - "insufficient": 슬라이드 텍스트/발화 맥락만으로 판단 부족
- error_origin:
  - "speech_error": 슬라이드는 맞지만 발화가 잘못됨
  - "slide_error": 슬라이드 자체 설명이 잘못됨
  - "slide_speech_mismatch": 슬라이드와 발화가 서로 충돌함
  - "unclear": 원인을 특정하기 어려움

응답 (JSON만):
```json
{{
  "valid": true | false,
  "reason": "슬라이드 맥락을 참고한 판단 이유",
  "supporting_slide_status": "supports" | "contradicts" | "neutral" | "insufficient",
  "error_origin": "speech_error" | "slide_error" | "slide_speech_mismatch" | "unclear"
}}
```

JSON 외 텍스트를 출력하지 마세요.
"""


def _coerce_bool(value, default: bool = True) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value or "").strip().lower()
    if text in {"true", "yes", "1", "valid"}:
        return True
    if text in {"false", "no", "0", "invalid", "rejected"}:
        return False
    return default


def _normalize_choice(value: str, allowed: set[str], default: str) -> str:
    normalized = str(value or "").strip().lower()
    return normalized if normalized in allowed else default


def _slide_recheck_issue(
    issue: dict,
    slide_ctx: dict,
    slides: list[dict],
    hint: dict,
) -> tuple[dict, dict]:
    from . import claim_common as cv

    prompt = _build_slide_recheck_prompt(issue, slide_ctx, slides, hint)

    try:
        text, call_usage = cv._call_llm(
            prompt,
            max_tokens=2048,
            temperature=0.0,
            thinking_budget=1024,
            stage="recheck",
        )
        payload = json.loads(cv._strip_json_fence(text.strip()))
        valid = _coerce_bool(payload.get("valid", True), default=True)
        reason = str(payload.get("reason", "") or "")
        issue["slide_recheck_status"] = "completed"
        issue["slide_recheck_valid"] = valid
        issue["slide_recheck_reason"] = reason
        issue["supporting_slide_status"] = _normalize_choice(
            payload.get("supporting_slide_status"),
            _SLIDE_SUPPORT_STATUSES,
            "insufficient",
        )
        issue["error_origin"] = _normalize_choice(
            payload.get("error_origin"),
            _ERROR_ORIGINS,
            "unclear",
        )
        issue["slide_recheck_api_failed"] = False
        token_usage = cv._empty_token_usage()
        cv._add_call_usage(token_usage, call_usage)
        return issue, token_usage
    except Exception as e:
        issue["slide_recheck_status"] = "failed"
        issue["slide_recheck_valid"] = True
        issue["slide_recheck_reason"] = f"재검증 호출 실패: {e}"
        issue["supporting_slide_status"] = "insufficient"
        issue["error_origin"] = "unclear"
        issue["slide_recheck_api_failed"] = True
        return issue, cv._empty_token_usage()


def slide_recheck_all_issues(
    issues: list[dict],
    slide_ctx: dict,
    slides: list[dict],
    hint: dict,
    max_workers: int = 4,
) -> tuple[list[dict], list[dict], int, int, dict]:
    """모든 이슈를 슬라이드 맥락으로 재검증, 통과/기각 분류."""
    from . import claim_common as cv

    if not issues:
        return [], [], 0, 0, cv._empty_token_usage()

    print(f"\n  ── 3단계: 슬라이드 맥락 재검증 ({len(issues)}건) ──")
    verified, rejected = [], []
    api_calls = 0
    failed_calls = 0
    token_usage = cv._empty_token_usage()

    def process(i, issue):
        claim_preview = str(issue.get("claim_text", issue.get("problematic_content", "")))[:50]
        print(f"    재검증 [{i+1}/{len(issues)}] 슬라이드 {issue.get('slide_number', '?')} | {claim_preview}...")
        return _slide_recheck_issue(issue.copy(), slide_ctx, slides, hint)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(process, i, iss): i for i, iss in enumerate(issues)}
        for f in as_completed(futures):
            api_calls += 1
            try:
                result, call_usage = f.result()
                token_usage = cv._merge_token_usage(token_usage, call_usage)
                if result.get("slide_recheck_api_failed"):
                    failed_calls += 1
                if result.get("slide_recheck_valid") is False:
                    rejected.append(result)
                    print(f"      ❌ 기각: {result.get('slide_recheck_reason', '')[:80]}")
                else:
                    verified.append(result)
                    print(f"      ✅ 유지")
            except Exception as e:
                idx = futures[f]
                issue_copy = issues[idx].copy()
                issue_copy["slide_recheck_status"] = "failed"
                issue_copy["slide_recheck_valid"] = True
                issue_copy["slide_recheck_reason"] = f"재검증 실패: {e}"
                issue_copy["supporting_slide_status"] = "insufficient"
                issue_copy["error_origin"] = "unclear"
                issue_copy["slide_recheck_api_failed"] = True
                verified.append(issue_copy)
                failed_calls += 1

    verified.sort(key=lambda x: float(x.get("start_time", 0) or 0))
    rejected.sort(key=lambda x: float(x.get("start_time", 0) or 0))
    print(f"  슬라이드 재검증 결과: {len(verified)}건 유지, {len(rejected)}건 기각")
    return verified, rejected, api_calls, failed_calls, token_usage
