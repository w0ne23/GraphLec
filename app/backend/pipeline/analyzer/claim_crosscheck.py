from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

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


def _pack_crosscheck_payload(payload: dict) -> dict:
    result = {
        "verdict": str(payload.get("verdict", "") or "").lower().strip(),
        "reason": str(payload.get("reason", "") or "").strip(),
    }
    for field in _CROSSCHECK_EXTRA_FIELDS:
        value = str(payload.get(field, "") or "").strip()
        if value:
            result[field] = value
    return result


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
                result = _pack_crosscheck_payload(payload)
                if result["verdict"] in _CROSSCHECK_VERDICTS:
                    return result

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
            result = _pack_crosscheck_payload(payload)
            if result["verdict"] in _CROSSCHECK_VERDICTS:
                return result

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

            rows = payload.get("verdicts") if isinstance(payload, dict) else payload
            if not isinstance(rows, list):
                continue

            parsed: dict[str, dict] = {}
            for row in rows:
                if not isinstance(row, dict):
                    continue
                issue_id = str(row.get("issue_id", "") or row.get("id", "") or "").strip()
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
            start = float(u.get("start_time", 0) or 0)
            uid = str(u.get("utterance_id", "") or "").strip()
            text = str(u.get("text", "") or "").strip()
            if text:
                transcript_lines.append(f"  {uid} [{start:.1f}s] {text}")
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
    return (
        f"### {issue_id}\n"
        f"- utterance_id: {issue.get('utterance_id', '')}\n"
        f"- issue_unit_id: {issue.get('issue_unit_id') or issue.get('canonical_issue_id') or ''}\n"
        f"- 유형: {issue_label} ({issue_type})\n"
        f"- claim: {issue.get('claim_text', '')}\n"
        f"- 문제: {issue.get('issue', '')}"
        f"{source_block}"
    )


def judge_single_claim(issue: dict, ctx: dict) -> tuple[dict, dict]:
    """단일 이슈를 재검증하여 agree/disagree/inconclusive 반환."""
    from . import claim_common as cv

    uid = issue.get("utterance_id", "")
    claim_text = issue.get("claim_text", "")
    issue_desc = issue.get("issue", "")
    issue_type = cv.normalize_issue_type(issue.get("type", ""))
    issue_label = issue.get("issue_type_label") or cv.issue_type_label(issue_type)

    utterances = ctx["utterances"]
    slides = ctx["slides"]
    slide_ctx = ctx["slide_ctx"]
    hint = ctx["hint"]
    current_date = ctx["current_date"]

    utt_map = {u["utterance_id"]: (i, u) for i, u in enumerate(utterances)}
    if uid not in utt_map:
        return {"verdict": "inconclusive", "reason": "utterance_id를 찾을 수 없음"}, cv._empty_token_usage()

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
    transcript_block = _build_multi_slide_transcript_block(slides, slide_numbers, utterances)
    context_text = _crosscheck_context_text(
        f"{title} ({time_range})",
        (prev_title + f" ({prev_time_range})") if prev_slide_num > 0 else "없음",
        transcript_block,
    )

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
- 유형: {issue_label} ({issue_type})
- claim: {claim_text}
- 문제: {issue_desc}

## 판정 절차
이전 judge의 문제 제기를 그대로 믿지 말고, 아래 순서로 다시 판단하세요.

중요: 실재 대상의 구체 수치/비율/연도/규모를 다루는 이슈에서는 "핵심 설명용 예시라서 학생이 암기하지 않을 것"만으로
disagree하지 마세요. 강의의 핵심이 다른 개념이어도, 현실 대상에 붙은 수치가 틀리거나 오래되었으면 교수에게 확인 대상으로
올릴 수 있습니다. 이 경우 문맥이 해결했다는 판단은 "해당 숫자가 임의값/가상값/변수값이라고 명시됨" 또는
"같은 문맥에서 정확한 값이나 최신 값으로 바로 정정됨"일 때만 가능합니다.

1. **문제 제기가 전제하는 잘못된 명제 재구성**
   이 지적이 맞으려면 학생이 어떤 잘못된 명제를 외워야 하는지 한 문장으로 재구성하세요.
   그 명제가 구체적으로 재구성되지 않으면 disagree입니다.

2. **그 잘못된 명제가 원문+문맥에 실제로 남아 있는지 확인**
   원문 발화, 슬라이드 텍스트, 앞뒤 발화가 함께 학생에게 남기는 의미를 판단하세요.
   주변 문맥이나 슬라이드가 설명을 보충해서 대상, 관계, 순서, 주체, 조건, 범위가 충분히 이해되고,
   그 결과 학생이 잘못된 명제로 외울 가능성이 낮아지면 disagree입니다.
   완전한 정정 문장이 없더라도, 앞뒤 발화나 슬라이드 구조가 생략된 주어/대상/관계/범위를 자연스럽게 지지하면
   그 문맥상 지지되는 해석을 우선하세요. 문제가 되는 해석이 더 약하거나 과도한 추측이면 disagree입니다.
   단, 실재하는 대상, 집단, 기관, 지역, 제품, 시장, 인구, 통계값 등에 구체적인 수치/비율/연도/규모가 붙은 경우에는
   예시 문맥이라는 이유만으로 disagree하지 마세요. 그 수치가 실제와 다르거나 오래되었고 학생이 예시 수치로 받아들일 수 있으면
   agree 또는 inconclusive로 유지하세요. 명시적으로 가상의 대상/임의의 숫자/변수 예시라고 밝힌 경우에만 이 이유로 기각할 수 있습니다.
   단, 앞에서 올바른 설명이 한 번 나왔거나 슬라이드에 관련 키워드가 있다는 이유만으로 자동 해소하지 마세요.
   뒤따르는 발화 흐름이 다시 다른 오해를 만들면 그 흐름 기준으로 판단하세요.
   특정 단어만 떼어내야 문제가 생기면 disagree이고, 발화 흐름 전체가 같은 오해를 남기면 agree 또는 inconclusive입니다.

3. **문맥 보충 판단**
   슬라이드와 전사문은 서로 보완 근거입니다. 둘을 함께 봤을 때 학생이 자연스럽게 이해할 최종 의미를 판단하세요.
   문맥이 문제 발화를 보충해 오해 가능성을 실질적으로 낮추면 disagree입니다.
   문맥과 슬라이드가 오히려 같은 혼동을 반복하거나, 서로 다른 설명이 충돌해 학생이 잘못된 관계/순서/주체를 외울 수 있으면 agree 또는 inconclusive입니다.
   보충 또는 충돌의 근거는 reason이나 evidence_in_context에 실제 발화나 슬라이드 표현으로 설명하세요.
   생략된 주어/지시어가 있는 경우에는 먼저 "문맥상 가장 자연스러운 대상"과 그 근거를 찾으세요.
   그 자연스러운 해석에서는 발화가 맞고, 문제 제기 쪽 해석이 특정 단어나 직전 용어를 과도하게 잡아당겨야만 성립하면 disagree입니다.

4. **판단 기준의 수준 맞추기**
   반박 근거는 학생이 이 강의 구간에서 배우는 개념 수준과 맞아야 합니다.
   강의가 설명하지 않는 세부 구현, 특수 상황, 예외적 전제만으로는 issue를 유지하지 마세요.
   반대로 문제 제기가 같은 수준의 기본 개념, 관계, 순서, 주체, 조건, 범위 혼동을 지적한다면 issue를 유지하세요.
   단순한 세부 생략이나 표현 취향이 아니라, 학생이 실제로 잘못 외울 명제가 남는지가 기준입니다.

5. **유형별 확인**
   - factual_error/temporal_error: 발화가 실제로 틀린 사실/현재성 주장을 남겼을 때만 agree입니다.
     현실 대상에 붙은 예시 수치가 실제와 다르거나 오래된 경우도 factual_error/temporal_error 후보입니다.
     예시라는 점은 severity나 professor_check 여부를 낮출 수는 있지만, 그 자체로 기각 사유는 아닙니다.
     "학생이 암기하지 않을 것"이라는 추정만으로는 disagree하지 말고, 정확한 값 확인이 필요하면 inconclusive로 유지하세요.
   - confusing_explanation: 학생이 서로 다른 개념, 주체, 과정, 조건을 같은 것으로 외우게 되는 구체 명제가 남을 때만 agree입니다.
     단, "얘/이거/그거/이 값/해당 항목" 같은 지시어를 특정 선행사 하나로 강제 해석해야만 문제가 생기고,
     원문 발화와 슬라이드를 함께 보면 맞는 설명이면 disagree입니다.
   - scope_overclaim: 발화와 문맥이 학생에게 닫힌 범위의 명제로 남을 때만 agree입니다.
   - 더 자세히 말할 수 있다는 정도, 더 엄밀한 표현 가능성, 표현 취향만으로는 agree하지 마세요.

## verdict 기준
- agree: 원문+슬라이드+주변 발화를 함께 봐도 학생이 잘못 외울 명제가 실제로 남습니다.
- disagree: 문맥과 슬라이드가 설명을 충분히 보충해 오해 가능성이 낮거나, 문제 제기가 특정 단어/지시어를 과확장한 것입니다.
- inconclusive: 오해 가능성은 있지만 문맥만으로 확정하기 어려워 교수 확인 대상으로 둘 필요가 있습니다.

agree 또는 inconclusive로 판단하는 경우에는 교수에게 보여줄 수 있는 설명 필드도 작성하세요.
disagree로 판단하는 경우에는 reason에 기각 이유를 쓰고 나머지 설명 필드는 비워도 됩니다.
inconclusive는 확정 표현을 피하고, "교수 확인 후보" 관점으로 작성하세요.

응답 규칙:
- 반드시 JSON object 하나만 출력
- markdown/code fence 금지
- verdict 값은 agree / disagree / inconclusive 중 하나만 사용

응답 예시:
{{
  "verdict": "agree",
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
}}"""

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
                max_tokens=2048,
                thinking_budget=1024,
                response_format=response_format,
                stage="cross_recheck",
            )
            cv._add_call_usage(token_usage, call_usage)
            payload = _parse_crosscheck_payload(text)
            verdict = str(payload.get("verdict", "") or "").lower().strip()
            if verdict not in _CROSSCHECK_VERDICTS:
                return {"verdict": "inconclusive", "reason": f"알 수 없는 verdict: {verdict}"}, token_usage
            payload["crosscheck_context_text"] = context_text
            return payload, token_usage
        except Exception as e:
            last_error = e
            if attempt < _CROSSCHECK_PARSE_RETRIES:
                print(f"    ↺ 교차 재검증 JSON 파싱 재시도 ({attempt+1}/{_CROSSCHECK_PARSE_RETRIES})")

    return {
        "verdict": "inconclusive",
        "reason": f"교차 재검증 실패: {last_error}",
        "crosscheck_context_text": context_text,
    }, token_usage


def judge_claim_batch(issues: list[dict], ctx: dict) -> tuple[dict[str, dict], dict]:
    """같은 슬라이드 문맥의 여러 이슈를 한 번에 crosscheck한다."""
    from . import claim_common as cv

    if not issues:
        return {}, cv._empty_token_usage()

    utterances = ctx["utterances"]
    hint = ctx["hint"]
    utt_map = {u["utterance_id"]: (i, u) for i, u in enumerate(utterances)}

    valid: list[tuple[str, dict]] = []
    payloads: dict[str, dict] = {}
    for idx, issue in enumerate(issues, 1):
        issue_id = f"i{idx:04d}"
        if issue.get("utterance_id", "") not in utt_map:
            payloads[issue_id] = {"verdict": "inconclusive", "reason": "utterance_id를 찾을 수 없음"}
            continue
        valid.append((issue_id, issue))

    if not valid:
        return payloads, cv._empty_token_usage()

    _, first_utt = utt_map[valid[0][1].get("utterance_id", "")]
    slide_num = int(first_utt.get("slide_number", 0) or 0)
    target_label, prev_label, transcript_block = _crosscheck_slide_context(ctx, slide_num)
    context_text = _crosscheck_context_text(target_label, prev_label, transcript_block)
    issue_block = "\n\n".join(_issue_line_for_batch(issue_id, issue) for issue_id, issue in valid)

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

## 지적 목록
{issue_block}

## 판정 절차
각 issue_id는 하나의 claim이 아니라 같은 오해를 만들 수 있는 문맥 단위 issue일 수 있습니다.
issue 안에 묶인 발화/claim이 여러 개 있으면, 개별 문장 하나가 아니라 그 발화 흐름 전체가 학생에게 남기는
잘못된 명제 또는 오해를 판단하세요.
서로 다른 issue_id끼리는 독립적으로 판단하세요. 새로운 이슈를 만들지 말고, 제공된 issue_id 각각에 대해서만 verdict를 작성하세요.

중요: 실재 대상의 구체 수치/비율/연도/규모를 다루는 이슈에서는 "핵심 설명용 예시라서 학생이 암기하지 않을 것"만으로
disagree하지 마세요. 강의의 핵심이 다른 개념이어도, 현실 대상에 붙은 수치가 틀리거나 오래되었으면 교수에게 확인 대상으로
올릴 수 있습니다. 이 경우 문맥이 해결했다는 판단은 "해당 숫자가 임의값/가상값/변수값이라고 명시됨" 또는
"같은 문맥에서 정확한 값이나 최신 값으로 바로 정정됨"일 때만 가능합니다.

1. **문제 제기가 전제하는 잘못된 명제 재구성**
   이 지적이 맞으려면 학생이 어떤 잘못된 명제를 외워야 하는지 한 문장으로 재구성하세요.
   그 명제가 구체적으로 재구성되지 않으면 disagree입니다.

2. **그 잘못된 명제가 원문+문맥에 실제로 남아 있는지 확인**
   원문 발화, 묶인 발화 흐름, 슬라이드 텍스트, 앞뒤 발화가 함께 학생에게 남기는 의미를 판단하세요.
   주변 문맥이나 슬라이드가 설명을 보충해서 대상, 관계, 순서, 주체, 조건, 범위가 충분히 이해되고,
   그 결과 학생이 잘못된 명제로 외울 가능성이 낮아지면 disagree입니다.
   완전한 정정 문장이 없더라도, 앞뒤 발화나 슬라이드 구조가 생략된 주어/대상/관계/범위를 자연스럽게 지지하면
   그 문맥상 지지되는 해석을 우선하세요. 문제가 되는 해석이 더 약하거나 과도한 추측이면 disagree입니다.
   단, 실재하는 대상, 집단, 기관, 지역, 제품, 시장, 인구, 통계값 등에 구체적인 수치/비율/연도/규모가 붙은 경우에는
   예시 문맥이라는 이유만으로 disagree하지 마세요. 그 수치가 실제와 다르거나 오래되었고 학생이 예시 수치로 받아들일 수 있으면
   agree 또는 inconclusive로 유지하세요. 명시적으로 가상의 대상/임의의 숫자/변수 예시라고 밝힌 경우에만 이 이유로 기각할 수 있습니다.
   단, 앞에서 올바른 설명이 한 번 나왔거나 슬라이드에 관련 키워드가 있다는 이유만으로 자동 해소하지 마세요.
   뒤따르는 발화 흐름이 다시 다른 오해를 만들면 그 흐름 기준으로 판단하세요.
   특정 단어만 떼어내야 문제가 생기면 disagree이고, 발화 흐름 전체가 같은 오해를 남기면 agree 또는 inconclusive입니다.
   슬라이드 텍스트가 판단에 영향을 주면 reason/evidence_in_context에서 실제 표현을 들어 반영하세요.

3. **문맥 보충 판단**
   슬라이드와 전사문은 서로 보완 근거입니다. 둘을 함께 봤을 때 학생이 자연스럽게 이해할 최종 의미를 판단하세요.
   문맥이 문제 발화를 보충해 오해 가능성을 실질적으로 낮추면 disagree입니다.
   문맥과 슬라이드가 오히려 같은 혼동을 반복하거나, 서로 다른 설명이 충돌해 학생이 잘못된 관계/순서/주체를 외울 수 있으면 agree 또는 inconclusive입니다.
   보충 또는 충돌의 근거는 reason이나 evidence_in_context에 실제 발화나 슬라이드 표현으로 설명하세요.
   생략된 주어/지시어가 있는 경우에는 먼저 "문맥상 가장 자연스러운 대상"과 그 근거를 찾으세요.
   그 자연스러운 해석에서는 발화가 맞고, 문제 제기 쪽 해석이 특정 단어나 직전 용어를 과도하게 잡아당겨야만 성립하면 disagree입니다.

4. **판단 기준의 수준 맞추기**
   반박 근거는 학생이 이 강의 구간에서 배우는 개념 수준과 맞아야 합니다.
   강의가 설명하지 않는 세부 구현, 특수 상황, 예외적 전제만으로는 issue를 유지하지 마세요.
   반대로 문제 제기가 같은 수준의 기본 개념, 관계, 순서, 주체, 조건, 범위 혼동을 지적한다면 issue를 유지하세요.
   단순한 세부 생략이나 표현 취향이 아니라, 학생이 실제로 잘못 외울 명제가 남는지가 기준입니다.

5. **유형별 확인**
   - factual_error/temporal_error: 발화가 실제로 틀린 사실/현재성 주장을 남겼을 때만 agree입니다.
     현실 대상에 붙은 예시 수치가 실제와 다르거나 오래된 경우도 factual_error/temporal_error 후보입니다.
     예시라는 점은 severity나 professor_check 여부를 낮출 수는 있지만, 그 자체로 기각 사유는 아닙니다.
     "학생이 암기하지 않을 것"이라는 추정만으로는 disagree하지 말고, 정확한 값 확인이 필요하면 inconclusive로 유지하세요.
   - confusing_explanation: 학생이 서로 다른 개념, 주체, 과정, 조건을 같은 것으로 외우게 되는 구체 명제가 남을 때만 agree입니다.
     단, "얘/이거/그거/이 값/해당 항목" 같은 지시어를 특정 선행사 하나로 강제 해석해야만 문제가 생기고,
     원문 발화와 슬라이드를 함께 보면 맞는 설명이면 disagree입니다.
   - scope_overclaim: 발화와 문맥이 학생에게 닫힌 범위의 명제로 남을 때만 agree입니다.
   - 더 자세히 말할 수 있다는 정도, 더 엄밀한 표현 가능성, 표현 취향만으로는 agree하지 마세요.

## verdict 기준
- agree: 원문+슬라이드+주변 발화를 함께 봐도 학생이 잘못 외울 명제가 실제로 남습니다.
- disagree: 문맥과 슬라이드가 설명을 충분히 보충해 오해 가능성이 낮거나, 문제 제기가 특정 단어/지시어를 과확장한 것입니다.
- inconclusive: 오해 가능성은 있지만 문맥만으로 확정하기 어려워 교수 확인 대상으로 둘 필요가 있습니다.

agree 또는 inconclusive로 판단하는 경우에는 교수에게 보여줄 수 있는 설명 필드도 작성하세요.
disagree로 판단하는 경우에는 reason에 기각 이유를 쓰고 나머지 설명 필드는 비워도 됩니다.

응답 규칙:
- 반드시 JSON object 하나만 출력
- markdown/code fence 금지
- verdict 값은 agree / disagree / inconclusive 중 하나만 사용
- 입력된 모든 issue_id에 대해 정확히 하나의 verdict를 출력

응답 예시:
{{
  "verdicts": [
    {{
      "issue_id": "i0001",
      "verdict": "disagree",
      "reason": "원문과 문맥 기준의 판단 이유"
    }},
    {{
      "issue_id": "i0002",
      "verdict": "agree",
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
    dynamic_marker = "## 도메인"
    instruction_marker = "## 판정 절차"
    dynamic_start = prompt.find(dynamic_marker)
    instruction_start = prompt.find(instruction_marker)
    system_prompt = None
    if 0 <= dynamic_start < instruction_start:
        system_prompt = prompt[:dynamic_start].rstrip() + "\n\n" + prompt[instruction_start:].lstrip()
        prompt = prompt[dynamic_start:instruction_start].strip()

    model = str(cv._resolve_stage_model("cross_recheck") or "").strip()
    response_format = {"type": "json_object"} if (
        model.startswith("gpt") or model.startswith("o1") or model.startswith("o3")
    ) else None

    token_usage = cv._empty_token_usage()
    issue_ids = {issue_id for issue_id, _ in valid}
    parsed: dict[str, dict] = {}
    last_error = None
    for attempt in range(_CROSSCHECK_PARSE_RETRIES + 1):
        try:
            text, call_usage = cv._call_llm(
                prompt,
                system_prompt=system_prompt,
                max_tokens=min(8192, max(2048, 900 * len(valid))),
                thinking_budget=1024,
                response_format=response_format,
                stage="cross_recheck",
            )
            cv._add_call_usage(token_usage, call_usage)
            parsed = _parse_crosscheck_batch_payload(text, issue_ids)
            for payload in parsed.values():
                payload.setdefault("crosscheck_context_text", context_text)
            break
        except Exception as e:
            last_error = e
            if attempt < _CROSSCHECK_PARSE_RETRIES:
                print(f"    ↺ 교차 재검증 batch JSON 파싱 재시도 ({attempt+1}/{_CROSSCHECK_PARSE_RETRIES})")

    missing = [pair for pair in valid if pair[0] not in parsed]
    for issue_id, issue in missing:
        try:
            payload, call_usage = judge_single_claim(issue, ctx)
        except Exception as e:
            payload, call_usage = {"verdict": "inconclusive", "reason": f"교차 재검증 실패: {e}"}, cv._empty_token_usage()
        cv._add_call_usage(token_usage, call_usage)
        payload.setdefault("crosscheck_context_text", context_text)
        parsed[issue_id] = payload

    if last_error and not parsed:
        for issue_id, _ in valid:
            parsed[issue_id] = {
                "verdict": "inconclusive",
                "reason": f"교차 재검증 batch 실패: {last_error}",
                "crosscheck_context_text": context_text,
            }

    payloads.update(parsed)
    return payloads, token_usage


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
   슬라이드에 있는 구체적 조건, 표기, 구조, 자료 범위를 정확히 확인하세요.
   발화가 슬라이드의 특정 상황을 설명하는 것이면, 그 상황에서 발화가 맞는지 판단하세요.
   claim 또는 이전 단계의 지적이 발화 원문보다 넓게 일반화되었다면, 발화 원문과 주변 문맥의 범위로 다시 좁혀 판단하세요.
   **슬라이드에 정량 자료나 비교 기준이 있으면 직접 확인하여 발화와 비교하세요.**

2. **전사 전문의 전후 맥락 확인**
   강의자가 이후 발화에서 정정하거나 보충 설명했는지 확인하세요.

3. **슬라이드-발화 일치 여부**
   슬라이드 텍스트와 발화가 같은 내용을 말하고 있으면 → 발화는 맞는 설명
   슬라이드 텍스트와 발화가 직접 모순되면 → 이슈 유지
   슬라이드 텍스트 기준으로 약어차이, 영어-한글 혼동되어 잡히는 문제점이라면 이슈에서 탈락.

4. **배경 설명/상대 시점/애매한 지시어 재판단**
   - 복잡한 현상을 설명하면서 여러 요인 중 하나를 대표 축으로 든 경우,
     배타적 단정이 없다면 오류로 보지 마세요.
   - "최근/요즘/현재/당시/주류/추세" 같은 상대 시점 표현은 강의 발화 시점 기준으로 해석하세요.
     녹화 시점을 알 수 없으면 그것만으로 오류를 유지하지 마세요.
   - "하나는/이것/그것/마찬가지고"처럼 선행사가 애매한 표현은
     슬라이드 전체 맥락을 봐도 단일 해석이 확실할 때만 이슈를 유지하세요.
   - 원문 발화와 슬라이드가 함께 보면 맞는 설명인데, 이전 단계의 claim 해소가 지시어를 특정 선행사로 과하게 확정해서
     오류처럼 보이는 경우는 이슈를 유지하지 마세요.
   - 완전한 정정 문장이 없더라도, 슬라이드 구조나 앞뒤 발화가 생략된 주어/대상/관계/범위를 자연스럽게 지지하면
     그 문맥상 지지되는 해석을 우선하세요.

5. **설명적 요약과 배타적 단정 구분**
   - 강의자가 복잡한 현상, 이론, 제도, 관점을 설명하면서 하나의 설명 축이나 대표 요인을 말하는 경우,
     그것이 더 완전한 설명이 아니라고 해서 곧바로 오류가 되지는 않습니다.
   - 여러 요인 중 하나를 설명한 것과, 그 요인이 유일한 원인이라고 단정한 것은 다릅니다.
     "오직", "전적으로", "유일한 원인", "~만으로", "반드시 이것 때문이다" 같은 배타적 단정이 없다면
     기본적으로 설명적 요약으로 해석하세요.
   - 단지 더 포괄적이고 더 정확한 보충 설명이 가능하다는 이유만으로 이슈를 유지하지 마세요.
   - 슬라이드가 같은 교육적 단순화를 명시하고 발화가 그 슬라이드를 설명하는 경우, 강의 수준 밖의 세부 예외만으로 이슈를 유지하지 마세요.
   - 강의 수준에 맞춘 큰 그림 설명, 비유, 세부 계층 생략은 핵심 결론이 맞으면 이슈를 유지하지 마세요.
   - 다만 실재 대상에 붙은 구체 수치/비율/연도/규모가 실제와 다르거나 오래된 경우에는,
     예시 문맥이라는 이유만으로 이슈를 기각하지 마세요. 명시적으로 가상의 대상/임의 숫자/변수 예시라고 밝힌 경우만 제외합니다.

6. **이론/관점 소개 구간 처리**
   - 슬라이드나 직전 발화가 특정 이론, 관점, 설명틀을 소개하고 있다면,
     바로 뒤의 발화는 그 관점 안에서 해석하세요.
   - 매 문장마다 "라고 본다", "라고 주장한다", "~의 관점에서는" 같은 한정 표현이 반복되지 않았더라도,
     주변 맥락이 이미 그 관점을 세워주고 있으면 이를 사실 오류로 보지 마세요.
   - 발화의 유일한 문제점이 "관점/주장임을 더 명시적으로 말하지 않았다"는 것뿐이라면,
     학생이 전체 맥락상 올바르게 이해할 수 있는지 먼저 판단하고, 그렇다면 이슈를 기각하세요.

7. **일반화된 표현과 구체적 표기의 관계**
   - 슬라이드가 하위 분류, 구체 예시, 특정 표기를 제시하고, 발화는 그보다 상위 범주나 더 일반적인 명칭으로 설명할 수 있습니다.
   - 이 경우 발화의 일반적 표현이 슬라이드의 구체 정보와 양립 가능하면 이슈를 유지하지 마세요.
   - "더 구체적으로 말할 수 있었다", "세부 명칭을 다 말하지 않았다", "예시를 하나만 말하지 않았다"는 이유만으로는 factual_error가 아닙니다.
   - 일반적 표현이 잘못된 상위 범주이거나, 슬라이드의 구체 대상을 배제하거나, 직접 모순될 때만 이슈를 유지하세요.

결론:
- 슬라이드 전체 맥락을 봐도 여전히 틀리면 → "valid": true
- 슬라이드 맥락을 보면 실제로는 맞는 설명이었으면 → "valid": false

응답 (JSON만):
```json
{{
  "valid": true | false,
  "reason": "슬라이드 맥락을 참고한 판단 이유"
}}
```

JSON 외 텍스트를 출력하지 마세요.
"""


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
        valid = payload.get("valid", True)
        reason = str(payload.get("reason", "") or "")
        issue["slide_recheck_valid"] = bool(valid)
        issue["slide_recheck_reason"] = reason
        issue["slide_recheck_api_failed"] = False
        token_usage = cv._empty_token_usage()
        cv._add_call_usage(token_usage, call_usage)
        return issue, token_usage
    except Exception as e:
        issue["slide_recheck_valid"] = True
        issue["slide_recheck_reason"] = f"재검증 호출 실패: {e}"
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
                issue_copy["slide_recheck_valid"] = True
                issue_copy["slide_recheck_reason"] = f"재검증 실패: {e}"
                issue_copy["slide_recheck_api_failed"] = True
                verified.append(issue_copy)
                failed_calls += 1

    verified.sort(key=lambda x: float(x.get("start_time", 0) or 0))
    rejected.sort(key=lambda x: float(x.get("start_time", 0) or 0))
    print(f"  슬라이드 재검증 결과: {len(verified)}건 유지, {len(rejected)}건 기각")
    return verified, rejected, api_calls, failed_calls, token_usage
