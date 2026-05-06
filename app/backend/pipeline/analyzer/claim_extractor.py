from __future__ import annotations

import json
from collections import OrderedDict


_CANONICAL_CLAIM_TYPES = {
    "definition",
    "numeric",
    "causal",
    "relationship",
    "currentness",
}


def _build_slide_and_utterance_context(utterances: list[dict], slide_ctx: dict) -> tuple[str, str]:
    """슬라이드 참조와 발화 카드를 한 번의 순회로 생성."""
    from . import claim_common as cv

    total = len(utterances)
    seen_slides = OrderedDict()
    cards = []

    for i, u in enumerate(utterances):
        current_sn = int(u.get("slide_number", 0) or 0)
        if current_sn not in seen_slides:
            seen_slides[current_sn] = True

        prev_context = utterances[max(0, i - 3):i]
        next_context = utterances[i + 1:min(total, i + 4)]
        same_slide_prev = [x for x in utterances[max(0, i - 6):i] if int(x.get("slide_number", 0) or 0) == current_sn]
        same_slide_next = [x for x in utterances[i + 1:min(total, i + 7)] if int(x.get("slide_number", 0) or 0) == current_sn]

        parts = [
            f"### 대상 발화 {u.get('utterance_id', '?')} (슬라이드 {current_sn})",
            f"[현재 발화]\n{cv._format_utterance_for_prompt(u)}",
        ]
        if prev_context:
            parts.append("[직전 문맥]")
            parts.extend(f"  {cv._format_utterance_for_prompt(x)}" for x in prev_context)
        if next_context:
            parts.append("[직후 문맥]")
            parts.extend(f"  {cv._format_utterance_for_prompt(x)}" for x in next_context)
        if same_slide_prev or same_slide_next:
            parts.append("[같은 슬라이드 추가 문맥]")
            for x in same_slide_prev[-3:]:
                parts.append(f"  {cv._format_utterance_for_prompt(x)}")
            for x in same_slide_next[:3]:
                parts.append(f"  {cv._format_utterance_for_prompt(x)}")
        cards.append("\n".join(parts))

    refs = []
    for sn in seen_slides:
        ctx = slide_ctx.get(sn, {})
        title = ctx.get("title", f"슬라이드 {sn}")
        time_range = ctx.get("time_range", "")
        slide_text = str(ctx.get("slide_text", "") or "")
        if len(slide_text) > 1200:
            slide_text = slide_text[:1200] + "\n... (이하 생략)"
        refs.append(
            f"[슬라이드 {sn}] {title} ({time_range})\n"
            + (slide_text if slide_text else "(슬라이드 텍스트 없음)")
        )

    return "\n\n".join(refs), "\n\n".join(cards)


def _build_extract_prompt(utterances: list[dict], current_date: str, hint: dict, slide_ctx: dict) -> str:
    slide_references, utterance_cards = _build_slide_and_utterance_context(utterances, slide_ctx)

    return f"""당신은 강의 발화에서 검증 가능한 사실 주장(claim)을 추출하는 전문가입니다.
오늘 날짜: {current_date}
강의 도메인: {hint['label']}

아래는 슬라이드 참조 정보와, 각 대상 발화별 로컬 문맥입니다.
강의자는 슬라이드를 보여주면서 발화합니다. 학생은 슬라이드와 발화를 동시에 받습니다.

[슬라이드 참조]
{slide_references}

[대상 발화별 로컬 문맥]
{utterance_cards}

### 추출 대상
- definition: 정의, 의미, 용어, 기호, 개념 설명
- numeric: 수치, 단위, 기준값, 정량적 비교
- causal: 원인, 결과, 메커니즘, 작동 방식
- relationship: 분류, 포함, 비교, 상하위, 대응 관계
- currentness: 현재 시점의 유효성, 현행성, 주류성, 최신성

### 추출 제외
- 의견/감상, 교육적 지시, 구어적 필러
- 단순한 질문 제시만 있고 강의자가 답이나 기준을 제시하지 않은 경우
- "약/대략/정도" 붙은 수치는 근사 claim(is_approximate=true)으로 표시

### 문맥 사용 원칙
- 반드시 **현재 발화 자체가 주장한 내용만** claim으로 추출하세요.
- 직전/직후 문맥은 지시어 해소, 예시 여부 판단, 수사/강조 표현 판별에만 사용하세요.
- 주변 문맥에 있는 더 강한 일반 명제를 현재 발화에 덧씌우지 마세요.
- 현재 발화가 예시/가정/비유/수사적 요약이면, resolved_claim에도 그 범위를 유지하세요.
- 현재 발화가 특정 예시를 설명하는 문장인데 이를 일반 법칙처럼 바꾸지 마세요.
- 특히 "보통", "항상", "모든", "전부", "정해져 있다" 같은 표현은, 문맥상
  예시 설명/강조/요약인지 먼저 확인한 뒤에만 일반 claim으로 추출하세요.

### 지시어 해소 + 슬라이드 반영
- "이것", "여기", "해당 항목", "얘", "이거", "그거" 같은 지시어는 슬라이드/앞뒤 발화를 참고하되,
  **단일 선행사가 확실할 때만** 구체적 대상으로 바꿔 resolved_claim을 작성하세요.
- 지시어가 슬라이드의 기호/행/도형/목록 항목을 가리킬 수도 있고 직전 발화의 용어를 가리킬 수도 있는 경우,
  직전 발화의 용어 하나로 과하게 확정하지 마세요. 슬라이드에 보이는 대상이 더 자연스러우면
  "슬라이드의 σ 기호", "슬라이드의 해당 항목"처럼 시각 자료 기준으로 좁혀 쓰세요.
- 둘 이상의 합리적 해석이 가능하면 특정 개념으로 단정하지 말고, 원문이 실제로 말한 범위만 유지하세요.
  이 경우 resolved_claim에는 "직전 용어 A 자체가 B다"처럼 강한 동일시 문장을 만들지 마세요.
- "그게", "이거", "그래서", "~은요"처럼 바로 앞 발화를 이어받는 조각 문장은 직전 발화를 이용해 주어를 복원할 수 있으면 추출하되,
  복원 결과가 슬라이드 표기나 발화 흐름과 충돌하면 원문 범위를 보존하세요.
- 슬라이드의 구체적 조건, 표기, 구조, 자료 범위도 resolved_claim에 반영하세요.

예시:
- 원문: "얘를 시그마라고 하는 건데 이거는 모수고요."
  슬라이드에 모수 행의 "σ"가 보이면 resolved_claim은 "슬라이드의 σ 기호는 모수 항목이다."처럼 작성하세요.
  직전 발화에 "분산"이 있었다는 이유만으로 "분산의 모수는 σ다"라고 확정하지 마세요.

### 복수 claim 추출
하나의 발화에 여러 주장이 섞여 있으면 반드시 각각 별도 claim으로 추출하세요.
특히 "요즘/현재/최근/추세/주류" 표현은 별도 currentness claim으로 추출하세요.
하나의 발화 안에 수치가 2개 이상 나오면, 각각이 독립적으로 검증 가능한 값인지 확인하고 가능한 한 분리하세요.
문장이 불완전해 보여도 직전 발화와 결합하면 검증 가능한 수치/기준 claim이 되면 추출하세요.

주의: 말실수나 용어 착각으로 보이더라도, 학생이 그대로 믿으면 틀린 지식이 되는 경우는 추출 대상입니다.
주의: 강의자가 예시 상황 안에서 기준값이나 정량적 조건을 말하면, 그 예시 범위 안의 검증 가능한 claim으로 추출하세요.
주의: 문장이 질문형으로 시작하더라도, 뒤에서 강의자가 특정 값이나 기준을 제시하면 그 제시된 값/기준은 claim으로 추출하세요.
주의: 다만 예시 속 기준값을 일반 상식/보편 법칙으로 확대 해석하지 마세요. 예시의 범위가 드러나면 resolved_claim에도 그 예시 범위를 남기세요.

### 출력 (JSON만)
```json
{{
  "claims": [
    {{
      "utterance_id": "U0001",
      "claim_type": "definition",
      "claim_text": "claim 원문 (발화 그대로)",
      "resolved_claim": "지시어 해소 + 슬라이드 조건 반영한 claim",
      "is_approximate": false
    }}
  ]
}}
```

지침:
- 검증 불가능한 주장은 추출하지 마세요.
- 하나의 발화에서 여러 claim이 나올 수 있습니다.
- claim_type은 반드시 `definition`, `numeric`, `causal`, `relationship`, `currentness` 중 하나만 사용하세요.
- verification_question은 생성하지 마세요. 검증 질문은 후속 판정 단계에서 필요한 claim에만 만듭니다.
- claim이 없으면 {{"claims": []}}만 출력하세요.
- JSON 외 텍스트를 출력하지 마세요.
"""


def _normalize_claim_type(value: str) -> str | None:
    raw = str(value or "").strip()
    if not raw:
        return "definition"

    lowered = raw.lower()
    korean = raw.strip()

    if lowered == "no_claim":
        return None

    if lowered in _CANONICAL_CLAIM_TYPES:
        return lowered

    if any(token in lowered for token in ("current", "outdated", "deprecated", "trend", "recent")):
        return "currentness"

    if any(token in lowered for token in ("causal", "cause", "effect", "mechanism")):
        return "causal"

    if (
        any(token in lowered for token in (
            "numeric", "numerical", "number", "quantity", "quantitative",
            "specification", "standard", "example_numeric", "standard/quantity",
        ))
        or korean in {"수치", "정량/기준값", "실무 예시 속 사실 주장"}
    ):
        return "numeric"

    if any(token in lowered for token in ("relation", "relationship", "relational", "comparison", "compare")) or korean == "관계":
        return "relationship"

    return "definition"


def _extract_claims(
    utterances: list[dict],
    current_date: str,
    hint: dict,
    slide_ctx: dict,
) -> tuple[list[dict], bool, int, dict]:
    from . import claim_common as cv

    if not utterances:
        return [], False, 0, cv._empty_token_usage()

    prompt = _build_extract_prompt(utterances, current_date, hint, slide_ctx)
    api_calls = 0
    token_usage = cv._empty_token_usage()

    for attempt in range(cv.VERIFIER_PARSE_RETRIES + 1):
        text, call_usage = cv._call_llm(
            prompt,
            max_tokens=8192,
            temperature=0.0,
            thinking_budget=1024,
            stage="extract",
        )
        api_calls += 1
        cv._add_call_usage(token_usage, call_usage)
        try:
            payload = json.loads(cv._strip_json_fence(text.strip()))
            claims = payload.get("claims", [])
            cleaned = []
            for c in claims:
                if not isinstance(c, dict) or not c.get("utterance_id"):
                    continue
                normalized = _normalize_claim_type(c.get("claim_type"))
                if normalized is None:
                    continue
                c["claim_type"] = normalized
                c.pop("verification_question", None)
                c.pop("verificationQuestion", None)
                cleaned.append(c)
            return cleaned, False, api_calls, token_usage
        except (json.JSONDecodeError, AttributeError):
            if attempt < cv.VERIFIER_PARSE_RETRIES:
                print(f"    ↺ claim 추출 JSON 파싱 재시도 ({attempt+1}/{cv.VERIFIER_PARSE_RETRIES})")

    return [], True, api_calls, token_usage


def recover_claim_extraction(
    utterances: list[dict],
    current_date: str,
    hint: dict,
    slide_ctx: dict,
    label: str,
) -> tuple[list[dict], bool, int, dict, bool]:
    from . import claim_common as cv

    total_api_calls = 0
    total_token_usage = cv._empty_token_usage()
    last_claims: list[dict] = []
    parse_failed = False
    last_exc = None

    for attempt in range(cv.VERIFIER_BATCH_RECOVERY_RETRIES + 1):
        if attempt > 0:
            print(f"    ↺ {label} claim 추출 재처리 ({attempt}/{cv.VERIFIER_BATCH_RECOVERY_RETRIES})")
        try:
            claims, parse_failed, api_calls, token_usage = _extract_claims(
                utterances, current_date, hint, slide_ctx
            )
            total_api_calls += api_calls
            total_token_usage = cv._merge_token_usage(total_token_usage, token_usage)
            last_claims = claims
            if not parse_failed:
                return claims, False, total_api_calls, total_token_usage, True
        except Exception as e:
            last_exc = e

    if last_exc and not last_claims and total_api_calls == 0:
        raise last_exc
    return last_claims, True, total_api_calls, total_token_usage, False


def extract_claims_only(
    utterances: list[dict],
    current_date: str,
    hint: dict,
    slide_ctx: dict,
    batch_size: int | None = None,
) -> tuple[list[tuple], int, dict]:
    """1단계만 실행: claim 추출. (batch_list, total_claims) 반환."""
    from . import claim_common as cv

    if batch_size is None:
        batch_size = cv.BATCH_SIZE

    batches = [utterances[i:i + batch_size] for i in range(0, len(utterances), batch_size)]
    all_claims_by_batch = []
    total_api = 0
    total_token_usage = cv._empty_token_usage()

    for i, batch in enumerate(batches):
        ids = f"{batch[0]['utterance_id']}..{batch[-1]['utterance_id']}"
        print(f"    추출 [{i+1}/{len(batches)}] {ids}")
        claims, parse_failed, api_calls, token_usage, ok = recover_claim_extraction(
            batch, current_date, hint, slide_ctx, f"배치 {i+1} {ids}"
        )
        all_claims_by_batch.append((batch, claims))
        total_api += api_calls
        total_token_usage = cv._merge_token_usage(total_token_usage, token_usage)

    total_claims = sum(len(c) for _, c in all_claims_by_batch)
    print(f"  추출된 claim: {total_claims}개")
    return all_claims_by_batch, total_api, total_token_usage
