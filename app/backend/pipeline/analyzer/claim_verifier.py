from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed


JUDGE_MIN_CONFIDENCE = 0.55


def _issue_judge_min_confidence() -> float:
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
        or claim.get("utterance_id")
        or ""
    ).strip()


def _context_id(claim: dict) -> str:
    return str(claim.get("context_id") or claim.get("utterance_id") or "").strip()


def _context_ids(claim: dict) -> list[str]:
    values = claim.get("context_ids")
    if isinstance(values, list):
        ids = [str(value).strip() for value in values if str(value).strip()]
        if ids:
            return ids
    cid = _context_id(claim)
    return [cid] if cid else []


def _related_context_ids(claim: dict) -> list[str]:
    ids = []
    for key in ("antecedent_context_ids", "context_ids", "utterance_ids"):
        values = claim.get(key)
        if isinstance(values, list):
            ids.extend(str(value).strip() for value in values if str(value).strip())
    cid = _context_id(claim)
    if cid:
        ids.append(cid)
    return list(dict.fromkeys(ids))


def _format_utterance_for_prompt(u: dict) -> str:
    uid = u["utterance_id"]
    ts = f"{u['start_time']:.1f}s"
    corr = str(u.get("text_corrected", "") or "").strip()

    if corr:
        return f"{uid} | {ts} | {corr}"

    return f"{uid} | {ts} | {u['text']}"


def _normalize_judge_context_mode(value: str | None = None) -> str:
    from . import claim_common as cv

    return cv.normalize_judge_context_mode(value)


def _judge_batch_max_workers(value: int | None = None) -> int:
    if value is not None:
        try:
            return max(1, int(value))
        except (TypeError, ValueError):
            return 1
    try:
        configured = int(os.getenv("VERIFIER_JUDGE_BATCH_MAX_WORKERS", "2") or "2")
    except ValueError:
        configured = 2
    return max(1, configured)


def _build_shared_judge_context(utterances: list[dict], context_mode: str = "batch") -> str:
    utterance_lines = [
        f"- {_format_utterance_for_prompt(u)}"
        for u in utterances
    ]

    return (
        "[배치 공통 발화]\n"
        + ("\n".join(utterance_lines) if utterance_lines else "(발화 없음)")
    )


def _source_text_from_claim(claim: dict, utterance_by_id: dict[str, dict]) -> str:
    ids = [
        str(value).strip()
        for value in claim.get("source_span_ids", []) or []
        if str(value).strip()
    ]
    if not ids:
        context_id = _context_id(claim)
        if context_id:
            ids = [context_id]
    pieces = []
    for source_id in ids:
        ref = utterance_by_id.get(source_id)
        if not ref:
            continue
        text = str(ref.get("text_corrected") or ref.get("text") or "").strip()
        if text:
            pieces.append(text)
    return " ".join(pieces).strip()


def _reference_utterance_from_claim(claim: dict, utterance_by_id: dict[str, dict]) -> tuple[str, dict | None]:
    candidate_ids: list[str] = []
    for key in ("context_id", "utterance_id"):
        value = str(claim.get(key) or "").strip()
        if value:
            candidate_ids.append(value)
    for key in (
        "source_span_ids",
        "anchor_utterance_ids",
        "utterance_ids",
        "context_ids",
        "antecedent_context_ids",
    ):
        values = claim.get(key)
        if isinstance(values, list):
            candidate_ids.extend(str(value).strip() for value in values if str(value).strip())

    for candidate_id in dict.fromkeys(candidate_ids):
        ref = utterance_by_id.get(candidate_id)
        if ref:
            return candidate_id, ref
    return "", None


def build_issue_detection_prompt(
    current_date,
    hint,
    min_confidence,
    shared_context,
    claim_lines,
):
    return f"""목표는 최종 판정이 아니라, 후속 issue classification과 crosscheck가 검증할 가치가 있는 issue 후보를 빠뜨리지 않는 것입니다.

이 단계는 claim별 true/false 판정 단계가 아닙니다.
이 단계의 역할은 입력 claim 목록과 배치 문맥을 보고, 실제로 검증할 가치가 있는 issue 후보를 구조화하여 생성하는 것입니다.

단순 표현 취향, 강의 범위 제한, 예시 제한, 강조 표현, 즉시 재표현으로 완전히 해소되는 표현은 issue 후보로 만들지 마세요.
하지만 문맥 후에도 학생이 잘못 외울 수 있는 구체 명제나 오개념 가능성이 남으면, 최종 확정하지 말고 issue 후보로 생성하세요.

배치 공통 문맥:
오늘 날짜: {current_date}
강의 도메인: {hint.get('label', '')}
최소 후보 신뢰도 기준: {min_confidence}

{shared_context}

판정 대상 claim 목록:
{chr(10).join(claim_lines)}

## 이 단계의 입력 해석
각 claim의 실제 판정 대상은 source_span_ids가 가리키는 원문 context와 위 배치 공통 발화 문맥입니다.
source_slice가 제공된 경우에는 해당 claim을 만든 최소 원문 조각이므로 먼저 source_slice의 명제를 보세요.
claim_text는 extractor가 만든 위치 표식 또는 요약입니다.

claim_text가 원문 context보다 더 넓거나 강하면 claim_text를 그대로 믿지 말고 원문 context 기준으로 판단하세요.
다만 원문 문맥 기준으로 더 좁은 실제 오류 가능성이 남으면, 그 좁은 문제를 issue 후보로 생성하세요.

한 context 안에 여러 claim이 있을 수 있습니다.
한 context 안에 서로 다른 잘못된 명제 후보가 여러 개 있으면 각각 독립적으로 판단하세요.
같은 source_span_ids나 같은 context라는 이유만으로 서로 다른 issue를 하나로 합치지 마세요.

## 이 단계에서 해야 할 일
각 claim에 대해 다음을 판단하세요.

1. 원문 context에서 실제로 어떤 명제가 전달되는지 확인합니다.
2. 그 명제가 학생이 잘못 외울 수 있는 구체 명제인지 확인합니다.
3. 문맥 안에서 명시적으로 정정되었는지 확인합니다.
4. 정정되지 않았고 검증 가치가 있으면 issue 후보를 생성합니다.
5. 같은 문제를 여러 claim이 반복하면 가장 직접적인 claim_id를 대표로 삼습니다.

이 단계는 후속 crosscheck보다 넓게 잡는 단계입니다.
따라서 “확실히 틀렸다”가 아니라 “검증할 가치가 있는 구체 후보”이면 출력하세요.

## issue 후보로 생성할 수 있는 경우
다음 중 하나라도 구체적으로 남으면 issue 후보를 생성하세요.

### 1. 직접 내용 오류 후보
- 정의, 분류, 포함 관계, 주체, 과정, 원인-결과, 작동 방식이 잘못 연결된 경우
- 수치, 순서, 조건, 과정 설명이 실제와 다를 가능성이 있는 경우
- 학자, 학파, 이론, 개념, 저작, 시대, 사건, 인물 간의 귀속 관계가 잘못 연결되었을 가능성이 있는 경우

### 2. 용어/분류명 불일치 후보
- 발화가 특정 용어 또는 분류명을 명시했는데, 같은 문맥의 정의·예시·기능·목적이 실제로는 다른 용어/분류를 가리키는 경우
- 슬라이드에는 한 범주가 제시되어 있는데, 발화가 다른 범주명을 반복 사용하여 학생이 두 개념을 동일시할 가능성이 있는 경우
- 여러 구체 대상을 하나의 범주로 묶어 말했는데 그 범주가 부정확할 가능성이 있는 경우

단순히 더 좋은 용어가 있다는 이유만으로는 issue 후보가 아닙니다.
하지만 잘못된 용어/분류명이 명시적으로 발화되고 정정 없이 남아 학생이 잘못 외울 수 있으면 후보로 생성하세요.

### 3. 범위·조건 과잉 후보
- 전체/일부, 가능/불가능, 항상/가끔, 유일/복수, 일시/영구 같은 범위가 실제로 뒤바뀐 경우
- 특정 조건에서만 맞는 말을 일반 사실처럼 말한 경우
- 결과를 과장하여 학생이 잘못된 일반 규칙이나 배제 관계를 외울 수 있는 경우

다음 표현은 주의 신호일 뿐, 그 자체로 issue 근거가 아닙니다.

- 오직, ~만, 항상, 전부, 모든, 반드시, 유일하게, 절대, 불가능, 없다
- 독점, 배타적, 전적으로, 완전히, 무조건
- only, always, all, never, impossible, exclusively

단정어가 있더라도 다음이 어느 정도 구체적으로 보일 때만 후보로 생성하세요.

1. 실제로 무엇이 배제되거나 일반화되었는지
2. 그 배제나 일반화가 강의 범위 제한, 예시 제한, 대비, 강조가 아니라 일반 사실처럼 전달되는지
3. 학생이 잘못 외울 만한 반례, 예외, 누락 대상이 있을 가능성이 있는지
4. 같은 배치 문맥에서 의미가 완전히 좁혀지거나 해소되지 않았는지

이 단계에서는 C형 여부를 확정하지 않습니다.
범위 문제가 구체적으로 의심되면 issue 후보로 생성하고, 후속 classification/crosscheck가 판단하게 하세요.

### 4. 시간성·최신성 후보
- 현재성, 최신성, 지원 여부, 사용 여부, 버전, 정책, 통계, 시장 상황처럼 강의 제공 시점 또는 오늘 날짜 기준 확인이 필요한 경우
- "현재", "요즘", "최근", "최신", "현재 기준", "일반적으로" 같은 표현이 작동 방식, 상태, 지원 여부, 사용 여부, 기술/제도/시장 관행 설명에 붙어 있는 경우

이 단계에서는 최신 사실을 확정하지 말고, 후속 확인이 필요할 수 있는 후보로 생성하세요.

### 5. 오개념 유발 설명 후보
- 명확한 사실 오류로 단정하기 어렵더라도, 설명 흐름 때문에 학생이 개념, 주체, 과정, 원인, 조건을 잘못 연결해 외울 가능성이 있는 경우
- 발화가 애매하여 학생이 “누가 무엇을 수행하는지”, “어떤 과정이 실제로 일어나는지”, “어떤 조건에서 성립하는지”를 잘못 이해할 수 있는 경우

단순히 더 친절하게 설명할 수 있다는 이유만으로는 후보가 아닙니다.
다만 학생이 잘못 외울 수 있는 구체 오개념 문장을 만들 수 있으면 후보로 생성하세요.

## 제외 조건
다음 경우는 issue 후보로 생성하지 마세요.

- 현재 배치 발화 문맥 안에서 명시적으로 정정된 경우
- 바로 뒤 문장이 같은 대상, 같은 관계, 같은 조건을 정확히 풀어 최종 의미가 올바르게 좁혀진 경우
- 잘못된 분류명이나 용어가 즉시 정정되어 최종적으로 어떤 범주를 말하는지 명확한 경우
- 강의 범위 제한, 예시 제한, 문제 풀이 조건, 설명 편의상 제한을 일반 사실처럼 확장해야만 문제가 되는 경우
- 특정 학자, 학파, 연구자, 해석 전통의 견해로 명확히 귀속되어 소개된 경우
- 학술적 가설, 논쟁적 견해, 텍스트에 대한 다양한 해석을 소개하는 발화임이 문맥상 명확한 경우
- 강의 수준 밖의 고급 예외나 매우 엄밀한 보충 가능성만으로 문제를 만들 수 있는 경우
- 표현 취향, 더 좋은 설명 가능성만 있고 학생이 잘못 외울 구체 명제가 남지 않는 경우

단, 견해 소개 형식이더라도 귀속 대상, 개념 설명, 시대, 인물, 학파, 저작명, 이론명이 잘못 연결된 가능성이 구체적으로 남으면 issue 후보로 생성하세요.

## 중복 처리
이 단계에서 중복 제거를 너무 강하게 하지 마세요.
후속 crosscheck에서 merged 처리가 가능하므로, 서로 다른 학생 오개념으로 보일 수 있으면 후보를 유지하세요.

다음 경우에만 중복으로 보고 하나만 출력하세요.

- issue_candidate_text가 실질적으로 동일함
- student_wrong_takeaway가 실질적으로 동일함
- 같은 원문 표현을 같은 방식으로 반복한 후보임

같은 context 안에 있더라도 다음은 별도 후보로 유지하세요.

- 분류 오류와 작동 방식 오류가 다름
- 작동 주체 오류와 결과 과장 오류가 다름
- 용어 불일치와 범위 과잉 단정이 다름
- 같은 발화 안에서 서로 다른 오개념을 만들 수 있음

## candidate_confidence 의미
candidate_confidence는 최종 오류 확정도가 아니라, 후속 classification/crosscheck가 검증할 가치가 있는 후보 가능성입니다.

- 0.00~0.30: issue 후보로 보기 어려움
- 0.31~0.49: 매우 약한 의심, 보통 출력하지 않음
- 0.50~0.60: 약하지만 구체적 검증 후보
- 0.61~0.80: 후속 검증할 가치가 있는 구체적 issue 후보
- 0.81~1.00: 강하게 검증해야 할 issue 후보

candidate_confidence가 {min_confidence:.2f} 미만인 issue 후보는 출력하지 마세요.
다만 이 단계는 후보 누락 방지가 목적이므로, 학생 오개념이 구체적으로 남는 경우에는 과도하게 낮게 주지 마세요.

## 출력 필드 설명
- issue_candidate_id: 이번 응답 안에서 ic001, ic002처럼 순서대로 부여
- source_claim_id: 가장 직접적으로 대응하는 입력 claim_id
- related_claim_ids: 같은 issue 후보와 관련 있는 claim_id 목록. 없으면 source_claim_id 하나만 포함
- source_span_ids: 입력 claim에 있는 source_span_ids를 복사
- issue_candidate_text: 후속 단계가 검증할 문제 명제. 원문보다 과장하지 말 것
- student_wrong_takeaway: 학생이 잘못 외울 수 있는 구체 명제
- detection_basis: 왜 issue 후보인지 짧게 표현
- candidate_confidence: 0.0~1.0 숫자
- duplicate_key: 같은 실제 문제를 묶을 수 있는 짧은 키. 예: "utility_vs_application", "pdf_rendering_by_os"

## 응답 형식
마크다운 code fence 없이 순수 JSON 객체만 출력하세요.

{{
  "issue_candidates": [
    {{
      "issue_candidate_id": "ic001",
      "source_claim_id": "입력 claim_id",
      "related_claim_ids": ["입력 claim_id"],
      "source_span_ids": ["원문 source_span_id"],
      "issue_candidate_text": "후속 검증할 issue 후보 명제",
      "student_wrong_takeaway": "학생이 잘못 외울 수 있는 구체 명제",
      "detection_basis": "직접 내용 오류 후보 | 용어/분류명 불일치 후보 | 범위·조건 과잉 후보 | 시간성·최신성 후보 | 오개념 유발 설명 후보",
      "candidate_confidence": 0.75,
      "duplicate_key": "short_key"
    }}
  ]
}}

## 출력 규칙
1. issue_candidates만 출력하세요.
2. 문제가 없으면 {{"issue_candidates": []}}만 출력하세요.
3. 출력 순서는 입력 claim 순서를 따르세요.
4. 하나의 claim에서 서로 다른 issue 후보가 실제로 여러 개 남으면 여러 issue_candidate를 출력할 수 있습니다.
5. 같은 issue 후보를 여러 claim이 반복하면 가장 직접적인 source_claim_id 하나를 대표로 두고, 나머지는 related_claim_ids에 넣으세요.
6. issue_candidate_text는 원문보다 더 강한 명제로 만들지 마세요.
7. student_wrong_takeaway는 학생이 잘못 외울 수 있는 명제로 쓰세요.
8. detection_basis는 위에 제시된 다섯 후보 범주 중 하나로 쓰세요. A/B/C/D 최종 분류는 하지 마세요.
9. JSON 포맷 외의 어떠한 설명 텍스트도 덧붙이지 마세요.
"""


def _build_judge_prompt(
    claims: list[dict],
    utterances: list[dict],
    current_date: str,
    hint: dict,
    context_mode: str | None = None,
) -> str:
    context_mode = _normalize_judge_context_mode(context_mode)
    shared_context = _build_shared_judge_context(utterances, context_mode)
    min_confidence = _issue_judge_min_confidence()
    claim_lines = []
    for i, c in enumerate(claims, 1):
        claim_id = _claim_id(c) or f"claim_{i}"
        context_id = _context_id(c)
        claim_text = str(c.get("claim_text") or "").strip()
        source_span_ids = [
            str(value).strip()
            for value in c.get("source_span_ids", []) or []
            if str(value).strip()
        ]
        lines = [
            f"{i}. claim_id: {claim_id}",
            f"   claim_text: {claim_text}",
        ]
        source_slice = " ".join(str(c.get("source_slice") or "").split()).strip()
        if source_slice and source_slice != claim_text:
            lines.append(f"   source_slice: {source_slice}")
        if source_span_ids:
            lines.append(f"   source_span_ids: {', '.join(source_span_ids)}")
        claim_lines.append("\n".join(lines))

    return build_issue_detection_prompt(
        current_date,
        hint,
        min_confidence,
        shared_context,
        claim_lines,
    )

    return f"""목표는 최종 판정이 아니라, crosscheck가 검증할 가치가 있는 후보를 빠뜨리지 않는 것입니다.
다만 단순 표현 취향, 강의 범위 제한, 예시 제한, 강조 표현, 즉시 재표현으로 해소되는 표현은 issue 후보로 출력하지 마세요.

배치 공통 문맥:
오늘 날짜: {current_date}
강의 도메인: {hint.get('label', '')}
최소 신뢰도 기준: {min_confidence}

{shared_context}

판정 대상 claim 목록:
{chr(10).join(claim_lines)}

### 판정 기준
각 claim의 실제 판정 대상은 `source_slice`가 있으면 `source_slice`, 없으면 `claim_text`입니다.
`source_span_ids`가 가리키는 원문 context와 위 배치 공통 발화 문맥은 지시어, 생략, 즉시 정정 여부를 확인하는 보조 문맥입니다.
`claim_text`는 extractor가 만든 위치 표식/요약이므로, 원문 context보다 강하거나 다르게 쓰였으면
원문 context 기준으로 낮게 채점하세요.
`source_slice`가 제공된 경우에는 같은 context 안의 다른 문장이 아니라 source_slice의 명제를 먼저 독립적으로 판단하세요.

이 단계에서는 같은 문맥 안의 claim들을 병합하거나 대표 claim을 고르지 않습니다.
각 claim_id를 서로 독립된 검토 문제로 보고, 다른 claim의 존재 여부와 무관하게 점수를 매기세요.
이미 같은 context 안의 다른 claim이 높은 점수를 받았더라도, 현재 claim에 별도 오류 가능성이 남으면
현재 claim_id에도 독립적으로 높은 candidate_confidence를 줄 수 있습니다.
반대로 같은 context 안의 다른 claim이 문제라고 해서, 현재 claim까지 자동으로 높게 주면 안 됩니다.

should_verify=true로 두려면 다음 조건을 모두 만족해야 합니다.

1. 학생이 잘못 외울 수 있는 구체 명제가 남아야 합니다.
2. 그 명제가 source_slice, claim_text, 원문 context 중 하나에 실제로 뒷받침되어야 합니다.
3. 단순히 더 엄밀하게 말할 수 있다는 수준이 아니라, 강의 도메인과 수준에서 crosscheck할 가치가 있어야 합니다.
4. 현재 배치 발화 문맥 안에서 명확히 정정되거나 바로 뒤 문장으로 의미가 정확히 풀리지 않아야 합니다.

다음 중 하나라도 구체적으로 남으면 should_verify=true 후보로 보세요.

- 정의, 분류, 포함 관계, 주체, 과정, 원인-결과, 작동 방식이 잘못 연결된 경우
- 학자, 학파, 이론, 개념, 저작, 시대, 사건, 인물 간의 귀속 관계가 잘못 연결되었을 가능성이 있는 경우
- 여러 구체 대상을 하나의 범주로 묶어 말했는데 그 범주가 부정확할 가능성이 있는 경우
- 발화가 특정 용어/분류명을 명시했고, 같은 문맥의 정의·예시·이어지는 설명이 실제로는 다른 용어/분류를 가리키는 경우
- 전체/일부, 가능/불가능, 항상/가끔, 유일/복수, 일시/영구 같은 범위가 실제로 뒤바뀌어 학생이 잘못된 일반 규칙이나 배제 관계를 외울 가능성이 남는 경우
- 특정 조건에서만 맞는 말을 일반 사실처럼 말하거나, 결과를 과장해 학생이 잘못 외울 수 있는 경우
- 충돌, 오류, 손상, 실패, 불능 같은 결과 설명이 실제 강의 수준의 문제보다 더 영구적이거나 전체적인 결과처럼 전달되는 경우
- 현재성, 최신성, 지원 여부, 사용 여부처럼 강의 제공 시점 또는 오늘 날짜 기준으로 확인이 필요한 경우
- "현재", "요즘", "최근", "최신", "현재 기준", "일반적으로" 같은 시점 의존 표현이
  작동 방식, 상태, 지원 여부, 사용 여부, 기술/제도/시장 관행 설명에 붙어 있고
  검증 기준일의 실제 상태나 통상 관행과 충돌할 가능성이 있으면 최종 확정하지 말고 issue 후보로 올리세요.

### 단정어/범위 표현 주의 규칙
다음 표현은 issue 후보를 찾기 위한 주의 신호일 뿐, 그 자체로 issue 근거가 아닙니다.

- 오직, ~만, 항상, 전부, 모든, 반드시, 유일하게, 절대, 불가능, 없다
- 독점, 배타적, 전적으로, 완전히, 무조건
- only, always, all, never, impossible, exclusively

이 표현이 있더라도 다음이 명확하지 않으면 should_verify=false로 두세요.

1. 실제로 무엇이 배제되거나 일반화되었는지
2. 그 배제나 일반화가 강의 범위 제한, 예시 제한, 대비, 강조가 아니라 일반 사실처럼 전달되는지
3. 학생이 잘못 외울 만한 구체적인 반례, 예외, 누락 대상이 있는지
4. 같은 배치 문맥에서 의미가 좁혀지거나 해소되지 않았는지

단정어 또는 범위 표현이 핵심 근거인 후보는 실제 배제 명제가 구체적으로 남을 때만 candidate_confidence를 0.61 이상으로 주세요.
단정어는 있지만 실제 배제 대상이나 반례가 구체적이지 않으면 candidate_confidence를 0.45 이하로 판단하고 출력하지 마세요.

### claim_text 과장 금지
claim_text가 배치 공통 발화의 원문 문맥보다 더 넓거나 강한 명제로 보이면 claim_text를 그대로 믿지 말고 원문 context 기준으로 판단하세요.

예를 들어 다음과 같은 변환은 과장입니다.

- "여기서는 A만 보겠습니다" → "A만 존재한다"
- "대표적으로 A입니다" → "A가 유일한 예입니다"
- "주로 A입니다" → "항상 A입니다"
- "이 경우에는 A입니다" → "모든 경우에 A입니다"

claim_text가 원문보다 일반화, 단정, 배제, 인과, 귀속을 강하게 만든 경우에는 그 강해진 명제로 should_verify=true를 주지 마세요.
다만 원문 문맥 기준으로 더 좁은 실제 오류가 명확히 남는 경우에는 그 좁은 오류 가능성만 기준으로 판단하세요.

### 용어/분류명 불일치 처리
- 단순히 더 좋은 용어가 있다는 이유만으로는 issue 후보가 아닙니다.
- 그러나 발화가 A라는 용어/분류명을 명시하고, 곧이어 든 정의·예시·기능·목적이 실제로는 B를 설명하는 흐름이면 후보로 올리세요.
- 넓은 문맥을 보면 화자가 B를 말하려던 의도가 추정되더라도, A라는 잘못된 용어/분류명이 정정 없이 남으면 후보입니다.
- 같은 대상이나 대상 묶음이 한 문맥 안에서 서로 다른 범주명으로 호명되어,
  학생이 그 대상이 어느 범주에 속하는지 구체적으로 혼동할 수 있으면 단순 용어 취향이 아니라 분류 관계 후보입니다.
- 바로 뒤에서 "A가 아니라 B"처럼 명시적으로 고쳐 최종 전달 명제가 분명해진 경우만 제외하세요.
- 같은 context 안에 서로 다른 용어/분류명 불일치가 여러 개 있으면, 각각의 claim_id를 독립적으로 채점하세요.
- 같은 대상 묶음에 대해 서로 다른 분류명이 이어서 붙는 경우, 앞뒤에 더 적절한 분류명이 한 번 등장했다는 이유만으로
  중간의 부정확한 분류명을 자동 해소하지 마세요.
  그 부정확한 분류명이 학생에게 독립적인 분류 명제로 남으면 해당 claim_id에 높은 candidate_confidence를 주세요.

### 제외 조건
다음 경우는 should_verify=false로 두세요.

- 현재 배치 발화 문맥 안에서 명확히 정정된 경우
- 발화 자체는 어색하지만 바로 뒤 문장이 같은 의미를 정확히 풀어 오해 가능성을 낮춘 경우
- 잘못된 분류명이나 용어가 즉시 정정되어 최종적으로 어떤 범주를 말하는지 명확한 경우
- 강의 범위 제한, 예시 제한, 문제 풀이 조건, 설명 편의상 제한을 일반 사실처럼 확장해야만 문제가 되는 경우
- 특정 학자, 학파, 연구자, 해석 전통의 견해로 명확히 귀속되어 소개된 경우
- 학술적 가설, 논쟁적 견해, 텍스트에 대한 다양한 해석을 소개하는 발화임이 문맥상 명확한 경우
- 강의 수준 밖의 고급 예외나 매우 엄밀한 보충 가능성만으로 문제를 만들 수 있는 경우
- 표현 취향, 더 좋은 설명 가능성만 있고 학생이 잘못 외울 구체 명제가 남지 않는 경우

단, 견해 소개 형식이더라도 귀속 대상, 개념 설명, 시대, 인물, 학파, 저작명, 이론명이 잘못 연결된 가능성이 구체적으로 남으면 should_verify=true 후보로 보세요.

### candidate_confidence 의미
candidate_confidence는 최종 오류 확정도가 아니라, crosscheck가 검증할 가치가 있는 오류 가능성의 강도입니다.

- 0.00~0.45: issue 후보로 삼기 어려움
- 0.46~0.60: 애매하지만 검토 가능성이 약간 있음
- 0.61~0.80: crosscheck가 검증할 가치가 있는 구체적 오류 가능성이 남음
- 0.81~1.00: 문맥 후에도 명백히 검증해야 할 오류 가능성이 강함

candidate_confidence가 {min_confidence:.2f} 미만이어도 claim_scores에는 반드시 출력하세요.
다만 이 경우 should_verify=false로 두세요.
단정어 또는 범위 표현이 핵심 근거인 후보는 candidate_confidence가 0.61 이상이라고 판단될 때만 should_verify=true로 두세요.

### 응답 형식
마크다운 code fence 없이 순수 JSON 객체만 출력하세요.

{{
  "claim_scores": [
    {{
      "claim_id": "입력 claim_id를 그대로 복사",
      "should_verify": true,
      "candidate_confidence": 0.85
    }}
  ]
}}

### 출력 규칙
1. 입력된 모든 claim_id를 claim_scores에 정확히 한 번씩 출력하세요.
   문제가 없어도 생략하지 말고 should_verify=false, candidate_confidence=0.0~0.45로 출력하세요.
2. 각 claim_id는 독립적으로 채점하세요.
   같은 source_span_ids, 같은 context, 같은 주제, 같은 오류 유형이라는 이유로 다른 claim의 점수를 참고하거나 낮추지 마세요.
   중복 병합과 대표 선택은 후속 단계에서 처리하므로, 이 단계에서는 하지 마세요.
3. 출력 순서는 입력 claim 순서를 따르세요.
4. candidate_confidence는 0.0~1.0 사이 숫자로 쓰세요.
5. 출력 필드는 claim_id, should_verify, candidate_confidence만 허용합니다.
6. type, issue_type, reason, 반례, 설명 문장은 출력하지 마세요.
7. JSON 포맷 외의 어떠한 설명 텍스트도 덧붙이지 마세요.
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
    judge_model = str(cv._resolve_stage_model("judge") or "").strip()
    response_format = {"type": "json_object"} if cv._is_anthropic_model(judge_model) else None
    api_calls = 0
    token_usage = cv._empty_token_usage()
    claim_by_id = {_claim_id(claim): claim for claim in claims if _claim_id(claim)}

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
                print(f"    ↺ claim 판정 JSON 파싱 재시도 ({attempt+1}/{cv.VERIFIER_PARSE_RETRIES})")
                continue
            return [], True, api_calls, token_usage

        if isinstance(payload, list):
            raw = payload
        elif isinstance(payload, dict):
            raw = (
                payload.get("issue_candidates")
                or payload.get("claim_scores")
                or payload.get("issues")
                or payload.get("results")
                or payload.get("items")
                or []
            )
        else:
            raw = []
        if not isinstance(raw, list):
            return [], False, api_calls, token_usage

        issues = []
        for raw_issue in raw:
            if not isinstance(raw_issue, dict):
                continue
            raw_claim_id = str(
                raw_issue.get("source_claim_id")
                or raw_issue.get("claim_id")
                or ""
            ).strip()
            source_claim = claim_by_id.get(raw_claim_id)
            if not source_claim:
                continue
            claim_id = _claim_id(source_claim)
            context_id = _context_id(source_claim)
            ref_id, ref = _reference_utterance_from_claim(source_claim, utt_map)
            if not ref:
                continue
            try:
                confidence = float(
                    raw_issue.get("candidate_confidence", raw_issue.get("confidence", 1.0))
                    or 0.0
                )
            except (TypeError, ValueError):
                confidence = 0.0
            if 1.0 < confidence <= 100.0:
                confidence /= 100.0
            confidence = max(0.0, min(1.0, confidence))
            if confidence < _issue_judge_min_confidence():
                continue
            issue_candidate_id = str(raw_issue.get("issue_candidate_id") or "").strip()
            issue_candidate_text = " ".join(str(raw_issue.get("issue_candidate_text") or "").split()).strip()
            student_wrong_takeaway = " ".join(str(raw_issue.get("student_wrong_takeaway") or "").split()).strip()
            detection_basis = " ".join(str(raw_issue.get("detection_basis") or "").split()).strip()
            duplicate_key = " ".join(str(raw_issue.get("duplicate_key") or "").split()).strip()
            issue_candidate_key = ""
            if issue_candidate_text or issue_candidate_id or duplicate_key:
                key_tail = detection_basis or duplicate_key or issue_candidate_text[:120] or issue_candidate_id
                issue_candidate_key = f"{claim_id}::{key_tail}"
            source_span_ids = [
                str(value).strip()
                for value in raw_issue.get("source_span_ids", source_claim.get("source_span_ids", [])) or []
                if str(value).strip()
            ]
            source_text = (
                str(source_claim.get("source_slice") or "").strip()
                or
                str(source_claim.get("source_text") or source_claim.get("claim_context_text") or "").strip()
                or _source_text_from_claim(source_claim, utt_map)
                or str(ref.get("text") or "").strip()
            )
            issue = {
                "claim_id": claim_id,
                "source_claim_id": claim_id,
                "issue_candidate_id": issue_candidate_id,
                "issue_candidate_key": issue_candidate_key,
                "issue_candidate_text": issue_candidate_text,
                "student_wrong_takeaway": student_wrong_takeaway,
                "detection_basis": detection_basis,
                "duplicate_key": duplicate_key,
                "utterance_id": ref_id or context_id,
                "context_id": context_id,
                "context_ids": _related_context_ids(source_claim),
                "utterance_ids": _related_context_ids(source_claim),
                "anchor_utterance_ids": [
                    str(value).strip()
                    for value in source_claim.get("anchor_utterance_ids", []) or []
                    if str(value).strip()
                ],
                "source_span_ids": source_span_ids,
                "source_text": source_text,
                "source_slice": str(source_claim.get("source_slice") or "").strip(),
                "source_block_id": str(source_claim.get("source_block_id", "") or "").strip(),
                "claim_context_ids": [
                    str(value).strip()
                    for value in source_claim.get("claim_context_ids", source_claim.get("source_span_ids", [])) or []
                    if str(value).strip()
                ],
                "claim_context_text": source_text,
                "claim_context_block_id": str(
                    source_claim.get("claim_context_block_id") or source_claim.get("source_block_id", "") or ""
                ).strip(),
                "antecedent_context_ids": [
                    str(value).strip()
                    for value in source_claim.get("antecedent_context_ids", []) or []
                    if str(value).strip()
                ],
                "claim_text": issue_candidate_text or str(source_claim.get("claim_text", "") or "").strip(),
                "source_claim_text": str(source_claim.get("claim_text", "") or "").strip(),
                "issue": student_wrong_takeaway or issue_candidate_text or detection_basis,
                "confidence": confidence,
                "candidate_confidence": confidence,
            }
            if issue["source_span_ids"]:
                issue["utterance_ids"] = issue["source_span_ids"]
                issue["context_ids"] = issue["source_span_ids"]
            issue["claim_type"] = source_claim.get("claim_type", "")
            issue["is_approximate"] = bool(source_claim.get("is_approximate"))
            issue["needs_context"] = bool(source_claim.get("needs_context"))
            issue["resolution_status"] = source_claim.get("resolution_status", "")
            issue["start_time"] = ref["start_time"]
            issue["timestamp"] = f"[{ref['start_time']:.1f}s]"
            issue["slide_number"] = ref["slide_number"]
            if not isinstance(issue.get("evidence_sources"), list):
                issue["evidence_sources"] = []
            if "severity" not in issue:
                issue["severity"] = "major"
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
    max_workers: int | None = None,
) -> tuple[list[dict], int, dict]:
    """2단계만 실행: claim 판정. (이슈 리스트, api_calls) 반환."""
    from . import claim_common as cv

    run_results = []
    total_api = 0
    total_token_usage = cv._empty_token_usage()

    context_mode = _normalize_judge_context_mode(context_mode)
    prefix = f"[{log_prefix}] " if log_prefix else ""
    total_batches = sum(1 for _, claims in all_claims_by_batch if claims)
    worker_count = min(_judge_batch_max_workers(max_workers), max(1, total_batches))
    print(f"  {prefix}claim 판정 문맥 모드: {context_mode}", flush=True)
    if worker_count > 1 and total_batches > 1:
        print(f"  {prefix}claim 판정 병렬 처리: max_workers={worker_count}", flush=True)

    for run in range(num_runs):
        if num_runs > 1:
            print(f"\n  {prefix}판정 run {run+1}/{num_runs}")
        run_issues = []
        run_token_usage = cv._empty_token_usage()
        active_batch_idx = 0
        jobs = []
        for batch_idx, (batch, claims) in enumerate(all_claims_by_batch, start=1):
            if not claims:
                continue
            active_batch_idx += 1
            batch_map = {u["utterance_id"]: u for u in batch}
            ids = f"{batch[0]['utterance_id']}..{batch[-1]['utterance_id']}"
            jobs.append((active_batch_idx, batch_idx, batch, claims, batch_map, ids))

        def _run_job(job):
            active_idx, batch_idx, batch, claims, batch_map, ids = job
            print(f"  {prefix}({active_idx}/{total_batches}) {ids}", flush=True)
            issues, parse_failed, api_calls, token_usage, ok = recover_claim_judgement(
                claims, batch, current_date, hint, batch_map,
                f"판정 {run+1} 배치 {batch_idx} {ids}",
                context_mode=context_mode,
            )
            return active_idx, issues, api_calls, token_usage

        if worker_count > 1 and len(jobs) > 1:
            print(f"  {prefix}cache warm-up: 첫 batch 선실행 후 나머지 병렬 처리", flush=True)
            results = {}
            warmup_result = _run_job(jobs[0])
            results[warmup_result[0]] = warmup_result
            remaining_jobs = jobs[1:]
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                futures = {executor.submit(_run_job, job): job[0] for job in remaining_jobs}
                for future in as_completed(futures):
                    active_idx = futures[future]
                    try:
                        results[active_idx] = future.result()
                    except Exception as e:
                        raise RuntimeError(f"claim judge batch {active_idx} failed: {e}") from e
            ordered_results = [results[job[0]] for job in jobs]
        else:
            ordered_results = [_run_job(job) for job in jobs]

        for _active_idx, issues, api_calls, token_usage in ordered_results:
            run_issues.extend(issues)
            total_api += api_calls
            run_token_usage = cv._merge_token_usage(run_token_usage, token_usage)
        run_results.append(cv._make_result(cv._dedupe_issues(run_issues), 0, token_usage=run_token_usage))

    print(f"\n  📊 통합 중 (합의 기준 {min_detection_rate:.0%})...")
    merged = cv.merge_multiple_runs(run_results, num_runs, min_detection_rate)
    total_token_usage = cv._merge_token_usage(*(r.get("token_usage") for r in run_results))
    return merged.get("issues", []), total_api, total_token_usage
