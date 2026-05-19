from __future__ import annotations

import json
import hashlib
import os
import re
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed


_CANONICAL_CLAIM_TYPES = {
    "definition",
    "numeric",
    "causal",
    "relationship",
    "currentness",
}

def _stable_claim_id(claim: dict) -> str:
    base_id = str(claim.get("context_id") or claim.get("utterance_id") or "claim").strip() or "claim"
    # The stable identity must be anchored to the extracted source span, not a
    # model-rewritten restatement.
    text = str(claim.get("claim_text") or claim.get("source_text") or "").strip()
    digest = hashlib.sha1(f"{base_id}\n{text}".encode("utf-8")).hexdigest()[:12]
    return f"{base_id}::{digest}"


def _normalize_prompt_ref(value) -> str:
    raw = str(value or "").strip()
    match = re.match(r"^((?:U\d{3,5}|S\d{3}(?:-C\d{3})?))(?:#seg\d+)?$", raw)
    return match.group(1) if match else raw


def _dedupe_ids(values: list[str]) -> list[str]:
    return list(dict.fromkeys(
        ref for ref in (_normalize_prompt_ref(value) for value in values) if ref
    ))


def _ids_from_context_note(note: str) -> list[str]:
    return re.findall(r"\b(?:U\d{3,5}|S\d{3}(?:-C\d{3})?)\b", str(note or ""))


def _has_model_added_parenthetical(claim_text: str, source_text: str) -> bool:
    claim_text = str(claim_text or "")
    source_text = str(source_text or "")
    bracket_chars = ("(", ")", "（", "）")
    if not any(ch in claim_text for ch in bracket_chars):
        return False
    return not any(ch in source_text for ch in bracket_chars)


def _remove_parenthetical_text(text: str) -> str:
    output: list[str] = []
    depth = 0
    for ch in str(text or ""):
        if ch in {"(", "（"}:
            depth += 1
            continue
        if ch in {")", "）"}:
            if depth > 0:
                depth -= 1
                continue
        if depth == 0:
            output.append(ch)
    return " ".join("".join(output).split()).strip()


def _source_sentence_units(source_text: str) -> list[str]:
    units: list[str] = []
    current: list[str] = []
    for ch in str(source_text or ""):
        current.append(ch)
        if ch in {".", "?", "!", "。", "？", "！"}:
            unit = " ".join("".join(current).split()).strip()
            if unit:
                units.append(unit)
            current = []
    tail = " ".join("".join(current).split()).strip()
    if tail:
        units.append(tail)
    return units


def _tokenize_for_overlap(text: str) -> set[str]:
    tokens: set[str] = set()
    current: list[str] = []
    for ch in str(text or "").lower():
        if ch.isalnum() or "\uac00" <= ch <= "\ud7a3" or ch in {"#", "+", "."}:
            current.append(ch)
            continue
        if current:
            token = "".join(current).strip(".")
            if len(token) >= 2:
                tokens.add(token)
            current = []
    if current:
        token = "".join(current).strip(".")
        if len(token) >= 2:
            tokens.add(token)
    return tokens


def _restore_parenthetical_claim_to_source(claim_text: str, source_text: str) -> str:
    if not _has_model_added_parenthetical(claim_text, source_text):
        return " ".join(str(claim_text or "").split()).strip()

    target_tokens = _tokenize_for_overlap(_remove_parenthetical_text(claim_text))
    units = _source_sentence_units(source_text)
    if not target_tokens or not units:
        return ""

    best_text = ""
    best_score = 0.0
    best_span_len = 0
    max_span = min(3, len(units))
    for start in range(len(units)):
        for end in range(start, min(len(units), start + max_span)):
            candidate = " ".join(units[start : end + 1]).strip()
            candidate_tokens = _tokenize_for_overlap(candidate)
            if not candidate_tokens:
                continue
            overlap = len(target_tokens & candidate_tokens)
            recall = overlap / max(1, len(target_tokens))
            precision = overlap / max(1, len(candidate_tokens))
            score = (recall * 0.7) + (precision * 0.3)
            span_len = end - start + 1
            if (
                score > best_score
                or (score == best_score and span_len < best_span_len)
                or (score == best_score and span_len == best_span_len and len(candidate) < len(best_text))
            ):
                best_text = candidate
                best_score = score
                best_span_len = span_len

    if best_score < 0.35:
        return ""
    return best_text


def _normalize_source_slice(value, claim_text: str, source_text: str) -> str:
    """Keep a compact claim-local quote for downstream scoring."""
    source_text = " ".join(str(source_text or "").split()).strip()
    claim_text = " ".join(str(claim_text or "").split()).strip()
    candidate = " ".join(str(value or "").split()).strip()
    if not candidate:
        candidate = claim_text
    if not candidate:
        return ""

    if source_text:
        compact_source = re.sub(r"\s+", "", source_text)
        compact_candidate = re.sub(r"\s+", "", candidate)
        if compact_candidate and compact_candidate in compact_source:
            return candidate

        compact_claim = re.sub(r"\s+", "", claim_text)
        if compact_claim and compact_claim in compact_source:
            return claim_text

    return candidate


def _claim_extract_batch_mode() -> str:
    mode = str(os.getenv("VERIFIER_CLAIM_EXTRACT_BATCH_MODE", "context") or "context").strip().lower()
    if mode in {"context", "contexts", "context_unit", "context-unit"}:
        return "context"
    if mode in {"slide", "slides", "scene", "scenes"}:
        return "slide"
    return "utterance"


def _claim_extract_context_window() -> tuple[int, int]:
    def _safe_int(name: str, default: int) -> int:
        try:
            return max(0, int(os.getenv(name, str(default)) or default))
        except (TypeError, ValueError):
            return default

    return (
        _safe_int("VERIFIER_CLAIM_EXTRACT_CONTEXT_PREV", 2),
        _safe_int("VERIFIER_CLAIM_EXTRACT_CONTEXT_NEXT", 1),
    )


def _claim_extract_context_group_size() -> int:
    try:
        return max(1, int(os.getenv("VERIFIER_CLAIM_EXTRACT_CONTEXT_GROUP_SIZE", "4") or "4"))
    except (TypeError, ValueError):
        return 4


def _force_context_unit_batching() -> bool:
    raw = str(os.getenv("VERIFIER_CLAIM_EXTRACT_FORCE_CONTEXT_UNITS", "1") or "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _claim_extract_prompt_profile() -> str:
    raw = str(os.getenv("VERIFIER_CLAIM_EXTRACT_PROMPT_PROFILE", "auto") or "auto").strip().lower()
    if raw in {"gpt", "gpt_strict", "gpt-strict", "openai_strict", "openai-strict"}:
        return "gpt_strict"
    if raw in {"default", "base", "common", "none", "off"}:
        return "default"

    model = (
        os.getenv("VERIFIER_CLAIM_EXTRACT_MODEL", "")
        or os.getenv("VERIFIER_MODEL", "")
        or ""
    ).strip().lower()
    if model.startswith(("gpt", "o1", "o3")):
        return "gpt_strict"
    return "default"


def _claim_extract_max_workers(value: int | None = None) -> int:
    if value is not None:
        try:
            return max(1, int(value))
        except (TypeError, ValueError):
            return 1
    try:
        configured = int(os.getenv("VERIFIER_CLAIM_EXTRACT_MAX_WORKERS", "4") or "4")
    except ValueError:
        configured = 4
    return max(1, configured)


def _model_specific_extract_rules() -> str:
    if _claim_extract_prompt_profile() != "gpt_strict":
        return ""
    return """
### GPT 계열 추가 규칙
GPT 계열 모델은 "빠뜨리지 말라"는 지시를 과하게 해석해 문장 조각이나 강의 진행 발언까지 claim으로 만들 수 있습니다.
따라서 아래 규칙을 우선 적용하세요.

1. 출력 기준
- claim은 학생이 그대로 외웠을 때 참/거짓을 검증할 수 있는 **완성 명제**여야 합니다.
- 단순히 강의자가 설명을 시작함, 예시를 들겠다고 함, 다음 내용을 예고함, 질문을 던짐, 강의 운영을 안내함은 claim이 아닙니다.
- 술어가 끝나지 않은 문장 조각은 단독 claim으로 출력하지 마세요.

2. 전수 확인의 의미
- 검사 대상 발화마다 claim 후보가 있는지 확인하라는 뜻이지, 모든 검사 대상 발화를 claim으로 만들라는 뜻이 아닙니다.
- 검증 가능한 완성 명제가 없는 검사 대상 발화는 과감히 출력하지 마세요.
- 강의 일정, 연락 방법, 수업 준비물, 공지 확인처럼 강의 운영에 관한 발언은 수치나 정책 자체가 핵심 검증 대상일 때만 추출하세요.

3. 문맥 결합
- 문맥은 문장 조각을 완성하거나 지시어 의존 여부를 표시하기 위한 보조 정보입니다.
- 문맥을 이용해 현재 발화가 실제로 말하지 않은 주체, 조건, 인과, 일반 법칙을 새로 만들지 마세요.
- 여러 발화가 하나의 문장을 이룰 때는 하나의 claim으로 합치고, utterance_ids에 포함하세요.

4. claim_text 작성
- claim_text는 원문에서 직접 가져온 검증 대상 span이어야 합니다.
- 하나의 원문에 서로 다른 명제가 있으면 하나로 요약하지 말고 분리하세요.
- 원문에 있는 중요한 부정/한정 표현을 덮어쓰지 마세요. 예: "A와 직접 관련 없다"와 "B를 제공한다"는 별도 claim입니다.
"""


def _format_seconds(value) -> str:
    try:
        return f"{float(value):.1f}s"
    except (TypeError, ValueError):
        return "?s"


def _segment_text(seg: dict) -> str:
    return str(
        seg.get("text_corrected")
        or seg.get("text")
        or ""
    ).strip()


def _format_source_segments_for_prompt(u: dict) -> str:
    source_segments = u.get("source_segments") or []
    if not source_segments:
        return ""

    uid = str(u.get("utterance_id") or u.get("context_id") or "?")
    lines = ["  발화:"]
    for idx, seg in enumerate(source_segments, 1):
        text = _segment_text(seg)
        if not text:
            continue
        start = _format_seconds(seg.get("start", seg.get("start_time")))
        lines.append(f"    - {uid}#seg{idx:02d} [{start}] {text}")
    return "\n".join(lines) if len(lines) > 1 else ""


def _format_utterance_for_prompt(u: dict) -> str:
    uid = u["utterance_id"]
    ts = _format_seconds(u.get("start_time"))
    corr = str(u.get("text_corrected", "") or "").strip()
    source_block = _format_source_segments_for_prompt(u)

    if source_block:
        return f"{uid} | {ts} | context\n{source_block}"

    if corr:
        return f"{uid} | {ts} | {corr}"

    return f"{uid} | {ts} | {u['text']}"


def _build_slide_references(slide_numbers: list[int], slide_ctx: dict) -> str:
    refs = []
    for sn in slide_numbers:
        ctx = slide_ctx.get(sn, {})
        title = ctx.get("title", f"슬라이드 {sn}")
        time_range = ctx.get("time_range", "")
        slide_text = str(ctx.get("slide_text", "") or "")
        limit = 1200
        if len(slide_text) > limit:
            slide_text = slide_text[:limit] + "\n... (이하 생략)"
        refs.append(
            f"[슬라이드 {sn}] {title} ({time_range})\n"
            + (slide_text if slide_text else "(슬라이드 텍스트 없음)")
        )
    return "\n\n".join(refs)


def _build_slide_and_utterance_context(
    utterances: list[dict],
    slide_ctx: dict,
    target_utterance_ids: set[str] | None = None,
) -> tuple[str, str]:
    """슬라이드 참조와 발화 문맥을 한 번의 순회로 생성."""
    seen_slides = OrderedDict()
    target_ids = target_utterance_ids or {str(u.get("utterance_id") or "") for u in utterances}
    transcript_lines = []
    target_lines = []
    for u in utterances:
        current_sn = int(u.get("slide_number", 0) or 0)
        if current_sn not in seen_slides:
            seen_slides[current_sn] = True
        uid = str(u.get("utterance_id") or "")
        marker = ">>" if uid in target_ids else "  "
        transcript_lines.append(f"{marker} {_format_utterance_for_prompt(u)}")
        if uid in target_ids:
            target_lines.append(f"- {uid} (슬라이드 {current_sn})")

    context = (
        "[배치 전체 전사]\n"
        + "\n".join(transcript_lines)
        + "\n\n[검사 대상 context/발화]\n"
        + "\n".join(target_lines)
        + "\n\n[추출 규칙]\n"
        + "- '>>' 표시된 검사 대상만 claim 출력 대상으로 삼으세요.\n"
        + "- 표시 없는 항목은 앞뒤 문맥, 지시어 해소, 조각 문장 완결에만 사용하세요.\n"
        + "- 표시 없는 항목을 대표 utterance_id로 새 claim을 만들지 마세요.\n"
        + "- 검사 대상이 context인 경우 context는 원문 발화를 보존한 묶음입니다. context 대표 요약이 아니라 내부 원문 발화의 원자 명제를 추출하세요.\n"
        + "- 같은 context 안에 서로 다른 정의/수치/인과/관계/현재성 주장이 있으면 각각 별도 claim으로 출력하세요.\n"
        + "- 전사 분할 때문에 한 문장이 여러 원문 발화로 끊긴 경우만 결합하고, 서로 다른 명제를 하나의 claim_text로 합치지 마세요."
    )
    return _build_slide_references(list(seen_slides.keys()), slide_ctx), context


def _build_extract_prompt(
    utterances: list[dict],
    current_date: str,
    hint: dict,
    slide_ctx: dict,
    target_utterance_ids: set[str] | None = None,
) -> str:
    slide_references, utterance_context = _build_slide_and_utterance_context(
        utterances,
        slide_ctx,
        target_utterance_ids=target_utterance_ids,
    )

    return f"""당신은 강의 발화에서 검증 가능한 사실 주장(claim)의 원문 목록을 추출하는 전문가입니다.
오늘 날짜: {current_date}
강의 도메인: {hint['label']}

아래는 슬라이드 참조 정보와 발화 문맥입니다.
강의자는 슬라이드를 보여주면서 발화합니다. 학생은 슬라이드와 발화를 동시에 받습니다.

[슬라이드 참조]
{slide_references}

[발화 문맥]
{utterance_context}

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

### 핵심 원칙: 1단계는 raw claim inventory입니다
- 이 단계에서는 오류 여부, 오해 가능성, 강의자 피드백, 반례를 판단하지 마세요.
- 입력 ID가 `S001-C001`처럼 context 단위이면, output의 `utterance_id`와 `context_id`도 같은 context ID를 그대로 쓰세요.
- `#seg01` 표시는 원문 위치 참고용입니다. 출력 ID에는 `#seg`를 붙이지 마세요.
- claim은 context 요약이 아니라 context 안 원문 발화가 말한 원자 명제 단위로 추출하세요.
- claim_text는 원문에서 직접 가져온 가장 짧은 검증 대상 span입니다.
- claim_text를 자연스럽게 고치거나, 표준 용어로 바꾸거나, 괄호로 주어/대상을 보충하지 마세요.
- source_slice는 detector가 이 claim을 우선 판단할 때 볼 최소 원문 조각입니다.
- source_slice에는 claim_text를 만든 가장 직접적인 원문 구간만 넣고, 같은 context의 다른 문제나 앞선 예시를 불필요하게 포함하지 마세요.
- 한 context 안에 여러 관계/분류/작동 방식 claim이 있으면 각 claim의 source_slice도 서로 독립적으로 좁게 잡으세요.
- source_slice는 원문 표현을 그대로 인용하되, 전사 분할로 끊긴 조각을 이어야 할 때만 필요한 조각을 결합하세요.
- 서로 다른 정의, 수치, 원인, 기능, 분류, 예시는 각각 별도 claim으로 출력하세요.
- 전사 분할 때문에 하나의 문장이 여러 발화로 끊긴 경우에만 원문 조각을 결합하세요.
- 바로 뒤 문장이 앞 표현을 이어 완성하는 경우에는 가장 짧은 원문 구간으로 하나의 claim만 추출하세요.
- 한 발화가 검증 대상의 이름/예시/대상 목록만 제시하고, 바로 뒤 발화가 그 대상의 분류명, 명칭, 포함 관계, 기능, 목적을 말하면 두 원문 조각을 결합해 하나의 claim으로 추출하세요.
- 이 경우 claim_text와 source_slice에는 결합된 원문 구간만 쓰세요.
- 바로 뒤 문장이 앞 표현을 정정/보완/해소하는지 판단해서 claim을 버리지 마세요. 그런 판단은 Issue_detection과 crosscheck 단계에서 합니다.
- 문맥과 슬라이드는 claim인지 아닌지, 예시 범위가 있는지, 지시어가 문맥 의존적인지 판단하는 보조 정보입니다. 문맥의 더 강한 일반 명제를 claim_text에 덧씌우지 마세요.

### 지시어/생략 처리
- "이것", "여기", "해당 항목", "얘", "걔", "이거", "그거", "이것들" 같은 지시어는 claim_text 안에서 풀어 쓰지 마세요.
- 지시어의 선행사가 명확해 보여도 claim_text에는 원문 지시어를 그대로 남기세요.
- 지시어를 괄호로 보충하지 마세요. 예: `얘네들(컴퓨터 하드웨어)`처럼 쓰지 마세요.
- 모든 claim은 후속 단계에서 문맥과 함께 검증됩니다. needs_context=true, resolution_status="unresolved"로 표시하세요.
- 선행사 판단에 참고한 context가 있으면 antecedent_context_ids에 ID만 넣고, claim_text는 바꾸지 마세요.
- 지시어만 있고 검증 가능한 술어/정의/수치/관계가 없으면 추출하지 마세요.

### 복수 claim 추출
하나의 발화에 여러 주장이 섞여 있으면 반드시 각각 별도 claim으로 추출하세요.
특히 "요즘/현재/최근/추세/주류" 표현은 별도 currentness claim으로 추출하세요.
하나의 발화 안에 수치가 2개 이상 나오면, 각각이 독립적으로 검증 가능한 값인지 확인하고 가능한 한 분리하세요.
문장이 불완전해 보여도 직전 발화와 결합하면 검증 가능한 수치/기준 claim이 되면 추출하세요.

### 인접 조각 발화 병합
- 전사 분할 때문에 하나의 문장이 여러 utterance로 나뉜 경우, 조각을 각각 claim으로 만들지 마세요.
- 전후 조각을 합쳐야 하나의 검증 가능한 정의/관계/수치/인과 명제가 되는 경우에만 하나의 claim으로 합쳐 추출하세요.
- 대상명/예시 목록만 있는 조각과, 바로 뒤의 분류명/명칭/포함 관계 술어 조각은 결합해야 검증 가능한 claim이 됩니다.
- 이때 utterance_id는 claim의 시작 발화로 두고, 가능하면 utterance_ids에 포함된 발화 ID 배열을 넣으세요.
- 술어가 끝나지 않은 조각, 연결어로 끝난 조각, 목적어만 남은 조각은 단독 claim으로 출력하지 마세요.

주의: 말실수나 용어 착각으로 보이더라도, 학생이 그대로 믿으면 틀린 지식이 되는 경우는 추출 대상입니다.
주의: 강의자가 예시 상황 안에서 기준값이나 정량적 조건을 말하면, 그 예시 범위 안의 검증 가능한 claim으로 추출하세요.
주의: 문장이 질문형으로 시작하더라도, 뒤에서 강의자가 특정 값이나 기준을 제시하면 그 제시된 값/기준은 claim으로 추출하세요.
주의: 다만 예시 속 기준값을 일반 상식/보편 법칙으로 확대 해석하지 마세요. 예시의 범위가 드러나면 claim_text에도 그 예시 범위를 남기세요.

{_model_specific_extract_rules()}

### 출력 (JSON만)
```json
{{
  "claims": [
    {{
      "utterance_id": "S001-C001",
      "context_id": "S001-C001",
      "claim_type": "definition",
      "claim_text": "현재 발화에서 직접 가져온 claim 원문",
      "source_slice": "이 claim을 직접 만든 최소 원문 조각",
      "utterance_ids": ["S001-C001"],
      "context_ids": ["S001-C001"],
      "antecedent_context_ids": [],
      "is_approximate": false,
      "needs_context": true,
      "resolution_status": "unresolved",
      "context_note": ""
    }}
  ]
}}
```

지침:
- 검증 불가능한 주장은 추출하지 마세요.
- 하나의 발화에서 여러 claim이 나올 수 있습니다.
- claim_type은 반드시 `definition`, `numeric`, `causal`, `relationship`, `currentness` 중 하나만 사용하세요.
- resolution_status는 `resolved` 또는 `unresolved`만 사용하세요.
- 입력 ID가 `S001-C001`이면 출력 ID도 `S001-C001` 형식으로 쓰세요. 입력 ID가 `U0001`이면 `U0001` 형식으로 그대로 쓰세요.
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


def _utterance_number(uid: str) -> int | None:
    match = re.search(r"(\d+)$", str(uid or ""))
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _utterance_index(utterances: list[dict]) -> dict[str, tuple[int, dict]]:
    return {
        str(utterance.get("utterance_id") or "").strip(): (idx, utterance)
        for idx, utterance in enumerate(utterances)
        if str(utterance.get("utterance_id") or "").strip()
    }


def _continuous_source_span_ids(
    ids: list[str],
    utterances: list[dict],
    max_gap: int = 12,
) -> list[str]:
    indexed = _utterance_index(utterances)
    present = [(indexed[uid][0], uid, indexed[uid][1]) for uid in _dedupe_ids(ids) if uid in indexed]
    if not present:
        return _dedupe_ids(ids)
    present.sort(key=lambda item: item[0])
    start_idx = present[0][0]
    end_idx = present[-1][0]
    if end_idx < start_idx or end_idx - start_idx > max_gap:
        return _dedupe_ids([uid for _, uid, _ in present])

    slides = {
        int(utterance.get("slide_number", 0) or 0)
        for idx in range(start_idx, end_idx + 1)
        for utterance in [utterances[idx]]
        if int(utterance.get("slide_number", 0) or 0)
    }
    if len(slides) > 1:
        return _dedupe_ids([uid for _, uid, _ in present])

    return _dedupe_ids(
        str(utterance.get("utterance_id") or "").strip()
        for utterance in utterances[start_idx : end_idx + 1]
    )


def _source_text_for_ids(ids: list[str], utterances: list[dict]) -> str:
    indexed = _utterance_index(utterances)
    rows = [(indexed[uid][0], indexed[uid][1]) for uid in _dedupe_ids(ids) if uid in indexed]
    rows.sort(key=lambda item: item[0])
    pieces = [
        " ".join(str(utterance.get("text") or "").split()).strip()
        for _, utterance in rows
    ]
    return " ".join(piece for piece in pieces if piece).strip()


def _attach_source_span_fields(claim: dict, utterances: list[dict]) -> dict:
    updated = dict(claim)
    uid = str(updated.get("utterance_id") or "").strip()
    antecedent_ids = _dedupe_ids([
        str(value).strip()
        for value in updated.get("antecedent_context_ids", []) or []
        if str(value).strip()
    ])
    raw_utterance_ids = updated.get("utterance_ids")
    if isinstance(raw_utterance_ids, list) and raw_utterance_ids:
        direct_ids = _dedupe_ids([str(value).strip() for value in raw_utterance_ids if str(value).strip()])
    else:
        direct_ids = [uid] if uid else []

    anchor_ids = _dedupe_ids(updated.get("anchor_utterance_ids", []) or [])
    if not anchor_ids:
        anchor_ids = _dedupe_ids(antecedent_ids + direct_ids + ([uid] if uid else []))
    source_span_ids = _dedupe_ids(updated.get("source_span_ids", []) or [])
    if not source_span_ids:
        source_span_ids = _continuous_source_span_ids(anchor_ids or direct_ids or ([uid] if uid else []), utterances)
    source_text = _source_text_for_ids(source_span_ids, utterances)

    updated["anchor_utterance_ids"] = anchor_ids
    updated["source_span_ids"] = source_span_ids
    # Keep source text only as a temporary post-processing aid. Persisting the
    # full context on every claim duplicates the batch context used downstream.
    updated["_source_text_for_restore"] = source_text

    # Backward compatibility: existing downstream stages read utterance_ids as
    # the related source range. The precise claim anchors remain in
    # anchor_utterance_ids.
    if source_span_ids:
        updated["utterance_ids"] = source_span_ids
    if source_span_ids and str(updated.get("context_id") or "").startswith("S"):
        updated["context_ids"] = source_span_ids

    claim_text = " ".join(str(updated.get("claim_text") or "").split()).strip()
    updated["claim_text"] = claim_text
    return updated


def _claim_utterance_id_set(claim: dict) -> set[str]:
    values = claim.get("utterance_ids")
    if not isinstance(values, list) or not values:
        values = [claim.get("utterance_id")]
    return {str(value) for value in values if str(value or "").strip()}


def _text_tokens(value: str) -> set[str]:
    return {token for token in re.split(r"\s+", str(value or "").strip()) if token}


def _similar_claim_text(a: dict, b: dict) -> bool:
    a_text = str(a.get("claim_text") or "")
    b_text = str(b.get("claim_text") or "")
    a_compact = re.sub(r"\s+", "", a_text)
    b_compact = re.sub(r"\s+", "", b_text)
    if not a_compact or not b_compact:
        return False
    if a_compact in b_compact or b_compact in a_compact:
        return True
    a_tokens = _text_tokens(a_text)
    b_tokens = _text_tokens(b_text)
    if not a_tokens or not b_tokens:
        return False
    return len(a_tokens & b_tokens) / max(1, min(len(a_tokens), len(b_tokens))) >= 0.65


def _dedupe_overlapping_claims(claims: list[dict]) -> list[dict]:
    deduped: list[dict] = []
    for claim in claims:
        claim_ids = _claim_utterance_id_set(claim)
        replaced = False
        for idx, existing in enumerate(deduped):
            if str(existing.get("claim_type") or "") != str(claim.get("claim_type") or ""):
                continue
            existing_ids = _claim_utterance_id_set(existing)
            if not (claim_ids <= existing_ids or existing_ids <= claim_ids):
                continue
            if not _similar_claim_text(existing, claim):
                continue
            existing_span = len(existing_ids)
            claim_span = len(claim_ids)
            existing_text_len = len(str(existing.get("claim_text") or ""))
            claim_text_len = len(str(claim.get("claim_text") or ""))
            if (claim_span, claim_text_len) > (existing_span, existing_text_len):
                deduped[idx] = claim
            replaced = True
            break
        if not replaced:
            deduped.append(claim)
    return deduped


def _extract_claims(
    utterances: list[dict],
    current_date: str,
    hint: dict,
    slide_ctx: dict,
    target_utterance_ids: set[str] | None = None,
) -> tuple[list[dict], bool, int, dict]:
    from . import claim_common as cv

    if not utterances:
        return [], False, 0, cv._empty_token_usage()

    prompt = _build_extract_prompt(
        utterances,
        current_date,
        hint,
        slide_ctx,
        target_utterance_ids=target_utterance_ids,
    )
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
                c["utterance_id"] = _normalize_prompt_ref(c.get("utterance_id"))
                claim_text = str(c.get("claim_text") or "").strip()
                if not claim_text:
                    continue
                normalized = _normalize_claim_type(c.get("claim_type"))
                if normalized is None:
                    continue
                c["claim_type"] = normalized
                c["claim_text"] = claim_text
                utterance_ids = c.get("utterance_ids")
                if isinstance(utterance_ids, list):
                    c["utterance_ids"] = _dedupe_ids(utterance_ids)
                else:
                    c["utterance_ids"] = [str(c.get("utterance_id") or "")]
                context_note = str(c.get("context_note") or "").strip()
                antecedent_ids = c.get("antecedent_context_ids")
                if isinstance(antecedent_ids, list):
                    antecedent_ids = _dedupe_ids(antecedent_ids)
                else:
                    antecedent_ids = []
                antecedent_ids = _dedupe_ids(antecedent_ids + _ids_from_context_note(context_note))
                if antecedent_ids:
                    c["antecedent_context_ids"] = antecedent_ids
                c["anchor_utterance_ids"] = _dedupe_ids(antecedent_ids + c["utterance_ids"])
                context_id = _normalize_prompt_ref(c.get("context_id"))
                if not context_id and str(c.get("utterance_id") or "").startswith("S"):
                    context_id = str(c.get("utterance_id") or "").strip()
                if context_id:
                    c["context_id"] = context_id
                    context_ids = c.get("context_ids")
                    if isinstance(context_ids, list):
                        c["context_ids"] = _dedupe_ids(context_ids)
                    else:
                        c["context_ids"] = [context_id]
                    if antecedent_ids:
                        c["context_ids"] = _dedupe_ids(antecedent_ids + c["context_ids"])
                c = _attach_source_span_fields(c, utterances)
                source_text_for_restore = str(c.get("_source_text_for_restore", "") or "")
                if _has_model_added_parenthetical(c.get("claim_text", ""), source_text_for_restore):
                    restored_claim_text = _restore_parenthetical_claim_to_source(
                        c.get("claim_text", ""),
                        source_text_for_restore,
                    )
                    if not restored_claim_text:
                        continue
                    c["claim_text"] = restored_claim_text
                c["source_slice"] = _normalize_source_slice(
                    c.get("source_slice"),
                    c.get("claim_text", ""),
                    source_text_for_restore,
                )
                c.pop("_source_text_for_restore", None)
                c.pop("source_text", None)
                c.pop("claim_context_text", None)
                c.pop("source_block_id", None)
                c.pop("claim_context_block_id", None)
                c.pop("claim_context_ids", None)
                c["claim_id"] = str(c.get("claim_id") or "").strip() or _stable_claim_id(c)
                c["claim_fingerprint"] = (
                    str(c.get("claim_fingerprint") or "").strip()
                    or c["claim_id"]
                )
                c["needs_context"] = True
                c["resolution_status"] = "unresolved"
                c["context_note"] = ""
                allowed_claim_keys = {
                    "utterance_id",
                    "context_id",
                    "claim_type",
                    "claim_text",
                    "source_slice",
                    "utterance_ids",
                    "context_ids",
                    "antecedent_context_ids",
                    "anchor_utterance_ids",
                    "source_span_ids",
                    "is_approximate",
                    "needs_context",
                    "resolution_status",
                    "context_note",
                    "claim_id",
                    "claim_fingerprint",
                }
                c = {k: v for k, v in c.items() if k in allowed_claim_keys}
                cleaned.append(c)
            cleaned = _dedupe_overlapping_claims(cleaned)
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
    target_utterance_ids: set[str] | None = None,
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
                utterances, current_date, hint, slide_ctx, target_utterance_ids=target_utterance_ids
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
    max_workers: int | None = None,
) -> tuple[list[tuple], int, dict]:
    """1단계만 실행: claim 추출. (batch_list, total_claims) 반환."""
    from . import claim_common as cv

    if batch_size is None:
        batch_size = cv.BATCH_SIZE

    all_claims_by_batch = []
    total_api = 0
    total_token_usage = cv._empty_token_usage()
    total = len(utterances)
    has_context_units = any(u.get("context_id") for u in utterances)
    batch_mode = (
        "context"
        if has_context_units and _force_context_unit_batching()
        else (_claim_extract_batch_mode() if has_context_units else "utterance")
    )
    context_prev, context_next = _claim_extract_context_window()

    if has_context_units and batch_mode == "context":
        context_group_size = _claim_extract_context_group_size()
        core_ranges = [
            (i, min(total, i + context_group_size))
            for i in range(0, total, context_group_size)
        ]
    elif has_context_units and batch_mode == "slide":
        core_ranges = []
        start = 0
        while start < total:
            slide_no = int(utterances[start].get("slide_number", 0) or 0)
            end = start + 1
            while end < total and int(utterances[end].get("slide_number", 0) or 0) == slide_no:
                end += 1
            core_ranges.append((start, end))
            start = end
    else:
        core_ranges = [(i, min(total, i + batch_size)) for i in range(0, total, batch_size)]
    worker_count = min(_claim_extract_max_workers(max_workers), max(1, len(core_ranges)))

    jobs = []
    for i, (start, end) in enumerate(core_ranges):
        core_batch = utterances[start:end]
        if not core_batch:
            continue
        if has_context_units and batch_mode in {"context", "slide"}:
            context_batch = utterances[max(0, start - context_prev):min(total, end + context_next)]
            core_uids = {str(u.get("utterance_id") or "") for u in core_batch}
        else:
            context_batch = core_batch
            core_uids = None

        ids = f"{core_batch[0]['utterance_id']}..{core_batch[-1]['utterance_id']}"
        jobs.append((i, context_batch, core_uids, ids))

    def _run_job(job):
        i, context_batch, core_uids, ids = job
        print(f"    추출 [{i+1}/{len(core_ranges)}] {ids}")
        claims, parse_failed, api_calls, token_usage, ok = recover_claim_extraction(
            context_batch,
            current_date,
            hint,
            slide_ctx,
            f"배치 {i+1} {ids}",
            target_utterance_ids=core_uids if has_context_units else None,
        )
        if core_uids is not None:
            claims = [c for c in claims if str(c.get("utterance_id") or "") in core_uids]
        return i, context_batch, claims, api_calls, token_usage

    if worker_count > 1 and len(jobs) > 1:
        print(f"  claim 추출 병렬 처리: max_workers={worker_count}", flush=True)
        results = {}
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {executor.submit(_run_job, job): job[0] for job in jobs}
            for future in as_completed(futures):
                job_idx = futures[future]
                try:
                    results[job_idx] = future.result()
                except Exception as e:
                    raise RuntimeError(f"claim extraction batch {job_idx + 1} failed: {e}") from e
        ordered_results = [results[job[0]] for job in jobs]
    else:
        ordered_results = [_run_job(job) for job in jobs]

    for _i, context_batch, claims, api_calls, token_usage in ordered_results:
        all_claims_by_batch.append((context_batch, claims))
        total_api += api_calls
        total_token_usage = cv._merge_token_usage(total_token_usage, token_usage)

    total_claims = sum(len(c) for _, c in all_claims_by_batch)
    print(f"  추출된 claim: {total_claims}개")
    return all_claims_by_batch, total_api, total_token_usage
