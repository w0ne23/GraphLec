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


def _claim_extract_context_mode() -> str:
    mode = str(os.getenv("VERIFIER_CLAIM_EXTRACT_CONTEXT_MODE", "cards") or "cards").strip().lower()
    if mode in {"compact", "batch", "transcript"}:
        return "compact"
    if mode in {"card_lite", "card-lite", "lite", "cards_lite", "cards-lite"}:
        return "card_lite"
    return "cards"


def _claim_extract_batch_mode() -> str:
    mode = str(os.getenv("VERIFIER_CLAIM_EXTRACT_BATCH_MODE", "context") or "context").strip().lower()
    if mode in {"slide", "slides", "slide_batch", "slide-batch", "by_slide", "by-slide"}:
        return "slide"
    return "context"


def _claim_extract_context_window() -> tuple[int, int]:
    return (
        _read_int_env("VERIFIER_CLAIM_EXTRACT_CONTEXT_PREV", 2, minimum=0),
        _read_int_env("VERIFIER_CLAIM_EXTRACT_CONTEXT_NEXT", 1, minimum=0),
    )


def _read_int_env(name: str, default: int, *, minimum: int = 1) -> int:
    raw = str(os.getenv(name, str(default)) or str(default)).strip()
    try:
        return max(minimum, int(raw))
    except (TypeError, ValueError):
        return default


def _claim_extract_batch_size(default: int | None = None) -> int:
    if default is not None:
        return max(1, int(default))
    fallback = _read_int_env("VERIFIER_BATCH_SIZE", 2, minimum=1)
    return _read_int_env("VERIFIER_CLAIM_EXTRACT_BATCH_SIZE", fallback, minimum=1)


def _claim_extract_max_workers(default: int | None = None) -> int:
    if default is not None:
        return max(1, int(default))
    return _read_int_env("VERIFIER_CLAIM_EXTRACT_MAX_WORKERS", 4, minimum=1)


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
- "정의한다면", "원가라는 것은", "재화나 용역을", "얻기 위해서 희생한"처럼 술어가 끝나지 않은 조각은 단독 claim으로 출력하지 마세요.

2. 전수 확인의 의미
- 검사 대상 발화마다 claim 후보가 있는지 확인하라는 뜻이지, 모든 검사 대상 발화를 claim으로 만들라는 뜻이 아닙니다.
- 검증 가능한 완성 명제가 없는 검사 대상 발화는 과감히 출력하지 마세요.
- 강의 일정, 연락 방법, 수업 준비물, 공지 확인처럼 강의 운영에 관한 발언은 수치나 정책 자체가 핵심 검증 대상일 때만 추출하세요.

3. 문맥 결합
- 문맥은 문장 조각을 완성하거나 지시어를 보수적으로 해소하기 위한 보조 정보입니다.
- 문맥을 이용해 현재 발화가 실제로 말하지 않은 주체, 조건, 인과, 일반 법칙을 새로 만들지 마세요.
- 여러 발화가 하나의 문장을 이룰 때는 하나의 claim으로 합치고, utterance_ids에 포함하세요.

4. resolved_claim 작성
- resolved_claim은 원문보다 넓어지면 안 됩니다.
- 하나의 원문에 서로 다른 명제가 있으면 하나로 요약하지 말고 분리하세요.
- 원문에 있는 중요한 부정/한정 표현을 덮어쓰지 마세요. 예: "A와 직접 관련 없다"와 "B를 제공한다"는 별도 claim입니다.
"""


def _build_slide_references(slide_numbers: list[int], slide_ctx: dict) -> str:
    refs = []
    mode = _claim_extract_context_mode()
    for sn in slide_numbers:
        ctx = slide_ctx.get(sn, {})
        title = ctx.get("title", f"슬라이드 {sn}")
        time_range = ctx.get("time_range", "")
        slide_text = str(ctx.get("slide_text", "") or "")
        limit = 900 if mode in {"compact", "card_lite"} else 1200
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
    from . import claim_common as cv

    total = len(utterances)
    seen_slides = OrderedDict()
    cards = []
    mode = _claim_extract_context_mode()

    if mode == "compact":
        transcript_lines = []
        for u in utterances:
            current_sn = int(u.get("slide_number", 0) or 0)
            if current_sn not in seen_slides:
                seen_slides[current_sn] = True
            transcript_lines.append(f"  {cv._format_utterance_for_prompt(u)}")
        context = (
            "[배치 발화 전체]\n"
            + "\n".join(transcript_lines)
            + "\n\n[문맥 사용 규칙]\n"
            + "- 각 claim은 반드시 위 발화의 utterance_id에 귀속하세요.\n"
            + "- 앞뒤 문맥은 위 발화 순서 안에서 참고하세요.\n"
            + "- 지시어 선행사가 이 배치 밖에 있어 확정되지 않으면 원문 claim을 유지하고 needs_context=true로 표시하세요."
        )
        return _build_slide_references(list(seen_slides.keys()), slide_ctx), context

    if mode == "card_lite":
        target_ids = target_utterance_ids or {str(u.get("utterance_id") or "") for u in utterances}
        transcript_lines = []
        target_lines = []
        for u in utterances:
            current_sn = int(u.get("slide_number", 0) or 0)
            if current_sn not in seen_slides:
                seen_slides[current_sn] = True
            uid = str(u.get("utterance_id") or "")
            marker = "*" if uid in target_ids else " "
            line = f"{marker} {cv._format_utterance_for_prompt(u)}"
            transcript_lines.append(line)
            if uid in target_ids:
                target_lines.append(f"- {uid} (슬라이드 {current_sn})")
        context = (
            "[배치 전체 전사]\n"
            + "\n".join(transcript_lines)
            + "\n\n[검사 대상 발화]\n"
            + "\n".join(target_lines)
            + "\n\n[card-lite 추출 규칙]\n"
            + "- '*' 표시된 검사 대상 발화는 각각 독립적으로 검토하세요.\n"
            + "- claim이 있는 검사 대상 발화는 빠뜨리지 말고 claims 배열에 출력하세요.\n"
            + "- claim이 없는 검사 대상 발화는 출력하지 않아도 됩니다.\n"
            + "- '*'가 없는 발화는 앞뒤 문맥, 지시어 해소, 조각 문장 완결에만 사용하세요.\n"
            + "- '*'가 없는 발화를 대표 utterance_id로 새 claim을 만들지 마세요.\n"
            + "- 검사 대상 발화의 claim이 문맥 발화와 결합되어야 완성되면 대표 utterance_id는 검사 대상 발화로 두고, 결합된 발화들은 utterance_ids에 포함하세요.\n"
            + "- 긴 배치 전체에서 중요한 것만 선별하지 말고, 검사 대상 발화마다 검증 가능한 정의/수치/인과/관계/현재성 주장이 있는지 전수 확인하세요."
        )
        return _build_slide_references(list(seen_slides.keys()), slide_ctx), context

    if target_utterance_ids:
        target_lines = []
        reference_lines = []
        for u in utterances:
            current_sn = int(u.get("slide_number", 0) or 0)
            if current_sn not in seen_slides:
                seen_slides[current_sn] = True
            uid = str(u.get("utterance_id") or "")
            line = f"  {cv._format_utterance_for_prompt(u)}"
            if uid in target_utterance_ids:
                target_lines.append(line)
            else:
                reference_lines.append(line)

        context = (
            "[검사 대상 context]\n"
            + ("\n".join(target_lines) if target_lines else "(없음)")
        )
        if reference_lines:
            context += (
                "\n\n[참고 context - 지시어 해소/문장 완결용]\n"
                + "\n".join(reference_lines)
            )
        context += (
            "\n\n[context-window 추출 규칙]\n"
            "- claim은 반드시 검사 대상 context에서 직접 말한 내용만 추출하세요.\n"
            "- 참고 context는 지시어 선행사 해소, 생략된 주어 확인, 조각 문장 완결에만 사용하세요.\n"
            "- 참고 context에만 있는 새 claim을 만들지 마세요.\n"
            "- 지시어가 단일 후보로 확실히 해소되면 resolved_claim에 최소한으로 반영하세요.\n"
            "- 참고 context로 지시어를 해소한 경우 antecedent_context_ids에 사용한 context id를 넣으세요.\n"
            "- 후보가 둘 이상 가능하거나 검사 대상/참고 context/슬라이드 안에서 확정되지 않으면 unresolved로 두세요."
        )
        return _build_slide_references(list(seen_slides.keys()), slide_ctx), context

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

    return _build_slide_references(list(seen_slides.keys()), slide_ctx), "\n\n".join(cards)


def _build_extract_prompt(
    utterances: list[dict],
    current_date: str,
    hint: dict,
    slide_ctx: dict,
    target_utterance_ids: set[str] | None = None,
) -> str:
    slide_references, utterance_cards = _build_slide_and_utterance_context(
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

### 핵심 원칙: 1단계는 raw claim inventory입니다
- 이 단계에서는 오류 여부, 오해 가능성, 교수 피드백, 반례를 판단하지 마세요.
- 반드시 **현재 발화 자체가 명시한 주장**만 claim으로 추출하세요.
- claim_text는 현재 발화에서 직접 가져온 가장 짧은 원문 조각으로 쓰세요.
- resolved_claim은 원문 claim의 범위를 보존한 짧은 정리문입니다.
- resolved_claim에서 새로운 주체, 조건, 원인, 반례, 일반 법칙을 만들지 마세요.
- 주변 문맥과 슬라이드는 현재 발화가 claim인지, 예시인지, 지시어가 명확한지만 판단하는 보조 정보입니다.
- 주변 문맥에 있는 더 강한 일반 명제를 현재 발화에 덧씌우지 마세요.
- 현재 발화가 예시/가정/비유/수사적 요약이면, resolved_claim에도 그 예시/가정/비유/요약 범위를 유지하세요.

### 지시어 처리
- "이것", "여기", "해당 항목", "얘", "이거", "그거" 같은 지시어는 단일 선행사가 확실할 때만 최소한으로 풀어 쓰세요.
- 둘 이상의 합리적 해석이 가능하면 특정 대상으로 확정하지 말고 원문 지시어를 유지하세요.
- 지시어 선행사가 제공된 문맥보다 앞에 있을 수 있어도, 현재 발화에 검증 가능한 술어/정의/수치/관계가 있으면
  claim을 버리지 말고 원문 그대로 추출하세요. 이때 resolved_claim은 claim_text와 같게 두고,
  needs_context=true, resolution_status="unresolved"로 표시하세요.
- 슬라이드나 앞뒤 발화가 보이더라도, 현재 발화가 실제로 말하지 않은 관계를 resolved_claim에 추가하지 마세요.
- 불완전한 조각 문장은 직전 발화와 결합해야 검증 가능한 값/정의/관계가 될 때만 추출하세요.
- 복원 결과가 애매하다는 이유만으로 "이것은 X이다", "얘는 Y로 처리된다" 같은 원문 claim을 버리지 마세요.
- 단, 현재 발화에 검증 가능한 술어가 없고 지시어만 남은 경우는 추출하지 마세요.

### 복수 claim 추출
하나의 발화에 여러 주장이 섞여 있으면 반드시 각각 별도 claim으로 추출하세요.
특히 "요즘/현재/최근/추세/주류" 표현은 별도 currentness claim으로 추출하세요.
하나의 발화 안에 수치가 2개 이상 나오면, 각각이 독립적으로 검증 가능한 값인지 확인하고 가능한 한 분리하세요.
문장이 불완전해 보여도 직전 발화와 결합하면 검증 가능한 수치/기준 claim이 되면 추출하세요.

### 인접 조각 발화 병합
- 전사 분할 때문에 하나의 문장이 여러 utterance로 나뉜 경우, 조각을 각각 claim으로 만들지 마세요.
- 예: "원가라는 것은" / "재화라든가 용역을" / "얻기 위해서 희생한" / "경제적 자원을 화폐 단위로" / "측정한 거다"
  → 하나의 definition claim으로 합쳐 추출하세요.
- 이때 utterance_id는 claim의 시작 발화로 두고, 가능하면 utterance_ids에 포함된 발화 ID 배열을 넣으세요.
- "X는", "X라는 것은", "A를", "B하기 위해서", "희생한"처럼 술어가 끝나지 않은 조각은 단독 claim으로 출력하지 마세요.

주의: 말실수나 용어 착각으로 보이더라도, 학생이 그대로 믿으면 틀린 지식이 되는 경우는 추출 대상입니다.
주의: 강의자가 예시 상황 안에서 기준값이나 정량적 조건을 말하면, 그 예시 범위 안의 검증 가능한 claim으로 추출하세요.
주의: 문장이 질문형으로 시작하더라도, 뒤에서 강의자가 특정 값이나 기준을 제시하면 그 제시된 값/기준은 claim으로 추출하세요.
주의: 다만 예시 속 기준값을 일반 상식/보편 법칙으로 확대 해석하지 마세요. 예시의 범위가 드러나면 resolved_claim에도 그 예시 범위를 남기세요.

{_model_specific_extract_rules()}

### 출력 (JSON만)
```json
{{
  "claims": [
    {{
      "utterance_id": "U0001",
      "claim_type": "definition",
      "claim_text": "현재 발화에서 직접 가져온 claim 원문",
      "resolved_claim": "원문 범위를 보존한 최소 정리문",
      "utterance_ids": ["U0001"],
      "antecedent_context_ids": [],
      "is_approximate": false,
      "needs_context": false,
      "resolution_status": "resolved",
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
- verification_question은 생성하지 마세요. 검증 질문은 후속 판정 단계에서 필요한 claim에만 만듭니다.
- resolved_claim을 쓰기 애매하면 claim_text와 동일하게 두세요.
- 참고 context로 지시어를 해소한 경우 antecedent_context_ids에 사용한 context id를 넣으세요.
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


def _build_claim_fingerprint(claim: dict) -> str:
    anchor = str(claim.get("context_id") or claim.get("utterance_id") or "CTX").strip()
    anchor = re.sub(r"[^A-Za-z0-9_-]+", "-", anchor).strip("-") or "CTX"
    source = "|".join(
        str(claim.get(key) or "")
        for key in ("claim_type", "claim_text", "resolved_claim")
    )
    digest = hashlib.sha1(source.encode("utf-8")).hexdigest()[:10]
    return f"CLM-{anchor}-{digest}"


def assign_claim_display_ids(claims_by_batch: list[tuple]) -> None:
    """최종 추출 순서 기준으로 사람이 읽는 claim_id를 부여한다."""
    sequence = 1
    for _batch, claims in claims_by_batch:
        for claim in claims:
            claim["claim_id"] = f"CL{sequence:04d}"
            _order_claim_fields(claim)
            sequence += 1


def _order_claim_fields(claim: dict) -> None:
    preferred_keys = (
        "claim_id",
        "claim_text",
        "resolved_claim",
        "claim_type",
        "context_id",
        "context_ids",
        "antecedent_context_ids",
        "claim_fingerprint",
        "utterance_id",
        "utterance_ids",
        "is_approximate",
        "needs_context",
        "resolution_status",
        "context_note",
    )
    ordered = {key: claim[key] for key in preferred_keys if key in claim}
    ordered.update({key: value for key, value in claim.items() if key not in ordered})
    claim.clear()
    claim.update(ordered)


def _utterance_number(uid: str) -> int | None:
    match = re.search(r"(\d+)$", str(uid or ""))
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _is_fragment_like_claim(claim: dict) -> bool:
    text = str(claim.get("claim_text") or claim.get("resolved_claim") or "").strip()
    compact = re.sub(r"\s+", "", text)
    if not compact:
        return False
    if len(compact) <= 18:
        return True
    fragment_endings = (
        "것은",
        "것이",
        "것을",
        "용역을",
        "재화를",
        "자원을",
        "위해서",
        "희생한",
        "통해서",
        "가운데서",
        "경우는",
        "한다면",
        "그리고",
        "또는",
        "라든가",
    )
    return compact.endswith(fragment_endings)


def _cut_complete_sentence(text: str) -> tuple[str, bool]:
    stripped = " ".join(str(text or "").split()).strip()
    if not stripped:
        return "", False

    punctuation_positions = [pos for pos in (stripped.find("."), stripped.find("?"), stripped.find("!")) if pos >= 0]
    if punctuation_positions:
        end = min(punctuation_positions) + 1
        return stripped[:end].strip(), True

    sentence_endings = ("입니다", "됩니다", "합니다", "합니다", "이다", "된다", "한다", "했다", "했다", "거다", "겁니다", "겠죠", "이죠", "죠")
    for ending in sentence_endings:
        if stripped.endswith(ending):
            return stripped, True
    return stripped, False


def _expand_fragment_claim(claim: dict, utterances: list[dict]) -> tuple[str, list[str], bool]:
    uid = str(claim.get("utterance_id") or "")
    start_idx = None
    for idx, utterance in enumerate(utterances):
        if str(utterance.get("utterance_id") or "") == uid:
            start_idx = idx
            break
    if start_idx is None:
        return str(claim.get("claim_text") or "").strip(), [uid] if uid else [], False

    pieces = []
    ids = []
    start_slide = int(utterances[start_idx].get("slide_number", 0) or 0)
    for utterance in utterances[start_idx:min(len(utterances), start_idx + 8)]:
        current_slide = int(utterance.get("slide_number", 0) or 0)
        if ids and start_slide and current_slide and current_slide != start_slide:
            break
        ids.append(str(utterance.get("utterance_id") or ""))
        pieces.append(str(utterance.get("text") or "").strip())
        joined, complete = _cut_complete_sentence(" ".join(pieces))
        if complete:
            return joined, [x for x in ids if x], True
        if len(joined) >= 220:
            return joined, [x for x in ids if x], False
    joined, complete = _cut_complete_sentence(" ".join(pieces))
    return joined, [x for x in ids if x], complete


def _merge_fragment_claims(claims: list[dict], utterances: list[dict]) -> list[dict]:
    if _claim_extract_context_mode() != "compact":
        return claims

    covered_fragment_uids: set[str] = set()
    merged: list[dict] = []
    for claim in claims:
        uid = str(claim.get("utterance_id") or "")
        is_fragment = _is_fragment_like_claim(claim)
        if uid in covered_fragment_uids:
            continue
        if not is_fragment:
            merged.append(claim)
            continue

        expanded_text, utterance_ids, complete = _expand_fragment_claim(claim, utterances)
        if len(utterance_ids) <= 1 or not expanded_text:
            merged.append(claim)
            continue

        updated = dict(claim)
        updated["claim_text"] = expanded_text
        updated["resolved_claim"] = expanded_text
        updated["utterance_ids"] = utterance_ids
        updated["needs_context"] = not complete
        updated["resolution_status"] = "resolved" if complete else "unresolved"
        note = "인접 조각 발화를 원본 전사 순서로 자동 병합"
        if updated.get("context_note"):
            updated["context_note"] = f"{updated['context_note']}; {note}"
        else:
            updated["context_note"] = note
        covered_fragment_uids.update(utterance_ids[1:])
        merged.append(updated)

    return merged


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


def _context_batches_by_slide(utterances: list[dict]) -> list[list[dict]]:
    batches: list[list[dict]] = []
    current_slide = None
    current_batch: list[dict] = []
    for utterance in utterances:
        slide_no = int(utterance.get("slide_number", 0) or 0)
        if current_batch and slide_no != current_slide:
            batches.append(current_batch)
            current_batch = []
        current_slide = slide_no
        current_batch.append(utterance)
    if current_batch:
        batches.append(current_batch)
    return batches


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
                claim_text = str(c.get("claim_text") or "").strip()
                if not claim_text:
                    continue
                resolved_claim = str(c.get("resolved_claim") or "").strip() or claim_text
                normalized = _normalize_claim_type(c.get("claim_type"))
                if normalized is None:
                    continue
                resolution_status = str(c.get("resolution_status") or "").strip().lower()
                if resolution_status not in {"resolved", "unresolved"}:
                    resolution_status = "unresolved" if c.get("needs_context") else "resolved"
                c["claim_type"] = normalized
                c["claim_text"] = claim_text
                c["resolved_claim"] = resolved_claim
                if not c.get("context_id"):
                    c["context_id"] = str(c.get("utterance_id") or "")
                utterance_ids = c.get("utterance_ids")
                if isinstance(utterance_ids, list):
                    c["utterance_ids"] = [str(x) for x in utterance_ids if str(x).strip()]
                else:
                    c["utterance_ids"] = [str(c.get("utterance_id") or "")]
                if not c.get("context_ids"):
                    c["context_ids"] = list(c["utterance_ids"])
                antecedent_ids = c.get("antecedent_context_ids")
                if isinstance(antecedent_ids, list):
                    c["antecedent_context_ids"] = [str(x) for x in antecedent_ids if str(x).strip()]
                else:
                    c["antecedent_context_ids"] = []
                c["claim_fingerprint"] = str(c.get("claim_fingerprint") or _build_claim_fingerprint(c))
                c.pop("claim_id", None)
                c["needs_context"] = bool(c.get("needs_context") or resolution_status == "unresolved")
                c["resolution_status"] = resolution_status
                c["context_note"] = str(c.get("context_note") or "").strip()
                c.pop("verification_question", None)
                c.pop("verificationQuestion", None)
                cleaned.append(c)
            cleaned = _merge_fragment_claims(cleaned, utterances)
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

    batch_size = _claim_extract_batch_size(batch_size)
    max_workers = _claim_extract_max_workers(max_workers)

    all_claims_by_batch = []
    total_api = 0
    total_token_usage = cv._empty_token_usage()
    total = len(utterances)
    has_contexts = any(u.get("context_id") for u in utterances)
    context_window_prev, context_window_next = _claim_extract_context_window()
    if has_contexts:
        batch_mode = _claim_extract_batch_mode()
        if batch_mode == "slide":
            core_batches = _context_batches_by_slide(utterances)
        else:
            core_batches = [
                utterances[i:min(total, i + batch_size)]
                for i in range(0, total, batch_size)
            ]
        print(
            f"  claim 추출 batch mode: {batch_mode}"
            + (f" (prev={context_window_prev}, next={context_window_next})" if batch_mode == "context" else "")
        )
    else:
        batch_mode = "utterance"
        core_batches = [
            utterances[i:min(total, i + batch_size)]
            for i in range(0, total, batch_size)
        ]
    context_mode = _claim_extract_context_mode()
    shared_context_mode = context_mode in {"compact", "card_lite"}

    cursor = 0
    work_items = []
    for i, core_batch in enumerate(core_batches):
        if not core_batch:
            continue
        start = cursor
        end = cursor + len(core_batch)
        cursor = end
        if has_contexts and batch_mode == "context":
            context_batch = utterances[
                max(0, start - context_window_prev):min(total, end + context_window_next)
            ]
            core_uids = {str(u.get("utterance_id") or "") for u in core_batch}
        elif shared_context_mode and not any(u.get("context_id") for u in core_batch):
            context_batch = utterances[max(0, start - 3):min(total, end + 5)]
            core_uids = {str(u.get("utterance_id") or "") for u in core_batch}
        else:
            context_batch = core_batch
            core_uids = None

        ids = f"{core_batch[0]['utterance_id']}..{core_batch[-1]['utterance_id']}"
        work_items.append({
            "index": i,
            "context_batch": context_batch,
            "core_uids": core_uids,
            "ids": ids,
        })

    def _run_item(item: dict) -> tuple[int, list[dict], list[dict], int, dict]:
        ids = item["ids"]
        print(f"    추출 [{item['index']+1}/{len(core_batches)}] {ids}", flush=True)
        claims, parse_failed, api_calls, token_usage, ok = recover_claim_extraction(
            item["context_batch"],
            current_date,
            hint,
            slide_ctx,
            f"배치 {item['index']+1} {ids}",
            target_utterance_ids=item["core_uids"],
        )
        if not ok:
            print(f"    ⚠️ claim 추출 실패: {ids} — 이 batch는 빈 결과로 기록됩니다.")
        if item["core_uids"] is not None:
            claims = [c for c in claims if str(c.get("utterance_id") or "") in item["core_uids"]]
        return item["index"], item["context_batch"], claims, api_calls, token_usage

    if work_items:
        worker_count = min(max_workers, len(work_items))
        print(
            f"  claim 추출 병렬 실행: batch_size={batch_size}, workers={worker_count}, batches={len(work_items)}",
            flush=True,
        )
    if len(work_items) <= 1 or max_workers <= 1:
        results = [_run_item(item) for item in work_items]
    else:
        results = [None] * len(work_items)
        with ThreadPoolExecutor(max_workers=min(max_workers, len(work_items))) as executor:
            futures = {executor.submit(_run_item, item): item for item in work_items}
            for future in as_completed(futures):
                index, context_batch, claims, api_calls, token_usage = future.result()
                results[index] = (index, context_batch, claims, api_calls, token_usage)

    for _index, context_batch, claims, api_calls, token_usage in [r for r in results if r is not None]:
        all_claims_by_batch.append((context_batch, claims))
        total_api += api_calls
        total_token_usage = cv._merge_token_usage(total_token_usage, token_usage)

    total_claims = sum(len(c) for _, c in all_claims_by_batch)
    assign_claim_display_ids(all_claims_by_batch)
    print(f"  추출된 claim: {total_claims}개")
    return all_claims_by_batch, total_api, total_token_usage
