"""Final verifier for classified issue candidates.

This module consumes ``classified_issue_input.v1`` produced by
``issue_type_classifier.py`` and runs a category-specific final verifier over each
already-classified issue.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import json
import os
import re
from pathlib import Path
from typing import Any

from .issue_type_classifier import (
    ISSUE_TYPES,
    TOKEN_USAGE_FIELDS,
    _aggregate_token_usage,
    _call_llm,
    _load_env,
    _parse_model_weights,
    _resolve_model_spec,
    _split_csv,
)


SCHEMA_VERSION = "classified_issue_verifier.v1"
DEFAULT_MODEL_WEIGHTS = "gpt=0.4,claude=0.4,grok=0.2"
DEFAULT_MODELS = ("gpt", "claude", "grok")
DEFAULT_CONTEXT_WINDOW = 3
CATEGORY_MODEL_WEIGHT_OVERRIDES = {
    "scope_overclaim": {"gpt": 0.3, "openai": 0.3, "claude": 0.5, "anthropic": 0.5, "grok": 0.2, "xai": 0.2},
    "confusing_explanation": {"gpt": 0.3, "openai": 0.3, "claude": 0.5, "anthropic": 0.5, "grok": 0.2, "xai": 0.2},
}
JUDGMENTS = {
    "valid_issue",
    "partially_resolved",
    "not_issue",
    "insufficient_context",
}

CATEGORY_LABELS = {
    "factual_error": "사실 오류",
    "temporal_error": "시대적 오류",
    "confusing_explanation": "혼동 오류",
    "scope_overclaim": "범위 오류",
}


CATEGORY_DESCRIPTIONS = {
    "factual_error": (
        "정의, 용어, 동작 원리, 관계, 순서, 메커니즘, 수식, 인과관계 등 "
        "기준일과 무관하게 명제 자체가 객관적으로 틀린 사실 오류 resolved_claim입니다."
    ),
    "temporal_error": (
        "과거 어느 시점에는 맞았거나 자연스러웠을 수 있지만, 현재 기준으로는 "
        "더 이상 맞지 않거나 현재 학습자에게 outdated 정보로 전달될 수 있는 resolved_claim입니다."
    ),
    "confusing_explanation": (
        "명제가 명백히 틀렸다고 단정하기보다는, 비유/예시/생략/표현 방식 때문에 "
        "학생이 다른 의미로 해석하거나 잘못된 오개념을 만들 위험이 있는 resolved_claim입니다."
    ),
    "scope_overclaim": (
        "조건, 예외, 범위, 적용 대상을 닫아버려 과도하게 일반화한 resolved_claim입니다. "
        '"항상/오직/모든/유일한/전부/완전히/~만" 같은 범위 표현을 완화하면 '
        "대체로 맞는 명제가 되는 경우를 포함합니다."
    ),
}


CATEGORY_SCORE_GUIDES = {
    "factual_error": {
        "is_valid_issue": (
            "resolved_claim이 정의, 용어, 동작 원리, 관계, 순서, 수식, 인과관계 측면에서 "
            "객관적으로 틀렸을 가능성을 평가하세요. 문맥과 슬라이드를 포함해도 같은 잘못된 "
            "명제가 남아 있으면 높게 주고, 문맥상 표현이 바로 정정되었거나 정확한 의미로 "
            "좁혀지면 낮게 주세요. 영문/외래어 용어를 한글로 옮긴 발음 표기, 음차, 전사 "
            "흔들림만 문제이고 문맥상 어떤 원어와 개념을 가리키는지 명확하면 사실 오류로 "
            "높게 채점하지 마세요. 수치 표현에서 약, 한, 대략, 정도, 조금 같은 근사 표현이 "
            "있고 그 수치가 핵심 개념이 아니라 보조 설명, 감각적 환산, 예시로 쓰였으며 "
            "일반적으로 통용되는 근사라면 정확한 수치와 차이가 있어도 사실 오류로 높게 "
            "채점하지 마세요."
        ),
        "category_severity": (
            "학생이 핵심 개념, 작동 원리, 문제 풀이 방식, 구현 판단, 후속 개념 이해를 "
            "잘못 학습할 위험이 클수록 높게 주세요. 강의 흐름에 큰 영향을 주지 않는 "
            "사소한 부정확성, 발음 표기 차이, 음차/전사 흔들림, 보조 예시의 일반적 근사 "
            "표현은 낮게 주세요."
        ),
        "context_resolution": (
            "앞뒤 context와 슬라이드 텍스트가 잘못된 의미를 얼마나 정정, 보완, "
            "조건화하는지 평가하세요. 명시적 정정이나 충분한 보완이 있으면 높게 주고, "
            "문맥을 봐도 같은 오류가 그대로 남으면 낮게 주세요."
        ),
    },
    "temporal_error": {
        "is_valid_issue": (
            "resolved_claim이 기준일 현재 더 이상 맞지 않거나, 현재 학습자에게 outdated 정보로 "
            "전달될 가능성을 평가하세요. 단순히 강의가 오래되었거나 날짜 표현이 있다는 "
            "이유만으로 높게 주지 마세요."
        ),
        "category_severity": (
            "현재 학습자의 도구 선택, 구현 방식, 지원 여부, 정책/버전 판단, 통계나 시장 상황 "
            "이해에 실제 영향을 줄수록 높게 주세요. 역사적 배경 설명이거나 현재 학습에 영향이 적다면 낮게 주세요."
        ),
        "context_resolution": (
            "문맥상 과거 시점, 역사적 상황, 당시 기준의 설명으로 명확히 제한되어 있으면 높게 주세요. "
            "현재 사실처럼 제시되고 보완 설명이 없으면 낮게 주세요."
        ),
    },
    "confusing_explanation": {
        "is_valid_issue": (
            "비유, 예시, 생략, 모호한 지시어, 압축된 표현 때문에 학생이 resolved_claim을 다른 의미로 "
            "해석하거나 잘못된 mental model을 만들 가능성을 평가하세요. 단순히 더 친절한 "
            "설명이 가능하다는 이유만으로 높게 주지 마세요."
            "일반적으로 맞는 표현이며, 일반적으로 맞는 설명이면 낮은 점수를 주세요."
        ),
        "category_severity": (
            "그 오해가 핵심 개념, 절차, 원인-결과, 구성 요소의 역할 이해를 크게 왜곡할수록 "
            "높게 주세요. 잠깐 헷갈릴 수 있으나 뒤 학습에 거의 영향을 주지 않는 표현은 낮게 주세요."
        ),
        "context_resolution": (
            "앞뒤 설명이 오해 가능성을 얼마나 풀어주는지 평가하세요. 같은 슬라이드나 인접 context에서 "
            "정확한 의미가 충분히 설명되면 높게 주고, 모호한 표현만 남아 있으면 낮게 주세요."
        ),
    },
    "scope_overclaim": {
        "is_valid_issue": (
            "조건, 예외, 적용 범위, 대상 집합을 닫아버려 일반적으로 맞지 않는 설명이 실제로 남아 있다면 점수를 높게 주세요."
            "범위 표현이 강조나 수사에 그치거나 일반적인 사례로 읽히면 낮게 주세요."
            "강조 표현, 범위 표현, 단정 표현이 문맥상 적절하게 사용되었는지 판단하세요."
            "이러한 표현이 포함되었다는 이유만으로 점수를 높게 주지 마세요."
            "점수를 높게 주려면, 해당 표현이 실제로 잘못된 범위 제한, 예외 배제, 조건 누락, 결과 과장으로 이어져야 합니다."
            "하지만 일반적으로 맞는 표현이며, 설명이면 낮은 점수를 주세요."
            "단정, 과장으로 인해 일어날 수 있는 반례가 일반적이지 않은 상황에 대한 반례라면, 굉장히 낮은 점수를 주세요."
        ),
        "category_severity": (
            "그 과잉 단정이 학생의 적용 범위 판단, 예외, 가능/불가능 판단, 전체/일부 구분을 "
            "크게 틀리게 만들수록 높게 주세요."
            "하지만 그 과잉 단정 표현을 포함하여도, 통상적으로 맞는 지식이고, 일반적으로 맞는 설명이면 점수를 낮게 주세요."
        ),
        "context_resolution": (
            "앞뒤 context와 슬라이드가 조건, 예외, 적용 대상, 범위를 충분히 복원하는지 평가하세요. "
            "문맥상 범위 단정이 명확히 완화되면 높게 주고, 닫힌 범위가 그대로 남으면 낮게 주세요."
            "주어진 문맥을 포함하였을 때, 교육상 맥락에서 허용이 가능한 범위라면 점수를 높게 주세요."
            "앞뒤 context가 해당 resolved_claim에 대해서 계속 설명하고 있으며, 해당 resolved_claim이 앞뒤 문맥과 슬라이드에서 설명하는 것으로 보완이 되거나, 맞는 설명으로 귀결된다면 낮게 주세요."
        ),
    },
}


def _status_from_severity(score: float) -> str:
    confirmed = _safe_float(os.getenv("CLASSIFIED_ISSUE_VERIFIER_CONFIRMED_THRESHOLD"), 0.80)
    rejected = _safe_float(os.getenv("CLASSIFIED_ISSUE_VERIFIER_REJECTED_THRESHOLD"), 0.40)
    if score >= confirmed:
        return "confirmed"
    if score <= rejected:
        return "rejected"
    return "professor_check"


def _confirmed_threshold() -> float:
    return _safe_float(os.getenv("CLASSIFIED_ISSUE_VERIFIER_CONFIRMED_THRESHOLD"), 0.80)


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _strip_json_fence(text: str) -> str:
    stripped = (text or "").strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number != number or number in {float("inf"), float("-inf")}:
        return default
    return number


def _clamp01(value: Any, default: float = 0.0) -> float:
    return round(max(0.0, min(1.0, _safe_float(value, default))), 6)


def _default_models() -> list[str]:
    _load_env()
    configured = _split_csv(os.getenv("CLASSIFIED_ISSUE_VERIFIER_MODELS"))
    return configured or list(DEFAULT_MODELS)


def _chunk(items: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def _bundle_slide_number(item: dict[str, Any]) -> int | None:
    issue = item.get("issue") if isinstance(item.get("issue"), dict) else {}
    return _slide_number(issue)


def _chunk_by_slide_and_category(items: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    grouped: dict[tuple[int | str, str], list[dict[str, Any]]] = {}
    order: list[tuple[int | str, str]] = []
    for item in items:
        slide_number = _bundle_slide_number(item)
        slide_key: int | str = slide_number if slide_number is not None else "__unknown_slide__"
        category = str(item.get("category") or "").strip() or "__unknown_category__"
        key = (slide_key, category)
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(item)

    batches: list[list[dict[str, Any]]] = []
    for key in order:
        batches.extend(_chunk(grouped[key], size))
    return batches


def _batch_category_label(items: list[dict[str, Any]]) -> str:
    categories = []
    for item in items:
        category = str(item.get("category") or "").strip()
        if category and category not in categories:
            categories.append(category)
    if not categories:
        return "mixed"
    if len(categories) == 1:
        return categories[0]
    raise ValueError(f"final verifier batch contains mixed categories: {categories}")


def _load_json(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    target = Path(path)
    if not target.exists():
        return {}
    payload = json.loads(target.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _slide_number(issue: dict[str, Any]) -> int | None:
    location = issue.get("location") if isinstance(issue.get("location"), dict) else {}
    value = location.get("slide_number", issue.get("slide_number"))
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _issue_context_ids(issue: dict[str, Any]) -> list[str]:
    context = issue.get("context") if isinstance(issue.get("context"), dict) else {}
    values = context.get("context_ids") or issue.get("context_ids")
    if isinstance(values, list):
        ids = [str(value).strip() for value in values if str(value).strip()]
        if ids:
            return ids
    single = str(context.get("context_id") or issue.get("context_id") or "").strip()
    return [single] if single else []


def _flatten_issues(payload: dict[str, Any], *, limit: int | None = None) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    issues_by_type = payload.get("issues_by_type") or {}
    if not isinstance(issues_by_type, dict):
        return refs
    for category in ISSUE_TYPES:
        rows = issues_by_type.get(category) or []
        if not isinstance(rows, list):
            continue
        for index, issue in enumerate(rows):
            if not isinstance(issue, dict):
                continue
            refs.append({
                "id": str(issue.get("issue_id") or f"{category}:{index + 1}"),
                "category": category,
                "category_label": CATEGORY_LABELS.get(category, category),
                "index": index,
                "issue": issue,
            })
    if limit is not None:
        return refs[: max(0, limit)]
    return refs


def _build_slide_lookup(*payloads: dict[str, Any]) -> dict[int, dict[str, Any]]:
    lookup: dict[int, dict[str, Any]] = {}
    for payload in payloads:
        candidates = []
        if isinstance(payload.get("slides"), list):
            candidates.extend(payload.get("slides") or [])
        if isinstance(payload.get("scenes"), list):
            candidates.extend(payload.get("scenes") or [])
        for slide in candidates:
            if not isinstance(slide, dict):
                continue
            try:
                number = int(slide.get("slide_number") or slide.get("slide_canonical_number") or 0)
            except (TypeError, ValueError):
                continue
            if number <= 0:
                continue
            base = lookup.setdefault(number, {})
            for key, value in slide.items():
                if value not in (None, "", [], {}):
                    base[key] = value
    return lookup


def _build_context_lookup(merged_payload: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[int, list[dict[str, Any]]]]:
    by_id: dict[str, dict[str, Any]] = {}
    by_slide: dict[int, list[dict[str, Any]]] = {}
    containers: list[dict[str, Any]] = []
    for key in ("slides", "scenes"):
        rows = merged_payload.get(key) or []
        if isinstance(rows, list):
            containers.extend(row for row in rows if isinstance(row, dict))

    next_index_by_slide: dict[int, int] = {}
    for container in containers:
        if not isinstance(container, dict):
            continue
        try:
            slide_number = int(container.get("slide_number") or container.get("slide_canonical_number") or 0)
        except (TypeError, ValueError):
            continue
        contexts = container.get("contexts") or []
        if not isinstance(contexts, list):
            continue
        for context in contexts:
            if not isinstance(context, dict):
                continue
            global_index = next_index_by_slide.get(slide_number, 0)
            next_index_by_slide[slide_number] = global_index + 1
            row = dict(context)
            row.setdefault("slide_number", slide_number)
            row["context_index"] = global_index
            context_id = str(row.get("context_id") or "").strip()
            if not context_id:
                context_id = f"S{slide_number:03d}-C{global_index + 1:03d}"
                row["context_id"] = context_id
            if context_id:
                by_id[context_id] = row
            by_slide.setdefault(slide_number, []).append(row)
    for rows in by_slide.values():
        rows.sort(key=lambda item: (
            int(item.get("context_index", 0) or 0),
            float(item.get("start_time", 0) or 0),
        ))
    return by_id, by_slide


def _compact_context(context: dict[str, Any]) -> dict[str, Any]:
    return {
        "context_id": context.get("context_id", ""),
        "context_index": context.get("context_index"),
        "text": context.get("text", ""),
    }


def _joined_context_text(contexts: list[dict[str, Any]]) -> str:
    lines = []
    for context in contexts:
        context_id = str(context.get("context_id") or "").strip()
        text = str(context.get("text") or "").strip()
        if not text:
            continue
        prefix = f"{context_id}: " if context_id else ""
        lines.append(f"{prefix}{text}")
    return "\n".join(lines).strip()


def _adjacent_slide_number(contexts_by_slide: dict[int, list[dict[str, Any]]], slide_number: int, step: int) -> int | None:
    numbers = sorted(number for number, rows in contexts_by_slide.items() if rows)
    if slide_number not in numbers:
        return None
    index = numbers.index(slide_number) + step
    if index < 0 or index >= len(numbers):
        return None
    return numbers[index]


def _context_window(
    issue: dict[str, Any],
    *,
    context_by_id: dict[str, dict[str, Any]],
    contexts_by_slide: dict[int, list[dict[str, Any]]],
    window: int,
) -> dict[str, Any]:
    context_ids = _issue_context_ids(issue)
    center_contexts = [context_by_id[cid] for cid in context_ids if cid in context_by_id]
    slide_number = _slide_number(issue)
    if slide_number is None and center_contexts:
        try:
            slide_number = int(center_contexts[0].get("slide_number") or 0)
        except (TypeError, ValueError):
            slide_number = None
    if not center_contexts and slide_number:
        rows = contexts_by_slide.get(slide_number) or []
        if rows:
            center_contexts = rows[:1]

    current_slide_contexts = contexts_by_slide.get(slide_number or -1) or []
    local_contexts = current_slide_contexts
    if current_slide_contexts:
        indices: list[int] = []
        for context in center_contexts:
            try:
                indices.append(int(context.get("context_index", 0) or 0))
            except (TypeError, ValueError):
                continue
        if indices:
            start = max(0, min(indices) - max(0, window))
            end = min(len(current_slide_contexts), max(indices) + max(0, window) + 1)
            local_contexts = current_slide_contexts[start:end]

    return {
        "target_context_ids": context_ids,
        "current_slide_transcript": _joined_context_text(local_contexts),
        "window_contexts": [_compact_context(item) for item in local_contexts],
        "previous_slide_tail_contexts": [],
        "next_slide_head_contexts": [],
    }


def _compact_slide(slide: dict[str, Any]) -> dict[str, Any]:
    return {
        key: slide.get(key)
        for key in ("slide_number", "title", "slide_text")
        if slide.get(key) not in (None, "", [], {})
    }


def _build_context_bundle(
    ref: dict[str, Any],
    *,
    domain: str,
    subdomain: str,
    slide_lookup: dict[int, dict[str, Any]],
    context_by_id: dict[str, dict[str, Any]],
    contexts_by_slide: dict[int, list[dict[str, Any]]],
    context_window: int,
) -> dict[str, Any]:
    issue = ref["issue"]
    slide_number = _slide_number(issue)
    slide = slide_lookup.get(slide_number or -1, {})
    return {
        "id": ref["id"],
        "category": ref["category"],
        "category_label": ref["category_label"],
        "domain": domain,
        "subdomain": subdomain,
        "issue": issue,
        "slide": _compact_slide(slide),
        "context_bundle": _context_window(
            issue,
            context_by_id=context_by_id,
            contexts_by_slide=contexts_by_slide,
            window=context_window,
        ),
    }


def _prompt_issue_brief(item: dict[str, Any]) -> dict[str, Any]:
    issue = item.get("issue") or {}
    context_bundle = item.get("context_bundle") if isinstance(item.get("context_bundle"), dict) else {}
    return {
        "id": item.get("id"),
        "issue_id": issue.get("issue_id", ""),
        "claim_id": issue.get("claim_id", ""),
        "category": item.get("category", ""),
        "category_label": item.get("category_label", ""),
        "claim_text": issue.get("claim_text", ""),
        "resolved_claim": issue.get("resolved_claim", ""),
        "target_context_ids": context_bundle.get("target_context_ids", []),
    }


def _prompt_batch_context(items: list[dict[str, Any]]) -> dict[str, Any]:
    first = items[0] if items else {}
    issue = first.get("issue") if isinstance(first.get("issue"), dict) else {}
    location = issue.get("location") if isinstance(issue.get("location"), dict) else {}
    merged_contexts: dict[str, dict[str, Any]] = {}
    for item in items:
        context_bundle = item.get("context_bundle") if isinstance(item.get("context_bundle"), dict) else {}
        for context in context_bundle.get("window_contexts", []) or []:
            if not isinstance(context, dict):
                continue
            context_id = str(context.get("context_id") or "").strip()
            if not context_id:
                continue
            merged_contexts[context_id] = context
    ordered_contexts = sorted(
        merged_contexts.values(),
        key=lambda item: (
            int(item.get("context_index", 0) or 0),
            str(item.get("context_id") or ""),
        ),
    )
    return {
        "domain": first.get("domain", ""),
        "subdomain": first.get("subdomain", ""),
        "location": {"slide_number": location.get("slide_number")},
        "slide": first.get("slide", {}),
        "context_bundle": {
            "current_slide_transcript": _joined_context_text(ordered_contexts),
            "previous_slide_tail_contexts": [],
            "next_slide_head_contexts": [],
        },
    }


def _prompt_payload(items: list[dict[str, Any]]) -> str:
    return (
        "batch_context:\n"
        f"{json.dumps(_prompt_batch_context(items), ensure_ascii=False, indent=2)}\n\n"
        "issues:\n"
        f"{json.dumps([_prompt_issue_brief(item) for item in items], ensure_ascii=False, indent=2)}"
    )


def _build_prompt(category: str, items: list[dict[str, Any]], current_date: str) -> str:
    description = CATEGORY_DESCRIPTIONS.get(category, "")
    score_guide = CATEGORY_SCORE_GUIDES.get(category, {})
    rows = [_prompt_issue_brief(item) for item in items]
    return f"""당신은 강의 verifier의 최종 FACT check 심사자입니다.


def _prompt_payload(items: list[dict[str, Any]]) -> str:
    return (
        "batch_context:\n"
        f"{json.dumps(_prompt_batch_context(items), ensure_ascii=False, indent=2)}\n\n"
        "issues:\n"
        f"{json.dumps([_prompt_issue_brief(item) for item in items], ensure_ascii=False, indent=2)}"
    )


def _response_contract() -> str:
    return """응답은 JSON 객체 하나만 출력하세요. markdown fence는 쓰지 마세요.
모든 입력 id에 대해 judgments 항목을 하나씩 포함하세요.

{
  "judgments": [
    {
      "id": "입력 id",
      "judgment": "valid_issue | partially_resolved | not_issue | insufficient_context",
      "is_valid_issue": 0.0,
      "category_severity": 0.0,
      "context_unresolved": 0.0,
      "context_evidence": "판단에 직접 사용한 전사 문맥 근거 1~2문장",
      "slide_text_observation": "슬라이드 텍스트에서 판단에 영향을 준 요소 1~2문장, 없으면 빈 문자열",
      "score_basis": {
        "is_valid_issue": "이 점수를 준 이유 1문장",
        "category_severity": "이 점수를 준 이유 1문장",
        "context_unresolved": "이 점수를 준 이유 1문장"
      },
      "reason": "판단 근거 1~2문장",
      "minimal_fix": "필요한 경우만 짧게, 없으면 빈 문자열"
    }
  ]
}

공통 출력 규칙:
- is_valid_issue, category_severity, context_unresolved는 0.0 이상 1.0 이하 숫자입니다.
- context_unresolved는 문맥을 본 뒤에도 문제가 남는 정도입니다. 0.0은 문맥에서 해소됨, 1.0은 해소 안 됨입니다.
- context_evidence에는 current_slide_transcript 또는 인접 슬라이드 문맥 중 실제로 점수 판단에 사용한 근거만 쓰세요.
- slide_text_observation에는 slide.t1_structure나 슬라이드 표/대비 항목에서 무엇을 읽었는지 쓰세요. 슬라이드가 실질 근거가 아니면 빈 문자열로 두세요.
- score_basis는 세 점수 각각을 왜 그렇게 줬는지 분리해서 작성하세요. 같은 문장을 복붙하지 말고, 각 점수축의 판단 이유를 따로 쓰세요.
- 바로 뒤 또는 같은 슬라이드의 설명이 같은 대상/관계/조건을 정확히 풀어주면 context_unresolved를 낮게 주세요."""

공통 문맥 해소 판단 순서:
0. resolved_claim과 claim_text/source_context의 관계를 먼저 확인하세요.
   resolved_claim이 source_context의 최종 전달 의미를 올바르게 정리한 문장이고,
   claim_text의 어색함이 말실수, 전사 흔들림, 즉시 재표현, 자기수정 수준이라면
   claim_text의 표면적 어색함만으로 점수를 높이지 마세요.
   반대로 resolved_claim이 source_context의 실제 전달 의미보다 더 강하거나 넓게 정리되었다면,
   resolved_claim의 강해진 부분을 그대로 믿지 말고 source_context 기준으로 낮게 판단하세요.

1. 먼저 target context 안에서 claim이 실제로 어떤 의미로 사용되었는지 판단하세요.
2. 바로 앞뒤 context가 같은 대상, 같은 관계, 같은 조건을 설명하는 경우에만 해소 근거로 사용하세요.
3. 문맥이 단순히 같은 주제를 말하거나 일반 배경을 제공하는 정도라면 해소 근거로 보지 마세요.
4. 문맥이 claim의 강한 표현을 예시, 대비, 강조, 교육적 단순화로 좁혀 주면 context_resolution을 높게 주세요.
5. 반대로 문맥이 같은 강한 표현을 반복하거나 강화하면 context_resolution을 낮게 주세요.
6. 문맥이 양쪽으로 읽히면, claim 자체가 일반적으로 맞는 설명인지 먼저 보세요. 일반적으로 맞는 설명이면 해소 쪽으로, 일반적으로 틀린 설명이면 미해소 쪽으로 판단하세요.
7. slide_text는 target claim을 해석하고 문맥 해소 여부를 판단하기 위한 보조 근거입니다.
   slide_text에 관련 개념이나 강한 표현이 있다는 이유만으로 is_valid_issue를 높이지 마세요.
   slide_text가 target claim의 대상, 관계, 조건, 범위를 더 정확하게 설명하면 context_resolution을 높게 주어 issue를 낮추세요.
   다만 slide_text 자체가 target claim과 같은 잘못된 명제를 직접 반복하거나 강화할 때만 issue를 높이는 근거로 사용할 수 있습니다.
8. 잘못된 용어/분류명을 직접 발화한 경우, 뒤에서 상위 범주나 포함 관계를 설명하더라도 그 설명이 해당 용어/분류명 자체를 바로잡는지 확인하세요.
   해당 용어가 직접 정정되지 않았고, 학생이 그 대상을 잘못된 범주명으로 외울 가능성이 남으면 context_resolution을 낮게 주세요.
   다만 뒤 문맥이나 slide_text가 같은 대상을 더 정확한 용어로 명시하고, 잘못된 용어가 단순 말실수나 재표현 과정으로 해소되면 context_resolution을 높게 줄 수 있습니다.
   단, 영문/외래어 용어의 한글 발음 표기, 음차, 전사 흔들림만 있고 문맥상 지칭하는 원어와 개념이 명확하면 잘못된 용어/분류명 오류로 보지 마세요.
9. 수치 claim에서 약, 한, 대략, 정도, 조금 같은 근사 표현이 있고, 해당 수치가 핵심 학습 대상이 아니라 보조 설명, 감각적 환산, 예시로 쓰인 경우에는 정확한 수치와 차이가 있어도 일반적으로 통용되는 근사인지 먼저 판단하세요.
   일반적으로 통용되는 근사이면 사실 오류로 높게 채점하지 말고, 문맥상 정확한 수치 판단이 핵심일 때만 높게 채점하세요.

입력으로 제공되는 정보:
- claim의 도메인/서브도메인
- resolved_claim과 원문 claim_text (전사본에서, claim단위로 구성하여 제공, resolved_claim은 지시어를 보강한 claim, claim_text는 원문기반 claim)
- 해당 context와 앞뒤 context (전사본을 문맥 단위로 나누어 제공)
- 해당 슬라이드의 slide_text(merged_clean에서 제공되는 기본 슬라이드 텍스트)
- 이전 분류 단계의 weighted_scores와 low_margin 정보

출력 점수:
- is_valid_issue: 이 분류 기준으로 실제 issue일 가능성. 0.0~1.0 issue일수록 1에 수렴.
- category_severity: 이 분류 안에서 오류가 얼마나 심각한지. 0.0~1.0 심각할수록 1에 수렴.
- context_resolution: 제공된 문맥이 issue를 얼마나 해소하는지. 0.0은 전혀 해소 안 됨, 1.0은 거의 완전히 해소됨.

판정 라벨:
- valid_issue: 이 분류 기준에서 유효한 issue
- partially_resolved: issue 가능성은 있으나 문맥으로 일부 해소됨
- not_issue: 이 분류 기준에서는 issue가 아님
- insufficient_context: 제공 자료만으로 판단하기 어려움

중요:
- 모든 입력 id에 대해 judgments 항목을 하나씩 포함하세요.
- 응답은 JSON 객체 하나만 출력하세요.
- 점수는 모두 0.0 이상 1.0 이하 숫자여야 합니다.
- reason은 한두 문장으로 쓰되, 반드시 다음 순서로 작성하세요: 1) claim이 일반 도메인 지식 기준으로 맞는지/틀린지, 2) 틀렸다면 현재 분류 기준 때문에 틀린 것인지, 3) 제공 문맥이 이를 해소했는지.
- minimal_fix는 가능하면 claim을 어떻게 완화/수정하면 되는지 짧게 쓰고, 없으면 빈 문자열로 두세요.
- 문맥이 issue를 해소하는 정도는 주로 context_resolution에 반영하세요.
- 문맥 해소를 이유로 is_valid_issue와 category_severity를 동시에 과도하게 낮추지 마세요.
- 다만 문맥을 포함했을 때 애초에 issue가 성립하지 않는다면 is_valid_issue도 낮출 수 있습니다.

```json
{{
  "judgments": [
    {
      "id": "입력 id",
      "judgment": "valid_issue | partially_resolved | not_issue | insufficient_context",
      "is_valid_issue": 0.0,
      "reason": "실제 전달 명제가 맞는지/틀린지와 C형 판단 이유 1문장",
      "evidence": "판단에 직접 사용한 context_id 또는 짧은 원문 근거"
    }
  ]
}

C형 출력 규칙:
- is_valid_issue는 0.0 이상 1.0 이하 숫자입니다.
- reason은 한 문장으로만 쓰세요.
- evidence는 실제로 본 전사 문맥 근거만 짧게 쓰세요.
- category_severity, context_unresolved, score_basis, slide_text_observation, minimal_fix는 출력하지 마세요."""


def _build_factual_error_prompt(items: list[dict[str, Any]], current_date: str) -> str:
    return f"""당신은 A형 factual_error 전용 최종 검증자입니다.
오늘 날짜: {current_date}

이 batch의 모든 issue는 이미 factual_error로 분류되어 들어왔습니다.
당신의 역할은 재분류가 아니라, 제공 문맥 안에서 factual_error가 실제로 남는지 판단하는 것입니다.

판정 대상:
- 정의, 분류, 포함 관계, 주체, 과정, 원인-결과, 작동 방식, 귀속 관계가 직접 잘못 연결된 오류
- 원문 발화 또는 슬라이드가 학생에게 틀린 객관 명제를 남기는 경우

A형에서 높게 채점하세요:
1. 원문 발화의 핵심 명제가 일반 도메인 지식과 직접 충돌함 -> is_valid_issue 0.80 이상
2. 슬라이드 또는 같은 문맥이 그 충돌을 바로잡지 않음 -> context_unresolved 0.70 이상
3. 학생이 그대로 외우면 틀린 정의/관계/작동 방식을 외우게 됨 -> category_severity 0.70 이상
4. 더 엄밀한 taxonomy가 아니라 강의 수준에서도 구분해야 하는 오류임 -> category_severity 0.80 이상 가능

A형에서 낮게 채점하세요:
- 표현은 거칠지만 같은 문맥이 올바른 의미로 좁힘 -> context_unresolved 0.30 이하
- 관리/중재/대행/권한 통제 설명을 실제 수행 주체 오류로 과해석해야만 문제가 됨 -> is_valid_issue 0.40 이하
- 자원, 권한, 접근, 요청, 허용, 할당을 설명하는 문맥에서
  "모든", "배타적", "독점" 같은 표현이 실제 자원 소비 주체가 하나라는 뜻이 아니라
  접근 권한을 중앙에서 관리/통제한다는 뜻으로 자연스럽게 읽히면 factual_error로 보지 말고 is_valid_issue 0.30 이하로 주세요.
- 단순 용어 엄밀성, 표현 취향, 고급 구현 예외에 가까움 -> category_severity 0.40 이하
- 다른 category 문제처럼 보이지만 factual_error로 직접 확정하기 어려움 -> is_valid_issue 0.50 이하

A형 점수 산정:
- is_valid_issue는 "factual_error로 성립하는가"만 봅니다. 객관 충돌이 직접 있으면 0.80~1.00, 애매하면 0.40~0.69, 문맥상 맞으면 0.00~0.30.
- category_severity는 "그 사실 오류가 강의 수준에서 얼마나 중요한가"입니다. 핵심 정의/관계/작동 방식이면 0.70~1.00, 보조 설명이면 0.40~0.69, 표현 보완 수준이면 0.00~0.30.
- context_unresolved는 "문맥이 그 사실 오류를 해소하지 못한 정도"입니다. 정정 없음/반복 강화면 0.70~1.00, 일부 보완이면 0.30~0.69, 바로 해소되면 0.00~0.25.

판정 라벨:
- valid_issue: 명확한 사실/정의/관계 오류가 문맥 후에도 남음
- partially_resolved: 오류 가능성은 있지만 문맥이 일부 완화함
- not_issue: 문맥상 올바른 설명이거나 factual_error가 아님
- insufficient_context: 실제 발화나 슬라이드 맥락이 부족함

{_response_contract()}

{_prompt_payload(items)}
"""


def _build_temporal_error_prompt(items: list[dict[str, Any]], current_date: str) -> str:
    return f"""당신은 B형 temporal_error 전용 최종 검증자입니다.
오늘 날짜: {current_date}

이 batch의 모든 issue는 이미 temporal_error로 분류되어 들어왔습니다.
당신의 역할은 제공 문맥 안에서 시점 의존 정보가 현재 학습자에게 확인 필요 정보로 남는지 판단하는 것입니다.

판정 대상:
- 현재/요즘/최근/최신/지원 여부/버전/정책/통계/시장 상황/기술 관행처럼 시간에 따라 참거짓이 달라질 수 있는 설명
- 강의 제공 시점 또는 오늘 날짜 기준으로 확인이 필요한 정보

B형에서 높게 채점하세요:
1. 발화가 현재 사실처럼 시점 의존 정보를 말함 -> is_valid_issue 0.70 이상
2. 문맥이 과거 사례, 역사 설명, 당시 기준 설명으로 제한하지 않음 -> context_unresolved 0.70 이상
3. 강의 도메인과 수준에서 현재 학습자가 잘못된 최신 정보를 가져갈 수 있음 -> category_severity 0.60 이상
4. 슬라이드 또는 주변 설명이 기준 시점을 보완하지 않음 -> context_unresolved 0.75 이상

B형에서 낮게 채점하세요:
- 명시적으로 과거/역사/당시 기준 설명임 -> is_valid_issue 0.30 이하, context_unresolved 0.25 이하
- 최신성 확인이 강의 핵심 이해와 거의 무관함 -> category_severity 0.30 이하
- 시점 문제가 아니라 정의/범위/혼동 문제가 핵심임 -> is_valid_issue 0.40 이하
- 단순 예시나 대표 사례일 뿐 현재 일반 사실로 남지 않음 -> is_valid_issue 0.45 이하

B형 점수 산정:
- is_valid_issue는 "현재성/시점 의존 issue로 성립하는가"만 봅니다. 현재 사실처럼 말한 시점 의존 정보면 0.70~1.00, 시점성이 약하면 0.40~0.69, 시점 issue가 아니면 0.00~0.30.
- category_severity는 "현재 학습자에게 오래된 정보가 될 위험"입니다. 학습 결정이나 개념 이해에 영향이 크면 0.60~0.90, 보조 사례면 0.30~0.59, 사소하면 0.00~0.29.
- context_unresolved는 "문맥이 기준 시점/역사 맥락을 보완하지 못한 정도"입니다. 기준 시점 보완 없음이면 0.70~1.00, 일부 보완이면 0.30~0.69, 과거/당시 기준으로 명확히 제한되면 0.00~0.25.

판정 라벨:
- valid_issue: 현재성 확인이 강하게 필요하고 문맥도 보완하지 않음
- partially_resolved: 확인 필요성은 있으나 예시/당시 맥락 가능성이 있음
- not_issue: 시점 의존 issue가 아니거나 문맥에서 해소됨
- insufficient_context: 기준 시점 또는 대상 기술을 특정하기 어려움

{_response_contract()}

{_prompt_payload(items)}
"""


def _build_scope_overclaim_prompt(items: list[dict[str, Any]], current_date: str) -> str:
    return f"""당신은 C형 scope_overclaim 전용 최종 검증자입니다.
오늘 날짜: {current_date}

이 batch의 모든 issue는 이미 scope_overclaim로 분류되어 들어왔습니다.
당신의 역할은 제공 문맥을 읽고, 실제 전달 내용이 C형 scope_overclaim으로 남는지 판단하는 것입니다.

판정 대상:
- 전체/일부, 항상/가끔, 오직/복수, 가능/불가능, 일시/영구, 조건부/필연이 뒤바뀐 범위 오류
- 특정 조건에서 생길 수 있는 결과를 전체적, 영구적, 회복 불가능한 결과처럼 말하는 결과 과장 오류
- 강의 도메인과 수준에서 학생이 실제 발화 기준으로 잘못 외울 수 있는 배제 관계, 일반화, 조건부 명제가 남는 경우

C형 판단 방식

is_valid_issue 하나로만 판단하세요.
이 점수는 “target_context_ids가 가리키는 context 전체와 제공된 slide.t1_structure를 읽었을 때,
실제 전달 내용이 일반 도메인 지식 기준으로 틀리고, 그 틀림의 원인이 범위/조건/예외/결과를 닫아 말한 데 있는 정도”입니다.

resolved_claim을 1차 검증 명제로 사용하세요.
claim_text는 원문 표현과 검토 위치를 확인하기 위한 보조 표식입니다.
resolved_claim 문장만 단독으로 보지 말고, 반드시 batch_context의 해당 context 전체를 읽어
resolved_claim이 실제 context에서 전달되는 명제를 충실히 보존하는지 확인한 뒤 판단하세요.
resolved_claim이 context의 실제 발화 흐름보다 강하게 일반화했으면 그 강해진 부분은 낮게 보세요.

판단 순서:
1. 제공 문맥에서 실제 전달되는 명제를 먼저 재구성하세요.
2. 그 명제가 일반 도메인 지식과 강의 수준 기준으로 맞는 설명인지 판단하세요.
3. 맞는 설명이면 단정어가 있거나 더 엄밀한 예외가 떠올라도 is_valid_issue를 0.39 이하로 주세요.
4. 틀린 설명이면, 그 틀림이 범위/조건/예외/결과를 닫아 말해서 생긴 것인지 판단하세요.
5. 틀린 이유가 C형 범위 과잉이면 그때만 0.40 이상을 줄 수 있습니다.

context가 바로 의미를 좁히거나, 예시/대비/강의 범위 제한/교육적 단순화로 자연스럽게 읽히게 만들면
그 효과까지 포함해서 is_valid_issue를 낮게 주세요.

역할, 책임, 권한, 관리, 중재, 표준 절차를 설명하는 문맥에서는
강한 표현을 문자 그대로의 독점 수행, 모든 내부 동작의 대행, 또는 모든 예외의 배제로 바로 해석하지 마세요.
제공 문맥상 일반적인 역할 설명이나 표준 접근 경로 설명으로 자연스럽게 성립하면 is_valid_issue는 0.39 이하로 주세요.
문제를 만들기 위해 "관리한다/권한이 있다/요청한다"를 "해당 주체만 모든 실제 동작을 직접 수행한다"로 바꿔 읽어야 한다면 is_valid_issue는 0.29 이하로 주세요.

단정어는 주의 신호일 뿐입니다.
"항상", "모든", "오직", "~만", "반드시", "독점", "배타적" 같은 표현이 있어도
실제 배제 명제나 잘못된 일반 규칙이 명확하지 않으면 높은 점수를 주지 마세요.
반대로 "다시는", "영구적으로", "완전히", "항상 ... 된다"처럼 결과를 영구적/전체적/필연적으로 닫아 말하고,
제공 문맥이 그 결과를 일시적, 부분적, 조건부 결과로 좁히지 않으면 C형 근거로 볼 수 있습니다.

반례를 먼저 찾지 마세요.
먼저 제공 문맥에서 resolved_claim이 실제로 무엇을 배제하거나 일반화했는지 판단하세요.
배제 대상이나 일반화 명제가 명확할 때만, 그 명제와 직접 충돌하는 강의 수준의 반례/예외를 검토하세요.
배제 명제가 명확하지 않으면 반례를 만들지 말고 낮게 채점하세요.
반례를 찾기 위해 resolved_claim이나 원문 context를 더 강하게 해석해야 하면 not_issue에 가깝게 판단하세요.

상한 규칙:

- 실제 배제 명제나 일반 규칙을 한 문장으로 쓸 수 없으면 is_valid_issue는 0.39 이하로 주세요.
- 단정어만 있고 무엇이 배제되었는지 불명확하면 is_valid_issue는 0.19 이하로 주세요.
- 실제 전달 명제가 일반 도메인 지식 기준으로 맞는 설명이면 is_valid_issue는 0.39 이하로 주세요.
- "절대 규칙처럼 들릴 수 있음", "과하게 해석될 수 있음", "예외 없는 규칙처럼 보일 수 있음"만으로는 is_valid_issue를 0.40 이상 주지 마세요.
- 강의 범위 제한, 예시 제한, 대표 설명, 대비, 강조로 자연스럽게 읽히면 is_valid_issue는 0.19 이하로 주세요.
- 반례나 예외가 강의 도메인 밖, 강의 수준 밖 고급 예외, 특수 환경에서만 성립하는 경우 is_valid_issue는 0.29 이하로 주세요.
- resolved_claim이 해당 context의 실제 발화 흐름보다 강하게 일반화한 경우, context에 그 강한 일반화가 남지 않으면 is_valid_issue는 0.39 이하로 주세요.
- 입문 수준에서 허용 가능한 단순화이면 is_valid_issue는 0.39 이하로 주세요.

- 0.80~1.00:
  실제 전달 명제가 일반 도메인 지식 기준으로 명확히 틀림.
  그 틀림의 원인이 범위, 조건, 예외, 가능성, 결과를 닫아 말한 데 있음.
  무엇이 배제되거나 일반화되었는지 명확함.
  그 일반화가 강의 도메인과 수준 안에서 중요한 조건, 예외, 가능성을 실제로 배제함.
  조건부 결과를 영구적/전체적/필연적 결과처럼 말해 잘못된 결과 규칙이 발화에 남음.
  예시, 대비, 강조, 강의 범위 제한, 입문 수준 단순화로 보기 어려움.

- 0.60~0.79:
  실제 전달 명제가 일반 도메인 지식 기준으로 틀릴 가능성이 비교적 분명함.
  틀린 이유가 범위/조건/예외/결과를 닫아 말한 데 있음.
  강의 수준 안의 반례, 예외, 조건 차이도 구체적으로 있음.
  또는 결과 과장이 있으나 발화의 강조/예시 성격도 일부 남아 있음.
  다만 발화의 일반화 강도, 적용 범위, 반례의 중요성이 0.80 이상만큼 명확하지는 않음.
  일반적인 역할, 책임, 권한, 관리, 중재, 표준 절차 설명으로 자연스럽게 성립하는 경우는 이 구간에 두지 마세요.

- 0.40~0.59:
  실제 전달 명제가 틀릴 가능성은 있으나 약함.
  C형 문제로 볼 수 있는 근거가 일부 있지만, 문맥의 보완이나 교육적 단순화도 함께 강함.
  이 구간은 "들릴 수 있음"만으로 주는 구간이 아니라, 실제로 틀린 명제가 약하게라도 남는 경우입니다.
  단, 역할/권한/중재/표준 절차 설명을 문자 그대로 과해석해야만 문제가 되면 0.39 이하로 두세요.

- 0.20~0.39:
  단정어 또는 강한 표현은 있으나, 실제 배제 명제나 잘못된 일반 규칙이 명확하지 않음.
  문제를 만들려면 원문보다 강하게 해석해야 함.
  반례가 있더라도 고급 예외, 특수 환경, 다른 도메인, 구현 세부사항에 가까움.
  실제 전달 명제는 대체로 맞는 설명임.
  강의 맥락상 표현이 다소 강하지만 C형 issue로 유지하기 어려움.

- 0.00~0.19:
  C형 scope_overclaim으로 보기 어려움.
  발화가 일반 도메인 지식 기준으로 자연스럽게 성립하는 일반 설명임.
  일반적인 강의 설명, 예시, 대비, 강조, 교육적 단순화로 자연스럽게 이해됨.
  실제 배제 대상이나 일반화 명제가 남지 않음.

C형 추가 출력 지침:
- reason에는 실제 전달 명제가 맞는 설명인지 틀린 설명인지와, C형 issue로 남는지 여부를 한 문장으로 쓰세요.
- evidence에는 판단에 직접 사용한 context_id 또는 핵심 원문 근거만 짧게 쓰세요.

{_scope_response_contract()}

{_prompt_payload(items)}
"""


def _build_confusing_explanation_prompt(items: list[dict[str, Any]], current_date: str) -> str:
    return f"""당신은 D형 confusing_explanation 전용 최종 검증자입니다.
오늘 날짜: {current_date}

이 batch의 모든 issue는 이미 confusing_explanation로 분류되어 들어왔습니다.
당신의 역할은 설명 흐름이 실제 문맥 안에서 구체적인 오개념을 남기는지 판단하는 것입니다.
이 단계는 "오해할 수도 있다"를 상상하는 단계가 아닙니다.
제공 문맥을 읽고, 실제 전달된 설명이 어떤 mental model을 남기는지 먼저 재구성하세요.

판정 대상:
- 학생이 주체, 과정, 원인, 조건, 개념 관계를 잘못 연결해 외울 수 있는 설명
- 단일 문장만 보면 거칠 수 있으나, 문맥 후에도 구체적으로 잘못된 mental model이 남는 설명

D형 판단 방식

먼저 target_context_ids가 가리키는 context 전체와 slide.t1_structure를 읽고,
실제로 전달되는 설명 흐름을 재구성하세요.

resolved_claim을 1차 검토 명제로 사용하되, resolved_claim 문장만 단독으로 판단하지 마세요.
claim_text는 원문 표현과 검토 위치를 확인하기 위한 보조 표식입니다.
resolved_claim이 실제 context보다 더 강한 오개념으로 바뀌었으면, 그 강해진 부분은 낮게 보세요.

판단 순서:
1. 제공 문맥에서 실제로 학생에게 남는 설명 흐름을 재구성합니다.
2. 그 설명 흐름에서 학생이 잘못 외울 구체 오개념을 한 문장으로 쓸 수 있는지 봅니다.
3. 그 오개념이 일반 도메인 지식과 강의 수준 기준으로 실제로 잘못된 mental model인지 봅니다.
4. 바로 뒤 문장, 같은 context, 같은 슬라이드 설명이 같은 대상/관계/조건을 올바르게 풀어주면 낮게 봅니다.
5. 구체 오개념 문장을 쓰려면 원문보다 강하게 해석해야 하거나, 단순히 더 자세히 설명하면 좋겠다는 수준이면 낮게 봅니다.
6. 같은 context 안에 더 직접적인 다른 오류가 있더라도, 그 오류가 현재 issue의 claim_text/resolved_claim이 가리키는 문제 자체가 아니면 현재 issue를 높게 채점하지 마세요.

우선 적용 규칙:
- 같은 context 또는 바로 인접 context에서 target 개념의 올바른 정의/전체 조건이 먼저 제시되어 있고,
  target 발화가 그 뒤에 이어지는 짧은 재진술, 생성 과정, 준비 단계, 말줄임 표현이라면
  target 발화만 떼어 독립 정의로 재해석하지 마세요.
- 이 경우 target 발화가 명시적으로 "그 단계만으로 충분하다", "그 조건 없이도 해당 개념이다",
  "앞의 정의가 아니라 이것이 정의다"라고 말하지 않는 한 not_issue로 판단하고
  is_valid_issue는 0.29 이하, context_unresolved는 0.25 이하로 주세요.
- "학생이 그렇게 오해할 수도 있다"는 가능성만으로 위 상한을 넘기지 마세요.

명시적 용어/분류명 오류 처리:
- target context 안에서 어떤 대상을 특정 용어, 범주, 분류명으로 명시적으로 부르고,
  그 용어/분류명이 일반 도메인 지식 기준으로 다른 범주를 가리킨다면,
  학생이 "그 대상은 그 범주에 속한다"는 오개념을 외울 수 있는지 판단하세요.
- 같은 문맥에 올바른 관련 설명이나 상위 범주 설명이 있어도,
  잘못된 용어/분류명을 직접 정정하지 않는다면 자동 해소로 보지 마세요.
- 다만 문맥이 "일반 사용자는 그렇게 생각하지만 실제로는 아니다"처럼
  잘못된 명칭을 소개한 뒤 바로 구분하거나 정정하면 낮게 보세요.
- 이 경우 A형과 겹칠 수 있지만, 현재 batch가 D형으로 들어온 상태에서는
  "잘못된 라벨 때문에 생기는 구체적 mental model"이 남는지를 기준으로 review 후보를 남길 수 있습니다.

C형에서처럼, 먼저 실제 전달 내용이 틀린지 보세요.
설명이 일반 도메인 지식 기준으로 맞고, 단지 생략되었거나 입문 수준으로 단순화된 것이라면 D형 issue로 높게 채점하지 마세요.

Anchor 귀속 규칙:
- 현재 issue의 claim_text/resolved_claim이 검토 anchor입니다.
- 제공 문맥은 anchor를 해석하고 해소 여부를 판단하기 위한 근거입니다.
- 같은 context 안에서 다른 문장이 더 직접적인 사실 오류나 범위 오류를 만들더라도, 그 오류를 현재 anchor의 D형 오개념으로 옮겨 붙이지 마세요.
- 현재 anchor 자체는 맞거나 교육적 단순화인데, 주변의 다른 claim 때문에 context가 혼란스럽다면 현재 anchor는 낮게 채점하세요.
- 주변의 다른 오류가 핵심이면 reason에 "현재 anchor 자체보다는 같은 context의 다른 문장이 직접 오류임"이라고 쓰고 is_valid_issue는 0.39 이하로 주세요.
- 문맥은 현재 anchor의 의미를 확인하거나 해소할 수는 있지만, 현재 anchor에 없던 문제를 새로 만들어 붙일 수는 없습니다.
- 후속 문장이 현재 anchor와 모순되더라도, 현재 anchor 자체가 일반 도메인 지식 기준으로 맞는 설명이면 현재 anchor는 not_issue 또는 낮은 partially_resolved입니다.
- 모순의 원인이 후속 문장 자체의 오류라면, 그 후속 문장을 별도 issue로 보아야 하며 현재 anchor의 점수를 올리는 근거로 쓰지 마세요.
- "anchor는 맞지만 뒤 문장이 틀려서 혼란스럽다"는 경우 is_valid_issue는 0.39 이하로 주세요.
- 후속 문장이 anchor를 "수정한다"고 가정하지 마세요. 후속 문장이 틀린 설명이면, 그것은 후속 문장 자체의 문제이지 현재 anchor가 D형 issue라는 증거가 아닙니다.
- 현재 anchor가 맞는 중재/접근 경로 설명인데 후속 문장이 틀리거나 거칠어서 모순처럼 보이는 경우, 현재 anchor의 is_valid_issue는 0.29 이하로 주세요.

역할, 책임, 권한, 관리, 중재, 표준 절차를 설명하는 문맥에서는
강한 표현을 문맥보다 넓은 절대 수행, 모든 내부 동작의 대행, 또는 모든 예외의 배제로 바로 해석하지 마세요.
제공 문맥상 일반적인 역할 설명이나 표준 경로 설명으로 자연스럽게 성립하면 낮게 채점하세요.
문제를 만들기 위해 원문 표현을 더 강한 독점 수행/직접 동일시/절대 배제 명제로 바꿔 읽어야 한다면 낮게 채점하세요.

중간 과정이 압축된 표현은 앞뒤 문맥에서 실제 전달되는 관계를 기준으로 판단하세요.
단순한 생략이나 요약만으로 D형을 유지하려면, 학생이 실제 주체/과정/조건을 잘못 예측하게 되는 구체 오개념이 문맥에 남아야 합니다.

전체 과정의 한 단계를 짧게 말한 표현을 전체 정의로 확대하지 마세요.
같은 문맥에서 전체 정의나 전후 단계가 함께 제시되어 있으면, 특정 단계 표현 하나를 "그 단계가 곧 전체 개념"이라는 오개념으로 만들지 마세요.
과정 설명이 "A를 하고 B라고 부른 뒤 C를 한다"처럼 거칠어도, 주변 문맥이 전체 흐름을 보여주면 입문 수준 단순화로 보고 is_valid_issue는 0.39 이하로 주세요.
같은 context 또는 바로 인접 context에서 올바른 정의가 먼저 제시되어 있으면,
뒤따르는 생성 과정/준비 단계의 느슨한 표현을 그 정의를 뒤집는 새 오개념으로 보지 마세요.
이 경우 명시적으로 "그 준비 단계만으로 충분하다"고 말하지 않는 한 is_valid_issue는 0.39 이하로 주세요.
같은 context 또는 바로 인접 context에서 전체 정의가 이미 제시되어 있고,
target 발화가 그 정의 다음에 나오는 짧은 재진술, 생성 단계, 준비 단계, 말줄임 표현이면
그 표현을 독립 정의로 재해석하지 마세요.
명시적으로 "그 단계만으로 충분하다", "그 조건 없이도 해당 개념이다"라고 말하지 않으면
is_valid_issue는 0.29 이하, context_unresolved는 0.25 이하로 주세요.

가능성, 관찰, 접근, 통제, 원인 설명이 거칠더라도, 뒤 문맥이 실제 의미를 좁히고 결론 자체가 일반 도메인 지식 기준으로 맞으면 낮게 보세요.
문제를 만들기 위해 원문보다 강한 인과, 절대 불가능, 독점 수행, 실제 소비 주체 변경 명제를 새로 만들어야 한다면 낮게 보세요.
거친 원인 설명을 D형으로 높게 채점하려면, 학생이 그 원인을 일반 규칙으로 적용해 잘못된 예측을 하게 되는 구체 오개념이 문맥 후에도 남아야 합니다.
결론은 맞고 원인 표현만 덜 정교한 경우에는, 그 원인 표현이 실제 행동/판단을 잘못 예측하게 만드는 경우에만 0.40 이상을 주세요.

D형에서 높게 채점하세요:
1. 제공 문맥을 읽어도 학생이 잘못 외울 구체 오개념을 한 문장으로 명확히 쓸 수 있음 -> is_valid_issue 0.70 이상
2. 그 오개념이 단순 표현 어색함이 아니라 주체/과정/원인/조건/개념 관계 이해를 실제로 바꿈 -> category_severity 0.65 이상
3. 같은 문맥의 재표현, 예시, 슬라이드가 그 오개념을 해소하지 않음 -> context_unresolved 0.70 이상
4. 바로 뒤 설명이 오히려 같은 오개념을 반복하거나 강화함 -> context_unresolved 0.80 이상 가능
5. 명시적인 잘못된 용어/분류명이 target context에 남고, 바로 정정되지 않아 학생이 대상의 범주를 잘못 외울 수 있음 -> is_valid_issue 0.40 이상 가능

D형에서 낮게 채점하세요:
- 구체 오개념 문장을 쓰기 어렵고 막연히 헷갈릴 수 있다는 수준임 -> is_valid_issue 0.35 이하
- 표현이 어색하거나 생략되었지만 핵심 의미는 문맥상 올바르게 전달됨 -> is_valid_issue 0.39 이하
- 바로 뒤 문장이나 슬라이드가 대상/관계/조건을 정확히 풀어줌 -> context_unresolved 0.25 이하
- A/B/C형으로 명확히 판단해야 할 문제를 D형으로 우회하는 경우 -> is_valid_issue 0.39 이하
- 다만 현재 batch가 D형으로 들어왔고, 제공 문맥 안에서 구체 오개념이 실제로 남는다면 A/B/C와 겹치더라도 D형 review 후보로 남길 수 있습니다.
- 정의, 분류, 인과, 작동 방식의 참거짓을 직접 판정해야만 문제가 되는 경우는 낮게 보되, 그 참거짓 문제 때문에 학생이 구체적으로 잘못 연결해 외울 mental model이 남으면 0.40 이상을 줄 수 있습니다.
- 더 자세히 설명하면 좋겠다는 수준임 -> category_severity 0.30 이하
- 일반적인 강의 설명, 예시, 대비, 강조, 교육적 단순화로 자연스럽게 이해됨 -> is_valid_issue 0.29 이하
- 고급 구현 세부사항이나 강의 수준 밖 엄밀성을 알아야만 문제가 됨 -> is_valid_issue 0.29 이하

D형 점수 산정:
- is_valid_issue는 "제공 문맥 안에서 구체 오개념 유발 설명으로 성립하는가"만 봅니다.
  오개념 문장을 명확히 쓸 수 있고 그 오개념이 강의 수준에서 실제로 틀리면 0.70~1.00,
  오개념 가능성은 있으나 문맥 보완이나 단순화 성격도 있으면 0.40~0.69,
  막연한 혼동 가능성/표현 보완/강의 수준 밖 엄밀성이면 0.00~0.39입니다.
- category_severity는 "그 오개념이 학습에 주는 영향"입니다.
  핵심 개념/과정/주체/조건 혼동이면 0.65~0.90,
  보조 개념 혼동이면 0.30~0.64,
  표현 보완 수준이면 0.00~0.29입니다.
- context_unresolved는 "문맥 후에도 그 오개념이 남는 정도"입니다.
  오개념이 반복/강화되면 0.75~1.00,
  일부 보완이면 0.30~0.74,
  바로 해소되거나 올바른 의미로 좁혀지면 0.00~0.25입니다.

상한 규칙:
- 구체 오개념 문장을 한 문장으로 쓸 수 없으면 is_valid_issue는 0.35 이하로 주세요.
- 실제 전달 의미가 일반 도메인 지식 기준으로 맞는 설명이면 is_valid_issue는 0.39 이하로 주세요.
- 현재 anchor 자체가 아니라 같은 context의 다른 claim이 직접 문제라면 is_valid_issue는 0.39 이하로 주세요.
- 현재 anchor 자체는 맞고 후속 문장의 오류 때문에만 문맥이 모순된다면 is_valid_issue는 0.39 이하로 주세요.
- 현재 anchor가 문맥상 표준 절차나 일반 역할 설명으로 자연스럽게 읽히면 is_valid_issue는 0.29 이하로 주세요.
- 원문보다 강한 직접 동일시나 절대 규칙으로 바꿔야만 문제가 되면 is_valid_issue는 0.29 이하로 주세요.
- 전체 과정의 한 단계 표현을 전체 정의 오류로 확대해야만 문제가 되면 is_valid_issue는 0.39 이하로 주세요.
- 올바른 정의가 바로 앞뒤에 있고, 준비/생성 단계의 느슨한 표현만 문제라면 is_valid_issue는 0.39 이하로 주세요.
- 올바른 정의가 바로 앞뒤에 있고, 짧은 후속 표현을 독립 정의로 떼어내야만 문제가 되면 is_valid_issue는 0.29 이하로 주세요.
- 관찰/접근/통제 설명을 원문보다 강한 절대 불가능이나 실제 수행 주체 오류로 바꿔 읽어야만 문제가 되면 is_valid_issue는 0.29 이하로 주세요.
- 거친 원인 표현이 뒤 문맥에서 대비 설명으로 좁혀지고 구체 오개념 예측이 남지 않으면 is_valid_issue는 0.39 이하로 주세요.
- 결론은 맞고 원인 표현만 덜 정교한 경우, 구체적 오답 예측이 없으면 is_valid_issue는 0.39 이하로 주세요.
- 현재 issue를 유지하려면 "이 설명이 사실상 맞는가/틀린가"를 먼저 판정해야 한다면 D형이 아닙니다. 이 경우 is_valid_issue는 0.39 이하로 주세요.
- 단, 현재 issue가 명시적 용어/분류명 때문에 생긴 오개념이고 그 라벨이 문맥에서 직접 정정되지 않았다면,
  A형과 겹친다는 이유만으로 0.39 이하 상한을 적용하지 마세요.
- "오해할 수 있음", "혼동될 수 있음", "더 정확히 말하면 좋음"만으로는 is_valid_issue를 0.40 이상 주지 마세요.
- resolved_claim이 실제 context보다 오개념을 강하게 만든 경우, context에 그 강한 오개념이 남지 않으면 is_valid_issue는 0.39 이하로 주세요.
- 입문 수준에서 허용 가능한 생략이나 교육적 단순화이면 is_valid_issue는 0.39 이하로 주세요.

판정 라벨:
- valid_issue: 구체 오개념이 문맥 후에도 뚜렷하게 남음
- partially_resolved: 오개념 가능성은 있으나 문맥이 일부 보완함. 일부 보완 후에도 구체 오개념이 남으면 review 가능한 점수를 줄 수 있음
- not_issue: 막연한 혼동 가능성 또는 문맥에서 해소됨
- insufficient_context: 오개념 여부를 판단할 문맥이 부족함

출력 지침:
- reason에는 실제로 남는 구체 오개념이 무엇인지, 또는 왜 오개념으로 남지 않는지 1~2문장으로 쓰세요.
- score_basis.is_valid_issue에는 "실제 남는 오개념 문장" 또는 "오개념 없음/문맥 해소 이유"를 적으세요.

{_response_contract()}

{_prompt_payload(items)}
"""


def _build_prompt(category: str, items: list[dict[str, Any]], current_date: str) -> str:
    if category == "factual_error":
        return _build_factual_error_prompt(items, current_date)
    if category == "temporal_error":
        return _build_temporal_error_prompt(items, current_date)
    if category == "scope_overclaim":
        return _build_scope_overclaim_prompt(items, current_date)
    if category == "confusing_explanation":
        return _build_confusing_explanation_prompt(items, current_date)
    return _build_confusing_explanation_prompt(items, current_date)


def _parse_response(text: str) -> list[dict[str, Any]]:
    payload = json.loads(_strip_json_fence(text))
    rows = payload.get("judgments", [])
    return rows if isinstance(rows, list) else []


def _final_model_score(
    *,
    category: str,
    judgment: str,
    is_valid_issue: float,
    category_severity: float,
    context_unresolved: float,
) -> float:
    return _clamp01(is_valid_issue * category_severity * context_unresolved)


def _normalize_judgment_row(
    row: dict[str, Any],
    *,
    ref: dict[str, Any],
    model: str,
    resolved: dict[str, str],
) -> dict[str, Any]:
    raw_judgment = str(row.get("judgment") or "").strip()
    judgment = raw_judgment if raw_judgment in JUDGMENTS else "insufficient_context"
    is_valid_issue = _clamp01(row.get("is_valid_issue"))
    category_severity = _clamp01(row.get("category_severity"))
    context_resolution = _clamp01(row.get("context_resolution"))
    context_unresolved = _clamp01(1.0 - context_resolution)
    final_model_score = _final_model_score(
        category=ref["category"],
        judgment=judgment,
        is_valid_issue=is_valid_issue,
        category_severity=category_severity,
        context_unresolved=context_unresolved,
    )
    reason = str(row.get("reason", "") or "").strip()
    return {
        "id": ref["id"],
        "model": model,
        "provider": resolved.get("provider", ""),
        "resolved_model": resolved.get("resolved_model", model),
        "category": ref["category"],
        "judgment": judgment,
        "is_valid_issue": is_valid_issue,
        "category_severity": category_severity,
        "context_resolution": context_resolution,
        "context_unresolved": context_unresolved,
        "final_model_score": final_model_score,
        "reason": reason,
        "minimal_fix": str(row.get("minimal_fix", "") or "").strip(),
        "status": "ok",
        "parse_error": "",
    }
    if ref["category"] == "scope_overclaim":
        return {
            "id": normalized["id"],
            "model": normalized["model"],
            "provider": normalized["provider"],
            "resolved_model": normalized["resolved_model"],
            "category": normalized["category"],
            "judgment": normalized["judgment"],
            "is_valid_issue": normalized["is_valid_issue"],
            "final_model_score": normalized["final_model_score"],
            "context_evidence": normalized["context_evidence"],
            "reason": normalized["reason"],
            "status": normalized["status"],
            "parse_error": normalized["parse_error"],
        }
    return normalized


def _parse_failed_row(ref: dict[str, Any], model: str, resolved: dict[str, str], error: str) -> dict[str, Any]:
    row = {
        "id": ref["id"],
        "model": model,
        "provider": resolved.get("provider", ""),
        "resolved_model": resolved.get("resolved_model", model),
        "category": ref["category"],
        "judgment": "insufficient_context",
        "is_valid_issue": 0.0,
        "category_severity": 0.0,
        "context_resolution": 0.0,
        "context_unresolved": 1.0,
        "final_model_score": 0.0,
        "context_evidence": "",
        "slide_text_observation": "",
        "score_basis": {
            "is_valid_issue": "",
            "category_severity": "",
            "context_unresolved": "",
        },
        "reason": "",
        "minimal_fix": "",
        "status": "parse_failed",
        "parse_error": error,
    }
    return row


def _call_model_for_batch(
    *,
    model: str,
    category: str,
    batch: list[dict[str, Any]],
    current_date: str,
    max_tokens: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prompt = _build_prompt(category, batch, current_date)
    text, usage, resolved = _call_llm(model_spec=model, prompt=prompt, max_tokens=max_tokens)
    try:
        rows = _parse_response(text)
        by_id = {str(row.get("id") or ""): row for row in rows if isinstance(row, dict)}
    except Exception as exc:
        by_id = {}
        parse_error = str(exc)
    else:
        parse_error = ""

    normalized = []
    for ref in batch:
        row = by_id.get(ref["id"])
        if isinstance(row, dict):
            normalized.append(_normalize_judgment_row(row, ref=ref, model=model, resolved=resolved))
        else:
            normalized.append(_parse_failed_row(ref, model, resolved, parse_error or "missing judgment row"))
    return normalized, usage


def _batch_worker(args: tuple) -> dict[str, Any]:
    model, category, batch, batch_index, total_batches, current_date, max_tokens = args
    resolved = _resolve_model_spec(model)
    ids = f"{batch[0]['id']}..{batch[-1]['id']}" if batch else "-"
    slide_number = _bundle_slide_number(batch[0]) if batch else None
    slide_label = f"slide {slide_number}" if slide_number else "slide ?"
    print(
        f"  [{model}] {slide_label} {category} batch {batch_index}/{total_batches} 요청 중: {ids}",
        flush=True,
    )
    rows, usage = _call_model_for_batch(
        model=model,
        category=category,
        batch=batch,
        current_date=current_date,
        max_tokens=max_tokens,
    )
    for row in rows:
        row["batch_index"] = batch_index
    ok_count = sum(1 for row in rows if row.get("status") == "ok")
    print(
        f"  [{model}] {slide_label} {category} batch {batch_index}/{total_batches} 완료: {ok_count}/{len(rows)} parsed",
        flush=True,
    )
    return {
        "model": model,
        "provider": resolved["provider"],
        "resolved_model": resolved["resolved_model"],
        "category": category,
        "batch_index": batch_index,
        "judgments": rows,
        "token_usage": usage,
    }


def _dry_run_model_results(models: list[str]) -> dict[str, dict[str, Any]]:
    results = {}
    for model in models:
        try:
            resolved = _resolve_model_spec(model)
        except Exception:
            resolved = {"provider": "unknown", "resolved_model": model}
        results[model] = {
            "model": model,
            "provider": resolved["provider"],
            "resolved_model": resolved["resolved_model"],
            "status": "dry_run",
            "judgments": [],
            "token_usage": {},
        }
    return results


def _weighted_final_score(
    verdicts: list[dict[str, Any]],
    model_weights: dict[str, float],
) -> tuple[float, dict[str, float], float, float, bool]:
    score = 0.0
    used_weights: dict[str, float] = {}
    missing_weight = 0.0
    model_scores = []
    for verdict in verdicts:
        model = str(verdict.get("model") or "")
        weight = float(model_weights.get(model, 0.0) or 0.0)
        if verdict.get("status") != "ok":
            missing_weight += weight
            continue
        model_score = _clamp01(verdict.get("final_model_score"))
        score += model_score * weight
        used_weights[model] = weight
        model_scores.append(model_score)
    if not used_weights:
        return 0.0, {}, round(missing_weight, 6), 0.0, False
    disagreement = max(model_scores) - min(model_scores) if len(model_scores) > 1 else 0.0
    return round(score, 6), used_weights, round(missing_weight, 6), round(disagreement, 6), bool(disagreement >= 0.35)


def _verdict_model_family(verdict: dict[str, Any]) -> str:
    model = str(verdict.get("model") or "").strip().lower()
    provider = str(verdict.get("provider") or "").strip().lower()
    resolved_model = str(verdict.get("resolved_model") or "").strip().lower()
    if provider in {"openai", "anthropic", "xai"}:
        return provider
    if model.startswith(("gpt", "o1", "o3")) or resolved_model.startswith(("gpt", "o1", "o3")):
        return "gpt"
    if (
        model.startswith("claude")
        or resolved_model.startswith("claude")
        or any(token in model for token in ("sonnet", "haiku", "opus"))
        or any(token in resolved_model for token in ("sonnet", "haiku", "opus"))
    ):
        return "claude"
    if model.startswith("grok") or resolved_model.startswith("grok"):
        return "xai"
    return model


def _effective_model_weights(
    category: str,
    verdicts: list[dict[str, Any]],
    base_weights: dict[str, float],
) -> dict[str, float]:
    override = CATEGORY_MODEL_WEIGHT_OVERRIDES.get(category)
    if not override:
        return base_weights

    weights: dict[str, float] = {}
    seen_models = {str(verdict.get("model") or "") for verdict in verdicts if str(verdict.get("model") or "")}
    for model in seen_models:
        verdict = next((row for row in verdicts if str(row.get("model") or "") == model), {})
        family = _verdict_model_family(verdict)
        weights[model] = float(override.get(family, 0.0) or 0.0)

    total = sum(weights.values())
    if total <= 0:
        return base_weights
    return {model: round(weight / total, 6) for model, weight in weights.items()}


def _issue_result_record(
    ref: dict[str, Any],
    verdicts: list[dict[str, Any]],
    *,
    model_weights: dict[str, float],
) -> dict[str, Any]:
    issue = ref["issue"]
    effective_model_weights = _effective_model_weights(ref["category"], verdicts, model_weights)
    final_score, used_weights, missing_weight, disagreement, needs_manual_review = _weighted_final_score(
        verdicts,
        effective_model_weights,
    )
    final_status = _status_from_severity(final_score)
    ok_verdicts = [row for row in verdicts if row.get("status") == "ok"]
    avg_is_valid = sum(_clamp01(row.get("is_valid_issue")) for row in ok_verdicts) / len(ok_verdicts) if ok_verdicts else 0.0
    include_axis_scores = ref["category"] != "scope_overclaim"
    if include_axis_scores:
        avg_severity = (
            sum(_clamp01(row.get("category_severity")) for row in ok_verdicts) / len(ok_verdicts)
            if ok_verdicts
            else 0.0
        )
        avg_context_resolution = (
            sum(_clamp01(row.get("context_resolution")) for row in ok_verdicts) / len(ok_verdicts)
            if ok_verdicts
            else 0.0
        )
    record = {
        "id": ref["id"],
        "issue_id": issue.get("issue_id", ""),
        "claim_id": issue.get("claim_id", ""),
        "resolved_claim": issue.get("resolved_claim", ""),
        "claim_text": issue.get("claim_text", ""),
        "category": ref["category"],
        "category_label": ref["category_label"],
        "location": issue.get("location", {}),
        "context": issue.get("context", {}),
        "judge_context": {
            "domain": ref.get("domain", ""),
            "subdomain": ref.get("subdomain", ""),
            "slide": ref.get("slide", {}),
            "context_bundle": ref.get("context_bundle", {}),
        },
        "previous_classification": {
            "weighted_scores": issue.get("weighted_scores", {}),
            "ensemble_confidence": issue.get("ensemble_confidence", 0.0),
            "low_margin": bool(issue.get("low_margin")),
            "margin": issue.get("margin", 0.0),
        },
        "final_severity_score": final_score,
        "final_severity_percent": round(final_score * 100.0, 2),
        "average_is_valid_issue": round(avg_is_valid, 6),
        "model_weights": used_weights,
        "missing_model_weight": missing_weight,
        "model_disagreement": disagreement,
        "model_disagreement_needs_review": needs_manual_review,
        "needs_manual_review": final_status == "professor_check",
        "model_judgments": verdicts,
    }
    if include_axis_scores:
        record["average_category_severity"] = round(avg_severity, 6)
        record["average_context_resolution"] = round(avg_context_resolution, 6)
    return record


def _group_issue_results(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped = {category: [] for category in ISSUE_TYPES}
    for record in records:
        category = record.get("category")
        if category in grouped:
            grouped[category].append(record)
    for rows in grouped.values():
        rows.sort(key=lambda item: (
            -float(item.get("final_severity_score") or 0.0),
            float((item.get("location") or {}).get("start_time") or 0.0),
            str(item.get("issue_id") or ""),
        ))
    return grouped


def _summary(records: list[dict[str, Any]], model_results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    by_type = Counter(record.get("category") or "unknown" for record in records)
    manual_review_count = sum(1 for record in records if record.get("needs_manual_review"))
    high_count = sum(
        1
        for record in records
        if float(record.get("final_severity_score") or 0.0) >= _confirmed_threshold()
    )
    return {
        "total_issue_count": len(records),
        "breakdown_by_type": {category: by_type.get(category, 0) for category in ISSUE_TYPES},
        "high_severity_count": high_count,
        "needs_manual_review_count": manual_review_count,
        "model_breakdown": {
            model: {
                "status": result.get("status", ""),
                "provider": result.get("provider", ""),
                "resolved_model": result.get("resolved_model", ""),
                "judgment_count": len(result.get("judgments", []) or []),
                "parse_failed_count": sum(
                    1 for row in result.get("judgments", []) or [] if row.get("status") != "ok"
                ),
            }
            for model, result in model_results.items()
        },
    }


def build_content_verification_view(result: dict[str, Any]) -> dict[str, Any]:
    """Convert severity output into the web verifier response shape.

    The frontend already knows how to render ``content_verification.v2`` style
    ``feedback_items``. Keeping this adapter here lets the new classified issue
    pipeline show up in the existing verifier page without a separate UI pass.
    """

    all_issues = result.get("all_issues", []) or []
    feedback_items = []
    claims = []
    for index, issue in enumerate(all_issues, start=1):
        if not isinstance(issue, dict):
            continue
        score = _clamp01(issue.get("final_severity_score"))
        status = _status_from_severity(score)
        issue_id = str(issue.get("issue_id") or f"I{index:04d}")
        claim_id = str(issue.get("claim_id") or issue_id)
        category = str(issue.get("category") or "")
        category_label = str(issue.get("category_label") or CATEGORY_LABELS.get(category, category))
        model_judgments = []
        for row in issue.get("model_judgments", []) or []:
            if not isinstance(row, dict):
                continue
            model_row = {
                "model": row.get("model", ""),
                "resolved_model": row.get("resolved_model", ""),
                "provider": row.get("provider", ""),
                "decision": row.get("judgment", ""),
                "status": row.get("status", ""),
                "confidence": row.get("final_model_score", 0.0),
                "vote_score": row.get("final_model_score", 0.0),
                "score": row.get("final_model_score", 0.0),
                "model_weight": (issue.get("model_weights") or {}).get(row.get("model", ""), 0.0),
                "reason": row.get("reason", ""),
                "is_valid_issue": row.get("is_valid_issue", 0.0),
            }
            if "minimal_fix" in row:
                model_row["minimal_fix"] = row.get("minimal_fix", "")
            if "category_severity" in row:
                model_row["category_severity"] = row.get("category_severity", 0.0)
            if "context_resolution" in row:
                model_row["context_resolution"] = row.get("context_resolution", 0.0)
            model_judgments.append(model_row)
        reason = " / ".join(
            row.get("reason", "")
            for row in model_judgments
            if str(row.get("reason", "")).strip()
        )
        minimal_fix = next(
            (
                row.get("minimal_fix", "")
                for row in model_judgments
                if str(row.get("minimal_fix", "")).strip()
            ),
            "",
        )
        location = issue.get("location") if isinstance(issue.get("location"), dict) else {}
        context = issue.get("context") if isinstance(issue.get("context"), dict) else {}
        claim = {
            "claim_id": claim_id,
            "claim_text": issue.get("claim_text", ""),
            "resolved_claim": issue.get("resolved_claim", ""),
            "claim_type": category,
            "context_id": context.get("context_id", ""),
            "context_ids": context.get("context_ids", []),
            "slide_number": location.get("slide_number"),
            "start_time": location.get("start_time"),
            "end_time": location.get("end_time"),
        }
        claims.append(claim)
        feedback_items.append(
            {
                "feedback_id": f"F{index:04d}",
                "issue_id": issue_id,
                "source_claim_id": claim_id,
                "status": status,
                "feedback_type": category,
                "feedback_label": category_label,
                "claim_text": issue.get("claim_text", ""),
                "resolved_claim": issue.get("resolved_claim", ""),
                "location": location,
                "severity_score": score,
                "severity_score_percent": round(score * 100.0, 2),
                "severity_status": status,
                "problem": {
                    "problematic_content": issue.get("resolved_claim") or issue.get("claim_text", ""),
                    "summary": reason or f"{category_label} 후보입니다.",
                    "why_wrong": reason,
                    "issue_basis": category_label,
                    "recommendation": minimal_fix,
                    "correct_info": minimal_fix,
                },
                "professor_feedback": {
                    "summary": reason,
                    "why_wrong": reason,
                    "teaching_note": minimal_fix,
                    "suggested_rephrase": minimal_fix,
                },
                "evidence": {
                    "slide_number": location.get("slide_number"),
                    "evidence_in_context": reason,
                    "source_issues": [issue],
                },
                "checks": {
                    "severity": {
                        "score": score,
                        "score_percent": round(score * 100.0, 2),
                        "status_by_score": status,
                        "verdict": status,
                        "model_results": model_judgments,
                    }
                },
                "classified_issue_verifier": {
                    "final_severity_score": score,
                    "final_severity_percent": issue.get("final_severity_percent", round(score * 100.0, 2)),
                    "average_is_valid_issue": issue.get("average_is_valid_issue", 0.0),
                    "model_disagreement": issue.get("model_disagreement", 0.0),
                    "needs_manual_review": bool(issue.get("needs_manual_review")),
                },
            }
        )
        if "average_context_resolution" in issue:
            feedback_items[-1]["problem"]["context_resolution"] = f"{issue.get('average_context_resolution', 0.0):.2f}"
            feedback_items[-1]["classified_issue_verifier"]["average_context_resolution"] = issue.get(
                "average_context_resolution",
                0.0,
            )
        if "average_category_severity" in issue:
            feedback_items[-1]["classified_issue_verifier"]["average_category_severity"] = issue.get(
                "average_category_severity",
                0.0,
            )

    confirmed = [item for item in feedback_items if item.get("status") == "confirmed"]
    review = [item for item in feedback_items if item.get("status") == "professor_check"]
    rejected = [item for item in feedback_items if item.get("status") == "rejected"]
    breakdown = Counter(item.get("feedback_type") or "unknown" for item in feedback_items)
    return {
        "schema_version": "content_verification.v2",
        "mode": "classified_issue_verifier",
        "verification_date": result.get("generated_at", ""),
        "models": list((result.get("model_weights") or {}).keys()),
        "verifier_source_models": list((result.get("model_weights") or {}).keys()),
        "verifier_model_weights": result.get("model_weights", {}),
        "summary": {
            "total_feedback_count": len(feedback_items),
            "confirmed_feedback_count": len(confirmed),
            "review_needed_feedback_count": len(review),
            "rejected_feedback_count": len(rejected),
            "breakdown_by_type": dict(breakdown),
        },
        "counts": {
            "final_confirmed": len(confirmed),
            "needs_review": len(review),
            "rejected": len(rejected),
        },
        "claims": claims,
        "feedback_items": feedback_items,
        "views": {
            "classified_issue_verifier": result,
        },
        "final_confirmed_claims": confirmed,
        "needs_review_claims": review,
        "verifier_rejected_claims": rejected,
        "claim_decision_flow_summary": {
            "final_confirmed_claim_count": len(confirmed),
            "needs_review_claim_count": len(review),
            "verifier_rejected_claim_count": len(rejected),
        },
        "issues": all_issues,
        "classified_issue_verifier_path": result.get("output_path", ""),
    }


def judge_classified_issues(
    payload: dict[str, Any],
    *,
    input_path: str | Path,
    merged_clean_path: str | Path | None,
    slide_textualized_path: str | Path | None,
    slide_classified_path: str | Path | None,
    models: list[str],
    batch_size: int,
    current_date: str,
    max_tokens: int,
    max_workers: int,
    context_window: int,
    limit: int | None = None,
    dry_run: bool = False,
    model_weights_spec: str | None = None,
) -> dict[str, Any]:
    _load_env()
    refs = _flatten_issues(payload, limit=limit)
    merged_payload = _load_json(merged_clean_path)
    slide_lookup = _build_slide_lookup(merged_payload)
    context_by_id, contexts_by_slide = _build_context_lookup(merged_payload)
    domain = str(merged_payload.get("domain") or "")
    subdomain = str(merged_payload.get("subdomain") or "")
    bundles = [
        _build_context_bundle(
            ref,
            domain=domain,
            subdomain=subdomain,
            slide_lookup=slide_lookup,
            context_by_id=context_by_id,
            contexts_by_slide=contexts_by_slide,
            context_window=context_window,
        )
        for ref in refs
    ]

    model_results: dict[str, dict[str, Any]]
    if dry_run:
        model_results = _dry_run_model_results(models)
    else:
        model_results = {}
        for model in models:
            try:
                resolved = _resolve_model_spec(model)
            except Exception:
                resolved = {"provider": "unknown", "resolved_model": model}
            model_results[model] = {
                "model": model,
                "provider": resolved["provider"],
                "resolved_model": resolved["resolved_model"],
                "status": "ok",
                "judgments": [],
                "token_usage_by_batch": [],
                "batch_errors": [],
            }

        work_items = []
        for model in models:
            batches = _chunk_by_slide_and_category(bundles, batch_size)
            for batch_index, batch in enumerate(batches, start=1):
                category = _batch_category_label(batch)
                work_items.append((model, category, batch, batch_index, len(batches), current_date, max_tokens))

        if max_workers <= 1:
            for args in work_items:
                try:
                    result = _batch_worker(args)
                except Exception as exc:
                    model = args[0]
                    model_results[model]["status"] = "partial_failed"
                    model_results[model]["batch_errors"].append({"category": args[1], "batch_index": args[3], "error": str(exc)})
                    continue
                model_results[result["model"]]["judgments"].extend(result["judgments"])
                model_results[result["model"]]["token_usage_by_batch"].append(result["token_usage"])
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(_batch_worker, args): args for args in work_items}
                for future in as_completed(futures):
                    args = futures[future]
                    try:
                        result = future.result()
                    except Exception as exc:
                        model = args[0]
                        model_results[model]["status"] = "partial_failed"
                        model_results[model]["batch_errors"].append({
                            "category": args[1],
                            "batch_index": args[3],
                            "error": str(exc),
                        })
                        continue
                    model_results[result["model"]]["judgments"].extend(result["judgments"])
                    model_results[result["model"]]["token_usage_by_batch"].append(result["token_usage"])

        for result in model_results.values():
            usages = result.pop("token_usage_by_batch", [])
            result["token_usage"] = _aggregate_token_usage(usages)

    model_weights = _parse_model_weights(model_weights_spec, models, model_results)
    verdicts_by_id: dict[str, list[dict[str, Any]]] = {}
    for result in model_results.values():
        for row in result.get("judgments", []) or []:
            verdicts_by_id.setdefault(str(row.get("id") or ""), []).append(row)

    records = [
        _issue_result_record(bundle, verdicts_by_id.get(bundle["id"], []), model_weights=model_weights)
        for bundle in bundles
    ]
    grouped = _group_issue_results(records)
    token_usage = Counter()
    for result in model_results.values():
        usage = result.get("token_usage") or {}
        for key in TOKEN_USAGE_FIELDS:
            token_usage[key] += int(usage.get(key, 0) or 0)

    return {
        "schema_version": SCHEMA_VERSION,
        "stage": "classified_issue_verifier",
        "source_input_path": str(input_path),
        "source_classification_path": payload.get("source_classification_path", ""),
        "source_issue_path": payload.get("source_issue_path", ""),
        "merged_clean_path": str(merged_clean_path or ""),
        "slide_textualized_path": str(slide_textualized_path or ""),
        "slide_classified_path": str(slide_classified_path or ""),
        "generated_at": _now_iso(),
        "current_date": current_date,
        "categories": {category: CATEGORY_LABELS.get(category, category) for category in ISSUE_TYPES},
        "model_weights": model_weights,
        "summary": _summary(records, model_results),
        "issues_by_type": grouped,
        "all_issues": records,
        "model_results": model_results,
        "token_usage": dict(token_usage),
    }


def _default_output_path(input_path: Path) -> Path:
    stem = input_path.stem
    if stem.endswith("_classified_issues"):
        stem = stem[: -len("_classified_issues")]
    elif stem.endswith("_issue_judge"):
        stem = stem[: -len("_issue_judge")]
    return input_path.with_name(f"{stem}_classified_issue_verifier.json")


def _guess_related_path(input_path: Path, suffix: str) -> Path:
    stem = input_path.stem
    if stem.endswith("_classified_issues"):
        prefix = stem[: -len("_classified_issues")]
    else:
        prefix = stem
    if suffix.startswith("../"):
        return (input_path.parent / suffix).resolve()
    return input_path.with_name(f"{prefix}{suffix}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the category-specific final verifier over classified issues.",
    )
    parser.add_argument("input_json", help="classified_issue_input.v1 JSON path")
    parser.add_argument("-o", "--output", help="output JSON path")
    parser.add_argument("--merged-clean", help="merged_clean JSON path")
    parser.add_argument("--slide-textualized", help="slide_textualized JSON path")
    parser.add_argument("--slide-classified", help="slide_classified JSON path")
    parser.add_argument(
        "--models",
        default=",".join(_default_models()),
        help="comma/space separated model list. Default: CLASSIFIED_ISSUE_VERIFIER_MODELS or preset",
    )
    parser.add_argument(
        "--model-weights",
        default=os.getenv("CLASSIFIED_ISSUE_VERIFIER_MODEL_WEIGHTS", DEFAULT_MODEL_WEIGHTS),
        help="comma/space separated weights, e.g. gpt=0.4,claude=0.4,grok=0.2",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(os.getenv("CLASSIFIED_ISSUE_VERIFIER_BATCH_SIZE", "4")),
    )
    parser.add_argument("--max-tokens", type=int, default=int(os.getenv("CLASSIFIED_ISSUE_VERIFIER_MAX_TOKENS", "8192")))
    parser.add_argument("--max-workers", type=int, default=int(os.getenv("CLASSIFIED_ISSUE_VERIFIER_MAX_WORKERS", "1")))
    parser.add_argument("--context-window", type=int, default=int(os.getenv("CLASSIFIED_ISSUE_VERIFIER_CONTEXT_WINDOW", str(DEFAULT_CONTEXT_WINDOW))))
    parser.add_argument("--current-date", default=os.getenv("CLASSIFIED_ISSUE_VERIFIER_CURRENT_DATE", "2026-05-14"))
    parser.add_argument("--limit", type=int, default=None, help="optional issue count limit for quick tests")
    parser.add_argument("--dry-run", action="store_true", help="validate input/output shape without calling LLMs")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    input_path = Path(args.input_json)
    output_path = Path(args.output) if args.output else _default_output_path(input_path)
    merged_clean_path = Path(args.merged_clean) if args.merged_clean else _guess_related_path(input_path, "_merged_clean.json")
    slide_textualized_path = (
        Path(args.slide_textualized)
        if args.slide_textualized
        else _guess_related_path(input_path, "../" + input_path.parent.parent.name + "_slide_textualized.json")
    )
    slide_classified_path = (
        Path(args.slide_classified)
        if args.slide_classified
        else _guess_related_path(input_path, "../" + input_path.parent.parent.name + "_slide_classified.json")
    )

    payload = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("입력 JSON 최상위 객체는 dict여야 합니다.")

    models = _split_csv(args.models)
    if not models:
        raise ValueError("사용할 모델이 없습니다. --models 또는 CLASSIFIED_ISSUE_VERIFIER_MODELS를 설정하세요.")

    result = judge_classified_issues(
        payload,
        input_path=input_path,
        merged_clean_path=merged_clean_path,
        slide_textualized_path=slide_textualized_path,
        slide_classified_path=slide_classified_path,
        models=models,
        batch_size=max(1, args.batch_size),
        current_date=args.current_date,
        max_tokens=max(256, args.max_tokens),
        max_workers=max(1, args.max_workers),
        context_window=max(0, args.context_window),
        limit=args.limit,
        dry_run=args.dry_run,
        model_weights_spec=args.model_weights,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = result["summary"]
    print(f"입력 issue: {summary['total_issue_count']}건")
    print(f"모델: {', '.join(models)}")
    print(f"모델 가중치: {json.dumps(result.get('model_weights', {}), ensure_ascii=False)}")
    print(f"출력: {output_path}")
    print(f"유형별 분포: {json.dumps(summary['breakdown_by_type'], ensure_ascii=False)}")
    print(f"high severity: {summary['high_severity_count']}건")
    print(f"needs manual review: {summary['needs_manual_review_count']}건")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
