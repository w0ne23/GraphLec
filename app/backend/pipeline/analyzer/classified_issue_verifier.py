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
DEFAULT_CONTEXT_WINDOW = 2
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

CATEGORY_RUBRICS = {
    "factual_error": """# 사실 오류

기본: 정의, 용어, 동작 원리, 관계, 순서, 메커니즘, 수식, 인과관계 등 객관적으로 틀린 사실 오류를 판단한다. 기준일과 무관하게 명제 자체가 틀렸는지 본다.

+요소: 문맥, 해당 슬라이드, 앞뒤 context, t1_structure를 포함해도 claim이 참으로 복원되지 않는다면 점수를 높인다.

-요소: 문맥을 포함해 판단했을 때 정확한 의미가 복원되거나, 앞뒤 설명이 문제를 해소한다면 점수를 낮춘다.""",
    "temporal_error": """# 시대적 오류

기본: 현재 기준으로 업데이트되지 않은 정보인지 판단한다. 과거 어느 시점에는 맞았거나 자연스러웠을 수 있지만, 현재 기준으로 더 이상 맞지 않는 정보인지 본다.

+요소: 문맥을 포함해 판단했을 때 해당 영상이 오래된 강의이고, claim이 현재 사실처럼 제시되어 현재 학습자에게 outdated 정보를 줄 가능성이 크다면 점수를 높인다.

-요소: 강의 자체나 내용이 단순히 업데이트되지 않은 문제가 아니라, 문맥상 과거 시점/역사/당시 상황을 설명하는 내용이거나 현재성 claim이 아니라면 점수를 낮춘다.""",
    "confusing_explanation": """# 혼동 오류

기본: 명제가 명백히 틀렸다고 단정하기보다는, 비유/예시/생략/표현 방식 때문에 학생이 해당 명제를 다른 의미로 해석할 위험이 있는 설명인지 판단한다.

+요소: 문맥을 이용해서 봤을 때도 학생의 오해가 해소되지 않고, 구체적으로 잘못된 mental model을 만들 위험이 있다면 점수를 높인다.

-요소: 문맥을 봤을 때 학생이 오해할 만한 부분이 해소되거나, 단순히 더 친절한 설명이 가능한 정도라면 점수를 낮춘다.""",
    "scope_overclaim": """# 범위 오류

기본: 조건, 예외, 범위, 적용 대상을 닫아버려 과도하게 일반화한 오류인지 판단한다. "항상/오직/모든/유일한/전부/완전히/~만" 같은 범위 표현을 제거하거나 완화하면 대체로 맞는 명제가 되는 경우를 본다.

+요소: 문맥을 봤는데도 강조를 위한 표현이 아니라 실제 범위 단정 표현으로 남고, 조건/예외/적용 대상이 닫혀 있다면 점수를 높인다.

-요소: 문맥상 범위 단정의 단어나 표현이 강조를 위한 표현임이 확실하거나, 앞뒤 문맥에서 조건/예외가 충분히 복원된다면 점수를 낮춘다.""",
}


def _status_from_severity(score: float) -> str:
    confirmed = _safe_float(os.getenv("CLASSIFIED_ISSUE_VERIFIER_CONFIRMED_THRESHOLD"), 0.50)
    rejected = _safe_float(os.getenv("CLASSIFIED_ISSUE_VERIFIER_REJECTED_THRESHOLD"), 0.20)
    if score >= confirmed:
        return "confirmed"
    if score <= rejected:
        return "rejected"
    return "professor_check"


def _confirmed_threshold() -> float:
    return _safe_float(os.getenv("CLASSIFIED_ISSUE_VERIFIER_CONFIRMED_THRESHOLD"), 0.50)


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
    for slide in merged_payload.get("slides") or []:
        if not isinstance(slide, dict):
            continue
        try:
            slide_number = int(slide.get("slide_number") or 0)
        except (TypeError, ValueError):
            continue
        contexts = slide.get("contexts") or []
        if not isinstance(contexts, list):
            continue
        for index, context in enumerate(contexts):
            if not isinstance(context, dict):
                continue
            row = dict(context)
            row.setdefault("slide_number", slide_number)
            row.setdefault("context_index", index)
            context_id = str(row.get("context_id") or "").strip()
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
        "slide_number": context.get("slide_number"),
        "context_index": context.get("context_index"),
        "start_time": context.get("start_time"),
        "end_time": context.get("end_time"),
        "text": context.get("text", ""),
    }


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
    if not center_contexts and slide_number:
        rows = contexts_by_slide.get(slide_number) or []
        if rows:
            center_contexts = rows[:1]

    around: list[dict[str, Any]] = []
    seen: set[str] = set()
    for center in center_contexts:
        try:
            sn = int(center.get("slide_number") or slide_number or 0)
            idx = int(center.get("context_index") or 0)
        except (TypeError, ValueError):
            continue
        rows = contexts_by_slide.get(sn) or []
        for item in rows[max(0, idx - window) : idx + window + 1]:
            key = str(item.get("context_id") or f"{sn}:{item.get('context_index')}")
            if key in seen:
                continue
            seen.add(key)
            around.append(_compact_context(item))

    return {
        "target_context_ids": context_ids,
        "target_contexts": [_compact_context(item) for item in center_contexts],
        "neighbor_contexts": around,
    }


def _compact_slide(slide: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "slide_number",
        "title",
        "time_range",
        "slide_text",
        "t1",
        "t1_structure",
        "slide_type",
        "role",
        "image_path",
        "slide_topic_keywords",
        "slide_emphasis",
    )
    return {key: slide.get(key) for key in keys if slide.get(key) not in (None, "", [], {})}


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
    location = issue.get("location") if isinstance(issue.get("location"), dict) else {}
    context_meta = issue.get("context") if isinstance(issue.get("context"), dict) else {}
    return {
        "id": item.get("id"),
        "issue_id": issue.get("issue_id", ""),
        "claim_id": issue.get("claim_id", ""),
        "resolved_claim": issue.get("resolved_claim", ""),
        "claim_text": issue.get("claim_text", ""),
        "previous_classification": {
            "final_issue_type": issue.get("final_issue_type"),
            "final_issue_type_label": issue.get("final_issue_type_label", ""),
            "weighted_scores": issue.get("weighted_scores", {}),
            "ensemble_confidence": issue.get("ensemble_confidence", 0.0),
            "low_margin": bool(issue.get("low_margin")),
            "margin": issue.get("margin", 0.0),
        },
        "location": location,
        "context_meta": context_meta,
        "domain": item.get("domain", ""),
        "subdomain": item.get("subdomain", ""),
        "slide": item.get("slide", {}),
        "context_bundle": item.get("context_bundle", {}),
    }


def _build_prompt(category: str, items: list[dict[str, Any]], current_date: str) -> str:
    rubric = CATEGORY_RUBRICS.get(category, "")
    rows = [_prompt_issue_brief(item) for item in items]
    return f"""당신은 강의 verifier의 최종 FACT check 심사자입니다.

오늘 날짜: {current_date}
판정 분류: {CATEGORY_LABELS.get(category, category)} ({category})

아래 rubric은 이 요청에서 유일하게 적용할 분류 기준입니다. 다른 분류로 재분류하지 말고, 이 분류 기준 안에서만 issue의 유효성과 심각도를 판단하세요.

{rubric}

입력으로 제공되는 정보:
- claim의 도메인/서브도메인
- resolved_claim과 원문 claim_text
- 해당 context와 앞뒤 context
- 해당 슬라이드의 텍스트/t1/t1_structure/역할/이미지 경로
- 이전 분류 단계의 weighted_scores와 low_margin 정보

출력 점수:
- is_valid_issue: 이 분류 기준으로 실제 issue일 가능성. 0.0~1.0.
- category_severity: 이 분류 안에서 오류가 얼마나 심각한지. 0.0~1.0.
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
- reason은 한두 문장으로 쓰세요.
- minimal_fix는 가능하면 claim을 어떻게 완화/수정하면 되는지 짧게 쓰고, 없으면 빈 문자열로 두세요.

```json
{{
  "judgments": [
    {{
      "id": "입력 id",
      "judgment": "valid_issue",
      "is_valid_issue": 0.0,
      "category_severity": 0.0,
      "context_resolution": 0.0,
      "reason": "판단 근거",
      "minimal_fix": "수정안"
    }}
  ]
}}
```

입력 issue:
{json.dumps(rows, ensure_ascii=False, indent=2)}
"""


def _parse_response(text: str) -> list[dict[str, Any]]:
    payload = json.loads(_strip_json_fence(text))
    rows = payload.get("judgments", [])
    return rows if isinstance(rows, list) else []


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
    final_model_score = _clamp01(is_valid_issue * category_severity * (1.0 - context_resolution))
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
        "final_model_score": final_model_score,
        "reason": str(row.get("reason", "") or "").strip(),
        "minimal_fix": str(row.get("minimal_fix", "") or "").strip(),
        "status": "ok",
        "parse_error": "",
    }


def _parse_failed_row(ref: dict[str, Any], model: str, resolved: dict[str, str], error: str) -> dict[str, Any]:
    return {
        "id": ref["id"],
        "model": model,
        "provider": resolved.get("provider", ""),
        "resolved_model": resolved.get("resolved_model", model),
        "category": ref["category"],
        "judgment": "insufficient_context",
        "is_valid_issue": 0.0,
        "category_severity": 0.0,
        "context_resolution": 0.0,
        "final_model_score": 0.0,
        "reason": "",
        "minimal_fix": "",
        "status": "parse_failed",
        "parse_error": error,
    }


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
    print(
        f"  [{model}] {category} batch {batch_index}/{total_batches} 요청 중: {ids}",
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
        f"  [{model}] {category} batch {batch_index}/{total_batches} 완료: {ok_count}/{len(rows)} parsed",
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


def _issue_result_record(
    ref: dict[str, Any],
    verdicts: list[dict[str, Any]],
    *,
    model_weights: dict[str, float],
) -> dict[str, Any]:
    issue = ref["issue"]
    final_score, used_weights, missing_weight, disagreement, needs_manual_review = _weighted_final_score(
        verdicts,
        model_weights,
    )
    ok_verdicts = [row for row in verdicts if row.get("status") == "ok"]
    avg_is_valid = sum(_clamp01(row.get("is_valid_issue")) for row in ok_verdicts) / len(ok_verdicts) if ok_verdicts else 0.0
    avg_severity = sum(_clamp01(row.get("category_severity")) for row in ok_verdicts) / len(ok_verdicts) if ok_verdicts else 0.0
    avg_context_resolution = (
        sum(_clamp01(row.get("context_resolution")) for row in ok_verdicts) / len(ok_verdicts)
        if ok_verdicts
        else 0.0
    )
    return {
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
        "average_category_severity": round(avg_severity, 6),
        "average_context_resolution": round(avg_context_resolution, 6),
        "model_weights": used_weights,
        "missing_model_weight": missing_weight,
        "model_disagreement": disagreement,
        "needs_manual_review": needs_manual_review or bool(issue.get("low_margin")),
        "model_judgments": verdicts,
    }


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
        model_judgments = [
            {
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
                "minimal_fix": row.get("minimal_fix", ""),
                "is_valid_issue": row.get("is_valid_issue", 0.0),
                "category_severity": row.get("category_severity", 0.0),
                "context_resolution": row.get("context_resolution", 0.0),
            }
            for row in issue.get("model_judgments", []) or []
            if isinstance(row, dict)
        ]
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
                    "context_resolution": f"{issue.get('average_context_resolution', 0.0):.2f}",
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
                    "average_category_severity": issue.get("average_category_severity", 0.0),
                    "average_context_resolution": issue.get("average_context_resolution", 0.0),
                    "model_disagreement": issue.get("model_disagreement", 0.0),
                    "needs_manual_review": bool(issue.get("needs_manual_review")),
                },
            }
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
    textualized_payload = _load_json(slide_textualized_path)
    classified_payload = _load_json(slide_classified_path)
    slide_lookup = _build_slide_lookup(merged_payload, textualized_payload, classified_payload)
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
            for category in ISSUE_TYPES:
                category_items = [item for item in bundles if item["category"] == category]
                batches = _chunk(category_items, batch_size)
                for batch_index, batch in enumerate(batches, start=1):
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
        default=int(
            os.getenv(
                "VERIFIER_CROSSCHECK_MAX_ISSUES_PER_BATCH",
                os.getenv("CLASSIFIED_ISSUE_VERIFIER_BATCH_SIZE", "5"),
            )
        ),
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
