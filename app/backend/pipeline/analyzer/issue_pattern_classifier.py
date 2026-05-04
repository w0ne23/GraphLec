from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


ALLOWED_ISSUE_PATTERNS = {
    "scope_overstatement",
    "terminology_confusion",
    "ambiguous_reference",
    "internal_contradiction",
    "missing_prerequisite",
}
CACHE_VERSION = 1

ISSUE_LIST_KEYS = (
    "issues",
    "needs_review_issues",
    "grounding_rejected_issues",
    "slide_rejected_issues",
    "crosscheck_rejected_issues",
    "crosscheck_inconclusive_issues",
)


def _issue_identity(list_key: str, index: int, issue: dict) -> str:
    seed = {
        "list_key": list_key,
        "index": index,
        "utterance_id": issue.get("utterance_id", ""),
        "claim_text": issue.get("claim_text") or issue.get("problematic_content", ""),
        "issue": issue.get("issue", ""),
        "correct_info": issue.get("correct_info", ""),
    }
    digest = hashlib.sha1(json.dumps(seed, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    return f"{list_key}:{index}:{digest}"


def _collect_issue_refs(result: dict) -> list[dict]:
    refs = []
    for list_key in ISSUE_LIST_KEYS:
        for index, issue in enumerate(result.get(list_key, []) or []):
            if not isinstance(issue, dict):
                continue
            refs.append(
                {
                    "id": _issue_identity(list_key, index, issue),
                    "list_key": list_key,
                    "index": index,
                    "issue": issue,
                }
            )
    return refs


def _fingerprint(refs: list[dict]) -> str:
    payload = [
        {
            "id": ref["id"],
            "claim_text": ref["issue"].get("claim_text") or ref["issue"].get("problematic_content", ""),
            "issue": ref["issue"].get("issue", ""),
            "correct_info": ref["issue"].get("correct_info", ""),
            "grounding_status": ref["issue"].get("grounding_status", ""),
            "slide_recheck_status": ref["issue"].get("slide_recheck_status", ""),
        }
        for ref in refs
    ]
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def _safe_int_env(name: str, default: int, *, min_value: int = 1) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except Exception:
        value = default
    return max(min_value, value)


def _chunks(items: list[dict], size: int) -> list[list[dict]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def _cache_path(cache_dir: str | Path | None, batch_index: int) -> Path | None:
    if not cache_dir:
        return None
    return Path(cache_dir) / f"issue_pattern_classifier_batch_{batch_index:03d}.json"


def _load_cache(
    cache_dir: str | Path | None,
    batch_index: int,
    expected_fingerprint: str,
    expected_model: str,
    *,
    resume: bool,
) -> dict | None:
    path = _cache_path(cache_dir, batch_index)
    if not resume or path is None or not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"    ⚠️ issue_pattern 배치 {batch_index} 캐시 로드 실패, 재분류: {e}")
        return None
    if payload.get("cache_version") != CACHE_VERSION:
        print(f"    ⚠️ issue_pattern 배치 {batch_index} 캐시 버전 불일치, 재분류")
        return None
    if payload.get("fingerprint") != expected_fingerprint:
        print(f"    ⚠️ issue_pattern 배치 {batch_index} 캐시 입력 불일치, 재분류")
        return None
    if payload.get("model") != expected_model:
        print(f"    ⚠️ issue_pattern 배치 {batch_index} 캐시 모델 불일치, 재분류")
        return None
    print(f"    ↻ issue_pattern 배치 {batch_index} 캐시 사용")
    return payload


def _save_cache(
    cache_dir: str | Path | None,
    batch_index: int,
    fingerprint: str,
    classifications: list[dict],
    summary: dict,
) -> None:
    path = _cache_path(cache_dir, batch_index)
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_text(
        json.dumps(
            {
                "cache_version": CACHE_VERSION,
                "batch_index": batch_index,
                "model": summary.get("model"),
                "fingerprint": fingerprint,
                "classifications": classifications,
                "summary": summary,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    tmp_path.replace(path)


def _build_prompt(items: list[dict]) -> str:
    allowed = " | ".join(sorted(ALLOWED_ISSUE_PATTERNS))
    rows = []
    for item in items:
        issue = item["issue"]
        rows.append(
            {
                "id": item["id"],
                "claim_text": issue.get("claim_text") or issue.get("problematic_content", ""),
                "issue": issue.get("issue", ""),
                "correct_info": issue.get("correct_info", ""),
                "grounding_status": issue.get("grounding_status", ""),
                "slide_recheck_status": issue.get("slide_recheck_status", ""),
                "supporting_slide_status": issue.get("supporting_slide_status", ""),
                "error_origin": issue.get("error_origin", ""),
            }
        )

    return f"""당신은 강의 검증 시스템의 issue_pattern 분류기입니다.

아래 이슈 각각에 대해 오류 표현 패턴을 보수적으로 분류하세요.
확실하지 않으면 반드시 null을 사용하세요. 키워드 하나만 보고 억지로 분류하지 마세요.

허용 패턴:
- scope_overstatement: 항상/모든/오직/반드시처럼 범위를 과도하게 일반화한 오류
- terminology_confusion: 서로 다른 용어/개념을 동일시하거나 잘못 정의한 오류
- ambiguous_reference: 주어, 지시 대상, 생략된 표현 때문에 무엇을 말하는지 불명확한 오류
- internal_contradiction: 같은 강의 내부의 앞뒤 설명/슬라이드가 실제로 충돌하는 오류
- missing_prerequisite: 필요한 선행 개념이나 전제가 제공되지 않아 결론이 오해되는 오류
- null: 위 패턴으로 충분히 확정할 수 없음

주의:
- internal_contradiction은 이슈 텍스트가 강의 내부 충돌을 명시할 때만 선택하세요.
- missing_prerequisite은 선행 개념/전제 부족이 명시될 때만 선택하세요.
- 용어가 등장한다는 이유만으로 terminology_confusion을 선택하지 마세요.
- 패턴 분류는 참고 라벨이며, 낮은 확신은 null로 두세요.

응답은 JSON만 출력하세요.

```json
{{
  "classifications": [
    {{
      "id": "입력 id",
      "issue_pattern": "{allowed} | null",
      "issue_pattern_reason": "분류 이유. null이면 null",
      "issue_pattern_confidence": 0.0
    }}
  ]
}}
```

입력 이슈:
{json.dumps(rows, ensure_ascii=False, indent=2)}
"""


def _parse_classifications(text: str) -> list[dict]:
    from . import claim_common as cc

    payload = json.loads(cc._strip_json_fence((text or "").strip()))
    rows = payload.get("classifications", [])
    return rows if isinstance(rows, list) else []


def _normalize_pattern(value) -> str | None:
    if value is None:
        return None
    text = str(value or "").strip()
    if not text or text.lower() == "null":
        return None
    return text if text in ALLOWED_ISSUE_PATTERNS else None


def _apply_classifications(refs: list[dict], classifications: list[dict], *, min_confidence: float) -> dict:
    by_id = {str(row.get("id", "")): row for row in classifications if isinstance(row, dict)}
    classified = 0
    kept_null = 0
    for ref in refs:
        issue = ref["issue"]
        row = by_id.get(ref["id"], {})
        pattern = _normalize_pattern(row.get("issue_pattern"))
        try:
            confidence = float(row.get("issue_pattern_confidence", 0) or 0)
        except Exception:
            confidence = 0.0

        if pattern and confidence >= min_confidence:
            issue["issue_pattern"] = pattern
            issue["issue_pattern_reason"] = str(row.get("issue_pattern_reason", "") or "").strip() or None
            issue["issue_pattern_confidence"] = confidence
            issue["issue_pattern_source"] = "llm_classifier"
            classified += 1
        else:
            issue["issue_pattern"] = None
            issue["issue_pattern_reason"] = None
            issue["issue_pattern_confidence"] = confidence
            issue["issue_pattern_source"] = "llm_classifier"
            kept_null += 1
    return {"classified_count": classified, "null_count": kept_null}


def _mark_classifier_failed(refs: list[dict]) -> None:
    for ref in refs:
        ref["issue"]["issue_pattern"] = None
        ref["issue"]["issue_pattern_reason"] = None
        ref["issue"]["issue_pattern_confidence"] = 0.0
        ref["issue"]["issue_pattern_source"] = "classifier_failed"


def classify_issue_patterns(
    result: dict,
    *,
    cache_dir: str | Path | None = None,
    resume: bool = False,
    min_confidence: float = 0.70,
) -> dict:
    from . import claim_common as cc

    refs = _collect_issue_refs(result)
    batch_size = _safe_int_env("VERIFIER_ISSUE_PATTERN_BATCH_SIZE", 12)
    max_tokens = _safe_int_env("VERIFIER_ISSUE_PATTERN_MAX_TOKENS", 8192)
    if not refs:
        return {
            "status": "skipped_no_issues",
            "model": cc._resolve_stage_model("issue_pattern"),
            "issue_count": 0,
            "batch_size": batch_size,
            "batch_count": 0,
            "max_tokens": max_tokens,
            "classified_count": 0,
            "null_count": 0,
            "cached_batch_count": 0,
            "failed_batch_count": 0,
            "failed": False,
        }

    model = cc._resolve_stage_model("issue_pattern")
    response_format = {"type": "json_object"} if model.startswith(("gpt", "o1", "o3")) else None
    batches = _chunks(refs, batch_size)

    print(f"\n  ── issue_pattern LLM 분류 ({len(refs)}건, {len(batches)}배치) ──")
    total_usage = cc._empty_token_usage()
    classified_count = 0
    null_count = 0
    cached_batch_count = 0
    completed_batch_count = 0
    failed_batches: list[dict] = []

    for batch_index, batch_refs in enumerate(batches, start=1):
        batch_fingerprint = _fingerprint(batch_refs)
        cached = _load_cache(cache_dir, batch_index, batch_fingerprint, model, resume=resume)
        if cached is not None:
            summary = _apply_classifications(
                batch_refs,
                cached.get("classifications", []),
                min_confidence=min_confidence,
            )
            cached_batch_count += 1
            classified_count += summary["classified_count"]
            null_count += summary["null_count"]
            total_usage = cc._merge_token_usage(total_usage, (cached.get("summary") or {}).get("token_usage", {}))
            continue

        ids = f"{batch_refs[0]['id']}..{batch_refs[-1]['id']}"
        print(f"    issue_pattern [{batch_index}/{len(batches)}] {ids}")
        prompt = _build_prompt(batch_refs)
        try:
            text, token_usage = cc._call_llm(
                prompt,
                max_tokens=max_tokens,
                temperature=0.0,
                thinking_budget=0,
                response_format=response_format,
                stage="issue_pattern",
            )
            classifications = _parse_classifications(text)
            summary = _apply_classifications(batch_refs, classifications, min_confidence=min_confidence)
            batch_summary = {
                "status": "completed",
                "model": model,
                "issue_count": len(batch_refs),
                "classified_count": summary["classified_count"],
                "null_count": summary["null_count"],
                "failed": False,
                "token_usage": token_usage,
            }
            _save_cache(cache_dir, batch_index, batch_fingerprint, classifications, batch_summary)
            classified_count += summary["classified_count"]
            null_count += summary["null_count"]
            completed_batch_count += 1
            total_usage = cc._merge_token_usage(total_usage, token_usage)
        except Exception as e:
            print(f"    ⚠️ issue_pattern 배치 {batch_index} 분류 실패: {e}")
            _mark_classifier_failed(batch_refs)
            null_count += len(batch_refs)
            failed_batches.append(
                {
                    "batch_index": batch_index,
                    "issue_count": len(batch_refs),
                    "error": str(e),
                }
            )

    status = "completed"
    if failed_batches:
        status = "failed" if len(failed_batches) == len(batches) else "completed_with_failures"
    elif cached_batch_count == len(batches):
        status = "cached"

    return {
        "status": status,
        "model": model,
        "issue_count": len(refs),
        "batch_size": batch_size,
        "batch_count": len(batches),
        "max_tokens": max_tokens,
        "classified_count": classified_count,
        "null_count": null_count,
        "completed_batch_count": completed_batch_count,
        "cached_batch_count": cached_batch_count,
        "failed_batch_count": len(failed_batches),
        "failed_batches": failed_batches,
        "failed": bool(failed_batches),
        "token_usage": total_usage,
    }
