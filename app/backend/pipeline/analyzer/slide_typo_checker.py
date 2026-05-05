from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional


def _build_slide_typo_prompt(slide_no: int, title: str, slide_text: str) -> str:
    return f"""당신은 강의 슬라이드에서 눈에 보이는 오타를 찾는 교정자입니다.

중요:
- 슬라이드 이미지가 원본입니다.
- 아래 OCR 텍스트는 보조 정보일 뿐이며, 이미지와 다르면 이미지를 우선하세요.
- 내용의 사실성, 더 좋은 표현, 문체 개선, 디자인 문제는 보고하지 마세요.
- 애매하면 보고하지 마세요.

대상 슬라이드:
- 번호: {slide_no}
- OCR 제목: {title}
- OCR 본문:
{slide_text[:3000]}

보고할 것:
- 이미지에서 명백하게 보이는 한글 철자 오타
- 영문 철자 오류
- 숫자/단위 오기

보고하지 말 것:
- OCR이 잘못 읽은 텍스트 자체
- 용어 선택/문체/표현 선호
- 사실 오류나 개념 오류
- 띄어쓰기, 줄바꿈, 글자 간격, 디자인 문제
- 약어, 고유명사, 표기 관례처럼 오타로 단정하기 어려운 것
- 복합어 띄어쓰기 관례
- 조사/어미/접속 표현 교정
- 외래어를 한국어로 순화하는 교정
- 쉼표 추가, 문장 자연화, 더 좋은 표현 제안

출력 형식은 JSON만 허용합니다.

```json
{{
  "typos": [
    {{
      "problematic_text": "슬라이드에 보이는 문제 표현",
      "corrected_text": "수정 표현",
      "reason": "왜 오타라고 보는지",
      "confidence": 0.0
    }}
  ]
}}
```

지침:
1. 확신이 0.90 미만이면 출력하지 마세요.
2. "더 자연스럽다", "더 적절하다" 수준이면 출력하지 마세요.
3. 오타가 없으면 {{"typos": []}}만 출력하세요.
4. JSON 외 텍스트 금지.
"""


def _compact_no_space(value: str) -> str:
    return re.sub(r"\s+", "", str(value or "").strip().lower())


def is_reportable_slide_typo(problematic: str, corrected: str, reason: str = "") -> bool:
    """True only for visually clear typos; reject style/terminology polish."""
    p = str(problematic or "").strip()
    c = str(corrected or "").strip()
    r = str(reason or "").strip().lower()
    if not p or not c or p == c:
        return False

    p_compact = _compact_no_space(p)
    c_compact = _compact_no_space(c)
    if not p_compact or p_compact == c_compact:
        return False
    if re.sub(r"\S+", "", p) != re.sub(r"\S+", "", c):
        return False

    style_markers = (
        "더 적절",
        "더 자연",
        "자연스럽",
        "어색",
        "문법",
        "조사",
        "순화",
        "용어",
        "표현 선호",
        "더 좋은 표현",
        "문체",
        "선택의 의미",
        "나열",
        "병렬",
        "쉼표",
        "구분",
        "별개",
        "띄어쓰기",
        "공백",
        "줄바꿈",
        "글자 간격",
    )
    if any(marker in r for marker in style_markers):
        return False
    if "중복" in r and re.search(r"(을|를|이|가|은|는)\s+\S+(을|를|이|가|은|는)", p):
        return False

    return True


def _check_single_slide(slide: dict, img_dir: Optional[str], run_index: int = 1) -> tuple[list[dict], bool, int, dict]:
    from . import claim_common as cc

    slide_no = int(slide.get("slide_number", 0) or 0)
    title = str(slide.get("title", "") or "")
    slide_text = str(slide.get("slide_text", "") or "")

    img_bytes = None
    if img_dir:
        start_path = Path(img_dir) / f"slide_{slide_no:03d}_start.jpg"
        end_path = Path(img_dir) / f"slide_{slide_no:03d}_end.jpg"
        img_path = start_path if start_path.exists() else end_path
        if img_path.exists():
            img_bytes = img_path.read_bytes()

    prompt = _build_slide_typo_prompt(slide_no, title, slide_text)
    model = str(cc._resolve_stage_model("recheck") or "").strip()
    response_format = {"type": "json_object"} if (
        model.startswith("gpt") or model.startswith("o1") or model.startswith("o3")
    ) else None

    api_calls = 0
    token_usage = cc._empty_token_usage()

    for attempt in range(cc.VERIFIER_PARSE_RETRIES + 1):
        text, call_usage = cc._call_llm(
            prompt,
            max_tokens=2048,
            temperature=0.0,
            image_bytes=img_bytes,
            thinking_budget=0,
            response_format=response_format,
            stage="recheck",
        )
        api_calls += 1
        cc._add_call_usage(token_usage, call_usage)
        try:
            payload = json.loads(cc._strip_json_fence((text or "").strip()))
            raw = payload.get("typos", [])
            if not isinstance(raw, list):
                raw = []
            cleaned = []
            for item in raw:
                if not isinstance(item, dict):
                    continue
                try:
                    conf = float(item.get("confidence", 0) or 0)
                except Exception:
                    conf = 0.0
                if conf < 0.80:
                    continue
                problematic = str(item.get("problematic_text", "") or "").strip()
                corrected = str(item.get("corrected_text", "") or "").strip()
                reason = str(item.get("reason", "") or "").strip()
                if not problematic or not corrected:
                    continue
                if not is_reportable_slide_typo(problematic, corrected, reason):
                    continue
                cleaned.append({
                    "slide_number": slide_no,
                    "slide_title": title,
                    "problematic_text": problematic,
                    "corrected_text": corrected,
                    "reason": reason,
                    "confidence": conf,
                    "run_index": run_index,
                })
            return cleaned, False, api_calls, token_usage
        except Exception:
            if attempt < cc.VERIFIER_PARSE_RETRIES:
                print(f"    ↺ 슬라이드 {slide_no} 오타 JSON 파싱 재시도 ({attempt+1}/{cc.VERIFIER_PARSE_RETRIES})")
    return [], True, api_calls, token_usage


def _safe_rate(value, default: float) -> float:
    try:
        rate = float(value)
    except Exception:
        rate = default
    return max(0.0, min(1.0, rate))


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def _typo_key(typo: dict) -> tuple:
    # Future: consider conservative prefix/substring grouping for problematic_text.
    # Avoid semantic/fuzzy merging here because typo reporting favors precision.
    return (
        int(typo.get("slide_number", 0) or 0),
        _normalize_text(typo.get("problematic_text", "")),
    )


def _meets_rate(support_count: int, run_count: int, threshold: float) -> bool:
    if run_count <= 0:
        return False
    return round(support_count / run_count, 2) >= threshold


def _collect_reasons(items: list[dict]) -> list[str]:
    reasons = []
    for item in items:
        reason = str(item.get("reason", "") or "").strip()
        if reason and reason not in reasons:
            reasons.append(reason)
    return reasons


def _correction_candidates(group: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for item in group:
        grouped.setdefault(_normalize_text(item.get("corrected_text", "")), []).append(item)

    candidates = []
    for items in grouped.values():
        best = max(items, key=lambda item: float(item.get("confidence", 0) or 0))
        confidences = [float(item.get("confidence", 0) or 0) for item in items]
        run_indices = sorted({
            int(item.get("run_index", 0) or 0)
            for item in items
            if int(item.get("run_index", 0) or 0) > 0
        })
        reasons = _collect_reasons(items)
        candidate = {
            "corrected_text": best.get("corrected_text", ""),
            "support_count": len(run_indices),
            "confidence": max(confidences) if confidences else 0.0,
            "average_confidence": sum(confidences) / len(confidences) if confidences else 0.0,
            "supporting_runs": run_indices,
        }
        if reasons:
            candidate["reason"] = reasons[0]
        if len(reasons) > 1:
            candidate["reasons"] = reasons
        candidates.append(candidate)

    return sorted(
        candidates,
        key=lambda item: (
            -int(item.get("support_count", 0) or 0),
            -float(item.get("confidence", 0) or 0),
            str(item.get("corrected_text", "")),
        ),
    )


def _merge_typo_group(group: list[dict], *, run_count: int, status: str) -> dict:
    best = max(group, key=lambda item: float(item.get("confidence", 0) or 0))
    confidences = [float(item.get("confidence", 0) or 0) for item in group]
    reasons = _collect_reasons(group)
    candidates = _correction_candidates(group)
    top_candidate = candidates[0] if candidates else {}
    run_indices = sorted({
        int(item.get("run_index", 0) or 0)
        for item in group
        if int(item.get("run_index", 0) or 0) > 0
    })
    support_count = len(run_indices)
    merged = {
        "slide_number": best.get("slide_number"),
        "slide_title": best.get("slide_title", ""),
        "problematic_text": best.get("problematic_text", ""),
        "corrected_text": top_candidate.get("corrected_text") or best.get("corrected_text", ""),
        "correction_candidates": candidates,
        "reason": top_candidate.get("reason") or (reasons[0] if reasons else str(best.get("reason", "") or "")),
        "confidence": max(confidences) if confidences else 0.0,
        "average_confidence": sum(confidences) / len(confidences) if confidences else 0.0,
        "support_count": support_count,
        "run_count": run_count,
        "detection_rate": support_count / run_count if run_count else 0.0,
        "supporting_runs": run_indices,
        "consensus_status": status,
    }
    if len(reasons) > 1:
        merged["reasons"] = reasons
    return merged


def _split_by_consensus(
    typos: list[dict],
    *,
    run_count: int,
    min_detection_rate: float,
    review_min_detection_rate: float,
) -> tuple[list[dict], list[dict]]:
    grouped: dict[tuple, list[dict]] = {}
    for typo in typos:
        grouped.setdefault(_typo_key(typo), []).append(typo)

    confirmed: list[dict] = []
    needs_review: list[dict] = []
    for group in grouped.values():
        support_count = len({
            int(item.get("run_index", 0) or 0)
            for item in group
            if int(item.get("run_index", 0) or 0) > 0
        })
        if _meets_rate(support_count, run_count, min_detection_rate):
            confirmed.append(_merge_typo_group(group, run_count=run_count, status="confirmed"))
        elif _meets_rate(support_count, run_count, review_min_detection_rate):
            review_item = _merge_typo_group(group, run_count=run_count, status="needs_review")
            review_item["review_stage"] = "slide_typo"
            review_item["review_reason_code"] = "low_typo_consensus"
            needs_review.append(review_item)

    sort_key = lambda x: (
        int(x.get("slide_number", 0) or 0),
        -float(x.get("detection_rate", 0) or 0),
        -float(x.get("confidence", 0) or 0),
    )
    return sorted(confirmed, key=sort_key), sorted(needs_review, key=sort_key)


def detect_slide_typos(
    slides: list[dict],
    img_dir: Optional[str] = None,
    max_workers: int = 4,
) -> tuple[list[dict], list[dict], int, int, dict]:
    from . import claim_common as cc

    if not slides:
        return [], [], 0, 0, cc._empty_token_usage()

    results: list[dict] = []
    api_calls = 0
    failures = 0
    token_usage = cc._empty_token_usage()
    num_runs = max(1, int(getattr(cc, "VERIFIER_SLIDE_TYPO_RUNS", 1) or 1))
    min_detection_rate = _safe_rate(getattr(cc, "VERIFIER_SLIDE_TYPO_MIN_RATE", 1.0), 1.0)
    review_min_detection_rate = _safe_rate(
        getattr(cc, "VERIFIER_SLIDE_TYPO_REVIEW_MIN_RATE", 0.5),
        0.5,
    )
    review_min_detection_rate = min(review_min_detection_rate, min_detection_rate)

    def process(slide: dict, run_index: int):
        sn = slide.get("slide_number", "?")
        suffix = f" ({run_index}/{num_runs})" if num_runs > 1 else ""
        print(f"    슬라이드 오타 검사 [{sn}]{suffix}")
        return _check_single_slide(slide, img_dir, run_index=run_index)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(process, slide, run_index): (slide, run_index)
            for slide in slides
            for run_index in range(1, num_runs + 1)
        }
        for future in as_completed(futures):
            try:
                typos, parse_failed, calls, usage = future.result()
                results.extend(typos)
                api_calls += calls
                token_usage = cc._merge_token_usage(token_usage, usage)
                if parse_failed:
                    failures += 1
            except Exception:
                failures += 1

    confirmed, needs_review = _split_by_consensus(
        results,
        run_count=num_runs,
        min_detection_rate=min_detection_rate,
        review_min_detection_rate=review_min_detection_rate,
    )
    return confirmed, needs_review, api_calls, failures, token_usage
