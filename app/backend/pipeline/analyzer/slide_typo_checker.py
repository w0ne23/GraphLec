from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional


def _norm_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _existing_path(raw_path: object) -> Optional[Path]:
    if not raw_path:
        return None
    path = Path(str(raw_path))
    candidates = [path]
    if str(path).startswith("/pipeline/"):
        try:
            candidates.append(Path.cwd() / path.relative_to("/pipeline"))
        except ValueError:
            pass
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return path if path.is_absolute() else None


def _find_slide_image(img_dir: str, slide_no: int) -> Optional[Path]:
    base = Path(img_dir)
    candidates = (
        base / f"slide_{slide_no:03d}_base.jpg",
        base / f"slide_{slide_no:03d}_base.png",
        base / f"slide_{slide_no:03d}_start.jpg",
        base / f"slide_{slide_no:03d}_start.png",
        base / f"slide_{slide_no:03d}_end.jpg",
        base / f"slide_{slide_no:03d}_end.png",
    )
    return next((path for path in candidates if path.exists()), None)


def _load_textualized_slides(merged_path: str | Path | None) -> list[dict]:
    if not merged_path:
        return []
    base = Path(merged_path).resolve().parent
    files = sorted(base.glob("*_slide_textualized.json"))
    for path in files:
        try:
            with path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            slides = payload.get("slides", [])
            if isinstance(slides, list):
                return [s for s in slides if isinstance(s, dict)]
        except Exception:
            continue
    return []


def attach_slide_image_paths(slides: list[dict], merged_path: str | Path | None) -> list[dict]:
    """Attach actual slide image paths from slide_textualized.json when merged_clean lacks them.

    merged_clean.slide_number can be a logical slide number, while slide_### image files are
    scene indices. Matching by title/text avoids sending the wrong scene image to the typo model.
    """
    textualized = _load_textualized_slides(merged_path)
    if not textualized:
        return slides

    candidates: list[dict] = []
    for item in textualized:
        image_path = str(item.get("image_path", "") or "").strip()
        if not image_path:
            continue
        title = _norm_text(item.get("title"))
        text = _norm_text("\n".join(
            part for part in (
                str(item.get("t1", "") or ""),
                str(item.get("t1_structure", "") or ""),
            )
            if part
        ))
        candidates.append({
            "title": title,
            "text": text,
            "image_path": image_path,
        })

    if not candidates:
        return slides

    enriched: list[dict] = []
    for slide in slides:
        if not isinstance(slide, dict):
            enriched.append(slide)
            continue
        item = dict(slide)
        if item.get("image_path"):
            enriched.append(item)
            continue

        title = _norm_text(item.get("title") or item.get("slide_title"))
        slide_text = _norm_text(item.get("slide_text") or item.get("text"))
        matches = [c for c in candidates if c["title"] == title]
        if not matches:
            enriched.append(item)
            continue

        def score(candidate: dict) -> tuple[int, int]:
            candidate_text = candidate["text"]
            overlap = 0
            if candidate_text and (candidate_text in slide_text or slide_text in candidate_text):
                overlap = max(len(candidate_text), len(slide_text))
            else:
                slide_tokens = set(slide_text.split())
                candidate_tokens = set(candidate_text.split())
                overlap = len(slide_tokens & candidate_tokens)
            return (overlap, len(candidate_text))

        best = max(matches, key=score)
        item["image_path"] = best["image_path"]
        enriched.append(item)

    return enriched


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
1. 확신이 0.80 미만이면 출력하지 마세요.
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


def _check_single_slide(slide: dict, img_dir: Optional[str]) -> tuple[list[dict], bool, int, dict]:
    from . import claim_common as cc

    slide_no = int(slide.get("slide_number", 0) or 0)
    title = str(slide.get("title", "") or "")
    slide_text = str(slide.get("slide_text", "") or "")

    img_bytes = None
    img_path = _existing_path(slide.get("image_path"))
    if not img_path and img_dir:
        img_path = _find_slide_image(img_dir, slide_no)
    if img_path and img_path.exists():
        img_bytes = img_path.read_bytes()

    prompt = _build_slide_typo_prompt(slide_no, title, slide_text)
    model = str(cc._resolve_stage_model("slide_typo") or "").strip()
    response_format = {"type": "json_object"} if cc._supports_json_object_response_format(model) else None

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
            stage="slide_typo",
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
                })
            return cleaned, False, api_calls, token_usage
        except Exception:
            if attempt < cc.VERIFIER_PARSE_RETRIES:
                print(f"    ↺ 슬라이드 {slide_no} 오타 JSON 파싱 재시도 ({attempt+1}/{cc.VERIFIER_PARSE_RETRIES})")
    return [], True, api_calls, token_usage


def detect_slide_typos(
    slides: list[dict],
    img_dir: Optional[str] = None,
    max_workers: int = 4,
    merged_path: str | Path | None = None,
) -> tuple[list[dict], int, int, dict]:
    from . import claim_common as cc

    if not slides:
        return [], 0, 0, cc._empty_token_usage()

    slides = attach_slide_image_paths(slides, merged_path)

    results: list[dict] = []
    api_calls = 0
    failures = 0
    token_usage = cc._empty_token_usage()

    def process(slide: dict):
        sn = slide.get("slide_number", "?")
        print(f"    슬라이드 오타 검사 [{sn}]")
        return _check_single_slide(slide, img_dir)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(process, slide): slide for slide in slides}
        for future in as_completed(futures):
            slide = futures[future]
            try:
                typos, parse_failed, calls, usage = future.result()
                results.extend(typos)
                api_calls += calls
                token_usage = cc._merge_token_usage(token_usage, usage)
                if parse_failed:
                    failures += 1
            except Exception:
                failures += 1

    dedup = {}
    for typo in results:
        key = (
            typo.get("slide_number"),
            typo.get("problematic_text", "").strip().lower(),
            typo.get("corrected_text", "").strip().lower(),
        )
        prev = dedup.get(key)
        if prev is None or float(typo.get("confidence", 0) or 0) > float(prev.get("confidence", 0) or 0):
            dedup[key] = typo

    final = sorted(
        dedup.values(),
        key=lambda x: (int(x.get("slide_number", 0) or 0), -float(x.get("confidence", 0) or 0)),
    )
    return final, api_calls, failures, token_usage
