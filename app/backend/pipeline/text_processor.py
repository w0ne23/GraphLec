"""
텍스트 교정 엔진.

Pass 1: 슬라이드 제목 + 용어 사전만으로 ASR 오인식 교정
Pass 2: 슬라이드 전체 컨텍스트로 교정 → Pass 1과 merge하여 안전하게 적용
"""

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types

from .config import GEMINI_GENERATIVE_MODEL, gemini_client_2

load_dotenv()

GEMINI_MODEL = GEMINI_GENERATIVE_MODEL

BATCH_SIZE = int(os.getenv("MERGE_CORRECTION_BATCH_SIZE", "50"))
TRANSITION_LEAD_SEC = float(os.getenv("MERGE_TRANSITION_LEAD_SEC", "1.0"))
TRANSITION_TAIL_SEC = float(os.getenv("MERGE_TRANSITION_TAIL_SEC", "0.2"))
ASSIGN_MAX_GAP_SEC = float(os.getenv("MERGE_ASSIGN_MAX_GAP_SEC", "3.0"))

DOMAIN_CHOICES = ["공학", "자연과학", "인문학", "사회과학", "예술"]
SUBDOMAIN_CHOICES = [
    "컴퓨터공학", "전자공학", "기계공학",
    "물리학", "화학", "생물학",
    "역사학", "철학", "문학",
    "경제학", "정치학", "사회학",
]

_token_usage: dict[str, int] = {"input": 0, "output": 0, "calls": 0}
_override_client: Optional[genai.Client] = gemini_client_2

_KO_PARTICLES = re.compile(
    r"(은|는|이|가|을|를|에|에서|의|와|과|로|으로|도|만|부터|까지|에게|한테|께|처럼|보다|마다|이나|나|라|이라|든지|이든지|라고|이라고|라는|이라는)$"
)


def set_client(client: genai.Client) -> None:
    global _override_client
    _override_client = client


def _get_client() -> genai.Client:
    if _override_client is None:
        raise RuntimeError("Gemini client가 설정되지 않았습니다.")
    return _override_client


def _add_usage(response) -> None:
    usage = getattr(response, "usage_metadata", None)
    if usage:
        _token_usage["input"] += getattr(usage, "prompt_token_count", 0) or 0
        _token_usage["output"] += getattr(usage, "candidates_token_count", 0) or 0
    _token_usage["calls"] += 1


def api_call_with_retry(func, max_retries: int = 5, initial_wait: int = 10):
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as exc:
            err = str(exc)
            retryable = ["429", "503", "500", "RESOURCE_EXHAUSTED", "UNAVAILABLE", "overloaded"]
            if any(code in err for code in retryable) and attempt < max_retries - 1:
                wait = initial_wait * (attempt + 1)
                print(f"  재시도 ({attempt + 1}): {err[:60]}, {wait}초 대기")
                time.sleep(wait)
                continue
            raise


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _is_spacing_only(orig: str, corrected: str) -> bool:
    return orig.replace(" ", "") == corrected.replace(" ", "")


def _strip_particle(token: str) -> str:
    return _KO_PARTICLES.sub("", token)


def _is_english_term_fix(orig: str, corrected: str) -> bool:
    orig_no_space = re.sub(r"[A-Za-z0-9_.()]+", "", orig).replace(" ", "")
    corr_no_space = re.sub(r"[A-Za-z0-9_.()]+", "", corrected).replace(" ", "")
    if orig_no_space != corr_no_space:
        return False
    orig_eng = set(re.findall(r"[A-Za-z_][A-Za-z0-9_.]*", orig))
    corr_eng = set(re.findall(r"[A-Za-z_][A-Za-z0-9_.]*", corrected))
    return orig_eng != corr_eng


def _is_variable_or_abbrev_fix(orig: str, corrected: str) -> bool:
    orig_tokens = set(orig.split())
    corr_tokens = set(corrected.split())
    new_tokens = corr_tokens - orig_tokens
    if not new_tokens:
        return False
    code_pattern = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*$")
    for token in new_tokens:
        stripped = _strip_particle(token)
        if stripped and code_pattern.match(stripped):
            return True
    return False


def classify_lecture_domain(slide_titles: list[str], transcript_sample: str) -> dict:
    titles_block = "\n".join(f"- {title}" for title in slide_titles[:10] if title)
    transcript_block = transcript_sample[:1500]

    prompt = f"""아래 강의 슬라이드 제목과 전사 내용을 보고 도메인과 서브도메인을 분류하세요.

## 슬라이드 제목
{titles_block}

## 전사 내용 (일부)
{transcript_block}

## 도메인 선택지
{", ".join(DOMAIN_CHOICES)}

## 서브도메인 선택지
{", ".join(SUBDOMAIN_CHOICES)}

위 선택지에서 가장 적합한 것을 하나씩 골라 JSON만 출력하세요.
선택지에 없으면 가장 가까운 것을 고르세요.
{{"domain": "...", "subdomain": "..."}}"""

    def call():
        return _get_client().models.generate_content(
            model=GEMINI_MODEL,
            contents=[types.Part.from_text(text=prompt)],
            config=types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=256,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )

    try:
        response = api_call_with_retry(call)
        _add_usage(response)
        raw = (response.text or "").strip()
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0].strip()
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0].strip()
        parsed = json.loads(raw)
        domain = parsed.get("domain", "")
        subdomain = parsed.get("subdomain", "")
        if domain not in DOMAIN_CHOICES:
            domain = ""
        if subdomain not in SUBDOMAIN_CHOICES:
            subdomain = ""
        return {"domain": domain, "subdomain": subdomain}
    except Exception as exc:
        print(f"  [도메인 분류 오류] {exc}")
        return {"domain": "", "subdomain": ""}


def _build_occurrence_index(slide_occurrences: dict[int, list[dict]]) -> list[dict]:
    occ_index = []
    for slide_no, occs in slide_occurrences.items():
        for occ in occs:
            occ_index.append(
                {
                    "slide_no": int(slide_no),
                    "start_sec": float(occ["start_sec"]),
                    "end_sec": float(occ["end_sec"]),
                }
            )
    occ_index.sort(key=lambda item: (item["start_sec"], item["end_sec"], item["slide_no"]))
    return occ_index


def _assign_segment_occurrence(seg: dict, occ_index: list[dict]) -> Optional[int]:
    if not occ_index:
        return None

    seg_start = float(seg.get("start", 0) or 0)
    seg_end = float(seg.get("end", seg_start) or seg_start)
    if seg_end < seg_start:
        seg_end = seg_start

    overlaps: list[tuple[int, float]] = []
    for idx, occ in enumerate(occ_index):
        overlap = min(seg_end, occ["end_sec"]) - max(seg_start, occ["start_sec"])
        if overlap > 0:
            overlaps.append((idx, overlap))

    if overlaps:
        overlap_map = {idx: overlap for idx, overlap in overlaps}
        overlaps.sort(key=lambda item: (item[1], occ_index[item[0]]["start_sec"]), reverse=True)
        best_idx, best_overlap = overlaps[0]

        later_candidates = [
            idx
            for idx, _ in overlaps
            if occ_index[idx]["start_sec"] > seg_start and occ_index[idx]["start_sec"] <= seg_end
        ]
        if later_candidates:
            later_idx = min(later_candidates, key=lambda item: occ_index[item]["start_sec"])
            later_start = occ_index[later_idx]["start_sec"]
            later_overlap = overlap_map.get(later_idx, 0.0)
            started_near_boundary = seg_start >= (later_start - TRANSITION_LEAD_SEC)
            carried_after_boundary = (seg_end - later_start) >= TRANSITION_TAIL_SEC
            if started_near_boundary or later_overlap >= (best_overlap * 0.75) or carried_after_boundary:
                return later_idx

        return best_idx

    anchor = seg_start + (seg_end - seg_start) * 0.65
    best_idx = None
    best_dist = float("inf")
    for idx, occ in enumerate(occ_index):
        if occ["start_sec"] <= anchor <= occ["end_sec"]:
            return idx
        dist = min(abs(anchor - occ["start_sec"]), abs(anchor - occ["end_sec"]))
        if dist < best_dist:
            best_dist = dist
            best_idx = idx
    if best_idx is None or best_dist > ASSIGN_MAX_GAP_SEC:
        return None
    return best_idx


def parse_batch_response(text: str) -> dict[int, dict]:
    if not text:
        return {}
    raw = text.strip()
    if "```json" in raw:
        raw = raw.split("```json")[1].split("```")[0].strip()
    elif "```" in raw:
        raw = raw.split("```")[1].split("```")[0].strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    if not isinstance(parsed, dict):
        return {}

    corrections: dict[int, dict] = {}
    for correction in parsed.get("corrections", []):
        if not isinstance(correction, dict):
            continue
        idx = correction.get("index")
        text_value = correction.get("text", "")
        decision = str(correction.get("decision", "") or "").strip().lower()
        risk = str(correction.get("risk", "") or "").strip().lower()
        reason = str(correction.get("reason", "") or "").strip()
        if isinstance(idx, int) and text_value:
            corrections[idx] = {
                "text": text_value,
                "decision": decision,
                "risk": risk,
                "reason": reason,
            }
    return corrections


def extract_glossary_terms(slide_texts: dict[int, dict]) -> list[str]:
    all_text = ""
    for _, extracted in slide_texts.items():
        all_text += f" {extracted.get('title', '')} {extracted.get('text', '')}"
    code_terms = set(re.findall(r"\b[A-Za-z_][A-Za-z0-9_.()]*\b", all_text))
    func_terms = set(re.findall(r"\b[A-Za-z_]\w*\s*\(", all_text))
    func_terms = {term.strip().rstrip("(").strip() for term in func_terms}
    return sorted({term for term in (code_terms | func_terms) if len(term) >= 2})


def _load_slide_occurrences_from_metadata(metadata: list[dict]) -> tuple[dict[int, list[dict]], dict[int, int]]:
    slide_occurrences: dict[int, list[dict]] = {}
    slide_det_no: dict[int, int] = {}
    seen: set[int] = set()

    for entry in metadata:
        slide_no = entry.get("scene_index", entry.get("slide_index"))
        start_sec = entry.get("slide_start_sec")
        end_sec = entry.get("slide_end_sec")
        if not isinstance(slide_no, int) or slide_no in seen:
            continue
        if start_sec is None or end_sec is None:
            continue
        start = float(start_sec)
        end = float(end_sec)
        if end < start:
            end = start
        slide_occurrences[slide_no] = [{
            "start_sec": start,
            "end_sec": end,
            "duration": round(end - start, 3),
            "is_dup": False,
        }]
        slide_det_no[slide_no] = slide_no
        seen.add(slide_no)

    return slide_occurrences, slide_det_no


def _build_scene_metadata_index(metadata: list[dict]) -> dict[int, dict]:
    scene_meta: dict[int, dict] = {}
    for entry in metadata:
        if entry.get("capture_type") != "base" and int(entry.get("annot_index", 0) or 0) != 0:
            continue
        scene_idx = entry.get("scene_index", entry.get("slide_index"))
        if not isinstance(scene_idx, int) or scene_idx in scene_meta:
            continue
        slide_number = entry.get("slide_number", entry.get("slide_canonical_index"))
        slide_canonical_index = entry.get(
            "slide_canonical_index",
            entry.get("same_slide_canonical"),
        )
        scene_meta[scene_idx] = {
            "scene_index": scene_idx,
            "slide_number": slide_number if isinstance(slide_number, int) else None,
            "slide_canonical_index": (
                slide_canonical_index if isinstance(slide_canonical_index, int) else None
            ),
            "slide_visit_order": int(entry.get("slide_visit_order", entry.get("same_slide_visit_order", 1)) or 1),
            "slide_is_revisit": bool(
                entry.get("slide_is_revisit", entry.get("same_slide_is_revisit", False))
            ),
        }
    return scene_meta


def _load_integrated_slide_texts(
    integrated_data: dict,
    base_dir: Path,
    scene_meta_by_index: Optional[dict[int, dict]] = None,
) -> dict[int, dict]:
    result: dict[int, dict] = {}
    for slide in integrated_data.get("slides", []):
        scene_no = slide.get("scene_number", slide.get("slide_number"))
        slide_no = slide.get("slide_number")
        if scene_meta_by_index and isinstance(scene_no, int):
            logical_slide_no = scene_meta_by_index.get(scene_no, {}).get("slide_number")
        else:
            logical_slide_no = slide_no
        if not isinstance(logical_slide_no, int):
            continue
        raw_image_path = str(slide.get("image_path", "") or "")
        image_path = ""
        if raw_image_path:
            candidate = Path(raw_image_path)
            if not candidate.is_absolute():
                candidate = (base_dir / candidate).resolve()
            image_path = str(candidate)
        text_parts = []
        if slide.get("t1"):
            text_parts.append(str(slide.get("t1")))
        if slide.get("t1_structure"):
            text_parts.append(str(slide.get("t1_structure")))
        current = result.get(logical_slide_no)
        candidate_entry = {
            "title": str(slide.get("title", "") or ""),
            "text": "\n".join(part for part in text_parts if part),
            "image_path": image_path,
            "scene_number": scene_no if isinstance(scene_no, int) else None,
        }
        if current is None:
            result[logical_slide_no] = candidate_entry
            continue
        current_len = len(current.get("text", "")) + len(current.get("title", ""))
        candidate_len = len(candidate_entry.get("text", "")) + len(candidate_entry.get("title", ""))
        if candidate_len > current_len:
            result[logical_slide_no] = candidate_entry
    return result


def _correct_batch_pass1(
    batch: list[tuple[int, dict]],
    slide_title: str = "",
    glossary: str = "",
) -> dict[int, str]:
    if not batch:
        return {}

    seg_text = "\n".join(
        f"[{local_i}] {seg.get('text_original', seg['text'])}"
        for local_i, (_, seg) in enumerate(batch)
    )
    topic_hint = f"\n## 현재 구간 주제\n{slide_title}\n" if slide_title.strip() else ""
    glossary_block = f"\n{glossary}\n" if glossary.strip() else ""

    prompt = f"""강의 음성 전사본을 문맥만 보고 교정하세요. 슬라이드 원본은 제공되지 않습니다.
{topic_hint}{glossary_block}
## 전사 (교정 대상)
{seg_text}

## 출력 (JSON만)
{{"corrections": [{{"index": 0, "text": "교정된 텍스트"}}, ...]}}

### 교정 범위
- ASR 오인식 교정 (깨진 텍스트를 문맥에 맞는 단어로 복원)
- 맞춤법, 띄어쓰기, 조사 오류 교정
- 불필요한 추임새(자, 뭐, 어, 그) 제거 (의미가 유지될 때만)
- 코드에 실제로 등장하는 영문 토큰, 함수명, 클래스명, 변수명, 라이브러리명이 한국어식 발음 또는 ASR 오인식으로 들어온 경우에만 glossary를 참고하여 해당 영문 표기로 복원
- 강의자가 실제로 한국어 일반 용어로 말한 경우에는 glossary에 대응되는 영문 용어가 있더라도 영어로 번역하지 말 것
- 발화가 외래어/영문 토큰 자체를 읽는 맥락이면 영문 표기를 사용할 수 있다
- 발화가 한국어 설명 문장이라면 한국어 표현을 유지할 것
- ASR이 영문 용어를 잘못 인식한 경우 glossary를 참고하여 교정

### 핵심 원칙
- 강의자의 발화 구조와 의미를 절대적으로 보존하라
- 문장 길이와 정보량은 원문과 거의 똑같게 유지
- 문맥상 말이 안 되는 단어(ASR 깨짐)만 복원. 의미가 통하는 단어는 그대로 둘 것
- 강의자의 단어 선택에 있어, 문맥적으로 맞는데, 오탈자가 있다면, 문맥에만 맞게 바꾸어주라(해당 내용이 틀리던 말던, 문맥에는 맞으면 됨)
- 절대 강의자가 주어와 서술어를 반대되는 개념으로 설명하여도, 바꾸지 말아라. 강의자가 잘못된 내용을 말한 것이다.
- 각 index의 원문만 수정. 다른 index 내용과 섞지 말 것"""

    def call():
        return _get_client().models.generate_content(
            model=GEMINI_MODEL,
            contents=[types.Part.from_text(text=prompt)],
            config=types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=8192,
                thinking_config=types.ThinkingConfig(thinking_budget=1024),
            ),
        )

    try:
        response = api_call_with_retry(call)
        _add_usage(response)
        local_corrections = parse_batch_response(response.text or "")
    except Exception as exc:
        print(f"  [Pass1 오류 무시] {exc}")
        return {}

    result: dict[int, str] = {}
    for local_i, corr_payload in local_corrections.items():
        if 0 <= local_i < len(batch):
            global_i = batch[local_i][0]
            original = batch[local_i][1].get("text_original", batch[local_i][1]["text"])
            cleaned = _normalize_text(str(corr_payload.get("text", "") or ""))
            if cleaned and cleaned != _normalize_text(original):
                result[global_i] = cleaned
    return result


def _correct_batch_pass2(
    batch: list[tuple[int, dict]],
    slide_context: str,
    slide_image_path: Optional[str] = None,
) -> dict[int, str]:
    if not batch:
        return {}

    seg_text = "\n".join(
        f"[{local_i}] {seg.get('text_original', seg['text'])}"
        for local_i, (_, seg) in enumerate(batch)
    )
    has_image = bool(slide_image_path and Path(slide_image_path).exists())
    if has_image:
        ref_block = "\n## 강의 슬라이드 이미지 (첨부됨)\n이미지에 보이는 용어, 수식, 다이어그램을 참고하여 전사를 교정하세요.\n"
        if slide_context.strip():
            ref_block += f"\n## 강의자료 텍스트 (추가 참조)\n{slide_context[:1500]}\n"
    else:
        ref_block = f"\n## 강의자료 (용어 참조)\n{slide_context[:2000]}\n" if slide_context.strip() else ""

    prompt = f"""강의 음성 전사본을 슬라이드와 문맥을 참고하여 교정하세요.
{ref_block}
## 전사 (교정 대상)
{seg_text}

## 출력 (JSON만)
{{"corrections": [{{"index": 0, "text": "교정된 텍스트"}}, ...]}}

### 교정 범위
- ASR 오인식 교정 (깨진 텍스트를 문맥에 맞는 단어로 복원)
- 전문용어 철자 교정 (슬라이드를 참고하여 정확한 표기로)
- 맞춤법, 띄어쓰기, 조사 오류 교정
- 불필요한 추임새(자, 뭐, 어, 그) 제거 (의미가 유지될 때만)

### 핵심 원칙 — 강의자의 실제 발화 의미를 보존하라
- 전사 원문의 의미가 기준이다. 슬라이드는 용어 철자 확인용 참고 자료일 뿐이다
- 문장 길이와 정보량은 원문과 거의 똑같게 유지
- 요약, 재서술, 슬라이드 bullet 복사 금지
- 강의자의 발화 중 오인식된 단어가 있다면 해당 단어에 대해서만 교체하는 수준이다
- 슬라이드와 발화가 완전히 다르다면, 발화를 따르도록 할 것.
- 전사본을 따라갔을 때, 해당 강의 자체의 문맥에 맞지 않을 때는 반드시 문맥에 맞는 단어로 바꾸어야 한다.
- 각 index의 원문만 수정. 다른 index 내용과 섞지 말 것

### 중요 원칙
- 강의자의 발화 구조를 절대적으로 따라가라
- 강의자의 단어 선택에 있어, 문맥적으로 맞는데, 오탈자가 있다면, 문맥에만 맞게 바꾸어주라(해당 내용이 틀리던 말던, 문맥에는 맞으면 됨)"""

    contents = []
    if has_image:
        with open(slide_image_path, "rb") as f:
            img_bytes = f.read()
        contents.append(types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"))
    contents.append(types.Part.from_text(text=prompt))

    def call():
        return _get_client().models.generate_content(
            model=GEMINI_MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=8192,
                thinking_config=types.ThinkingConfig(thinking_budget=1024),
            ),
        )

    try:
        response = api_call_with_retry(call)
        _add_usage(response)
        local_corrections = parse_batch_response(response.text or "")
    except Exception as exc:
        print(f"  [Pass2 오류 무시] {exc}")
        return {}

    result: dict[int, str] = {}
    for local_i, corr_payload in local_corrections.items():
        if 0 <= local_i < len(batch):
            global_i = batch[local_i][0]
            original = batch[local_i][1].get("text_original", batch[local_i][1]["text"])
            cleaned = _normalize_text(str(corr_payload.get("text", "") or ""))
            if cleaned and cleaned != _normalize_text(original):
                result[global_i] = cleaned
    return result


def merge_two_passes(
    batch: list[tuple[int, dict]],
    pass1: dict[int, str],
    pass2: dict[int, str],
    subdomain: str = "",
) -> dict[int, dict]:
    is_cs = subdomain in ("컴퓨터공학", "소프트웨어공학", "정보통신", "전산학")
    corrections: dict[int, dict] = {}
    all_indices = set(pass1.keys()) | set(pass2.keys())
    orig_map = {global_i: seg.get("text_original", seg["text"]) for global_i, seg in batch}

    for global_i in all_indices:
        p1 = pass1.get(global_i)
        p2 = pass2.get(global_i)
        original = orig_map.get(global_i, "")

        if p1 and p2:
            if _normalize_text(p1) == _normalize_text(p2):
                corrections[global_i] = {
                    "candidate_text": p1,
                    "applied_text": p1,
                    "risk": "low",
                    "apply": True,
                    "reason": "pass1+pass2 일치",
                }
            else:
                corrections[global_i] = {
                    "candidate_text": p2,
                    "applied_text": p1,
                    "risk": "medium",
                    "apply": True,
                    "reason": "pass1 채택 (pass2 상이)",
                }
        elif p1 and not p2:
            corrections[global_i] = {
                "candidate_text": p1,
                "applied_text": p1,
                "risk": "low",
                "apply": True,
                "reason": "pass1만 교정 (문맥 기반)",
            }
        elif p2 and not p1:
            if _is_spacing_only(original, p2):
                corrections[global_i] = {
                    "candidate_text": p2,
                    "applied_text": p2,
                    "risk": "low",
                    "apply": True,
                    "reason": "pass2 띄어쓰기 교정만 (안전)",
                }
            elif is_cs and _is_variable_or_abbrev_fix(original, p2):
                corrections[global_i] = {
                    "candidate_text": p2,
                    "applied_text": p2,
                    "risk": "low",
                    "apply": True,
                    "reason": "pass2 변수명/약어 표기 반영 (CS)",
                }
            elif _is_english_term_fix(original, p2):
                corrections[global_i] = {
                    "candidate_text": p2,
                    "applied_text": p2,
                    "risk": "low",
                    "apply": True,
                    "reason": "pass2 영문 용어 교정 (ASR 오인식)",
                }
            else:
                corrections[global_i] = {
                    "candidate_text": p2,
                    "applied_text": "",
                    "risk": "high",
                    "apply": False,
                    "reason": "pass2만 교정 (슬라이드 영향 가능성)",
                }
    return corrections


def correct_segments_two_pass(
    segments: list[dict],
    metadata: list[dict],
    textualized_data: dict,
    textualized_dir: Path,
    use_pass2: bool = True,
) -> list[dict]:
    if not segments:
        return []

    slide_occurrences, _ = _load_slide_occurrences_from_metadata(metadata)
    scene_meta_by_index = _build_scene_metadata_index(metadata)
    extracted_slide_texts = _load_integrated_slide_texts(
        textualized_data,
        textualized_dir,
        scene_meta_by_index=scene_meta_by_index,
    )

    occ_index = _build_occurrence_index(slide_occurrences)
    seg_scene: dict[int, int] = {}
    seg_logical_slide: dict[int, int] = {}
    for i, seg in enumerate(segments):
        scene_no = seg.get("scene_index", seg.get("slide_index"))
        if isinstance(scene_no, int):
            seg_scene[i] = scene_no
            logical_slide_no = scene_meta_by_index.get(scene_no, {}).get("slide_number")
            if isinstance(logical_slide_no, int):
                seg_logical_slide[i] = logical_slide_no
            continue
        occ_idx = _assign_segment_occurrence(seg, occ_index)
        if occ_idx is not None:
            scene_no = occ_index[occ_idx]["slide_no"]
            seg_scene[i] = scene_no
            logical_slide_no = scene_meta_by_index.get(scene_no, {}).get("slide_number")
            if isinstance(logical_slide_no, int):
                seg_logical_slide[i] = logical_slide_no

    groups: dict[int, list[tuple[int, dict]]] = {}
    group_scene_indices: dict[int, set[int]] = {}
    no_slide: list[tuple[int, dict]] = []
    for i, seg in enumerate(segments):
        if i in seg_logical_slide:
            logical_slide_no = seg_logical_slide[i]
            groups.setdefault(logical_slide_no, []).append((i, seg))
            if i in seg_scene:
                group_scene_indices.setdefault(logical_slide_no, set()).add(seg_scene[i])
        else:
            no_slide.append((i, seg))

    slide_titles = [extracted_slide_texts.get(sno, {}).get("title", "") for sno in sorted(extracted_slide_texts.keys())]
    transcript_sample = " ".join(seg.get("text", "") for seg in segments[:30])
    domain_info = classify_lecture_domain(slide_titles, transcript_sample)
    subdomain = domain_info.get("subdomain", "")
    print(f"    도메인: {domain_info['domain']}, 서브도메인: {subdomain}")

    glossary_terms = extract_glossary_terms(extracted_slide_texts)
    glossary = "## 슬라이드 용어 사전 (표기 참조용)\n" + ", ".join(glossary_terms) if glossary_terms else ""
    if glossary:
        print(f"    용어 사전: {len(glossary_terms)}개 용어")

    all_corrections: dict[int, dict] = {}
    for logical_slide_no in sorted(extracted_slide_texts.keys()):
        group = groups.get(logical_slide_no, [])
        if not group:
            continue
        extracted = extracted_slide_texts.get(logical_slide_no, {})
        context = f"슬라이드 제목: {extracted.get('title', '')}\n{extracted.get('text', '')}"
        slide_title = extracted.get("title", "")
        scene_list = sorted(group_scene_indices.get(logical_slide_no, set()))
        if len(scene_list) <= 3:
            scene_label = ",".join(f"{scene_idx}" for scene_idx in scene_list)
        else:
            scene_label = f"{scene_list[0]},{scene_list[1]},...,{scene_list[-1]}"
        print(
            f"    slide {logical_slide_no:3d} (scenes {scene_label:>7s}, {slide_title[:22]:22s}): {len(group):3d}개",
            end="",
            flush=True,
        )

        sub_batches = [group[b:b + BATCH_SIZE] for b in range(0, len(group), BATCH_SIZE)]
        for sub in sub_batches:
            if use_pass2:
                with ThreadPoolExecutor(max_workers=2) as pool:
                    future_pass1 = pool.submit(_correct_batch_pass1, sub, slide_title=slide_title, glossary=glossary)
                    future_pass2 = pool.submit(_correct_batch_pass2, sub, context)
                    pass1 = future_pass1.result()
                    print("①", end="", flush=True)
                    pass2 = future_pass2.result()
                    print("②", end="", flush=True)
                merged = merge_two_passes(sub, pass1, pass2, subdomain=subdomain)
            else:
                pass1 = _correct_batch_pass1(sub, slide_title=slide_title, glossary=glossary)
                print("①", end="", flush=True)
                merged = {
                    global_i: {
                        "candidate_text": text_value,
                        "applied_text": text_value,
                        "risk": "low",
                        "apply": True,
                        "reason": "pass1 only",
                    }
                    for global_i, text_value in pass1.items()
                }
            all_corrections.update(merged)
        print()

    if no_slide:
        print(f"    미매핑 {len(no_slide)}개 교정 중...", end="", flush=True)
        for offset in range(0, len(no_slide), BATCH_SIZE):
            sub = no_slide[offset:offset + BATCH_SIZE]
            pass1 = _correct_batch_pass1(sub, glossary=glossary)
            all_corrections.update({
                global_i: {
                    "candidate_text": text_value,
                    "applied_text": text_value,
                    "risk": "low",
                    "apply": True,
                    "reason": "pass1 only (미매핑)",
                }
                for global_i, text_value in pass1.items()
            })
            print(".", end="", flush=True)
        print()

    corrected_segments: list[dict] = []
    applied_count = 0
    for i, seg in enumerate(segments):
        corrected = seg.copy()
        scene_idx = seg_scene.get(i)
        if isinstance(scene_idx, int):
            corrected["slide_index"] = scene_idx  # 하위 호환: 기존 pipeline은 scene을 slide_index로 본다.
            corrected["scene_index"] = scene_idx
            scene_meta = scene_meta_by_index.get(scene_idx, {})
            logical_slide_no = scene_meta.get("slide_number")
            if isinstance(logical_slide_no, int):
                corrected["slide_number"] = logical_slide_no
            slide_canonical_index = scene_meta.get("slide_canonical_index")
            if isinstance(slide_canonical_index, int):
                corrected["slide_canonical_index"] = slide_canonical_index
            corrected["slide_visit_order"] = int(scene_meta.get("slide_visit_order", 1) or 1)
            corrected["slide_is_revisit"] = bool(scene_meta.get("slide_is_revisit", False))
        original = seg.get("text_original", seg["text"])
        corrected["text_raw"] = original
        corrected["text_corrected"] = original
        corrected["text_original"] = original

        corr_info = all_corrections.get(i)
        if corr_info:
            corrected["correction_risk"] = str(corr_info.get("risk", "none") or "none")
            corrected["correction_reason"] = str(corr_info.get("reason", "") or "")
            candidate_text = str(corr_info.get("candidate_text", "") or "").strip()
            if candidate_text:
                corrected["text_corrected_candidate"] = candidate_text

            if corr_info.get("apply") and corr_info.get("applied_text"):
                corrected_text = str(corr_info["applied_text"])
                corrected["text"] = corrected_text
                corrected["text_corrected"] = corrected_text
                corrected["correction_status"] = "applied"
                applied_count += 1
            else:
                corrected["text"] = original
                corrected["correction_status"] = "candidate_only"
        else:
            corrected["text"] = original
            corrected["correction_status"] = "unchanged"
            corrected["correction_risk"] = "none"
            corrected["correction_reason"] = ""

        corrected_segments.append(corrected)

    print(f"    교정 적용: {applied_count}건 / 전체 {len(segments)}건")
    return corrected_segments
