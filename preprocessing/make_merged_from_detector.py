#!/usr/bin/env python3
"""
slide_detector 결과 + (PPTX) + 전사본 → 교정 + merged JSON 생성

1. (PPTX 있을 때) 슬라이드 이미지 → Gemini로 PPTX 번호 식별 (캐시)
   (PPTX 없을 때) detected slide_no 직접 사용
2. 타임라인 기반 슬라이드별 세그먼트 배분
3. 슬라이드 컨텍스트로 전사 교정 (Gemini, LLM이 직접 apply 판정)
4. 교정본으로 merged JSON + transcribed.json 저장
"""

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from google import genai
from google.genai import types

from config import get_gemini_client, get_gemini_client_count

load_dotenv()

GEMINI_MODEL   = "gemini-2.5-flash"

BATCH_SIZE = int(os.getenv("MERGE_CORRECTION_BATCH_SIZE", "50"))
TRANSITION_LEAD_SEC = float(os.getenv("MERGE_TRANSITION_LEAD_SEC", "1.0"))
TRANSITION_TAIL_SEC = float(os.getenv("MERGE_TRANSITION_TAIL_SEC", "0.2"))
ASSIGN_MAX_GAP_SEC = float(os.getenv("MERGE_ASSIGN_MAX_GAP_SEC", "3.0"))


_token_usage: dict[str, int] = {"input": 0, "output": 0, "calls": 0}


def _get_client() -> genai.Client:
    return get_gemini_client()


def _add_usage(response) -> None:
    u = getattr(response, "usage_metadata", None)
    if u:
        _token_usage["input"]  += getattr(u, "prompt_token_count", 0) or 0
        _token_usage["output"] += getattr(u, "candidates_token_count", 0) or 0
    _token_usage["calls"] += 1


def api_call_with_retry(func, max_retries=5, initial_wait=10):
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            err = str(e)
            if any(c in err for c in ["429", "503", "500", "RESOURCE_EXHAUSTED", "UNAVAILABLE", "overloaded"]) and attempt < max_retries - 1:
                if get_gemini_client_count() > 1 and any(c in err for c in ["503", "UNAVAILABLE"]):
                    print(f"  Gemini 503 감지 → 다음 API 키로 즉시 전환 ({attempt+1})")
                    continue
                wait = initial_wait * (attempt + 1)
                print(f"  재시도 ({attempt+1}): {err[:60]}, {wait}초 대기")
                time.sleep(wait)
            else:
                raise


def fmt_ts(sec: float) -> str:
    m, s = int(sec) // 60, int(sec) % 60
    return f"{m:02d}:{s:02d}"


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())



def _is_spacing_only(orig: str, corrected: str) -> bool:
    """공백/띄어쓰기만 다른지 확인."""
    return orig.replace(" ", "") == corrected.replace(" ", "")


_KO_PARTICLES = re.compile(
    r'(은|는|이|가|을|를|에|에서|의|와|과|로|으로|도|만|부터|까지|에게|한테|께|처럼|보다|마다|이나|나|라|이라|든지|이든지|라고|이라고|라는|이라는)$'
)


def _strip_particle(token: str) -> str:
    """한국어 조사 제거: 'regr에' → 'regr', 'data를' → 'data'"""
    return _KO_PARTICLES.sub('', token)




def _is_english_term_fix(orig: str, corrected: str) -> bool:
    """
    영문 토큰이 다른 영문 토큰으로만 바뀐 교정인지 확인.
    예: 'PIT 함수'→'fit 함수', 'Regression에'→'regression에'
    한글 부분은 동일하고 영문 부분만 다를 때 True.
    """
    orig_no_space = re.sub(r'[A-Za-z0-9_.()]+', '', orig).replace(" ", "")
    corr_no_space = re.sub(r'[A-Za-z0-9_.()]+', '', corrected).replace(" ", "")
    if orig_no_space != corr_no_space:
        return False  # 한글 부분이 다름 → 영문만 바꾼 게 아님
    # 영문 부분이 실제로 달라야 함
    orig_eng = set(re.findall(r'[A-Za-z_][A-Za-z0-9_.]*', orig))
    corr_eng = set(re.findall(r'[A-Za-z_][A-Za-z0-9_.]*', corrected))
    return orig_eng != corr_eng


def _is_variable_or_abbrev_fix(orig: str, corrected: str) -> bool:
    """
    슬라이드의 약어/변수명 표기를 반영한 교정인지 확인.
    예: 'x 데이터'→'X_data', 'regr에'→변수명 regr 등
    조사가 붙은 경우도 분리해서 체크.
    """
    orig_tokens = set(orig.split())
    corr_tokens = set(corrected.split())
    new_tokens = corr_tokens - orig_tokens
    if not new_tokens:
        return False
    code_pattern = re.compile(r'^[A-Za-z_][A-Za-z0-9_.]*$')
    for t in new_tokens:
        stripped = _strip_particle(t)
        if stripped and code_pattern.match(stripped):
            return True
    return False


# ── 도메인 분류 ──────────────────────────────────────────────────────

DOMAIN_CHOICES = [
    "공학", "자연과학", "인문학", "사회과학", "예술",
]
SUBDOMAIN_CHOICES = [
    "컴퓨터공학", "전자공학", "기계공학",
    "물리학", "화학", "생물학",
    "역사학", "철학", "문학",
    "경제학", "정치학", "사회학",
]


def classify_lecture_domain(slide_titles: list[str], transcript_sample: str) -> dict:
    """슬라이드 제목 + 전사 샘플로 강의 도메인/서브도메인 분류."""
    titles_block = "\n".join(f"- {t}" for t in slide_titles[:10] if t)
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
    except Exception as e:
        print(f"  [도메인 분류 오류] {e}")
        return {"domain": "", "subdomain": ""}


# ── Gemini 관련 함수들 ──────────────────────────────────────────────

def extract_slide_text(img_path: str) -> dict:
    """슬라이드 이미지에서 Gemini Vision으로 제목 + 본문 텍스트 추출"""
    with open(img_path, "rb") as f:
        img_bytes = f.read()

    prompt = """이 강의 슬라이드 이미지에서 텍스트를 추출하세요.

## 출력 형식 (JSON만)
{"title": "슬라이드 제목", "text": "본문 내용 전체"}

### 규칙
- 제목: 슬라이드 상단의 큰 글씨 (없으면 빈 문자열)
- 본문: 제목을 제외한 나머지 텍스트 전부 (줄바꿈은 \\n으로)
- 다이어그램/표의 텍스트도 포함
- 이미지 속 텍스트를 있는 그대로 옮길 것"""

    def call():
        return _get_client().models.generate_content(
            model=GEMINI_MODEL,
            contents=[
                types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
                types.Part.from_text(text=prompt),
            ],
            config=types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=4096,
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
        return {"title": parsed.get("title", ""), "text": parsed.get("text", "")}
    except Exception as e:
        print(f"  [텍스트 추출 오류] {e}")
        return {"title": "", "text": ""}


def load_pptx_slides(pptx_path: str) -> list[dict]:
    from pptx import Presentation
    prs = Presentation(pptx_path)
    slides = []
    for i in range(len(prs.slides)):
        slide = prs.slides[i]
        texts = [s.text.strip() for s in slide.shapes
                 if hasattr(s, "text") and s.text.strip()]
        title = texts[0] if texts else f"슬라이드 {i+1}"
        body  = "\n".join(texts[1:]) if len(texts) > 1 else ""
        slides.append({"slide_number": i + 1, "title": title, "text": body})
    return slides


def identify_pptx_slide(img_path: str, pptx_slides: list[dict]) -> Optional[int]:
    """슬라이드 이미지를 Gemini에 보내 PPTX 슬라이드 번호 식별"""
    with open(img_path, "rb") as f:
        img_bytes = f.read()

    slides_list = "\n".join(
        f"{s['slide_number']}. {s['title']} / {s['text'][:60]}"
        for s in pptx_slides
    )

    prompt = f"""아래 슬라이드 이미지를 보고, 다음 목록에서 일치하는 슬라이드 번호를 찾아주세요.
숫자 하나만 출력하세요. 일치하는 것이 없으면 0을 출력하세요.

## 슬라이드 목록
{slides_list}

## 출력 형식
숫자만 (예: 54)
"""

    def call():
        return _get_client().models.generate_content(
            model=GEMINI_MODEL,
            contents=[
                types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
                types.Part.from_text(text=prompt),
            ],
            config=types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=1024,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )

    response = api_call_with_retry(call)
    _add_usage(response)
    if response is None or not response.candidates:
        return None
    raw = (response.text or "").strip()
    try:
        return int(raw)
    except ValueError:
        m = re.search(r"\d+", raw)
        return int(m.group()) if m else None


# ── 세그먼트 배분 ───────────────────────────────────────────────────

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
    occ_index.sort(key=lambda x: (x["start_sec"], x["end_sec"], x["slide_no"]))
    return occ_index


def _assign_segment_occurrence(seg: dict, occ_index: list[dict]) -> Optional[int]:
    """
    세그먼트를 하나의 슬라이드 occurrence에 주배정한다.
    - 기본: overlap 최대 occurrence
    - 전환 경계(세그먼트가 다음 occurrence 시작을 가로지름): 다음 occurrence 우선
    """
    if not occ_index:
        return None

    seg_start = float(seg.get("start", 0) or 0)
    seg_end = float(seg.get("end", seg_start) or seg_start)
    if seg_end < seg_start:
        seg_end = seg_start

    overlaps: list[tuple[int, float]] = []
    for idx, occ in enumerate(occ_index):
        ov = min(seg_end, occ["end_sec"]) - max(seg_start, occ["start_sec"])
        if ov > 0:
            overlaps.append((idx, ov))

    if overlaps:
        overlap_map = {idx: ov for idx, ov in overlaps}
        overlaps.sort(key=lambda x: (x[1], occ_index[x[0]]["start_sec"]), reverse=True)
        best_idx, best_ov = overlaps[0]

        # 경계 발화(다음 슬라이드 시작을 가로지르는 경우)는 다음 슬라이드 우선
        later_candidates = [
            idx
            for idx, _ in overlaps
            if occ_index[idx]["start_sec"] > seg_start and occ_index[idx]["start_sec"] <= seg_end
        ]
        if later_candidates:
            later_idx = min(later_candidates, key=lambda i: occ_index[i]["start_sec"])
            later_start = occ_index[later_idx]["start_sec"]
            later_ov = overlap_map.get(later_idx, 0.0)
            started_near_boundary = seg_start >= (later_start - TRANSITION_LEAD_SEC)
            carried_after_boundary = (seg_end - later_start) >= TRANSITION_TAIL_SEC
            if started_near_boundary or later_ov >= (best_ov * 0.75) or carried_after_boundary:
                return later_idx

        return best_idx

    # overlap이 없으면 세그먼트 끝쪽(anchor) 기준으로 가장 가까운 occurrence 선택
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


# ── 전사 교정 (LLM 판정) ────────────────────────────────────────────

def parse_batch_response(text: str) -> dict[int, dict]:
    """LLM 응답에서 corrections 파싱"""
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
    for c in parsed.get("corrections", []):
        if not isinstance(c, dict):
            continue
        idx = c.get("index")
        txt = c.get("text", "")
        decision = str(c.get("decision", "") or "").strip().lower()
        risk = str(c.get("risk", "") or "").strip().lower()
        reason = str(c.get("reason", "") or "").strip()
        if isinstance(idx, int) and txt:
            corrections[idx] = {
                "text": txt,
                "decision": decision,
                "risk": risk,
                "reason": reason,
            }

    return corrections


def extract_glossary_terms(slide_texts: dict[int, dict], pptx_slides: list[dict], use_pptx: bool) -> list[str]:
    """전체 슬라이드에서 영문 용어, 함수명, 약어를 수집."""
    all_text = ""
    if use_pptx:
        for s in pptx_slides:
            all_text += f" {s.get('title', '')} {s.get('text', '')}"
    else:
        for sno, ext in slide_texts.items():
            all_text += f" {ext.get('title', '')} {ext.get('text', '')}"

    # 영문 용어/함수명 추출 (2글자 이상, 코드 패턴 포함)
    code_terms = set(re.findall(r'\b[A-Za-z_][A-Za-z0-9_.()]*\b', all_text))
    # 괄호 포함 함수명 (예: fit(), predict())
    func_terms = set(re.findall(r'\b[A-Za-z_]\w*\s*\(', all_text))
    func_terms = {t.strip().rstrip('(').strip() for t in func_terms}

    return sorted({t for t in (code_terms | func_terms) if len(t) >= 2})


def extract_glossary(slide_texts: dict[int, dict], pptx_slides: list[dict], use_pptx: bool) -> str:
    """
    전체 슬라이드에서 전문용어 glossary를 추출.
    LLM 호출 없이 슬라이드 텍스트에서 영문 용어, 함수명, 약어를 수집.
    """
    terms = extract_glossary_terms(slide_texts, pptx_slides, use_pptx)
    if not terms:
        return ""
    return "## 슬라이드 용어 사전 (표기 참조용)\n" + ", ".join(terms)




def _correct_batch_pass1(
    batch: list[tuple[int, dict]],
    slide_title: str = "",
    glossary: str = "",
) -> dict[int, str]:
    """Pass 1: 슬라이드 없이 문맥만으로 ASR 교정. {global_idx: corrected_text} 반환."""
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
- ASR이 영문 용어를 잘못 인식한 경우(예: 함수명/클래스명/라이브러리명) glossary를 참고하여 교정



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
    except Exception as e:
        print(f"  [Pass1 오류 무시] {e}")
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
    """Pass 2: 슬라이드 포함 교정. {global_idx: corrected_text} 반환."""
    if not batch:
        return {}

    seg_text = "\n".join(
        f"[{local_i}] {seg.get('text_original', seg['text'])}"
        for local_i, (_, seg) in enumerate(batch)
    )

    has_image = slide_image_path and Path(slide_image_path).exists()
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
    except Exception as e:
        print(f"  [Pass2 오류 무시] {e}")
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
    """
    Pass 1(문맥만)과 Pass 2(슬라이드 포함) 결과를 비교하여 최종 판정.

    - 둘 다 같은 교정 → apply (확실한 ASR 오류)
    - Pass 1만 교정 → apply (문맥상 명확한 오류)
    - Pass 2만 교정 → candidate_only (슬라이드 영향 가능성)
      ※ 예외: 띄어쓰기만 다른 경우 → apply
      ※ 예외: 컴퓨터공학 등에서 변수명/약어 표기 반영 → apply
    - 둘 다 교정했지만 다름 → Pass 1 채택 apply (문맥 우선)
    """
    is_cs = subdomain in ("컴퓨터공학", "소프트웨어공학", "정보통신", "전산학")

    corrections: dict[int, dict] = {}
    all_indices = set(pass1.keys()) | set(pass2.keys())

    # 원문 조회용 맵
    orig_map = {}
    for global_i, seg in batch:
        orig_map[global_i] = seg.get("text_original", seg["text"])

    for global_i in all_indices:
        p1 = pass1.get(global_i)
        p2 = pass2.get(global_i)
        original = orig_map.get(global_i, "")

        if p1 and p2:
            if _normalize_text(p1) == _normalize_text(p2):
                # 둘 다 같은 교정 → 확실한 ASR 오류
                corrections[global_i] = {
                    "candidate_text": p1,
                    "applied_text": p1,
                    "risk": "low",
                    "apply": True,
                    "reason": "pass1+pass2 일치",
                }
            else:
                # 둘 다 교정했지만 다름 → Pass 1(문맥 기반) 우선
                corrections[global_i] = {
                    "candidate_text": p2,
                    "applied_text": p1,
                    "risk": "medium",
                    "apply": True,
                    "reason": "pass1 채택 (pass2 상이)",
                }
        elif p1 and not p2:
            # Pass 1만 교정 → 문맥상 명확한 오류
            corrections[global_i] = {
                "candidate_text": p1,
                "applied_text": p1,
                "risk": "low",
                "apply": True,
                "reason": "pass1만 교정 (문맥 기반)",
            }
        elif p2 and not p1:
            # Pass 2만 교정 → 기본적으로 candidate_only
            # 예외 1: 띄어쓰기만 다른 경우 → 안전하게 적용
            if _is_spacing_only(original, p2):
                corrections[global_i] = {
                    "candidate_text": p2,
                    "applied_text": p2,
                    "risk": "low",
                    "apply": True,
                    "reason": "pass2 띄어쓰기 교정만 (안전)",
                }
            # 예외 2: CS 도메인에서 변수명/약어 표기 반영
            elif is_cs and _is_variable_or_abbrev_fix(original, p2):
                corrections[global_i] = {
                    "candidate_text": p2,
                    "applied_text": p2,
                    "risk": "low",
                    "apply": True,
                    "reason": "pass2 변수명/약어 표기 반영 (CS)",
                }
            # 예외 3: 영문→영문 교정 (ASR 영문 오인식, 예: PIT→fit)
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


# ── 메인 프로세스 ────────────────────────────────────────────────────

def process(
    stem: str,
    pptx_path: Optional[str] = None,
    output_dir: str = "./output",
    work_dir: str = ".",
) -> Path:
    """
    슬라이드 감지 결과 + 전사본 → 교정 + merged JSON 생성.

    Args:
        stem:       영상 파일 이름 (확장자 제외)
        pptx_path:  PPTX 파일 경로 (없으면 None)
        output_dir: slide_detector 출력 디렉토리 (output/{stem}/ 상위)
        work_dir:   transcribed.json / merged.json 위치

    Returns:
        생성된 merged JSON 경로
    """
    global _token_usage
    _token_usage = {"input": 0, "output": 0, "calls": 0}

    use_pptx  = bool(pptx_path)
    work      = Path(work_dir).resolve()
    out       = Path(output_dir).resolve()

    detector_log    = out / stem / f"{stem}_log.json"
    img_dir         = out / stem
    transcript_path = work / f"{stem}_transcribed.json"
    output_path     = work / f"{stem}_merged.json"
    checkpoint_path = work / f"{stem}_merge_checkpoint.json"


    # ── [1/4] 데이터 로드
    print("[1/4] 데이터 로드")
    with open(detector_log, encoding="utf-8") as f:
        log = json.load(f)
    with open(transcript_path, encoding="utf-8") as f:
        segments = json.load(f)

    if use_pptx:
        pptx_slides = load_pptx_slides(pptx_path)
        print(f"  PPTX 슬라이드: {len(pptx_slides)}개")
    else:
        pptx_slides = []
        print("  PPTX 없음 — detected slide_no 직접 사용")
    print(f"  전사 세그먼트: {len(segments)}개")

    saved_slides = [r for r in log["slides"] if r["status"] == "SAVED"]
    timeline     = log["timeline"]
    detector_signature = {
        "path": str(detector_log),
        "mtime_ns": detector_log.stat().st_mtime_ns if detector_log.exists() else 0,
        "size": detector_log.stat().st_size if detector_log.exists() else 0,
        "saved_slides": len(saved_slides),
        "timeline_entries": len(timeline),
    }

    # ── [2/4] 슬라이드 번호 식별
    cache_path = img_dir / "slide_id_cache.json"

    if not use_pptx:
        detected_to_pptx = {row["slide_no"]: row["slide_no"] for row in saved_slides}
        print(f"\n[2/4] PPTX 없음 — slide_no 직접 매핑 ({len(detected_to_pptx)}개)")
    elif cache_path.exists():
        with open(cache_path, encoding="utf-8") as f:
            raw = json.load(f)
        detected_to_pptx = {int(k): v for k, v in raw.items()}
        print(f"\n[2/4] 슬라이드 식별 캐시 로드 ({len(detected_to_pptx)}개, Gemini 생략)")
    else:
        print(f"\n[2/4] Gemini 슬라이드 번호 식별 ({len(saved_slides)}개)")
        detected_to_pptx = {}
        for row in saved_slides:
            det_no   = row["slide_no"]
            img_path = str(img_dir / f"slide_{det_no:03d}_start.jpg")
            if not Path(img_path).exists():
                print(f"  slide_{det_no:03d}: 이미지 없음, 스킵")
                detected_to_pptx[det_no] = None
                continue
            pptx_no = identify_pptx_slide(img_path, pptx_slides)
            detected_to_pptx[det_no] = pptx_no
            title = next((s["title"] for s in pptx_slides if s["slide_number"] == pptx_no), "?")
            print(f"  slide_{det_no:03d} → PPTX {pptx_no}: {title[:40]}")
            time.sleep(0.3)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(detected_to_pptx, f, ensure_ascii=False)
        print(f"  캐시 저장 → {cache_path}")

    # ── [3/4] 세그먼트 배분 + 교정
    print(f"\n[3/4] 슬라이드별 세그먼트 배분 + 교정")

    slide_occurrences: dict[int, list[dict]] = {}
    slide_det_no: dict[int, int] = {}
    for entry in timeline:
        pptx_no = detected_to_pptx.get(entry["slide_no"])
        if not pptx_no:
            continue
        slide_det_no.setdefault(pptx_no, entry["slide_no"])
        slide_occurrences.setdefault(pptx_no, []).append({
            "start_sec": entry["start_sec"],
            "end_sec":   entry["end_sec"],
            "duration":  entry["duration"],
            "is_dup":    entry["is_dup"],
        })

    # PPTX 없을 때 슬라이드 이미지에서 텍스트 미리 추출 (교정 시 slide_context로 사용)
    extracted_slide_texts: dict[int, dict] = {}
    if not use_pptx:
        print("  슬라이드 이미지 텍스트 사전 추출 중...")
        for slide_no in sorted(slide_occurrences.keys()):
            det_no = slide_det_no.get(slide_no, slide_no)
            img_path = str(img_dir / f"slide_{det_no:03d}_start.jpg")
            if Path(img_path).exists():
                extracted = extract_slide_text(img_path)
                extracted_slide_texts[slide_no] = extracted
                print(f"    슬라이드 {slide_no:3d}: {(extracted['title'] or '(제목 없음)')[:30]}")
                time.sleep(0.3)

    occ_index = _build_occurrence_index(slide_occurrences)

    seg_slide: dict[int, int] = {}
    for i, seg in enumerate(segments):
        occ_idx = _assign_segment_occurrence(seg, occ_index)
        if occ_idx is None:
            continue
        seg_slide[i] = occ_index[occ_idx]["slide_no"]

    groups: dict[int, list[tuple[int, dict]]] = {}
    no_slide: list[tuple[int, dict]] = []
    for i, seg in enumerate(segments):
        if i in seg_slide:
            groups.setdefault(seg_slide[i], []).append((i, seg))
        else:
            no_slide.append((i, seg))

    # 체크포인트 로드
    if checkpoint_path.exists():
        with open(checkpoint_path, encoding="utf-8") as f:
            ckpt = json.load(f)
        ckpt_sig = ckpt.get("detector_signature")
        if ckpt_sig == detector_signature:
            all_corrections = {}
            for k, v in ckpt.get("corrections", {}).items():
                idx = int(k)
                if isinstance(v, dict):
                    all_corrections[idx] = v
                elif isinstance(v, str):
                    all_corrections[idx] = {
                        "candidate_text": v,
                        "applied_text": v,
                        "risk": "low",
                        "apply": True,
                        "reason": "legacy_checkpoint",
                    }
            done_slides = set(ckpt.get("done_slides", []))
            print(f"  체크포인트 재개: {len(done_slides)}개 슬라이드 완료")
        else:
            all_corrections = {}
            done_slides = set()
            print("  체크포인트 무효화: detector 로그가 변경되어 처음부터 재처리")
    else:
        all_corrections = {}
        done_slides = set()

    def save_checkpoint():
        with open(checkpoint_path, "w", encoding="utf-8") as f:
            json.dump({
                "corrections": all_corrections,
                "done_slides": list(done_slides),
                "detector_signature": detector_signature,
            }, f, ensure_ascii=False)

    # 도메인 분류 (교정 전에 먼저 수행 — merge 판정에 subdomain 필요)
    print("\n  도메인 분류 중...")
    _slide_titles_for_domain = []
    for sno in sorted(slide_occurrences.keys()):
        if use_pptx:
            pm = next((s for s in pptx_slides if s["slide_number"] == sno), {})
            _slide_titles_for_domain.append(pm.get("title", ""))
        else:
            _slide_titles_for_domain.append(extracted_slide_texts.get(sno, {}).get("title", ""))
    _transcript_sample = " ".join(seg.get("text", "") for seg in segments[:30])
    domain_info = classify_lecture_domain(_slide_titles_for_domain, _transcript_sample)
    subdomain = domain_info.get("subdomain", "")
    print(f"  → 도메인: {domain_info['domain']}, 서브도메인: {subdomain}")

    # 용어 사전 추출 (Pass 1에 전달)
    glossary_terms = extract_glossary_terms(extracted_slide_texts, pptx_slides, use_pptx)
    glossary = "## 슬라이드 용어 사전 (표기 참조용)\n" + ", ".join(glossary_terms) if glossary_terms else ""
    if glossary:
        term_count = len(glossary_terms)
        print(f"  → 용어 사전: {term_count}개 용어 추출")

    for slide_no in sorted(slide_occurrences.keys()):
        if slide_no in done_slides:
            print(f"  슬라이드 {slide_no:3d}: 스킵 (체크포인트)")
            continue
        group = groups.get(slide_no, [])
        if not group:
            done_slides.add(slide_no)
            continue
        pptx_meta = next((s for s in pptx_slides if s["slide_number"] == slide_no), {})
        if use_pptx:
            context = f"슬라이드 제목: {pptx_meta.get('title', '')}\n{pptx_meta.get('text', '')}"
            slide_title = pptx_meta.get('title', '')
        else:
            ext = extracted_slide_texts.get(slide_no, {})
            context = f"슬라이드 제목: {ext.get('title', '')}\n{ext.get('text', '')}"
            slide_title = ext.get('title', '')
        det_no = slide_det_no.get(slide_no)
        img_path = str(img_dir / f"slide_{det_no:03d}_end.jpg") if det_no else None
        img_flag = " +IMG" if img_path and Path(img_path).exists() else ""
        print(f"  슬라이드 {slide_no:3d} ({pptx_meta.get('title', '')[:30]:30s}): {len(group):3d}개{img_flag}", end="", flush=True)
        slide_corrections: dict[int, dict] = {}
        sub_batches = [group[b:b + BATCH_SIZE] for b in range(0, len(group), BATCH_SIZE)]
        for sub in sub_batches:
            with ThreadPoolExecutor(max_workers=2) as pool:
                f1 = pool.submit(_correct_batch_pass1, sub, slide_title=slide_title, glossary=glossary)
                f2 = pool.submit(_correct_batch_pass2, sub, context, slide_image_path=img_path)
                p1 = f1.result()
                print("①", end="", flush=True)
                p2 = f2.result()
                print("②", end="", flush=True)
            merged = merge_two_passes(sub, p1, p2, subdomain=subdomain)
            slide_corrections.update(merged)
        all_corrections.update(slide_corrections)
        done_slides.add(slide_no)
        save_checkpoint()
        print()

    if no_slide:
        print(f"  미매핑 {len(no_slide)}개 교정 중...", end="", flush=True)
        for b in range(0, len(no_slide), BATCH_SIZE):
            sub = no_slide[b:b + BATCH_SIZE]
            p1 = _correct_batch_pass1(sub, glossary=glossary)
            all_corrections.update({
                gi: {"candidate_text": txt, "applied_text": txt, "risk": "low", "apply": True, "reason": "pass1 only (미매핑)"}
                for gi, txt in p1.items()
            })
            print(".", end="", flush=True)
        print()

    # 교정 적용
    corrected_segments = []
    correction_summary = {
        "applied": 0,
        "candidate_only": 0,
        "unchanged": 0,
        "risk_breakdown": {"low": 0, "medium": 0, "high": 0, "none": 0},
    }
    for i, seg in enumerate(segments):
        s = seg.copy()
        s["text_original"] = seg.get("text_original", seg["text"])
        corr_info = all_corrections.get(i)
        if isinstance(corr_info, str):
            corr_info = {
                "candidate_text": corr_info,
                "applied_text": corr_info,
                "risk": "low",
                "apply": True,
                "reason": "legacy_string",
            }

        if corr_info:
            risk = str(corr_info.get("risk", "none") or "none")
            correction_summary["risk_breakdown"][risk] = correction_summary["risk_breakdown"].get(risk, 0) + 1
            candidate_text = corr_info.get("candidate_text") or s["text_original"]
            applied = bool(corr_info.get("apply"))
            if applied and corr_info.get("applied_text"):
                s["text"] = corr_info["applied_text"]
                s["correction_status"] = "applied"
                correction_summary["applied"] += 1
            else:
                s["text"] = s["text_original"]
                s["text_corrected_candidate"] = candidate_text
                s["correction_status"] = "candidate_only"
                correction_summary["candidate_only"] += 1
            s["correction_risk"] = risk
            s["correction_reason"] = str(corr_info.get("reason", "") or "")
        else:
            s["text"] = s["text_original"]
            s["correction_status"] = "unchanged"
            s["correction_risk"] = "none"
            correction_summary["risk_breakdown"]["none"] += 1
            correction_summary["unchanged"] += 1
        corrected_segments.append(s)

    with open(transcript_path, "w", encoding="utf-8") as f:
        json.dump(corrected_segments, f, ensure_ascii=False, indent=2)

    checkpoint_path.unlink(missing_ok=True)

    # ── [4/4] merged JSON 구성
    print(f"\n[4/4] merged JSON 구성")
    total_duration = corrected_segments[-1]["end"] if corrected_segments else 0

    merged_slides = []
    for slide_no in sorted(slide_occurrences.keys()):
        occs      = slide_occurrences[slide_no]
        pptx_meta = next((s for s in pptx_slides if s["slide_number"] == slide_no), {})

        all_start = min(o["start_sec"] for o in occs)
        all_end   = max(o["end_sec"]   for o in occs)
        total_dur = sum(o["duration"]  for o in occs)

        # 세그먼트는 "주배정된 슬라이드"에만 포함해 경계 중복/오배치를 줄인다.
        assigned = [
            corrected_segments[i]
            for i in sorted(seg_slide.keys())
            if seg_slide.get(i) == slide_no
        ]
        seen_starts: set = set()
        unique_segs = []
        for s in sorted(assigned, key=lambda x: x["start"]):
            if s["start"] not in seen_starts:
                seen_starts.add(s["start"])
                unique_segs.append(s)

        if use_pptx:
            title      = pptx_meta.get("title", f"슬라이드 {slide_no}")
            slide_text = pptx_meta.get("text", "")
        else:
            ext = extracted_slide_texts.get(slide_no, {})
            if ext:
                title      = ext["title"] or f"슬라이드 {slide_no}"
                slide_text = ext["text"]
            else:
                title      = f"슬라이드 {slide_no}"
                slide_text = ""
        time_range = f"{fmt_ts(all_start)} ~ {fmt_ts(all_end)}"
        label      = "PPTX" if use_pptx else "슬라이드"
        print(f"  {label} {slide_no}: {title[:35]:35s} {time_range}  ({len(occs)}회, {total_dur:.0f}s)")

        merged_slides.append({
            "slide_number":        slide_no,
            "title":               title,
            "time_range":          time_range,
            "time_range_seconds":  [all_start, all_end],
            "total_duration":      round(total_dur, 1),
            "occurrences":         occs,
            "slide_text":          slide_text,
            "transcript":          " ".join(s.get("text", "") for s in unique_segs),
            "transcript_segments": unique_segs,
            "segment_count":       len(unique_segs),
        })

    # 도메인 분류는 교정 전에 이미 완료됨 (subdomain → merge 판정에 사용)

    result = {
        "description":               "슬라이드+전사 통합 JSON (slide_detector 기반, LLM 판정 교정)",
        "source_slides":             pptx_path or "(없음)",
        "source_transcript":         str(transcript_path),
        "source_detector_log":       str(detector_log),
        "domain":                    domain_info["domain"],
        "subdomain":                 domain_info["subdomain"],
        "correction_summary":        correction_summary,
        "total_slides":              len(merged_slides),
        "total_transcript_segments": len(corrected_segments),
        "total_duration_formatted":  fmt_ts(total_duration),
        "slides": merged_slides,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # ── clean 버전 생성 (교정 메타데이터 제거, content_verifier용) ──
    _CORRECTION_KEYS = {
        "text_original", "correction_status", "correction_risk",
        "correction_reason", "text_corrected_candidate",
    }
    clean_slides = []
    for slide in merged_slides:
        clean_slide = {k: v for k, v in slide.items() if k != "transcript_segments"}
        clean_segs = []
        for seg in slide.get("transcript_segments", []):
            clean_segs.append({k: v for k, v in seg.items() if k not in _CORRECTION_KEYS})
        clean_slide["transcript_segments"] = clean_segs
        clean_slides.append(clean_slide)

    clean_result = {
        "description":               "슬라이드+전사 통합 JSON (교정 완료, 검증용)",
        "domain":                    domain_info["domain"],
        "subdomain":                 domain_info["subdomain"],
        "total_slides":              len(clean_slides),
        "total_transcript_segments": len(corrected_segments),
        "total_duration_formatted":  fmt_ts(total_duration),
        "slides": clean_slides,
    }

    clean_path = str(output_path).replace("_merged.json", "_merged_clean.json")
    with open(clean_path, "w", encoding="utf-8") as f:
        json.dump(clean_result, f, ensure_ascii=False, indent=2)

    print(f"\n완료! → {output_path} ({len(merged_slides)}개 슬라이드)")
    print(f"       → {clean_path} (검증용 clean)")
    print(f"\n[토큰 사용량]")
    print(f"  API 호출 횟수 : {_token_usage['calls']}회")
    print(f"  입력 토큰     : {_token_usage['input']:,}")
    print(f"  출력 토큰     : {_token_usage['output']:,}")
    print(f"  합계          : {_token_usage['input'] + _token_usage['output']:,}")

    return Path(output_path)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("stem", help="영상 파일 이름 (확장자 제외, 예: 2장3절2)")
    parser.add_argument("pptx", nargs="?", default=None, help="PPTX 파일 경로 (없으면 생략)")
    args = parser.parse_args()
    process(args.stem, args.pptx)


if __name__ == "__main__":
    main()
