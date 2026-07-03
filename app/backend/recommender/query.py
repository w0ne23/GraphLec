"""
recommender/query.py
────────────────────
자연어 질의 분석 (LLM 호출 + 캐시 + fallback).
"""

from __future__ import annotations

import json
import re
from typing import Optional

from openai import OpenAI

from recommender.config import (
    GEMINI_MODEL,
    OPENAI_API_KEY,
    QUERY_LLM_PROVIDER,
    QUERY_OPENAI_MODEL,
    _client,
)
from recommender.scoring import (
    _VISUAL_CONDITION_TERMS,
    _APPLICATION_CONDITION_TERMS,
    _DELIVERY_CONDITION_TERMS,
    _RECENCY_CONDITION_TERMS,
    _content_terms_only,
)
from recommender.utils import (
    _append_topic_expansions,
    _compact_term,
    _infer_domain_filters,
    _normalize_term,
    _query_term_base,
    _term_lookup_keys,
)


# ── OpenAI 클라이언트 ─────────────────────────────────────────────────────────

def _openai_client() -> OpenAI:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY 환경 변수가 없습니다.")
    return OpenAI(api_key=OPENAI_API_KEY)


def _call_query_analysis_llm(prompt: str) -> str:
    if QUERY_LLM_PROVIDER == "openai":
        response = _openai_client().chat.completions.create(
            model=QUERY_OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "너는 강의 추천 질의를 분석해 JSON만 반환하는 분류기다.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_completion_tokens=2048,
        )
        return (response.choices[0].message.content or "").strip()

    response = _client.models.generate_content(
        model    = GEMINI_MODEL,
        contents = prompt,
        config   = {"temperature": 0.0},
    )
    return response.text.strip()


# ── 분석 캐시 ─────────────────────────────────────────────────────────────────

_ANALYZE_QUERY_CACHE: dict[tuple[str, tuple[str, ...], int], tuple] = {}


def _analysis_cache_key(
    query: str,
    available_domains: list[str],
    available_keywords: list[str],
) -> tuple[str, tuple[str, ...], int]:
    return (
        re.sub(r"\s+", " ", str(query or "").strip()),
        tuple(sorted(available_domains)),
        len(available_keywords),
    )


def _copy_analysis_result(result: tuple) -> tuple:
    intent, search_text, query_keywords, inferred_keywords, domain, focus_concept, duration_max_sec, conditions = result
    return (
        intent,
        search_text,
        list(query_keywords),
        list(inferred_keywords),
        domain,
        focus_concept,
        duration_max_sec,
        dict(conditions),
    )


# ── list_by_topic 빠른 감지 ───────────────────────────────────────────────────

def _fast_list_query_analysis(
    query: str,
    available_domains: list[str],
    available_subdomains: list[str],
) -> Optional[tuple[str, str, list[str], list[str], Optional[str], Optional[str], Optional[int], dict]]:
    normalized = _normalize_term(query)
    has_list_signal = any(
        signal in normalized
        for signal in (
            "뭐 있어", "뭐있어", "목록", "리스트",
            "전체 강의", "모든 강의", "강의 보여", "강의 알려",
        )
    )
    if not has_list_signal:
        return None

    query_terms = [query]
    if "전체 강의" in normalized or "모든 강의" in normalized:
        return "recommend", query, query_terms, [], None, None, None, {}

    domain, subdomain = _infer_domain_filters(query, available_domains, available_subdomains)
    if domain:
        return "recommend", query, query_terms, [], domain, None, None, {"subdomain": subdomain}

    return None


# ── fallback 분석 ─────────────────────────────────────────────────────────────

def _fallback_query_analysis(
    query: str,
    available_domains: list[str],
    available_keywords: list[str],
) -> tuple[str, str, list[str], list[str], Optional[str], Optional[str], Optional[int], dict]:
    """LLM 질의 분석 실패 시 metadata keyword pool만으로 안전하게 질의를 해석한다."""
    domain, subdomain = _infer_domain_filters(query, available_domains, [])
    query_base = _query_term_base(query)
    query_keys = _term_lookup_keys(query_base or query)
    query_compact = _compact_term(query_base or query)

    matched: list[str] = []
    seen: set[str] = set()
    for keyword in sorted(available_keywords, key=lambda value: (-len(str(value)), str(value))):
        normalized = _normalize_term(keyword)
        if not normalized or normalized in seen:
            continue
        keyword_keys = _term_lookup_keys(normalized)
        keyword_compact = _compact_term(normalized)
        if (
            keyword_keys & query_keys
            or (keyword_compact and keyword_compact in query_compact)
            or (query_compact and query_compact in keyword_compact)
        ):
            seen.add(normalized)
            matched.append(normalized)
        if len(matched) >= 4:
            break

    if not matched and query_base:
        matched = [query_base]

    inferred = _append_topic_expansions(matched, [])
    inferred = [term for term in inferred if _normalize_term(term) not in {_normalize_term(m) for m in matched}]
    search_text = " ".join(matched + inferred) or query
    focus_concept = matched[0] if len(matched) == 1 and matched[0] in available_keywords else None
    _has_duration = any(term in query for term in ("짧은", "짧게", "분 이내", "분 이하", "분 내외"))
    conditions = {
        "issue_free": False,
        "prefers_visual": any(term in query for term in _VISUAL_CONDITION_TERMS),
        "prefers_application": any(term in query for term in _APPLICATION_CONDITION_TERMS),
        "prefers_slow_speech": any(term in query for term in _DELIVERY_CONDITION_TERMS),
        "prefers_listenability": any(term in query for term in _DELIVERY_CONDITION_TERMS),
        "prefers_recency": any(term in query for term in _RECENCY_CONDITION_TERMS),
        "query_type": "condition_first" if _has_duration else "topic_browse",
        "query_specificity": "broad",
    }
    if subdomain:
        conditions["subdomain"] = subdomain

    return (
        "recommend",
        search_text,
        matched,
        inferred[:6],
        domain,
        focus_concept,
        None,
        conditions,
    )


# ── 메인 질의 분석 ────────────────────────────────────────────────────────────

def analyze_query(
    query:              str,
    available_domains:  list[str],
    available_keywords: list[str],
) -> tuple[str, str, list[str], list[str], Optional[str], Optional[str], Optional[int], dict]:
    """
    질의 → intent + search_text + query_keywords + inferred_keywords + domain + focus_concept + duration_max_sec 추출.

    반환:
      intent             : "recommend" | "list_by_topic"
      search_text        : 벡터 임베딩용 전체 텍스트 (query + inferred 합산)
      query_keywords     : 원본 질의에서 직접 추출한 핵심 용어 (dm 100% 반영)
      inferred_keywords  : Gemini가 의미 확장한 연관 용어 (dm 50% 반영)
      domain             : available_domains 중 하나, 없으면 None
      focus_concept      : 깊이를 측정할 핵심 개념, 없으면 None
      duration_max_sec   : 최대 강의 길이(초), 언급 없으면 None
      conditions         : 조건 질의 플래그
    """
    cache_key = _analysis_cache_key(query, available_domains, available_keywords)
    cached = _ANALYZE_QUERY_CACHE.get(cache_key)
    if cached:
        return _copy_analysis_result(cached)

    domain_list  = ", ".join(available_domains)
    keyword_list = ", ".join(available_keywords)

    prompt = f"""다음 강의 검색 질의를 분석해줘.

질의: "{query}"

다음 JSON 형식으로만 출력해 (설명 없이):
{{
  "query_keywords": ["원본 질의 핵심 용어1", ...],
  "inferred_keywords": ["확장 연관 용어1", ...],
  "intent": "recommend",
  "query_type": "topic_browse | concept_depth | condition_first | related_search",
  "query_specificity": "broad | specific",
  "domain": "도메인 문자열 또는 null",
  "focus_concept": "개념 문자열 또는 null",
  "duration_max_sec": 숫자 또는 null,
  "conditions": {{
    "issue_free": true 또는 false,
    "prefers_visual": true 또는 false,
    "prefers_application": true 또는 false,
    "prefers_slow_speech": true 또는 false,
    "prefers_listenability": true 또는 false,
    "prefers_recency": true 또는 false
  }}
}}

[query_type]: 질의 유형 분류
  - "topic_browse": 넓은 주제 탐색 — 여러 강의 나와도 됨
    예) "운영체제 강의 추천", "AI 강의 뭐 있어", "미적분 강의 알려줘"
  - "concept_depth": 특정 개념의 원리·동작 방식 심화 요청
    예) "가상 메모리 동작 원리 자세히", "커널 구조 깊게 설명하는 강의"
  - "condition_first": 길이·최신성 등 조건이 명시적 주 요청
    예) "짧은 AI 강의", "30분 이내 머신러닝 강의", "최근 올라온 딥러닝 강의"
  - "related_search": 특정 개념이 포함된 강의를 탐색 — 핵심 주제가 아닌 연관 탐색
    예) "커널 다루는 강의 있어?", "TCP 언급하는 강의 있나요?"

[query_specificity]: 질의어 특이성
  - "broad": 분야·주제 수준의 일반 용어 — 여러 강의에서 공통으로 다룰 법한 개념
  - "specific": 특정 알고리즘명·인물·사건·고유 기법처럼 일부 강의만 다룰 법한 용어
  - 판단 기준: query_keywords가 교과서 목차 수준이면 broad, 특정 챕터나 인물 이름 수준이면 specific

[query_keywords]: 원본 질의에서 직접 등장하는 핵심 학술·기술 용어
- "찾아줘", "알려줘", "강의", "어떻게" 같은 메타·구어체 표현 제외
- 질의에 명시된 개념만 포함 (1~4개)
- 예) "가상 메모리 페이징 방식 설명하는 강의" → ["가상 메모리", "페이징"]
- 예) "TCP와 UDP 차이를 다루는 강의" → ["TCP", "UDP"]
- 예) "비동기 처리할 때 막혀" → ["비동기"]

[inferred_keywords]: 질의 의도에서 연관성이 높은 확장 용어 (2~6개)
- query_keywords와 겹치지 않을 것
- 영문 약어·영문 기술 용어가 query_keywords에 포함된 경우, 반드시 한국어 학술 용어를 inferred_keywords에 포함할 것
  (예: "os" → "운영체제", "db" → "데이터베이스", "ml" → "머신러닝", "dl" → "딥러닝",
       "nlp" → "자연어 처리", "cv" → "컴퓨터 비전", "nn" → "신경망", "oop" → "객체지향 프로그래밍")
- 예) "가상 메모리 페이징" → ["페이지 폴트", "페이지 교체", "TLB", "운영체제"]
- 예) "TCP UDP 차이" → ["프로토콜", "전송 계층", "3-way 핸드셰이크", "흐름 제어"]
- 예) "비동기" → ["Promise", "async/await", "이벤트 루프", "콜백"]

[domain]: 반드시 아래 목록 중 하나: {domain_list}
  - 확신할 수 없으면 null

[focus_concept]: 특정 개념의 자세한 설명·원리·동작 방식을 명시적으로 요청할 때만 설정
  - 반드시 아래 [키워드 목록]에서만 선택, 해당 없으면 null
  - 예) "스레드 자세히 설명해주는 강의" → "스레드"
  - 예) "비동기 처리할 때 막혀" → null

[duration_max_sec]: 질의에 강의 길이 조건이 있으면 초 단위 기준값으로 변환
  - "30분 이내" → 1800
  - "30분 내외" → 1800  (내외는 기준값만 추출, ±5분 여유는 코드에서 처리)
  - "1시간 이하" → 3600
  - "짧은", "빠르게" 등 모호한 표현 → 1800
  - 길이 언급 없으면 → null

[conditions]: 내용 조건이 아니라 강의 상태/형식에 대한 선호를 의미 단위로 추출
  - issue_free: "오류 없는", "검증된", "이슈 없는", "틀린 내용 없는" 등
  - prefers_visual: "그림/도식/표/그래프/시각 자료 위주", "시각적으로 설명" 등
  - prefers_application: "예제/사례/문제풀이/실습/시연/데모/적용/활용 중심", "이론만 말고" 등
  - prefers_slow_speech: "말이 느렸으면", "급하게 설명하지 않는", "여유있게 진행", "빠르지 않은" 등
  - prefers_listenability: "음질 좋은", "잘 들리는", "듣기 편한", "녹음 상태 좋은", "전달이 명료한" 등
  - prefers_recency: "최근 업로드된", "최신 강의", "새로 올라온 강의"처럼 강의 업로드 시점이 최근이기를 선호하는 경우
    * 단순히 "최신 Transformer", "최신 기술 동향"처럼 주제 자체의 최신성을 말하는 경우는 업로드 시점 선호가 명확할 때만 true

[키워드 목록]: {keyword_list}"""

    try:
        text   = _call_query_analysis_llm(prompt).replace("```json", "").replace("```", "").strip()
        parsed = json.loads(text)
    except Exception as exc:
        print(f"  ⚠ 질의 LLM 분석 실패 — metadata 기반 fallback 사용: {exc}")
        result = _fallback_query_analysis(query, available_domains, available_keywords)
        _ANALYZE_QUERY_CACHE[cache_key] = _copy_analysis_result(result)
        return result

    intent            = parsed.get("intent") or "recommend"
    query_keywords    = parsed.get("query_keywords", [])
    inferred_keywords = parsed.get("inferred_keywords", [])
    domain            = parsed.get("domain") or None
    focus_concept     = parsed.get("focus_concept") or None
    duration_max_sec  = parsed.get("duration_max_sec") or None
    conditions        = parsed.get("conditions") if isinstance(parsed.get("conditions"), dict) else {}

    _valid_query_types = {"topic_browse", "concept_depth", "condition_first", "related_search"}
    query_type = parsed.get("query_type") or "topic_browse"
    if query_type not in _valid_query_types:
        query_type = "topic_browse"
    query_specificity = "specific" if parsed.get("query_specificity") == "specific" else "broad"

    normalized_conditions = {
        "issue_free":            bool(conditions.get("issue_free")),
        "prefers_visual":        bool(conditions.get("prefers_visual")),
        "prefers_application":   bool(conditions.get("prefers_application")),
        "prefers_slow_speech":   bool(conditions.get("prefers_slow_speech")),
        "prefers_listenability": bool(conditions.get("prefers_listenability")),
        "prefers_recency":       bool(conditions.get("prefers_recency")),
        "query_type":            query_type,
        "query_specificity":     query_specificity,
    }
    query_keywords    = _content_terms_only(query_keywords, normalized_conditions)
    inferred_keywords = _content_terms_only(inferred_keywords, normalized_conditions)

    search_text = " ".join(query_keywords + inferred_keywords) or query

    if domain not in available_domains:
        domain = None
    if duration_max_sec is not None:
        try:
            duration_max_sec = int(duration_max_sec)
        except (ValueError, TypeError):
            duration_max_sec = None
    if intent != "recommend":
        intent = "recommend"

    result = (
        intent,
        search_text,
        query_keywords,
        inferred_keywords,
        domain,
        focus_concept,
        duration_max_sec,
        normalized_conditions,
    )
    _ANALYZE_QUERY_CACHE[cache_key] = _copy_analysis_result(result)
    return result
