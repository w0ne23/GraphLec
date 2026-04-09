"""
recommender.py
──────────────
강의 추천 시스템 — 자연어 질의 기반

변경 이력:
  v3: Gemini가 메타데이터 keyword pool에서 직접 선택 (자유 생성 → vocabulary 고정)
      inferred_keywords 제거, semantic gap 해소
  v4: prerequisite 타입 제거
      focus_concept 추출 → depth_score (concept_roles/relations/하위키워드 기반)
      깊이 기반 추천 지원 ("스레드 자세히 설명하는 강의")
  v5: 인텐트별 동적 가중치 (INTENT_WEIGHTS)
      후보 집합 내 Min-Max 정규화 (keyword / title / summary 각각)
      파편화 패널티 — concept_roles 기반 (core 비율 낮으면 패널티)
      query_direct_match — 원본 질의어를 제목과 직접 매칭, title_score에 블렌딩
        → "운영체제 기능" 질의 시 상위 개념 강의 우선 추천
      MIN_SCORE 필터 — 점수 미달 강의 추천 목록에서 제외

CLI 실행:
  python recommender.py --query "메모리 관리 방법 알고 싶어"
  python recommender.py --metadata_dir metadata/ --query "운영체제란 무엇인가"

모듈로 사용:
  from recommender import Recommender
  rec = Recommender("metadata/")
  results = rec.recommend_from_query("프로세스 스케줄링 알고 싶어")
"""

import json
import os
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from google import genai
from dotenv import load_dotenv

load_dotenv()
_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY_2"))
MODEL = "gemini-2.5-flash"


# ============================================================================
#  질의 유형 정의
# ============================================================================
#
#  direct   : "운영체제 강의 추천해줘"
#             target_keywords = 관련 키워드 (pool에서 선택)
#             context_keywords = []
#             focus_concept = 깊이 측정 대상 개념 (없으면 None)
#
#  followup : "선형대수 들었는데 연계 강의 추천해줘"
#             target_keywords = 다음 단계 키워드 (pool에서 선택)
#             context_keywords = 이미 들은 강의 키워드 — 해당 강의 penalty
#             focus_concept = None (연계 흐름이 목적)
#
#  career   : "AI 연구자가 되고 싶어"
#             target_keywords = 역할에 필요한 기술 키워드 (pool에서 선택)
#             context_keywords = []
#             focus_concept = None
#
#  unknown  : 강의 추천 의도 없음

QUERY_TYPES = ("direct", "followup", "career", "unknown")

# context_keywords 강의에 적용할 score 배율 (낮을수록 강하게 억제)
CONTEXT_PENALTY_BY_TYPE: dict[str, float] = {
    "followup": 0.15,
}

# ============================================================================
#  인텐트별 동적 가중치 (v5)
# ============================================================================
#
#  (w_keyword, w_title, w_summary, domain_boost_strength)
#
#  direct   : 제목 비중을 0.40으로 상향
#             → "운영체제 기능" 같은 broad 질의에서 상위 개념이 제목에 있는 강의 우선
#  followup : summary(맥락) 중심 — 다음 단계 개념이 요약에 설명된 강의 선호
#  career   : domain_boost 강화 — 직무 연관 도메인 강의 우선
#  unknown  : 균형

INTENT_WEIGHTS: dict[str, tuple[float, float, float, float, float]] = {
    "direct":   (0.35, 0.35, 0.10, 0.10, 0.10),
    "followup": (0.20, 0.15, 0.45, 0.10, 0.10),
    "career":   (0.30, 0.15, 0.20, 0.20, 0.15),
    "unknown":  (0.40, 0.20, 0.20, 0.10, 0.10),
}


# ============================================================================
#  설정
# ============================================================================

@dataclass
class RecommenderConfig:
    # depth: multiplicative boost (focus_concept 지정 시에만 활성)
    # total × (1 + W_DEPTH_BOOST × depth_score)
    W_DEPTH_BOOST:       float = 0.30
    # 파편화 패널티 강도 λ (0.0 ~ 1.0)
    # content_score -= λ × fragmentation_penalty
    FRAG_PENALTY_WEIGHT: float = 0.10
    # 추천 최소 점수 — 이 점수 미만인 강의는 추천 결과에서 제외
    # 0.0으로 설정하면 모든 강의 반환 (기존 동작)
    MIN_SCORE:           float = 0.05


# ============================================================================
#  데이터 모델
# ============================================================================

@dataclass
class LectureMetadata:
    video_id:          str
    title:             str
    instructor_id:     str
    domain:            str
    duration_sec:      float
    summary:           str
    keywords:          list[dict]
    concept_roles:     list[dict]   # [{"concept": "스레드", "role": "core"}, ...]
    concept_relations: list[dict]   # [{"source": "프로세스", "target": "스레드", ...}, ...]


# ============================================================================
#  메타데이터 컬렉션
# ============================================================================

class MetadataCollection:
    def __init__(self, metadata_dir: str):
        self.lectures: dict[str, LectureMetadata] = {}
        self._load(Path(metadata_dir))

    def _load(self, directory: Path):
        files = list(directory.glob("*_metadata.json"))
        if not files:
            raise FileNotFoundError(f"메타데이터 파일 없음: {directory}")
        for path in files:
            with open(path, encoding="utf-8") as f:
                raw = json.load(f)
            items = raw if isinstance(raw, list) else [raw]
            for item in items:
                lec = LectureMetadata(
                    video_id          = item["video_id"],
                    title             = item["title"],
                    instructor_id     = item.get("instructor_id", ""),
                    domain            = item.get("domain", "unknown"),
                    duration_sec      = item.get("duration_sec", 0.0),
                    summary           = item.get("summary", ""),
                    keywords          = item.get("keywords", []),
                    concept_roles     = item.get("concept_roles", []),
                    concept_relations = item.get("concept_relations", []),
                )
                self.lectures[lec.video_id] = lec
        print(f"[로드] {len(self.lectures)}개 강의 메타데이터 로드 완료\n")

    def get(self, video_id: str) -> Optional[LectureMetadata]:
        return self.lectures.get(video_id)

    def all(self) -> list[LectureMetadata]:
        return list(self.lectures.values())

    def available_domains(self) -> list[str]:
        """메타데이터에 실제 존재하는 도메인 목록"""
        return sorted({lec.domain for lec in self.lectures.values()})

    def available_keywords(self) -> list[str]:
        """메타데이터 전체 keyword 배열에서 unique 키워드 수집.

        Gemini가 이 목록에서만 선택하게 함으로써
        semantic gap(확률 vs 확률 분포)을 원천 제거.

        ※ 강의 수가 수천 개 이상으로 늘어나면 프롬프트 토큰 한계 도달
          → LanceDB 벡터 검색으로 교체 예정.
        """
        return sorted({
            k["keyword"]
            for lec in self.lectures.values()
            for k in lec.keywords
        })


# ============================================================================
#  유사도 계산
# ============================================================================

import re as _re


def _strip_josa(t: str) -> str:
    return _re.sub(r'(의|은|는|이|가|을|를|에서|에|로|으로|과|와|도|만|까지|부터)$', '', t)


def _kw_match(meta_kw: str, query_kws: list[str]) -> bool:
    """양방향 substring 매칭 (pool 선택 후 안전망 역할)."""
    for qk in query_kws:
        if qk in meta_kw or meta_kw in qk:
            return True
    return False


def _concept_match(concept: str, focus: str) -> bool:
    """concept_roles / concept_relations 매칭: 양방향 substring."""
    return focus in concept or concept in focus


def _compute_depth_score(focus_concept: str, target: LectureMetadata) -> float:
    """
    focus_concept이 강의에서 얼마나 깊이 다뤄지는지 추정.

    3가지 신호:
      1. concept_roles  — 해당 개념의 role (core=1.0 / prerequisite=0.5 / introduced=0.1)
      2. concept_relations — 해당 개념이 관여하는 엣지 수 (많을수록 세부 내용 포함)
      3. keywords 하위 개념  — focus와 관련된 더 구체적인 키워드 등장 수

    가중합: role×0.5 + relation×0.3 + subkw×0.2
    """
    if not focus_concept:
        return 0.0

    # 1. role score
    # concept_roles는 두 가지 형식을 모두 지원:
    #   dict: {"core": [...], "introduced": [...]}
    #   list: [{"concept": "...", "role": "..."}]
    role_score = 0.0
    concept_roles = target.concept_roles
    if isinstance(concept_roles, dict):
        ROLE_WEIGHTS = {"core": 1.0, "prerequisite": 0.5, "introduced": 0.1}
        for role, concepts in concept_roles.items():
            weight = ROLE_WEIGHTS.get(role, 0.0)
            for concept in (concepts or []):
                if _concept_match(concept, focus_concept):
                    role_score = max(role_score, weight)
    else:
        for cr in (concept_roles or []):
            if not isinstance(cr, dict):
                continue
            if _concept_match(cr.get("concept", ""), focus_concept):
                role = cr.get("role", "")
                if role == "core":
                    role_score = 1.0
                elif role == "prerequisite":
                    role_score = max(role_score, 0.5)
                elif role == "introduced":
                    role_score = max(role_score, 0.1)

    # 2. relation score — focus와 연결된 엣지 수 (10개 이상 → 1.0)
    # from/to 키와 source/target 키 모두 지원
    edge_count = sum(
        1 for rel in (target.concept_relations or [])
        if _concept_match(rel.get("from", rel.get("source", "")), focus_concept)
        or _concept_match(rel.get("to",   rel.get("target", "")), focus_concept)
    )
    relation_score = min(edge_count / 10.0, 1.0)

    # 3. 하위 키워드 score — focus보다 구체적인 키워드(ex. "스레드 풀") 등장
    sub_kw_count = sum(
        1 for k in target.keywords
        if focus_concept in k["keyword"] and k["keyword"] != focus_concept
    )
    subkw_score = min(sub_kw_count / 3.0, 1.0)

    depth = role_score * 0.5 + relation_score * 0.3 + subkw_score * 0.2
    return round(depth, 4)


def _context_match(meta_kw: str, context_kws: list[str]) -> bool:
    """Context penalty 전용: prefix 방향만 허용.
    "고유벡터".startswith("벡터") → False (false positive 방지)
    "선형대수학".startswith("선형대수") → True
    """
    for qk in context_kws:
        if meta_kw == qk or meta_kw.startswith(qk):
            return True
    return False


def _compute_fragmentation_penalty(concept_roles) -> float:
    """
    강의 내용의 파편화 정도를 측정. (v5 신규)

    핵심(core) 개념 비율이 낮고 도입(introduced) 개념 비율이 높을수록
    내용이 파편적이라고 판단하여 패널티를 부여한다.

    segment_importance.py의 std_score 패널티와 같은 역할:
      std_score가 높으면(점수 변동이 크면) 패널티 → 응집된 강의 선호

    반환: 0.0(응집) ~ 1.0(파편화)
    """
    if isinstance(concept_roles, dict):
        n_core  = len(concept_roles.get("core", []))
        n_intro = len(concept_roles.get("introduced", []))
        n_pre   = len(concept_roles.get("prerequisite", []))
    elif isinstance(concept_roles, list):
        n_core  = sum(1 for cr in concept_roles if isinstance(cr, dict) and cr.get("role") == "core")
        n_intro = sum(1 for cr in concept_roles if isinstance(cr, dict) and cr.get("role") == "introduced")
        n_pre   = sum(1 for cr in concept_roles if isinstance(cr, dict) and cr.get("role") == "prerequisite")
    else:
        return 0.0

    total = n_core + n_intro + n_pre
    if total == 0:
        return 0.0

    core_ratio  = n_core  / total
    intro_ratio = n_intro / total
    # 핵심 비율 낮고 도입 비율 높을수록 패널티 증가
    penalty = intro_ratio * (1.0 - core_ratio)
    return float(min(penalty, 1.0))


def _minmax_normalize(values: list[float]) -> list[float]:
    """
    후보 집합 내 Min-Max 정규화. (v5 신규)

    각 점수 차원(keyword/title/summary)을 독립적으로 정규화하여
    단위·분포가 다른 점수들을 동일 선상에서 비교 가능하게 한다.

    모든 값이 동일하면(분모=0) 전부 1.0으로 반환.
    """
    lo, hi = min(values), max(values)
    if hi == lo:
        return [1.0] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


def _compute_raw_scores(
    target_keywords:  list[str],
    context_keywords: list[str],
    query_type:       str,
    query_domain:     Optional[str],
    focus_concept:    Optional[str],
    target:           LectureMetadata,
    raw_query:        str = "",
) -> dict:
    """
    정규화·가중치 적용 이전의 원시 점수를 계산한다. (v5 내부 함수)

    recommend_from_query에서 후보 전체를 모은 뒤 배치 정규화에 사용.

    반환 키:
      keyword_score, title_score, summary_score  — 원시 [0,1] 점수
      domain_score, depth_score                  — boost 계산용
      context_penalized                          — bool
    """
    domain_score = 1.0 if (query_domain and target.domain == query_domain) else 0.0

    kw_list = target.keywords
    total_w = sum(k["score"] for k in kw_list) or 1.0

    # keyword_score: target_keywords 가중합
    matched_w     = sum(k["score"] for k in kw_list if _kw_match(k["keyword"], target_keywords))
    keyword_score = matched_w / total_w

    # title_score: (1) Gemini 키워드 기반 히트 + (2) 원본 질의어 직접 매칭 블렌딩
    title_tokens  = {_strip_josa(t) for t in target.title.replace(",", " ").replace("·", " ").split()}
    summary_lower = target.summary.lower()

    # (1) Gemini 키워드 기반 히트 (기존 방식)
    title_hit = 0.0
    if target_keywords:
        for qk in target_keywords:
            if any(qk in tok or tok in qk for tok in title_tokens):
                title_hit += 1.0 if qk in summary_lower else 0.1
        gemini_title_score = title_hit / len(target_keywords)
    else:
        gemini_title_score = 0.0

    # (2) 원본 질의어 직접 매칭 — Gemini 키워드 확장과 무관하게
    #     "운영체제 기능을 알고 싶어" → query_tokens: {"운영체제", "기능"}
    #     → 제목에 "운영체제" 포함되면 direct hit
    query_tokens = {
        _strip_josa(t)
        for t in raw_query.replace(",", " ").split()
        if len(t) > 1
    }
    if query_tokens:
        direct_hits = sum(
            1 for qt in query_tokens
            if any(qt in tok or tok in qt for tok in title_tokens)
        )
        direct_title_score = direct_hits / len(query_tokens)
    else:
        direct_title_score = 0.0

    # 7:3 블렌딩 — Gemini 확장 키워드(의미 폭) + 원본 질의어(정확도)
    title_score = 0.70 * gemini_title_score + 0.30 * direct_title_score

    # summary_score: keyword 배열 가중합 + 직접 텍스트 히트 합산
    if target_keywords:
        matched_sum_w  = sum(
            k["score"] for k in kw_list
            if _kw_match(k["keyword"], target_keywords) and k["keyword"] in summary_lower
        )
        kw_summary     = matched_sum_w / total_w
        direct_summary = sum(1 for kw in target_keywords if kw in summary_lower) / len(target_keywords)
        summary_score  = min(kw_summary + direct_summary * 0.5, 1.0)
    else:
        summary_score = 0.0

    # depth score
    depth_score = _compute_depth_score(focus_concept, target) if focus_concept else 0.0

    # context penalty 여부
    context_penalized = False
    if context_keywords and query_type in CONTEXT_PENALTY_BY_TYPE:
        if any(_context_match(k["keyword"], context_keywords) for k in kw_list):
            context_penalized = True

    return {
        "keyword_score":    keyword_score,
        "title_score":      title_score,
        "summary_score":    summary_score,
        "domain_score":     domain_score,
        "depth_score":      depth_score,
        "context_penalized": context_penalized,
    }


def compute_query_similarity(
    target_keywords:  list[str],
    context_keywords: list[str],
    query_type:       str,
    query_domain:     Optional[str],
    focus_concept:    Optional[str],
    target:           LectureMetadata,
    cfg:              RecommenderConfig,
    raw_query:        str = "",
) -> dict:
    """
    단일 강의에 대한 최종 점수를 계산한다.

    recommend_from_query는 배치 정규화를 위해 이 함수 대신
    _compute_raw_scores → 정규화 → 가중합 순서로 호출한다.
    이 함수는 단일 강의 테스트·디버깅용으로 유지한다.
    (정규화 없이 raw 가중합으로 계산하므로 추천 결과와 점수가 다를 수 있음)

    content_score = w_k·keyword + w_t·title + w_s·summary - λ·frag
    score = content_score × domain_boost × depth_boost [× context_penalty]
    """
    if not target_keywords:
        return {"score": 0.0, "query_type": query_type,
                "domain_score": 0.0, "domain_boost": 1.0,
                "keyword_score": 0.0, "title_score": 0.0,
                "summary_score": 0.0, "depth_score": 0.0, "depth_boost": 1.0,
                "frag_penalty": 0.0, "context_penalty": False}

    raw = _compute_raw_scores(
        target_keywords, context_keywords, query_type,
        query_domain, focus_concept, target, raw_query
    )

    w_k, w_t, w_s, w_d, w_dp = INTENT_WEIGHTS.get(query_type, INTENT_WEIGHTS["unknown"])
    domain_boost_str = w_d
    frag    = _compute_fragmentation_penalty(target.concept_roles)
    content = max(
        w_k * raw["keyword_score"]
        + w_t * raw["title_score"]
        + w_s * raw["summary_score"]
        - cfg.FRAG_PENALTY_WEIGHT * frag,
        0.0
    )

    domain_boost = 1.0 + domain_boost_str * raw["domain_score"]
    depth_boost  = 1.0 + cfg.W_DEPTH_BOOST * raw["depth_score"]
    total        = content * domain_boost * depth_boost

    if raw["context_penalized"]:
        total *= CONTEXT_PENALTY_BY_TYPE[query_type]

    return {
        "score":           round(total, 4),
        "query_type":      query_type,
        "domain_score":    round(raw["domain_score"], 4),
        "domain_boost":    round(domain_boost, 4),
        "keyword_score":   round(raw["keyword_score"], 4),
        "title_score":     round(raw["title_score"], 4),
        "summary_score":   round(raw["summary_score"], 4),
        "depth_score":     round(raw["depth_score"], 4),
        "depth_boost":     round(depth_boost, 4),
        "frag_penalty":    round(frag, 4),
        "context_penalty": raw["context_penalized"],
    }


# ============================================================================
#  Gemini — 자연어 질의 분석
# ============================================================================

def analyze_query(
    query:              str,
    available_domains:  list[str],
    available_keywords: list[str],
) -> tuple[str, list[str], list[str], Optional[str], Optional[str]]:
    """
    질의 유형·키워드·도메인·focus_concept을 Gemini 1회 호출로 추출.

    target/context_keywords, focus_concept은 available_keywords pool에서만 선택.
    → semantic gap 없음, hallucination 방지.

    반환:
      query_type       : "direct" | "followup" | "career" | "unknown"
      target_keywords  : 추천 대상 강의를 찾을 키워드 (pool 내 선택)
      context_keywords : 억제할 맥락 키워드 (pool 내 선택)
      domain           : available_domains 중 하나, 해당 없으면 None
      focus_concept    : 깊이를 측정할 핵심 개념 (pool 내 선택), 없으면 None
    """
    domain_list  = ", ".join(available_domains)
    keyword_list = ", ".join(available_keywords)

    prompt = f"""다음 강의 검색 질의를 분석해줘.

질의: "{query}"

다음 JSON 형식으로만 출력해 (설명 없이):
{{
  "query_type": "질의 유형",
  "target_keywords": ["키워드1", ...],
  "context_keywords": ["키워드1", ...],
  "domain": "도메인 문자열 또는 null",
  "focus_concept": "개념 문자열 또는 null"
}}

──────────────────────────────────────────────────────
[query_type] 네 가지 중 하나:

  "direct"   특정 주제 강의를 직접 요청
             예) "운영체제 강의 추천해줘", "스레드 자세히 설명해주는 강의"
             문제 상황·증상으로 시작해도 기술 키워드로 변환하여 direct 처리
             예) "비동기 처리할 때 막혀" → target: [Promise, 비동기, 이벤트루프]

  "followup" 이미 들은 강의 이후 연계 강의 요청
             예) "선형대수 벡터 강의 들었는데 다음은?"
             target = 다음 단계, context = 들은 강의

  "career"   직무·진로 목표 기반 강의 요청
             예) "AI 연구자가 되고 싶어", "백엔드 취업 준비"

  "unknown"  강의 추천 의도가 없는 질의
             예) "운영체제가 뭐야?", "안녕", "파이썬 코드 짜줘"
             → target_keywords = [], context_keywords = [], domain = null, focus_concept = null

──────────────────────────────────────────────────────
[target_keywords] — 추천받을 강의를 찾기 위한 키워드:
- 반드시 아래 [키워드 목록]에서만 선택 (최대 8개)
- direct/career: 질의 의도에 맞는 관련 키워드를 넉넉하게 선택
- followup: 들은 강의의 다음 단계 키워드 (들은 강의 키워드 제외)

★ [중요] broad 질의 처리 규칙:
  "운영체제 기능", "파이썬 기초", "자료구조 개요"처럼
  상위 개념 자체를 묻는 질의는 해당 개념 키워드("운영체제", "파이썬", "자료구조")를
  target_keywords의 첫 번째에 반드시 포함하라.
  세부 하위 개념(프로세스, 메모리, 스케줄링 등)만 나열하지 말 것.

[context_keywords] — 억제할 맥락 키워드:
- 반드시 아래 [키워드 목록]에서만 선택 (최대 5개)
- direct/career: 항상 []
- followup: 이미 들은 강의 관련 키워드

[domain] — 추천받을 강의의 도메인:
- 반드시 아래 목록 중 하나: {domain_list}
- 확신할 수 없으면 null

[focus_concept] — 깊이를 측정할 핵심 개념:
- 질의가 특정 개념의 '자세한 설명', '깊은 이해', '원리', '동작 방식'을 명시적으로 요청할 때만 설정
- 반드시 아래 [키워드 목록]에서만 선택, 해당 없으면 null
- 예) "스레드 자세히 설명해주는 강의" → "스레드"
- 예) "딥러닝 원리 설명해주는 강의" → "딥러닝"
- 예) "운영체제 강의 추천해줘" → null (깊이 특정 없음)

──────────────────────────────────────────────────────
[키워드 목록] — target/context_keywords, focus_concept은 이 중에서만 선택:
{keyword_list}"""

    response = _client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config={"temperature": 0.0},
    )
    text   = response.text.strip().replace("```json", "").replace("```", "").strip()
    parsed = json.loads(text)

    query_type       = parsed.get("query_type", "direct")
    target_keywords  = parsed.get("target_keywords", [])
    context_keywords = parsed.get("context_keywords", [])
    domain           = parsed.get("domain") or None
    focus_concept    = parsed.get("focus_concept") or None

    # pool 외 키워드 필터링
    kw_set           = set(available_keywords)
    target_keywords  = [k for k in target_keywords  if k in kw_set]
    context_keywords = [k for k in context_keywords if k in kw_set]
    if focus_concept and focus_concept not in kw_set:
        focus_concept = None

    if query_type not in QUERY_TYPES:
        query_type = "direct"
    if domain not in available_domains:
        domain = None

    return query_type, target_keywords, context_keywords, domain, focus_concept


# ============================================================================
#  추천 결과
# ============================================================================

@dataclass
class RecommendResult:
    video_id:     str
    title:        str
    domain:       str
    instructor:   str
    score:        float
    score_detail: dict
    reason:       str
    summary:      str


def _build_reason(detail: dict) -> str:
    parts = []
    if detail.get("context_penalty"):
        parts.append("⚑ 맥락 억제")
    if detail.get("domain_score", 0) == 1.0:
        parts.append("도메인 일치")
    if detail.get("depth_score", 0) > 0.3:
        parts.append(f"개념 깊이 {detail['depth_score']:.0%}")
    if detail.get("title_score", 0) >= 0.5:
        parts.append(f"제목 일치 {detail['title_score']:.0%}")
    if detail.get("keyword_score", 0) > 0.05:
        parts.append(f"키워드 {detail['keyword_score']:.0%}")
    if detail.get("summary_score", 0) > 0.3:
        parts.append(f"요약 일치 {detail['summary_score']:.0%}")
    if detail.get("frag_penalty", 0) > 0.3:
        parts.append(f"파편화 -{detail['frag_penalty']:.0%}")
    return " · ".join(parts) if parts else "관련 강의"


# ============================================================================
#  추천 엔진
# ============================================================================

class Recommender:
    def __init__(self, metadata_dir: str = "metadata/", config: Optional[RecommenderConfig] = None):
        self.collection          = MetadataCollection(metadata_dir)
        self.cfg                 = config or RecommenderConfig()
        self._available_domains  = self.collection.available_domains()
        self._available_keywords = self.collection.available_keywords()
        print(f"[도메인]    {self._available_domains}")
        print(f"[키워드 풀] {len(self._available_keywords)}개\n")

    def recommend_from_query(
        self,
        query:     str,
        top_k:     int            = 5,
        min_score: Optional[float] = None,
    ) -> list[RecommendResult]:
        """
        자연어 질의를 분석하고 관련 강의를 추천한다.

        Args:
            query     : 자연어 질의
            top_k     : 최대 반환 강의 수
            min_score : 이 점수 미만 강의 제외 (None이면 cfg.MIN_SCORE 사용)

        v5 변경:
          1. _compute_raw_scores로 후보 전체 원시 점수 수집
          2. keyword / title / summary 각각 Min-Max 정규화
          3. INTENT_WEIGHTS 동적 가중치 + 파편화 패널티 적용
          4. min_score 미만 강의 필터링 후 top_k 반환
        """
        print(f"[질의 분석] {query}")
        query_type, target_kw, context_kw, domain, focus_concept = analyze_query(
            query, self._available_domains, self._available_keywords
        )
        print(f"[질의 유형]   {query_type}")
        print(f"[대상 키워드] {target_kw}")
        print(f"[맥락 키워드] {context_kw}")
        print(f"[추론 도메인] {domain or '미확정'}")
        print(f"[깊이 개념]   {focus_concept or '없음'}\n")

        if query_type == "unknown":
            print("  ※ 강의 추천 질의가 아닙니다.")
            return []

        threshold = min_score if min_score is not None else self.cfg.MIN_SCORE

        # ── Pass 1: 후보 전체 원시 점수 수집 ──────────────────────────
        all_targets  = self.collection.all()
        raw_scores   = [
            _compute_raw_scores(
                target_kw, context_kw, query_type,
                domain, focus_concept, t, query
            )
            for t in all_targets
        ]

        # ── Pass 2: 차원별 Min-Max 정규화 ─────────────────────────────
        kw_norm = _minmax_normalize([r["keyword_score"] for r in raw_scores])
        tt_norm = _minmax_normalize([r["title_score"]   for r in raw_scores])
        ss_norm = _minmax_normalize([r["summary_score"] for r in raw_scores])

        # ── Pass 3: 가중합 + boost + penalty → 최종 점수 ──────────────
        w_k, w_t, w_s, w_d, w_dp = INTENT_WEIGHTS.get(
            query_type, INTENT_WEIGHTS["unknown"]
        )
        λ = self.cfg.FRAG_PENALTY_WEIGHT

        candidates = []
        for i, (target, raw) in enumerate(zip(all_targets, raw_scores)):
            # 1. 가중합 기반의 기본 내용 점수 (0.0 ~ 1.0)
            # 기존 가중치에 도메인과 깊이 점수를 합산 항목으로 포함
            total = (
                w_k * kw_norm[i] +
                w_t * tt_norm[i] +
                w_s * ss_norm[i] +
                w_d * raw["domain_score"] +   # 도메인 일치 시 가산
                w_dp * raw["depth_score"]     # 깊이 점수만큼 가산
            )

            # 2. 파편화 패널티 차감
            frag = _compute_fragmentation_penalty(target.concept_roles)
            total = max(total - (λ * frag), 0.0)

            # 3. 맥락 억제 (Followup 시 이미 들은 강의 감점)
            if raw["context_penalized"]:
                total *= CONTEXT_PENALTY_BY_TYPE[query_type]

            # 4. (중요) 기존에 에러가 났던 detail 기록용 변수 설정
            # 가중합 방식에서는 곱연산 boost가 없으므로, 
            # '실제로 얼마나 가산되었는지'를 시각적으로 보여주는 용도로 정의합니다.
            domain_boost = 1.0 + (w_d * raw["domain_score"])
            depth_boost  = 1.0 + (w_dp * raw["depth_score"])

            detail = {
                "score":              round(total, 4),
                "query_type":         query_type,
                "keyword_score":      round(raw["keyword_score"], 4),
                "title_score":        round(raw["title_score"],   4),
                "summary_score":      round(raw["summary_score"], 4),
                "keyword_score_norm": round(kw_norm[i], 4),
                "title_score_norm":   round(tt_norm[i], 4),
                "summary_score_norm": round(ss_norm[i], 4),
                "domain_score":       round(raw["domain_score"],  4),
                "domain_boost":       round(domain_boost, 4),  # 이제 에러가 나지 않습니다!
                "depth_score":        round(raw["depth_score"],   4),
                "depth_boost":        round(depth_boost, 4),   # 이제 에러가 나지 않습니다!
                "frag_penalty":       round(frag, 4),
                "context_penalty":    raw["context_penalized"],
            }
            candidates.append((target, detail))

        # ── 점수 기준 내림차순 정렬 ────────────────────────────────
        candidates.sort(key=lambda x: x[1]["score"], reverse=True)

        results = []
        for t, d in candidates:
            if d["score"] < threshold:
                break   # 정렬 후이므로 이후 항목도 모두 미달 → 조기 종료
            results.append(RecommendResult(
                video_id     = t.video_id,
                title        = t.title,
                domain       = t.domain,
                instructor   = t.instructor_id,
                score        = d["score"],
                score_detail = d,
                reason       = _build_reason(d),
                summary      = t.summary,
            ))
            if len(results) >= top_k:
                break

        return results

    def print_results(self, results: list[RecommendResult], title: str = "추천 결과", top_k: int = 3):
        # results는 이미 min_score 필터링·top_k 적용 완료
        top = results[:top_k]

        print(f"\n{'='*62}")
        print(f"  {title}")
        print(f"{'='*62}")

        if not top:
            print("  ※ 질의와 일치하는 강의를 찾지 못했습니다.")
        else:
            qtype = top[0].score_detail.get("query_type", "direct")
            w_k, w_t, w_s, w_d, w_dp = INTENT_WEIGHTS.get(qtype, INTENT_WEIGHTS["unknown"])
            print(f"  ▶ 추천 강의 (Top {len(top)})  [유형: {qtype}]")
            print(f"  가중치: keyword {w_k:.0%}  title {w_t:.0%}  summary {w_s:.0%}")
            print(f"  {'-'*58}")
            for i, r in enumerate(top, 1):
                d         = r.score_detail
                score_100 = round(r.score * 100, 1)
                # 정규화된 값으로 기여도 표시
                k_contrib = round(w_k * d.get("keyword_score_norm", d.get("keyword_score", 0)) * 100, 1)
                t_contrib = round(w_t * d.get("title_score_norm",   d.get("title_score",   0)) * 100, 1)
                s_contrib = round(w_s * d.get("summary_score_norm", d.get("summary_score", 0)) * 100, 1)
                boost_pct = round((d.get("domain_boost", 1.0) - 1.0) * 100, 1)
                depth_pct = round((d.get("depth_boost",  1.0) - 1.0) * 100, 1)
                frag_pct  = round(d.get("frag_penalty", 0.0) * 100, 1)
                penalty   = "  ※억제됨" if d.get("context_penalty") else ""
                print(f"  {i}. [{r.video_id}] {r.title}{penalty}")
                print(f"     점수:   {score_100}점  |  {r.reason}")
                print(f"     기여:   keyword {k_contrib}점  title {t_contrib}점  "
                      f"summary {s_contrib}점  domain_boost +{boost_pct}%  depth_boost +{depth_pct}%")
                if frag_pct > 0:
                    print(f"     파편화 패널티: -{frag_pct}%")
                print(f"     요약:   {r.summary[:80]}...")
                print()

        print(f"  {'─'*58}")
        print(f"  ▶ 추천된 강의 점수 전체 (min_score 필터 적용됨)")
        print(f"  {'─'*58}")
        print(f"  {'video_id':<12} {'제목':<22} {'도메인':<18} {'점수':>6} {'억제':>4}")
        print(f"  {'─'*58}")
        for r in results:
            marker  = " ◀" if r in top else ""
            penalty = "  ⚑" if r.score_detail.get("context_penalty") else ""
            print(f"  {r.video_id:<12} {r.title[:20]:<22} {r.domain:<18} "
                  f"{round(r.score*100,1):>5.1f}점{marker}{penalty}")
        print()


# ============================================================================
#  CLI
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="강의 추천 시스템",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""예시:
  python recommender.py --query "메모리 관리 방법 알고 싶어"
  python recommender.py --metadata_dir metadata/ --query "운영체제란 무엇인가"
  python recommender.py --query "운영체제 기능 알고 싶어" --min_score 0.1
"""
    )
    parser.add_argument("--metadata_dir", default="metadata/",
                        help="메타데이터 디렉토리 (기본: metadata/)")
    parser.add_argument("--query", default=None,
                        help="자연어 질의 (예: '메모리 관리 알고 싶어')")
    parser.add_argument("--top_k", type=int, default=5,
                        help="추천 결과 수 (기본: 5)")
    parser.add_argument("--min_score", type=float, default=None,
                        help="추천 최소 점수 0~1 (기본: cfg.MIN_SCORE=0.1). "
                             "0 지정 시 필터 없음.")
    args = parser.parse_args()

    rec = Recommender(metadata_dir=args.metadata_dir)

    if args.query:
        results = rec.recommend_from_query(
            args.query, top_k=args.top_k, min_score=args.min_score
        )
        rec.print_results(results, f"질의 기반 추천: '{args.query}'", top_k=3)
    else:
        print("대화형 모드 (종료: q)\n")
        while True:
            try:
                query = input("질의 입력 > ").strip()
                if query.lower() in ("q", "quit", "exit"):
                    break
                if not query:
                    continue
                results = rec.recommend_from_query(
                    query, top_k=args.top_k, min_score=args.min_score
                )
                rec.print_results(results, f"질의: '{query}'", top_k=3)
            except KeyboardInterrupt:
                print("\n종료")
                break