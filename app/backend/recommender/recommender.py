"""
recommender.py
──────────────
강의 추천 시스템 — 자연어 질의 기반

변경 이력:
  v3: Gemini가 메타데이터 keyword pool에서 직접 선택 (자유 생성 → vocabulary 고정)
  v4: prerequisite 타입 제거, focus_concept → depth_score
  v5: 인텐트별 동적 가중치, Min-Max 정규화, 파편화 패널티, MIN_SCORE 필터
  v6: 벡터 유사도 방식 전환 (ko-sroberta + LanceDB)
  v7: 필드별 분리 벡터 + Gemini search_text 변환
  v8: 직접 매칭(dm_score) 추가 — 벡터 유사도 보조
      content_score = VEC_BLEND×vec_score + (1-VEC_BLEND)×dm_score
  v9: 임베딩 모델 교체 (ko-sroberta → gemini-embedding-001)
      SentenceTransformer 제거, Gemini API 단일 클라이언트로 통합
      keyword vec threshold 필터 적용 (sim_keyword < 0.60 → 기여 0)

CLI 실행:
  python recommender.py --query "메모리 관리 방법 알고 싶어"
  python recommender.py --metadata_dir app/backend/metadata --query "운영체제란 무엇인가"

모듈로 사용:
  from recommender.recommender import Recommender
  rec = Recommender("app/backend/metadata")
  results = rec.recommend_from_query("프로세스 스케줄링 알고 싶어")
"""

import json
import os
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import numpy as np
import lancedb
from google import genai
from dotenv import load_dotenv

load_dotenv()
_client        = genai.Client(api_key=os.getenv("GOOGLE_API_KEY_2"))
GEMINI_MODEL   = "gemini-2.5-flash"
EMBED_MODEL    = "gemini-embedding-001"


def _resolve_repo_root() -> Path:
    """Resolve project root for both local and container runs."""
    env_root = os.getenv("GRAPHLEC_ROOT") or os.getenv("PIPELINE_ROOT")
    if env_root:
        return Path(env_root).resolve()
    here = Path(__file__).resolve()
    # Local path: <repo>/app/backend/recommender/recommender.py
    # Container path: /app/recommender/recommender.py
    return here.parents[3] if len(here.parents) > 3 else here.parents[1]


_REPO_ROOT     = _resolve_repo_root()
DEFAULT_METADATA_DIR = str(_REPO_ROOT / "app" / "backend" / "metadata")
DEFAULT_DB_DIR = str(_REPO_ROOT / "data" / "lancedb")


# ============================================================================
#  설정
# ============================================================================

@dataclass
class RecommenderConfig:
    # 필드별 가중치 (합계 = 1.0)
    W_TITLE:             float = 0.20
    W_KEYWORD:           float = 0.55
    W_SUMMARY:           float = 0.25
    # 벡터 유사도 vs 직접 매칭 블렌딩 비율 (벡터:직접 = VEC_BLEND : 1-VEC_BLEND)
    VEC_BLEND:           float = 0.70
    # keyword vec 유사도 threshold — 미만이면 기여 0으로 처리
    KW_VEC_THRESHOLD:    float = 0.60
    # depth boost (focus_concept 지정 시에만 활성)
    W_DEPTH_BOOST:       float = 0.30
    # 파편화 패널티 강도 λ
    FRAG_PENALTY_WEIGHT: float = 0.10
    # 추천 최소 점수 — 이 점수 이하인 강의는 추천 결과에서 제외
    MIN_SCORE:           float = 0.30
    # 벡터 DB 경로
    DB_DIR:              str   = DEFAULT_DB_DIR
    # domain boost 강도
    W_DOMAIN_BOOST:      float = 0.20
    # difficulty boost 강도 (질의 난이도 힌트 일치 시)
    W_DIFFICULTY_BOOST:  float = 0.15


# ============================================================================
#  데이터 모델
# ============================================================================

@dataclass
class LectureMetadata:
    video_id:          str
    title:             str
    instructor_id:     str
    domain:            str
    difficulty:        str   # "beginner" | "intermediate" | "advanced"
    duration_sec:      float
    summary:           str
    keywords:          list[dict]
    concept_roles:     list[dict]
    concept_relations: list[dict]


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
                    difficulty        = item.get("difficulty", "unknown"),
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
        return sorted({lec.domain for lec in self.lectures.values()})

    def available_keywords(self) -> list[str]:
        """focus_concept 선택용 keyword pool"""
        return sorted({
            k["keyword"]
            for lec in self.lectures.values()
            for k in lec.keywords
        })


# ============================================================================
#  임베딩
# ============================================================================

def _embed(text: str) -> list[float]:
    """단일 텍스트 → Gemini embedding-001 벡터."""
    text = text.strip() or " "
    response = _client.models.embed_content(
        model    = EMBED_MODEL,
        contents = text,
    )
    return response.embeddings[0].values


# ============================================================================
#  점수 계산 유틸
# ============================================================================

def _cosine_sim(a: list[float], b: list[float]) -> float:
    """두 벡터 간 코사인 유사도."""
    va, vb = np.array(a), np.array(b)
    denom  = np.linalg.norm(va) * np.linalg.norm(vb)
    if denom == 0:
        return 0.0
    return float(np.dot(va, vb) / denom)


def _direct_match_score(
    query_keywords:    list[str],
    inferred_keywords: list[str],
    lec:               LectureMetadata,
    inferred_weight:   float = 0.5,
) -> dict:
    """
    원본 질의 키워드 / 추론 키워드를 각 필드에 직접 매칭.

    query_keywords    → 100% 반영 (원본 질의에서 직접 추출)
    inferred_keywords → inferred_weight 비율 반영 (Gemini 확장, 기본 50%)

    keyword 매칭: 완전 일치 또는 토큰이 keyword의 prefix인 경우만 허용
    """
    q_tokens = set(query_keywords)
    i_tokens = set(inferred_keywords)
    all_tokens = q_tokens | i_tokens
    if not all_tokens:
        return {"title": 0.0, "keyword": 0.0, "summary": 0.0}

    # ── title 매칭 ─────────────────────────────────────────────────
    title_words = set(lec.title.split())
    def title_hit(tokens: set) -> float:
        hits = sum(1 for t in tokens if any(t in w or w in t for w in title_words))
        return hits / len(tokens) if tokens else 0.0

    title_match = (
        title_hit(q_tokens) * 1.0 +
        title_hit(i_tokens) * inferred_weight
    ) / (1.0 + inferred_weight) if (q_tokens or i_tokens) else 0.0

    # ── keyword 매칭 — 완전 일치 또는 prefix, 2글자 이상 토큰만 ────
    total_kw_score = sum(k["score"] for k in lec.keywords) or 1.0

    def kw_matched_score(tokens: set) -> float:
        return sum(
            k["score"] for k in lec.keywords
            if any(
                t == k["keyword"] or k["keyword"].startswith(t)
                for t in tokens
                if len(t) > 1
            )
        )

    q_kw  = kw_matched_score(q_tokens)
    i_kw  = kw_matched_score(i_tokens)
    kw_match = (q_kw * 1.0 + i_kw * inferred_weight) / (total_kw_score * (1.0 + inferred_weight))

    # ── summary 매칭 ───────────────────────────────────────────────
    def summary_hit(tokens: set) -> float:
        hits = sum(1 for t in tokens if t in lec.summary)
        return hits / len(tokens) if tokens else 0.0

    sum_match = (
        summary_hit(q_tokens) * 1.0 +
        summary_hit(i_tokens) * inferred_weight
    ) / (1.0 + inferred_weight) if (q_tokens or i_tokens) else 0.0

    return {"title": title_match, "keyword": kw_match, "summary": sum_match}


def _concept_match(concept: str, focus: str) -> bool:
    return focus in concept or concept in focus


def _compute_depth_score(focus_concept: str, target: LectureMetadata) -> float:
    """
    focus_concept이 강의에서 얼마나 깊이 다뤄지는지 추정.
    가중합: role×0.5 + relation×0.3 + subkw×0.2
    """
    if not focus_concept:
        return 0.0

    role_score    = 0.0
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

    edge_count = sum(
        1 for rel in (target.concept_relations or [])
        if _concept_match(rel.get("from", rel.get("source", "")), focus_concept)
        or _concept_match(rel.get("to",   rel.get("target", "")), focus_concept)
    )
    relation_score = min(edge_count / 10.0, 1.0)

    sub_kw_count = sum(
        1 for k in target.keywords
        if focus_concept in k["keyword"] and k["keyword"] != focus_concept
    )
    subkw_score = min(sub_kw_count / 3.0, 1.0)

    depth = role_score * 0.5 + relation_score * 0.3 + subkw_score * 0.2
    return round(depth, 4)


def _compute_fragmentation_penalty(concept_roles) -> float:
    """
    core 비율 낮고 introduced 비율 높을수록 패널티.
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
    return float(min(intro_ratio * (1.0 - core_ratio), 1.0))


# ============================================================================
#  Gemini — 자연어 질의 분석
# ============================================================================

def analyze_query(
    query:              str,
    available_domains:  list[str],
    available_keywords: list[str],
) -> tuple[str, list[str], list[str], Optional[str], Optional[str], Optional[int], Optional[str]]:
    """
    질의 → search_text + query_keywords + inferred_keywords + domain + focus_concept + duration_max_sec + difficulty_hint 추출.

    반환:
      search_text        : 벡터 임베딩용 전체 텍스트 (query + inferred 합산)
      query_keywords     : 원본 질의에서 직접 추출한 핵심 용어 (dm 100% 반영)
      inferred_keywords  : Gemini가 의미 확장한 연관 용어 (dm 50% 반영)
      domain             : available_domains 중 하나, 없으면 None
      focus_concept      : 깊이를 측정할 핵심 개념, 없으면 None
      duration_max_sec   : 최대 강의 길이(초), 언급 없으면 None
      difficulty_hint    : "beginner" | "intermediate" | "advanced" | None
    """
    domain_list  = ", ".join(available_domains)
    keyword_list = ", ".join(available_keywords)

    prompt = f"""다음 강의 검색 질의를 분석해줘.

질의: "{query}"

다음 JSON 형식으로만 출력해 (설명 없이):
{{
  "query_keywords": ["원본 질의 핵심 용어1", ...],
  "inferred_keywords": ["확장 연관 용어1", ...],
  "domain": "도메인 문자열 또는 null",
  "focus_concept": "개념 문자열 또는 null",
  "duration_max_sec": 숫자 또는 null,
  "difficulty_hint": "beginner" 또는 "intermediate" 또는 "advanced" 또는 null
}}

[query_keywords]: 원본 질의에서 직접 등장하는 핵심 학술·기술 용어
- "찾아줘", "알려줘", "강의", "어떻게" 같은 메타·구어체 표현 제외
- 질의에 명시된 개념만 포함 (1~4개)
- 예) "가상 메모리 페이징 방식 설명하는 강의" → ["가상 메모리", "페이징"]
- 예) "TCP와 UDP 차이를 다루는 강의" → ["TCP", "UDP"]
- 예) "비동기 처리할 때 막혀" → ["비동기"]

[inferred_keywords]: 질의 의도에서 연관성이 높은 확장 용어 (2~6개)
- query_keywords와 겹치지 않을 것
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

[difficulty_hint]: 질의에서 난이도·범위 표현을 감지하여 추출
  - "입문", "기초", "개론", "개요", "훑어주는", "전반", "처음", "쉽게" 등 → "beginner"
  - "심화", "자세히", "깊게", "원리", "내부 동작" 등 → "advanced"
  - "응용", "실습", "프로젝트" 등 중급 신호 → "intermediate"
  - 난이도·범위 표현 없으면 → null

[키워드 목록]: {keyword_list}"""

    response = _client.models.generate_content(
        model    = GEMINI_MODEL,
        contents = prompt,
        config   = {"temperature": 0.0},
    )
    text   = response.text.strip().replace("```json", "").replace("```", "").strip()
    parsed = json.loads(text)

    query_keywords    = parsed.get("query_keywords", [])
    inferred_keywords = parsed.get("inferred_keywords", [])
    domain            = parsed.get("domain") or None
    focus_concept     = parsed.get("focus_concept") or None
    duration_max_sec  = parsed.get("duration_max_sec") or None
    difficulty_hint   = parsed.get("difficulty_hint") or None

    # 벡터 임베딩용 search_text — 전체 합산
    search_text = " ".join(query_keywords + inferred_keywords) or query

    kw_set = set(available_keywords)
    if domain not in available_domains:
        domain = None
    if focus_concept and focus_concept not in kw_set:
        focus_concept = None
    if duration_max_sec is not None:
        try:
            duration_max_sec = int(duration_max_sec)
        except (ValueError, TypeError):
            duration_max_sec = None
    if difficulty_hint not in ("beginner", "intermediate", "advanced"):
        difficulty_hint = None

    return search_text, query_keywords, inferred_keywords, domain, focus_concept, duration_max_sec, difficulty_hint


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
    duration_sec: float
    score_detail: dict
    reason:       str
    summary:      str


def _build_reason(detail: dict) -> str:
    parts = []
    if detail.get("domain_score", 0) == 1.0:
        parts.append("도메인 일치")
    if detail.get("depth_score", 0) > 0.3:
        parts.append(f"개념 깊이 {detail['depth_score']:.0%}")
    if detail.get("sim_keyword", 0) >= 0.6:
        parts.append(f"키워드 유사도 {detail['sim_keyword']:.0%}")
    if detail.get("dm_keyword", 0) > 0.1:
        parts.append(f"키워드 직접 매칭 {detail['dm_keyword']:.0%}")
    if detail.get("frag_penalty", 0) > 0.3:
        parts.append(f"파편화 -{detail['frag_penalty']:.0%}")
    return " · ".join(parts) if parts else "관련 강의"


# ============================================================================
#  추천 엔진
# ============================================================================

class Recommender:
    def __init__(self, metadata_dir: str = DEFAULT_METADATA_DIR, config: Optional[RecommenderConfig] = None):
        self.collection          = MetadataCollection(metadata_dir)
        self.cfg                 = config or RecommenderConfig()
        self._available_domains  = self.collection.available_domains()
        self._available_keywords = self.collection.available_keywords()
        # LanceDB 전체 레코드 사전 로드 (요청마다 디스크 읽기 방지)
        print("[LanceDB 레코드 로드 중...]")
        db               = lancedb.connect(self.cfg.DB_DIR)
        table            = db.open_table("lectures")
        self._index_rows = table.to_arrow().to_pylist()
        print(f"  → {len(self._index_rows)}개 레코드 로드\n")
        print(f"[도메인]    {self._available_domains}")
        print(f"[키워드 풀] {len(self._available_keywords)}개\n")

    def recommend_from_query(
        self,
        query:     str,
        top_k:     int             = 5,
        min_score: Optional[float] = None,
    ) -> list[RecommendResult]:
        """
        자연어 질의를 분석하고 관련 강의를 추천한다.

        v9 흐름:
          1. analyze_query — search_text / domain / focus_concept 추출
          2. search_text → Gemini embedding-001 벡터화
          3. 모든 강의에 대해:
             - 필드별 코사인 유사도 (vec_score), keyword threshold 필터 적용
             - 필드별 직접 토큰 매칭 (dm_score)
             - content_score = VEC_BLEND×vec_score + (1-VEC_BLEND)×dm_score
          4. domain_boost × depth_boost 적용, frag_penalty 차감
          5. 정렬 + min_score 필터 후 top_k 반환
        """
        print(f"[질의 분석] {query}")
        search_text, query_keywords, inferred_keywords, domain, focus_concept, duration_max_sec, difficulty_hint = analyze_query(
            query, self._available_domains, self._available_keywords
        )
        print(f"[원본 키워드] {query_keywords}")
        print(f"[확장 키워드] {inferred_keywords}")
        print(f"[추론 도메인] {domain or '미확정'}")
        print(f"[깊이 개념]   {focus_concept or '없음'}")
        print(f"[난이도 힌트] {difficulty_hint or '없음'}")
        if duration_max_sec:
            print(f"[길이 조건]   기준 {duration_max_sec//60}분 ({duration_max_sec}초) — 소프트 패널티 적용")
        print()

        threshold = min_score if min_score is not None else self.cfg.MIN_SCORE

        # ── 질의 벡터화 ───────────────────────────────────────────────
        query_vec = _embed(search_text)

        # ── 강의별 점수 계산 ──────────────────────────────────────────
        candidates = []
        for row in self._index_rows:
            lec = self.collection.get(row["video_id"])
            if lec is None:
                continue

            # ── 길이 소프트 패널티 ────────────────────────────────────
            # 기준 ±5분(300초) 이내 → 1.0
            # 초과량에 따라 선형 감쇄 → 최소 0.1
            if duration_max_sec:
                over_sec = lec.duration_sec - (duration_max_sec + 300)
                if over_sec <= 0:
                    duration_score = 1.0
                else:
                    # 300초(5분) 초과부터 감쇄, 1200초(20분) 초과 시 0.1
                    duration_score = max(1.0 - (over_sec / 1200) * 0.9, 0.1)
            else:
                duration_score = 1.0

            # 필드별 코사인 유사도
            sim_title   = _cosine_sim(query_vec, row["title_vec"])
            sim_keyword = _cosine_sim(query_vec, row["keyword_vec"])
            sim_summary = _cosine_sim(query_vec, row["summary_vec"])

            # keyword vec threshold 필터
            sim_keyword_filtered = (
                sim_keyword if sim_keyword >= self.cfg.KW_VEC_THRESHOLD else 0.0
            )
            vec_score = (
                self.cfg.W_TITLE   * sim_title            +
                self.cfg.W_KEYWORD * sim_keyword_filtered +
                self.cfg.W_SUMMARY * sim_summary
            )

            # 직접 토큰 매칭 — 원본 키워드 100%, 추론 키워드 50% 반영
            dm       = _direct_match_score(query_keywords, inferred_keywords, lec)
            dm_score = (
                self.cfg.W_TITLE   * dm["title"]   +
                self.cfg.W_KEYWORD * dm["keyword"]  +
                self.cfg.W_SUMMARY * dm["summary"]
            )

            # 블렌딩 — content 최대 0.85로 제한 (boost 여유 확보)
            content_score = min(
                self.cfg.VEC_BLEND       * vec_score +
                (1 - self.cfg.VEC_BLEND) * dm_score,
                0.85
            )

            # domain boost 신호
            domain_score = 1.0 if (domain and lec.domain == domain) else 0.0

            # difficulty boost 신호
            difficulty_match = 1.0 if (difficulty_hint and lec.difficulty == difficulty_hint) else 0.0

            # depth boost 신호
            depth_score = _compute_depth_score(focus_concept, lec) if focus_concept else 0.0

            # ── 가중합 구조 점수 (total ≤ 1.0 수학적 보장) ──────────────
            # total = ALPHA × content_score + BETA × boost_signal
            #   ALPHA = 0.80 : 내용 점수 예산
            #   BETA  = 0.20 : boost 예산
            #   boost_signal : domain/difficulty/depth 신호 가중합을 최대값으로 정규화 (0~1)
            ALPHA = 0.80
            BETA  = 0.20
            MAX_BOOST = (
                self.cfg.W_DOMAIN_BOOST +
                self.cfg.W_DIFFICULTY_BOOST +
                self.cfg.W_DEPTH_BOOST
            )  # 최대 가능 boost 합산값 → 정규화 기준
            raw_boost = (
                self.cfg.W_DOMAIN_BOOST     * domain_score    +
                self.cfg.W_DIFFICULTY_BOOST * difficulty_match +
                self.cfg.W_DEPTH_BOOST      * depth_score
            )
            boost_signal = raw_boost / MAX_BOOST if MAX_BOOST > 0 else 0.0  # 0~1 정규화

            # 파편화 패널티
            frag = _compute_fragmentation_penalty(lec.concept_roles)

            total = max(
                ALPHA * content_score * duration_score
                + BETA * boost_signal
                - self.cfg.FRAG_PENALTY_WEIGHT * frag,
                0.0
            )

            # dm_keyword == 0 패널티
            # 키워드 직접 매칭이 전혀 없으면 내용 관련성 없는 강의로 판단
            # domain/difficulty boost만으로 올라온 false positive 제거
            if dm["keyword"] == 0:
                total *= 0.6

            detail = {
                "score":                round(total, 4),
                "content_score":        round(content_score, 4),
                "content_pct":          round(content_score * 100, 1),
                "vec_score":            round(vec_score, 4),
                "dm_score":             round(dm_score, 4),
                "sim_title":            round(sim_title, 4),
                "sim_keyword":          round(sim_keyword, 4),
                "sim_keyword_filtered": round(sim_keyword_filtered, 4),
                "sim_summary":          round(sim_summary, 4),
                "dm_title":             round(dm["title"], 4),
                "dm_keyword":           round(dm["keyword"], 4),
                "dm_summary":           round(dm["summary"], 4),
                "domain_score":         round(domain_score, 4),
                "difficulty_match":     round(difficulty_match, 4),
                "depth_score":          round(depth_score, 4),
                "boost_signal":         round(boost_signal, 4),
                "duration_score":       round(duration_score, 4),
                "duration_mismatch":    duration_score < 1.0,
                "frag_penalty":         round(frag, 4),
            }
            candidates.append((lec, detail))

        # ── 점수 기준 내림차순 정렬 ───────────────────────────────────
        candidates.sort(key=lambda x: x[1]["score"], reverse=True)

        # ── 전체 후보 점수 출력 (디버그) ─────────────────────────────
        print(f"  {'video_id':<10} {'제목':<24} {'sim_t':>5} {'sim_k':>5} {'sim_k*':>6} {'sim_s':>5} "
              f"{'vec':>5} {'dm':>5} {'boost':>6} {'dur':>5} {'→score':>7}")
        for lec, d in candidates:
            print(f"  {lec.video_id:<10} {lec.title[:22]:<24} "
                  f"{d['sim_title']:>5.3f} {d['sim_keyword']:>5.3f} "
                  f"{d['sim_keyword_filtered']:>6.3f} {d['sim_summary']:>5.3f} "
                  f"{d['vec_score']:>5.3f} {d['dm_score']:>5.3f} "
                  f"{d['boost_signal']:>6.3f} {d['duration_score']:>5.3f} {d['score']:>7.3f}")

        # ── min_score 필터 + top_k 반환 ──────────────────────────────
        results = []
        for lec, detail in candidates:
            if detail["score"] <= threshold:
                break
            detail["duration_sec"] = lec.duration_sec
            results.append(RecommendResult(
                video_id     = lec.video_id,
                title        = lec.title,
                domain       = lec.domain,
                instructor   = lec.instructor_id,
                score        = detail["score"],
                duration_sec = lec.duration_sec,
                score_detail = detail,
                reason       = _build_reason(detail),
                summary      = lec.summary,
            ))
            if len(results) >= top_k:
                break

        return results

    def print_results(self, results: list[RecommendResult], title: str = "추천 결과", top_k: int = 3):
        top = results[:top_k]

        print(f"\n{'='*62}")
        print(f"  {title}")
        print(f"{'='*62}")

        if not top:
            print("  ※ 질의와 일치하는 강의를 찾지 못했습니다.")
        else:
            print(f"  ▶ 추천 강의 (Top {len(top)})")
            print(f"  {'-'*58}")
            for i, r in enumerate(top, 1):
                d         = r.score_detail
                score_100 = round(r.score * 100, 1)
                t_pct     = round(d.get("sim_title",   0) * 100, 1)
                k_pct     = round(d.get("sim_keyword", 0) * 100, 1)
                s_pct     = round(d.get("sim_summary", 0) * 100, 1)
                dm_t      = round(d.get("dm_title",    0) * 100, 1)
                dm_k      = round(d.get("dm_keyword",  0) * 100, 1)
                dm_s      = round(d.get("dm_summary",  0) * 100, 1)
                boost_pct    = round((d.get("domain_boost",   1.0) - 1.0) * 100, 1)
                depth_pct    = round((d.get("depth_boost",    1.0) - 1.0) * 100, 1)
                duration_pct = round((d.get("duration_score", 1.0)) * 100, 1)
                frag_pct     = round(d.get("frag_penalty", 0.0) * 100, 1)
                print(f"  {i}. [{r.video_id}] {r.title}")
                print(f"     점수:   {score_100}점  |  {r.reason}")
                print(f"     벡터:   title {t_pct}점  keyword {k_pct}점  summary {s_pct}점")
                print(f"     직접:   title {dm_t}점  keyword {dm_k}점  summary {dm_s}점")
                print(f"     boost:  domain +{boost_pct}%  depth +{depth_pct}%  duration {duration_pct}%")
                if frag_pct > 0:
                    print(f"     파편화 패널티: -{frag_pct}%")
                print(f"     요약:   {r.summary[:80]}...")
                print()

        print(f"  {'─'*58}")
        print(f"  ▶ 추천된 강의 점수 전체 (min_score 필터 적용됨)")
        print(f"  {'─'*58}")
        print(f"  {'video_id':<12} {'제목':<22} {'도메인':<18} {'점수':>6}")
        print(f"  {'─'*58}")
        for r in results:
            marker = " ◀" if r in top else ""
            print(f"  {r.video_id:<12} {r.title[:20]:<22} {r.domain:<18} "
                  f"{round(r.score*100,1):>5.1f}점{marker}")
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
  python recommender.py --metadata_dir app/backend/metadata --query "운영체제란 무엇인가"
  python recommender.py --query "운영체제 기능 알고 싶어" --min_score 0.1
"""
    )
    parser.add_argument("--metadata_dir", default=DEFAULT_METADATA_DIR,
                        help=f"메타데이터 디렉토리 (기본: {DEFAULT_METADATA_DIR})")
    parser.add_argument("--query", default=None,
                        help="자연어 질의 (예: '메모리 관리 알고 싶어')")
    parser.add_argument("--top_k", type=int, default=5,
                        help="추천 결과 수 (기본: 5)")
    parser.add_argument("--min_score", type=float, default=None,
                        help="추천 최소 점수 0~1 (기본: cfg.MIN_SCORE=0.05). "
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