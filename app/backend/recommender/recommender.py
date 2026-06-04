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
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass
from collections import Counter, defaultdict
from typing import Any, Iterable, Optional

import numpy as np
import lancedb
from google import genai
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
_client        = genai.Client(api_key=os.getenv("GOOGLE_API_KEY_2"))
GEMINI_MODEL   = "gemini-2.5-flash"
EMBED_MODEL    = "gemini-embedding-001"
QUERY_LLM_PROVIDER = (
    os.getenv("RECOMMENDER_LLM_PROVIDER")
    or os.getenv("QUERY_SERVICE_LLM_PROVIDER")
    or "openai"
).strip().lower()
QUERY_OPENAI_MODEL = (
    os.getenv("RECOMMENDER_OPENAI_MODEL")
    or os.getenv("QUERY_SERVICE_OPENAI_MODEL")
    or "gpt-5.4"
).strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()


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
_TERM_RE = re.compile(r"[0-9A-Za-z가-힣_#+./-]+")
_DOMAIN_ALIASES = {
    "컴퓨터공학": "engineering",
    "컴공": "engineering",
    "computer science": "engineering",
    "cs": "engineering",
    "공학": "engineering",
    "경제학": "social_science",
    "경제": "social_science",
    "경영": "social_science",
    "비즈니스": "social_science",
    "마케팅": "social_science",
    "수학": "natural_science",
    "자연과학": "natural_science",
    "의학": "health_sciences",
    "의료": "health_sciences",
    "보건": "health_sciences",
    "생물학": "natural_science",
    "생명과학": "natural_science",
    "화학": "natural_science",
    "물리학": "natural_science",
    "물리": "natural_science",
    "환경": "natural_science",
    "기후": "natural_science",
    "철학": "humanities",
    "역사": "humanities",
    "세계사": "humanities",
    "한국사": "humanities",
    "언어학": "humanities",
    "교육": "education",
    "교육학": "education",
    "사회학": "social_science",
    "사회": "social_science",
    "디자인": "arts",
    "예술": "arts",
    "체육": "sports",
    "스포츠": "sports",
}
_SUBDOMAIN_ALIASES = {
    "컴퓨터공학": ("engineering", "computer_science"),
    "컴공": ("engineering", "computer_science"),
    "computer science": ("engineering", "computer_science"),
    "cs": ("engineering", "computer_science"),
    "전기공학": ("engineering", "electrical_engineering"),
    "전자공학": ("engineering", "electrical_engineering"),
    "기계공학": ("engineering", "mechanical_engineering"),
    "화학공학": ("engineering", "chemical_engineering"),
    "산업공학": ("engineering", "industrial_engineering"),
    "경제학": ("social_science", "economics"),
    "경제": ("social_science", "economics"),
    "경영학": ("social_science", "business"),
    "경영": ("social_science", "business"),
    "비즈니스": ("social_science", "business"),
    "법학": ("social_science", "law"),
    "정치학": ("social_science", "political_science"),
    "사회학": ("social_science", "sociology"),
    "심리학": ("social_science", "psychology"),
    "교육학": ("education", "education"),
    "물리학": ("natural_science", "physics"),
    "물리": ("natural_science", "physics"),
    "화학": ("natural_science", "chemistry"),
    "수학": ("natural_science", "mathematics"),
    "mathematics": ("natural_science", "mathematics"),
    "math": ("natural_science", "mathematics"),
    "생물학": ("natural_science", "biology"),
    "생명과학": ("natural_science", "biology"),
    "천문학": ("natural_science", "astronomy"),
    "생태학": ("natural_science", "ecology"),
    "철학": ("humanities", "philosophy"),
    "역사학": ("humanities", "history"),
    "역사": ("humanities", "history"),
    "언어학": ("humanities", "linguistics"),
    "문학": ("humanities", "literature"),
    "종교학": ("humanities", "religion"),
    "미술": ("arts", "fine_arts"),
    "음악": ("arts", "music"),
    "디자인": ("arts", "design"),
    "영화": ("arts", "film"),
    "연극": ("arts", "theater"),
    "해부학": ("health_sciences", "anatomy"),
    "생리학": ("health_sciences", "physiology"),
    "약리학": ("health_sciences", "pharmacology"),
    "공중보건": ("health_sciences", "public_health"),
    "간호학": ("health_sciences", "nursing"),
    "체육": ("sports", "physical_education"),
    "스포츠과학": ("sports", "sports_science"),
}
_DOMAIN_LABELS = {
    "engineering": "공학",
    "natural_science": "자연과학",
    "humanities": "인문학",
    "social_science": "사회과학",
    "arts": "예술",
    "health_sciences": "보건의료",
    "sports": "스포츠",
    "education": "교육",
    "etc": "기타",
}


def _canonical_domain(value: str) -> str:
    token = str(value or "").strip().lower().replace("-", "_")
    if token in _DOMAIN_LABELS:
        return token
    top = token.split("/", 1)[0]
    return {
        "eng": "engineering",
        "sci": "natural_science",
        "math": "natural_science",
        "hum": "humanities",
        "soc": "social_science",
        "med": "health_sciences",
        "art": "arts",
        "gen": "etc",
    }.get(top, "etc")


def _normalize_subdomain(value: str) -> str:
    return str(value or "").strip().lower().replace("-", "_")


def _infer_domain_filters(
    query: str,
    available_domains: list[str],
    available_subdomains: list[str],
) -> tuple[Optional[str], Optional[str]]:
    normalized = str(query or "").lower()
    compact = re.sub(r"\s+", "", normalized)
    for alias, (domain, subdomain) in _SUBDOMAIN_ALIASES.items():
        alias_normalized = alias.lower()
        alias_compact = re.sub(r"\s+", "", alias_normalized)
        if (
            (alias_normalized in normalized or alias_compact in compact)
            and domain in available_domains
            and subdomain in available_subdomains
        ):
            return domain, subdomain
    for alias, domain in _DOMAIN_ALIASES.items():
        alias_normalized = alias.lower()
        alias_compact = re.sub(r"\s+", "", alias_normalized)
        if (alias_normalized in normalized or alias_compact in compact) and domain in available_domains:
            return domain, None
    return None, None


_TOPIC_EXPANSIONS = {
    "파이썬": [
        "Python",
    ],
    "python": [
        "파이썬",
    ],
    "웹": [
        "리액트",
        "컴포넌트",
        "자바스크립트",
        "비동기",
        "Promise",
        "async/await",
        "fetch",
        "JSX",
        "가상 DOM",
    ],
    "웹서비스": [
        "리액트",
        "컴포넌트",
        "자바스크립트",
        "비동기",
        "Promise",
        "async/await",
        "fetch",
        "JSX",
        "가상 DOM",
    ],
    "프론트엔드": [
        "리액트",
        "컴포넌트",
        "자바스크립트",
        "JSX",
        "가상 DOM",
        "렌더링",
    ],
    "데이터베이스": [
        "SQL",
        "JOIN",
        "트랜잭션",
        "인덱스",
        "정규화",
        "스키마",
    ],
    "DB": [
        "데이터베이스",
        "SQL",
        "JOIN",
        "트랜잭션",
        "인덱스",
        "정규화",
    ],
}


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
    # 메인 점수 컴포넌트 가중치
    W_CONTENT:           float = 0.60
    W_GRAPH:             float = 0.20
    W_COMMUNITY:         float = 0.10
    W_VISUAL:            float = 0.03
    W_BOOST:             float = 0.07
    W_CONTENT_VISUAL_QUERY:   float = 0.52
    W_GRAPH_VISUAL_QUERY:     float = 0.18
    W_COMMUNITY_VISUAL_QUERY: float = 0.10
    W_VISUAL_QUERY:           float = 0.15
    W_BOOST_VISUAL_QUERY:     float = 0.05
    # graph_score role 가중치
    GRAPH_CORE_WEIGHT:   float = 1.00
    GRAPH_INTRO_WEIGHT:  float = 0.35
    # depth boost (focus_concept 지정 시에만 활성)
    W_DEPTH_BOOST:       float = 0.30
    # 파편화 패널티 강도 λ
    FRAG_PENALTY_WEIGHT: float = 0.10
    # 이중 레이어 임계값
    DIRECT_RATIO:            float = 0.90
    RELATED_RATIO:           float = 0.55
    ABS_MIN_SCORE:           float = 0.30
    ABS_DIRECT_FLOOR:        float = 0.50
    CORE_MATCH_DIRECT_FLOOR: float = 0.35
    GAP_THRESHOLD:           float = 0.08
    DIRECT_GRAPH_FLOOR:      float = 0.55
    RELATED_GRAPH_FLOOR:     float = 0.25
    RELATED_COMMUNITY_FLOOR: float = 0.20
    # 패널티
    DOMAIN_MISMATCH_PENALTY: float = 0.60  # 도메인 상위 카테고리 불일치
    Q_KW_MISMATCH_PENALTY:   float = 0.60  # 원본 query_keyword 완전 미매칭
    # 비교 의도 × 강의 대조 관계 보너스
    W_CONTRAST_BOOST:        float = 0.10
    # 벡터 DB 경로
    DB_DIR:                  str   = DEFAULT_DB_DIR
    # domain boost 강도
    W_DOMAIN_BOOST:      float = 0.20
    W_DURATION_BOOST:    float = 0.08
    W_APPLICATION_BOOST: float = 0.10
    W_LISTENABILITY_BOOST: float = 0.08
    W_SPEECH_RATE_BOOST: float = 0.07
    W_RECENCY_BOOST_FAST: float = 0.10
    W_RECENCY_BOOST_MEDIUM: float = 0.05
    W_RECENCY_BOOST_SLOW: float = 0.02
    RECENCY_HALF_LIFE_DAYS: float = 180.0
    # BM25 후보 검색 파라미터
    BM25_K1:             float = 1.2
    BM25_B:              float = 0.75
    BM25_TITLE_WEIGHT:   float = 1.5
    BM25_KEYWORD_WEIGHT: float = 2.0
    BM25_SUMMARY_WEIGHT: float = 0.5
    BM25_RETRIEVE_TOP_N: int = 80
    VECTOR_RETRIEVE_TOP_N: int = 80
    RRF_K:               int = 60
    HYBRID_CANDIDATE_TOP_N: int = 50
    MIN_HYBRID_CANDIDATES: int = 10
    USE_TF_IRF_DM:       bool = True
    TF_IRF_IDF_FLOOR:    float = 0.05
    USE_METADATA_PREFILTER: bool = True
    METADATA_DURATION_GRACE_SEC: int = 300
    LIST_QUERY_TOP_K:    int = 50


# ============================================================================
#  데이터 모델
# ============================================================================

@dataclass
class LectureMetadata:
    video_id:          str
    title:             str
    instructor_id:     str
    uploaded_at:       Optional[str]
    domain:            str
    graph_subdomain:   str
    difficulty:        str   # "beginner" | "intermediate" | "advanced"
    duration_sec:      float
    summary:           str
    keywords:          list[dict]
    concept_roles:     list[dict]
    concept_relations: list[dict]
    communities:       list[dict]
    pedagogy:          dict
    diagnostics:       dict
    visual_concept_terms: list[str]


@dataclass
class LectureLexicalDocument:
    video_id:  str
    field_tf:  dict[str, Counter]
    term_tf:   Counter
    doc_len:   float


@dataclass
class LexicalStats:
    total_docs:  int
    doc_freq:    dict[str, int]
    irf:         dict[str, float]
    bm25_idf:    dict[str, float]
    documents:   dict[str, LectureLexicalDocument]
    avg_doc_len: float


@dataclass
class VectorSearchIndex:
    video_ids:       list[str]
    title_matrix:    np.ndarray
    keyword_matrix:  np.ndarray
    summary_matrix:  np.ndarray


@dataclass
class CommunityReportDocument:
    video_id:    str
    title_terms: Counter
    summary_terms: Counter
    rank:        float
    level:       int


def _database_url_sync() -> str:
    url = os.getenv("DATABASE_URL", "")
    return url.replace("+asyncpg", "") if url else ""


def _video_id_from_db_row(row: dict) -> str:
    metadata_uri = str(row.get("metadata_uri") or "").strip()
    if metadata_uri:
        name = Path(metadata_uri).name
        if name.endswith("_metadata.json"):
            return name[:-len("_metadata.json")]
        if name:
            return Path(name).stem

    video_path = str(row.get("video_path") or "").strip()
    if video_path:
        return Path(video_path).stem

    return str(row.get("lecture_id") or "")


def _fetch_lecture_metadata_rows(database_url: str) -> list[dict]:
    import psycopg2
    from psycopg2.extras import RealDictCursor

    sql = """
        SELECT
            lm.lecture_id,
            CASE
                WHEN lm.title IS NULL
                  OR btrim(lm.title) = ''
                  OR lm.title = 'Untitled lecture'
                THEN l.title
                ELSE lm.title
            END AS title,
            lm.instructor_id,
            lm.domain,
            lm.graph_domain,
            lm.graph_subdomain,
            lm.difficulty,
            lm.summary,
            lm.learning_objectives,
            lm.keywords,
            lm.concept_roles,
            lm.concept_relations,
            lm.communities,
            lm.visual_concept_terms,
            lm.pedagogy,
            lm.diagnostics,
            lm.duration_sec,
            lm.uploaded_at,
            lm.metadata_version,
            lm.metadata_uri,
            lm.created_at,
            lm.updated_at,
            l.video_path,
            l.created_at AS lecture_created_at
        FROM lecture_metadata lm
        JOIN lectures l ON l.id = lm.lecture_id
        WHERE l.is_published IS TRUE
        ORDER BY lm.updated_at DESC NULLS LAST, lm.created_at DESC NULLS LAST
    """

    conn = psycopg2.connect(database_url)
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql)
            return [dict(row) for row in cur.fetchall()]
    finally:
        conn.close()


# ============================================================================
#  메타데이터 컬렉션
# ============================================================================

class MetadataCollection:
    def __init__(self, metadata_dir: str):
        self.lectures: dict[str, LectureMetadata] = {}
        use_file_metadata = os.getenv("RECOMMENDER_USE_FILE_METADATA", "1").lower() not in {"0", "false", "no"}
        if use_file_metadata:
            self._load_files(Path(metadata_dir))
        else:
            print("[파일 로드] RECOMMENDER_USE_FILE_METADATA=0 — 파일 metadata 로드 생략")
        self._load_db()

    def _load_files(self, directory: Path):
        if not directory.exists():
            print(f"[Recommender] 메타데이터 디렉토리 없음, 빈 컬렉션으로 시작: {directory}")
            return
        files = list(directory.glob("*_metadata.json"))
        if not files:
            print(f"[Recommender] 메타데이터 파일 없음, 빈 컬렉션으로 시작: {directory}")
            return
        for path in files:
            with open(path, encoding="utf-8") as f:
                raw = json.load(f)
            items = raw if isinstance(raw, list) else [raw]
            for item in items:
                if not self._metadata_item_is_published(item):
                    continue
                lec = self._from_metadata_item(item)
                self.lectures[lec.video_id] = lec
        print(f"[파일 로드] {len(self.lectures)}개 강의 메타데이터 로드 완료")

    def _load_db(self):
        database_url = _database_url_sync()
        if not database_url:
            print("[DB 로드] DATABASE_URL 없음 — 파일 metadata만 사용\n")
            return

        try:
            rows = _fetch_lecture_metadata_rows(database_url)
        except Exception as exc:
            print(f"[DB 로드] lecture_metadata 로드 실패 — 파일 metadata만 사용: {exc}\n")
            return

        added = 0
        merged = 0
        for row in rows:
            lec = self._from_db_row(row)
            existing = self.lectures.get(lec.video_id)
            if existing:
                self.lectures[lec.video_id] = self._merge_db_with_existing(existing, lec)
                merged += 1
            else:
                self.lectures[lec.video_id] = lec
                added += 1
        print(
            f"[DB 로드] lecture_metadata {len(rows)}개 로드 "
            f"(추가 {added}, 병합 {merged}) — 총 {len(self.lectures)}개\n"
        )

    @staticmethod
    def _metadata_item_is_published(item: dict) -> bool:
        if "is_published" in item:
            value = item.get("is_published")
            if isinstance(value, bool):
                return value
            return str(value).strip().lower() in {"1", "true", "yes", "y"}

        publication_status = item.get("publication_status")
        if publication_status is not None:
            return str(publication_status).strip().lower() == "published"

        return True

    @staticmethod
    def _from_metadata_item(item: dict) -> LectureMetadata:
        return LectureMetadata(
            video_id          = item["video_id"],
            title             = item["title"],
            instructor_id     = item.get("instructor_id", ""),
            uploaded_at       = item.get("uploaded_at"),
            domain            = _canonical_domain(item.get("graph_domain") or item.get("domain")),
            graph_subdomain   = _normalize_subdomain(item.get("graph_subdomain")),
            difficulty        = item.get("difficulty", "unknown"),
            duration_sec      = item.get("duration_sec", 0.0),
            summary           = item.get("summary", ""),
            keywords          = item.get("keywords", []),
            concept_roles     = item.get("concept_roles", []),
            concept_relations = item.get("concept_relations", []),
            communities       = item.get("communities", []),
            pedagogy          = item.get("pedagogy", {}),
            diagnostics       = item.get("diagnostics", {}),
            visual_concept_terms = (
                item.get("visual_concept_terms")
                or item.get("pedagogy", {}).get("visual_concept_terms", [])
            ),
        )

    @staticmethod
    def _from_db_row(row: dict) -> LectureMetadata:
        uploaded_at = row.get("uploaded_at") or row.get("lecture_created_at")
        if hasattr(uploaded_at, "isoformat"):
            uploaded_at = uploaded_at.isoformat()
        return LectureMetadata(
            video_id          = _video_id_from_db_row(row),
            title             = row.get("title") or "Untitled lecture",
            instructor_id     = row.get("instructor_id") or "",
            uploaded_at       = uploaded_at,
            domain            = _canonical_domain(row.get("graph_domain") or row.get("domain")),
            graph_subdomain   = _normalize_subdomain(row.get("graph_subdomain")),
            difficulty        = row.get("difficulty") or "unknown",
            duration_sec      = row.get("duration_sec") or 0.0,
            summary           = row.get("summary") or "",
            keywords          = row.get("keywords") or [],
            concept_roles     = row.get("concept_roles") or {},
            concept_relations = row.get("concept_relations") or [],
            communities       = row.get("communities") or [],
            pedagogy          = row.get("pedagogy") or {},
            diagnostics       = row.get("diagnostics") or {},
            visual_concept_terms = row.get("visual_concept_terms") or [],
        )

    @staticmethod
    def _merge_db_with_existing(existing: LectureMetadata, db_lecture: LectureMetadata) -> LectureMetadata:
        db_concept_roles = db_lecture.concept_roles
        concept_roles = (
            db_concept_roles
            if (
                isinstance(db_concept_roles, dict)
                and (db_concept_roles.get("core") or db_concept_roles.get("introduced"))
            )
            else existing.concept_roles
        )
        return LectureMetadata(
            video_id          = existing.video_id,
            title             = db_lecture.title or existing.title,
            instructor_id     = db_lecture.instructor_id or existing.instructor_id,
            uploaded_at       = db_lecture.uploaded_at or existing.uploaded_at,
            domain            = _canonical_domain(db_lecture.domain or existing.domain),
            graph_subdomain   = db_lecture.graph_subdomain or existing.graph_subdomain,
            difficulty        = db_lecture.difficulty or existing.difficulty,
            duration_sec      = db_lecture.duration_sec or existing.duration_sec,
            summary           = db_lecture.summary or existing.summary,
            keywords          = db_lecture.keywords or existing.keywords,
            concept_roles     = concept_roles,
            concept_relations = db_lecture.concept_relations or existing.concept_relations,
            communities       = db_lecture.communities or existing.communities,
            pedagogy          = db_lecture.pedagogy or existing.pedagogy,
            diagnostics       = db_lecture.diagnostics or existing.diagnostics,
            visual_concept_terms = db_lecture.visual_concept_terms or existing.visual_concept_terms,
        )

    def get(self, video_id: str) -> Optional[LectureMetadata]:
        return self.lectures.get(video_id)

    def all(self) -> list[LectureMetadata]:
        return list(self.lectures.values())

    def available_domains(self) -> list[str]:
        return sorted({lec.domain for lec in self.lectures.values()})

    def available_subdomains(self) -> list[str]:
        return sorted({lec.graph_subdomain for lec in self.lectures.values() if lec.graph_subdomain})

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


def _normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return matrix
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def _normalize_vector(vector: list[float]) -> np.ndarray:
    arr = np.array(vector, dtype=np.float32)
    norm = np.linalg.norm(arr)
    if norm == 0:
        return arr
    return arr / norm


def _normalize_term(text: str) -> str:
    """lexical retrieval에서 공유할 term 정규화."""
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def _tokenize_text(text: str) -> list[str]:
    """한글/영문/숫자 혼합 텍스트를 가볍게 토큰화."""
    normalized = _normalize_term(text)
    return [
        token
        for token in _TERM_RE.findall(normalized)
        if len(token) > 1
    ]


def _expanded_topic_terms(terms: Iterable[str]) -> list[str]:
    """
    LLM이 일반 표현을 세부 강의 키워드로 확장하지 못한 경우를 위한
    작은 도메인 사전. 예: 웹 → 리액트/자바스크립트/비동기.
    """
    expanded: list[str] = []
    seen: set[str] = set()

    def add(term: str) -> None:
        normalized = _normalize_term(term)
        if not normalized or normalized in seen:
            return
        seen.add(normalized)
        expanded.append(term)

    for term in terms:
        add(term)
        normalized = _normalize_term(term)
        for alias, aliases in _TOPIC_EXPANSIONS.items():
            normalized_alias = _normalize_term(alias)
            if normalized == normalized_alias or normalized_alias in normalized or normalized in normalized_alias:
                for expanded_term in aliases:
                    add(expanded_term)

    return expanded


def _append_topic_expansions(
    query_keywords: list[str],
    inferred_keywords: list[str],
    seed_terms: Optional[list[str]] = None,
) -> list[str]:
    existing = {_normalize_term(term) for term in query_keywords + inferred_keywords}
    seed_only = {
        _normalize_term(term)
        for term in (seed_terms or [])
        if _normalize_term(term)
    }
    expanded = list(inferred_keywords)
    for term in _expanded_topic_terms(query_keywords + inferred_keywords + (seed_terms or [])):
        normalized = _normalize_term(term)
        if normalized in seed_only:
            continue
        if normalized and normalized not in existing:
            existing.add(normalized)
            expanded.append(term)
    return expanded


def _community_report_terms(text: str) -> Counter:
    terms = Counter()
    normalized = _normalize_term(text)
    if not normalized:
        return terms
    for token in _tokenize_text(normalized):
        terms[token] += 1.0
    return terms


def _keyword_terms(lec: LectureMetadata) -> list[tuple[str, float]]:
    terms = []
    for item in lec.keywords or []:
        term = _normalize_term(item.get("keyword", ""))
        if not term:
            continue
        try:
            weight = float(item.get("score", 1.0))
        except (TypeError, ValueError):
            weight = 1.0
        terms.append((term, max(weight, 0.0)))
    return terms


def _build_lexical_stats(lectures: list[LectureMetadata]) -> LexicalStats:
    """
    BM25 후보 검색과 TF-IRF dm_score가 공유할 lexical statistics.
    현재 단계에서는 추천 점수에 연결하지 않고, 다음 단계의 기반만 만든다.
    """
    documents: dict[str, LectureLexicalDocument] = {}
    doc_freq: defaultdict[str, int] = defaultdict(int)

    for lec in lectures:
        field_tf = {
            "title":   Counter(_tokenize_text(lec.title)),
            "keyword": Counter(),
            "summary": Counter(_tokenize_text(lec.summary)),
        }

        for term, weight in _keyword_terms(lec):
            field_tf["keyword"][term] += weight
            # 복합 키워드는 phrase term과 구성 토큰을 함께 보존한다.
            for token in _tokenize_text(term):
                field_tf["keyword"][token] += weight * 0.5

        term_tf = Counter()
        for field_counter in field_tf.values():
            term_tf.update(field_counter)

        doc_len = float(sum(term_tf.values()))
        documents[lec.video_id] = LectureLexicalDocument(
            video_id = lec.video_id,
            field_tf = field_tf,
            term_tf  = term_tf,
            doc_len  = doc_len,
        )

        for term in term_tf:
            doc_freq[term] += 1

    total_docs = len(documents)
    safe_total = max(total_docs, 1)
    irf = {
        term: math.log(safe_total / max(df, 1))
        for term, df in doc_freq.items()
    }
    bm25_idf = {
        term: math.log(1.0 + (safe_total - df + 0.5) / (df + 0.5))
        for term, df in doc_freq.items()
    }
    avg_doc_len = (
        sum(doc.doc_len for doc in documents.values()) / total_docs
        if total_docs else 0.0
    )

    return LexicalStats(
        total_docs  = total_docs,
        doc_freq    = dict(doc_freq),
        irf         = irf,
        bm25_idf    = bm25_idf,
        documents   = documents,
        avg_doc_len = avg_doc_len,
    )


def _build_vector_search_index(index_rows: list[dict]) -> VectorSearchIndex:
    video_ids = [row["video_id"] for row in index_rows]
    if not index_rows:
        empty = np.empty((0, 0), dtype=np.float32)
        return VectorSearchIndex(video_ids, empty, empty, empty)

    title_matrix = np.array(
        [row["title_vec"] for row in index_rows],
        dtype=np.float32,
    )
    keyword_matrix = np.array(
        [row["keyword_vec"] for row in index_rows],
        dtype=np.float32,
    )
    summary_matrix = np.array(
        [row["summary_vec"] for row in index_rows],
        dtype=np.float32,
    )

    return VectorSearchIndex(
        video_ids      = video_ids,
        title_matrix   = _normalize_matrix(title_matrix),
        keyword_matrix = _normalize_matrix(keyword_matrix),
        summary_matrix = _normalize_matrix(summary_matrix),
    )


class CommunityIndex:
    """
    metadata.communities 기반 reranking 보조 신호.
    community 정보가 없는 강의는 0.0으로 동작해 기존 데모 강의에는 영향을 주지 않는다.
    """

    def __init__(self, lectures: list[LectureMetadata]):
        self.reports_by_video_id: dict[str, list[CommunityReportDocument]] = {}
        self._load(lectures)

    def _load(self, lectures: list[LectureMetadata]) -> None:
        report_count = 0
        for lec in lectures:
            reports = []
            for row in lec.communities or []:
                if not isinstance(row, dict):
                    continue
                title = str(row.get("title") or "")
                summary = str(row.get("summary") or "")
                if not title and not summary:
                    continue

                try:
                    rank = float(row.get("rank") or 0.0)
                except (TypeError, ValueError):
                    rank = 0.0
                if not math.isfinite(rank):
                    rank = 0.0

                try:
                    level = int(row.get("level") or 0)
                except (TypeError, ValueError):
                    level = 0

                reports.append(CommunityReportDocument(
                    video_id      = lec.video_id,
                    title_terms   = _community_report_terms(title),
                    summary_terms = _community_report_terms(summary),
                    rank          = rank,
                    level         = level,
                ))

            if reports:
                self.reports_by_video_id[lec.video_id] = reports
                report_count += len(reports)

        print(
            f"[Community] metadata 기반 {len(self.reports_by_video_id)}개 강의, "
            f"{report_count}개 report 로드\n"
        )

    @staticmethod
    def _coverage(query_terms: Counter, report_terms: Counter) -> float:
        if not query_terms or not report_terms:
            return 0.0
        denom = sum(max(weight, 0.0) for weight in query_terms.values()) or 1.0
        hit = sum(
            max(weight, 0.0)
            for term, weight in query_terms.items()
            if report_terms.get(term, 0.0) > 0
        )
        return min(hit / denom, 1.0)

    def score(self, video_id: str, query_terms: Counter) -> float:
        reports = self.reports_by_video_id.get(video_id)
        if not reports or not query_terms:
            return 0.0

        best = 0.0
        for report in reports:
            title_overlap = self._coverage(query_terms, report.title_terms)
            summary_overlap = self._coverage(query_terms, report.summary_terms)
            base = 0.60 * title_overlap + 0.40 * summary_overlap
            rank_norm = min(max(report.rank / 10.0, 0.0), 1.0)
            rank_weight = 0.85 + 0.15 * rank_norm
            best = max(best, base * rank_weight)

        return round(min(best, 1.0), 4)


def _fast_list_by_domain_analysis(
    query: str,
    available_domains: list[str],
    available_subdomains: list[str],
) -> Optional[tuple[str, str, list[str], list[str], Optional[str], Optional[str], Optional[int], dict]]:
    normalized = _normalize_term(query)
    has_list_signal = any(
        signal in normalized
        for signal in (
            "뭐 있어",
            "뭐있어",
            "목록",
            "리스트",
            "전체 강의",
            "모든 강의",
            "강의 보여",
            "강의 알려",
        )
    )
    if not has_list_signal:
        return None

    if "전체 강의" in normalized or "모든 강의" in normalized:
        return "list_by_domain", query, [], [], None, None, None, {}

    domain, subdomain = _infer_domain_filters(query, available_domains, available_subdomains)
    if domain:
        return "list_by_domain", query, [], [], domain, None, None, {"subdomain": subdomain}

    return None


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
    q_tokens = set(_expanded_topic_terms(query_keywords))
    i_tokens = set(_expanded_topic_terms(inferred_keywords))
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
                t == k["keyword"] or k["keyword"].startswith(t) or t.startswith(k["keyword"])
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

    return {
        "title":        title_match,
        "keyword":      kw_match,
        "summary":      sum_match,
        "q_kw_matched": (not q_tokens) or q_kw > 0,  # 원본 query_keyword 매칭 여부
    }


def _direct_match_score_tfirf(
    query_keywords:    list[str],
    inferred_keywords: list[str],
    lec:               LectureMetadata,
    stats:             LexicalStats,
    idf_floor:         float = 0.05,
    inferred_weight:   float = 0.5,
) -> dict:
    """
    기존 직접 매칭 중 keyword 성분만 TF-IRF 방식으로 보정한다.
    title/summary는 기존 휴리스틱을 유지해 동작 변화 폭을 줄인다.
    """
    base = _direct_match_score(
        query_keywords,
        inferred_keywords,
        lec,
        inferred_weight=inferred_weight,
    )

    q_tokens = {_normalize_term(t) for t in _expanded_topic_terms(query_keywords) if _normalize_term(t)}
    i_tokens = {_normalize_term(t) for t in _expanded_topic_terms(inferred_keywords) if _normalize_term(t)}
    if not (q_tokens or i_tokens):
        base["keyword_legacy"] = base["keyword"]
        return base

    keyword_weights = []
    for item in lec.keywords or []:
        keyword = _normalize_term(item.get("keyword", ""))
        if not keyword:
            continue
        try:
            score = float(item.get("score", 1.0))
        except (TypeError, ValueError):
            score = 1.0
        irf = max(stats.irf.get(keyword, 0.0), idf_floor)
        keyword_weights.append((keyword, max(score, 0.0) * irf))

    total_weight = sum(weight for _keyword, weight in keyword_weights) or 1.0

    def matched_weight(tokens: set[str]) -> float:
        return sum(
            weight
            for keyword, weight in keyword_weights
            if any(
                token == keyword or keyword.startswith(token) or token.startswith(keyword)
                for token in tokens
                if len(token) > 1
            )
        )

    q_kw = matched_weight(q_tokens)
    i_kw = matched_weight(i_tokens)
    tfirf_keyword = (
        q_kw * 1.0 + i_kw * inferred_weight
    ) / (total_weight * (1.0 + inferred_weight))

    base["keyword_legacy"] = base["keyword"]
    base["keyword"] = tfirf_keyword
    base["q_kw_matched"] = (not q_tokens) or q_kw > 0
    return base


def _concept_match(concept: str, focus: str) -> bool:
    return focus in concept or concept in focus


def _append_terms(target: list[str], values) -> None:
    """metadata의 list/dict/string 혼합 필드에서 문자열 term만 평탄화한다."""
    if values is None:
        return
    if isinstance(values, str):
        if values.strip():
            target.append(values)
        return
    if isinstance(values, dict):
        for value in values.values():
            _append_terms(target, value)
        return
    if isinstance(values, (list, tuple, set)):
        for value in values:
            _append_terms(target, value)
        return
    if isinstance(values, (int, float)):
        target.append(str(values))


def _role_weight_for_concept(
    lec: LectureMetadata,
    concept: str,
    core_weight: float = 1.0,
    intro_weight: float = 0.35,
) -> float:
    """강의 내 concept role 기반 매칭 가중치."""
    if not concept:
        return 0.0

    role_weights = {
        "core": core_weight,
        "introduced": intro_weight,
        "prerequisite": 0.60,
    }
    concept_roles = lec.concept_roles
    best = 0.0

    if isinstance(concept_roles, dict):
        for role, concepts in concept_roles.items():
            weight = role_weights.get(role, 0.0)
            for role_concept in (concepts or []):
                if _concept_match(str(role_concept), concept):
                    best = max(best, weight)
    elif isinstance(concept_roles, list):
        for cr in (concept_roles or []):
            if not isinstance(cr, dict):
                continue
            if _concept_match(str(cr.get("concept", "")), concept):
                best = max(best, role_weights.get(cr.get("role", ""), 0.0))

    return best


def _query_concept_in_role(
    lec: LectureMetadata,
    query_concepts: set[str],
    target_role: str,
) -> bool:
    """질의 개념이 특정 concept_roles bucket에 포함되는지 확인한다."""
    concepts = {
        _normalize_term(concept)
        for concept in query_concepts
        if _normalize_term(concept)
    }
    if not concepts:
        return False

    role_terms: list[str] = []
    concept_roles = lec.concept_roles
    if isinstance(concept_roles, dict):
        _append_terms(role_terms, concept_roles.get(target_role, []))
    elif isinstance(concept_roles, list):
        for item in concept_roles:
            if not isinstance(item, dict) or item.get("role") != target_role:
                continue
            _append_terms(role_terms, item.get("concept"))

    normalized_roles = [
        _normalize_term(term)
        for term in role_terms
        if _normalize_term(term)
    ]
    return any(
        _concept_match(role_term, query_term)
        for role_term in normalized_roles
        for query_term in concepts
    )


def _compute_graph_score(
    lec: LectureMetadata,
    query_concepts: set[str],
    core_weight: float = 1.0,
    intro_weight: float = 0.35,
) -> float:
    """
    질의 개념이 강의의 concept_roles/concept_relations에 얼마나 걸리는지 계산.
    반환: 0.0 ~ 1.0
    """
    query_concepts = {c for c in query_concepts if c}
    if not query_concepts:
        return 0.0

    role_hits = [
        _role_weight_for_concept(lec, concept, core_weight, intro_weight)
        for concept in query_concepts
    ]
    role_score = sum(role_hits) / len(query_concepts)

    graph: dict[str, list[str]] = {}
    for rel in (lec.concept_relations or []):
        src = rel.get("from", rel.get("source", ""))
        dst = rel.get("to", rel.get("target", ""))
        if src and dst:
            graph.setdefault(src, []).append(dst)
            graph.setdefault(dst, []).append(src)

    relation_scores = []
    for node, neighbors in graph.items():
        if not any(_concept_match(node, qc) for qc in query_concepts):
            continue
        if not neighbors:
            continue
        related_neighbors = sum(
            1
            for neighbor in neighbors
            if any(_concept_match(neighbor, qc) for qc in query_concepts)
        )
        relation_scores.append(related_neighbors / len(neighbors))

    relation_score = (
        sum(relation_scores) / len(relation_scores)
        if relation_scores else 0.0
    )

    return round(min(role_score * 0.65 + relation_score * 0.35, 1.0), 4)


def _compute_depth_score(focus_concept: str, target: LectureMetadata) -> float:
    """
    focus_concept이 강의에서 얼마나 깊이 다뤄지는지 추정.

    v2: concept_relations로 로컬 그래프를 구성하고 BFS 홉 거리 기반 점수 계산.
      - focus_concept이 그래프 내 노드면 BFS로 인접 개념 분포를 측정
      - 그래프에 없으면 keyword 기반 fallback
      가중합: role×0.5 + hop×0.5
    """
    if not focus_concept:
        return 0.0

    # ── 1. role 점수 (concept_roles 기반) ────────────────────────
    role_score    = 0.0
    concept_roles = target.concept_roles
    if isinstance(concept_roles, dict):
        ROLE_WEIGHTS = {"core": 1.0, "introduced": 0.1}
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
                elif role == "introduced":
                    role_score = max(role_score, 0.1)

    # ── 2. 로컬 그래프 구성 ──────────────────────────────────────
    graph: dict[str, list[str]] = {}
    for rel in (target.concept_relations or []):
        src = rel.get("from", rel.get("source", ""))
        dst = rel.get("to",   rel.get("target", ""))
        if src and dst:
            graph.setdefault(src, []).append(dst)
            graph.setdefault(dst, []).append(src)  # 양방향

    # focus_concept과 매칭되는 그래프 내 노드 탐색 (부분 일치 허용)
    focus_node = next(
        (n for n in graph if _concept_match(n, focus_concept)),
        None
    )

    # ── 3. BFS 홉 거리 점수 ──────────────────────────────────────
    if focus_node:
        # BFS: 최대 3홉 이내 노드와 거리 계산
        dist_map: dict[str, int] = {focus_node: 0}
        frontier = [focus_node]
        for _ in range(3):
            next_frontier = []
            for node in frontier:
                for neighbor in graph.get(node, []):
                    if neighbor and neighbor not in dist_map:
                        dist_map[neighbor] = dist_map[node] + 1
                        next_frontier.append(neighbor)
            frontier = next_frontier
            if not frontier:
                break

        # 거리 d 노드 기여: 1/(d+1)  [1홉=0.5, 2홉=0.33, 3홉=0.25]
        # 정규화 기준: 5개 인접 노드가 모두 1홉 → hop_score=1.0
        hop_score = sum(
            1.0 / (d + 1)
            for n, d in dist_map.items()
            if n != focus_node and d > 0
        )
        hop_score = min(hop_score / 5.0, 1.0)
    else:
        # focus_node가 로컬 그래프에 없으면 하위 키워드 기반 fallback
        sub_kw_count = sum(
            1 for k in target.keywords
            if focus_concept in k["keyword"] and k["keyword"] != focus_concept
        )
        hop_score = min(sub_kw_count / 3.0, 1.0)

    depth = role_score * 0.5 + hop_score * 0.5
    return round(depth, 4)


# ── 대조·비교형 relation type 집합 ───────────────────────────────────
_CONTRAST_TYPES = frozenset({
    "contrasts", "lacks", "differs", "vs", "versus",
    "compared_to", "unlike", "opposes", "excludes",
})

# ── 비교 의도 감지 신호 ───────────────────────────────────────────────
_COMPARISON_SIGNALS = frozenset({
    "차이", "비교", "vs", "versus", "차이점", "대비",
    "구분", "다른점", "차이를", "비교해", "비교한",
})


_VISUAL_CONDITION_TERMS = frozenset({
    "그림", "도식", "도표", "시각", "시각자료", "시각 자료",
    "이미지", "다이어그램", "위주",
})

_APPLICATION_CONDITION_TERMS = frozenset({
    "예제", "예시", "사례", "실습", "시연", "데모", "적용", "활용",
    "문제풀이", "풀이", "이론만 말고",
})

_DELIVERY_CONDITION_TERMS = frozenset({
    "음질", "녹음", "청취", "듣기", "발화", "말", "빠르지", "느리",
    "천천히", "천천", "여유", "명료",
})

_RECENCY_CONDITION_TERMS = frozenset({
    "최근", "최신", "새로운", "새로", "새로 올라온", "업로드", "업데이트",
    "요즘", "근래", "최근 업로드", "최신 강의",
})

_NON_CONTENT_QUERY_TERMS = frozenset({
    "강의", "추천", "내용", "설명", "요약", "개념", "주제", "관련",
    "기초", "입문", "초급", "쉬운", "쉽게", "쉬움", "설명이 쉬운",
    "알려줘", "찾아줘", "보여줘", "내외", "이내", "이하", "정도",
})


def _strip_non_content_modifiers(term: str, blocked: set[str]) -> str:
    cleaned = _normalize_term(term)
    for blocked_term in sorted(blocked, key=len, reverse=True):
        cleaned = cleaned.replace(blocked_term, " ")
    return re.sub(r"\s+", " ", cleaned).strip()


def _content_terms_only(terms: list[str], conditions: dict) -> list[str]:
    """
    LLM이 조건 표현(그림/도식/음질/예제 등)을 query_keywords에 넣어도
    원본 키워드 미매칭 패널티가 내용 검색을 죽이지 않도록 제거한다.
    조건 감지는 LLM에 맡기고, 여기서는 이미 감지된 조건의 수식어만 정리한다.
    """
    blocked: set[str] = set()
    if conditions.get("prefers_visual"):
        blocked.update(_VISUAL_CONDITION_TERMS)
    if conditions.get("prefers_application"):
        blocked.update(_APPLICATION_CONDITION_TERMS)
    if conditions.get("prefers_slow_speech") or conditions.get("prefers_listenability"):
        blocked.update(_DELIVERY_CONDITION_TERMS)
    if conditions.get("prefers_recency"):
        blocked.update(_RECENCY_CONDITION_TERMS)
    blocked.update(_NON_CONTENT_QUERY_TERMS)

    normalized_blocked = {_normalize_term(term) for term in blocked}
    cleaned = []
    seen = set()
    for term in terms or []:
        normalized = _normalize_term(term)
        if not normalized or normalized in normalized_blocked:
            continue
        stripped = _strip_non_content_modifiers(normalized, normalized_blocked)
        if stripped and stripped not in normalized_blocked and stripped not in seen:
            seen.add(stripped)
            cleaned.append(stripped)
    return cleaned


def _required_subject_terms(ctx) -> set[str]:
    if ctx.intent != "recommend":
        return set()
    return {
        _normalize_term(term)
        for term in _expanded_topic_terms(ctx.query_keywords)
        if _normalize_term(term)
    }


def _lecture_subject_terms(lec: LectureMetadata) -> list[str]:
    terms: list[str] = []
    _append_terms(terms, lec.title)
    _append_terms(terms, lec.summary)
    _append_terms(terms, [item.get("keyword", "") for item in (lec.keywords or []) if isinstance(item, dict)])
    _append_terms(terms, lec.concept_roles)
    for relation in lec.concept_relations or []:
        if not isinstance(relation, dict):
            continue
        _append_terms(terms, relation.get("from"))
        _append_terms(terms, relation.get("to"))
        _append_terms(terms, relation.get("source"))
        _append_terms(terms, relation.get("target"))
    for community in lec.communities or []:
        if not isinstance(community, dict):
            continue
        _append_terms(terms, community.get("title"))
        _append_terms(terms, community.get("summary"))
        _append_terms(terms, community.get("nodes"))
    return [_normalize_term(term) for term in terms if _normalize_term(term)]


def _has_required_subject_match(lec: LectureMetadata, required_terms: set[str]) -> bool:
    """
    길이/도메인/전달 조건이 내용 적합성을 대체하지 못하도록,
    원본 질의의 명시 주제가 강의 메타데이터에 직접 걸리는지 확인한다.
    """
    if not required_terms:
        return True
    lecture_terms = _lecture_subject_terms(lec)
    return any(
        _concept_match(lecture_term, required_term)
        for lecture_term in lecture_terms
        for required_term in required_terms
    )


def _parse_uploaded_at(uploaded_at: Optional[str]) -> Optional[datetime]:
    if not uploaded_at:
        return None
    text = str(uploaded_at).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _compute_recency_score(
    uploaded_at: Optional[str],
    half_life_days: float,
    now: Optional[datetime] = None,
) -> float:
    uploaded_dt = _parse_uploaded_at(uploaded_at)
    if uploaded_dt is None:
        return 0.0
    now_dt = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    age_days = max((now_dt - uploaded_dt).total_seconds() / 86400.0, 0.0)
    half_life = max(float(half_life_days or 1.0), 1.0)
    return round(math.exp(-age_days * math.log(2) / half_life), 4)


def _compute_contrast_signal(lec: LectureMetadata) -> float:
    """
    강의 내 대조·비교형 concept_relations 비율.
    반환: 0.0(비교 요소 없음) ~ 1.0(전체가 대조 관계)

    비교형 강의(net1-1: TCP vs UDP 차이 분석)와
    소개형 강의(net1-2: TCP와 UDP 각각 소개)를 구분하는 핵심 신호.
    """
    relations = lec.concept_relations or []
    if not relations:
        return 0.0
    contrast_count = sum(
        1 for rel in relations
        if rel.get("type", "").lower() in _CONTRAST_TYPES
    )
    return round(min(contrast_count / len(relations), 1.0), 4)


def _detect_comparison_intent(query_keywords: list[str], query: str) -> bool:
    """질의에 비교/대조 의도가 있는지 감지"""
    q = query.lower()
    return (
        any(sig in q for sig in _COMPARISON_SIGNALS) or
        any(kw in _COMPARISON_SIGNALS for kw in (query_keywords or []))
    )


def _compute_fragmentation_penalty(concept_roles) -> float:
    """
    core 비율 낮고 introduced 비율 높을수록 패널티.
    반환: 0.0(응집) ~ 1.0(파편화)
    """
    if isinstance(concept_roles, dict):
        n_core  = len(concept_roles.get("core", []))
        n_intro = len(concept_roles.get("introduced", []))
    elif isinstance(concept_roles, list):
        n_core  = sum(1 for cr in concept_roles if isinstance(cr, dict) and cr.get("role") == "core")
        n_intro = sum(1 for cr in concept_roles if isinstance(cr, dict) and cr.get("role") == "introduced")
    else:
        return 0.0

    total = n_core + n_intro
    if total == 0:
        return 0.0

    core_ratio  = n_core  / total
    intro_ratio = n_intro / total
    return float(min(intro_ratio * (1.0 - core_ratio), 1.0))


# ============================================================================
#  OpenAI — 자연어 질의 분석
# ============================================================================

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


_ANALYZE_QUERY_CACHE: dict[tuple[str, tuple[str, ...], int], tuple] = {}


def _analysis_cache_key(query: str, available_domains: list[str], available_keywords: list[str]) -> tuple[str, tuple[str, ...], int]:
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


def analyze_query(
    query:              str,
    available_domains:  list[str],
    available_keywords: list[str],
) -> tuple[str, str, list[str], list[str], Optional[str], Optional[str], Optional[int], dict]:
    """
    질의 → intent + search_text + query_keywords + inferred_keywords + domain + focus_concept + duration_max_sec 추출.

    반환:
      intent             : "recommend" | "list_by_domain" | "list_by_topic"
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
  "intent": "recommend 또는 list_by_domain 또는 list_by_topic",
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

[intent]: 질의 목적 분류
  - "recommend": 특정 강의를 추천받고 싶은 일반 질의
    예) "가상 메모리 자세히 설명하는 강의 추천해줘"
  - "list_by_domain": 분야/도메인 강의 목록을 묻는 질의
    예) "컴퓨터공학 강의 뭐 있어?", "경제학 강의 목록 보여줘", "전체 강의 뭐 있어?"
  - "list_by_topic": 특정 주제와 관련된 강의를 넓게/모두 보고 싶은 질의
    예) "딥러닝 관련 강의 모두 알려줘", "운영체제 관련 강의 다 보여줘"

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

[conditions]: 내용 조건이 아니라 강의 상태/형식에 대한 선호를 의미 단위로 추출
  - issue_free: "오류 없는", "검증된", "이슈 없는", "틀린 내용 없는" 등
  - prefers_visual: "그림/도식/표/그래프/시각 자료 위주", "시각적으로 설명" 등
  - prefers_application: "예제/사례/문제풀이/실습/시연/데모/적용/활용 중심", "이론만 말고" 등
  - prefers_slow_speech: "말이 느렸으면", "급하게 설명하지 않는", "여유있게 진행", "빠르지 않은" 등
  - prefers_listenability: "음질 좋은", "잘 들리는", "듣기 편한", "녹음 상태 좋은", "전달이 명료한" 등
  - prefers_recency: "최근 업로드된", "최신 강의", "새로 올라온 강의"처럼 강의 업로드 시점이 최근이기를 선호하는 경우
    * 단순히 "최신 Transformer", "최신 기술 동향"처럼 주제 자체의 최신성을 말하는 경우는 업로드 시점 선호가 명확할 때만 true

[키워드 목록]: {keyword_list}"""

    text   = _call_query_analysis_llm(prompt).replace("```json", "").replace("```", "").strip()
    parsed = json.loads(text)

    intent            = parsed.get("intent") or "recommend"
    query_keywords    = parsed.get("query_keywords", [])
    inferred_keywords = parsed.get("inferred_keywords", [])
    domain            = parsed.get("domain") or None
    focus_concept     = parsed.get("focus_concept") or None
    duration_max_sec  = parsed.get("duration_max_sec") or None
    conditions        = parsed.get("conditions") if isinstance(parsed.get("conditions"), dict) else {}

    normalized_conditions = {
        "issue_free": bool(conditions.get("issue_free")),
        "prefers_visual": bool(conditions.get("prefers_visual")),
        "prefers_application": bool(conditions.get("prefers_application")),
        "prefers_slow_speech": bool(conditions.get("prefers_slow_speech")),
        "prefers_listenability": bool(conditions.get("prefers_listenability")),
        "prefers_recency": bool(conditions.get("prefers_recency")),
    }
    query_keywords = _content_terms_only(query_keywords, normalized_conditions)
    inferred_keywords = _content_terms_only(inferred_keywords, normalized_conditions)

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
    if intent not in ("recommend", "list_by_domain", "list_by_topic"):
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
    display_score: Optional[int]
    duration_sec: float
    score_detail: dict
    reason:       str
    summary:      str
    tier:         str   # "direct" | "related"
    keywords:     list[dict]


@dataclass
class QueryContext:
    query:             str
    intent:            str
    search_text:       str
    query_keywords:    list[str]
    inferred_keywords: list[str]
    domain:            Optional[str]
    subdomain:         Optional[str]
    focus_concept:     Optional[str]
    duration_max_sec:  Optional[int]
    comparison_intent: bool
    issue_free_preference: bool
    visual_preference: bool
    application_preference: bool
    listenability_preference: bool
    slow_speech_preference: bool
    recency_preference: bool


def _build_reason(detail: dict, tier: str = "direct") -> str:
    if tier == "related":
        parts = []
        if detail.get("dm_keyword", 0) > 0.1:
            parts.append(f"키워드 일부 포함 {detail['dm_keyword']:.0%}")
        if detail.get("sim_keyword", 0) >= 0.6:
            parts.append(f"주제 근접 {detail['sim_keyword']:.0%}")
        if detail.get("domain_score", 0) == 1.0:
            parts.append("도메인 일치")
        note = " · ".join(parts) if parts else "주변 개념 포함"
        return f"질의 주제와 직접 일치하지 않지만 관련 개념을 포함합니다 ({note})"

    parts = []
    if detail.get("domain_score", 0) == 1.0:
        parts.append("도메인 일치")
    if detail.get("depth_score", 0) > 0.3:
        parts.append(f"개념 깊이 {detail['depth_score']:.0%}")
    if detail.get("contrast_bonus", 0) > 0:
        parts.append("비교 분석형")
    if detail.get("application_preference") and detail.get("application_score", 0) > 0:
        parts.append(f"예제/시연 지향 {detail['application_score']:.0%}")
    if detail.get("listenability_preference") and detail.get("listenability_score", 0) > 0:
        parts.append(f"청취 품질 {detail['listenability_score']:.0%}")
    if detail.get("slow_speech_preference") and detail.get("speech_rate_score", 0) > 0:
        parts.append(f"발화 속도 적합 {detail['speech_rate_score']:.0%}")
    if detail.get("recency_preference") and detail.get("recency_score", 0) >= 0.6:
        parts.append("최근 업로드")
    if detail.get("sim_keyword", 0) >= 0.6:
        parts.append(f"키워드 유사도 {detail['sim_keyword']:.0%}")
    if detail.get("dm_keyword", 0) > 0.1:
        parts.append(f"키워드 직접 매칭 {detail['dm_keyword']:.0%}")
    if detail.get("frag_penalty", 0) > 0.3:
        parts.append(f"파편화 -{detail['frag_penalty']:.0%}")
    return " · ".join(parts) if parts else "관련 강의"


DISPLAY_SCORE_FULL_RATIO = 0.8
CONDITION_DISPLAY_SHARE = 20
CONTENT_MEANING_DISPLAY_SHARE = 100 - CONDITION_DISPLAY_SHARE


def _ratio(value: Any) -> float:
    try:
        numeric = float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return numeric / 100.0 if numeric > 1.0 else numeric


def _display_component_score(value: Any) -> float:
    return max(0.0, min(_ratio(value) / DISPLAY_SCORE_FULL_RATIO, 1.0)) * 100.0


def _condition_score_ratio(detail: dict) -> float:
    condition_scores: list[float] = []
    warnings = detail.get("condition_warnings") if isinstance(detail.get("condition_warnings"), list) else []
    has_duration_condition = bool(_ratio(detail.get("duration_score")) > 0.0 or detail.get("duration_mismatch") is True)

    if has_duration_condition:
        condition_scores.append(_ratio(detail.get("duration_score")))
    if detail.get("visual_preference"):
        condition_scores.append(max(
            _ratio(detail.get("visual_score")),
            _ratio(detail.get("visual_density_score")),
            _ratio(detail.get("visual_concept_score")),
        ))
    if detail.get("application_preference"):
        condition_scores.append(_ratio(detail.get("application_score")))
    if detail.get("listenability_preference"):
        condition_scores.append(_ratio(detail.get("listenability_score")))
    if detail.get("slow_speech_preference"):
        condition_scores.append(_ratio(detail.get("speech_rate_score")))
    if detail.get("recency_preference"):
        condition_scores.append(_ratio(detail.get("recency_score")))

    if condition_scores:
        combined = _ratio(detail.get("combined_boost"))
        if combined > 0.0:
            return combined
        return sum(condition_scores) / len(condition_scores)

    return _ratio(detail.get("combined_boost")) if warnings else 1.0


def _content_display_share(content_ratio: float, meaning_ratio: float) -> int:
    total = max(content_ratio, 0.0) + max(meaning_ratio, 0.0)
    if total <= 0.0:
        return CONTENT_MEANING_DISPLAY_SHARE // 2
    return int(round((max(content_ratio, 0.0) / total) * CONTENT_MEANING_DISPLAY_SHARE))


def _display_score(internal_score: float, tier: str, detail: Optional[dict] = None) -> int:
    """
    내부 랭킹 점수를 사용자 표시용 추천 적합도로 변환한다.
    추천 순위와 tier 판단에는 영향을 주지 않는다.
    """
    if not detail:
        return int(round(max(0.0, min(internal_score, 1.0)) * 100))

    content_ratio = _ratio(detail.get("content_score", _ratio(detail.get("content_pct"))))
    meaning_ratio = max(
        _ratio(detail.get("vec_score")),
        _ratio(detail.get("graph_score")),
        _ratio(detail.get("sim_keyword")),
    )
    condition_ratio = _condition_score_ratio(detail)

    content_share = _content_display_share(content_ratio, meaning_ratio)
    meaning_share = CONTENT_MEANING_DISPLAY_SHARE - content_share

    score = (
        _display_component_score(content_ratio) * (content_share / 100.0)
        + _display_component_score(meaning_ratio) * (meaning_share / 100.0)
        + _display_component_score(condition_ratio) * (CONDITION_DISPLAY_SHARE / 100.0)
    )
    return int(round(score))


# ============================================================================
#  추천 엔진
# ============================================================================

class Recommender:
    def __init__(self, metadata_dir: str = DEFAULT_METADATA_DIR, config: Optional[RecommenderConfig] = None):
        self.collection          = MetadataCollection(metadata_dir)
        self.cfg                 = config or RecommenderConfig()
        self._available_domains  = self.collection.available_domains()
        self._available_subdomains = self.collection.available_subdomains()
        self._available_keywords = self.collection.available_keywords()

        # LanceDB 전체 레코드 사전 로드 (요청마다 디스크 읽기 방지).
        # 테이블이 아직 생성되지 않은 초기 상태에서는 벡터 검색만 비활성화하고,
        # metadata 기반 BM25/직접매칭/그래프 점수로 추천을 계속 제공한다.
        print("[LanceDB 레코드 로드 중...]")
        self._index_rows = [
            row
            for row in self._load_lancedb_rows()
            if self.collection.get(str(row.get("video_id") or "")) is not None
        ]
        self._row_by_video_id = {
            row["video_id"]: row
            for row in self._index_rows
        }
        self._vector_search_index = _build_vector_search_index(self._index_rows)
        indexed_lectures = [
            lec
            for video_id in self._row_by_video_id
            if (lec := self.collection.get(video_id)) is not None
        ]
        metadata_lectures = self.collection.all()
        scoring_lectures = indexed_lectures if indexed_lectures else metadata_lectures
        self._lexical_stats = _build_lexical_stats(scoring_lectures)
        self._community_index = CommunityIndex(scoring_lectures)
        print(f"  → {len(self._index_rows)}개 레코드 로드\n")
        if not self._index_rows and metadata_lectures:
            print("  ⚠ LanceDB lectures 테이블 없음/비어 있음 — 벡터 검색 없이 metadata 기반 추천으로 동작합니다.\n")
        elif len(indexed_lectures) < len(metadata_lectures):
            print(
                f"  ⚠ 벡터 인덱스에 없는 metadata 강의 "
                f"{len(metadata_lectures) - len(indexed_lectures)}개는 metadata 기반으로만 점수화합니다.\n"
            )
        print(f"[도메인]    {self._available_domains}")
        print(f"[세부도메인] {self._available_subdomains}")
        print(f"[키워드 풀] {len(self._available_keywords)}개\n")
        print(
            f"[Lexical]   {len(self._lexical_stats.doc_freq)}개 term, "
            f"avg_len={self._lexical_stats.avg_doc_len:.1f}\n"
        )
        print(f"[Vector]    matrix rows={len(self._vector_search_index.video_ids)}\n")

    def _load_lancedb_rows(self) -> list[dict]:
        try:
            db = lancedb.connect(self.cfg.DB_DIR)
            table_names = set(db.table_names())
            if "lectures" not in table_names:
                print(f"  ⚠ LanceDB 테이블 없음: lectures ({self.cfg.DB_DIR})")
                return []
            table = db.open_table("lectures")
            return table.to_arrow().to_pylist()
        except Exception as exc:
            print(f"  ⚠ LanceDB 로드 실패: {exc}")
            return []

    def _prepare_query_context(self, query: str) -> QueryContext:
        print(f"[질의 분석] {query}")
        fast_analysis = _fast_list_by_domain_analysis(query, self._available_domains, self._available_subdomains)
        if fast_analysis:
            intent, search_text, query_keywords, inferred_keywords, domain, focus_concept, duration_max_sec, conditions = fast_analysis
        else:
            intent, search_text, query_keywords, inferred_keywords, domain, focus_concept, duration_max_sec, conditions = analyze_query(
                query, self._available_domains, self._available_keywords
            )
        subdomain = _normalize_subdomain(conditions.get("subdomain"))
        if not subdomain:
            inferred_domain, inferred_subdomain = _infer_domain_filters(
                query,
                self._available_domains,
                self._available_subdomains,
            )
            domain = inferred_domain or domain
            subdomain = _normalize_subdomain(inferred_subdomain)
        if subdomain not in self._available_subdomains:
            subdomain = None
        inferred_keywords = _append_topic_expansions(
            query_keywords,
            inferred_keywords,
            seed_terms=[query],
        )
        search_text = " ".join(query_keywords + inferred_keywords) or search_text or query
        comparison_intent = _detect_comparison_intent(query_keywords, query)
        issue_free_preference = bool(conditions.get("issue_free"))
        visual_preference = bool(conditions.get("prefers_visual"))
        application_preference = bool(conditions.get("prefers_application"))
        listenability_preference = bool(conditions.get("prefers_listenability"))
        slow_speech_preference = bool(conditions.get("prefers_slow_speech"))
        recency_preference = bool(conditions.get("prefers_recency"))

        print(f"[질의 의도]   {intent}")
        print(f"[원본 키워드] {query_keywords}")
        print(f"[확장 키워드] {inferred_keywords}")
        print(f"[추론 도메인] {domain or '미확정'}")
        print(f"[추론 세부]   {subdomain or '미확정'}")
        print(f"[깊이 개념]   {focus_concept or '없음'}")
        print(f"[비교 의도]   {'있음' if comparison_intent else '없음'}")
        print(f"[검증 조건]   {'있음' if issue_free_preference else '없음'}")
        print(f"[시각 선호]   {'있음' if visual_preference else '없음'}")
        print(f"[적용/시연]   {'있음' if application_preference else '없음'}")
        print(f"[청취 품질]   {'있음' if listenability_preference else '없음'}")
        print(f"[발화 속도]   {'빠르지 않음 선호' if slow_speech_preference else '없음'}")
        print(f"[최신성]     {'최근 업로드 선호' if recency_preference else '없음'}")
        if duration_max_sec:
            print(f"[길이 조건]   기준 {duration_max_sec//60}분 ({duration_max_sec}초) — 조건 boost + warning 적용")
        print()

        return QueryContext(
            query             = query,
            intent            = intent,
            search_text       = search_text,
            query_keywords    = query_keywords,
            inferred_keywords = inferred_keywords,
            domain            = domain,
            subdomain         = subdomain,
            focus_concept     = focus_concept,
            duration_max_sec  = duration_max_sec,
            comparison_intent = comparison_intent,
            issue_free_preference = issue_free_preference,
            visual_preference = visual_preference,
            application_preference = application_preference,
            listenability_preference = listenability_preference,
            slow_speech_preference = slow_speech_preference,
            recency_preference = recency_preference,
        )

    def _all_candidate_ids(self) -> list[str]:
        return [lec.video_id for lec in self.collection.all()]

    def _list_by_domain_results(self, ctx: QueryContext, top_k: int) -> list[RecommendResult]:
        lectures = []
        for lec in self.collection.all():
            if ctx.domain and lec.domain != ctx.domain:
                continue
            if ctx.subdomain and lec.graph_subdomain != ctx.subdomain:
                continue
            lectures.append(lec)

        lectures.sort(key=lambda lec: lec.title or lec.video_id)

        scope = _DOMAIN_LABELS.get(ctx.domain, ctx.domain) if ctx.domain else "전체"
        if ctx.subdomain:
            scope = f"{scope} / {ctx.subdomain}"
        reason = "전체 강의 목록입니다." if not ctx.domain else f"{scope} 분야 강의 목록입니다."
        results = []
        for lec in lectures[:top_k]:
            results.append(RecommendResult(
                video_id      = lec.video_id,
                title         = lec.title,
                domain        = lec.domain,
                instructor    = lec.instructor_id,
                score         = 0.0,
                display_score = None,
                duration_sec  = lec.duration_sec,
                score_detail  = {},
                reason        = reason,
                summary       = lec.summary,
                tier          = "list",
                keywords      = lec.keywords or [],
            ))
        return results

    def _rrf_fuse(
        self,
        rankings: list[list[str]],
        top_n: Optional[int] = None,
    ) -> list[str]:
        scores: defaultdict[str, float] = defaultdict(float)
        best_rank: dict[str, int] = {}

        for ranking in rankings:
            for rank, video_id in enumerate(ranking, start=1):
                scores[video_id] += 1.0 / (self.cfg.RRF_K + rank)
                best_rank[video_id] = min(best_rank.get(video_id, rank), rank)

        fused = sorted(
            scores,
            key=lambda video_id: (
                -scores[video_id],
                best_rank.get(video_id, 10**9),
                video_id,
            ),
        )
        limit = top_n or self.cfg.HYBRID_CANDIDATE_TOP_N
        return fused[:limit]

    def _metadata_preferred_ids(self, ctx: QueryContext) -> Optional[set[str]]:
        """
        metadata 조건을 만족하는 우선 후보군.
        hard filter가 아니라 RRF에 preferred ranking을 추가하는 soft signal로만 쓴다.
        """
        if not self.cfg.USE_METADATA_PREFILTER:
            return None
        if not (ctx.domain or ctx.subdomain or ctx.duration_max_sec):
            return None

        preferred = set()
        for lec in self.collection.all():
            if ctx.domain and lec.domain != ctx.domain:
                continue
            if ctx.subdomain and lec.graph_subdomain != ctx.subdomain:
                continue
            if ctx.duration_max_sec and lec.duration_sec > (
                ctx.duration_max_sec + self.cfg.METADATA_DURATION_GRACE_SEC
            ):
                continue

            preferred.add(lec.video_id)

        return preferred or None

    def _get_initial_candidate_ids(self, ctx: QueryContext, query_vec: list[float]) -> list[str]:
        """
        BM25 lexical 후보와 vector semantic 후보를 RRF로 통합한다.
        후보가 너무 적으면 기존 전체 후보 방식으로 되돌린다.
        """
        preferred_ids = self._metadata_preferred_ids(ctx)
        bm25_candidates = self._retrieve_bm25_candidates(
            ctx,
            top_n=self.cfg.BM25_RETRIEVE_TOP_N,
        )
        vector_candidates = self._retrieve_vector_candidates(
            query_vec,
            top_n=self.cfg.VECTOR_RETRIEVE_TOP_N,
        )
        rankings = [
            [video_id for video_id, _score in bm25_candidates],
            [video_id for video_id, _score in vector_candidates],
        ]

        pref_bm25_candidates = []
        pref_vector_candidates = []
        if preferred_ids:
            pref_bm25_candidates = self._retrieve_bm25_candidates(
                ctx,
                eligible_ids=preferred_ids,
                top_n=self.cfg.BM25_RETRIEVE_TOP_N,
            )
            pref_vector_candidates = self._retrieve_vector_candidates(
                query_vec,
                eligible_ids=preferred_ids,
                top_n=self.cfg.VECTOR_RETRIEVE_TOP_N,
            )
            rankings.extend([
                [video_id for video_id, _score in pref_bm25_candidates],
                [video_id for video_id, _score in pref_vector_candidates],
            ])

        candidate_ids = self._rrf_fuse(
            rankings,
            top_n=self.cfg.HYBRID_CANDIDATE_TOP_N,
        )

        print(
            f"[후보 검색] BM25 {len(bm25_candidates)}개, "
            f"Vector {len(vector_candidates)}개, RRF {len(candidate_ids)}개"
        )
        if preferred_ids:
            print(
                f"[메타데이터 우선 후보] {len(preferred_ids)}개 "
                f"(BM25 {len(pref_bm25_candidates)}개, "
                f"Vector {len(pref_vector_candidates)}개)"
            )

        if len(candidate_ids) < self.cfg.MIN_HYBRID_CANDIDATES:
            all_ids = self._all_candidate_ids()
            print(
                f"[후보 검색] RRF 후보 부족({len(candidate_ids)}개) — "
                f"전체 후보 {len(all_ids)}개로 fallback"
            )
            return all_ids

        return candidate_ids

    def _query_lexical_terms(self, ctx: QueryContext) -> Counter:
        """
        BM25/TF-IRF에서 공유할 질의 term 구성.
        원본 키워드는 강하게, LLM 확장 키워드는 약하게 반영한다.
        """
        terms = Counter()

        def add_term(text: str, weight: float) -> None:
            phrase = _normalize_term(text)
            if not phrase:
                return
            terms[phrase] += weight
            for token in _tokenize_text(phrase):
                if token != phrase:
                    terms[token] += weight * 0.5

        for keyword in ctx.query_keywords:
            add_term(keyword, 1.0)
        for keyword in ctx.inferred_keywords:
            add_term(keyword, 0.5)

        if not terms:
            for token in _tokenize_text(ctx.search_text or ctx.query):
                terms[token] += 1.0

        return terms

    def _resolve_rerank_weights(self, ctx: QueryContext) -> dict[str, float]:
        if ctx.visual_preference:
            return {
                "content": self.cfg.W_CONTENT_VISUAL_QUERY,
                "graph": self.cfg.W_GRAPH_VISUAL_QUERY,
                "community": self.cfg.W_COMMUNITY_VISUAL_QUERY,
                "visual": self.cfg.W_VISUAL_QUERY,
                "boost": self.cfg.W_BOOST_VISUAL_QUERY,
            }
        return {
            "content": self.cfg.W_CONTENT,
            "graph": self.cfg.W_GRAPH,
            "community": self.cfg.W_COMMUNITY,
            "visual": self.cfg.W_VISUAL,
            "boost": self.cfg.W_BOOST,
        }

    @staticmethod
    def _visual_density_score(lec: LectureMetadata) -> float:
        pedagogy = lec.pedagogy or {}
        teaching_style = (lec.diagnostics or {}).get("teaching_style", {})

        def as_float(value) -> float:
            try:
                parsed = float(value)
                return parsed if math.isfinite(parsed) else 0.0
            except (TypeError, ValueError):
                return 0.0

        visual_ratio = as_float(teaching_style.get("visual_ratio", pedagogy.get("visual_ratio")))
        structure_ratio = as_float(pedagogy.get("structure_ratio"))
        return round(min(max(0.5 * visual_ratio + 0.5 * structure_ratio, 0.0), 1.0), 4)

    @staticmethod
    def _diagnostic_score(lec: LectureMetadata, section: str, key: str) -> float:
        value = ((lec.diagnostics or {}).get(section, {}) or {}).get(key)
        try:
            parsed = float(value)
            return min(max(parsed, 0.0), 1.0) if math.isfinite(parsed) else 0.0
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _speech_rate_fit_score(lec: LectureMetadata) -> float:
        delivery = ((lec.diagnostics or {}).get("delivery", {}) or {})
        try:
            spm = float(delivery.get("speech_rate_spm"))
        except (TypeError, ValueError):
            return 0.0
        if not math.isfinite(spm) or spm <= 0:
            return 0.0
        # "천천히/빠르지 않게" 조건은 낮은 SPM일수록 더 적합하게 본다.
        # 임계값은 더미 데이터 분포 기준 임시값이며, 실제 데이터 분포로 재보정한다.
        if spm <= 240:
            return 1.0
        if spm >= 330:
            return 0.0
        return round(1.0 - ((spm - 240) / 90), 4)

    def _recency_boost_weight(self, domain: str) -> float:
        normalized = _normalize_term(domain)
        if normalized == "engineering":
            return self.cfg.W_RECENCY_BOOST_FAST
        if normalized in {"social_science", "health_sciences"}:
            return self.cfg.W_RECENCY_BOOST_MEDIUM
        return self.cfg.W_RECENCY_BOOST_SLOW

    @staticmethod
    def _visual_concept_score(lec: LectureMetadata, query_terms: Counter) -> float:
        if not query_terms:
            return 0.0
        visual_terms = set()
        for term in lec.visual_concept_terms or []:
            normalized = _normalize_term(term)
            if not normalized:
                continue
            visual_terms.add(normalized)
            visual_terms.update(_tokenize_text(normalized))
        if not visual_terms:
            return 0.0

        denom = sum(max(weight, 0.0) for weight in query_terms.values()) or 1.0
        hit = 0.0
        for term, weight in query_terms.items():
            if term in visual_terms:
                hit += max(weight, 0.0)
        return round(min(hit / denom, 1.0), 4)

    def _compute_visual_score(
        self,
        lec: LectureMetadata,
        ctx: QueryContext,
        query_terms: Counter,
    ) -> tuple[float, float, float]:
        density_score = self._visual_density_score(lec)
        concept_score = self._visual_concept_score(lec, query_terms)
        if ctx.visual_preference:
            visual_score = 0.65 * density_score + 0.35 * concept_score
        else:
            visual_score = 0.5 * density_score
        return (
            round(min(max(visual_score, 0.0), 1.0), 4),
            density_score,
            concept_score,
        )

    def _bm25_tf(self, doc: LectureLexicalDocument, term: str) -> float:
        return (
            self.cfg.BM25_TITLE_WEIGHT   * doc.field_tf["title"].get(term, 0.0) +
            self.cfg.BM25_KEYWORD_WEIGHT * doc.field_tf["keyword"].get(term, 0.0) +
            self.cfg.BM25_SUMMARY_WEIGHT * doc.field_tf["summary"].get(term, 0.0)
        )

    def _bm25_score_document(self, doc: LectureLexicalDocument, query_terms: Counter) -> float:
        stats = self._lexical_stats
        avg_len = stats.avg_doc_len or 1.0
        doc_len = doc.doc_len or avg_len
        norm = self.cfg.BM25_K1 * (
            1.0 - self.cfg.BM25_B + self.cfg.BM25_B * (doc_len / avg_len)
        )

        score = 0.0
        for term, query_weight in query_terms.items():
            tf = self._bm25_tf(doc, term)
            if tf <= 0:
                continue
            idf = stats.bm25_idf.get(term, 0.0)
            score += query_weight * idf * (
                tf * (self.cfg.BM25_K1 + 1.0)
                / (tf + norm)
            )
        return score

    def _retrieve_bm25_candidates(
        self,
        ctx: QueryContext,
        eligible_ids: Optional[set[str]] = None,
        top_n: int = 50,
    ) -> list[tuple[str, float]]:
        """
        lexical 후보 검색기. vector 후보와 RRF로 통합해 1차 후보군을 만든다.
        """
        query_terms = self._query_lexical_terms(ctx)
        if not query_terms:
            return []

        scores = []
        for video_id, doc in self._lexical_stats.documents.items():
            if eligible_ids is not None and video_id not in eligible_ids:
                continue
            score = self._bm25_score_document(doc, query_terms)
            if score > 0:
                scores.append((video_id, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_n]

    def _retrieve_vector_candidates(
        self,
        query_vec: list[float],
        eligible_ids: Optional[set[str]] = None,
        top_n: Optional[int] = None,
    ) -> list[tuple[str, float]]:
        """
        semantic 후보 검색기. 기존 vec_score와 같은 필드 가중치로 top-N을 뽑는다.
        """
        index = self._vector_search_index
        if not index.video_ids or not query_vec:
            return []

        q = _normalize_vector(query_vec)
        sim_title = index.title_matrix @ q
        sim_keyword = index.keyword_matrix @ q
        sim_summary = index.summary_matrix @ q
        sim_keyword_filtered = np.where(
            sim_keyword >= self.cfg.KW_VEC_THRESHOLD,
            sim_keyword,
            0.0,
        )
        vec_scores = (
            self.cfg.W_TITLE   * sim_title +
            self.cfg.W_KEYWORD * sim_keyword_filtered +
            self.cfg.W_SUMMARY * sim_summary
        )

        limit = top_n or self.cfg.VECTOR_RETRIEVE_TOP_N
        scored = []
        for idx, video_id in enumerate(index.video_ids):
            if eligible_ids is not None and video_id not in eligible_ids:
                continue
            score = float(vec_scores[idx])
            if score > 0:
                scored.append((video_id, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:limit]

    def _score_candidate(
        self,
        row: Optional[dict],
        lec: LectureMetadata,
        ctx: QueryContext,
        query_vec: list[float],
        query_concepts: set[str],
        query_terms: Counter,
        weights: dict[str, float],
    ) -> dict:
        # ── 길이 조건 boost + warning ─────────────────────────────
        # 내용 적합성을 깎지 않고, 요청 길이(+grace)를 만족하는 후보만 boost한다.
        condition_warnings: list[str] = []
        if ctx.duration_max_sec:
            duration_fit_score = (
                1.0
                if lec.duration_sec <= ctx.duration_max_sec + self.cfg.METADATA_DURATION_GRACE_SEC
                else 0.0
            )
            if lec.duration_sec > ctx.duration_max_sec:
                over_min = math.ceil((lec.duration_sec - ctx.duration_max_sec) / 60.0)
                condition_warnings.append(f"요청한 길이보다 약 {over_min}분 깁니다.")
        else:
            duration_fit_score = 0.0

        # 필드별 코사인 유사도. LanceDB row가 없으면 벡터 성분만 0으로 둔다.
        has_vector_row = row is not None and bool(query_vec)
        if has_vector_row:
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
        else:
            sim_title = sim_keyword = sim_summary = sim_keyword_filtered = vec_score = 0.0

        # 직접 토큰 매칭 — 원본 키워드 100%, 추론 키워드 50% 반영
        if self.cfg.USE_TF_IRF_DM:
            dm = _direct_match_score_tfirf(
                ctx.query_keywords,
                ctx.inferred_keywords,
                lec,
                self._lexical_stats,
                idf_floor=self.cfg.TF_IRF_IDF_FLOOR,
            )
        else:
            dm = _direct_match_score(ctx.query_keywords, ctx.inferred_keywords, lec)
        dm_score = (
            self.cfg.W_TITLE   * dm["title"]   +
            self.cfg.W_KEYWORD * dm["keyword"]  +
            self.cfg.W_SUMMARY * dm["summary"]
        )

        # 블렌딩 — content 최대 0.85로 제한 (boost 여유 확보)
        vec_blend = self.cfg.VEC_BLEND if has_vector_row else 0.0
        content_score = min(
            vec_blend       * vec_score +
            (1 - vec_blend) * dm_score,
            0.85
        )

        # domain boost 신호
        domain_score = 0.0
        subdomain_score = 0.0
        if ctx.domain and lec.domain == ctx.domain:
            domain_score = 1.0
        if ctx.subdomain and lec.graph_subdomain == ctx.subdomain:
            subdomain_score = 1.0
            domain_score = 1.0

        # depth boost 신호 — BFS 홉 거리 기반
        depth_score = _compute_depth_score(ctx.focus_concept, lec) if ctx.focus_concept else 0.0

        graph_score = _compute_graph_score(
            lec,
            query_concepts,
            core_weight=self.cfg.GRAPH_CORE_WEIGHT,
            intro_weight=self.cfg.GRAPH_INTRO_WEIGHT,
        )
        community_score = self._community_index.score(
            lec.video_id,
            query_terms,
        )
        visual_score, visual_density_score, visual_concept_score = self._compute_visual_score(
            lec,
            ctx,
            query_terms,
        )
        application_score = self._diagnostic_score(
            lec,
            "teaching_style",
            "application_orientation_score",
        )
        listenability_score = self._diagnostic_score(
            lec,
            "delivery",
            "listenability_score",
        )
        speech_rate_score = self._speech_rate_fit_score(lec)
        recency_score = _compute_recency_score(
            lec.uploaded_at,
            half_life_days=self.cfg.RECENCY_HALF_LIFE_DAYS,
        )
        recency_weight = (
            self._recency_boost_weight(lec.domain)
            if ctx.recency_preference and lec.uploaded_at
            else 0.0
        )
        if ctx.visual_preference and visual_density_score < 0.3:
            condition_warnings.append("시각 자료 비중이 높지 않습니다.")
        if ctx.application_preference and application_score < 0.2:
            condition_warnings.append("예제/시연 지향 신호가 약합니다.")
        if ctx.listenability_preference and listenability_score < 0.4:
            condition_warnings.append("청취 품질 신호가 높지 않습니다.")
        if ctx.slow_speech_preference and speech_rate_score < 0.3:
            condition_warnings.append("발화 속도가 빠른 편일 수 있습니다.")

        # ── 가중합 구조 점수 ──────────────────────────────────
        MAX_BOOST = (
            self.cfg.W_DOMAIN_BOOST +
            self.cfg.W_DEPTH_BOOST +
            (self.cfg.W_DURATION_BOOST if ctx.duration_max_sec else 0.0) +
            (self.cfg.W_APPLICATION_BOOST if ctx.application_preference else 0.0) +
            (self.cfg.W_LISTENABILITY_BOOST if ctx.listenability_preference else 0.0) +
            (self.cfg.W_SPEECH_RATE_BOOST if ctx.slow_speech_preference else 0.0) +
            recency_weight
        )
        raw_boost = (
            self.cfg.W_DOMAIN_BOOST     * (subdomain_score or domain_score) +
            self.cfg.W_DEPTH_BOOST      * depth_score +
            (self.cfg.W_DURATION_BOOST * duration_fit_score if ctx.duration_max_sec else 0.0) +
            (self.cfg.W_APPLICATION_BOOST * application_score if ctx.application_preference else 0.0) +
            (self.cfg.W_LISTENABILITY_BOOST * listenability_score if ctx.listenability_preference else 0.0) +
            (self.cfg.W_SPEECH_RATE_BOOST * speech_rate_score if ctx.slow_speech_preference else 0.0) +
            recency_weight * recency_score
        )
        boost_signal = raw_boost / MAX_BOOST if MAX_BOOST > 0 else 0.0

        # 파편화 패널티
        frag = _compute_fragmentation_penalty(lec.concept_roles)

        total = max(
            weights["content"] * content_score
            + weights["graph"] * graph_score
            + weights["community"] * community_score
            + weights["visual"] * visual_score
            + weights["boost"] * boost_signal
            - self.cfg.FRAG_PENALTY_WEIGHT * frag,
            0.0
        )

        # ── 비교 분석형 보너스 ────────────────────────────────
        # 비교 의도 질의 + 강의 내 대조형 relation 비율 → 가산
        contrast_signal = _compute_contrast_signal(lec)
        contrast_bonus  = (
            self.cfg.W_CONTRAST_BOOST * contrast_signal
            if ctx.comparison_intent else 0.0
        )
        total = min(total + contrast_bonus, 1.0)

        # ── 패널티 체계 ───────────────────────────────────────
        # 도메인 상위 카테고리 불일치 패널티
        if ctx.subdomain and lec.graph_subdomain != ctx.subdomain:
            total *= self.cfg.DOMAIN_MISMATCH_PENALTY
        elif ctx.domain:
            if ctx.domain != lec.domain:
                total *= self.cfg.DOMAIN_MISMATCH_PENALTY

        return {
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
            "dm_keyword_legacy":    round(dm.get("keyword_legacy", dm["keyword"]), 4),
            "dm_summary":           round(dm["summary"], 4),
            "domain_score":         round(domain_score, 4),
            "subdomain_score":      round(subdomain_score, 4),
            "q_kw_matched":         dm.get("q_kw_matched", True),
            "domain_mismatch":      bool(
                (ctx.subdomain and ctx.subdomain != lec.graph_subdomain)
                or (ctx.domain and ctx.domain != lec.domain)
            ),
            "graph_score":          round(graph_score, 4),
            "community_score":      round(community_score, 4),
            "visual_score":         round(visual_score, 4),
            "visual_density_score": round(visual_density_score, 4),
            "visual_concept_score": round(visual_concept_score, 4),
            "visual_preference":    ctx.visual_preference,
            "application_score":    round(application_score, 4),
            "application_preference": ctx.application_preference,
            "listenability_score":  round(listenability_score, 4),
            "listenability_preference": ctx.listenability_preference,
            "speech_rate_score":    round(speech_rate_score, 4),
            "slow_speech_preference": ctx.slow_speech_preference,
            "recency_score":        round(recency_score, 4),
            "recency_preference":   ctx.recency_preference,
            "recency_weight":       round(recency_weight, 4),
            "weight_content":       round(weights["content"], 4),
            "weight_graph":         round(weights["graph"], 4),
            "weight_community":     round(weights["community"], 4),
            "weight_visual":        round(weights["visual"], 4),
            "weight_boost":         round(weights["boost"], 4),
            "depth_score":          round(depth_score, 4),
            "contrast_signal":      round(contrast_signal, 4),
            "contrast_bonus":       round(contrast_bonus, 4),
            "boost_signal":         round(boost_signal, 4),
            "combined_boost":       round(boost_signal, 4),
            "duration_score":       round(duration_fit_score, 4),
            "duration_mismatch":    bool(ctx.duration_max_sec and lec.duration_sec > ctx.duration_max_sec),
            "condition_warnings":   condition_warnings,
            "frag_penalty":         round(frag, 4),
        }

    def _rank_candidates(
        self,
        candidate_ids: list[str],
        ctx: QueryContext,
        query_vec: list[float],
    ) -> list[tuple[LectureMetadata, dict]]:
        query_concepts = set(ctx.query_keywords) | set(ctx.inferred_keywords)
        query_terms = self._query_lexical_terms(ctx)
        required_subject_terms = _required_subject_terms(ctx)
        weights = self._resolve_rerank_weights(ctx)
        print(
            "[Rerank weights] "
            f"content={weights['content']:.2f} graph={weights['graph']:.2f} "
            f"community={weights['community']:.2f} visual={weights['visual']:.2f} "
            f"boost={weights['boost']:.2f}"
        )
        candidates = []
        for video_id in candidate_ids:
            row = self._row_by_video_id.get(video_id)
            lec = self.collection.get(video_id)
            if lec is None:
                continue
            if not _has_required_subject_match(lec, required_subject_terms):
                continue
            detail = self._score_candidate(
                row,
                lec,
                ctx,
                query_vec,
                query_concepts,
                query_terms,
                weights,
            )
            candidates.append((lec, detail))

        candidates.sort(key=lambda x: x[1]["score"], reverse=True)
        return candidates

    def _print_candidate_scores(self, candidates: list[tuple[LectureMetadata, dict]]) -> None:
        print(f"  {'video_id':<10} {'제목':<24} {'sim_t':>5} {'sim_k':>5} {'sim_k*':>6} {'sim_s':>5} "
              f"{'vec':>5} {'dm':>5} {'graph':>5} {'comm':>5} {'vis':>5} {'boost':>6} {'ctr':>5} {'dur':>5} {'→score':>7}")
        for lec, d in candidates:
            flags = ""
            if d.get("domain_mismatch"):          flags += " !dom"
            if not d.get("q_kw_matched", True):   flags += " !qkw"
            if d.get("contrast_bonus", 0) > 0:    flags += f" +ctr{d['contrast_bonus']:.2f}"
            print(f"  {lec.video_id:<10} {lec.title[:22]:<24} "
                  f"{d['sim_title']:>5.3f} {d['sim_keyword']:>5.3f} "
                  f"{d['sim_keyword_filtered']:>6.3f} {d['sim_summary']:>5.3f} "
                  f"{d['vec_score']:>5.3f} {d['dm_score']:>5.3f} "
                  f"{d['graph_score']:>5.3f} {d['community_score']:>5.3f} "
                  f"{d['visual_score']:>5.3f} "
                  f"{d['boost_signal']:>6.3f} {d['contrast_signal']:>5.3f} "
                  f"{d['duration_score']:>5.3f} {d['score']:>7.3f}{flags}")

    def _classify_tiers(
        self,
        candidates: list[tuple[LectureMetadata, dict]],
        ctx: QueryContext,
        top_k: int,
    ) -> list[RecommendResult]:
        if not candidates:
            return []

        max_score = candidates[0][1]["score"]
        if max_score < self.cfg.ABS_MIN_SCORE:
            print(f"  → top 점수 {max_score:.3f} < ABS_MIN_SCORE {self.cfg.ABS_MIN_SCORE} — 결과 없음")
            return []

        scores = [d["score"] for _, d in candidates]
        if len(scores) >= 2 and (scores[0] - scores[1]) >= self.cfg.GAP_THRESHOLD:
            # 자연 경계도 ABS_DIRECT_FLOOR 이상이어야 함
            direct_threshold = max(
                (scores[0] + scores[1]) / 2,
                self.cfg.ABS_DIRECT_FLOOR,
            )
            print(f"  → 자연 경계 감지 (gap={scores[0]-scores[1]:.3f}): "
                  f"direct_threshold={direct_threshold:.3f}")
        else:
            direct_threshold = max(
                max_score * self.cfg.DIRECT_RATIO,
                self.cfg.ABS_DIRECT_FLOOR,
            )

        related_threshold = max_score * self.cfg.RELATED_RATIO
        print(f"  [임계값] direct ≥ {direct_threshold:.3f} (floor={self.cfg.ABS_DIRECT_FLOOR})  "
              f"core_direct_floor={self.cfg.CORE_MATCH_DIRECT_FLOOR:.2f}  "
              f"related ≥ {related_threshold:.3f}  "
              f"direct_graph ≥ {self.cfg.DIRECT_GRAPH_FLOOR:.2f}  "
              f"related_graph/community ≥ {self.cfg.RELATED_GRAPH_FLOOR:.2f}/{self.cfg.RELATED_COMMUNITY_FLOOR:.2f}  "
              f"(top={max_score:.3f})\n")

        query_concepts = set(ctx.query_keywords) | set(ctx.inferred_keywords)
        results = []
        for lec, detail in candidates:
            score      = detail["score"]
            dm_kw      = detail["dm_keyword"]
            graph_sc   = detail.get("graph_score", 0.0)
            community_sc = detail.get("community_score", 0.0)
            core_match = _query_concept_in_role(lec, query_concepts, "core")
            introduced_match = _query_concept_in_role(lec, query_concepts, "introduced")
            strong_graph = graph_sc >= self.cfg.DIRECT_GRAPH_FLOOR
            related_graph = graph_sc >= self.cfg.RELATED_GRAPH_FLOOR
            related_community = community_sc >= self.cfg.RELATED_COMMUNITY_FLOOR
            direct_threshold_for_candidate = (
                min(direct_threshold, self.cfg.CORE_MATCH_DIRECT_FLOOR)
                if core_match
                else direct_threshold
            )

            if score >= direct_threshold_for_candidate and (core_match or strong_graph):
                tier = "direct"
                if core_match:
                    detail["tier_reason"] = "core_match"
                else:
                    detail["tier_reason"] = "strong_graph"
            elif score >= related_threshold and (introduced_match or related_graph or related_community):
                tier = "related"
                if introduced_match:
                    detail["tier_reason"] = "introduced_match"
                elif related_graph:
                    detail["tier_reason"] = "related_graph"
                else:
                    detail["tier_reason"] = "related_community"
            else:
                if score < related_threshold:
                    break
                continue

            detail["duration_sec"] = lec.duration_sec
            results.append(RecommendResult(
                video_id     = lec.video_id,
                title        = lec.title,
                domain       = lec.domain,
                instructor   = lec.instructor_id,
                score        = detail["score"],
                display_score= _display_score(detail["score"], tier, detail),
                duration_sec = lec.duration_sec,
                score_detail = detail,
                reason       = _build_reason(detail, tier),
                summary      = lec.summary,
                tier         = tier,
                keywords     = lec.keywords or [],
            ))
            if len(results) >= top_k:
                break

        return results

    def recommend_from_query(
        self,
        query:     str,
        top_k:     int             = 5,
        min_score: Optional[float] = None,
    ) -> list[RecommendResult]:
        """
        자연어 질의를 분석하고 관련 강의를 추천한다.

        v12 흐름 (BM25/vector 후보 검색 + RRF 통합):
          1. analyze_query — search_text / domain / focus_concept 추출
          2. 비교 의도 감지 (_detect_comparison_intent)
          3. search_text → Gemini embedding-001 벡터화
          4. BM25 후보 + vector 후보 → RRF 통합
          5. RRF candidate_ids에 대해:
             - vec_score / dm_score 계산
             - depth_score: BFS 홉 거리 기반 (concept_relations 활용)
             - contrast_bonus: 비교 의도 × 대조형 relation 비율
          6. 패널티 체계:
             - dm_keyword==0 패널티 (×0.6)
             - 원본 query_keyword 완전 미매칭 패널티 (×Q_KW_MISMATCH_PENALTY)
             - 도메인 상위 카테고리 불일치 패널티 (×DOMAIN_MISMATCH_PENALTY)
          7. 이중 레이어 임계값 + gap 자연 경계 → direct/related 분류
        """
        ctx = self._prepare_query_context(query)

        if ctx.intent == "list_by_domain":
            return self._list_by_domain_results(ctx, max(top_k, self.cfg.LIST_QUERY_TOP_K))

        restored_cfg = None
        effective_top_k = top_k
        if ctx.intent == "list_by_topic":
            effective_top_k = max(top_k, self.cfg.LIST_QUERY_TOP_K)
            restored_cfg = {
                "ABS_MIN_SCORE": self.cfg.ABS_MIN_SCORE,
                "Q_KW_MISMATCH_PENALTY": self.cfg.Q_KW_MISMATCH_PENALTY,
                "RELATED_RATIO": self.cfg.RELATED_RATIO,
                "HYBRID_CANDIDATE_TOP_N": self.cfg.HYBRID_CANDIDATE_TOP_N,
            }
            self.cfg.ABS_MIN_SCORE = 0.05
            self.cfg.Q_KW_MISMATCH_PENALTY = 1.0
            self.cfg.RELATED_RATIO = 0.35
            self.cfg.HYBRID_CANDIDATE_TOP_N = max(
                self.cfg.HYBRID_CANDIDATE_TOP_N,
                self.cfg.LIST_QUERY_TOP_K,
            )

        if min_score is not None:
            self.cfg.ABS_MIN_SCORE = min_score

        try:
            # ── 질의 벡터화 ───────────────────────────────────────────────
            query_vec = (
                _embed(ctx.search_text)
                if self._vector_search_index.video_ids
                else []
            )
            candidate_ids = self._get_initial_candidate_ids(ctx, query_vec)
            candidates = self._rank_candidates(candidate_ids, ctx, query_vec)

            # ── 전체 후보 점수 출력 (디버그) ─────────────────────────────
            self._print_candidate_scores(candidates)
            return self._classify_tiers(candidates, ctx, effective_top_k)
        finally:
            if restored_cfg:
                for key, value in restored_cfg.items():
                    setattr(self.cfg, key, value)

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
                tier_label = {
                    "direct": "✅ 직접 추천",
                    "related": "🔸 간접 관련",
                }.get(r.tier, r.tier)
                print(f"  {i}. [{r.video_id}] {r.title}  {tier_label}")
                print(f"     점수:   {score_100}점  |  {r.reason}")
                print(f"     벡터:   title {t_pct}점  keyword {k_pct}점  summary {s_pct}점")
                print(f"     직접:   title {dm_t}점  keyword {dm_k}점  summary {dm_s}점")
                print(f"     boost:  domain +{boost_pct}%  depth +{depth_pct}%  duration {duration_pct}%")
                if d.get("contrast_bonus", 0) > 0:
                    print(f"     대조형: contrast_signal {d['contrast_signal']:.2f}  bonus +{d['contrast_bonus']:.3f}")
                if frag_pct > 0:
                    print(f"     파편화 패널티: -{frag_pct}%")
                print(f"     요약:   {r.summary[:80]}...")
                print()

        print(f"  {'─'*62}")
        print(f"  ▶ 추천된 강의 점수 전체 (이중 레이어 필터 적용됨)")
        print(f"  {'─'*62}")
        print(f"  {'video_id':<12} {'제목':<22} {'도메인':<14} {'티어':<8} {'점수':>6}")
        print(f"  {'─'*62}")
        for r in results:
            tier_label = {
                "direct": "직접",
                "related": "간접",
            }.get(r.tier, r.tier)
            marker = " ◀" if r in top else ""
            print(f"  {r.video_id:<12} {r.title[:20]:<22} {r.domain:<14} "
                  f"{tier_label:<8} {round(r.score*100,1):>5.1f}점{marker}")
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
                        help="ABS_MIN_SCORE 오버라이드 0~1 (기본: cfg.ABS_MIN_SCORE=0.30).")
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
