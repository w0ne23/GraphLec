"""
recommender/config.py
─────────────────────
추천 시스템 전역 설정 및 RecommenderConfig.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from google import genai

load_dotenv()


# ── 경로 ─────────────────────────────────────────────────────────────────────

def _resolve_repo_root() -> Path:
    """Resolve project root for both local and container runs."""
    env_root = os.getenv("GRAPHLEC_ROOT") or os.getenv("PIPELINE_ROOT")
    if env_root:
        return Path(env_root).resolve()
    here = Path(__file__).resolve()
    return here.parents[3] if len(here.parents) > 3 else here.parents[1]


_REPO_ROOT           = _resolve_repo_root()
DEFAULT_METADATA_DIR = str(_REPO_ROOT / "app" / "backend" / "metadata")
DEFAULT_DB_DIR       = str(_REPO_ROOT / "data" / "lancedb")


# ── API 클라이언트 ────────────────────────────────────────────────────────────

_client            = genai.Client(api_key=os.getenv("GOOGLE_API_KEY_2"))
GEMINI_MODEL       = "gemini-2.5-flash"
EMBED_MODEL        = "gemini-embedding-001"
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


# ── RecommenderConfig ─────────────────────────────────────────────────────────

@dataclass
class RecommenderConfig:
    # 필드별 가중치 (합계 = 1.0)
    W_TITLE:             float = 0.20
    W_KEYWORD:           float = 0.55
    W_SUMMARY:           float = 0.25
    # 벡터 유사도 vs 직접 매칭 블렌딩 비율 (벡터:직접 = VEC_BLEND : 1-VEC_BLEND)
    VEC_BLEND:           float = 0.70
    # keyword vec 유사도 threshold — floor~threshold 구간은 soft decay
    KW_VEC_THRESHOLD:    float = 0.60
    KW_VEC_SOFT_FLOOR:   float = 0.45
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
    DOMAIN_MISMATCH_PENALTY: float = 0.60
    Q_KW_MISMATCH_PENALTY:   float = 0.60
    SUBJECT_PARTIAL_MATCH_PENALTY: float = 0.85
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
