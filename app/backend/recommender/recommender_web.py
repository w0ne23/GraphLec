"""
recommender_web.py
──────────────────
FastAPI 기반 강의 추천 웹 서버

실행:
  uvicorn recommender.recommender_web:app --port 8002 --reload

엔드포인트:
  POST /recommend   { "query": "스레드 자세히 설명하는 강의", "top_k": 3 }
  GET  /health
"""

import os
import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from recommender.recommender import Recommender, RecommenderConfig


# ============================================================================
#  앱 초기화
# ============================================================================

_BACKEND_ROOT = Path(__file__).resolve().parents[1]


def _resolve_repo_root() -> Path:
    env_root = os.getenv("GRAPHLEC_ROOT") or os.getenv("PIPELINE_ROOT")
    if env_root:
        return Path(env_root).resolve()
    here = Path(__file__).resolve()
    return here.parents[3] if len(here.parents) > 3 else here.parents[1]


_REPO_ROOT = _resolve_repo_root()
METADATA_DIR = os.getenv("METADATA_DIR", str(_BACKEND_ROOT / "metadata"))
RECOMMENDER_DB_DIR = os.getenv("RECOMMENDER_DB_DIR", str(_REPO_ROOT / "data" / "lancedb"))
_recommender: Optional[Recommender] = None
_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


def _thumbnail_url(video_id: str) -> Optional[str]:
    if not _UUID_RE.match(str(video_id)):
        return None
    result_dir = Path("/pipeline/local_storage/results") / str(video_id)
    if not result_dir.exists():
        return f"/api/files/results/{video_id}/slides/scene_001_base.jpg"
    candidates = (
        sorted(result_dir.glob("slides/scene_001_base.*"))
        or sorted(result_dir.glob("slides/scene_*_base.*"))
    )
    if not candidates:
        return None
    return f"/api/files/results/{video_id}/slides/{candidates[0].name}"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _recommender
    print(f"[시작] 메타데이터 로드: {METADATA_DIR}")
    _recommender = Recommender(
        metadata_dir=METADATA_DIR,
        config=RecommenderConfig(DB_DIR=RECOMMENDER_DB_DIR),
    )
    yield
    print("[종료]")


app = FastAPI(
    title="GraphLEC Recommender",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
#  요청 / 응답 스키마
# ============================================================================

class RecommendRequest(BaseModel):
    query: str
    top_k: int = 3


class ScoreDetail(BaseModel):
    content_pct:       float
    vec_score:         float
    dm_score:          float
    sim_title:         float
    sim_keyword:       float
    sim_summary:       float
    dm_keyword:        float
    graph_score:       float
    community_score:   float
    visual_score:      float
    visual_density_score: float
    visual_concept_score: float
    visual_preference: bool
    application_score: float = 0.0
    application_preference: bool = False
    listenability_score: float = 0.0
    listenability_preference: bool = False
    speech_rate_score: float = 0.0
    slow_speech_preference: bool = False
    recency_score: float = 0.0
    recency_preference: bool = False
    recency_weight: float = 0.0
    domain_score:      float
    depth_score:       float
    tier_reason:       str = ""
    combined_boost:    float
    duration_score:    float
    duration_mismatch: bool
    condition_warnings: list[str] = []
    frag_penalty:      float


class LectureResult(BaseModel):
    video_id:          str
    title:             str
    domain:            str
    instructor:        str
    score:             float
    display_score:     Optional[int] = None
    duration_sec:      float
    reason:            str
    summary:           str
    tier:              str
    thumbnail_url:     Optional[str] = None
    keywords:          list[dict] = []
    score_detail:      Optional[ScoreDetail] = None


class RecommendResponse(BaseModel):
    query:   str
    results: list[LectureResult]


# ============================================================================
#  엔드포인트
# ============================================================================

@app.get("/")
def root():
    return {
        "service": "GraphLEC Recommender",
        "endpoints": ["/health", "/recommend"],
    }


@app.get("/health")
def health():
    loaded = _recommender is not None
    count  = len(_recommender.collection.lectures) if loaded else 0
    return {"status": "ok", "lectures_loaded": count}


@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    if not _recommender:
        raise HTTPException(status_code=503, detail="추천 엔진 초기화 중")
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="query가 비어 있습니다")
    if not _recommender.collection.lectures:
        return RecommendResponse(query=req.query, results=[])

    results = _recommender.recommend_from_query(req.query, top_k=req.top_k)

    return RecommendResponse(
        query=req.query,
        results=[
            LectureResult(
                video_id     = r.video_id,
                title        = r.title,
                domain       = r.domain,
                instructor   = r.instructor,
                score        = r.score,
                display_score= r.display_score,
                duration_sec = r.duration_sec,
                reason       = r.reason,
                summary      = r.summary,
                tier         = r.tier,
                thumbnail_url= _thumbnail_url(r.video_id),
                keywords     = r.keywords or [],
                score_detail = None if not r.score_detail else ScoreDetail(
                    content_pct       = r.score_detail.get("content_pct",       0.0),
                    vec_score         = r.score_detail.get("vec_score",          0.0),
                    dm_score          = r.score_detail.get("dm_score",           0.0),
                    sim_title         = r.score_detail.get("sim_title",          0.0),
                    sim_keyword       = r.score_detail.get("sim_keyword",        0.0),
                    sim_summary       = r.score_detail.get("sim_summary",        0.0),
                    dm_keyword        = r.score_detail.get("dm_keyword",         0.0),
                    graph_score       = r.score_detail.get("graph_score",        0.0),
                    community_score   = r.score_detail.get("community_score",    0.0),
                    visual_score      = r.score_detail.get("visual_score",       0.0),
                    visual_density_score = r.score_detail.get("visual_density_score", 0.0),
                    visual_concept_score = r.score_detail.get("visual_concept_score", 0.0),
                    visual_preference = r.score_detail.get("visual_preference",  False),
                    application_score = r.score_detail.get("application_score", 0.0),
                    application_preference = r.score_detail.get("application_preference", False),
                    listenability_score = r.score_detail.get("listenability_score", 0.0),
                    listenability_preference = r.score_detail.get("listenability_preference", False),
                    speech_rate_score = r.score_detail.get("speech_rate_score", 0.0),
                    slow_speech_preference = r.score_detail.get("slow_speech_preference", False),
                    recency_score = r.score_detail.get("recency_score", 0.0),
                    recency_preference = r.score_detail.get("recency_preference", False),
                    recency_weight = r.score_detail.get("recency_weight", 0.0),
                    domain_score      = r.score_detail.get("domain_score",       0.0),
                    depth_score       = r.score_detail.get("depth_score",        0.0),
                    tier_reason      = r.score_detail.get("tier_reason", ""),
                    combined_boost    = r.score_detail.get("combined_boost",     1.0),
                    duration_score    = r.score_detail.get("duration_score",     1.0),
                    duration_mismatch = r.score_detail.get("duration_mismatch",  False),
                    condition_warnings = r.score_detail.get("condition_warnings", []),
                    frag_penalty      = r.score_detail.get("frag_penalty",       0.0),
                ),
            )
            for r in results
        ],
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("recommender.recommender_web:app", host="0.0.0.0", port=8002, reload=True)
