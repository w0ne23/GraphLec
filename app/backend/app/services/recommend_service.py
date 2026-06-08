import os
import uuid
from typing import Any, Dict

import httpx
from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Lecture, LectureMetadata
from app.services.lecture_service import normalize_domain_value, _lecture_thumbnail_url

RECOMMENDER_SERVICE_URL = os.getenv("RECOMMENDER_SERVICE_URL", "http://recommender:8002")
RECOMMEND_TIMEOUT_SEC = float(os.getenv("RECOMMEND_TIMEOUT_SEC", "10"))


def _valid_recommendation_items(raw_results: list[Any]) -> list[tuple[uuid.UUID, dict[str, Any]]]:
    seen: set[uuid.UUID] = set()
    items: list[tuple[uuid.UUID, dict[str, Any]]] = []
    for item in raw_results:
        if not isinstance(item, dict):
            continue
        try:
            lecture_id = uuid.UUID(str(item.get("video_id") or ""))
        except (TypeError, ValueError):
            continue
        if lecture_id in seen:
            continue
        seen.add(lecture_id)
        items.append((lecture_id, item))
    return items


def _metadata_domain(
    lecture: Lecture,
    lecture_metadata: LectureMetadata | None,
) -> str:
    metadata_domain = None
    if lecture_metadata:
        metadata_domain = lecture_metadata.graph_domain or lecture_metadata.domain
    return normalize_domain_value(metadata_domain or lecture.category)


async def _load_lectures(
    db: AsyncSession,
    lecture_ids: list[uuid.UUID],
) -> dict[uuid.UUID, tuple[Lecture, LectureMetadata | None]]:
    result = await db.execute(
        select(Lecture, LectureMetadata)
        .outerjoin(LectureMetadata, LectureMetadata.lecture_id == Lecture.id)
        .where(Lecture.id.in_(lecture_ids))
    )
    return {
        lecture.id: (lecture, lecture_metadata)
        for lecture, lecture_metadata in result.all()
    }


async def recommend(
    db: AsyncSession,
    query: str,
    top_k: int = 3,
) -> Dict[str, Any]:
    overfetch_top_k = min(top_k * 3, 30)
    try:
        async with httpx.AsyncClient(timeout=RECOMMEND_TIMEOUT_SEC) as client:
            resp = await client.post(
                f"{RECOMMENDER_SERVICE_URL}/recommend",
                json={"query": query, "top_k": overfetch_top_k},
            )
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=503,
            detail="Recommender service unavailable",
        ) from exc

    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail="Recommender service error")

    payload = resp.json()
    raw_results = payload.get("results", [])
    if not isinstance(raw_results, list):
        raw_results = []

    candidate_items = _valid_recommendation_items(raw_results)
    if not candidate_items:
        return {"query": query, "results": []}

    lecture_rows = await _load_lectures(
        db,
        [lecture_id for lecture_id, _item in candidate_items],
    )

    enriched: list[dict[str, Any]] = []
    passthrough_fields = (
        "score",
        "display_score",
        "tier",
        "reason",
        "summary",
        "score_detail",
        "keywords",
        "instructor",
        "duration_sec",
    )
    for lecture_id, item in candidate_items:
        row = lecture_rows.get(lecture_id)
        if not row:
            continue

        lecture, lecture_metadata = row
        if not bool(getattr(lecture, "is_published", False)):
            continue

        result_item = {field: item.get(field) for field in passthrough_fields}
        result_item.update({
            "video_id": str(lecture.id),
            "title": lecture.title or str(lecture.id),
            "domain": _metadata_domain(lecture, lecture_metadata),
            "thumbnail_url": _lecture_thumbnail_url(lecture.output_dir),
        })
        enriched.append(result_item)
        if len(enriched) >= top_k:
            break

    return {"query": query, "results": enriched}
