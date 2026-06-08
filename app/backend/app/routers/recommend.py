from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db
from app.services import recommend_service

router = APIRouter(prefix="/recommend")


class RecommendRequest(BaseModel):
    query: str
    top_k: int = Field(3, ge=1, le=50)


@router.post("")
async def recommend_lectures(
    request: RecommendRequest,
    db: AsyncSession = Depends(get_db),
):
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="query is required")
    return await recommend_service.recommend(db, query=query, top_k=request.top_k)
