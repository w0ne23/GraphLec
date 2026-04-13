from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.db import get_db
from app.models import LectureContent
from app.services.job_service import make_file_url

router = APIRouter(prefix="/results")

@router.get("")
async def list_lectures(db: AsyncSession = Depends(get_db)):
    """분석이 완료된 실제 강의 목록 반환"""
    result = await db.execute(select(LectureContent).order_by(LectureContent.created_at.desc()))
    lectures = result.scalars().all()
    
    return [
        {
            "id":          str(l.id),
            "job_id":      str(l.job_id) if l.job_id else None,
            "stem":        l.stem,
            "title":       l.title or l.stem,
            "category":    l.category or "일반",
            "description": l.description,
            "video_url":   make_file_url(l.video_path),
            "created_at":  l.created_at.isoformat(),
        }
        for l in lectures
    ]

@router.get("/{id}")
async def get_lecture(id: str, db: AsyncSession = Depends(get_db)):
    """특정 강의 상세 정보 반환"""
    result = await db.execute(select(LectureContent).where(LectureContent.id == id))
    lecture = result.scalar_one_or_none()
    
    if not lecture:
        raise HTTPException(status_code=404, detail="Lecture not found")
        
    return {
        "id":          str(lecture.id),
        "stem":        lecture.stem,
        "title":       lecture.title or lecture.stem,
        "category":    lecture.category or "일반",
        "description": lecture.description,
        "video_url":   make_file_url(lecture.video_path),
    }
