import logging
from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db
from app.services import job_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/results")


@router.get("")
async def list_results(db: AsyncSession = Depends(get_db)):
    """강의 목록 조회 (Lecture ID 기준)"""
    return await job_service.list_all_results(db)


@router.get("/{lecture_id}/timeline")
async def get_timeline(lecture_id: str, db: AsyncSession = Depends(get_db)):
    """강의 타임라인 데이터 조회"""
    return await job_service.get_timeline(db, lecture_id)


@router.get("/{lecture_id}/graph")
async def get_knowledge_graph(lecture_id: str, db: AsyncSession = Depends(get_db)):
    """강의 지식 그래프 데이터 조회"""
    return await job_service.get_knowledge_graph(db, lecture_id)


@router.post("/{lecture_id}/graph/graphrag/ingest")
async def ingest_graphrag_concept_graph(lecture_id: str, db: AsyncSession = Depends(get_db)):
    """GraphRAG 개념 그래프를 기존 Neo4j 강의 그래프에 적재"""
    return await job_service.ingest_graphrag_concept_graph(db, lecture_id)


@router.get("/{lecture_id}/graph/graphrag/status")
async def get_graphrag_ingest_status(lecture_id: str, db: AsyncSession = Depends(get_db)):
    """GraphRAG 개념 그래프 Neo4j 적재 상태 조회"""
    return await job_service.get_graphrag_ingest_status(db, lecture_id)


@router.get("/{lecture_id}/verifier")
async def get_content_verification(lecture_id: str, db: AsyncSession = Depends(get_db)):
    """강의 verifier 결과 조회"""
    return await job_service.get_content_verification(db, lecture_id)


@router.post("/{lecture_id}/query")
async def ask_question(lecture_id: str, request: Request, db: AsyncSession = Depends(get_db)):
    """강의 내용 질의응답 (Lecture ID 기준)"""
    body = await request.json()
    question = body.get("question")
    if not question:
        raise HTTPException(status_code=400, detail="Question is required")
    return await job_service.ask_question(db, lecture_id, question)


@router.get("/{lecture_id}")
async def get_result_detail(lecture_id: str, db: AsyncSession = Depends(get_db)):
    """특정 강의 상세 정보 조회"""
    detail = await job_service.get_lecture_detail(db, lecture_id)
    if not detail:
        raise HTTPException(status_code=404, detail="Lecture not found")
    return detail
