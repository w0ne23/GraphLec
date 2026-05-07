import logging
from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db
from app.services import lecture_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/results")


@router.get("")
async def list_results(db: AsyncSession = Depends(get_db)):
    """강의 목록 조회 (Lecture ID 기준)"""
    return await lecture_service.list_all_results(db)


@router.get("/{lecture_id}/timeline")
async def get_timeline(lecture_id: str, db: AsyncSession = Depends(get_db)):
    """강의 타임라인 데이터 조회"""
    return await lecture_service.get_timeline(db, lecture_id)


@router.get("/{lecture_id}/graph")
async def get_knowledge_graph(lecture_id: str, db: AsyncSession = Depends(get_db)):
    """강의 지식 그래프 데이터 조회"""
    return await lecture_service.get_knowledge_graph(db, lecture_id)


@router.post("/{lecture_id}/graph/graphrag/ingest")
async def ingest_graphrag_concept_graph(lecture_id: str, db: AsyncSession = Depends(get_db)):
    """구조 그래프와 GraphRAG 그래프를 Neo4j에 적재"""
    return await lecture_service.ingest_graphrag_concept_graph(db, lecture_id)


@router.post("/{lecture_id}/graph/activate")
async def activate_lecture_graph(lecture_id: str, db: AsyncSession = Depends(get_db)):
    """강의 화면 진입 시 구조 그래프와 GraphRAG 그래프를 Neo4j에 적재"""
    return await lecture_service.ensure_graphrag_concept_graph_loaded(db, lecture_id)


@router.post("/{lecture_id}/graph/unload")
async def unload_lecture_graph(lecture_id: str, db: AsyncSession = Depends(get_db)):
    """강의 화면 이탈 시 해당 강의 그래프를 Neo4j에서 제거"""
    return await lecture_service.unload_graphrag_concept_graph(db, lecture_id)


@router.post("/{lecture_id}/graph/graphrag/activate")
async def activate_graphrag_concept_graph(lecture_id: str, db: AsyncSession = Depends(get_db)):
    """이전 프론트 호환용: 구조 그래프와 GraphRAG 그래프를 Neo4j에 적재"""
    return await lecture_service.ensure_graphrag_concept_graph_loaded(db, lecture_id)


@router.post("/{lecture_id}/graph/graphrag/unload")
async def unload_graphrag_concept_graph(lecture_id: str, db: AsyncSession = Depends(get_db)):
    """이전 프론트 호환용: 해당 강의 그래프를 Neo4j에서 제거"""
    return await lecture_service.unload_graphrag_concept_graph(db, lecture_id)


@router.get("/{lecture_id}/graph/graphrag/status")
async def get_graphrag_ingest_status(lecture_id: str, db: AsyncSession = Depends(get_db)):
    """GraphRAG 개념 그래프 Neo4j 적재 상태 조회"""
    return await lecture_service.get_graphrag_ingest_status(db, lecture_id)


@router.get("/{lecture_id}/verifier")
async def get_content_verification(lecture_id: str, db: AsyncSession = Depends(get_db)):
    """강의 verifier 결과 조회"""
    return await lecture_service.get_content_verification(db, lecture_id)


@router.post("/{lecture_id}/query")
async def ask_question(lecture_id: str, request: Request, db: AsyncSession = Depends(get_db)):
    """강의 내용 질의응답 (Lecture ID 기준)"""
    body = await request.json()
    question = body.get("question")
    if not question:
        raise HTTPException(status_code=400, detail="Question is required")
    return await lecture_service.ask_question(db, lecture_id, question)


@router.post("/{lecture_id}/graph/enter")
async def graph_enter(lecture_id: str, request: Request, db: AsyncSession = Depends(get_db)):
    body = await request.json() if request.headers.get("content-type", "").startswith("application/json") else {}
    session_id = (body.get("session_id") or "").strip()
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required")
    return await lecture_service.graph_enter(db, lecture_id, session_id)


@router.post("/{lecture_id}/graph/heartbeat")
async def graph_heartbeat(lecture_id: str, request: Request, db: AsyncSession = Depends(get_db)):
    body = await request.json() if request.headers.get("content-type", "").startswith("application/json") else {}
    session_id = (body.get("session_id") or "").strip()
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required")
    return await lecture_service.graph_heartbeat(db, lecture_id, session_id)


@router.post("/{lecture_id}/graph/leave")
async def graph_leave(lecture_id: str, request: Request, db: AsyncSession = Depends(get_db)):
    body = await request.json() if request.headers.get("content-type", "").startswith("application/json") else {}
    session_id = (body.get("session_id") or "").strip()
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required")
    return await lecture_service.graph_leave(db, lecture_id, session_id)


@router.get("/{lecture_id}/graph/status")
async def graph_status(lecture_id: str, db: AsyncSession = Depends(get_db)):
    return await lecture_service.graph_status(db, lecture_id)


@router.get("/{lecture_id}")
async def get_result_detail(lecture_id: str, db: AsyncSession = Depends(get_db)):
    """특정 강의 상세 정보 조회"""
    detail = await lecture_service.get_lecture_detail(db, lecture_id)
    if not detail:
        raise HTTPException(status_code=404, detail="Lecture not found")
    return detail
