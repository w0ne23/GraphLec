import uuid
import shutil
import logging
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
import asyncio
import json

from app.db import AsyncSessionLocal, get_db
from app.models import Job, LectureContent
from app.services import job_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/jobs")


@router.get("/{job_id}/stream")
async def stream_job_status(job_id: str, request: Request):
    async def event_generator():
        while True:
            if await request.is_disconnected():
                break
            try:
                async with AsyncSessionLocal() as db:
                    job = await job_service.get_job(db, job_id)
                if not job:
                    yield f"data: {json.dumps({'error': 'Job not found'})}\n\n"
                    break
                payload = {
                    "job_id":          str(job.id),
                    "lecture_status":  job.status,
                    "current_stage":   job.current_stage,
                    "error_message":   job.error_message,
                    "pipeline_stages": job.pipeline_stages or [],
                }
                yield f"data: {json.dumps(payload)}\n\n"
                if job.status in ("done", "error"):
                    break
            except Exception as e:
                logger.error(f"SSE error for {job_id}: {e}")
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
                break
            await asyncio.sleep(1)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.get("")
async def list_jobs(db: AsyncSession = Depends(get_db)):
    return await job_service.list_jobs(db)


@router.get("/{job_id}/graph_status")
async def check_graph_status(job_id: str, db: AsyncSession = Depends(get_db)):
    return await job_service.get_graph_info(db, job_id)


@router.get("/{job_id}")
async def get_job_detail(job_id: str, db: AsyncSession = Depends(get_db)):
    job_detail = await job_service.get_job_detail(db, job_id)
    if not job_detail:
        raise HTTPException(status_code=404, detail="Job not found")
    job_detail["video_url"] = job_service.make_file_url(job_detail.get("input_path"))
    return job_detail


@router.post("")
async def create_job(
    video: UploadFile = File(...),
    title: str = Form(...),
    category: str = Form("컴퓨터 과학"),
    description: str = Form(""),
    db: AsyncSession = Depends(get_db),
):
    job_id   = str(uuid.uuid4())
    base_dir = Path(job_service.LOCAL_STORAGE_DIR)

    input_dir  = base_dir / "inputs"  / job_id
    output_dir = base_dir / "results" / job_id

    # 원본 파일명 보존 및 안전한 파일명(UUID) 생성
    original_stem = Path(video.filename).stem
    extension = Path(video.filename).suffix
    safe_filename = f"{job_id}{extension}"
    
    # title이 비어있으면 원본 파일명으로 대체
    final_title = title.strip() if title and title.strip() else original_stem

    try:
        input_dir.mkdir(parents=True, exist_ok=True)
        input_path = input_dir / safe_filename
        with open(input_path, "wb") as f:
            shutil.copyfileobj(video.file, f)
    except Exception as e:
        shutil.rmtree(input_dir, ignore_errors=True)
        logger.error(f"File save failed for job {job_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to save uploaded file")

    try:
        new_job = Job(
            id=job_id,
            input_path=str(input_path),
            status="pending",
        )
        db.add(new_job)

        # 시스템 전반에서 쓰이는 stem은 고유한 job_id(UUID)를 사용하여 충돌을 방지함
        new_content = LectureContent(
            id=uuid.uuid4(),
            job_id=job_id,
            stem=job_id,
            title=final_title,
            category=category,
            description=description,
            video_path=str(input_path),
            output_dir=str(output_dir),
            graphrag_workspace=str(output_dir / "graphrag"),
        )
        db.add(new_content)
        await db.commit()
    except Exception as e:
        await db.rollback()
        shutil.rmtree(input_dir, ignore_errors=True)
        logger.error(f"DB commit failed for job {job_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to create job")

    return {"job_id": job_id}


@router.delete("/{job_id}")
async def delete_job(job_id: str, db: AsyncSession = Depends(get_db)):
    success = await job_service.delete_job_and_content(db, job_id)
    if not success:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"status": "success"}


@router.post("/{job_id}/retry")
async def retry_job(job_id: str, db: AsyncSession = Depends(get_db)):
    success = await job_service.retry_job(db, job_id)
    if not success:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"status": "success"}


@router.post("/{job_id}/retry_graph")
async def retry_graph_ingestion(job_id: str, db: AsyncSession = Depends(get_db)):
    success = await job_service.retry_graph_only(db, job_id)
    if not success:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"status": "success"}
