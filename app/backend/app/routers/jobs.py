import os
import uuid
import shutil
import logging
import json
import asyncio
import httpx
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import AsyncSessionLocal, get_db
from app.models import Job
from app.services import job_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/jobs")


@router.get("/{job_id}/stream")
async def stream_job_status(job_id: str, request: Request):
    async def event_generator():
        while True:
            if await request.is_disconnected(): break
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
                if job.status in ("done", "error"): break
            except Exception as e:
                logger.error(f"SSE error for {job_id}: {e}")
                break
            await asyncio.sleep(1)
    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.get("")
async def list_jobs(db: AsyncSession = Depends(get_db)):
    jobs = await job_service.list_jobs(db)
    return jobs


@router.post("")
async def create_job(
    video: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    job_id = str(uuid.uuid4())
    input_dir = Path(job_service.LOCAL_STORAGE_DIR) / "inputs" / job_id
    input_dir.mkdir(parents=True, exist_ok=True)
    input_path = input_dir / video.filename

    with open(input_path, "wb") as f:
        shutil.copyfileobj(video.file, f)

    new_job = Job(id=job_id, input_path=str(input_path), status="pending")
    db.add(new_job)
    await db.commit()
    return {"job_id": job_id}


@router.delete("/{job_id}")
async def delete_job(job_id: str, db: AsyncSession = Depends(get_db)):
    success = await job_service.delete_job_and_content(db, job_id)
    if not success:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"status": "success"}


@router.post("/{job_id}/qa")
async def ask_question(job_id: str, request: Request, db: AsyncSession = Depends(get_db)):
    body = await request.json()
    question = body.get("question")
    job = await job_service.get_job(db, job_id)
    if not job or not job.input_path:
        raise HTTPException(status_code=404, detail="Job not found")

    stem = Path(job.input_path).stem
    query_url = os.getenv("QUERY_SERVICE_URL", "http://query_service:8001")

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(f"{query_url}/internal/query", json={"stem": stem, "question": question})
            return resp.json()
    except Exception as e:
        raise HTTPException(status_code=503, detail="Query service unreachable")
