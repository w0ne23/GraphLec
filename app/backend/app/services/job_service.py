import os
import shutil
import logging
from pathlib import Path
from typing import Optional, List

from sqlalchemy import select, delete, text
from sqlalchemy.ext.asyncio import AsyncSession
from neo4j import GraphDatabase

from app.models import Job, LectureContent

logger = logging.getLogger(__name__)

# ── 경로 설정 ──────────────────────────────────────────────────
# Docker 환경을 우선으로 고려한 경로 설정
PROJECT_ROOT     = Path("/app") if Path("/app").exists() else Path(__file__).resolve().parents[3]
LOCAL_STORAGE_DIR = os.getenv("LOCAL_STORAGE_DIR", str(PROJECT_ROOT / "local_storage"))


# ── Neo4j 드라이버 헬퍼 ──────────────────────────────────────────

def get_neo4j_driver():
    uri  = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USER")
    pw   = os.getenv("NEO4J_PASSWORD")
    if not all([uri, user, pw]):
        return None
    return GraphDatabase.driver(uri, auth=(user, pw))


# ── 조회 ───────────────────────────────────────────────────────

async def get_job(db: AsyncSession, job_id: str) -> Optional[Job]:
    result = await db.execute(select(Job).where(Job.id == job_id))
    return result.scalar_one_or_none()

async def list_jobs(db: AsyncSession) -> List[Job]:
    result = await db.execute(select(Job).order_by(Job.created_at.desc()))
    return result.scalars().all()

async def get_lecture_by_stem(db: AsyncSession, stem: str) -> Optional[LectureContent]:
    result = await db.execute(select(LectureContent).where(LectureContent.stem == stem))
    return result.scalar_one_or_none()


# ── 삭제 (강의 콘텐츠 및 작업 데이터 파기) ──────────────────────────

async def delete_job_and_content(db: AsyncSession, job_id: str) -> bool:
    """
    작업 데이터와 분석된 강의 콘텐츠를 완전히 삭제합니다.
    """
    job = await get_job(db, job_id)
    if not job:
        return False

    stem = Path(job.input_path).stem if job.input_path else None

    # 1. 로컬 파일 시스템 정리
    if job.input_path:
        input_dir = Path(job.input_path).parent
        if input_dir.exists() and "inputs" in str(input_dir):
            shutil.rmtree(input_dir, ignore_errors=True)

    if job.output_dir:
        output_dir = Path(job.output_dir)
        if output_dir.exists() and "results" in str(output_dir):
            shutil.rmtree(output_dir, ignore_errors=True)

    # 2. Neo4j 그래프 데이터 정리
    if stem:
        driver = get_neo4j_driver()
        if driver:
            try:
                with driver.session() as session:
                    session.run("MATCH (n {stem: $stem}) DETACH DELETE n", stem=stem)
                logger.info(f"Deleted Neo4j nodes for stem: {stem}")
            except Exception as e:
                logger.warning(f"Neo4j cleanup failed for {job_id}: {e}")
            finally:
                driver.close()

    # 3. DB 레코드 삭제 (Job 삭제 시 LectureContent는 FK 설정에 따라 처리)
    await db.execute(delete(Job).where(Job.id == job_id))
    await db.commit()
    return True


# ── 유틸리티 ───────────────────────────────────────────────────

def make_file_url(abs_path: str) -> Optional[str]:
    if not abs_path: return None
    try:
        p = Path(abs_path).as_posix()
        if "local_storage/" in p:
            return f"/files/{p.split('local_storage/')[1]}"
    except Exception: pass
    return None
