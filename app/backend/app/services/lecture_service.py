import asyncio
import os
import shutil
import logging
import json
import hashlib
import uuid
from datetime import datetime, timedelta, timezone
import pandas as pd
import httpx
from pathlib import Path
from typing import Optional, List, Any, Dict

from sqlalchemy import select, delete, and_, or_
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import HTTPException
from neo4j import GraphDatabase

from app.models import Lecture, ProcessingJob, GraphSession
from pipeline.graphrag_neo4j_ingest import (
    delete_graphrag_layer_tx,
    find_graphrag_output_dir,
    load_graphrag_layer_tx,
)
from pipeline.graphrag_emphasis import (
    compute_keyword_match,
    compute_visual_match,
    compute_annotation_match,
    compute_audio_segment_match,
    compute_final_weight,
    compute_relation_boost,
)
from pipeline.neo4j_ingest import ingest_parquet_to_neo4j

logger = logging.getLogger(__name__)
_RUNTIME_GRAPH_LABELS = {
    "AnnotationEmphasis",
    "ConceptGraph",
    "Context",
    "Domain",
    "GraphRAGCommunity",
    "GraphRAGEntity",
    "GraphRAGTextUnit",
    "Scene",
    "Segment",
    "Slide",
    "Video",
}

# ── 경로 설정 ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path("/pipeline") if Path("/pipeline").exists() else Path(__file__).resolve().parents[4]
LOCAL_STORAGE_DIR = os.getenv("LOCAL_STORAGE_DIR", str(PROJECT_ROOT / "local_storage"))
GRAPH_SESSION_TTL_SEC = int(os.getenv("GRAPH_SESSION_TTL_SEC", "180"))


# ── Neo4j ────────────────────────────────────────────────────────────────────
def get_neo4j_driver():
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USER")
    pw = os.getenv("NEO4J_PASSWORD")
    if not all([uri, user, pw]):
        return None
    return GraphDatabase.driver(uri, auth=(user, pw))


def _graphrag_layer_counts(session, stem: str) -> Dict[str, int]:
    record = session.run(
        """
        CALL {
            MATCH (e:GraphRAGEntity {stem: $stem})
            RETURN count(e) AS entities
        }
        CALL {
            MATCH (:GraphRAGEntity {stem: $stem})-[r:GRAPHRAG_RELATES_TO]->(:GraphRAGEntity {stem: $stem})
            RETURN count(r) AS relationships
        }
        CALL {
            MATCH (:GraphRAGEntity {stem: $stem})-[s:GRAPHRAG_APPEARS_IN]->(:Slide {stem: $stem})
            RETURN count(s) AS slide_links
        }
        CALL {
            MATCH (:GraphRAGEntity {stem: $stem})-[s:GRAPHRAG_APPEARS_IN_SCENE]->(:Scene {stem: $stem})
            RETURN count(s) AS scene_links
        }
        CALL {
            MATCH (tu:GraphRAGTextUnit {stem: $stem})
            RETURN count(tu) AS text_units
        }
        CALL {
            MATCH (c:GraphRAGCommunity {stem: $stem})
            RETURN count(c) AS communities
        }
        RETURN entities, relationships, slide_links, scene_links, text_units, communities
        """,
        stem=stem,
    ).single()
    if not record:
        return {
            "entities": 0,
            "relationships": 0,
            "slide_links": 0,
            "scene_links": 0,
            "text_units": 0,
            "communities": 0,
        }
    return {
        "entities": int(record["entities"]),
        "relationships": int(record["relationships"]),
        "slide_links": int(record["slide_links"]),
        "scene_links": int(record["scene_links"]),
        "text_units": int(record["text_units"]),
        "communities": int(record["communities"]),
    }


def _stem_graph_counts(session, stem: str) -> Dict[str, int]:
    record = session.run(
        """
        CALL {
            MATCH (n {stem: $stem})
            RETURN count(n) AS nodes
        }
        CALL {
            MATCH (a {stem: $stem})-[r]->(b {stem: $stem})
            RETURN count(r) AS relationships
        }
        CALL {
            MATCH (e:GraphRAGEntity {stem: $stem})
            RETURN count(e) AS concepts
        }
        RETURN nodes, relationships, concepts
        """,
        stem=stem,
    ).single()
    if not record:
        return {"nodes": 0, "relationships": 0, "concepts": 0}
    return {
        "nodes": int(record["nodes"]),
        "relationships": int(record["relationships"]),
        "concepts": int(record["concepts"]),
    }


def _delete_stem_graph_tx(tx, stem: str) -> None:
    tx.run("MATCH (n {stem: $stem}) DETACH DELETE n", stem=stem)


def clear_runtime_lecture_graphs() -> Dict[str, int]:
    """앱 시작 시 Neo4j에 남아 있는 강의 런타임 그래프를 비운다."""
    driver = get_neo4j_driver()
    if not driver:
        logger.info("Neo4j connection is not configured; skip runtime graph cleanup")
        return {"before": 0, "after": 0}

    try:
        with driver.session() as session:
            before_record = session.run(
                """
                MATCH (n)
                WHERE n.stem IS NOT NULL OR any(label IN labels(n) WHERE label IN $labels)
                RETURN count(n) AS count
                """,
                labels=sorted(_RUNTIME_GRAPH_LABELS),
            ).single()
            before = int(before_record["count"]) if before_record else 0
            session.run(
                """
                MATCH (n)
                WHERE n.stem IS NOT NULL OR any(label IN labels(n) WHERE label IN $labels)
                DETACH DELETE n
                """,
                labels=sorted(_RUNTIME_GRAPH_LABELS),
            )
            after_record = session.run(
                """
                MATCH (n)
                WHERE n.stem IS NOT NULL OR any(label IN labels(n) WHERE label IN $labels)
                RETURN count(n) AS count
                """,
                labels=sorted(_RUNTIME_GRAPH_LABELS),
            ).single()
            after = int(after_record["count"]) if after_record else 0
            logger.info("Cleared Neo4j runtime lecture graphs: before=%s after=%s", before, after)
            return {"before": before, "after": after}
    except Exception as e:
        logger.warning("Failed to clear Neo4j runtime lecture graphs: %s", e)
        return {"before": 0, "after": 0}
    
    
def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _session_expired(cutoff: datetime):
    return and_(
        GraphSession.ended_at.is_(None),
        GraphSession.last_heartbeat_at < cutoff,
    )


async def _cleanup_stale_sessions(db: AsyncSession, lecture_id) -> int:
    cutoff = _utcnow() - timedelta(seconds=GRAPH_SESSION_TTL_SEC)
    q = select(GraphSession).where(
        GraphSession.stem == str(lecture_id),
        _session_expired(cutoff),
    )
    res = await db.execute(q)
    rows = res.scalars().all()
    for row in rows:
        row.ended_at = _utcnow()
    if rows:
        await db.commit()
    return len(rows)


async def _active_session_count(db: AsyncSession, lecture_id) -> int:
    await _cleanup_stale_sessions(db, lecture_id)
    q = select(GraphSession).where(
        GraphSession.stem == str(lecture_id),
        GraphSession.ended_at.is_(None),
    )
    res = await db.execute(q)
    return len(res.scalars().all())


async def _touch_or_create_graph_session(
    db: AsyncSession,
    *,
    lecture_id,
    session_id: str,
    now: datetime,
) -> None:
    """
    (lecture_id, session_id) 세션을 upsert처럼 갱신한다.
    - 기존 중복 행이 있으면 최신 1개만 활성 유지하고 나머지는 ended 처리
    """
    q = (
        select(GraphSession)
        .where(
            GraphSession.lecture_id == lecture_id,
            GraphSession.session_id == session_id,
        )
        .order_by(GraphSession.created_at.desc(), GraphSession.id.desc())
    )
    res = await db.execute(q)
    rows = res.scalars().all()

    if not rows:
        db.add(
            GraphSession(
                lecture_id=lecture_id,
                stem=str(lecture_id),
                session_id=session_id,
                last_heartbeat_at=now,
                ended_at=None,
            )
        )
        return

    primary = rows[0]
    primary.last_heartbeat_at = now
    primary.ended_at = None
    for extra in rows[1:]:
        extra.ended_at = now


async def _commit_graph_session_touch(
    db: AsyncSession,
    *,
    lecture_id,
    session_id: str,
    now: datetime,
) -> None:
    try:
        await _touch_or_create_graph_session(
            db,
            lecture_id=lecture_id,
            session_id=session_id,
            now=now,
        )
        await db.commit()
    except IntegrityError:
        await db.rollback()
        await _touch_or_create_graph_session(
            db,
            lecture_id=lecture_id,
            session_id=session_id,
            now=now,
        )
        await db.commit()


def _is_stem_loaded(stem: str) -> bool:
    driver = get_neo4j_driver()
    if not driver:
        raise HTTPException(status_code=503, detail="Neo4j 설정이 없습니다.")
    try:
        with driver.session() as session:
            record = session.run(
                "MATCH (n {stem: $stem}) RETURN count(n) AS count",
                stem=stem,
            ).single()
            return bool(record and int(record["count"] or 0) > 0)
    finally:
        driver.close()


def _unload_stem_from_neo4j(stem: str) -> None:
    driver = get_neo4j_driver()
    if not driver:
        raise HTTPException(status_code=503, detail="Neo4j 설정이 없습니다.")
    try:
        with driver.session() as session:
            session.run("MATCH (n {stem: $stem}) DETACH DELETE n", stem=stem)
    finally:
        driver.close()


def _load_graphrag_layer_for_stem(
    stem: str,
    output_dir: Path,
    graphrag_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    graphrag_dir = graphrag_dir or find_graphrag_output_dir(stem, output_dir)
    if not graphrag_dir:
        return {"graphrag_output_dir": None, "graphrag": {}}

    driver = get_neo4j_driver()
    if not driver:
        raise HTTPException(status_code=503, detail="Neo4j 설정이 없습니다.")

    try:
        with driver.session() as session:
            graphrag_counts = session.execute_write(
                lambda tx: (
                    delete_graphrag_layer_tx(tx, stem),
                    load_graphrag_layer_tx(tx, stem, graphrag_dir),
                )[1]
            )
            fused_path = output_dir / f"{stem}_fused.json"
            if fused_path.is_file():
                graphrag_counts.update(compute_keyword_match(session, stem, fused_path))
                graphrag_counts.update(compute_visual_match(session, stem, fused_path))
            graphrag_counts.update(compute_annotation_match(session, stem))
            graphrag_counts.update(compute_audio_segment_match(session, stem))
            graphrag_counts.update(compute_final_weight(session, stem))
            graphrag_counts.update(compute_relation_boost(session, stem))
            graph_counts = _stem_graph_counts(session, stem)
    finally:
        driver.close()

    return {
        "graphrag_output_dir": str(graphrag_dir),
        "graphrag": graphrag_counts,
        "node_count": graph_counts["nodes"],
        "edge_count": graph_counts["relationships"],
        "concept_count": graph_counts["concepts"],
    }


def _ensure_stem_loaded(stem: str, output_dir: str) -> Dict[str, Any]:
    output_path = Path(output_dir)
    if _is_stem_loaded(stem):
        driver = get_neo4j_driver()
        if not driver:
            raise HTTPException(status_code=503, detail="Neo4j 설정이 없습니다.")
        try:
            with driver.session() as session:
                graphrag_counts = _graphrag_layer_counts(session, stem)
        finally:
            driver.close()
        if graphrag_counts.get("entities", 0) == 0:
            loaded = _load_graphrag_layer_for_stem(stem, output_path)
            return {"loaded_now": False, "graphrag_loaded_now": bool(loaded.get("graphrag")), **loaded}
        return {"loaded_now": False, "graphrag": graphrag_counts}
    try:
        ingest_result = ingest_parquet_to_neo4j(stem=stem, output_dir=output_path)
        graphrag_loaded = _load_graphrag_layer_for_stem(stem, output_path)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Neo4j 적재 실패: {e}")
    return {
        "loaded_now": True,
        "node_count": ingest_result.get("node_count", 0),
        "edge_count": ingest_result.get("edge_count", 0),
        **graphrag_loaded,
    }


# ── 직렬화 헬퍼 ──────────────────────────────────────────────────────────────
def format_job_dict(job: ProcessingJob, lecture: Optional[Lecture]) -> Dict[str, Any]:
    """ProcessingJob과 Lecture를 하나의 딕셔너리로 직렬화"""
    stages = job.pipeline_stages
    if isinstance(stages, str):
        try:
            stages = json.loads(stages)
        except Exception:
            stages = []
    elif stages is None:
        stages = []

    res = {
        "job_id": str(job.id),
        "status": job.status,
        "current_stage": job.current_stage,
        "error_message": job.error_message,
        "pipeline_stages": stages,
        "created_at": job.created_at.isoformat() if job.created_at else None,
        "content": [],
    }
    if lecture:
        res["video_path"] = lecture.video_path
        res["content"].append({
            "id": str(lecture.id),
            "title": lecture.title,
            "category": lecture.category,
            "description": lecture.description,
            "stem": str(lecture.id),
            "output_dir": lecture.output_dir,
        })
    return res


def make_file_url(abs_path: Optional[str]) -> Optional[str]:
    if not abs_path:
        return None
    try:
        p = Path(abs_path).as_posix()
        if "local_storage/" in p:
            return f"/files/{p.split('local_storage/')[1]}"
    except Exception:
        pass
    return None


# ── 그래프 유틸 ──────────────────────────────────────────────────────────────
def _hex_color(key: str) -> str:
    h = hashlib.md5(key.encode("utf-8")).hexdigest()
    return f"#{h[:6]}"


def _str_cell(x: Any) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x)


# ── ProcessingJob CRUD ───────────────────────────────────────────────────────
async def get_job(db: AsyncSession, job_id: str) -> Optional[ProcessingJob]:
    try:
        # UUID 형식 검증
        uuid.UUID(str(job_id))
    except (ValueError, TypeError):
        return None
        
    result = await db.execute(
        select(ProcessingJob).where(ProcessingJob.id == job_id)
    )
    return result.scalar_one_or_none()


async def get_latest_job(db: AsyncSession, lecture_id: str) -> Optional[ProcessingJob]:
    """lecture_id로 가장 최근 job을 반환"""
    try:
        ident_uuid = uuid.UUID(str(lecture_id))
    except (ValueError, TypeError):
        return None
    result = await db.execute(
        select(ProcessingJob)
        .where(ProcessingJob.lecture_id == ident_uuid)
        .order_by(ProcessingJob.created_at.desc())
        .limit(1)
    )
    return result.scalar_one_or_none()


async def get_job_detail(db: AsyncSession, job_id: str) -> Optional[Dict[str, Any]]:
    query = (
        select(ProcessingJob, Lecture)
        .join(Lecture, ProcessingJob.lecture_id == Lecture.id)
        .where(ProcessingJob.id == job_id)
    )
    result = await db.execute(query)
    row = result.unique().one_or_none()
    if not row:
        return None
    return format_job_dict(row[0], row[1])


ACTIVE_STATUSES = {'pending', 'running'}

async def list_jobs(db: AsyncSession, status_filter: Optional[str] = None):
    query = (
        select(Lecture, ProcessingJob)
        .outerjoin(ProcessingJob, ProcessingJob.lecture_id == Lecture.id)
        .order_by(Lecture.created_at.desc(), ProcessingJob.created_at.desc())
    )
    result = await db.execute(query)
    rows = result.unique().all()

    seen = set()
    out = []
    for lecture, job in rows:
        if lecture.id in seen:
            continue
        seen.add(lecture.id)
        job_status = job.status if job else 'unknown'
        if status_filter == 'active' and job_status not in ACTIVE_STATUSES:
            continue
        is_done = job_status == "done"
        out.append({
            "id": str(lecture.id),
            "status": job_status,
            "current_stage": job.current_stage if job and not is_done else None,
            "error_message": job.error_message if job else None,
            "pipeline_stages": job.pipeline_stages or [] if job and not is_done else [],
            "title": lecture.title or str(lecture.id),
            "category": lecture.category or "기타",
            "created_at": lecture.created_at.isoformat() if lecture.created_at else None,
        })
    return out


async def retry_lecture(db: AsyncSession, lecture_id: str):
    """lecture_id로 새 ProcessingJob을 INSERT하여 재시도 이력을 누적.
    성공 시 { status, job_id } dict 반환, 실패 시 None.
    """
    try:
        ident_uuid = uuid.UUID(str(lecture_id))
    except (ValueError, TypeError):
        return None

    lecture = await _get_lecture(db, ident_uuid)
    if not lecture:
        return None

    new_job = ProcessingJob(
        id=uuid.uuid4(),
        lecture_id=ident_uuid,
        status="pending",
        current_stage="Resuming pipeline...",
        error_message=None,
        pipeline_stages=[],
    )
    db.add(new_job)
    await db.commit()
    await db.refresh(new_job)
    return {"status": "success", "job_id": str(new_job.id)}


async def delete_lecture(db: AsyncSession, lecture_id: str) -> bool:
    """lecture_id로 강의 전체 삭제 — DB, 로컬 파일, Neo4j 모두 정리"""
    try:
        ident_uuid = uuid.UUID(str(lecture_id))
    except (ValueError, TypeError):
        return False

    lecture_result = await db.execute(
        select(Lecture).where(Lecture.id == ident_uuid)
    )
    lecture = lecture_result.scalar_one_or_none()
    if not lecture:
        return False

    stem = str(lecture.id) # stem = lecture_id

    # 파일 정리 — inputs/{lecture_id}/ 와 results/{lecture_id}/
    if lecture.video_path:
        shutil.rmtree(Path(lecture.video_path).parent, ignore_errors=True)
    if lecture.output_dir:
        shutil.rmtree(Path(lecture.output_dir), ignore_errors=True)

    # Neo4j 정리
    driver = get_neo4j_driver()
    if driver:
        try:
            with driver.session() as session:
                session.run("MATCH (n {stem: $stem}) DETACH DELETE n", stem=stem)
        finally:
            driver.close()

    # Lecture CASCADE로 ProcessingJob, GraphSession 함께 삭제
    await db.execute(delete(Lecture).where(Lecture.id == lecture.id))
    await db.commit()
    return True


# ── 결과 조회 (Lecture ID 기준) ───────────────────────────────────────────────
async def list_all_results(
    db: AsyncSession,
    page: int = 1,
    limit: int = 12,
    category: Optional[str] = None,
    search: Optional[str] = None,
    scope: str = 'browse',
) -> Dict[str, Any]:
    query = (
        select(Lecture, ProcessingJob)
        .outerjoin(ProcessingJob, ProcessingJob.lecture_id == Lecture.id)
        .order_by(Lecture.created_at.desc(), ProcessingJob.created_at.desc())
    )
    result = await db.execute(query)
    rows = result.unique().all()

    seen = set()
    out = []
    for lecture, job in rows:
        if lecture.id in seen:
            continue
        seen.add(lecture.id)
        job_status = job.status if job else 'unknown'

        if scope == 'browse' and job_status != 'done':
            continue
        if scope == 'upload' and job_status in ACTIVE_STATUSES:
            continue

        if category and lecture.category != category:
            continue
        if search and search.lower() not in (lecture.title or '').lower():
            continue

        out.append({
            "id": str(lecture.id),
            "job_id": str(job.id) if job else None,
            "status": job_status,
            "title": lecture.title or str(lecture.id),
            "category": lecture.category or "기타",
            "created_at": lecture.created_at.isoformat() if lecture.created_at else None,
            "error_message": job.error_message if job else None,
            "pipeline_stages": job.pipeline_stages or [] if job else [],
        })

    total_items = len(out)
    start = (page - 1) * limit
    paginated = out[start: start + limit]
    return {"items": paginated, "total_items": total_items}


async def get_lecture_detail(db: AsyncSession, lecture_id: str) -> Optional[Dict[str, Any]]:
    """강의 상세 정보 조회.

    프론트가 업로드 직후 job_id를 들고 있는 경우가 있어 Lecture.id,
    Job.id, Lecture.job_id, stem을 모두 허용한다.
    """
    row = await _get_lecture_row(db, lecture_id)
    if not row:
        return None
    lecture, job = row
    stem = str(lecture.id)
    return {
        "id": str(lecture.id),
        "job_id": str(job.id) if job else None,
        "status": job.status if job else "unknown",
        "title": lecture.title or stem,
        "category": lecture.category or "기타",
        "description": lecture.description,
        "stem": stem,
        "video_url": make_file_url(lecture.video_path),
        "output_dir": lecture.output_dir,
        "graphrag_workspace": str(Path(lecture.output_dir) / "graphrag") if lecture.output_dir else None,
        "created_at": lecture.created_at.isoformat() if lecture.created_at else None,
    }



async def _get_lecture_row(db: AsyncSession, lecture_id: str):
    """lecture_id로 Lecture + 최신 ProcessingJob을 함께 반환"""
    try:
        ident_uuid = uuid.UUID(str(lecture_id))
    except (ValueError, TypeError):
        return None
    query = (
        select(Lecture, ProcessingJob)
        .outerjoin(ProcessingJob, ProcessingJob.lecture_id == Lecture.id)
        .where(Lecture.id == ident_uuid)
        .order_by(ProcessingJob.created_at.desc())
    )
    result = await db.execute(query)
    return result.unique().first()


async def _get_lecture(db: AsyncSession, lecture_id: str) -> Optional[Lecture]:
    """lecture_id로 Lecture 객체를 반환"""
    try:
        ident_uuid = uuid.UUID(str(lecture_id))
    except (ValueError, TypeError):
        return None
    result = await db.execute(select(Lecture).where(Lecture.id == ident_uuid))
    return result.scalar_one_or_none()


# ── GraphSession 관련 ────────────────────────────────────────────────────────
async def graph_enter(db: AsyncSession, lecture_id: str, session_id: str) -> Dict[str, Any]:
    lecture = await _get_lecture(db, lecture_id)
    if not lecture:
        raise HTTPException(status_code=404, detail="Lecture not found")

    stem = str(lecture.id)
    output_dir = str(lecture.output_dir)
    now = _utcnow()

    await _cleanup_stale_sessions(db, lecture.id)
    await _commit_graph_session_touch(
        db,
        lecture_id=lecture.id,
        session_id=session_id,
        now=now,
    )

    loop = asyncio.get_event_loop()
    load_info = await loop.run_in_executor(None, _ensure_stem_loaded, stem, output_dir)
    active_count = await _active_session_count(db, lecture.id)
    return {
        "lecture_id": str(lecture.id),
        "stem": stem,
        "session_id": session_id,
        "active_sessions": active_count,
        "loaded": True,
        **load_info,
    }


async def graph_heartbeat(db: AsyncSession, lecture_id: str, session_id: str) -> Dict[str, Any]:
    lecture = await _get_lecture(db, lecture_id)
    if not lecture:
        raise HTTPException(status_code=404, detail="Lecture not found")

    now = _utcnow()
    await _commit_graph_session_touch(
        db,
        lecture_id=lecture.id,
        session_id=session_id,
        now=now,
    )
    return {
        "lecture_id": str(lecture.id),
        "stem": str(lecture.id),
        "session_id": session_id,
        "active_sessions": await _active_session_count(db, lecture.id),
    }


async def graph_leave(db: AsyncSession, lecture_id: str, session_id: str) -> Dict[str, Any]:
    lecture = await _get_lecture(db, lecture_id)
    if not lecture:
        raise HTTPException(status_code=404, detail="Lecture not found")

    stem = str(lecture.id)
    q = select(GraphSession).where(
        GraphSession.lecture_id == lecture.id,
        GraphSession.session_id == session_id,
        GraphSession.ended_at.is_(None),
    )
    res = await db.execute(q)
    rows = res.scalars().all()
    if rows:
        now = _utcnow()
        for row in rows:
            row.ended_at = now
        await db.commit()

    active_count = await _active_session_count(db, lecture.id)
    loop = asyncio.get_event_loop()
    unloaded_now = False
    if active_count == 0 and await loop.run_in_executor(None, _is_stem_loaded, stem):
        await loop.run_in_executor(None, _unload_stem_from_neo4j, stem)
        unloaded_now = True

    loaded = await loop.run_in_executor(None, _is_stem_loaded, stem)
    return {
        "lecture_id": str(lecture.id),
        "stem": stem,
        "session_id": session_id,
        "active_sessions": active_count,
        "unloaded_now": unloaded_now,
        "loaded": loaded,
    }


async def graph_status(db: AsyncSession, lecture_id: str) -> Dict[str, Any]:
    lecture = await _get_lecture(db, lecture_id)
    if not lecture:
        raise HTTPException(status_code=404, detail="Lecture not found")
    stem = str(lecture.id)
    loop = asyncio.get_event_loop()
    loaded = await loop.run_in_executor(None, _is_stem_loaded, stem)
    active_count = await _active_session_count(db, lecture.id)
    return {
        "lecture_id": str(lecture.id),
        "stem": stem,
        "loaded": loaded,
        "active_sessions": active_count,
        "session_ttl_sec": GRAPH_SESSION_TTL_SEC,
    }


async def ask_question(db: AsyncSession, lecture_id: str, question: str) -> Dict[str, Any]:
    lecture = await _get_lecture(db, lecture_id)
    if not lecture:
        raise HTTPException(status_code=404, detail="Lecture not found")

    stem = str(lecture.id)
    query_url = os.getenv("QUERY_SERVICE_URL", "http://query_service:8001")
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _ensure_stem_loaded, stem, lecture.output_dir)

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{query_url}/internal/query",
                json={"stem": stem, "question": question},
            )
            if resp.status_code != 200:
                raise HTTPException(status_code=resp.status_code, detail="Query service error")
            qr = resp.json()
            return {
                "answer": qr.get("answer"),
                "timestamps": qr.get("timestamps", []),
                "graph": qr.get("graph", {"nodes": [], "edges": []}),
                "retrieved_chunks": qr.get("retrieved_chunks", []),
            }
    except httpx.HTTPError:
        raise HTTPException(status_code=503, detail="Query service unreachable")


async def get_timeline(db: AsyncSession, lecture_id: str) -> List[Dict[str, Any]]:
    """타임라인 조회 (Lecture ID 기준)"""
    detail = await get_lecture_detail(db, lecture_id)
    if not detail or not detail.get("output_dir"):
        raise HTTPException(status_code=404, detail="Lecture result not found")

    output_dir = Path(detail["output_dir"])
    json_files = list(output_dir.glob("*_slide_classified.json"))
    if not json_files:
        raise HTTPException(status_code=404, detail="Timeline file not found")

    try:
        with open(json_files[0], "r", encoding="utf-8") as f:
            data = json.load(f)

        scenes = []
        for s in data.get("scenes", []):
            img_url = make_file_url(s.get("image_path"))
            ts = s.get("timestamp_formatted", "00:00").split(".")[0]
            if ts.startswith("00:"): ts = ts[3:]

            scenes.append({
                "timestamp":    ts,
                "type":         "emphasis" if s.get("role") == "elaborated" else "slide",
                "text":         s.get("title") or f"Slide {s.get('slide_number')}",
                "image_url":    img_url,
                "scene_number": s.get("scene_number", s.get("scene_index")),
                "slide_number": s.get("slide_number"),
            })
        return scenes
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading timeline: {e}")


async def get_knowledge_graph(db: AsyncSession, lecture_id: str) -> Dict[str, Any]:
    """지식 그래프 조회 (Lecture ID 기준)"""
    detail = await get_lecture_detail(db, lecture_id)
    if not detail or not detail.get("output_dir"):
        raise HTTPException(status_code=404, detail="Lecture result not found")

    output_dir = Path(detail["output_dir"])
    nodes_paths = list(output_dir.glob("*_nodes.parquet"))
    edges_paths = list(output_dir.glob("*_edges.parquet"))

    if not nodes_paths:
        raise HTTPException(status_code=404, detail="Graph files not found")

    try:
        ndf = pd.read_parquet(nodes_paths[0])
        edf = (
            pd.read_parquet(edges_paths[0])
            if edges_paths
            else pd.DataFrame(columns=["src_id", "rel_type", "tgt_id", "properties_json"])
        )

        nodes_out = []
        for _, row in ndf.iterrows():
            nid = _str_cell(row.get("node_id"))
            if not nid:
                continue
            label = _str_cell(row.get("label")) or nid
            props = _str_cell(row.get("properties_json")) or "{}"
            try:
                ntype = json.loads(props).get("type", "node")
            except:
                ntype = "node"
            nodes_out.append({
                "id": nid, "label": label[:120], "title": props[:800],
                "type": ntype, "color": _hex_color(label),
            })

        seen_ids = {n["id"] for n in nodes_out}
        edges_out = []
        for _, row in edf.iterrows():
            src, tgt = _str_cell(row.get("src_id")), _str_cell(row.get("tgt_id"))
            if not src or not tgt:
                continue
            edges_out.append({"from": src, "to": tgt, "label": _str_cell(row.get("rel_type")) or "related"})
            for x in (src, tgt):
                if x not in seen_ids:
                    seen_ids.add(x)
                    nodes_out.append({"id": x, "label": x[:80], "type": "orphan", "color": "#9ca3af"})

        return {
            "node_count": len(nodes_out),
            "edge_count": len(edges_out),
            "graph": {"nodes": nodes_out, "edges": edges_out},
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading graph: {e}")


async def ingest_graphrag_concept_graph(db: AsyncSession, lecture_id: str) -> Dict[str, Any]:
    """강의 시청 화면 진입 시 구조 그래프와 GraphRAG 그래프를 Neo4j에 적재한다."""
    detail = await get_lecture_detail(db, lecture_id)
    if not detail or not detail.get("output_dir") or not detail.get("stem"):
        raise HTTPException(status_code=404, detail="Lecture result not found")

    stem = str(detail["stem"])
    output_dir = Path(detail["output_dir"])
    structural_counts = ingest_parquet_to_neo4j(stem=stem, output_dir=output_dir)
    aliases = [a for a in [_str_cell(detail.get("graphrag_workspace")), _str_cell(detail.get("title"))] if a]
    graphrag_dir = find_graphrag_output_dir(
        stem,
        output_dir,
        aliases=aliases,
    )

    graphrag_counts: Dict[str, int] = {}
    if graphrag_dir:
        driver = get_neo4j_driver()
        if not driver:
            raise HTTPException(status_code=503, detail="Neo4j connection is not configured")

        try:
            with driver.session() as session:
                graphrag_counts = session.execute_write(
                    lambda tx: (
                        delete_graphrag_layer_tx(tx, stem),
                        load_graphrag_layer_tx(tx, stem, graphrag_dir),
                    )[1]
                )
                fused_path = output_dir / f"{stem}_fused.json"
                if fused_path.is_file():
                    graphrag_counts.update(compute_keyword_match(session, stem, fused_path))
                    graphrag_counts.update(compute_visual_match(session, stem, fused_path))
                graphrag_counts.update(compute_annotation_match(session, stem))
                graphrag_counts.update(compute_audio_segment_match(session, stem))
                graphrag_counts.update(compute_final_weight(session, stem))
                graphrag_counts.update(compute_relation_boost(session, stem))
                summary = _stem_graph_counts(session, stem)
        finally:
            driver.close()
    else:
        driver = get_neo4j_driver()
        if not driver:
            raise HTTPException(status_code=503, detail="Neo4j connection is not configured")
        try:
            with driver.session() as session:
                summary = _stem_graph_counts(session, stem)
        finally:
            driver.close()

    return {
        "status": "loaded",
        "lecture_id": lecture_id,
        "stem": stem,
        "graphrag_output_dir": str(graphrag_dir) if graphrag_dir else None,
        "structural": structural_counts,
        "graphrag": graphrag_counts,
        "node_count": summary["nodes"],
        "edge_count": summary["relationships"],
        "concept_count": summary["concepts"],
    }


async def get_graphrag_ingest_status(db: AsyncSession, lecture_id: str) -> Dict[str, Any]:
    """Neo4j에 적재된 GraphRAG 개념 그래프 상태를 확인한다."""
    detail = await get_lecture_detail(db, lecture_id)
    if not detail or not detail.get("stem"):
        raise HTTPException(status_code=404, detail="Lecture result not found")

    stem = str(detail["stem"])
    driver = get_neo4j_driver()
    if not driver:
        raise HTTPException(status_code=503, detail="Neo4j connection is not configured")

    try:
        with driver.session() as session:
            counts = _graphrag_layer_counts(session, stem)
            custom_record = session.run(
                "MATCH (c:GraphRAGEntity {stem: $stem}) RETURN count(c) AS custom_concepts",
                stem=stem,
            ).single()
    finally:
        driver.close()

    graphrag_dir = find_graphrag_output_dir(
        stem,
        Path(detail["output_dir"]) if detail.get("output_dir") else None,
        aliases=[_str_cell(detail.get("title"))],
    )
    return {
        "stem": stem,
        "graphrag_output_dir": str(graphrag_dir) if graphrag_dir else None,
        **counts,
        "custom_concepts": int(custom_record["custom_concepts"]) if custom_record else 0,
    }


async def ensure_graphrag_concept_graph_loaded(db: AsyncSession, lecture_id: str) -> Dict[str, Any]:
    """강의 화면 진입 시 구조 그래프와 GraphRAG 그래프가 Neo4j에 올라와 있도록 보장한다."""
    detail = await get_lecture_detail(db, lecture_id)
    if not detail or not detail.get("output_dir") or not detail.get("stem"):
        raise HTTPException(status_code=404, detail="Lecture result not found")

    stem = str(detail["stem"])
    output_dir = Path(detail["output_dir"])
    aliases = [a for a in [_str_cell(detail.get("graphrag_workspace")), _str_cell(detail.get("title"))] if a]
    graphrag_dir = find_graphrag_output_dir(
        stem,
        output_dir,
        aliases=aliases,
    )

    driver = get_neo4j_driver()
    if not driver:
        raise HTTPException(status_code=503, detail="Neo4j connection is not configured")

    try:
        with driver.session() as session:
            graph_counts = _stem_graph_counts(session, stem)
            graphrag_counts = _graphrag_layer_counts(session, stem)
            if graph_counts["nodes"] > 0:
                if graphrag_dir and graphrag_counts.get("entities", 0) == 0:
                    loaded = _load_graphrag_layer_for_stem(stem, output_dir, graphrag_dir)
                    return {
                        "status": "graphrag_layer_loaded",
                        "lecture_id": lecture_id,
                        "stem": stem,
                        **loaded,
                    }
                return {
                    "status": "already_loaded",
                    "lecture_id": lecture_id,
                    "stem": stem,
                    "graphrag_output_dir": str(graphrag_dir) if graphrag_dir else None,
                    "node_count": graph_counts["nodes"],
                    "edge_count": graph_counts["relationships"],
                    "concept_count": graph_counts["concepts"],
                    "graphrag": graphrag_counts,
                }
    finally:
        driver.close()

    return await ingest_graphrag_concept_graph(db, lecture_id)


async def unload_graphrag_concept_graph(db: AsyncSession, lecture_id: str) -> Dict[str, Any]:
    """활성 시청자가 없을 때만 해당 stem의 구조 그래프와 GraphRAG 그래프를 Neo4j에서 제거한다."""
    detail = await get_lecture_detail(db, lecture_id)
    if not detail or not detail.get("stem"):
        raise HTTPException(status_code=404, detail="Lecture result not found")

    stem = str(detail["stem"])
    active_count = await _active_session_count(db, stem)
    driver = get_neo4j_driver()
    if not driver:
        raise HTTPException(status_code=503, detail="Neo4j connection is not configured")

    try:
        with driver.session() as session:
            before = _stem_graph_counts(session, stem)
            if active_count > 0:
                return {
                    "status": "still_active",
                    "lecture_id": lecture_id,
                    "stem": stem,
                    "active_sessions": active_count,
                    "before": before,
                    "after": before,
                }
            session.execute_write(lambda tx: _delete_stem_graph_tx(tx, stem))
            after = _stem_graph_counts(session, stem)
    finally:
        driver.close()

    return {
        "status": "unloaded",
        "lecture_id": lecture_id,
        "stem": stem,
        "active_sessions": active_count,
        "before": before,
        "after": after,
    }


def _filter_served_slide_typos(items: list[dict]) -> list[dict]:
    try:
        from pipeline.analyzer.slide_typo_checker import is_reportable_slide_typo
    except Exception:
        return items

    filtered = []
    for item in items or []:
        if not isinstance(item, dict):
            continue
        if is_reportable_slide_typo(
            str(item.get("problematic_text", "") or ""),
            str(item.get("corrected_text", "") or ""),
            str(item.get("reason", "") or ""),
        ):
            filtered.append(item)
    return filtered


async def get_content_verification(db: AsyncSession, lecture_id: str) -> Dict[str, Any]:
    """verifier 결과 조회 (Lecture ID 기준)."""
    detail = await get_lecture_detail(db, lecture_id)
    if not detail or not detail.get("output_dir") or not detail.get("stem"):
        raise HTTPException(status_code=404, detail="Lecture result not found")

    output_dir = Path(detail["output_dir"])
    stem = str(detail["stem"])

    candidate_paths = [
        output_dir / f"{stem}_analyzer" / f"{stem}_content_verification.json",
        output_dir / f"{stem}_content_verification.json",
    ]
    verifier_path = next((path for path in candidate_paths if path.exists()), None)
    if not verifier_path:
        raise HTTPException(status_code=404, detail="Content verification file not found")

    try:
        with open(verifier_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading content verification: {e}")

    flow = data.get("claim_decision_flow", {}) or {}
    summary = data.get("claim_decision_flow_summary", {}) or {}
    final_claims = flow.get("final_confirmed_claims", []) or []
    needs_review_claims = flow.get("needs_review_claims", []) or []
    crosscheck_rejected_claims = flow.get("crosscheck_rejected_claims", []) or []
    crosscheck_inconclusive_claims = flow.get("crosscheck_inconclusive_claims", []) or []
    slide_rejected_claims = flow.get("slide_rejected_claims", []) or []
    grounding_rejected_claims = flow.get("grounding_rejected_claims", []) or []
    first_stage_rejected_claims = flow.get("first_stage_rejected_claims", []) or []
    slide_typos = _filter_served_slide_typos(data.get("slide_typos", []) or [])
    slide_typo_needs_review = _filter_served_slide_typos(data.get("slide_typo_needs_review", []) or [])

    return {
        "lecture_id": str(detail["id"]),
        "stem": stem,
        "verification_path": str(verifier_path),
        "mode": data.get("mode", ""),
        "verification_date": data.get("verification_date", ""),
        "models": data.get("models", []) or [],
        "primary_model": data.get("primary_model", ""),
        "summary": summary,
        "overview": data.get("claim_decision_overview", []) or [],
        "counts": {
            "final_confirmed": int(summary.get("final_confirmed_claim_count", len(final_claims)) or 0),
            "needs_review": int(summary.get("needs_review_claim_count", len(needs_review_claims)) or 0),
            "slide_typos": len(slide_typos),
            "slide_typo_needs_review": len(slide_typo_needs_review),
            "crosscheck_rejected": int(summary.get("crosscheck_rejected_claim_count", len(crosscheck_rejected_claims)) or 0),
            "crosscheck_inconclusive": int(summary.get("crosscheck_inconclusive_claim_count", len(crosscheck_inconclusive_claims)) or 0),
            "slide_rejected": int(summary.get("slide_rejected_claim_count", len(slide_rejected_claims)) or 0),
            "grounding_rejected": int(summary.get("grounding_rejected_claim_count", len(grounding_rejected_claims)) or 0),
            "first_stage_rejected": int(summary.get("first_stage_rejected_claim_count", len(first_stage_rejected_claims)) or 0),
        },
        "final_confirmed_claim_count": int(
            summary.get("final_confirmed_claim_count", len(final_claims))
        ),
        "final_confirmed_claims": final_claims,
        "needs_review_claims": needs_review_claims,
        "crosscheck_rejected_claims": crosscheck_rejected_claims,
        "crosscheck_inconclusive_claims": crosscheck_inconclusive_claims,
        "slide_rejected_claims": slide_rejected_claims,
        "grounding_rejected_claims": grounding_rejected_claims,
        "first_stage_rejected_claims": first_stage_rejected_claims,
        "unmatched_issue_records": flow.get("unmatched_issue_records", []) or [],
        "issues": data.get("issues", []) or [],
        "needs_review_issues": data.get("needs_review_issues", []) or [],
        "slide_typos": slide_typos,
        "slide_typo_needs_review": slide_typo_needs_review,
        "slide_typo_consensus": data.get("slide_typo_consensus", {}) or {},
        "slide_typo_status": data.get("slide_typo_status", ""),
        "rejected_issues": data.get("rejected_issues", []) or [],
        "crosscheck_rejected_issues": data.get("crosscheck_rejected_issues", []) or [],
        "crosscheck_inconclusive_issues": data.get("crosscheck_inconclusive_issues", []) or [],
        "slide_rejected_issues": data.get("slide_rejected_issues", []) or [],
        "grounding_rejected_issues": data.get("grounding_rejected_issues", []) or [],
        "claim_decision_flow_summary": summary,
    }


# ── 기타 유틸 ───────────────────────────────────────────────────────────────
async def get_graph_info(db: AsyncSession, lecture_id: str) -> Dict[str, Any]:
    lecture = await _get_lecture(db, lecture_id)
    if not lecture:
        return {"error": "Not Found"}
    stem = str(lecture.id)
    node_count = 0
    driver = get_neo4j_driver()
    if driver:
        try:
            with driver.session() as session:
                record = session.run(
                    "MATCH (n {stem: $stem}) RETURN count(n) AS count", stem=stem
                ).single()
                if record:
                    node_count = record["count"]
        finally:
            driver.close()
    return {"lecture_id": lecture_id, "stem": stem, "graph_exists": node_count > 0, "node_count": node_count}


async def retry_graph_only(db: AsyncSession, lecture_id: str) -> bool:
    """lecture_id로 최신 job을 그래프 재적재 상태로 초기화"""
    job = await get_latest_job(db, lecture_id)
    if not job:
        return False
    job.status = "pending"
    job.current_stage = "Retrying Graph Ingestion..."
    job.error_message = None
    await db.commit()
    return True