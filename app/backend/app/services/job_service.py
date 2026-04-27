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

from sqlalchemy import select, delete, and_
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import HTTPException
from neo4j import GraphDatabase

from app.models import Job, LectureContent, GraphSession
from pipeline.neo4j_ingest import ingest_parquet_to_neo4j

logger = logging.getLogger(__name__)

# ── 경로 설정 ────────────────────────────────────────────────────────────────
PROJECT_ROOT      = Path("/pipeline") if Path("/pipeline").exists() else Path(__file__).resolve().parents[4]
LOCAL_STORAGE_DIR = os.getenv("LOCAL_STORAGE_DIR", str(PROJECT_ROOT / "local_storage"))
GRAPH_SESSION_TTL_SEC = int(os.getenv("GRAPH_SESSION_TTL_SEC", "180"))


# ── Neo4j ────────────────────────────────────────────────────────────────────
def get_neo4j_driver():
    uri  = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USER")
    pw   = os.getenv("NEO4J_PASSWORD")
    if not all([uri, user, pw]):
        return None
    return GraphDatabase.driver(uri, auth=(user, pw))


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _session_expired(cutoff: datetime):
    return and_(
        GraphSession.ended_at.is_(None),
        GraphSession.last_heartbeat_at < cutoff,
    )


async def _cleanup_stale_sessions(db: AsyncSession, stem: str) -> int:
    cutoff = _utcnow() - timedelta(seconds=GRAPH_SESSION_TTL_SEC)
    q = select(GraphSession).where(
        GraphSession.stem == stem,
        _session_expired(cutoff),
    )
    res = await db.execute(q)
    rows = res.scalars().all()
    for row in rows:
        row.ended_at = _utcnow()
    if rows:
        await db.commit()
    return len(rows)


async def _active_session_count(db: AsyncSession, stem: str) -> int:
    await _cleanup_stale_sessions(db, stem)
    q = select(GraphSession).where(
        GraphSession.stem == stem,
        GraphSession.ended_at.is_(None),
    )
    res = await db.execute(q)
    return len(res.scalars().all())


async def _touch_or_create_graph_session(
    db: AsyncSession,
    *,
    lecture_id,
    stem: str,
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
                stem=stem,
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


def _ensure_stem_loaded(stem: str, output_dir: str) -> Dict[str, Any]:
    if _is_stem_loaded(stem):
        return {"loaded_now": False}
    try:
        ingest_result = ingest_parquet_to_neo4j(stem=stem, output_dir=Path(output_dir))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Neo4j 적재 실패: {e}")
    return {
        "loaded_now": True,
        "node_count": ingest_result.get("node_count", 0),
        "edge_count": ingest_result.get("edge_count", 0),
    }


# ── 직렬화 헬퍼 ──────────────────────────────────────────────────────────────
def format_job_dict(job: Job, content: Optional[LectureContent]) -> Dict[str, Any]:
    """Job과 LectureContent를 하나의 딕셔너리로 직렬화"""
    stages = job.pipeline_stages
    if isinstance(stages, str):
        try:
            stages = json.loads(stages)
        except Exception:
            stages = []
    elif stages is None:
        stages = []

    res = {
        "job_id":          str(job.id),
        "status":          job.status,
        "input_path":      job.input_path,
        "current_stage":   job.current_stage,
        "error_message":   job.error_message,
        "pipeline_stages": stages,
        "created_at":      job.created_at.isoformat() if job.created_at else None,
        "content":         [],
    }
    if content:
        res["content"].append({
            "id":          str(content.id),
            "title":       content.title,
            "category":    content.category,
            "description": content.description,
            "stem":        content.stem,
            "output_dir":  content.output_dir,
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


# ── Job CRUD (분석 프로세스 관리) ─────────────────────────────────────────────
async def get_job(db: AsyncSession, job_id: str) -> Optional[Job]:
    result = await db.execute(select(Job).where(Job.id == job_id))
    return result.scalar_one_or_none()


async def get_job_detail(db: AsyncSession, job_id: str) -> Optional[Dict[str, Any]]:
    query = (
        select(Job, LectureContent)
        .outerjoin(LectureContent, Job.id == LectureContent.job_id)
        .where(Job.id == job_id)
    )
    result = await db.execute(query)
    row = result.unique().one_or_none()
    if not row:
        return None
    return format_job_dict(row[0], row[1])


async def list_jobs(db: AsyncSession) -> List[Dict[str, Any]]:
    query = (
        select(Job, LectureContent)
        .outerjoin(LectureContent, Job.id == LectureContent.job_id)
        .order_by(Job.created_at.desc())
    )
    result = await db.execute(query)
    rows = result.unique().all()
    return [format_job_dict(row[0], row[1]) for row in rows]


async def retry_job(db: AsyncSession, job_id: str) -> bool:
    job = await get_job(db, job_id)
    if not job:
        return False
    result = await db.execute(select(LectureContent).where(LectureContent.job_id == job_id))
    content = result.scalar_one_or_none()
    if content and content.output_dir:
        output_dir = Path(content.output_dir)
        if output_dir.exists() and "results" in output_dir.as_posix():
            shutil.rmtree(output_dir, ignore_errors=True)
            logger.info("Cleared output dir for retry: %s", output_dir)
    job.status          = "pending"
    job.error_message   = None
    job.current_stage   = "Retrying..."
    job.pipeline_stages = []
    await db.commit()
    return True


async def delete_job_and_content(db: AsyncSession, job_id: str) -> bool:
    """모든 연관 데이터(DB, 로컬 파일, Neo4j) 정리"""
    job = await get_job(db, job_id)
    if not job:
        return False

    stem = Path(job.input_path).stem if job.input_path else None
    result  = await db.execute(select(LectureContent).where(LectureContent.job_id == job_id))
    content = result.scalar_one_or_none()

    if job.input_path:
        shutil.rmtree(Path(job.input_path).parent, ignore_errors=True)
    if content and content.output_dir:
        shutil.rmtree(Path(content.output_dir), ignore_errors=True)

    if stem:
        driver = get_neo4j_driver()
        if driver:
            try:
                with driver.session() as session:
                    session.run("MATCH (n {stem: $stem}) DETACH DELETE n", stem=stem)
            finally:
                driver.close()

    await db.execute(delete(Job).where(Job.id == job_id))
    await db.commit()
    return True


# ── 결과 조회 (Lecture ID 기준) ───────────────────────────────────────────────
async def list_all_results(db: AsyncSession) -> List[Dict[str, Any]]:
    """완료된 강의 목록 — Lecture ID 및 Job 상태 포함"""
    query = (
        select(Job, LectureContent)
        .join(LectureContent, Job.id == LectureContent.job_id)
        .order_by(Job.created_at.desc())
    )
    result = await db.execute(query)
    rows = result.unique().all()
    return [
        {
            "id":          str(row[1].id),      # Lecture ID
            "job_id":      str(row[0].id),      # Job ID
            "status":      row[0].status,
            "title":       row[1].title or row[1].stem,
            "category":    row[1].category or "기타",
            "description": row[1].description,
            "stem":        row[1].stem,
            "video_url":   make_file_url(row[1].video_path),
            "created_at":  row[0].created_at.isoformat() if row[0].created_at else None,
        }
        for row in rows
    ]


async def get_lecture_detail(db: AsyncSession, lecture_id: str) -> Optional[Dict[str, Any]]:
    """강의 상세 정보 조회 (Lecture ID 기준)"""
    query = (
        select(Job, LectureContent)
        .join(LectureContent, Job.id == LectureContent.job_id)
        .where(LectureContent.id == lecture_id)
    )
    result = await db.execute(query)
    row = result.unique().one_or_none()
    if not row:
        return None
    job, content = row[0], row[1]
    return {
        "id":          str(content.id),
        "job_id":      str(job.id),
        "status":      job.status,
        "title":       content.title or content.stem,
        "category":    content.category or "기타",
        "description": content.description,
        "stem":        content.stem,
        "video_url":   make_file_url(content.video_path),
        "output_dir":  content.output_dir,
        "created_at":  job.created_at.isoformat() if job.created_at else None,
    }


async def _get_lecture_content(db: AsyncSession, lecture_id: str) -> Optional[LectureContent]:
    try:
        lec_uuid = uuid.UUID(str(lecture_id))
    except (ValueError, TypeError):
        return None
    result = await db.execute(select(LectureContent).where(LectureContent.id == lec_uuid))
    return result.scalar_one_or_none()


async def graph_enter(db: AsyncSession, lecture_id: str, session_id: str) -> Dict[str, Any]:
    content = await _get_lecture_content(db, lecture_id)
    if not content:
        raise HTTPException(status_code=404, detail="Lecture not found")

    stem = str(content.stem)
    output_dir = str(content.output_dir)
    now = _utcnow()

    await _cleanup_stale_sessions(db, stem)
    await _touch_or_create_graph_session(
        db,
        lecture_id=content.id,
        stem=stem,
        session_id=session_id,
        now=now,
    )
    await db.commit()

    load_info = _ensure_stem_loaded(stem, output_dir)
    active_count = await _active_session_count(db, stem)
    return {
        "lecture_id": str(content.id),
        "stem": stem,
        "session_id": session_id,
        "active_sessions": active_count,
        "loaded": True,
        **load_info,
    }


async def graph_heartbeat(db: AsyncSession, lecture_id: str, session_id: str) -> Dict[str, Any]:
    content = await _get_lecture_content(db, lecture_id)
    if not content:
        raise HTTPException(status_code=404, detail="Lecture not found")
    stem = str(content.stem)
    now = _utcnow()

    await _touch_or_create_graph_session(
        db,
        lecture_id=content.id,
        stem=stem,
        session_id=session_id,
        now=now,
    )
    await db.commit()

    return {
        "lecture_id": str(content.id),
        "stem": stem,
        "session_id": session_id,
        "active_sessions": await _active_session_count(db, stem),
    }


async def graph_leave(db: AsyncSession, lecture_id: str, session_id: str) -> Dict[str, Any]:
    content = await _get_lecture_content(db, lecture_id)
    if not content:
        raise HTTPException(status_code=404, detail="Lecture not found")
    stem = str(content.stem)

    q = select(GraphSession).where(
        GraphSession.lecture_id == content.id,
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

    active_count = await _active_session_count(db, stem)
    unloaded_now = False
    if active_count == 0 and _is_stem_loaded(stem):
        _unload_stem_from_neo4j(stem)
        unloaded_now = True

    return {
        "lecture_id": str(content.id),
        "stem": stem,
        "session_id": session_id,
        "active_sessions": active_count,
        "unloaded_now": unloaded_now,
        "loaded": _is_stem_loaded(stem),
    }


async def graph_status(db: AsyncSession, lecture_id: str) -> Dict[str, Any]:
    content = await _get_lecture_content(db, lecture_id)
    if not content:
        raise HTTPException(status_code=404, detail="Lecture not found")
    stem = str(content.stem)
    return {
        "lecture_id": str(content.id),
        "stem": stem,
        "loaded": _is_stem_loaded(stem),
        "active_sessions": await _active_session_count(db, stem),
        "session_ttl_sec": GRAPH_SESSION_TTL_SEC,
    }


async def ask_question(db: AsyncSession, lecture_id: str, question: str) -> Dict[str, Any]:
    """질의응답 (Lecture ID 기준)"""
    result = await db.execute(select(LectureContent).where(LectureContent.id == lecture_id))
    content = result.scalar_one_or_none()
    if not content:
        raise HTTPException(status_code=404, detail="Lecture not found")

    stem      = content.stem
    query_url = os.getenv("QUERY_SERVICE_URL", "http://query_service:8001")
    _ensure_stem_loaded(stem, content.output_dir)

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
                "answer":           qr.get("answer"),
                "timestamps":       qr.get("timestamps", []),
                "graph":            qr.get("graph", {"nodes": [], "edges": []}),
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
        for s in data.get("slides", []):
            img_url = make_file_url(s.get("image_path"))
            ts = s.get("timestamp_formatted", "00:00").split(".")[0]
            if ts.startswith("00:"): ts = ts[3:]

            scenes.append({
                "timestamp":    ts,
                "type":         "emphasis" if s.get("role") == "elaborated" else "slide",
                "text":         s.get("title") or f"Slide {s.get('slide_number')}",
                "image_url":    img_url,
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

    output_dir  = Path(detail["output_dir"])
    nodes_paths = list(output_dir.glob("*_nodes.parquet"))
    edges_paths = list(output_dir.glob("*_edges.parquet"))

    if not nodes_paths:
        raise HTTPException(status_code=404, detail="Graph files not found")

    try:
        ndf = pd.read_parquet(nodes_paths[0])
        edf = pd.read_parquet(edges_paths[0]) if edges_paths else pd.DataFrame(columns=["src_id", "rel_type", "tgt_id", "properties_json"])

        nodes_out = []
        for _, row in ndf.iterrows():
            nid = _str_cell(row.get("node_id"))
            if not nid: continue
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
            if not src or not tgt: continue
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

    return {
        "lecture_id": str(lecture_id),
        "stem": stem,
        "verification_path": str(verifier_path),
        "final_confirmed_claim_count": int(
            summary.get("final_confirmed_claim_count", len(final_claims))
        ),
        "final_confirmed_claims": final_claims,
    }


# ── 기타 유틸 ───────────────────────────────────────────────────────────────
async def get_graph_info(db: AsyncSession, job_id: str) -> Dict[str, Any]:
    job = await get_job(db, job_id)
    if not job: return {"error": "Not Found"}
    stem = Path(job.input_path).stem
    node_count = 0
    driver = get_neo4j_driver()
    if driver:
        try:
            with driver.session() as session:
                record = session.run("MATCH (n {stem: $stem}) RETURN count(n) AS count", stem=stem).single()
                if record: node_count = record["count"]
        finally: driver.close()
    return {"job_id": job_id, "stem": stem, "graph_exists": node_count > 0, "node_count": node_count}


async def retry_graph_only(db: AsyncSession, job_id: str) -> bool:
    job = await get_job(db, job_id)
    if not job: return False
    job.status, job.current_stage, job.error_message = "pending", "Retrying Graph Ingestion...", None
    await db.commit()
    return True
