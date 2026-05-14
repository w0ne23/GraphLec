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

from app.models import Lecture, ProcessingJob, GraphSession
from app.services.neo4j_service import (
    neo4j_session,
    _is_stem_loaded,
    _unload_stem_from_neo4j,
    _ensure_stem_loaded,
)

logger = logging.getLogger(__name__)

# ── 경로 설정 ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path("/pipeline") if Path("/pipeline").exists() else Path(__file__).resolve().parents[4]
LOCAL_STORAGE_DIR = os.getenv("LOCAL_STORAGE_DIR", str(PROJECT_ROOT / "local_storage"))
GRAPH_SESSION_TTL_SEC = int(os.getenv("GRAPH_SESSION_TTL_SEC", "180"))
METADATA_DIR = Path(os.getenv("GRAPHLEC_METADATA_DIR", str(PROJECT_ROOT / "app" / "backend" / "metadata")))


def _metadata_category(domain: str) -> str:
    text = (domain or "").lower()
    if "math" in text:
        return "수학"
    if any(token in text for token in ("cs", "eng", "software", "data")):
        return "컴퓨터 과학"
    return "기타"


def _metadata_tags(meta: dict, limit: int = 3) -> List[str]:
    tags: List[str] = []
    for keyword in (meta.get("keywords") or [])[:limit]:
        if isinstance(keyword, dict) and keyword.get("keyword"):
            tags.append(str(keyword["keyword"]))
        elif isinstance(keyword, str):
            tags.append(keyword)
    return tags


def _metadata_created_at(meta: dict, video_id: str) -> str:
    for key in ("created_at", "updated_at", "date", "published_at"):
        value = meta.get(key)
        if value:
            return str(value)

    digest = hashlib.sha1(video_id.encode("utf-8")).hexdigest()
    offset_days = int(digest[:4], 16) % 180
    base = datetime.now(timezone.utc).replace(hour=9, minute=0, second=0, microsecond=0)
    return (base - timedelta(days=offset_days)).isoformat()


def _load_metadata_by_video_id() -> Dict[str, dict]:
    if not METADATA_DIR.exists():
        return {}

    by_id: Dict[str, dict] = {}
    for path in sorted(METADATA_DIR.glob("*_metadata.json")):
        try:
            with path.open(encoding="utf-8") as f:
                raw = json.load(f)
        except Exception as exc:
            logger.warning("metadata lecture load failed: %s (%s)", path, exc)
            continue

        metas = raw if isinstance(raw, list) else [raw]
        for meta in metas:
            if not isinstance(meta, dict):
                continue
            video_id = str(meta.get("video_id") or "").strip()
            if video_id:
                by_id[video_id] = meta
    return by_id


def _load_metadata_lecture_items(existing_ids: set[str]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for video_id, meta in _load_metadata_by_video_id().items():
        if video_id in existing_ids:
            continue

        items.append({
            "id": video_id,
            "job_id": None,
            "status": "done",
            "title": meta.get("title") or video_id,
            "category": _metadata_category(meta.get("domain", "")),
            "created_at": _metadata_created_at(meta, video_id),
            "error_message": None,
            "pipeline_stages": [],
            "tags": _metadata_tags(meta),
            "is_dummy": True,
            "source": "metadata",
        })

    return items


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
    try:
        with neo4j_session() as session:
            session.run("MATCH (n {stem: $stem}) DETACH DELETE n", stem=stem)
    except HTTPException:
        pass

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
    metadata_by_id = _load_metadata_by_video_id() if scope == 'browse' else {}

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

        lecture_id = str(lecture.id)
        metadata = metadata_by_id.get(lecture_id, {})
        out.append({
            "id": lecture_id,
            "job_id": str(job.id) if job else None,
            "status": job_status,
            "title": lecture.title or str(lecture.id),
            "category": lecture.category or "기타",
            "created_at": lecture.created_at.isoformat() if lecture.created_at else None,
            "error_message": job.error_message if job else None,
            "pipeline_stages": job.pipeline_stages or [] if job else [],
            "tags": _metadata_tags(metadata) if metadata else [],
            "is_dummy": False,
            "source": "database",
        })

    if scope == 'browse':
        existing_ids = {str(lecture.id) for lecture, _job in rows}
        metadata_items = _load_metadata_lecture_items(existing_ids)
        for item in metadata_items:
            if category and item["category"] != category:
                continue
            if search and search.lower() not in (item["title"] or '').lower():
                continue
            out.append(item)

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


# ── GraphSession 헬퍼 ────────────────────────────────────────────────────────
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


# ── GraphSession 관련 ────────────────────────────────────────────────────────
async def graph_enter(db: AsyncSession, lecture_id: str, session_id: str) -> Dict[str, Any]:
    lecture = await _get_lecture(db, lecture_id)
    if not lecture:
        raise HTTPException(status_code=404, detail="Lecture not found")

    lecture_id_uuid = lecture.id
    stem = str(lecture_id_uuid)
    output_dir = str(lecture.output_dir)
    now = _utcnow()

    await _cleanup_stale_sessions(db, lecture_id_uuid)
    await _commit_graph_session_touch(
        db,
        lecture_id=lecture_id_uuid,
        session_id=session_id,
        now=now,
    )

    loop = asyncio.get_event_loop()
    load_info = await loop.run_in_executor(None, _ensure_stem_loaded, stem, output_dir)
    active_count = await _active_session_count(db, lecture_id_uuid)
    return {
        "lecture_id": stem,
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

    lecture_id_uuid = lecture.id
    stem = str(lecture_id_uuid)
    now = _utcnow()

    await _commit_graph_session_touch(
        db,
        lecture_id=lecture_id_uuid,
        session_id=session_id,
        now=now,
    )
    return {
        "lecture_id": stem,
        "stem": stem,
        "session_id": session_id,
        "active_sessions": await _active_session_count(db, lecture_id_uuid),
    }


async def graph_leave(db: AsyncSession, lecture_id: str, session_id: str) -> Dict[str, Any]:
    lecture = await _get_lecture(db, lecture_id)
    if not lecture:
        raise HTTPException(status_code=404, detail="Lecture not found")

    lecture_id_uuid = lecture.id
    stem = str(lecture_id_uuid)
    q = select(GraphSession).where(
        GraphSession.lecture_id == lecture_id_uuid,
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

    active_count = await _active_session_count(db, lecture_id_uuid)
    loop = asyncio.get_event_loop()
    unloaded_now = False
    if active_count == 0 and await loop.run_in_executor(None, _is_stem_loaded, stem):
        await loop.run_in_executor(None, _unload_stem_from_neo4j, stem)
        unloaded_now = True

    loaded = await loop.run_in_executor(None, _is_stem_loaded, stem)
    return {
        "lecture_id": stem,
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

    lecture_id_uuid = lecture.id
    stem = str(lecture_id_uuid)
    loop = asyncio.get_event_loop()
    loaded = await loop.run_in_executor(None, _is_stem_loaded, stem)
    active_count = await _active_session_count(db, lecture_id_uuid)
    return {
        "lecture_id": stem,
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

def _filter_served_slide_errors(items: list[dict]) -> list[dict]:
    filtered = []
    for item in items or []:
        if not isinstance(item, dict):
            continue
        problematic = str(item.get("problematic_text", "") or "").strip()
        corrected = str(item.get("corrected_text", "") or "").strip()
        if not problematic or not corrected or problematic == corrected:
            continue
        filtered.append(item)
    return filtered


def _slide_number_key(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_count(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _load_slide_image_url_map(output_dir: Path) -> dict[str, dict[Any, str]]:
    classified_paths = list(output_dir.glob("*_slide_classified.json"))
    if not classified_paths:
        return {"by_number": {}, "by_title": {}}

    image_urls: dict[str, dict[Any, str]] = {"by_number": {}, "by_title": {}}
    for classified_path in classified_paths:
        try:
            with open(classified_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            logger.warning("Failed to read slide image metadata: %s", classified_path, exc_info=True)
            continue

        slides = []
        if isinstance(data.get("slides"), list):
            slides.extend(data.get("slides") or [])
        if isinstance(data.get("scenes"), list):
            slides.extend(data.get("scenes") or [])

        for slide in slides:
            if not isinstance(slide, dict):
                continue

            image_url = make_file_url(slide.get("image_path"))
            if not image_url:
                continue

            slide_number = _slide_number_key(slide.get("slide_number"))
            if slide_number is not None:
                image_urls["by_number"][slide_number] = image_url

            title = str(slide.get("title") or "").strip()
            if title:
                image_urls["by_title"][title] = image_url

    return image_urls


def _attach_slide_image_urls(items: list[dict], image_urls: dict[str, dict[Any, str]]) -> list[dict]:
    by_number = image_urls.get("by_number", {}) or {}
    by_title = image_urls.get("by_title", {}) or {}
    enriched = []
    for item in items or []:
        if not isinstance(item, dict):
            continue

        copied = dict(item)
        slide_number = _slide_number_key(copied.get("slide_number"))
        slide_title = str(copied.get("slide_title") or "").strip()
        image_url = copied.get("slide_image_url") or copied.get("image_url")
        if not image_url:
            image_url = make_file_url(copied.get("slide_image_path"))
        if not image_url:
            image_url = by_title.get(slide_title)
            if not image_url and slide_number is not None:
                image_url = by_number.get(slide_number)
        if image_url:
            copied.setdefault("slide_image_url", image_url)
            copied.setdefault("image_url", image_url)
        enriched.append(copied)

    return enriched


async def get_content_verification(db: AsyncSession, lecture_id: str) -> Dict[str, Any]:
    """verifier 결과 조회 (Lecture ID 기준)."""
    detail = await get_lecture_detail(db, lecture_id)
    if not detail or not detail.get("output_dir") or not detail.get("stem"):
        raise HTTPException(status_code=404, detail="Lecture result not found")

    output_dir = Path(detail["output_dir"])
    stem = str(detail["stem"])
    analyzer_dir = output_dir / f"{stem}_analyzer"

    candidate_paths = [
        analyzer_dir / f"{stem}_verification_final.json",
        output_dir / f"{stem}_verification_final.json",
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
    content_summary = data.get("summary", {}) or summary
    feedback_items = data.get("feedback_items", []) or []
    confirmed_feedback_items = [
        item for item in feedback_items
        if isinstance(item, dict) and item.get("status") == "confirmed"
    ]
    professor_check_feedback_items = [
        item for item in feedback_items
        if isinstance(item, dict) and item.get("status") in {"professor_check", "review_needed"}
    ]
    rejected_feedback_items = [
        item for item in feedback_items
        if isinstance(item, dict) and item.get("status") == "rejected"
    ]
    final_claims = flow.get("final_confirmed_claims", []) or data.get("final_confirmed_claims", []) or []
    needs_review_claims = flow.get("needs_review_claims", []) or data.get("needs_review_claims", []) or []
    verifier_rejected_claims = flow.get("verifier_rejected_claims", []) or data.get("verifier_rejected_claims", []) or []
    slide_image_urls = _load_slide_image_url_map(output_dir)
    slide_errors = _attach_slide_image_urls(
        _filter_served_slide_errors(data.get("slide_errors", []) or []),
        slide_image_urls,
    )
    slide_error_needs_review = _attach_slide_image_urls(
        _filter_served_slide_errors(data.get("slide_error_needs_review", []) or []),
        slide_image_urls,
    )

    return {
        "lecture_id": str(detail["id"]),
        "stem": stem,
        "verification_path": str(verifier_path),
        "schema_version": data.get("schema_version"),
        "mode": data.get("mode", ""),
        "verification_date": data.get("verification_date", ""),
        "models": data.get("models", {}) or [],
        "pipeline_models": data.get("pipeline_models", {}) or {},
        "primary_model": data.get("primary_model", ""),
        "verifier_source_models": data.get("verifier_source_models", []) or [],
        "verifier_model_weights": data.get("verifier_model_weights", {}) or {},
        "severity_score_report": data.get("severity_score_report", {}) or {},
        "summary": content_summary,
        "overview": data.get("claim_decision_overview", []) or [],
        "counts": {
            "final_confirmed": _safe_count(
                content_summary.get(
                    "confirmed_feedback_count",
                    summary.get("final_confirmed_claim_count", len(confirmed_feedback_items) or len(final_claims)),
                )
            ),
            "needs_review": _safe_count(
                content_summary.get(
                    "review_needed_feedback_count",
                    summary.get("needs_review_claim_count", len(professor_check_feedback_items) or len(needs_review_claims)),
                )
            ),
            "rejected": _safe_count(
                content_summary.get("rejected_feedback_count", len(rejected_feedback_items))
            ),
            "slide_errors": _safe_count(content_summary.get("slide_error_count", len(slide_errors))),
            "slide_error_needs_review": len(slide_error_needs_review),
            "verifier_rejected": _safe_count(summary.get("verifier_rejected_claim_count", len(verifier_rejected_claims))),
        },
        "final_confirmed_claim_count": _safe_count(
            content_summary.get(
                "confirmed_feedback_count",
                summary.get("final_confirmed_claim_count", len(confirmed_feedback_items) or len(final_claims)),
            )
        ),
        "claims": data.get("claims", []) or [],
        "feedback_groups": data.get("feedback_groups", []) or [],
        "feedback_items": feedback_items,
        "views": data.get("views", {}) or {},
        "final_confirmed_claims": final_claims,
        "needs_review_claims": needs_review_claims,
        "verifier_rejected_claims": verifier_rejected_claims,
        "issues": data.get("issues", []) or [],
        "slide_errors": slide_errors,
        "slide_error_needs_review": slide_error_needs_review,
        "slide_error_consensus": data.get("slide_error_consensus", {}) or {},
        "slide_error_status": data.get("slide_error_status", ""),
        "slide_error_summary": data.get("slide_error_summary", {}) or {},
        "slide_error_path": data.get("slide_error_path", ""),
        "claim_decision_flow_summary": summary,
        "classified_issue_artifacts": data.get("classified_issue_artifacts", {}) or {},
        "classified_issue_verifier_path": data.get("classified_issue_verifier_path", ""),
        "classified_issue_verifier": (data.get("views", {}) or {}).get("classified_issue_verifier", {}),
    }


# ── 기타 유틸 ───────────────────────────────────────────────────────────────
async def get_graph_info(db: AsyncSession, lecture_id: str) -> Dict[str, Any]:
    lecture = await _get_lecture(db, lecture_id)
    if not lecture:
        return {"error": "Not Found"}
    stem = str(lecture.id)
    node_count = 0
    try:
        with neo4j_session() as session:
            record = session.run(
                "MATCH (n {stem: $stem}) RETURN count(n) AS count", stem=stem
            ).single()
            if record:
                node_count = record["count"]
    except Exception as e:
        logger.warning("Failed to get graph info for %s: %s", stem, e)

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
