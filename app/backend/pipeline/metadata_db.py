"""
metadata_db.py

Stage 8에서 생성된 추천 메타데이터 JSON을 PostgreSQL lecture_metadata 테이블에 저장한다.
파이프라인은 별도 프로세스에서 실행되므로 async SQLAlchemy 세션 대신 psycopg2 동기 연결을 사용한다.
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any

import psycopg2
from psycopg2.extras import Json


def _database_url_sync() -> str:
    url = os.getenv("DATABASE_URL", "")
    return url.replace("+asyncpg", "") if url else ""


def _text_list(value: Any) -> list[str] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        return None
    result = []
    for item in value:
        if isinstance(item, str):
            text = item.strip()
        elif isinstance(item, dict):
            text = str(item.get("text") or item.get("objective") or "").strip()
        else:
            text = str(item or "").strip()
        if text:
            result.append(text)
    return result


def load_metadata_json(metadata_path: str | Path) -> dict[str, Any]:
    path = Path(metadata_path)
    with path.open(encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"metadata JSON must be an object: {path}")
    return payload


def upsert_lecture_metadata_sync(
    lecture_id: str | None,
    metadata: dict[str, Any],
    metadata_uri: str | None = None,
) -> bool:
    """lecture_metadata를 생성 또는 갱신한다.

    lecture_id나 DATABASE_URL이 없으면 CLI 단독 실행으로 보고 조용히 건너뛴다.
    """
    if not lecture_id:
        return False

    database_url = _database_url_sync()
    if not database_url:
        return False

    values = {
        "lecture_id": uuid.UUID(lecture_id),
        "title": str(metadata.get("title") or "").strip() or "Untitled lecture",
        "instructor_id": metadata.get("instructor_id"),
        "domain": metadata.get("domain"),
        "graph_domain": metadata.get("graph_domain"),
        "graph_subdomain": metadata.get("graph_subdomain"),
        "difficulty": metadata.get("difficulty"),
        "summary": metadata.get("summary"),
        "learning_objectives": _text_list(metadata.get("learning_objectives")),
        "keywords": Json(metadata.get("keywords") or []),
        "concept_roles": Json(metadata.get("concept_roles") or {}),
        "concept_relations": Json(metadata.get("concept_relations") or []),
        "communities": Json(metadata.get("communities") or []),
        "visual_concept_terms": _text_list(metadata.get("visual_concept_terms")),
        "pedagogy": Json(metadata.get("pedagogy") or {}),
        "diagnostics": Json(metadata.get("diagnostics") or {}),
        "duration_sec": metadata.get("duration_sec"),
        "uploaded_at": metadata.get("uploaded_at"),
        "metadata_version": 1,
        "metadata_uri": metadata_uri,
    }

    sql = """
        INSERT INTO lecture_metadata (
            lecture_id,
            title,
            instructor_id,
            domain,
            graph_domain,
            graph_subdomain,
            difficulty,
            summary,
            learning_objectives,
            keywords,
            concept_roles,
            concept_relations,
            communities,
            visual_concept_terms,
            pedagogy,
            diagnostics,
            duration_sec,
            uploaded_at,
            metadata_version,
            metadata_uri,
            created_at,
            updated_at
        )
        VALUES (
            %(lecture_id)s,
            %(title)s,
            %(instructor_id)s,
            %(domain)s,
            %(graph_domain)s,
            %(graph_subdomain)s,
            %(difficulty)s,
            %(summary)s,
            %(learning_objectives)s,
            %(keywords)s,
            %(concept_roles)s,
            %(concept_relations)s,
            %(communities)s,
            %(visual_concept_terms)s,
            %(pedagogy)s,
            %(diagnostics)s,
            %(duration_sec)s,
            %(uploaded_at)s,
            %(metadata_version)s,
            %(metadata_uri)s,
            NOW(),
            NOW()
        )
        ON CONFLICT (lecture_id) DO UPDATE SET
            title = EXCLUDED.title,
            instructor_id = EXCLUDED.instructor_id,
            domain = EXCLUDED.domain,
            graph_domain = EXCLUDED.graph_domain,
            graph_subdomain = EXCLUDED.graph_subdomain,
            difficulty = EXCLUDED.difficulty,
            summary = EXCLUDED.summary,
            learning_objectives = EXCLUDED.learning_objectives,
            keywords = EXCLUDED.keywords,
            concept_roles = EXCLUDED.concept_roles,
            concept_relations = EXCLUDED.concept_relations,
            communities = EXCLUDED.communities,
            visual_concept_terms = EXCLUDED.visual_concept_terms,
            pedagogy = EXCLUDED.pedagogy,
            diagnostics = EXCLUDED.diagnostics,
            duration_sec = EXCLUDED.duration_sec,
            uploaded_at = EXCLUDED.uploaded_at,
            metadata_version = EXCLUDED.metadata_version,
            metadata_uri = EXCLUDED.metadata_uri,
            updated_at = NOW()
    """

    conn = psycopg2.connect(database_url)
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(sql, values)
    finally:
        conn.close()

    return True
