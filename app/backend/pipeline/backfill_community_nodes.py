"""
backfill_community_nodes.py
───────────────────────────
DB의 lecture_metadata.communities에 nodes 필드를 추가한다.

GraphRAG output parquet(communities + entities)에서 community별 entity 이름 목록을 추출해
이미 저장된 communities JSON에 "nodes" 키를 채워 넣는다.

사용법:
  python backfill_community_nodes.py --local_storage /path/to/local_storage [--dry_run]

옵션:
  --local_storage   local_storage 루트 경로 (기본: 환경변수 PIPELINE_ROOT/local_storage)
  --dry_run         DB 업데이트 없이 결과만 출력
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import psycopg2
from psycopg2.extras import Json


def _database_url() -> str:
    url = os.getenv("DATABASE_URL", "")
    if url:
        return url.replace("+asyncpg", "")
    user     = os.getenv("POSTGRES_USER", "user")
    password = os.getenv("POSTGRES_PASSWORD", "password")
    host     = os.getenv("POSTGRES_HOST", "db")
    port     = os.getenv("POSTGRES_PORT", "5432")
    db       = os.getenv("POSTGRES_DB", "graphlec")
    return f"postgresql://{user}:{password}@{host}:{port}/{db}"


def _build_community_nodes_map(graphrag_output_dir: Path) -> dict[str, list[str]]:
    """community ID(str) → entity 이름 목록."""
    try:
        import pyarrow.parquet as pq
        c_rows = pq.read_table(graphrag_output_dir / "communities.parquet").to_pylist()
        e_rows = pq.read_table(graphrag_output_dir / "entities.parquet").to_pylist()
    except Exception as exc:
        print(f"    ⚠ parquet 로드 실패: {exc}")
        return {}

    entity_title: dict[str, str] = {
        str(r.get("id") or ""): str(r.get("title") or "").strip()
        for r in e_rows
    }
    nodes_map: dict[str, list[str]] = {}
    for row in c_rows:
        cid = str(row.get("community") or "")
        nodes = [
            entity_title[str(eid)]
            for eid in (row.get("entity_ids") or [])
            if entity_title.get(str(eid))
        ]
        if nodes:
            nodes_map[cid] = nodes
    return nodes_map


def _patch_communities(communities: list, nodes_map: dict[str, list[str]]) -> tuple[list, int]:
    """communities 리스트에 nodes 필드를 채운다. 변경 수 반환."""
    patched = 0
    result = []
    for c in communities:
        c = dict(c)
        cid = str(c.get("community") or "")
        nodes = nodes_map.get(cid, [])
        if "nodes" not in c or c["nodes"] != nodes:
            c["nodes"] = nodes
            patched += 1
        result.append(c)
    return result, patched


def run(local_storage: Path, dry_run: bool = False) -> None:
    conn = psycopg2.connect(_database_url())
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT lm.lecture_id, lm.metadata_uri, lm.communities
                FROM lecture_metadata lm
                JOIN lectures l ON l.id = lm.lecture_id
                WHERE l.is_published IS TRUE
                ORDER BY lm.metadata_uri
            """)
            rows = cur.fetchall()

        print(f"DB published 강의: {len(rows)}개\n")
        updated = 0
        skipped = 0

        for lecture_id, metadata_uri, communities in rows:
            # metadata_uri 예: 9270142c-..._metadata.json
            if not metadata_uri:
                print(f"  [{lecture_id}] metadata_uri 없음 — 건너뜀")
                skipped += 1
                continue

            stem = Path(metadata_uri).name.replace("_metadata.json", "")
            results_dir = local_storage / "results" / stem
            graphrag_output = results_dir / "graphrag" / "output"

            if not graphrag_output.exists():
                print(f"  [{stem[:8]}...] graphrag output 없음 — 건너뜀")
                skipped += 1
                continue

            nodes_map = _build_community_nodes_map(graphrag_output)
            if not nodes_map:
                print(f"  [{stem[:8]}...] nodes_map 비어있음 — 건너뜀")
                skipped += 1
                continue

            current = communities if isinstance(communities, list) else []
            patched, count = _patch_communities(current, nodes_map)

            if count == 0:
                print(f"  [{stem[:8]}...] 변경 없음")
                skipped += 1
                continue

            print(f"  [{stem[:8]}...] communities {len(patched)}개, nodes 업데이트 {count}개", end="")
            if dry_run:
                print(" [dry_run]")
            else:
                with conn.cursor() as cur:
                    cur.execute(
                        "UPDATE lecture_metadata SET communities = %s WHERE lecture_id = %s",
                        (Json(patched), lecture_id),
                    )
                conn.commit()
                print(" ✓")
                updated += 1

        print(f"\n완료: {updated}개 업데이트, {skipped}개 건너뜀")
    finally:
        conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="lecture_metadata.communities에 nodes 필드 백필")
    parser.add_argument(
        "--local_storage",
        default=None,
        help="local_storage 루트 경로 (기본: PIPELINE_ROOT/local_storage 또는 ./local_storage)",
    )
    parser.add_argument("--dry_run", action="store_true", help="DB 업데이트 없이 결과만 출력")
    args = parser.parse_args()

    if args.local_storage:
        ls_path = Path(args.local_storage)
    else:
        repo_root = Path(os.getenv("PIPELINE_ROOT") or os.getenv("GRAPHLEC_ROOT") or ".")
        ls_path = repo_root / "local_storage"

    if not ls_path.exists():
        print(f"[오류] local_storage 경로 없음: {ls_path}")
        raise SystemExit(1)

    run(ls_path, dry_run=args.dry_run)
