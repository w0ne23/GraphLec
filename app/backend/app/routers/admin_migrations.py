import os
import shutil
import tarfile
import tempfile
from pathlib import Path

from fastapi import APIRouter, Header, HTTPException, UploadFile, File

from app.db import engine


router = APIRouter(prefix="/admin/migrations")

LOCAL_STORAGE_DIR = Path(os.getenv("LOCAL_STORAGE_DIR", "/pipeline/local_storage"))
MIGRATION_TOKEN = os.getenv("GRAPHLEC_MIGRATION_TOKEN", "")


def _require_token(token: str | None) -> None:
    if not MIGRATION_TOKEN:
        raise HTTPException(status_code=503, detail="Migration import is not configured")
    if token != MIGRATION_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid migration token")


def _safe_extract_tar(tar: tarfile.TarFile, dest: Path) -> None:
    dest_resolved = dest.resolve()
    for member in tar.getmembers():
        if member.issym() or member.islnk():
            raise HTTPException(status_code=400, detail="Invalid bundle link")
        member_path = (dest / member.name).resolve()
        if dest_resolved not in [member_path, *member_path.parents]:
            raise HTTPException(status_code=400, detail="Invalid bundle path")
    tar.extractall(dest)


def _copy_contents(src: Path, dst: Path) -> int:
    if not src.exists():
        return 0

    copied = 0
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.rglob("*"):
        rel = item.relative_to(src)
        target = dst / rel
        if item.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item, target)
        copied += 1
    return copied


async def _execute_migration_sql(sql_path: Path) -> int:
    statements = [
        line.strip()
        for line in sql_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and line.strip() not in {"BEGIN;", "COMMIT;"}
    ]

    async with engine.begin() as conn:
        for statement in statements:
            await conn.exec_driver_sql(statement)
        await conn.exec_driver_sql(
            """
            UPDATE lectures
            SET
              video_path = CASE
                WHEN video_path LIKE '%/local_storage/%'
                  THEN regexp_replace(video_path, '^.*/local_storage/', '/pipeline/local_storage/')
                ELSE video_path
              END,
              output_dir = CASE
                WHEN output_dir LIKE '%/local_storage/%'
                  THEN regexp_replace(output_dir, '^.*/local_storage/', '/pipeline/local_storage/')
                ELSE output_dir
              END
            """
        )
        await conn.exec_driver_sql(
            """
            UPDATE processing_jobs
            SET
              status = 'error',
              error_message = COALESCE(error_message, '마이그레이션 중 중단된 작업')
            WHERE status IN ('pending', 'running')
            """
        )

    return len(statements)


@router.post("/import")
async def import_migration_bundle(
    bundle: UploadFile = File(...),
    x_graphlec_migration_token: str | None = Header(default=None),
):
    _require_token(x_graphlec_migration_token)

    if not bundle.filename or not bundle.filename.endswith((".tgz", ".tar.gz")):
        raise HTTPException(status_code=400, detail="Bundle must be a .tgz or .tar.gz file")

    with tempfile.TemporaryDirectory(prefix="graphlec_http_import.") as tmp:
        workdir = Path(tmp)
        bundle_path = workdir / "bundle.tgz"

        with bundle_path.open("wb") as f:
            shutil.copyfileobj(bundle.file, f)

        extract_dir = workdir / "bundle"
        extract_dir.mkdir()
        try:
            with tarfile.open(bundle_path, "r:gz") as tar:
                _safe_extract_tar(tar, extract_dir)
        except tarfile.TarError as exc:
            raise HTTPException(status_code=400, detail=f"Invalid migration bundle: {exc}") from exc

        sql_path = extract_dir / "migration.sql"
        if not sql_path.exists():
            raise HTTPException(status_code=400, detail="Invalid bundle: migration.sql is missing")

        input_files = _copy_contents(
            extract_dir / "local_storage" / "inputs",
            LOCAL_STORAGE_DIR / "inputs",
        )
        result_files = _copy_contents(
            extract_dir / "local_storage" / "results",
            LOCAL_STORAGE_DIR / "results",
        )
        sql_statements = await _execute_migration_sql(sql_path)

    nodes_parquet = len(list((LOCAL_STORAGE_DIR / "results").rglob("*_nodes.parquet")))
    edges_parquet = len(list((LOCAL_STORAGE_DIR / "results").rglob("*_edges.parquet")))

    return {
        "status": "ok",
        "sql_statements": sql_statements,
        "input_files": input_files,
        "result_files": result_files,
        "nodes_parquet": nodes_parquet,
        "edges_parquet": edges_parquet,
    }
