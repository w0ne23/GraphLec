import asyncio
import os
import sys
import multiprocessing
import logging
from pathlib import Path
from dotenv import load_dotenv
from concurrent.futures import ProcessPoolExecutor
import concurrent.futures
import traceback
from contextlib import redirect_stdout, redirect_stderr

from sqlalchemy import text
from app.db import AsyncSessionLocal
from app.models import (
    JOB_STATUS_DONE,
    JOB_STATUS_WAITING_APPROVAL,
    JOB_TYPE_LEGACY_FULL,
    JOB_TYPE_GRAPH_UPLOAD,
    JOB_TYPE_VERIFIED_UPLOAD,
)
from app.services.job_service import update_job_stage_sync

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
root_env         = PROJECT_ROOT_DIR / ".env"

if root_env.exists():
    load_dotenv(dotenv_path=root_env)
else:
    load_dotenv()

PROJECT_ROOT      = Path("/pipeline") if Path("/pipeline").exists() else PROJECT_ROOT_DIR
LOCAL_STORAGE_DIR = os.getenv("LOCAL_STORAGE_DIR", str(PROJECT_ROOT / "local_storage"))

PIPELINE_STAGE_KEYS = [
    "preprocess_extract_media",
    "preprocess_textualize_transcribe",
    "preprocess_enrich_audio_annotation",
    "verifier_build_analyzer_input",
    "verifier_run",
    "graph_classify_scene",
    "graph_fusion",
    "graph_triples",
    "graph_lance_index",
    "graph_graphrag_index",
    "graph_metadata",
    "graph_recommender_index",
]

GRAPH_UPLOAD_PRECOMPLETED_STAGE_KEYS = {
    "preprocess_extract_media",
    "preprocess_textualize_transcribe",
    "preprocess_enrich_audio_annotation",
}


def _initial_stage_state(job_type: str) -> dict[str, str]:
    stages = {key: "wait" for key in PIPELINE_STAGE_KEYS}
    if job_type == JOB_TYPE_GRAPH_UPLOAD:
        for key in GRAPH_UPLOAD_PRECOMPLETED_STAGE_KEYS:
            stages[key] = "done"
    return stages


def pipeline_process(
    job_id: str,
    lecture_id: str,
    input_path: str,
    job_type: str,
    uploaded_at: str | None = None,
    title: str = "",
):
    pipeline_path = os.getenv("PIPELINE_ROOT", str(Path(__file__).resolve().parent.parent.parent.parent))

    # spawn된 자식 프로세스는 부모의 sys.path를 상속받지 않으므로 pipeline 패키지를 import하기 위해 명시적으로 경로를 추가한다.
    # os.chdir은 pipeline 내부의 상대 경로 참조를 위해 필요하다.
    logger.info(f"--- [Child Process {job_id}] Setting sys.path to: {pipeline_path} ---")
    if pipeline_path not in sys.path:
        sys.path.insert(0, pipeline_path)
    os.chdir(pipeline_path)

    try:
        video_path = Path(input_path)
        if not video_path.is_absolute():
            video_path = Path(pipeline_path) / input_path

        job_type = (job_type or JOB_TYPE_LEGACY_FULL).strip() or JOB_TYPE_LEGACY_FULL
        logger.info(f"--- [Child Process {job_id}] Target video: {video_path} ({job_type}) ---")

        output_dir    = Path(LOCAL_STORAGE_DIR) / "results" / lecture_id
        slides_dir    = output_dir / "slides"
        log_file_path = output_dir / "pipeline.log"

        output_dir.mkdir(parents=True, exist_ok=True)
        slides_dir.mkdir(parents=True, exist_ok=True)

        stages_state = _initial_stage_state(job_type)

        def on_progress(stage_key: str, status: str):
            if stage_key in stages_state:
                stages_state[stage_key] = status
            stages_array = [{"stage": k, "status": v} for k, v in stages_state.items()]
            stage_text   = f"Processing {stage_key}..." if status == "run" else f"Finished {stage_key}"
            update_job_stage_sync(job_id, stages_array, stage_text)
            logger.info(f"[{job_id}] Progress: {stage_key} -> {status}")

        with open(log_file_path, "w", encoding="utf-8", buffering=1) as log_file:
            try:
                log_file.reconfigure(line_buffering=True, write_through=True)
            except Exception:
                pass
            with redirect_stdout(log_file), redirect_stderr(log_file):
                try:
                    logger.info(f"[{job_id}] Importing pipeline...")
                    import pipeline.main as pipeline_main

                    init_array = [{"stage": k, "status": v} for k, v in stages_state.items()]
                    update_job_stage_sync(job_id, init_array, "Starting pipeline")

                    args = pipeline_main.get_parser().parse_args([
                        "--input",        str(video_path),
                        "--output",       str(output_dir),
                        "--slides",       str(slides_dir),
                        "--skip-neo4j",
                        "--metadata-dir", str(output_dir / "metadata"),
                        "--lance-root",   str(output_dir / "lancedb"),
                        "--lecture-id",   lecture_id,
                    ] + (["--title", title] if title else [])
                      + (["--uploaded-at", uploaded_at] if uploaded_at else []))
                    args.job_type = job_type
                    os.environ["PYTHONUNBUFFERED"] = "1"
                    logger.info(f"[{job_id}] Starting pipeline job_type={job_type}...")
                    pipeline_main.run_pipeline(args, progress_callback=on_progress)

                except ImportError as ie:
                    logger.error(f"[{job_id}] ImportError: {ie}. sys.path: {sys.path}")
                    raise

        return True, str(output_dir), None

    except BaseException as e:
        import traceback
        error_details = traceback.format_exc()
        logger.error(f"--- [Child Process {job_id}] FAILED: {e} ---")
        log_file_path = Path(LOCAL_STORAGE_DIR) / "results" / lecture_id / "pipeline.log"
        if log_file_path.parent.exists():
            with open(log_file_path, "a", encoding="utf-8") as log_file:
                log_file.write(f"\n[{job_id}] Pipeline failed: {error_details}\n")
        return False, None, str(e)


async def worker_loop():
    try:
        mp_context = multiprocessing.get_context("spawn")
        executor   = ProcessPoolExecutor(max_workers=1, mp_context=mp_context)

        async with AsyncSessionLocal() as db:
            result = await db.execute(text(
                "UPDATE processing_jobs SET status = 'error', error_message = '서버 재시작으로 인해 분석이 중단되었습니다.' "
                "WHERE status = 'running'"
            ))
            count = result.rowcount
            if count > 0:
                logger.info(f"--- [Worker Recovery] Marked {count} orphaned jobs as error. ---")
            await db.commit()

        logger.info(f"--- [Worker] Started. Storage: {LOCAL_STORAGE_DIR} ---")

        while True:
            try:
                job_id_val = None
                job_lecture_id = None
                job_input_path = None
                job_uploaded_at = None
                job_title = ""
                job_type_val = None

                async with AsyncSessionLocal() as db:
                    result = await db.execute(text("""
                        SELECT pj.id, pj.lecture_id, pj.job_type, l.video_path, l.created_at, l.title
                        FROM processing_jobs pj
                        JOIN lectures l ON l.id = pj.lecture_id
                        WHERE pj.status = 'pending'
                        FOR UPDATE SKIP LOCKED
                        LIMIT 1
                    """))
                    job = result.mappings().first()

                    if job:
                        job_id_val     = job["id"]
                        job_lecture_id = job["lecture_id"]
                        job_input_path = job["video_path"]
                        job_title      = job["title"] or ""
                        job_uploaded_at = (
                            job["created_at"].isoformat()
                            if job["created_at"]
                            else None
                        )
                        job_type_val   = job["job_type"] or JOB_TYPE_LEGACY_FULL
                        await db.execute(text("""
                            UPDATE processing_jobs
                            SET status = 'running', current_stage = 'Starting pipeline'
                            WHERE id = :id
                        """), {"id": job_id_val})
                        await db.commit()

                if not job_id_val:
                    await asyncio.sleep(5)
                    continue

                job_id_str     = str(job_id_val)
                job_lecture_str = str(job_lecture_id)
                job_type_str = str(job_type_val or JOB_TYPE_LEGACY_FULL)
                logger.info(f"--- [Worker] Starting pipeline: {job_id_str} (lecture: {job_lecture_str}, type: {job_type_str}) ---")

                try:
                    loop = asyncio.get_running_loop()
                    success, output_dir, error = await loop.run_in_executor(
                        executor,
                        pipeline_process,
                        job_id_str,
                        job_lecture_str,
                        job_input_path,
                        job_type_str,
                        job_uploaded_at,
                        job_title,
                    )
                except concurrent.futures.process.BrokenProcessPool as bp_err:
                    logger.error(f"--- [Worker EXECUTOR BROKEN] {job_id_str}: {bp_err} ---")
                    error = f"파이프라인 프로세스 강제 종료 (메모리 부족 등): {str(bp_err)}"
                    success = False
                    output_dir = None
                    # 손상된 Executor 재시작
                    executor.shutdown(wait=False)
                    executor = ProcessPoolExecutor(max_workers=1, mp_context=mp_context)
                    logger.info("--- [Worker] Executor restarted ---")
                except Exception as exec_err:
                    logger.error(f"--- [Worker EXECUTOR ERROR] {job_id_str}: {exec_err} ---")
                    error = f"시스템/프로세스 오류: {str(exec_err)}"
                    success = False
                    output_dir = None

                logger.info(f"--- [Worker] Pipeline done: {job_id_str} success={success} ---")

                async with AsyncSessionLocal() as db:
                    if success:
                        final_status = (
                            JOB_STATUS_WAITING_APPROVAL
                            if job_type_str == JOB_TYPE_VERIFIED_UPLOAD
                            else JOB_STATUS_DONE
                        )
                        final_stage = (
                            "Waiting approval"
                            if final_status == JOB_STATUS_WAITING_APPROVAL
                            else "Finished"
                        )
                        await db.execute(text("""
                            UPDATE processing_jobs
                            SET status = :status, current_stage = :stage
                            WHERE id = :id
                        """), {"id": job_id_val, "status": final_status, "stage": final_stage})
                    else:
                        logger.error(f"--- [Worker ERROR] {job_id_str}: {error} ---")
                        await db.execute(text("""
                            UPDATE processing_jobs
                            SET status = 'error', error_message = :err, current_stage = 'Failed'
                            WHERE id = :id
                        """), {"id": job_id_val, "err": error})
                    await db.commit()

            except asyncio.CancelledError:
                raise  # 바깥 try의 CancelledError 처리로 전달
            except Exception as e:
                logger.error(f"--- [Worker FATAL ERROR in loop]: {e} ---")
                # traceback.print_exc() 는 logger.error(..., exc_info=True)로 대체 가능
                logger.error(traceback.format_exc())
                await asyncio.sleep(5)

    except asyncio.CancelledError:
        logger.info("--- [Worker] Shutdown signal received. ---")
    except Exception as fatal_e:
        logger.error(f"--- [Worker FATAL ERROR ON STARTUP]: {fatal_e} ---")
        logger.error(traceback.format_exc())
    finally:
        try:
            executor.shutdown(wait=True)
        except:
            pass
        logger.info("--- [Worker] Executor shut down. ---")

if __name__ == "__main__":
    asyncio.run(worker_loop())
    
