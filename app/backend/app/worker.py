import asyncio
import os
import sys
import multiprocessing
import psycopg2
from psycopg2.extras import Json
from pathlib import Path
from dotenv import load_dotenv
from concurrent.futures import ProcessPoolExecutor
import concurrent.futures
import traceback
from contextlib import redirect_stdout, redirect_stderr

from sqlalchemy import text
from app.db import AsyncSessionLocal

PROJECT_ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
root_env         = PROJECT_ROOT_DIR / ".env"

if root_env.exists():
    load_dotenv(dotenv_path=root_env)
else:
    load_dotenv()

GEMINI_API_KEY_1 = os.getenv("GOOGLE_API_KEY_1")
GEMINI_API_KEY_2 = os.getenv("GOOGLE_API_KEY_2")
GROQ_API_KEY     = os.getenv("GROQ_API_KEY")

DATABASE_URL      = os.getenv("DATABASE_URL")
DATABASE_URL_SYNC = DATABASE_URL.replace("+asyncpg", "") if DATABASE_URL else ""

PROJECT_ROOT      = Path("/pipeline") if Path("/pipeline").exists() else PROJECT_ROOT_DIR
LOCAL_STORAGE_DIR = os.getenv("LOCAL_STORAGE_DIR", str(PROJECT_ROOT / "local_storage"))


def update_job_stage_sync(job_id: str, current_stages: list, current_stage_text: str):
    if not DATABASE_URL_SYNC:
        return
    try:
        conn = psycopg2.connect(DATABASE_URL_SYNC)
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE jobs SET pipeline_stages = %s, current_stage = %s WHERE id = %s",
                    (Json(current_stages), current_stage_text, job_id),
                )
        conn.close()
    except Exception as e:
        print(f"--- [Worker Sync DB Error] Failed to update stage: {e} ---", flush=True)


def pipeline_process(job_id: str, input_path: str):
    pipeline_path = os.getenv("PIPELINE_ROOT", str(Path(__file__).resolve().parent.parent.parent.parent))

    print(f"--- [Child Process {job_id}] Setting sys.path to: {pipeline_path} ---", flush=True)
    if pipeline_path not in sys.path:
        sys.path.insert(0, pipeline_path)
    os.chdir(pipeline_path)

    try:
        video_path = Path(input_path)
        if not video_path.is_absolute():
            video_path = Path(pipeline_path) / input_path

        print(f"--- [Child Process {job_id}] Target video: {video_path} ---", flush=True)

        output_dir    = Path(LOCAL_STORAGE_DIR) / "results" / job_id
        slides_dir    = output_dir / "slides"
        log_file_path = output_dir / "pipeline.log"

        output_dir.mkdir(parents=True, exist_ok=True)
        slides_dir.mkdir(parents=True, exist_ok=True)

        stages_state = {
            "scene": "wait", "voice": "wait", "stt": "wait",
            "integrate": "wait", "graph": "wait",
            "summarize": "wait", "metadata": "wait",
        }

        def on_progress(stage_key: str, status: str):
            if stage_key in stages_state:
                stages_state[stage_key] = status
            stages_array = [{"stage": k, "status": v} for k, v in stages_state.items()]
            stage_text   = f"Processing {stage_key}..." if status == "run" else f"Finished {stage_key}"
            update_job_stage_sync(job_id, stages_array, stage_text)
            print(f"[{job_id}] Progress: {stage_key} -> {status}", flush=True)

        with open(log_file_path, "w", encoding="utf-8") as log_file:
            with redirect_stdout(log_file), redirect_stderr(log_file):
                try:
                    print(f"[{job_id}] Importing pipeline...", flush=True)
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
                    ])
                    print(f"[{job_id}] Starting pipeline...", flush=True)
                    pipeline_main.run_pipeline(args, progress_callback=on_progress)

                except ImportError as ie:
                    print(f"[{job_id}] ImportError: {ie}. sys.path: {sys.path}", flush=True)
                    raise

        return True, str(output_dir), None

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"--- [Child Process {job_id}] FAILED: {e} ---", flush=True)
        log_file_path = Path(LOCAL_STORAGE_DIR) / "results" / job_id / "pipeline.log"
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
                "UPDATE jobs SET status = 'error', error_message = '서버 재시작으로 인해 분석이 중단되었습니다.' "
                "WHERE status = 'running'"
            ))
            count = result.rowcount
            if count > 0:
                print(f"--- [Worker Recovery] Marked {count} orphaned jobs as error. ---", flush=True)
            await db.commit()

        print(f"--- [Worker] Started. Storage: {LOCAL_STORAGE_DIR} ---", flush=True)

        while True:
            try:
                job_id_val = None
                job_input_path = None

                async with AsyncSessionLocal() as db:
                    result = await db.execute(text("""
                        SELECT id, input_path FROM jobs
                        WHERE status = 'pending'
                        FOR UPDATE SKIP LOCKED
                        LIMIT 1
                    """))
                    job = result.mappings().first()

                    if job:
                        job_id_val     = job["id"]
                        job_input_path = job["input_path"]
                        await db.execute(text("""
                            UPDATE jobs
                            SET status = 'running', current_stage = 'Starting pipeline', updated_at = now()
                            WHERE id = :id
                        """), {"id": job_id_val})
                        await db.commit()

                if not job_id_val:
                    await asyncio.sleep(5)
                    continue

                job_id_str = str(job_id_val)
                print(f"--- [Worker] Starting pipeline: {job_id_str} ---", flush=True)

                try:
                    loop = asyncio.get_running_loop()
                    success, output_dir, error = await loop.run_in_executor(
                        executor, pipeline_process, job_id_str, job_input_path
                    )
                except concurrent.futures.process.BrokenProcessPool as bp_err:
                    print(f"--- [Worker EXECUTOR BROKEN] {job_id_str}: {bp_err} ---", flush=True)
                    error = f"파이프라인 프로세스 강제 종료 (메모리 부족 등): {str(bp_err)}"
                    success = False
                    output_dir = None
                    # 손상된 Executor 재시작
                    executor.shutdown(wait=False)
                    executor = ProcessPoolExecutor(max_workers=1, mp_context=mp_context)
                    print("--- [Worker] Executor restarted ---", flush=True)
                except Exception as exec_err:
                    print(f"--- [Worker EXECUTOR ERROR] {job_id_str}: {exec_err} ---", flush=True)
                    error = f"시스템/프로세스 오류: {str(exec_err)}"
                    success = False
                    output_dir = None

                print(f"--- [Worker] Pipeline done: {job_id_str} success={success} ---", flush=True)

                async with AsyncSessionLocal() as db:
                    if success:
                        await db.execute(text("""
                            UPDATE lectures_content
                            SET output_dir = :out_dir
                            WHERE job_id = :id
                        """), {"out_dir": output_dir, "id": job_id_val})
                        await db.execute(text("""
                            UPDATE jobs
                            SET status = 'done', current_stage = 'Finished', updated_at = now()
                            WHERE id = :id
                        """), {"id": job_id_val})
                    else:
                        print(f"--- [Worker ERROR] {job_id_str}: {error} ---", flush=True)
                        await db.execute(text("""
                            UPDATE jobs
                            SET status = 'error', error_message = :err,
                                current_stage = 'Failed', updated_at = now()
                            WHERE id = :id
                        """), {"id": job_id_val, "err": error})
                    await db.commit()

            except asyncio.CancelledError:
                raise  # 바깥 try의 CancelledError 처리로 전달
            except Exception as e:
                print(f"--- [Worker FATAL ERROR in loop]: {e} ---", flush=True)
                traceback.print_exc()
                await asyncio.sleep(5)

    except asyncio.CancelledError:
        print("--- [Worker] Shutdown signal received. ---", flush=True)
    except Exception as fatal_e:
        print(f"--- [Worker FATAL ERROR ON STARTUP]: {fatal_e} ---", flush=True)
        traceback.print_exc()
    finally:
        try:
            executor.shutdown(wait=True)
        except:
            pass
        print("--- [Worker] Executor shut down. ---", flush=True)


if __name__ == "__main__":
    asyncio.run(worker_loop())