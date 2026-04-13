import asyncio
import os
import sys
import shutil
import json
import psycopg2
from psycopg2.extras import Json
from pathlib import Path
from dotenv import load_dotenv
from contextlib import redirect_stdout, redirect_stderr

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:pass@db/graphlec")

def get_conn():
    return psycopg2.connect(DATABASE_URL.replace("+asyncpg", ""))

def update_job_stage_sync(job_id, stages, current_text):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE jobs SET pipeline_stages = %s, current_stage = %s, updated_at = now() WHERE id = %s",
                (Json(stages), current_text, job_id)
            )
            conn.commit()

def finalize_job(job_id, status, error=None):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE jobs SET status = %s, error_message = %s, updated_at = now() WHERE id = %s",
                (status, error, job_id)
            )
            
            # 작업 성공 시 LectureContent 테이블에 등록
            if status == "done":
                cur.execute("SELECT input_path, output_dir FROM jobs WHERE id = %s", (job_id,))
                row = cur.fetchone()
                if row:
                    input_path, output_dir = row
                    stem = Path(input_path).stem
                    cur.execute("""
                        INSERT INTO lectures_content (id, job_id, stem, title, video_path, output_dir, created_at)
                        VALUES (gen_random_uuid(), %s, %s, %s, %s, %s, now())
                        ON CONFLICT (stem) DO UPDATE SET video_path = EXCLUDED.video_path, output_dir = EXCLUDED.output_dir
                    """, (job_id, stem, stem, input_path, output_dir))
            conn.commit()

async def pipeline_process(job_id, input_path):
    video_path = Path(input_path)
    output_dir = video_path.parent.parent.parent / "results" / job_id
    output_dir.mkdir(parents=True, exist_ok=True)
    slides_dir = output_dir / "slides"
    slides_dir.mkdir(parents=True, exist_ok=True)

    # 7단계 상태 초기화
    stages_state = {
        "scene": "wait", "voice": "wait", "stt": "wait",
        "integrate": "wait", "graph": "wait",
        "summarize": "wait", "metadata": "wait"
    }

    def on_progress(stage, status):
        if stage in stages_state: stages_state[stage] = status
        stages_array = [{"stage": k, "status": v} for k, v in stages_state.items()]
        update_job_stage_sync(job_id, stages_array, f"Processing {stage}...")

    # DB에 output_dir 업데이트
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("UPDATE jobs SET output_dir = %s, status = 'running' WHERE id = %s", (str(output_dir), job_id))
            conn.commit()

    try:
        import pipeline.main as pipeline
        args = pipeline.get_parser().parse_args([
            "--input", str(video_path), "--output", str(output_dir),
            "--slides", str(slides_dir), "--metadata-dir", str(output_dir / "metadata"),
            "--lance-root", str(output_dir / "lancedb"),
        ])
        
        # 파이프라인 실행
        pipeline.run_pipeline(args, progress_callback=on_progress)
        finalize_job(job_id, "done")
        print(f"--- [Worker] Job {job_id} COMPLETED ---")
    except Exception as e:
        finalize_job(job_id, "error", str(e))
        print(f"--- [Worker] Job {job_id} FAILED: {e} ---")

async def worker_loop():
    print("--- [Worker] Loop Started ---")
    while True:
        try:
            with get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT id, input_path FROM jobs WHERE status = 'pending' ORDER BY created_at LIMIT 1")
                    row = cur.fetchone()
            if row:
                await pipeline_process(row[0], row[1])
            await asyncio.sleep(3)
        except Exception as e:
            print(f"--- [Worker] Loop Error: {e} ---")
            await asyncio.sleep(5)

if __name__ == "__main__":
    asyncio.run(worker_loop())
