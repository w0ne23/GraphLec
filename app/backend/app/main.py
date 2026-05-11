import asyncio
import os
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Try to load .env from project root (../../.env) if it exists, or from the current directory
root_env = Path(__file__).resolve().parent.parent.parent / ".env"
if root_env.exists():
    load_dotenv(dotenv_path=root_env)
else:
    load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.db import init_db
from app.worker import worker_loop
from app.routers import jobs, results
from app.services.lecture_service import clear_runtime_lecture_graphs

# Cross-platform path handling for local_storage
# main.py is in app/backend/app/ -> 4 levels deep from root (including filename)
PROJECT_ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
PROJECT_ROOT = Path("/pipeline") if Path("/pipeline").exists() else PROJECT_ROOT_DIR
LOCAL_STORAGE_DIR = os.getenv("LOCAL_STORAGE_DIR", str(PROJECT_ROOT / "local_storage"))

# Global worker task registry for monitoring
worker_tasks = []

async def monitor_workers():
    """Periodically logs the number of active workers."""
    while True:
        try:
            await asyncio.sleep(5)
            if worker_tasks:
                active = len([t for t in worker_tasks if not t.done()])
                total = len(worker_tasks)
                logger.info(f"--- [Backend Heartbeat] Active Workers: {active}/{total} ---")
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"--- [Monitor Error]: {e} ---")

@asynccontextmanager
async def lifespan(app: FastAPI):
    global worker_tasks
    logger.info("--- [FastAPI] Starting lifespan events... ---")
    
    # Ensure local storage directory exists
    os.makedirs(LOCAL_STORAGE_DIR, exist_ok=True)
    logger.info(f"--- [FastAPI] Local storage directory ensured: {LOCAL_STORAGE_DIR} ---")
    
    try:
        logger.info("--- [FastAPI] Initializing DB... ---")
        await init_db()
        logger.info("--- [FastAPI] DB initialized successfully. ---")
    except Exception as e:
        logger.error(f"--- [FastAPI] ERROR initializing DB: {e} ---")

    if os.getenv("GRAPHLEC_CLEAR_NEO4J_ON_START", "0").lower() not in {"0", "false", "no"}:
        cleanup = clear_runtime_lecture_graphs()
        logger.info(
            f"--- [FastAPI] Neo4j runtime graph cleanup: before={cleanup['before']} after={cleanup['after']} ---"
        )
    
    logger.info("--- [FastAPI] Starting worker loops... ---")
    worker_tasks = [asyncio.create_task(worker_loop()) for _ in range(3)]
    monitor_task = asyncio.create_task(monitor_workers())
    logger.info(f"--- [FastAPI] {len(worker_tasks)} worker tasks and monitor created. ---")
    
    yield
    
    logger.info("--- [FastAPI] Shutting down... Cancelling workers. ---")
    monitor_task.cancel()
    for task in worker_tasks:
        task.cancel()
    await asyncio.gather(monitor_task, *worker_tasks, return_exceptions=True)
    logger.info("--- [FastAPI] Shutdown complete. ---")

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite 기본 포트
        "http://127.0.0.1:5173",
        "http://localhost:8000",  # 백엔드 자체
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount local storage for serving result files
app.mount("/files", StaticFiles(directory=LOCAL_STORAGE_DIR), name="files")

app.include_router(jobs.router)
app.include_router(results.router)

@app.get("/health")
def health_check():
    return {"status": "ok"}