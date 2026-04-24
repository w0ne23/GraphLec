"""
공통 유틸리티 함수
"""

import os
import subprocess
import time
from pathlib import Path


def resolve_backend_root() -> Path:
    """
    subprocess cwd / PYTHONPATH 기준 프로젝트 루트.

    - 로컬 graphLec: 저장소 루트 (`app/backend/pipeline` 경로)
    - Docker: /app (형제 패키지 `app`, `pipeline` — 위와 겹치지 않음)

    로컬 `.../app/backend` 는 `pipeline/` 과 `app/main.py` 를 동시에 가지므로,
    monorepo 판별을 Docker 평면 레이아웃보다 먼저 수행한다.
    """
    env_root = os.getenv("GRAPHLEC_ROOT") or os.getenv("PIPELINE_ROOT")
    if env_root:
        return Path(env_root).resolve()
    here = Path(__file__).resolve().parent  # pipeline/
    chain = [here.parent, *here.parents]
    for p in chain:
        if (p / "app" / "backend" / "pipeline").is_dir():
            return p
    for p in chain:
        if (p / "pipeline").is_dir() and (p / "app" / "main.py").exists():
            return p
    return here.parent


def resolve_pipeline_package_root() -> Path:
    """`pipeline` 패키지가 있는 디렉터리 (`python -m pipeline...` 실행 시 cwd)."""
    root = resolve_backend_root()
    nested = root / "app" / "backend"
    if (nested / "pipeline").is_dir():
        return nested
    return root


def api_call_with_retry(func, max_retries=5, initial_wait=10):
    """API 호출 재시도 (429, 503, 500 에러 처리)"""
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            error_msg = str(e)
            retry_errors = ["429", "503", "500", "RESOURCE_EXHAUSTED", "UNAVAILABLE", "overloaded"]
            if any(code in error_msg for code in retry_errors):
                print(f"API ERROR: {error_msg}")
                time.sleep(initial_wait * (attempt + 1))
            else:
                raise e
    raise Exception("API 호출 실패")


def is_retryable_api_error(error) -> bool:
    error_msg = str(error)
    retry_errors = ["429", "503", "500", "RESOURCE_EXHAUSTED", "UNAVAILABLE", "overloaded"]
    return any(code in error_msg for code in retry_errors)


def get_video_duration(file_path: str) -> float:
    """영상 길이 추출 (초)"""
    result = subprocess.run([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        file_path
    ], capture_output=True, text=True)
    return float(result.stdout.strip())
