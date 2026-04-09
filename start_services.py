"""
GraphLEC 서버(3개)를 한 번에 실행하는 전용 스크립트.

- query_service: FastAPI (8001)
- Django web: (8000)
- recommender_web: (8002)

메인 파이프라인 `main.py`에서 서버 실행 코드를 분리하기 위한 용도입니다.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


def _start(cmd: list[str], *, cwd: str) -> subprocess.Popen:
    # stdout/stderr를 리다이렉트하지 않아서, parent 터미널에 로그가 같이 출력되게 둔다.
    return subprocess.Popen(cmd, cwd=cwd)


def main() -> None:
    parser = argparse.ArgumentParser(description="GraphLEC 서비스 3개 동시 시작")
    parser.add_argument("--metadata-dir", default="metadata", help="추천 서비스/메타데이터 디렉토리")
    parser.add_argument("--host", default="127.0.0.1", help="FastAPI host (query_service)")
    parser.add_argument("--query-port", type=int, default=8001, help="query_service port")
    parser.add_argument("--django-port", type=int, default=8000, help="Django port")
    parser.add_argument("--reco-port", type=int, default=8002, help="recommender_web port")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    web_dir = repo_root / "web"

    cmds: dict[str, tuple[list[str], str]] = {
        "query_service (8001)": (
            [
                sys.executable,
                "-m",
                "uvicorn",
                "query_service.main:app",
                "--host",
                args.host,
                "--port",
                str(args.query_port),
            ],
            str(repo_root),
        ),
        "recommender_web (8002)": (
            [
                sys.executable,
                str(repo_root / "recommender_web.py"),
                "--metadata_dir",
                args.metadata_dir,
                "--port",
                str(args.reco_port),
            ],
            str(repo_root),
        ),
        "django (8000)": (
            [
                sys.executable,
                "manage.py",
                "runserver",
                str(args.django_port),
            ],
            str(web_dir),
        ),
    }

    print("서비스 시작 — query_service + Django + recommender_web (병렬)")
    procs: dict[str, subprocess.Popen] = {}
    for name, (cmd, cwd) in cmds.items():
        proc = _start(cmd, cwd=cwd)
        procs[name] = proc
        print(f"▶ {name} PID {proc.pid}")

    print()
    print(f"강의 질의 → http://{args.host}:{args.django_port}/")
    print(f"추천 서비스 → http://{args.host}:{args.reco_port}/")
    print("\n종료: Ctrl+C")
    print("-" * 70)

    try:
        while True:
            time.sleep(2)
            dead = [n for n, p in procs.items() if p.poll() is not None]
            if dead:
                print(f"\n⚠️ 비정상 종료: {', '.join(dead)}")
                break
    except KeyboardInterrupt:
        print("\n서비스 종료 중...")
    finally:
        for p in procs.values():
            try:
                p.terminate()
            except Exception:
                pass


if __name__ == "__main__":
    main()

