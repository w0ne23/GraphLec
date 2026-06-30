#!/usr/bin/env python3
"""Run GraphLec QnA over a JSONL evaluation set."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import httpx


SCRIPT_PATH = Path(__file__).resolve()


def find_backend_root() -> Path:
    """Return backend root for both host and Docker layouts.

    Host:   <repo>/app/backend/scripts/eval/run_qna_eval.py
    Docker: /app/scripts/eval/run_qna_eval.py
    """
    for parent in [SCRIPT_PATH.parent, *SCRIPT_PATH.parents]:
        if (parent / "app").is_dir() and (parent / "pipeline").is_dir():
            return parent
    return Path.cwd()


BACKEND_ROOT = find_backend_root()
REPO_ROOT = BACKEND_ROOT.parents[1] if BACKEND_ROOT.name == "backend" and len(BACKEND_ROOT.parents) >= 2 else BACKEND_ROOT


def default_output_dir(stem: str) -> Path:
    for path in [
        Path("/pipeline") / "local_storage" / "results" / stem,
        REPO_ROOT / "local_storage" / "results" / stem,
        REPO_ROOT / "output" / stem,
    ]:
        if path.exists():
            return path
    return REPO_ROOT / "local_storage" / "results" / stem


def ensure_neo4j_loaded(stem: str, output_dir: Path) -> dict[str, Any]:
    if str(BACKEND_ROOT) not in sys.path:
        sys.path.insert(0, str(BACKEND_ROOT))
    try:
        from dotenv import load_dotenv

        load_dotenv(REPO_ROOT / ".env")
        load_dotenv(BACKEND_ROOT / ".env")
    except Exception:
        pass
    try:
        from app.services.neo4j_service import _ensure_stem_loaded
    except Exception as exc:
        raise RuntimeError(
            "Neo4j 적재 함수를 불러오지 못했습니다. "
            "프로젝트 루트에서 실행 중인지, app/backend 의존성이 설치되어 있는지 확인하세요."
        ) from exc
    return _ensure_stem_loaded(stem, str(output_dir))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def wait_for_service(client: httpx.Client, base_url: str, timeout_sec: float = 60.0) -> None:
    deadline = time.time() + timeout_sec
    health_url = urljoin(base_url.rstrip("/") + "/", "health")
    last_error = ""
    while time.time() < deadline:
        try:
            resp = client.get(health_url)
            if resp.status_code == 200:
                return
            last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
        except Exception as exc:
            last_error = repr(exc)
        time.sleep(1)
    raise SystemExit(f"Service is not ready: {health_url} ({last_error})")


def graph_context_strings(core_graph: dict[str, Any]) -> list[str]:
    contexts: list[str] = []
    nodes = core_graph.get("nodes") or []
    edges = core_graph.get("edges") or []
    for node in nodes[:12]:
        label = node.get("label") or node.get("name") or node.get("id") or ""
        desc = node.get("description") or node.get("summary") or node.get("text") or ""
        if label or desc:
            contexts.append(f"그래프 노드: {label} {desc}".strip())
    for edge in edges[:12]:
        src = edge.get("source") or edge.get("src") or edge.get("from") or ""
        rel = edge.get("label") or edge.get("type") or edge.get("relationship") or ""
        tgt = edge.get("target") or edge.get("tgt") or edge.get("to") or ""
        evidence = edge.get("evidence") or edge.get("description") or ""
        text = " ".join(str(x) for x in [src, rel, tgt, evidence] if x)
        if text:
            contexts.append(f"그래프 관계: {text}")
    return contexts


def response_contexts(qr: dict[str, Any]) -> list[str]:
    prompt_contexts = qr.get("prompt_contexts") or []
    if prompt_contexts:
        contexts = clean_contexts(prompt_contexts)
        if contexts:
            return contexts

    contexts: list[str] = []
    for chunk in qr.get("retrieved_chunks") or []:
        text = (chunk.get("text") or "").strip()
        if not text:
            continue
        slide = chunk.get("slide_number")
        start = chunk.get("start_sec")
        end = chunk.get("end_sec")
        prefix_parts = []
        if slide is not None:
            prefix_parts.append(f"슬라이드 {slide}")
        if start is not None or end is not None:
            prefix_parts.append(f"{start}~{end}초")
        prefix = " / ".join(prefix_parts)
        contexts.append(f"{prefix}: {text}" if prefix else text)
    for slide in qr.get("related_slides") or []:
        text = (
            slide.get("summary")
            or slide.get("slide_summary")
            or slide.get("title")
            or slide.get("text")
            or ""
        )
        if text:
            number = slide.get("slide_number")
            contexts.append(f"관련 슬라이드 {number}: {text}" if number is not None else str(text))
    contexts.extend(graph_context_strings(qr.get("core_graph") or {}))
    seen: set[str] = set()
    unique: list[str] = []
    for ctx in contexts:
        ctx = str(ctx).strip()
        if ctx and ctx not in seen:
            unique.append(ctx)
            seen.add(ctx)
    return unique


def clean_contexts(contexts: list[Any]) -> list[str]:
    out: list[str] = []
    for ctx in contexts:
        text = str(ctx).strip()
        if text:
            out.append(text)
    return out


def run_one(client: httpx.Client, query_url: str, stem: str, row: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "stem": stem,
        "question": row["question"],
        "conversation_history": [],
    }
    if row.get("current_slide_number") is not None:
        payload["current_slide_number"] = row["current_slide_number"]
    if row.get("current_scene_number") is not None:
        payload["current_scene_number"] = row["current_scene_number"]

    started = time.time()
    try:
        resp = client.post(f"{query_url.rstrip('/')}/internal/query", json=payload)
        latency = time.time() - started
        result: dict[str, Any] = {
            **row,
            "latency_sec": round(latency, 3),
            "http_status": resp.status_code,
        }
        if resp.status_code == 200:
            qr = resp.json()
            contexts = response_contexts(qr)
            result.update(
                {
                    "response": qr.get("answer", ""),
                    "retrieved_contexts": contexts,
                    "retrieved_chunks": qr.get("retrieved_chunks", []),
                    "prompt_contexts": clean_contexts(qr.get("prompt_contexts", [])),
                    "related_slides": qr.get("related_slides", []),
                    "core_graph": qr.get("core_graph", {"nodes": [], "edges": []}),
                    "timestamps": qr.get("timestamps", []),
                    "source_mode": qr.get("source_mode", "default"),
                    "error": "",
                }
            )
        else:
            result.update(
                {
                    "response": "",
                    "retrieved_contexts": [],
                    "retrieved_chunks": [],
                    "prompt_contexts": [],
                    "related_slides": [],
                    "core_graph": {"nodes": [], "edges": []},
                    "timestamps": [],
                    "source_mode": "",
                    "error": resp.text,
                }
            )
        return result
    except Exception as exc:
        return {
            **row,
            "latency_sec": round(time.time() - started, 3),
            "http_status": None,
            "response": "",
            "retrieved_contexts": [],
            "retrieved_chunks": [],
            "prompt_contexts": [],
            "related_slides": [],
            "core_graph": {"nodes": [], "edges": []},
            "timestamps": [],
            "source_mode": "",
            "error": repr(exc),
        }


def run_one_backend(client: httpx.Client, backend_url: str, lecture_id: str, row: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "question": row["question"],
        "chat_session_id": f"eval-{row['id']}",
    }
    if row.get("current_slide_number") is not None:
        payload["current_slide_number"] = row["current_slide_number"]
    if row.get("current_scene_number") is not None:
        payload["current_scene_number"] = row["current_scene_number"]

    started = time.time()
    try:
        resp = client.post(f"{backend_url.rstrip('/')}/results/{lecture_id}/query", json=payload)
        latency = time.time() - started
        result: dict[str, Any] = {
            **row,
            "latency_sec": round(latency, 3),
            "http_status": resp.status_code,
        }
        if resp.status_code == 200:
            qr = resp.json()
            contexts = response_contexts(qr)
            result.update(
                {
                    "response": qr.get("answer", ""),
                    "retrieved_contexts": contexts,
                    "retrieved_chunks": qr.get("retrieved_chunks", []),
                    "prompt_contexts": clean_contexts(qr.get("prompt_contexts", [])),
                    "related_slides": qr.get("related_slides", []),
                    "core_graph": qr.get("core_graph", {"nodes": [], "edges": []}),
                    "timestamps": qr.get("timestamps", []),
                    "source_mode": qr.get("source_mode", "default"),
                    "query_type": qr.get("query_type", {}),
                    "error": "",
                }
            )
        else:
            result.update(
                {
                    "response": "",
                    "retrieved_contexts": [],
                    "retrieved_chunks": [],
                    "prompt_contexts": [],
                    "related_slides": [],
                    "core_graph": {"nodes": [], "edges": []},
                    "timestamps": [],
                    "source_mode": "",
                    "query_type": {},
                    "error": resp.text,
                }
            )
        return result
    except Exception as exc:
        return {
            **row,
            "latency_sec": round(time.time() - started, 3),
            "http_status": None,
            "response": "",
            "retrieved_contexts": [],
            "retrieved_chunks": [],
            "prompt_contexts": [],
            "related_slides": [],
            "core_graph": {"nodes": [], "edges": []},
            "timestamps": [],
            "source_mode": "",
            "query_type": {},
            "error": repr(exc),
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--stem", required=True)
    parser.add_argument("--query-url", default="http://localhost:8001")
    parser.add_argument(
        "--backend-url",
        help=(
            "Call the Django/FastAPI backend instead of query_service directly. "
            "Recommended because backend loads the lecture graph before QnA."
        ),
    )
    parser.add_argument("--type", choices=["all", "content", "structural"], default="all")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--timeout-sec", type=float, default=180.0)
    parser.add_argument("--ready-timeout-sec", type=float, default=60.0)
    parser.add_argument("--skip-ready-check", action="store_true")
    parser.add_argument(
        "--ensure-neo4j",
        action="store_true",
        help="Load the lecture graph into Neo4j before calling query_service.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Pipeline output directory. Defaults to local_storage/results/<stem> or output/<stem>.",
    )
    args = parser.parse_args()

    if args.ensure_neo4j and args.backend_url:
        print(
            "[preflight] --backend-url was provided; backend /results/{lecture_id}/query "
            "will load Neo4j automatically. Skipping local --ensure-neo4j.",
            file=sys.stderr,
        )
    elif args.ensure_neo4j:
        output_dir = args.output_dir or default_output_dir(args.stem)
        print(f"[preflight] loading Neo4j graph stem={args.stem} output_dir={output_dir}", file=sys.stderr)
        load_info = ensure_neo4j_loaded(args.stem, output_dir)
        print(f"[preflight] Neo4j loaded: {json.dumps(load_info, ensure_ascii=False)}", file=sys.stderr)

    rows = read_jsonl(args.dataset)
    if args.type != "all":
        rows = [row for row in rows if row.get("type") == args.type]
    if args.limit > 0:
        rows = rows[: args.limit]

    results: list[dict[str, Any]] = []
    with httpx.Client(timeout=args.timeout_sec) as client:
        if not args.skip_ready_check:
            wait_for_service(
                client,
                args.backend_url or args.query_url,
                timeout_sec=args.ready_timeout_sec,
            )
        for idx, row in enumerate(rows, start=1):
            print(f"[{idx}/{len(rows)}] {row['id']} {row['question']}", file=sys.stderr)
            if args.backend_url:
                result = run_one_backend(client, args.backend_url, args.stem, row)
            else:
                result = run_one(client, args.query_url, args.stem, row)
            results.append(result)
            write_jsonl(args.out, results)

    print(f"saved: {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
