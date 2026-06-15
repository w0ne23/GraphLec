"""Load lecture_metadata from recommender metadata artifacts for dev use.

This utility keeps recommender serving DB-only. It reads existing metadata
artifacts, verifies that each lecture exists, and delegates the actual upsert
mapping to pipeline.metadata_db.
"""

from __future__ import annotations

import argparse
import os
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


SCRIPT_PATH = Path(__file__).resolve()
BACKEND_ROOT = SCRIPT_PATH.parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

REPO_ROOT = SCRIPT_PATH.parents[3] if len(SCRIPT_PATH.parents) > 3 else BACKEND_ROOT

@dataclass(frozen=True)
class MetadataCandidate:
    path: Path
    source: str


@dataclass(frozen=True)
class PreparedMetadata:
    lecture_id: uuid.UUID
    metadata: dict[str, Any]


@dataclass
class Counters:
    inserted: int = 0
    updated: int = 0
    orphan: int = 0
    invalid: int = 0
    skipped_duplicate: int = 0
    failed: int = 0


def _database_url_sync() -> str:
    url = os.getenv("DATABASE_URL", "").strip()
    return url.replace("+asyncpg", "") if url else ""


def _metadata_helpers():
    from pipeline.metadata_db import load_metadata_json, upsert_lecture_metadata_sync

    return load_metadata_json, upsert_lecture_metadata_sync


def _resolve_metadata_dir(cli_value: str | None) -> Path | None:
    if cli_value:
        cli_path = Path(cli_value).expanduser()
        if not cli_path.is_dir():
            raise FileNotFoundError(f"metadata dir not found: {cli_path}")
        return cli_path

    candidates: list[Path] = []
    env_dir = os.getenv("GRAPHLEC_METADATA_DIR", "").strip()
    if env_dir:
        candidates.append(Path(env_dir))
    candidates.extend(
        [
            Path("/app/metadata"),
            BACKEND_ROOT / "metadata",
            REPO_ROOT / "app" / "backend" / "metadata",
        ]
    )

    seen: set[Path] = set()
    for candidate in candidates:
        path = candidate.expanduser()
        resolved = path.resolve() if path.exists() else path
        if resolved in seen:
            continue
        seen.add(resolved)
        if path.is_dir():
            return path
    return None


def _iter_metadata_dir(metadata_dir: Path | None) -> Iterable[MetadataCandidate]:
    if metadata_dir is None:
        return []
    return (
        MetadataCandidate(path=path, source="metadata-dir")
        for path in sorted(metadata_dir.glob("*_metadata.json"))
        if path.is_file()
    )


def _iter_results_dir(results_dir: Path | None) -> Iterable[MetadataCandidate]:
    if results_dir is None:
        return []
    return (
        MetadataCandidate(path=path, source="results-dir")
        for path in sorted(results_dir.glob("*/metadata/*_metadata.json"))
        if path.is_file()
    )


def _candidate_paths(metadata_dir: Path | None, results_dir: Path | None) -> list[MetadataCandidate]:
    return [*_iter_metadata_dir(metadata_dir), *_iter_results_dir(results_dir)]


def _prepare_metadata(path: Path, load_metadata_json) -> tuple[PreparedMetadata | None, str | None]:
    try:
        metadata = load_metadata_json(path)
    except Exception as exc:
        return None, f"load_error: {exc}"

    raw_video_id = str(metadata.get("video_id") or "").strip()
    if not raw_video_id:
        return None, "missing video_id"

    try:
        return PreparedMetadata(lecture_id=uuid.UUID(raw_video_id), metadata=metadata), None
    except (TypeError, ValueError):
        return None, f"invalid UUID: {raw_video_id}"


def _lecture_exists(cur, lecture_id: uuid.UUID) -> bool:
    cur.execute("SELECT 1 FROM lectures WHERE id = %s LIMIT 1", (str(lecture_id),))
    return cur.fetchone() is not None


def _metadata_exists(cur, lecture_id: uuid.UUID) -> bool:
    cur.execute("SELECT 1 FROM lecture_metadata WHERE lecture_id = %s LIMIT 1", (str(lecture_id),))
    return cur.fetchone() is not None


def _print_candidate(action: str, lecture_id: uuid.UUID | None, candidate: MetadataCandidate, detail: str = "") -> None:
    prefix = f"[{action}]"
    suffix = f" ({detail})" if detail else ""
    if lecture_id is None:
        print(f"{prefix} {candidate.path}{suffix}")
    else:
        print(f"{prefix} {lecture_id} <- {candidate.path}{suffix}")


def load_dev_metadata(
    *,
    metadata_dir: Path | None,
    results_dir: Path | None,
    dry_run: bool,
) -> Counters:
    import psycopg2

    load_metadata_json, upsert_lecture_metadata_sync = _metadata_helpers()
    database_url = _database_url_sync()
    if not database_url:
        raise RuntimeError("DATABASE_URL is not set")

    candidates = _candidate_paths(metadata_dir, results_dir)
    print(f"[scan] metadata_dir={metadata_dir or '(not found)'}")
    if results_dir:
        print(f"[scan] results_dir={results_dir}")
    print(f"[scan] candidates={len(candidates)}")

    counters = Counters()
    seen: set[uuid.UUID] = set()

    conn = psycopg2.connect(database_url)
    try:
        with conn:
            with conn.cursor() as cur:
                for candidate in candidates:
                    prepared, invalid_reason = _prepare_metadata(candidate.path, load_metadata_json)
                    if prepared is None:
                        counters.invalid += 1
                        _print_candidate("invalid", None, candidate, invalid_reason or "")
                        continue

                    lecture_id = prepared.lecture_id
                    if lecture_id in seen:
                        counters.skipped_duplicate += 1
                        _print_candidate("duplicate", lecture_id, candidate, candidate.source)
                        continue
                    seen.add(lecture_id)

                    if not _lecture_exists(cur, lecture_id):
                        counters.orphan += 1
                        _print_candidate("orphan", lecture_id, candidate)
                        continue

                    will_update = _metadata_exists(cur, lecture_id)
                    action = "updated" if will_update else "inserted"
                    if dry_run:
                        counters.updated += int(will_update)
                        counters.inserted += int(not will_update)
                        _print_candidate(f"would_{action}", lecture_id, candidate)
                        continue

                    try:
                        ok = upsert_lecture_metadata_sync(
                            lecture_id=str(lecture_id),
                            metadata=prepared.metadata,
                            metadata_uri=candidate.path.name,
                        )
                    except Exception as exc:
                        counters.failed += 1
                        _print_candidate("failed", lecture_id, candidate, str(exc))
                        continue

                    if not ok:
                        counters.failed += 1
                        _print_candidate("failed", lecture_id, candidate, "upsert returned false")
                        continue

                    counters.updated += int(will_update)
                    counters.inserted += int(not will_update)
                    _print_candidate(action, lecture_id, candidate)
    finally:
        conn.close()

    return counters


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "dev 환경에서 metadata artifact를 lecture_metadata DB에 로드하는 도구. "
            "운영 환경에서는 pipeline G6이 자동으로 upsert하므로 불필요. "
            "예시: docker compose exec backend python scripts/load_dev_metadata.py; "
            "docker compose exec backend python scripts/load_dev_metadata.py --dry-run; "
            "docker compose exec backend python scripts/load_dev_metadata.py "
            "--results-dir /pipeline/local_storage/results"
        )
    )
    parser.add_argument("--metadata-dir", help="metadata JSON 디렉토리")
    parser.add_argument("--results-dir", help="results 디렉토리. */metadata/*_metadata.json만 스캔")
    parser.add_argument("--dry-run", action="store_true", help="DB 변경 없이 예상 결과만 출력")
    parser.add_argument("--database-url", help="DATABASE_URL 오버라이드")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    if args.database_url:
        os.environ["DATABASE_URL"] = args.database_url

    metadata_dir = _resolve_metadata_dir(args.metadata_dir)
    results_dir = Path(args.results_dir).expanduser() if args.results_dir else None
    if results_dir is not None and not results_dir.is_dir():
        print(f"[error] results dir not found: {results_dir}", file=sys.stderr)
        return 2

    try:
        counters = load_dev_metadata(
            metadata_dir=metadata_dir,
            results_dir=results_dir,
            dry_run=args.dry_run,
        )
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    print(
        "[summary] "
        f"inserted={counters.inserted} "
        f"updated={counters.updated} "
        f"orphan={counters.orphan} "
        f"invalid={counters.invalid} "
        f"skipped_duplicate={counters.skipped_duplicate} "
        f"failed={counters.failed}"
    )
    return 1 if counters.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
