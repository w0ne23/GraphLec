import argparse
import hashlib
import json
import sys
from pathlib import Path

import lancedb

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

try:
    from recommender.build_index import TABLE_NAME, build_index, load_metadata
except ModuleNotFoundError:
    from build_index import TABLE_NAME, build_index, load_metadata


def _manifest_path(db_dir: str) -> Path:
    return Path(db_dir) / f"{TABLE_NAME}_manifest.json"


def _metadata_fingerprint(metas: list[dict]) -> str:
    payload = []
    for meta in metas:
        payload.append({
            "video_id": meta.get("video_id", ""),
            "title": meta.get("title", ""),
            "domain": meta.get("domain", "unknown"),
            "summary": meta.get("summary", ""),
            "keywords": meta.get("keywords", []),
        })
    payload.sort(key=lambda item: item["video_id"])
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _read_manifest(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        with path.open(encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _write_manifest(path: Path, metas: list[dict], fingerprint: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "fingerprint": fingerprint,
        "video_ids": sorted(str(meta.get("video_id", "")) for meta in metas),
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _table_matches_metadata(table, metas: list[dict], fingerprint: str, manifest: dict | None) -> bool:
    df = table.to_pandas()
    expected_ids = sorted(str(meta.get("video_id", "")) for meta in metas)
    table_ids = sorted(str(video_id) for video_id in df.get("video_id", []))
    if table_ids != expected_ids:
        return False

    if manifest and manifest.get("fingerprint") == fingerprint:
        return True

    by_id = {str(meta.get("video_id", "")): meta for meta in metas}
    for _, row in df.iterrows():
        video_id = str(row.get("video_id", ""))
        meta = by_id.get(video_id)
        if not meta:
            return False
        if str(row.get("title", "")) != str(meta.get("title", "")):
            return False
        if str(row.get("domain", "unknown")) != str(meta.get("domain", "unknown")):
            return False
    return True


def ensure_index(metadata_dir: str, db_dir: str) -> None:
    metas = load_metadata(Path(metadata_dir))
    if not metas:
        print("[recommender] metadata가 없어 인덱스 생성을 건너뜁니다.")
        return

    fingerprint = _metadata_fingerprint(metas)
    manifest_path = _manifest_path(db_dir)
    manifest = _read_manifest(manifest_path)

    db = lancedb.connect(db_dir)
    if TABLE_NAME in db.table_names():
        table = db.open_table(TABLE_NAME)
        if table.count_rows() > 0 and _table_matches_metadata(table, metas, fingerprint, manifest):
            _write_manifest(manifest_path, metas, fingerprint)
            print(f"[recommender] metadata와 일치하는 기존 인덱스 사용: {db_dir}/{TABLE_NAME} ({table.count_rows()}개)")
            return
        print("[recommender] metadata 변경 감지, 추천 인덱스를 재생성합니다.")

    build_index(metadata_dir, db_dir)
    _write_manifest(manifest_path, metas, fingerprint)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ensure recommender LanceDB index exists")
    parser.add_argument("--metadata_dir", required=True)
    parser.add_argument("--db_dir", required=True)
    args = parser.parse_args()
    ensure_index(args.metadata_dir, args.db_dir)
