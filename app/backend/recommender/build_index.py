"""
build_index.py
──────────────
강의 메타데이터 → LanceDB 벡터 인덱스 구축

임베딩 모델: gemini-embedding-001 (Google Gemini API)
필드별 벡터를 분리 저장:
  title_vec   : 강의 제목
  keyword_vec : 키워드 전체 concat
  summary_vec : 강의 요약

실행:
  python build_index.py
  python build_index.py --metadata_dir app/backend/metadata --db_dir data/lancedb

파이프라인 위치:
  import_graph.py → generate_metadata.py → build_index.py
"""

import json
import os
import time
import argparse
from pathlib import Path

from google import genai
from dotenv import load_dotenv
import lancedb

load_dotenv()
_client    = genai.Client(api_key=os.getenv("GOOGLE_API_KEY_2"))
MODEL_NAME = "gemini-embedding-001"
TABLE_NAME = "lectures"


def _resolve_repo_root() -> Path:
    """Resolve project root for both local and container runs."""
    env_root = os.getenv("GRAPHLEC_ROOT") or os.getenv("PIPELINE_ROOT")
    if env_root:
        return Path(env_root).resolve()
    here = Path(__file__).resolve()
    return here.parents[3] if len(here.parents) > 3 else here.parents[1]


_REPO_ROOT = _resolve_repo_root()
DEFAULT_METADATA_DIR = str(_REPO_ROOT / "app" / "backend" / "metadata")
DEFAULT_DB_DIR = str(_REPO_ROOT / "data" / "lancedb")


def embed_texts(texts: list[str], batch_size: int = 20) -> list[list[float]]:
    """
    Gemini embedding-001로 텍스트 배치 임베딩.

    API rate limit 대응:
      - batch_size 단위로 나눠서 요청
      - 배치 사이 0.5초 대기
      - 빈 문자열은 공백으로 대체 (API 오류 방지)
    """
    results = []
    for i in range(0, len(texts), batch_size):
        batch = [t if t.strip() else " " for t in texts[i:i + batch_size]]
        print(f"  임베딩 중... {i+1}~{min(i+len(batch), len(texts))} / {len(texts)}")
        response = _client.models.embed_content(
            model    = MODEL_NAME,
            contents = batch,
        )
        results.extend([e.values for e in response.embeddings])
        if i + batch_size < len(texts):
            time.sleep(0.5)
    return results


def load_metadata(metadata_dir: Path) -> list[dict]:
    if not metadata_dir.exists():
        print(f"[build_index] 메타데이터 디렉토리 없음, 인덱스 구축 생략: {metadata_dir}")
        return []
    records = []
    files   = list(metadata_dir.glob("*_metadata.json"))
    if not files:
        raise FileNotFoundError(f"메타데이터 파일 없음: {metadata_dir}")
    for path in files:
        with open(path, encoding="utf-8") as f:
            items = json.load(f)
        if isinstance(items, dict):
            items = [items]
        records.extend(items)
    return records


def build_index(metadata_dir: str = DEFAULT_METADATA_DIR, db_dir: str = DEFAULT_DB_DIR):
    print(f"[임베딩 모델] {MODEL_NAME}")
    print(f"[메타데이터 로드] {metadata_dir}")
    metas = load_metadata(Path(metadata_dir))
    if not metas:
        return
    print(f"  → {len(metas)}개 강의 로드\n")

    # ── 필드별 텍스트 분리 ────────────────────────────────────────────
    # title이 비어있으면 summary 앞부분으로 대체
    title_texts   = [
        m.get("title") or m.get("summary", "")[:50]
        for m in metas
    ]
    keyword_texts = [
        " ".join(k["keyword"] for k in m.get("keywords", []))
        for m in metas
    ]
    summary_texts = [m.get("summary", "") for m in metas]

    # ── 필드별 임베딩 생성 ────────────────────────────────────────────
    print("[title 임베딩 생성 중...]")
    title_vecs = embed_texts(title_texts)

    print("\n[keyword 임베딩 생성 중...]")
    keyword_vecs = embed_texts(keyword_texts)

    print("\n[summary 임베딩 생성 중...]")
    summary_vecs = embed_texts(summary_texts)

    # ── LanceDB 저장 ──────────────────────────────────────────────────
    records = []
    for meta, t_vec, k_vec, s_vec in zip(metas, title_vecs, keyword_vecs, summary_vecs):
        records.append({
            "video_id":    meta["video_id"],
            "title":       meta.get("title", ""),
            "domain":      meta.get("domain", "unknown"),
            "title_vec":   t_vec,
            "keyword_vec": k_vec,
            "summary_vec": s_vec,
        })

    print(f"\n[LanceDB 저장] {db_dir}  테이블: {TABLE_NAME}")
    db    = lancedb.connect(db_dir)
    table = db.create_table(TABLE_NAME, data=records, mode="overwrite")
    print(f"  → {table.count_rows()}개 레코드 저장 완료")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="강의 임베딩 인덱스 구축")
    parser.add_argument("--metadata_dir", default=DEFAULT_METADATA_DIR)
    parser.add_argument("--db_dir",       default=DEFAULT_DB_DIR)
    args = parser.parse_args()
    build_index(args.metadata_dir, args.db_dir)