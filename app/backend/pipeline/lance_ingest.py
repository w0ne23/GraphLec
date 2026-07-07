"""
fused.json → 청크 생성 → Gemini 임베딩 → Parquet 백업 + LanceDB 적재.

단일 LanceDB 테이블 `chunks` 에 stem 컬럼으로 강의를 구분한다.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import lancedb
import numpy as np
import pandas as pd

from .embedding_utils import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_PROVIDER,
    default_embedding_model,
    embed_documents,
    embed_query,
    get_embedding_client,
    normalize_embedding_provider,
)
from .utils import resolve_backend_root

CHUNKS_TABLE = "chunks"
BATCH_SIZE = 16


def safe_index_label(label: Optional[str]) -> Optional[str]:
    if not label:
        return None
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in label.strip().lower())
    cleaned = "_".join(part for part in cleaned.split("_") if part)
    return cleaned or None


def _format_visual_assets(slide: Dict[str, Any]) -> str:
    items: List[str] = []
    for idx, asset in enumerate(slide.get("visual_assets") or [], start=1):
        if isinstance(asset, str):
            asset = {"description": asset}
        if not isinstance(asset, dict):
            continue
        asset_type = str(asset.get("asset_type") or asset.get("type") or "visual").strip()
        title = str(asset.get("title") or "").strip()
        desc = str(asset.get("description") or "").strip()
        raw_text = str(asset.get("raw_text") or asset.get("text") or "").strip()
        elements = str(asset.get("visual_elements_text") or "").strip()
        relations = str(asset.get("visual_relations_text") or "").strip()
        layout = str(asset.get("layout_text") or "").strip()
        body = "\n".join(
            part for part in [
                desc,
                raw_text,
                f"시각 요소:\n{elements}" if elements else "",
                f"시각 관계:\n{relations}" if relations else "",
                f"배치:\n{layout}" if layout else "",
            ]
            if part
        )
        if not (title or body):
            continue
        label = f"{idx}. {asset_type}"
        if title:
            label += f" - {title}"
        items.append(f"{label}\n{body}".strip())

    if items:
        return "\n\n".join(items)
    return str(slide.get("t1_structure") or "").strip()


def default_lance_root() -> Path:
    env = os.getenv("GRAPHLEC_LANCE_ROOT")
    if env:
        return Path(env).resolve()
    # 레포 루트 기준 (CWD와 무관)
    repo_root = resolve_backend_root()
    return (repo_root / "data" / "lancedb").resolve()


def build_chunks_from_fused(fused: Dict[str, Any], stem: str) -> List[Dict[str, Any]]:
    """슬라이드 본문 + 세그먼트 전사를 검색용 청크로 만든다."""
    slides = fused.get("scenes") or []
    chunks: List[Dict[str, Any]] = []
    seg_idx = 0

    for slide in slides:
        sid = slide.get("slide_id", "")
        scene_id = slide.get("scene_id") or f"scene/{int(slide.get('scene_number', 0) or 0):04d}"
        sn = slide.get("slide_number")
        title = (slide.get("title") or "").strip()
        stext = (slide.get("slide_text") or "").strip()
        structure = _format_visual_assets(slide)
        body_parts = [f"[{sid}] {title}".strip()]
        if stext:
            body_parts.append("슬라이드 원문:\n" + stext)
        if structure:
            body_parts.append("시각자료 설명:\n" + structure)
        body = "\n".join(part for part in body_parts if part).strip()
        if body:
            chunks.append(
                {
                    "chunk_id": f"{scene_id}_slide_body",
                    "stem": stem,
                    "chunk_type": "slide",
                    "text": body[:12000],
                    "slide_id": sid,
                    "scene_id": scene_id,
                    "slide_number": sn,
                    "start_sec": slide.get("start_sec"),
                    "end_sec": slide.get("end_sec"),
                    "linked_node_id": sid,
                }
            )

        for ctx in slide.get("contexts") or []:
            for seg in ctx.get("segments") or []:
                seg_id = f"segment/{seg_idx:04d}"
                seg_idx += 1
                t = (seg.get("text") or "").strip()
                if not t:
                    continue
                chunks.append(
                    {
                        "chunk_id": seg_id,
                        "stem": stem,
                        "chunk_type": "segment",
                        "text": f"[{seg_id}] [{sid}] {t}"[:12000],
                        "slide_id": sid,
                        "scene_id": scene_id,
                        "slide_number": sn,
                        "start_sec": seg.get("start"),
                        "end_sec": seg.get("end"),
                        "linked_node_id": seg_id,
                    }
                )

    return chunks


def ingest_stem_to_lance(
    *,
    stem: str,
    fused_path: Path,
    output_dir: Path,
    lance_root: Optional[Path] = None,
    embedding_provider: str = DEFAULT_EMBEDDING_PROVIDER,
    embedding_model: Optional[str] = None,
    embedding_dimensions: Optional[int] = None,
    index_label: Optional[str] = None,
) -> Dict[str, Any]:
    """
    fused_path의 강의를 임베딩해 LanceDB에 넣는다.
    동일 stem 기존 행은 삭제 후 재삽입한다.
    """
    embedding_provider = normalize_embedding_provider(embedding_provider)
    embedding_model = embedding_model or default_embedding_model(embedding_provider)
    label = safe_index_label(index_label)
    lance_root = lance_root or default_lance_root()
    lance_root.mkdir(parents=True, exist_ok=True)

    if not fused_path.exists():
        raise FileNotFoundError(f"fused 파일 없음: {fused_path}")

    with open(fused_path, encoding="utf-8") as f:
        fused = json.load(f)

    chunks = build_chunks_from_fused(fused, stem)
    if not chunks:
        return {"ok": False, "reason": "no chunks", "count": 0}

    client = get_embedding_client(embedding_provider)
    texts = [c["text"] for c in chunks]
    vectors: List[List[float]] = []

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        vectors.extend(
            embed_documents(
                client,
                batch,
                model=embedding_model,
                provider=embedding_provider,
                dimensions=embedding_dimensions,
            )
        )

    rows: List[Dict[str, Any]] = []
    for c, vec in zip(chunks, vectors):
        rows.append(
            {
                **c,
                "vector": vec,
                "embedding_provider": embedding_provider,
                "embedding_model": embedding_model,
                "embedding_dimensions": int(embedding_dimensions) if embedding_dimensions else len(vec),
            }
        )

    df = pd.DataFrame(rows)
    suffix = f"_{label}" if label else ""
    parquet_path = output_dir / f"{stem}_chunks_lance{suffix}.parquet"
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet_path, index=False)

    db = lancedb.connect(str(lance_root))
    if CHUNKS_TABLE in db.table_names():
        table = db.open_table(CHUNKS_TABLE)
        stem_sql = stem.replace("'", "''")
        table.delete(f"stem = '{stem_sql}'")
        table.add(df)
    else:
        db.create_table(CHUNKS_TABLE, data=df)

    manifest_path: Optional[Path] = None
    if label:
        manifest_path = output_dir / f"{stem}_lance_{label}_manifest.json"
        manifest = {
            "stem": stem,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "provider": embedding_provider,
            "model": embedding_model,
            "dimensions": int(embedding_dimensions) if embedding_dimensions else (len(vectors[0]) if vectors else None),
            "index_label": label,
            "chunk_count": len(rows),
            "fused_path": str(fused_path),
            "parquet_path": str(parquet_path),
            "lance_root": str(lance_root),
            "table": CHUNKS_TABLE,
        }
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "ok": True,
        "stem": stem,
        "count": len(rows),
        "parquet_path": str(parquet_path),
        "lance_root": str(lance_root),
        "table": CHUNKS_TABLE,
        "embedding_provider": embedding_provider,
        "embedding_model": embedding_model,
        "embedding_dimensions": int(embedding_dimensions) if embedding_dimensions else (len(vectors[0]) if vectors else None),
        "index_label": label,
        "manifest_path": str(manifest_path) if manifest_path else None,
    }


def lance_search(
    *,
    stem: str,
    query: str,
    lance_root: Optional[Path] = None,
    top_k: int = 8,
    embedding_provider: str = DEFAULT_EMBEDDING_PROVIDER,
    embedding_model: Optional[str] = None,
    embedding_dimensions: Optional[int] = None,
) -> pd.DataFrame:
    """질의 벡터로 stem 범위 내 top-k 검색."""
    embedding_provider = normalize_embedding_provider(embedding_provider)
    embedding_model = embedding_model or default_embedding_model(embedding_provider)
    lance_root = lance_root or default_lance_root()
    if not lance_root.exists():
        raise FileNotFoundError(f"LanceDB 경로 없음: {lance_root}")

    db = lancedb.connect(str(lance_root))
    if CHUNKS_TABLE not in db.table_names():
        raise FileNotFoundError(f"LanceDB에 '{CHUNKS_TABLE}' 테이블이 없습니다. 먼저 파이프라인 Stage 7을 실행하세요.")

    client = get_embedding_client(embedding_provider)
    qv = np.array(
        embed_query(
            client,
            query,
            model=embedding_model,
            provider=embedding_provider,
            dimensions=embedding_dimensions,
        ),
        dtype=np.float32,
    )
    table = db.open_table(CHUNKS_TABLE)
    stem_sql = stem.replace("'", "''")
    return (
        table.search(qv)
        .where(f"stem = '{stem_sql}'", prefilter=True)
        .limit(top_k)
        .to_pandas()
    )


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="fused.json → Parquet + LanceDB 적재 (단일 강의)")
    p.add_argument("--stem", required=True, help="강의 stem (파일명과 동일)")
    p.add_argument("--output-dir", default="output", type=Path, help="분석 출력 디렉터리")
    p.add_argument("--slides-dir", default="output_slides", type=Path, help="슬라이드 디렉터리")
    p.add_argument("--lance-root", default=None, type=Path, help="LanceDB 루트 (기본: 레포 data/lancedb)")
    p.add_argument("--embedding-provider", choices=["gemini", "openai"], default=DEFAULT_EMBEDDING_PROVIDER)
    p.add_argument("--embedding-model", default=None, help="임베딩 모델 (기본: provider별 기본값)")
    p.add_argument("--embedding-dimensions", default=None, type=int, help="OpenAI 임베딩 차원 축소 옵션")
    p.add_argument("--index-label", default=None, help="별도 실험 인덱스 label. parquet/manifest suffix로 사용")
    args = p.parse_args()
    try:
        from .config import output_paths
    except ImportError:
        output_paths = None  # type: ignore

    if output_paths:
        paths = output_paths(args.stem, args.output_dir, args.slides_dir)
        fused = paths["fused"]
    else:
        fused = args.output_dir / f"{args.stem}_fused.json"

    lr = args.lance_root
    r = ingest_stem_to_lance(
        stem=args.stem,
        fused_path=fused,
        output_dir=args.output_dir,
        lance_root=lr,
        embedding_provider=args.embedding_provider,
        embedding_model=args.embedding_model,
        embedding_dimensions=args.embedding_dimensions,
        index_label=args.index_label,
    )
    print(r)
