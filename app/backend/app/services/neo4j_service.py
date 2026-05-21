import asyncio
import os
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import HTTPException
from neo4j import GraphDatabase

from pipeline.graphrag_neo4j_ingest import (
    delete_graphrag_layer_tx,
    find_graphrag_output_dir,
    load_graphrag_layer_tx,
)
from pipeline.graphrag_emphasis import (
    compute_keyword_match,
    compute_visual_match,
    compute_annotation_match,
    compute_audio_segment_match,
    compute_final_weight,
    compute_relation_boost,
)
from pipeline.neo4j_ingest import ingest_parquet_to_neo4j

logger = logging.getLogger(__name__)

_RUNTIME_GRAPH_LABELS = {
    "AnnotationEmphasis",
    "ConceptGraph",
    "Context",
    "Domain",
    "GraphRAGCommunity",
    "GraphRAGEntity",
    "GraphRAGTextUnit",
    "Scene",
    "Segment",
    "Slide",
    "VisualAsset",
    "Video",
}

# ── Neo4j ────────────────────────────────────────────────────────────────────
_neo4j_driver = None
_stem_load_locks: Dict[str, asyncio.Lock] = {}
_stem_load_locks_mutex: Optional[asyncio.Lock] = None
_stem_load_locks_loop: Optional[asyncio.AbstractEventLoop] = None


def _get_stem_load_locks_mutex() -> asyncio.Lock:
    """async 컨텍스트에서만 호출할 것. (asyncio.get_running_loop() 사용)"""
    global _stem_load_locks_mutex, _stem_load_locks_loop

    loop = asyncio.get_running_loop()
    if _stem_load_locks_mutex is None or _stem_load_locks_loop is not loop:
        _stem_load_locks.clear()
        _stem_load_locks_mutex = asyncio.Lock()
        _stem_load_locks_loop = loop
    return _stem_load_locks_mutex


async def get_stem_load_lock(stem: str) -> asyncio.Lock:
    """동일 stem의 Neo4j 적재/해제 작업을 직렬화하는 lock을 반환한다."""
    async with _get_stem_load_locks_mutex():
        if stem not in _stem_load_locks:
            _stem_load_locks[stem] = asyncio.Lock()
        return _stem_load_locks[stem]


def get_neo4j_driver():
    """Neo4j driver를 singleton으로 반환한다.

    환경 변수(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)가 설정되지 않은 경우 None을 반환한다.
    """
    global _neo4j_driver
    if _neo4j_driver is not None:
        return _neo4j_driver
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USER")
    pw = os.getenv("NEO4J_PASSWORD")
    if not all([uri, user, pw]):
        return None
    _neo4j_driver = GraphDatabase.driver(uri, auth=(user, pw))
    return _neo4j_driver


@contextmanager
def neo4j_session():
    """싱글톤 Neo4j driver에서 session을 열고 yield한다.

    Neo4j 설정이 없으면 HTTPException 503을 발생시킨다.
    """
    driver = get_neo4j_driver()
    if not driver:
        raise HTTPException(status_code=503, detail="Neo4j 설정이 없습니다.")
    with driver.session() as session:
        yield session


def _graphrag_layer_counts(session, stem: str) -> Dict[str, int]:
    record = session.run(
        """
        CALL {
            MATCH (e:GraphRAGEntity {stem: $stem})
            RETURN count(e) AS entities
        }
        CALL {
            MATCH (:GraphRAGEntity {stem: $stem})-[r:GRAPHRAG_RELATES_TO]->(:GraphRAGEntity {stem: $stem})
            RETURN count(r) AS relationships
        }
        CALL {
            MATCH (:GraphRAGEntity {stem: $stem})-[s:GRAPHRAG_APPEARS_IN]->(:Slide {stem: $stem})
            RETURN count(s) AS slide_links
        }
        CALL {
            MATCH (:GraphRAGEntity {stem: $stem})-[s:GRAPHRAG_APPEARS_IN_SCENE]->(:Scene {stem: $stem})
            RETURN count(s) AS scene_links
        }
        CALL {
            MATCH (tu:GraphRAGTextUnit {stem: $stem})
            RETURN count(tu) AS text_units
        }
        CALL {
            MATCH (c:GraphRAGCommunity {stem: $stem})
            RETURN count(c) AS communities
        }
        RETURN entities, relationships, slide_links, scene_links, text_units, communities
        """,
        stem=stem,
    ).single()
    if not record:
        return {
            "entities": 0,
            "relationships": 0,
            "slide_links": 0,
            "scene_links": 0,
            "text_units": 0,
            "communities": 0,
        }
    return {
        "entities": int(record["entities"]),
        "relationships": int(record["relationships"]),
        "slide_links": int(record["slide_links"]),
        "scene_links": int(record["scene_links"]),
        "text_units": int(record["text_units"]),
        "communities": int(record["communities"]),
    }


def _stem_graph_counts(session, stem: str) -> Dict[str, int]:
    record = session.run(
        """
        CALL {
            MATCH (n {stem: $stem})
            RETURN count(n) AS nodes
        }
        CALL {
            MATCH (a {stem: $stem})-[r]->(b {stem: $stem})
            RETURN count(r) AS relationships
        }
        CALL {
            MATCH (e:GraphRAGEntity {stem: $stem})
            RETURN count(e) AS concepts
        }
        RETURN nodes, relationships, concepts
        """,
        stem=stem,
    ).single()
    if not record:
        return {"nodes": 0, "relationships": 0, "concepts": 0}
    return {
        "nodes": int(record["nodes"]),
        "relationships": int(record["relationships"]),
        "concepts": int(record["concepts"]),
    }


def _delete_stem_graph_tx(tx, stem: str) -> None:
    tx.run("MATCH (n {stem: $stem}) DETACH DELETE n", stem=stem)


def clear_runtime_lecture_graphs() -> Dict[str, int]:
    """앱 시작 시 Neo4j에 남아 있는 강의 런타임 그래프를 비운다."""
    try:
        with neo4j_session() as session:
            before_record = session.run(
                """
                MATCH (n)
                WHERE n.stem IS NOT NULL OR any(label IN labels(n) WHERE label IN $labels)
                RETURN count(n) AS count
                """,
                labels=sorted(_RUNTIME_GRAPH_LABELS),
            ).single()
            before = int(before_record["count"]) if before_record else 0
            session.run(
                """
                MATCH (n)
                WHERE n.stem IS NOT NULL OR any(label IN labels(n) WHERE label IN $labels)
                DETACH DELETE n
                """,
                labels=sorted(_RUNTIME_GRAPH_LABELS),
            )
            after_record = session.run(
                """
                MATCH (n)
                WHERE n.stem IS NOT NULL OR any(label IN labels(n) WHERE label IN $labels)
                RETURN count(n) AS count
                """,
                labels=sorted(_RUNTIME_GRAPH_LABELS),
            ).single()
            after = int(after_record["count"]) if after_record else 0
            logger.info("Cleared Neo4j runtime lecture graphs: before=%s after=%s", before, after)
            return {"before": before, "after": after}
    except HTTPException:
        logger.info("Neo4j connection is not configured; skip runtime graph cleanup")
        return {"before": 0, "after": 0}
    except Exception as e:
        logger.warning("Failed to clear Neo4j runtime lecture graphs: %s", e)
        return {"before": 0, "after": 0}


def _is_stem_loaded(stem: str) -> bool:
    with neo4j_session() as session:
        record = session.run(
            """
            MATCH (n {stem: $stem})
            WHERE n:Slide OR n:Scene OR n:Segment OR n:VisualAsset OR n:Video
            RETURN count(n) AS count
            """,
            stem=stem,
        ).single()
        return bool(record and int(record["count"] or 0) > 0)


def _unload_stem_from_neo4j(stem: str) -> None:
    with neo4j_session() as session:
        session.run("MATCH (n {stem: $stem}) DETACH DELETE n", stem=stem)


def _load_graphrag_layer_for_stem(
    stem: str,
    output_dir: Path,
    graphrag_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    graphrag_dir = graphrag_dir or find_graphrag_output_dir(stem, output_dir)
    if not graphrag_dir:
        return {"graphrag_output_dir": None, "graphrag": {}}

    with neo4j_session() as session:
        graphrag_counts = session.execute_write(
            lambda tx: (
                delete_graphrag_layer_tx(tx, stem),
                load_graphrag_layer_tx(tx, stem, graphrag_dir),
            )[1]
        )
        fused_path = output_dir / f"{stem}_fused.json"
        if fused_path.is_file():
            graphrag_counts.update(compute_keyword_match(session, stem, fused_path))
            graphrag_counts.update(compute_visual_match(session, stem, fused_path))
        graphrag_counts.update(compute_annotation_match(session, stem))
        graphrag_counts.update(compute_audio_segment_match(session, stem))
        graphrag_counts.update(compute_final_weight(session, stem))
        graphrag_counts.update(compute_relation_boost(session, stem))
        graph_counts = _stem_graph_counts(session, stem)

    return {
        "graphrag_output_dir": str(graphrag_dir),
        "graphrag": graphrag_counts,
        "node_count": graph_counts["nodes"],
        "edge_count": graph_counts["relationships"],
        "concept_count": graph_counts["concepts"],
    }


def _ensure_stem_loaded(stem: str, output_dir: str) -> Dict[str, Any]:
    output_path = Path(output_dir)
    if _is_stem_loaded(stem):
        with neo4j_session() as session:
            graphrag_counts = _graphrag_layer_counts(session, stem)
        if graphrag_counts.get("entities", 0) == 0:
            loaded = _load_graphrag_layer_for_stem(stem, output_path)
            return {"loaded_now": False, "graphrag_loaded_now": bool(loaded.get("graphrag")), **loaded}
        return {"loaded_now": False, "graphrag": graphrag_counts}
    try:
        ingest_result = ingest_parquet_to_neo4j(stem=stem, output_dir=output_path)
        graphrag_loaded = _load_graphrag_layer_for_stem(stem, output_path)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Neo4j 적재 실패: {e}")
    return {
        "loaded_now": True,
        "node_count": ingest_result.get("node_count", 0),
        "edge_count": ingest_result.get("edge_count", 0),
        **graphrag_loaded,
    }
