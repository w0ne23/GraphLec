import os, sys
sys.path.insert(0, "app/backend")
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

from neo4j import GraphDatabase
from pipeline.graphrag_neo4j_ingest import (
    delete_custom_concept_layer_tx,
    delete_graphrag_layer_tx,
    load_graphrag_layer_tx,
    find_graphrag_output_dir,
)
from pipeline.graphrag_emphasis import (
    compute_keyword_match,
    compute_visual_match,
    compute_annotation_match,
    compute_audio_segment_match,
    compute_final_weight,
    compute_relation_boost,
)

STEM = "os1-1"
OUTPUT_DIR = Path("output")

graphrag_dir = find_graphrag_output_dir(STEM, OUTPUT_DIR)
print(f"GraphRAG dir: {graphrag_dir}")
if not graphrag_dir:
    print("ERROR: GraphRAG output을 찾지 못했습니다.")
    sys.exit(1)

driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))
)

with driver.session() as session:
    counts = session.execute_write(lambda tx: (
        delete_custom_concept_layer_tx(tx, STEM),
        delete_graphrag_layer_tx(tx, STEM),
        load_graphrag_layer_tx(tx, STEM, graphrag_dir),
    )[2])
    fused_path = OUTPUT_DIR / f"{STEM}_fused.json"
    if fused_path.is_file():
        counts.update(compute_keyword_match(session, STEM, fused_path))
        counts.update(compute_visual_match(session, STEM, fused_path))
    counts.update(compute_annotation_match(session, STEM))
    counts.update(compute_audio_segment_match(session, STEM))
    counts.update(compute_final_weight(session, STEM))
    counts.update(compute_relation_boost(session, STEM))
    summary = session.run(
        "MATCH (n {stem: $stem}) RETURN count(n) AS total", stem=STEM
    ).single()

driver.close()
print("적재 완료:", counts)
print(f"전체 노드 수: {summary['total']}")
