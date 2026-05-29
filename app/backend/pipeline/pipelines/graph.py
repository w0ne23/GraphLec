def run_graph_pipeline(args, runtime, preprocess_result: dict | None = None, *, helpers) -> dict:
    """Build graph/search/recommendation artifacts from shared preprocess output."""
    output_dir = runtime.output_dir
    slides_dir = runtime.slides_dir
    timings = runtime.timings
    r6: dict = {}
    r7: dict = {}
    r7b: dict = {}
    r8: dict = {}
    r11: dict = {}

    runtime.notify_stage("graph_triples", "run")
    if args.skip_graph_triples:
        print("\n  ⏭  G1 graph_triples — 그래프 트리플 생성 스킵")
        print("─" * 70)
        timings["G1 graph_triples — 그래프 트리플 생성"] = 0.0
        timings["G1 neo4j_load — Neo4j 적재"] = 0.0
    else:
        r6 = helpers.generate_graph_triples(args, output_dir, slides_dir)
        timings["G1 graph_triples — 그래프 트리플 생성"] = r6["elapsed"]
        print("\n  ⏭  Neo4j 적재 — 강의 시청 화면 진입 시 자동 적재")
        print("─" * 70)
        timings["G1 neo4j_load — Neo4j 적재"] = 0.0
    runtime.notify_stage("graph_triples", "done")

    if args.skip_lance_index:
        print("\n  ⏭  G2 lance_index — Lance 인덱스 생성 스킵")
        print("─" * 70)
        timings["G2 lance_index — Lance 인덱스 생성"] = 0.0
    else:
        runtime.notify_stage("graph_lance_index", "run")
        r7 = helpers.build_lance_index(args, output_dir, slides_dir)
        timings["G2 lance_index — Lance 인덱스 생성"] = r7.get("elapsed", 0.0)
        runtime.notify_stage("graph_lance_index", "done")

    if getattr(args, "skip_graphrag_index", False):
        print("\n  ⏭  G3 graphrag_index — GraphRAG 인덱스 생성 스킵")
        print("─" * 70)
        runtime.record_timing("G3 graphrag_index — GraphRAG 인덱스 생성", 0.0, "skipped")
    else:
        runtime.notify_stage("graph_graphrag_index", "run")
        runtime.stage_status["G3 graphrag_index — GraphRAG 인덱스 생성"] = "run"
        runtime.write_timings("G3 graphrag_index — GraphRAG 인덱스 생성")
        r7b = helpers.build_graphrag_index(args, output_dir)
        runtime.notify_stage("graph_graphrag_index", "done")
        runtime.record_timing("G3 graphrag_index — GraphRAG 인덱스 생성", r7b.get("elapsed", 0.0), "done")

    if getattr(args, "skip_metadata", False):
        print("\n  ⏭  G4 metadata — 메타데이터 생성 스킵")
        print("─" * 70)
        timings["G4 metadata — 메타데이터 생성"] = 0.0
    else:
        runtime.notify_stage("graph_metadata", "run")
        r8 = helpers.generate_metadata(args, output_dir, slides_dir)
        timings["G4 metadata — 메타데이터 생성"] = r8["elapsed"]
        runtime.notify_stage("graph_metadata", "done")

    if getattr(args, "skip_recommender_index", False):
        print("\n  ⏭  G5 recommender_index — 추천 인덱스 생성 스킵")
        print("─" * 70)
        timings["G5 recommender_index — 추천 인덱스 생성"] = 0.0
    else:
        runtime.notify_stage("graph_recommender_index", "run")
        r11 = helpers.build_recommender_index(args)
        timings["G5 recommender_index — 추천 인덱스 생성"] = r11["elapsed"]
        runtime.notify_stage("graph_recommender_index", "done")

    runtime.write_timings("graph_pipeline_done")
    return {
        "graph_triples_result": r6,
        "lance_result": r7,
        "graphrag_result": r7b,
        "metadata_result": r8,
        "recommender_result": r11,
    }
