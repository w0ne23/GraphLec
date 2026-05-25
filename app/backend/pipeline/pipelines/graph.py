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

    runtime.notify_stage("graph", "run")
    if args.skip_graph_triples:
        print("\n  ⏭  Stage 6 그래프 트리플 생성 — 사용자 옵션으로 스킵")
        print("─" * 70)
        timings["Stage 6 그래프 트리플"] = 0.0
        timings["Neo4j 적재"] = 0.0
    else:
        r6 = helpers.stage6_graph_triples(args, output_dir, slides_dir)
        timings["Stage 6 그래프 트리플"] = r6["elapsed"]
        print("\n  ⏭  Neo4j 적재 — 강의 시청 화면 진입 시 자동 적재")
        print("─" * 70)
        timings["Neo4j 적재"] = 0.0
    runtime.notify_stage("graph", "done")
    runtime.notify_stage("integrate", "done")

    if args.skip_lance_index:
        print("\n  ⏭  Stage 7 Lance 인덱스 — 사용자 옵션으로 스킵")
        print("─" * 70)
        timings["Stage 7 Lance 인덱스"] = 0.0
    else:
        runtime.notify_stage("summarize", "run")
        r7 = helpers.stage7_lance_index(args, output_dir, slides_dir)
        timings["Stage 7 Lance 인덱스"] = r7.get("elapsed", 0.0)
        runtime.notify_stage("summarize", "done")

    if getattr(args, "skip_graphrag_index", False):
        print("\n  ⏭  Stage 7B GraphRAG 인덱스 — 사용자 옵션으로 스킵")
        print("─" * 70)
        runtime.record_timing("Stage 7B GraphRAG 인덱스", 0.0, "skipped")
    else:
        runtime.stage_status["Stage 7B GraphRAG 인덱스"] = "run"
        runtime.write_timings("Stage 7B GraphRAG 인덱스")
        r7b = helpers.stage7b_graphrag_index(args, output_dir)
        runtime.record_timing("Stage 7B GraphRAG 인덱스", r7b.get("elapsed", 0.0), "done")

    if getattr(args, "skip_metadata", False):
        print("\n  ⏭  Stage 8 메타데이터 생성 — 사용자 옵션으로 스킵")
        print("─" * 70)
        timings["Stage 8 메타데이터 생성"] = 0.0
    else:
        r8 = helpers.stage8_generate_metadata(args, output_dir, slides_dir)
        timings["Stage 8 메타데이터 생성"] = r8["elapsed"]

    if getattr(args, "skip_recommender_index", False):
        print("\n  ⏭  Stage 11 추천 인덱스 생성 — 사용자 옵션으로 스킵")
        print("─" * 70)
        timings["Stage 11 추천 인덱스 생성"] = 0.0
    else:
        r11 = helpers.stage11_build_recommender_index(args)
        timings["Stage 11 추천 인덱스 생성"] = r11["elapsed"]

    runtime.write_timings("graph_pipeline_done")
    return {
        "graph_triples_result": r6,
        "lance_result": r7,
        "graphrag_result": r7b,
        "metadata_result": r8,
        "recommender_result": r11,
    }

