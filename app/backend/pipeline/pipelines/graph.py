import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def run_graph_pipeline(
    args,
    *,
    preprocess_result: dict,
    output_dir: Path,
    slides_dir: Path,
    paths: dict,
    timings: dict[str, float],
    stage_status: dict[str, str],
    notify_stage,
    write_timings,
    record_timing,
    helpers,
) -> dict:
    """Run graph/search/recommendation artifacts after shared preprocessing and verifier start."""
    meta_path = preprocess_result["meta_path"]
    textualized_path = preprocess_result["textualized_path"]
    annotation_result = preprocess_result["annotation_result"]
    audio_result = preprocess_result["audio_result"]

    helpers._banner("G1 classify_scene — 슬라이드 분류 + scene 구조 저장")
    t_parallel = time.time()
    classified_result: dict = {}
    by_scene_result: dict = {}

    silences_path = audio_result.get("silences_path", str(paths["silences"]))
    annotation_path = annotation_result.get("annotation_path", str(paths["annotation"]))

    notify_stage("graph_classify_scene", "run")
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_c = executor.submit(
            helpers.classify_slides, args, textualized_path, meta_path, silences_path, output_dir
        )
        future_d = executor.submit(helpers.save_scene_structure, args, audio_result, output_dir)
        for future in as_completed([future_c, future_d]):
            if future is future_c:
                classified_result = future.result()
                timings["G1A classify_slides — 슬라이드 분류"] = classified_result.get("elapsed", 0.0)
            else:
                by_scene_result = future.result()
                timings["G1B save_scene_structure — scene 구조 저장"] = by_scene_result.get("elapsed", 0.0)

    timings["G1 classify_scene total — 구조화 총합"] = time.time() - t_parallel
    notify_stage("graph_classify_scene", "done")
    print(f"\n  ✓ G1 classify_scene 완료 — 슬라이드 분류 + scene 구조 저장  ({timings['G1 classify_scene total — 구조화 총합']:.1f}초)")
    print("─" * 70)

    notify_stage("graph_fusion", "run")
    r5 = helpers.fuse_preprocessed_data(
        args,
        textualized_path=textualized_path,
        annotation_path=annotation_path,
        audio_result=audio_result,
        output_dir=output_dir,
    )
    timings["G2 fusion — 데이터 퓨전"] = r5["elapsed"]
    notify_stage("graph_fusion", "done")

    r6: dict = {}
    if args.skip_graph_triples:
        print("\n  ⏭  G3 graph_triples — 그래프 트리플 생성 스킵")
        print("─" * 70)
        timings["G3 graph_triples — 그래프 트리플 생성"] = 0.0
        timings["G3 neo4j_load — Neo4j 적재"] = 0.0
        notify_stage("graph_triples", "done")
    else:
        notify_stage("graph_triples", "run")
        r6 = helpers.generate_graph_triples(args, output_dir, slides_dir)
        timings["G3 graph_triples — 그래프 트리플 생성"] = r6["elapsed"]
        notify_stage("graph_triples", "done")

        print("\n  ⏭  Neo4j 적재 — 강의 시청 화면 진입 시 자동 적재")
        print("─" * 70)
        timings["G3 neo4j_load — Neo4j 적재"] = 0.0

    r7: dict = {}
    if args.skip_lance_index:
        print("\n  ⏭  G4 lance_index — Lance 인덱스 생성 스킵")
        print("─" * 70)
        timings["G4 lance_index — Lance 인덱스 생성"] = 0.0
        notify_stage("graph_lance_index", "done")
    else:
        notify_stage("graph_lance_index", "run")
        r7 = helpers.build_lance_index(args, output_dir, slides_dir)
        timings["G4 lance_index — Lance 인덱스 생성"] = r7.get("elapsed", 0.0)
        notify_stage("graph_lance_index", "done")

    r7b: dict = {}
    if getattr(args, "skip_graphrag_index", False):
        print("\n  ⏭  G5 graphrag_index — GraphRAG 인덱스 생성 스킵")
        print("─" * 70)
        record_timing("G5 graphrag_index — GraphRAG 인덱스 생성", 0.0, "skipped")
        notify_stage("graph_graphrag_index", "done")
    else:
        notify_stage("graph_graphrag_index", "run")
        stage_status["G5 graphrag_index — GraphRAG 인덱스 생성"] = "run"
        write_timings("G5 graphrag_index — GraphRAG 인덱스 생성")
        r7b = helpers.build_graphrag_index(args, output_dir)
        record_timing("G5 graphrag_index — GraphRAG 인덱스 생성", r7b.get("elapsed", 0.0), "done")
        notify_stage("graph_graphrag_index", "done")

    r8: dict = {}
    if getattr(args, "skip_metadata", False):
        print("\n  ⏭  G6 metadata — 메타데이터 생성 스킵")
        print("─" * 70)
        timings["G6 metadata — 메타데이터 생성"] = 0.0
        notify_stage("graph_metadata", "done")
    else:
        notify_stage("graph_metadata", "run")
        r8 = helpers.generate_metadata(args, output_dir, slides_dir)
        timings["G6 metadata — 메타데이터 생성"] = r8["elapsed"]
        notify_stage("graph_metadata", "done")

    r11: dict = {}
    if getattr(args, "skip_recommender_index", False):
        print("\n  ⏭  G7 recommender_index — 추천 인덱스 생성 스킵")
        print("─" * 70)
        timings["G7 recommender_index — 추천 인덱스 생성"] = 0.0
        notify_stage("graph_recommender_index", "done")
    else:
        notify_stage("graph_recommender_index", "run")
        r11 = helpers.build_recommender_index(args)
        timings["G7 recommender_index — 추천 인덱스 생성"] = r11["elapsed"]
        notify_stage("graph_recommender_index", "done")

    return {
        "classified_result": classified_result,
        "by_scene_result": by_scene_result,
        "fusion_result": r5,
        "graph_triples_result": r6,
        "lance_result": r7,
        "graphrag_result": r7b,
        "metadata_result": r8,
        "recommender_result": r11,
        "annotation_path": annotation_path,
    }

