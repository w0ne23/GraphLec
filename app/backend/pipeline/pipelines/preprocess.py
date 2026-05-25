import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock


def run_preprocess_pipeline(args, runtime, *, build_analyzer_input: bool = True, helpers) -> dict:
    """Build shared artifacts consumed by graph and verifier workflows."""
    stem = runtime.stem
    paths = runtime.paths
    output_dir = runtime.output_dir
    slides_dir = runtime.slides_dir
    timings = runtime.timings

    r9: dict = {}

    helpers._banner("Stage 1  —  병렬 실행 (슬라이드 추출 + 오디오 품질 분석)")
    t_parallel = time.time()
    audio_analyze_result: dict = {}
    runtime.notify_stage("preprocess_extract_media", "run")

    if args.skip_extract:
        helpers.log.info("Stage 1A 건너뜀 (--skip-extract)")
        meta_path = str(paths["metadata"])
        timings["Stage 1A 슬라이드 추출"] = 0.0
        audio_analyze_result = helpers.stage1b_audio_analyze(args, output_dir)
        timings["Stage 1B 오디오 품질 분석"] = audio_analyze_result["elapsed"]
    else:
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_1a = executor.submit(helpers.stage1_extract, args, slides_dir, output_dir)
            future_1b = executor.submit(helpers.stage1b_audio_analyze, args, output_dir)
            for future in as_completed([future_1a, future_1b]):
                if future is future_1a:
                    r1 = future.result()
                    meta_path = r1["meta_path"]
                    timings["Stage 1A 슬라이드 추출"] = r1["elapsed"]
                else:
                    audio_analyze_result = future.result()
                    timings["Stage 1B 오디오 품질 분석"] = audio_analyze_result["elapsed"]

    duration = audio_analyze_result.get("duration", 0.0)
    timings["Stage 1 병렬 총"] = time.time() - t_parallel
    runtime.notify_stage("preprocess_extract_media", "done")
    runtime.write_timings("Stage 1 병렬 총")
    print(f"\n  ✓ Stage 1 완료  ({timings['Stage 1 병렬 총']:.1f}초)")
    print("─" * 70)

    helpers._banner("Stage 2  —  병렬 실행 (슬라이드 텍스트화 + 전체 전사)")
    t_parallel = time.time()
    transcript_result: dict = {}
    runtime.notify_stage("preprocess_textualize_transcribe", "run")

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_2a = executor.submit(helpers.stage2_textualize, args, slides_dir, output_dir)
        future_2b = executor.submit(helpers.stage2b_transcribe, args, meta_path, duration, output_dir)
        for future in as_completed([future_2a, future_2b]):
            if future is future_2a:
                r2 = future.result()
                textualized_path = r2["textualized_path"]
                timings["Stage 2A 슬라이드 텍스트화"] = r2["elapsed"]
            else:
                transcript_result = future.result()
                timings["Stage 2B 전체 전사"] = transcript_result["elapsed"]

    timings["Stage 2 병렬 총"] = time.time() - t_parallel
    runtime.notify_stage("preprocess_textualize_transcribe", "done")
    runtime.write_timings("Stage 2 병렬 총")
    print(f"\n  ✓ Stage 2 완료  ({timings['Stage 2 병렬 총']:.1f}초)")
    print("─" * 70)

    helpers._banner("Stage 3  —  병렬 실행 (annotation + 오디오 후처리)")
    t_parallel = time.time()
    audio_result: dict = {}
    annotation_result: dict = {}
    analyzer_lock = Lock()
    analyzer_input_built = {"done": False}
    runtime.notify_stage("preprocess_enrich_audio_annotation", "run")

    transcript_raw_path = transcript_result.get(
        "transcript_raw_path",
        str(output_dir / f"{stem}_transcript_raw.json"),
    )

    def _build_analyzer_input_once(audio_payload: dict) -> None:
        with analyzer_lock:
            if analyzer_input_built["done"]:
                return
            runtime.notify_stage("verifier_build_analyzer_input", "run")
            local_r9 = helpers.stage9_build_analyzer_merged_clean(
                args,
                meta_path=meta_path,
                textualized_path=textualized_path,
                segments_path=audio_payload.get("segments_path", str(paths["segments"])),
                output_dir=output_dir,
                duration=audio_payload.get("duration", duration),
                slides_structure=audio_payload.get("slides_structure"),
            )
            timings["Stage 9 analyzer 입력 생성"] = local_r9["elapsed"]
            r9.update(local_r9)
            analyzer_input_built["done"] = True
            runtime.notify_stage("verifier_build_analyzer_input", "done")

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_a = executor.submit(helpers.stage3a_annotation, args, slides_dir, output_dir)
        future_b = executor.submit(
            helpers.stage3b_audio,
            args,
            meta_path,
            textualized_path,
            duration,
            output_dir,
            transcript_raw_path,
            _build_analyzer_input_once if build_analyzer_input else None,
        )
        for future in as_completed([future_a, future_b]):
            if future is future_a:
                annotation_result = future.result()
                timings["Stage 3A annotation"] = annotation_result["elapsed"]
            else:
                audio_result = future.result()
                if build_analyzer_input and not analyzer_input_built["done"]:
                    _build_analyzer_input_once(audio_result)

    timings["Stage 3 병렬 총"] = time.time() - t_parallel
    runtime.notify_stage("preprocess_enrich_audio_annotation", "done")
    runtime.write_timings("Stage 3 병렬 총")
    print(f"\n  ✓ Stage 3 완료  ({timings['Stage 3 병렬 총']:.1f}초)")
    print("─" * 70)

    helpers._banner("Stage 4  —  병렬 실행 (classifier + by_scene 저장)")
    t_parallel = time.time()
    classified_result: dict = {}
    by_scene_result: dict = {}
    silences_path = audio_result.get("silences_path", str(paths["silences"]))
    annotation_path = annotation_result.get("annotation_path", str(paths["annotation"]))
    runtime.notify_stage("preprocess_classify_scene", "run")

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_c = executor.submit(
            helpers.stage4a_classify, args, textualized_path, meta_path, silences_path, output_dir
        )
        future_d = executor.submit(helpers.stage4b_save_by_scene, args, audio_result, output_dir)
        for future in as_completed([future_c, future_d]):
            if future is future_c:
                classified_result = future.result()
                timings["Stage 4A 분류"] = classified_result.get("elapsed", 0.0)
            else:
                by_scene_result = future.result()
                timings["Stage 4B by_scene 저장"] = by_scene_result.get("elapsed", 0.0)

    timings["Stage 4 병렬 총"] = time.time() - t_parallel
    runtime.notify_stage("preprocess_classify_scene", "done")
    runtime.write_timings("Stage 4 병렬 총")
    print(f"\n  ✓ Stage 4 완료  ({timings['Stage 4 병렬 총']:.1f}초)")
    print("─" * 70)

    runtime.notify_stage("preprocess_fusion", "run")
    r5 = helpers.stage5_fusion(
        args,
        textualized_path=textualized_path,
        annotation_path=annotation_path,
        audio_result=audio_result,
        output_dir=output_dir,
    )
    timings["Stage 5 퓨전"] = r5["elapsed"]
    runtime.notify_stage("preprocess_fusion", "done")
    runtime.write_timings("Stage 5 퓨전")

    if build_analyzer_input and not timings.get("Stage 9 analyzer 입력 생성"):
        timings["Stage 9 analyzer 입력 생성"] = 0.0

    return {
        "meta_path": meta_path,
        "textualized_path": textualized_path,
        "transcript_result": transcript_result,
        "audio_result": audio_result,
        "annotation_path": annotation_path,
        "annotation_result": annotation_result,
        "classified_result": classified_result,
        "by_scene_result": by_scene_result,
        "fusion_result": r5,
        "analyzer_input_result": r9,
    }
