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

    helpers._banner("P1 extract_media — 슬라이드 추출 + 오디오 품질 분석")
    t_parallel = time.time()
    audio_analyze_result: dict = {}
    runtime.notify_stage("preprocess_extract_media", "run")

    if args.skip_extract:
        helpers.log.info("P1A extract_slides — 슬라이드 추출 건너뜀 (--skip-extract)")
        meta_path = str(paths["metadata"])
        timings["P1A extract_slides — 슬라이드 추출"] = 0.0
        audio_analyze_result = helpers.stage1b_audio_analyze(args, output_dir)
        timings["P1B analyze_audio_quality — 오디오 품질 분석"] = audio_analyze_result["elapsed"]
    else:
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_1a = executor.submit(helpers.stage1_extract, args, slides_dir, output_dir)
            future_1b = executor.submit(helpers.stage1b_audio_analyze, args, output_dir)
            for future in as_completed([future_1a, future_1b]):
                if future is future_1a:
                    r1 = future.result()
                    meta_path = r1["meta_path"]
                    timings["P1A extract_slides — 슬라이드 추출"] = r1["elapsed"]
                else:
                    audio_analyze_result = future.result()
                    timings["P1B analyze_audio_quality — 오디오 품질 분석"] = audio_analyze_result["elapsed"]

    duration = audio_analyze_result.get("duration", 0.0)
    timings["P1 extract_media total — 슬라이드 추출 + 오디오 품질 분석 총합"] = time.time() - t_parallel
    runtime.notify_stage("preprocess_extract_media", "done")
    runtime.write_timings("P1 extract_media total — 슬라이드 추출 + 오디오 품질 분석 총합")
    print(f"\n  ✓ P1 extract_media 완료 — 슬라이드 추출 + 오디오 품질 분석  ({timings['P1 extract_media total — 슬라이드 추출 + 오디오 품질 분석 총합']:.1f}초)")
    print("─" * 70)

    helpers._banner("P2 textualize_transcribe — 슬라이드 텍스트화 + 전체 전사")
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
                timings["P2A textualize_slides — 슬라이드 텍스트화"] = r2["elapsed"]
            else:
                transcript_result = future.result()
                timings["P2B transcribe_audio — 전체 전사"] = transcript_result["elapsed"]

    timings["P2 textualize_transcribe total — 텍스트화 + 전사 총합"] = time.time() - t_parallel
    runtime.notify_stage("preprocess_textualize_transcribe", "done")
    runtime.write_timings("P2 textualize_transcribe total — 텍스트화 + 전사 총합")
    print(f"\n  ✓ P2 textualize_transcribe 완료 — 슬라이드 텍스트화 + 전체 전사  ({timings['P2 textualize_transcribe total — 텍스트화 + 전사 총합']:.1f}초)")
    print("─" * 70)

    helpers._banner("P3 enrich_audio_annotation — 필기 강조 분석 + 오디오 후처리")
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
            timings["V1 build_analyzer_input — verifier 입력 생성"] = local_r9["elapsed"]
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
                timings["P3A analyze_annotation — 필기 강조 분석"] = annotation_result["elapsed"]
            else:
                audio_result = future.result()
                timings["P3B process_audio — 오디오 후처리"] = audio_result.get("elapsed", 0.0)
                if build_analyzer_input and not analyzer_input_built["done"]:
                    _build_analyzer_input_once(audio_result)

    timings["P3 enrich_audio_annotation total — 보강 분석 총합"] = time.time() - t_parallel
    runtime.notify_stage("preprocess_enrich_audio_annotation", "done")
    runtime.write_timings("P3 enrich_audio_annotation total — 보강 분석 총합")
    print(f"\n  ✓ P3 enrich_audio_annotation 완료 — 필기 강조 분석 + 오디오 후처리  ({timings['P3 enrich_audio_annotation total — 보강 분석 총합']:.1f}초)")
    print("─" * 70)

    helpers._banner("P4 classify_scene — 슬라이드 분류 + scene 구조 저장")
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
                timings["P4A classify_slides — 슬라이드 분류"] = classified_result.get("elapsed", 0.0)
            else:
                by_scene_result = future.result()
                timings["P4B save_scene_structure — scene 구조 저장"] = by_scene_result.get("elapsed", 0.0)

    timings["P4 classify_scene total — 구조화 총합"] = time.time() - t_parallel
    runtime.notify_stage("preprocess_classify_scene", "done")
    runtime.write_timings("P4 classify_scene total — 구조화 총합")
    print(f"\n  ✓ P4 classify_scene 완료 — 슬라이드 분류 + scene 구조 저장  ({timings['P4 classify_scene total — 구조화 총합']:.1f}초)")
    print("─" * 70)

    runtime.notify_stage("preprocess_fusion", "run")
    r5 = helpers.stage5_fusion(
        args,
        textualized_path=textualized_path,
        annotation_path=annotation_path,
        audio_result=audio_result,
        output_dir=output_dir,
    )
    timings["P5 fusion — 데이터 퓨전"] = r5["elapsed"]
    runtime.notify_stage("preprocess_fusion", "done")
    runtime.write_timings("P5 fusion — 데이터 퓨전")

    if build_analyzer_input and not timings.get("V1 build_analyzer_input — verifier 입력 생성"):
        timings["V1 build_analyzer_input — verifier 입력 생성"] = 0.0

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
