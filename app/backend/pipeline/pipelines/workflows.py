def run_direct_upload_workflow(args, progress_callback=None, *, helpers) -> dict:
    """Direct workflow: shared preprocess, then graph/upload artifacts."""
    runtime = helpers._create_pipeline_runtime(args, progress_callback, "Direct upload pipeline")
    preprocess_result: dict = {}
    graph_result: dict = {}
    try:
        preprocess_result = helpers.run_preprocess_pipeline(args, runtime, build_analyzer_input=False)
        runtime.timings["Stage 10 verifier 백그라운드 시작"] = 0.0
        graph_result = helpers.run_graph_pipeline(args, runtime, preprocess_result)
        helpers._print_generated_files(runtime, preprocess_result, graph_result, {})
        return {"preprocess": preprocess_result, "graph": graph_result}
    except Exception as e:
        print(f"\n❌ 파이프라인 오류: {e}")
        raise
    finally:
        helpers._finish_pipeline_run(runtime)


def run_verified_upload_workflow(args, progress_callback=None, *, helpers) -> dict:
    """Verified workflow: shared preprocess, then synchronous verifier; graph waits for approval."""
    runtime = helpers._create_pipeline_runtime(args, progress_callback, "Verified upload pipeline")
    preprocess_result: dict = {}
    verifier_result: dict = {}
    try:
        preprocess_result = helpers.run_preprocess_pipeline(args, runtime, build_analyzer_input=True)
        verifier_result = helpers.run_verifier_pipeline(args, runtime, preprocess_result, background=False)
        helpers._print_generated_files(runtime, preprocess_result, {}, verifier_result)
        return {"preprocess": preprocess_result, "verifier": verifier_result}
    except Exception as e:
        print(f"\n❌ 파이프라인 오류: {e}")
        raise
    finally:
        helpers._finish_pipeline_run(runtime)


def run_pipeline(args, progress_callback=None, *, helpers):
    runtime = helpers._create_pipeline_runtime(args, progress_callback)
    preprocess_result: dict = {}
    graph_result: dict = {}
    verifier_result: dict = {}
    try:
        should_stop_after_verifier = (
            getattr(args, "stop_after_claim_extract", False)
            or getattr(args, "stop_after_issue_judge", False)
            or getattr(args, "stop_after_verifier_start", False)
        )
        should_run_verifier = should_stop_after_verifier or not getattr(args, "skip_analyzer", False)
        preprocess_result = helpers.run_preprocess_pipeline(
            args,
            runtime,
            build_analyzer_input=should_run_verifier,
        )

        if should_run_verifier:
            verifier_result = helpers.run_verifier_pipeline(
                args,
                runtime,
                preprocess_result,
                background=not (
                    getattr(args, "stop_after_claim_extract", False)
                    or getattr(args, "stop_after_issue_judge", False)
                ),
            )
        else:
            runtime.timings["Stage 10 verifier 백그라운드 시작"] = 0.0

        if should_stop_after_verifier:
            helpers._print_stop_after_summary(args, runtime, verifier_result)
            return

        graph_result = helpers.run_graph_pipeline(args, runtime, preprocess_result)
        helpers._print_generated_files(runtime, preprocess_result, graph_result, verifier_result)

    except Exception as e:
        print(f"\n❌ 파이프라인 오류: {e}")
        raise
    finally:
        helpers._finish_pipeline_run(runtime)

