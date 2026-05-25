from pathlib import Path

def run_verifier_pipeline(
    args,
    runtime,
    preprocess_result: dict | None = None,
    *,
    background: bool = False,
    helpers,
) -> dict:
    """Run verifier stages from the shared analyzer input."""
    timings = runtime.timings
    if getattr(args, "skip_analyzer", False):
        timings["Stage 10 verifier 백그라운드 시작"] = 0.0
        return {}

    analyzer_result = (preprocess_result or {}).get("analyzer_input_result", {})
    merged_clean_path = analyzer_result.get(
        "merged_clean_path",
        str(runtime.output_dir / f"{runtime.stem}_analyzer" / f"{runtime.stem}_merged_clean.json"),
    )
    if not Path(merged_clean_path).exists():
        raise FileNotFoundError(f"verifier 입력 파일 없음: {merged_clean_path}")

    r10: dict = {}
    r10a: dict = {}
    r10b: dict = {}

    if getattr(args, "stop_after_claim_extract", False) or getattr(args, "stop_after_issue_judge", False):
        runtime.notify_stage("verifier_run", "run")
        r10a = helpers.stage10_extract_claims(args, merged_clean_path=merged_clean_path, output_dir=runtime.output_dir)
        timings["Stage 10A claim 추출"] = r10a["elapsed"]
        if getattr(args, "stop_after_issue_judge", False):
            r10b = helpers.stage10_issue_judge(
                args,
                merged_clean_path=merged_clean_path,
                output_dir=runtime.output_dir,
                claims_jsonl=r10a["claims_jsonl"],
            )
            timings["Stage 10B 1차 issue judge"] = r10b["elapsed"]
        timings["Stage 10 verifier 백그라운드 시작"] = 0.0
        runtime.notify_stage("verifier_run", "done")
    elif background:
        runtime.notify_stage("verifier_run", "run")
        r10 = helpers.stage10_spawn_analyzers_subprocess(args, merged_clean_path, runtime.output_dir)
        timings["Stage 10 verifier 백그라운드 시작"] = r10["elapsed"]
        runtime.notify_stage("verifier_run", "done")
    else:
        runtime.notify_stage("verifier_run", "run")
        r10 = helpers.stage10_run_verifier(args, merged_clean_path, runtime.output_dir)
        timings["Stage 10 verifier 실행"] = r10["elapsed"]
        runtime.notify_stage("verifier_run", "done")

    runtime.write_timings("verifier_pipeline_done")
    return {
        "verifier_result": r10,
        "claims_result": r10a,
        "issue_judge_result": r10b,
    }
