"""Worker functions for cross-model verification."""

from __future__ import annotations

import traceback
from pathlib import Path

from .cross_merge import _issue_match_key
from .cross_utils import _empty_token_usage, _merge_token_usage, _setup_worker


def extract_worker(args_tuple):
    try:
        merged_path, model, batch_size, root, env_vars, current_date = args_tuple
        _setup_worker(root, env_vars, model)
        import analyzer.claim_pipeline as cv

        ctx = cv.prepare_verification(merged_path, current_date=current_date)
        print(f"\n  [{model}] 1단계: claim 추출 시작", flush=True)

        claims_by_batch, api_calls, token_usage = cv.extract_claims_only(
            ctx["utterances"], ctx["current_date"], ctx["hint"], ctx["slide_ctx"], batch_size
        )

        serialized = []
        for batch, claims in claims_by_batch:
            serialized.append({"batch": batch, "claims": claims})

        total = sum(len(c) for _, c in claims_by_batch)
        print(f"  [{model}] claim 추출 완료: {total}개", flush=True)
        return {
            "model": model,
            "claims_by_batch": serialized,
            "api_calls": api_calls,
            "token_usage": token_usage,
        }
    except Exception as e:
        raise RuntimeError(f"[{args_tuple[1]}] extract worker failed:\n{traceback.format_exc()}") from e


def judge_worker(args_tuple):
    try:
        merged_path, model, claims_serialized, num_runs, min_rate, root, env_vars, current_date = args_tuple
        _setup_worker(root, env_vars, model)
        import analyzer.claim_pipeline as cv

        ctx = cv.prepare_verification(merged_path, current_date=current_date)
        print(f"\n  [{model}] 2단계: claim 판정 시작", flush=True)

        claims_by_batch = [(item["batch"], item["claims"]) for item in claims_serialized]
        issues, api_calls, token_usage = cv.judge_claims_only(
            claims_by_batch, ctx["current_date"], ctx["hint"], ctx["slide_ctx"], num_runs, min_rate
        )

        print(f"  [{model}] 판정 완료: {len(issues)}건 이슈", flush=True)
        return {
            "model": model,
            "issues": issues,
            "api_calls": api_calls,
            "token_usage": token_usage,
        }
    except Exception as e:
        raise RuntimeError(f"[{args_tuple[1]}] judge worker failed:\n{traceback.format_exc()}") from e


def cross_recheck_worker(args_tuple):
    """합집합 전체 이슈를 각 모델이 텍스트+문맥으로 재검증."""
    try:
        issues, merged_path, model, root, env_vars, current_date = args_tuple
        _setup_worker(root, env_vars, model)
        import analyzer.claim_pipeline as cv

        ctx = cv.prepare_verification(merged_path, current_date=current_date)
        print(f"\n  [{model}] 텍스트+문맥 교차검증: {len(issues)}건 확인 중...", flush=True)
        resolved_model = cv._resolve_stage_model("cross_recheck")

        token_usage = _empty_token_usage()
        verdicts = {}
        for issue in issues:
            verdict, reason, call_usage = cv.judge_single_claim(issue, ctx)
            token_usage = _merge_token_usage(token_usage, call_usage)
            verdicts[_issue_match_key(issue)] = {
                "verdict": verdict,
                "reason": reason,
                "resolved_model": resolved_model,
            }

        print(f"  [{model}] 텍스트+문맥 교차검증 완료", flush=True)
        return {
            "model": model,
            "verdicts": verdicts,
            "token_usage": token_usage,
        }
    except Exception as e:
        raise RuntimeError(f"[{args_tuple[2]}] cross recheck worker failed:\n{traceback.format_exc()}") from e


def stages_3_4_worker(args_tuple):
    """primary 모델로 grounding 실행."""
    try:
        merged_path, model, issues, root, env_vars, current_date = args_tuple
        _setup_worker(root, env_vars, model)
        import analyzer.claim_pipeline as cv

        ctx = cv.prepare_verification(merged_path, current_date=current_date)
        hint = ctx["hint"]
        slide_ctx = ctx["slide_ctx"]
        slides = ctx["slides"]
        token_usage = _empty_token_usage()

        slide_rejected = []
        grounding_rejected = []
        if issues:
            verified, g_rejected, g_calls, g_failures, grounding_token_usage = cv._ground_verify_all_issues(
                issues, hint, slide_ctx, slides
            )
            grounding_rejected = g_rejected
            issues = verified
            token_usage = _merge_token_usage(token_usage, grounding_token_usage)

        return {
            "issues": issues,
            "slide_rejected": slide_rejected,
            "grounding_rejected": grounding_rejected,
            "token_usage": token_usage,
        }
    except Exception as e:
        raise RuntimeError(f"[{args_tuple[1]}] stages 3-4 worker failed:\n{traceback.format_exc()}") from e


def slide_typo_worker(args_tuple):
    """슬라이드 이미지 기준 오타 검사."""
    try:
        merged_path, model, root, env_vars, current_date = args_tuple
        _setup_worker(root, env_vars, model)
        import analyzer.claim_pipeline as cv
        from analyzer.slide_typo_checker import detect_slide_typos

        ctx = cv.prepare_verification(merged_path, current_date=current_date)
        merged = ctx["merged"]
        detector_log = str(merged.get("source_detector_log", "") or "").strip()
        img_dir = str(Path(detector_log).parent) if detector_log else None
        if img_dir and not Path(img_dir).is_dir():
            img_dir = None

        print(f"\n  [{model}] 슬라이드 오타 검사 시작", flush=True)
        typos, api_calls, failures, token_usage = detect_slide_typos(
            ctx["slides"], img_dir=img_dir, max_workers=4
        )
        print(f"  [{model}] 슬라이드 오타 검사 완료: {len(typos)}건", flush=True)
        return {
            "model": model,
            "slide_typos": typos,
            "api_calls": api_calls,
            "failures": failures,
            "token_usage": token_usage,
        }
    except Exception as e:
        raise RuntimeError(f"[{args_tuple[1]}] slide typo worker failed:\n{traceback.format_exc()}") from e


def independent_worker(args_tuple):
    try:
        merged_path, model, num_runs, min_rate, root, env_vars = args_tuple
        _setup_worker(root, env_vars, model)
        import analyzer.claim_pipeline as cv

        print(f"\n{'='*60}", flush=True)
        print(f"  모델: {model}", flush=True)
        print(f"{'='*60}", flush=True)

        result = cv.verify_lecture_content(
            merged_path=merged_path, current_date=None, num_runs=num_runs, min_detection_rate=min_rate
        )
        result["_model"] = model
        return result
    except Exception as e:
        raise RuntimeError(f"[{args_tuple[1]}] independent worker failed:\n{traceback.format_exc()}") from e
