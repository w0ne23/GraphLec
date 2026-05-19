"""Worker functions for cross-model verification."""

from __future__ import annotations

import traceback
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import OrderedDict

from .cross_merge import _issue_match_key
from .cross_utils import _empty_token_usage, _merge_token_usage, _setup_worker


def _prepare_verification(merged_path: str):
    from analyzer.claim_pipeline import prepare_verification

    return prepare_verification(merged_path)


def _crosscheck_max_issues_per_batch() -> int:
    try:
        value = int(os.getenv("VERIFIER_CROSSCHECK_MAX_ISSUES_PER_BATCH", "5") or "5")
    except ValueError:
        value = 5
    return max(1, value)


def _crosscheck_group_max_workers() -> int:
    try:
        value = int(os.getenv("VERIFIER_CROSSCHECK_GROUP_MAX_WORKERS", "4") or "4")
    except ValueError:
        value = 4
    return max(1, value)


def _judge_crosscheck_group(group_item, ctx: dict, max_issues_per_batch: int, model: str, total_groups: int):
    """Run crosscheck for one slide-context group."""
    from analyzer.claim_crosscheck import judge_claim_batch

    group_idx, slide_number, group = group_item
    slide_label = slide_number if slide_number > 0 else "unknown"
    print(
        f"  [{model}] cross ({group_idx}/{total_groups}) slide {slide_label}: {len(group)}건",
        flush=True,
    )

    token_usage = _empty_token_usage()
    verdicts = {}
    chunks = [
        group[start : start + max_issues_per_batch]
        for start in range(0, len(group), max_issues_per_batch)
    ]
    for chunk_idx, chunk in enumerate(chunks, 1):
        if len(chunks) > 1:
            print(
                f"  [{model}] cross ({group_idx}/{total_groups}) slide {slide_label} "
                f"batch {chunk_idx}/{len(chunks)}: {len(chunk)}건",
                flush=True,
            )
        payloads, call_usage = judge_claim_batch(chunk, ctx)
        token_usage = _merge_token_usage(token_usage, call_usage)
        issue_id_to_key = {
            f"i{idx + 1:04d}": _issue_match_key(issue)
            for idx, issue in enumerate(chunk)
        }
        for idx, issue in enumerate(chunk, 1):
            issue_id = f"i{idx:04d}"
            payload = payloads.get(
                issue_id,
                {"verdict": "inconclusive", "reason": "crosscheck batch 결과 없음"},
            )
            issue_key = _issue_match_key(issue)
            if isinstance(payload, dict):
                payload = dict(payload)
                payload["source_issue_key"] = issue_key
                merged_into = str(payload.get("merged_into_issue_id", "") or "").strip()
                if merged_into and merged_into in issue_id_to_key:
                    payload["merged_into_issue_key"] = issue_id_to_key[merged_into]
            verdicts[issue_key] = payload
    return group_idx, verdicts, token_usage


def extract_worker(args_tuple):
    try:
        if len(args_tuple) >= 6:
            merged_path, model, batch_size, root, env_vars, max_workers = args_tuple
        else:
            merged_path, model, batch_size, root, env_vars = args_tuple
            max_workers = None
        _setup_worker(root, env_vars, model)
        from analyzer.claim_extractor import extract_claims_only

        ctx = _prepare_verification(merged_path)
        print(f"\n  [{model}] 1단계: claim 추출 시작", flush=True)

        claims_by_batch, api_calls, token_usage = extract_claims_only(
            ctx["utterances"],
            ctx["current_date"],
            ctx["hint"],
            ctx["slide_ctx"],
            batch_size,
            max_workers=max_workers,
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
        if len(args_tuple) >= 9:
            merged_path, model, claims_serialized, num_runs, min_rate, root, env_vars, context_mode, max_workers = args_tuple
        elif len(args_tuple) >= 8:
            merged_path, model, claims_serialized, num_runs, min_rate, root, env_vars, context_mode = args_tuple
            max_workers = None
        else:
            merged_path, model, claims_serialized, num_runs, min_rate, root, env_vars = args_tuple
            context_mode = None
            max_workers = None
        _setup_worker(root, env_vars, model)
        from analyzer.claim_verifier import judge_claims_only

        ctx = _prepare_verification(merged_path)
        print(
            f"\n  [{model}] 2단계: Issue_detection 시작 "
            f"(context={context_mode or 'batch'})",
            flush=True,
        )

        claims_by_batch = [(item["batch"], item["claims"]) for item in claims_serialized]
        issues, api_calls, token_usage = judge_claims_only(
            claims_by_batch, ctx["current_date"], ctx["hint"], num_runs, min_rate,
            log_prefix=model,
            context_mode=context_mode,
            max_workers=max_workers,
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
        issues, merged_path, model, root, env_vars = args_tuple
        _setup_worker(root, env_vars, model)
        from analyzer import claim_common as cv

        ctx = _prepare_verification(merged_path)
        resolved_model = cv._resolve_stage_model("cross_recheck")

        token_usage = _empty_token_usage()
        verdicts = {}
        groups: OrderedDict[int, list[dict]] = OrderedDict()
        for issue_idx, issue in enumerate(issues):
            try:
                slide_number = int(issue.get("slide_number", 0) or 0)
            except Exception:
                slide_number = 0
            if slide_number <= 0:
                slide_number = -(issue_idx + 1)
            groups.setdefault(slide_number, []).append(issue)

        total_groups = len(groups)
        print(
            f"\n  [{model}] 텍스트+문맥 교차검증: {len(issues)}건 / {total_groups}개 문맥 묶음 확인 중...",
            flush=True,
        )
        max_issues_per_batch = _crosscheck_max_issues_per_batch()
        group_items = [
            (group_idx, slide_number, group)
            for group_idx, (slide_number, group) in enumerate(groups.items(), 1)
        ]
        group_max_workers = min(_crosscheck_group_max_workers(), max(1, total_groups))
        if group_max_workers > 1 and total_groups > 1:
            print(
                f"  [{model}] cross 병렬 처리: max_workers={group_max_workers}",
                flush=True,
            )
            group_results = {}
            print(
                f"  [{model}] cross cache warm-up: 첫 문맥 묶음 선실행 후 나머지 병렬 처리",
                flush=True,
            )
            warmup_idx, warmup_verdicts, warmup_usage = _judge_crosscheck_group(
                group_items[0],
                ctx,
                max_issues_per_batch,
                model,
                total_groups,
            )
            group_results[warmup_idx] = (warmup_verdicts, warmup_usage)
            remaining_group_items = group_items[1:]
            with ThreadPoolExecutor(max_workers=group_max_workers) as executor:
                futures = {
                    executor.submit(
                        _judge_crosscheck_group,
                        group_item,
                        ctx,
                        max_issues_per_batch,
                        model,
                        total_groups,
                    ): group_item[0]
                    for group_item in remaining_group_items
                }
                for future in as_completed(futures):
                    group_idx = futures[future]
                    try:
                        _idx, group_verdicts, call_usage = future.result()
                    except Exception as e:
                        raise RuntimeError(f"crosscheck group {group_idx} failed: {e}") from e
                    group_results[group_idx] = (group_verdicts, call_usage)
            for group_idx, _slide_number, _group in group_items:
                group_verdicts, call_usage = group_results.get(group_idx, ({}, _empty_token_usage()))
                token_usage = _merge_token_usage(token_usage, call_usage)
                for key, payload in group_verdicts.items():
                    verdicts[key] = {
                        **payload,
                        "resolved_model": resolved_model,
                    }
        else:
            for group_item in group_items:
                _idx, group_verdicts, call_usage = _judge_crosscheck_group(
                    group_item,
                    ctx,
                    max_issues_per_batch,
                    model,
                    total_groups,
                )
                token_usage = _merge_token_usage(token_usage, call_usage)
                for key, payload in group_verdicts.items():
                    verdicts[key] = {
                        **payload,
                        "resolved_model": resolved_model,
                    }

        print(f"  [{model}] 텍스트+문맥 교차검증 완료", flush=True)
        return {
            "model": model,
            "verdicts": verdicts,
            "token_usage": token_usage,
        }
    except Exception as e:
        raise RuntimeError(f"[{args_tuple[2]}] crosscheck worker failed:\n{traceback.format_exc()}") from e


def stages_3_4_worker(args_tuple):
    """primary 모델로 grounding 실행."""
    try:
        merged_path, model, issues, root, env_vars = args_tuple
        _setup_worker(root, env_vars, model)
        from analyzer.claim_grounding import ground_verify_all_issues

        ctx = _prepare_verification(merged_path)
        hint = ctx["hint"]
        slide_ctx = ctx["slide_ctx"]
        slides = ctx["slides"]
        token_usage = _empty_token_usage()

        grounding_rejected = []
        if issues:
            verified, g_rejected, g_calls, g_failures, grounding_token_usage = ground_verify_all_issues(
                issues, hint, slide_ctx, slides
            )
            grounding_rejected = g_rejected
            issues = verified
            token_usage = _merge_token_usage(token_usage, grounding_token_usage)

        return {
            "issues": issues,
            "grounding_rejected": grounding_rejected,
            "token_usage": token_usage,
        }
    except Exception as e:
        raise RuntimeError(f"[{args_tuple[1]}] stages 3-4 worker failed:\n{traceback.format_exc()}") from e


def slide_typo_worker(args_tuple):
    """슬라이드 이미지 기준 오타 검사."""
    try:
        merged_path, model, root, env_vars = args_tuple
        _setup_worker(root, env_vars, model)
        from analyzer import claim_common as cv
        from analyzer.slide_typo_checker import detect_slide_typos

        ctx = _prepare_verification(merged_path)
        merged = ctx["merged"]
        from analyzer.claim_pipeline import _resolve_detector_img_dir

        img_dir = _resolve_detector_img_dir(merged, merged_path)
        typo_model = cv._resolve_stage_model("slide_typo")

        print(f"\n  [{typo_model}] 슬라이드 오타 검사 시작", flush=True)
        typos, api_calls, failures, token_usage = detect_slide_typos(
            ctx["slides"], img_dir=img_dir, max_workers=4, merged_path=merged_path
        )
        print(f"  [{typo_model}] 슬라이드 오타 검사 완료: {len(typos)}건", flush=True)
        return {
            "model": typo_model,
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
