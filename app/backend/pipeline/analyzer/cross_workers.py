"""Worker functions for cross-model verification."""

from __future__ import annotations

import json
import traceback
from pathlib import Path

from .cross_merge import _issue_match_key
from .cross_utils import _empty_token_usage, _merge_token_usage, _setup_worker


def _extract_batch_cache_path(cache_dir: str | None, index: int, first_uid: str, last_uid: str) -> Path | None:
    if not cache_dir:
        return None
    safe_first = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(first_uid or "start"))
    safe_last = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(last_uid or "end"))
    return Path(cache_dir) / f"phase1_extract_batch_{index + 1:03d}_{safe_first}_{safe_last}.json"


def _load_extract_batch_cache(
    cache_dir: str | None,
    index: int,
    batch: list[dict],
    expected_meta: dict,
    *,
    resume: bool,
) -> dict | None:
    if not batch:
        return None
    path = _extract_batch_cache_path(
        cache_dir,
        index,
        str(batch[0].get("utterance_id", "") or ""),
        str(batch[-1].get("utterance_id", "") or ""),
    )
    if not resume or path is None or not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"    ⚠️ claim batch 캐시 로드 실패, 재실행: {path.name} ({e})", flush=True)
        return None
    if payload.get("meta") != expected_meta:
        print(f"    ⚠️ claim batch 캐시 설정 불일치, 재실행: {path.name}", flush=True)
        return None
    print(f"    ↻ claim batch 캐시 사용 [{index + 1}] {path.name}", flush=True)
    return payload.get("payload")


def _save_extract_batch_cache(
    cache_dir: str | None,
    index: int,
    batch: list[dict],
    meta: dict,
    payload: dict,
) -> None:
    if not batch:
        return
    path = _extract_batch_cache_path(
        cache_dir,
        index,
        str(batch[0].get("utterance_id", "") or ""),
        str(batch[-1].get("utterance_id", "") or ""),
    )
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_text(
        json.dumps({"meta": meta, "payload": payload}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp_path.replace(path)


def extract_worker(args_tuple):
    try:
        if len(args_tuple) >= 9:
            merged_path, model, batch_size, root, env_vars, current_date, cache_dir, resume, cache_meta = args_tuple
        else:
            merged_path, model, batch_size, root, env_vars, current_date = args_tuple
            cache_dir, resume, cache_meta = None, False, {}
        _setup_worker(root, env_vars, model)
        import analyzer.claim_pipeline as cv
        from analyzer.claim_extractor import recover_claim_extraction

        ctx = cv.prepare_verification(merged_path, current_date=current_date)
        print(f"\n  [{model}] 1단계: claim 추출 시작", flush=True)

        utterances = ctx["utterances"]
        batches = [utterances[i:i + batch_size] for i in range(0, len(utterances), batch_size)]
        claims_by_batch = []
        api_calls = 0
        token_usage = cv.cc._empty_token_usage()

        for i, batch in enumerate(batches):
            ids = f"{batch[0]['utterance_id']}..{batch[-1]['utterance_id']}"
            batch_meta = {
                **(cache_meta or {}),
                "stage": "phase1_extract_batch",
                "batch_index": i,
                "batch_first_utterance_id": batch[0].get("utterance_id", ""),
                "batch_last_utterance_id": batch[-1].get("utterance_id", ""),
                "batch_utterance_count": len(batch),
            }
            cached = _load_extract_batch_cache(cache_dir, i, batch, batch_meta, resume=resume)
            if cached is not None:
                claims = cached.get("claims", [])
                batch_calls = int(cached.get("api_calls", 0) or 0)
                batch_usage = cached.get("token_usage") or cv.cc._empty_token_usage()
            else:
                print(f"    추출 [{i+1}/{len(batches)}] {ids}", flush=True)
                claims, parse_failed, batch_calls, batch_usage, ok = recover_claim_extraction(
                    batch, ctx["current_date"], ctx["hint"], ctx["slide_ctx"], f"배치 {i+1} {ids}"
                )
                if parse_failed or not ok:
                    raise RuntimeError(f"claim 추출 batch 실패: {ids}")
                _save_extract_batch_cache(
                    cache_dir,
                    i,
                    batch,
                    batch_meta,
                    {"claims": claims, "api_calls": batch_calls, "token_usage": batch_usage},
                )
            claims_by_batch.append((batch, claims))
            api_calls += batch_calls
            token_usage = cv.cc._merge_token_usage(token_usage, batch_usage)

        serialized = []
        for batch, claims in claims_by_batch:
            serialized.append({"batch": batch, "claims": claims})

        total = sum(len(c) for _, c in claims_by_batch)
        print(f"  추출된 claim: {total}개", flush=True)
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
        needs_review = []
        grounding_failures = 0
        slide_recheck_failures = 0
        slide_recheck_status = "skipped_no_issues"
        slide_recheck_reason = "crosscheck 통과 이슈가 없어 슬라이드 문맥 재검증을 건너뜀"
        if issues:
            verified, s_rejected, s_calls, s_failures, slide_token_usage = cv._slide_recheck_all_issues(
                issues, slide_ctx, slides, hint
            )
            slide_rejected = s_rejected
            slide_recheck_failures = s_failures
            slide_recheck_status = "completed_with_failures" if s_failures else "completed"
            slide_recheck_reason = f"슬라이드 문맥 재검증 완료: {len(verified)}건 유지, {len(s_rejected)}건 기각"
            issues = verified
            token_usage = _merge_token_usage(token_usage, slide_token_usage)

        if issues:
            verified, g_rejected, g_needs_review, g_calls, g_failures, grounding_token_usage = cv._ground_verify_all_issues(
                issues, hint, slide_ctx, slides
            )
            grounding_rejected = g_rejected
            needs_review = g_needs_review
            grounding_failures = g_failures
            issues = verified
            token_usage = _merge_token_usage(token_usage, grounding_token_usage)

        return {
            "issues": issues,
            "slide_rejected": slide_rejected,
            "slide_recheck_status": slide_recheck_status,
            "slide_recheck_reason": slide_recheck_reason,
            "slide_recheck_failures": slide_recheck_failures,
            "grounding_rejected": grounding_rejected,
            "needs_review": needs_review,
            "grounding_failures": grounding_failures,
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
