"""
교차 검증 (Cross-Model Verification)

1단계 claim 추출 → gemini-2.5-flash 단일 추출
2단계 claim 판정 → 다중 모델 후보 합집합
3단계 텍스트+문맥 교차검증 → 각 모델이 합집합 이슈를 재판정
4단계 grounding → 통과한 이슈만 primary 모델로 재검증

사용법:
    python -m analyzer.cross_pipeline <merged_clean.json>
    python -m analyzer.cross_pipeline <merged_clean.json> --models gemini-2.5-flash gpt-5.4
    python -m analyzer.cross_pipeline <merged_clean.json> --mode independent
"""

import argparse
import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from .cross_merge import _cluster_contextual_issues, _issue_match_key, rebuild_claim_batches, union_claims, union_issues
from .cross_utils import (
    CLAIM_EXTRACT_MODEL,
    _ROOT,
    _collect_env_vars,
    _empty_token_usage,
    _format_token_summary,
    _merge_token_usage,
    _write_claims_jsonl,
)
from .cross_workers import (
    cross_recheck_worker,
    extract_worker,
    independent_worker,
    judge_worker,
    slide_typo_worker,
    stages_3_4_worker,
)


_CACHE_VERSION = 1


def _safe_cache_name(value: str) -> str:
    text = str(value or "").strip()
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_") or "default"


def _cache_meta(
    stage: str,
    merged_path: str,
    models: list[str],
    current_date: str,
    num_runs: int,
    min_rate: float,
    batch_size: int,
    judge_batch_size: int | None,
    *,
    model: str | None = None,
) -> dict:
    merged_file = Path(merged_path).resolve()
    try:
        merged_mtime_ns = merged_file.stat().st_mtime_ns
    except OSError:
        merged_mtime_ns = None
    return {
        "cache_version": _CACHE_VERSION,
        "stage": stage,
        "merged_path": str(merged_file),
        "merged_mtime_ns": merged_mtime_ns,
        "models": list(models),
        "model": model or "",
        "current_date": current_date,
        "num_runs": num_runs,
        "min_rate": min_rate,
        "batch_size": batch_size,
        "judge_batch_size": judge_batch_size,
        "claim_extract_model": CLAIM_EXTRACT_MODEL,
    }


def _cache_path(cache_dir: str | Path | None, name: str) -> Path | None:
    if not cache_dir:
        return None
    return Path(cache_dir) / f"{name}.json"


def _load_cache(
    cache_dir: str | Path | None,
    name: str,
    expected_meta: dict,
    *,
    resume: bool,
) -> dict | None:
    path = _cache_path(cache_dir, name)
    if not resume or path is None or not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"  ⚠️ 캐시 로드 실패, 재실행: {path} ({e})")
        return None
    if payload.get("meta") != expected_meta:
        print(f"  ⚠️ 캐시 설정 불일치, 재실행: {path.name}")
        return None
    print(f"  ↻ 캐시 사용: {path.name}")
    return payload.get("payload")


def _save_cache(cache_dir: str | Path | None, name: str, meta: dict, payload: dict) -> None:
    path = _cache_path(cache_dir, name)
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_text(
        json.dumps({"meta": meta, "payload": payload}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp_path.replace(path)


# ── 교차 검증 메인 ────────────────────────────────────────

def cross_verify(
    merged_path: str,
    models: list[str],
    num_runs: int,
    min_rate: float,
    batch_size: int,
    env_vars: dict,
    judge_batch_size: int | None = None,
    current_date: str | None = None,
    skip_slide_typo: bool = False,
    resume: bool = False,
    cache_dir: str | None = None,
) -> dict:
    root = str(_ROOT)
    current_date = current_date or datetime.now().strftime("%Y-%m-%d")
    if cache_dir:
        print(f"  중간 캐시: {cache_dir} ({'resume' if resume else 'save-only'})")

    # ── Phase 1: claim 추출 (단일 모델) ──
    print(f"\n{'='*60}")
    print(f"  Phase 1: claim 추출 (단일) — {CLAIM_EXTRACT_MODEL}")
    print(f"{'='*60}")

    extract_args = (merged_path, CLAIM_EXTRACT_MODEL, batch_size, root, env_vars, current_date)
    extract_meta = _cache_meta(
        "phase1_extract",
        merged_path,
        models,
        current_date,
        num_runs,
        min_rate,
        batch_size,
        judge_batch_size,
        model=CLAIM_EXTRACT_MODEL,
    )

    extract_result = _load_cache(cache_dir, "phase1_extract", extract_meta, resume=resume)
    if extract_result is None:
        try:
            extract_result = extract_worker(extract_args)
            _save_cache(cache_dir, "phase1_extract", extract_meta, extract_result)
        except Exception as e:
            raise RuntimeError(f"[{CLAIM_EXTRACT_MODEL}] claim 추출 실패. 재실행 시 --resume을 사용할 수 있습니다.\n{e}") from e

    merged_claims = union_claims([extract_result])
    extract_claim_count = sum(len(item["claims"]) for item in extract_result["claims_by_batch"])

    print(f"\n  ── claim 추출 결과 ──")
    print(f"    [{CLAIM_EXTRACT_MODEL}]: {extract_claim_count}개")
    print(f"    판정 입력 claim 수: {len(merged_claims)}개")

    # utterances 복원 (첫 모델의 배치에서)
    utterances = []
    for item in extract_result["claims_by_batch"]:
        utterances.extend(item["batch"])
    # dedupe utterances by id
    seen_uids = set()
    unique_utts = []
    for u in utterances:
        uid = u.get("utterance_id")
        if uid not in seen_uids:
            seen_uids.add(uid)
            unique_utts.append(u)

    merged_batches = rebuild_claim_batches(merged_claims, unique_utts, judge_batch_size or batch_size)

    # ── Phase 2: claim 판정 (병렬) ──
    print(f"\n{'='*60}")
    print(f"  Phase 2: claim 판정 (병렬) — 합집합 {len(merged_claims)}개")
    print(f"{'='*60}")

    judge_args = [
        (merged_path, model, merged_batches, num_runs, min_rate, root, env_vars, current_date)
        for model in models
    ]

    judge_results = {}
    missing_judge_args = []
    for arg in judge_args:
        model = arg[1]
        meta = _cache_meta(
            "phase2_judge",
            merged_path,
            models,
            current_date,
            num_runs,
            min_rate,
            batch_size,
            judge_batch_size,
            model=model,
        )
        cache_name = f"phase2_judge_{_safe_cache_name(model)}"
        cached = _load_cache(cache_dir, cache_name, meta, resume=resume)
        if cached is not None:
            judge_results[model] = cached
        else:
            missing_judge_args.append((arg, meta, cache_name))

    judge_failures = []
    if missing_judge_args:
        with ProcessPoolExecutor(max_workers=len(missing_judge_args)) as executor:
            futures = {executor.submit(judge_worker, a): (a, meta, cache_name) for a, meta, cache_name in missing_judge_args}
            for future in as_completed(futures):
                arg, meta, cache_name = futures[future]
                model = arg[1]
                try:
                    result = future.result()
                    judge_results[model] = result
                    _save_cache(cache_dir, cache_name, meta, result)
                except Exception as e:
                    print(f"  ❌ [{model}] 판정 실패: {e}")
                    judge_failures.append(model)
    if judge_failures:
        raise RuntimeError(
            "claim 판정 실패: "
            + ", ".join(judge_failures)
            + ". 성공한 모델 결과는 캐시에 저장했습니다. 재실행 시 --resume을 사용하세요."
        )

    # 합집합 + 공통/단독 탐지 분류
    judge_list = [{"model": m, "issues": r["issues"]} for m, r in judge_results.items()]
    unioned, intersected, exclusive = union_issues(judge_list)

    total_union = len(unioned)
    total_exclusive = sum(len(v) for v in exclusive.values())

    print(f"\n  ── 이슈 분류 (합집합 {total_union}건) ──")
    for model, result in judge_results.items():
        print(f"    [{model}]: {len(result['issues'])}건 탐지")

    agreement_label = "양쪽 모두 탐지" if len(models) == 2 else "모든 모델 1차 탐지"
    print(f"    {agreement_label}: {len(intersected)}건 → 자동 확정")
    for model, issues in exclusive.items():
        if issues:
            print(f"    [{model}] 단독 {len(issues)}건")
            for issue in issues:
                print(f"      • {issue.get('claim_text','')[:80]}")

    # ── Phase 3: 합집합 전체 → 두 모델 모두 텍스트+문맥 crosscheck ──
    cross_recheck_verified = []
    cross_recheck_rejected = []
    cross_recheck_inconclusive = []
    cross_recheck_usage_per_model = {m: _empty_token_usage() for m in models}
    if total_union > 0 and len(models) >= 2:
        print(f"\n{'='*60}")
        print(f"  Phase 3: 텍스트+문맥 교차검증 — 합집합 {total_union}건을 두 모델이 재검증")
        print(f"{'='*60}")

        recheck_args = [(unioned, merged_path, m, root, env_vars, current_date) for m in models]
        cross_recheck_by_model = {}
        missing_recheck_args = []
        for arg in recheck_args:
            model = arg[2]
            meta = _cache_meta(
                "phase3_cross_recheck",
                merged_path,
                models,
                current_date,
                num_runs,
                min_rate,
                batch_size,
                judge_batch_size,
                model=model,
            )
            cache_name = f"phase3_cross_recheck_{_safe_cache_name(model)}"
            cached = _load_cache(cache_dir, cache_name, meta, resume=resume)
            if cached is not None:
                cross_recheck_by_model[cached["model"]] = cached["verdicts"]
                cross_recheck_usage_per_model[cached["model"]] = _merge_token_usage(
                    cross_recheck_usage_per_model[cached["model"]],
                    cached.get("token_usage"),
                )
            else:
                missing_recheck_args.append((arg, meta, cache_name))

        recheck_failures = []
        if missing_recheck_args:
            with ProcessPoolExecutor(max_workers=len(missing_recheck_args)) as executor:
                futures = {executor.submit(cross_recheck_worker, a): (a, meta, cache_name) for a, meta, cache_name in missing_recheck_args}
                for future in as_completed(futures):
                    arg, meta, cache_name = futures[future]
                    model = arg[2]
                    try:
                        result = future.result()
                        cross_recheck_by_model[result["model"]] = result["verdicts"]
                        cross_recheck_usage_per_model[result["model"]] = _merge_token_usage(
                            cross_recheck_usage_per_model[result["model"]],
                            result.get("token_usage"),
                        )
                        _save_cache(cache_dir, cache_name, meta, result)
                    except Exception as e:
                        print(f"  ❌ [{model}] 교차 재검증 실패: {e}")
                        recheck_failures.append(model)
        if recheck_failures:
            raise RuntimeError(
                "텍스트+문맥 교차검증 실패: "
                + ", ".join(recheck_failures)
                + ". 성공한 모델 결과는 캐시에 저장했습니다. 재실행 시 --resume을 사용하세요."
            )

        for issue in unioned:
            key = _issue_match_key(issue)
            verdict_rows = []
            reasons = []
            all_agree = True
            any_inconclusive = False
            for model in models:
                model_result = cross_recheck_by_model.get(model, {}).get(
                    key,
                    {"verdict": "inconclusive", "reason": "crosscheck 결과 없음", "resolved_model": model},
                )
                verdict = model_result.get("verdict", "inconclusive")
                reason = model_result.get("reason", "")
                verdict_rows.append({"model": model, **model_result})
                if verdict != "agree":
                    all_agree = False
                if verdict == "inconclusive":
                    any_inconclusive = True
                reasons.append(f"[{model}] {verdict}: {reason}")

            issue["crosscheck_details"] = verdict_rows
            if all_agree:
                issue["cross_model_agreement"] = len(models)
                cross_recheck_verified.append(issue)
            else:
                issue["cross_recheck"] = None if any_inconclusive else False
                issue["rejection_stage"] = "텍스트+문맥 교차검증"
                issue["rejection_reason"] = " / ".join(reasons)
                if any_inconclusive:
                    issue["rejection_reason_code"] = "crosscheck_inconclusive"
                    cross_recheck_inconclusive.append(issue)
                else:
                    issue["rejection_reason_code"] = "model_disagreement"
                    cross_recheck_rejected.append(issue)

        if cross_recheck_verified:
            cross_recheck_verified = _cluster_contextual_issues(cross_recheck_verified)
            print(f"\n    ✅ 텍스트+문맥 교차검증 통과: {len(cross_recheck_verified)}건")
        if cross_recheck_rejected:
            print(f"    ❌ 텍스트+문맥 교차검증 거부: {len(cross_recheck_rejected)}건")
        if cross_recheck_inconclusive:
            print(f"    ⚠️ 텍스트+문맥 교차검증 불확실: {len(cross_recheck_inconclusive)}건")

    all_confirmed = cross_recheck_verified

    # ── Phase 4: primary 모델 grounding ──
    primary = models[0]
    if all_confirmed:
        print(f"\n{'='*60}")
        print(f"  Phase 4: grounding — [{primary}]")
        print(f"{'='*60}")

        final_meta = _cache_meta(
            "phase4_slide_grounding",
            merged_path,
            models,
            current_date,
            num_runs,
            min_rate,
            batch_size,
            judge_batch_size,
            model=primary,
        )
        final = _load_cache(cache_dir, "phase4_slide_grounding", final_meta, resume=resume)
        if final is None:
            with ProcessPoolExecutor(max_workers=1) as executor:
                future = executor.submit(stages_3_4_worker,
                                         (merged_path, primary, all_confirmed, root, env_vars, current_date))
                final = future.result()
            _save_cache(cache_dir, "phase4_slide_grounding", final_meta, final)
    else:
        final = {
            "issues": [],
            "slide_rejected": [],
            "slide_recheck_status": "skipped_no_issues",
            "slide_recheck_reason": "crosscheck 통과 이슈가 없어 슬라이드 문맥 재검증을 건너뜀",
            "slide_recheck_failures": 0,
            "grounding_rejected": [],
            "needs_review": [],
            "grounding_failures": 0,
            "token_usage": _empty_token_usage(),
        }

    # ── 별도: 슬라이드 오타 검사 ──
    slide_typo_result = {
        "model": primary,
        "slide_typos": [],
        "api_calls": 0,
        "failures": 0,
        "token_usage": _empty_token_usage(),
    }
    if skip_slide_typo:
        print(f"\n  ⏭  슬라이드 오타 검사 스킵 (--skip-slide-typo)")
        slide_typo_result["skipped"] = True
        slide_typo_result["skip_reason"] = "skip_slide_typo"
    else:
        typo_meta = _cache_meta(
            "slide_typo",
            merged_path,
            models,
            current_date,
            num_runs,
            min_rate,
            batch_size,
            judge_batch_size,
            model=primary,
        )
        slide_typo_result = _load_cache(cache_dir, "slide_typo", typo_meta, resume=resume)
        if slide_typo_result is None:
            try:
                slide_typo_result = slide_typo_worker((merged_path, primary, root, env_vars, current_date))
                _save_cache(cache_dir, "slide_typo", typo_meta, slide_typo_result)
            except Exception as e:
                print(f"  ❌ [{primary}] 슬라이드 오타 검사 실패: {e}")
                slide_typo_result = {
                    "model": primary,
                    "slide_typos": [],
                    "api_calls": 0,
                    "failures": 1,
                    "token_usage": _empty_token_usage(),
                }

    # ── 결과 조합 ──
    for issue in final["slide_rejected"]:
        issue.setdefault("rejection_stage", "슬라이드 재검증")
        issue.setdefault("rejection_reason_code", "slide_context_rejected")
    for issue in final["grounding_rejected"]:
        issue.setdefault("rejection_stage", "grounding")
        issue.setdefault("rejection_reason_code", "grounding_rejected")
    for issue in final.get("needs_review", []):
        status = str(issue.get("grounding_status") or "").strip() or "grounding_unavailable"
        issue.setdefault("review_stage", "grounding")
        issue.setdefault("review_reason_code", status)

    all_rejected = final["slide_rejected"] + final["grounding_rejected"] + cross_recheck_rejected + cross_recheck_inconclusive
    token_usage_per_model = {m: _empty_token_usage() for m in models}
    extract_token_usage = extract_result.get("token_usage")
    for m in models:
        token_usage_per_model[m] = _merge_token_usage(
            token_usage_per_model[m],
            judge_results.get(m, {}).get("token_usage"),
            cross_recheck_usage_per_model.get(m),
        )
    token_usage_per_model[primary] = _merge_token_usage(
        token_usage_per_model.get(primary),
        final.get("token_usage"),
        slide_typo_result.get("token_usage"),
    )
    total_token_usage = _merge_token_usage(extract_token_usage, *(token_usage_per_model.values()))
    result = {
        "mode": "cross_verification",
        "verification_date": current_date,
        "models": models,
        "primary_model": primary,
        "resume_enabled": resume,
        "resume_cache_dir": str(cache_dir or ""),
        "claim_extract_model": CLAIM_EXTRACT_MODEL,
        "claims_per_model": {CLAIM_EXTRACT_MODEL: extract_claim_count},
        "merged_claims_count": len(merged_claims),
        "merged_claims": merged_claims,
        "issues_per_model": {m: len(r["issues"]) for m, r in judge_results.items()},
        "issue_detection_total": sum(len(r["issues"]) for r in judge_results.values()),
        "issue_union_count": total_union,
        "exclusive_count": total_exclusive,
        "intersected_count": len(intersected),
        "cross_recheck_verified_count": len(cross_recheck_verified),
        "cross_recheck_inconclusive_count": len(cross_recheck_inconclusive),
        "confirmed_count": len(all_confirmed),
        "issues": final["issues"],
        "slide_typos": slide_typo_result.get("slide_typos", []),
        "slide_typo_status": "skipped" if slide_typo_result.get("skipped") else "completed",
        "slide_typo_skip_reason": slide_typo_result.get("skip_reason", ""),
        "crosscheck_rejected_issues": cross_recheck_rejected,
        "crosscheck_inconclusive_issues": cross_recheck_inconclusive,
        "slide_recheck_status": final.get("slide_recheck_status", "completed"),
        "slide_recheck_reason": final.get("slide_recheck_reason", ""),
        "slide_rejected_issues": final["slide_rejected"],
        "grounding_rejected_issues": final["grounding_rejected"],
        "needs_review_issues": final.get("needs_review", []),
        "rejected_issues": all_rejected,
        "claim_extract_token_usage": extract_token_usage,
        "token_usage_per_model": token_usage_per_model,
        "token_usage": total_token_usage,
        "overall_assessment": {
            "has_issues": len(final["issues"]) > 0,
            "total_issues": len(final["issues"]),
            "severity_breakdown": _count_severity(final["issues"]),
        },
        "crosscheck_filtered": len(cross_recheck_rejected),
        "crosscheck_inconclusive_filtered": len(cross_recheck_inconclusive),
        "slide_recheck_filtered": len(final["slide_rejected"]),
        "grounding_filtered": len(final["grounding_rejected"]),
        "needs_review_count": len(final.get("needs_review", [])),
        "slide_recheck_failures": int(final.get("slide_recheck_failures", 0) or 0),
        "grounding_failures": int(final.get("grounding_failures", 0) or 0),
        "slide_typo_failures": int(slide_typo_result.get("failures", 0) or 0),
    }

    # ── 로그 기록 ──
    try:
        from .verification_logger import log_cross_result
        video_name = Path(merged_path).stem.replace("_merged_clean", "").replace("_merged", "")
        log_cross_result(video_name, result, judge_results)
    except ImportError:
        pass
    except Exception as e:
        print(f"  ⚠️ 로그 기록 실패: {e}")

    return result


def _count_severity(issues):
    bd = {}
    for i in issues:
        s = i.get("severity", "minor")
        bd[s] = bd.get(s, 0) + 1
    return bd


# ── 기존 독립 실행 ──────────────────────────────────────

# ── 출력 ────────────────────────────────────────────────

def print_cross_result(result: dict):
    print(f"\n\n{'='*60}")
    print(f"  📊 교차 검증 결과")
    print(f"{'='*60}\n")

    models = result["models"]
    ipm = result.get("issues_per_model", {})
    inter = result.get("intersected_count", 0)
    xr = result.get("cross_recheck_verified_count", 0)
    xi = result.get("cross_recheck_inconclusive_count", 0)
    confirmed = result.get("confirmed_count", 0)

    print(f"  모델: {', '.join(models)}")
    extract_model = result.get("claim_extract_model", CLAIM_EXTRACT_MODEL)
    extract_counts = result.get("claims_per_model", {})
    print(f"  claim 추출: [{extract_model}] {extract_counts.get(extract_model, result['merged_claims_count'])}개")
    print(f"  판정 입력 claim 수: {result['merged_claims_count']}개")
    print()
    print(f"  ── 판정 흐름 ──")
    for m, cnt in ipm.items():
        print(f"    [{m}] {cnt}건 탐지")
    detection_total = result.get("issue_detection_total", sum(ipm.values()))
    issue_union = result.get("issue_union_count", 0)
    exclusive = result.get("exclusive_count", max(issue_union - inter, 0))
    agreement_label = "양쪽 1차 탐지" if len(models) == 2 else "모든 모델 1차 탐지"
    print(f"    총 탐지 수: {detection_total}건")
    print(f"    이슈 합집합: {issue_union}건")
    print(f"    ├─ {agreement_label}: {inter}건")
    print(f"    ├─ 단독 1차 탐지: {exclusive}건")
    print(f"    ├─ 텍스트+문맥 교차검증 통과: {xr}건")
    print(f"    ├─ 텍스트+문맥 교차검증 불확실: {xi}건")
    print(f"    └─ 확정 이슈 (grounding 진입): {confirmed}건")

    sr = result.get("slide_recheck_filtered", 0)
    gr = result.get("grounding_filtered", 0)
    nr = result.get("needs_review_count", 0)
    if sr or gr or nr:
        print(f"    → 슬라이드 재검증 기각: {sr}건, grounding 기각: {gr}건, 리뷰 필요: {nr}건")
    sf = result.get("slide_recheck_failures", 0)
    gf = result.get("grounding_failures", 0)
    if sf or gf:
        print(f"    → 검사 실패: 슬라이드 재검증 {sf}건, grounding {gf}건")
    print()

    token_usage_per_model = result.get("token_usage_per_model", {})
    extract_usage = result.get("claim_extract_token_usage", {})
    if token_usage_per_model:
        print(f"  ── 토큰 사용량 ──")
        if extract_usage:
            print(f"    [claim 추출:{extract_model}] {_format_token_summary({'total': extract_usage.get('total', {})}) if 'total' in extract_usage else _format_token_summary(extract_usage)}")
        for model in models:
            usage = token_usage_per_model.get(model, {})
            print(f"    [{model}] {_format_token_summary(usage)}")
        print(f"    [전체] {_format_token_summary(result.get('token_usage', {}))}")
        print()

    issues = result.get("issues", [])
    if issues:
        print(f"  ✅ 최종 확정 이슈: {len(issues)}건")
        for i, issue in enumerate(issues):
            src = "양쪽" if issue.get("cross_model_agreement", 0) >= 2 else f"교차검증({issue.get('cross_recheck_model','?')} 동의)"
            print(f"    [{i+1}] {issue['type']} sev={issue['severity']} conf={issue.get('confidence',0):.2f} ({src})")
            print(f"        {issue.get('claim_text','')[:100]}")
            print(f"        → {issue.get('issue','')[:100]}")
    else:
        print(f"  최종 이슈: 0건")

    crosscheck_rejected = result.get("crosscheck_rejected_issues", [])
    crosscheck_inconclusive = result.get("crosscheck_inconclusive_issues", [])
    grounding_rejected = result.get("grounding_rejected_issues", [])
    needs_review = result.get("needs_review_issues", [])

    if crosscheck_rejected:
        print(f"\n  ❌ 텍스트+문맥 교차검증 기각: {len(crosscheck_rejected)}건")
        for i, issue in enumerate(crosscheck_rejected):
            reason = issue.get("rejection_reason", "텍스트+문맥 교차검증에서 유지되지 않음")
            print(f"    [{i+1}] {issue['type']} sev={issue['severity']}")
            print(f"        claim: {issue.get('claim_text','')[:100]}")
            print(f"        issue: {issue.get('issue','')[:100]}")
            print(f"        사유: {reason[:120]}")

    if crosscheck_inconclusive:
        print(f"\n  ⚠️ 텍스트+문맥 교차검증 불확실: {len(crosscheck_inconclusive)}건")
        for i, issue in enumerate(crosscheck_inconclusive):
            reason = issue.get("rejection_reason", "텍스트+문맥 교차검증에서 확정 판단 실패")
            print(f"    [{i+1}] {issue['type']} sev={issue['severity']}")
            print(f"        claim: {issue.get('claim_text','')[:100]}")
            print(f"        issue: {issue.get('issue','')[:100]}")
            print(f"        사유: {reason[:120]}")

    if grounding_rejected:
        print(f"\n  ❌ grounding 기각: {len(grounding_rejected)}건")
        for i, issue in enumerate(grounding_rejected):
            reason = issue.get("grounding_reason", "")
            print(f"    [{i+1}] {issue['type']} sev={issue['severity']}")
            print(f"        claim: {issue.get('claim_text','')[:100]}")
            print(f"        issue: {issue.get('issue','')[:100]}")
            if reason:
                print(f"        사유: {reason[:120]}")

    if needs_review:
        print(f"\n  ⚠️ 리뷰 필요: {len(needs_review)}건")
        for i, issue in enumerate(needs_review):
            reason = issue.get("grounding_reason", "")
            status = issue.get("grounding_status", "needs_review")
            print(f"    [{i+1}] {issue['type']} sev={issue['severity']} status={status}")
            print(f"        claim: {issue.get('claim_text','')[:100]}")
            print(f"        issue: {issue.get('issue','')[:100]}")
            if reason:
                print(f"        사유: {reason[:120]}")

    slide_typos = result.get("slide_typos", [])
    if slide_typos:
        print(f"\n  ✏️ 슬라이드 오타: {len(slide_typos)}건")
        for i, typo in enumerate(slide_typos[:10], 1):
            print(
                f"    [{i}] 슬라이드 {typo.get('slide_number', '?')} | "
                f"{typo.get('problematic_text', '')} -> {typo.get('corrected_text', '')}"
            )
    if result.get("slide_typo_failures", 0):
        print(f"  ⚠️ 슬라이드 오타 검사 실패: {result['slide_typo_failures']}건")


# ── main ────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="교차 검증 / 모델 비교")
    parser.add_argument("merged_path", help="merged_clean.json 경로")
    parser.add_argument("--models", nargs="+",
                        default=["gemini-2.5-flash", "gpt-5.4"],
                        help="판정/검증에 사용할 모델 (기본: gemini-2.5-flash gpt-5.4)")
    parser.add_argument("--mode", choices=["cross", "independent"], default="cross",
                        help="cross=교차검증(기본), independent=독립비교")
    parser.add_argument("--num-runs", type=int, default=1, help="1차 judge 반복 횟수 (기본 1)")
    parser.add_argument("--min-rate", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--date", default=None, help="검증 기준 날짜 (YYYY-MM-DD)")
    parser.add_argument("--skip-slide-typo", action="store_true", help="슬라이드 오타 검사를 건너뛰고 claim verifier만 실행")
    parser.add_argument("--resume", action="store_true", help="중간 캐시를 재사용해 실패 지점부터 재개")
    parser.add_argument("--cache-dir", default=None, help="중간 캐시 저장 디렉토리")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if not Path(args.merged_path).exists():
        print(f"❌ 파일 없음: {args.merged_path}")
        sys.exit(1)

    env_vars = _collect_env_vars()

    if args.mode == "cross":
        result = cross_verify(
            args.merged_path, args.models, args.num_runs,
            args.min_rate, args.batch_size, env_vars,
            current_date=args.date,
            skip_slide_typo=args.skip_slide_typo,
            resume=args.resume,
            cache_dir=args.cache_dir or (str(Path(args.output).with_suffix("")) + "_cache" if args.output else None),
        )
        print_cross_result(result)
    else:
        # 기존 독립 비교 모드
        worker_args = [
            (args.merged_path, m, args.num_runs, args.min_rate, str(_ROOT), env_vars)
            for m in args.models
        ]
        results = {}
        with ProcessPoolExecutor(max_workers=len(args.models)) as executor:
            futures = {executor.submit(independent_worker, a): a[1] for a in worker_args}
            for future in as_completed(futures):
                model = futures[future]
                try:
                    results[model] = future.result()
                except Exception as e:
                    print(f"  ❌ [{model}] 실패: {e}")
        result = {m: results[m] for m in args.models if m in results}

    if args.output:
        claims_log_path = None
        if args.mode == "cross":
            claims_log_path = _write_claims_jsonl(result.get("merged_claims", []), args.output)
        if claims_log_path:
            result["claims_log_path"] = claims_log_path
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\n💾 결과 저장: {args.output}")
        if claims_log_path:
            print(f"💾 claim 저장: {claims_log_path}")


if __name__ == "__main__":
    main()
