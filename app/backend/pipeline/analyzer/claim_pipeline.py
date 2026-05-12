"""Claim verification pipeline orchestration."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from . import claim_common as cc
from .claim_reporting import format_verification_report, write_claims_jsonl

# ── 상수 re-export (cross_workers 등 외부 모듈이 cv.BATCH_SIZE 로 접근) ──
BATCH_SIZE = cc.BATCH_SIZE
VERIFIER_MODEL = cc.VERIFIER_MODEL
VERIFIER_TEMPERATURE = cc.VERIFIER_TEMPERATURE
VERIFIER_REQUIRE_COMPLETE = cc.VERIFIER_REQUIRE_COMPLETE


# ── 단계별 공개 함수 (교차 검증용) ────────────────────────

def prepare_verification(merged_path: str, current_date: str = None):
    """merged.json 로드 + 공통 데이터 반환. 교차 검증에서 양쪽 모델이 공유."""
    if current_date is None:
        current_date = datetime.now().strftime("%Y-%m-%d")
    with open(merged_path, "r", encoding="utf-8") as f:
        merged = json.load(f)
    slides = merged.get("slides", [])
    domain, sub_domain = cc._resolve_domain_fields(merged)
    hint = cc._get_domain_hint(domain, sub_domain)
    utterances = cc._collect_utterances(slides)
    slide_ctx = cc._build_slide_context_map(slides)
    return {
        "merged": merged,
        "slides": slides,
        "domain": domain,
        "sub_domain": sub_domain,
        "hint": hint,
        "utterances": utterances,
        "slide_ctx": slide_ctx,
        "current_date": current_date,
    }


def extract_claims_only(
    utterances: list[dict], current_date: str, hint: dict,
    slide_ctx: dict, batch_size: int = BATCH_SIZE,
) -> tuple[list[tuple], int, dict]:
    """1단계: claim 추출."""
    from analyzer.claim_extractor import extract_claims_only as _run
    return _run(utterances, current_date, hint, slide_ctx, batch_size=batch_size)


def judge_claims_only(
    all_claims_by_batch: list[tuple], current_date: str, hint: dict,
    slide_ctx: dict, num_runs: int = 1, min_detection_rate: float = 0.5,
    log_prefix: str = "",
) -> tuple[list[dict], int, dict]:
    """2단계: claim 판정 (N회 반복 + 합의)."""
    from analyzer.claim_verifier import judge_claims_only as _run
    return _run(all_claims_by_batch, current_date, hint, slide_ctx,
                num_runs=num_runs, min_detection_rate=min_detection_rate,
                log_prefix=log_prefix)


def judge_issue_candidates_only(
    all_claims_by_batch: list[tuple], current_date: str, hint: dict,
    slide_ctx: dict, log_prefix: str = "",
) -> tuple[list[dict], int, dict]:
    """crosscheck 전 1차 issue 후보 judge만 실행."""
    from analyzer.claim_verifier import judge_issue_candidates_only as _run
    return _run(all_claims_by_batch, current_date, hint, slide_ctx, log_prefix=log_prefix)


def judge_single_claim(issue: dict, ctx: dict) -> tuple[dict, dict]:
    """3단계: 교차 모델 단건 판정."""
    from analyzer.claim_crosscheck import judge_single_claim as _run
    return _run(issue, ctx)


def judge_claim_batch(issues: list[dict], ctx: dict) -> tuple[dict[str, dict], dict]:
    """3단계: 같은 문맥 이슈 묶음 판정."""
    from analyzer.claim_crosscheck import judge_claim_batch as _run
    return _run(issues, ctx)


# ── 내부 헬퍼 (verify_lecture_content 전용) ──────────────

def _resolve_detector_img_dir(merged: dict, merged_path: str | Path | None = None) -> Optional[str]:
    detector_log = str(merged.get("source_detector_log", "") or "").strip()
    candidates = []
    if detector_log:
        candidates.append(Path(detector_log).parent)
    if merged_path:
        candidates.append(Path(merged_path).resolve().parent / "slides")
    for img_dir in candidates:
        if img_dir.is_dir() and any(img_dir.glob("slide_*_base.*")):
            return str(img_dir)
        if img_dir.is_dir() and any(img_dir.glob("slide_*_start.*")):
            return str(img_dir)
    return None


# cross_workers 가 cv._resolve_stage_model / cv._ground_verify_all_issues 로 접근
_resolve_stage_model = cc._resolve_stage_model


def _ground_verify_all_issues(
    issues: list[dict],
    hint: dict,
    slide_ctx: dict | None = None,
    slides: list[dict] | None = None,
    max_workers: int = 4,
) -> tuple[list[dict], list[dict], int, int, dict]:
    from analyzer.claim_grounding import ground_verify_all_issues
    return ground_verify_all_issues(issues, hint, slide_ctx, slides, max_workers=max_workers)


# ── 메인 진입점 ──────────────────────────────────────────

def verify_lecture_content(
    merged_path: str, current_date: str = None, num_runs: int = 1,
    min_detection_rate: float = 0.5, batch_size: int = BATCH_SIZE, max_workers: int = 4,
) -> dict:
    if current_date is None:
        current_date = datetime.now().strftime("%Y-%m-%d")

    with open(merged_path, "r", encoding="utf-8") as f:
        merged = json.load(f)

    slides = merged.get("slides", [])
    domain, sub_domain = cc._resolve_domain_fields(merged)
    hint = cc._get_domain_hint(domain, sub_domain)

    utterances = cc._collect_utterances(slides)
    slide_ctx = cc._build_slide_context_map(slides)
    total_chars = sum(len(u["text"]) for u in utterances)
    slides_with_text = sum(1 for s in slide_ctx.values() if s.get("slide_text"))

    print(f"\n  2단계 claim 파이프라인 검증 (슬라이드 맥락 포함)")
    print(f"  도메인: {hint['label']} (domain={domain}, sub_domain={sub_domain})")
    print(f"  발화: {len(utterances)}개, {total_chars:,}자")
    print(f"  슬라이드: {len(slide_ctx)}개 (텍스트 있음: {slides_with_text}개)")
    print(f"  기본 모델: {VERIFIER_MODEL}, temperature: {VERIFIER_TEMPERATURE}")
    print(
        "  단계별 모델:"
        f" extract={cc._resolve_stage_model('extract')}"
        f" | judge={cc._resolve_stage_model('judge')}"
        f" | slide_typo={cc._resolve_stage_model('slide_typo')}"
        f" | grounding={cc._resolve_stage_model('grounding')}"
    )

    # ── 1단계: claim 추출 (1회) ──
    from analyzer.claim_extractor import recover_claim_extraction

    print(f"\n  ── 1단계: claim 추출 (1회) ──")
    batches = [utterances[i:i + batch_size] for i in range(0, len(utterances), batch_size)]
    all_claims_by_batch = []
    extract_api_calls, extract_parse_failures, extract_failed_calls = 0, 0, 0
    extract_token_usage = cc._empty_token_usage()

    for i, batch in enumerate(batches):
        ids = f"{batch[0]['utterance_id']}..{batch[-1]['utterance_id']}"
        print(f"    추출 [{i+1}/{len(batches)}] {ids}")
        claims, parse_failed, api_calls, token_usage, ok = recover_claim_extraction(
            batch, current_date, hint, slide_ctx, f"배치 {i+1} {ids}"
        )
        all_claims_by_batch.append((batch, claims))
        extract_api_calls += api_calls
        extract_token_usage = cc._merge_token_usage(extract_token_usage, token_usage)
        if parse_failed:
            extract_parse_failures += 1
        if not ok and parse_failed:
            extract_failed_calls += 1

    total_claims = sum(len(c) for _, c in all_claims_by_batch)
    print(f"  추출된 claim: {total_claims}개")

    # ── 2단계: claim 판정 (N회 반복 + 합의) ──
    from analyzer.claim_verifier import recover_claim_judgement

    print(f"\n  ── 2단계: claim 판정 ({num_runs}회 반복) ──")
    run_results = []
    for run in range(num_runs):
        print(f"\n  판정 [{run+1}/{num_runs}]")
        run_issues = []
        run_token_usage = cc._empty_token_usage()
        for batch_idx, (batch, claims) in enumerate(all_claims_by_batch, start=1):
            if not claims:
                continue
            batch_map = {u["utterance_id"]: u for u in batch}
            ids = f"{batch[0]['utterance_id']}..{batch[-1]['utterance_id']}"
            issues, parse_failed, api_calls, token_usage, ok = recover_claim_judgement(
                claims, batch, current_date, hint, batch_map, slide_ctx,
                f"판정 {run+1} 배치 {batch_idx} {ids}",
            )
            run_issues.extend(issues)
            run_token_usage = cc._merge_token_usage(run_token_usage, token_usage)
        run_results.append(cc._make_result(
            cc._dedupe_issues(run_issues), 0, token_usage=run_token_usage,
        ))

    print(f"\n  📊 통합 중 (합의 기준 {min_detection_rate:.0%})...")
    result = cc.merge_multiple_runs(run_results, num_runs, min_detection_rate)
    result["api_calls"] = result.get("api_calls", 0) + extract_api_calls
    result["parse_failures"] = result.get("parse_failures", 0) + extract_parse_failures
    result["failed_calls"] = result.get("failed_calls", 0) + extract_failed_calls
    result["token_usage"] = cc._merge_token_usage(result.get("token_usage"), extract_token_usage)
    result["total_claims_extracted"] = total_claims
    all_extracted_claims = []
    for _batch, claims in all_claims_by_batch:
        all_extracted_claims.extend(claims)
    result["extracted_claims"] = all_extracted_claims

    img_dir = _resolve_detector_img_dir(merged, merged_path)

    # ── 3단계: grounding 검증 (Google Search로 재검증) ──
    pre_grounding_issues = list(result.get("issues", []))
    grounding_rejected = []
    if pre_grounding_issues:
        from analyzer.claim_grounding import ground_verify_all_issues

        verified, grounding_rejected, grounding_calls, grounding_failures, grounding_token_usage = ground_verify_all_issues(
            pre_grounding_issues, hint, slide_ctx, slides, max_workers=max_workers,
        )
        result["issues"] = verified
        result["api_calls"] = result.get("api_calls", 0) + grounding_calls
        result["grounding_failures"] = grounding_failures
        result["token_usage"] = cc._merge_token_usage(result.get("token_usage"), grounding_token_usage)
        result["overall_assessment"] = {
            "has_issues": len(verified) > 0,
            "total_issues": len(verified),
            "severity_breakdown": {
                s: sum(1 for i in verified if i.get("severity") == s)
                for s in ("critical", "major", "minor")
            },
        }
    else:
        result["overall_assessment"]["total_issues"] = 0
        result["overall_assessment"]["has_issues"] = False
        result["grounding_failures"] = 0

    # ── 슬라이드 오타 검사 ──
    from analyzer.slide_typo_checker import detect_slide_typos

    slide_typos, typo_calls, slide_typo_failures, typo_token_usage = detect_slide_typos(
        slides, img_dir=img_dir, max_workers=max_workers, merged_path=merged_path,
    )
    result["slide_typos"] = slide_typos
    result["api_calls"] = result.get("api_calls", 0) + typo_calls
    result["slide_typo_failures"] = slide_typo_failures
    result["token_usage"] = cc._merge_token_usage(result.get("token_usage"), typo_token_usage)

    # 단계별 기각 이슈 저장
    result["grounding_rejected_issues"] = grounding_rejected
    result["rejected_issues"] = grounding_rejected
    result["grounding_filtered"] = len(grounding_rejected)
    result["is_complete"] = not any([
        result.get("parse_failures", 0),
        result.get("failed_calls", 0),
        result.get("grounding_failures", 0),
        result.get("slide_typo_failures", 0),
    ])
    if not result["is_complete"]:
        result["completion_warning"] = (
            f"incomplete_result(parse_failures={result.get('parse_failures', 0)}, "
            f"failed_calls={result.get('failed_calls', 0)}, "
            f"grounding_failures={result.get('grounding_failures', 0)}, "
            f"slide_typo_failures={result.get('slide_typo_failures', 0)})"
        )

    result["metadata"] = {
        "total_slides": len(slides), "total_utterances": len(utterances),
        "total_characters": total_chars, "slides_with_text": slides_with_text,
        "verification_date": current_date,
        "total_duration_formatted": merged.get("total_duration_formatted", ""),
        "domain": domain, "sub_domain": sub_domain, "domain_hint": hint["label"],
        "num_runs": num_runs, "batch_size": batch_size, "max_workers": max_workers,
        "prompt_version": "v8.2-claim-slide-context",
        "verifier_model": VERIFIER_MODEL, "verifier_temperature": VERIFIER_TEMPERATURE,
        "verifier_extract_model": cc._resolve_stage_model("extract"),
        "verifier_judge_model": cc._resolve_stage_model("judge"),
        "verifier_slide_typo_model": cc._resolve_stage_model("slide_typo"),
        "verifier_grounding_model": cc._resolve_stage_model("grounding"),
        "total_claims_extracted": result.get("total_claims_extracted", 0),
        "slide_typo_count": len(result.get("slide_typos", [])),
        "token_usage_total": result.get("token_usage", {}).get("total", {}),
    }
    if VERIFIER_REQUIRE_COMPLETE and not result.get("is_complete", True):
        raise RuntimeError(result.get("completion_warning", "verification_incomplete"))
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="content_verifier v8 단독 실행")
    parser.add_argument("merged_json", help="검증할 merged.json 경로")
    parser.add_argument("--runs", type=int, default=1, help="1차 judge 반복 횟수 (기본 1)")
    parser.add_argument("--min-rate", type=float, default=0.5, help="합의 최소 비율")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--date", default=None, help="검증 기준 날짜 (YYYY-MM-DD)")
    parser.add_argument("--output", default=None, help="결과 JSON 저장 경로")
    args = parser.parse_args()

    merged_path = Path(args.merged_json).resolve()
    if not merged_path.exists():
        print(f"파일 없음: {merged_path}")
        raise SystemExit(1)

    result = verify_lecture_content(
        merged_path=str(merged_path),
        current_date=args.date,
        num_runs=args.runs,
        min_detection_rate=args.min_rate,
        batch_size=args.batch_size,
        max_workers=args.max_workers,
    )

    if args.output:
        out_json = Path(args.output).resolve()
    else:
        stem = merged_path.stem.replace("_merged", "")
        out_json = merged_path.with_name(f"{stem}_content_verification_v8.json")

    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    claims_log_path = write_claims_jsonl(result, out_json)
    if claims_log_path:
        result["claims_log_path"] = claims_log_path
        out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    out_report = out_json.with_suffix(".txt")
    out_report.write_text(format_verification_report(result), encoding="utf-8")

    assessment = result.get("overall_assessment", {})
    print(f"\n저장: {out_json}")
    if claims_log_path:
        print(f"저장: {claims_log_path}")
    print(f"저장: {out_report}")
    print(f"이슈: {assessment.get('total_issues', 0)}건")
