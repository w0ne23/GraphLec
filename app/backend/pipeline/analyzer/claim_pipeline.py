"""Compatibility facade for analyzer claim stages.

The production verifier is orchestrated by ``cross_pipeline.cross_verify``.
This module only keeps small public wrappers used by worker processes and old
CLI callers.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

from . import claim_common as cc

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
    max_workers: int | None = None,
) -> tuple[list[tuple], int, dict]:
    """1단계: claim 추출."""
    from analyzer.claim_extractor import extract_claims_only as _run
    return _run(
        utterances,
        current_date,
        hint,
        slide_ctx,
        batch_size=batch_size,
        max_workers=max_workers,
    )


def judge_claims_only(
    all_claims_by_batch: list[tuple], current_date: str, hint: dict,
    num_runs: int = 1, min_detection_rate: float = 0.5,
    log_prefix: str = "",
    context_mode: str | None = None,
    max_workers: int | None = None,
) -> tuple[list[dict], int, dict]:
    """2단계: claim 판정 (N회 반복 + 합의)."""
    from analyzer.claim_verifier import judge_claims_only as _run
    return _run(
        all_claims_by_batch,
        current_date,
        hint,
        num_runs=num_runs,
        min_detection_rate=min_detection_rate,
        log_prefix=log_prefix,
        context_mode=context_mode,
        max_workers=max_workers,
    )


def judge_single_claim(issue: dict, ctx: dict) -> tuple[dict, dict]:
    """3단계: 교차 모델 단건 판정."""
    from analyzer.claim_crosscheck import judge_single_claim as _run
    return _run(issue, ctx)


def judge_claim_batch(issues: list[dict], ctx: dict) -> tuple[dict[str, dict], dict]:
    """3단계: 같은 문맥 이슈 묶음 판정."""
    from analyzer.claim_crosscheck import judge_claim_batch as _run
    return _run(issues, ctx)


# ── 내부 헬퍼 (verify_lecture_content 전용) ──────────────

def _resolve_detector_img_dir(merged: dict, merged_path: str | Path | None = None) -> str | None:
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


# ── legacy 단독 진입점 ───────────────────────────────────

def verify_lecture_content(
    merged_path: str,
    current_date: str = None,
    num_runs: int = 1,
    min_detection_rate: float = 0.5,
    batch_size: int = BATCH_SIZE,
    max_workers: int = 4,
) -> dict:
    """Legacy wrapper around the current cross-verification pipeline."""
    from .cross_pipeline import cross_verify
    from .cross_utils import _collect_env_vars

    models = [
        model.strip()
        for model in (os.getenv("CROSS_VERIFY_MODELS", "gpt-5.4 claude-sonnet-4.5") or "").replace(",", " ").split()
        if model.strip()
    ]
    if not models:
        models = [cc._resolve_stage_model("judge")]
    env_vars = _collect_env_vars()
    return cross_verify(
        merged_path=merged_path,
        models=models,
        num_runs=num_runs,
        min_rate=min_detection_rate,
        batch_size=batch_size,
        env_vars=env_vars,
        claim_max_workers=max_workers,
        judge_max_workers=max_workers,
    )


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
    from .cross_utils import _write_claims_jsonl

    claims_log_path = _write_claims_jsonl(result.get("merged_claims", []), out_json)
    if claims_log_path:
        result["claims_log_path"] = claims_log_path
        out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    assessment = result.get("overall_assessment", {})
    print(f"\n저장: {out_json}")
    if claims_log_path:
        print(f"저장: {claims_log_path}")
    print(f"이슈: {assessment.get('total_issues', 0)}건")
