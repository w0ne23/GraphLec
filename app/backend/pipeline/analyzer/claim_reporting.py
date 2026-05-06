"""Reporting helpers for the single-model claim pipeline."""

from __future__ import annotations

from pathlib import Path

from .claim_common import TOKEN_USAGE_STAGES
from .cross_utils import _write_claims_jsonl


def format_timestamp(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}분 {s}초" if m > 0 else f"{s}초"


def _format_token_bucket(bucket: dict) -> str:
    if not isinstance(bucket, dict):
        return "input 0 / output 0 / reasoning 0 / total 0"

    parts = [
        f"input {bucket.get('input_tokens', 0):,}",
        f"output {bucket.get('output_tokens', 0):,}",
        f"reasoning {bucket.get('reasoning_tokens', 0):,}",
        f"total {bucket.get('total_tokens', 0):,}",
    ]
    if bucket.get("tool_input_tokens", 0):
        parts.append(f"tool {bucket.get('tool_input_tokens', 0):,}")
    if bucket.get("cached_input_tokens", 0):
        parts.append(f"cached {bucket.get('cached_input_tokens', 0):,}")
    if bucket.get("cache_creation_input_tokens", 0):
        parts.append(f"cache_write {bucket.get('cache_creation_input_tokens', 0):,}")
    return " / ".join(parts)


def format_verification_report(result: dict) -> str:
    lines = ["=" * 60, "강의 내용 검증 리포트 (claim 파이프라인)", "=" * 60]
    a = result["overall_assessment"]
    lines.append(f"\n전체: {'문제 발견' if a['has_issues'] else '문제 없음'} ({a['total_issues']}건)")
    if a["total_issues"] > 0:
        b = a["severity_breakdown"]
        lines.append(f"  Critical: {b['critical']} / Major: {b['major']} / Minor: {b['minor']}")
    lines.append(f"요약: {result.get('summary', 'N/A')}")
    if not result.get("is_complete", True):
        lines.append(f"경고: {result.get('completion_warning', 'verification_incomplete')}")
    meta = result.get("metadata", {})
    lines.append(f"도메인: {meta.get('domain_hint', 'N/A')}")
    if meta.get("verifier_extract_model") or meta.get("verifier_judge_model"):
        lines.append(
            "단계별 모델: "
            f"extract={meta.get('verifier_extract_model', meta.get('verifier_model', 'N/A'))}, "
            f"judge={meta.get('verifier_judge_model', meta.get('verifier_model', 'N/A'))}, "
            f"recheck={meta.get('verifier_recheck_model', meta.get('verifier_model', 'N/A'))}, "
            f"grounding={meta.get('verifier_grounding_model', meta.get('verifier_model', 'N/A'))}"
        )
    lines.append(f"추출된 claim: {meta.get('total_claims_extracted', 0)}개")
    token_usage = result.get("token_usage", {})
    lines.append(f"토큰 사용량(총): {_format_token_bucket(token_usage.get('total', {}))}")
    stage_parts = []
    for stage in TOKEN_USAGE_STAGES:
        bucket = token_usage.get(stage, {})
        if bucket.get("total_tokens", 0):
            stage_parts.append(f"{stage}={bucket.get('total_tokens', 0):,}")
    if stage_parts:
        lines.append(f"단계별 total 토큰: {', '.join(stage_parts)}")
    lines.append(f"슬라이드 텍스트: {meta.get('slides_with_text', 0)}/{meta.get('total_slides', 0)}개")

    for i, issue in enumerate(result.get("issues", []), 1):
        t = format_timestamp(issue.get("start_time", 0))
        type_label = issue.get("issue_type_label") or issue.get("pedagogical_label") or issue.get("type")
        feedback = issue.get("professor_feedback") if isinstance(issue.get("professor_feedback"), dict) else {}
        lines.append(f"\n[{i}] {issue['severity'].upper()} - {type_label} ({issue['type']})")
        lines.append(f"  시간: {t} ({issue.get('start_time', 0):.1f}초)")
        if issue.get("claim_text"):
            lines.append(f"  claim: {issue['claim_text']}")
        lines.append(f"  원문: {issue.get('problematic_content', '')}")
        lines.append(f"  문제: {issue.get('issue', '')}")
        if issue.get("cross_recheck_reason"):
            lines.append(f"  확정 사유: {issue['cross_recheck_reason']}")
        if issue.get("evidence_in_context"):
            lines.append(f"  문맥 근거: {issue['evidence_in_context']}")
        student_misunderstanding = (
            issue.get("student_misunderstanding")
            or feedback.get("student_misunderstanding")
            or ""
        )
        if student_misunderstanding:
            lines.append(f"  학생 오해 가능성: {student_misunderstanding}")
        lines.append(f"  올바른 정보: {issue.get('correct_info', '')}")
        lines.append(f"  설명: {issue.get('explanation', '')}")
        lines.append(f"  권장: {issue.get('recommendation', '')}")
        suggested_rephrase = issue.get("suggested_rephrase") or feedback.get("suggested_rephrase") or ""
        teaching_note = issue.get("teaching_note") or feedback.get("teaching_note") or ""
        if suggested_rephrase:
            lines.append(f"  대체 표현: {suggested_rephrase}")
        if teaching_note:
            lines.append(f"  교수 메모: {teaching_note}")
        if issue.get("verification_basis") or issue.get("evidence_need"):
            lines.append(
                "  검증 분류: "
                f"basis={issue.get('verification_basis', 'unclassified')}, "
                f"evidence={issue.get('evidence_need', 'unclassified')}, "
                f"scope={issue.get('claim_scope', 'unclassified')}, "
                f"priority={issue.get('review_priority', 'unclassified')}"
            )
        lines.append(f"  신뢰도: {issue.get('confidence', 0):.0%}")
        for s in (issue.get("evidence_sources") or [])[:2]:
            lines.append(f"  출처: {s}")
        if "detection_rate" in issue:
            lines.append(f"  탐지율: {issue['detection_rate']:.0%} ({issue['detection_count']}/{meta.get('num_runs', '?')}회)")
        if issue.get("grounding_verified") is not None:
            gv = "✅ 확인" if issue["grounding_verified"] else "❌ 기각"
            lines.append(f"  grounding: {gv}")
            if issue.get("grounding_reason"):
                lines.append(f"  grounding 근거: {issue['grounding_reason'][:200]}")
        if issue.get("slide_recheck_valid") is not None:
            sv = "✅ 유지" if issue["slide_recheck_valid"] else "❌ 기각"
            lines.append(f"  슬라이드 재검증: {sv}")
            if issue.get("slide_recheck_reason"):
                lines.append(f"  재검증 근거: {issue['slide_recheck_reason'][:200]}")

    if not result.get("issues"):
        lines.append("\n✅ 문제가 발견되지 않았습니다!")

    slide_rejected = result.get("slide_rejected_issues", [])
    grounding_rejected = result.get("grounding_rejected_issues", [])
    legacy_rejected = result.get("rejected_issues", [])

    if slide_rejected:
        lines.append(f"\n{'─' * 40}")
        lines.append(f"슬라이드 재검증 기각 ({len(slide_rejected)}건)")
        lines.append(f"{'─' * 40}")
        for i, issue in enumerate(slide_rejected, 1):
            t = format_timestamp(issue.get("start_time", 0))
            reason = issue.get("slide_recheck_reason") or "N/A"
            lines.append(f"  [{i}] {t} | {issue.get('claim_text', issue.get('problematic_content', ''))[:60]}")
            lines.append(f"       사유: {reason[:120]}")

    if grounding_rejected:
        lines.append(f"\n{'─' * 40}")
        lines.append(f"grounding 기각 ({len(grounding_rejected)}건)")
        lines.append(f"{'─' * 40}")
        for i, issue in enumerate(grounding_rejected, 1):
            t = format_timestamp(issue.get("start_time", 0))
            reason = issue.get("grounding_reason") or "N/A"
            lines.append(f"  [{i}] {t} | {issue.get('claim_text', issue.get('problematic_content', ''))[:60]}")
            lines.append(f"       사유: {reason[:120]}")

    if not slide_rejected and not grounding_rejected and legacy_rejected:
        lines.append(f"\n{'─' * 40}")
        lines.append(f"기각된 이슈 ({len(legacy_rejected)}건)")
        lines.append(f"{'─' * 40}")
        for i, issue in enumerate(legacy_rejected, 1):
            t = format_timestamp(issue.get("start_time", 0))
            reason = issue.get("slide_recheck_reason") or issue.get("grounding_reason") or "N/A"
            stage = "슬라이드 재검증" if issue.get("slide_recheck_valid") is False else "grounding"
            lines.append(f"  [{i}] {t} | {stage} 기각 | {issue.get('claim_text', issue.get('problematic_content', ''))[:60]}")
            lines.append(f"       사유: {reason[:120]}")

    extracted = result.get("extracted_claims", [])
    if extracted:
        lines.append(f"\n{'=' * 60}")
        lines.append(f"추출된 claim 목록 ({len(extracted)}개)")
        lines.append("=" * 60)
        for i, c in enumerate(extracted, 1):
            uid = c.get("utterance_id", "?")
            ctype = c.get("claim_type", "?")
            claim_text = c.get("claim_text", "")[:80]
            resolved = c.get("resolved_claim", "")[:80]
            approx = " [근사치]" if c.get("is_approximate") else ""
            lines.append(f"\n  [{i}] {uid} ({ctype}){approx}")
            lines.append(f"    원문: {claim_text}")
            if resolved and resolved != claim_text:
                lines.append(f"    해소: {resolved}")
            vq = c.get("verification_question", "")
            if vq:
                lines.append(f"    검증 질문: {vq[:100]}")

    slide_typos = result.get("slide_typos", [])
    if slide_typos:
        lines.append(f"\n{'=' * 60}")
        lines.append(f"슬라이드 오타 ({len(slide_typos)}건)")
        lines.append("=" * 60)
        for i, typo in enumerate(slide_typos, 1):
            lines.append(
                f"\n  [{i}] 슬라이드 {typo.get('slide_number', '?')} ({typo.get('slide_title', '')})"
            )
            lines.append(f"    문제: {typo.get('problematic_text', '')}")
            lines.append(f"    수정: {typo.get('corrected_text', '')}")
            lines.append(f"    이유: {typo.get('reason', '')}")
            lines.append(f"    신뢰도: {float(typo.get('confidence', 0) or 0):.0%}")

    lines.extend([
        f"\n{'=' * 60}", "메타데이터", "=" * 60,
        f"  검증일: {meta.get('verification_date', 'N/A')}",
        f"  도메인: {meta.get('domain', '')} > {meta.get('sub_domain', '')}",
        f"  발화: {meta.get('total_utterances', 0)}개 / {meta.get('total_characters', 0):,}자",
        f"  모델: {meta.get('verifier_model', 'N/A')} (t={meta.get('verifier_temperature', 'N/A')})",
        f"  프롬프트: {meta.get('prompt_version', 'N/A')}",
        f"  API: {result.get('api_calls', 0)}회 (실패: {result.get('failed_calls', 0)})",
        f"  토큰(총): {_format_token_bucket(token_usage.get('total', {}))}",
        f"  추출 claim: {meta.get('total_claims_extracted', 0)}개",
        f"  슬라이드 오타: {meta.get('slide_typo_count', len(result.get('slide_typos', [])))}건",
        f"  파싱 실패: {result.get('parse_failures', 0)}건",
        f"  슬라이드 재검증 실패: {result.get('slide_recheck_failures', 0)}건",
        f"  grounding 실패: {result.get('grounding_failures', 0)}건",
        f"  슬라이드 오타 검사 실패: {result.get('slide_typo_failures', 0)}건",
        f"  grounding 기각: {result.get('grounding_filtered', 0)}건",
        f"  슬라이드 재검증 기각: {result.get('slide_recheck_filtered', 0)}건",
    ])
    stage_lines = []
    for stage in TOKEN_USAGE_STAGES:
        bucket = token_usage.get(stage, {})
        if bucket.get("total_tokens", 0):
            stage_lines.append(f"  {stage}: {_format_token_bucket(bucket)}")
    if stage_lines:
        lines.append("  단계별 토큰:")
        lines.extend(stage_lines)
    return "\n".join(lines)


def write_claims_jsonl(result: dict, output_json_path: str | Path) -> str | None:
    claims = result.get("merged_claims")
    if claims is None:
        claims = result.get("extracted_claims")
    if claims is None:
        return None
    return _write_claims_jsonl(claims, output_json_path)
