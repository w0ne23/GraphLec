from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed


def _build_grounding_context(issue: dict, slide_ctx: dict | None, slides: list[dict] | None) -> str:
    if not slide_ctx or not slides:
        return "(강의 문맥 없음)"

    try:
        from .claim_crosscheck import _build_multi_slide_transcript_block
    except Exception:
        return "(강의 문맥 로드 실패)"

    sn = int(issue.get("slide_number", 0) or 0)
    prev_sn = max(0, sn - 1)
    ctx = slide_ctx.get(sn, {})
    prev_ctx = slide_ctx.get(prev_sn, {}) if prev_sn > 0 else {}
    title = ctx.get("title", f"슬라이드 {sn}")
    time_range = ctx.get("time_range", "")
    prev_title = prev_ctx.get("title", f"슬라이드 {prev_sn}") if prev_sn > 0 else ""
    prev_time_range = prev_ctx.get("time_range", "") if prev_sn > 0 else ""
    transcript_block = _build_multi_slide_transcript_block(slides, [prev_sn, sn] if prev_sn > 0 else [sn])

    return f"""대상 슬라이드: {title} ({time_range})
이전 슬라이드: {(prev_title + f" ({prev_time_range})") if prev_sn > 0 else "없음"}

[이전+현재 슬라이드 내용]
{transcript_block}"""


def _build_grounding_prompt(
    issue: dict,
    hint: dict,
    slide_ctx: dict | None = None,
    slides: list[dict] | None = None,
) -> str:
    claim = issue.get("claim_text", issue.get("problematic_content", ""))
    correct_info = issue.get("correct_info", "")
    issue_desc = issue.get("issue", "")
    domain_label = hint.get("label", "일반")
    lecture_context = _build_grounding_context(issue, slide_ctx, slides)

    return f"""당신은 강의 내용 검증 시스템의 최종 검증 단계입니다.
강의 도메인: {domain_label}

아래는 강의에서 발견된 잠재적 오류입니다.
Google Search 결과를 근거로, 이 지적이 **타당한지** 판단해주세요.

강의 문맥:
{lecture_context}

발화 원문: {claim}
지적 내용: {issue_desc}
제안된 정확한 정보: {correct_info}

판단 기준:
1. 먼저 강의 문맥에서 학생이 실제로 이해할 명제를 판단하세요.
2. claim이나 지적 내용이 발화 원문보다 넓게 일반화되었다면, 원문과 주변 문맥의 범위로 좁혀 판단하세요.
3. 검색 결과에서 그 문맥화된 지적 내용을 뒷받침하는 근거를 찾으세요.
4. 검색 결과가 오히려 발화 원문이 맞다고 지지하면, 이 지적은 기각합니다.
5. 검색 결과가 불충분하거나 모호하면, 이 지적은 보류(기각)합니다.
6. 슬라이드가 같은 교육적 단순화를 명시하고 발화가 이를 설명하는 경우, 강의 범위 밖의 고급 예외만으로 오류 처리하지 마세요.
7. 입문 운영체제 강의의 계층 구조 설명을 펌웨어, DMA, 하이퍼바이저, 장치 내부 컨트롤러 같은 예외만으로 반박하지 마세요.

★ 중요: 검색 쿼리는 반드시 강의 도메인({domain_label})의 맥락으로 검색하세요.
예: 특정 프로그래밍 언어 강의라면 해당 언어명을 포함해서 검색해야 하며,
    다른 언어의 문법/문서를 근거로 사용하면 안 됩니다.
예: 물리학 강의라면 물리학 맥락으로 검색해야 하며,
    화학이나 다른 분야의 용어 정의를 근거로 사용하면 안 됩니다.

응답 (JSON만):
```json
{{
  "is_valid": true | false,
  "reason": "검색 근거를 바탕으로 판단 이유를 한 줄로 설명",
  "evidence_sources": ["근거가 된 URL (있으면)"]
}}
```

JSON 외 텍스트를 출력하지 마세요.
reason에는 줄바꿈을 넣지 마세요.
"""


def _ground_verify_issue(
    issue: dict,
    hint: dict,
    slide_ctx: dict | None = None,
    slides: list[dict] | None = None,
) -> tuple[dict, dict]:
    from . import claim_common as cv

    prompt = _build_grounding_prompt(issue, hint, slide_ctx, slides)
    model = cv._resolve_stage_model("grounding")
    use_gs = not (model.startswith("gpt") or model.startswith("o1") or model.startswith("o3"))

    try:
        text, call_usage = cv._call_llm(
            prompt,
            max_tokens=2048,
            temperature=0.0,
            use_grounding=use_gs,
            stage="grounding",
        )
        payload = cv._parse_grounding_payload(text)
        is_valid = payload.get("is_valid", True)
        reason = str(payload.get("reason", "") or "")
        sources = payload.get("evidence_sources", [])
        if not isinstance(sources, list):
            sources = []

        issue["grounding_verified"] = bool(is_valid)
        issue["grounding_reason"] = reason
        issue["grounding_api_failed"] = False
        if sources:
            existing = issue.get("evidence_sources", [])
            issue["evidence_sources"] = list(dict.fromkeys(existing + sources))
        token_usage = cv._empty_token_usage()
        cv._add_call_usage(token_usage, call_usage)
        return issue, token_usage
    except Exception:
        issue["grounding_verified"] = None
        issue["grounding_reason"] = "grounding 응답 파싱 실패(이슈는 보수적으로 유지)"
        issue["grounding_api_failed"] = True
        return issue, cv._empty_token_usage()


def ground_verify_all_issues(
    issues: list[dict],
    hint: dict,
    slide_ctx: dict | None = None,
    slides: list[dict] | None = None,
    max_workers: int = 4,
) -> tuple[list[dict], list[dict], int, int, dict]:
    """모든 이슈를 grounding 검증하고, 통과/기각으로 분류."""
    from . import claim_common as cv

    if not issues:
        return [], [], 0, 0, cv._empty_token_usage()

    print(f"\n  ── 4단계: grounding 검증 ({len(issues)}건) ──")
    verified = []
    rejected = []
    api_calls = 0
    failed_calls = 0
    token_usage = cv._empty_token_usage()

    def process(i, issue):
        claim_preview = str(issue.get("claim_text", issue.get("problematic_content", "")))[:50]
        print(f"    grounding [{i+1}/{len(issues)}] {claim_preview}...")
        return _ground_verify_issue(issue.copy(), hint, slide_ctx, slides)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(process, i, iss): i for i, iss in enumerate(issues)}
        for f in as_completed(futures):
            api_calls += 1
            try:
                result, call_usage = f.result()
                token_usage = cv._merge_token_usage(token_usage, call_usage)
                if result.get("grounding_api_failed"):
                    failed_calls += 1
                if result.get("grounding_verified") is False:
                    rejected.append(result)
                    print(f"      ❌ 기각: {result.get('grounding_reason', '')[:80]}")
                else:
                    verified.append(result)
                    print(f"      ✅ 확인")
            except Exception as e:
                idx = futures[f]
                issue_copy = issues[idx].copy()
                issue_copy["grounding_verified"] = None
                issue_copy["grounding_reason"] = f"grounding 실패: {e}"
                issue_copy["grounding_api_failed"] = True
                verified.append(issue_copy)
                failed_calls += 1

    verified.sort(key=lambda x: float(x.get("start_time", 0) or 0))
    rejected.sort(key=lambda x: float(x.get("start_time", 0) or 0))
    print(f"  grounding 결과: {len(verified)}건 확인, {len(rejected)}건 기각")
    return verified, rejected, api_calls, failed_calls, token_usage
