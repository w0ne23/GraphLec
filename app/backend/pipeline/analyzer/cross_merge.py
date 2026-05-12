"""Claim/issue merge helpers for cross-model verification."""

from __future__ import annotations

import json
import os
import re


def _normalize_issue_contract(issue: dict) -> dict:
    from . import claim_common as cv

    return cv.normalize_issue_metadata(issue)


def _compact_issue_text(value: str) -> str:
    return " ".join(str(value or "").split()).strip().lower()


def _issue_anchor_text(issue: dict) -> str:
    """중복 판단에 쓸 핵심 발화 조각."""
    problematic = _compact_issue_text(issue.get("problematic_content", ""))
    if problematic:
        return problematic
    claim_text = _compact_issue_text(issue.get("claim_text", ""))
    if claim_text:
        return claim_text
    issue_hint = _compact_issue_text(issue.get("issue") or issue.get("correct_info") or "")
    return issue_hint[:160]


def _token_overlap_ratio(a: str, b: str) -> float:
    a_tokens = {tok for tok in a.split() if tok}
    b_tokens = {tok for tok in b.split() if tok}
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / max(1, min(len(a_tokens), len(b_tokens)))


def _time_distance_sec(a: dict, b: dict) -> float:
    try:
        return abs(float(a.get("start_time", 0) or 0) - float(b.get("start_time", 0) or 0))
    except Exception:
        return 999999.0


def _same_contextual_issue(a: dict, b: dict) -> bool:
    """같은 맥락에서 wording만 조금 다른 동일 이슈인지 판단."""
    same_utterance = (a.get("utterance_id") or "") == (b.get("utterance_id") or "")
    same_local_context = (
        not same_utterance
        and int(a.get("slide_number", 0) or 0) == int(b.get("slide_number", 0) or 0)
        and _time_distance_sec(a, b) <= 12.0
    )
    if not same_utterance and not same_local_context:
        return False
    if (a.get("type") or "") != (b.get("type") or ""):
        return False

    a_claim = _compact_issue_text(a.get("claim_text", ""))
    b_claim = _compact_issue_text(b.get("claim_text", ""))
    if same_utterance and a_claim and b_claim:
        if a_claim == b_claim:
            return True
        shorter, longer = sorted((a_claim, b_claim), key=len)
        if len(shorter) >= 12 and shorter in longer:
            return True

    a_anchor = _issue_anchor_text(a)
    b_anchor = _issue_anchor_text(b)
    if a_anchor and b_anchor:
        if a_anchor == b_anchor:
            return True
        shorter, longer = sorted((a_anchor, b_anchor), key=len)
        if len(shorter) >= 8 and shorter in longer:
            return True
        if _token_overlap_ratio(a_anchor, b_anchor) >= 0.8:
            return True

    a_issue = _compact_issue_text(a.get("issue", ""))
    b_issue = _compact_issue_text(b.get("issue", ""))
    if a_issue and b_issue:
        shorter, longer = sorted((a_issue, b_issue), key=len)
        if len(shorter) >= 16 and shorter in longer:
            return True
        local_threshold = 0.4 if _time_distance_sec(a, b) <= 5.0 else 0.55
        if _token_overlap_ratio(a_issue, b_issue) >= (local_threshold if same_local_context else 0.8):
            return True

    if same_local_context:
        a_correct = _compact_issue_text(a.get("correct_info", ""))
        b_correct = _compact_issue_text(b.get("correct_info", ""))
        if a_correct and b_correct and _token_overlap_ratio(a_correct, b_correct) >= 0.55:
            return True
    return False


def _merge_issue_payload(dst: dict, src: dict) -> dict:
    from . import claim_common as cv

    _normalize_issue_contract(dst)
    _normalize_issue_contract(src)
    merged = dict(dst)
    merged_models = set(merged.get("detected_by_models", []))
    merged_models.update(src.get("detected_by_models", []))
    merged["detected_by_models"] = sorted(merged_models)

    merged_claims = list(
        dict.fromkeys(
            [
                *(merged.get("merged_claim_texts", []) or [merged.get("claim_text", "")]),
                *(src.get("merged_claim_texts", []) or [src.get("claim_text", "")]),
            ]
        )
    )
    merged["merged_claim_texts"] = [x for x in merged_claims if x]

    merged_problematic = list(
        dict.fromkeys(
            [
                *(merged.get("merged_problematic_contents", []) or [merged.get("problematic_content", "")]),
                *(src.get("merged_problematic_contents", []) or [src.get("problematic_content", "")]),
            ]
        )
    )
    merged["merged_problematic_contents"] = [x for x in merged_problematic if x]

    try:
        dst_conf = float(merged.get("confidence", 0) or 0)
    except Exception:
        dst_conf = 0.0
    try:
        src_conf = float(src.get("confidence", 0) or 0)
    except Exception:
        src_conf = 0.0
    if src_conf > dst_conf:
        keep_lists = {
            "detected_by_models": merged["detected_by_models"],
            "merged_claim_texts": merged["merged_claim_texts"],
            "merged_problematic_contents": merged["merged_problematic_contents"],
        }
        merged = dict(src)
        merged.update(keep_lists)
    else:
        cv.copy_issue_metadata(merged, src)
    return merged


def _cluster_contextual_issues(issues: list[dict]) -> list[dict]:
    clustered: list[dict] = []
    for issue in issues:
        _normalize_issue_contract(issue)
        merged = False
        for idx, existing in enumerate(clustered):
            if _same_contextual_issue(existing, issue):
                clustered[idx] = _merge_issue_payload(existing, issue)
                merged = True
                break
        if not merged:
            seeded = dict(issue)
            seeded["merged_claim_texts"] = [issue.get("claim_text", "")] if issue.get("claim_text") else []
            seeded["merged_problematic_contents"] = [issue.get("problematic_content", "")] if issue.get("problematic_content") else []
            clustered.append(seeded)
    return clustered


def _issue_cluster_model(models: list[str]) -> str:
    configured = str(os.getenv("VERIFIER_ISSUE_CLUSTER_MODEL", "") or "").strip()
    if configured:
        return configured
    for model in models or []:
        lowered = str(model or "").lower()
        if lowered.startswith(("gpt", "o1", "o3")) or "claude" in lowered or "sonnet" in lowered or "opus" in lowered:
            return str(model)
    return str((models or [""])[0] or "")


def _issue_cluster_enabled() -> bool:
    return str(os.getenv("VERIFIER_LLM_ISSUE_CLUSTERING", "1") or "1").strip().lower() not in {
        "0",
        "false",
        "off",
        "none",
    }


def _issue_cluster_row(issue_id: str, issue: dict) -> str:
    detected = ", ".join(issue.get("detected_by_models", []) or [])
    merged_claims = [x for x in issue.get("merged_claim_texts", []) or [] if x]
    merged_problematic = [x for x in issue.get("merged_problematic_contents", []) or [] if x]
    extra_claims = ""
    if merged_claims:
        extra_claims = "\n  - related_claims: " + " | ".join(merged_claims[:4])
    if merged_problematic:
        extra_claims += "\n  - related_problematic: " + " | ".join(merged_problematic[:4])
    return (
        f"### {issue_id}\n"
        f"- utterance_id: {issue.get('utterance_id', '')}\n"
        f"- slide: {issue.get('slide_number', '')}\n"
        f"- start_time: {float(issue.get('start_time', 0) or 0):.1f}\n"
        f"- detected_by: {detected}\n"
        f"- tentative_type: {issue.get('type', '')}\n"
        f"- claim: {issue.get('claim_text', '')}\n"
        f"- problematic_content: {issue.get('problematic_content', '')}\n"
        f"- issue: {issue.get('issue', '')}\n"
        f"- student_error: {issue.get('student_error', '')}\n"
        f"- correct_info: {issue.get('correct_info', '')}"
        f"{extra_claims}"
    )


def _parse_issue_cluster_payload(text: str, issue_ids: set[str]) -> list[dict]:
    from . import claim_common as cv

    cleaned = cv._strip_json_fence((text or "").strip())
    candidates = [cleaned]
    obj = cv._extract_first_json_object(cleaned)
    if obj and obj not in candidates:
        candidates.append(obj)

    for candidate in candidates:
        if not candidate:
            continue
        fixed = re.sub(r",\s*([}\]])", r"\1", candidate)
        for payload_text in (candidate, fixed):
            try:
                payload = json.loads(payload_text)
            except json.JSONDecodeError:
                continue
            groups = payload.get("groups") if isinstance(payload, dict) else payload
            if not isinstance(groups, list):
                continue

            parsed = []
            seen: set[str] = set()
            for row in groups:
                if not isinstance(row, dict):
                    continue
                raw_members = row.get("member_issue_ids") or row.get("members") or []
                if not isinstance(raw_members, list):
                    continue
                members = []
                for member in raw_members:
                    member_id = str(member or "").strip()
                    if member_id in issue_ids and member_id not in seen and member_id not in members:
                        members.append(member_id)
                if not members:
                    continue
                seen.update(members)
                representative = str(row.get("representative_issue_id", "") or "").strip()
                if representative not in members:
                    representative = members[0]
                parsed.append(
                    {
                        "canonical_issue_id": str(row.get("canonical_issue_id", "") or "").strip(),
                        "member_issue_ids": members,
                        "representative_issue_id": representative,
                        "canonical_wrong_proposition": str(
                            row.get("canonical_wrong_proposition", "") or ""
                        ).strip(),
                        "merge_rationale": str(row.get("merge_rationale", "") or "").strip(),
                    }
                )
            for issue_id in issue_ids:
                if issue_id not in seen:
                    parsed.append(
                        {
                            "canonical_issue_id": "",
                            "member_issue_ids": [issue_id],
                            "representative_issue_id": issue_id,
                            "canonical_wrong_proposition": "",
                            "merge_rationale": "",
                        }
                    )
            if parsed:
                return parsed
    raise ValueError("issue_cluster_response_parse_failed")


def _individual_issue_key(issue: dict) -> str:
    uid = str(issue.get("utterance_id", "") or "")
    if uid:
        return uid
    claim = _compact_issue_text(issue.get("claim_text", "") or issue.get("problematic_content", ""))
    return claim


def _compact_source_issue(issue: dict) -> dict:
    return {
        "utterance_id": issue.get("utterance_id", ""),
        "slide_number": issue.get("slide_number"),
        "start_time": issue.get("start_time"),
        "end_time": issue.get("end_time"),
        "claim_type": issue.get("claim_type") or issue.get("type", ""),
        "claim_text": issue.get("claim_text", ""),
        "resolved_claim": issue.get("resolved_claim", ""),
        "problematic_content": issue.get("problematic_content", ""),
        "issue": issue.get("issue", ""),
        "correct_info": issue.get("correct_info", ""),
        "detected_by_models": issue.get("detected_by_models", []),
    }


def _build_issue_units(issues_by_id: dict[str, dict], groups: list[dict]) -> list[dict]:
    """canonical issue를 crosscheck가 검증할 실제 문맥 단위 issue로 만든다.

    최종 교수 피드백의 단위는 claim 하나가 아니라 같은 오해를 만드는 인접 발화 묶음이다.
    같은 utterance에서 모델별 wording만 다른 후보는 합치고, 서로 다른 utterance가 같은
    설명 오류를 구성하면 하나의 issue unit으로 승격한다.
    """
    issue_units: list[dict] = []
    for index, group in enumerate(groups, start=1):
        members = [mid for mid in group.get("member_issue_ids", []) if mid in issues_by_id]
        if not members:
            continue
        canonical_id = group.get("canonical_issue_id") or f"ci_{index:04d}"
        member_utterance_ids = list(
            dict.fromkeys(
                issues_by_id[mid].get("utterance_id", "")
                for mid in members
                if issues_by_id[mid].get("utterance_id")
            )
        )

        merged_by_source: dict[str, dict] = {}
        ordered_source_keys: list[str] = []
        for member_id in members:
            source_issue = dict(issues_by_id[member_id])
            source_key = _individual_issue_key(source_issue)
            if source_key in merged_by_source:
                merged_by_source[source_key] = _merge_issue_payload(merged_by_source[source_key], source_issue)
            else:
                merged_by_source[source_key] = source_issue
                ordered_source_keys.append(source_key)

        source_issues = [merged_by_source[source_key] for source_key in ordered_source_keys]
        source_issues.sort(key=lambda issue: float(issue.get("start_time", 0) or 0))
        representative = max(source_issues, key=lambda issue: float(issue.get("confidence", 0) or 0))
        issue_unit = dict(representative)

        detected_models: set[str] = set()
        claim_lines = []
        problem_lines = []
        correct_infos = []
        for source_issue in source_issues:
            detected_models.update(source_issue.get("detected_by_models", []) or [])
            uid = source_issue.get("utterance_id", "")
            claim = source_issue.get("claim_text", "") or source_issue.get("problematic_content", "")
            problem = source_issue.get("issue", "")
            correct = source_issue.get("correct_info", "")
            if claim:
                claim_lines.append(f"{uid}: {claim}" if uid else claim)
            if problem:
                problem_lines.append(f"{uid}: {problem}" if uid else problem)
            if correct:
                correct_infos.append(correct)

        start_times = [float(issue.get("start_time", 0) or 0) for issue in source_issues]
        end_times = [
            float(issue.get("end_time", 0) or 0)
            for issue in source_issues
            if issue.get("end_time") is not None
        ]
        issue_unit["issue_unit_id"] = canonical_id
        issue_unit["is_issue_unit"] = True
        issue_unit["canonical_issue_id"] = canonical_id
        issue_unit["canonical_member_issue_ids"] = members
        issue_unit["canonical_member_utterance_ids"] = member_utterance_ids
        issue_unit["canonical_issue_count"] = len(source_issues)
        issue_unit["source_issues"] = [_compact_source_issue(issue) for issue in source_issues]
        issue_unit["detected_by_models"] = sorted(detected_models)
        issue_unit["utterance_id"] = source_issues[0].get("utterance_id", "")
        issue_unit["utterance_ids"] = member_utterance_ids
        issue_unit["start_time"] = min(start_times) if start_times else issue_unit.get("start_time")
        if end_times:
            issue_unit["end_time"] = max(end_times)
        issue_unit["claim_text"] = "\n".join(dict.fromkeys(claim_lines))
        issue_unit["problematic_content"] = issue_unit["claim_text"]
        if problem_lines:
            issue_unit["issue"] = "\n".join(dict.fromkeys(problem_lines))
        elif group.get("canonical_wrong_proposition"):
            issue_unit["issue"] = group["canonical_wrong_proposition"]
        if correct_infos:
            issue_unit["correct_info"] = next(iter(dict.fromkeys(correct_infos)))
        if group.get("canonical_wrong_proposition"):
            issue_unit["canonical_wrong_proposition"] = group["canonical_wrong_proposition"]
            if not issue_unit.get("student_error"):
                issue_unit["student_error"] = group["canonical_wrong_proposition"]
        if group.get("merge_rationale"):
            issue_unit["canonical_merge_rationale"] = group["merge_rationale"]
        issue_units.append(issue_unit)
    return issue_units


def _rebuild_union_sets(issues: list[dict], models: list[str]) -> tuple[list[dict], list[dict], dict]:
    unioned = []
    intersected = []
    exclusive = {model: [] for model in models}

    for issue in issues:
        issue["detected_by_models"] = sorted(set(issue.get("detected_by_models", [])))
        issue["cross_model_agreement"] = len(issue["detected_by_models"])
        unioned.append(issue)
        if len(issue["detected_by_models"]) == len(models):
            intersected.append(issue)
        elif len(issue["detected_by_models"]) == 1 and issue["detected_by_models"][0] in exclusive:
            exclusive[issue["detected_by_models"][0]].append(issue)

    unioned.sort(key=lambda x: float(x.get("start_time", 0) or 0))
    intersected.sort(key=lambda x: float(x.get("start_time", 0) or 0))
    for model in models:
        exclusive[model].sort(key=lambda x: float(x.get("start_time", 0) or 0))
    return unioned, intersected, exclusive


def canonicalize_issues_with_llm(
    issues: list[dict],
    models: list[str],
) -> tuple[list[dict], list[dict], dict, dict]:
    """LLM으로 judge 후보를 canonical issue 단위로 묶는다.

    judge의 tentative type은 참고만 하고, 학생이 잘못 외울 명제와 수정 방향이
    같은지 기준으로 묶는다. 실패 시 원본 이슈를 그대로 반환한다.
    """
    from . import claim_common as cv

    if not _issue_cluster_enabled() or len(issues) <= 1:
        unioned, intersected, exclusive = _rebuild_union_sets(issues, models)
        return unioned, intersected, exclusive, cv._empty_token_usage()

    max_issues = int(os.getenv("VERIFIER_ISSUE_CLUSTER_MAX", "40") or "40")
    if len(issues) > max_issues:
        unioned, intersected, exclusive = _rebuild_union_sets(issues, models)
        return unioned, intersected, exclusive, cv._empty_token_usage()

    issue_ids = [f"i{i:04d}" for i in range(1, len(issues) + 1)]
    issues_by_id = dict(zip(issue_ids, issues))
    issue_block = "\n\n".join(_issue_cluster_row(issue_id, issue) for issue_id, issue in issues_by_id.items())
    prompt = f"""아래는 서로 다른 judge 모델이 강의에서 발견한 문제 후보 목록입니다.
당신의 작업은 최종 판정이 아니라, 같은 문제 후보를 canonical issue 단위로 묶는 것입니다.

canonical issue는 최종 항목을 삭제하기 위한 것이 아니라, 교수 화면에서
같은 근본 문제를 하나의 묶음으로 보여주기 위한 그룹입니다.

묶는 기준:
- 학생이 잘못 외울 수 있는 명제가 사실상 같음
- 문제 삼는 표현/개념이 같음
- 교수에게 권할 수정 방향이 같음
- 같은 발화, 인접 발화, 같은 슬라이드 흐름에서 반복/보강된 문제임
- 서로 다른 문장이라도 하나의 수정 안내로 함께 해결되는 결합된 오류임
- 대비되는 두 개념, 단계, 범주, 예시가 서로 뒤바뀐 것처럼 한 쌍의 설명 오류가
  여러 발화에 걸쳐 나타나면 같은 canonical issue로 묶으세요.

묶으면 안 되는 기준:
- tentative_type이 같다는 이유만으로 묶지 마세요.
- 키워드만 같고 학생 오답 명제나 수정 방향이 다르면 분리하세요.
- 같은 utterance라도 서로 다른 문제이면 분리하세요.
- 같은 주제라도 교수에게 서로 다른 수정 안내가 필요하면 분리하세요.
- 확신이 없으면 분리하세요.

입력:
{issue_block}

응답은 JSON object 하나만 출력하세요.
모든 issue_id는 정확히 한 번만 포함하세요.

형식:
{{
  "groups": [
    {{
      "canonical_issue_id": "ci_0001",
      "member_issue_ids": ["i0001", "i0003"],
      "representative_issue_id": "i0001",
      "canonical_wrong_proposition": "학생이 잘못 외울 수 있는 명제",
      "merge_rationale": "같은 문제로 묶은 짧은 이유"
    }}
  ]
}}"""

    cluster_model = _issue_cluster_model(models)
    old_model = os.environ.get("VERIFIER_CROSS_RECHECK_MODEL")
    if cluster_model:
        os.environ["VERIFIER_CROSS_RECHECK_MODEL"] = cluster_model
    token_usage = cv._empty_token_usage()
    try:
        text, call_usage = cv._call_llm(
            prompt,
            max_tokens=min(8192, max(2048, 350 * len(issues))),
            temperature=0.0,
            response_format=(
                {"type": "json_object"}
                if cv._supports_json_object_response_format(cluster_model)
                else None
            ),
            stage="cross_recheck",
        )
        cv._add_call_usage(token_usage, call_usage)
        groups = _parse_issue_cluster_payload(text, set(issue_ids))
        issue_units = _build_issue_units(issues_by_id, groups)
        unioned, intersected, exclusive = _rebuild_union_sets(issue_units, models)
        return unioned, intersected, exclusive, token_usage
    except Exception as e:
        print(f"  ⚠️ LLM issue clustering 실패, 기존 union 결과를 사용합니다: {e}")
        unioned, intersected, exclusive = _rebuild_union_sets(issues, models)
        return unioned, intersected, exclusive, token_usage
    finally:
        if old_model is None:
            os.environ.pop("VERIFIER_CROSS_RECHECK_MODEL", None)
        else:
            os.environ["VERIFIER_CROSS_RECHECK_MODEL"] = old_model


def _claim_key(claim: dict) -> str:
    if claim.get("claim_fingerprint"):
        return str(claim.get("claim_fingerprint"))
    if claim.get("claim_id"):
        return str(claim.get("claim_id"))
    uid = claim.get("utterance_id", "")
    text = claim.get("claim_text", "")[:60]
    return f"{uid}::{text}"


def union_claims(results: list[dict]) -> list[dict]:
    """여러 모델의 claim을 합집합 (claim_text + utterance_id 기준 dedupe)."""
    seen = {}
    for r in results:
        for item in r["claims_by_batch"]:
            for claim in item["claims"]:
                key = _claim_key(claim)
                if key not in seen:
                    seen[key] = claim
    return list(seen.values())


def rebuild_claim_batches(merged_claims: list[dict], utterances: list[dict], batch_size: int) -> list[dict]:
    """합집합 claim을 원래 배치 구조로 재구성."""
    batches = [utterances[i:i + batch_size] for i in range(0, len(utterances), batch_size)]
    batch_map = {}
    for batch in batches:
        uids = {u["utterance_id"] for u in batch}
        for uid in uids:
            batch_map[uid] = id(batch)

    batch_claims: dict[int, tuple] = {}
    for batch in batches:
        bid = id(batch)
        batch_claims[bid] = (batch, [])

    for claim in merged_claims:
        uid = claim.get("utterance_id", "") or claim.get("context_id", "")
        bid = batch_map.get(uid)
        if bid and bid in batch_claims:
            batch_claims[bid][1].append(claim)

    result = []
    for batch in batches:
        bid = id(batch)
        b, c = batch_claims[bid]
        if c:
            result.append({"batch": b, "claims": c})
    return result


def _issue_match_key(issue: dict) -> str:
    """claim-level 매칭 키."""
    uid = issue.get("utterance_id", "")
    itype = issue.get("type", "")
    anchor = _issue_anchor_text(issue)
    return f"{uid}::{itype}::{anchor}"


def _dedupe_model_issues(issues: list[dict]) -> dict[str, dict]:
    """같은 모델 안에서 같은 이슈 키는 confidence가 더 높은 것을 남긴다."""
    deduped: dict[str, dict] = {}
    for issue in issues or []:
        key = _issue_match_key(issue)
        current = deduped.get(key)
        if current is None:
            deduped[key] = issue
            continue
        try:
            new_conf = float(issue.get("confidence", 0) or 0)
        except Exception:
            new_conf = 0.0
        try:
            old_conf = float(current.get("confidence", 0) or 0)
        except Exception:
            old_conf = 0.0
        if new_conf >= old_conf:
            deduped[key] = issue
    return deduped


def union_issues(results: list[dict]) -> tuple[list[dict], list[dict], dict]:
    """여러 모델의 이슈를 claim-level 합집합으로 모은다."""
    if not results:
        return [], [], {}

    model_issues = {}
    for r in results:
        model = r["model"]
        normalized = [_normalize_issue_contract(issue) for issue in (r["issues"] or [])]
        model_issues[model] = _dedupe_model_issues(normalized)

    models = list(model_issues.keys())
    union_map = {}
    for model in models:
        for key, issue in model_issues[model].items():
            if key not in union_map:
                merged = dict(issue)
                merged["detected_by_models"] = [model]
                _normalize_issue_contract(merged)
                union_map[key] = merged
            else:
                union_map[key]["detected_by_models"].append(model)
                if float(issue.get("confidence", 0) or 0) > float(union_map[key].get("confidence", 0) or 0):
                    for k, v in issue.items():
                        if k != "detected_by_models":
                            union_map[key][k] = v

    exclusive = {model: [] for model in models}
    unioned = []
    intersected = []
    for issue in union_map.values():
        issue["detected_by_models"] = sorted(set(issue.get("detected_by_models", [])))
        issue["cross_model_agreement"] = len(issue["detected_by_models"])
        unioned.append(issue)
        if len(issue["detected_by_models"]) == len(models):
            intersected.append(issue)
        elif len(issue["detected_by_models"]) == 1:
            exclusive[issue["detected_by_models"][0]].append(issue)

    unioned.sort(key=lambda x: float(x.get("start_time", 0) or 0))
    intersected.sort(key=lambda x: float(x.get("start_time", 0) or 0))
    for model in models:
        exclusive[model].sort(key=lambda x: float(x.get("start_time", 0) or 0))

    return unioned, intersected, exclusive
