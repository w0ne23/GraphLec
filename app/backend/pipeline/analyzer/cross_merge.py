"""Claim/issue merge helpers for cross-model verification."""

from __future__ import annotations


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


def _char_bigram_overlap_ratio(a: str, b: str) -> float:
    def grams(text: str) -> set[str]:
        compact = "".join(str(text or "").split())
        if len(compact) < 2:
            return {compact} if compact else set()
        return {compact[i:i + 2] for i in range(len(compact) - 1)}

    a_grams = grams(a)
    b_grams = grams(b)
    if not a_grams or not b_grams:
        return 0.0
    return len(a_grams & b_grams) / max(1, min(len(a_grams), len(b_grams)))


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
        if same_local_context and _char_bigram_overlap_ratio(a_issue, b_issue) >= 0.5:
            return True

    if same_local_context:
        combined_a = _compact_issue_text(" ".join([
            str(a.get("claim_text", "") or ""),
            str(a.get("issue", "") or ""),
            str(a.get("correct_info", "") or ""),
        ]))
        combined_b = _compact_issue_text(" ".join([
            str(b.get("claim_text", "") or ""),
            str(b.get("issue", "") or ""),
            str(b.get("correct_info", "") or ""),
        ]))
        if _char_bigram_overlap_ratio(combined_a, combined_b) >= 0.45:
            return True
        a_correct = _compact_issue_text(a.get("correct_info", ""))
        b_correct = _compact_issue_text(b.get("correct_info", ""))
        if a_correct and b_correct and _token_overlap_ratio(a_correct, b_correct) >= 0.55:
            return True
    return False


def _merge_issue_payload(dst: dict, src: dict) -> dict:
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
    merged["evidence_sources"] = list(
        dict.fromkeys(
            [
                *(dst.get("evidence_sources", []) or []),
                *(src.get("evidence_sources", []) or []),
                *(merged.get("evidence_sources", []) or []),
            ]
        )
    )
    merged["crosscheck_details"] = _merge_crosscheck_details(
        dst.get("crosscheck_details", []),
        src.get("crosscheck_details", []),
        merged.get("crosscheck_details", []),
    )
    merged["merged_issue_count"] = int(dst.get("merged_issue_count", 1) or 1) + int(src.get("merged_issue_count", 1) or 1)
    if not merged.get("grounding_reason"):
        merged["grounding_reason"] = dst.get("grounding_reason") or src.get("grounding_reason") or ""
    if merged.get("grounding_status") != "verified_error":
        for candidate in (dst, src):
            if candidate.get("grounding_status") == "verified_error":
                merged["grounding_status"] = "verified_error"
                merged["grounding_verified"] = True
                break
    return merged


def _merge_crosscheck_details(*detail_lists) -> list[dict]:
    by_model: dict[str, dict] = {}
    unnamed: list[dict] = []
    for details in detail_lists:
        for row in details or []:
            if not isinstance(row, dict):
                continue
            model = str(row.get("model") or row.get("resolved_model") or "").strip()
            if not model:
                unnamed.append(row)
                continue
            by_model[model] = {**by_model.get(model, {}), **row}
    return [*by_model.values(), *unnamed]


def _cluster_contextual_issues(issues: list[dict]) -> list[dict]:
    clustered: list[dict] = []
    for issue in issues:
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


def _claim_key(claim: dict) -> str:
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
        uid = claim.get("utterance_id", "")
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
        model_issues[model] = _dedupe_model_issues(r["issues"])

    models = list(model_issues.keys())
    union_map = {}
    for model in models:
        for key, issue in model_issues[model].items():
            if key not in union_map:
                merged = dict(issue)
                merged["detected_by_models"] = [model]
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
    for issue in _cluster_contextual_issues(list(union_map.values())):
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
