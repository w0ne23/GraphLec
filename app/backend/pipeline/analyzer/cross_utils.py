"""Shared helpers for cross-model verification."""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path


def _find_repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "analyzer").is_dir():
            return candidate
    return start


_ROOT = _find_repo_root(Path(__file__).resolve())
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from .claim_common import _empty_token_usage, _merge_token_usage


_ENV_KEYS = [
    "GOOGLE_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY_1",
    "GOOGLE_API_KEY_2",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "XAI_API_KEY",
    "XAI_BASE_URL",
    "XAI_DEFAULT_MODEL",
    "XAI_TIMEOUT_SEC",
    "VERIFIER_XAI_DEFAULT_MODEL",
    "VERIFIER_XAI_TIMEOUT_SEC",
    "GROQ_API_KEY",
    "CROSS_VERIFY_MODELS",
    "CROSS_VERIFY_MODEL",
    "CROSS_CHECK_MODELS",
    "CROSS_VERIFY_MODEL_WEIGHTS",
    "CROSS_VERIFY_SCORE_CONFIRM_THRESHOLD",
    "CROSS_VERIFY_SCORE_PROFESSOR_CHECK_THRESHOLD",
    "VERIFIER_MODEL",
    "VERIFIER_CLAIM_EXTRACT_MODEL",
    "VERIFIER_CLAIM_EXTRACT_PROMPT_PROFILE",
    "VERIFIER_CLAIM_JUDGE_MODEL",
    "VERIFIER_ISSUE_CLASSIFIER_MODEL",
    "VERIFIER_ISSUE_CLASSIFIER_MODELS",
    "VERIFIER_ISSUE_CLASSIFIER_BATCH_SIZE",
    "VERIFIER_ISSUE_CLASSIFIER_MAX_WORKERS",
    "VERIFIER_CROSS_RECHECK_MODEL",
    "VERIFIER_CROSSCHECK_GEMINI_MODEL",
    "VERIFIER_SLIDE_TYPO_MODEL",
    "VERIFIER_GROUNDING_MODEL",
    "VERIFIER_BATCH_SIZE",
    "VERIFIER_TEMPERATURE",
    "VERIFIER_PARSE_RETRIES",
    "VERIFIER_BATCH_RECOVERY_RETRIES",
    "VERIFIER_REQUIRE_COMPLETE",
    "VERIFIER_JUDGE_CONTEXT_MODE",
    "VERIFIER_JUDGE_CONTEXT_OVERLAP",
    "VERIFIER_CROSSCHECK_CONTEXT_MODE",
    "VERIFIER_CROSSCHECK_FOCUS_WINDOW",
    "VERIFIER_OPENAI_PROMPT_CACHE_KEY",
    "VERIFIER_OPENAI_PROMPT_CACHE_RETENTION",
    "OPENAI_PROMPT_CACHE_KEY",
    "OPENAI_PROMPT_CACHE_RETENTION",
    "VERIFIER_ANTHROPIC_PROMPT_CACHE",
    "VERIFIER_ANTHROPIC_PROMPT_CACHE_TTL",
    "ANTHROPIC_PROMPT_CACHE",
    "ANTHROPIC_PROMPT_CACHE_TTL",
    "VERIFIER_LLM_ISSUE_CLUSTERING",
    "VERIFIER_ISSUE_CLUSTER_MODEL",
    "VERIFIER_ISSUE_CLUSTER_MAX",
]

CLAIM_EXTRACT_MODEL = (
    os.getenv("VERIFIER_CLAIM_EXTRACT_MODEL", "").strip()
    or os.getenv("VERIFIER_MODEL", "").strip()
    or "gemini-2.5-flash"
)


def _format_token_summary(usage: dict) -> str:
    total = (usage or {}).get("total", {})
    parts = [
        f"input {int(total.get('input_tokens', 0) or 0):,}",
        f"output {int(total.get('output_tokens', 0) or 0):,}",
        f"reasoning {int(total.get('reasoning_tokens', 0) or 0):,}",
        f"total {int(total.get('total_tokens', 0) or 0):,}",
    ]
    cached = int(total.get("cached_input_tokens", 0) or 0)
    if cached:
        parts.append(f"cached {cached:,}")
    cache_write = int(total.get("cache_creation_input_tokens", 0) or 0)
    if cache_write:
        parts.append(f"cache_write {cache_write:,}")
    return " / ".join(parts)


def _collect_env_vars() -> dict:
    from dotenv import load_dotenv

    load_dotenv()
    return {k: os.environ.get(k) for k in _ENV_KEYS if os.environ.get(k)}


def _write_claims_jsonl(claims: list[dict], output_json_path: str | Path) -> str | None:
    if claims is None:
        return None
    output_json_path = Path(output_json_path)
    stem = output_json_path.name
    if stem.endswith("_content_verification.json"):
        prefix = stem[: -len("_content_verification.json")]
    else:
        prefix = output_json_path.stem
    out_path = output_json_path.with_name(f"{prefix}_claims_extracted.jsonl")
    with out_path.open("w", encoding="utf-8") as f:
        for claim in claims:
            f.write(json.dumps(claim, ensure_ascii=False) + "\n")
    return str(out_path)


def _load_claims_jsonl(path: str | Path) -> list[dict]:
    jsonl_path = Path(path)
    if not jsonl_path.exists():
        raise FileNotFoundError(f"claims jsonl 파일 없음: {jsonl_path}")

    claims = []
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            claims.append(_normalize_loaded_claim(payload))
    return claims


def _dedupe_ids(values: list[str]) -> list[str]:
    return list(dict.fromkeys(str(value).strip() for value in values if str(value).strip()))


def _normalize_loaded_claim(claim: dict) -> dict:
    """Keep antecedent utterances from reused claim files.

    Older claims may only mention antecedents in context_note. Preserve those IDs
    so verifier/crosscheck see the same local context used by claim extraction.
    """
    claim = dict(claim)
    note_ids = re.findall(r"\b(?:U\d{3,5}|S\d{3}(?:-C\d{3})?)\b", str(claim.get("context_note", "") or ""))
    antecedent_ids = claim.get("antecedent_context_ids")
    if isinstance(antecedent_ids, list):
        antecedent_ids = _dedupe_ids(antecedent_ids + note_ids)
    else:
        antecedent_ids = _dedupe_ids(note_ids)
    if antecedent_ids:
        claim["antecedent_context_ids"] = antecedent_ids
    utterance_ids = claim.get("utterance_ids")
    if isinstance(utterance_ids, list):
        claim["utterance_ids"] = _dedupe_ids(utterance_ids)
    else:
        uid = str(claim.get("utterance_id") or "").strip()
        claim["utterance_ids"] = _dedupe_ids([uid] if uid else [])
    if antecedent_ids and not claim.get("anchor_utterance_ids"):
        claim["anchor_utterance_ids"] = _dedupe_ids(antecedent_ids + claim["utterance_ids"])
    context_ids = claim.get("context_ids")
    if isinstance(context_ids, list):
        claim["context_ids"] = _dedupe_ids(context_ids)
    return claim


def _setup_worker(root: str, env_vars: dict, model: str):
    """subprocess 공통 초기화."""
    if root not in sys.path:
        sys.path.insert(0, root)
    for k, v in env_vars.items():
        if v is not None:
            os.environ[k] = v

    base_model = str(os.environ.get("VERIFIER_MODEL", "") or "").strip()
    explicit_slide_typo = str(os.environ.get("VERIFIER_SLIDE_TYPO_MODEL", "") or "").strip()

    os.environ["VERIFIER_MODEL"] = model
    os.environ["VERIFIER_CLAIM_EXTRACT_MODEL"] = model
    os.environ["VERIFIER_CLAIM_JUDGE_MODEL"] = model
    os.environ["VERIFIER_CROSS_RECHECK_MODEL"] = model
    os.environ["VERIFIER_SLIDE_TYPO_MODEL"] = explicit_slide_typo or base_model or "gemini-2.5-flash"
    os.environ["VERIFIER_GROUNDING_MODEL"] = model
    os.environ.setdefault("VERIFIER_TEMPERATURE", "0.0")
