"""Shared helpers for cross-model verification."""

from __future__ import annotations

import json
import os
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
    "GROQ_API_KEY",
    "CROSS_VERIFY_MODEL",
    "VERIFIER_MODEL",
    "VERIFIER_CLAIM_EXTRACT_MODEL",
    "VERIFIER_CLAIM_JUDGE_MODEL",
    "VERIFIER_CROSS_RECHECK_MODEL",
    "VERIFIER_SLIDE_RECHECK_MODEL",
    "VERIFIER_GROUNDING_MODEL",
    "VERIFIER_BATCH_SIZE",
    "VERIFIER_TEMPERATURE",
    "VERIFIER_PARSE_RETRIES",
    "VERIFIER_BATCH_RECOVERY_RETRIES",
    "VERIFIER_REQUIRE_COMPLETE",
]

CLAIM_EXTRACT_MODEL = "gemini-2.5-flash"


def _format_token_summary(usage: dict) -> str:
    total = (usage or {}).get("total", {})
    return (
        f"input {int(total.get('input_tokens', 0) or 0):,} / "
        f"output {int(total.get('output_tokens', 0) or 0):,} / "
        f"reasoning {int(total.get('reasoning_tokens', 0) or 0):,} / "
        f"total {int(total.get('total_tokens', 0) or 0):,}"
    )


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


def _setup_worker(root: str, env_vars: dict, model: str):
    """subprocess 공통 초기화."""
    if root not in sys.path:
        sys.path.insert(0, root)
    for k, v in env_vars.items():
        if v is not None:
            os.environ[k] = v

    explicit_cross_recheck = str(os.environ.get("VERIFIER_CROSS_RECHECK_MODEL", "") or "").strip()
    explicit_slide_recheck = str(os.environ.get("VERIFIER_SLIDE_RECHECK_MODEL", "") or "").strip()

    os.environ["VERIFIER_MODEL"] = model
    os.environ["VERIFIER_CLAIM_EXTRACT_MODEL"] = model
    os.environ["VERIFIER_CLAIM_JUDGE_MODEL"] = model
    if model.startswith("gemini") and explicit_cross_recheck:
        os.environ["VERIFIER_CROSS_RECHECK_MODEL"] = explicit_cross_recheck
    else:
        os.environ["VERIFIER_CROSS_RECHECK_MODEL"] = model
    os.environ["VERIFIER_SLIDE_RECHECK_MODEL"] = explicit_slide_recheck or model
    os.environ["VERIFIER_GROUNDING_MODEL"] = model
    os.environ.setdefault("VERIFIER_TEMPERATURE", "0.0")
