"""Claim verification common helpers."""

import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from google.genai import types

from config import (
    get_anthropic_client,
    get_deepseek_client,
    get_gemini_client_sequence,
    get_openai_client,
    get_xai_client,
    resolve_anthropic_model,
)
from utils import api_call_with_retry, is_retryable_api_error


# ── LLM 추상화 레이어 ────────────────────────────────────────


def _default_judge_model(base_model: str) -> str:
    base = str(base_model or "").strip()
    if base == "gpt-5.4-mini":
        return "gpt-5.1-medium"
    return base


def _resolve_stage_model(stage: str) -> str:
    base = os.getenv("VERIFIER_MODEL", VERIFIER_MODEL).strip() or VERIFIER_MODEL
    extract_model = os.getenv("VERIFIER_CLAIM_EXTRACT_MODEL", VERIFIER_CLAIM_EXTRACT_MODEL).strip()
    judge_model = os.getenv("VERIFIER_CLAIM_JUDGE_MODEL", VERIFIER_CLAIM_JUDGE_MODEL).strip()
    cross_recheck_model = os.getenv("VERIFIER_CROSS_RECHECK_MODEL", VERIFIER_CROSS_RECHECK_MODEL).strip()
    slide_typo_model = os.getenv("VERIFIER_SLIDE_TYPO_MODEL", VERIFIER_SLIDE_TYPO_MODEL).strip()
    grounding_model = os.getenv("VERIFIER_GROUNDING_MODEL", VERIFIER_GROUNDING_MODEL).strip()
    strong = judge_model or _default_judge_model(base)

    if stage == "extract":
        return extract_model or base
    if stage == "judge":
        return strong
    if stage == "cross_recheck":
        return cross_recheck_model or strong
    if stage == "slide_typo":
        return slide_typo_model or strong
    if stage == "grounding":
        return grounding_model or strong
    return base


def _parse_openai_model_spec(model_spec: str) -> tuple[str, Optional[str]]:
    spec = str(model_spec or "").strip()
    if not spec:
        return spec, None

    match = re.match(
        r"^(?P<model>(?:gpt|o)[A-Za-z0-9.-]*?)-(?P<effort>low|medium|high|xhigh)$",
        spec,
    )
    if match:
        return match.group("model"), match.group("effort")
    return spec, None


def _should_send_openai_temperature(model: str, reasoning_effort: Optional[str]) -> bool:
    if reasoning_effort:
        return False
    return True


def _is_anthropic_model(model: str) -> bool:
    spec = str(model or "").strip().lower()
    return (
        spec.startswith("claude")
        or spec in {
            "haiku-4.5",
            "claude-haiku-4.5",
            "claude-haiku-4-5",
            "sonnet-4.5",
            "claude-sonnet-4.5",
            "claude-sonnet-4-5",
            "opus-4.5",
            "claude-opus-4.5",
            "claude-opus-4-5",
        }
    )


def _is_xai_model(model: str) -> bool:
    spec = str(model or "").strip().lower()
    return spec.startswith("grok") or spec.startswith("xai:")


def _is_deepseek_model(model: str) -> bool:
    spec = str(model or "").strip().lower()
    return spec.startswith("deepseek") or spec.startswith("deepseek:")


def _resolve_xai_model(model: str) -> str:
    spec = str(model or "").strip()
    lowered = spec.lower()
    if lowered.startswith("xai:"):
        return spec.split(":", 1)[1].strip()
    if lowered.startswith("xai/"):
        return spec.split("/", 1)[1].strip()
    return spec


def _resolve_deepseek_model(model: str) -> str:
    spec = str(model or "").strip()
    lowered = spec.lower()
    if lowered in {"deepseek", "deepseek-default"}:
        return "deepseek-v4-flash"
    if lowered.startswith("deepseek:"):
        return spec.split(":", 1)[1].strip()
    if lowered.startswith("deepseek/"):
        return spec.split("/", 1)[1].strip()
    return spec


def _supports_json_object_response_format(model: str) -> bool:
    lowered = str(model or "").strip().lower()
    return (
        lowered.startswith(("gpt", "o1", "o3"))
        or lowered.startswith("grok")
        or lowered.startswith(("xai:", "xai/"))
        or lowered.startswith("deepseek")
    )


TOKEN_USAGE_STAGES = ("extract", "judge", "slide_typo", "grounding", "cross_recheck")
TOKEN_USAGE_FIELDS = (
    "input_tokens",
    "output_tokens",
    "reasoning_tokens",
    "tool_input_tokens",
    "cached_input_tokens",
    "cache_creation_input_tokens",
    "total_tokens",
)


def _new_token_bucket() -> dict:
    return {field: 0 for field in TOKEN_USAGE_FIELDS}


def _empty_token_usage() -> dict:
    usage = {stage: _new_token_bucket() for stage in TOKEN_USAGE_STAGES}
    usage["total"] = _new_token_bucket()
    return usage


def _safe_int(value) -> int:
    try:
        if value is None:
            return 0
        return int(value)
    except Exception:
        return 0


def _usage_value(obj, *names) -> int:
    for name in names:
        if isinstance(obj, dict):
            value = obj.get(name)
        else:
            value = getattr(obj, name, None)
        if value is not None:
            return _safe_int(value)
    return 0


def _openai_prompt_cache_key(stage: str) -> str | None:
    base = (
        os.getenv("VERIFIER_OPENAI_PROMPT_CACHE_KEY", "")
        or os.getenv("OPENAI_PROMPT_CACHE_KEY", "")
        or "graphlec-verifier"
    ).strip()
    if base.lower() in {"", "0", "false", "off", "none"}:
        return None
    safe_base = re.sub(r"[^A-Za-z0-9_.:-]+", "-", base).strip("-") or "graphlec-verifier"
    safe_stage = re.sub(r"[^A-Za-z0-9_.:-]+", "-", str(stage or "default")).strip("-") or "default"
    return f"{safe_base}:{safe_stage}"[:128]


def _openai_prompt_cache_retention() -> str | None:
    raw = (
        os.getenv("VERIFIER_OPENAI_PROMPT_CACHE_RETENTION", "")
        or os.getenv("OPENAI_PROMPT_CACHE_RETENTION", "")
    ).strip().lower()
    if not raw or raw in {"0", "false", "off", "none"}:
        return None
    aliases = {
        "in_memory": "in-memory",
        "in-memory": "in-memory",
        "memory": "in-memory",
        "24h": "24h",
        "extended": "24h",
    }
    return aliases.get(raw)


def _anthropic_prompt_cache_control() -> dict | None:
    raw = (
        os.getenv("VERIFIER_ANTHROPIC_PROMPT_CACHE", "")
        or os.getenv("ANTHROPIC_PROMPT_CACHE", "")
        or "1"
    ).strip().lower()
    if raw in {"0", "false", "off", "none"}:
        return None
    ttl = (
        os.getenv("VERIFIER_ANTHROPIC_PROMPT_CACHE_TTL", "")
        or os.getenv("ANTHROPIC_PROMPT_CACHE_TTL", "")
        or "5m"
    ).strip().lower()
    control = {"type": "ephemeral"}
    if ttl == "1h":
        control["ttl"] = "1h"
    return control


def _deepseek_thinking_extra_body() -> dict | None:
    raw = (
        os.getenv("VERIFIER_DEEPSEEK_THINKING", "")
        or os.getenv("DEEPSEEK_THINKING", "")
        or "disabled"
    ).strip().lower()
    aliases = {
        "0": "disabled",
        "false": "disabled",
        "off": "disabled",
        "none": "disabled",
        "no": "disabled",
        "disable": "disabled",
        "disabled": "disabled",
        "1": "enabled",
        "true": "enabled",
        "on": "enabled",
        "yes": "enabled",
        "enable": "enabled",
        "enabled": "enabled",
    }
    thinking_type = aliases.get(raw, raw)
    if thinking_type not in {"enabled", "disabled"}:
        thinking_type = "disabled"
    return {"thinking": {"type": thinking_type}}


def _deepseek_reasoning_effort() -> str | None:
    raw = (
        os.getenv("VERIFIER_DEEPSEEK_REASONING_EFFORT", "")
        or os.getenv("DEEPSEEK_REASONING_EFFORT", "")
    ).strip().lower()
    if raw in {"high", "max"}:
        return raw
    return None


def _deepseek_request_timeout() -> float:
    raw = (
        os.getenv("VERIFIER_DEEPSEEK_TIMEOUT_SEC", "")
        or os.getenv("DEEPSEEK_TIMEOUT_SEC", "")
        or "180"
    )
    try:
        return max(30.0, float(raw))
    except (TypeError, ValueError):
        return 180.0


def _deepseek_api_retry_config() -> tuple[int, float]:
    try:
        max_retries = int(os.getenv("VERIFIER_DEEPSEEK_API_MAX_RETRIES", "4") or "4")
    except ValueError:
        max_retries = 4
    try:
        initial_wait = float(os.getenv("VERIFIER_DEEPSEEK_API_INITIAL_WAIT", "5") or "5")
    except ValueError:
        initial_wait = 5.0
    return max(1, max_retries), max(0.0, initial_wait)


def _join_system_and_prompt(system_prompt: str | None, prompt: str) -> str:
    if not system_prompt:
        return prompt
    return f"{system_prompt.rstrip()}\n\n{prompt.lstrip()}"


def _merge_token_usage(*usages: dict) -> dict:
    merged = _empty_token_usage()
    for usage in usages:
        if not isinstance(usage, dict):
            continue
        if usage.get("stage"):
            _add_call_usage(merged, usage)
            continue
        for stage in TOKEN_USAGE_STAGES:
            bucket = usage.get(stage)
            if not isinstance(bucket, dict):
                continue
            for field in TOKEN_USAGE_FIELDS:
                merged[stage][field] += _safe_int(bucket.get(field))

    merged["total"] = _new_token_bucket()
    for stage in TOKEN_USAGE_STAGES:
        for field in TOKEN_USAGE_FIELDS:
            merged["total"][field] += merged[stage][field]
    return merged


def _add_call_usage(token_usage: dict, call_usage: dict) -> dict:
    if not isinstance(token_usage, dict):
        token_usage = _empty_token_usage()
    if not isinstance(call_usage, dict):
        return token_usage

    stage = str(call_usage.get("stage", "") or "")
    if not stage:
        return token_usage
    if stage not in token_usage:
        token_usage[stage] = _new_token_bucket()

    for field in TOKEN_USAGE_FIELDS:
        value = _safe_int(call_usage.get(field))
        token_usage[stage][field] += value
        token_usage["total"][field] += value
    return token_usage


def _extract_openai_usage(resp, model: str, stage: str) -> dict:
    usage = getattr(resp, "usage", None)
    completion_details = getattr(usage, "completion_tokens_details", None)
    prompt_details = getattr(usage, "prompt_tokens_details", None)
    return {
        "provider": "openai",
        "model": model,
        "stage": stage,
        "input_tokens": _usage_value(usage, "prompt_tokens"),
        "output_tokens": _usage_value(usage, "completion_tokens"),
        "reasoning_tokens": _usage_value(completion_details, "reasoning_tokens"),
        "tool_input_tokens": 0,
        "cached_input_tokens": _usage_value(prompt_details, "cached_tokens"),
        "cache_creation_input_tokens": 0,
        "total_tokens": _usage_value(usage, "total_tokens"),
    }


def _extract_xai_usage(resp, model: str, stage: str) -> dict:
    usage = getattr(resp, "usage", None)
    completion_details = getattr(usage, "completion_tokens_details", None)
    prompt_details = getattr(usage, "prompt_tokens_details", None)
    input_tokens = _usage_value(usage, "prompt_tokens", "input_tokens")
    output_tokens = _usage_value(usage, "completion_tokens", "output_tokens")
    total_tokens = _usage_value(usage, "total_tokens")
    return {
        "provider": "xai",
        "model": model,
        "stage": stage,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "reasoning_tokens": _usage_value(completion_details, "reasoning_tokens"),
        "tool_input_tokens": 0,
        "cached_input_tokens": _usage_value(prompt_details, "cached_tokens"),
        "cache_creation_input_tokens": 0,
        "total_tokens": total_tokens or input_tokens + output_tokens,
    }


def _extract_deepseek_usage(resp, model: str, stage: str) -> dict:
    usage = getattr(resp, "usage", None)
    completion_details = getattr(usage, "completion_tokens_details", None)
    prompt_details = getattr(usage, "prompt_tokens_details", None)
    input_tokens = _usage_value(usage, "prompt_tokens", "input_tokens")
    output_tokens = _usage_value(usage, "completion_tokens", "output_tokens")
    total_tokens = _usage_value(usage, "total_tokens")
    return {
        "provider": "deepseek",
        "model": model,
        "stage": stage,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "reasoning_tokens": _usage_value(completion_details, "reasoning_tokens"),
        "tool_input_tokens": 0,
        "cached_input_tokens": _usage_value(prompt_details, "cached_tokens"),
        "cache_creation_input_tokens": 0,
        "total_tokens": total_tokens or input_tokens + output_tokens,
    }


def _extract_gemini_usage(resp, model: str, stage: str) -> dict:
    usage = getattr(resp, "usage_metadata", None) or getattr(resp, "usageMetadata", None)
    return {
        "provider": "gemini",
        "model": model,
        "stage": stage,
        "input_tokens": _safe_int(
            getattr(usage, "prompt_token_count", None) or getattr(usage, "promptTokenCount", None)
        ),
        "output_tokens": _safe_int(
            getattr(usage, "candidates_token_count", None) or getattr(usage, "candidatesTokenCount", None)
        ),
        "reasoning_tokens": _safe_int(
            getattr(usage, "thoughts_token_count", None) or getattr(usage, "thoughtsTokenCount", None)
        ),
        "tool_input_tokens": _safe_int(
            getattr(usage, "tool_use_prompt_token_count", None)
            or getattr(usage, "toolUsePromptTokenCount", None)
        ),
        "cached_input_tokens": _safe_int(
            getattr(usage, "cached_content_token_count", None)
            or getattr(usage, "cachedContentTokenCount", None)
        ),
        "total_tokens": _safe_int(
            getattr(usage, "total_token_count", None) or getattr(usage, "totalTokenCount", None)
        ),
    }


def _extract_anthropic_usage(resp, model: str, stage: str) -> dict:
    usage = getattr(resp, "usage", None)
    input_tokens = _usage_value(usage, "input_tokens")
    output_tokens = _usage_value(usage, "output_tokens")
    cached_tokens = _usage_value(usage, "cache_read_input_tokens")
    cache_creation_tokens = _usage_value(usage, "cache_creation_input_tokens")
    return {
        "provider": "anthropic",
        "model": model,
        "stage": stage,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "reasoning_tokens": 0,
        "tool_input_tokens": 0,
        "cached_input_tokens": cached_tokens,
        "cache_creation_input_tokens": cache_creation_tokens,
        "total_tokens": input_tokens + output_tokens + cached_tokens + cache_creation_tokens,
    }

def _call_llm(
    prompt: str,
    system_prompt: str = None,
    max_tokens: int = 8192,
    temperature: float = None,
    image_bytes: bytes = None,
    image_bytes_list: list[bytes] = None,
    use_grounding: bool = False,
    thinking_budget: int = 1024,
    thinking_level: str = None,
    response_format: dict = None,
    stage: str = "default",
) -> tuple[str, dict]:
    """모델에 무관한 단일 LLM 호출. (응답 텍스트, usage)를 반환."""
    temp = temperature if temperature is not None else VERIFIER_TEMPERATURE

    model_spec = _resolve_stage_model(stage)
    model, reasoning_effort = _parse_openai_model_spec(model_spec)

    # ── OpenAI ──────────────────────────────────────────────
    if model.startswith("gpt") or model.startswith("o1") or model.startswith("o3"):
        import base64
        client = get_openai_client()
        if client is None:
            raise RuntimeError("OPENAI_API_KEY가 설정되지 않았습니다.")

        image_payloads = []
        if image_bytes_list:
            image_payloads.extend([b for b in image_bytes_list if b])
        elif image_bytes:
            image_payloads.append(image_bytes)

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        if image_payloads:
            content = []
            for img in image_payloads:
                b64 = base64.b64encode(img).decode()
                content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
            content.append({"type": "text", "text": prompt})
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": prompt})

        def call_api():
            kwargs = dict(
                model=model,
                messages=messages,
                max_completion_tokens=max_tokens,
            )
            prompt_cache_key = _openai_prompt_cache_key(stage)
            if prompt_cache_key:
                kwargs["prompt_cache_key"] = prompt_cache_key
            prompt_cache_retention = _openai_prompt_cache_retention()
            if prompt_cache_retention:
                kwargs["prompt_cache_retention"] = prompt_cache_retention
            if _should_send_openai_temperature(model, reasoning_effort):
                kwargs["temperature"] = temp
            if reasoning_effort:
                kwargs["reasoning_effort"] = reasoning_effort
            if response_format is not None:
                kwargs["response_format"] = response_format
            try:
                return client.chat.completions.create(**kwargs)
            except TypeError as e:
                if "prompt_cache_" not in str(e):
                    raise
                kwargs.pop("prompt_cache_key", None)
                kwargs.pop("prompt_cache_retention", None)
                return client.chat.completions.create(**kwargs)

        resp = api_call_with_retry(call_api)
        return resp.choices[0].message.content or "", _extract_openai_usage(resp, model, stage)

    # ── xAI Grok ────────────────────────────────────────────
    if _is_xai_model(model):
        import base64

        resolved_model = _resolve_xai_model(model)
        client = get_xai_client()
        if client is None:
            raise RuntimeError("XAI_API_KEY가 설정되지 않았습니다.")

        image_payloads = []
        if image_bytes_list:
            image_payloads.extend([b for b in image_bytes_list if b])
        elif image_bytes:
            image_payloads.append(image_bytes)

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        if image_payloads:
            content = []
            for img in image_payloads:
                b64 = base64.b64encode(img).decode()
                content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
            content.append({"type": "text", "text": prompt})
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": prompt})

        def call_api():
            kwargs = dict(
                model=resolved_model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temp,
            )
            if response_format is not None:
                kwargs["response_format"] = response_format
            return client.chat.completions.create(**kwargs, timeout=_deepseek_request_timeout())

        resp = api_call_with_retry(call_api)
        return resp.choices[0].message.content or "", _extract_xai_usage(resp, resolved_model, stage)

    # ── DeepSeek ───────────────────────────────────────────
    if _is_deepseek_model(model):
        import base64

        resolved_model = _resolve_deepseek_model(model)
        client = get_deepseek_client()
        if client is None:
            raise RuntimeError("DEEPSEEK_API_KEY가 설정되지 않았습니다.")

        image_payloads = []
        if image_bytes_list:
            image_payloads.extend([b for b in image_bytes_list if b])
        elif image_bytes:
            image_payloads.append(image_bytes)

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        if image_payloads:
            content = []
            for img in image_payloads:
                b64 = base64.b64encode(img).decode()
                content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
            content.append({"type": "text", "text": prompt})
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": prompt})

        def call_api():
            thinking_extra_body = _deepseek_thinking_extra_body()
            kwargs = dict(
                model=resolved_model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temp,
                extra_body=thinking_extra_body,
            )
            reasoning_effort = _deepseek_reasoning_effort()
            if reasoning_effort and (thinking_extra_body or {}).get("thinking", {}).get("type") == "enabled":
                kwargs["reasoning_effort"] = reasoning_effort
            if response_format is not None:
                kwargs["response_format"] = response_format
            return client.chat.completions.create(**kwargs, timeout=_deepseek_request_timeout())

        max_retries, initial_wait = _deepseek_api_retry_config()
        resp = api_call_with_retry(call_api, max_retries=max_retries, initial_wait=initial_wait)
        return resp.choices[0].message.content or "", _extract_deepseek_usage(resp, resolved_model, stage)

    # ── Anthropic ───────────────────────────────────────────
    if _is_anthropic_model(model):
        import base64

        client = get_anthropic_client()
        if client is None:
            raise RuntimeError("ANTHROPIC_API_KEY가 설정되지 않았습니다.")

        resolved_model = resolve_anthropic_model(model)
        content = []
        if image_bytes_list:
            for img in image_bytes_list:
                if img:
                    content.append(
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": base64.b64encode(img).decode(),
                            },
                        }
                    )
        elif image_bytes:
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": base64.b64encode(image_bytes).decode(),
                    },
                }
            )
        content.append({"type": "text", "text": prompt})

        def call_api():
            kwargs = dict(
                model=resolved_model,
                max_tokens=max_tokens,
                temperature=temp,
                messages=[{"role": "user", "content": content}],
            )
            if system_prompt:
                system_block = {"type": "text", "text": system_prompt}
                cache_control = _anthropic_prompt_cache_control()
                if cache_control:
                    system_block["cache_control"] = cache_control
                kwargs["system"] = [system_block]
            return client.messages.create(**kwargs)

        resp = api_call_with_retry(call_api)
        text_blocks = [
            getattr(block, "text", "")
            for block in getattr(resp, "content", []) or []
            if getattr(block, "type", "") == "text"
        ]
        return "".join(text_blocks), _extract_anthropic_usage(resp, resolved_model, stage)

    # ── Gemini ──────────────────────────────────────────────
    contents = []
    if image_bytes_list:
        for img in image_bytes_list:
            if img:
                contents.append(types.Part.from_bytes(data=img, mime_type="image/jpeg"))
    elif image_bytes:
        contents.append(types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"))
    contents.append(types.Part.from_text(text=_join_system_and_prompt(system_prompt, prompt)))

    cfg_kwargs: dict = dict(temperature=temp, max_output_tokens=max_tokens)
    if use_grounding:
        cfg_kwargs["tools"] = [types.Tool(google_search=types.GoogleSearch())]
    else:
        if thinking_level is not None:
            cfg_kwargs["thinking_config"] = types.ThinkingConfig(thinking_level=thinking_level)
        else:
            cfg_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=thinking_budget)
        cfg_kwargs["response_mime_type"] = "application/json"

    client_sequence = get_gemini_client_sequence()
    if len(client_sequence) == 1:
        _client_name, client = client_sequence[0]

        def call_api():
            return client.models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(**cfg_kwargs),
            )

        resp = api_call_with_retry(call_api)
        return resp.text or "", _extract_gemini_usage(resp, model, stage)

    last_exc = None
    for idx, (client_name, client) in enumerate(client_sequence):
        try:
            resp = client.models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(**cfg_kwargs),
            )
            return resp.text or "", _extract_gemini_usage(resp, model, stage)
        except Exception as e:
            last_exc = e
            if is_retryable_api_error(e) and idx < len(client_sequence) - 1:
                print(f"GEMINI API ERROR [{client_name}]: {e}")
                print("  ↺ 다음 Gemini API 키로 전환")
                continue
            raise

    if last_exc is not None:
        raise last_exc
    raise RuntimeError("Gemini 호출 실패: 사용 가능한 API 키가 없습니다.")


# ── 설정 ──────────────────────────────────────────────────

BATCH_SIZE = int(os.getenv("VERIFIER_BATCH_SIZE", "15"))
VERIFIER_MODEL = os.getenv("VERIFIER_MODEL", "gemini-2.5-flash")
VERIFIER_CLAIM_EXTRACT_MODEL = os.getenv("VERIFIER_CLAIM_EXTRACT_MODEL", "")
VERIFIER_CLAIM_JUDGE_MODEL = os.getenv("VERIFIER_CLAIM_JUDGE_MODEL", "")
VERIFIER_CROSS_RECHECK_MODEL = os.getenv("VERIFIER_CROSS_RECHECK_MODEL", "")
VERIFIER_SLIDE_TYPO_MODEL = os.getenv("VERIFIER_SLIDE_TYPO_MODEL", "")
VERIFIER_GROUNDING_MODEL = os.getenv("VERIFIER_GROUNDING_MODEL", "")
VERIFIER_TEMPERATURE = float(os.getenv("VERIFIER_TEMPERATURE", "0.0"))
ISSUE_TYPE_LABELS = {
    "factual_error": "발언 자체 오류",
    "temporal_error": "시대적 오류",
    "confusing_explanation": "혼동 가능 설명",
    "scope_overclaim": "범위 과잉 단정",
}
FACT_GROUNDED_ISSUE_TYPES = {"factual_error", "temporal_error", "scope_overclaim"}
PEDAGOGICAL_ISSUE_TYPES = {"confusing_explanation"}
ALLOWED_ISSUE_TYPES = set(ISSUE_TYPE_LABELS)
ISSUE_VERIFICATION_BASIS_VALUES = {
    "external_factual",
    "temporal_factual",
    "lecture_context",
    "pedagogical_risk",
    "insufficient_context",
}
ISSUE_EVIDENCE_NEED_VALUES = {"external_grounding", "lecture_context_only", "professor_check"}
ISSUE_CLAIM_SCOPE_VALUES = {"explicit", "contextualized", "overgeneralized_from_context"}
ISSUE_REVIEW_PRIORITY_VALUES = {"high", "medium", "low"}
ISSUE_METADATA_FIELDS = (
    "verification_basis",
    "evidence_need",
    "claim_scope",
    "review_priority",
)
VERIFIER_PARSE_RETRIES = int(os.getenv("VERIFIER_PARSE_RETRIES", "2"))
VERIFIER_BATCH_RECOVERY_RETRIES = int(os.getenv("VERIFIER_BATCH_RECOVERY_RETRIES", "1"))
VERIFIER_REQUIRE_COMPLETE = os.getenv("VERIFIER_REQUIRE_COMPLETE", "1") != "0"


def normalize_issue_type(value: str) -> str:
    raw = str(value or "").strip().lower()
    aliases = {
        "outdated": "temporal_error",
        "currentness_error": "temporal_error",
        "temporal": "temporal_error",
        "missing_condition": "scope_overclaim",
        "clarification_needed": "scope_overclaim",
        "needs_clarification": "scope_overclaim",
        "overgeneralization": "scope_overclaim",
        "overclaim": "scope_overclaim",
        "ambiguous_expression": "confusing_explanation",
        "misleading_explanation": "confusing_explanation",
        "misconception_risk": "confusing_explanation",
        "ambiguous": "confusing_explanation",
        "ambiguity": "confusing_explanation",
        "misleading": "confusing_explanation",
        "misunderstanding_risk": "confusing_explanation",
        "student_misunderstanding": "confusing_explanation",
    }
    return aliases.get(raw, raw)


def issue_type_label(issue_type: str) -> str:
    return ISSUE_TYPE_LABELS.get(normalize_issue_type(issue_type), str(issue_type or "unknown"))


def normalize_issue_metadata(issue: dict, issue_type: str | None = None) -> dict:
    """Normalize stage-to-stage metadata so old/partial judge outputs do not break later phases."""
    if not isinstance(issue, dict):
        issue = {}
    normalized_type = normalize_issue_type(issue_type if issue_type is not None else issue.get("type", ""))

    basis_aliases = {
        "fact": "external_factual",
        "factual": "external_factual",
        "external": "external_factual",
        "temporal": "temporal_factual",
        "currentness": "temporal_factual",
        "context": "lecture_context",
        "pedagogical": "pedagogical_risk",
        "teaching": "pedagogical_risk",
        "review": "insufficient_context",
        "insufficient": "insufficient_context",
        "uncertain": "insufficient_context",
    }
    evidence_aliases = {
        "grounding": "external_grounding",
        "external": "external_grounding",
        "search": "external_grounding",
        "context": "lecture_context_only",
        "lecture": "lecture_context_only",
        "review": "professor_check",
        "human": "professor_check",
        "manual": "professor_check",
    }
    scope_aliases = {
        "explicit_claim": "explicit",
        "context": "contextualized",
        "contextual": "contextualized",
        "overgeneralized": "overgeneralized_from_context",
        "too_broad": "overgeneralized_from_context",
    }

    basis = str(issue.get("verification_basis", "") or "").strip().lower()
    basis = basis_aliases.get(basis, basis)
    evidence_need = str(issue.get("evidence_need", "") or "").strip().lower()
    evidence_need = evidence_aliases.get(evidence_need, evidence_need)
    claim_scope = str(issue.get("claim_scope", "") or "").strip().lower()
    claim_scope = scope_aliases.get(claim_scope, claim_scope)
    review_priority = str(issue.get("review_priority", "") or "").strip().lower()

    if basis not in ISSUE_VERIFICATION_BASIS_VALUES:
        if normalized_type == "temporal_error":
            basis = "temporal_factual"
        elif normalized_type in {"factual_error", "scope_overclaim"}:
            basis = "external_factual"
        elif normalized_type in PEDAGOGICAL_ISSUE_TYPES:
            basis = "pedagogical_risk"
        else:
            basis = "lecture_context"

    if evidence_need not in ISSUE_EVIDENCE_NEED_VALUES:
        if basis in {"external_factual", "temporal_factual"}:
            evidence_need = "external_grounding"
        elif basis == "insufficient_context":
            evidence_need = "professor_check"
        else:
            evidence_need = "lecture_context_only"

    if claim_scope not in ISSUE_CLAIM_SCOPE_VALUES:
        claim_scope = "explicit"

    if review_priority not in ISSUE_REVIEW_PRIORITY_VALUES:
        review_priority = "high"

    issue["verification_basis"] = basis
    issue["evidence_need"] = evidence_need
    issue["claim_scope"] = claim_scope
    issue["review_priority"] = review_priority
    return issue


def should_drop_issue_by_metadata(issue: dict) -> bool:
    normalize_issue_metadata(issue)
    return issue.get("review_priority") == "low"


def should_route_issue_to_professor_check(issue: dict) -> bool:
    normalize_issue_metadata(issue)
    return (
        issue.get("verification_basis") == "insufficient_context"
        or issue.get("evidence_need") == "professor_check"
        or issue.get("claim_scope") == "overgeneralized_from_context"
    )


def is_pedagogical_or_context_issue(issue: dict) -> bool:
    normalize_issue_metadata(issue)
    issue_type = normalize_issue_type(issue.get("type", ""))
    return (
        issue_type in PEDAGOGICAL_ISSUE_TYPES
        or issue.get("verification_basis") in {"pedagogical_risk", "lecture_context"}
        or issue.get("evidence_need") == "lecture_context_only"
    )


def should_review_single_model_pedagogical_issue(issue: dict, total_models: int) -> bool:
    normalize_issue_metadata(issue)
    detected_by = issue.get("detected_by_models")
    if not isinstance(detected_by, list):
        detected_by = []
    if total_models <= 1 or len(set(detected_by)) != 1:
        return False
    return is_pedagogical_or_context_issue(issue)


def is_fact_grounded_issue(issue: dict) -> bool:
    normalize_issue_metadata(issue)
    return normalize_issue_type(issue.get("type", "")) in FACT_GROUNDED_ISSUE_TYPES


def copy_issue_metadata(dst: dict, src: dict) -> dict:
    """Copy normalized issue metadata without overwriting already meaningful destination values."""
    if not isinstance(dst, dict):
        dst = {}
    if not isinstance(src, dict):
        return dst
    normalize_issue_metadata(src)
    for field in ISSUE_METADATA_FIELDS:
        if not dst.get(field) and src.get(field):
            dst[field] = src[field]
    return normalize_issue_metadata(dst)


def metadata_review_reason(issue: dict) -> str:
    normalize_issue_metadata(issue)
    reasons = []
    if issue.get("verification_basis") == "insufficient_context":
        reasons.append("문맥 부족")
    if issue.get("evidence_need") == "professor_check":
        reasons.append("교수 확인 필요")
    if issue.get("claim_scope") == "overgeneralized_from_context":
        reasons.append("원문보다 넓게 일반화된 claim")
    return ", ".join(reasons) or "자동 확정보다 교수 확인이 필요한 후보"



def build_verification_question(claim: dict) -> str:
    """후속 판정에서 살아남은 claim에만 짧은 검증 질문을 생성."""
    text = str(
        claim.get("resolved_claim")
        or claim.get("claim_text")
        or claim.get("problematic_content")
        or ""
    ).strip()
    if not text:
        return ""

    text = " ".join(text.split()).strip(" \t\r\n.。?？!！")
    if not text:
        return ""

    claim_type = str(claim.get("claim_type", "") or "").strip()
    if claim_type == "numeric":
        return f"'{text}'라는 수치나 기준이 정확한가?"
    if claim_type == "causal":
        return f"'{text}'라는 인과 또는 작동 방식 설명이 타당한가?"
    if claim_type == "relationship":
        return f"'{text}'라는 개념 간 관계 설명이 타당한가?"
    if claim_type == "currentness":
        return f"'{text}'라는 현행성 설명이 현재 기준으로 타당한가?"
    return f"'{text}'라는 설명이 타당한가?"


# ── 도메인 힌트 ──────────────────────────────────────────


def _normalize_sub_domain(value: str) -> str:
    return str(value or "").strip()


def _get_domain_hint(domain: str, sub_domain: str) -> dict:
    domain_value = str(domain or "").strip()
    sub_domain_value = _normalize_sub_domain(sub_domain)
    label_parts = [part for part in (domain_value, sub_domain_value) if part]
    return {"label": " > ".join(label_parts) if label_parts else "일반"}


def _resolve_domain_fields(merged: dict) -> tuple[str, str]:
    """merged.json에서 domain/sub_domain을 읽고 정규화."""
    domain = str(
        merged.get("domain")
        or merged.get("primary_domain")
        or ""
    ).strip()
    sub_domain = str(
        merged.get("sub_domain")
        or merged.get("subdomain")
        or merged.get("secondary_domain")
        or ""
    ).strip()
    sub_domain = _normalize_sub_domain(sub_domain)
    return domain, sub_domain


# ── 유틸리티 ──────────────────────────────────────────────

def _compact_text(value: str) -> str:
    return re.sub(r"\s+", "", (value or "")).lower()


def _issue_key(issue: dict) -> tuple:
    return (
        issue.get("type", ""),
        int(issue.get("slide_number", 0) or 0),
        int(round(float(issue.get("start_time", 0) or 0))),
        _compact_text(str(issue.get("problematic_content", "") or ""))[:60],
    )


def _dedupe_issues(issues: list[dict]) -> list[dict]:
    dedup = {}
    for issue in issues:
        key = _issue_key(issue)
        cur = dedup.get(key)
        if cur is None or float(issue.get("confidence", 0) or 0) > float(cur.get("confidence", 0) or 0):
            dedup[key] = issue
    return sorted(dedup.values(), key=lambda x: float(x.get("start_time", 0) or 0))


def _normalize_severity(issue: dict) -> None:
    severity = str(issue.get("severity", "")).lower()
    if severity not in {"critical", "major", "minor"}:
        severity = "major"
    if severity == "critical" and float(issue.get("confidence", 0) or 0) < 0.9:
        severity = "major"
    issue["severity"] = severity


def _is_asr_artifact(issue: dict, utt_map: dict) -> bool:
    uid = issue.get("utterance_id", "")
    ref = utt_map.get(uid)
    if not ref:
        return False
    orig = _compact_text(ref.get("text_original", ""))
    corr = _compact_text(ref.get("text_corrected", ""))
    if not orig or not corr or orig == corr:
        return False
    snippet = _compact_text(str(issue.get("problematic_content", "") or ""))
    if len(snippet) < 6:
        return False
    if snippet in corr:
        return False
    if snippet in orig:
        return True
    return False


def _make_result(
    issues: list[dict],
    api_calls: int,
    parse_failures: int = 0,
    failed_calls: int = 0,
    token_usage: Optional[dict] = None,
) -> dict:
    issues = sorted(issues, key=lambda x: float(x.get("start_time", 0) or 0))
    return {
        "overall_assessment": {
            "has_issues": len(issues) > 0,
            "total_issues": len(issues),
            "severity_breakdown": {
                "critical": sum(1 for i in issues if i.get("severity") == "critical"),
                "major": sum(1 for i in issues if i.get("severity") == "major"),
                "minor": sum(1 for i in issues if i.get("severity") == "minor"),
            },
        },
        "issues": issues,
        "api_calls": api_calls,
        "parse_failures": parse_failures,
        "failed_calls": failed_calls,
        "token_usage": _merge_token_usage(token_usage),
    }


def merge_multiple_runs(
    run_results: list[dict],
    num_runs: int,
    min_detection_rate: float = 0.5,
    time_window_sec: float = 30,
) -> dict:
    issues_by_key = {}
    for result in run_results:
        for issue in result.get("issues", []):
            # ② claim_id 기반 매칭: utterance_id + claim_text 우선, fallback으로 기존 fuzzy key
            claim_text = _compact_text(str(issue.get("claim_text", "") or ""))
            uid = issue.get("utterance_id", "")
            if uid and claim_text:
                key = (uid, claim_text[:80])
            else:
                start = float(issue.get("start_time", 0) or 0)
                bucket = round(start / time_window_sec) * time_window_sec
                key = (
                    issue.get("type", ""),
                    int(issue.get("slide_number", 0) or 0),
                    bucket,
                    _compact_text(str(issue.get("problematic_content", "") or ""))[:60],
                )
            if key not in issues_by_key:
                issues_by_key[key] = {
                    "issue": issue,
                    "count": 1,
                    "conf_sum": float(issue.get("confidence", 0) or 0),
                }
            else:
                issues_by_key[key]["count"] += 1
                issues_by_key[key]["conf_sum"] += float(issue.get("confidence", 0) or 0)
                if float(issue.get("confidence", 0) or 0) > float(issues_by_key[key]["issue"].get("confidence", 0) or 0):
                    issues_by_key[key]["issue"] = issue

    all_issues, filtered = [], []
    for data in issues_by_key.values():
        issue = data["issue"].copy()
        issue["detection_count"] = data["count"]
        issue["detection_rate"] = data["count"] / num_runs
        issue["avg_confidence"] = data["conf_sum"] / data["count"]
        all_issues.append(issue)
        if issue["detection_rate"] >= min_detection_rate or (data["count"] >= 2 and issue["avg_confidence"] >= 0.9):
            filtered.append(issue)

    filtered.sort(key=lambda x: float(x.get("start_time", 0) or 0))
    dropped = len(all_issues) - len(filtered)
    return {
        "overall_assessment": {
            "has_issues": len(filtered) > 0,
            "total_issues": len(filtered),
            "severity_breakdown": {
                s: sum(1 for i in filtered if i.get("severity") == s)
                for s in ("critical", "major", "minor")
            },
        },
        "issues": filtered,
        "summary": f"{num_runs}회 판정, 합의 기준 {min_detection_rate:.0%} (시간 창 {time_window_sec:.0f}초). 총 {len(all_issues)}개 후보 중 {len(filtered)}개 확정 ({dropped}개 제외).",
        "verification_method": "claim_consensus_v8",
        "consensus_threshold": min_detection_rate,
        "api_calls": sum(r.get("api_calls", 0) for r in run_results),
        "parse_failures": sum(r.get("parse_failures", 0) for r in run_results),
        "failed_calls": sum(r.get("failed_calls", 0) for r in run_results),
        "token_usage": _merge_token_usage(*(r.get("token_usage") for r in run_results)),
    }


def _strip_json_fence(text: str) -> str:
    if "```json" in text:
        return text.split("```json")[1].split("```")[0].strip()
    if "```" in text:
        return text.split("```")[1].split("```")[0].strip()
    return text.strip()


def _extract_first_json_object(text: str) -> str:
    """응답 문자열에서 첫 JSON object 블록을 안전하게 추출."""
    s = text or ""
    start = s.find("{")
    if start < 0:
        return ""

    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(s)):
        ch = s[i]
        if escaped:
            escaped = False
            continue
        if ch == "\\" and in_string:
            escaped = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[start:i + 1]
    return ""


def _extract_json_like_string_field(text: str, key: str) -> str:
    """불완전 JSON에서도 key의 string 값을 최대한 복구."""
    m = re.search(rf'"{re.escape(key)}"\s*:\s*', text)
    if not m:
        return ""
    i = m.end()
    while i < len(text) and text[i].isspace():
        i += 1
    if i >= len(text):
        return ""

    if text[i] == '"':
        i += 1
        out = []
        escaped = False
        while i < len(text):
            ch = text[i]
            if escaped:
                out.append(ch)
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                break
            else:
                out.append(ch)
            i += 1
        return "".join(out).strip()

    j = i
    while j < len(text) and text[j] not in ",}\n\r":
        j += 1
    return text[i:j].strip()


def _extract_json_like_bool_field(text: str, keys: list[str]) -> Optional[bool]:
    for key in keys:
        m = re.search(rf'"{re.escape(key)}"\s*:\s*(true|false)', text, flags=re.IGNORECASE)
        if m:
            return m.group(1).lower() == "true"
    return None


def _extract_urls(text: str) -> list[str]:
    urls = re.findall(r'https?://[^\s"\'<>]+', text or "")
    return list(dict.fromkeys(urls))


def _parse_grounding_payload(text: str) -> dict:
    """
    grounding 응답을 최대한 복구해서 파싱.
    1) strict JSON
    2) object 블록 추출 + trailing comma 정리
    3) 최소 필드(is_valid/reason/evidence_sources) regex 복구
    """
    cleaned = _strip_json_fence((text or "").strip())
    candidates = [cleaned]

    obj = _extract_first_json_object(cleaned)
    if obj and obj not in candidates:
        candidates.append(obj)

    for candidate in candidates:
        if not candidate:
            continue
        try:
            payload = json.loads(candidate)
            if isinstance(payload, dict):
                return payload
        except json.JSONDecodeError:
            fixed = re.sub(r",\s*([}\]])", r"\1", candidate)
            try:
                payload = json.loads(fixed)
                if isinstance(payload, dict):
                    return payload
            except json.JSONDecodeError:
                pass

    recovered = {}
    is_valid = _extract_json_like_bool_field(
        cleaned,
        ["is_valid", "issue_is_valid", "claim_is_true", "claim_is_valid"],
    )
    if is_valid is not None:
        recovered["is_valid"] = is_valid

    reason = _extract_json_like_string_field(cleaned, "reason")
    if reason:
        recovered["reason"] = reason

    sources = _extract_urls(cleaned)
    if sources:
        recovered["evidence_sources"] = sources

    if recovered:
        return recovered
    raise ValueError("grounding_response_parse_failed")


# ── 발화 수집 + 슬라이드 맥락 ────────────────────────────

def _collect_utterances(slides: list[dict]) -> list[dict]:
    utterances = []
    has_contexts = any(slide.get("contexts") for slide in slides)
    if has_contexts:
        for slide in slides:
            slide_no = int(slide.get("slide_number", 0) or 0)
            for ctx in slide.get("contexts", []) or []:
                text = str(ctx.get("text", "") or "").strip()
                if not text:
                    continue
                context_id = str(ctx.get("context_id", "") or "").strip()
                if not context_id:
                    context_id = f"S{slide_no:03d}-C{len(utterances) + 1:04d}"
                utterances.append({
                    "context_id": context_id,
                    "utterance_id": context_id,
                    "slide_number": slide_no,
                    "scene_index": ctx.get("scene_index"),
                    "context_index": ctx.get("context_index"),
                    "start_time": float(ctx.get("start_time", ctx.get("start", 0)) or 0),
                    "end_time": float(ctx.get("end_time", ctx.get("end", ctx.get("start", 0))) or 0),
                    "text": text,
                    "text_corrected": text,
                    "text_original": "",
                    "text_corrected_candidate": "",
                    "correction_status": "context",
                    "correction_risk": "",
                    "correction_reason": "",
                    "source_segment_indices": ctx.get("source_segment_indices", []),
                })
        utterances.sort(key=lambda u: u["start_time"])
        return utterances

    for slide in slides:
        slide_no = int(slide.get("slide_number", 0) or 0)
        for seg in slide.get("transcript_segments", []) or []:
            corr = str(seg.get("text", "") or "").strip()
            orig = str(seg.get("text_original", "") or "").strip()
            candidate = str(seg.get("text_corrected_candidate", "") or "").strip()
            text = corr or orig
            if not text:
                continue
            utterances.append({
                "slide_number": slide_no,
                "start_time": float(seg.get("start", 0) or 0),
                "text": text,
                "text_corrected": corr,
                "text_original": orig,
                "text_corrected_candidate": candidate,
                "correction_status": str(seg.get("correction_status", "") or "").strip(),
                "correction_risk": str(seg.get("correction_risk", "") or "").strip(),
                "correction_reason": str(seg.get("correction_reason", "") or "").strip(),
            })
    utterances.sort(key=lambda u: u["start_time"])
    for i, u in enumerate(utterances, start=1):
        u["utterance_id"] = f"U{i:04d}"
    return utterances


def _build_slide_context_map(slides: list[dict]) -> dict:
    """slides 데이터에서 slide_number → {title, time_range, slide_text} 매핑."""
    ctx = {}
    for slide in slides:
        sn = int(slide.get("slide_number", 0) or 0)
        if sn <= 0:
            continue
        ctx[sn] = {
            "title": str(slide.get("title", "") or ""),
            "time_range": str(slide.get("time_range", "") or ""),
            "slide_text": str(slide.get("slide_text", "") or "").strip(),
        }
    return ctx


def _format_utterance_for_prompt(u: dict) -> str:
    uid = u["utterance_id"]
    ts = f"{u['start_time']:.1f}s"
    corr = str(u.get("text_corrected", "") or "").strip()
    orig = str(u.get("text_original", "") or "").strip()

    if str(u.get("correction_status", "") or "").strip() == "candidate_only":
        return f"{uid} | {ts} | {orig or u.get('text', '')}"

    if corr and orig and corr != orig:
        return f"{uid} | {ts} | 교정: {corr} | 원문: {orig}"

    return f"{uid} | {ts} | {u['text']}"
