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
    OPENAI_SDK_IMPORT_ERROR,
    get_anthropic_client,
    get_gemini_client_sequence,
    get_openai_client,
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
    slide_recheck_model = os.getenv("VERIFIER_SLIDE_RECHECK_MODEL", VERIFIER_SLIDE_RECHECK_MODEL).strip()
    grounding_model = os.getenv("VERIFIER_GROUNDING_MODEL", VERIFIER_GROUNDING_MODEL).strip()
    strong = judge_model or _default_judge_model(base)

    if stage == "extract":
        return extract_model or base
    if stage == "judge":
        return strong
    if stage == "cross_recheck":
        return cross_recheck_model or strong
    if stage == "recheck":
        return slide_recheck_model or strong
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


TOKEN_USAGE_STAGES = ("extract", "judge", "recheck", "grounding", "cross_recheck")
TOKEN_USAGE_FIELDS = (
    "input_tokens",
    "output_tokens",
    "reasoning_tokens",
    "tool_input_tokens",
    "cached_input_tokens",
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


def _merge_token_usage(*usages: dict) -> dict:
    merged = _empty_token_usage()
    for usage in usages:
        if not isinstance(usage, dict):
            continue
        for stage in TOKEN_USAGE_STAGES:
            bucket = usage.get(stage)
            if not isinstance(bucket, dict):
                continue
            for field in TOKEN_USAGE_FIELDS:
                merged[stage][field] += _safe_int(bucket.get(field))

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
        "input_tokens": _safe_int(getattr(usage, "prompt_tokens", None)),
        "output_tokens": _safe_int(getattr(usage, "completion_tokens", None)),
        "reasoning_tokens": _safe_int(getattr(completion_details, "reasoning_tokens", None)),
        "tool_input_tokens": 0,
        "cached_input_tokens": _safe_int(getattr(prompt_details, "cached_tokens", None)),
        "total_tokens": _safe_int(getattr(usage, "total_tokens", None)),
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
    input_tokens = _safe_int(getattr(usage, "input_tokens", None))
    output_tokens = _safe_int(getattr(usage, "output_tokens", None))
    call_bucket = _new_token_bucket()
    call_bucket["input_tokens"] = input_tokens
    call_bucket["output_tokens"] = output_tokens
    call_bucket["total_tokens"] = input_tokens + output_tokens
    merged = _empty_token_usage()
    merged[stage] = call_bucket
    merged["total"] = dict(call_bucket)
    return merged

def _call_llm(
    prompt: str,
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
            if OPENAI_SDK_IMPORT_ERROR is not None:
                raise RuntimeError(
                    "openai 패키지가 설치되지 않았습니다. "
                    "venv/bin/python -m pip install -r app/backend/pipeline/requirements.txt "
                    "또는 venv/bin/python -m pip install 'openai>=1.0.0' 실행 후 다시 시도하세요."
                )
            if not os.getenv("OPENAI_API_KEY"):
                raise RuntimeError("OPENAI_API_KEY가 설정되지 않았습니다.")
            raise RuntimeError("OpenAI client 초기화에 실패했습니다.")

        image_payloads = []
        if image_bytes_list:
            image_payloads.extend([b for b in image_bytes_list if b])
        elif image_bytes:
            image_payloads.append(image_bytes)

        if image_payloads:
            content = []
            for img in image_payloads:
                b64 = base64.b64encode(img).decode()
                content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
            content.append({"type": "text", "text": prompt})
            messages = [{"role": "user", "content": content}]
        else:
            messages = [{"role": "user", "content": prompt}]

        def call_api():
            kwargs = dict(
                model=model,
                messages=messages,
                max_completion_tokens=max_tokens,
            )
            if _should_send_openai_temperature(model, reasoning_effort):
                kwargs["temperature"] = temp
            if reasoning_effort:
                kwargs["reasoning_effort"] = reasoning_effort
            if response_format is not None:
                kwargs["response_format"] = response_format
            return client.chat.completions.create(**kwargs)

        resp = api_call_with_retry(call_api)
        return resp.choices[0].message.content or "", _extract_openai_usage(resp, model, stage)

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
    contents.append(types.Part.from_text(text=prompt))

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

BATCH_SIZE = int(os.getenv("VERIFIER_BATCH_SIZE", "20"))
VERIFIER_MODEL = os.getenv("VERIFIER_MODEL", "gemini-2.5-flash")
VERIFIER_CLAIM_EXTRACT_MODEL = os.getenv("VERIFIER_CLAIM_EXTRACT_MODEL", "")
VERIFIER_CLAIM_JUDGE_MODEL = os.getenv("VERIFIER_CLAIM_JUDGE_MODEL", "")
VERIFIER_CROSS_RECHECK_MODEL = os.getenv("VERIFIER_CROSS_RECHECK_MODEL", "")
VERIFIER_SLIDE_RECHECK_MODEL = os.getenv("VERIFIER_SLIDE_RECHECK_MODEL", "")
VERIFIER_GROUNDING_MODEL = os.getenv("VERIFIER_GROUNDING_MODEL", "")
VERIFIER_TEMPERATURE = float(os.getenv("VERIFIER_TEMPERATURE", "0.0"))
ALLOWED_ISSUE_TYPES = {"factual_error", "outdated"}
VERIFIER_PARSE_RETRIES = int(os.getenv("VERIFIER_PARSE_RETRIES", "2"))
VERIFIER_BATCH_RECOVERY_RETRIES = int(os.getenv("VERIFIER_BATCH_RECOVERY_RETRIES", "1"))
VERIFIER_REQUIRE_COMPLETE = os.getenv("VERIFIER_REQUIRE_COMPLETE", "1") != "0"


# ── 도메인 힌트 ──────────────────────────────────────────

DOMAIN_HINTS = {
    ("공학", "CS"): {
        "label": "공학 > 컴퓨터공학",
        "concept_examples": "운영체제, 자료구조, 알고리즘, 네트워크 프로토콜, 프로세스, 메모리 관리",
        "outdated_guidance": (
            "아키텍처/모델 설명에 사용되는 고유명사(32비트, x86, 16비트 세그먼트 등)는 "
            "개념 설명용이므로 outdated로 잡지 마세요. "
            "단, 특정 소프트웨어/OS 버전을 설치·사용하라고 권장하는 맥락이면 검출 대상입니다."
        ),
    },
    ("공학", "전자공학"): {
        "label": "공학 > 전자공학",
        "concept_examples": "회로, 신호처리, 반도체, 전력, 제어 시스템",
        "outdated_guidance": "구형 부품/규격을 개념 설명용으로 언급하는 것은 outdated가 아닙니다.",
    },
    ("자연과학", "물리학"): {
        "label": "자연과학 > 물리학",
        "concept_examples": "뉴턴 역학, 열역학, 전자기학, 양자역학, 상대성이론",
        "outdated_guidance": "고전 물리 법칙을 교육적으로 설명하는 것은 outdated가 아닙니다.",
    },
    ("자연과학", "화학"): {
        "label": "자연과학 > 화학",
        "concept_examples": "원자 구조, 화학 결합, 반응 속도론, 유기화학, 열화학",
        "outdated_guidance": "고전 모델(보어 모델 등)을 개념 도입용으로 설명하는 것은 outdated가 아닙니다.",
    },
    ("자연과학", "생물학"): {
        "label": "자연과학 > 생물학",
        "concept_examples": "세포, 유전, 진화, 생태계, 분자생물학",
        "outdated_guidance": "과거 실험/발견을 역사적으로 소개하는 것은 outdated가 아닙니다.",
    },
    ("인문학", "역사학"): {
        "label": "인문학 > 역사학",
        "concept_examples": "사건, 인물, 시대, 사료, 역사 해석",
        "outdated_guidance": "역사적 사실의 서술은 outdated 대상이 아닙니다. 현행 학술 합의가 바뀐 해석만 해당됩니다.",
    },
    ("인문학", "철학"): {
        "label": "인문학 > 철학",
        "concept_examples": "존재론, 인식론, 윤리학, 논리학, 사상가",
        "outdated_guidance": "고전 철학 이론의 소개는 outdated가 아닙니다.",
    },
    ("사회과학", "경제학"): {
        "label": "사회과학 > 경제학",
        "concept_examples": "수요·공급, GDP, 인플레이션, 통화정책, 시장 구조",
        "outdated_guidance": "경제 모델의 교육적 설명은 outdated가 아닙니다. 변경된 법률·세율·기준금리 등은 검출 대상입니다.",
    },
    ("사회과학", "정치학"): {
        "label": "사회과학 > 정치학",
        "concept_examples": "정치체제, 선거, 정당, 국제관계, 정책",
        "outdated_guidance": "정치 이론의 설명은 outdated가 아닙니다. 변경된 법률·제도·현직 인물 정보는 검출 대상입니다.",
    },
    ("예술", "음악"): {
        "label": "예술 > 음악",
        "concept_examples": "음악 이론, 작곡, 악기, 장르, 음악사",
        "outdated_guidance": "과거 음악 양식/작곡 기법의 설명은 outdated가 아닙니다.",
    },
}

DEFAULT_HINT = {
    "label": "일반",
    "concept_examples": "",
    "outdated_guidance": "개념 설명용 언급과 실제 사용 권장을 구분하세요.",
}

SUBDOMAIN_ALIASES = {
    "cs": "CS",
    "computer_science": "CS",
    "컴퓨터공학": "CS",
    "컴공": "CS",
    "전자공학": "전자공학",
    "electrical_engineering": "전자공학",
    "물리학": "물리학",
    "physics": "물리학",
    "화학": "화학",
    "chemistry": "화학",
    "생물학": "생물학",
    "biology": "생물학",
    "역사학": "역사학",
    "history": "역사학",
    "철학": "철학",
    "philosophy": "철학",
    "경제학": "경제학",
    "economics": "경제학",
    "정치학": "정치학",
    "politics": "정치학",
    "political_science": "정치학",
    "음악": "음악",
    "music": "음악",
}


def _normalize_sub_domain(value: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    return SUBDOMAIN_ALIASES.get(raw, SUBDOMAIN_ALIASES.get(raw.lower(), raw))


def _get_domain_hint(domain: str, sub_domain: str) -> dict:
    key = (domain.strip(), _normalize_sub_domain(sub_domain))
    return DOMAIN_HINTS.get(key, DEFAULT_HINT)


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
    3) 최소 필드(status/is_valid/reason/evidence_sources) regex 복구
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
    status = _extract_json_like_string_field(cleaned, "status").lower().strip()
    if status:
        recovered["status"] = status

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
