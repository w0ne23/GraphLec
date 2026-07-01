"""
임베딩 헬퍼 (Gemini/OpenAI provider 지원, 문서/질의용 task_type 분리, 429 재시도).
"""

from __future__ import annotations

import os
import time
from typing import Any, List, Optional, Sequence, Union

from dotenv import load_dotenv

load_dotenv()

RETRY_DELAYS_SEC = [0, 5, 15, 30]

DEFAULT_EMBEDDING_PROVIDER = os.getenv("GRAPHLEC_EMBEDDING_PROVIDER", "openai").strip().lower()
DEFAULT_GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
DEFAULT_OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")

# Backward-compatible name used by existing call sites.
DEFAULT_EMBEDDING_MODEL = (
    DEFAULT_OPENAI_EMBEDDING_MODEL
    if DEFAULT_EMBEDDING_PROVIDER in {"openai", "gpt"}
    else DEFAULT_GEMINI_EMBEDDING_MODEL
)


def normalize_embedding_provider(provider: str | None = None) -> str:
    value = (provider or DEFAULT_EMBEDDING_PROVIDER or "gemini").strip().lower()
    if value in {"google", "gemini"}:
        return "gemini"
    if value in {"openai", "gpt"}:
        return "openai"
    raise ValueError(f"지원하지 않는 임베딩 provider입니다: {provider}")


def default_embedding_model(provider: str | None = None) -> str:
    provider = normalize_embedding_provider(provider)
    if provider == "openai":
        return DEFAULT_OPENAI_EMBEDDING_MODEL
    return DEFAULT_GEMINI_EMBEDDING_MODEL


def gemini_api_key() -> str:
    key = os.getenv("GOOGLE_API_KEY_2") or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GOOGLE_API_KEY (또는 GOOGLE_API_KEY_2) 환경 변수가 필요합니다.")
    return key


def get_genai_client() -> Any:
    from google import genai

    return genai.Client(api_key=gemini_api_key())


def openai_api_key() -> str:
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("OPENAI_API_KEY 환경 변수가 필요합니다.")
    return key


def get_openai_client() -> Any:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("openai 패키지가 설치되어 있지 않습니다.") from exc
    return OpenAI(api_key=openai_api_key())


def get_embedding_client(provider: str | None = None) -> Any:
    provider = normalize_embedding_provider(provider)
    if provider == "openai":
        return get_openai_client()
    return get_genai_client()


def _embed_with_retry(
    client: Any,
    *,
    model: str,
    contents: Union[Sequence[str], str],
    task_type: str,
) -> List[List[float]]:
    last_err: Optional[Exception] = None
    for delay in RETRY_DELAYS_SEC:
        if delay:
            time.sleep(delay)
        try:
            from google.genai import types

            r = client.models.embed_content(
                model=model,
                contents=contents,
                config=types.EmbedContentConfig(task_type=task_type),
            )
            try:
                from .cost_report import record_model_call

                item_count = len(contents) if isinstance(contents, Sequence) and not isinstance(contents, str) else 1
                prompt_chars = (
                    sum(len(str(x)) for x in contents)
                    if isinstance(contents, Sequence) and not isinstance(contents, str)
                    else len(str(contents))
                )
                record_model_call(
                    stage="stage7_lance_embedding" if task_type == "RETRIEVAL_DOCUMENT" else "embedding_query",
                    provider="google",
                    model=model,
                    response=r,
                    item_count=item_count,
                    prompt_chars=prompt_chars,
                )
            except Exception:
                pass
            return [list(e.values) for e in r.embeddings]
        except Exception as e:
            last_err = e
            err_s = str(e)
            if "429" not in err_s and "RESOURCE_EXHAUSTED" not in err_s:
                raise
    assert last_err is not None
    raise last_err


def _embed_openai_with_retry(
    client: Any,
    *,
    model: str,
    contents: Union[Sequence[str], str],
    stage: str,
    dimensions: Optional[int] = None,
) -> List[List[float]]:
    input_items = list(contents) if isinstance(contents, Sequence) and not isinstance(contents, str) else [str(contents)]
    last_err: Optional[Exception] = None
    for delay in RETRY_DELAYS_SEC:
        if delay:
            time.sleep(delay)
        try:
            kwargs: dict[str, Any] = {
                "model": model,
                "input": input_items,
                "encoding_format": "float",
            }
            if dimensions:
                kwargs["dimensions"] = int(dimensions)
            r = client.embeddings.create(**kwargs)
            try:
                from .cost_report import record_model_call

                prompt_chars = sum(len(str(x)) for x in input_items)
                record_model_call(
                    stage=stage,
                    provider="openai",
                    model=model,
                    response=r,
                    item_count=len(input_items),
                    prompt_chars=prompt_chars,
                    metadata={"dimensions": dimensions} if dimensions else None,
                )
            except Exception:
                pass
            data = sorted(r.data, key=lambda item: getattr(item, "index", 0))
            return [list(item.embedding) for item in data]
        except Exception as e:
            last_err = e
            err_s = str(e)
            if "429" not in err_s and "rate limit" not in err_s.lower() and "RESOURCE_EXHAUSTED" not in err_s:
                raise
    assert last_err is not None
    raise last_err


def embed_documents(
    client: Any,
    texts: Sequence[str],
    *,
    model: str = DEFAULT_EMBEDDING_MODEL,
    provider: str = DEFAULT_EMBEDDING_PROVIDER,
    dimensions: Optional[int] = None,
) -> List[List[float]]:
    if not texts:
        return []
    provider = normalize_embedding_provider(provider)
    if provider == "openai":
        return _embed_openai_with_retry(
            client,
            model=model,
            contents=list(texts),
            stage="stage7_lance_embedding",
            dimensions=dimensions,
        )
    return _embed_with_retry(
        client, model=model, contents=list(texts), task_type="RETRIEVAL_DOCUMENT"
    )


def embed_query(
    client: Any,
    text: str,
    *,
    model: str = DEFAULT_EMBEDDING_MODEL,
    provider: str = DEFAULT_EMBEDDING_PROVIDER,
    dimensions: Optional[int] = None,
) -> List[float]:
    provider = normalize_embedding_provider(provider)
    if provider == "openai":
        vecs = _embed_openai_with_retry(
            client,
            model=model,
            contents=text,
            stage="embedding_query",
            dimensions=dimensions,
        )
        return vecs[0]
    vecs = _embed_with_retry(
        client, model=model, contents=text, task_type="RETRIEVAL_QUERY"
    )
    return vecs[0]
