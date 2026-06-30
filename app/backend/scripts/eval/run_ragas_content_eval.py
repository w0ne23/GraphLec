#!/usr/bin/env python3
"""Evaluate content QnA results with RAGAS."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import re
import sys
import types
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def first_metric_value(record: dict[str, Any], *names: str, prefixes: str | tuple[str, ...] = ()) -> Any:
    for name in names:
        if name in record:
            return record.get(name)
    if isinstance(prefixes, str):
        prefixes = (prefixes,)
    for key, value in record.items():
        if any(key.startswith(prefix) for prefix in prefixes):
            return value
    return None


def result_records(result: Any) -> list[dict[str, Any]]:
    try:
        df = result.to_pandas()
        return df.to_dict(orient="records")
    except Exception:
        return []


def is_example_question(question: str) -> bool:
    compact = re.sub(r"\s+", "", str(question or ""))
    return any(token in compact for token in ("예시", "사례", "예를들", "예는", "예가", "예로"))


def trim_to_max_chars(text: str, max_chars: int) -> str:
    text = str(text or "").strip()
    if len(text) <= max_chars:
        return text
    sentences = re.split(r"(?<=[.!?。]|다\.)\s+", text)
    compact = ""
    for sentence in sentences:
        candidate = (compact + " " + sentence).strip()
        if len(candidate) > max_chars:
            break
        compact = candidate
    return (compact or text[:max_chars]).strip()


def clean_focused_response(text: str, max_chars: int) -> str:
    text = str(text or "").strip()
    text = re.sub(r"^```(?:\w+)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = re.sub(r"^\s*(?:답변|평가용 답변)\s*[:：]\s*", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return trim_to_max_chars(text, max_chars)


def example_items_to_sentence(question: str, response: str, max_chars: int) -> str | None:
    if not is_example_question(question):
        return None

    bullet_re = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*(.+?)\s*$")
    items: list[str] = []
    for line in str(response or "").replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        match = bullet_re.match(line)
        if not match:
            continue
        item = match.group(1).strip()
        item = re.sub(r"^\*\*(.+?)\*\*", r"\1", item)
        item = re.split(r"\s*[:：]\s*", item, maxsplit=1)[0].strip()
        particle_match = re.match(r"(.+?)(?:은|는)\s+", item)
        if particle_match:
            item = particle_match.group(1).strip()
        item = re.sub(r"[.。]\s*$", "", item).strip()
        if item and item not in items:
            items.append(item)

    if len(items) < 2:
        return None

    subject_match = re.search(r"(.+?)의\s*예시", str(question or "").strip())
    subject = subject_match.group(1).strip() if subject_match else ""
    joined = ", ".join(items)
    if subject:
        sentence = f"{subject}의 예시로는 {joined} 등이 있습니다."
    else:
        sentence = f"예시로는 {joined} 등이 있습니다."
    return trim_to_max_chars(sentence, max_chars)


def response_for_relevancy(question: str, response: str, mode: str, max_chars: int) -> str:
    """Return the part of the answer used only for answer relevancy scoring."""
    text = str(response or "").strip()
    if mode == "full" or not text:
        return text
    if mode == "focused":
        raise ValueError("focused mode must be handled by build_relevancy_responses")
    if is_example_question(question):
        return trim_to_max_chars(text, max_chars)

    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    lead_lines: list[str] = []
    bullet_re = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if lead_lines:
                break
            continue
        if bullet_re.match(stripped):
            break
        lead_lines.append(stripped)

    lead = " ".join(lead_lines).strip()
    if not lead:
        lead = text.split("\n", 1)[0].strip()
    if not lead:
        return text[:max_chars].strip()
    return trim_to_max_chars(lead, max_chars)


def focus_response_with_llm(
    *,
    client: Any,
    model: str,
    question: str,
    response: str,
    max_chars: int,
) -> str:
    example_sentence = example_items_to_sentence(question, response, max_chars)
    if example_sentence:
        return example_sentence

    system = (
        "너는 평가용 답변 정리기다. 사용자의 질문과 이미 생성된 답변만 보고, "
        "answer_relevancy 평가에 사용할 짧은 한국어 답변을 만든다.\n"
        "규칙:\n"
        "1. 새 사실, 일반화, 평가, 부가 설명을 추가하지 말고, 반드시 생성된 답변 안에 있는 정보만 사용한다.\n"
        "2. 질문에 직접 답하는 내용만 남긴다.\n"
        "3. 정의를 물으면 정의만 남기고 목적, 기능, 예시는 빼라.\n"
        "4. 예시/사례를 물으면 단순 목록으로 쓰지 말고, 예시 항목들을 포함한 완전한 문장으로 답하라. "
        "원답변이 목록만 제공하면 목록 항목만 한 문장으로 합치고 설명을 덧붙이지 않는다.\n"
        "5. 이유를 물으면 이유만, 차이를 물으면 차이만, 역할/기능을 물으면 해당 역할/기능만 남겨라.\n"
        "6. 출력은 반드시 1~2개의 완전한 문장이어야 한다. 불릿, 번호 목록, 표, 줄바꿈 목록은 쓰지 않는다.\n"
        "7. 내부 평가용 텍스트만 출력하고 설명, 주석, 출처 문구는 쓰지 않는다."
    )
    user = (
        f"질문:\n{question.strip()}\n\n"
        f"생성된 답변:\n{str(response or '').strip()}\n\n"
        f"위 생성된 답변만 사용해서 질문에 직접 답하는 평가용 답변을 "
        f"1~2개의 완전한 문장, {max_chars}자 이내로 작성해."
    )
    kwargs = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_completion_tokens": 512,
    }
    try:
        result = client.chat.completions.create(**kwargs)
    except Exception:
        kwargs.pop("max_completion_tokens", None)
        kwargs["max_tokens"] = 512
        result = client.chat.completions.create(**kwargs)
    text = clean_focused_response(result.choices[0].message.content or "", max_chars)
    return text or response_for_relevancy(
        question,
        response,
        "lead",
        max_chars,
    )


def build_relevancy_responses(
    rows: list[dict[str, Any]],
    mode: str,
    max_chars: int,
    focus_model: str,
) -> list[str]:
    if mode != "focused":
        return [
            response_for_relevancy(row["question"], row["response"], mode, max_chars)
            for row in rows
        ]

    try:
        from openai import OpenAI
    except Exception as exc:
        raise SystemExit(
            "focused answer_relevancy mode requires the openai package. "
            f"Original error: {type(exc).__name__}: {exc}"
        ) from exc

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "").strip() or None)
    responses: list[str] = []
    for idx, row in enumerate(rows, start=1):
        print(
            f"[focus {idx}/{len(rows)}] {row.get('id')} {row.get('question')}",
            file=sys.stderr,
        )
        try:
            focused = focus_response_with_llm(
                client=client,
                model=focus_model,
                question=row["question"],
                response=row["response"],
                max_chars=max_chars,
            )
        except Exception as exc:
            print(
                f"[focus fallback] {row.get('id')}: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            focused = response_for_relevancy(
                row["question"],
                row["response"],
                "lead",
                max_chars,
            )
        responses.append(focused)
    return responses


def result_to_rows(
    core_result: Any,
    source_rows: list[dict[str, Any]],
    relevancy_responses: list[str],
    relevancy_debugs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    records = result_records(core_result)
    out: list[dict[str, Any]] = []
    for idx, src in enumerate(source_rows):
        rec = records[idx] if idx < len(records) else {}
        rel_debug = relevancy_debugs[idx] if idx < len(relevancy_debugs) else {}
        out.append(
            {
                "id": src.get("id"),
                "question": src.get("question"),
                "answerable": src.get("answerable"),
                "response": src.get("response", ""),
                "faithfulness": to_float(rec.get("faithfulness")),
                "answer_relevancy": to_float(rel_debug.get("answer_relevancy")),
                "answer_relevancy_response": (
                    relevancy_responses[idx] if idx < len(relevancy_responses) else ""
                ),
                "answer_relevancy_question_language": rel_debug.get(
                    "question_language",
                    "",
                ),
                "answer_relevancy_generated_questions": json.dumps(
                    rel_debug.get("generated_questions", []),
                    ensure_ascii=False,
                ),
                "answer_relevancy_similarities": json.dumps(
                    rel_debug.get("similarities", []),
                    ensure_ascii=False,
                ),
                "answer_relevancy_noncommittal": json.dumps(
                    rel_debug.get("noncommittal", []),
                    ensure_ascii=False,
                ),
                "answer_relevancy_all_noncommittal": rel_debug.get(
                    "all_noncommittal",
                    "",
                ),
                "context_precision": to_float(
                    first_metric_value(
                        rec,
                        "context_precision",
                        "llm_context_precision_with_reference",
                        "llm_context_precision_without_reference",
                        prefixes=("context_precision", "llm_context_precision"),
                    )
                ),
                "context_recall": to_float(rec.get("context_recall")),
                "factual_correctness": to_float(
                    first_metric_value(
                        rec,
                        "factual_correctness",
                        "answer_correctness",
                        prefixes=("factual_correctness", "answer_correctness"),
                    )
                ),
            }
        )
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "id",
        "question",
        "answerable",
        "faithfulness",
        "answer_relevancy",
        "answer_relevancy_response",
        "answer_relevancy_question_language",
        "answer_relevancy_generated_questions",
        "answer_relevancy_similarities",
        "answer_relevancy_noncommittal",
        "answer_relevancy_all_noncommittal",
        "context_precision",
        "context_recall",
        "factual_correctness",
        "response",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def install_ragas_import_shims() -> None:
    """Patch optional legacy imports required by some RAGAS/LangChain combos.

    RAGAS 0.4.x may import `langchain_community.chat_models.vertexai` at module
    import time even when this evaluation does not use VertexAI. Recent
    langchain-community versions removed that legacy module, so provide a tiny
    placeholder to let non-VertexAI metrics load.
    """
    module_name = "langchain_community.chat_models.vertexai"
    if module_name in sys.modules:
        return
    shim = types.ModuleType(module_name)

    class ChatVertexAI:  # pragma: no cover - only used as an import placeholder.
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError(
                "ChatVertexAI is not available in this evaluation environment. "
                "Configure a non-VertexAI evaluator model or install the VertexAI integration."
            )

    shim.ChatVertexAI = ChatVertexAI
    sys.modules[module_name] = shim


def answer_relevance_prompt_string(
    *,
    response: str,
    question_language: str,
    ragas_prompt: Any,
    ragas_input_cls: Any,
) -> str:
    if question_language == "ragas":
        return ragas_prompt.to_string(ragas_input_cls(response=response))

    return (
        "주어진 답변이 답이 될 수 있는 질문 하나를 생성하고, 답변이 회피적인지도 판단하라.\n"
        "규칙:\n"
        "1. question은 반드시 자연스러운 한국어 질문이어야 한다.\n"
        "2. question은 주어진 답변에서 직접 답할 수 있는 핵심 질문이어야 한다.\n"
        "3. 영어 질문을 만들지 않는다.\n"
        "4. 답변이 회피적이거나 모호하거나 알 수 없다는 내용이면 noncommittal=1, "
        "실질적인 답변이면 noncommittal=0으로 판단한다.\n\n"
        f"답변:\n{response}"
    )


async def build_answer_relevancy_debugs(
    *,
    rows: list[dict[str, Any]],
    relevancy_responses: list[str],
    model: str,
    embedding_model: str,
    strictness: int,
    question_language: str,
) -> list[dict[str, Any]]:
    import numpy as np
    from openai import AsyncOpenAI
    from ragas.embeddings.base import embedding_factory
    from ragas.llms import llm_factory
    from ragas.metrics.collections.answer_relevancy.metric import (
        AnswerRelevanceInput,
        AnswerRelevanceOutput,
        AnswerRelevancePrompt,
    )

    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY", "").strip() or None)
    llm = llm_factory(
        model,
        client=client,
        temperature=0.01,
        max_tokens=1024,
    )
    embeddings = embedding_factory(
        "openai",
        model=embedding_model,
        client=client,
    )
    prompt = AnswerRelevancePrompt()

    debug_rows: list[dict[str, Any]] = []
    for idx, (row, relevancy_response) in enumerate(
        zip(rows, relevancy_responses),
        start=1,
    ):
        print(
            f"[answer_relevancy {idx}/{len(rows)}] {row.get('id')} {row.get('question')}",
            file=sys.stderr,
        )
        generated_questions: list[str] = []
        noncommittal_flags: list[int] = []
        prompt_string = answer_relevance_prompt_string(
            response=str(relevancy_response or ""),
            question_language=question_language,
            ragas_prompt=prompt,
            ragas_input_cls=AnswerRelevanceInput,
        )
        for _ in range(strictness):
            result = await llm.agenerate(prompt_string, AnswerRelevanceOutput)
            if result.question:
                generated_questions.append(result.question)
                noncommittal_flags.append(int(result.noncommittal))

        if not generated_questions:
            debug_rows.append(
                {
                    "answer_relevancy": 0.0,
                    "question_language": question_language,
                    "generated_questions": [],
                    "similarities": [],
                    "noncommittal": noncommittal_flags,
                    "all_noncommittal": bool(noncommittal_flags),
                }
            )
            continue

        question_vec = np.asarray(
            await embeddings.aembed_text(str(row["question"]))
        ).reshape(1, -1)
        generated_vec = np.asarray(
            await embeddings.aembed_texts(generated_questions)
        ).reshape(len(generated_questions), -1)
        norm = np.linalg.norm(generated_vec, axis=1) * np.linalg.norm(
            question_vec,
            axis=1,
        )
        similarities = (
            np.dot(generated_vec, question_vec.T).reshape(-1) / norm
        )
        all_noncommittal = bool(np.all(noncommittal_flags))
        score = float(similarities.mean() * int(not all_noncommittal))
        debug_rows.append(
            {
                "answer_relevancy": score,
                "question_language": question_language,
                "generated_questions": generated_questions,
                "similarities": [float(value) for value in similarities],
                "noncommittal": noncommittal_flags,
                "all_noncommittal": all_noncommittal,
            }
        )
    return debug_rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--include-unanswerable",
        action="store_true",
        help="Include no-answer style rows. By default only answerable content rows are evaluated.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N eligible content rows. Useful for smoke tests.",
    )
    parser.add_argument(
        "--answer-relevancy-response-mode",
        choices=["focused", "lead", "full"],
        default="focused",
        help=(
            "Response text used only for answer_relevancy. "
            "'focused' rewrites the existing answer into a question-focused evaluation answer, "
            "'lead' uses the answer lead before bullet/list details, and 'full' uses the full answer. "
            "Other metrics always use the full response."
        ),
    )
    parser.add_argument(
        "--answer-relevancy-max-chars",
        type=int,
        default=600,
        help="Maximum characters used only for answer_relevancy.",
    )
    parser.add_argument(
        "--answer-relevancy-focus-model",
        default=(
            os.getenv("RAGAS_RELEVANCY_FOCUS_MODEL")
            or os.getenv("QUERY_SERVICE_OPENAI_MODEL")
            or "gpt-4o-mini"
        ),
        help="OpenAI model used to create focused answer_relevancy_response in focused mode.",
    )
    parser.add_argument(
        "--ragas-answer-relevancy-model",
        default=os.getenv("RAGAS_ANSWER_RELEVANCY_MODEL", "gpt-4o-mini"),
        help="OpenAI chat model used by RAGAS to score answer_relevancy.",
    )
    parser.add_argument(
        "--answer-relevancy-strictness",
        type=int,
        default=3,
        help="Number of generated questions used to score answer_relevancy.",
    )
    parser.add_argument(
        "--answer-relevancy-question-language",
        choices=["ko", "ragas"],
        default=os.getenv("RAGAS_ANSWER_RELEVANCY_QUESTION_LANGUAGE", "ko"),
        help=(
            "Language/prompt used for generated questions. "
            "'ko' forces Korean generated questions; 'ragas' uses the stock RAGAS prompt."
        ),
    )
    parser.add_argument(
        "--only-answer-relevancy",
        action="store_true",
        help="Evaluate only answer_relevancy and skip slower core RAGAS metrics.",
    )
    parser.add_argument(
        "--ragas-timeout",
        type=int,
        default=int(os.getenv("RAGAS_TIMEOUT", "300")),
        help="Timeout in seconds for each RAGAS evaluation job.",
    )
    parser.add_argument(
        "--ragas-max-retries",
        type=int,
        default=int(os.getenv("RAGAS_MAX_RETRIES", "10")),
        help="Maximum retries for each RAGAS evaluation job.",
    )
    parser.add_argument(
        "--ragas-max-workers",
        type=int,
        default=int(os.getenv("RAGAS_MAX_WORKERS", "4")),
        help="Maximum parallel workers for RAGAS evaluation jobs.",
    )
    parser.add_argument(
        "--ragas-batch-size",
        type=int,
        default=int(os.getenv("RAGAS_BATCH_SIZE", "4")),
        help="Batch size passed to RAGAS evaluate().",
    )
    args = parser.parse_args()

    rows = [row for row in read_jsonl(args.input) if row.get("type") == "content"]
    if not args.include_unanswerable:
        rows = [row for row in rows if row.get("answerable", True)]
    rows = [row for row in rows if row.get("response") and row.get("retrieved_contexts")]
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit("No content rows with response and retrieved_contexts found.")

    samples = [
        {
            "user_input": row["question"],
            "response": row["response"],
            "retrieved_contexts": row.get("retrieved_contexts") or [],
            "reference": row["reference"],
        }
        for row in rows
    ]
    relevancy_responses = build_relevancy_responses(
        rows,
        args.answer_relevancy_response_mode,
        args.answer_relevancy_max_chars,
        args.answer_relevancy_focus_model,
    )
    try:
        install_ragas_import_shims()
        from ragas import EvaluationDataset, evaluate
        import ragas.metrics as ragas_metrics
        from ragas.run_config import RunConfig
        from langchain_openai import OpenAIEmbeddings
    except Exception as exc:
        raise SystemExit(
            "RAGAS is not installed or the installed version is incompatible. "
            f"Install/fix it in the runtime first. Original error: {type(exc).__name__}: {exc}"
        ) from exc

    embedding_model = os.getenv("RAGAS_EMBEDDING_MODEL", "text-embedding-3-small")
    relevancy_debugs = asyncio.run(
        build_answer_relevancy_debugs(
            rows=rows,
            relevancy_responses=relevancy_responses,
            model=args.ragas_answer_relevancy_model,
            embedding_model=embedding_model,
            strictness=args.answer_relevancy_strictness,
            question_language=args.answer_relevancy_question_language,
        )
    )
    if args.only_answer_relevancy:
        write_csv(args.out, result_to_rows(None, rows, relevancy_responses, relevancy_debugs))
        print(f"saved: {args.out}")
        valid_scores = [
            debug["answer_relevancy"]
            for debug in relevancy_debugs
            if isinstance(debug.get("answer_relevancy"), (int, float))
        ]
        if valid_scores:
            print({"answer_relevancy": sum(valid_scores) / len(valid_scores)})
        return 0

    faithfulness_cls = getattr(ragas_metrics, "Faithfulness")
    context_precision_cls = getattr(
        ragas_metrics,
        "LLMContextPrecisionWithReference",
        getattr(
            ragas_metrics,
            "LLMContextPrecisionWithoutReference",
            getattr(ragas_metrics, "ContextPrecision", None),
        ),
    )
    context_recall_cls = getattr(
        ragas_metrics,
        "LLMContextRecall",
        getattr(ragas_metrics, "ContextRecall", None),
    )
    factual_correctness_cls = getattr(
        ragas_metrics,
        "FactualCorrectness",
        getattr(ragas_metrics, "AnswerCorrectness", None),
    )
    missing = [
        name
        for name, cls in [
            ("ContextPrecision", context_precision_cls),
            ("ContextRecall", context_recall_cls),
            ("FactualCorrectness/AnswerCorrectness", factual_correctness_cls),
        ]
        if cls is None
    ]
    if missing:
        raise SystemExit(f"RAGAS metrics not found in this version: {', '.join(missing)}")

    metrics = [
        faithfulness_cls(),
        context_precision_cls(),
        context_recall_cls(),
        factual_correctness_cls(),
    ]
    run_config = RunConfig(
        timeout=args.ragas_timeout,
        max_retries=args.ragas_max_retries,
        max_workers=args.ragas_max_workers,
    )
    embeddings = OpenAIEmbeddings(model=embedding_model)
    dataset = EvaluationDataset.from_list(samples)
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        embeddings=embeddings,
        run_config=run_config,
        batch_size=args.ragas_batch_size,
    )
    write_csv(args.out, result_to_rows(result, rows, relevancy_responses, relevancy_debugs))
    print(f"saved: {args.out}")
    print(result)
    valid_scores = [
        debug["answer_relevancy"]
        for debug in relevancy_debugs
        if isinstance(debug.get("answer_relevancy"), (int, float))
    ]
    if valid_scores:
        print({"answer_relevancy": sum(valid_scores) / len(valid_scores)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
