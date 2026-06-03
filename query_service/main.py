"""
GraphLEC 질의 전용 FastAPI 서비스.

UI(React) 또는 내부 서비스로부터 stem + question 을 전달받아 처리한다.

/internal/query: 질문 유형에 따라
  - 구조(structural): LLM이 Cypher 생성 → Neo4j 조회(읽기 전용 검증, $stem 필수)
  - 내용(content): 의도 추론(LLM JSON) → Neo4j 광범위 조회 → 임베딩 재순위·MMR → Lance 2-pass → Gemini 답변
  그래프 근거가 없으면 답변 거부.

/internal/query_graph_evidence: 키워드 기반 엣지 근거(디버그·경량 조회용, 기존 유지).
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError, ServiceUnavailable
from openai import OpenAI
from pydantic import BaseModel, Field

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from pipeline.lance_ingest import default_lance_root, lance_search  # noqa: E402

from .content_retrieval import (  # noqa: E402
    EvidenceItem,
    build_sectioned_context,
    infer_intents_json,
    run_enhanced_content_pipeline,
)
app = FastAPI(title="GraphLEC Query Service", version="0.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY_2") or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
GEMINI_ANSWER_MODEL = os.getenv("GEMINI_ANSWER_MODEL", "gemini-2.5-flash")
QUERY_LLM_PROVIDER = os.getenv("QUERY_SERVICE_LLM_PROVIDER", "openai").strip().lower()
QUERY_OPENAI_MODEL = os.getenv("QUERY_SERVICE_OPENAI_MODEL", "gpt-5.4").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
TOP_K = int(os.getenv("GRAPHLEC_TOP_K", "8"))
GRAPH_TOP_K = int(os.getenv("GRAPHLEC_GRAPH_TOP_K", "12"))
RETRY_DELAYS = [0, 5, 15, 30]
NEO4J_URI = os.getenv("NEO4J_URI", "").strip()
NEO4J_USER = os.getenv("NEO4J_USER", "").strip()
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")

GRAPH_SCHEMA = """
노드 타입과 주요 프로퍼티:
- Video          : id, stem, title, domain, subdomain
- Slide          : id, stem, slide_number, title, slide_text, t1_structure, visual_asset_text, slide_type, emphasis_total, emphasis_score, emphasis_keywords_text
- VisualAsset    : id, stem, asset_type, description, raw_text, slide_number, scene_number, title, image_path
- Scene          : id, stem, source_slide_id, scene_number, slide_number, start_sec, end_sec, role, emphasis_total
- Context        : id, stem, slide_id, scene_id, context_index, start, end, stressed, text
- Segment        : id, stem, start, end, text, stressed
- AnnotationEmphasis : id, stem, type, target_content, score, confidence, timestamp_sec
- GraphRAGEntity : id, stem, title, type, description, degree, frequency
- GraphRAGTextUnit : id, stem, text, n_tokens
- GraphRAGCommunity : id, stem, title, summary, rank, size

관계 (방향 중요):
- (Video)-[:HAS_SCENE]->(Scene)
- (Video)-[:HAS_SLIDE]->(Slide)
- (Slide)-[:HAS_VISUAL_ASSET]->(VisualAsset)
- (Scene)-[:USES_SLIDE]->(Slide)
- (Scene)-[:HAS_CONTEXT]->(Context)
- (Context)-[:HAS_SEGMENT]->(Segment)
- (Scene)-[:HAS_ANNOTATION]->(AnnotationEmphasis)
- (Segment)-[:REFERS_TO]->(AnnotationEmphasis)
- (GraphRAGEntity)-[:GRAPHRAG_RELATES_TO]->(GraphRAGEntity)
- (GraphRAGEntity)-[:GRAPHRAG_SUPPORTED_BY]->(GraphRAGTextUnit)
- (GraphRAGTextUnit)-[:GRAPHRAG_MENTIONS_SLIDE]->(Slide)
- (GraphRAGTextUnit)-[:GRAPHRAG_MENTIONS_SCENE]->(Scene)
- (GraphRAGEntity)-[:GRAPHRAG_APPEARS_IN]->(Slide)
- (GraphRAGEntity)-[:GRAPHRAG_APPEARS_IN_SCENE]->(Scene)
- (GraphRAGCommunity)-[:GRAPHRAG_HAS_ENTITY]->(GraphRAGEntity)

금지 패턴:
- Concept 라벨은 사용하지 않는다. 개념 레이어는 GraphRAGEntity가 담당한다.
- (Segment)-[:APPEARS_IN]->(...)
- (...)-[:MENTIONS]->(Slide)

모든 노드 패턴에는 반드시 {{stem: $stem}} 를 포함한다. 쿼리 실행 시 stem 파라미터가 전달된다.
"""

CYPHER_SYSTEM_PROMPT = f"""
너는 강의 영상 분석 그래프 DB를 조회하는 Cypher 전문가다.
아래 스키마를 기반으로 사용자 질문에 맞는 Cypher만 생성한다.

{GRAPH_SCHEMA}

규칙:
1. 읽기 전용: MATCH / OPTIONAL MATCH / WITH / WHERE / RETURN / ORDER BY / LIMIT / SKIP 만 사용한다. CREATE, MERGE, DELETE, SET, REMOVE, LOAD CSV, FOREACH, CALL 은 쓰지 않는다.
2. 모든 노드는 반드시 {{stem: $stem}} 조건을 포함한다. 예: MATCH (s:Slide {{stem: $stem}})
3. 파라미터는 $stem 만 외부에서 넣는다. 사용자 입력 문자열을 쿼리 문자열에 직접 이어붙이지 않는다. 검색은 $needle 등 추가 파라미터를 쓸 수 있다.
4. Cypher만 출력한다. 코드 블록(```cypher ... ```) 안에 작성한다.
5. RETURN에 필요한 필드만 명시한다. LIMIT는 30 이하로 둔다.
6. 타임스탬프가 필요하면 Segment.start/end 또는 Scene.start_sec/end_sec을 활용한다. Slide에는 시간 정보가 없습니다.
"""

ANSWER_SYSTEM_PROMPT = """
너는 강의 영상 분석 지식 그래프(및 선택적 보강 텍스트)를 바탕으로 질문에 답하는 어시스턴트다.

[근거 규칙]
- 답변의 사실·관계·용어는 반드시 제공된 "근거" 안에 있을 때만 쓴다.
- 근거에 없으면 추측하지 말고 짧게 "제공된 근거만으로는 답변하기 어렵습니다."라고 한다.
- 근거에 "[질문 의도(모델 추론)]"가 있으면 참고만 하되, 답의 내용은 반드시 그 아래 실제 인용 근거에만 기대어라.
- 의미 검색(보조)로 표시된 텍스트은 그래프 근거와 모순 없을 때만 보강에 사용한다.

[복합 질문]
- 질문이 정의와 예시·사례 등을 동시에 요구하면, 근거에서 가능한 범위로 각 요구를 모두 다룬다.
- 특정 요구에 해당하는 근거가 없으면 그 한 가지만 정중하게 짧게 밝힌다.

[답변 스타일]
1. 질문에 직접 필요한 내용만 답한다. 근거가 많아도 전부 나열하지 말고 핵심만 추린다.
2. 기본 답변은 1문장 요약 + 최대 4개 항목으로 작성한다. 각 항목은 한 문장으로 짧게 쓴다.
3. 사용자가 "자세히", "구체적으로", "전부", "비교표"처럼 확장을 요청한 경우에만 더 길게 답한다.
4. 초 단위 시간·구간·슬라이드 번호는 UI의 출처 버튼으로 따로 제공된다. 답변 본문에는 시간·구간·슬라이드 확인 안내를 쓰지 않는다.
5. "어디", "장면", "구간"이라는 단어만으로는 시간값을 본문에 쓰지 않는다. 사용자가 "정확히 몇 초", "전체 구간을 모두", "시작/끝 시간을 표로"처럼 명시적으로 시간 목록을 요구한 경우에만 시간 범위를 본문에 나열한다.
6. "장면 4:", "슬라이드 8:"처럼 출처 위치를 항목 제목으로 쓰지 않는다. 위치는 UI 버튼에서만 제공된다.
7. "(GraphRAG 개념)", "(GraphRAG 개념 관계)" 같은 내부 근거 종류명은 사용자에게 노출하지 않는다.
8. 질문에 사례·예시·예를 들어 등이 있으면 근거에서 1개 사례만 넣고, 없으면 "제공된 근거만으로는 사례를 확인하기 어렵습니다."라고 말한다.
9. "정의:", "음성 발췌:" 같은 인위적 소제목 블록을 만들지 않는다.
10. 한국어로 답한다.
"""


class InternalQueryRequest(BaseModel):
    stem: str = Field(..., min_length=1)
    question: str = Field(..., min_length=1)
    current_scene_number: Optional[int] = None
    current_slide_number: Optional[int] = None
    conversation_history: list[dict[str, str]] = Field(default_factory=list)


class InternalGraphQueryRequest(BaseModel):
    stem: str = Field(..., min_length=1)
    question: str = Field(..., min_length=1)
    top_k: int = Field(default=GRAPH_TOP_K, ge=1, le=100)


class RetrievedChunk(BaseModel):
    chunk_id: str = ""
    stem: str = ""
    chunk_type: str = ""
    text: str = ""
    score: Optional[float] = None
    slide_number: Optional[int] = None
    start_sec: Optional[float] = None
    end_sec: Optional[float] = None
    linked_node_id: Optional[str] = None
    score_breakdown: Optional[dict[str, float]] = None


class QueryResponse(BaseModel):
    answer: str
    timestamps: list[dict]
    graph: dict
    core_graph: dict = Field(default_factory=lambda: {"nodes": [], "edges": []})
    retrieved_chunks: list[RetrievedChunk]
    related_slides: list[dict] = Field(default_factory=list)
    source_mode: str = "default"


class GraphEvidence(BaseModel):
    src_id: str
    src_labels: list[str]
    rel_type: str
    tgt_id: str
    tgt_labels: list[str]
    src_text: str = ""
    tgt_text: str = ""


class GraphEvidenceResponse(BaseModel):
    stem: str
    question: str
    keywords: list[str]
    evidence: list[GraphEvidence]
    count: int


def _to_float_or_none(v: Any) -> Optional[float]:
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _to_int_or_none(v: Any) -> Optional[int]:
    try:
        return int(v) if v is not None else None
    except (TypeError, ValueError):
        return None


STRUCTURAL_KEYWORDS = [
    "몇 번",
    "슬라이드",
    "시간",
    "타임",
    "언제",
    "어디서",
    "강조",
    "몇 초",
    "구간",
    "씬",
    "장면",
    "scene",
    "몇 분",
]

STOPWORDS = {
    "무엇",
    "무슨",
    "어떤",
    "왜",
    "어떻게",
    "언제",
    "어디",
    "누가",
    "몇",
    "이유",
    "방법",
    "설명",
    "의미",
    "정의",
    "개념",
    "차이",
    "관계",
    "있나요",
    "있어요",
    "인가요",
    "인지",
    "하는",
    "하나요",
    "해줘",
    "알려줘",
    "대해",
    "대한",
    "위한",
    "위해",
    "그리고",
    "또는",
    "하지만",
    "그러나",
    "교수",
    "내용",
    "지금",
    "강의",
    "강의에서",
    "가장",
    "가지",
    "슬라이드",
    "뭐야",
    "거야",
    "어디있지",
    "설명해줘",
    "소개하",
    "알려줘",
}

_CYPHER_FORBIDDEN = re.compile(
    r"\b(CREATE|MERGE|DELETE|DETACH|SET|REMOVE|LOAD\s+CSV|FOREACH|DROP|INSERT|CALL)\b",
    re.IGNORECASE,
)


def _is_emphasis_content_question(question: str) -> bool:
    return "강조" in question and any(
        k in question
        for k in ("내용", "뭐", "무엇", "어떤", "핵심", "중요", "키워드", "개념", "정리", "요약")
    )


def classify_question(question: str) -> str:
    if _is_visual_question(question):
        return "content"
    if _is_emphasis_content_question(question):
        return "content"
    if any(k in question for k in ("설명", "내용", "기능", "역할", "정의", "차이", "이유", "의미", "개념")) and any(
        k in question for k in ("구간", "어디", "어디서", "언제", "장면", "씬")
    ):
        return "content"
    if "슬라이드" in question and any(
        k in question
        for k in ("뭐", "무엇", "어떤", "어느", "찾", "알려", "소개", "관련", "다루", "중요", "핵심", "강조", "비중", "순위", "랭킹")
    ):
        return "content"
    for kw in STRUCTURAL_KEYWORDS:
        if kw in question:
            return "structural"
    return "content"


def extract_keywords_from_question(question: str) -> list[str]:
    keywords: list[str] = []
    quoted = re.findall(
        r"['\u2018\u2019\u201c\u201d\u300c\u300d]"
        r"([^'\u2018\u2019\u201c\u201d\u300c\u300d]+)"
        r"['\u2018\u2019\u201c\u201d\u300c\u300d]",
        question,
    )
    keywords.extend(quoted)

    clean = re.sub(r"['\u2018\u2019\u201c\u201d\u300c\u300d]", " ", question)
    clean = re.sub(r"[?？!！.,，。·]", " ", clean)
    tokens = clean.split()
    for tok in tokens:
        tok = re.sub(
            r"(은|는|이|가|을|를|의|에|에서|에게|로|으로|와|과|도|만|까지|부터|처럼|이란|란|으로서|로서)$",
            "",
            tok,
        )
        tok = tok.strip()
        if tok in {"중요한", "중요하게", "중요도"}:
            tok = "중요"
        elif tok in {"핵심적인", "핵심적"}:
            tok = "핵심"
        elif tok.startswith("비교"):
            tok = "비교"
        elif tok.startswith("강조"):
            tok = "강조"
        if (len(tok) >= 2 or tok in {"표", "그림"}) and tok not in STOPWORDS and tok not in keywords:
            keywords.append(tok)
            if tok == "응용프로그램" and "응용소프트웨어" not in keywords:
                keywords.append("응용소프트웨어")

    seen: set[str] = set()
    out: list[str] = []
    for k in keywords:
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out


def _validate_readonly_cypher(cypher: str) -> None:
    if not cypher or not cypher.strip():
        raise ValueError("빈 Cypher")
    if _CYPHER_FORBIDDEN.search(cypher):
        raise ValueError("허용되지 않는 Cypher 키워드가 포함되어 있습니다.")
    if "$stem" not in cypher:
        raise ValueError("Cypher에 $stem 파라미터가 필요합니다.")


def _gemini_client() -> genai.Client:
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GOOGLE_API_KEY 환경 변수가 없습니다.")
    return genai.Client(api_key=GEMINI_API_KEY)


def _openai_client() -> OpenAI:
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY 환경 변수가 없습니다.")
    return OpenAI(api_key=OPENAI_API_KEY)


def _is_retryable_llm_error(error: Exception) -> bool:
    err_s = str(error)
    return any(k in err_s for k in ("429", "500", "503", "RESOURCE_EXHAUSTED", "UNAVAILABLE", "high demand", "overloaded"))


def _call_llm_raw(contents: str, system_instruction: str, *, temperature: float = 0.0, max_tokens: int = 2048) -> str:
    last_err: Optional[Exception] = None
    for delay in RETRY_DELAYS:
        if delay:
            time.sleep(delay)
        try:
            if QUERY_LLM_PROVIDER == "openai":
                r = _openai_client().chat.completions.create(
                    model=QUERY_OPENAI_MODEL,
                    messages=[
                        {"role": "system", "content": system_instruction},
                        {"role": "user", "content": contents},
                    ],
                    temperature=temperature,
                    max_completion_tokens=max_tokens,
                )
                return (r.choices[0].message.content or "").strip()
            client = _gemini_client()
            r = client.models.generate_content(
                model=GEMINI_ANSWER_MODEL,
                contents=contents,
                config={
                    "system_instruction": system_instruction,
                    "temperature": temperature,
                    "max_output_tokens": max_tokens,
                },
            )
            return (r.text or "").strip()
        except Exception as e:
            last_err = e
            if not _is_retryable_llm_error(e):
                raise
    assert last_err is not None
    raise last_err


def _call_gemini_raw(contents: str, system_instruction: str) -> str:
    return _call_llm_raw(contents, system_instruction, temperature=0.0, max_tokens=2048)


def _format_conversation_history(history: list[dict[str, str]], max_chars: int = 1600) -> str:
    lines: list[str] = []
    for turn in history[-12:]:
        role = "사용자" if turn.get("role") == "user" else "답변"
        content = str(turn.get("content") or "").strip()
        if not content:
            continue
        content = re.sub(r"\s+", " ", content)
        lines.append(f"{role}: {content[:300]}")
    text = "\n".join(lines)
    return text[-max_chars:]


def _last_history_user_question(history: list[dict[str, str]]) -> str:
    fallback = ""
    for turn in reversed(history or []):
        if turn.get("role") != "user":
            continue
        content = re.sub(r"\s+", " ", str(turn.get("content") or "")).strip()
        if not content:
            continue
        if not fallback:
            fallback = content
        if not _looks_like_followup_question(content):
            return content
    return fallback


def _looks_like_followup_question(question: str) -> bool:
    compact = re.sub(r"\s+", "", question or "")
    if not compact:
        return False
    followup_terms = (
        "그이유",
        "왜",
        "그건",
        "그게",
        "그거",
        "그것",
        "이건",
        "이게",
        "이거",
        "이것",
        "앞에서",
        "방금",
        "좀더",
        "자세히",
        "구체적",
        "예시",
        "그러면",
        "그럼",
    )
    has_followup_marker = any(term in compact for term in followup_terms)
    has_topic_hint = len(re.findall(r"[가-힣A-Za-z0-9]{2,}", question or "")) >= 3
    return (len(compact) <= 18 and has_followup_marker) or (has_followup_marker and not has_topic_hint)


def _resolve_followup_question(question: str, conversation_history: Optional[list[dict[str, str]]] = None) -> str:
    question = (question or "").strip()
    if not question or not _looks_like_followup_question(question):
        return question
    previous_question = _last_history_user_question(conversation_history or [])
    if not previous_question:
        return question
    compact = re.sub(r"\s+", "", question)
    if "이유" in compact or "왜" in compact:
        if previous_question.endswith("?"):
            previous_question = previous_question[:-1].strip()
        return f"{previous_question} 이유"
    return f"{previous_question} {question}"


def _call_gemini_answer(
    context: str,
    question: str,
    conversation_history: Optional[list[dict[str, str]]] = None,
    resolved_question: Optional[str] = None,
) -> str:
    history = _format_conversation_history(conversation_history or [])
    history_block = f"이전 대화:\n{history}\n\n" if history else ""
    resolved_block = ""
    if resolved_question and resolved_question.strip() and resolved_question.strip() != question.strip():
        resolved_block = f"이전 대화를 반영한 검색 질문: {resolved_question.strip()}\n\n"
    contents = f"{history_block}현재 질문: {question}\n\n{resolved_block}근거:\n{context}"
    return _call_llm_raw(contents, ANSWER_SYSTEM_PROMPT, temperature=0.2, max_tokens=2048)


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _clean_answer_text(text: str) -> str:
    prev = None
    while prev != text:
        prev = text
        text = re.sub(r"\([^()]*GraphRAG[^()]*\)", "", text)
    text = re.sub(r"\(음성 발췌\)", "", text)
    text = re.sub(r"\s*,\s*(?=[),])", "", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(re.sub(r"[ \t]{2,}", " ", line).strip() for line in text.split("\n"))
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _first_sentence(text: str, max_chars: int = 95) -> str:
    text = _clean_answer_text(text)
    if not text:
        return ""
    m = re.search(r"(.+?[.!?。]|.+?입니다\.|.+?합니다\.|.+?합니다)", text)
    sent = (m.group(1) if m else text).strip()
    if len(sent) <= max_chars:
        return sent
    return sent[:max_chars].rsplit(" ", 1)[0].rstrip(" ,.") + "."


def _source_labels_from_chunks(chunks: list[RetrievedChunk]) -> list[str]:
    def sort_key(c: RetrievedChunk) -> tuple[float, float]:
        score = c.score if c.score is not None else float("-inf")
        start = c.start_sec if c.start_sec is not None else float("inf")
        return (-score, start)

    labels: list[str] = []
    seen: set[str] = set()
    for c in sorted(chunks, key=sort_key):
        if c.start_sec is not None:
            key = f"t:{round(c.start_sec)}"
        elif c.slide_number is not None:
            key = f"s:{c.slide_number}"
        else:
            continue
        if key in seen:
            continue
        seen.add(key)
        if c.slide_number is not None:
            labels.append(f"슬라이드 {c.slide_number}")
        if c.start_sec is not None:
            labels.append(f"약 {c.start_sec:.0f}초")
        if len(seen) >= 3:
            break
    return labels


def _format_time_label(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    return f"{total // 60:02d}:{total % 60:02d}"


def _format_source_block(sources: list[str]) -> str:
    slides: list[str] = []
    times: list[str] = []
    for src in sources:
        slide_match = re.search(r"슬라이드\s*(\d+)", src)
        if slide_match:
            slides.append(slide_match.group(1))
            continue
        sec_match = re.search(r"약\s*(\d+(?:\.\d+)?)초", src)
        if sec_match:
            times.append(_format_time_label(float(sec_match.group(1))))

    lines = ["출처"]
    if slides:
        lines.append(f"- 슬라이드: {', '.join(_dedupe_preserve_order(slides)[:3])}")
    if times:
        lines.append(f"- 시간: {', '.join(_dedupe_preserve_order(times)[:3])}")
    return "\n".join(lines) if len(lines) > 1 else ""


def _is_refusal_answer(text: str) -> bool:
    cleaned = _clean_answer_text(text or "")
    refusal_patterns = (
        "제공된 근거만으로는 답변하기 어렵습니다",
        "제공된 그래프 근거만으로는 답변하기 어렵습니다",
        "근거가 없어 답변할 수 없습니다",
        "답할 수 없습니다",
    )
    return any(p in cleaned for p in refusal_patterns)


def _concise_bullet_body(title: str, body: str) -> str:
    t = title.replace(" ", "")
    if "프로세스" in t or "스레드" in t:
        return "프로그램 실행과 스케줄링을 관리합니다."
    if "메모리" in t:
        return "메모리를 할당하고 보호합니다."
    if "파일" in t or "저장" in t:
        return "파일과 저장장치를 관리합니다."
    if "입출력" in t or "장치" in t:
        return "하드웨어 장치 입출력을 관리합니다."
    if "네트워크" in t:
        return "네트워크 입출력을 관리합니다."
    if "보안" in t or "계정" in t:
        return "사용자 계정과 시스템 보안을 관리합니다."
    if "오류" in t:
        return "오류를 탐지하고 대응합니다."
    return _first_sentence(body, max_chars=45)


def _compact_answer(
    answer: str,
    question: str,
    chunks: Optional[list[RetrievedChunk]] = None,
    source_mode: str = "default",
) -> str:
    """Clean presentation-only noise without truncating or summarizing model content."""
    raw = answer.strip()
    if not raw:
        return raw

    cleaned = _clean_answer_text(raw)
    cleaned = cleaned.replace("**", "")

    cleaned = re.sub(r"[ \t]+[*-]\s+([^:：\n]{1,40})[:：]\s*", r"\n- \1: ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r"\s*출처\s*:.*$", "", cleaned, flags=re.S).strip()
    cleaned = re.sub(r"\n*\s*출처\s*\n(?:\s*[-*].*(?:\n|$))+\s*$", "", cleaned).strip()
    cleaned = re.sub(
        r"\s*\((?:슬라이드\s*\d+\s*,?\s*)?(?:약\s*)?\d+(?:\.\d+)?초\)\s*",
        " ",
        cleaned,
    )
    cleaned = re.sub(
        r"\s*\(슬라이드\s*\d+\)\s*",
        " ",
        cleaned,
    )
    cleaned = re.sub(
        r"\s*\(?\s*관련\s*영상\s*구간에서\s*확인할\s*수\s*있습니다\.?\s*\)?",
        "",
        cleaned,
    )
    cleaned = re.sub(
        r"\n?\s*관련\s*영상\s*구간에서\s*확인할\s*수\s*있습니다\.?\s*",
        "\n",
        cleaned,
    )
    cleaned = re.sub(
        r"\s*\(?\s*관련\s*(?:장면|구간|출처)에서\s*확인할\s*수\s*있습니다\.?\s*\)?",
        "",
        cleaned,
    )
    if source_mode == "visual_location":
        cleaned = re.sub(
            r"(?:은|는)?\s*\d+(?:\.\d+)?초부터\s*\d+(?:\.\d+)?초(?:까지)?(?:,\s*(?:그리고\s*)?\d+(?:\.\d+)?초부터\s*\d+(?:\.\d+)?초(?:까지)?)*\s*(?:나옵니다|등장합니다|확인됩니다|확인할\s*수\s*있습니다)\.?",
            "입니다.",
            cleaned,
        )
        cleaned = re.sub(
            r"\s*\d+(?:\.\d+)?초부터\s*\d+(?:\.\d+)?초(?:까지)?",
            "",
            cleaned,
        )
        cleaned = re.sub(
            r"\n?\s*해당\s*내용은\s*관련\s*영상\s*구간\s*\([^)]*\)\s*에서\s*확인할\s*수\s*있습니다\.?\s*",
            "\n",
            cleaned,
        )
        cleaned = re.sub(
            r"\n?\s*해당\s*내용은\s*관련\s*영상\s*구간에서\s*확인할\s*수\s*있습니다\.?\s*",
            "\n",
            cleaned,
        )
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
        cleaned = _compact_visual_location_answer(cleaned)

    if _is_refusal_answer(cleaned):
        return cleaned

    return cleaned.strip()


def generate_cypher(question: str, stem: str, intent_hint: str = "") -> str:
    extra = ""
    if intent_hint.strip():
        extra = f"\n질문 의도 힌트(참고): {intent_hint.strip()}\n"
    text = _call_gemini_raw(
        f"stem 파라미터 값은 실행 시 전달된다.{extra}질문: {question}\n\n위 질문에 맞는 Cypher만 생성하라.",
        CYPHER_SYSTEM_PROMPT,
    )
    m = re.search(r"```(?:cypher)?\s*(.*?)```", text, re.DOTALL)
    return m.group(1).strip() if m else text.strip()


def _run_cypher_dicts(session, cypher: str, params: dict[str, Any]) -> list[dict[str, Any]]:
    result = session.run(cypher, **params)
    return [dict(record) for record in result]


def run_cypher_structural(session, cypher: str, stem: str) -> list[dict[str, Any]]:
    _validate_readonly_cypher(cypher)
    return _run_cypher_dicts(session, cypher, {"stem": stem})


def run_cypher_safe_structural(
    session, question: str, stem: str, intent_hint: str = ""
) -> tuple[list[dict[str, Any]], str]:
    cypher = generate_cypher(question, stem, intent_hint=intent_hint)
    try:
        return run_cypher_structural(session, cypher, stem), cypher
    except Exception as e:
        fix_prompt = (
            f"다음 Cypher 실행 또는 검증에 실패했다:\n```cypher\n{cypher}\n```\n"
            f"오류: {e}\n\n읽기 전용만 사용하고 모든 노드에 {{stem: $stem}} 를 포함하라. "
            f"수정된 Cypher만 코드 블록(```cypher```)으로 출력하라."
        )
        fixed_text = _call_gemini_raw(fix_prompt, CYPHER_SYSTEM_PROMPT)
        m = re.search(r"```(?:cypher)?\s*(.*?)```", fixed_text, re.DOTALL)
        fixed_cypher = m.group(1).strip() if m else fixed_text.strip()
        return run_cypher_structural(session, fixed_cypher, stem), fixed_cypher


def _build_structural_context(question: str, raw_rows: list[dict[str, Any]], cypher: str) -> str:
    lines = [
        f"질문: {question}",
        "",
        "[지식 그래프 조회 결과 — 구조 질문]",
        f"실행 Cypher:\n{cypher[:2000]}",
        "",
        "결과(JSON, 상위 25행):",
        json.dumps(raw_rows[:25], ensure_ascii=False, default=str),
    ]
    return "\n".join(lines)


def _collect_ids_from_content(structured: dict[str, list[dict[str, Any]]]) -> set[str]:
    ids: set[str] = set()
    for key in ("sub_concepts", "segments", "slides"):
        for r in structured.get(key, []):
            for fld in ("sub_id", "concept_id", "slide_id", "segment_id"):
                v = r.get(fld)
                if v:
                    ids.add(str(v).strip())
    return {x for x in ids if x}


def _collect_ids_from_raw_rows(rows: list[dict[str, Any]]) -> set[str]:
    ids: set[str] = set()
    id_like = re.compile(r"^(slide_|segment/|annotation/|graphrag/)")

    def walk(v: Any) -> None:
        if isinstance(v, str) and id_like.match(v):
            ids.add(v)
        elif isinstance(v, dict):
            for x in v.values():
                walk(x)
        elif isinstance(v, list):
            for x in v:
                walk(x)

    for row in rows:
        walk(row)
    return ids


def _timestamps_from_content(structured: dict[str, list[dict[str, Any]]]) -> list[dict]:
    ts: list[dict] = []
    for r in structured.get("segments", []):
        start = r.get("start")
        if start is None:
            continue
        try:
            s = float(start)
        except (TypeError, ValueError):
            continue
        end = r.get("end")
        try:
            e = float(end) if end is not None else s
        except (TypeError, ValueError):
            e = s
        label = str(r.get("segment_text", ""))[:50]
        ts.append({"label": label, "start": s, "end": e})
    ts.sort(key=lambda x: x["start"])
    return ts[:20]


def _timestamps_from_structural_rows(rows: list[dict[str, Any]]) -> list[dict]:
    ts: list[dict] = []
    for row in rows:
        for key, val in row.items():
            if key in ("start", "start_sec") and val is not None:
                try:
                    s = float(val)
                except (TypeError, ValueError):
                    continue
                end_val = row.get("end") or row.get("end_sec") or s
                try:
                    e = float(end_val)
                except (TypeError, ValueError):
                    e = s
                label = str(row.get("text") or row.get("title") or row.get("name") or "구간")[:50]
                ts.append({"label": label, "start": s, "end": e})
    seen: set[tuple[float, float]] = set()
    out: list[dict] = []
    for t in sorted(ts, key=lambda x: x["start"]):
        k = (t["start"], t["end"])
        if k not in seen:
            seen.add(k)
            out.append(t)
    return out[:20]


def _graph_from_content_structured(structured: dict[str, list[dict[str, Any]]]) -> dict:
    nodes: dict[str, dict] = {}
    edges: list[dict] = []

    def add_node(nid: str, label: str, ntype: str, title: str = "") -> None:
        if nid and nid not in nodes:
            color_map = {
                "GraphRAGEntity": "#FF6B6B",
                "GraphRAGCommunity": "#FF9F43",
                "Slide": "#4ECDC4",
                "Scene": "#A29BFE",
                "Context": "#81ECEC",
                "Segment": "#45B7D1",
                "VisualAsset": "#F59E0B",
            }
            nodes[nid] = {
                "id": nid,
                "label": (label or nid)[:30],
                "color": color_map.get(ntype, "#4ECDC4"),
                "title": title[:300],
                "type": ntype,
            }

    for r in structured.get("segments", []):
        slid = str(r.get("slide_id", ""))
        scene_id = str(r.get("scene_id", ""))
        ctx_id = str(r.get("context_id", ""))
        segid = str(r.get("segment_id", ""))
        if slid:
            add_node(slid, f"S{r.get('slide_number')}", "Slide", str(r.get("segment_text", ""))[:200])
        if scene_id:
            add_node(scene_id, "scene", "Scene", str(r.get("segment_text", ""))[:200])
        if ctx_id:
            add_node(ctx_id, "ctx", "Context", str(r.get("segment_text", ""))[:200])
        if segid:
            add_node(segid, "seg", "Segment", str(r.get("segment_text", ""))[:200])
        if scene_id and slid:
            edges.append({"from": scene_id, "to": slid, "label": "USES_SLIDE"})
        if scene_id and ctx_id:
            edges.append({"from": scene_id, "to": ctx_id, "label": "HAS_CONTEXT"})
        if ctx_id and segid:
            edges.append({"from": ctx_id, "to": segid, "label": "HAS_SEGMENT"})

    for r in structured.get("slides", []):
        slid = str(r.get("slide_id", ""))
        if slid:
            add_node(
                slid,
                f"S{r.get('slide_number')}",
                "Slide",
                str(r.get("slide_text", ""))[:200],
            )

    for r in structured.get("visual_assets", []):
        aid = str(r.get("visual_asset_id", ""))
        slid = str(r.get("slide_id", ""))
        if slid:
            add_node(slid, f"S{r.get('slide_number')}", "Slide")
        if aid:
            add_node(
                aid,
                str(r.get("asset_type", "visual")),
                "VisualAsset",
                (str(r.get("description", "")) + "\n" + str(r.get("raw_text", ""))).strip()[:300],
            )
        if slid and aid:
            edges.append({"from": slid, "to": aid, "label": "HAS_VISUAL_ASSET"})

    for r in structured.get("graphrag_entities", []):
        eid = str(r.get("graphrag_entity_id", ""))
        if eid:
            add_node(
                eid,
                str(r.get("graphrag_title", eid)),
                "GraphRAGEntity",
                str(r.get("graphrag_description", ""))[:300],
            )
        for sid, sn in zip(r.get("slide_ids") or [], r.get("slide_numbers") or []):
            sid = str(sid or "")
            if sid:
                add_node(sid, f"S{sn}", "Slide")
                edges.append({"from": eid, "to": sid, "label": "GRAPHRAG_APPEARS_IN"})
        for scene_id in r.get("scene_ids") or []:
            scene_id = str(scene_id or "")
            if scene_id:
                add_node(scene_id, "scene", "Scene")
                edges.append({"from": eid, "to": scene_id, "label": "GRAPHRAG_APPEARS_IN_SCENE"})

    for r in structured.get("graphrag_relationships", []):
        sid, tid = str(r.get("src_id", "")), str(r.get("tgt_id", ""))
        if sid:
            add_node(sid, str(r.get("src_title", sid)), "GraphRAGEntity")
        if tid:
            add_node(tid, str(r.get("tgt_title", tid)), "GraphRAGEntity")
        if sid and tid:
            edges.append({"from": sid, "to": tid, "label": "GRAPHRAG_RELATES_TO"})

    return {"nodes": list(nodes.values()), "edges": edges}


_GRAPH_COLOR_MAP = {
    "GraphRAGEntity": "#FF6B6B",
    "GraphRAGCommunity": "#FF9F43",
    "Slide": "#4ECDC4",
    "Scene": "#A29BFE",
    "Context": "#81ECEC",
    "Segment": "#45B7D1",
    "VisualAsset": "#F59E0B",
}


def _core_reason_score(reason: str) -> float:
    return {
        "answer_location": 100.0,
        "selected_visual_asset": 96.0,
        "selected_relationship": 92.0,
        "selected_entity": 88.0,
        "source_anchor": 78.0,
        "visual_grounding": 74.0,
        "scene_anchor": 72.0,
        "supporting_concept": 58.0,
        "structural_evidence": 50.0,
    }.get(reason, 10.0)


def _node_payload(
    node_id: str,
    label: str,
    node_type: str,
    title: str = "",
    reason: str = "supporting_concept",
    score: Optional[float] = None,
    evidence: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    payload = {
        "id": node_id,
        "label": (label or node_id)[:48],
        "color": _GRAPH_COLOR_MAP.get(node_type, "#64748b"),
        "title": (title or label or node_id)[:500],
        "type": node_type,
        "reason": reason,
        "score": _core_reason_score(reason) if score is None else score,
    }
    if evidence:
        payload["evidence"] = evidence
    return payload


def _merge_node(nodes: dict[str, dict[str, Any]], node: dict[str, Any]) -> None:
    node_id = str(node.get("id") or "")
    if not node_id:
        return
    prev = nodes.get(node_id)
    if not prev or float(node.get("score") or 0) > float(prev.get("score") or 0):
        nodes[node_id] = node


def _edge_payload(
    src: str,
    tgt: str,
    label: str,
    reason: str = "supporting_concept",
    score: Optional[float] = None,
    evidence: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    payload = {
        "from": src,
        "to": tgt,
        "label": label,
        "reason": reason,
        "score": _core_reason_score(reason) if score is None else score,
    }
    if evidence:
        payload["evidence"] = evidence
    return payload


def _slide_label(slide_number: Any, fallback: str = "slide") -> str:
    n = _to_int_or_none(slide_number)
    return f"S{n}" if n is not None else fallback


def _full_graph_lookup(graph: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[tuple[str, str, str], dict[str, Any]]]:
    nodes = {str(n.get("id")): n for n in graph.get("nodes", []) if n.get("id")}
    edges = {}
    for e in graph.get("edges", []):
        src = str(e.get("from") or e.get("src_id") or "")
        tgt = str(e.get("to") or e.get("tgt_id") or "")
        label = str(e.get("label") or e.get("rel_type") or "")
        if src and tgt and label:
            edges[(src, label, tgt)] = e
    return nodes, edges


def _build_core_graph(
    *,
    stem: str,
    graph: dict[str, Any],
    source_items: list[EvidenceItem],
    source_mode: str,
    related_slides: list[dict[str, Any]],
    retrieved_chunks: list[RetrievedChunk],
    max_nodes: int = 12,
    max_edges: int = 16,
) -> dict[str, Any]:
    """Build the small graph that actually explains the answer evidence."""
    full_nodes, full_edges = _full_graph_lookup(graph)
    nodes: dict[str, dict[str, Any]] = {}
    edges_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    related_slide_numbers = {
        int(s["slide_number"]) for s in related_slides
        if s.get("slide_number") is not None and _to_int_or_none(s.get("slide_number")) is not None
    }
    for c in retrieved_chunks:
        if c.slide_number is not None:
            related_slide_numbers.add(int(c.slide_number))
    evidence_text = "\n".join(
        [it.text for it in source_items if it.text]
        + [c.text for c in retrieved_chunks if c.text]
    ).lower()

    def item_evidence(item: EvidenceItem, note: str = "") -> dict[str, Any]:
        ev: dict[str, Any] = {
            "source_uid": item.uid,
            "source_kind": item.kind,
        }
        if item.retrieval_score is not None:
            ev["retrieval_score"] = item.retrieval_score
        if item.score_breakdown:
            ev["score_breakdown"] = item.score_breakdown
        if item.slide_number is not None:
            ev["slide_number"] = item.slide_number
        if item.start_sec is not None:
            ev["start_sec"] = item.start_sec
        if note:
            ev["note"] = note
        return ev

    def add_node(
        node_id: str,
        label: str,
        node_type: str,
        title: str = "",
        reason: str = "supporting_concept",
        score: Optional[float] = None,
        evidence: Optional[dict[str, Any]] = None,
    ) -> None:
        node_id = str(node_id or "")
        if not node_id:
            return
        existing = full_nodes.get(node_id, {})
        _merge_node(
            nodes,
            _node_payload(
                node_id=node_id,
                label=str(existing.get("label") or label or node_id),
                node_type=str(existing.get("type") or node_type),
                title=str(existing.get("title") or title or label or node_id),
                reason=reason,
                score=score,
                evidence=evidence,
            ),
        )

    def add_edge(
        src: str,
        tgt: str,
        label: str,
        reason: str = "supporting_concept",
        score: Optional[float] = None,
        evidence: Optional[dict[str, Any]] = None,
    ) -> None:
        src, tgt, label = str(src or ""), str(tgt or ""), str(label or "")
        if not src or not tgt or not label:
            return
        key = (src, label, tgt)
        edge = _edge_payload(src, tgt, label, reason=reason, score=score, evidence=evidence)
        prev = edges_by_key.get(key)
        if not prev or float(edge.get("score") or 0) > float(prev.get("score") or 0):
            edges_by_key[key] = edge

    def add_full_edge_if_exists(
        src: str,
        label: str,
        tgt: str,
        reason: str,
        evidence: Optional[dict[str, Any]] = None,
    ) -> None:
        if (src, label, tgt) in full_edges:
            add_edge(src, tgt, label, reason=reason, evidence=evidence)

    for slide_number in related_slide_numbers:
        slide_id = f"slide_{slide_number:03d}"
        add_node(
            slide_id,
            f"S{slide_number}",
            "Slide",
            reason="answer_location",
            evidence={"note": "selected from related_slides/retrieved_chunks"},
        )

    include_scene = source_mode in {"visual_location", "scene_location"}
    include_segment = source_mode == "scene_location"

    for item in source_items:
        row = item.row or {}
        if item.kind == "visual_asset":
            ev = item_evidence(item, "visual asset selected for answer context")
            aid = str(row.get("visual_asset_id") or "")
            slide_id = str(row.get("slide_id") or "")
            slide_number = row.get("slide_number") or item.slide_number
            if slide_id:
                add_node(slide_id, _slide_label(slide_number), "Slide", reason="answer_location", evidence=ev)
            if aid:
                add_node(
                    aid,
                    str(row.get("asset_type") or "visual"),
                    "VisualAsset",
                    (str(row.get("description") or "") + "\n" + str(row.get("raw_text") or "")).strip(),
                    reason="selected_visual_asset",
                    evidence=ev,
                )
            if slide_id and aid:
                add_edge(slide_id, aid, "HAS_VISUAL_ASSET", reason="visual_grounding", evidence=ev)

        elif item.kind in {"slide_text", "slide_concept"}:
            ev = item_evidence(item, "slide evidence selected for answer context")
            slide_id = str(row.get("slide_id") or "")
            if slide_id:
                add_node(
                    slide_id,
                    _slide_label(row.get("slide_number") or item.slide_number),
                    "Slide",
                    str(row.get("slide_text") or item.text),
                    reason="source_anchor",
                    evidence=ev,
                )

        elif item.kind == "segment":
            ev = item_evidence(item, "segment evidence selected for answer context")
            slide_id = str(row.get("slide_id") or "")
            scene_id = str(row.get("scene_id") or "")
            segment_id = str(row.get("segment_id") or "")
            segment_text = str(row.get("segment_text") or item.text)
            if slide_id:
                add_node(slide_id, _slide_label(row.get("slide_number") or item.slide_number), "Slide", reason="source_anchor", evidence=ev)
            if include_scene and scene_id:
                add_node(scene_id, "scene", "Scene", segment_text, reason="scene_anchor", evidence=ev)
            if include_scene and scene_id and slide_id:
                add_edge(scene_id, slide_id, "USES_SLIDE", reason="scene_anchor", evidence=ev)
            if include_segment and segment_id:
                add_node(segment_id, "segment", "Segment", segment_text, reason="structural_evidence", evidence=ev)

        elif item.kind == "graphrag_entity":
            ev = item_evidence(item, "GraphRAG entity selected for answer context")
            eid = str(row.get("graphrag_entity_id") or "")
            if eid:
                add_node(
                    eid,
                    str(row.get("graphrag_title") or eid),
                    "GraphRAGEntity",
                    str(row.get("graphrag_description") or item.text),
                    reason="selected_entity",
                    score=_core_reason_score("selected_entity") + (_to_float_or_none(row.get("final_weight")) or 0.0),
                    evidence=ev,
                )
                if source_mode in {"visual_location", "scene_location", "overview", "default"}:
                    for sid, sn in zip(row.get("slide_ids") or [], row.get("slide_numbers") or []):
                        if related_slide_numbers and _to_int_or_none(sn) not in related_slide_numbers:
                            continue
                        sid = str(sid or "")
                        if sid:
                            add_node(sid, _slide_label(sn), "Slide", reason="source_anchor", evidence=ev)
                            add_edge(eid, sid, "GRAPHRAG_APPEARS_IN", reason="source_anchor", evidence=ev)

        elif item.kind == "graphrag_relationship":
            ev = item_evidence(item, "GraphRAG relationship selected for answer context")
            src = str(row.get("src_id") or "")
            tgt = str(row.get("tgt_id") or "")
            if src:
                add_node(src, str(row.get("src_title") or src), "GraphRAGEntity", reason="selected_relationship", evidence=ev)
            if tgt:
                add_node(tgt, str(row.get("tgt_title") or tgt), "GraphRAGEntity", reason="selected_relationship", evidence=ev)
            if src and tgt:
                add_edge(src, tgt, "GRAPHRAG_RELATES_TO", reason="selected_relationship", evidence=ev)

    if source_mode in {"visual_location", "scene_location"}:
        # Keep location graphs readable: connect selected concepts to chosen scenes only when already selected.
        selected_scene_ids = [nid for nid, node in nodes.items() if node.get("type") == "Scene"]
        selected_entity_ids = [nid for nid, node in nodes.items() if node.get("type") == "GraphRAGEntity"]
        for eid in selected_entity_ids:
            for scene_id in selected_scene_ids[:2]:
                add_full_edge_if_exists(
                    eid,
                    "GRAPHRAG_APPEARS_IN_SCENE",
                    scene_id,
                    "source_anchor",
                    evidence={"note": "selected concept appears in selected scene"},
                )

    if source_mode == "visual_location" and related_slide_numbers:
        slide_ids = {f"slide_{n:03d}" for n in related_slide_numbers}
        candidate_entities: list[tuple[float, str, dict[str, Any]]] = []
        for eid, node in full_nodes.items():
            if node.get("type") != "GraphRAGEntity":
                continue
            label = str(node.get("label") or "").strip()
            title = str(node.get("title") or "").strip()
            hay_terms = [x.lower() for x in (label, title) if len(x.strip()) >= 2]
            text_hit = any(term and term in evidence_text for term in hay_terms)
            linked_slide = any((eid, "GRAPHRAG_APPEARS_IN", sid) in full_edges for sid in slide_ids)
            if not text_hit and not linked_slide:
                continue
            score = _core_reason_score("supporting_concept")
            if linked_slide:
                score += 12.0
            if text_hit:
                score += 18.0
            score += min(6.0, sum(1 for term in hay_terms if term and term in evidence_text) * 2.0)
            candidate_entities.append((score, eid, node, {"text_hit": text_hit, "linked_slide": linked_slide}))

        already = {nid for nid, node in nodes.items() if node.get("type") == "GraphRAGEntity"}
        for score, eid, node, evidence_flags in sorted(candidate_entities, reverse=True):
            if eid in already:
                continue
            add_node(
                eid,
                str(node.get("label") or eid),
                "GraphRAGEntity",
                str(node.get("title") or node.get("label") or eid),
                reason="supporting_concept",
                score=score,
                evidence={
                    "note": "concept matched selected visual evidence and/or related slide",
                    **evidence_flags,
                },
            )
            for sid in slide_ids:
                add_full_edge_if_exists(
                    eid,
                    "GRAPHRAG_APPEARS_IN",
                    sid,
                    "source_anchor",
                    evidence={
                        "note": "supporting concept appears in selected slide",
                        **evidence_flags,
                    },
                )
            already.add(eid)
            if len(already) >= 5:
                break

        selected_entity_ids = [nid for nid, node in nodes.items() if node.get("type") == "GraphRAGEntity"]
        for i, src in enumerate(selected_entity_ids):
            for tgt in selected_entity_ids[i + 1:]:
                add_full_edge_if_exists(
                    src,
                    "GRAPHRAG_RELATES_TO",
                    tgt,
                    "selected_relationship",
                    evidence={"note": "relationship exists between selected/supporting concepts"},
                )
                add_full_edge_if_exists(
                    tgt,
                    "GRAPHRAG_RELATES_TO",
                    src,
                    "selected_relationship",
                    evidence={"note": "relationship exists between selected/supporting concepts"},
                )

    selected_entity_ids = [nid for nid, node in nodes.items() if node.get("type") == "GraphRAGEntity"]
    selected_slide_ids = [
        nid for nid, node in nodes.items()
        if node.get("type") == "Slide"
    ]
    if stem and selected_entity_ids and selected_slide_ids:
        try:
            driver = _neo4j_driver()
            try:
                with driver.session() as session:
                    rows = session.run(
                        """
                        MATCH (e:GraphRAGEntity {stem: $stem})-[r:GRAPHRAG_APPEARS_IN]->(s:Slide {stem: $stem})
                        WHERE e.id IN $entity_ids AND s.id IN $slide_ids
                        RETURN e.id AS entity_id,
                               s.id AS slide_id,
                               s.slide_number AS slide_number
                        LIMIT 48
                        """,
                        stem=stem,
                        entity_ids=selected_entity_ids,
                        slide_ids=selected_slide_ids,
                    )
                    for row in rows:
                        add_edge(
                            str(row.get("entity_id") or ""),
                            str(row.get("slide_id") or ""),
                            "GRAPHRAG_APPEARS_IN",
                            reason="source_anchor",
                            evidence={
                                "note": "selected core concept appears in selected slide",
                                "slide_number": _to_int_or_none(row.get("slide_number")),
                            },
                        )
            finally:
                driver.close()
        except Exception:
            pass

    for i, src in enumerate(selected_entity_ids):
        for tgt in selected_entity_ids[i + 1:]:
            add_full_edge_if_exists(
                src,
                "GRAPHRAG_RELATES_TO",
                tgt,
                "selected_relationship",
                evidence={"note": "relationship exists between selected core concepts"},
            )
            add_full_edge_if_exists(
                tgt,
                "GRAPHRAG_RELATES_TO",
                src,
                "selected_relationship",
                evidence={"note": "relationship exists between selected core concepts"},
            )

    if stem and selected_entity_ids:
        try:
            driver = _neo4j_driver()
            try:
                with driver.session() as session:
                    rows = session.run(
                        """
                        MATCH (a:GraphRAGEntity {stem: $stem})-[r:GRAPHRAG_RELATES_TO]->(b:GraphRAGEntity {stem: $stem})
                        WHERE a.id IN $ids AND b.id IN $ids
                        RETURN a.id AS src_id, b.id AS tgt_id,
                               coalesce(r.description, '') AS description,
                               coalesce(r.emphasis_edge_weight, r.weight, 0) AS weight
                        ORDER BY weight DESC
                        LIMIT 24
                        """,
                        stem=stem,
                        ids=selected_entity_ids,
                    )
                    for row in rows:
                        add_edge(
                            str(row.get("src_id") or ""),
                            str(row.get("tgt_id") or ""),
                            "GRAPHRAG_RELATES_TO",
                            reason="selected_relationship",
                            score=_core_reason_score("selected_relationship") + (_to_float_or_none(row.get("weight")) or 0.0),
                            evidence={
                                "note": "Neo4j relationship between selected core concepts",
                                "description": str(row.get("description") or "")[:300],
                            },
                        )
            finally:
                driver.close()
        except Exception:
            pass

    sorted_nodes = sorted(nodes.values(), key=lambda n: float(n.get("score") or 0), reverse=True)[:max_nodes]
    node_ids = {str(n["id"]) for n in sorted_nodes}
    sorted_edges = sorted(edges_by_key.values(), key=lambda e: float(e.get("score") or 0), reverse=True)
    sorted_edges = [e for e in sorted_edges if str(e.get("from")) in node_ids and str(e.get("to")) in node_ids][:max_edges]
    connected_ids = {str(e.get("from")) for e in sorted_edges} | {str(e.get("to")) for e in sorted_edges}
    if sorted_edges:
        sorted_nodes = [
            n for n in sorted_nodes
            if str(n.get("id")) in connected_ids or n.get("reason") in {"answer_location", "selected_visual_asset"}
        ]
    return {
        "mode": source_mode,
        "nodes": sorted_nodes,
        "edges": sorted_edges,
        "node_count": len(sorted_nodes),
        "edge_count": len(sorted_edges),
    }


def _graph_from_structural_rows(rows: list[dict[str, Any]]) -> dict:
    """LLM Cypher 결과가 다양해 완전한 그래프는 어렵고, 노드 id 문자열만 수집."""
    nodes: dict[str, dict] = {}
    id_like = re.compile(r"^(slide_|segment/|annotation/|graphrag/)")

    def add_from_val(v: Any) -> None:
        if isinstance(v, str) and id_like.match(v) and v not in nodes:
            nodes[v] = {"id": v, "label": v[:30], "color": "#B0C4DE", "title": "", "type": "node"}

    for row in rows:
        for v in row.values():
            add_from_val(v)
            if isinstance(v, dict):
                for x in v.values():
                    add_from_val(x)
    return {"nodes": list(nodes.values()), "edges": []}


def _distance_to_score(dist: float) -> float:
    if dist is None or math.isnan(dist):
        return 0.0
    return 1.0 / (1.0 + float(dist))


def _rows_to_chunks(df) -> list[RetrievedChunk]:
    out: list[RetrievedChunk] = []
    for _, row in df.iterrows():
        dist = row.get("_distance")
        try:
            d = float(dist) if dist is not None else None
        except (TypeError, ValueError):
            d = None
        text = str(row.get("text", ""))
        out.append(
            RetrievedChunk(
                chunk_id=str(row.get("chunk_id", "")),
                stem=str(row.get("stem", "")),
                chunk_type=str(row.get("chunk_type", "")),
                text=text[:2000],
                score=_distance_to_score(d) if d is not None else None,
                slide_number=row.get("slide_number"),
                start_sec=float(row["start_sec"]) if row.get("start_sec") is not None else None,
                end_sec=float(row["end_sec"]) if row.get("end_sec") is not None else None,
                linked_node_id=str(row["linked_node_id"]) if row.get("linked_node_id") is not None else None,
            )
        )
    return out


def _evidence_to_retrieved_chunks(stem: str, items: list[EvidenceItem]) -> list[RetrievedChunk]:
    """LLM 컨텍스트에 실제로 선택된 근거를 API 출처 청크로 변환한다."""
    out: list[RetrievedChunk] = []
    for it in items:
        if it.start_sec is None and it.slide_number is None:
            continue
        cid = it.uid
        if cid.startswith("lance:"):
            cid = cid[6:]
        out.append(
            RetrievedChunk(
                chunk_id=cid[:500],
                stem=stem,
                chunk_type=it.chunk_type or it.kind,
                text=it.text[:2000],
                score=it.retrieval_score if it.retrieval_score is not None else it.lance_score,
                slide_number=it.slide_number,
                start_sec=it.start_sec,
                end_sec=it.end_sec,
                linked_node_id=it.linked_node_id,
                score_breakdown=it.score_breakdown,
            )
        )
    return out


def _chunk_needs_segment_label(chunk: RetrievedChunk) -> bool:
    if chunk.start_sec is None:
        return False
    text = (chunk.text or "").strip()
    if not text:
        return True
    compact = re.sub(r"\s+", "", text)
    if text.startswith("{") or text.startswith("["):
        return True
    return (
        text.startswith("슬라이드")
        or "슬라이드강조점수" in compact
        or text.startswith("시각자료(")
        or chunk.chunk_type in {"slide", "slide_text", "slide_concept", "visual_asset", "lance_strict", "lance_soft"}
    )


def _hydrate_chunk_segment_labels(
    stem: str,
    chunks: list[RetrievedChunk],
    structured: dict[str, list[dict[str, Any]]],
) -> list[RetrievedChunk]:
    segments: list[dict[str, Any]] = []
    for row in structured.get("segments", []):
        text = str(row.get("segment_text") or "").strip()
        start = _to_float_or_none(row.get("start"))
        if not text or start is None:
            continue
        segments.append(
            {
                "text": text,
                "start": start,
                "end": _to_float_or_none(row.get("end")),
                "slide_number": _to_int_or_none(row.get("slide_number")),
                "segment_id": str(row.get("segment_id") or ""),
            }
        )
    if not segments:
        segments = []

    def nearest_segment(chunk: RetrievedChunk) -> Optional[dict[str, Any]]:
        if chunk.start_sec is None:
            return None
        try:
            chunk_slide = int(chunk.slide_number) if chunk.slide_number is not None else None
        except (TypeError, ValueError):
            chunk_slide = None
        candidates = segments
        if chunk_slide is not None:
            slide_candidates = [s for s in segments if s.get("slide_number") == chunk_slide]
            if slide_candidates:
                candidates = slide_candidates
        best = None
        best_gap = float("inf")
        for seg in candidates:
            start = float(seg["start"])
            end = seg.get("end")
            if end is not None and start <= float(chunk.start_sec) <= float(end):
                gap = 0.0
            else:
                gap = abs(start - float(chunk.start_sec))
            if gap < best_gap:
                best_gap = gap
                best = seg
        return best if best is not None and best_gap <= 45.0 else None

    def neo4j_nearest_segment(chunk: RetrievedChunk) -> Optional[dict[str, Any]]:
        if chunk.start_sec is None:
            return None
        try:
            driver = _neo4j_driver()
            try:
                with driver.session() as session:
                    row = session.run(
                        """
                        MATCH (seg:Segment {stem: $stem})
                        WHERE seg.start IS NOT NULL
                          AND abs(toFloat(seg.start) - $target) <= 75.0
                        OPTIONAL MATCH (ctx:Context {stem: $stem})-[:HAS_SEGMENT]->(seg)
                        OPTIONAL MATCH (scene:Scene {stem: $stem})-[:HAS_CONTEXT]->(ctx)
                        OPTIONAL MATCH (scene)-[:USES_SLIDE]->(slide:Slide {stem: $stem})
                        WITH seg, slide,
                             CASE
                               WHEN $slide_number IS NOT NULL
                                    AND slide.slide_number IS NOT NULL
                                    AND toInteger(slide.slide_number) = $slide_number
                               THEN 0 ELSE 1
                             END AS slide_penalty,
                             abs(toFloat(seg.start) - $target) AS gap
                        RETURN coalesce(seg.text, '') AS text,
                               seg.start AS start,
                               seg.end AS end,
                               coalesce(seg.id, '') AS segment_id,
                               slide.slide_number AS slide_number
                        ORDER BY slide_penalty ASC, gap ASC
                        LIMIT 1
                        """,
                        stem=stem,
                        target=float(chunk.start_sec),
                        slide_number=_to_int_or_none(chunk.slide_number),
                    ).single()
                    if not row:
                        return None
                    text = str(row.get("text") or "").strip()
                    start = _to_float_or_none(row.get("start"))
                    if not text or start is None:
                        return None
                    return {
                        "text": text,
                        "start": start,
                        "end": _to_float_or_none(row.get("end")),
                        "slide_number": _to_int_or_none(row.get("slide_number")),
                        "segment_id": str(row.get("segment_id") or ""),
                    }
            finally:
                driver.close()
        except Exception:
            return None

    hydrated: list[RetrievedChunk] = []
    for chunk in chunks:
        if not _chunk_needs_segment_label(chunk):
            hydrated.append(chunk)
            continue
        seg = nearest_segment(chunk) or neo4j_nearest_segment(chunk)
        if not seg:
            hydrated.append(chunk)
            continue
        data = chunk.model_dump() if hasattr(chunk, "model_dump") else chunk.dict()
        data["text"] = seg["text"]
        data["chunk_type"] = "segment"
        data["chunk_id"] = seg.get("segment_id") or data.get("chunk_id", "")
        data["start_sec"] = seg["start"]
        data["end_sec"] = seg.get("end") if seg.get("end") is not None else chunk.end_sec
        if seg.get("slide_number") is not None:
            data["slide_number"] = seg["slide_number"]
        hydrated.append(RetrievedChunk(**data))
    return hydrated


def _structural_rows_to_retrieved_chunks(stem: str, rows: list[dict[str, Any]]) -> list[RetrievedChunk]:
    out: list[RetrievedChunk] = []
    for i, row in enumerate(rows[:20], start=1):
        start_val = row.get("start") if row.get("start") is not None else row.get("start_sec")
        end_val = row.get("end") if row.get("end") is not None else row.get("end_sec")
        start = _to_float_or_none(start_val)
        end = _to_float_or_none(end_val)
        slide_number = _to_int_or_none(row.get("slide_number"))
        label = row.get("text") or row.get("title") or row.get("name")
        text = str(label).strip() if label else json.dumps(row, ensure_ascii=False, default=str)
        out.append(
            RetrievedChunk(
                chunk_id=f"structural:{i}",
                stem=stem,
                chunk_type="structural_row",
                text=text[:2000],
                slide_number=slide_number,
                start_sec=start,
                end_sec=end if end is not None else start,
            )
        )
    return out


def _chunks_to_timestamps(chunks: list[RetrievedChunk]) -> list[dict]:
    ts: list[dict] = []
    seen: set[tuple[float, float, str]] = set()
    for c in chunks:
        if c.start_sec is None:
            continue
        label = (c.text or "")[:50]
        end = c.end_sec if c.end_sec is not None else c.start_sec
        key = (round(c.start_sec, 2), round(end, 2), label)
        if key in seen:
            continue
        seen.add(key)
        ts.append(
            {
                "label": label,
                "start": c.start_sec,
                "end": end,
            }
        )
    return ts[:20]


def _is_visual_question(question: str) -> bool:
    return any(k in question for k in ("시각자료", "그림", "이미지", "표", "도표", "비교표", "다이어그램", "구조도", "화살표", "양방향"))


def _is_visual_interpretation_question(question: str) -> bool:
    if not _is_visual_question(question):
        return False
    q = question.replace(" ", "")
    return any(k in q for k in ("뭘의미", "무엇을의미", "의미해", "의미야", "왜", "이유", "설명해", "나타내", "말하는"))


def _is_visual_list_question(question: str) -> bool:
    if not _is_visual_question(question):
        return False
    q = question.replace(" ", "")
    return (
        any(k in q for k in ("이강의에서", "전체", "모두", "전부"))
        and any(k in q for k in ("나오는", "등장하는", "있는장면", "있는슬라이드", "장면들", "슬라이드들", "슬라이드만"))
    )


def _is_lecture_overview_question(question: str) -> bool:
    return _has_lecture_overview_scope(question)


def _has_lecture_overview_scope(question: str) -> bool:
    q = question.replace(" ", "")
    overview_terms = (
        "강의내용요약",
        "내용요약",
        "전체요약",
        "전체내용",
        "전체흐름",
        "내용정리",
        "강의정리",
        "수업정리",
        "큰그림",
        "뭘배웠",
        "무엇을배웠",
        "어떤내용",
        "주로뭘",
        "주로무엇",
        "주로다뤄",
        "뭘다뤄",
        "무엇을다뤄",
        "전반적으로",
    )
    scoped_summary = (
        ("요약" in q or "정리" in q or "흐름" in q)
        and any(scope in q for scope in ("강의", "수업", "전체", "내용", "전반"))
    )
    return any(
        k in q
        for k in overview_terms
    ) or scoped_summary


def _is_location_question(question: str) -> bool:
    return any(k in question for k in ("구간", "어디", "어디서", "언제", "장면", "씬"))


def _is_definition_question(question: str) -> bool:
    q = question.replace(" ", "")
    return any(k in q for k in ("뭐야", "무엇", "정의", "개념", "뜻이", "의미"))


def _is_comparison_question(question: str) -> bool:
    q = question.replace(" ", "").lower()
    return any(k in q for k in ("차이", "비교", "다른점", "반면", "vs", "versus"))


def _is_core_content_question(question: str) -> bool:
    q = question.replace(" ", "")
    return (
        any(k in q for k in ("핵심", "중요", "주요", "시험", "출제", "나올것같", "나올만한"))
        and any(k in q for k in ("내용", "개념", "키워드", "용어", "알려줘", "뭐야", "정리"))
    )


def _is_emphasis_overview_question(question: str) -> bool:
    q = question.replace(" ", "")
    return "강조" in q and any(k in q for k in ("내용", "뭐", "무엇", "어떤", "핵심", "중요", "키워드", "개념", "정리"))


def _is_emphasis_location_question(question: str) -> bool:
    return "강조" in question and any(k in question for k in ("장면", "씬", "구간", "어디", "언제"))


def _evidence_slide_numbers(items: list[EvidenceItem], kinds: set[str]) -> set[int]:
    slides: set[int] = set()
    for it in items:
        if it.kind not in kinds or it.slide_number is None:
            continue
        try:
            slides.add(int(it.slide_number))
        except (TypeError, ValueError):
            continue
    return slides


def _chunk_score(item: EvidenceItem) -> float:
    score = item.retrieval_score if item.retrieval_score is not None else item.lance_score
    try:
        return float(score) if score is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


def _emphasis_score_for_item(item: EvidenceItem) -> float:
    row = item.row or {}
    if item.kind == "segment":
        return max(
            _to_float_or_none(row.get("scene_emphasis_total")) or 0.0,
            _to_float_or_none(row.get("slide_emphasis_total")) or 0.0,
        )
    return _to_float_or_none(row.get("emphasis_total")) or 0.0


def _is_overview_item(item: EvidenceItem) -> bool:
    text = item.text.replace(" ", "").lower()
    return any(
        k in text
        for k in (
            "강의목표",
            "강의의목표",
            "학습목표",
            "목차",
            "개요",
            "chapter",
            "이장의목적",
            "강의의목적",
            "학습의목적",
        )
    )


def _is_overview_text(text: str) -> bool:
    compact = (text or "").replace(" ", "").lower()
    return any(
        k in compact
        for k in (
            "강의목표",
            "강의의목표",
            "학습목표",
            "목차",
            "개요",
            "chapter",
            "이장의목적",
            "강의의목적",
            "학습의목적",
        )
    )


def _is_title_or_intro_text(text: str) -> bool:
    compact = (text or "").replace(" ", "").lower()
    if _is_overview_text(text):
        return True
    if any(k in compact for k in ("[slide_001]", "운영체제의시작과발전")):
        return True
    intro_markers = (
        "알아보도록하죠",
        "알아보도록하겠습니다",
        "알아보고",
        "보겠습니다",
        "시작하겠습니다",
        "들어보셨을",
        "말을많이들었",
    )
    if any(k in compact for k in intro_markers) and len(compact) < 180:
        return True
    return False


def _is_overview_target_question(question: str) -> bool:
    q = question.replace(" ", "")
    return any(k in q for k in ("강의목표", "학습목표", "목차", "개요", "이장의목적", "강의의목적"))


def _is_low_signal_slide_item(item: EvidenceItem) -> bool:
    if item.kind not in {"slide_text", "slide_concept", "visual_asset"}:
        return False
    return _overview_related_score(item) < 3.0 and _chunk_score(item) < 0.75


def _focus_keywords_for_location(question: str) -> list[str]:
    drop = {
        "설명",
        "설명하는",
        "구간",
        "어디",
        "어디서",
        "언제",
        "장면",
        "장면들",
        "씬",
        "교수",
        "알려줘",
    }
    return [k for k in extract_keywords_from_question(question) if k not in drop]


def _source_items_for_question(question: str, items: list[EvidenceItem]) -> list[EvidenceItem]:
    if _is_visual_question(question):
        visual_items = [
            it for it in items
            if it.kind == "visual_asset" and it.slide_number is not None
        ]
        if visual_items:
            if _is_visual_list_question(question):
                visual_items.sort(key=lambda it: int(it.slide_number or 10**9))
            else:
                visual_items.sort(key=_chunk_score, reverse=True)
            return visual_items[:8 if _is_visual_list_question(question) else 4]

    if _is_core_content_question(question):
        candidates = [
            it for it in items
            if it.kind in {"graphrag_entity", "graphrag_relationship", "slide_text", "slide_concept", "visual_asset"}
            and not _is_overview_item(it)
            and not _is_low_signal_slide_item(it)
        ]
        if candidates:
            candidates.sort(
                key=lambda it: (
                    _overview_related_score(it),
                    _emphasis_score_for_item(it),
                    _chunk_score(it),
                ),
                reverse=True,
            )
            return candidates[:8]

    if _is_emphasis_overview_question(question):
        candidates = [
            it for it in items
            if it.kind in {"graphrag_entity", "graphrag_relationship", "slide_text", "slide_concept", "visual_asset"}
            and not _is_overview_item(it)
            and not _is_low_signal_slide_item(it)
        ]
        if candidates:
            candidates.sort(
                key=lambda it: (
                    _emphasis_score_for_item(it),
                    _overview_related_score(it),
                    _chunk_score(it),
                ),
                reverse=True,
            )
            return candidates[:8]

    if not _is_location_question(question):
        if _is_overview_target_question(question):
            return items
        non_overview = [it for it in items if not _is_overview_item(it)]
        return non_overview or items

    keywords = _focus_keywords_for_location(question)

    if not _is_emphasis_location_question(question):
        candidates = [
            it for it in items
            if it.kind in {"segment", "slide_text", "slide_concept", "visual_asset", "lance_strict"}
            and it.slide_number is not None
            and not _is_overview_item(it)
        ]
        if keywords:
            focused = [it for it in candidates if any(k.lower() in it.text.lower() for k in keywords)]
            if focused:
                candidates = focused
        if not candidates:
            return items
        candidates.sort(key=_chunk_score, reverse=True)
        return candidates[:4]

    keywords = [k for k in keywords if k not in {"강조"}]
    emphasized = [it for it in items if _emphasis_score_for_item(it) > 0]
    non_overview = [it for it in emphasized if not _is_overview_item(it)]
    if non_overview:
        emphasized = non_overview
    if keywords:
        focused = [it for it in emphasized if any(k.lower() in it.text.lower() for k in keywords)]
        if focused:
            emphasized = focused
    if not emphasized:
        return items
    emphasized.sort(key=lambda it: (_emphasis_score_for_item(it), _chunk_score(it)), reverse=True)
    return emphasized[:4]


def _visual_asset_slide_numbers(items: list[EvidenceItem], question: str = "") -> set[int]:
    visual_items = [
        it for it in items
        if it.kind == "visual_asset" and it.slide_number is not None
    ]
    if not visual_items:
        return set()

    if _is_visual_list_question(question):
        slides: set[int] = set()
        for it in visual_items:
            try:
                slides.add(int(it.slide_number))
            except (TypeError, ValueError):
                continue
        return slides

    def score_of(it: EvidenceItem) -> float:
        score = it.retrieval_score if it.retrieval_score is not None else it.lance_score
        try:
            return float(score) if score is not None else 0.0
        except (TypeError, ValueError):
            return 0.0

    best_score = max(score_of(it) for it in visual_items)
    slides: set[int] = set()
    for it in visual_items:
        if score_of(it) < best_score - 1e-9:
            continue
        try:
            slides.add(int(it.slide_number))
        except (TypeError, ValueError):
            continue
    return slides


def _filter_location_question_chunks(
    question: str,
    items: list[EvidenceItem],
    chunks: list[RetrievedChunk],
) -> list[RetrievedChunk]:
    if not _is_location_question(question):
        return chunks

    evidence_slides = _evidence_slide_numbers(items, {"slide_text", "slide_concept", "visual_asset"})
    if not evidence_slides:
        return chunks

    filtered: list[RetrievedChunk] = []
    for c in chunks:
        try:
            slide_number = int(c.slide_number) if c.slide_number is not None else None
        except (TypeError, ValueError):
            slide_number = None
        if slide_number in evidence_slides:
            filtered.append(c)
    return filtered


def _filter_visual_question_chunks(
    question: str,
    items: list[EvidenceItem],
    chunks: list[RetrievedChunk],
) -> list[RetrievedChunk]:
    visual_slides = _visual_asset_slide_numbers(items, question)
    if not visual_slides or not _is_visual_question(question):
        return chunks

    filtered: list[RetrievedChunk] = []
    for c in chunks:
        try:
            slide_number = int(c.slide_number) if c.slide_number is not None else None
        except (TypeError, ValueError):
            slide_number = None
        if slide_number in visual_slides and c.chunk_type in {"slide", "visual_asset"}:
            filtered.append(c)
    return filtered


def _source_mode_for(question: str, items: list[EvidenceItem]) -> str:
    if _is_lecture_overview_question(question):
        return "overview"
    if _is_core_content_question(question) or _is_emphasis_overview_question(question):
        return "overview"
    if _is_visual_question(question) and _visual_asset_slide_numbers(items, question):
        return "visual_location"
    if _is_emphasis_location_question(question):
        return "scene_location"
    return "default"


def _overview_related_score(item: EvidenceItem) -> float:
    row = item.row or {}
    if item.kind in {"slide_text", "slide_concept"}:
        return (
            (_to_float_or_none(row.get("emphasis_total")) or 0.0)
            + (3.0 if row.get("visual_asset_text") or row.get("t1_structure") else 0.0)
        )
    if item.kind == "visual_asset":
        return 12.0
    if item.kind == "segment":
        return max(
            _to_float_or_none(row.get("scene_emphasis_total")) or 0.0,
            _to_float_or_none(row.get("slide_emphasis_total")) or 0.0,
        )
    if item.kind == "graphrag_entity":
        return (
            (_to_float_or_none(row.get("final_weight")) or 0.0)
            + 0.75 * (_to_float_or_none(row.get("emphasis_boost_local")) or 0.0)
            + 0.20 * (_to_float_or_none(row.get("frequency")) or 0.0)
            + 0.10 * (_to_float_or_none(row.get("degree")) or 0.0)
        )
    if item.kind == "graphrag_relationship":
        return (
            (_to_float_or_none(row.get("emphasis_edge_weight")) or 0.0)
            + 0.10 * (_to_float_or_none(row.get("combined_degree")) or 0.0)
        )
    return 0.0


def _related_slides_from_evidence(
    items: list[EvidenceItem],
    chunks: list[RetrievedChunk],
    max_items: int = 6,
    preserve_item_order: bool = False,
    filter_to_visual_slides: bool = True,
    overview_rank: bool = False,
) -> list[dict]:
    scored: list[dict] = []
    visual_slides = _visual_asset_slide_numbers(items) if filter_to_visual_slides else set()
    order_by_slide: dict[int, int] = {}
    for idx, it in enumerate(items):
        if it.slide_number is None:
            continue
        try:
            order_by_slide.setdefault(int(it.slide_number), idx)
        except (TypeError, ValueError):
            continue
    for it in items:
        if visual_slides and it.kind != "visual_asset" and it.slide_number not in visual_slides:
            continue
        score = it.retrieval_score if it.retrieval_score is not None else it.lance_score
        if it.slide_number is not None:
            if overview_rank:
                score = _overview_related_score(it)
            scored.append(
                {
                    "slide_number": it.slide_number,
                    "start_sec": it.start_sec,
                    "score": score,
                    "label": f"슬라이드 {it.slide_number}",
                    "_order": order_by_slide.get(int(it.slide_number), 10**9),
                }
            )
        elif it.kind == "graphrag_entity":
            row = it.row or {}
            item_score = score if score is not None else _overview_related_score(it)
            for idx, slide_number in enumerate(row.get("slide_numbers") or []):
                try:
                    sn = int(slide_number)
                except (TypeError, ValueError):
                    continue
                if visual_slides and sn not in visual_slides:
                    continue
                start_sec = None
                scene_starts = row.get("scene_start_secs") or []
                if idx < len(scene_starts):
                    start_sec = scene_starts[idx]
                scored.append(
                    {
                        "slide_number": sn,
                        "start_sec": start_sec,
                        "score": item_score,
                        "label": f"슬라이드 {sn}",
                        "_order": order_by_slide.get(sn, 10**9),
                    }
                )

    for c in chunks:
        try:
            chunk_slide_number = int(c.slide_number) if c.slide_number is not None else None
        except (TypeError, ValueError):
            chunk_slide_number = None
        if visual_slides and chunk_slide_number not in visual_slides:
            continue
        if c.slide_number is not None and c.chunk_type == "slide":
            score = c.score
            if overview_rank and chunk_slide_number is not None:
                score = float("-inf")
            scored.append(
                {
                    "slide_number": c.slide_number,
                    "start_sec": c.start_sec,
                    "score": score,
                    "label": f"슬라이드 {c.slide_number}",
                    "_order": order_by_slide.get(int(c.slide_number), 10**9),
                }
            )

    if preserve_item_order:
        def _order_value(row: dict) -> int:
            value = row.get("_order")
            return int(value) if value is not None else 10**9

        scored.sort(
            key=lambda r: (
                _order_value(r),
                int(r.get("slide_number") or 10**9),
            )
        )
    else:
        scored.sort(
            key=lambda r: (
                -(float(r.get("score")) if r.get("score") is not None else float("-inf")),
                float(r.get("start_sec")) if r.get("start_sec") is not None else float("inf"),
                int(r.get("slide_number") or 10**9),
            )
        )
    out: list[dict] = []
    seen: set[int] = set()
    for row in scored:
        slide_number = int(row["slide_number"])
        if slide_number in seen:
            continue
        seen.add(slide_number)
        row.pop("_order", None)
        out.append(row)
        if len(out) >= max_items:
            break
    return out


def _topic_keywords_for_related(question: str) -> list[str]:
    instruction_terms = {
        "요약",
        "요약해서",
        "정리",
        "설명",
        "설명해줘",
        "알려줘",
        "과정",
        "내용",
        "강의",
        "수업",
        "특징",
        "종류",
        "개념",
        "정의",
        "뭐야",
        "무엇",
    }
    out: list[str] = []
    compact_q = question.replace(" ", "").lower()
    for kw in extract_keywords_from_question(question):
        k = kw.strip().lower()
        if not k or k in instruction_terms:
            continue
        if k.startswith("실행"):
            k = "실행"
        elif k.startswith("부팅"):
            k = "부팅"
        elif k.startswith("적재"):
            k = "적재"
        if k not in out:
            out.append(k)
    if "응용" in compact_q and "소프트웨어" in compact_q:
        for k in ("응용소프트웨어", "응용 소프트웨어"):
            if k not in out:
                out.append(k)
    if _is_comparison_question(question) and "운영체제" in compact_q and "운영체제" not in out:
        out.insert(0, "운영체제")
    return out[:8]


def _definition_focus_terms(question: str) -> list[str]:
    if not _is_definition_question(question):
        return []
    terms = _topic_keywords_for_related(question)
    q = question.replace(" ", "")
    if "운영체제" in q and "운영체제" not in terms:
        terms.insert(0, "운영체제")
    return terms[:6]


def _definition_signal_hits(text: str) -> int:
    signals = (
        "정의",
        "개념",
        "시스템소프트웨어",
        "소프트웨어",
        "중개",
        "자원관리",
        "자원을관리",
        "관리하고제어",
        "관리",
        "제어",
        "독점",
        "배타",
        "메모리",
        "부팅",
    )
    compact = re.sub(r"\s+", "", (text or "").lower())
    return sum(1 for s in signals if s in compact)


def _strong_definition_signal_hits(text: str) -> int:
    signals = (
        "정의",
        "시스템소프트웨어",
        "중개",
        "핵심단어",
        "실체가있는소프트웨어",
        "컴퓨터가아닙니다",
        "소프트웨어입니다",
    )
    compact = re.sub(r"\s+", "", (text or "").lower())
    return sum(1 for s in signals if s in compact)


def _filter_chunks_to_topic(question: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    if _is_location_question(question) or _is_lecture_overview_question(question):
        return chunks
    if _is_core_content_question(question) or _is_emphasis_overview_question(question) or _is_visual_question(question):
        return chunks

    non_intro = [c for c in chunks if not _is_title_or_intro_text(c.text or "")]
    if non_intro:
        chunks = non_intro

    definition_terms = _definition_focus_terms(question)
    if definition_terms:
        topic_matched = [c for c in chunks if _chunk_keyword_hits(c.text or "", definition_terms) > 0]
        if topic_matched:
            strong_signal = [c for c in topic_matched if _strong_definition_signal_hits(c.text or "") > 0]
            if strong_signal:
                return strong_signal
            with_signal = [c for c in topic_matched if _definition_signal_hits(c.text or "") > 0]
            return with_signal or topic_matched

    keywords = _topic_keywords_for_related(question)
    if not keywords:
        return chunks

    filtered = [c for c in chunks if _chunk_keyword_hits(c.text or "", keywords) > 0]
    if _is_comparison_question(question) and filtered:
        max_hits = max(_chunk_keyword_hits(c.text or "", keywords) for c in filtered)
        if max_hits >= 2:
            focused = [c for c in filtered if _chunk_keyword_hits(c.text or "", keywords) >= 2]
            if focused:
                return focused
    return filtered or chunks


def _chunk_keyword_hits(text: str, keywords: list[str]) -> int:
    if not keywords:
        return 0
    compact = re.sub(r"\s+", "", (text or "").lower())
    hits = 0
    for kw in keywords:
        k = kw.lower()
        if k in compact or k in (text or "").lower():
            hits += 1
    return hits


def _related_slides_from_chunks_for_topic(
    question: str,
    chunks: list[RetrievedChunk],
    max_items: int = 4,
) -> list[dict]:
    keywords = _topic_keywords_for_related(question)
    candidates: list[dict] = []
    for c in chunks:
        if c.slide_number is None:
            continue
        try:
            slide_number = int(c.slide_number)
        except (TypeError, ValueError):
            continue
        text = c.text or ""
        hits = _chunk_keyword_hits(text, keywords)
        base = _to_float_or_none(c.score) or 0.0
        type_bonus = 0.0
        if c.chunk_type in {"segment", "audio", "structural_row"}:
            type_bonus = 0.18
        elif c.chunk_type in {"slide", "slide_text", "slide_concept"}:
            type_bonus = 0.08
        score = base + 0.28 * hits + type_bonus
        candidates.append(
            {
                "slide_number": slide_number,
                "start_sec": c.start_sec,
                "score": score,
                "label": f"슬라이드 {slide_number}",
                "_hits": hits,
            }
        )

    if not candidates:
        return []

    max_hits = max(int(r.get("_hits") or 0) for r in candidates)
    if keywords and max_hits > 0:
        candidates = [r for r in candidates if int(r.get("_hits") or 0) > 0]
    if _is_comparison_question(question) and max_hits >= 2:
        focused = [r for r in candidates if int(r.get("_hits") or 0) >= 2]
        if focused:
            candidates = focused

    candidates.sort(
        key=lambda r: (
            -(float(r.get("score")) if r.get("score") is not None else float("-inf")),
            float(r.get("start_sec")) if r.get("start_sec") is not None else float("inf"),
            int(r.get("slide_number") or 10**9),
        )
    )

    out: list[dict] = []
    seen: set[int] = set()
    for row in candidates:
        slide_number = int(row["slide_number"])
        if slide_number in seen:
            continue
        seen.add(slide_number)
        row.pop("_hits", None)
        out.append(row)
        if len(out) >= max_items:
            break
    return out


def _compact_visual_location_answer(answer: str) -> str:
    lines = [ln.strip() for ln in answer.splitlines()]
    kept: list[str] = []
    for ln in lines:
        if not ln:
            if kept and kept[-1]:
                kept.append("")
            continue
        line = re.sub(r"^\d+\.\s*", "- ", ln)
        line = re.sub(r"^[*-]\s*", "- ", line)
        line = re.sub(r"^-\s*(?:장면|씬|Scene)\s*\d+\s*[:：]\s*", "- ", line, flags=re.IGNORECASE)
        line = re.sub(r"^-\s*슬라이드\s*\d+\s*[:：]\s*", "- ", line)
        line = re.sub(r"^(?:장면|씬|Scene)\s*\d+\s*[:：]\s*", "", line, flags=re.IGNORECASE)
        line = re.sub(r"^슬라이드\s*\d+\s*[:：]\s*", "", line)
        kept.append(line)

    text = "\n".join(kept).strip()
    text = re.sub(r"\b(?:장면|씬|Scene)\s*\d+\b", "해당 장면", text, flags=re.IGNORECASE)
    text = re.sub(r"\b슬라이드\s*\d+\b", "해당 슬라이드", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def _visual_asset_label(asset_type: str) -> str:
    t = (asset_type or "").lower()
    if t == "diagram":
        return "다이어그램"
    if t == "table":
        return "표"
    if t == "chart":
        return "차트"
    if t in {"figure", "image"}:
        return "그림"
    if t == "list":
        return "목록"
    return "시각자료"


def _short_visual_description(text: str) -> str:
    body = re.sub(r"^시각자료\([^)]*\)\s*-\s*슬라이드\s*\d+\s*[^\n]*\n?", "", text or "").strip()
    body = re.sub(r"^이\s*슬라이드는\s*", "", body)
    body = re.sub(r"\s+", " ", body)
    if not body:
        return ""
    m = re.search(r"(.+?(?:다\.|요\.|[.!?。]))(?:\s+|$)", body)
    first = (m.group(1) if m else body).strip()
    return _first_sentence(first, max_chars=72)


def _visual_list_answer(items: list[EvidenceItem]) -> str:
    visual_items = [it for it in items if it.kind == "visual_asset" and it.slide_number is not None]
    if not visual_items:
        return ""

    asset_label = _visual_asset_label(str((visual_items[0].row or {}).get("asset_type", "")))
    lines = [f"이 강의에서 {asset_label}은 총 {len(visual_items)}곳에 나옵니다.", ""]
    for it in visual_items:
        row = it.row or {}
        title = str(row.get("title") or "").strip()
        desc = _short_visual_description(it.text)
        label = f"슬라이드 {it.slide_number}"
        if title:
            label += f": {title}"
        if desc:
            label += f" - {desc}"
        lines.append(f"- {label}")
    return "\n".join(lines).strip()


def _slides_explicitly_mentioned_in_answer(answer: str) -> set[int]:
    slides: set[int] = set()
    for m in re.finditer(r"슬라이드\s*(\d+)", answer or ""):
        try:
            slides.add(int(m.group(1)))
        except (TypeError, ValueError):
            continue
    return slides


def _filter_related_slides_to_answer(
    related_slides: list[dict],
    answer: str,
    question: str,
) -> list[dict]:
    if not related_slides or not _is_visual_question(question):
        return related_slides
    if _is_visual_interpretation_question(question):
        return related_slides
    mentioned = _slides_explicitly_mentioned_in_answer(answer)
    if len(mentioned) != 1:
        return related_slides
    target = next(iter(mentioned))
    filtered = [
        row for row in related_slides
        if int(row.get("slide_number") or -1) == target
    ]
    return filtered or related_slides


def _ensure_visual_asset_related_slides(
    related_slides: list[dict],
    source_items: list[EvidenceItem],
    question: str,
) -> list[dict]:
    if not _is_visual_question(question):
        return related_slides
    out = list(related_slides or [])
    seen = {
        int(row["slide_number"]) for row in out
        if row.get("slide_number") is not None and _to_int_or_none(row.get("slide_number")) is not None
    }
    visual_rows: list[dict] = []
    for idx, it in enumerate(source_items):
        if it.kind != "visual_asset" or it.slide_number is None:
            continue
        try:
            slide_number = int(it.slide_number)
        except (TypeError, ValueError):
            continue
        if slide_number in seen:
            continue
        seen.add(slide_number)
        visual_rows.append(
            {
                "slide_number": slide_number,
                "start_sec": it.start_sec,
                "score": it.retrieval_score if it.retrieval_score is not None else it.lance_score,
                "label": f"슬라이드 {slide_number}",
                "_visual_order": idx,
            }
        )
    if not visual_rows:
        return related_slides
    visual_rows.sort(key=lambda row: (row.get("_visual_order", 10**9), int(row.get("slide_number") or 10**9)))
    merged = out + visual_rows
    for row in merged:
        row.pop("_visual_order", None)
    return merged[:6]


def _align_chunks_to_related_slides(
    chunks: list[RetrievedChunk],
    related_slides: list[dict],
    question: str,
) -> list[RetrievedChunk]:
    if not chunks or not related_slides:
        return chunks
    slide_numbers = {
        int(row["slide_number"]) for row in related_slides
        if row.get("slide_number") is not None and _to_int_or_none(row.get("slide_number")) is not None
    }
    if not slide_numbers:
        return chunks

    def chunk_slide(c: RetrievedChunk) -> Optional[int]:
        try:
            return int(c.slide_number) if c.slide_number is not None else None
        except (TypeError, ValueError):
            return None

    aligned = [c for c in chunks if chunk_slide(c) in slide_numbers]
    if not aligned:
        return chunks
    if _is_comparison_question(question) or len(slide_numbers) <= 3:
        return aligned
    return chunks


def _prune_related_slides(
    related_slides: list[dict],
    chunks: list[RetrievedChunk],
    question: str,
    max_items: int = 2,
) -> list[dict]:
    if not related_slides:
        return related_slides
    if _is_lecture_overview_question(question) or _is_visual_list_question(question):
        return related_slides

    keywords = _topic_keywords_for_related(question)
    chunk_by_slide: dict[int, list[RetrievedChunk]] = {}
    for c in chunks:
        if c.slide_number is None:
            continue
        try:
            sn = int(c.slide_number)
        except (TypeError, ValueError):
            continue
        chunk_by_slide.setdefault(sn, []).append(c)

    scored: list[tuple[float, dict]] = []
    for idx, row in enumerate(related_slides):
        try:
            sn = int(row.get("slide_number"))
        except (TypeError, ValueError):
            continue
        row_chunks = chunk_by_slide.get(sn, [])
        chunk_text = " ".join(c.text or "" for c in row_chunks)
        if _is_title_or_intro_text(chunk_text or str(row.get("label") or "")):
            continue
        keyword_hits = _chunk_keyword_hits(chunk_text, keywords)
        chunk_score = max((_to_float_or_none(c.score) or 0.0) for c in row_chunks) if row_chunks else 0.0
        base = _to_float_or_none(row.get("score")) or 0.0
        has_timed_chunk = any(c.start_sec is not None for c in row_chunks)
        score = base + chunk_score + 0.45 * keyword_hits + (0.35 if has_timed_chunk else 0.0) - 0.03 * idx
        scored.append((score, row))

    if not scored:
        return related_slides[:max_items]

    if keywords:
        max_hits = max(
            _chunk_keyword_hits(" ".join(c.text or "" for c in chunk_by_slide.get(int(row.get("slide_number")), [])), keywords)
            for _, row in scored
            if row.get("slide_number") is not None
        )
        if max_hits > 0:
            focused = [
                (score, row) for score, row in scored
                if _chunk_keyword_hits(
                    " ".join(c.text or "" for c in chunk_by_slide.get(int(row.get("slide_number")), [])),
                    keywords,
                ) > 0
            ]
            if focused:
                scored = focused

    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [row for _, row in scored[:max_items]]


def _extract_keywords(question: str) -> list[str]:
    parts = re.split(r"[^0-9A-Za-z가-힣_]+", (question or "").lower())
    out: list[str] = []
    for p in parts:
        if len(p) < 2:
            continue
        if p not in out:
            out.append(p)
    return out[:8]


def _select_lance_supporting_chunks(
    chunks: list[RetrievedChunk], allowed_node_ids: set[str], max_items: int = 4
) -> list[RetrievedChunk]:
    if not chunks or not allowed_node_ids:
        return []
    selected: list[RetrievedChunk] = []
    for c in chunks:
        linked = (c.linked_node_id or "").strip()
        if linked and linked in allowed_node_ids:
            selected.append(c)
            if len(selected) >= max_items:
                break
    return selected


def _build_hybrid_context_base(graph_context: str, supporting_chunks: list[RetrievedChunk]) -> str:
    lines = [graph_context]
    if supporting_chunks:
        lines.extend(["", "[Lance 보강 텍스트]"])
        for i, c in enumerate(supporting_chunks, start=1):
            meta = f"(슬라이드 {c.slide_number}" if c.slide_number is not None else "(출처"
            if c.start_sec is not None:
                meta += f", {c.start_sec:.1f}초"
            meta += ")"
            lines.append(f"{i}. {meta} {c.text[:500]}")
        lines.append("Lance 텍스트는 그래프 근거를 보강하는 범위에서만 사용한다.")
    return "\n".join(lines)


def _augment_answer_context_for_question(context: str, question: str) -> str:
    if not _is_visual_interpretation_question(question):
        return context
    return (
        context
        + "\n\n[답변 지시]\n"
        + "- 이 질문은 시각자료의 위치를 묻는 것이 아니라, 그림/화살표/표현이 의미하는 관계를 묻는 시각 해석 질문이다.\n"
        + "- 본문 첫 문장에 핵심 의미를 답하고, 이어서 근거에 있는 시각 관계를 바탕으로 2~4개 항목으로 설명한다.\n"
        + "- 위치 정보는 본문에 쓰지 말고 UI의 확인 위치 버튼으로만 제공한다.\n"
    )


def _neo4j_driver():
    if not NEO4J_URI or not NEO4J_USER:
        raise HTTPException(
            status_code=503,
            detail="Neo4j 설정이 없습니다. NEO4J_URI/NEO4J_USER/NEO4J_PASSWORD를 확인하세요.",
        )
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        driver.verify_connectivity()
        return driver
    except (ServiceUnavailable, Neo4jError) as e:
        raise HTTPException(status_code=503, detail=f"Neo4j 연결 실패: {e}") from e


def graph_search_by_stem_question(stem: str, question: str, top_k: int = GRAPH_TOP_K) -> list[GraphEvidence]:
    keywords = _extract_keywords(question)
    if not keywords:
        return []

    driver = _neo4j_driver()
    try:
        with driver.session() as session:
            rows = session.run(
                """
                MATCH (a {stem: $stem})-[r]->(b {stem: $stem})
                WITH a, r, b,
                     toLower(
                        coalesce(a.id, '') + ' ' +
                        coalesce(a.title, '') + ' ' +
                        coalesce(a.text, '') + ' ' +
                        coalesce(a.name, '') + ' ' +
                        coalesce(b.id, '') + ' ' +
                        coalesce(b.title, '') + ' ' +
                        coalesce(b.text, '') + ' ' +
                        coalesce(b.name, '')
                     ) AS haystack
                WITH a, r, b, haystack, [kw IN $keywords WHERE haystack CONTAINS kw] AS hits
                WHERE size(hits) > 0
                RETURN
                    coalesce(a.id, '') AS src_id,
                    labels(a) AS src_labels,
                    type(r) AS rel_type,
                    coalesce(b.id, '') AS tgt_id,
                    labels(b) AS tgt_labels,
                    coalesce(a.text, a.title, a.name, '') AS src_text,
                    coalesce(b.text, b.title, b.name, '') AS tgt_text,
                    size(hits) AS score
                ORDER BY score DESC, rel_type ASC
                LIMIT $top_k
                """,
                stem=stem,
                keywords=keywords,
                top_k=int(top_k),
            )
            out: list[GraphEvidence] = []
            for row in rows:
                out.append(
                    GraphEvidence(
                        src_id=str(row.get("src_id", "")),
                        src_labels=[str(x) for x in (row.get("src_labels") or [])],
                        rel_type=str(row.get("rel_type", "")),
                        tgt_id=str(row.get("tgt_id", "")),
                        tgt_labels=[str(x) for x in (row.get("tgt_labels") or [])],
                        src_text=str(row.get("src_text", "") or "")[:800],
                        tgt_text=str(row.get("tgt_text", "") or "")[:800],
                    )
                )
            return out
    finally:
        driver.close()


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/internal/query_graph_evidence", response_model=GraphEvidenceResponse)
async def internal_query_graph_evidence(req: InternalGraphQueryRequest) -> GraphEvidenceResponse:
    stem = req.stem.strip()
    question = req.question.strip()
    if not stem or not question:
        raise HTTPException(status_code=400, detail="stem/question은 비어 있을 수 없습니다.")

    evidence = graph_search_by_stem_question(stem=stem, question=question, top_k=req.top_k)
    return GraphEvidenceResponse(
        stem=stem,
        question=question,
        keywords=_extract_keywords(question),
        evidence=evidence,
        count=len(evidence),
    )


def _empty_query_response() -> QueryResponse:
    return QueryResponse(
        answer="제공된 그래프 근거만으로는 답변하기 어렵습니다. 질문을 더 구체화하거나 강의 그래프 생성/적재 상태를 확인해주세요.",
        timestamps=[],
        graph={"nodes": [], "edges": []},
        core_graph={"nodes": [], "edges": []},
        retrieved_chunks=[],
        related_slides=[],
        source_mode="default",
    )


@app.post("/internal/query", response_model=QueryResponse)
async def internal_query(req: InternalQueryRequest) -> QueryResponse:
    stem = req.stem.strip()
    raw_question = req.question.strip()
    question = _resolve_followup_question(raw_question, req.conversation_history)
    if not stem or not raw_question:
        raise HTTPException(status_code=400, detail="stem/question은 비어 있을 수 없습니다.")

    driver = _neo4j_driver()
    q_type = classify_question(question)
    graph_context = ""
    allowed_ids: set[str] = set()
    timestamps: list[dict] = []
    graph: dict = {"nodes": [], "edges": []}
    selected_items: list[EvidenceItem] = []
    raw_rows: list[dict[str, Any]] = []
    intent_weights_for_context: dict[str, float] = {"general": 1.0}

    try:
        with driver.session() as session:
            if q_type == "content":
                graph_context, intent_weights_for_context, allowed_ids, structured, selected_items = run_enhanced_content_pipeline(
                    session,
                    stem,
                    question,
                    extract_keywords_from_question,
                    _call_gemini_raw,
                    current_slide_number=req.current_slide_number,
                )
                if not graph_context.strip():
                    return _empty_query_response()
                timestamps = _timestamps_from_content(structured)
                graph = _graph_from_content_structured(structured)
            else:
                try:
                    iw, _ = infer_intents_json(question, _call_gemini_raw)
                    intent_hint = ", ".join(
                        f"{k}:{v:.2f}" for k, v in sorted(iw.items(), key=lambda x: -x[1]) if v >= 0.04
                    )
                except Exception:
                    intent_hint = ""
                try:
                    raw_rows, cypher_used = run_cypher_safe_structural(
                        session, question, stem, intent_hint=intent_hint
                    )
                except Exception:
                    return _empty_query_response()
                if not raw_rows:
                    return _empty_query_response()
                graph_context = _build_structural_context(question, raw_rows, cypher_used)
                allowed_ids = _collect_ids_from_raw_rows(raw_rows)
                timestamps = _timestamps_from_structural_rows(raw_rows)
                graph = _graph_from_structural_rows(raw_rows)
    finally:
        driver.close()

    retrieved_chunks: list[RetrievedChunk] = []
    supporting_chunks: list[RetrievedChunk] = []
    source_items = _source_items_for_question(question, selected_items)
    if q_type == "content" and source_items != selected_items:
        graph_context = build_sectioned_context(
            question,
            intent_weights_for_context,
            source_items,
            max_chars=900,
        )
    if q_type == "content":
        try:
            retrieved_chunks = _evidence_to_retrieved_chunks(stem, source_items)
            retrieved_chunks = _filter_location_question_chunks(question, source_items, retrieved_chunks)
            retrieved_chunks = _filter_visual_question_chunks(question, source_items, retrieved_chunks)
            retrieved_chunks = _filter_chunks_to_topic(question, retrieved_chunks)
        except Exception:
            retrieved_chunks = []
    else:
        retrieved_chunks = _structural_rows_to_retrieved_chunks(stem, raw_rows)
        try:
            lance_root = default_lance_root()
            if lance_root.exists() and allowed_ids:
                df = lance_search(
                    stem=stem,
                    query=question,
                    lance_root=lance_root,
                    top_k=TOP_K,
                )
                if df is not None and len(df) > 0:
                    supporting_chunks = _select_lance_supporting_chunks(
                        _rows_to_chunks(df),
                        allowed_node_ids=allowed_ids,
                        max_items=4,
                    )
                    retrieved_chunks.extend(supporting_chunks)
        except Exception:
            supporting_chunks = []

    context = (
        graph_context
        if q_type == "content"
        else _build_hybrid_context_base(graph_context, supporting_chunks)
    )
    context = _augment_answer_context_for_question(context, question)
    source_mode = _source_mode_for(question, source_items)
    if _is_visual_list_question(question):
        answer = _visual_list_answer(source_items)
    else:
        try:
            answer = _call_gemini_answer(context, raw_question, req.conversation_history, resolved_question=question)
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"LLM 답변 생성 실패: {e}") from e
        answer = _compact_answer(answer, question, retrieved_chunks, source_mode=source_mode)
    if _is_refusal_answer(answer):
        retrieved_chunks = []
        timestamps = []
        graph = {"nodes": [], "edges": []}
    if (
        q_type == "content"
        and source_mode == "default"
        and not _is_location_question(question)
        and not _is_lecture_overview_question(question)
        and not _is_core_content_question(question)
        and not _is_emphasis_overview_question(question)
    ):
        related_slides = _related_slides_from_evidence(source_items, [], max_items=4, filter_to_visual_slides=False)
        if not related_slides:
            related_slides = _related_slides_from_chunks_for_topic(question, retrieved_chunks, max_items=4)
        elif _is_comparison_question(question):
            focused = _related_slides_from_chunks_for_topic(question, retrieved_chunks, max_items=4)
            focused_slides = {int(r.get("slide_number")) for r in focused if r.get("slide_number") is not None}
            if focused_slides:
                related_slides = [r for r in related_slides if int(r.get("slide_number") or -1) in focused_slides] or focused
    else:
        related_slides = _related_slides_from_evidence(
            source_items,
            retrieved_chunks,
            max_items=6 if (_is_lecture_overview_question(question) or _is_core_content_question(question) or _is_emphasis_overview_question(question)) else (3 if _is_location_question(question) else 6),
            preserve_item_order=_is_visual_list_question(question),
            filter_to_visual_slides=not (
                _is_lecture_overview_question(question)
                or _is_core_content_question(question)
                or _is_emphasis_overview_question(question)
            ) and _is_visual_question(question),
            overview_rank=_is_lecture_overview_question(question) or _is_core_content_question(question) or _is_emphasis_overview_question(question),
        )
    related_slides = _filter_related_slides_to_answer(related_slides, answer, question)
    related_slides = _ensure_visual_asset_related_slides(related_slides, source_items, question)
    retrieved_chunks = _align_chunks_to_related_slides(retrieved_chunks, related_slides, question)
    related_slides = _prune_related_slides(related_slides, retrieved_chunks, question)
    related_slides = _ensure_visual_asset_related_slides(related_slides, source_items, question)
    retrieved_chunks = _align_chunks_to_related_slides(retrieved_chunks, related_slides, question)
    if q_type == "content":
        retrieved_chunks = _hydrate_chunk_segment_labels(stem, retrieved_chunks, structured)
    core_source_items = list(source_items)
    if q_type == "content" and source_mode in {"visual_location", "scene_location", "overview"}:
        seen_uids = {it.uid for it in core_source_items}
        for it in selected_items:
            if it.uid in seen_uids or it.kind not in {"graphrag_entity", "graphrag_relationship"}:
                continue
            core_source_items.append(it)
            seen_uids.add(it.uid)
            if len([x for x in core_source_items if x.kind in {"graphrag_entity", "graphrag_relationship"}]) >= 8:
                break
    core_graph = (
        {"nodes": [], "edges": []}
        if _is_refusal_answer(answer)
        else _build_core_graph(
            stem=stem,
            graph=graph,
            source_items=core_source_items if q_type == "content" else [],
            source_mode=source_mode,
            related_slides=related_slides,
            retrieved_chunks=retrieved_chunks,
        )
    )

    return QueryResponse(
        answer=answer,
        timestamps=(
            []
            if source_mode in {"visual_location", "scene_location", "overview"} or _is_visual_question(question)
            else (_chunks_to_timestamps(retrieved_chunks) or ([] if _is_location_question(question) else timestamps))
        ),
        graph=graph,
        core_graph=core_graph,
        retrieved_chunks=retrieved_chunks,
        related_slides=related_slides,
        source_mode=source_mode,
    )
