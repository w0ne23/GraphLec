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
from pydantic import BaseModel, Field

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from pipeline.lance_ingest import default_lance_root, lance_search  # noqa: E402

from .content_retrieval import EvidenceItem, infer_intents_json, run_enhanced_content_pipeline  # noqa: E402
app = FastAPI(title="GraphLEC Query Service", version="0.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY_2") or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
GEMINI_ANSWER_MODEL = os.getenv("GEMINI_ANSWER_MODEL", "gemini-2.5-flash")
TOP_K = int(os.getenv("GRAPHLEC_TOP_K", "8"))
GRAPH_TOP_K = int(os.getenv("GRAPHLEC_GRAPH_TOP_K", "12"))
RETRY_DELAYS = [0, 5, 15, 30]
NEO4J_URI = os.getenv("NEO4J_URI", "").strip()
NEO4J_USER = os.getenv("NEO4J_USER", "").strip()
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")

GRAPH_SCHEMA = """
노드 타입과 주요 프로퍼티:
- Video          : id, stem, title
- Slide          : id, stem, slide_number, title, slide_text
- Domain         : id, stem, name, subdomain
- Scene          : id, stem, source_slide_id, scene_number, slide_number, start_sec, end_sec, role, emphasis_total
- Context        : id, stem, slide_id, scene_id, context_index, start, end, stressed, text
- Segment        : id, stem, start, end, text, stressed
- AnnotationEmphasis : id, stem, type, target_content, score, confidence, timestamp_sec
- GraphRAGEntity : id, stem, title, type, description, degree, frequency
- GraphRAGTextUnit : id, stem, text, n_tokens
- GraphRAGCommunity : id, stem, title, summary, rank, size

관계 (방향 중요):
- (Video)-[:HAS_SCENE]->(Scene)
- (Video)-[:HAS_DOMAIN]->(Domain)
- (Video)-[:HAS_SLIDE]->(Slide)
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
6. 타임스탬프가 필요하면 Segment.start/end 또는 Scene.start_sec/end_sec을 활용한다. Slide에는 시간 정보가 없다.
"""

ANSWER_SYSTEM_PROMPT = """
너는 강의 영상 분석 지식 그래프(및 선택적 보강 텍스트)를 바탕으로 질문에 답하는 어시스턴트다.

[근거 규칙]
- 답변의 사실·관계·용어는 반드시 제공된 "근거" 안에 있을 때만 쓴다.
- 근거에 없으면 추측하지 말고 짧게 "근거에 없어 답할 수 없다"고 한다.
- 근거에 "[질문 의도(모델 추론)]"가 있으면 참고만 하되, 답의 내용은 반드시 그 아래 실제 인용 근거에만 기대어라.
- 의미 검색(보조)로 표시된 텍스트은 그래프 근거와 모순 없을 때만 보강에 사용한다.

[복합 질문]
- 질문이 정의와 예시·사례 등을 동시에 요구하면, 근거에서 가능한 범위로 각 요구를 모두 다룬다.
- 특정 요구에 해당하는 근거가 없으면 그 한 가지만 짧게 밝힌다.

[답변 스타일]
1. 질문에 직접 필요한 내용만 답한다. 근거가 많아도 전부 나열하지 말고 핵심만 추린다.
2. 기본 답변은 1문장 요약 + 최대 4개 항목으로 작성한다. 각 항목은 한 문장으로 짧게 쓴다.
3. 사용자가 "자세히", "구체적으로", "전부", "비교표"처럼 확장을 요청한 경우에만 더 길게 답한다.
4. 출처는 답변 끝에 한 번만 짧게 묶어 쓴다. 예: 출처: 슬라이드 34, 약 120초.
5. "(GraphRAG 개념)", "(GraphRAG 개념 관계)" 같은 내부 근거 종류명은 사용자에게 노출하지 않는다.
6. 질문에 사례·예시·예를 들어 등이 있으면 근거에서 1개 사례만 넣고, 없으면 없다고 말한다.
7. "정의:", "음성 발췌:" 같은 인위적 소제목 블록을 만들지 않는다.
8. 한국어로 답한다.
"""


class InternalQueryRequest(BaseModel):
    stem: str = Field(..., min_length=1)
    question: str = Field(..., min_length=1)


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


class QueryResponse(BaseModel):
    answer: str
    timestamps: list[dict]
    graph: dict
    retrieved_chunks: list[RetrievedChunk]


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
}

_CYPHER_FORBIDDEN = re.compile(
    r"\b(CREATE|MERGE|DELETE|DETACH|SET|REMOVE|LOAD\s+CSV|FOREACH|DROP|INSERT|CALL)\b",
    re.IGNORECASE,
)


def classify_question(question: str) -> str:
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
        if len(tok) >= 2 and tok not in STOPWORDS and tok not in keywords:
            keywords.append(tok)

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


def _call_gemini_raw(contents: str, system_instruction: str) -> str:
    client = _gemini_client()
    last_err: Optional[Exception] = None
    for delay in RETRY_DELAYS:
        if delay:
            time.sleep(delay)
        try:
            r = client.models.generate_content(
                model=GEMINI_ANSWER_MODEL,
                contents=contents,
                config={"system_instruction": system_instruction},
            )
            return (r.text or "").strip()
        except Exception as e:
            last_err = e
            err_s = str(e)
            if "429" not in err_s and "RESOURCE_EXHAUSTED" not in err_s:
                raise
    assert last_err is not None
    raise last_err


def _call_gemini_answer(context: str, question: str) -> str:
    client = _gemini_client()
    contents = f"질문: {question}\n\n근거:\n{context}"
    last_err: Optional[Exception] = None
    for delay in RETRY_DELAYS:
        if delay:
            time.sleep(delay)
        try:
            r = client.models.generate_content(
                model=GEMINI_ANSWER_MODEL,
                contents=contents,
                config={
                    "system_instruction": ANSWER_SYSTEM_PROMPT,
                    "temperature": 0.2,
                    "max_output_tokens": 2048,
                },
            )
            return (r.text or "").strip()
        except Exception as e:
            last_err = e
            err_s = str(e)
            if "429" not in err_s and "RESOURCE_EXHAUSTED" not in err_s:
                raise
    assert last_err is not None
    raise last_err


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
    labels: list[str] = []
    for c in chunks:
        if c.slide_number is not None:
            labels.append(f"슬라이드 {c.slide_number}")
        if c.start_sec is not None:
            labels.append(f"약 {c.start_sec:.0f}초")
    return _dedupe_preserve_order(labels)


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


def _compact_answer(answer: str, question: str, chunks: Optional[list[RetrievedChunk]] = None) -> str:
    """Clean presentation-only noise without truncating or summarizing model content."""
    raw = answer.strip()
    if not raw:
        return raw

    sources = _source_labels_from_chunks(chunks or [])
    if not sources:
        sources = _dedupe_preserve_order(re.findall(r"슬라이드\s*\d+|약\s*\d+(?:\.\d+)?초", raw))
    cleaned = _clean_answer_text(raw)
    cleaned = cleaned.replace("**", "")

    cleaned = re.sub(r"[ \t]+[*-]\s+([^:：\n]{1,40})[:：]\s*", r"\n- \1: ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r"\s*출처\s*:.*$", "", cleaned, flags=re.S).strip()
    cleaned = re.sub(r"\n*\s*출처\s*\n(?:\s*[-*].*(?:\n|$))+\s*$", "", cleaned).strip()

    source_block = _format_source_block(sources)
    if source_block:
        cleaned += f"\n\n{source_block}"
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
            )
        )
    return out


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
        answer="그래프 근거가 없어 답변할 수 없습니다. 질문을 더 구체화하거나 강의 그래프 생성/적재 상태를 확인해주세요.",
        timestamps=[],
        graph={"nodes": [], "edges": []},
        retrieved_chunks=[],
    )


@app.post("/internal/query", response_model=QueryResponse)
async def internal_query(req: InternalQueryRequest) -> QueryResponse:
    stem = req.stem.strip()
    question = req.question.strip()
    if not stem or not question:
        raise HTTPException(status_code=400, detail="stem/question은 비어 있을 수 없습니다.")

    driver = _neo4j_driver()
    q_type = classify_question(question)
    graph_context = ""
    allowed_ids: set[str] = set()
    timestamps: list[dict] = []
    graph: dict = {"nodes": [], "edges": []}
    selected_items: list[EvidenceItem] = []
    raw_rows: list[dict[str, Any]] = []

    try:
        with driver.session() as session:
            if q_type == "content":
                graph_context, _intent_w, allowed_ids, structured, selected_items = run_enhanced_content_pipeline(
                    session,
                    stem,
                    question,
                    extract_keywords_from_question,
                    _call_gemini_raw,
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
    if q_type == "content":
        try:
            retrieved_chunks = _evidence_to_retrieved_chunks(stem, selected_items)
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
    try:
        answer = _call_gemini_answer(context, question)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Gemini 답변 생성 실패: {e}") from e
    answer = _compact_answer(answer, question, retrieved_chunks)

    return QueryResponse(
        answer=answer,
        timestamps=_chunks_to_timestamps(retrieved_chunks) or timestamps,
        graph=graph,
        retrieved_chunks=retrieved_chunks,
    )
