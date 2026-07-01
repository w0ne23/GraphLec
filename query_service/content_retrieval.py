"""
내용형 질의: 의도 추론(LLM JSON) → Neo4j 광범위 조회 → 임베딩 재순위 → MMR → Lance 2-pass → 섹션 근거 문자열.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from pipeline.embedding_utils import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_PROVIDER,
    embed_documents,
    embed_query,
    get_embedding_client,
)
from pipeline.lance_ingest import default_lance_root, lance_search

from .neo4j_content_queries import run_content_queries, run_overview_queries


def _load_cfg() -> dict[str, Any]:
    p = Path(__file__).resolve().parent / "intent_config.json"
    if not p.is_file():
        return {}
    with open(p, encoding="utf-8") as f:
        return json.load(f)


_CFG = _load_cfg()


def _add_intent(weights: dict[str, float], name: str, weight: float = 1.0) -> None:
    weights[name] = max(float(weight), weights.get(name, 0.0))


def _renormalize_intents(weights: dict[str, float]) -> dict[str, float]:
    weights = {k: float(v) for k, v in weights.items() if v and float(v) > 0}
    if not weights:
        return {"general": 1.0}
    total = sum(weights.values())
    return {k: v / total for k, v in weights.items()}


def _apply_rule_intent_overrides(
    question: str,
    weights: dict[str, float],
    hints: list[Any],
) -> tuple[dict[str, float], list[Any]]:
    """LLM intent를 기본으로 쓰되, 고신뢰 표현은 GraphLec 전용 intent로 보정한다."""
    q = question.replace(" ", "")
    q_lower = question.lower()
    out = dict(weights)

    overview_terms = (
        "강의내용요약",
        "내용요약",
        "전체요약",
        "전체내용",
        "전체흐름",
        "내용정리",
        "강의정리",
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
    if _has_lecture_overview_scope(question) or any(t in q for t in overview_terms):
        _add_intent(out, "lecture_overview", 1.0)
    elif out.get("lecture_overview", 0) > 0:
        out.pop("lecture_overview", None)

    if any(t in q for t in ("핵심키워드", "중요키워드", "주요키워드", "핵심개념", "주요개념", "핵심용어")):
        _add_intent(out, "core_concepts", 1.0)

    if _is_exam_prep_query(question):
        _add_intent(out, "core_concepts", 1.0)

    if "강조" in q or any(t in q for t in ("중요하게다룬", "중요하게말한", "교수가중요", "강의자가중요")):
        _add_intent(out, "emphasis_overview", 0.9)

    if any(t in q_lower for t in ("시각자료", "그림", "이미지", "표", "도표", "비교표", "다이어그램", "구조도", "화살표", "양방향", "table", "diagram", "arrow")):
        _add_intent(out, "visual_location", 0.95)

    if any(t in q for t in ("장면", "구간", "어디", "언제", "몇초", "몇분", "씬")):
        _add_intent(out, "scene_location", 0.7)

    if "슬라이드" in q and any(t in q for t in ("중요", "핵심", "강조", "비중", "순위", "랭킹")):
        _add_intent(out, "slide_importance", 0.95)

    if out.keys() != weights.keys():
        hints = list(hints or [])
        if out.get("lecture_overview", 0) > 0:
            for h in ("운영체제", "강의 목표", "핵심 개념", "주요 내용"):
                if h not in hints:
                    hints.append(h)
        if out.get("core_concepts", 0) > 0 and _is_exam_prep_query(question):
            for h in ("운영체제 정의", "운영체제 목적", "운영체제 기능", "응용 소프트웨어 차이"):
                if h not in hints:
                    hints.append(h)
    return _renormalize_intents(out), hints


INTENT_SYSTEM_PROMPT = """
너는 강의 질의응답용 검색 의도 분류기다. 사용자 질문을 보고 JSON만 출력한다.

출력 형식(반드시 이 키만):
{
  "intents": [
    {"name": "definition", "weight": 0.0},
    ...
  ],
  "keywords_hint": ["추가로 검색에 쓸 한국어 키워드", "..."]
}

name은 반드시 아래 중에서만 고른다:
- lecture_overview: 강의 전체 요약·전체 흐름·무엇을 배웠는지·큰 그림
- core_concepts: 핵심 개념·핵심 키워드·주요 용어
- emphasis_overview: 교수/강의자가 강조한 내용·중요하게 다룬 부분
- visual_location: 표·그림·다이어그램·시각자료가 어디 있는지
- scene_location: 관련 장면·구간·시간 위치 찾기
- slide_importance: 중요한 슬라이드·핵심 슬라이드 순위
- definition: 정의·개념·무엇인지
- example: 예시·사례·예를 들어
- explanation: 이유·설명·왜·어떻게 동작
- comparison: 비교·차이·대조
- temporal: 시간·언제·몇 초·구간
- location: 슬라이드 번호·어디·장면
- general: 위에 해당하지 않거나 복합·일반

규칙:
- 질문이 여러 요구를 동시에 하면 intents를 여러 개 넣고 weight를 나눈다 (합은 대략 1.0).
- keywords_hint는 질문에 없지만 검색에 도움이 될 한국어 명사 위주로, 없으면 [].
- JSON 외 텍스트는 쓰지 않는다.
"""


def infer_intents_json(question: str, call_gemini_raw: Callable[[str, str], str]) -> tuple[dict[str, float], list[str]]:
    """LLM으로 의도 가중치 + 키워드 힌트. 실패 시 general=1.0."""
    try:
        raw = call_gemini_raw(
            f"사용자 질문:\n{question}\n",
            INTENT_SYSTEM_PROMPT,
        )
    except Exception:
        return _apply_rule_intent_overrides(question, {"general": 1.0}, [])
    m = re.search(r"\{[\s\S]*\}", raw)
    if not m:
        return _apply_rule_intent_overrides(question, {"general": 1.0}, [])
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return _apply_rule_intent_overrides(question, {"general": 1.0}, [])
    intents_in = obj.get("intents") or []
    weights: dict[str, float] = {}
    for it in intents_in:
        if not isinstance(it, dict):
            continue
        name = str(it.get("name", "")).strip()
        w = it.get("weight", 0.0)
        try:
            wf = float(w)
        except (TypeError, ValueError):
            continue
        if name and wf > 0:
            weights[name] = weights.get(name, 0.0) + wf
    if not weights:
        return _apply_rule_intent_overrides(question, {"general": 1.0}, obj.get("keywords_hint") or [])
    s = sum(weights.values())
    if s > 0:
        weights = {k: v / s for k, v in weights.items()}
    weights, hints = _apply_rule_intent_overrides(question, weights, obj.get("keywords_hint") or [])
    hints = hints or []
    hints = [str(h).strip() for h in hints if isinstance(h, str) and len(str(h).strip()) >= 2][:12]
    return weights, hints


@dataclass
class EvidenceItem:
    uid: str
    kind: str  # slide_text, slide_concept, segment, sub_concept, graphrag_*, lance_*
    text: str
    row: Optional[dict[str, Any]] = None
    lance_score: Optional[float] = None
    linked_node_id: Optional[str] = None
    chunk_type: str = ""
    slide_number: Optional[int] = None
    start_sec: Optional[float] = None
    end_sec: Optional[float] = None
    retrieval_score: Optional[float] = None
    score_breakdown: Optional[dict[str, float]] = None


def _to_float(v: Any) -> Optional[float]:
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _to_int(v: Any) -> Optional[int]:
    try:
        return int(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _structured_to_items(structured: dict[str, list[dict[str, Any]]]) -> list[EvidenceItem]:
    items: list[EvidenceItem] = []
    seen: set[str] = set()

    for r in structured.get("sub_concepts", []):
        sid, cid = str(r.get("sub_id", "")), str(r.get("concept_id", ""))
        sub = str(r.get("sub_concept", ""))
        par = str(r.get("parent_concept", ""))
        text = f"{sub} —(관계)→ {par}".strip()
        uid = f"sc:{sid}:{cid}"
        if uid in seen or not text:
            continue
        seen.add(uid)
        items.append(EvidenceItem(uid=uid, kind="sub_concept", text=text, row=r))

    for r in structured.get("segments", []):
        st = str(r.get("segment_text", ""))
        co = str(r.get("concept", "") or "")
        t0 = r.get("start")
        uid = f"seg:{r.get('segment_id','')}:{t0}"
        if uid in seen or not st:
            continue
        seen.add(uid)
        text = f"{st} (개념: {co})" if co else st
        items.append(
            EvidenceItem(
                uid=uid,
                kind="segment",
                text=text,
                row=r,
                chunk_type="segment",
                slide_number=_to_int(r.get("slide_number")),
                start_sec=_to_float(r.get("start")),
                end_sec=_to_float(r.get("end")),
            )
        )

    for r in structured.get("slides", []):
        sn = r.get("slide_number")
        body = str(r.get("slide_text", "") or "")
        structure = str(r.get("t1_structure") or r.get("visual_asset_text") or "")
        tit = str(r.get("title", "") or "")
        cid = str(r.get("concept_id", "") or "")
        emphasis_total = _row_float(r, "emphasis_total")
        title_relevance = int(r.get("title_relevance") or 0)
        relevance = int(r.get("relevance") or 0)
        meta = (
            f"슬라이드 강조 점수(emphasis_total): {emphasis_total:.3f}. "
            f"제목 일치: {title_relevance}, 검색 일치: {relevance}."
        )
        if cid:
            uid = f"slc:{r.get('slide_id')}:{cid}"
            if uid in seen:
                continue
            seen.add(uid)
            visual = f"\n시각자료 설명:\n{structure}" if structure.strip() else ""
            text = f"슬라이드 {sn} {tit}\n{meta}\n{body}{visual}".strip()
            if text:
                items.append(
                    EvidenceItem(
                        uid=uid,
                        kind="slide_concept",
                        text=text,
                        row=r,
                        chunk_type="slide",
                        slide_number=_to_int(sn),
                        start_sec=_to_float(r.get("start_sec")),
                        end_sec=_to_float(r.get("end_sec")),
                    )
                )
        else:
            uid = f"slt:{r.get('slide_id')}"
            if uid in seen:
                continue
            seen.add(uid)
            visual = f"\n시각자료 설명:\n{structure}" if structure.strip() else ""
            text = f"슬라이드 {sn} {tit}\n{meta}\n{body}{visual}".strip()
            if text:
                items.append(
                    EvidenceItem(
                        uid=uid,
                        kind="slide_text",
                        text=text,
                        row=r,
                        chunk_type="slide",
                        slide_number=_to_int(sn),
                        start_sec=_to_float(r.get("start_sec")),
                        end_sec=_to_float(r.get("end_sec")),
                    )
                )

    for r in structured.get("visual_assets", []):
        aid = str(r.get("visual_asset_id", ""))
        sn = r.get("slide_number")
        asset_type = str(r.get("asset_type", "") or "visual")
        title = str(r.get("title", "") or "")
        desc = str(r.get("description", "") or "")
        raw_text = str(r.get("raw_text", "") or "")
        elements = str(r.get("visual_elements_text", "") or "")
        relations = str(r.get("visual_relations_text", "") or "")
        layout = str(r.get("layout_text", "") or "")
        uid = f"vis:{aid or sn}:{asset_type}"
        if uid in seen or not (desc or raw_text or elements or relations or layout):
            continue
        seen.add(uid)
        body = "\n".join(
            part for part in [
                desc,
                raw_text,
                f"시각 요소:\n{elements}" if elements else "",
                f"시각 관계:\n{relations}" if relations else "",
                f"배치:\n{layout}" if layout else "",
            ]
            if part
        )
        text = f"시각자료({asset_type}) - 슬라이드 {sn} {title}\n{body}".strip()
        items.append(
            EvidenceItem(
                uid=uid,
                kind="visual_asset",
                text=text,
                row=r,
                chunk_type="slide",
                slide_number=_to_int(sn),
                start_sec=_to_float(r.get("start_sec")),
                end_sec=_to_float(r.get("end_sec")),
            )
        )

    for r in structured.get("graphrag_entities", []):
        eid = str(r.get("graphrag_entity_id", ""))
        title = str(r.get("graphrag_title", "") or "")
        desc = str(r.get("graphrag_description", "") or "")
        gtype = str(r.get("graphrag_type", "") or "")
        slides = [x for x in (r.get("slide_numbers") or []) if x not in (None, "")]
        concepts = [x for x in (r.get("concept_names") or []) if x]
        scene_ids = [str(x) for x in (r.get("scene_ids") or []) if x]
        uid = f"gre:{eid}"
        if uid in seen or not (title or desc):
            continue
        seen.add(uid)
        meta = []
        if gtype:
            meta.append(f"유형: {gtype}")
        boost = _to_float(r.get("emphasis_boost_local"))
        final_weight = _to_float(r.get("final_weight"))
        emphasis_sources = [str(x) for x in (r.get("emphasis_sources") or []) if x]
        emphasis_keywords = []
        for key in ("emphasis_matched_keywords", "emphasis_visual_keywords"):
            for keyword in r.get(key) or []:
                keyword = str(keyword).strip()
                if keyword and keyword not in emphasis_keywords:
                    emphasis_keywords.append(keyword)
        if boost and boost > 0:
            desc_parts = [f"boost {boost:.2f}"]
            if final_weight is not None:
                desc_parts.append(f"final_weight {final_weight:.2f}")
            if emphasis_sources:
                desc_parts.append("sources: " + ", ".join(emphasis_sources[:4]))
            meta.append("강조 신호: " + " / ".join(desc_parts))
        if emphasis_keywords:
            meta.append("강조 키워드: " + ", ".join(emphasis_keywords[:8]))
        if concepts:
            meta.append("기존 개념 연결: " + ", ".join(str(x) for x in concepts[:4]))
        if slides:
            meta.append("관련 슬라이드: " + ", ".join(str(x) for x in slides[:6]))
        text = f"{title}\n{desc}".strip()
        if meta:
            text += "\n" + " / ".join(meta)
        items.append(
            EvidenceItem(
                uid=uid,
                kind="graphrag_entity",
                text=text,
                row=r,
                chunk_type="graphrag_entity",
                linked_node_id=scene_ids[0] if scene_ids else eid,
            )
        )

    for r in structured.get("graphrag_relationships", []):
        sid, tid = str(r.get("src_id", "")), str(r.get("tgt_id", ""))
        src = str(r.get("src_title", "") or "")
        tgt = str(r.get("tgt_title", "") or "")
        desc = str(r.get("rel_description", "") or "")
        uid = f"grr:{sid}:{tid}:{desc[:40]}"
        if uid in seen or not (src or tgt or desc):
            continue
        seen.add(uid)
        text = f"{src} —(GraphRAG 관계)→ {tgt}"
        if desc:
            text += f"\n{desc}"
        edge_weight = _to_float(r.get("emphasis_edge_weight"))
        if edge_weight and edge_weight > _to_float(r.get("weight") or 0):
            text += f"\n강조 반영 관계 가중치: {edge_weight:.2f}"
        items.append(
            EvidenceItem(
                uid=uid,
                kind="graphrag_relationship",
                text=text,
                row=r,
                chunk_type="graphrag_relationship",
            )
        )

    return items


def _collect_ids(structured: dict[str, list[dict[str, Any]]]) -> set[str]:
    ids: set[str] = set()
    for key in ("sub_concepts", "segments", "slides", "visual_assets"):
        for r in structured.get(key, []):
            for fld in ("sub_id", "concept_id", "slide_id", "segment_id", "visual_asset_id"):
                v = r.get(fld)
                if v:
                    ids.add(str(v).strip())
    for r in structured.get("graphrag_entities", []):
        for fld in ("graphrag_entity_id",):
            v = r.get(fld)
            if v:
                ids.add(str(v).strip())
        for fld in ("concept_ids", "slide_ids", "scene_ids"):
            for v in r.get(fld) or []:
                if v:
                    ids.add(str(v).strip())
    for r in structured.get("graphrag_relationships", []):
        for fld in ("src_id", "tgt_id"):
            v = r.get(fld)
            if v:
                ids.add(str(v).strip())
        for fld in ("src_concept_ids", "tgt_concept_ids"):
            for v in r.get(fld) or []:
                if v:
                    ids.add(str(v).strip())
    return {x for x in ids if x}


def _intent_prior_for_kind(kind: str, intent_weights: dict[str, float], prior_map: dict[str, dict[str, float]]) -> float:
    acc = 0.0
    for intent_name, w in intent_weights.items():
        row = prior_map.get(intent_name) or prior_map.get("general", {})
        acc += w * float(row.get(kind, 0.7))
    return acc


def _kw_score(text: str, keywords: list[str]) -> float:
    if not keywords:
        return 0.0
    t = text.lower()
    hits = sum(1 for k in keywords if k.lower() in t)
    return min(1.0, hits / max(3, len(keywords) * 0.5))


def _is_slide_importance_query(question: str) -> bool:
    return "슬라이드" in question and any(k in question for k in ("중요", "핵심", "강조", "비중", "순위", "랭킹"))


def _is_exam_prep_query(question: str) -> bool:
    q = question.replace(" ", "")
    return any(k in q for k in ("시험", "출제", "시험대비", "나올것같", "나올만한"))


def _is_example_query(question: str, intent_weights: dict[str, float] | None = None) -> bool:
    if intent_weights and _intent_weight(intent_weights, "example") >= 0.25:
        return True
    q = question.replace(" ", "")
    return any(k in q for k in ("예시", "사례", "예를들", "예는", "예가", "예로"))


def _example_subject_terms(question: str, keywords: list[str]) -> list[str]:
    terms: list[str] = []
    m = re.search(r"(.+?)의\s*(?:예시|사례|예는|예가|예로)", question.strip())
    if m:
        subject = m.group(1).strip(" ?!.,，。")
        if subject:
            terms.append(subject)
    for kw in keywords:
        kw = str(kw or "").strip()
        if len(kw) >= 2 and kw not in {"예시", "사례"} and kw not in terms:
            terms.append(kw)
    return terms[:5]


def _example_directness_score(it: EvidenceItem, subject_terms: list[str]) -> float:
    raw = it.text or ""
    compact = re.sub(r"\s+", "", raw.lower())
    if not compact or not subject_terms:
        return 0.0
    has_subject = any(
        term.lower() in raw.lower() or term.lower().replace(" ", "") in compact
        for term in subject_terms
    )
    if not has_subject:
        return 0.0
    if any(marker in compact for marker in ("예시", "사례", "예를들", "예로", "대표적예")):
        return 1.0
    if "등" in compact:
        return 0.75
    if re.search(r"\([^)]*,[^)]*\)", raw):
        return 0.6
    return 0.0


def _is_overview_item(it: EvidenceItem) -> bool:
    text = (it.text or "").replace(" ", "").lower()
    return any(k in text for k in ("강의목표", "강의의목표", "학습목표", "목차", "개요", "chapter"))


def _is_visual_query(question: str) -> bool:
    return any(k in question for k in ("시각자료", "그림", "이미지", "표", "도표", "비교표", "다이어그램", "구조도", "화살표", "양방향"))


def _is_current_visual_query(question: str) -> bool:
    if _is_visual_list_query(question):
        return False
    return _is_visual_query(question) and any(
        k in question for k in ("이 ", "이것", "이거", "저 ", "저것", "저거", "현재", "지금", "보고 있는", "보고있는")
    )


def _is_visual_list_query(question: str) -> bool:
    if not _is_visual_query(question):
        return False
    q = question.replace(" ", "")
    return (
        any(k in q for k in ("이강의에서", "전체", "모두", "전부"))
        and any(k in q for k in ("나오는", "등장하는", "있는장면", "있는슬라이드", "장면들", "슬라이드들", "슬라이드만"))
    )


def _is_visual_interpretation_query(question: str) -> bool:
    if not _is_visual_query(question):
        return False
    q = question.replace(" ", "")
    return any(k in q for k in ("뭘의미", "무엇을의미", "의미해", "의미야", "왜", "이유", "설명해", "나타내", "말하는"))


def _is_emphasis_overview_query(question: str) -> bool:
    return "강조" in question and any(k in question for k in ("내용", "뭐", "무엇", "어떤", "핵심", "중요", "키워드", "개념"))


def _is_core_keyword_query(question: str) -> bool:
    return _is_exam_prep_query(question) or any(k in question for k in ("핵심 키워드", "중요 키워드", "주요 키워드", "핵심 개념", "주요 개념")) or (
        any(k in question for k in ("핵심", "중요", "주요"))
        and any(k in question for k in ("키워드", "개념", "용어", "내용"))
    )


def _intent_weight(intent_weights: dict[str, float], name: str) -> float:
    try:
        return float(intent_weights.get(name, 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _is_lecture_overview_query(question: str, intent_weights: dict[str, float] | None = None) -> bool:
    if intent_weights and _intent_weight(intent_weights, "lecture_overview") >= 0.25 and _has_lecture_overview_scope(question):
        return True
    if intent_weights and _intent_weight(intent_weights, "lecture_overview") >= 0.25:
        return False
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


def _row_float(row: dict[str, Any] | None, key: str, default: float = 0.0) -> float:
    if not row:
        return default
    try:
        return float(row.get(key) or default)
    except (TypeError, ValueError):
        return default


def _entity_core_score(row: dict[str, Any] | None) -> float:
    """GraphRAG 개념 노드의 강의 중심성 + 강조 신호를 하나의 비교 점수로 묶는다."""
    return (
        _row_float(row, "final_weight")
        + 0.75 * _row_float(row, "emphasis_boost_local")
        + 0.20 * _row_float(row, "frequency")
        + 0.10 * _row_float(row, "degree")
    )


def _select_lecture_overview_items(items: list[EvidenceItem]) -> list[EvidenceItem]:
    """전체 요약용 근거: 슬라이드 흐름을 보존하고, 핵심 개념/관계/강조/시각자료를 보강한다."""
    slides = [it for it in items if it.kind in {"slide_text", "slide_concept"} and it.slide_number is not None]
    slides.sort(key=lambda it: int(it.slide_number or 10**9))

    entities = [it for it in items if it.kind == "graphrag_entity"]
    entities.sort(key=lambda it: _entity_core_score(it.row), reverse=True)

    rels = [it for it in items if it.kind == "graphrag_relationship"]
    rels.sort(
        key=lambda it: (
            _row_float(it.row, "emphasis_edge_weight"),
            _row_float(it.row, "combined_degree"),
            _row_float(it.row, "weight"),
        ),
        reverse=True,
    )

    segments = [it for it in items if it.kind == "segment"]
    segments.sort(
        key=lambda it: (
            max(_row_float(it.row, "scene_emphasis_total"), _row_float(it.row, "slide_emphasis_total")),
            -(it.start_sec or 0.0),
        ),
        reverse=True,
    )

    visuals = [it for it in items if it.kind == "visual_asset"]
    visuals.sort(key=lambda it: int(it.slide_number or 10**9))

    selected: list[EvidenceItem] = []
    selected.extend(slides[:14])
    selected.extend(entities[:8])
    selected.extend(rels[:6])
    selected.extend(segments[:5])
    selected.extend(visuals[:5])

    seen: set[str] = set()
    out: list[EvidenceItem] = []
    for it in selected:
        if it.uid in seen:
            continue
        seen.add(it.uid)
        out.append(it)
    return out


def _cosine_mat(vectors: list[np.ndarray]) -> np.ndarray:
    """rows normalized, returns sim matrix len x len."""
    if not vectors:
        return np.array([])
    m = np.stack([v / (np.linalg.norm(v) + 1e-9) for v in vectors], axis=0)
    return m @ m.T


def _mmr(
    order_by_score: list[int],
    sim_to_q: np.ndarray,
    sim_all: np.ndarray,
    k: int,
    lambda_: float,
) -> list[int]:
    """order_by_score: indices sorted by pre-score desc. Returns selected indices."""
    if not order_by_score:
        return []
    if k <= 0:
        return []
    if len(order_by_score) == 1 or sim_all.size == 0:
        return order_by_score[:k]
    selected: list[int] = []
    pool = list(order_by_score)
    while len(selected) < k and pool:
        best_i = -1
        best_val = -1e9
        for i in pool:
            div = 0.0
            if selected:
                div = max(float(sim_all[i, j]) for j in selected)
            val = lambda_ * float(sim_to_q[i]) - (1.0 - lambda_) * div
            if val > best_val:
                best_val = val
                best_i = i
        selected.append(best_i)
        pool.remove(best_i)
    return selected


def _is_media_evidence(it: EvidenceItem) -> bool:
    return it.kind in {"segment", "slide_text", "slide_concept", "visual_asset"} or it.start_sec is not None


def _distance_to_score(dist: Optional[float]) -> float:
    if dist is None or (isinstance(dist, float) and math.isnan(dist)):
        return 0.0
    return 1.0 / (1.0 + float(dist))


def _df_to_lance_items(df, *, strict: bool, allowed_ids: set[str]) -> list[EvidenceItem]:
    out: list[EvidenceItem] = []
    if df is None or len(df) == 0:
        return out
    for _, row in df.iterrows():
        dist = row.get("_distance")
        try:
            d = float(dist) if dist is not None else None
        except (TypeError, ValueError):
            d = None
        text = str(row.get("text", ""))[:4000]
        linked = str(row.get("linked_node_id", "") or "").strip()
        kind = "lance_strict" if (linked and linked in allowed_ids) else "lance_soft"
        if strict and kind != "lance_strict":
            continue
        if not strict and kind != "lance_soft":
            continue
        sc = _distance_to_score(d) if d is not None else 0.5
        uid = f"lance:{row.get('chunk_id','')}"
        out.append(
            EvidenceItem(
                uid=uid,
                kind=kind,
                text=text,
                row=None,
                lance_score=sc,
                linked_node_id=linked or None,
                chunk_type=str(row.get("chunk_type", "") or ""),
                slide_number=row.get("slide_number"),
                start_sec=float(row["start_sec"]) if row.get("start_sec") is not None else None,
                end_sec=float(row["end_sec"]) if row.get("end_sec") is not None else None,
            )
        )
    return out


def _lance_two_pass(
    stem: str,
    question: str,
    lance_root: Path,
    allowed_ids: set[str],
    top1: int,
    top2: int,
    min_strict: int,
) -> tuple[list[EvidenceItem], list[EvidenceItem]]:
    """strict: linked in allowed_ids. soft: all stem hits, for gap filling."""
    strict_items: list[EvidenceItem] = []
    soft_items: list[EvidenceItem] = []
    try:
        df1 = lance_search(stem=stem, query=question, lance_root=lance_root, top_k=top1)
        strict_items = _df_to_lance_items(df1, strict=True, allowed_ids=allowed_ids)
        if len(strict_items) < min_strict:
            df2 = lance_search(stem=stem, query=question, lance_root=lance_root, top_k=top2)
            soft_items = _df_to_lance_items(df2, strict=False, allowed_ids=allowed_ids)
            # dedupe by chunk id
            seen = {i.uid for i in strict_items}
            soft_items = [x for x in soft_items if x.uid not in seen]
    except Exception:
        pass
    return strict_items, soft_items


def build_sectioned_context(
    question: str,
    intent_weights: dict[str, float],
    items: list[EvidenceItem],
    max_chars: int,
) -> str:
    lines = [
        f"질문: {question}",
        "",
        "[질문 의도(모델 추론)]",
        ", ".join(f"{k}: {v:.2f}" for k, v in sorted(intent_weights.items(), key=lambda x: -x[1]) if v >= 0.03),
        "",
        "[검색·재순위로 선택된 근거]",
        "아래 내용만 사실로 사용한다. 서로 다른 출처를 골랐다.",
    ]
    for context in build_prompt_contexts(intent_weights, items, max_chars):
        lines.append(context)
        lines.append("")
    if _intent_weight(intent_weights, "lecture_overview") >= 0.25:
        lines.append(
            "강의 전체 요약 질문이다. 슬라이드 순서를 중심으로 전체 흐름을 요약하고, 핵심 개념과 중요한 시각자료/강조 근거는 보조로만 사용한다. "
            "답변은 한 문장 요약, 강의 흐름 3~5개, 핵심 개념 3~5개로 간결하게 작성한다."
        )
    else:
        lines.append(
            "질문에 정의·예시·설명 등 여러 요구가 섞여 있으면, 위 근거에서 가능한 범위로 각각에 답하고 "
            "특정 유형에 근거가 없으면 그 점을 정중하게 짧게 밝힌다."
        )
    return "\n".join(lines).strip()


def _prompt_ordered_items(
    intent_weights: dict[str, float],
    items: list[EvidenceItem],
) -> list[EvidenceItem]:
    """Return evidence in the same order used inside the answer prompt."""
    buckets: dict[str, list[EvidenceItem]] = {}
    for it in items:
        buckets.setdefault(it.kind, []).append(it)

    if _intent_weight(intent_weights, "lecture_overview") >= 0.25:
        order = [
            "slide_text",
            "slide_concept",
            "graphrag_entity",
            "graphrag_relationship",
            "segment",
            "visual_asset",
            "sub_concept",
            "lance_strict",
            "lance_soft",
        ]
    else:
        order = [
            "visual_asset",
            "graphrag_entity",
            "graphrag_relationship",
            "sub_concept",
            "slide_text",
            "slide_concept",
            "segment",
            "lance_strict",
            "lance_soft",
        ]
    ordered: list[EvidenceItem] = []
    for bk in order:
        ordered.extend(buckets.get(bk, []))
    return ordered


def build_prompt_contexts(
    intent_weights: dict[str, float],
    items: list[EvidenceItem],
    max_chars: int,
) -> list[str]:
    """Build evidence snippets in the exact evidence order used by the prompt."""
    contexts: list[str] = []
    for it in _prompt_ordered_items(intent_weights, items):
        tag = {
            "sub_concept": "개념 관계",
            "graphrag_entity": "GraphRAG 개념",
            "graphrag_relationship": "GraphRAG 개념 관계",
            "visual_asset": "시각자료",
            "slide_text": "슬라이드 본문",
            "slide_concept": "슬라이드-개념",
            "segment": "음성 구간",
            "lance_strict": "의미 검색(그래프 연동)",
            "lance_soft": "의미 검색(보조)",
        }.get(it.kind, it.kind)
        meta = f"[{tag}]"
        if it.slide_number is not None:
            meta += f" 슬라이드 {it.slide_number}"
        if it.start_sec is not None:
            meta += f" · 약 {it.start_sec:.1f}초"
        if it.kind == "lance_soft":
            meta += " (그래프 id 미일치 보조)"
        contexts.append(f"{meta}\n{it.text[:max_chars]}")
    return contexts


def run_enhanced_content_pipeline(
    session,
    stem: str,
    question: str,
    extract_keywords_fn: Callable[[str], list[str]],
    call_gemini_raw: Callable[[str, str], str],
    current_slide_number: int | None = None,
) -> tuple[str, dict[str, float], set[str], dict[str, list[dict[str, Any]]], list[EvidenceItem]]:
    """
    Returns:
      context_str, intent_weights, allowed_ids, structured (full graph용), selected_items (표시/청크용)
    """
    cfg = _CFG or {}
    lim = cfg.get("limits") or {}
    score_cfg = cfg.get("scoring") or {}
    mmr_cfg = cfg.get("mmr") or {}
    prior_map = cfg.get("intent_source_prior") or {}

    intent_weights, hints = infer_intents_json(question, call_gemini_raw)
    lecture_overview_query = _is_lecture_overview_query(question, intent_weights)
    kws = extract_keywords_fn(question)
    for h in hints:
        if h not in kws:
            kws.append(h)
    # dedupe preserve order
    seen_k: set[str] = set()
    keywords: list[str] = []
    for x in kws:
        if x and x not in seen_k:
            seen_k.add(x)
            keywords.append(x)
    if lecture_overview_query:
        structured, _raw = run_overview_queries(session, stem)
        n_total = sum(len(v) for v in structured.values())
        if n_total == 0:
            return "", intent_weights, set(), structured, []
        allowed_ids = _collect_ids(structured)
        graph_items = _structured_to_items(structured)
        selected = _select_lecture_overview_items(graph_items)
        if not selected:
            return "", intent_weights, allowed_ids, structured, []
        context = build_sectioned_context(question, intent_weights, selected, int(lim.get("max_context_chars_per_item", 900)))
        return context, intent_weights, allowed_ids, structured, selected

    if not keywords:
        return "", intent_weights, set(), {}, []

    structured, _raw = run_content_queries(session, stem, keywords)
    n_total = sum(len(v) for v in structured.values())
    if n_total == 0:
        return "", intent_weights, set(), structured, []

    allowed_ids = _collect_ids(structured)
    graph_items = _structured_to_items(structured)

    cap = int(lim.get("graph_candidate_cap", 96))
    graph_items = graph_items[:cap]

    # Lance
    lance_root = default_lance_root()
    top1 = int(lim.get("lance_pass1_top_k", 40))
    top2 = int(lim.get("lance_pass2_top_k", 24))
    min_strict = int(lim.get("min_strict_lance", 4))
    strict_l, soft_l = _lance_two_pass(stem, question, lance_root, allowed_ids, top1, top2, min_strict)

    max_lance_ctx = int(lim.get("max_lance_in_context", 10))
    all_items = graph_items + strict_l + soft_l[: max(0, max_lance_ctx - len(strict_l))]

    if current_slide_number is not None and _is_current_visual_query(question):
        try:
            current_sn = int(current_slide_number)
        except (TypeError, ValueError):
            current_sn = None
        if current_sn is not None:
            current_items = [
                it for it in all_items
                if it.slide_number is not None and int(it.slide_number) == current_sn
            ]
            if current_items:
                all_items = current_items

    if not all_items:
        return "", intent_weights, allowed_ids, structured, []

    # Embedding rerank + MMR
    client = get_embedding_client(DEFAULT_EMBEDDING_PROVIDER)
    model = DEFAULT_EMBEDDING_MODEL
    texts = [it.text[:8000] for it in all_items]
    q_emb = np.array(embed_query(client, question, model=model, provider=DEFAULT_EMBEDDING_PROVIDER), dtype=np.float32)
    doc_embs: list[np.ndarray] = []
    bs = 32
    for i in range(0, len(texts), bs):
        batch = texts[i : i + bs]
        vecs = embed_documents(client, batch, model=model, provider=DEFAULT_EMBEDDING_PROVIDER)
        doc_embs.extend([np.array(v, dtype=np.float32) for v in vecs])

    sim_to_q = np.array(
        [
            float(np.dot(q_emb, d) / ((np.linalg.norm(q_emb) + 1e-9) * (np.linalg.norm(d) + 1e-9)))
            for d in doc_embs
        ]
    )
    sim_all = _cosine_mat(doc_embs) if doc_embs else np.array([])

    sw = float(score_cfg.get("semantic_weight", 0.52))
    iw = float(score_cfg.get("intent_weight", 0.33))
    kw_w = float(score_cfg.get("keyword_weight", 0.15))
    slide_importance_query = _is_slide_importance_query(question)
    visual_query = _is_visual_query(question)
    current_visual_query = _is_current_visual_query(question)
    emphasis_overview_query = _is_emphasis_overview_query(question)
    core_keyword_query = _is_core_keyword_query(question)
    example_query = _is_example_query(question, intent_weights)
    example_subject_terms = _example_subject_terms(question, keywords) if example_query else []
    max_slide_emphasis = max(
        [_row_float(it.row, "emphasis_total") for it in all_items if it.kind in {"slide_text", "slide_concept"}] or [0.0]
    )
    max_entity_weight = max(
        [_row_float(it.row, "final_weight") for it in all_items if it.kind == "graphrag_entity"] or [0.0]
    )
    max_rel_weight = max(
        [_row_float(it.row, "emphasis_edge_weight") for it in all_items if it.kind == "graphrag_relationship"] or [0.0]
    )
    max_entity_core = max(
        [_entity_core_score(it.row) for it in all_items if it.kind == "graphrag_entity"] or [0.0]
    )

    combined = np.zeros(len(all_items))
    for i, it in enumerate(all_items):
        ip = _intent_prior_for_kind(
            "lance" if it.kind.startswith("lance") else it.kind,
            intent_weights,
            prior_map,
        )
        if it.kind == "lance_soft":
            ip *= 0.72
        kw = _kw_score(all_items[i].text, keywords)
        semantic_component = sw * float(sim_to_q[i])
        intent_component = iw * ip
        keyword_component = kw_w * kw
        bonus_total = 0.0
        breakdown = {
            "semantic": semantic_component,
            "intent_prior": intent_component,
            "keyword": keyword_component,
            "bonus": 0.0,
        }
        combined[i] = semantic_component + intent_component + keyword_component
        if slide_importance_query and it.kind in {"slide_text", "slide_concept"} and max_slide_emphasis > 0:
            bonus = 0.45 * (_row_float(it.row, "emphasis_total") / max_slide_emphasis)
            combined[i] += bonus
            bonus_total += bonus
        if visual_query:
            if it.kind == "visual_asset":
                combined[i] += 0.55
                bonus_total += 0.55
            elif it.kind in {"slide_text", "slide_concept"} and (it.row or {}).get("t1_structure"):
                combined[i] += 0.25
                bonus_total += 0.25
            if current_visual_query and current_slide_number is not None and it.slide_number == current_slide_number:
                combined[i] += 0.60
                bonus_total += 0.60
        if emphasis_overview_query:
            if it.kind in {"slide_text", "slide_concept"} and max_slide_emphasis > 0:
                bonus = 0.30 * (_row_float(it.row, "emphasis_total") / max_slide_emphasis)
                combined[i] += bonus
                bonus_total += bonus
            elif it.kind == "graphrag_entity" and max_entity_weight > 0:
                bonus = 0.35 * (_row_float(it.row, "final_weight") / max_entity_weight)
                combined[i] += bonus
                bonus_total += bonus
            elif it.kind == "graphrag_relationship" and max_rel_weight > 0:
                bonus = 0.25 * (_row_float(it.row, "emphasis_edge_weight") / max_rel_weight)
                combined[i] += bonus
                bonus_total += bonus
        if core_keyword_query:
            if it.kind == "graphrag_entity" and max_entity_core > 0:
                bonus = 0.55 * (_entity_core_score(it.row) / max_entity_core)
                combined[i] += bonus
                bonus_total += bonus
            elif it.kind == "graphrag_relationship" and max_rel_weight > 0:
                bonus = 0.20 * (_row_float(it.row, "emphasis_edge_weight") / max_rel_weight)
                combined[i] += bonus
                bonus_total += bonus
            elif it.kind in {"slide_text", "slide_concept"} and max_slide_emphasis > 0:
                bonus = 0.15 * (_row_float(it.row, "emphasis_total") / max_slide_emphasis)
                combined[i] += bonus
                bonus_total += bonus
        if example_query:
            directness = _example_directness_score(it, example_subject_terms)
            if directness > 0:
                if it.kind in {"visual_asset", "slide_text", "slide_concept"}:
                    bonus = 0.45 * directness
                elif it.kind in {"graphrag_entity", "graphrag_relationship"}:
                    bonus = 0.35 * directness
                else:
                    bonus = 0.18 * directness
                combined[i] += bonus
                bonus_total += bonus
                breakdown["example_directness"] = directness
        breakdown["bonus"] = bonus_total
        breakdown["total"] = float(combined[i])
        breakdown["raw_semantic"] = float(sim_to_q[i])
        breakdown["raw_intent_prior"] = float(ip)
        breakdown["raw_keyword"] = float(kw)
        it.retrieval_score = float(combined[i])
        it.score_breakdown = breakdown

    order = list(np.argsort(-combined))
    mmr_k = int(lim.get("mmr_pick_k", 18))
    ex_w = intent_weights.get("example", 0.0) + intent_weights.get("definition", 0.0)
    lambda_mmr = float(mmr_cfg.get("lambda_example_heavy", 0.62)) if ex_w > 0.45 else float(mmr_cfg.get("lambda_default", 0.78))
    picked_idx = _mmr(order, sim_to_q, sim_all, min(mmr_k, len(all_items)), lambda_mmr)
    selected = [all_items[i] for i in picked_idx]
    if visual_query:
        visual_items = [it for it in all_items if it.kind == "visual_asset"]
        visual_items.sort(key=lambda it: it.retrieval_score if it.retrieval_score is not None else 0.0, reverse=True)
        if _is_visual_list_query(question) and visual_items:
            selected = visual_items[: min(8, len(visual_items))]
        elif _is_visual_interpretation_query(question) and visual_items:
            keep = visual_items[: min(3, len(visual_items))]
            keep_uids = {it.uid for it in keep}
            selected = keep + [it for it in selected if it.uid not in keep_uids]
        elif visual_items and not any(it.kind == "visual_asset" for it in selected):
            selected.insert(0, visual_items[0])
    selected_uids = {it.uid for it in selected}
    if not any(_is_media_evidence(it) for it in selected):
        for i in order:
            candidate = all_items[i]
            if _is_media_evidence(candidate) and candidate.uid not in selected_uids:
                selected.append(candidate)
                break

    max_c = int(lim.get("max_context_chars_per_item", 900))
    context = build_sectioned_context(question, intent_weights, selected, max_c)
    return context, intent_weights, allowed_ids, structured, selected
