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

from pipeline.embedding_utils import DEFAULT_EMBEDDING_MODEL, embed_documents, embed_query, get_genai_client
from pipeline.lance_ingest import default_lance_root, lance_search

from .neo4j_content_queries import run_content_queries


def _load_cfg() -> dict[str, Any]:
    p = Path(__file__).resolve().parent / "intent_config.json"
    if not p.is_file():
        return {}
    with open(p, encoding="utf-8") as f:
        return json.load(f)


_CFG = _load_cfg()


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
    raw = call_gemini_raw(
        f"사용자 질문:\n{question}\n",
        INTENT_SYSTEM_PROMPT,
    )
    m = re.search(r"\{[\s\S]*\}", raw)
    if not m:
        return {"general": 1.0}, []
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {"general": 1.0}, []
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
        return {"general": 1.0}, []
    s = sum(weights.values())
    if s > 0:
        weights = {k: v / s for k, v in weights.items()}
    hints = obj.get("keywords_hint") or []
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
        tit = str(r.get("title", "") or "")
        cid = str(r.get("concept_id", "") or "")
        if cid:
            uid = f"slc:{r.get('slide_id')}:{cid}"
            if uid in seen:
                continue
            seen.add(uid)
            text = f"슬라이드 {sn} {tit}\n{body}".strip()
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
            text = f"슬라이드 {sn} {tit}\n{body}".strip()
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
    for key in ("sub_concepts", "segments", "slides"):
        for r in structured.get(key, []):
            for fld in ("sub_id", "concept_id", "slide_id", "segment_id"):
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
    return it.kind in {"segment", "slide_text", "slide_concept"} or it.start_sec is not None


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
    # group by kind for readability
    buckets: dict[str, list[EvidenceItem]] = {}
    for it in items:
        buckets.setdefault(it.kind, []).append(it)

    order = [
        "graphrag_entity",
        "graphrag_relationship",
        "sub_concept",
        "slide_text",
        "slide_concept",
        "segment",
        "lance_strict",
        "lance_soft",
    ]
    for bk in order:
        for it in buckets.get(bk, []):
            tag = {
                "sub_concept": "개념 관계",
                "graphrag_entity": "GraphRAG 개념",
                "graphrag_relationship": "GraphRAG 개념 관계",
                "slide_text": "슬라이드 본문",
                "slide_concept": "슬라이드-개념",
                "segment": "음성 구간",
                "lance_strict": "의미 검색(그래프 연동)",
                "lance_soft": "의미 검색(보조)",
            }.get(bk, bk)
            meta = f"[{tag}]"
            if it.slide_number is not None:
                meta += f" 슬라이드 {it.slide_number}"
            if it.start_sec is not None:
                meta += f" · 약 {it.start_sec:.1f}초"
            if it.kind == "lance_soft":
                meta += " (그래프 id 미일치 보조)"
            chunk = it.text[:max_chars]
            lines.append(f"{meta}\n{chunk}")
            lines.append("")
    lines.append(
        "질문에 정의·예시·설명 등 여러 요구가 섞여 있으면, 위 근거에서 가능한 범위로 각각에 답하고 "
        "특정 유형에 근거가 없으면 그 점을 짧게 밝힌다."
    )
    return "\n".join(lines).strip()


def run_enhanced_content_pipeline(
    session,
    stem: str,
    question: str,
    extract_keywords_fn: Callable[[str], list[str]],
    call_gemini_raw: Callable[[str, str], str],
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

    if not all_items:
        return "", intent_weights, allowed_ids, structured, []

    # Embedding rerank + MMR
    client = get_genai_client()
    model = DEFAULT_EMBEDDING_MODEL
    texts = [it.text[:8000] for it in all_items]
    q_emb = np.array(embed_query(client, question, model=model), dtype=np.float32)
    doc_embs: list[np.ndarray] = []
    bs = 32
    for i in range(0, len(texts), bs):
        batch = texts[i : i + bs]
        vecs = embed_documents(client, batch, model=model)
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
        combined[i] = sw * sim_to_q[i] + iw * ip + kw_w * kw
        it.retrieval_score = float(combined[i])

    order = list(np.argsort(-combined))
    mmr_k = int(lim.get("mmr_pick_k", 18))
    ex_w = intent_weights.get("example", 0.0) + intent_weights.get("definition", 0.0)
    lambda_mmr = float(mmr_cfg.get("lambda_example_heavy", 0.62)) if ex_w > 0.45 else float(mmr_cfg.get("lambda_default", 0.78))
    picked_idx = _mmr(order, sim_to_q, sim_all, min(mmr_k, len(all_items)), lambda_mmr)
    selected = [all_items[i] for i in picked_idx]
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
