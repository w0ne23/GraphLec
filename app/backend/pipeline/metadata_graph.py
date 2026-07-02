"""
pipeline/metadata_graph.py
──────────────────────────
GraphRAG 산출물 기반 graph-first 메타데이터 빌더.

  - 파일 기반 그래프 artifact 로딩 (Parquet, GraphRAG output)
  - graph-first keyword 후보 구성 및 병합
  - concept_roles / concept_relations 구성

LLM 호출(generate_graph_first_summary 등)과 메인 오케스트레이터
(try_generate_graph_first_metadata_parts)는 generate_metadata.py에 유지한다.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path


# ── 언어 규칙 기반 노이즈 필터 상수 ──────────────────────────────────────────

# 동사 어미·조사 패턴 (한국어 언어 규칙 — 도메인 무관)
_NOISE_SUFFIX_RE = re.compile(
    r"(하고|하며|하여|하는|하면서|하고자|이라고|이며|이나|이고"
    r"|대해|에서|으로|에게|으로써|이다|아니다|있다|없다|한다"
    r"|하거나|하지|하면|이지|하니|않고|이후|이전"
    r"|있도록|있음|없던|만든|만드|된|될|할|함|함으로|함께"
    r"|하다$|된다$|되다$|가능하다$|하$|않는다$|아님$|준다$|포함한$|보여주$|느끼$|제공받$"
    r"|반영하$|전달하$|구현하$|쫓아가$)$"
)

# 영어 기능어 (문법적 역할만 하는 단어 — 도메인 무관)
_ENGLISH_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "and", "or", "but", "if", "in", "on", "at", "to", "for", "of",
    "with", "by", "from", "as", "into", "about",
}


# ── 공유 유틸리티 ─────────────────────────────────────────────────────────────

def _normalize(d: dict[str, float]) -> dict[str, float]:
    if not d:
        return d
    max_v = max(d.values()) or 1.0
    return {k: v / max_v for k, v in d.items()}


def _is_valid_concept(name: str) -> bool:
    """
    언어 규칙만으로 명백한 노이즈를 걸러내는 필터.
    도메인·강의 무관하게 적용 가능한 규칙만 포함.

    제거 대상:
      - 길이 2자 미만
      - 숫자만으로 구성된 토큰 (예: "001", "123")
      - URL 잔재 (예: "proprofsurvey.com", "tistory.com")
      - 동사 어미·조사 패턴으로 끝나는 한글 토큰
      - 영어 기능어 (소문자 단독 단어)
    """
    name = name.strip()
    if len(name) < 2:
        return False
    if re.fullmatch(r"\d+", name):
        return False
    if re.search(r"\.(com|kr|org|net|io|tistory|brunch|github)", name):
        return False
    if re.fullmatch(r"[a-z]+", name):
        return name not in _ENGLISH_STOPWORDS
    if _NOISE_SUFFIX_RE.search(name):
        return False
    return True


# ── 그래프 파일 경로 탐색 ─────────────────────────────────────────────────────

def _read_parquet_records(path: Path) -> tuple[list[dict], str | None]:
    if not path.is_file():
        return [], "missing"
    try:
        import pandas as pd
        return pd.read_parquet(path).to_dict(orient="records"), None
    except Exception as exc:
        return [], str(exc)


def _first_existing_path(paths: list[Path]) -> Path | None:
    seen: set[Path] = set()
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        if path.is_file():
            return path
    return None


def _candidate_stage_graph_dirs(stem: str, output_dir: Path) -> list[Path]:
    return [
        output_dir,
        output_dir / "parquet",
        output_dir / stem,
        output_dir.parent / stem,
    ]


def _find_stage_graph_file(stem: str, output_dir: Path, suffix: str) -> Path | None:
    candidates = [
        directory / f"{stem}_{suffix}.parquet"
        for directory in _candidate_stage_graph_dirs(stem, output_dir)
    ]
    return _first_existing_path(candidates)


def _candidate_graphrag_output_dirs(stem: str, output_dir: Path) -> list[Path]:
    candidates = [
        output_dir / "graphrag" / "output",
        output_dir / stem / "graphrag" / "output",
        output_dir.parent / stem / "graphrag" / "output",
    ]
    try:
        candidates.extend(
            path / "graphrag" / "output"
            for path in sorted(output_dir.parent.glob(f"{stem}*"))
            if path.is_dir()
        )
    except Exception:
        pass
    return candidates


def _find_graphrag_output_dir(stem: str, output_dir: Path) -> Path | None:
    seen: set[Path] = set()
    for directory in _candidate_graphrag_output_dirs(stem, output_dir):
        if directory in seen:
            continue
        seen.add(directory)
        if (directory / "entities.parquet").is_file() or (directory / "community_reports.parquet").is_file():
            return directory
    return None


def _count_label(rows: list[dict], key: str) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        value = str(row.get(key) or "").strip()
        if value:
            counts[value] += 1
    return dict(counts)


def _to_float(value, default: float = 0.0) -> float:
    try:
        parsed = float(value)
        return parsed if math.isfinite(parsed) else default
    except (TypeError, ValueError):
        return default


def _iter_values(value) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    try:
        return value.tolist()
    except AttributeError:
        return [value]


# ── artifact 로딩 ─────────────────────────────────────────────────────────────

def load_graph_artifacts_for_metadata(
    stem: str,
    output_dir: Path,
    fused: dict,
    emphasized: dict[str, float],
) -> dict:
    """
    추천 메타데이터 graph-first 경로에서 사용할 파일 기반 그래프 산출물.

    Neo4j는 metadata 생성 시점의 필수 입력이 아니므로 여기서 호출하지 않는다.
    emphasized는 호출자(generate_metadata.py)가 collect_emphasized(fused)로 준비해 전달한다.
    """
    graph_dir = _find_graphrag_output_dir(stem, output_dir)
    graphrag: dict[str, dict] = {}
    if graph_dir:
        for name in ("entities", "relationships", "communities", "community_reports", "text_units"):
            path = graph_dir / f"{name}.parquet"
            records, error = _read_parquet_records(path)
            graphrag[name] = {
                "path": str(path),
                "rows": records,
                "count": len(records),
                "error": error,
            }
    else:
        graphrag["error"] = "graphrag output not found"

    stage_graph: dict[str, dict] = {}
    for name, suffix in (
        ("nodes", "nodes"),
        ("edges", "edges"),
        ("graph_triples", "graph_triples"),
    ):
        path = _find_stage_graph_file(stem, output_dir, suffix)
        records, error = _read_parquet_records(path) if path else ([], "missing")
        stage_graph[name] = {
            "path": str(path) if path else "",
            "rows": records,
            "count": len(records),
            "error": error,
        }

    return {
        "graphrag_output_dir": str(graph_dir) if graph_dir else "",
        "graphrag": graphrag,
        "stage_graph": stage_graph,
        "emphasis": emphasized,
        "summary": {
            "graphrag_entities": graphrag.get("entities", {}).get("count", 0),
            "graphrag_relationships": graphrag.get("relationships", {}).get("count", 0),
            "graphrag_communities": graphrag.get("communities", {}).get("count", 0),
            "graphrag_reports": graphrag.get("community_reports", {}).get("count", 0),
            "stage_nodes": stage_graph.get("nodes", {}).get("count", 0),
            "stage_edges": stage_graph.get("edges", {}).get("count", 0),
            "stage_triples": stage_graph.get("graph_triples", {}).get("count", 0),
            "stage_node_labels": _count_label(stage_graph.get("nodes", {}).get("rows", []), "label"),
            "stage_edge_types": _count_label(stage_graph.get("edges", {}).get("rows", []), "rel_type"),
            "emphasis_terms": len(emphasized),
        },
    }


def _log_graph_artifact_summary(stem: str, artifacts: dict) -> None:
    summary = artifacts.get("summary", {})
    print(
        f"[{stem}] graph artifacts: "
        f"GraphRAG entities={summary.get('graphrag_entities', 0)}, "
        f"relationships={summary.get('graphrag_relationships', 0)}, "
        f"communities={summary.get('graphrag_communities', 0)}, "
        f"reports={summary.get('graphrag_reports', 0)}"
    )
    print(
        f"[{stem}] stage graph: "
        f"nodes={summary.get('stage_nodes', 0)}, "
        f"edges={summary.get('stage_edges', 0)}, "
        f"triples={summary.get('stage_triples', 0)}, "
        f"emphasis_terms={summary.get('emphasis_terms', 0)}"
    )


# ── entity 점수 산출 ──────────────────────────────────────────────────────────

def _norm_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def _stringify_for_match(value) -> str:
    values = _iter_values(value)
    if len(values) == 1 and values[0] is value:
        return str(value or "")
    return " ".join(str(item or "") for item in values)


def _term_matches_text(term: str, text: str) -> bool:
    term_norm = _norm_text(term)
    text_norm = _norm_text(text)
    return bool(term_norm and text_norm and term_norm in text_norm)


def _score_entity_emphasis(entity: dict, emphasis: dict[str, float]) -> float:
    title = str(entity.get("title") or "")
    description = str(entity.get("description") or "")
    score = 0.0
    for keyword, weight in emphasis.items():
        if _term_matches_text(keyword, title):
            score += float(weight)
        elif _term_matches_text(keyword, description):
            score += float(weight) * 0.5
    return score


def _score_entity_text_grounding(
    entity: dict,
    slide_texts: list[str],
    transcript_texts: list[str],
) -> tuple[float, float, float]:
    title = str(entity.get("title") or "")
    if not title:
        return 0.0, 0.0, 0.0
    slide_hits = sum(1 for text in slide_texts if title in str(text or ""))
    transcript_hits = sum(1 for text in transcript_texts if title in str(text or ""))
    slide_grounding = float(slide_hits)
    transcript_grounding = float(transcript_hits)
    combined_grounding = slide_grounding + transcript_grounding * 0.5
    return slide_grounding, transcript_grounding, combined_grounding


def _community_report_mention_score(title: str, report: dict | None) -> float:
    if not title or not report:
        return 0.0
    score = 0.0
    if _term_matches_text(title, str(report.get("title") or "")):
        score += 1.0
    if _term_matches_text(title, str(report.get("summary") or "")):
        score += 0.7
    if _term_matches_text(title, _stringify_for_match(report.get("findings"))):
        score += 0.6
    if _term_matches_text(title, str(report.get("full_content") or "")):
        score += 0.4
    return score


# ── keyword 후보 병합 판단 상수·헬퍼 ─────────────────────────────────────────

_EXAMPLE_RELATION_TERMS = (
    "example", "instance", "instance_of", "kind", "type",
    "종류", "예시", "사례", "대표적인", "중 하나", "중의 하나",
    "예로", "예에는", "포함된다", "포함되는",
)

_OPERATIONAL_TITLE_TERMS = (
    "교수", "강사", "시험", "출석", "평가", "학점", "과제", "퀴즈",
    "중간고사", "기말고사", "참고 문헌", "참고문헌", "교재", "주차",
    "공지", "대학교", "대학", "학과", "처장", "총장",
)

_GENERIC_SUFFIX_TERMS = (
    "개념", "정의", "기능", "목적", "특징", "원리", "이해", "소개", "기초", "관련",
)

_MERGE_BRIDGE_TERMS = (
    "와", "과", "및", "또는", "그리고", "차이", "관계", "비교", "대조",
)


def _has_example_relation_terms(rel: dict) -> bool:
    source = str(rel.get("source") or "").strip()
    target = str(rel.get("target") or "").strip()
    description = str(rel.get("description") or "")
    rel_type = str(rel.get("type") or rel.get("rel_type") or rel.get("predicate") or "")
    haystack = _norm_text(" ".join([source, target, description, rel_type]))
    return any(term in haystack for term in _EXAMPLE_RELATION_TERMS)


def _example_child_title(rel: dict, importance_by_title: dict[str, float]) -> str:
    if not _has_example_relation_terms(rel):
        return ""
    source = str(rel.get("source") or "").strip()
    target = str(rel.get("target") or "").strip()
    if not source or not target or source == target:
        return ""
    source_importance = importance_by_title.get(source, 0.0)
    target_importance = importance_by_title.get(target, 0.0)
    if source_importance != target_importance:
        return source if source_importance < target_importance else target
    description = _norm_text(str(rel.get("description") or ""))
    source_norm = _norm_text(source)
    target_norm = _norm_text(target)
    if source_norm and target_norm and source_norm in description and target_norm in description:
        return source
    return target


def _candidate_canonical_key(title: str) -> str:
    text = _norm_text(title)
    text = re.sub(r"\([^)]*\)", "", text)
    text = re.sub(r"[\s_./#+:;,\-·'\"`~!?\[\]{}<>]", "", text)
    text = re.sub(r"(에대한|에관한|관련된|관련)$", "", text)
    text = text.replace("의", "")
    for suffix in _GENERIC_SUFFIX_TERMS:
        suffix_key = _norm_text(suffix).replace(" ", "")
        if len(text) > len(suffix_key) + 2 and text.endswith(suffix_key):
            text = text[: -len(suffix_key)]
            break
    return text


def _operational_penalty(title: str, entity: dict) -> float:
    title_norm = _norm_text(title)
    entity_type = _norm_text(entity.get("type"))
    penalty = 0.0
    if entity_type in {"person", "organization", "geo"}:
        penalty = max(penalty, 0.45)
    if any(term in title_norm for term in _OPERATIONAL_TITLE_TERMS):
        penalty = max(penalty, 0.65)
    return penalty


def _operational_score_cap(
    title: str,
    entity: dict,
    graph_importance: float,
    emphasis_boost: float,
    text_grounding: float,
) -> float | None:
    title_norm = _norm_text(title)
    entity_type = _norm_text(entity.get("type"))
    if any(term in title_norm for term in _OPERATIONAL_TITLE_TERMS):
        return 0.16
    if (
        entity_type == "person"
        and graph_importance < 0.65
        and max(emphasis_boost, text_grounding) < 0.25
    ):
        return 0.22
    if entity_type in {"organization", "geo"} and max(emphasis_boost, text_grounding) < 0.20:
        return 0.24
    if entity_type in {"organization", "geo"} and graph_importance < 0.35:
        return 0.24
    return None


def _code_like_penalty(title: str) -> float:
    stripped = re.sub(r"[\s_\-./#+]", "", str(title or ""))
    if not stripped:
        return 0.0
    has_alpha = bool(re.search(r"[A-Za-z]", stripped))
    if not has_alpha:
        return 0.0
    if re.fullmatch(r"[A-Za-z]+[0-9]+", stripped) and len(stripped) <= 8:
        return 1.0
    if stripped.isupper() and 2 <= len(stripped) <= 3:
        return 0.35
    if stripped.isupper() and 2 <= len(stripped) <= 6:
        return 0.75
    if re.fullmatch(r"[A-Za-z]+[0-9]*", stripped) and len(stripped) <= 6:
        return 0.55
    return 0.0


def _looks_like_merge_bridge(title: str) -> bool:
    title_norm = _norm_text(title)
    return any(term in title_norm for term in _MERGE_BRIDGE_TERMS)


def _is_suffix_expansion(base_key: str, expanded_key: str) -> bool:
    if base_key == expanded_key:
        return True
    if len(expanded_key) <= len(base_key):
        return False
    if not expanded_key.startswith(base_key):
        return False
    remainder = expanded_key[len(base_key):]
    if not remainder:
        return True
    return any(
        remainder == _norm_text(suffix).replace(" ", "")
        for suffix in _GENERIC_SUFFIX_TERMS
    )


def _sequence_similarity(a_key: str, b_key: str) -> float:
    if not a_key or not b_key:
        return 0.0
    return SequenceMatcher(None, a_key, b_key).ratio()


def _same_length_single_char_variant(a_key: str, b_key: str) -> bool:
    if len(a_key) != len(b_key) or len(a_key) < 4:
        return False
    return sum(1 for left, right in zip(a_key, b_key) if left != right) == 1


def _has_structural_variant_marker(title: str) -> bool:
    text = str(title or "").strip()
    return bool(
        re.match(r"^\d+\s*[.)]", text)
        or re.search(r"[①-⑳]", text)
        or ("(" in text and ")" in text)
        or ("（" in text and "）" in text)
    )


def _looks_like_transcription_variant(a: dict, b: dict) -> bool:
    a_key = str(a.get("canonical_key") or "")
    b_key = str(b.get("canonical_key") or "")
    if len(a_key) < 4 or len(b_key) < 4:
        return False
    a_title = str(a.get("keyword") or "")
    b_title = str(b.get("keyword") or "")
    if _looks_like_merge_bridge(a_title) or _looks_like_merge_bridge(b_title):
        return False
    if _has_structural_variant_marker(a_title) or _has_structural_variant_marker(b_title):
        return False
    if a_key in b_key or b_key in a_key:
        return False
    sequence_sim = _sequence_similarity(a_key, b_key)
    typo_like = (
        _same_length_single_char_variant(a_key, b_key)
        and sequence_sim >= 0.84
    )
    return typo_like


def _has_independent_graph_signal(a: dict, b: dict) -> bool:
    return (
        float(a.get("graph_importance", 0.0)) >= 0.12
        and float(b.get("graph_importance", 0.0)) >= 0.12
        and float(a.get("community_inner_representative", 0.0)) >= 0.20
        and float(b.get("community_inner_representative", 0.0)) >= 0.20
    )


def _should_merge_keyword_candidates(a: dict, b: dict) -> bool:
    a_key = str(a.get("canonical_key") or "")
    b_key = str(b.get("canonical_key") or "")
    if not a_key or not b_key:
        return False
    if len(a_key) < 3 or len(b_key) < 3:
        return False
    a_title = str(a.get("keyword") or "")
    b_title = str(b.get("keyword") or "")
    if _has_structural_variant_marker(a_title) or _has_structural_variant_marker(b_title):
        return False
    if a_key == b_key:
        return True
    if _looks_like_merge_bridge(a_title) or _looks_like_merge_bridge(b_title):
        return False
    if _looks_like_transcription_variant(a, b):
        return True
    if _has_independent_graph_signal(a, b):
        return False
    return _is_suffix_expansion(a_key, b_key) or _is_suffix_expansion(b_key, a_key)


def _candidate_representative_score(item: dict) -> float:
    return (
        float(item.get("graph_importance", 0.0))
        + float(item.get("community_inner_representative", 0.0))
        + float(item.get("text_grounding", 0.0))
        - float(item.get("example_penalty", 0.0))
        - float(item.get("operational_penalty", 0.0))
        - float(item.get("code_like_penalty", 0.0))
    )


def _merge_graph_keyword_candidates(candidates: list[dict], limit: int) -> list[dict]:
    clusters: list[list[dict]] = []
    for candidate in sorted(candidates, key=lambda item: item["score"], reverse=True):
        target_cluster = None
        for cluster in clusters:
            if any(_should_merge_keyword_candidates(candidate, existing) for existing in cluster):
                target_cluster = cluster
                break
        if target_cluster is None:
            clusters.append([candidate])
        else:
            target_cluster.append(candidate)

    merged = []
    for cluster in clusters:
        representative = max(
            cluster,
            key=lambda item: (
                _candidate_representative_score(item),
                float(item.get("score", 0.0)),
                -len(str(item.get("keyword") or "")),
            ),
        ).copy()
        alternates = [
            item["keyword"]
            for item in sorted(cluster, key=lambda row: row["score"], reverse=True)
            if item["keyword"] != representative["keyword"]
        ]
        if alternates:
            representative["merged_variants"] = alternates[:6]
            representative["merged_count"] = len(cluster)
        merged.append(representative)

    merged.sort(key=lambda item: item["score"], reverse=True)
    for item in merged:
        item.pop("canonical_key", None)
    return merged[:limit]


# ── graph-first keyword 후보 빌더 ─────────────────────────────────────────────

def build_graph_keyword_candidates(
    artifacts: dict,
    slide_texts: list[str],
    transcript_texts: list[str],
    limit: int = 30,
) -> list[dict]:
    """GraphRAG entity 중심의 추천 키워드 후보와 점수 breakdown."""
    entities = artifacts.get("graphrag", {}).get("entities", {}).get("rows", []) or []
    relationships = artifacts.get("graphrag", {}).get("relationships", {}).get("rows", []) or []
    communities = artifacts.get("graphrag", {}).get("communities", {}).get("rows", []) or []
    reports = artifacts.get("graphrag", {}).get("community_reports", {}).get("rows", []) or []
    emphasis = artifacts.get("emphasis", {}) or {}
    if not entities:
        return []

    entity_by_id = {
        str(entity.get("id") or ""): entity
        for entity in entities
        if str(entity.get("id") or "").strip()
    }
    title_by_id = {
        entity_id: str(entity.get("title") or "").strip()
        for entity_id, entity in entity_by_id.items()
    }
    report_by_community = {
        str(report.get("community") or ""): report
        for report in reports
    }
    importance_by_title = {
        str(entity.get("title") or "").strip(): (
            _to_float(entity.get("degree"))
            + math.log1p(max(_to_float(entity.get("frequency")), 0.0))
        )
        for entity in entities
        if str(entity.get("title") or "").strip()
    }

    raw_community_quality: Counter[str] = Counter()
    raw_community_report_presence: Counter[str] = Counter()
    community_memberships: dict[str, set[str]] = {}
    community_rel_endpoint_signal: dict[str, Counter[str]] = {}
    relationship_by_id = {
        str(rel.get("id") or ""): rel
        for rel in relationships
        if str(rel.get("id") or "").strip()
    }

    for community in communities:
        community_id = str(community.get("community") or "")
        report = report_by_community.get(community_id)
        rank = _to_float((report or {}).get("rank"))
        size = _to_float(community.get("size"))
        quality = rank / math.sqrt(max(size, 1.0))
        endpoint_counter: Counter[str] = Counter()
        for rel_id in _iter_values(community.get("relationship_ids")):
            rel = relationship_by_id.get(str(rel_id))
            if not rel:
                continue
            signal = _to_float(rel.get("weight")) + math.log1p(max(_to_float(rel.get("combined_degree")), 0.0))
            for key in ("source", "target"):
                title = str(rel.get(key) or "").strip()
                if title:
                    endpoint_counter[title] += signal
        community_rel_endpoint_signal[community_id] = endpoint_counter

        for entity_id in _iter_values(community.get("entity_ids")):
            entity_id = str(entity_id)
            title = title_by_id.get(entity_id)
            if not title:
                continue
            raw_community_quality[title] += quality
            raw_community_report_presence[title] += _community_report_mention_score(title, report)
            community_memberships.setdefault(title, set()).add(community_id)

    relation_endpoint_signal: Counter[str] = Counter()
    example_relation_signal: Counter[str] = Counter()
    for rel in relationships:
        weight = _to_float(rel.get("weight"))
        combined_degree = _to_float(rel.get("combined_degree"))
        signal = weight + math.log1p(max(combined_degree, 0.0))
        example_child = _example_child_title(rel, importance_by_title)
        for key in ("source", "target"):
            title = str(rel.get(key) or "").strip()
            if title:
                relation_endpoint_signal[title] += signal
                if title == example_child:
                    example_relation_signal[title] += signal

    raw_graph_importance: dict[str, float] = {}
    raw_community: dict[str, float] = {}
    raw_community_member_quality: dict[str, float] = {}
    raw_community_inner_representative: dict[str, float] = {}
    raw_community_cross_bridge: dict[str, float] = {}
    raw_emphasis: dict[str, float] = {}
    raw_bridge: dict[str, float] = {}
    raw_slide_grounding: dict[str, float] = {}
    raw_transcript_grounding: dict[str, float] = {}
    raw_grounding: dict[str, float] = {}
    entity_by_title: dict[str, dict] = {}

    norm_member_quality = _normalize(dict(raw_community_quality))
    norm_report_presence = _normalize(dict(raw_community_report_presence))
    raw_inner_relation: dict[str, float] = {}
    for title, memberships in community_memberships.items():
        raw_inner_relation[title] = sum(
            community_rel_endpoint_signal.get(community_id, Counter()).get(title, 0.0)
            for community_id in memberships
        )
    norm_inner_relation = _normalize(raw_inner_relation)
    raw_cross_membership = {
        title: float(max(len(memberships) - 1, 0))
        for title, memberships in community_memberships.items()
    }
    norm_cross_membership = _normalize(raw_cross_membership)

    for entity in entities:
        title = str(entity.get("title") or "").strip()
        if not title or not _is_valid_concept(title):
            continue
        entity_by_title[title] = entity
        raw_graph_importance[title] = (
            _to_float(entity.get("degree"))
            + math.log1p(max(_to_float(entity.get("frequency")), 0.0))
        )
        member_quality = norm_member_quality.get(title, 0.0)
        inner_representative = (
            0.60 * norm_report_presence.get(title, 0.0)
            + 0.40 * norm_inner_relation.get(title, 0.0)
        )
        cross_bridge = norm_cross_membership.get(title, 0.0)
        raw_community_member_quality[title] = member_quality
        raw_community_inner_representative[title] = inner_representative
        raw_community_cross_bridge[title] = cross_bridge
        raw_community[title] = (
            0.40 * member_quality
            + 0.40 * inner_representative
            + 0.20 * cross_bridge
        )
        raw_emphasis[title] = _score_entity_emphasis(entity, emphasis)
        raw_bridge[title] = (
            relation_endpoint_signal.get(title, 0.0)
            * (1.0 + 0.25 * max(len(community_memberships.get(title, set())) - 1, 0))
        )
        slide_grounding, transcript_grounding, combined_grounding = _score_entity_text_grounding(
            entity, slide_texts, transcript_texts,
        )
        raw_slide_grounding[title] = slide_grounding
        raw_transcript_grounding[title] = transcript_grounding
        raw_grounding[title] = combined_grounding

    norm_graph = _normalize(raw_graph_importance)
    norm_community = _normalize(raw_community)
    norm_emphasis = _normalize(raw_emphasis)
    norm_bridge = _normalize(raw_bridge)
    norm_slide_grounding = _normalize(raw_slide_grounding)
    norm_transcript_grounding = _normalize(raw_transcript_grounding)
    norm_grounding = _normalize(raw_grounding)
    norm_example_penalty = _normalize(dict(example_relation_signal))

    candidates = []
    for title in norm_graph:
        entity = entity_by_title.get(title, {})
        base_score = (
            0.45 * norm_graph.get(title, 0.0)
            + 0.25 * norm_community.get(title, 0.0)
            + 0.15 * norm_emphasis.get(title, 0.0)
            + 0.10 * norm_bridge.get(title, 0.0)
            + 0.05 * norm_grounding.get(title, 0.0)
        )
        example_penalty = norm_example_penalty.get(title, 0.0)
        protected_core_signal = max(
            norm_graph.get(title, 0.0),
            norm_grounding.get(title, 0.0),
        )
        operational_penalty = _operational_penalty(title, entity)
        code_like_penalty = _code_like_penalty(title)
        penalty_strength = (
            0.55 * example_penalty
            + 0.35 * operational_penalty
            + 0.65 * code_like_penalty
        ) * (1.0 - 0.5 * protected_core_signal)
        score = base_score * (1.0 - min(max(penalty_strength, 0.0), 0.65))
        if code_like_penalty >= 0.75 and norm_graph.get(title, 0.0) < 0.60:
            score = min(score, 0.18)
        elif code_like_penalty >= 0.55 and norm_graph.get(title, 0.0) < 0.35:
            score = min(score, 0.20)
        operational_score_cap = _operational_score_cap(
            title,
            entity,
            norm_graph.get(title, 0.0),
            norm_emphasis.get(title, 0.0),
            norm_grounding.get(title, 0.0),
        )
        if operational_score_cap is not None:
            score = min(score, operational_score_cap)
        candidates.append({
            "keyword": title,
            "score": round(score, 4),
            "canonical_key": _candidate_canonical_key(title),
            "base_score": round(base_score, 4),
            "graph_importance": round(norm_graph.get(title, 0.0), 4),
            "community_representativeness": round(norm_community.get(title, 0.0), 4),
            "community_quality": round(raw_community_member_quality.get(title, 0.0), 4),
            "community_inner_representative": round(raw_community_inner_representative.get(title, 0.0), 4),
            "community_cross_bridge": round(raw_community_cross_bridge.get(title, 0.0), 4),
            "emphasis_boost": round(norm_emphasis.get(title, 0.0), 4),
            "relation_bridge_score": round(norm_bridge.get(title, 0.0), 4),
            "slide_grounding": round(norm_slide_grounding.get(title, 0.0), 4),
            "transcript_grounding": round(norm_transcript_grounding.get(title, 0.0), 4),
            "text_grounding": round(norm_grounding.get(title, 0.0), 4),
            "example_penalty": round(example_penalty, 4),
            "operational_penalty": round(operational_penalty, 4),
            "operational_score_cap": (
                round(operational_score_cap, 4) if operational_score_cap is not None else None
            ),
            "code_like_penalty": round(code_like_penalty, 4),
            "penalty_strength": round(penalty_strength, 4),
            "raw_degree": round(_to_float(entity.get("degree")), 4),
            "raw_frequency": round(_to_float(entity.get("frequency")), 4),
        })

    candidates.sort(key=lambda item: item["score"], reverse=True)
    return _merge_graph_keyword_candidates(candidates, limit)


def _log_graph_keyword_candidates(stem: str, candidates: list[dict], limit: int = 12) -> None:
    print(f"[{stem}] graph-first keyword candidates 상위 {min(limit, len(candidates))}개")
    for item in candidates[:limit]:
        print(
            "  "
            f"{item['keyword']:<24} score={item['score']:.4f} "
            f"graph={item['graph_importance']:.3f} "
            f"comm={item['community_representativeness']:.3f} "
            f"(q={item['community_quality']:.2f},in={item['community_inner_representative']:.2f},x={item['community_cross_bridge']:.2f}) "
            f"emph={item['emphasis_boost']:.3f} "
            f"bridge={item['relation_bridge_score']:.3f} "
            f"text={item['text_grounding']:.3f} "
            f"(s={item.get('slide_grounding', 0.0):.2f},t={item.get('transcript_grounding', 0.0):.2f}) "
            f"penalty={item['penalty_strength']:.3f}"
        )


def _merge_graph_keywords_with_legacy(
    graph_candidates: list[dict],
    legacy_keywords: list[dict],
    target_count: int,
) -> list[dict]:
    """검증된 그래프 후보를 우선 사용하고 부족분만 legacy 키워드로 보충한다."""
    merged: list[dict] = []
    seen: set[str] = set()

    def add_keyword(keyword: str, score: float, **breakdown: float) -> None:
        key = _norm_text(keyword)
        if not keyword or not key or key in seen:
            return
        seen.add(key)
        entry: dict = {"keyword": keyword, "score": round(score, 4)}
        entry.update({k: round(v, 4) for k, v in breakdown.items() if v > 0.0})
        merged.append(entry)

    for candidate in graph_candidates:
        if len(merged) >= target_count:
            break
        add_keyword(
            str(candidate.get("keyword") or "").strip(),
            _to_float(candidate.get("score")),
            community_representativeness=_to_float(candidate.get("community_representativeness")),
            text_grounding=_to_float(candidate.get("text_grounding")),
        )

    for keyword in legacy_keywords:
        if len(merged) >= target_count:
            break
        add_keyword(
            str(keyword.get("keyword") or "").strip(),
            _to_float(keyword.get("score")),
        )

    return merged


def _graph_keyword_debug_summary(keyword_candidates: list[dict], artifacts: dict) -> dict:
    return {
        "graph_keyword_source": "graphrag_entity_community",
        "graph_keyword_candidate_count": len(keyword_candidates),
        "graph_keyword_top_preview": [
            {
                "keyword": str(candidate.get("keyword") or ""),
                "score": round(_to_float(candidate.get("score")), 4),
            }
            for candidate in keyword_candidates[:5]
        ],
        "graph_artifact_summary": artifacts.get("summary", {}),
    }


# ── alias map / 정규화 ────────────────────────────────────────────────────────

def build_graph_candidate_alias_map(keyword_candidates: list[dict]) -> dict[str, str]:
    alias_map: dict[str, str] = {}
    for candidate in keyword_candidates:
        representative = str(candidate.get("keyword") or "").strip()
        if not representative:
            continue
        for variant in candidate.get("merged_variants") or []:
            variant = str(variant or "").strip()
            if variant and _norm_text(variant) != _norm_text(representative):
                alias_map[variant] = representative
    return alias_map


def _canonicalize_graph_name(name: str, alias_map: dict[str, str]) -> str:
    name = str(name or "").strip()
    if not name:
        return ""
    if name in alias_map:
        return alias_map[name]
    norm_name = _norm_text(name)
    for alias, representative in alias_map.items():
        if _norm_text(alias) == norm_name:
            return representative
    return name


def _canonicalize_graph_name_list(names: list[str], alias_map: dict[str, str]) -> list[str]:
    canonicalized: list[str] = []
    seen: set[str] = set()
    for name in names:
        representative = _canonicalize_graph_name(name, alias_map)
        key = _norm_text(representative)
        if representative and key and key not in seen:
            seen.add(key)
            canonicalized.append(representative)
    return canonicalized


# ── concept_roles / concept_relations 빌더 ───────────────────────────────────

def build_graph_concept_roles(
    keyword_candidates: list[dict],
    core_keywords: list[str] | None = None,
) -> dict[str, list[str]]:
    """GraphRAG 후보 점수 기반으로 core/introduced 역할을 구성한다."""
    if not keyword_candidates:
        return {"core": [], "introduced": []}

    core: list[str] = []
    introduced: list[str] = []
    seen: set[str] = set()
    candidate_by_key = {
        _norm_text(str(candidate.get("keyword") or "").strip()): candidate
        for candidate in keyword_candidates
        if str(candidate.get("keyword") or "").strip()
    }

    def is_role_candidate(keyword: str, candidate: dict | None) -> bool:
        if not _is_valid_concept(keyword):
            return False
        if not candidate:
            return True
        if candidate.get("operational_score_cap") is not None:
            return False
        if _to_float(candidate.get("code_like_penalty")) >= 0.75:
            return False
        if (
            _to_float(candidate.get("example_penalty")) >= 0.65
            and _to_float(candidate.get("graph_importance")) < 0.30
        ):
            return False
        return True

    if core_keywords:
        for keyword in core_keywords:
            keyword = str(keyword or "").strip()
            key = _norm_text(keyword)
            if not keyword or not key or key in seen:
                continue
            if not is_role_candidate(keyword, candidate_by_key.get(key)):
                continue
            core.append(keyword)
            seen.add(key)
            if len(core) >= 12:
                break

    for candidate in keyword_candidates:
        keyword = str(candidate.get("keyword") or "").strip()
        key = _norm_text(keyword)
        if not keyword or not key or key in seen:
            continue
        if not is_role_candidate(keyword, candidate):
            seen.add(key)
            continue
        if core_keywords:
            if len(introduced) < 40:
                introduced.append(keyword)
            seen.add(key)
            continue
        score = _to_float(candidate.get("score"))
        graph_importance = _to_float(candidate.get("graph_importance"))
        community_inner = _to_float(candidate.get("community_inner_representative"))
        example_penalty = _to_float(candidate.get("example_penalty"))
        operational_cap = candidate.get("operational_score_cap")
        code_penalty = _to_float(candidate.get("code_like_penalty"))

        core_like = (
            score >= 0.30
            and graph_importance >= 0.20
            and community_inner >= 0.20
            and example_penalty < 0.35
            and operational_cap is None
            and code_penalty < 0.75
        )
        if core_like and len(core) < 12:
            core.append(keyword)
        elif len(introduced) < 40:
            introduced.append(keyword)
        seen.add(key)

    if not core:
        for candidate in keyword_candidates[:5]:
            keyword = str(candidate.get("keyword") or "").strip()
            key = _norm_text(keyword)
            if keyword and key and key not in seen:
                core.append(keyword)
                seen.add(key)

    return {"core": core, "introduced": introduced[:40]}


def build_graph_role_scores(
    keyword_candidates: list[dict],
) -> tuple[dict[str, float], dict[str, float], dict[str, float], dict[str, float], dict[str, float]]:
    """GraphRAG 후보 breakdown을 기존 metadata score/norm shape에 맞춘다."""
    scored: dict[str, float] = {}
    norm_slide_freq: dict[str, float] = {}
    norm_trans_freq: dict[str, float] = {}
    norm_emph: dict[str, float] = {}
    norm_cent: dict[str, float] = {}

    for candidate in keyword_candidates:
        keyword = str(candidate.get("keyword") or "").strip()
        if not keyword:
            continue
        scored[keyword] = _to_float(candidate.get("score"))
        norm_slide_freq[keyword] = _to_float(candidate.get("slide_grounding"))
        norm_trans_freq[keyword] = _to_float(candidate.get("transcript_grounding"))
        norm_emph[keyword] = _to_float(candidate.get("emphasis_boost"))
        norm_cent[keyword] = _to_float(candidate.get("graph_importance"))

    return scored, norm_slide_freq, norm_trans_freq, norm_emph, norm_cent


def _graph_relation_type(rel: dict) -> str:
    haystack = _norm_text(" ".join([
        str(rel.get("source") or ""),
        str(rel.get("target") or ""),
        str(rel.get("description") or ""),
    ]))
    if any(term in haystack for term in ("비교", "대조", "차이", "vs", "versus")):
        return "CONTRASTS_WITH"
    return "RELATED_TO"


def build_graph_concept_relations(
    artifacts: dict,
    concept_roles: dict[str, list[str]],
    alias_map: dict[str, str] | None = None,
    limit: int = 120,
) -> list[dict]:
    """GraphRAG relationships에서 metadata concept_relations를 구성한다."""
    relationships = artifacts.get("graphrag", {}).get("relationships", {}).get("rows", []) or []
    alias_map = alias_map or {}
    role_names = {
        str(name).strip()
        for names in concept_roles.values()
        for name in (names or [])
        if str(name).strip()
    }
    if not relationships or not role_names:
        return []

    selected = []
    seen: set[tuple[str, str, str]] = set()
    for rel in sorted(
        relationships,
        key=lambda row: (
            _to_float(row.get("weight"))
            + math.log1p(max(_to_float(row.get("combined_degree")), 0.0))
        ),
        reverse=True,
    ):
        source = _canonicalize_graph_name(str(rel.get("source") or "").strip(), alias_map)
        target = _canonicalize_graph_name(str(rel.get("target") or "").strip(), alias_map)
        if not source or not target or source == target:
            continue
        if source not in role_names or target not in role_names:
            continue
        rel_type = _graph_relation_type(rel)
        dedupe_key = (source, rel_type, target)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        selected.append({
            "from": source,
            "type": rel_type,
            "to": target,
            "weight": round(_to_float(rel.get("weight")), 4),
        })
        if len(selected) >= limit:
            break

    return selected
