"""
recommender/types.py
────────────────────
추천 시스템 데이터 모델 (dataclass).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Optional

import numpy as np

from recommender.utils import _query_term_base, _term_lookup_keys


@dataclass
class LectureMetadata:
    video_id:          str
    title:             str
    instructor_id:     str
    uploaded_at:       Optional[str]
    domain:            str
    graph_subdomain:   str
    # difficulty: DB 컬럼명 그대로 사용. 실제 의미는 개념 복잡도(낮을수록 쉬운 개념 구성)이며,
    # 실제 강의 난이도가 아니다. 프론트에서 이 필드를 쓸 때 주의.
    difficulty: str
    duration_sec:      float
    summary:           str
    keywords:          list[dict]
    concept_roles:     list[dict]
    concept_relations: list[dict]
    communities:       list[dict]
    pedagogy:          dict
    diagnostics:       dict
    visual_concept_terms: list[str]


@dataclass
class LectureLexicalDocument:
    video_id:  str
    field_tf:  dict[str, Counter]
    term_tf:   Counter
    doc_len:   float


@dataclass
class LexicalStats:
    total_docs:  int
    doc_freq:    dict[str, int]
    irf:         dict[str, float]
    bm25_idf:    dict[str, float]
    documents:   dict[str, LectureLexicalDocument]
    avg_doc_len: float


@dataclass
class VectorSearchIndex:
    video_ids:       list[str]
    title_matrix:    np.ndarray
    keyword_matrix:  np.ndarray
    summary_matrix:  np.ndarray


@dataclass
class CommunityReportDocument:
    video_id:    str
    title_terms: Counter
    summary_terms: Counter
    rank:        float
    level:       int


@dataclass
class QueryConceptIndex:
    """metadata에서 자동 수집한 concept lookup index."""
    alias_to_canonical: dict[str, str]
    canonical_weight:   dict[str, float]

    def canonicalize(self, term: str) -> tuple[str, bool]:
        base = _query_term_base(term)
        if not base:
            return "", False
        best = ""
        best_weight = -1.0
        for key in _term_lookup_keys(base):
            canonical = self.alias_to_canonical.get(key)
            if not canonical:
                continue
            weight = self.canonical_weight.get(canonical, 0.0)
            if weight > best_weight:
                best = canonical
                best_weight = weight
        return (best, True) if best else (base, False)


@dataclass
class RecommendResult:
    video_id:      str
    title:         str
    domain:        str
    instructor:    str
    score:         float
    display_score: Optional[int]
    duration_sec:  float
    score_detail:  dict
    reason:        str
    summary:       str
    tier:          str   # "direct" | "related"
    keywords:      list[dict]


@dataclass
class QueryContext:
    query:                  str
    intent:                 str
    search_text:            str
    query_keywords:         list[str]
    inferred_keywords:      list[str]
    raw_query_keywords:     list[str]
    raw_inferred_keywords:  list[str]
    canonical_matches:      dict[str, str]
    unmatched_query_terms:  list[str]
    domain:                 Optional[str]
    subdomain:              Optional[str]
    focus_concept:          Optional[str]
    duration_max_sec:       Optional[int]
    query_type:             str   # topic_browse | concept_depth | condition_first | related_search
    query_specificity:      str   # broad | specific
    comparison_intent:      bool
    issue_free_preference:  bool
    visual_preference:      bool
    application_preference: bool
    listenability_preference: bool
    slow_speech_preference: bool
    recency_preference:     bool
