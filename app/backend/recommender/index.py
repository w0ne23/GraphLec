"""
recommender/index.py
────────────────────
MetadataCollection, CommunityIndex, 인덱스 빌더, DB 로딩.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import numpy as np

from recommender.config import _client, EMBED_MODEL
from recommender.types import (
    CommunityReportDocument,
    LectureLexicalDocument,
    LectureMetadata,
    LexicalStats,
    QueryConceptIndex,
    VectorSearchIndex,
)
from recommender.utils import (
    _add_lexical_variants,
    _canonical_domain,
    _community_report_terms,
    _database_url_sync,
    _normalize_matrix,
    _normalize_subdomain,
    _normalize_term,
    _term_lookup_keys,
    _tokenize_text,
)


# ── 임베딩 ────────────────────────────────────────────────────────────────────

def _embed(text: str) -> list[float]:
    text = text.strip() or " "
    response = _client.models.embed_content(
        model=EMBED_MODEL,
        contents=text,
    )
    return response.embeddings[0].values


# ── DB 로딩 ───────────────────────────────────────────────────────────────────

def _video_id_from_db_row(row: dict) -> str:
    metadata_uri = str(row.get("metadata_uri") or "").strip()
    if metadata_uri:
        name = Path(metadata_uri).name
        if name.endswith("_metadata.json"):
            return name[: -len("_metadata.json")]
        if name:
            return Path(name).stem
    video_path = str(row.get("video_path") or "").strip()
    if video_path:
        return Path(video_path).stem
    return str(row.get("lecture_id") or "")


def _fetch_lecture_metadata_rows(database_url: str) -> list[dict]:
    import psycopg2
    from psycopg2.extras import RealDictCursor

    sql = """
        SELECT
            lm.lecture_id,
            CASE
                WHEN lm.title IS NULL
                  OR btrim(lm.title) = ''
                  OR lm.title = 'Untitled lecture'
                THEN l.title
                ELSE lm.title
            END AS title,
            lm.instructor_id,
            lm.domain,
            lm.graph_domain,
            lm.graph_subdomain,
            lm.difficulty,
            lm.summary,
            lm.learning_objectives,
            lm.keywords,
            lm.concept_roles,
            lm.concept_relations,
            lm.communities,
            lm.visual_concept_terms,
            lm.pedagogy,
            lm.diagnostics,
            lm.duration_sec,
            lm.uploaded_at,
            lm.metadata_version,
            lm.metadata_uri,
            lm.created_at,
            lm.updated_at,
            l.video_path,
            l.created_at AS lecture_created_at
        FROM lecture_metadata lm
        JOIN lectures l ON l.id = lm.lecture_id
        WHERE l.is_published IS TRUE
        ORDER BY lm.updated_at DESC NULLS LAST, lm.created_at DESC NULLS LAST
    """

    conn = psycopg2.connect(database_url)
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql)
            return [dict(row) for row in cur.fetchall()]
    finally:
        conn.close()


# ── MetadataCollection ────────────────────────────────────────────────────────

class MetadataCollection:
    def __init__(self, _metadata_dir: str | None = None):
        # metadata_dir is accepted for backward compatibility. Runtime serving
        # uses the DB as its declared source and does not read artifact files.
        self.lectures: dict[str, LectureMetadata] = {}
        self._load_db()

    def _load_db(self):
        database_url = _database_url_sync()
        if not database_url:
            print("[DB 로드] DATABASE_URL 없음 — 추천 가능한 강의 메타데이터를 로드하지 못했습니다.\n")
            return

        try:
            rows = _fetch_lecture_metadata_rows(database_url)
        except Exception as exc:
            print(f"[DB 로드] lecture_metadata 로드 실패 — 추천 가능한 강의 메타데이터를 로드하지 못했습니다: {exc}\n")
            return

        for row in rows:
            lec = self._from_db_row(row)
            self.lectures[lec.video_id] = lec
        print(
            f"[DB 로드] published lecture_metadata {len(rows)}개 로드 "
            f"— 총 {len(self.lectures)}개\n"
        )

    @staticmethod
    def _from_db_row(row: dict) -> LectureMetadata:
        uploaded_at = row.get("uploaded_at") or row.get("lecture_created_at")
        if hasattr(uploaded_at, "isoformat"):
            uploaded_at = uploaded_at.isoformat()
        return LectureMetadata(
            video_id           = _video_id_from_db_row(row),
            title              = row.get("title") or "Untitled lecture",
            instructor_id      = row.get("instructor_id") or "",
            uploaded_at        = uploaded_at,
            domain             = _canonical_domain(row.get("graph_domain") or row.get("domain")),
            graph_subdomain    = _normalize_subdomain(row.get("graph_subdomain")),
            difficulty         = row.get("difficulty") or "unknown",
            duration_sec       = row.get("duration_sec") or 0.0,
            summary            = row.get("summary") or "",
            keywords           = row.get("keywords") or [],
            concept_roles      = row.get("concept_roles") or {},
            concept_relations  = row.get("concept_relations") or [],
            communities        = row.get("communities") or [],
            pedagogy           = row.get("pedagogy") or {},
            diagnostics        = row.get("diagnostics") or {},
            visual_concept_terms = row.get("visual_concept_terms") or [],
        )

    def get(self, video_id: str) -> Optional[LectureMetadata]:
        return self.lectures.get(video_id)

    def all(self) -> list[LectureMetadata]:
        return list(self.lectures.values())

    def available_domains(self) -> list[str]:
        return sorted({lec.domain for lec in self.lectures.values()})

    def available_subdomains(self) -> list[str]:
        return sorted({lec.graph_subdomain for lec in self.lectures.values() if lec.graph_subdomain})

    def available_keywords(self) -> list[str]:
        return sorted({
            k["keyword"]
            for lec in self.lectures.values()
            for k in lec.keywords
        })


# ── 어휘 통계 ─────────────────────────────────────────────────────────────────

def _keyword_terms(lec: LectureMetadata) -> list[tuple[str, float]]:
    terms = []
    for item in lec.keywords or []:
        term = _normalize_term(item.get("keyword", ""))
        if not term:
            continue
        try:
            weight = float(item.get("score", 1.0))
        except (TypeError, ValueError):
            weight = 1.0
        terms.append((term, max(weight, 0.0)))
    return terms


def _build_lexical_stats(lectures: list[LectureMetadata]) -> LexicalStats:
    documents: dict[str, LectureLexicalDocument] = {}
    doc_freq: defaultdict[str, int] = defaultdict(int)

    for lec in lectures:
        field_tf = {
            "title":   Counter(_tokenize_text(lec.title)),
            "keyword": Counter(),
            "summary": Counter(_tokenize_text(lec.summary)),
        }

        for term, weight in _keyword_terms(lec):
            _add_lexical_variants(field_tf["keyword"], term, weight)
            for token in _tokenize_text(term):
                field_tf["keyword"][token] += weight * 0.5
                for key in _term_lookup_keys(token):
                    if key != token:
                        field_tf["keyword"][key] += weight * 0.4

        _COMMUNITY_NODE_WEIGHT = 0.4
        for community in (lec.communities or []):
            for node in (community.get("nodes") or []):
                node_term = _normalize_term(node)
                if not node_term:
                    continue
                _add_lexical_variants(field_tf["keyword"], node_term, _COMMUNITY_NODE_WEIGHT)
                for token in _tokenize_text(node_term):
                    field_tf["keyword"][token] += _COMMUNITY_NODE_WEIGHT * 0.5
                    for key in _term_lookup_keys(token):
                        if key != token:
                            field_tf["keyword"][key] += _COMMUNITY_NODE_WEIGHT * 0.4

        term_tf = Counter()
        for field_counter in field_tf.values():
            term_tf.update(field_counter)

        doc_len = float(sum(term_tf.values()))
        documents[lec.video_id] = LectureLexicalDocument(
            video_id = lec.video_id,
            field_tf = field_tf,
            term_tf  = term_tf,
            doc_len  = doc_len,
        )

        for term in term_tf:
            doc_freq[term] += 1

    total_docs = len(documents)
    safe_total = max(total_docs, 1)
    irf = {
        term: math.log(safe_total / max(df, 1))
        for term, df in doc_freq.items()
    }
    bm25_idf = {
        term: math.log(1.0 + (safe_total - df + 0.5) / (df + 0.5))
        for term, df in doc_freq.items()
    }
    avg_doc_len = (
        sum(doc.doc_len for doc in documents.values()) / total_docs
        if total_docs else 0.0
    )

    return LexicalStats(
        total_docs  = total_docs,
        doc_freq    = dict(doc_freq),
        irf         = irf,
        bm25_idf    = bm25_idf,
        documents   = documents,
        avg_doc_len = avg_doc_len,
    )


def _build_vector_search_index(index_rows: list[dict]) -> VectorSearchIndex:
    video_ids = [row["video_id"] for row in index_rows]
    if not index_rows:
        empty = np.empty((0, 0), dtype=np.float32)
        return VectorSearchIndex(video_ids, empty, empty, empty)

    title_matrix = np.array(
        [row["title_vec"] for row in index_rows], dtype=np.float32
    )
    keyword_matrix = np.array(
        [row["keyword_vec"] for row in index_rows], dtype=np.float32
    )
    summary_matrix = np.array(
        [row["summary_vec"] for row in index_rows], dtype=np.float32
    )

    return VectorSearchIndex(
        video_ids      = video_ids,
        title_matrix   = _normalize_matrix(title_matrix),
        keyword_matrix = _normalize_matrix(keyword_matrix),
        summary_matrix = _normalize_matrix(summary_matrix),
    )


# ── CommunityIndex ────────────────────────────────────────────────────────────

class CommunityIndex:
    """
    metadata.communities 기반 reranking 보조 신호.
    community 정보가 없는 강의는 0.0으로 동작해 기존 데모 강의에는 영향을 주지 않는다.
    """

    def __init__(self, lectures: list[LectureMetadata]):
        self.reports_by_video_id: dict[str, list[CommunityReportDocument]] = {}
        self._load(lectures)

    def _load(self, lectures: list[LectureMetadata]) -> None:
        report_count = 0
        for lec in lectures:
            reports = []
            for row in lec.communities or []:
                if not isinstance(row, dict):
                    continue
                title = str(row.get("title") or "")
                summary = str(row.get("summary") or "")
                if not title and not summary:
                    continue

                try:
                    rank = float(row.get("rank") or 0.0)
                except (TypeError, ValueError):
                    rank = 0.0
                if not math.isfinite(rank):
                    rank = 0.0

                try:
                    level = int(row.get("level") or 0)
                except (TypeError, ValueError):
                    level = 0

                reports.append(CommunityReportDocument(
                    video_id      = lec.video_id,
                    title_terms   = _community_report_terms(title),
                    summary_terms = _community_report_terms(summary),
                    rank          = rank,
                    level         = level,
                ))

            if reports:
                self.reports_by_video_id[lec.video_id] = reports
                report_count += len(reports)

        print(
            f"[Community] metadata 기반 {len(self.reports_by_video_id)}개 강의, "
            f"{report_count}개 report 로드\n"
        )

    @staticmethod
    def _coverage(query_terms: Counter, report_terms: Counter) -> float:
        if not query_terms or not report_terms:
            return 0.0
        denom = sum(max(weight, 0.0) for weight in query_terms.values()) or 1.0
        hit = sum(
            max(weight, 0.0)
            for term, weight in query_terms.items()
            if report_terms.get(term, 0.0) > 0
        )
        return min(hit / denom, 1.0)

    def score(self, video_id: str, query_terms: Counter) -> float:
        reports = self.reports_by_video_id.get(video_id)
        if not reports or not query_terms:
            return 0.0

        best = 0.0
        for report in reports:
            title_overlap = self._coverage(query_terms, report.title_terms)
            summary_overlap = self._coverage(query_terms, report.summary_terms)
            base = 0.60 * title_overlap + 0.40 * summary_overlap
            rank_norm = min(max(report.rank / 10.0, 0.0), 1.0)
            # level 0=최상위(광역) → 강의 대주제 표현력 높음, 낮을수록 importance 증가
            level_signal = max(1.0 - report.level * 0.25, 0.25)
            importance = 0.70 * rank_norm + 0.30 * level_signal
            rank_weight = 0.85 + 0.15 * importance
            best = max(best, base * rank_weight)

        return round(min(best, 1.0), 4)


# ── QueryConceptIndex 빌더 ────────────────────────────────────────────────────

def _metadata_concept_terms(lec: LectureMetadata) -> list[tuple[str, float]]:
    from recommender.utils import _append_terms
    terms: list[tuple[str, float]] = []

    for item in lec.keywords or []:
        if not isinstance(item, dict):
            continue
        term = str(item.get("keyword") or "").strip()
        if not term:
            continue
        try:
            score = float(item.get("score", 1.0))
        except (TypeError, ValueError):
            score = 1.0
        terms.append((term, 3.0 + max(score, 0.0)))

    concept_roles = lec.concept_roles
    if isinstance(concept_roles, dict):
        for term in concept_roles.get("core", []) or []:
            terms.append((str(term), 3.0))
        for term in concept_roles.get("introduced", []) or []:
            terms.append((str(term), 1.5))
    elif isinstance(concept_roles, list):
        for item in concept_roles or []:
            if not isinstance(item, dict):
                continue
            weight = 3.0 if item.get("role") == "core" else 1.5
            terms.append((str(item.get("concept") or ""), weight))

    for relation in lec.concept_relations or []:
        if not isinstance(relation, dict):
            continue
        for key in ("from", "to", "source", "target"):
            value = str(relation.get(key) or "").strip()
            if value:
                terms.append((value, 1.25))

    for community in lec.communities or []:
        if not isinstance(community, dict):
            continue
        for key in ("title", "nodes"):
            values: list[str] = []
            _append_terms(values, community.get(key))
            for value in values:
                terms.append((value, 0.75))

    for term in lec.visual_concept_terms or []:
        terms.append((str(term), 0.75))

    return [(term, weight) for term, weight in terms if _normalize_term(term)]


def _build_query_concept_index(lectures: list[LectureMetadata]) -> QueryConceptIndex:
    alias_to_canonical: dict[str, str] = {}
    alias_weight: dict[str, float] = {}
    canonical_weight: defaultdict[str, float] = defaultdict(float)

    for lec in lectures:
        for term, weight in _metadata_concept_terms(lec):
            canonical = _normalize_term(term)
            if not canonical:
                continue
            canonical_weight[canonical] += weight
            for key in _term_lookup_keys(canonical):
                prev_weight = alias_weight.get(key, -1.0)
                prev = alias_to_canonical.get(key)
                if weight > prev_weight or (
                    weight == prev_weight and prev and len(canonical) < len(prev)
                ):
                    alias_to_canonical[key] = canonical
                    alias_weight[key] = weight

        # graph-first 전사/OCR 변형 alias (diagnostics.keyword_aliases) 반영
        if isinstance(lec.diagnostics, dict):
            for alias, canonical_raw in (lec.diagnostics.get("keyword_aliases") or {}).items():
                alias_norm = _normalize_term(alias)
                canonical_norm = _normalize_term(canonical_raw)
                if not alias_norm or not canonical_norm:
                    continue
                w = canonical_weight.get(canonical_norm, 0.5)
                for key in _term_lookup_keys(alias_norm):
                    if alias_weight.get(key, -1.0) < w:
                        alias_to_canonical[key] = canonical_norm
                        alias_weight[key] = w

    return QueryConceptIndex(
        alias_to_canonical=alias_to_canonical,
        canonical_weight=dict(canonical_weight),
    )
