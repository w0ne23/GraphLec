"""
recommender.py
──────────────
강의 추천 시스템 — 자연어 질의 기반

변경 이력:
  v3: Gemini가 메타데이터 keyword pool에서 직접 선택 (자유 생성 → vocabulary 고정)
  v4: prerequisite 타입 제거, focus_concept → depth_score
  v5: 인텐트별 동적 가중치, Min-Max 정규화, 파편화 패널티, MIN_SCORE 필터
  v6: 벡터 유사도 방식 전환 (ko-sroberta + LanceDB)
  v7: 필드별 분리 벡터 + Gemini search_text 변환
  v8: 직접 매칭(dm_score) 추가 — 벡터 유사도 보조
      content_score = VEC_BLEND×vec_score + (1-VEC_BLEND)×dm_score
  v9: 임베딩 모델 교체 (ko-sroberta → gemini-embedding-001)
      SentenceTransformer 제거, Gemini API 단일 클라이언트로 통합
      keyword vec threshold 필터 적용 (sim_keyword < 0.60 → 기여 0)

CLI 실행:
  python recommender.py --query "메모리 관리 방법 알고 싶어"
  python recommender.py --metadata_dir app/backend/metadata --query "운영체제란 무엇인가"

모듈로 사용:
  from recommender.recommender import Recommender
  rec = Recommender("app/backend/metadata")
  results = rec.recommend_from_query("프로세스 스케줄링 알고 싶어")
"""

import argparse
import math
from collections import Counter, defaultdict
from typing import Optional

import lancedb

from recommender.config import DEFAULT_METADATA_DIR, RecommenderConfig
from recommender.display import _build_reason, _display_score
from recommender.index import (
    CommunityIndex,
    MetadataCollection,
    _build_lexical_stats,
    _build_query_concept_index,
    _build_vector_search_index,
    _embed,
)
from recommender.query import _fast_list_query_analysis, analyze_query
from recommender.scoring import (
    _compute_contrast_signal,
    _compute_depth_score,
    _compute_fragmentation_penalty,
    _compute_graph_score,
    _compute_recency_score,
    _detect_comparison_intent,
    _direct_match_score,
    _direct_match_score_tfirf,
    _required_subject_match_type,
    _required_subject_terms,
    _topic_centrality_profile,
    _topic_score_cap,
)
from recommender.types import (
    LectureLexicalDocument,
    LectureMetadata,
    QueryContext,
    RecommendResult,
)
from recommender.utils import (
    _add_lexical_variants,
    _append_topic_expansions,
    _cosine_sim,
    _infer_domain_filters,
    _normalize_subdomain,
    _normalize_term,
    _normalize_vector,
    _soft_threshold_similarity,
    _term_lookup_keys,
    _tokenize_text,
)


# ============================================================================
#  추천 엔진
# ============================================================================

class Recommender:
    def __init__(self, config: Optional[RecommenderConfig] = None):
        self.collection          = MetadataCollection()
        self.cfg                 = config or RecommenderConfig()
        self._available_domains  = self.collection.available_domains()
        self._available_subdomains = self.collection.available_subdomains()
        self._available_keywords = self.collection.available_keywords()

        # LanceDB 전체 레코드 사전 로드 (요청마다 디스크 읽기 방지).
        # 테이블이 아직 생성되지 않은 초기 상태에서는 벡터 검색만 비활성화하고,
        # metadata 기반 BM25/직접매칭/그래프 점수로 추천을 계속 제공한다.
        print("[LanceDB 레코드 로드 중...]")
        self._index_rows = [
            row
            for row in self._load_lancedb_rows()
            if self.collection.get(str(row.get("video_id") or "")) is not None
        ]
        self._row_by_video_id = {
            row["video_id"]: row
            for row in self._index_rows
        }
        self._vector_search_index = _build_vector_search_index(self._index_rows)
        indexed_lectures = [
            lec
            for video_id in self._row_by_video_id
            if (lec := self.collection.get(video_id)) is not None
        ]
        metadata_lectures = self.collection.all()
        scoring_lectures = indexed_lectures if indexed_lectures else metadata_lectures
        self._concept_index = _build_query_concept_index(metadata_lectures)
        self._lexical_stats = _build_lexical_stats(scoring_lectures)
        self._community_index = CommunityIndex(scoring_lectures)
        print(f"  → {len(self._index_rows)}개 레코드 로드\n")
        if not self._index_rows and metadata_lectures:
            print("  ⚠ LanceDB lectures 테이블 없음/비어 있음 — 벡터 검색 없이 metadata 기반 추천으로 동작합니다.\n")
        elif len(indexed_lectures) < len(metadata_lectures):
            print(
                f"  ⚠ 벡터 인덱스에 없는 metadata 강의 "
                f"{len(metadata_lectures) - len(indexed_lectures)}개는 metadata 기반으로만 점수화합니다.\n"
            )
        print(f"[도메인]    {self._available_domains}")
        print(f"[세부도메인] {self._available_subdomains}")
        print(f"[키워드 풀] {len(self._available_keywords)}개\n")
        print(
            f"[Lexical]   {len(self._lexical_stats.doc_freq)}개 term, "
            f"avg_len={self._lexical_stats.avg_doc_len:.1f}\n"
        )
        print(f"[ConceptIndex] {len(self._concept_index.alias_to_canonical)}개 alias key\n")
        print(f"[Vector]    matrix rows={len(self._vector_search_index.video_ids)}\n")

    def _load_lancedb_rows(self) -> list[dict]:
        try:
            db = lancedb.connect(self.cfg.DB_DIR)
            table_names = set(db.table_names())
            if "lectures" not in table_names:
                print(f"  ⚠ LanceDB 테이블 없음: lectures ({self.cfg.DB_DIR})")
                return []
            table = db.open_table("lectures")
            return table.to_arrow().to_pylist()
        except Exception as exc:
            print(f"  ⚠ LanceDB 로드 실패: {exc}")
            return []

    def _canonicalize_query_terms(
        self,
        terms: list[str],
        existing: Optional[set[str]] = None,
    ) -> tuple[list[str], dict[str, str], list[str]]:
        canonical_terms: list[str] = []
        matches: dict[str, str] = {}
        unmatched: list[str] = []
        seen = set(existing or set())

        for raw in terms or []:
            normalized_raw = _normalize_term(raw)
            canonical, matched = self._concept_index.canonicalize(normalized_raw)
            if not canonical:
                continue
            if matched:
                matches[normalized_raw] = canonical
            else:
                unmatched.append(canonical)
            if canonical not in seen:
                seen.add(canonical)
                canonical_terms.append(canonical)

        return canonical_terms, matches, unmatched

    def _prepare_query_context(self, query: str) -> QueryContext:
        print(f"[질의 분석] {query}")
        fast_analysis = _fast_list_query_analysis(query, self._available_domains, self._available_subdomains)
        if fast_analysis:
            intent, search_text, query_keywords, inferred_keywords, domain, focus_concept, duration_max_sec, conditions = fast_analysis
        else:
            intent, search_text, query_keywords, inferred_keywords, domain, focus_concept, duration_max_sec, conditions = analyze_query(
                query, self._available_domains, self._available_keywords
            )
        raw_query_keywords = list(query_keywords)
        raw_inferred_keywords = list(inferred_keywords)
        raw_focus_concept = focus_concept
        query_keywords, query_matches, query_unmatched = self._canonicalize_query_terms(query_keywords)
        inferred_keywords, inferred_matches, inferred_unmatched = self._canonicalize_query_terms(
            inferred_keywords,
            existing=set(query_keywords),
        )
        canonical_matches = {**query_matches, **inferred_matches}
        unmatched_query_terms = query_unmatched + [
            term for term in inferred_unmatched if term not in query_unmatched
        ]
        if focus_concept:
            focus_concept_canonical, focus_matched = self._concept_index.canonicalize(focus_concept)
            if focus_matched:
                canonical_matches[_normalize_term(raw_focus_concept)] = focus_concept_canonical
                focus_concept = focus_concept_canonical
            elif focus_concept_canonical:
                focus_concept = focus_concept_canonical
            else:
                focus_concept = None
        query_type        = conditions.get("query_type", "topic_browse")
        query_specificity = conditions.get("query_specificity", "broad")
        subdomain = _normalize_subdomain(conditions.get("subdomain"))
        if not subdomain:
            inferred_domain, inferred_subdomain = _infer_domain_filters(
                query,
                self._available_domains,
                self._available_subdomains,
            )
            domain = inferred_domain or domain
            subdomain = _normalize_subdomain(inferred_subdomain)
        if subdomain not in self._available_subdomains:
            subdomain = None
        inferred_keywords = _append_topic_expansions(
            query_keywords,
            inferred_keywords,
            seed_terms=[query],
        )
        inferred_keywords, expansion_matches, expansion_unmatched = self._canonicalize_query_terms(
            inferred_keywords,
            existing=set(query_keywords),
        )
        canonical_matches.update(expansion_matches)
        for term in expansion_unmatched:
            if term not in unmatched_query_terms:
                unmatched_query_terms.append(term)
        search_text = " ".join(query_keywords + inferred_keywords) or search_text or query
        comparison_intent = _detect_comparison_intent(query_keywords, query)
        issue_free_preference = bool(conditions.get("issue_free"))
        visual_preference = bool(conditions.get("prefers_visual"))
        application_preference = bool(conditions.get("prefers_application"))
        listenability_preference = bool(conditions.get("prefers_listenability"))
        slow_speech_preference = bool(conditions.get("prefers_slow_speech"))
        recency_preference = bool(conditions.get("prefers_recency"))

        print(f"[질의 의도]   {intent}")
        print(f"[LLM 원본]    {raw_query_keywords}")
        print(f"[원본 키워드] {query_keywords}")
        print(f"[LLM 확장]    {raw_inferred_keywords}")
        print(f"[확장 키워드] {inferred_keywords}")
        if canonical_matches:
            print(f"[정규화 매칭] {canonical_matches}")
        if unmatched_query_terms:
            print(f"[미매칭 용어] {unmatched_query_terms}")
        print(f"[추론 도메인] {domain or '미확정'}")
        print(f"[추론 세부]   {subdomain or '미확정'}")
        print(f"[깊이 개념]   {focus_concept or '없음'}")
        print(f"[비교 의도]   {'있음' if comparison_intent else '없음'}")
        print(f"[검증 조건]   {'있음' if issue_free_preference else '없음'}")
        print(f"[시각 선호]   {'있음' if visual_preference else '없음'}")
        print(f"[적용/시연]   {'있음' if application_preference else '없음'}")
        print(f"[청취 품질]   {'있음' if listenability_preference else '없음'}")
        print(f"[발화 속도]   {'빠르지 않음 선호' if slow_speech_preference else '없음'}")
        print(f"[최신성]     {'최근 업로드 선호' if recency_preference else '없음'}")
        if duration_max_sec:
            print(f"[질의 유형]   {query_type} / {query_specificity}")
        print(f"[길이 조건]   기준 {duration_max_sec//60}분 ({duration_max_sec}초)" if duration_max_sec else "")
        print()

        return QueryContext(
            query             = query,
            intent            = intent,
            search_text       = search_text,
            query_keywords    = query_keywords,
            inferred_keywords = inferred_keywords,
            raw_query_keywords = raw_query_keywords,
            raw_inferred_keywords = raw_inferred_keywords,
            canonical_matches = canonical_matches,
            unmatched_query_terms = unmatched_query_terms,
            domain            = domain,
            subdomain         = subdomain,
            focus_concept     = focus_concept,
            duration_max_sec  = duration_max_sec,
            comparison_intent = comparison_intent,
            issue_free_preference = issue_free_preference,
            visual_preference = visual_preference,
            application_preference = application_preference,
            listenability_preference = listenability_preference,
            slow_speech_preference = slow_speech_preference,
            recency_preference = recency_preference,
            query_type        = query_type,
            query_specificity = query_specificity,
        )

    def _all_candidate_ids(self) -> list[str]:
        return [lec.video_id for lec in self.collection.all()]

    def _rrf_fuse(
        self,
        rankings: list[list[str]],
        top_n: Optional[int] = None,
    ) -> list[str]:
        scores: defaultdict[str, float] = defaultdict(float)
        best_rank: dict[str, int] = {}

        for ranking in rankings:
            for rank, video_id in enumerate(ranking, start=1):
                scores[video_id] += 1.0 / (self.cfg.RRF_K + rank)
                best_rank[video_id] = min(best_rank.get(video_id, rank), rank)

        fused = sorted(
            scores,
            key=lambda video_id: (
                -scores[video_id],
                best_rank.get(video_id, 10**9),
                video_id,
            ),
        )
        limit = top_n or self.cfg.HYBRID_CANDIDATE_TOP_N
        return fused[:limit]

    def _metadata_preferred_ids(self, ctx: QueryContext) -> Optional[set[str]]:
        """
        metadata 조건을 만족하는 우선 후보군.
        hard filter가 아니라 RRF에 preferred ranking을 추가하는 soft signal로만 쓴다.
        """
        if not self.cfg.USE_METADATA_PREFILTER:
            return None
        if not (ctx.domain or ctx.subdomain or ctx.duration_max_sec):
            return None

        preferred = set()
        for lec in self.collection.all():
            if ctx.domain and lec.domain != ctx.domain:
                continue
            if ctx.subdomain and lec.graph_subdomain != ctx.subdomain:
                continue
            if ctx.duration_max_sec and lec.duration_sec > (
                ctx.duration_max_sec + self.cfg.METADATA_DURATION_GRACE_SEC
            ):
                continue

            preferred.add(lec.video_id)

        return preferred or None

    def _get_initial_candidate_ids(
        self, ctx: QueryContext, query_vec: list[float]
    ) -> tuple[list[str], set[str]]:
        """
        BM25 lexical 후보와 vector semantic 후보를 RRF로 통합한다.
        반환: (candidate_ids, bm25_matched_ids)
        """
        preferred_ids = self._metadata_preferred_ids(ctx)
        bm25_candidates = self._retrieve_bm25_candidates(
            ctx,
            top_n=self.cfg.BM25_RETRIEVE_TOP_N,
        )
        vector_candidates = self._retrieve_vector_candidates(
            query_vec,
            top_n=self.cfg.VECTOR_RETRIEVE_TOP_N,
        )
        rankings = [
            [video_id for video_id, _ in bm25_candidates],
            [video_id for video_id, _ in vector_candidates],
        ]

        pref_bm25_candidates = []
        pref_vector_candidates = []
        if preferred_ids:
            pref_bm25_candidates = self._retrieve_bm25_candidates(
                ctx,
                eligible_ids=preferred_ids,
                top_n=self.cfg.BM25_RETRIEVE_TOP_N,
            )
            pref_vector_candidates = self._retrieve_vector_candidates(
                query_vec,
                eligible_ids=preferred_ids,
                top_n=self.cfg.VECTOR_RETRIEVE_TOP_N,
            )
            rankings.extend([
                [video_id for video_id, _ in pref_bm25_candidates],
                [video_id for video_id, _ in pref_vector_candidates],
            ])

        bm25_ids: set[str] = {vid for vid, _ in bm25_candidates}
        if pref_bm25_candidates:
            bm25_ids |= {vid for vid, _ in pref_bm25_candidates}

        candidate_ids = self._rrf_fuse(
            rankings,
            top_n=self.cfg.HYBRID_CANDIDATE_TOP_N,
        )

        print(
            f"[후보 검색] BM25 {len(bm25_candidates)}개, "
            f"Vector {len(vector_candidates)}개, RRF {len(candidate_ids)}개"
        )
        if preferred_ids:
            print(
                f"[메타데이터 우선 후보] {len(preferred_ids)}개 "
                f"(BM25 {len(pref_bm25_candidates)}개, "
                f"Vector {len(pref_vector_candidates)}개)"
            )

        return candidate_ids, bm25_ids

    def _query_lexical_terms(self, ctx: QueryContext) -> Counter:
        """
        BM25/TF-IRF에서 공유할 질의 term 구성.
        원본 키워드는 강하게, LLM 확장 키워드는 약하게 반영한다.
        """
        terms: Counter = Counter()

        def add_term(text: str, weight: float) -> None:
            phrase = _normalize_term(text)
            if not phrase:
                return
            _add_lexical_variants(terms, phrase, weight)
            for token in _tokenize_text(phrase):
                if token != phrase:
                    terms[token] += weight * 0.5
                    for key in _term_lookup_keys(token):
                        if key != token:
                            terms[key] += weight * 0.4

        for keyword in ctx.query_keywords:
            add_term(keyword, 1.0)
        for keyword in ctx.inferred_keywords:
            add_term(keyword, 0.5)

        if not terms:
            for token in _tokenize_text(ctx.search_text or ctx.query):
                terms[token] += 1.0

        return terms

    def _resolve_rerank_weights(self, ctx: QueryContext) -> dict[str, float]:
        if ctx.visual_preference:
            return {
                "content": self.cfg.W_CONTENT_VISUAL_QUERY,
                "graph": self.cfg.W_GRAPH_VISUAL_QUERY,
                "visual": self.cfg.W_VISUAL_QUERY,
                "boost": self.cfg.W_BOOST_VISUAL_QUERY,
            }
        return {
            "content": self.cfg.W_CONTENT,
            "graph": self.cfg.W_GRAPH,
            "visual": self.cfg.W_VISUAL,
            "boost": self.cfg.W_BOOST,
        }

    @staticmethod
    def _visual_density_score(lec: LectureMetadata) -> float:
        pedagogy = lec.pedagogy or {}
        teaching_style = (lec.diagnostics or {}).get("teaching_style", {})

        def as_float(value) -> float:
            try:
                parsed = float(value)
                return parsed if math.isfinite(parsed) else 0.0
            except (TypeError, ValueError):
                return 0.0

        visual_ratio = as_float(teaching_style.get("visual_ratio", pedagogy.get("visual_ratio")))
        structure_ratio = as_float(pedagogy.get("structure_ratio"))
        return round(min(max(0.5 * visual_ratio + 0.5 * structure_ratio, 0.0), 1.0), 4)

    @staticmethod
    def _diagnostic_score(lec: LectureMetadata, section: str, key: str) -> float:
        value = ((lec.diagnostics or {}).get(section, {}) or {}).get(key)
        try:
            parsed = float(value)
            return min(max(parsed, 0.0), 1.0) if math.isfinite(parsed) else 0.0
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _speech_rate_fit_score(lec: LectureMetadata) -> float:
        delivery = ((lec.diagnostics or {}).get("delivery", {}) or {})
        try:
            spm = float(delivery.get("speech_rate_spm"))
        except (TypeError, ValueError):
            return 0.0
        if not math.isfinite(spm) or spm <= 0:
            return 0.0
        # "천천히/빠르지 않게" 조건은 낮은 SPM일수록 더 적합하게 본다.
        # 임계값은 더미 데이터 분포 기준 임시값이며, 실제 데이터 분포로 재보정한다.
        if spm <= 240:
            return 1.0
        if spm >= 330:
            return 0.0
        return round(1.0 - ((spm - 240) / 90), 4)

    def _recency_boost_weight(self, domain: str) -> float:
        normalized = _normalize_term(domain)
        if normalized == "engineering":
            return self.cfg.W_RECENCY_BOOST_FAST
        if normalized in {"social_science", "health_sciences"}:
            return self.cfg.W_RECENCY_BOOST_MEDIUM
        return self.cfg.W_RECENCY_BOOST_SLOW

    @staticmethod
    def _visual_concept_score(lec: LectureMetadata, query_terms: Counter) -> float:
        if not query_terms:
            return 0.0
        visual_terms = set()
        for term in lec.visual_concept_terms or []:
            normalized = _normalize_term(term)
            if not normalized:
                continue
            visual_terms.add(normalized)
            visual_terms.update(_tokenize_text(normalized))
        if not visual_terms:
            return 0.0

        denom = sum(max(weight, 0.0) for weight in query_terms.values()) or 1.0
        hit = 0.0
        for term, weight in query_terms.items():
            if term in visual_terms:
                hit += max(weight, 0.0)
        return round(min(hit / denom, 1.0), 4)

    def _compute_visual_score(
        self,
        lec: LectureMetadata,
        ctx: QueryContext,
        query_terms: Counter,
    ) -> tuple[float, float, float]:
        density_score = self._visual_density_score(lec)
        concept_score = self._visual_concept_score(lec, query_terms)
        if ctx.visual_preference:
            visual_score = 0.65 * density_score + 0.35 * concept_score
        else:
            visual_score = 0.5 * density_score
        return (
            round(min(max(visual_score, 0.0), 1.0), 4),
            density_score,
            concept_score,
        )

    def _bm25_tf(self, doc: LectureLexicalDocument, term: str) -> float:
        return (
            self.cfg.BM25_TITLE_WEIGHT   * doc.field_tf["title"].get(term, 0.0) +
            self.cfg.BM25_KEYWORD_WEIGHT * doc.field_tf["keyword"].get(term, 0.0) +
            self.cfg.BM25_SUMMARY_WEIGHT * doc.field_tf["summary"].get(term, 0.0)
        )

    def _bm25_score_document(self, doc: LectureLexicalDocument, query_terms: Counter) -> float:
        stats = self._lexical_stats
        avg_len = stats.avg_doc_len or 1.0
        doc_len = doc.doc_len or avg_len
        norm = self.cfg.BM25_K1 * (
            1.0 - self.cfg.BM25_B + self.cfg.BM25_B * (doc_len / avg_len)
        )

        score = 0.0
        for term, query_weight in query_terms.items():
            tf = self._bm25_tf(doc, term)
            if tf <= 0:
                continue
            idf = stats.bm25_idf.get(term, 0.0)
            score += query_weight * idf * (
                tf * (self.cfg.BM25_K1 + 1.0)
                / (tf + norm)
            )
        return score

    def _retrieve_bm25_candidates(
        self,
        ctx: QueryContext,
        eligible_ids: Optional[set[str]] = None,
        top_n: int = 50,
    ) -> list[tuple[str, float]]:
        """
        lexical 후보 검색기. vector 후보와 RRF로 통합해 1차 후보군을 만든다.
        """
        query_terms = self._query_lexical_terms(ctx)
        if not query_terms:
            return []

        scores = []
        for video_id, doc in self._lexical_stats.documents.items():
            if eligible_ids is not None and video_id not in eligible_ids:
                continue
            score = self._bm25_score_document(doc, query_terms)
            if score > 0:
                scores.append((video_id, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_n]

    def _retrieve_vector_candidates(
        self,
        query_vec: list[float],
        eligible_ids: Optional[set[str]] = None,
        top_n: Optional[int] = None,
    ) -> list[tuple[str, float]]:
        """
        semantic 후보 검색기. 기존 vec_score와 같은 필드 가중치로 top-N을 뽑는다.
        """
        index = self._vector_search_index
        if not index.video_ids or not query_vec:
            return []

        q = _normalize_vector(query_vec)
        sim_title = index.title_matrix @ q
        sim_keyword = index.keyword_matrix @ q
        sim_summary = index.summary_matrix @ q
        vec_scores = (
            self.cfg.W_TITLE   * sim_title +
            self.cfg.W_KEYWORD * sim_keyword +
            self.cfg.W_SUMMARY * sim_summary
        )

        limit = top_n or self.cfg.VECTOR_RETRIEVE_TOP_N
        scored = []
        for idx, video_id in enumerate(index.video_ids):
            if eligible_ids is not None and video_id not in eligible_ids:
                continue
            score = float(vec_scores[idx])
            if score > 0:
                scored.append((video_id, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:limit]

    def _score_candidate(
        self,
        row: Optional[dict],
        lec: LectureMetadata,
        ctx: QueryContext,
        query_vec: list[float],
        query_concepts: set[str],
        query_terms: Counter,
        weights: dict[str, float],
    ) -> dict:
        # ── 길이 조건 boost + warning ─────────────────────────────
        # 내용 적합성을 깎지 않고, 요청 길이(+grace)를 만족하는 후보만 boost한다.
        condition_warnings: list[str] = []
        if ctx.duration_max_sec:
            duration_fit_score = (
                1.0
                if lec.duration_sec <= ctx.duration_max_sec + self.cfg.METADATA_DURATION_GRACE_SEC
                else 0.0
            )
            if lec.duration_sec > ctx.duration_max_sec:
                over_min = math.ceil((lec.duration_sec - ctx.duration_max_sec) / 60.0)
                condition_warnings.append(f"요청한 길이보다 약 {over_min}분 깁니다.")
        else:
            duration_fit_score = 0.0

        # 필드별 코사인 유사도. LanceDB row가 없으면 벡터 성분만 0으로 둔다.
        has_vector_row = row is not None and bool(query_vec)
        if has_vector_row:
            sim_title   = _cosine_sim(query_vec, row["title_vec"])
            sim_keyword = _cosine_sim(query_vec, row["keyword_vec"])
            sim_summary = _cosine_sim(query_vec, row["summary_vec"])

            # keyword vec soft threshold
            sim_keyword_filtered = float(
                _soft_threshold_similarity(
                    sim_keyword,
                    self.cfg.KW_VEC_THRESHOLD,
                    self.cfg.KW_VEC_SOFT_FLOOR,
                )
            )
            vec_score = (
                self.cfg.W_TITLE   * sim_title            +
                self.cfg.W_KEYWORD * sim_keyword_filtered +
                self.cfg.W_SUMMARY * sim_summary
            )
        else:
            sim_title = sim_keyword = sim_summary = sim_keyword_filtered = vec_score = 0.0

        # 직접 토큰 매칭 — 원본 키워드 100%, 추론 키워드 50% 반영
        if self.cfg.USE_TF_IRF_DM:
            dm = _direct_match_score_tfirf(
                ctx.query_keywords,
                ctx.inferred_keywords,
                lec,
                self._lexical_stats,
                idf_floor=self.cfg.TF_IRF_IDF_FLOOR,
            )
        else:
            dm = _direct_match_score(ctx.query_keywords, ctx.inferred_keywords, lec)
        dm_score = (
            self.cfg.W_TITLE   * dm["title"]   +
            self.cfg.W_KEYWORD * dm["keyword"]  +
            self.cfg.W_SUMMARY * dm["summary"]
        )

        # 블렌딩 — content 최대 0.85로 제한 (boost 여유 확보)
        vec_blend = self.cfg.VEC_BLEND if has_vector_row else 0.0
        content_score = min(
            vec_blend       * vec_score +
            (1 - vec_blend) * dm_score,
            0.85
        )

        # domain boost 신호
        domain_score = 0.0
        subdomain_score = 0.0
        if ctx.domain and lec.domain == ctx.domain:
            domain_score = 1.0
        if ctx.subdomain and lec.graph_subdomain == ctx.subdomain:
            subdomain_score = 1.0
            domain_score = 1.0

        # depth boost 신호 — BFS 홉 거리 기반
        depth_score = _compute_depth_score(ctx.focus_concept, lec) if ctx.focus_concept else 0.0

        graph_score = _compute_graph_score(
            lec,
            query_concepts,
            core_weight=self.cfg.GRAPH_CORE_WEIGHT,
            intro_weight=self.cfg.GRAPH_INTRO_WEIGHT,
        )
        community_score = self._community_index.score(
            lec.video_id,
            query_terms,
        )
        visual_score, visual_density_score, visual_concept_score = self._compute_visual_score(
            lec,
            ctx,
            query_terms,
        )
        application_score = self._diagnostic_score(
            lec,
            "teaching_style",
            "application_orientation_score",
        )
        listenability_score = self._diagnostic_score(
            lec,
            "delivery",
            "listenability_score",
        )
        speech_rate_score = self._speech_rate_fit_score(lec)
        recency_score = _compute_recency_score(
            lec.uploaded_at,
            half_life_days=self.cfg.RECENCY_HALF_LIFE_DAYS,
        )
        recency_weight = (
            self._recency_boost_weight(lec.domain)
            if ctx.recency_preference and lec.uploaded_at
            else 0.0
        )
        if ctx.visual_preference and visual_density_score < 0.3:
            condition_warnings.append("시각 자료 비중이 높지 않습니다.")
        if ctx.application_preference and application_score < 0.2:
            condition_warnings.append("예제/시연 지향 신호가 약합니다.")
        if ctx.listenability_preference and listenability_score < 0.4:
            condition_warnings.append("청취 품질 신호가 높지 않습니다.")
        if ctx.slow_speech_preference and speech_rate_score < 0.3:
            condition_warnings.append("발화 속도가 빠른 편일 수 있습니다.")

        # ── 가중합 구조 점수 ──────────────────────────────────
        MAX_BOOST = (
            (self.cfg.W_DOMAIN_BOOST if ctx.domain or ctx.subdomain else 0.0) +
            (self.cfg.W_DEPTH_BOOST if ctx.focus_concept else 0.0) +
            (self.cfg.W_DURATION_BOOST if ctx.duration_max_sec else 0.0) +
            (self.cfg.W_APPLICATION_BOOST if ctx.application_preference else 0.0) +
            (self.cfg.W_LISTENABILITY_BOOST if ctx.listenability_preference else 0.0) +
            (self.cfg.W_SPEECH_RATE_BOOST if ctx.slow_speech_preference else 0.0) +
            recency_weight
        )
        raw_boost = (
            self.cfg.W_DOMAIN_BOOST     * (subdomain_score or domain_score) +
            self.cfg.W_DEPTH_BOOST      * depth_score +
            (self.cfg.W_DURATION_BOOST * duration_fit_score if ctx.duration_max_sec else 0.0) +
            (self.cfg.W_APPLICATION_BOOST * application_score if ctx.application_preference else 0.0) +
            (self.cfg.W_LISTENABILITY_BOOST * listenability_score if ctx.listenability_preference else 0.0) +
            (self.cfg.W_SPEECH_RATE_BOOST * speech_rate_score if ctx.slow_speech_preference else 0.0) +
            recency_weight * recency_score
        )
        boost_signal = raw_boost / MAX_BOOST if MAX_BOOST > 0 else 0.0

        # 파편화 패널티
        frag = _compute_fragmentation_penalty(lec.concept_roles)

        total = max(
            weights["content"] * content_score
            + weights["graph"] * graph_score
            + weights["visual"] * visual_score
            + weights["boost"] * boost_signal
            - self.cfg.FRAG_PENALTY_WEIGHT * frag,
            0.0
        )

        # ── 비교 분석형 보너스 ────────────────────────────────
        # 비교 의도 질의 + 강의 내 대조형 relation 비율 → 가산
        contrast_signal = _compute_contrast_signal(lec)
        contrast_bonus  = (
            self.cfg.W_CONTRAST_BOOST * contrast_signal
            if ctx.comparison_intent else 0.0
        )
        total = min(total + contrast_bonus, 1.0)

        # ── 패널티 체계 ───────────────────────────────────────
        # 도메인 상위 카테고리 불일치 패널티
        if ctx.subdomain and lec.graph_subdomain != ctx.subdomain:
            if ctx.query_type != "topic_browse":
                total *= self.cfg.DOMAIN_MISMATCH_PENALTY
        elif ctx.domain:
            if ctx.domain != lec.domain:
                total *= self.cfg.DOMAIN_MISMATCH_PENALTY

        return {
            "score":                round(total, 4),
            "raw_query_keywords":   ctx.raw_query_keywords,
            "raw_inferred_keywords": ctx.raw_inferred_keywords,
            "canonical_query_keywords": ctx.query_keywords,
            "canonical_inferred_keywords": ctx.inferred_keywords,
            "canonical_matches":    ctx.canonical_matches,
            "unmatched_query_terms": ctx.unmatched_query_terms,
            "graph_query_concepts": sorted(query_concepts),
            "content_score":        round(content_score, 4),
            "content_pct":          round(content_score * 100, 1),
            "vec_score":            round(vec_score, 4),
            "dm_score":             round(dm_score, 4),
            "sim_title":            round(sim_title, 4),
            "sim_keyword":          round(sim_keyword, 4),
            "sim_keyword_filtered": round(sim_keyword_filtered, 4),
            "sim_summary":          round(sim_summary, 4),
            "dm_title":             round(dm["title"], 4),
            "dm_keyword":           round(dm["keyword"], 4),
            "dm_summary":           round(dm["summary"], 4),
            "domain_score":         round(domain_score, 4),
            "subdomain_score":      round(subdomain_score, 4),
            "q_kw_matched":         dm.get("q_kw_matched", True),
            "domain_mismatch":      bool(
                (ctx.subdomain and ctx.subdomain != lec.graph_subdomain)
                or (ctx.domain and ctx.domain != lec.domain)
            ),
            "graph_score":          round(graph_score, 4),
            "community_score":      round(community_score, 4),
            "visual_score":         round(visual_score, 4),
            "visual_density_score": round(visual_density_score, 4),
            "visual_concept_score": round(visual_concept_score, 4),
            "visual_preference":    ctx.visual_preference,
            "application_score":    round(application_score, 4),
            "application_preference": ctx.application_preference,
            "listenability_score":  round(listenability_score, 4),
            "listenability_preference": ctx.listenability_preference,
            "speech_rate_score":    round(speech_rate_score, 4),
            "slow_speech_preference": ctx.slow_speech_preference,
            "recency_score":        round(recency_score, 4),
            "recency_preference":   ctx.recency_preference,
            "recency_weight":       round(recency_weight, 4),
            "weight_content":       round(weights["content"], 4),
            "weight_graph":         round(weights["graph"], 4),
            "weight_visual":        round(weights["visual"], 4),
            "weight_boost":         round(weights["boost"], 4),
            "depth_score":          round(depth_score, 4),
            "contrast_signal":      round(contrast_signal, 4),
            "contrast_bonus":       round(contrast_bonus, 4),
            "boost_signal":         round(boost_signal, 4),
            "combined_boost":       round(boost_signal, 4),
            "duration_score":       round(duration_fit_score, 4),
            "duration_mismatch":    bool(ctx.duration_max_sec and lec.duration_sec > ctx.duration_max_sec),
            "condition_warnings":   condition_warnings,
            "frag_penalty":         round(frag, 4),
        }

    def _rank_candidates(
        self,
        candidate_ids: list[str],
        ctx: QueryContext,
        query_vec: list[float],
        bm25_ids: Optional[set[str]] = None,
    ) -> list[tuple[LectureMetadata, dict]]:
        query_concepts = set(ctx.query_keywords) | set(ctx.inferred_keywords)
        query_terms = self._query_lexical_terms(ctx)
        required_subject_terms = _required_subject_terms(ctx)
        weights = self._resolve_rerank_weights(ctx)
        print(
            "[Rerank weights] "
            f"content={weights['content']:.2f} graph={weights['graph']:.2f} "
            f"visual={weights['visual']:.2f} boost={weights['boost']:.2f}"
        )
        candidates = []
        for video_id in candidate_ids:
            row = self._row_by_video_id.get(video_id)
            lec = self.collection.get(video_id)
            if lec is None:
                continue
            subject_match_type = _required_subject_match_type(lec, required_subject_terms)
            detail = self._score_candidate(
                row,
                lec,
                ctx,
                query_vec,
                query_concepts,
                query_terms,
                weights,
            )
            detail["required_subject_terms"] = sorted(required_subject_terms)
            detail["required_subject_match"] = subject_match_type
            topic_profile = _topic_centrality_profile(
                lec,
                required_subject_terms or query_concepts,
            )
            detail.update(topic_profile)
            if subject_match_type == "none":
                detail["score"] = round(detail["score"] * self.cfg.Q_KW_MISMATCH_PENALTY, 4)
                detail["subject_mismatch_penalty"] = self.cfg.Q_KW_MISMATCH_PENALTY
            else:
                detail["subject_mismatch_penalty"] = 1.0
            if required_subject_terms:
                cap = _topic_score_cap(
                    detail["topic_match_level"],
                    detail["topic_centrality"],
                    detail["topic_all_terms_matched"],
                )
                detail["topic_score_cap"] = cap
                if detail["score"] > cap:
                    detail["score"] = round(cap, 4)
                    detail["topic_cap_applied"] = True
                else:
                    detail["topic_cap_applied"] = False
            else:
                detail["topic_score_cap"] = 1.0
                detail["topic_cap_applied"] = False
            if (
                ctx.query_specificity == "specific"
                and bm25_ids is not None
                and video_id not in bm25_ids
            ):
                detail["score"] = round(detail["score"] * self.cfg.SPECIFICITY_BM25_MISS_PENALTY, 4)
                detail["specificity_penalty_applied"] = True
            else:
                detail["specificity_penalty_applied"] = False
            if (
                ctx.query_type == "concept_depth"
                and detail.get("topic_match_level") not in {"core", "keyword_high"}
            ):
                detail["score"] = round(detail["score"] * self.cfg.CONCEPT_DEPTH_WEAK_TOPIC_PENALTY, 4)
                detail["concept_depth_penalty_applied"] = True
            else:
                detail["concept_depth_penalty_applied"] = False
            candidates.append((lec, detail))

        candidates.sort(key=lambda x: x[1]["score"], reverse=True)
        return candidates

    def _print_candidate_scores(self, candidates: list[tuple[LectureMetadata, dict]]) -> None:
        print(f"  {'video_id':<10} {'제목':<24} {'sim_t':>5} {'sim_k':>5} {'sim_k*':>6} {'sim_s':>5} "
              f"{'vec':>5} {'dm':>5} {'graph':>5} {'comm':>5} {'vis':>5} {'boost':>6} {'ctr':>5} {'dur':>5} {'→score':>7}")
        for lec, d in candidates:
            flags = ""
            if d.get("domain_mismatch"):          flags += " !dom"
            if not d.get("q_kw_matched", True):   flags += " !qkw"
            if d.get("contrast_bonus", 0) > 0:    flags += f" +ctr{d['contrast_bonus']:.2f}"
            print(f"  {lec.video_id:<10} {lec.title[:22]:<24} "
                  f"{d['sim_title']:>5.3f} {d['sim_keyword']:>5.3f} "
                  f"{d['sim_keyword_filtered']:>6.3f} {d['sim_summary']:>5.3f} "
                  f"{d['vec_score']:>5.3f} {d['dm_score']:>5.3f} "
                  f"{d['graph_score']:>5.3f} {d['community_score']:>5.3f} "
                  f"{d['visual_score']:>5.3f} "
                  f"{d['boost_signal']:>6.3f} {d['contrast_signal']:>5.3f} "
                  f"{d['duration_score']:>5.3f} {d['score']:>7.3f}{flags}")

    def _classify_tiers(
        self,
        candidates: list[tuple[LectureMetadata, dict]],
        top_k: int,
        min_score: Optional[float] = None,
    ) -> list[RecommendResult]:
        if not candidates:
            return []

        effective_min = min_score if min_score is not None else self.cfg.ABS_MIN_SCORE
        max_score = max(d["score"] for _, d in candidates)
        if max_score < effective_min:
            print(f"  → top 점수 {max_score:.3f} < ABS_MIN_SCORE {effective_min} — 결과 없음")
            return []

        results = []
        for lec, detail in candidates:
            score = detail["score"]
            if score < effective_min:
                continue

            topic_level = detail.get("topic_match_level", "none")
            tier = "direct" if topic_level in {"core", "keyword_high"} else "related"
            detail["tier_reason"] = f"topic_{topic_level}"
            detail["duration_sec"] = lec.duration_sec
            results.append(RecommendResult(
                video_id     = lec.video_id,
                title        = lec.title,
                domain       = lec.domain,
                instructor   = lec.instructor_id,
                score        = detail["score"],
                display_score= _display_score(detail["score"], tier, detail),
                duration_sec = lec.duration_sec,
                score_detail = detail,
                reason       = _build_reason(detail, tier),
                summary      = lec.summary,
                tier         = tier,
                keywords     = lec.keywords or [],
            ))
            if len(results) >= top_k:
                break

        return results

    def recommend_from_query(
        self,
        query:     str,
        top_k:     int             = 5,
        min_score: Optional[float] = None,
    ) -> list[RecommendResult]:
        """
        자연어 질의를 분석하고 관련 강의를 추천한다.

        v12 흐름 (BM25/vector 후보 검색 + RRF 통합):
          1. analyze_query — search_text / domain / focus_concept 추출
          2. 비교 의도 감지 (_detect_comparison_intent)
          3. search_text → Gemini embedding-001 벡터화
          4. BM25 후보 + vector 후보 → RRF 통합
          5. RRF candidate_ids에 대해:
             - vec_score / dm_score 계산
             - depth_score: BFS 홉 거리 기반 (concept_relations 활용)
             - contrast_bonus: 비교 의도 × 대조형 relation 비율
          6. 패널티 체계:
             - dm_keyword==0 패널티 (×0.6)
             - 원본 query_keyword 완전 미매칭 패널티 (×Q_KW_MISMATCH_PENALTY)
             - 도메인 상위 카테고리 불일치 패널티 (×DOMAIN_MISMATCH_PENALTY)
          7. 이중 레이어 임계값 + gap 자연 경계 → direct/related 분류
        """
        ctx = self._prepare_query_context(query)

        # ── 질의 벡터화 ───────────────────────────────────────────────
        query_vec = []
        if self._vector_search_index.video_ids:
            try:
                query_vec = _embed(ctx.search_text)
            except Exception as exc:
                print(f"  ⚠ 질의 임베딩 실패 — BM25/metadata 기반으로 계속 진행: {exc}")
        candidate_ids, bm25_ids = self._get_initial_candidate_ids(ctx, query_vec)
        candidates = self._rank_candidates(candidate_ids, ctx, query_vec, bm25_ids)

        # ── condition_first: content 필터 후 duration 재정렬 ─────────
        if ctx.query_type == "condition_first" and ctx.duration_max_sec:
            candidates = [
                (lec, d) for lec, d in candidates
                if d.get("content_score", 0.0) >= self.cfg.CONDITION_FIRST_MIN_CONTENT
            ]
            candidates.sort(key=lambda x: x[0].duration_sec)

        # ── 전체 후보 점수 출력 (디버그) ─────────────────────────────
        self._print_candidate_scores(candidates)
        effective_min_score = (
            min_score if min_score is not None
            else self.cfg.RELATED_SEARCH_ABS_MIN_SCORE if ctx.query_type == "related_search"
            else self.cfg.ABS_MIN_SCORE
        )
        return self._classify_tiers(candidates, top_k, min_score=effective_min_score)

    def print_results(self, results: list[RecommendResult], title: str = "추천 결과", top_k: int = 3):
        top = results[:top_k]

        print(f"\n{'='*62}")
        print(f"  {title}")
        print(f"{'='*62}")

        if not top:
            print("  ※ 질의와 일치하는 강의를 찾지 못했습니다.")
        else:
            print(f"  ▶ 추천 강의 (Top {len(top)})")
            print(f"  {'-'*58}")
            for i, r in enumerate(top, 1):
                d         = r.score_detail
                score_100 = round(r.score * 100, 1)
                t_pct     = round(d.get("sim_title",   0) * 100, 1)
                k_pct     = round(d.get("sim_keyword", 0) * 100, 1)
                s_pct     = round(d.get("sim_summary", 0) * 100, 1)
                dm_t      = round(d.get("dm_title",    0) * 100, 1)
                dm_k      = round(d.get("dm_keyword",  0) * 100, 1)
                dm_s      = round(d.get("dm_summary",  0) * 100, 1)
                boost_pct    = round(d.get("boost_signal", 0.0) * 100, 1)
                depth_pct    = round(d.get("depth_score",   0.0) * 100, 1)
                duration_pct = round(d.get("duration_score", 0.0) * 100, 1)
                frag_pct     = round(d.get("frag_penalty", 0.0) * 100, 1)
                tier_label = {
                    "direct": "✅ 직접 추천",
                    "related": "🔸 간접 관련",
                }.get(r.tier, r.tier)
                print(f"  {i}. [{r.video_id}] {r.title}  {tier_label}")
                print(f"     점수:   {score_100}점  |  {r.reason}")
                print(f"     벡터:   title {t_pct}점  keyword {k_pct}점  summary {s_pct}점")
                print(f"     직접:   title {dm_t}점  keyword {dm_k}점  summary {dm_s}점")
                print(f"     boost:  combined {boost_pct}%  depth {depth_pct}%  duration {duration_pct}%")
                if d.get("contrast_bonus", 0) > 0:
                    print(f"     대조형: contrast_signal {d['contrast_signal']:.2f}  bonus +{d['contrast_bonus']:.3f}")
                if frag_pct > 0:
                    print(f"     파편화 패널티: -{frag_pct}%")
                print(f"     요약:   {r.summary[:80]}...")
                print()

        print(f"  {'─'*62}")
        print(f"  ▶ 추천된 강의 점수 전체 (이중 레이어 필터 적용됨)")
        print(f"  {'─'*62}")
        print(f"  {'video_id':<12} {'제목':<22} {'도메인':<14} {'티어':<8} {'점수':>6}")
        print(f"  {'─'*62}")
        for r in results:
            tier_label = {
                "direct": "직접",
                "related": "간접",
            }.get(r.tier, r.tier)
            marker = " ◀" if r in top else ""
            print(f"  {r.video_id:<12} {r.title[:20]:<22} {r.domain:<14} "
                  f"{tier_label:<8} {round(r.score*100,1):>5.1f}점{marker}")
        print()


# ============================================================================
#  CLI
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="강의 추천 시스템",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""예시:
  python recommender.py --query "메모리 관리 방법 알고 싶어"
  python recommender.py --metadata_dir app/backend/metadata --query "운영체제란 무엇인가"
  python recommender.py --query "운영체제 기능 알고 싶어" --min_score 0.1
"""
    )
    parser.add_argument("--metadata_dir", default=DEFAULT_METADATA_DIR,
                        help=f"메타데이터 디렉토리 (기본: {DEFAULT_METADATA_DIR})")
    parser.add_argument("--query", default=None,
                        help="자연어 질의 (예: '메모리 관리 알고 싶어')")
    parser.add_argument("--top_k", type=int, default=5,
                        help="추천 결과 수 (기본: 5)")
    parser.add_argument("--min_score", type=float, default=None,
                        help="ABS_MIN_SCORE 오버라이드 0~1 (기본: cfg.ABS_MIN_SCORE=0.30).")
    args = parser.parse_args()

    rec = Recommender()

    if args.query:
        results = rec.recommend_from_query(
            args.query, top_k=args.top_k, min_score=args.min_score
        )
        rec.print_results(results, f"질의 기반 추천: '{args.query}'", top_k=3)
    else:
        print("대화형 모드 (종료: q)\n")
        while True:
            try:
                query = input("질의 입력 > ").strip()
                if query.lower() in ("q", "quit", "exit"):
                    break
                if not query:
                    continue
                results = rec.recommend_from_query(
                    query, top_k=args.top_k, min_score=args.min_score
                )
                rec.print_results(results, f"질의: '{query}'", top_k=3)
            except KeyboardInterrupt:
                print("\n종료")
                break
