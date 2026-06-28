import unittest
from contextlib import redirect_stdout
from io import StringIO

import app.backend.recommender.recommender as recmod
from app.backend.recommender.recommender import (
    CommunityIndex,
    LectureMetadata,
    QueryContext,
    Recommender,
    RecommenderConfig,
    _build_query_concept_index,
    _build_lexical_stats,
    _compute_graph_score,
    _direct_match_score,
    _fallback_query_analysis,
    _query_term_base,
    _required_subject_match_type,
)


def _lecture(**overrides):
    base = {
        "video_id": "v1",
        "title": "운영체제 소개",
        "instructor_id": "i1",
        "uploaded_at": None,
        "domain": "engineering",
        "graph_subdomain": "computer_science",
        "difficulty": "unknown",
        "duration_sec": 600.0,
        "summary": "운영체제는 사용자와 하드웨어 사이에서 중계 역할을 한다.",
        "keywords": [{"keyword": "운영체제", "score": 1.0}],
        "concept_roles": {"core": ["운영체제"], "introduced": ["응용 소프트웨어"]},
        "concept_relations": [{"from": "운영체제", "to": "응용 소프트웨어", "type": "CONTRASTS_WITH"}],
        "communities": [{"title": "운영체제와 시스템 소프트웨어", "nodes": ["운영체제"]}],
        "pedagogy": {},
        "diagnostics": {},
        "visual_concept_terms": [],
    }
    base.update(overrides)
    return LectureMetadata(**base)


class _MemoryCollection:
    def __init__(self, lectures):
        self._lectures = {lec.video_id: lec for lec in lectures}

    def get(self, video_id):
        return self._lectures.get(video_id)


class RecommenderQueryCanonicalizationTest(unittest.TestCase):
    def test_query_meta_and_particle_stripping(self):
        self.assertEqual(_query_term_base("운영체제의 강의 추천해줘"), "운영체제")
        self.assertEqual(_query_term_base("운영 체제에 대해 알려줘"), "운영 체제")

    def test_metadata_concept_index_uses_automatic_terms(self):
        index = _build_query_concept_index([_lecture()])

        self.assertEqual(index.canonicalize("운영체제의")[0], "운영체제")
        self.assertEqual(index.canonicalize("응용소프트웨어")[0], "응용 소프트웨어")

    def test_metadata_concept_index_does_not_over_merge_substrings(self):
        index = _build_query_concept_index([_lecture()])

        canonical, matched = index.canonicalize("소프트웨어")

        self.assertEqual(canonical, "소프트웨어")
        self.assertFalse(matched)

    def test_direct_match_accepts_particle_variant(self):
        score = _direct_match_score(["운영체제의"], [], _lecture())

        self.assertGreater(score["keyword"], 0.0)
        self.assertTrue(score["q_kw_matched"])

    def test_graph_score_accepts_particle_variant(self):
        score = _compute_graph_score(_lecture(), {"운영체제의"})

        self.assertGreater(score, 0.0)

    def test_required_subject_match_is_canonical_not_hard_brittle(self):
        lec = _lecture()

        self.assertEqual(_required_subject_match_type(lec, {"운영체제의"}), "canonical")
        self.assertEqual(_required_subject_match_type(lec, {"네트워크"}), "none")

    def test_prepare_query_context_canonicalizes_llm_output(self):
        original = recmod.analyze_query

        def fake_analyze_query(query, available_domains, available_keywords):
            return (
                "recommend",
                query,
                ["운영체제의 강의"],
                ["응용소프트웨어"],
                None,
                None,
                None,
                {},
            )

        recmod.analyze_query = fake_analyze_query
        try:
            recommender = Recommender.__new__(Recommender)
            recommender._available_domains = []
            recommender._available_subdomains = []
            recommender._available_keywords = []
            recommender._concept_index = _build_query_concept_index([_lecture()])

            with redirect_stdout(StringIO()):
                ctx = recommender._prepare_query_context("운영체제의 강의 추천해줘")

            self.assertEqual(ctx.raw_query_keywords, ["운영체제의 강의"])
            self.assertEqual(ctx.query_keywords, ["운영체제"])
            self.assertEqual(ctx.inferred_keywords, ["응용 소프트웨어"])
            self.assertEqual(ctx.canonical_matches["운영체제의 강의"], "운영체제")
            self.assertEqual(ctx.canonical_matches["응용소프트웨어"], "응용 소프트웨어")
        finally:
            recmod.analyze_query = original

    def test_fallback_query_analysis_uses_metadata_keywords(self):
        result = _fallback_query_analysis(
            "운영체제의 강의 추천해줘",
            ["engineering"],
            ["운영체제", "응용 소프트웨어"],
        )

        self.assertEqual(result[0], "recommend")
        self.assertEqual(result[2], ["운영체제"])
        self.assertIn("운영체제", result[1])

    def test_ranking_keeps_subject_mismatch_soft_and_orders_canonical_match_first(self):
        os_lecture = _lecture(video_id="os")
        network_lecture = _lecture(
            video_id="net",
            title="네트워크 소개",
            summary="네트워크 프로토콜과 전송 계층을 설명한다.",
            keywords=[{"keyword": "네트워크", "score": 1.0}],
            concept_roles={"core": ["네트워크"], "introduced": ["프로토콜"]},
            concept_relations=[{"from": "네트워크", "to": "프로토콜", "type": "RELATED_TO"}],
            communities=[{"title": "네트워크와 프로토콜", "nodes": ["네트워크"]}],
        )
        lectures = [os_lecture, network_lecture]
        recommender = Recommender.__new__(Recommender)
        recommender.cfg = RecommenderConfig()
        recommender.collection = _MemoryCollection(lectures)
        recommender._row_by_video_id = {}
        recommender._lexical_stats = _build_lexical_stats(lectures)
        with redirect_stdout(StringIO()):
            recommender._community_index = CommunityIndex(lectures)

        ctx = QueryContext(
            query="운영체제의 강의 추천해줘",
            intent="recommend",
            search_text="운영체제",
            query_keywords=["운영체제"],
            inferred_keywords=[],
            raw_query_keywords=["운영체제의 강의"],
            raw_inferred_keywords=[],
            canonical_matches={"운영체제의 강의": "운영체제"},
            unmatched_query_terms=[],
            domain=None,
            subdomain=None,
            focus_concept=None,
            duration_max_sec=None,
            comparison_intent=False,
            issue_free_preference=False,
            visual_preference=False,
            application_preference=False,
            listenability_preference=False,
            slow_speech_preference=False,
            recency_preference=False,
        )

        with redirect_stdout(StringIO()):
            ranked = recommender._rank_candidates(["net", "os"], ctx, [])

        self.assertEqual(ranked[0][0].video_id, "os")
        self.assertEqual(ranked[0][1]["required_subject_match"], "canonical")
        self.assertEqual(ranked[1][1]["required_subject_match"], "none")
        self.assertGreater(ranked[0][1]["score"], ranked[1][1]["score"])


if __name__ == "__main__":
    unittest.main()
