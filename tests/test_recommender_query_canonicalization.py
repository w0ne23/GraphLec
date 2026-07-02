import sys
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1] / "app" / "backend"
sys.path.insert(0, str(_BACKEND))

import recommender.recommender as recmod
from recommender.recommender import (
    CommunityIndex,
    LectureMetadata,
    QueryContext,
    Recommender,
    RecommenderConfig,
    _build_query_concept_index,
    _build_lexical_stats,
    _compute_graph_score,
    _direct_match_score,
    _build_reason,
    _required_subject_match_type,
    _soft_threshold_similarity,
)
from recommender.display import _display_score
from recommender.scoring import _topic_centrality_profile, _topic_score_cap
from recommender.query import _fallback_query_analysis
from recommender.utils import _query_term_base


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

    def test_keyword_vector_soft_decay_avoids_threshold_cliff(self):
        self.assertEqual(_soft_threshold_similarity(0.44, 0.60, 0.45), 0.0)
        self.assertGreater(_soft_threshold_similarity(0.55, 0.60, 0.45), 0.0)
        self.assertEqual(_soft_threshold_similarity(0.60, 0.60, 0.45), 0.60)

    def test_user_reason_hides_internal_fragmentation_penalty(self):
        reason = _build_reason(
            {
                "sim_keyword": 0.7,
                "dm_keyword": 0.2,
                "frag_penalty": 0.75,
            }
        )

        self.assertIn("키워드", reason)
        self.assertNotIn("파편화", reason)

    def test_display_score_has_no_free_condition_points_without_conditions(self):
        score = _display_score(
            0.5,
            "related",
            {
                "content_score": 0.4,
                "vec_score": 0.4,
                "graph_score": 0.0,
                "sim_keyword": 0.0,
                "combined_boost": 1.0,
                "duration_score": 0.0,
                "duration_mismatch": False,
                "condition_warnings": [],
            },
        )

        self.assertLess(score, 70)

    def test_topic_centrality_distinguishes_summary_mention_from_core(self):
        summary_only = _lecture(
            summary="운영체제의 역사에서 커널이라는 용어가 잠깐 언급된다.",
            keywords=[{"keyword": "운영체제", "score": 1.0}],
            concept_roles={"core": ["운영체제"], "introduced": []},
            concept_relations=[],
            communities=[],
        )
        core = _lecture(
            summary="커널 공간과 사용자 공간을 설명한다.",
            keywords=[{"keyword": "커널", "score": 1.0}],
            concept_roles={"core": ["커널"], "introduced": []},
            concept_relations=[],
            communities=[],
        )

        weak_profile = _topic_centrality_profile(summary_only, {"커널"})
        strong_profile = _topic_centrality_profile(core, {"커널"})

        self.assertEqual(weak_profile["topic_match_level"], "summary")
        self.assertLess(_topic_score_cap(**{
            "level": weak_profile["topic_match_level"],
            "centrality": weak_profile["topic_centrality"],
            "all_terms_matched": weak_profile["topic_all_terms_matched"],
        }), 0.5)
        self.assertEqual(strong_profile["topic_match_level"], "core")
        self.assertEqual(_topic_score_cap(
            strong_profile["topic_match_level"],
            strong_profile["topic_centrality"],
            strong_profile["topic_all_terms_matched"],
        ), 1.0)

    def test_ranking_caps_summary_only_subject_match(self):
        lec = _lecture(
            video_id="os",
            summary="운영체제 강의에서 커널이라는 용어를 배경으로 언급한다.",
            keywords=[{"keyword": "운영체제", "score": 1.0}],
            concept_roles={"core": ["운영체제"], "introduced": []},
            concept_relations=[],
            communities=[],
        )
        recommender = Recommender.__new__(Recommender)
        recommender.cfg = RecommenderConfig()
        recommender.collection = _MemoryCollection([lec])
        recommender._row_by_video_id = {}

        def fake_score_candidate(*_args, **_kwargs):
            return {"score": 0.9}

        recommender._score_candidate = fake_score_candidate
        ctx = QueryContext(
            query="커널 관련 강의 있어?",
            intent="recommend",
            search_text="커널",
            query_keywords=["커널"],
            inferred_keywords=["운영체제"],
            raw_query_keywords=["커널"],
            raw_inferred_keywords=["운영체제"],
            canonical_matches={},
            unmatched_query_terms=[],
            domain=None,
            subdomain=None,
            focus_concept=None,
            duration_max_sec=None,
            query_type="topic_browse",
            query_specificity="broad",
            comparison_intent=False,
            issue_free_preference=False,
            visual_preference=False,
            application_preference=False,
            listenability_preference=False,
            slow_speech_preference=False,
            recency_preference=False,
        )

        with redirect_stdout(StringIO()):
            ranked = recommender._rank_candidates(["os"], ctx, [])

        detail = ranked[0][1]
        self.assertEqual(detail["topic_match_level"], "summary")
        self.assertTrue(detail["topic_cap_applied"])
        self.assertLessEqual(detail["score"], 0.42)

    def test_weak_related_candidate_is_shown_with_low_score(self):
        lec = _lecture(
            video_id="os",
            summary="운영체제는 하드웨어 자원을 관리한다.",
            keywords=[{"keyword": "운영체제", "score": 1.0}],
            concept_roles={"core": ["운영체제"], "introduced": []},
            concept_relations=[],
            communities=[],
        )
        recommender = Recommender.__new__(Recommender)
        recommender.cfg = RecommenderConfig()
        recommender.collection = _MemoryCollection([lec])

        ctx = QueryContext(
            query="커널 관련 강의 있어?",
            intent="recommend",
            search_text="커널",
            query_keywords=["커널"],
            inferred_keywords=[],
            raw_query_keywords=["커널"],
            raw_inferred_keywords=[],
            canonical_matches={},
            unmatched_query_terms=[],
            domain=None,
            subdomain=None,
            focus_concept=None,
            duration_max_sec=None,
            query_type="topic_browse",
            query_specificity="broad",
            comparison_intent=False,
            issue_free_preference=False,
            visual_preference=False,
            application_preference=False,
            listenability_preference=False,
            slow_speech_preference=False,
            recency_preference=False,
        )
        detail = {
            "score": 0.30,
            "content_score": 0.30,
            "dm_keyword": 0.0,
            "sim_keyword": 0.0,
            "graph_score": 0.0,
            "community_score": 0.0,
            "domain_score": 0.0,
            "depth_score": 0.0,
            "contrast_bonus": 0.0,
            "application_preference": False,
            "listenability_preference": False,
            "slow_speech_preference": False,
            "recency_preference": False,
            "visual_preference": False,
            "duration_score": 0.0,
            "duration_mismatch": False,
            "condition_warnings": [],
            "combined_boost": 0.0,
            "frag_penalty": 0.0,
            "topic_match_level": "none",
            "required_subject_match": "none",
        }

        with redirect_stdout(StringIO()):
            results = recommender._classify_tiers([(lec, detail)], top_k=3)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].tier, "related")
        self.assertEqual(results[0].score, 0.30)
        self.assertEqual(results[0].score_detail["tier_reason"], "topic_none")

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
            query_type="topic_browse",
            query_specificity="broad",
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
