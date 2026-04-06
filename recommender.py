"""
recommender.py
──────────────
강의 추천 시스템 — 자연어 질의 기반

변경 이력:
  v3: Gemini가 메타데이터 keyword pool에서 직접 선택 (자유 생성 → vocabulary 고정)
      inferred_keywords 제거, semantic gap 해소

CLI 실행:
  python recommender.py --query "메모리 관리 방법 알고 싶어"
  python recommender.py --metadata_dir metadata/ --query "운영체제란 무엇인가"

모듈로 사용:
  from recommender import Recommender
  rec = Recommender("metadata/")
  results = rec.recommend_from_query("프로세스 스케줄링 알고 싶어")
"""

import json
import os
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from google import genai
from dotenv import load_dotenv

load_dotenv()
_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY_2"))
MODEL = "gemini-2.5-flash"


# ============================================================================
#  질의 유형 정의
# ============================================================================
#
#  direct      : "운영체제 강의 추천해줘"
#                target_keywords = 관련 키워드 (pool에서 선택)
#                context_keywords = []
#
#  prerequisite: "머신러닝 배우기 전에 알아야 할 수학"
#                target_keywords = 먼저 들을 강의 키워드 (수학 관련)
#                context_keywords = 최종 목표 키워드 (머신러닝) — 해당 강의 penalty
#
#  followup    : "선형대수 들었는데 연계 강의 추천해줘"
#                target_keywords = 다음 단계 키워드 (pool에서 선택)
#                context_keywords = 이미 들은 강의 키워드 — 해당 강의 penalty
#
#  career      : "AI 연구자가 되고 싶어"
#                target_keywords = 역할에 필요한 기술 키워드 (pool에서 선택)
#                context_keywords = []

QUERY_TYPES = ("direct", "prerequisite", "followup", "career", "unknown")

# context_keywords 강의에 적용할 score 배율 (낮을수록 강하게 억제)
CONTEXT_PENALTY_BY_TYPE: dict[str, float] = {
    "prerequisite": 0.25,
    "followup":     0.15,
}


# ============================================================================
#  설정
# ============================================================================

@dataclass
class RecommenderConfig:
    # content 가중치 (합계 = 1.0)
    W_KEYWORD:      float = 0.50
    W_TITLE:        float = 0.35
    W_SUMMARY:      float = 0.15
    # domain: multiplicative boost
    # content_score × (1 + W_DOMAIN_BOOST × domain_score)
    # content가 0이면 domain도 0
    W_DOMAIN_BOOST: float = 0.20


# ============================================================================
#  데이터 모델
# ============================================================================

@dataclass
class LectureMetadata:
    video_id:      str
    title:         str
    instructor_id: str
    domain:        str
    duration_sec:  float
    summary:       str
    keywords:      list[dict]


# ============================================================================
#  메타데이터 컬렉션
# ============================================================================

class MetadataCollection:
    def __init__(self, metadata_dir: str):
        self.lectures: dict[str, LectureMetadata] = {}
        self._load(Path(metadata_dir))

    def _load(self, directory: Path):
        files = list(directory.glob("*_metadata.json"))
        if not files:
            raise FileNotFoundError(f"메타데이터 파일 없음: {directory}")
        for path in files:
            with open(path, encoding="utf-8") as f:
                raw = json.load(f)
            items = raw if isinstance(raw, list) else [raw]
            for item in items:
                lec = LectureMetadata(
                    video_id      = item["video_id"],
                    title         = item["title"],
                    instructor_id = item.get("instructor_id", ""),
                    domain        = item.get("domain", "unknown"),
                    duration_sec  = item.get("duration_sec", 0.0),
                    summary       = item.get("summary", ""),
                    keywords      = item.get("keywords", []),
                )
                self.lectures[lec.video_id] = lec
        print(f"[로드] {len(self.lectures)}개 강의 메타데이터 로드 완료\n")

    def get(self, video_id: str) -> Optional[LectureMetadata]:
        return self.lectures.get(video_id)

    def all(self) -> list[LectureMetadata]:
        return list(self.lectures.values())

    def available_domains(self) -> list[str]:
        """메타데이터에 실제 존재하는 도메인 목록"""
        return sorted({lec.domain for lec in self.lectures.values()})

    def available_keywords(self) -> list[str]:
        """메타데이터 전체 keyword 배열에서 unique 키워드 수집.

        Gemini가 이 목록에서만 선택하게 함으로써
        semantic gap(확률 vs 확률 분포)을 원천 제거.

        ※ 강의 수가 수천 개 이상으로 늘어나면 프롬프트 토큰 한계 도달
          → LanceDB 벡터 검색으로 교체 예정.
        """
        return sorted({
            k["keyword"]
            for lec in self.lectures.values()
            for k in lec.keywords
        })


# ============================================================================
#  유사도 계산
# ============================================================================

import re as _re


def _strip_josa(t: str) -> str:
    return _re.sub(r'(의|은|는|이|가|을|를|에서|에|로|으로|과|와|도|만|까지|부터)$', '', t)


def _kw_match(meta_kw: str, query_kws: list[str]) -> bool:
    """양방향 substring 매칭 (pool 선택 후 안전망 역할)."""
    for qk in query_kws:
        if qk in meta_kw or meta_kw in qk:
            return True
    return False


def _context_match(meta_kw: str, context_kws: list[str]) -> bool:
    """Context penalty 전용: prefix 방향만 허용.
    "고유벡터".startswith("벡터") → False (false positive 방지)
    "선형대수학".startswith("선형대수") → True
    """
    for qk in context_kws:
        if meta_kw == qk or meta_kw.startswith(qk):
            return True
    return False


def compute_query_similarity(
    target_keywords:  list[str],
    context_keywords: list[str],
    query_type:       str,
    query_domain:     Optional[str],
    target:           LectureMetadata,
    cfg:              RecommenderConfig,
) -> dict:
    """
    질의 기반 유사도 계산.

    content_score = W_KEYWORD × keyword_score
                  + W_TITLE   × title_score
                  + W_SUMMARY × summary_score

    score = content_score × (1 + W_DOMAIN_BOOST × domain_score)
    """
    if not target_keywords:
        return {"score": 0.0, "query_type": query_type,
                "domain_score": 0.0, "domain_boost": 1.0,
                "keyword_score": 0.0, "title_score": 0.0,
                "summary_score": 0.0, "context_penalty": False}

    domain_score = 1.0 if (query_domain and target.domain == query_domain) else 0.0

    kw_list = target.keywords
    total_w = sum(k["score"] for k in kw_list) or 1.0

    # keyword_score: target_keywords 가중합
    matched_w     = sum(k["score"] for k in kw_list if _kw_match(k["keyword"], target_keywords))
    keyword_score = matched_w / total_w

    # title_score: 제목 토큰 hit → summary 검증
    title_tokens = {_strip_josa(t) for t in target.title.replace(",", " ").replace("·", " ").split()}
    summary_lower = target.summary.lower()

    title_hit = 0.0
    for qk in target_keywords:
        if any(qk in tok or tok in qk for tok in title_tokens):
            title_hit += 1.0 if qk in summary_lower else 0.1
    title_score = title_hit / len(target_keywords)

    # summary_score: keyword 배열 가중합 + 직접 텍스트 히트 합산
    matched_sum_w    = sum(
        k["score"] for k in kw_list
        if _kw_match(k["keyword"], target_keywords) and k["keyword"] in summary_lower
    )
    kw_summary       = matched_sum_w / total_w
    direct_summary   = sum(1 for kw in target_keywords if kw in summary_lower) / len(target_keywords)
    summary_score    = min(kw_summary + direct_summary * 0.5, 1.0)

    # content + domain boost
    content_score = (
        cfg.W_KEYWORD * keyword_score
        + cfg.W_TITLE   * title_score
        + cfg.W_SUMMARY * summary_score
    )
    domain_boost = 1.0 + cfg.W_DOMAIN_BOOST * domain_score
    total        = content_score * domain_boost

    # context penalty
    context_penalized = False
    if context_keywords and query_type in CONTEXT_PENALTY_BY_TYPE:
        if any(_context_match(k["keyword"], context_keywords) for k in kw_list):
            total *= CONTEXT_PENALTY_BY_TYPE[query_type]
            context_penalized = True

    return {
        "score":           round(total, 4),
        "query_type":      query_type,
        "domain_score":    round(domain_score, 4),
        "domain_boost":    round(domain_boost, 4),
        "keyword_score":   round(keyword_score, 4),
        "title_score":     round(title_score, 4),
        "summary_score":   round(summary_score, 4),
        "context_penalty": context_penalized,
    }


# ============================================================================
#  Gemini — 자연어 질의 분석
# ============================================================================

def analyze_query(
    query:              str,
    available_domains:  list[str],
    available_keywords: list[str],
) -> tuple[str, list[str], list[str], Optional[str]]:
    """
    질의 유형·키워드·도메인을 Gemini 1회 호출로 추출.

    target/context_keywords는 available_keywords pool에서만 선택.
    → semantic gap 없음, hallucination 방지.

    반환:
      query_type       : "direct" | "prerequisite" | "followup" | "career"
      target_keywords  : 추천 대상 강의를 찾을 키워드 (pool 내 선택)
      context_keywords : 억제할 맥락 키워드 (pool 내 선택)
      domain           : available_domains 중 하나, 해당 없으면 None
    """
    domain_list  = ", ".join(available_domains)
    keyword_list = ", ".join(available_keywords)

    prompt = f"""다음 강의 검색 질의를 분석해줘.

질의: "{query}"

다음 JSON 형식으로만 출력해 (설명 없이):
{{
  "query_type": "질의 유형",
  "target_keywords": ["키워드1", ...],
  "context_keywords": ["키워드1", ...],
  "domain": "도메인 문자열 또는 null"
}}

──────────────────────────────────────────────────────
[query_type] 네 가지 중 하나:

  "direct"       특정 주제 강의를 직접 요청
                 예) "운영체제 강의 추천해줘"

  "prerequisite" X를 배우기 위해 먼저 알아야 할 강의 요청
                 예) "머신러닝 배우기 전에 수학 강의 뭐가 좋아?"
                 target = 먼저 들을 것(수학), context = 목표(머신러닝)

  "followup"     이미 들은 강의 이후 연계 강의 요청
                 예) "선형대수 벡터 강의 들었는데 다음은?"
                 target = 다음 단계, context = 들은 강의

  "career"       직무·진로 목표 기반 강의 요청
                 예) "AI 연구자가 되고 싶어"

  "unknown"      강의 추천 의도가 없는 질의
                 예) "운영체제가 뭐야?", "안녕", "파이썬 코드 짜줘", "강의 목록 보여줘"
                 → target_keywords = [], context_keywords = [], domain = null

──────────────────────────────────────────────────────
[target_keywords] — 추천받을 강의를 찾기 위한 키워드:
- 반드시 아래 [키워드 목록]에서만 선택 (최대 8개)
- direct/career: 질의 의도에 맞는 관련 키워드를 넉넉하게 선택
  예) "운영체제 강의" → 운영체제와 직접 연관된 키워드 전부
- prerequisite: 먼저 들을 강의 키워드 (목표 강의 키워드 제외)
- followup: 들은 강의의 다음 단계 키워드 (들은 강의 키워드 제외)

[context_keywords] — 억제할 맥락 키워드:
- 반드시 아래 [키워드 목록]에서만 선택 (최대 5개)
- direct/career: 항상 []
- prerequisite: 목표 강의 관련 키워드 (ex. 머신러닝 관련)
- followup: 이미 들은 강의 관련 키워드

[domain] — 추천받을 강의의 도메인:
- 반드시 아래 목록 중 하나: {domain_list}
- 확신할 수 없으면 null

──────────────────────────────────────────────────────
[키워드 목록] — target/context_keywords는 이 중에서만 선택:
{keyword_list}"""

    response = _client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config={"temperature": 0.0},
    )
    text   = response.text.strip().replace("```json", "").replace("```", "").strip()
    parsed = json.loads(text)

    query_type       = parsed.get("query_type", "direct")
    target_keywords  = parsed.get("target_keywords", [])
    context_keywords = parsed.get("context_keywords", [])
    domain           = parsed.get("domain") or None

    # pool 외 키워드 필터링
    kw_set           = set(available_keywords)
    target_keywords  = [k for k in target_keywords  if k in kw_set]
    context_keywords = [k for k in context_keywords if k in kw_set]

    if query_type not in QUERY_TYPES:
        query_type = "direct"
    if domain not in available_domains:
        domain = None

    return query_type, target_keywords, context_keywords, domain


# ============================================================================
#  추천 결과
# ============================================================================

@dataclass
class RecommendResult:
    video_id:     str
    title:        str
    domain:       str
    instructor:   str
    score:        float
    score_detail: dict
    reason:       str
    summary:      str


def _build_reason(detail: dict) -> str:
    parts = []
    if detail.get("context_penalty"):
        parts.append("⚑ 맥락 억제")
    if detail.get("domain_score", 0) == 1.0:
        parts.append("도메인 일치")
    if detail.get("title_score", 0) >= 0.5:
        parts.append(f"제목 일치 {detail['title_score']:.0%}")
    if detail.get("keyword_score", 0) > 0.05:
        parts.append(f"키워드 {detail['keyword_score']:.0%}")
    if detail.get("summary_score", 0) > 0.3:
        parts.append(f"요약 일치 {detail['summary_score']:.0%}")
    return " · ".join(parts) if parts else "관련 강의"


# ============================================================================
#  추천 엔진
# ============================================================================

class Recommender:
    def __init__(self, metadata_dir: str = "metadata/", config: Optional[RecommenderConfig] = None):
        self.collection          = MetadataCollection(metadata_dir)
        self.cfg                 = config or RecommenderConfig()
        self._available_domains  = self.collection.available_domains()
        self._available_keywords = self.collection.available_keywords()
        print(f"[도메인]    {self._available_domains}")
        print(f"[키워드 풀] {len(self._available_keywords)}개\n")

    def recommend_from_query(self, query: str, top_k: int = 5) -> list[RecommendResult]:
        print(f"[질의 분석] {query}")
        query_type, target_kw, context_kw, domain = analyze_query(
            query, self._available_domains, self._available_keywords
        )
        print(f"[질의 유형]   {query_type}")
        print(f"[대상 키워드] {target_kw}")
        print(f"[맥락 키워드] {context_kw}")
        print(f"[추론 도메인] {domain or '미확정'}\n")

        if query_type == "unknown":
            print("  ※ 강의 추천 질의가 아닙니다.")
            return []

        candidates = []
        for target in self.collection.all():
            detail = compute_query_similarity(
                target_kw, context_kw, query_type, domain, target, self.cfg
            )
            candidates.append((target, detail))

        candidates.sort(key=lambda x: x[1]["score"], reverse=True)
        return [
            RecommendResult(
                video_id     = t.video_id,
                title        = t.title,
                domain       = t.domain,
                instructor   = t.instructor_id,
                score        = d["score"],
                score_detail = d,
                reason       = _build_reason(d),
                summary      = t.summary,
            )
            for t, d in candidates
        ]

    def print_results(self, results: list[RecommendResult], title: str = "추천 결과", top_k: int = 3):
        scored = [r for r in results if r.score > 0]
        top    = scored[:top_k]

        print(f"\n{'='*62}")
        print(f"  {title}")
        print(f"{'='*62}")

        if not top:
            print("  ※ 질의와 일치하는 강의를 찾지 못했습니다.")
        else:
            qtype = top[0].score_detail.get("query_type", "direct")
            print(f"  ▶ 추천 강의 (Top {min(top_k, len(top))})  [유형: {qtype}]")
            print(f"  {'-'*58}")
            for i, r in enumerate(top, 1):
                d         = r.score_detail
                cfg       = self.cfg
                score_100 = round(r.score * 100, 1)
                k_contrib = round(cfg.W_KEYWORD * d.get("keyword_score", 0) * 100, 1)
                t_contrib = round(cfg.W_TITLE   * d.get("title_score",   0) * 100, 1)
                s_contrib = round(cfg.W_SUMMARY * d.get("summary_score", 0) * 100, 1)
                boost_pct = round((d.get("domain_boost", 1.0) - 1.0) * 100, 1)
                penalty   = "  ※억제됨" if d.get("context_penalty") else ""
                print(f"  {i}. [{r.video_id}] {r.title}{penalty}")
                print(f"     점수:   {score_100}점  |  {r.reason}")
                print(f"     기여:   keyword {k_contrib}점  title {t_contrib}점  "
                      f"summary {s_contrib}점  domain_boost +{boost_pct}%")
                print(f"     요약:   {r.summary[:80]}...")
                print()

        print(f"  {'─'*58}")
        print(f"  ▶ 전체 강의 점수")
        print(f"  {'─'*58}")
        print(f"  {'video_id':<12} {'제목':<22} {'도메인':<18} {'점수':>6} {'억제':>4}")
        print(f"  {'─'*58}")
        for r in results:
            marker  = " ◀" if r in top else ""
            penalty = "  ⚑" if r.score_detail.get("context_penalty") else ""
            print(f"  {r.video_id:<12} {r.title[:20]:<22} {r.domain:<18} "
                  f"{round(r.score*100,1):>5.1f}점{marker}{penalty}")
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
  python recommender.py --metadata_dir metadata/ --query "운영체제란 무엇인가"
"""
    )
    parser.add_argument("--metadata_dir", default="metadata/",
                        help="메타데이터 디렉토리 (기본: metadata/)")
    parser.add_argument("--query", default=None,
                        help="자연어 질의 (예: '메모리 관리 알고 싶어')")
    parser.add_argument("--top_k", type=int, default=5,
                        help="추천 결과 수 (기본: 5)")
    args = parser.parse_args()

    rec = Recommender(metadata_dir=args.metadata_dir)

    if args.query:
        results = rec.recommend_from_query(args.query, top_k=args.top_k)
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
                results = rec.recommend_from_query(query, top_k=args.top_k)
                rec.print_results(results, f"질의: '{query}'", top_k=3)
            except KeyboardInterrupt:
                print("\n종료")
                break