"""
generate_metadata.py — 강의 메타데이터 추출

입력:
  - {stem}_fused.json      → 슬라이드 텍스트, 전사, 강조 키워드, 길이
  - Neo4j                  → Concept 노드 degree (중심성)

출력:
  - metadata/{stem}_metadata.json

사용법:
  python generate_metadata.py --stem os1-1 --title "운영체제 개론" --instructor_id prof_001
  python generate_metadata.py --stem os1-1 --title "운영체제 개론" --instructor_id prof_001 \\
      --output_dir output --metadata_dir metadata
"""

import os
import re
import json
import math
import argparse
from pathlib import Path
from collections import Counter

from neo4j import GraphDatabase
from google import genai
from dotenv import load_dotenv

from .config import GEMINI_GENERATIVE_MODEL

load_dotenv(override=True)

# ── 설정 ──────────────────────────────────────────────────────────────────────

NEO4J_URI      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER",     "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")

print(f"[디버그] NEO4J URI={NEO4J_URI}  USER={NEO4J_USER}  PW={'*'*len(NEO4J_PASSWORD)}")

_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY_1"))
MODEL   = GEMINI_GENERATIVE_MODEL


def _sample_uniform(texts: list[str], n: int) -> list[str]:
    """리스트에서 균등 간격으로 n개 샘플링"""
    if len(texts) <= n:
        return texts
    step = len(texts) / n
    return [texts[int(i * step)] for i in range(n)]

DOMAIN_LIST = """
ENG: eng/cs, eng/electrical, eng/mechanical, eng/civil, eng/chemical,
     eng/industrial, eng/biomedical, eng/aerospace, eng/materials, eng/environmental
SCI: sci/physics, sci/chemistry, sci/biology, sci/earth_science, sci/astronomy, sci/ecology
MATH: math/calculus, math/linear_algebra, math/discrete, math/probability,
      math/statistics, math/numerical, math/optimization
HUM: hum/philosophy, hum/history, hum/linguistics, hum/literature,
     hum/art_history, hum/religion
SOC: soc/economics, soc/business, soc/law, soc/political_science,
     soc/sociology, soc/psychology, soc/education
MED: med/anatomy, med/physiology, med/pharmacology, med/clinical,
     med/public_health, med/nursing
ART: art/fine_arts, art/music, art/design, art/film, art/theater,
     art/physical_education, art/sports_science
GEN: gen/writing, gen/critical_thinking, gen/career, gen/ethics,
     gen/language, gen/interdisciplinary, gen/other
"""

# 키워드 점수 가중치
W_FREQ       = 0.4
W_EMPHASIS   = 0.2
W_CENTRALITY = 0.4

# concept_roles 분류 임계값
CORE_CENT_THRESHOLD  = 0.5   # norm_cent 이 이상이면 core 후보
CORE_EMPH_THRESHOLD  = 0.35  # norm_emph + norm_slide_freq 조합 기준
CORE_FREQ_THRESHOLD  = 0.25


# ── fused.json 파싱 ────────────────────────────────────────────────────────────

def load_fused(stem: str, output_dir: Path) -> dict:
    try:
        from .config import output_paths
        paths = output_paths(stem, output_dir, output_dir)
        path  = paths["fused"]
    except ImportError:
        path = output_dir / f"{stem}_fused.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def get_duration(fused: dict) -> float:
    """마지막 슬라이드 end_sec 기준 총 길이(초)"""
    slides = fused.get("slides", [])
    if not slides:
        return 0.0
    last = slides[-1]
    ctxs = last.get("contexts", [])
    if ctxs:
        segs = ctxs[-1].get("segments", [])
        if segs:
            return float(segs[-1].get("end", 0.0))
        return float(ctxs[-1].get("end", 0.0))
    return float(last.get("end_sec", 0.0))


def collect_texts(fused: dict) -> tuple[list[str], list[str], list[str], list[str]]:
    """
    slide_texts, transcript_texts: 전체 (키워드 점수용)
    core_slide_texts, core_trans_texts: core/elaborated만 (요약용)
    """
    slide_texts      = []
    transcript_texts = []
    core_slide_texts = []
    core_trans_texts = []

    for slide in fused.get("slides", []):
        # objectives 슬라이드 스킵
        if slide.get("role") == "objectives":
            continue

        role = slide.get("role", "")
        parts = []
        if slide.get("title"):
            parts.append(slide["title"])
        if slide.get("slide_text"):
            parts.append(slide["slide_text"])
        text = " ".join(parts)
        slide_texts.append(text)
        if role == "core":
            core_slide_texts.append(text)

        for ctx in slide.get("contexts", []):
            for seg in ctx.get("segments", []):
                t = seg.get("text", "")
                if t:
                    transcript_texts.append(t)
                    if role in ("core", "elaborated"):
                        core_trans_texts.append(t)

    # core/elaborated가 하나도 없으면 전체로 fallback
    if not core_slide_texts:
        core_slide_texts = slide_texts
    if not core_trans_texts:
        core_trans_texts = transcript_texts

    return slide_texts, transcript_texts, core_slide_texts, core_trans_texts


def collect_emphasized(fused: dict) -> dict[str, float]:
    """키워드 → 강조 점수 합산 (emphasized_keywords 기반)"""
    scores: dict[str, float] = {}
    for slide in fused.get("slides", []):
        for kw in slide.get("emphasized_keywords", []):
            name = kw.get("keyword", "")
            if not name:
                continue
            score = (
                kw.get("audio_score", 0.0)
                + kw.get("annotation_score", 0.0)
                + kw.get("visual_score", 0.0)
                + kw.get("slide_text_score", 0.0)
            )
            scores[name] = scores.get(name, 0.0) + score
    return scores


def collect_slide_role_freq(
    fused: dict,
    all_names: set[str],
) -> dict[str, dict[str, int]]:
    """
    개념명 → 슬라이드 role별 텍스트 등장 횟수
    반환: {개념명: {"core": n, "elaborated": n, "supplementary": n, "other": n}}

    concept_roles 분류 시 슬라이드 내 역할 분포 판단에 사용
    """
    TRACKED_ROLES = ("core", "elaborated", "supplementary")
    result = {n: {"core": 0, "elaborated": 0, "supplementary": 0, "other": 0}
              for n in all_names}

    for slide in fused.get("slides", []):
        role = slide.get("role", "other")
        role_key = role if role in TRACKED_ROLES else "other"

        text = " ".join(filter(None, [
            slide.get("title", ""),
            slide.get("slide_text", ""),
        ]))

        for name in all_names:
            if name in text:
                result[name][role_key] += text.count(name)

    return result


# ── Neo4j: Concept 노드 degree 조회 ──────────────────────────────────────────

def fetch_concept_degrees(stem: str) -> dict[str, int]:
    """
    stem에 해당하는 Concept 노드 name → degree 반환
    lecture_video/{stem} 노드로부터 연결된 Concept 탐색
    """
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    result = {}
    with driver.session() as session:
        records = session.run(
            """
            MATCH (v:Video {id: $vid})
                  -[:HAS_SLIDES]->(:Slides)
                  -[:CONTAINS]->(s:Slide)
                  -[:MENTIONS|APPEARS_IN]-(c:Concept)
            RETURN c.name AS name, COUNT { (c)-[]-() } AS degree
            """,
            vid=f"lecture_video/{stem}",
        )
        for r in records:
            if r["name"]:
                result[r["name"]] = r["degree"]
    driver.close()
    return result


def fetch_concept_degrees(stem: str) -> dict[str, int]:
    """
    stem에 해당하는 Concept 노드 name → degree 반환
    lecture_video/{stem} 노드로부터 연결된 Concept 탐색
    """
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    result = {}
    with driver.session() as session:
        records = session.run(
            """
            MATCH (v:Video {id: $vid})
                  -[:HAS_SLIDES]->(:Slides)
                  -[:CONTAINS]->(s:Slide)
                  -[:MENTIONS|APPEARS_IN]-(c:Concept)
            RETURN c.name AS name, COUNT { (c)-[]-() } AS degree
            """,
            vid=f"lecture_video/{stem}",
        )
        for r in records:
            if r["name"]:
                result[r["name"]] = r["degree"]
    driver.close()
    return result


def fetch_concept_relations(
    stem:            str,
    role_candidates: set[str],
) -> list[dict]:
    """
    role_candidates 내 Concept 노드 간 방향성 있는 관계 조회

    반환: [{"from": "A", "to": "B", "type": "LEADS_TO"}, ...]

    - 강의 범위 내 Concept만 대상 (Video → Slides → Slide → Concept 경유)
    - role_candidates 바깥 개념은 제외 → 메타데이터 외부 참조 없음
    - 관계 수 상한 200개 (단일 강의에서 그 이상은 노이즈)
    """
    if not role_candidates:
        return []

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    result = []
    names_list = list(role_candidates)

    with driver.session() as session:
        records = session.run(
            """
            MATCH (v:Video {id: $vid})
                  -[:HAS_SLIDES]->(:Slides)
                  -[:CONTAINS]->(s:Slide)
                  -[:MENTIONS|APPEARS_IN]-(c1:Concept)
            WHERE c1.name IN $names
            MATCH (c1)-[r]->(c2:Concept)
            WHERE c2.name IN $names
              AND c1 <> c2
            RETURN DISTINCT c1.name AS from_name,
                            type(r)  AS rel_type,
                            c2.name  AS to_name
            LIMIT 200
            """,
            vid=f"lecture_video/{stem}",
            names=names_list,
        )
        for r in records:
            if r["from_name"] and r["to_name"]:
                result.append({
                    "from": r["from_name"],
                    "type": r["rel_type"],
                    "to":   r["to_name"],
                })
    driver.close()
    print(f"       → concept_relations {len(result)}개")
    return result
    """
    stem에 해당하는 Concept 노드 name → degree 반환
    lecture_video/{stem} 노드로부터 연결된 Concept 탐색
    """
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    result = {}
    with driver.session() as session:
        records = session.run(
            """
            MATCH (v:Video {id: $vid})
                  -[:HAS_SLIDES]->(:Slides)
                  -[:CONTAINS]->(s:Slide)
                  -[:MENTIONS|APPEARS_IN]-(c:Concept)
            RETURN c.name AS name, COUNT { (c)-[]-() } AS degree
            """,
            vid=f"lecture_video/{stem}",
        )
        for r in records:
            if r["name"]:
                result[r["name"]] = r["degree"]
    driver.close()
    return result


# ── 키워드 점수 산출 ───────────────────────────────────────────────────────────

def _normalize(d: dict[str, float]) -> dict[str, float]:
    if not d:
        return d
    max_v = max(d.values()) or 1.0
    return {k: v / max_v for k, v in d.items()}


def _count_freq_split(
    name: str,
    slide_texts: list[str],
    transcript_texts: list[str],
) -> tuple[float, float]:
    """(슬라이드 등장 횟수, 전사 등장 횟수) 분리 반환 — concept_roles 분류용"""
    slide_cnt = sum(t.count(name) for t in slide_texts)
    trans_cnt = sum(t.count(name) for t in transcript_texts)
    return float(slide_cnt), float(trans_cnt)


def compute_keyword_scores(
    concept_degrees:  dict[str, int],
    emphasized:       dict[str, float],
    slide_texts:      list[str],
    transcript_texts: list[str],
) -> tuple[dict[str, float], dict[str, float], dict[str, float], dict[str, float], dict[str, float]]:
    """
    키워드별 정규화 점수를 계산하고 반환

    반환값 (모두 {name: float}, 0~1 정규화):
        scored         — 최종 가중 합산 점수 (추천용)
        norm_slide_freq — 슬라이드 빈도만
        norm_trans_freq — 전사 빈도만
        norm_emph       — 강조 점수
        norm_cent       — KG 중심성 (degree)
    """
    all_names = set(concept_degrees) | set(emphasized)
    if not all_names:
        empty: dict[str, float] = {}
        return empty, empty, empty, empty, empty

    # 슬라이드·전사 빈도 분리 수집
    raw_slide_freq: dict[str, float] = {}
    raw_trans_freq: dict[str, float] = {}
    for n in all_names:
        s, t = _count_freq_split(n, slide_texts, transcript_texts)
        raw_slide_freq[n] = s
        raw_trans_freq[n] = t

    # 기존 합산 빈도 (슬라이드×1.0 + 전사×0.5) → 최종 점수용
    raw_freq = {n: raw_slide_freq[n] * 1.0 + raw_trans_freq[n] * 0.5 for n in all_names}

    raw_emph = {n: emphasized.get(n, 0.0)         for n in all_names}
    raw_cent = {n: float(concept_degrees.get(n, 0)) for n in all_names}

    norm_freq       = _normalize(raw_freq)
    norm_slide_freq = _normalize(raw_slide_freq)
    norm_trans_freq = _normalize(raw_trans_freq)
    norm_emph       = _normalize(raw_emph)
    norm_cent       = _normalize(raw_cent)

    scored = {
        n: W_FREQ * norm_freq[n] + W_EMPHASIS * norm_emph[n] + W_CENTRALITY * norm_cent[n]
        for n in all_names
    }

    return scored, norm_slide_freq, norm_trans_freq, norm_emph, norm_cent


def score_keywords(
    concept_degrees:  dict[str, int],
    emphasized:       dict[str, float],
    slide_texts:      list[str],
    transcript_texts: list[str],
    duration_sec:     float,
    debug:            bool = False,
) -> tuple[list[dict], dict, dict, dict, dict, dict]:
    """
    score = 0.4·freq + 0.2·emphasis + 0.4·centrality
    추천 메타데이터용 키워드 목록과 중간 norm값을 함께 반환

    반환:
        keywords        — [{"keyword": ..., "score": ...}, ...]  score 내림차순
        scored          — {name: float}  전체 점수 (concept_roles 분류용)
        norm_slide_freq — {name: float}
        norm_trans_freq — {name: float}
        norm_emph       — {name: float}
        norm_cent       — {name: float}
    """
    all_names = set(concept_degrees) | set(emphasized)
    if not all_names:
        empty: dict[str, float] = {}
        return [], empty, empty, empty, empty, empty

    scored, norm_slide_freq, norm_trans_freq, norm_emph, norm_cent = compute_keyword_scores(
        concept_degrees, emphasized, slide_texts, transcript_texts
    )

    if debug:
        print("\n[디버그] 키워드 점수 상세 (상위 15개)")
        top15 = sorted(scored.items(), key=lambda x: -x[1])[:15]
        for n, s in top15:
            s_raw = sum(t.count(n) for t in slide_texts)
            t_raw = sum(t.count(n) for t in transcript_texts)
            e_raw = emphasized.get(n, 0.0)
            c_raw = concept_degrees.get(n, 0)
            print(
                f"{n:<20} score={s:>6.4f}  "
                f"slide_freq={norm_slide_freq[n]:>5.3f}(raw={s_raw:>4})  "
                f"trans_freq={norm_trans_freq[n]:>5.3f}(raw={t_raw:>4})  "
                f"emph={norm_emph[n]:>5.3f}(raw={e_raw:>5.2f})  "
                f"cent={norm_cent[n]:>5.3f}(raw={c_raw:>3d})"
            )
        print()

    # 개수 결정: duration 기반 [min_k, max_k] + score 임계값 컷
    # min: 7개 하한, 8분당 1개 추가 / max: 30개 상한, 3분당 1개 추가
    duration_min = duration_sec / 60.0
    min_k = max(7, int(duration_min // 8))
    max_k = min(30, int(duration_min // 3))
    if max_k < min_k:
        max_k = min_k

    threshold = sum(scored.values()) / len(scored) * 1.2

    def _keep(n: str) -> bool:
        return (
            _is_valid_concept(n)
            and not _is_background_noise(n, norm_slide_freq, norm_trans_freq, norm_cent)
        )

    candidates = sorted(
        [(n, s) for n, s in scored.items() if s >= threshold and _keep(n)],
        key=lambda x: -x[1],
    )

    if len(candidates) < min_k:
        # 임계값 완화 fallback — 노이즈 필터는 유지
        candidates = sorted(
            [(n, s) for n, s in scored.items() if _keep(n)],
            key=lambda x: -x[1],
        )[:min_k]
    elif len(candidates) > max_k:
        candidates = candidates[:max_k]

    keywords = [{"keyword": n, "score": round(s, 4)} for n, s in candidates]

    return keywords, scored, norm_slide_freq, norm_trans_freq, norm_emph, norm_cent


# ── concept_roles 분류 ────────────────────────────────────────────────────────

# ── 언어 규칙 기반 노이즈 필터 ───────────────────────────────────────────────
#
# 설계 원칙:
#   하드코딩 도메인 불용어 목록(KOREAN_STOPWORDS) 제거.
#   대신 두 가지 자동 필터를 사용한다:
#
#   1) _is_valid_concept  — 순수 언어 규칙 (길이, 동사 어미, 숫자, URL)
#                           어떤 도메인·강의에도 동일하게 적용 가능
#
#   2) _is_background_noise — 신호 기반 자동 탐지 (계산된 norm 값 필요)
#                           슬라이드에만 등장하고 전사·KG에 없는 항목
#                           = 워터마크, 반복 헤더, 출처 표기 등
#                           어떤 도메인이든 '실제로 강의한 내용'과
#                           '슬라이드 장식'을 신호로 구분

# 동사 어미·조사 패턴 (한국어 언어 규칙 — 도메인 무관)
_NOISE_SUFFIX_RE = re.compile(
    r"(하고|하며|하여|하는|하면서|하고자|이라고|이며|이나|이고"
    r"|대해|에서|으로|에게|으로써|이다|아니다|있다|없다|한다"
    r"|하거나|하지|하면|이지|하니|않고|이후|이전"
    r"|있도록|있음|없던|만든|만드|된|될|할|함|함으로|함께"
    r"|하$|않는다$|아님$|준다$|포함한$|보여주$|느끼$|제공받$"
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

    # 숫자만으로 구성
    if re.fullmatch(r"\d+", name):
        return False

    # URL 잔재 (점+도메인 패턴)
    if re.search(r"\.(com|kr|org|net|io|tistory|brunch|github)", name):
        return False

    # 영어 소문자 단독 단어 → 기능어 체크
    if re.fullmatch(r"[a-z]+", name):
        return name not in _ENGLISH_STOPWORDS

    # 동사 어미·조사 패턴
    if _NOISE_SUFFIX_RE.search(name):
        return False

    return True


def _is_background_noise(
    name: str,
    norm_slide_freq: dict[str, float],
    norm_trans_freq: dict[str, float],
    norm_cent:       dict[str, float],
) -> bool:
    """
    신호 기반 배경 노이즈 자동 탐지.

    슬라이드에는 반복 등장하지만 실제로 강의하지 않은 항목을 탐지.
    = 워터마크, 기관명, 반복 헤더, 출처 표기, 잘린 토큰 등

    판단 기준:
      슬라이드 빈도 상위 25% 이상  (많이 등장)
      + 전사 빈도 하위 5%          (말로는 거의 안 함)
      + KG 중심성 하위 10%         (개념 그래프에서도 주변부)

    → 세 조건이 모두 성립하면 강의 내용이 아닌 슬라이드 장식으로 판단.

    반례 방지:
      "운영체제"처럼 주제 핵심어는 슬라이드 빈도 높아도
      전사 빈도·중심성도 높으므로 통과.
    """
    s = norm_slide_freq.get(name, 0.0)
    t = norm_trans_freq.get(name, 0.0)
    c = norm_cent.get(name, 0.0)
    return s >= 0.25 and t <= 0.05 and c <= 0.10


def classify_concept_roles(
    all_names:       set[str],
    norm_slide_freq: dict[str, float],
    norm_trans_freq: dict[str, float],
    norm_emph:       dict[str, float],
    norm_cent:       dict[str, float],
    slide_role_freq: dict[str, dict[str, int]],
) -> dict[str, list[str]]:
    """
    각 개념을 core / introduced 중 하나로 분류

    분류 기준:
      core      — KG 중심성이 높거나 (강조 + 슬라이드 빈도) 조합이 충분한 핵심 개념
      introduced — 나머지 전부 (덜 다뤄지는 개념, 예시성 언급 포함)

    prerequisite(선수 지식)은 텍스트 신호만으로 구별 불가능하므로 제거.
    대신 추천 시스템 런타임에서 concept_relations 체이닝으로 추론:
      "다른 강의의 introduced → 이 강의의 core" 관계가 있으면 선수 강의로 판단.
    """
    core:      list[str] = []
    introduced: list[str] = []

    for name in all_names:
        s_freq = norm_slide_freq.get(name, 0.0)
        t_freq = norm_trans_freq.get(name, 0.0)
        emph   = norm_emph.get(name, 0.0)
        cent   = norm_cent.get(name, 0.0)

        # ── 분류 규칙 ────────────────────────────────────────────────────────

        # core: 두 조건 중 하나 만족
        #   A) 중심성 높고 강조도 일정 이상 — KG 핵심 + 실제 강의에서 강조됨
        #   B) 강조·슬라이드빈도 모두 충분 — 강의에서 직접 비중있게 다뤄짐
        # emph를 필수로 두는 이유: cent 단독으로 허용하면 범용어(프로그램 등)가
        # KG 연결 수만으로 core에 오분류됨
        if (cent >= CORE_CENT_THRESHOLD and emph >= CORE_EMPH_THRESHOLD) or (
            emph >= CORE_EMPH_THRESHOLD and s_freq >= CORE_FREQ_THRESHOLD
        ):
            core.append(name)

        # 2. introduced: 나머지 전부
        #    - 슬라이드에만 등장, 전사에만 등장, 둘 다 적게 등장 모두 포함
        #    - "덜 다뤄지는 개념"을 통일해서 관리
        #    - 진짜 prerequisite(선수 지식)은 concept_relations 체이닝으로 런타임 추론
        else:
            introduced.append(name)

    # 정렬: 각 그룹 내부를 중심성 내림차순으로 정렬
    sort_key = lambda n: -norm_cent.get(n, 0.0)

    # introduced는 상한선 40개 — 이 이상은 추천 시스템 노이즈
    MAX_INTRODUCED = 40
    return {
        "core":      sorted(core,                          key=sort_key),
        "introduced": sorted(introduced, key=sort_key)[:MAX_INTRODUCED],
    }


# ── learning_objectives 수집 ──────────────────────────────────────────────────
_TOC_LINE_RE = re.compile(r"^\d+\s*[.)]?\s+\S+")

def collect_learning_objectives(fused: dict) -> list[str]:
    objectives = []
    for slide in fused.get("slides", []):      # slides 리스트 순회
        if slide.get("role") != "objectives":
            continue
        slide_text = slide.get("slide_text", "")
        texts = [t.strip() for t in slide_text.splitlines() if t.strip()]

        toc_lines = [t for t in texts
                     if _TOC_LINE_RE.match(t)
                     and not t.startswith(("•", "-", "*"))]
        if len(toc_lines) >= 5:
            objectives.extend(toc_lines)
            continue

        # 학습목표형: 모든 줄 수집
        objectives.extend(texts)

    return objectives


# ── difficulty 추정 ────────────────────────────────────────────────────────────

def estimate_difficulty(
    concept_roles: dict[str, list[str]],
    norm_cent:     dict[str, float],
) -> str:
    """
    core 개념들의 평균 KG 중심성으로 난이도 추정

    직관: core 개념이 KG에서 촘촘하게 연결될수록 → 개념 밀도 높은 강의 → 어려움
    low_intro_ratio 제거: 개론 강의일수록 예시가 많아 저중심성 introduced가 많아지므로
                         오히려 역방향으로 작동하는 구조적 결함 있음

    기준 (avg_core_cent 기준):
      advanced     : >= 0.65
      intermediate : >= 0.35
      beginner     : 그 외
    """
    cores = concept_roles.get("core", [])
    avg_core_cent = (
        sum(norm_cent.get(n, 0.0) for n in cores) / len(cores)
        if cores else 0.0
    )

    if avg_core_cent >= 0.65:
        difficulty = "advanced"
    elif avg_core_cent >= 0.35:
        difficulty = "intermediate"
    else:
        difficulty = "beginner"

    print(
        f"[디버그] difficulty: {difficulty}  "
        f"(core={len(cores)}, avg_core_cent={avg_core_cent:.3f})"
    )
    return difficulty


# ── LLM 호출 ──────────────────────────────────────────────────────────────────

def _gemini(prompt: str) -> str:
    resp = _client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config={"temperature": 0.0},
    )
    return resp.text.strip()


def classify_domain(slide_texts: list[str], transcript_texts: list[str]) -> str:
    sample_slide = " ".join(slide_texts[:5])[:800]
    sample_trans = " ".join(transcript_texts[:20])[:800]
    prompt = f"""아래는 강의 내용 일부다.

슬라이드: {sample_slide}
전사: {sample_trans}

다음 도메인 목록 중 가장 적합한 것 하나만 출력하라. 설명 없이 도메인 코드만 출력.
예시 출력: eng/cs

도메인 목록:
{DOMAIN_LIST}
"""
    result = _gemini(prompt).strip().lower()
    valid = re.findall(r"[a-z]+/[a-z_]+", result)
    return valid[0] if valid else "gen/other"


def generate_summary(
    core_slide_texts: list[str],
    core_trans_texts: list[str],
    concept_degrees:  dict | None = None,
) -> str:
    MAX_PER_SLIDE = 150
    slide_block = " / ".join(
        s.strip()[:MAX_PER_SLIDE] for s in core_slide_texts if s.strip()
    )[:MAX_PER_SLIDE * len(core_slide_texts)]

    n_sample = min(40, len(core_trans_texts))
    sampled_trans = _sample_uniform(core_trans_texts, n_sample)
    trans_block = " ".join(sampled_trans)[:1500]

    concept_block = ""
    if concept_degrees:
        top = sorted(concept_degrees.items(), key=lambda x: -x[1])[:10]
        concept_block = f"\n핵심 개념 (지식그래프 기반): {', '.join(n for n, _ in top)}\n"

    prompt = f"""아래는 강의의 슬라이드 텍스트, 전사본, 핵심 개념 목록이다.

    슬라이드 (강의 전체 균등 샘플):
    {slide_block}

    전사 (강의 전체 균등 샘플):
    {trans_block}
    {concept_block}두 소스 모두에서 언급된 내용을 중심으로 3문장으로 요약하라.

    작성 규칙:
    - 단순 나열이 아니라 강의의 핵심 흐름과 개념 간 관계가 드러나도록 작성하라.
    - 복숭아, 사과, 파인애플처럼 구체적인 하위 항목을 나열하지 말고,
    "과일", "음식" 등 상위 개념으로 추상화하여 표현하라.
    - 한국어로 작성하고 설명 없이 요약문만 출력하라.
    """

    print(f"\n[디버그] 요약 입력 슬라이드 ({len(core_slide_texts)}장):")
    for i, s in enumerate(core_slide_texts):
        print(f"  [{i+1}] {s.strip()[:80]}")
    print(f"\n[디버그] 요약 입력 전사 segment 수: {len(core_trans_texts)}")
    if concept_degrees:
        top = sorted(concept_degrees.items(), key=lambda x: -x[1])[:10]
        print(f"[디버그] KG 핵심 개념: {[n for n, _ in top]}")
    print(f"\n[디버그] slide_block ({len(slide_block)}자):\n{slide_block[:500]}")
    print(f"\n[디버그] trans_block ({len(trans_block)}자):\n{trans_block[:300]}\n")
    return _gemini(prompt)


# ── 메인 ──────────────────────────────────────────────────────────────────────

def generate_metadata(
    stem:          str,
    title:         str,
    instructor_id: str,
    output_dir:    Path,
    metadata_dir:  Path,
) -> dict:
    print(f"[{stem}] fused.json 로드 중...")
    fused = load_fused(stem, output_dir)

    duration_sec = get_duration(fused)
    slide_texts, transcript_texts, core_slide_texts, core_trans_texts = collect_texts(fused)
    emphasized   = collect_emphasized(fused)

    # objectives 슬라이드는 요약/키워드 연산과 독립적으로 별도 수집
    learning_objectives = collect_learning_objectives(fused)
    print(f"[{stem}] 학습 목표 {len(learning_objectives)}개 수집")

    print(f"[{stem}] Neo4j Concept 노드 조회 중...")
    concept_degrees = fetch_concept_degrees(stem)
    print(f"       → Concept {len(concept_degrees)}개")

    print(f"[{stem}] 도메인 분류 중...")
    domain = classify_domain(slide_texts, transcript_texts)

    print(f"[{stem}] 요약 생성 중...")
    summary = generate_summary(core_slide_texts, core_trans_texts, concept_degrees)

    print(f"[{stem}] 키워드 점수 산출 중...")
    keywords, scored, norm_slide_freq, norm_trans_freq, norm_emph, norm_cent = score_keywords(
        concept_degrees, emphasized, slide_texts, transcript_texts, duration_sec,
        debug=True,
    )
    print(f"       → {len(keywords)}개 선택")

    # concept_roles: scored 평균 × 0.6 이상 + 노이즈 필터 통과한 개념만 분류
    # (keywords 임계값 1.2보다 낮게 → prerequisite/introduced도 충분히 포함)
    print(f"[{stem}] concept_roles 분류 중...")
    role_threshold = (sum(scored.values()) / len(scored) * 0.6) if scored else 0.0
    role_candidates = {
        n for n, s in scored.items()
        if s >= role_threshold
        and _is_valid_concept(n)
        and not _is_background_noise(n, norm_slide_freq, norm_trans_freq, norm_cent)
    }
    print(f"       → 분류 대상 {len(role_candidates)}개 (전체 {len(scored)}개 중)")
    slide_role_freq = collect_slide_role_freq(fused, role_candidates)
    concept_roles = classify_concept_roles(
        role_candidates,
        norm_slide_freq,
        norm_trans_freq,
        norm_emph,
        norm_cent,
        slide_role_freq,
    )
    print(
        f"       → core {len(concept_roles['core'])}개 | "
        f"introduced {len(concept_roles['introduced'])}개"
    )
    print(f"\n[디버그] concept_roles 상세:")
    for role, names in concept_roles.items():
        print(f"  [{role}] {names[:10]}")
    print()

    print(f"[{stem}] concept_relations 조회 중...")
    concept_relations = fetch_concept_relations(stem, role_candidates)

    # difficulty: concept_roles 결과 + norm_cent 활용 (LLM 호출 없음)
    difficulty = estimate_difficulty(concept_roles, norm_cent)

    metadata = {
        "video_id":            stem,
        "title":               title,
        "instructor_id":       instructor_id,
        "duration_sec":        round(duration_sec, 1),
        "domain":              domain,
        "difficulty":          difficulty,
        "summary":             summary,
        "learning_objectives": learning_objectives,
        "keywords":            keywords,
        "concept_roles":       concept_roles,
        "concept_relations":   concept_relations,
    }

    # 저장
    metadata_dir.mkdir(parents=True, exist_ok=True)
    out_path = metadata_dir / f"{stem}_metadata.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"[{stem}] 완료 → {out_path}")

    return metadata


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="강의 메타데이터 생성")
    parser.add_argument("--stem",          required=True,    help="강의 식별자 (예: os1-1)")
    parser.add_argument("--title",         required=True,    help="강의 제목")
    parser.add_argument("--instructor_id", required=True,    help="교수 ID")
    parser.add_argument("--output_dir",    default="output", help="fused.json 위치")
    parser.add_argument("--metadata_dir",  default="metadata", help="메타데이터 저장 디렉토리")
    args = parser.parse_args()

    generate_metadata(
        stem          = args.stem,
        title         = args.title,
        instructor_id = args.instructor_id,
        output_dir    = Path(args.output_dir),
        metadata_dir  = Path(args.metadata_dir),
    )
