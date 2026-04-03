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

load_dotenv(override=True)

# ── 설정 ──────────────────────────────────────────────────────────────────────

NEO4J_URI      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER",     "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")

print(f"[디버그] NEO4J URI={NEO4J_URI}  USER={NEO4J_USER}  PW={'*'*len(NEO4J_PASSWORD)}")

_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY_1"))
MODEL   = "gemini-2.5-flash"


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


# ── fused.json 파싱 ────────────────────────────────────────────────────────────

def load_fused(stem: str, output_dir: Path) -> dict:
    try:
        from config import output_paths
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
    # contexts가 있으면 마지막 context의 end, 없으면 slide의 end_sec
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


# ── Neo4j: Concept 노드 degree 조회 ──────────────────────────────────────────

def fetch_concept_degrees(stem: str) -> dict[str, int]:
    """
    stem에 해당하는 Concept 노드 name → degree 반환
    lecture_video/{stem} 노드로부터 연결된 Concept 탐색
    """
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    result = {}
    with driver.session() as session:
        # Video 노드 경유로 해당 강의의 Concept만 조회
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


def _count_freq(name: str, slide_texts: list[str], transcript_texts: list[str]) -> float:
    """슬라이드 등장 횟수 × 1.0 + 전사 등장 횟수 × 0.5"""
    slide_cnt = sum(t.count(name) for t in slide_texts)
    trans_cnt = sum(t.count(name) for t in transcript_texts)
    return slide_cnt * 1.0 + trans_cnt * 0.5


def score_keywords(
    concept_degrees: dict[str, int],
    emphasized:      dict[str, float],
    slide_texts:     list[str],
    transcript_texts: list[str],
    duration_sec:    float,
    debug:           bool = False,
) -> list[dict]:
    """
    score = 0.4·freq + 0.2·emphasis + 0.4·centrality
    반환: [{"keyword": ..., "score": ...}, ...]  (score 내림차순)
    """
    all_names = set(concept_degrees) | set(emphasized)
    if not all_names:
        return []

    # 원시값 수집
    raw_freq  = {n: _count_freq(n, slide_texts, transcript_texts) for n in all_names}
    raw_emph  = {n: emphasized.get(n, 0.0) for n in all_names}
    raw_cent  = {n: float(concept_degrees.get(n, 0)) for n in all_names}

    # [0,1] 정규화
    norm_freq  = _normalize(raw_freq)
    norm_emph  = _normalize(raw_emph)
    norm_cent  = _normalize(raw_cent)

    # 최종 점수
    scored = {
        n: W_FREQ * norm_freq[n] + W_EMPHASIS * norm_emph[n] + W_CENTRALITY * norm_cent[n]
        for n in all_names
    }

    if debug:
        print("\n[디버그] 키워드 점수 상세 (상위 15개)")
        print(f"{'키워드':<20} {'최종':>6}  {'freq':>6}(raw={'{:>5}'.format('')})  {'emph':>6}(raw={'{:>5}'.format('')})  {'cent':>6}(raw)")
        top15 = sorted(scored.items(), key=lambda x: -x[1])[:15]
        for n, s in top15:
            print(
                f"{n:<20} {s:>6.4f}  "
                f"freq={norm_freq[n]:>5.3f}(raw={raw_freq[n]:>5.1f})  "
                f"emph={norm_emph[n]:>5.3f}(raw={raw_emph[n]:>5.2f})  "
                f"cent={norm_cent[n]:>5.3f}(raw={int(raw_cent[n]):>3d})"
            )
        print()

    # 개수 결정: [min_k, max_k] + score 임계값 컷
    duration_min = duration_sec / 60.0
    min_k = max(5, int(duration_min // 10))
    max_k = min(20, int(duration_min // 5))
    if max_k < min_k:
        max_k = min_k  # 5분 미만 초단편 강의 안전장치

    threshold = sum(scored.values()) / len(scored) * 1.2
    candidates = sorted(
        [(n, s) for n, s in scored.items() if s >= threshold],
        key=lambda x: -x[1],
    )

    # 범위 클리핑
    if len(candidates) < min_k:
        candidates = sorted(scored.items(), key=lambda x: -x[1])[:min_k]
    elif len(candidates) > max_k:
        candidates = candidates[:max_k]

    return [{"keyword": n, "score": round(s, 4)} for n, s in candidates]


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
    # 목록에 없는 값이면 gen/other 반환
    valid = re.findall(r"[a-z]+/[a-z_]+", result)
    return valid[0] if valid else "gen/other"


def generate_summary(
    core_slide_texts: list[str],
    core_trans_texts: list[str],
    concept_degrees:  dict | None = None,
) -> str:
    # core/elaborated 슬라이드: 장당 150자 제한, 전체 한도 = 장수 × 150
    MAX_PER_SLIDE = 150
    slide_block = " / ".join(
        s.strip()[:MAX_PER_SLIDE] for s in core_slide_texts if s.strip()
    )[:MAX_PER_SLIDE * len(core_slide_texts)]

    # core/elaborated 전사: 균등 샘플링 후 1500자 컷
    n_sample = min(40, len(core_trans_texts))
    sampled_trans = _sample_uniform(core_trans_texts, n_sample)
    trans_block = " ".join(sampled_trans)[:1500]

    # KG 핵심 concept 상위 10개
    concept_block = ""
    if concept_degrees:
        top = sorted(concept_degrees.items(), key=lambda x: -x[1])[:10]
        concept_block = f"\n핵심 개념 (지식그래프 기반): {', '.join(n for n, _ in top)}\n"

    prompt = f"""아래는 강의의 슬라이드 텍스트, 전사본, 핵심 개념 목록이다.

슬라이드 (강의 전체 균등 샘플):
{slide_block}

전사 (강의 전체 균등 샘플):
{trans_block}
{concept_block}
두 소스 모두에서 언급된 내용을 중심으로 3문장으로 요약하라.
단순 나열이 아니라 강의의 핵심 흐름이 드러나도록 작성하라.
한국어로 작성하고 설명 없이 요약문만 출력하라.
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

    duration_sec                     = get_duration(fused)
    slide_texts, transcript_texts, core_slide_texts, core_trans_texts = collect_texts(fused)
    emphasized                       = collect_emphasized(fused)

    print(f"[{stem}] Neo4j Concept 노드 조회 중...")
    concept_degrees = fetch_concept_degrees(stem)
    print(f"       → Concept {len(concept_degrees)}개")

    print(f"[{stem}] 도메인 분류 중...")
    domain = classify_domain(slide_texts, transcript_texts)

    print(f"[{stem}] 요약 생성 중...")
    summary = generate_summary(core_slide_texts, core_trans_texts, concept_degrees)

    print(f"[{stem}] 키워드 점수 산출 중...")
    keywords = score_keywords(
        concept_degrees, emphasized, slide_texts, transcript_texts, duration_sec,
        debug=True,
    )
    print(f"       → {len(keywords)}개 선택")

    metadata = {
        "video_id":      stem,
        "title":         title,
        "instructor_id": instructor_id,
        "duration_sec":  round(duration_sec, 1),
        "domain":        domain,
        "summary":       summary,
        "keywords":      keywords,
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
    parser.add_argument("--stem",          required=True,           help="강의 식별자 (예: os1-1)")
    parser.add_argument("--title",         required=True,           help="강의 제목")
    parser.add_argument("--instructor_id", required=True,           help="교수 ID")
    parser.add_argument("--output_dir",    default="output",        help="fused.json 위치")
    parser.add_argument("--metadata_dir",  default="metadata",      help="메타데이터 저장 디렉토리")
    args = parser.parse_args()

    generate_metadata(
        stem          = args.stem,
        title         = args.title,
        instructor_id = args.instructor_id,
        output_dir    = Path(args.output_dir),
        metadata_dir  = Path(args.metadata_dir),
    )