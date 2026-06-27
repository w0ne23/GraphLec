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

from __future__ import annotations

import os
import re
import json
import math
import argparse
import random
import time
from pathlib import Path
from collections import Counter
from difflib import SequenceMatcher

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
GEMINI_METADATA_MAX_ATTEMPTS = int(os.getenv("GRAPHLEC_STAGE8_GEMINI_MAX_ATTEMPTS", "5"))
GEMINI_METADATA_BACKOFF_BASE_SEC = float(os.getenv("GRAPHLEC_STAGE8_GEMINI_BACKOFF_BASE_SEC", "10"))
GEMINI_METADATA_BACKOFF_MAX_SEC = float(os.getenv("GRAPHLEC_STAGE8_GEMINI_BACKOFF_MAX_SEC", "90"))


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _sample_uniform(texts: list[str], n: int) -> list[str]:
    """리스트에서 균등 간격으로 n개 샘플링"""
    if len(texts) <= n:
        return texts
    step = len(texts) / n
    return [texts[int(i * step)] for i in range(n)]

DOMAIN_TYPES = {
    "engineering",
    "natural_science",
    "humanities",
    "social_science",
    "arts",
    "health_sciences",
    "sports",
    "education",
    "etc",
}

DOMAIN_LIST = "engineering, natural_science, humanities, social_science, arts, health_sciences, sports, education, etc"

GRAPH_DOMAIN_TO_METADATA_DOMAIN = {
    ("engineering", "computer_science"): "eng/cs",
    ("engineering", "electrical_engineering"): "eng/electrical",
    ("engineering", "mechanical_engineering"): "eng/mechanical",
    ("engineering", "civil_engineering"): "eng/civil",
    ("engineering", "chemical_engineering"): "eng/chemical",
    ("engineering", "industrial_engineering"): "eng/industrial",
    ("engineering", "biomedical_engineering"): "eng/biomedical",
    ("engineering", "aerospace_engineering"): "eng/aerospace",
    ("engineering", "materials_engineering"): "eng/materials",
    ("engineering", "environmental_engineering"): "eng/environmental",
    ("natural_science", "physics"): "sci/physics",
    ("natural_science", "chemistry"): "sci/chemistry",
    ("natural_science", "biology"): "sci/biology",
    ("natural_science", "earth_science"): "sci/earth_science",
    ("natural_science", "astronomy"): "sci/astronomy",
    ("natural_science", "ecology"): "sci/ecology",
    ("humanities", "philosophy"): "hum/philosophy",
    ("humanities", "history"): "hum/history",
    ("humanities", "linguistics"): "hum/linguistics",
    ("humanities", "literature"): "hum/literature",
    ("humanities", "art_history"): "hum/art_history",
    ("humanities", "religion"): "hum/religion",
    ("social_science", "economics"): "soc/economics",
    ("social_science", "business"): "soc/business",
    ("social_science", "law"): "soc/law",
    ("social_science", "political_science"): "soc/political_science",
    ("social_science", "sociology"): "soc/sociology",
    ("social_science", "psychology"): "soc/psychology",
    ("social_science", "education"): "soc/education",
    ("health_sciences", "anatomy"): "med/anatomy",
    ("health_sciences", "physiology"): "med/physiology",
    ("health_sciences", "pharmacology"): "med/pharmacology",
    ("health_sciences", "clinical"): "med/clinical",
    ("health_sciences", "public_health"): "med/public_health",
    ("health_sciences", "nursing"): "med/nursing",
    ("arts", "fine_arts"): "art/fine_arts",
    ("arts", "music"): "art/music",
    ("arts", "design"): "art/design",
    ("arts", "film"): "art/film",
    ("arts", "theater"): "art/theater",
    ("sports", "physical_education"): "art/physical_education",
    ("sports", "sports_science"): "art/sports_science",
    ("education", "education"): "soc/education",
}

GRAPH_DOMAIN_DEFAULTS = {
    "arts": "art/fine_arts",
    "education": "soc/education",
    "health_sciences": "med/public_health",
    "humanities": "hum/philosophy",
    "natural_science": "sci/biology",
    "social_science": "soc/sociology",
    "sports": "art/sports_science",
    "etc": "gen/other",
}

# 키워드 점수 가중치
W_FREQ       = 0.4
W_EMPHASIS   = 0.2
W_CENTRALITY = 0.4

# concept_roles 분류 임계값
CORE_CENT_THRESHOLD  = 0.5   # norm_cent 이 이상이면 core 후보
CORE_EMPH_THRESHOLD  = 0.35  # norm_emph + norm_slide_freq 조합 기준
CORE_FREQ_THRESHOLD  = 0.25
MAX_COMMUNITIES_IN_METADATA = int(os.getenv("GRAPHLEC_METADATA_MAX_COMMUNITIES", "12"))
MAX_COMMUNITY_SUMMARY_CHARS = int(os.getenv("GRAPHLEC_METADATA_COMMUNITY_SUMMARY_CHARS", "800"))
MAX_VISUAL_CONCEPT_TERMS = int(os.getenv("GRAPHLEC_METADATA_MAX_VISUAL_TERMS", "80"))
_VISUAL_TERM_RE = re.compile(r"[0-9A-Za-z가-힣_#+./-]+")
_KOREAN_SYLLABLE_RE = re.compile(r"[가-힣]")
METADATA_DOMAIN_ALIASES = {
    "eng": "engineering",
    "sci": "natural_science",
    "hum": "humanities",
    "soc": "social_science",
    "med": "health_sciences",
    "art": "arts",
    "gen": "etc",
    "math": "natural_science",
}

APPLICATION_KEYWORDS = (
    "예를 들어",
    "예시",
    "예제",
    "사례",
    "적용",
    "활용",
    "실행",
    "시연",
    "데모",
    "실습",
    "문제",
    "풀이",
    "풀어보",
    "계산해",
    "구현",
    "코드",
    "실험",
    "시뮬레이션",
)


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


def _normalize_graph_domain_token(value: str) -> str:
    return str(value or "").strip().lower().replace("-", "_")


def _metadata_domain_from_graph_domain(domain: str, subdomain: str) -> str:
    domain = _normalize_graph_domain_token(domain)
    subdomain = _normalize_graph_domain_token(subdomain)
    if domain in DOMAIN_TYPES:
        return domain
    if not domain:
        return "etc"
    mapped = GRAPH_DOMAIN_TO_METADATA_DOMAIN.get((domain, subdomain))
    if mapped:
        mapped_top = _normalize_graph_domain_token(mapped.split("/", 1)[0])
        return METADATA_DOMAIN_ALIASES.get(mapped_top, "etc")
    if domain == "engineering" and subdomain:
        if "computer" in subdomain or subdomain in {"cs", "software", "operating_systems"}:
            return "engineering"
        if "electrical" in subdomain:
            return "engineering"
        if "mechanical" in subdomain:
            return "engineering"
        if "civil" in subdomain:
            return "engineering"
        if "chemical" in subdomain:
            return "engineering"
        if "biomedical" in subdomain:
            return "engineering"
        if "aerospace" in subdomain:
            return "engineering"
        if "material" in subdomain:
            return "engineering"
        if "environment" in subdomain:
            return "engineering"
        if "industrial" in subdomain:
            return "engineering"
    return domain if domain in DOMAIN_TYPES else "etc"


def load_graph_domain(stem: str, output_dir: Path) -> tuple[str, str, str]:
    nodes_path = output_dir / f"{stem}_nodes.parquet"
    if not nodes_path.is_file():
        return "", "", ""
    try:
        import pandas as pd

        ndf = pd.read_parquet(nodes_path)
    except Exception:
        return "", "", ""
    video_id = f"lecture_video/{stem}"
    for _, row in ndf.iterrows():
        if str(row.get("node_id") or "").strip() != video_id:
            continue
        try:
            props = json.loads(str(row.get("properties_json") or "{}"))
        except Exception:
            props = {}
        graph_domain = _normalize_graph_domain_token(props.get("domain"))
        graph_subdomain = _normalize_graph_domain_token(props.get("subdomain"))
        metadata_domain = _metadata_domain_from_graph_domain(graph_domain, graph_subdomain)
        return metadata_domain, graph_domain, graph_subdomain
    return "", "", ""


def fused_scene_entries(fused: dict) -> list[dict]:
    return fused.get("scenes", [])


def get_duration(fused: dict) -> float:
    """마지막 슬라이드 end_sec 기준 총 길이(초)"""
    slides = fused_scene_entries(fused)
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

    for slide in fused_scene_entries(fused):
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
    for slide in fused_scene_entries(fused):
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


def collect_pedagogy(fused: dict) -> dict:
    """
    슬라이드 타입 분포로 강의 전달 방식(pedagogy) 메타데이터 추출.

    slide_type 기준:
      - image_only : 이미지·다이어그램만 있는 슬라이드 (시각 자료 전용)
      - mixed      : 텍스트 + 이미지 혼합 (절반 기여)
      - text       : 텍스트만 있는 슬라이드

    visual_ratio 계산:
      (image_only 수 + mixed 수 × 0.5) / 전체 슬라이드 수

    style_tags:
      visual_ratio ≥ 0.5 → "diagram-heavy"
      visual_ratio ≥ 0.3 → "visual-supported"
      그 외              → 태그 없음
    """
    slides = [
        s for s in fused_scene_entries(fused)
        if s.get("role") != "objectives"  # 학습목표 슬라이드 제외
    ]
    total = len(slides)
    if total == 0:
        return {"visual_ratio": 0.0, "style_tags": []}

    image_only_count = sum(1 for s in slides if s.get("slide_type") == "image_only")
    mixed_count      = sum(1 for s in slides if s.get("slide_type") == "mixed")
    structure_count  = sum(
        1 for s in slides
        if str(s.get("t1_structure") or "").strip()
    )

    # image_only는 전체 기여, mixed는 절반 기여
    visual_ratio = round((image_only_count + mixed_count * 0.5) / total, 3)
    structure_ratio = round(structure_count / total, 3)

    style_tags: list[str] = []
    if visual_ratio >= 0.5:
        style_tags.append("diagram-heavy")
    elif visual_ratio >= 0.3:
        style_tags.append("visual-supported")

    print(
        f"[디버그] pedagogy: visual_ratio={visual_ratio:.3f} "
        f"structure_ratio={structure_ratio:.3f} "
        f"(image_only={image_only_count}, mixed={mixed_count}, "
        f"structure={structure_count}, total={total}) "
        f"tags={style_tags}"
    )

    return {
        "visual_ratio":    visual_ratio,
        "structure_ratio": structure_ratio,
        "style_tags":      style_tags,
    }


def _load_json_optional(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _segments_from_fused(fused: dict) -> list[dict]:
    segments: list[dict] = []
    for scene in fused_scene_entries(fused):
        for ctx in scene.get("contexts", []) or []:
            for seg in ctx.get("segments", []) or []:
                if isinstance(seg, dict):
                    segments.append(seg)
    return segments


def _load_segments_for_diagnostics(stem: str, output_dir: Path, fused: dict) -> tuple[list[dict], str]:
    payload = _load_json_optional(output_dir / f"{stem}_segments.json")
    segments = payload.get("segments")
    if isinstance(segments, list):
        return [s for s in segments if isinstance(s, dict)], "segments_json"
    return _segments_from_fused(fused), "fused_contexts"


def _count_korean_syllables(text: str) -> int:
    return len(_KOREAN_SYLLABLE_RE.findall(str(text or "")))


def _collect_speech_rate_spm(segments: list[dict]) -> dict:
    syllable_count = 0
    speech_duration_sec = 0.0
    usable_segments = 0
    total_segments = len(segments)

    for seg in segments:
        text = str(seg.get("text") or "")
        count = _count_korean_syllables(text)
        try:
            start = float(seg.get("start", 0.0) or 0.0)
            end = float(seg.get("end", start) or start)
        except Exception:
            start = 0.0
            end = 0.0
        duration = max(0.0, end - start)
        if duration <= 0 or count <= 0:
            continue
        syllable_count += count
        speech_duration_sec += duration
        usable_segments += 1

    speech_duration_min = speech_duration_sec / 60.0
    spm = syllable_count / speech_duration_min if speech_duration_min > 0 else None
    return {
        "speech_rate_spm": round(spm, 2) if spm is not None else None,
        "korean_syllable_count": syllable_count,
        "speech_duration_sec": round(speech_duration_sec, 3),
        "speech_segment_count": usable_segments,
        "total_segment_count": total_segments,
        "excluded_segment_count": max(0, total_segments - usable_segments),
    }


def _has_application_keyword(text: str) -> bool:
    lowered = str(text or "").lower()
    return any(keyword.lower() in lowered for keyword in APPLICATION_KEYWORDS)


def _ratio_with_application_keywords(texts: list[str]) -> tuple[float, int, int]:
    cleaned = [str(t or "").strip() for t in texts if str(t or "").strip()]
    if not cleaned:
        return 0.0, 0, 0
    hits = sum(1 for text in cleaned if _has_application_keyword(text))
    return round(hits / len(cleaned), 4), hits, len(cleaned)


def _collect_application_orientation(fused: dict, segments: list[dict]) -> dict:
    transcript_texts = [str(seg.get("text") or "") for seg in segments]
    slide_texts: list[str] = []
    for scene in fused_scene_entries(fused):
        if scene.get("role") == "objectives":
            continue
        parts = [
            str(scene.get("title") or ""),
            str(scene.get("slide_text") or ""),
            str(scene.get("t1_structure") or ""),
        ]
        for asset in scene.get("visual_assets") or []:
            if not isinstance(asset, dict):
                continue
            parts.extend(str(asset.get(key) or "") for key in ("type", "label", "role", "meaning", "description"))
        slide_texts.append(" ".join(parts))

    transcript_ratio, transcript_hits, transcript_total = _ratio_with_application_keywords(transcript_texts)
    slide_ratio, slide_hits, slide_total = _ratio_with_application_keywords(slide_texts)
    score = round(0.5 * transcript_ratio + 0.5 * slide_ratio, 4)
    return {
        "application_orientation_score": score,
        "transcript_application_signal": transcript_ratio,
        "slide_application_signal": slide_ratio,
        "matched_transcript_segments": transcript_hits,
        "transcript_segment_count": transcript_total,
        "matched_slides": slide_hits,
        "slide_count": slide_total,
        "keyword_set": list(APPLICATION_KEYWORDS),
    }


def _collect_visual_ratio(fused: dict, pedagogy: dict) -> dict:
    slides = [
        s for s in fused_scene_entries(fused)
        if s.get("role") != "objectives"
    ]
    total = len(slides)
    asset_count = sum(1 for s in slides if s.get("visual_assets"))
    if total > 0 and asset_count > 0:
        return {
            "visual_ratio": round(asset_count / total, 4),
            "visual_slide_count": asset_count,
            "slide_count": total,
            "source": "visual_assets",
        }
    return {
        "visual_ratio": round(float(pedagogy.get("visual_ratio", 0.0) or 0.0), 4),
        "visual_slide_count": asset_count,
        "slide_count": total,
        "source": "pedagogy_visual_ratio",
    }


def _collect_listenability(stem: str, output_dir: Path) -> dict:
    audio_quality = _load_json_optional(output_dir / f"{stem}_audio_quality.json")
    details = audio_quality.get("details", {}) if isinstance(audio_quality.get("details"), dict) else {}
    rms = details.get("rms_energy", {}) if isinstance(details.get("rms_energy"), dict) else {}
    mfcc = details.get("mfcc_stability", {}) if isinstance(details.get("mfcc_stability"), dict) else {}

    def _score(row: dict) -> float | None:
        try:
            return max(0.0, min(100.0, float(row.get("score")))) / 100.0
        except Exception:
            return None

    volume_score = _score(rms)
    stability_score = _score(mfcc)
    scores = [s for s in (volume_score, stability_score) if s is not None]
    listenability_score = round(sum(scores) / len(scores), 4) if scores else None
    return {
        "listenability_score": listenability_score,
        "volume_score": volume_score,
        "voice_stability_score": stability_score,
        "rms_energy": rms.get("value"),
        "mfcc_stability": mfcc.get("value"),
        "source": "audio_quality" if scores else "missing_audio_quality",
    }


def collect_diagnostics(stem: str, output_dir: Path, fused: dict, pedagogy: dict) -> dict:
    segments, segment_source = _load_segments_for_diagnostics(stem, output_dir, fused)
    speech_rate = _collect_speech_rate_spm(segments)
    application = _collect_application_orientation(fused, segments)
    visual = _collect_visual_ratio(fused, pedagogy)
    listenability = _collect_listenability(stem, output_dir)

    delivery = {
        **speech_rate,
        **listenability,
        "segment_source": segment_source,
    }
    teaching_style = {
        **visual,
        **application,
    }
    print(
        f"[디버그] diagnostics: spm={delivery.get('speech_rate_spm')} "
        f"listenability={delivery.get('listenability_score')} "
        f"visual_ratio={teaching_style.get('visual_ratio')} "
        f"application={teaching_style.get('application_orientation_score')}"
    )
    return {
        "delivery": delivery,
        "teaching_style": teaching_style,
    }


def _iter_t1_structure_texts(fused: dict) -> list[str]:
    """scene 기준 t1_structure 수집. 구형 fused는 slides에서 fallback한다."""
    texts = []
    seen = set()
    slide_by_id = {
        slide.get("slide_id"): slide
        for slide in fused.get("slides", [])
        if slide.get("slide_id")
    }

    for scene in fused_scene_entries(fused):
        if scene.get("role") == "objectives":
            continue
        text = str(scene.get("t1_structure") or "").strip()
        if not text:
            slide = slide_by_id.get(scene.get("slide_id"))
            text = str((slide or {}).get("t1_structure") or "").strip()
        if text and text not in seen:
            seen.add(text)
            texts.append(text)

    if texts:
        return texts

    for slide in fused.get("slides", []):
        text = str(slide.get("t1_structure") or "").strip()
        if text and text not in seen:
            seen.add(text)
            texts.append(text)
    return texts


def collect_visual_concept_terms(fused: dict, concept_terms: list[str]) -> list[str]:
    """
    t1_structure 안에서 실제 강의 개념으로 등장하는 term을 미리 저장한다.
    추천 시점에는 query term과 set intersection만 수행하도록 비용을 앞당긴다.
    """
    structure_text = "\n".join(_iter_t1_structure_texts(fused))
    if not structure_text.strip():
        return []

    structure_lower = structure_text.lower()
    candidates = []
    for term in concept_terms:
        term = str(term or "").strip()
        if term and _is_valid_concept(term):
            candidates.append(term)

    matched = []
    seen = set()
    for term in candidates:
        key = term.lower()
        if key in seen:
            continue
        if key in structure_lower:
            seen.add(key)
            matched.append(term)
            if len(matched) >= MAX_VISUAL_CONCEPT_TERMS:
                break

    return matched


def collect_graphrag_communities(output_dir: Path) -> list[dict]:
    """GraphRAG community_reports를 추천용 metadata에 저장할 compact 구조로 변환."""
    reports_path = output_dir / "graphrag" / "output" / "community_reports.parquet"
    if not reports_path.exists():
        return []

    try:
        import pyarrow.parquet as pq
    except ImportError:
        print("[경고] pyarrow 없음 — GraphRAG communities metadata 저장 생략")
        return []

    try:
        table = pq.read_table(reports_path)
    except Exception as exc:
        print(f"[경고] community_reports 로드 실패: {exc}")
        return []

    rows = table.to_pylist()
    if not rows:
        return []

    if "rank" in table.column_names:
        rows.sort(
            key=lambda row: as_float(row.get("rank")),
            reverse=True,
        )

    communities = []
    for row in rows[:MAX_COMMUNITIES_IN_METADATA]:
        title = str(row.get("title") or "").strip()
        summary = str(row.get("summary") or "").strip()
        if not title and not summary:
            continue
        if len(summary) > MAX_COMMUNITY_SUMMARY_CHARS:
            summary = summary[:MAX_COMMUNITY_SUMMARY_CHARS].rstrip() + "..."

        def as_int(value, default=0):
            try:
                return int(value)
            except (TypeError, ValueError):
                return default

        communities.append({
            "id":        str(row.get("id") or row.get("community") or ""),
            "community": str(row.get("community") or ""),
            "level":     as_int(row.get("level")),
            "title":     title,
            "summary":   summary,
            "rank":      round(as_float(row.get("rank")), 4),
            "size":      as_int(row.get("size")),
        })

    print(f"[디버그] GraphRAG communities metadata 저장: {len(communities)}개")
    return communities


def as_float(value, default=0.0):
    try:
        parsed = float(value)
        return parsed if math.isfinite(parsed) else default
    except (TypeError, ValueError):
        return default


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

    for slide in fused_scene_entries(fused):
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
    for slide in fused_scene_entries(fused):
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


# ── concept complexity 추정 ───────────────────────────────────────────────────

def estimate_concept_complexity(
    concept_roles: dict[str, list[str]],
    norm_cent:     dict[str, float],
) -> str:
    """
    core 개념들의 평균 그래프 중심성으로 개념 복잡도 추정

    직관: core 개념이 그래프에서 촘촘하게 연결될수록 → 개념 밀도 높은 강의
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
        concept_complexity = "high"
    elif avg_core_cent >= 0.35:
        concept_complexity = "medium"
    else:
        concept_complexity = "low"

    print(
        f"[디버그] concept_complexity: {concept_complexity}  "
        f"(core={len(cores)}, avg_core_cent={avg_core_cent:.3f})"
    )
    return concept_complexity


# ── LLM 호출 ──────────────────────────────────────────────────────────────────

def _is_retryable_gemini_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code in {429, 500, 502, 503, 504}:
        return True

    text = str(exc).lower()
    retryable_markers = (
        "429",
        "500",
        "502",
        "503",
        "504",
        "unavailable",
        "high demand",
        "rate limit",
        "resource exhausted",
        "deadline exceeded",
        "temporarily",
    )
    return any(marker in text for marker in retryable_markers)


def _gemini(prompt: str) -> str:
    max_attempts = max(1, GEMINI_METADATA_MAX_ATTEMPTS)
    last_error: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            resp = _client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config={"temperature": 0.0},
            )
            break
        except Exception as exc:
            last_error = exc
            if attempt >= max_attempts or not _is_retryable_gemini_error(exc):
                raise

            delay = min(
                GEMINI_METADATA_BACKOFF_MAX_SEC,
                GEMINI_METADATA_BACKOFF_BASE_SEC * (2 ** (attempt - 1)),
            )
            jitter = random.uniform(0.0, min(3.0, delay * 0.15))
            wait_sec = delay + jitter
            print(
                f"[경고] Gemini metadata 호출 실패({type(exc).__name__}: {exc}). "
                f"{wait_sec:.1f}초 후 재시도 {attempt + 1}/{max_attempts}"
            )
            time.sleep(wait_sec)
    else:
        raise RuntimeError("Gemini metadata 호출 재시도에 실패했습니다.") from last_error

    try:
        from .cost_report import record_model_call

        record_model_call(
            stage="stage8_metadata",
            provider="google",
            model=MODEL,
            response=resp,
            prompt_chars=len(prompt),
        )
    except Exception:
        pass
    return resp.text.strip()


def classify_domain(slide_texts: list[str], transcript_texts: list[str]) -> str:
    sample_slide = " ".join(slide_texts[:5])[:800]
    sample_trans = " ".join(transcript_texts[:20])[:800]
    prompt = f"""아래는 강의 내용 일부다.

슬라이드: {sample_slide}
전사: {sample_trans}

다음 도메인 목록 중 가장 적합한 것 하나만 출력하라. 설명 없이 도메인 코드만 출력.
판단이 애매하면 etc를 출력하라.
예시 출력: engineering

도메인 목록:
{DOMAIN_LIST}
"""
    result = _gemini(prompt).strip().lower()
    token = result.strip().split()[0].strip("`\"'.,:{}[]()").replace("-", "_")
    return token if token in DOMAIN_TYPES else "etc"


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
        normalized = path
        if normalized in seen:
            continue
        seen.add(normalized)
        if normalized.is_file():
            return normalized
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


def load_graph_artifacts_for_metadata(stem: str, output_dir: Path, fused: dict) -> dict:
    """
    추천 메타데이터 graph-first 경로에서 사용할 파일 기반 그래프 산출물.

    Neo4j는 metadata 생성 시점의 필수 입력이 아니므로 여기서 호출하지 않는다.
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

    emphasis = collect_emphasized(fused)
    return {
        "graphrag_output_dir": str(graph_dir) if graph_dir else "",
        "graphrag": graphrag,
        "stage_graph": stage_graph,
        "emphasis": emphasis,
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
            "emphasis_terms": len(emphasis),
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


_EXAMPLE_RELATION_TERMS = (
    "example",
    "instance",
    "instance_of",
    "kind",
    "type",
    "종류",
    "예시",
    "사례",
    "대표적인",
    "중 하나",
    "중의 하나",
    "예로",
    "예에는",
    "포함된다",
    "포함되는",
)


_OPERATIONAL_TITLE_TERMS = (
    "교수",
    "강사",
    "시험",
    "출석",
    "평가",
    "학점",
    "과제",
    "퀴즈",
    "중간고사",
    "기말고사",
    "참고 문헌",
    "참고문헌",
    "교재",
    "주차",
    "공지",
    "대학교",
    "대학",
    "학과",
    "처장",
    "총장",
)

_GENERIC_SUFFIX_TERMS = (
    "개념",
    "정의",
    "기능",
    "목적",
    "특징",
    "원리",
    "이해",
    "소개",
    "기초",
    "관련",
)

_MERGE_BRIDGE_TERMS = (
    "와",
    "과",
    "및",
    "또는",
    "그리고",
    "차이",
    "관계",
    "비교",
    "대조",
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
    if not typo_like:
        return False
    return True


def _has_independent_graph_signal(a: dict, b: dict) -> bool:
    return (
        float(a.get("graph_importance", 0.0)) >= 0.12
        and float(b.get("graph_importance", 0.0)) >= 0.12
        and float(a.get("community_inner_representative", 0.0)) >= 0.20
        and float(b.get("community_inner_representative", 0.0)) >= 0.20
    )


def _should_merge_keyword_candidates(
    a: dict,
    b: dict,
) -> bool:
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


def _merge_graph_keyword_candidates(
    candidates: list[dict],
    limit: int,
) -> list[dict]:
    clusters: list[list[dict]] = []

    for candidate in sorted(candidates, key=lambda item: item["score"], reverse=True):
        target_cluster = None
        for cluster in clusters:
            if any(
                _should_merge_keyword_candidates(
                    candidate,
                    existing,
                )
                for existing in cluster
            ):
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
            entity,
            slide_texts,
            transcript_texts,
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
                round(operational_score_cap, 4)
                if operational_score_cap is not None else None
            ),
            "code_like_penalty": round(code_like_penalty, 4),
            "penalty_strength": round(penalty_strength, 4),
            "raw_degree": round(_to_float(entity.get("degree")), 4),
            "raw_frequency": round(_to_float(entity.get("frequency")), 4),
        })

    candidates.sort(key=lambda item: item["score"], reverse=True)
    return _merge_graph_keyword_candidates(
        candidates,
        limit,
    )


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

    def add_keyword(keyword: str, score: float) -> None:
        key = _norm_text(keyword)
        if not keyword or not key or key in seen:
            return
        seen.add(key)
        merged.append({
            "keyword": keyword,
            "score": round(score, 4),
        })

    for candidate in graph_candidates:
        if len(merged) >= target_count:
            break
        add_keyword(
            str(candidate.get("keyword") or "").strip(),
            _to_float(candidate.get("score")),
        )

    for keyword in legacy_keywords:
        if len(merged) >= target_count:
            break
        add_keyword(
            str(keyword.get("keyword") or "").strip(),
            _to_float(keyword.get("score")),
        )

    return merged


def _graph_keyword_debug_summary(
    keyword_candidates: list[dict],
    artifacts: dict,
) -> dict:
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

    return {
        "core": core,
        "introduced": introduced[:40],
    }


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


def try_generate_graph_first_metadata_parts(
    *,
    stem: str,
    output_dir: Path,
    fused: dict,
    concept_degrees: dict[str, int],
    emphasized: dict[str, float],
    slide_texts: list[str],
    transcript_texts: list[str],
    core_slide_texts: list[str],
    core_trans_texts: list[str],
    duration_sec: float,
) -> dict | None:
    """
    Graph-first 추천 메타데이터 생성 진입점.

    GraphRAG 파일 산출물로 keywords, concept_roles, concept_relations를 구성한다.
    summary는 아직 기존 텍스트 요약 경로를 재사용한다.
    """
    artifacts = load_graph_artifacts_for_metadata(stem, output_dir, fused)
    _log_graph_artifact_summary(stem, artifacts)
    keyword_candidates = build_graph_keyword_candidates(
        artifacts,
        slide_texts,
        transcript_texts,
    )
    _log_graph_keyword_candidates(stem, keyword_candidates)
    alias_map = build_graph_candidate_alias_map(keyword_candidates)
    if alias_map:
        preview = list(alias_map.items())[:8]
        print(
            f"[{stem}] graph variant aliases {len(alias_map)}개 감지: "
            f"{preview}"
        )

    if not keyword_candidates:
        print(f"[{stem}] graph keyword candidates 없음 — 기존 로직으로 fallback")
        return {
            "metadata_parts": None,
            "debug": {
                "keyword_source": "legacy_text_graph_mixed",
                **_graph_keyword_debug_summary(keyword_candidates, artifacts),
            },
        }

    print(f"[{stem}] graph-first summary는 기존 로직 재사용, roles/relations는 GraphRAG 기반 생성")
    summary = generate_summary(core_slide_texts, core_trans_texts, concept_degrees)

    legacy_keywords, _legacy_scored, _legacy_slide_freq, _legacy_trans_freq, _legacy_emph, _legacy_cent = score_keywords(
        concept_degrees,
        emphasized,
        slide_texts,
        transcript_texts,
        duration_sec,
        debug=False,
    )
    target_count = len(legacy_keywords) or min(30, max(7, len(keyword_candidates)))
    keywords = _merge_graph_keywords_with_legacy(
        keyword_candidates,
        legacy_keywords,
        target_count,
    )
    print(
        f"[{stem}] graph-first keywords {len(keywords)}개 선택 "
        f"(target={target_count}, graph_candidates={len(keyword_candidates)}, "
        f"legacy_candidates={len(legacy_keywords)})"
    )
    concept_roles = build_graph_concept_roles(
        keyword_candidates,
        [item.get("keyword", "") for item in keywords],
    )
    concept_roles = {
        role: _canonicalize_graph_name_list(names, alias_map)
        for role, names in concept_roles.items()
    }
    role_candidates = set(concept_roles.get("core", [])) | set(concept_roles.get("introduced", []))
    concept_relations = build_graph_concept_relations(artifacts, concept_roles, alias_map)
    scored, norm_slide_freq, norm_trans_freq, norm_emph, norm_cent = build_graph_role_scores(
        keyword_candidates
    )
    print(
        f"[{stem}] graph-first concept_roles "
        f"core={len(concept_roles.get('core', []))}, "
        f"introduced={len(concept_roles.get('introduced', []))}, "
        f"relations={len(concept_relations)}"
    )

    visual_concept_terms = collect_visual_concept_terms(
        fused,
        [k.get("keyword", "") for k in keywords]
        + list(concept_degrees.keys())
        + list(scored.keys())
        + list(alias_map.keys()),
    )
    visual_concept_terms = _canonicalize_graph_name_list(visual_concept_terms, alias_map)

    return {
        "metadata_parts": {
            "summary": summary,
            "keywords": keywords,
            "scored": scored,
            "norm_slide_freq": norm_slide_freq,
            "norm_trans_freq": norm_trans_freq,
            "norm_emph": norm_emph,
            "norm_cent": norm_cent,
            "role_candidates": role_candidates,
            "concept_roles": concept_roles,
            "concept_relations": concept_relations,
            "visual_concept_terms": visual_concept_terms,
        },
        "debug": {
            "keyword_source": "graph_first_with_legacy_fallback",
            **_graph_keyword_debug_summary(keyword_candidates, artifacts),
        },
    }


# ── 메인 ──────────────────────────────────────────────────────────────────────

def generate_metadata(
    stem:          str,
    title:         str,
    instructor_id: str,
    output_dir:    Path,
    metadata_dir:  Path,
    uploaded_at:   str | None = None,
) -> dict:
    print(f"[{stem}] fused.json 로드 중...")
    fused = load_fused(stem, output_dir)

    duration_sec = get_duration(fused)
    slide_texts, transcript_texts, core_slide_texts, core_trans_texts = collect_texts(fused)
    emphasized   = collect_emphasized(fused)
    pedagogy     = collect_pedagogy(fused)
    diagnostics  = collect_diagnostics(stem, output_dir, fused, pedagogy)

    # objectives 슬라이드는 요약/키워드 연산과 독립적으로 별도 수집
    learning_objectives = collect_learning_objectives(fused)
    print(f"[{stem}] 학습 목표 {len(learning_objectives)}개 수집")

    graph_first_enabled = _env_flag("GRAPHLEC_METADATA_GRAPH_FIRST")
    if graph_first_enabled:
        print(f"[{stem}] graph-first metadata 모드: Neo4j Concept 조회 건너뜀")
        concept_degrees = {}
    else:
        print(f"[{stem}] Neo4j Concept 노드 조회 중...")
        concept_degrees = fetch_concept_degrees(stem)
        print(f"       → Concept {len(concept_degrees)}개")

    metadata_domain, graph_domain, graph_subdomain = load_graph_domain(stem, output_dir)
    if metadata_domain:
        print(
            f"[{stem}] Stage 6 도메인 재사용: "
            f"{graph_domain}/{graph_subdomain or '-'} → {metadata_domain}"
        )
        domain = metadata_domain
    else:
        print(f"[{stem}] 도메인 분류 중...")
        domain = classify_domain(slide_texts, transcript_texts)

    graph_first_parts = None
    graph_first_debug = {}
    if graph_first_enabled:
        print(f"[{stem}] graph-first metadata 모드 활성화")
        try:
            graph_first_result = try_generate_graph_first_metadata_parts(
                stem=stem,
                output_dir=output_dir,
                fused=fused,
                concept_degrees=concept_degrees,
                emphasized=emphasized,
                slide_texts=slide_texts,
                transcript_texts=transcript_texts,
                core_slide_texts=core_slide_texts,
                core_trans_texts=core_trans_texts,
                duration_sec=duration_sec,
            )
            if graph_first_result:
                graph_first_parts = graph_first_result.get("metadata_parts")
                graph_first_debug = graph_first_result.get("debug") or {}
        except Exception as exc:
            print(f"[경고] graph-first metadata 생성 실패 — 기존 로직으로 fallback: {exc}")

    if graph_first_parts:
        summary = graph_first_parts["summary"]
        keywords = graph_first_parts["keywords"]
        scored = graph_first_parts["scored"]
        norm_slide_freq = graph_first_parts["norm_slide_freq"]
        norm_trans_freq = graph_first_parts["norm_trans_freq"]
        norm_emph = graph_first_parts["norm_emph"]
        norm_cent = graph_first_parts["norm_cent"]
        role_candidates = graph_first_parts["role_candidates"]
        concept_roles = graph_first_parts["concept_roles"]
        concept_relations = graph_first_parts.get("concept_relations", [])
        visual_concept_terms = graph_first_parts["visual_concept_terms"]
    else:
        if graph_first_enabled and not concept_degrees:
            print(f"[{stem}] graph-first fallback: Neo4j Concept 노드 조회 중...")
            concept_degrees = fetch_concept_degrees(stem)
            print(f"       → Concept {len(concept_degrees)}개")
        print(f"[{stem}] 요약 생성 중...")
        summary = generate_summary(core_slide_texts, core_trans_texts, concept_degrees)

        print(f"[{stem}] 키워드 점수 산출 중...")
        keywords, scored, norm_slide_freq, norm_trans_freq, norm_emph, norm_cent = score_keywords(
            concept_degrees, emphasized, slide_texts, transcript_texts, duration_sec,
            debug=True,
        )
        print(f"       → {len(keywords)}개 선택")
        visual_concept_terms = collect_visual_concept_terms(
            fused,
            [k.get("keyword", "") for k in keywords]
            + list(concept_degrees.keys())
            + list(scored.keys()),
        )
        print(f"       → visual_concept_terms {len(visual_concept_terms)}개")

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

    if graph_first_parts:
        print(f"[{stem}] graph-first concept_relations {len(concept_relations)}개 사용")
    else:
        print(f"[{stem}] concept_relations 조회 중...")
        concept_relations = fetch_concept_relations(stem, role_candidates)

    # concept_complexity: concept_roles 결과 + norm_cent 활용 (LLM 호출 없음)
    concept_complexity = estimate_concept_complexity(concept_roles, norm_cent)

    print(f"[{stem}] GraphRAG community report 수집 중...")
    communities = collect_graphrag_communities(output_dir)

    metadata = {
        "video_id":            stem,
        "title":               title,
        "instructor_id":       instructor_id,
        "uploaded_at":         uploaded_at,
        "duration_sec":        round(duration_sec, 1),
        "domain":              domain,
        "graph_domain":        graph_domain,
        "graph_subdomain":     graph_subdomain,
        "concept_complexity":  concept_complexity,
        "summary":             summary,
        "learning_objectives": learning_objectives,
        "keywords":            keywords,
        "concept_roles":       concept_roles,
        "concept_relations":   concept_relations,
        "communities":         communities,
        "visual_concept_terms": visual_concept_terms,
        "pedagogy":            pedagogy,
        "diagnostics":         diagnostics,
    }
    if graph_first_debug:
        metadata.update(graph_first_debug)

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
    parser.add_argument("--uploaded-at", "--uploaded_at", dest="uploaded_at", default=None,
                        help="업로드 시각 ISO 문자열")
    args = parser.parse_args()

    generate_metadata(
        stem          = args.stem,
        title         = args.title,
        instructor_id = args.instructor_id,
        output_dir    = Path(args.output_dir),
        metadata_dir  = Path(args.metadata_dir),
        uploaded_at   = args.uploaded_at,
    )
