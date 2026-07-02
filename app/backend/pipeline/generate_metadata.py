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

from neo4j import GraphDatabase
from google import genai
from dotenv import load_dotenv

from .config import GEMINI_GENERATIVE_MODEL
from .metadata_graph import (
    _is_valid_concept,
    _normalize,
    _norm_text,
    _stringify_for_match,
    _to_float,
    load_graph_artifacts_for_metadata,
    _log_graph_artifact_summary,
    build_graph_keyword_candidates,
    _log_graph_keyword_candidates,
    build_graph_candidate_alias_map,
    _graph_keyword_debug_summary,
    _merge_graph_keywords_with_legacy,
    build_graph_concept_roles,
    _canonicalize_graph_name_list,
    build_graph_concept_relations,
    build_graph_role_scores,
)

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
    norm_emph:       dict[str, float],
    norm_cent:       dict[str, float],
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


_COMMUNITY_BOILERPLATE_TERMS = (
    "법적",
    "규제",
    "평판",
    "분쟁",
    "악의적",
    "위반",
    "리스크",
    "위험",
)


def _clean_community_summary(summary: str, max_chars: int = 360) -> str:
    sentences = re.split(r"(?<=[.!?。！？])\s+|(?<=다\.)\s*", str(summary or "").strip())
    kept = []
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if any(term in sentence for term in _COMMUNITY_BOILERPLATE_TERMS):
            continue
        kept.append(sentence)
    cleaned = " ".join(kept) if kept else str(summary or "").strip()
    return cleaned[:max_chars].strip()


def _select_summary_community_reports(
    artifacts: dict,
    concept_roles: dict[str, list[str]],
    limit: int = 5,
) -> list[dict]:
    reports = artifacts.get("graphrag", {}).get("community_reports", {}).get("rows", []) or []
    core_names = concept_roles.get("core", []) or []
    introduced_names = concept_roles.get("introduced", []) or []
    terms = core_names + introduced_names[:8]
    if not reports or not terms:
        return []

    selected = []
    for report in reports:
        title = str(report.get("title") or "").strip()
        summary = _clean_community_summary(str(report.get("summary") or ""))
        if not title and not summary:
            continue
        haystack = _norm_text(" ".join([title, summary, _stringify_for_match(report.get("findings"))]))
        mention_score = 0.0
        for index, term in enumerate(terms):
            term_norm = _norm_text(term)
            if term_norm and term_norm in haystack:
                mention_score += 2.0 if index < len(core_names) else 1.0
        if mention_score <= 0:
            continue
        selected.append({
            "title": title,
            "summary": summary,
            "rank": _to_float(report.get("rank")),
            "score": mention_score + _to_float(report.get("rank")) * 0.2,
        })

    selected.sort(key=lambda row: row["score"], reverse=True)
    return selected[:limit]


def generate_graph_first_summary(
    *,
    artifacts: dict,
    keyword_candidates: list[dict],
    concept_roles: dict[str, list[str]],
    concept_relations: list[dict],
    core_slide_texts: list[str],
    core_trans_texts: list[str],
) -> str:
    core_names = concept_roles.get("core", [])[:10]
    introduced_names = concept_roles.get("introduced", [])[:12]
    top_candidates = [
        str(candidate.get("keyword") or "").strip()
        for candidate in keyword_candidates[:12]
        if str(candidate.get("keyword") or "").strip()
    ]
    relation_lines = [
        f"{rel.get('from')} -[{rel.get('type')}]-> {rel.get('to')}"
        for rel in concept_relations[:14]
        if rel.get("from") and rel.get("to")
    ]
    community_reports = _select_summary_community_reports(
        artifacts,
        concept_roles,
    )
    community_block = "\n".join(
        f"- {report.get('title')}: {report.get('summary')}"
        for report in community_reports
    )

    MAX_PER_SLIDE = 120
    slide_block = " / ".join(
        s.strip()[:MAX_PER_SLIDE] for s in core_slide_texts if s.strip()
    )[:1200]

    n_sample = min(24, len(core_trans_texts))
    sampled_trans = _sample_uniform(core_trans_texts, n_sample)
    trans_block = " ".join(sampled_trans)[:1200]

    prompt = f"""아래는 강의의 그래프 기반 메타데이터와 텍스트 근거다.

그래프 핵심 개념:
{', '.join(core_names)}

그래프 보조 개념:
{', '.join(introduced_names)}

그래프 상위 후보:
{', '.join(top_candidates)}

주요 개념 관계:
{chr(10).join(relation_lines)}

커뮤니티 요약:
{community_block}

텍스트 근거 슬라이드:
{slide_block}

텍스트 근거 전사:
{trans_block}

위 정보를 바탕으로 강의 내용을 3문장으로 요약하라.

작성 규칙:
- 그래프 핵심 개념과 관계를 우선 반영하라.
- 텍스트 근거는 그래프 정보가 실제 강의 내용과 맞는지 보조 검증용으로만 사용하라.
- 커뮤니티 보고서의 법적/평판/리스크 관련 boilerplate는 요약하지 마라.
- 단순 키워드 나열이 아니라 강의의 핵심 흐름과 개념 간 관계가 드러나도록 작성하라.
- 한국어로 작성하고 설명 없이 요약문만 출력하라.
"""

    print(f"\n[디버그] graph-first 요약 입력 core={core_names}")
    print(f"[디버그] graph-first 요약 관계 수: {len(relation_lines)}")
    print(f"[디버그] graph-first 요약 community report 수: {len(community_reports)}")
    print(f"[디버그] graph-first slide_block ({len(slide_block)}자):\n{slide_block[:400]}")
    print(f"[디버그] graph-first trans_block ({len(trans_block)}자):\n{trans_block[:240]}\n")
    return _gemini(prompt)


def _fallback_summary_from_graph(concept_roles: dict[str, list[str]]) -> str:
    core = [name for name in concept_roles.get("core", []) if str(name or "").strip()]
    introduced = [name for name in concept_roles.get("introduced", []) if str(name or "").strip()]
    primary = ", ".join(core[:5])
    secondary = ", ".join(introduced[:5])
    if primary and secondary:
        return (
            f"이 강의는 {primary}를 중심으로 관련 개념을 설명한다. "
            f"또한 {secondary}를 함께 다루며 핵심 개념 간 관계를 정리한다."
        )
    if primary:
        return f"이 강의는 {primary}를 중심으로 핵심 개념과 관계를 설명한다."
    return "이 강의는 그래프 기반 핵심 개념과 관계를 중심으로 내용을 설명한다."


def generate_graph_first_summary_with_fallback(
    *,
    artifacts: dict,
    keyword_candidates: list[dict],
    concept_roles: dict[str, list[str]],
    concept_relations: list[dict],
    core_slide_texts: list[str],
    core_trans_texts: list[str],
    concept_degrees: dict[str, int],
) -> tuple[str, str]:
    try:
        return generate_graph_first_summary(
            artifacts=artifacts,
            keyword_candidates=keyword_candidates,
            concept_roles=concept_roles,
            concept_relations=concept_relations,
            core_slide_texts=core_slide_texts,
            core_trans_texts=core_trans_texts,
        ), "graph_first"
    except Exception as exc:
        print(f"[경고] graph-first summary 생성 실패 — 기존 텍스트 요약으로 fallback: {exc}")

    try:
        return generate_summary(core_slide_texts, core_trans_texts, concept_degrees), "legacy_text_fallback"
    except Exception as exc:
        print(f"[경고] legacy summary fallback 실패 — 그래프 핵심 개념 기반 안전 요약 사용: {exc}")
        return _fallback_summary_from_graph(concept_roles), "graph_core_safe_fallback"



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
    artifacts = load_graph_artifacts_for_metadata(stem, output_dir, fused, emphasized)
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

    print(f"[{stem}] graph-first keywords/roles/relations 생성 중")

    legacy_keywords, *_ = score_keywords(
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
    print(f"[{stem}] graph-first summary 생성 중")
    summary, summary_source = generate_graph_first_summary_with_fallback(
        artifacts=artifacts,
        keyword_candidates=keyword_candidates,
        concept_roles=concept_roles,
        concept_relations=concept_relations,
        core_slide_texts=core_slide_texts,
        core_trans_texts=core_trans_texts,
        concept_degrees=concept_degrees,
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
            "keyword_aliases": alias_map,
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
            "summary_source": summary_source,
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
        keyword_aliases = graph_first_parts.get("keyword_aliases") or {}
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
        keyword_aliases = {}
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
        concept_roles = classify_concept_roles(
            role_candidates,
            norm_slide_freq,
            norm_emph,
            norm_cent,
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
        "keyword_aliases":     keyword_aliases,
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
