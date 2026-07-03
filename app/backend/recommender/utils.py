"""
recommender/utils.py
────────────────────
순수 텍스트/수학 유틸리티. 로컬 모듈 의존성 없음.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Iterable

import numpy as np


# ── 텍스트 처리 상수 ──────────────────────────────────────────────────────────

_TERM_RE = re.compile(r"[0-9A-Za-z가-힣_#+./-]+")
_TRAILING_PARTICLES = (
    "으로부터", "로부터", "에서", "에게", "한테", "으로", "까지", "부터",
    "처럼", "보다", "마다", "에서", "으로", "로", "와", "과", "의",
    "을", "를", "이", "가", "은", "는", "에", "도", "만",
)
_QUERY_META_PHRASES = (
    "추천해줘", "추천해 주세요", "추천해주세요", "추천해", "추천",
    "알려줘", "알려 주세요", "알려주세요", "알려", "찾아줘", "찾아 주세요",
    "찾아주세요", "찾아", "보여줘", "보여 주세요", "보여주세요", "보여",
    "설명해줘", "설명해 주세요", "설명해주세요", "설명", "강의", "관련",
    "대해서", "대해", "관한", "다루는", "배우는", "학습", "수업",
)
_DOMAIN_ALIASES = {
    "컴퓨터공학": "engineering",
    "컴공": "engineering",
    "computer science": "engineering",
    "cs": "engineering",
    "공학": "engineering",
    "경제학": "social_science",
    "경제": "social_science",
    "경영": "social_science",
    "비즈니스": "social_science",
    "마케팅": "social_science",
    "수학": "natural_science",
    "자연과학": "natural_science",
    "의학": "health_sciences",
    "의료": "health_sciences",
    "보건": "health_sciences",
    "생물학": "natural_science",
    "생명과학": "natural_science",
    "화학": "natural_science",
    "물리학": "natural_science",
    "물리": "natural_science",
    "환경": "natural_science",
    "기후": "natural_science",
    "철학": "humanities",
    "역사": "humanities",
    "세계사": "humanities",
    "한국사": "humanities",
    "언어학": "humanities",
    "교육": "education",
    "교육학": "education",
    "사회학": "social_science",
    "사회": "social_science",
    "디자인": "arts",
    "예술": "arts",
    "체육": "sports",
    "스포츠": "sports",
}
_SUBDOMAIN_ALIASES: dict[str, tuple[str, str]] = {
    "컴퓨터공학": ("engineering", "computer_science"),
    "컴공": ("engineering", "computer_science"),
    "computer science": ("engineering", "computer_science"),
    "cs": ("engineering", "computer_science"),
    "전기공학": ("engineering", "electrical_engineering"),
    "전자공학": ("engineering", "electrical_engineering"),
    "기계공학": ("engineering", "mechanical_engineering"),
    "화학공학": ("engineering", "chemical_engineering"),
    "산업공학": ("engineering", "industrial_engineering"),
    "경제학": ("social_science", "economics"),
    "경제": ("social_science", "economics"),
    "경영학": ("social_science", "business"),
    "경영": ("social_science", "business"),
    "비즈니스": ("social_science", "business"),
    "법학": ("social_science", "law"),
    "정치학": ("social_science", "political_science"),
    "사회학": ("social_science", "sociology"),
    "심리학": ("social_science", "psychology"),
    "교육학": ("education", "education"),
    "물리학": ("natural_science", "physics"),
    "물리": ("natural_science", "physics"),
    "화학": ("natural_science", "chemistry"),
    "수학": ("natural_science", "mathematics"),
    "mathematics": ("natural_science", "mathematics"),
    "math": ("natural_science", "mathematics"),
    "통계학": ("natural_science", "statistics"),
    "통계": ("natural_science", "statistics"),
    "statistics": ("natural_science", "statistics"),
    "생물학": ("natural_science", "biology"),
    "생명과학": ("natural_science", "biology"),
    "천문학": ("natural_science", "astronomy"),
    "생태학": ("natural_science", "ecology"),
    "철학": ("humanities", "philosophy"),
    "역사학": ("humanities", "history"),
    "역사": ("humanities", "history"),
    "언어학": ("humanities", "linguistics"),
    "문학": ("humanities", "literature"),
    "종교학": ("humanities", "religion"),
    "미술": ("arts", "fine_arts"),
    "음악": ("arts", "music"),
    "디자인": ("arts", "design"),
    "영화": ("arts", "film"),
    "연극": ("arts", "theater"),
    "해부학": ("health_sciences", "anatomy"),
    "생리학": ("health_sciences", "physiology"),
    "약리학": ("health_sciences", "pharmacology"),
    "공중보건": ("health_sciences", "public_health"),
    "간호학": ("health_sciences", "nursing"),
    "체육": ("sports", "physical_education"),
    "스포츠과학": ("sports", "sports_science"),
}
_DOMAIN_LABELS = {
    "engineering": "공학",
    "natural_science": "자연과학",
    "humanities": "인문학",
    "social_science": "사회과학",
    "arts": "예술",
    "health_sciences": "보건의료",
    "sports": "스포츠",
    "education": "교육",
    "etc": "기타",
}

_TOPIC_EXPANSIONS: dict[str, list[str]] = {
    "파이썬": ["Python"],
    "python": ["파이썬"],
    "웹": ["리액트", "컴포넌트", "자바스크립트", "비동기", "Promise", "async/await", "fetch", "JSX", "가상 DOM"],
    "웹서비스": ["리액트", "컴포넌트", "자바스크립트", "비동기", "Promise", "async/await", "fetch", "JSX", "가상 DOM"],
    "프론트엔드": ["리액트", "컴포넌트", "자바스크립트", "JSX", "가상 DOM", "렌더링"],
    "데이터베이스": ["SQL", "JOIN", "트랜잭션", "인덱스", "정규화", "스키마"],
    "DB": ["데이터베이스", "SQL", "JOIN", "트랜잭션", "인덱스", "정규화"],
}


# ── 도메인 유틸 ───────────────────────────────────────────────────────────────

def _canonical_domain(value: str) -> str:
    token = str(value or "").strip().lower().replace("-", "_")
    if token in _DOMAIN_LABELS:
        return token
    top = token.split("/", 1)[0]
    return {
        "eng": "engineering",
        "sci": "natural_science",
        "math": "natural_science",
        "hum": "humanities",
        "soc": "social_science",
        "med": "health_sciences",
        "art": "arts",
        "gen": "etc",
    }.get(top, "etc")


def _normalize_subdomain(value: str) -> str:
    return str(value or "").strip().lower().replace("-", "_")


def _infer_domain_filters(
    query: str,
    available_domains: list[str],
    available_subdomains: list[str],
) -> tuple[str | None, str | None]:
    normalized = str(query or "").lower()
    compact = re.sub(r"\s+", "", normalized)
    for alias, (domain, subdomain) in _SUBDOMAIN_ALIASES.items():
        alias_normalized = alias.lower()
        alias_compact = re.sub(r"\s+", "", alias_normalized)
        if (
            (alias_normalized in normalized or alias_compact in compact)
            and domain in available_domains
            and subdomain in available_subdomains
        ):
            return domain, subdomain
    for alias, domain in _DOMAIN_ALIASES.items():
        alias_normalized = alias.lower()
        alias_compact = re.sub(r"\s+", "", alias_normalized)
        if (alias_normalized in normalized or alias_compact in compact) and domain in available_domains:
            return domain, None
    return None, None


# ── 텍스트 정규화 ─────────────────────────────────────────────────────────────

def _normalize_term(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def _compact_term(text: str) -> str:
    return re.sub(r"\s+", "", _normalize_term(text))


def _strip_query_meta_phrases(text: str) -> str:
    cleaned = _normalize_term(text)
    for phrase in sorted(_QUERY_META_PHRASES, key=len, reverse=True):
        cleaned = re.sub(rf"(?<![0-9A-Za-z가-힣]){re.escape(phrase)}(?![0-9A-Za-z가-힣])", " ", cleaned)
        cleaned = cleaned.replace(phrase, " ")
    return re.sub(r"\s+", " ", cleaned).strip()


def _strip_trailing_particles(text: str) -> str:
    cleaned = _normalize_term(text)
    changed = True
    while changed and len(cleaned) > 2:
        changed = False
        for particle in sorted(_TRAILING_PARTICLES, key=len, reverse=True):
            if cleaned.endswith(particle) and len(cleaned) - len(particle) >= 2:
                cleaned = cleaned[:-len(particle)].strip()
                changed = True
                break
    return cleaned


def _query_term_base(text: str) -> str:
    cleaned = _strip_query_meta_phrases(text)
    cleaned = _strip_trailing_particles(cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def _term_lookup_keys(text: str) -> set[str]:
    keys: set[str] = set()
    normalized = _normalize_term(text)
    base = _query_term_base(normalized)
    for value in (normalized, base):
        value = _normalize_term(value)
        if not value:
            continue
        keys.add(value)
        compact = _compact_term(value)
        if compact:
            keys.add(compact)
    return {key for key in keys if len(key) > 1}


def _tokenize_text(text: str) -> list[str]:
    normalized = _normalize_term(text)
    return [token for token in _TERM_RE.findall(normalized) if len(token) > 1]


def _concept_match(concept: str, focus: str) -> bool:
    concept_norm = _normalize_term(concept)
    focus_norm = _normalize_term(focus)
    if not concept_norm or not focus_norm:
        return False
    if concept_norm == focus_norm:
        return True
    if _compact_term(concept_norm) == _compact_term(focus_norm):
        return True
    if _term_lookup_keys(concept_norm) & _term_lookup_keys(focus_norm):
        return True
    return focus_norm in concept_norm or concept_norm in focus_norm


def _append_terms(target: list[str], values) -> None:
    if values is None:
        return
    if isinstance(values, str):
        if values.strip():
            target.append(values)
        return
    if isinstance(values, dict):
        for value in values.values():
            _append_terms(target, value)
        return
    if isinstance(values, (list, tuple, set)):
        for value in values:
            _append_terms(target, value)
        return
    if isinstance(values, (int, float)):
        target.append(str(values))


def _expanded_topic_terms(terms: Iterable[str]) -> list[str]:
    expanded: list[str] = []
    seen: set[str] = set()

    def add(term: str) -> None:
        normalized = _normalize_term(term)
        if not normalized or normalized in seen:
            return
        seen.add(normalized)
        expanded.append(term)

    for term in terms:
        add(term)
        normalized = _normalize_term(term)
        for alias, aliases in _TOPIC_EXPANSIONS.items():
            normalized_alias = _normalize_term(alias)
            if normalized == normalized_alias or normalized_alias in normalized or normalized in normalized_alias:
                for expanded_term in aliases:
                    add(expanded_term)

    return expanded


def _expanded_lookup_terms(terms: Iterable[str]) -> set[str]:
    expanded: set[str] = set()
    for term in _expanded_topic_terms(terms):
        normalized = _normalize_term(term)
        if not normalized:
            continue
        expanded.add(normalized)
        expanded.update(_term_lookup_keys(normalized))
    return expanded


def _append_topic_expansions(
    query_keywords: list[str],
    inferred_keywords: list[str],
    seed_terms: list[str] | None = None,
) -> list[str]:
    existing = {_normalize_term(term) for term in query_keywords + inferred_keywords}
    seed_only = {
        _normalize_term(term)
        for term in (seed_terms or [])
        if _normalize_term(term)
    }
    expanded = list(inferred_keywords)
    for term in _expanded_topic_terms(query_keywords + inferred_keywords + (seed_terms or [])):
        normalized = _normalize_term(term)
        if normalized in seed_only:
            continue
        if normalized and normalized not in existing:
            existing.add(normalized)
            expanded.append(term)
    return expanded


def _community_report_terms(text: str) -> Counter:
    terms: Counter = Counter()
    normalized = _normalize_term(text)
    if not normalized:
        return terms
    for token in _tokenize_text(normalized):
        terms[token] += 1.0
        for key in _term_lookup_keys(token):
            if key != token:
                terms[key] += 0.75
    return terms


def _add_lexical_variants(counter: Counter, text: str, weight: float) -> None:
    phrase = _normalize_term(text)
    if not phrase:
        return
    counter[phrase] += weight
    for key in _term_lookup_keys(phrase):
        if key != phrase:
            counter[key] += weight * 0.8


# ── 수학 유틸 ─────────────────────────────────────────────────────────────────

def _cosine_sim(a: list[float], b: list[float]) -> float:
    va, vb = np.array(a), np.array(b)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    if denom == 0:
        return 0.0
    return float(np.dot(va, vb) / denom)


def _normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return matrix
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def _normalize_vector(vector: list[float]) -> np.ndarray:
    arr = np.array(vector, dtype=np.float32)
    norm = np.linalg.norm(arr)
    if norm == 0:
        return arr
    return arr / norm


def _soft_threshold_similarity(value, threshold: float, floor: float):
    """
    floor 미만은 0, threshold 이상은 원래 값을 유지하고,
    사이 구간은 점진적 감쇠. numpy array와 scalar 모두 지원.
    """
    safe_threshold = max(float(threshold), 0.0)
    safe_floor = min(max(float(floor), 0.0), safe_threshold)
    if safe_threshold <= safe_floor:
        return np.where(value >= safe_threshold, value, 0.0)
    ratio = (value - safe_floor) / (safe_threshold - safe_floor)
    ratio = np.clip(ratio, 0.0, 1.0)
    return value * ratio


def _database_url_sync() -> str:
    import os
    url = os.getenv("DATABASE_URL", "")
    return url.replace("+asyncpg", "") if url else ""
