"""
json_to_graph_triples.py — fused.json → 그래프 Parquet

입력: {stem}_fused.json  (fusion.py 출력)
출력: {stem}_graph_triples.parquet, {stem}_nodes.parquet, {stem}_edges.parquet

레이어:
  - 구조 레이어: 결정론적 (Slide, Segment, AnnotationEmphasis 노드 + 관계)
  - 개념 레이어: Gemini 기반 (Concept 노드, MENTIONS/APPEARS_IN/관계 엣지)

사용법:
  python json_to_graph_triples.py --stem os1-1
"""

import os
import re
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from google import genai
from dotenv import load_dotenv

from .config import GEMINI_GENERATIVE_MODEL
load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
#  설정
# ============================================================================

@dataclass
class Config:
    stem:        str  = "lecture"
    output_dir:  Path = Path("output")
    slides_dir:  Path = Path("output_slides")

    fused_path:  Path = field(default=None)
    output_triples_parquet: Path = field(default=None)

    google_api_key: str = field(default_factory=lambda: os.getenv('GOOGLE_API_KEY_1', ''))
    gemini_model:   str = GEMINI_GENERATIVE_MODEL
    lecture_title:  str = "강의"

    def __post_init__(self):
        try:
            from .config import output_paths
            paths = output_paths(self.stem, self.output_dir, self.slides_dir)
            if self.fused_path is None: self.fused_path = paths["fused"]
        except ImportError:
            if self.fused_path is None:
                self.fused_path = self.output_dir / f"{self.stem}_fused.json"
        if self.output_triples_parquet is None:
            self.output_triples_parquet = self.output_dir / f"{self.stem}_graph_triples.parquet"


RELATION_TYPES = {
    "is_a", "part_of", "instance_of", "has_attribute",
    "prerequisite_of", "causes", "influences",
    "uses", "applies",
    "compared_to", "illustrates",
    "abstracts",
    "solves", "optimizes",
    "implements", "replaces",
}

ENTITY_TYPES = {
    "concept", "agent", "system", "artifact",
    "method", "event", "phenomenon", "metric",
    "location", "time_period"
}

# 상위 도메인 (프롬프트·후처리와 동일 집합 유지)
DOMAIN_TYPES = frozenset({
    "engineering",
    "natural_science",
    "humanities",
    "social_science",
    "arts",
    "health_sciences",
    "sports",
    "education",
    "etc",
})
DOMAIN_FALLBACK = "etc"


def _normalize_domain_token(raw: str) -> str:
    s = raw.strip().lower().replace("-", "_")
    return s


def resolve_domain_from_api(domain_result: Optional[Dict]) -> Tuple[str, str]:
    """
    Gemini 도메인 JSON → (domain, subdomain).
    API 실패·누락·허용 목록 밖 값은 DOMAIN_FALLBACK(etc), engineering 등 임의 기본값은 쓰지 않음.
    """
    if not domain_result or not isinstance(domain_result, dict):
        return DOMAIN_FALLBACK, ""

    sub = domain_result.get("subdomain", "")
    subdomain = sub.strip() if isinstance(sub, str) else ""

    dom = domain_result.get("domain")
    if not dom or not isinstance(dom, str):
        return DOMAIN_FALLBACK, subdomain

    token = _normalize_domain_token(dom)
    if token in DOMAIN_TYPES:
        return token, subdomain
    return DOMAIN_FALLBACK, subdomain


DOMAIN_PROMPT = """
아래 강의 내용을 보고 도메인을 분류해라. 설명 없이 JSON만 출력.

[본문]
{content}

출력 형식:
{{
  "domain": "engineering|natural_science|humanities|social_science|arts|health_sciences|sports|education|etc",
  "subdomain": "세부 분야를 짧은 영어 스네이크케이스로 (예: computer_science, cardiology). 애매하면 빈 문자열 \"\""
}}

도메인 기준 (위 목록 중 정확히 하나만 선택):
- engineering: 설계·시스템·구현·최적화·공학적 문제 해결 중심
- natural_science: 자연 현상·법칙·실험·측정·이론 모델링 중심
- humanities: 해석·사상·텍스트·역사·철학·언어·문화 비평 중심
- social_science: 사회 구조·제도·정책·행위자·지표·조사·경제·법 중심
- arts: 작품·표현·매체·양식·창작·미학·공연·시각예술 중심
- health_sciences: 의학·간호·보건·역학·임상·신체·질병·치료·예방 중심
- sports: 체육·운동생리·경기·훈련법·스포츠 과학·신체 활동 교육 중심
- education: 교수학습·교육과정·평가·교육 심리·교수법 일반(특정 교과 내용이 주가 아닐 때)
- etc: 위 어디에도 단일하게 속하기 어렵거나, 학제간·소개·행정 안내 위주 등 판단이 애매할 때

반드시 domain 값은 위 파이프(|)로 나열한 토큰 중 하나와 정확히 일치해야 한다.
"""

EXTRACTION_PROMPT = """
아래는 강의 전체 내용이다. 도메인: {domain} / {subdomain}
설명 없이 JSON만 출력.

각 텍스트 단위에는 [source_id] 태그가 붙어 있다.
- [slide_XXX]      : 슬라이드 제목/텍스트
- [segment/NNNN]   : 오디오 전사 세그먼트

[본문]
{content}

출력 형식:
{{
  "entities": [
    {{"name": "엔티티명", "type": "엔티티타입"}},
    ...
  ],
  "relations": [
    {{
      "from": "출발 엔티티명",
      "to": "도착 엔티티명",
      "type": "관계 타입"
    }}
  ],
  "mentions": {{
    "엔티티명": ["segment/0001", "slide_002"],
    ...
  }}
}}

엔티티 타입 (10가지만 사용):
- concept   : 추상 개념, 이론, 원리, 분류
- agent     : 행위 주체, 사람, 집단, 조직, 기관
- system    : 구조적으로 결합된 체계
- artifact  : 인공물, 문서, 도구, 소프트웨어
- method    : 절차, 알고리즘, 기법, 실험법
- event     : 사건, 실행, 변화, 실험, 시행
- phenomenon: 현상, 문제, 효과, 상태
- metric    : 변수, 지표, 측정값, 성능값
- location  : 장소, 영역, 공간, 구간
- time_period: 시점, 기간, 시대, 단계

엔티티 추출 규칙:
- 강의 전체에서 등장하는 모든 핵심 엔티티 추출, 동일 엔티티 통합
- 정의·용어·기술적 메커니즘·문제·현상·구체적 식별자·행위자·도구 모두 포함
- 슬라이드 레이블(코드 번호, 그림 번호), 교육 메타 표현("다음 슬라이드", "예제") 제외
- 수량 제한 없음: 강의에 등장하는 모든 의미있는 엔티티 추출
- 자기 자신과의 관계 제외
- 상위 개념의 구성 요소·기능·하위 항목은 반드시 개별 엔티티로 분리해서 추출
  예) "마의 효능: 위벽 보호, 소화 촉진" → 기능(X), 위벽 보호(O), 소화 촉진(O) 각각 별도 엔티티
- 목록·열거 형태로 나오는 항목들은 전부 개별 엔티티로 추출

관계 타입 (아래 16가지만 사용; 다른 문자열 금지):
is_a, part_of, instance_of, has_attribute,
prerequisite_of, causes, influences,
uses, applies,
compared_to, illustrates,
abstracts,
solves, optimizes,
implements, replaces

각 관계 의미 (방향: from → to):
- is_a: from은 to의 한 종류·범주·유형이다. (from=하위, to=상위)
- part_of: from은 to의 구성 요소·부분·하위 단위다. (from=부분, to=전체)
- instance_of: from은 to의 구체적 사례·실례·표본이다. (from=사례, to=범주)
- has_attribute: from은 속성·특성·조건으로 to를 갖는다 (정의·성질·전제).
- prerequisite_of: from을 이해·다루기 전에 to가 필요하다 (선행 지식).
- causes: from이 to를 일으키거나 강한 인과로 이끈다 (메커니즘·직접 원인).
- influences: from이 to에 영향을 준다 (causes보다 약하거나 다방향·맥락적 영향).
- uses: from이 to를 수단·도구·방법·자료로 쓴다.
- applies: from(이론·규칙·방법)이 to(상황·대상·문제)에 적용된다.
- compared_to: from과 to가 대조·비교된다.
- illustrates: from(사례·예)가 to(개념·주장)를 설명·뒷받침한다. (단순 분류가 아닐 때; 분류면 instance_of/is_a)
- abstracts: from(상위·일반 개념)이 to(하위·구체 세부)를 포괄·일반화한다. (from=상위·일반, to=하위·구체)
- solves: from이 to(문제·과제)를 해결한다.
- optimizes: from이 to(목표·지표·과정)를 개선·최적화한다.
- implements: from이 to(명세·아이디어·요구)를 실현·구현한다.
- replaces: from이 to를 대체한다.

관계 작성 규칙:
- 위 16개 외 타입 금지. 애매하면 엣지를 생략한다.
- causes vs influences: 직접·강한 인과면 causes, 약하거나 상호·맥락적이면 influences.
- uses vs applies: 도구·자료 활용은 uses; 이론·규칙의 적용은 applies.
- illustrates 남용 금지: 명백한 예시·사례 설명일 때만.

mentions 작성 규칙:
- 각 엔티티가 직접 언급·설명·예시로 다뤄지는 모든 source_id 나열
- 정확한 단어 일치 불필요, 관련 설명·예시도 포함
- segment와 slide 양쪽 모두 포함 가능
- 모든 entity는 최소 1개 이상의 source_id 필수
"""


# ============================================================================
#  트리플 수집기
# ============================================================================

class TripleCollector:
    def __init__(self):
        self.triples: List[Tuple] = []
        self._edge_set: set = set()

    def add(self, subject: str, predicate: str, obj: str, properties: dict = None):
        key = (subject, predicate, obj)
        if key in self._edge_set:
            return
        self._edge_set.add(key)
        props_str = json.dumps(properties, ensure_ascii=False) if properties else ''
        self.triples.append((subject, predicate, obj, props_str))

    def postprocess_abstracts(self):
        """
        abstracts 엣지 방향 후처리.

        abstracts 정의: from(상위·일반) → to(하위·구체)
        Gemini가 가끔 역전시킴 → concept 노드 out-degree 기반으로 교정.

        교정 기준:
          - out-degree가 높을수록 다른 노드와 많이 연결된 상위 개념
          - abstracts 엣지에서 src_degree < tgt_degree 이면 from이 더 구체적 → swap
          - degree가 같으면 판단 불가 → 그대로 유지
        """
        # concept 노드의 out-degree 계산
        degree: Dict[str, int] = {}
        for subj, pred, obj, _ in self.triples:
            if subj.startswith('concept/'):
                degree[subj] = degree.get(subj, 0) + 1

        corrected: List[Tuple] = []
        swap_count = 0

        for subj, pred, obj, props in self.triples:
            if (pred == 'abstracts'
                    and subj.startswith('concept/')
                    and obj.startswith('concept/')):
                src_deg = degree.get(subj, 0)
                tgt_deg = degree.get(obj, 0)
                if src_deg < tgt_deg:
                    # from이 더 구체적 → swap
                    corrected.append((obj, pred, subj, props))
                    swap_count += 1
                    logger.info(
                        f"  [abstracts 교정] {subj} → {obj} 역전 "
                        f"(degree: {src_deg} < {tgt_deg})"
                    )
                    continue
            corrected.append((subj, pred, obj, props))

        self.triples = corrected
        self._edge_set = {(s, p, o) for s, p, o, _ in self.triples}
        logger.info(f"✓ abstracts 방향 교정 완료: {swap_count}개 역전")

    def write_parquet_outputs(self, stem: str, output_dir: Path) -> dict:
        from .graph_parquet_export import write_graph_parquet_bundle

        return write_graph_parquet_bundle(self.triples, stem, output_dir)


# ============================================================================
#  전처리: fused.json 인덱싱
# ============================================================================

class Preprocessor:
    """
    fused.json → 구조 레이어·개념 레이어에서 사용할 인덱스 생성

    인덱스:
      - slide_map:    slide_id → slide dict
      - unique_slides: logical slide_id별 대표 slide dict
      - segment_data: segment_id → segment dict (slide_id 포함)
      - annot_data:   annot_id  → annotation_summary dict (slide_id 포함)
    """

    def __init__(self, slides: List[Dict]):
        self.slides = slides
        self.slide_map:    Dict[str, Dict] = {}
        self.unique_slides: Dict[str, Dict] = {}
        self.segment_data: Dict[str, Dict] = {}  # segment/NNNN → dict
        self.annot_data:   Dict[str, Dict] = {}  # annotation/NNNN → dict
        self.scene_data:   Dict[str, Dict] = {}  # scene/0000 → dict
        self.context_data: Dict[str, Dict] = {}  # scene_id/context/NN → dict
        self._build()

    @staticmethod
    def _content_score(slide: Dict) -> int:
        return len(str(slide.get('slide_text') or '')) + len(str(slide.get('title') or ''))

    def _build(self):
        seg_idx = 0
        ann_idx = 0

        for slide in self.slides:
            sid = slide['slide_id']
            self.slide_map[sid] = slide
            current = self.unique_slides.get(sid)
            if current is None or self._content_score(slide) > self._content_score(current):
                self.unique_slides[sid] = slide

            scene_number = slide.get('scene_number', slide.get('scene_index'))
            if isinstance(scene_number, int):
                scene_id = slide.get('scene_id') or f"scene/{scene_number:04d}"
            else:
                scene_id = slide.get('scene_id') or f"scene/{len(self.scene_data):04d}"
            self.scene_data[scene_id] = {
                'slide_id':       sid,
                'scene_number':   scene_number,
                'start':          slide.get('start_sec'),
                'end':            slide.get('end_sec'),
                'slide_number':   slide.get('slide_number'),
                'role':           slide.get('role'),
                'emphasis_total': (slide.get('emphasis_score') or {}).get('total', 0.0),
            }

            for ctx in slide.get('contexts', []):
                context_id = f"{scene_id}/context/{ctx['context_index']:02d}"
                for seg in ctx.get('segments', []):
                    seg_id = seg.get('segment_id') or f'segment/{seg_idx:04d}'
                    self.segment_data[seg_id] = {
                        **seg,
                        'slide_id':      sid,
                        'scene_id':      scene_id,
                        'context_id':    context_id,
                        'context_index': ctx['context_index'],
                    }
                    seg_idx += 1

            for ctx in slide.get('contexts', []):
                context_id = f"{scene_id}/context/{ctx['context_index']:02d}"
                self.context_data[context_id] = {
                    'slide_id':      sid,
                    'scene_id':      scene_id,
                    'context_index': ctx['context_index'],
                    'start':         ctx.get('start'),
                    'end':           ctx.get('end'),
                    'stressed':      ctx.get('stressed', False),
                    'text':          ctx.get('text', ''),
                }

            for ann in slide.get('annotations_summary', []):
                ann_id = f'annotation/{ann_idx:04d}'
                # scene_id 보존 — _build_annotations에서 재계산 없이 직접 참조
                self.annot_data[ann_id] = {**ann, 'slide_id': sid, 'scene_id': scene_id}
                ann_idx += 1

        logger.info(f"✓ 전처리 완료: slide {len(self.slides)}개, "
                    f"segment {seg_idx}개, annotation {ann_idx}개")


# ============================================================================
#  개념 정규화
# ============================================================================

class ConceptNormalizer:

    def __init__(self):
        pass

    def normalize(self, concept: str) -> str:
        concept = concept.strip()
        if '(' in concept:
            concept = concept.split('(')[0].strip()
        return concept

    @staticmethod
    def to_slug(name: str) -> str:
        return name.strip().lower()

    @staticmethod
    def to_display_name(name: str) -> str:
        return name.strip().upper()


# ============================================================================
#  구조 레이어 빌더
# ============================================================================

class StructureLayerBuilder:

    def __init__(self, pre: Preprocessor, collector: TripleCollector, config: Config):
        self.pre = pre
        self.c   = collector
        self.cfg = config
        self.vid = f'lecture_video/{config.stem}'

    def build(self):
        self._build_root()
        self._build_slides()
        self._build_scenes()
        self._build_segments()
        self._build_annotations()
        logger.info("✓ 구조 레이어 완료")

    def _build_root(self):
        self.c.add(self.vid, 'type', 'Video', {'title': self.cfg.lecture_title, 'stem': self.cfg.stem})

    def _build_slides(self):
        for slide in self.pre.unique_slides.values():
            sid = slide['slide_id']
            self.c.add(self.vid, 'HAS_SLIDE', sid)
            self.c.add(sid, 'type', 'Slide', {
                'slide_number': slide.get('slide_number'),
                'title':        slide.get('title', ''),
                'slide_text':   slide.get('slide_text', ''),
            })

    def _build_scenes(self):
        """Scene 노드 생성 (슬라이드 등장 구간, 1 per scene occurrence)."""
        for scene_id, data in self.pre.scene_data.items():
            sid = data['slide_id']
            self.c.add(self.vid, 'HAS_SCENE', scene_id)
            self.c.add(scene_id, 'USES_SLIDE', sid)
            self.c.add(scene_id, 'type', 'Scene', {
                'source_slide_id': sid,
                'scene_number':    data.get('scene_number'),
                'slide_number':    data.get('slide_number'),
                'start_sec':       data['start'],
                'end_sec':         data['end'],
                'role':            data.get('role'),
                'emphasis_total':  data.get('emphasis_total', 0.0),
            })
        """Context 노드 생성 (발화 문맥 묶음, Scene 내부)"""
        for context_id, data in self.pre.context_data.items():
            self.c.add(data['scene_id'], 'HAS_CONTEXT', context_id)
            self.c.add(context_id, 'type', 'Context', {
                'slide_id':      data['slide_id'],
                'scene_id':      data['scene_id'],
                'context_index': data['context_index'],
                'start':         data['start'],
                'end':           data['end'],
                'stressed':      data['stressed'],
                'text':          data['text'],
            })

    def _build_segments(self):
        """segment → Segment 노드 + Context에 HAS_SEGMENT 연결"""
        for seg_id, data in self.pre.segment_data.items():
            self.c.add(data['context_id'], 'HAS_SEGMENT', seg_id)
            self.c.add(seg_id, 'type', 'Segment', {
                'start':    data['start'],
                'end':      data['end'],
                'text':     data['text'],
                'stressed': data.get('stressed', False),
            })

    def _build_annotations(self):
        """annotation_summary → AnnotationEmphasis 노드 + Scene에 HAS_ANNOTATION 연결
        주석은 특정 영상 장면에서 발생한 시점 이벤트이므로 Scene에 귀속."""
        for ann_id, data in self.pre.annot_data.items():
            self.c.add(data['scene_id'], 'HAS_ANNOTATION', ann_id)
            props = {
                'type':           data.get('type'),
                'target_content': data.get('target_content'),
                'score':          data.get('score', 0.0),
                'confidence':     data.get('confidence'),
                'timestamp_sec':  data.get('timestamp_sec'),
            }
            if data.get('bbox'):
                props['bbox'] = data['bbox']
            if data.get('handwritten_content'):
                props['handwritten_content'] = data['handwritten_content']
            self.c.add(ann_id, 'type', 'AnnotationEmphasis', props)

# ============================================================================
#  개념 레이어 빌더 (Gemini)
# ============================================================================

class ConceptLayerBuilder:

    def __init__(self, pre: Preprocessor, collector: TripleCollector, config: Config):
        self.pre        = pre
        self.c          = collector
        self.cfg        = config
        self.normalizer = ConceptNormalizer()
        self.client     = genai.Client(api_key=config.google_api_key)

    def build(self):
        content = self._build_full_content()
        if not content.strip():
            logger.warning("전체 콘텐츠가 비어있음")
            return

        # ── 0. 도메인 감지 ───────────────────────────────────────────────────
        # 토큰 절약: 앞 3000자만 사용
        domain_result = self._call_gemini(DOMAIN_PROMPT.format(content=content[:3000]))
        domain, subdomain = resolve_domain_from_api(domain_result)
        logger.info(f"  도메인: {domain} / {subdomain}")

        # 도메인 노드 저장
        self.c.add('lecture_video', 'HAS_DOMAIN', f'domain/{domain}')
        self.c.add(f'domain/{domain}', 'type', 'Domain', {'name': domain, 'subdomain': subdomain})

        # ── 1. 엔티티 추출 ───────────────────────────────────────────────────
        logger.info("엔티티 레이어 추출 중...")
        result = self._call_gemini(EXTRACTION_PROMPT.format(
            domain=domain, subdomain=subdomain, content=content
        ))
        if not result:
            return

        entities_raw: List[Dict]          = result.get('entities', [])
        relations_raw: List[Dict]         = result.get('relations', [])
        mentions_raw: Dict[str, List[str]] = result.get('mentions', {})

        logger.info(f"  Gemini 반환: entities={len(entities_raw)}, "
                    f"relations={len(relations_raw)}, "
                    f"mentions_entries={len(mentions_raw)}")

        # ── 2. Entity 노드 ───────────────────────────────────────────────────
        registered: Dict[str, str] = {}  # raw_name → slug
        for entry in entities_raw:
            name      = entry.get('name', '') if isinstance(entry, dict) else entry
            ent_type  = entry.get('type', 'concept') if isinstance(entry, dict) else 'concept'
            if ent_type not in ENTITY_TYPES:
                ent_type = 'concept'
            norm  = self.normalizer.normalize(name)
            slug  = ConceptNormalizer.to_slug(norm)
            if not slug:
                continue
            registered[name] = slug
            display = ConceptNormalizer.to_display_name(norm)
            self.c.add(f'concept/{slug}', 'type', 'Concept', {
                'name':        display,
                'entity_type': ent_type,
            })

        slug_set = set(registered.values())

        # ── 3. MENTIONS / APPEARS_IN 트리플 ─────────────────────────────────
        for entity_name, source_ids in mentions_raw.items():
            norm = self.normalizer.normalize(entity_name)
            slug = ConceptNormalizer.to_slug(norm)
            if slug not in slug_set:
                continue
            for source_id in source_ids:
                if source_id.startswith('segment/'):
                    seg_data = self.pre.segment_data.get(source_id)
                    if not seg_data:
                        continue
                    self.c.add(source_id, 'MENTIONS', f'concept/{slug}', {
                        'stressed': seg_data.get('stressed', False),
                        'start':    seg_data['start'],
                        'end':      seg_data['end'],
                    })
                elif source_id.startswith('slide_'):
                    if source_id in self.pre.slide_map:
                        self.c.add(source_id, 'APPEARS_IN', f'concept/{slug}')

        # ── 4. entity↔entity 관계 ────────────────────────────────────────────
        for rel in relations_raw:
            from_raw = rel.get('from', '')
            to_raw   = rel.get('to', '')
            rel_type = rel.get('type', '')
            if rel_type not in RELATION_TYPES:
                continue
            from_slug = ConceptNormalizer.to_slug(self.normalizer.normalize(from_raw))
            to_slug   = ConceptNormalizer.to_slug(self.normalizer.normalize(to_raw))
            if not from_slug or not to_slug or from_slug == to_slug:
                continue
            if from_slug not in slug_set or to_slug not in slug_set:
                continue
            self.c.add(f'concept/{from_slug}', rel_type, f'concept/{to_slug}')

        logger.info(f"✓ 엔티티 레이어 완료: entity {len(slug_set)}개")

    def _build_full_content(self) -> str:
        """
        fused.json → [source_id] 태그 포함 전체 텍스트 조합
        슬라이드 제목([slide_XXX]) + 세그먼트([segment/NNNN])
        """
        lines = []
        seg_idx = 0

        for slide in self.pre.slides:
            sid   = slide['slide_id']
            title = slide.get('title', '')
            lines.append(f'\n--- {sid}: {title} ---')
            lines.append(f'[{sid}] {title}')

            # 강조 키워드도 슬라이드 소스로 추가
            for kw_entry in slide.get('emphasized_keywords', []):
                kw = kw_entry.get('keyword', '')
                if kw:
                    lines.append(f'[{sid}] {kw}')

            # 세그먼트 (오디오 전사)
            for ctx in slide.get('contexts', []):
                for seg in ctx.get('segments', []):
                    seg_id = f'segment/{seg_idx:04d}'
                    lines.append(f'[{seg_id}] {seg["text"]}')
                    seg_idx += 1

        return '\n'.join(lines)

    def _call_gemini(self, prompt: str) -> Optional[Dict]:
        try:
            response = self.client.models.generate_content(
                model=self.cfg.gemini_model, contents=prompt
            )
            try:
                from .cost_report import record_model_call

                record_model_call(
                    stage="stage6_graph_triples",
                    provider="google",
                    model=self.cfg.gemini_model,
                    response=response,
                    prompt_chars=len(prompt),
                )
            except Exception:
                pass
            text = response.text
            if '```json' in text:
                text = text.split('```json')[1].split('```')[0]
            elif '```' in text:
                text = text.split('```')[1].split('```')[0]

            # 탭·개행을 제외한 제어 문자 제거 (JSON 파싱 실패 방지)
            text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', ' ', text)

            return json.loads(text.strip())

        except json.JSONDecodeError as e:
            logger.error(f"  ✗ JSON 파싱 실패: {e}")
            try:
                logger.error(
                    f"  ✗ 실패 지점 전후: {response.text[max(0, e.pos-100):e.pos+100]!r}"
                )
            except Exception:
                pass
            return None
        except Exception as e:
            logger.error(f"  ✗ Gemini 호출 실패: {e}")
            return None


# ============================================================================
#  파이프라인
# ============================================================================

class GraphPipeline:

    def __init__(self, config: Config = None):
        self.config = config or Config()

    def run(self):
        start = time.time()
        cfg = self.config

        print('\n' + '='*70)
        print('📐 그래프 트리플 생성 파이프라인')
        print('='*70)
        print(f'  fused      : {cfg.fused_path}')
        print(f'  output     : {cfg.output_triples_parquet} (+ nodes/edges parquet)')

        # ── 데이터 로드 ──────────────────────────────────────────────────────
        print('\n[Step 1] 데이터 로드')
        with open(cfg.fused_path, encoding='utf-8') as f:
            fused = json.load(f)
        slides = fused.get('scenes') or fused['slides']
        logical_slide_count = len({
            slide.get('slide_number')
            for slide in slides
            if slide.get('slide_number') is not None
        })
        logger.info(f"✓ scene {len(slides)}개 / slide {logical_slide_count}개")

        # ── 전처리 ───────────────────────────────────────────────────────────
        print('\n[Step 2] 전처리')
        pre = Preprocessor(slides)
        collector = TripleCollector()

        # ── 구조 레이어 ───────────────────────────────────────────────────────
        print('\n[Step 3] 구조 레이어 생성')
        struct_builder = StructureLayerBuilder(pre, collector, cfg)
        struct_builder.build()
        struct_count = len(collector.triples)
        logger.info(f"✓ 구조 트리플: {struct_count}개")

        # ── 개념 레이어 ───────────────────────────────────────────────────────
        print('\n[Step 4] 개념 레이어 생성 (Gemini)')
        concept_builder = ConceptLayerBuilder(pre, collector, cfg)
        concept_builder.build()
        concept_count = len(collector.triples) - struct_count
        logger.info(f"✓ 개념 트리플: {concept_count}개")

        # ── 후처리: abstracts 방향 교정 ──────────────────────────────────────
        print('\n[Step 4.5] abstracts 방향 후처리')
        collector.postprocess_abstracts()

        # ── Parquet 출력 (triples + nodes + edges) ───────────────────────────
        print('\n[Step 5] Parquet 출력')
        pq_paths = collector.write_parquet_outputs(cfg.stem, cfg.output_dir)

        elapsed = time.time() - start
        print('\n' + '='*70)
        print('✅ 완료')
        print('='*70)
        print(f'  구조 트리플 : {struct_count}개')
        print(f'  개념 트리플 : {concept_count}개')
        print(f'  전체 트리플 : {len(collector.triples)}개')
        print(f'  처리 시간   : {elapsed:.2f}초')
        print(f'  출력 파일   :')
        print(f'    triples : {pq_paths["triples"]}')
        print(f'    nodes   : {pq_paths["nodes"]}')
        print(f'    edges   : {pq_paths["edges"]}')


# ============================================================================
#  메인
# ============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description='멀티모달 강의 그래프 트리플 생성')
    parser.add_argument('--stem',       required=True,              help='강의 파일 stem (예: os1-1)')
    parser.add_argument('--output_dir', default='output',           help='출력 디렉토리 (기본: output)')
    parser.add_argument('--slides_dir', default='output_slides',    help='슬라이드 디렉토리')
    parser.add_argument('--title',      default='강의',              help='강의 제목')
    parser.add_argument('--model',      default=GEMINI_GENERATIVE_MODEL)
    args = parser.parse_args()

    cfg = Config(
        stem          = args.stem,
        output_dir    = Path(args.output_dir),
        slides_dir    = Path(args.slides_dir),
        lecture_title = args.title,
        gemini_model  = args.model,
    )

    if not cfg.fused_path.exists():
        print(f'❌ fused 파일 없음: {cfg.fused_path}')
        return
    if not cfg.google_api_key:
        print('❌ GOOGLE_API_KEY 환경변수가 설정되지 않았습니다')
        return

    GraphPipeline(cfg).run()


if __name__ == '__main__':
    main()
