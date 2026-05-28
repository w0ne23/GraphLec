export const VERIFY_PIPELINE_FLOW_NODES = [
  { id: 'verify_start', label: '검증 시작', type: 'major', weight: 2 },
  {
    id: 'preprocess_extract',
    label: '데이터 추출',
    type: 'minor',
    weight: 16,
    stages: [
      { key: 'preprocess_extract_media', label: '슬라이드 추출 및 오디오 품질 분석' },
    ],
  },
  {
    id: 'preprocess_text',
    label: '텍스트화',
    type: 'minor',
    weight: 14,
    stages: [
      { key: 'preprocess_textualize_transcribe', label: '슬라이드 텍스트화 및 전체 전사' },
    ],
  },
  {
    id: 'preprocess_enrichment',
    label: '강의 보강 분석',
    type: 'minor',
    weight: 12,
    stages: [
      { key: 'preprocess_enrich_audio_annotation', label: '필기 강조 및 오디오 후처리' },
    ],
  },
  {
    id: 'preprocess_structure',
    label: '강의 구조 파악',
    type: 'minor',
    weight: 14,
    stages: [
      { key: 'preprocess_classify_scene', label: '슬라이드 분류 및 장면별 데이터 정리' },
    ],
  },
  {
    id: 'preprocess_fusion',
    label: '데이터 통합',
    type: 'minor',
    weight: 10,
    stages: [
      { key: 'preprocess_fusion', label: '전체 데이터 통합' },
    ],
  },
  {
    id: 'analyzer_input',
    label: '검증 입력 생성',
    type: 'minor',
    weight: 10,
    stages: [
      { key: 'verifier_build_analyzer_input', label: 'analyzer 입력 생성' },
    ],
  },
  {
    id: 'verifier_run',
    label: '검증 실행',
    type: 'minor',
    weight: 18,
    stages: [
      { key: 'verifier_run', label: '검증 보고서 생성' },
    ],
  },
  { id: 'verified', label: '검증 결과 확인', type: 'major', weight: 4 },
]

export const PIPELINE_FLOW_NODES = VERIFY_PIPELINE_FLOW_NODES

export const UPLOAD_PIPELINE_FLOW_NODES = [
  { id: 'upload', label: '업로드', type: 'major', weight: 2 },
  {
    id: 'extract',
    label: '데이터 추출',
    type: 'minor',
    weight: 16,
    stages: [
      { key: 'preprocess_extract_media', label: '슬라이드 추출 및 오디오 품질 분석' },
    ],
  },
  {
    id: 'text',
    label: '텍스트화',
    type: 'minor',
    weight: 14,
    stages: [
      { key: 'preprocess_textualize_transcribe', label: '슬라이드 텍스트화 및 전체 전사' },
    ],
  },
  {
    id: 'enrichment',
    label: '강의 보강 분석',
    type: 'minor',
    weight: 12,
    stages: [
      { key: 'preprocess_enrich_audio_annotation', label: '필기 강조 및 오디오 후처리' },
    ],
  },
  {
    id: 'structure',
    label: '강의 구조 파악',
    type: 'minor',
    weight: 12,
    stages: [
      { key: 'preprocess_classify_scene', label: '슬라이드 분류 및 장면별 데이터 정리' },
    ],
  },
  {
    id: 'fusion',
    label: '데이터 통합',
    type: 'minor',
    weight: 10,
    stages: [
      { key: 'preprocess_fusion', label: '전체 데이터 통합' },
    ],
  },
  {
    id: 'graph_build',
    label: '그래프 생성',
    type: 'minor',
    weight: 6,
    stages: [
      { key: 'graph_triples', label: '그래프 데이터 생성' },
    ],
  },
  {
    id: 'lance_index',
    label: 'Lance 인덱스',
    type: 'minor',
    weight: 6,
    stages: [
      { key: 'graph_lance_index', label: '벡터 검색 인덱스 생성' },
    ],
  },
  {
    id: 'graphrag_index',
    label: 'GraphRAG 인덱스',
    type: 'minor',
    weight: 6,
    stages: [
      { key: 'graph_graphrag_index', label: 'GraphRAG 인덱스 생성' },
    ],
  },
  {
    id: 'metadata',
    label: '메타데이터 생성',
    type: 'minor',
    weight: 6,
    stages: [
      { key: 'graph_metadata', label: '강의 메타데이터 생성' },
    ],
  },
  {
    id: 'recommender_index',
    label: '추천 인덱스',
    type: 'minor',
    weight: 6,
    stages: [
      { key: 'graph_recommender_index', label: '강의 추천 인덱스 생성' },
    ],
  },
  { id: 'done', label: '완료', type: 'major', weight: 2 },
]

export const VERIFIER_PREPROCESS_FLOW_NODES = [
  ...UPLOAD_PIPELINE_FLOW_NODES.filter(node => (
    node.id === 'upload' ||
    ['extract', 'text', 'enrichment', 'structure', 'fusion'].includes(node.id)
  )),
  { id: 'verify_start', label: '검증하기', type: 'major', weight: 2 },
]

export const FINALIZE_PIPELINE_FLOW_NODES = [
  { id: 'verify_start', label: '검증하기', type: 'major', weight: 2 },
  ...UPLOAD_PIPELINE_FLOW_NODES.filter(node => (
    ['graph_build', 'lance_index', 'graphrag_index', 'metadata', 'recommender_index', 'done'].includes(node.id)
  )),
]

function getStageKeys(node) {
  return node.stages?.map(stage => stage.key) ?? []
}

function getLogStages(flowNodes) {
  return flowNodes.flatMap(node => {
    return node.stages?.map(stage => ({
      ...stage,
      groupId: node.id,
      groupLabel: node.label,
    })) ?? []
  })
}

function uniqueValues(values) {
  return Array.from(new Set(values))
}

export const VERIFY_STAGE_KEYS = VERIFY_PIPELINE_FLOW_NODES.flatMap(getStageKeys)
export const UPLOAD_STAGE_KEYS = UPLOAD_PIPELINE_FLOW_NODES.flatMap(getStageKeys)
export const STAGE_KEYS = uniqueValues([...VERIFY_STAGE_KEYS, ...UPLOAD_STAGE_KEYS])

export const VERIFY_PIPELINE_LOG_STAGES = getLogStages(VERIFY_PIPELINE_FLOW_NODES)
export const UPLOAD_PIPELINE_LOG_STAGES = getLogStages(UPLOAD_PIPELINE_FLOW_NODES)
export const PIPELINE_LOG_STAGES = [
  ...VERIFY_PIPELINE_LOG_STAGES,
  ...UPLOAD_PIPELINE_LOG_STAGES,
]

export const PHASES = {
  UPLOAD: 'upload',
  VERIFY_CHOICE: 'verifyChoice',
  PIPELINE1: 'pipeline1',
  VERIFY_READY: 'verifyReady',
  REVIEWED: 'reviewed',
  UPLOAD_RESUME: 'uploadResume',
  PIPELINE2: 'pipeline2',
  DONE: 'done',
  ERROR: 'error',
}

export function createEmptyStages() {
  return STAGE_KEYS.map(stage => ({ stage, status: 'wait' }))
}

export function normalizePipelineStages(stages = []) {
  if (!Array.isArray(stages) || stages.length === 0) return createEmptyStages()

  const byStage = new Map(createEmptyStages().map(item => [item.stage, item.status]))

  stages.forEach(item => {
    if (!item?.stage) return
    if (byStage.has(item.stage)) byStage.set(item.stage, item.status)
  })

  return Array.from(byStage, ([stage, status]) => ({ stage, status }))
}
