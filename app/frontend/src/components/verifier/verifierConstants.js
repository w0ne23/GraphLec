export const VERIFY_PIPELINE_FLOW_NODES = [
  { id: 'verify_start', label: '검증 시작', type: 'major', weight: 2 },
  {
    id: 'claim_extraction',
    label: '주장 추출',
    type: 'minor',
    weight: 20,
    stages: [
      { key: 'verify_claim_extraction', label: '주장 후보 추출' },
    ],
  },
  {
    id: 'issue_judge',
    label: '이슈 후보 판단',
    type: 'minor',
    weight: 22,
    stages: [
      { key: 'verify_issue_judge', label: '이슈 후보 판단' },
    ],
  },
  {
    id: 'issue_classification',
    label: '이슈 유형 분류',
    type: 'minor',
    weight: 20,
    stages: [
      { key: 'verify_issue_classification', label: '이슈 유형 분류' },
    ],
  },
  {
    id: 'final_verification',
    label: '최종 평가',
    type: 'minor',
    weight: 22,
    stages: [
      { key: 'verify_final_report', label: '최종 평가' },
    ],
  },
  {
    id: 'slide_review',
    label: '슬라이드 오류',
    type: 'minor',
    weight: 14,
    stages: [
      { key: 'verify_slide_errors', label: '슬라이드 오류 검사' },
    ],
  },
  { id: 'verified', label: '검증 결과 확인', type: 'major', weight: 4 },
]

export const PIPELINE_FLOW_NODES = VERIFY_PIPELINE_FLOW_NODES

export const UPLOAD_PIPELINE_FLOW_NODES = [
  { id: 'upload', label: '업로드', type: 'major', weight: 2 },
  {
    id: 'structure',
    label: '강의 구조 파악',
    type: 'minor',
    weight: 12,
    stages: [
      { key: 'stage4a_classify', label: '슬라이드 유형 분류' },
      { key: 'stage4b_save_by_scene', label: '장면별 데이터 정리' },
    ],
  },
  {
    id: 'fusion',
    label: '데이터 통합',
    type: 'minor',
    weight: 10,
    stages: [
      { key: 'stage5_fusion', label: '전체 데이터 통합' },
    ],
  },
  {
    id: 'graph_build',
    label: '그래프 생성',
    type: 'minor',
    weight: 6,
    stages: [
      { key: 'stage6_graph_triples', label: '그래프 데이터 생성' },
    ],
  },
  {
    id: 'search_index',
    label: '검색 인덱스 생성',
    type: 'minor',
    weight: 8,
    stages: [
      { key: 'stage7a_lance_index', label: '벡터 검색 인덱스 생성' },
      { key: 'stage7b_graphrag_index', label: 'GraphRAG 인덱스 생성' },
    ],
  },
  {
    id: 'metadata',
    label: '메타데이터 생성',
    type: 'minor',
    weight: 6,
    stages: [
      { key: 'stage8_generate_metadata', label: '강의 메타데이터 생성' },
      { key: 'stage11_build_recommender_index', label: '강의 추천 인덱스 생성' },
    ],
  },
  { id: 'done', label: '완료', type: 'major', weight: 2 },
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
