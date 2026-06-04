function createNode({ id, label, stageKey = id, stageLabel = label }) {
  return {
    id,
    label,
    stages: [{ key: stageKey, label: stageLabel }],
  }
}

export const PIPELINE_NODES = {
  data_extract: createNode({
    id: 'data_extract',
    label: '데이터 추출',
    stageKey: 'preprocess_extract_media',
    stageLabel: '슬라이드 추출 및 오디오 품질 분석',
  }),
  content_extract: createNode({
    id: 'content_extract',
    label: '텍스트화',
    stageKey: 'preprocess_textualize_transcribe',
    stageLabel: '슬라이드 텍스트화 및 전체 전사',
  }),
  context_analysis: createNode({
    id: 'context_analysis',
    label: '강의 보강 분석',
    stageKey: 'preprocess_enrich_audio_annotation',
    stageLabel: '필기 강조 및 오디오 후처리',
  }),
  verifier_data: createNode({
    id: 'verifier_data',
    label: '검증 데이터 구성',
    stageKey: 'verifier_build_analyzer_input',
    stageLabel: '검증 입력 데이터 구성',
  }),
  verifier_run: createNode({
    id: 'verifier_run',
    label: '검증 실행',
    stageLabel: '강의 내용 검증 실행',
  }),
  claim_extraction: createNode({
    id: 'claim_extraction',
    label: '주장 추출',
    stageKey: 'verifier_claim_extraction',
    stageLabel: '주장 후보 추출',
  }),
  issue_judge: createNode({
    id: 'issue_judge',
    label: '이슈 후보 판단',
    stageKey: 'verifier_issue_judge',
  }),
  issue_classification: createNode({
    id: 'issue_classification',
    label: '이슈 유형 분류',
    stageKey: 'verifier_issue_classification',
  }),
  final_verification: createNode({
    id: 'final_verification',
    label: '멀티 LLM 검증',
    stageKey: 'verifier_final_verification',
  }),
  slide_review: createNode({
    id: 'slide_review',
    label: '슬라이드 오류',
    stageKey: 'verify_slide_errors',
    stageLabel: '슬라이드 오류 검사',
  }),
  structure: createNode({
    id: 'structure',
    label: '강의 구조 파악',
    stageKey: 'graph_classify_scene',
    stageLabel: '슬라이드 분류 및 장면별 데이터 정리',
  }),
  fusion: createNode({
    id: 'fusion',
    label: '데이터 통합',
    stageKey: 'graph_fusion',
    stageLabel: '전체 데이터 통합',
  }),
  graph_build: createNode({
    id: 'graph_build',
    label: '그래프 생성',
    stageKey: 'graph_triples',
    stageLabel: '그래프 데이터 생성',
  }),
  graph_lance_index: createNode({
    id: 'graph_lance_index',
    label: '벡터 검색 인덱스 생성',
  }),
  graph_graphrag_index: createNode({
    id: 'graph_graphrag_index',
    label: 'GraphRAG 인덱스 생성',
  }),
  graph_metadata: createNode({
    id: 'graph_metadata',
    label: '강의 메타데이터 생성',
  }),
  graph_recommender_index: createNode({
    id: 'graph_recommender_index',
    label: '강의 추천 인덱스 생성',
  }),
}

export const VERIFY_DETAIL_PIPELINE_NODE_IDS = [
  'claim_extraction',
  'issue_judge',
  'issue_classification',
  'final_verification',
  'slide_review',
]

export const VERIFY_STEPS = VERIFY_DETAIL_PIPELINE_NODE_IDS.map(key => ({
  key,
  label: PIPELINE_NODES[key].label,
}))

const VERIFY_PROGRESS_PIPELINE_NODE_IDS = [
  'data_extract',
  'content_extract',
  'context_analysis',
  'verifier_data',
  ...VERIFY_DETAIL_PIPELINE_NODE_IDS,
]

const FINALIZE_PIPELINE_NODE_IDS = [
  'structure',
  'fusion',
  'graph_build',
  'graph_lance_index',
  'graph_graphrag_index',
  'graph_metadata',
  'graph_recommender_index',
]

const SKIP_VERIFY_PIPELINE_NODE_IDS = [
  'data_extract',
  'content_extract',
  'context_analysis',
  'structure',
  'fusion',
  'graph_build',
  'graph_lance_index',
  'graph_graphrag_index',
  'graph_metadata',
  'graph_recommender_index',
]

function createPipeline(nodeIds) {
  return nodeIds.map(id => PIPELINE_NODES[id]).filter(Boolean)
}

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

export const VERIFY_PROGRESS_PIPELINE_FLOW_NODES = createPipeline(VERIFY_PROGRESS_PIPELINE_NODE_IDS)
export const VERIFY_PIPELINE_FLOW_NODES = VERIFY_PROGRESS_PIPELINE_FLOW_NODES
export const FINALIZE_PIPELINE_FLOW_NODES = createPipeline(FINALIZE_PIPELINE_NODE_IDS)
export const SKIP_VERIFY_PIPELINE_FLOW_NODES = createPipeline(SKIP_VERIFY_PIPELINE_NODE_IDS)
export const UPLOAD_PIPELINE_FLOW_NODES = SKIP_VERIFY_PIPELINE_FLOW_NODES
export const PIPELINE_FLOW_NODES = VERIFY_PROGRESS_PIPELINE_FLOW_NODES

export const VERIFY_PROGRESS_NODE_IDS = VERIFY_PROGRESS_PIPELINE_NODE_IDS
export const VERIFY_PROGRESS_STAGE_KEYS = VERIFY_PROGRESS_PIPELINE_FLOW_NODES.flatMap(getStageKeys)
export const FINALIZE_STAGE_KEYS = FINALIZE_PIPELINE_FLOW_NODES.flatMap(getStageKeys)
export const UPLOAD_STAGE_KEYS = UPLOAD_PIPELINE_FLOW_NODES.flatMap(getStageKeys)
export const VERIFY_STAGE_KEYS = uniqueValues([...VERIFY_PROGRESS_STAGE_KEYS, 'verifier_run'])
export const STAGE_KEYS = uniqueValues([...VERIFY_STAGE_KEYS, ...FINALIZE_STAGE_KEYS, ...UPLOAD_STAGE_KEYS])

export const PIPELINE_LOG_STAGES = getLogStages(PIPELINE_FLOW_NODES)

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
