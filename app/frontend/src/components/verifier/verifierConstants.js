export const PIPELINE_STAGES = {
  stage1a_extract: { key: 'stage1a_extract', label: '슬라이드 프레임 추출' },
  stage1b_audio_analyze: { key: 'stage1b_audio_analyze', label: '오디오 품질 분석' },
  stage2a_textualize: { key: 'stage2a_textualize', label: '슬라이드 내 정보 추출' },
  stage2b_transcribe: { key: 'stage2b_transcribe', label: '강의 음성 텍스트 전사' },
  stage3a_annotation: { key: 'stage3a_annotation', label: '화면 필기 강조 분석' },
  stage3b_audio: { key: 'stage3b_audio', label: '음성 전사본 정제 및 강조 분석' },
  stage9_build_analyzer_merged_clean: { key: 'stage9_build_analyzer_merged_clean', label: '강의 검증 입력 데이터 구성' },
  stage10_run_analyzers: { key: 'stage10_run_analyzers', label: '강의 내용 검증 실행' },
  verify_claim_extraction: { key: 'verify_claim_extraction', label: '주장 후보 추출' },
  verify_issue_judge: { key: 'verify_issue_judge', label: '이슈 후보 판단' },
  verify_issue_classification: { key: 'verify_issue_classification', label: '이슈 유형 분류' },
  verify_final_report: { key: 'verify_final_report', label: '최종 평가' },
  verify_slide_errors: { key: 'verify_slide_errors', label: '슬라이드 오류 검사' },
  stage4a_classify: { key: 'stage4a_classify', label: '슬라이드 유형 분류' },
  stage4b_save_by_scene: { key: 'stage4b_save_by_scene', label: '장면별 데이터 정리' },
  stage5_fusion: { key: 'stage5_fusion', label: '전체 데이터 통합' },
  stage6_graph_triples: { key: 'stage6_graph_triples', label: '그래프 데이터 생성' },
  stage7a_lance_index: { key: 'stage7a_lance_index', label: '벡터 검색 인덱스 생성' },
  stage7b_graphrag_index: { key: 'stage7b_graphrag_index', label: 'GraphRAG 인덱스 생성' },
  stage8_generate_metadata: { key: 'stage8_generate_metadata', label: '강의 메타데이터 생성' },
  stage11_build_recommender_index: { key: 'stage11_build_recommender_index', label: '강의 추천 인덱스 생성' },
}

function createNode({ id, label, type = 'minor', stageKeys = [] }) {
  return {
    id,
    label,
    type,
    stages: stageKeys.map(key => PIPELINE_STAGES[key]).filter(Boolean),
  }
}

export const PIPELINE_NODES = {
  upload: createNode({ id: 'upload', label: '업로드', type: 'major' }),
  data_extract: createNode({
    id: 'data_extract',
    label: '데이터 추출',
    stageKeys: [
      'stage1a_extract',
      'stage1b_audio_analyze',
    ],
  }),
  content_extract: createNode({
    id: 'content_extract',
    label: '컨텐츠 추출',
    stageKeys: [
      'stage2a_textualize',
      'stage2b_transcribe',
    ],
  }),
  context_analysis: createNode({
    id: 'context_analysis',
    label: '맥락 분석',
    stageKeys: [
      'stage3a_annotation',
      'stage3b_audio',
    ],
  }),
  verifier_context: createNode({
    id: 'verifier_context',
    label: '맥락 분석',
    stageKeys: ['stage3a_annotation'],
  }),
  verifier_data: createNode({
    id: 'verifier_data',
    label: '검증 데이터 구성',
    stageKeys: ['stage9_build_analyzer_merged_clean'],
  }),
  claim_extraction: createNode({
    id: 'claim_extraction',
    label: '주장 추출',
    stageKeys: ['verify_claim_extraction'],
  }),
  issue_judge: createNode({
    id: 'issue_judge',
    label: '이슈 후보 판단',
    stageKeys: ['verify_issue_judge'],
  }),
  issue_classification: createNode({
    id: 'issue_classification',
    label: '이슈 유형 분류',
    stageKeys: ['verify_issue_classification'],
  }),
  final_verification: createNode({
    id: 'final_verification',
    label: '최종 평가',
    stageKeys: ['verify_final_report'],
  }),
  slide_review: createNode({
    id: 'slide_review',
    label: '슬라이드 오류',
    stageKeys: ['verify_slide_errors'],
  }),
  verified: createNode({ id: 'verified', label: '검증 결과 확인', type: 'major' }),
  structure: createNode({
    id: 'structure',
    label: '강의 구조 파악',
    stageKeys: [
      'stage4a_classify',
      'stage4b_save_by_scene',
    ],
  }),
  fusion: createNode({
    id: 'fusion',
    label: '데이터 통합',
    stageKeys: ['stage5_fusion'],
  }),
  graph_build: createNode({
    id: 'graph_build',
    label: '그래프 생성',
    stageKeys: ['stage6_graph_triples'],
  }),
  search_index: createNode({
    id: 'search_index',
    label: '검색 인덱스 생성',
    stageKeys: [
      'stage7a_lance_index',
      'stage7b_graphrag_index',
    ],
  }),
  metadata: createNode({
    id: 'metadata',
    label: '메타데이터 생성',
    stageKeys: [
      'stage8_generate_metadata',
      'stage11_build_recommender_index',
    ],
  }),
  done: createNode({ id: 'done', label: '완료', type: 'major' }),
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
  'upload',
  'data_extract',
  'content_extract',
  'verifier_context',
  'verifier_data',
  ...VERIFY_DETAIL_PIPELINE_NODE_IDS,
  'verified',
]

const FINALIZE_PIPELINE_NODE_IDS = [
  'structure',
  'fusion',
  'graph_build',
  'search_index',
  'metadata',
  'done',
]

const SKIP_VERIFY_PIPELINE_NODE_IDS = [
  'upload',
  'data_extract',
  'content_extract',
  'context_analysis',
  ...FINALIZE_PIPELINE_NODE_IDS,
]

const FULL_VERIFY_PIPELINE_NODE_IDS = [
  ...VERIFY_PROGRESS_PIPELINE_NODE_IDS,
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
export const PIPELINE_FLOW_NODES = createPipeline(FULL_VERIFY_PIPELINE_NODE_IDS)

export const VERIFY_PROGRESS_NODE_IDS = VERIFY_PROGRESS_PIPELINE_NODE_IDS.filter(id => id !== 'upload')
export const VERIFY_PROGRESS_STAGE_KEYS = VERIFY_PROGRESS_PIPELINE_FLOW_NODES.flatMap(getStageKeys)
export const FINALIZE_STAGE_KEYS = FINALIZE_PIPELINE_FLOW_NODES.flatMap(getStageKeys)
export const UPLOAD_STAGE_KEYS = UPLOAD_PIPELINE_FLOW_NODES.flatMap(getStageKeys)
export const VERIFY_STAGE_KEYS = VERIFY_DETAIL_PIPELINE_NODE_IDS.flatMap(id => getStageKeys(PIPELINE_NODES[id]))
export const STAGE_KEYS = uniqueValues([...VERIFY_PROGRESS_STAGE_KEYS, ...FINALIZE_STAGE_KEYS, ...UPLOAD_STAGE_KEYS])

export const PIPELINE_LOG_STAGES = getLogStages(PIPELINE_FLOW_NODES)

export const PHASES = {
  UPLOAD: 'upload',
  VERIFY_CHOICE: 'verifyChoice',
  PIPELINE1: 'pipeline1',
  VERIFY_READY: 'verifyReady',
  REVIEWED: 'reviewed',
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
