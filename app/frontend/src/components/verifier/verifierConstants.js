export const PIPELINE_FLOW_NODES = [
  { id: 'upload', label: '업로드', type: 'major', weight: 2 },
  {
    id: 'extract',
    label: '데이터 추출',
    type: 'minor',
    weight: 24,
    stages: [
      { key: 'stage1a_extract', label: '슬라이드 프레임 추출' },
      { key: 'stage1b_audio_analyze', label: '오디오 품질 분석' },
      { key: 'stage2a_textualize', label: '슬라이드 내 정보 추출' },
      { key: 'stage2b_transcribe', label: '강의 음성 텍스트 전사' },
    ],
  },
  {
    id: 'content_verify',
    label: '강의 내용 검증',
    type: 'minor',
    weight: 30,
    stages: [
      { key: 'stage3a_annotation', label: '필기 강조 분석' },
      { key: 'stage3b_audio', label: '음성 강조 분석' },
      { key: 'stage9_build_analyzer_merged_clean', label: '검증 입력 데이터 구성' },
      { key: 'stage10_run_analyzers', label: '강의 내용 검증 실행' },
    ],
  },
  { id: 'verified', label: '검증 결과 확인', type: 'major', weight: 4 },
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

export const STAGE_KEYS = PIPELINE_FLOW_NODES.flatMap(node => {
  return node.stages?.map(stage => stage.key) ?? []
})

export const PIPELINE_LOG_STAGES = PIPELINE_FLOW_NODES.flatMap(node => {
  return node.stages?.map(stage => ({
    ...stage,
    groupId: node.id,
    groupLabel: node.label,
  })) ?? []
})

export const PHASES = {
  UPLOAD: 'upload',
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

export function formatAnalysisTime(seconds) {
  const safe = Number.isFinite(seconds) ? Math.max(0, seconds) : 0
  const m = Math.floor(safe / 60)
  const s = Math.floor(safe % 60)
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`
}
