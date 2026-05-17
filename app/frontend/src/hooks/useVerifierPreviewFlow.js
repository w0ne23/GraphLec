import { useEffect, useMemo, useState } from 'react'
import { MOCK_CLAIMS, MOCK_FILE, MOCK_LECTURE } from '../mocks/verifierMock'
import {
  PHASES,
  PIPELINE_LOG_STAGES,
  normalizePipelineStages,
} from '../components/verifier/verifierConstants'

const PREVIEW_PIPELINE1_STAGE_KEYS = [
  'stage1a_extract',
  'stage1b_audio_analyze',
  'stage2a_textualize',
  'stage2b_transcribe',
  'stage3a_annotation',
  'stage3b_audio',
  'stage9_build_analyzer_merged_clean',
  'stage10_run_analyzers',
]

const PREVIEW_PIPELINE2_STAGE_KEYS = [
  'stage4a_classify',
  'stage4b_save_by_scene',
  'stage5_fusion',
  'stage6_graph_triples',
  'stage7a_lance_index',
  'stage7b_graphrag_index',
  'stage8_generate_metadata',
  'stage11_build_recommender_index',
]

function getStageGroupKey(stageKey) {
  return stageKey.match(/^stage(\d+)[ab]_/i)?.[1] ?? stageKey
}

function createStageGroups(stageKeys) {
  const groups = []
  const groupByKey = new Map()

  stageKeys.forEach(stageKey => {
    const groupKey = getStageGroupKey(stageKey)
    const existingGroup = groupByKey.get(groupKey)
    if (existingGroup) {
      existingGroup.push(stageKey)
      return
    }

    const group = [stageKey]
    groupByKey.set(groupKey, group)
    groups.push(group)
  })

  return groups
}

const PREVIEW_PIPELINE1_STAGE_GROUPS = createStageGroups(PREVIEW_PIPELINE1_STAGE_KEYS)
const PREVIEW_PIPELINE2_STAGE_GROUPS = createStageGroups(PREVIEW_PIPELINE2_STAGE_KEYS)

function getPreviewStageGroups(phase) {
  if (phase === PHASES.PIPELINE1) return PREVIEW_PIPELINE1_STAGE_GROUPS
  if (phase === PHASES.PIPELINE2) return PREVIEW_PIPELINE2_STAGE_GROUPS
  return []
}

function createPreviewStages(phase, stageGroupIndex = -1) {
  const stageGroups = getPreviewStageGroups(phase)

  if (stageGroupIndex >= 0 && stageGroups[stageGroupIndex]) {
    const doneStageKeys = [
      ...(phase === PHASES.PIPELINE2 ? PREVIEW_PIPELINE1_STAGE_KEYS : []),
      ...stageGroups.slice(0, stageGroupIndex).flat(),
    ]
    const activeStageKeys = stageGroups[stageGroupIndex]

    return normalizePipelineStages([
      ...doneStageKeys.map(stage => ({ stage, status: 'done' })),
      ...activeStageKeys.map(stage => ({ stage, status: 'run' })),
    ])
  }

  if (phase === PHASES.PIPELINE1) {
    return normalizePipelineStages(PREVIEW_PIPELINE1_STAGE_GROUPS[0].map(stage => ({ stage, status: 'run' })))
  }

  if (phase === PHASES.VERIFY_READY || phase === PHASES.REVIEWED) {
    return normalizePipelineStages(PREVIEW_PIPELINE1_STAGE_KEYS.map(stage => ({ stage, status: 'done' })))
  }

  if (phase === PHASES.PIPELINE2) {
    return normalizePipelineStages([
      ...PREVIEW_PIPELINE1_STAGE_KEYS.map(stage => ({ stage, status: 'done' })),
      ...PREVIEW_PIPELINE2_STAGE_GROUPS[0].map(stage => ({ stage, status: 'run' })),
    ])
  }

  if (phase === PHASES.DONE) {
    return normalizePipelineStages(
      [...PREVIEW_PIPELINE1_STAGE_KEYS, ...PREVIEW_PIPELINE2_STAGE_KEYS].map(stage => ({ stage, status: 'done' }))
    )
  }

  return normalizePipelineStages()
}

function getCurrentStageMessage(phase, stageGroupIndex) {
  const selectedGroup = getPreviewStageGroups(phase)[stageGroupIndex] ?? []
  const selectedStage = selectedGroup.length > 0
    ? PIPELINE_LOG_STAGES.find(stage => stage.key === selectedGroup[0])
    : null

  if (selectedStage) return `${selectedStage.groupLabel} 진행 중`
  if (phase === PHASES.PIPELINE1) return '데이터 추출 진행 중'
  if (phase === PHASES.VERIFY_READY) return '검증 결과가 준비되었습니다.'
  if (phase === PHASES.PIPELINE2) return '강의 구조 파악 진행 중'
  return ''
}

export function useVerifierPreviewFlow() {
  const [title, setTitle] = useState(MOCK_LECTURE.title)
  const [file, setFile] = useState(MOCK_FILE)

  const [phase, setPhase] = useState(PHASES.UPLOAD)
  const [stageGroupIndex, setStageGroupIndex] = useState(-1)
  const [errorMessage, setErrorMessage] = useState('')

  const [expandedClaimKey, setExpandedClaimKey] = useState('')
  const [isVideoMode, setIsVideoMode] = useState(false)
  const [seekToSeconds, setSeekToSeconds] = useState(null)

  const pipelineStages = useMemo(
    () => createPreviewStages(phase, stageGroupIndex),
    [phase, stageGroupIndex]
  )
  const claims = phase === PHASES.VERIFY_READY || phase === PHASES.REVIEWED ? MOCK_CLAIMS : []
  const currentStage = getCurrentStageMessage(phase, stageGroupIndex)

  useEffect(() => {
    const stageGroups = getPreviewStageGroups(phase)
    if (stageGroups.length === 0) return

    if (stageGroupIndex < 0 || stageGroupIndex >= stageGroups.length) {
      setStageGroupIndex(0)
      return
    }

    const timer = setTimeout(() => {
      const nextIndex = stageGroupIndex + 1
      if (stageGroups[nextIndex]) {
        setStageGroupIndex(nextIndex)
        return
      }
      setStageGroupIndex(-1)
      setPhase(phase === PHASES.PIPELINE1 ? PHASES.VERIFY_READY : PHASES.DONE)
    }, 2000)

    return () => clearTimeout(timer)
  }, [phase, stageGroupIndex])

  function selectFile(nextFile) {
    if (!nextFile) return
    setFile(nextFile)
    setTitle(prev => prev.trim() ? prev : nextFile.name.replace(/\.[^.]+$/, ''))
  }

  function upload() {
    if (!file) return
    setErrorMessage('')
    setStageGroupIndex(-1)
    setPhase(PHASES.PIPELINE1)
  }

  function reset() {
    setPhase(PHASES.UPLOAD)
    setErrorMessage('')
    setStageGroupIndex(-1)
    setExpandedClaimKey('')
    setIsVideoMode(false)
    setSeekToSeconds(null)
    setFile(MOCK_FILE)
    setTitle(MOCK_LECTURE.title)
  }

  function retry() {
    setErrorMessage('')
    setStageGroupIndex(-1)
    setPhase(PHASES.PIPELINE1)
  }

  return {
    phase,
    lecture: MOCK_LECTURE,
    title,
    file,
    pipelineStages,
    currentStage,
    errorMessage,
    claims,
    expandedClaimKey,
    isVideoMode,
    seekToSeconds,
    actions: {
      setTitle,
      selectFile,
      upload,
      openReview: () => setPhase(PHASES.REVIEWED),
      confirmReview: () => {
        setStageGroupIndex(-1)
        setPhase(PHASES.PIPELINE2)
      },
      retry,
      reset,
      toggleClaim: key => setExpandedClaimKey(prev => prev === key ? '' : key),
      watchClaim: startTime => {
        setIsVideoMode(true)
        setSeekToSeconds(Number(startTime) || 0)
      },
      exitVideo: () => setIsVideoMode(false),
    },
  }
}
