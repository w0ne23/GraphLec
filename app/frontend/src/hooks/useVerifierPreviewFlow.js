import { useEffect, useMemo, useState } from 'react'
import {
  PHASES,
  PIPELINE_LOG_STAGES,
  normalizePipelineStages,
} from '../components/verifier/verifierConstants'

const DEFAULT_PREVIEW_ID = 'd5e73475-c1be-42ba-a526-4fd6eb420aac'
const DEFAULT_PREVIEW_VIDEO_TITLE = '운영체제 강의 영상'
const DEFAULT_PREVIEW_VIDEO_URL = `/files/inputs/${DEFAULT_PREVIEW_ID}/${DEFAULT_PREVIEW_ID}.mp4`
const DEFAULT_PREVIEW_RESULT_URL = `/files/results/${DEFAULT_PREVIEW_ID}/${DEFAULT_PREVIEW_ID}_verification.json`
const DEFAULT_PREVIEW_SLIDE_DATA_URL = `/files/results/${DEFAULT_PREVIEW_ID}/${DEFAULT_PREVIEW_ID}_slide_classified.json`

const EMPTY_VERIFIER_PREVIEW = {
  lecture: {
    id: DEFAULT_PREVIEW_ID,
    title: DEFAULT_PREVIEW_VIDEO_TITLE,
    video_url: DEFAULT_PREVIEW_VIDEO_URL,
  },
  file: null,
  verifier: null,
}

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
  return stageKey.match(/^stage(\d+)(?:[ab])?_/i)?.[1] ?? stageKey
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

function createPreviewResult(result, scenes = []) {
  return {
    lecture: {
      id: DEFAULT_PREVIEW_ID,
      title: DEFAULT_PREVIEW_VIDEO_TITLE,
      video_url: DEFAULT_PREVIEW_VIDEO_URL,
      scenes,
    },
    file: {
      name: DEFAULT_PREVIEW_VIDEO_TITLE,
      size: null,
    },
    verifier: result,
  }
}

function toPreviewFileUrl(value) {
  const raw = String(value ?? '').trim()
  if (!raw) return ''
  if (/^(https?:|data:|blob:)/.test(raw)) return raw

  const normalized = raw.replace(/\\/g, '/')
  const resultMatch = normalized.match(/(?:local_storage|\/?files)\/results\/[^/]+\/(.+)$/)
  if (resultMatch) return `/files/results/${DEFAULT_PREVIEW_ID}/${resultMatch[1]}`
  if (normalized.startsWith('/files/')) return normalized

  const marker = 'local_storage/'
  const markerIndex = normalized.indexOf(marker)
  return markerIndex >= 0 ? `/files/${normalized.slice(markerIndex + marker.length)}` : raw
}

function timestampFromSlide(slide) {
  const value = Number(slide?.timestamp ?? slide?.scene_start_sec ?? slide?.start_time)
  return Number.isFinite(value) ? value : 0
}

function createPreviewScenesFromSlideData(data) {
  const scenes = Array.isArray(data?.scenes) ? data.scenes : []
  return scenes.map(slide => ({
    timestamp: timestampFromSlide(slide),
    end_time: Number(slide.scene_end_sec ?? slide.end_time) || undefined,
    type: slide.role === 'elaborated' ? 'emphasis' : 'slide',
    text: slide.title || `Slide ${slide.slide_number}`,
    image_url: toPreviewFileUrl(slide.image_path),
    scene_number: slide.scene_number ?? slide.scene_index,
    slide_number: slide.slide_number,
  }))
}

async function loadPreviewScenes() {
  const response = await fetch(DEFAULT_PREVIEW_SLIDE_DATA_URL).catch(() => null)
  if (!response?.ok) return []
  const slideData = await response.json()
  return createPreviewScenesFromSlideData(slideData)
}

export function useVerifierPreviewFlow() {
  const [preview, setPreview] = useState(EMPTY_VERIFIER_PREVIEW)
  const [title, setTitle] = useState(EMPTY_VERIFIER_PREVIEW.lecture.title)
  const [file, setFile] = useState(EMPTY_VERIFIER_PREVIEW.file)

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
  const currentStage = getCurrentStageMessage(phase, stageGroupIndex)

  useEffect(() => {
    let active = true

    async function loadVerificationResult() {
      try {
        const response = await fetch(DEFAULT_PREVIEW_RESULT_URL)
        if (!response.ok) throw new Error('verification result load failed')
        const result = await response.json()
        if (!active) return

        const nextPreview = createPreviewResult(result)
        setPreview(nextPreview)
        setFile(nextPreview.file)
        setTitle(nextPreview.lecture.title)

        const scenes = await loadPreviewScenes()
        if (!active || scenes.length === 0) return
        setPreview(prev => ({
          ...prev,
          lecture: {
            ...prev.lecture,
            scenes,
          },
        }))
      } catch (error) {
        if (active) setErrorMessage('verification 결과를 불러오지 못했습니다.')
      }
    }

    loadVerificationResult()

    return () => {
      active = false
    }
  }, [])

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
    setFile(preview.file)
    setTitle(preview.lecture.title)
  }

  function retry() {
    setErrorMessage('')
    setStageGroupIndex(-1)
    setPhase(PHASES.PIPELINE1)
  }

  function backToVerifyReady() {
    setExpandedClaimKey('')
    setIsVideoMode(false)
    setSeekToSeconds(null)
    setPhase(PHASES.VERIFY_READY)
  }

  return {
    phase,
    lecture: preview.lecture,
    verifier: preview.verifier,
    title,
    file,
    pipelineStages,
    currentStage,
    errorMessage,
    expandedClaimKey,
    isVideoMode,
    seekToSeconds,
    actions: {
      setTitle,
      selectFile,
      upload,
      openReview: () => setPhase(PHASES.REVIEWED),
      backToVerifyReady,
      confirmReview: () => {
        setStageGroupIndex(-1)
        setPhase(PHASES.PIPELINE2)
      },
      retry,
      reset,
      toggleClaim: key => setExpandedClaimKey(prev => prev === key ? '' : key),
      watchClaim: startTime => {
        setIsVideoMode(true)
        setSeekToSeconds({ seconds: Number(startTime) || 0, time: Date.now() })
      },
      exitVideo: () => setIsVideoMode(false),
    },
  }
}
