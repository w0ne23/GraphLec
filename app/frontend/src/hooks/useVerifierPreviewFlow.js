import { useEffect, useMemo, useState } from 'react'
import {
  FINALIZE_STAGE_KEYS,
  PIPELINE_FLOW_NODES,
  PHASES,
  PIPELINE_LOG_STAGES,
  UPLOAD_PIPELINE_FLOW_NODES,
  UPLOAD_STAGE_KEYS,
  VERIFY_PROGRESS_STAGE_KEYS,
  VERIFY_PROGRESS_PIPELINE_FLOW_NODES,
  normalizePipelineStages,
} from '../components/verifier/verifierConstants'

const DEV_FILE_BASE = typeof window !== 'undefined' && window.location?.hostname
  ? `http://${window.location.hostname}:8000`
  : ''
const FILE_BASE = (import.meta.env.VITE_API_BASE_URL || DEV_FILE_BASE || '').replace(/\/$/, '')
const DEFAULT_PREVIEW_ID = 'preview'
const DEFAULT_PREVIEW_VIDEO_TITLE = '운영체제 강의 영상'
const DEFAULT_PREVIEW_ANALYZER_DIR = 'preview_analyzer'

function fileUrl(path) {
  if (path.startsWith('/files/')) return path
  return `${FILE_BASE}${path}`
}

const DEFAULT_PREVIEW_VIDEO_URL = fileUrl(`/files/inputs/${DEFAULT_PREVIEW_ID}/${DEFAULT_PREVIEW_ID}.mp4`)
const DEFAULT_PREVIEW_RESULT_URL = fileUrl(`/files/results/${DEFAULT_PREVIEW_ID}/${DEFAULT_PREVIEW_ANALYZER_DIR}/${DEFAULT_PREVIEW_ID}_verification_final.json`)
const DEFAULT_PREVIEW_SLIDE_ERROR_URL = fileUrl(`/files/results/${DEFAULT_PREVIEW_ID}/${DEFAULT_PREVIEW_ANALYZER_DIR}/${DEFAULT_PREVIEW_ID}_slide_errors.json`)
const DEFAULT_PREVIEW_SLIDE_DATA_URL = fileUrl(`/files/results/${DEFAULT_PREVIEW_ID}/${DEFAULT_PREVIEW_ID}_slide_classified.json`)

function analyzerFileUrl(suffix) {
  return fileUrl(`/files/results/${DEFAULT_PREVIEW_ID}/${DEFAULT_PREVIEW_ANALYZER_DIR}/${DEFAULT_PREVIEW_ID}${suffix}`)
}

const DEFAULT_PREVIEW_ARTIFACT_URLS = {
  mergedClean: analyzerFileUrl('_merged_clean.json'),
  claims: analyzerFileUrl('_claims.json'),
  claimsJsonl: analyzerFileUrl('_claims.jsonl'),
  issueJudge: analyzerFileUrl('_issue_judge.json'),
  issueJudgeSummary: analyzerFileUrl('_issue_judge_summary.json'),
  issueJudgeCompare: analyzerFileUrl('_issue_judge_compare.json'),
  issueTypes: analyzerFileUrl('_issue_types.json'),
  classifiedIssues: analyzerFileUrl('_classified_issues.json'),
  classifiedIssueVerifier: analyzerFileUrl('_classified_issue_verifier.json'),
  slideErrors: DEFAULT_PREVIEW_SLIDE_ERROR_URL,
  slideClassified: DEFAULT_PREVIEW_SLIDE_DATA_URL,
  verification: DEFAULT_PREVIEW_RESULT_URL,
}

const FALLBACK_VERIFIER_RESULT = {
  schema_version: 'content_verification.v2',
  mode: 'classified_issue_verifier',
  models: ['gpt', 'claude', 'grok'],
  summary: {
    total_feedback_count: 2,
    confirmed_feedback_count: 1,
    review_needed_feedback_count: 1,
    rejected_feedback_count: 0,
    breakdown_by_type: {
      factual_error: 1,
      confusing_explanation: 1,
    },
  },
  counts: {
    final_confirmed: 1,
    needs_review: 1,
    rejected: 0,
    slide_errors: 1,
  },
  claims: [
    {
      claim_id: 'CL0001',
      claim_type: 'definition',
      claim_text: '프로세스는 실행 중인 프로그램이다.',
      resolved_claim: '프로세스는 실행 중인 프로그램의 실행 단위다.',
      verification_question: '프로세스 정의가 정확한가?',
      context_id: 'SC001',
      slide_number: 3,
      start_time: 42,
    },
    {
      claim_id: 'CL0002',
      claim_type: 'relationship',
      claim_text: '스레드는 항상 독립된 주소 공간을 가진다.',
      resolved_claim: '스레드는 일반적으로 같은 프로세스의 주소 공간을 공유한다.',
      verification_question: '스레드와 주소 공간 설명이 맞는가?',
      context_id: 'SC002',
      slide_number: 5,
      start_time: 128,
    },
  ],
  feedback_items: [
    {
      feedback_id: 'F0001',
      issue_id: 'I0001',
      source_claim_id: 'CL0002',
      status: 'professor_check',
      feedback_type: 'factual_error',
      feedback_label: '사실 오류',
      claim_text: '스레드는 항상 독립된 주소 공간을 가진다.',
      resolved_claim: '스레드는 일반적으로 같은 프로세스의 주소 공간을 공유한다.',
      location: { slide_number: 5, start_time: 128 },
      severity_score: 0.82,
      severity_score_percent: 82,
      problem: {
        problematic_content: '스레드는 항상 독립된 주소 공간을 가진다.',
        summary: '스레드의 주소 공간 설명 확인 필요',
        issue_basis: '핵심 개념',
        correct_info: '같은 프로세스의 스레드는 주소 공간을 공유한다.',
        recommendation: '프로세스와 스레드의 메모리 관계를 구분해 설명',
      },
      professor_feedback: {
        summary: '주소 공간 공유 여부가 반대로 전달될 수 있음',
        teaching_note: '프로세스 단위 자원과 스레드 실행 단위를 나눠 설명',
      },
      classified_issue_verifier: {
        final_severity_score: 0.82,
        final_severity_percent: 82,
        average_is_valid_issue: 0.9,
        average_category_severity: 0.78,
        average_context_resolution: 0.72,
        model_disagreement: 0.08,
        needs_manual_review: true,
      },
    },
    {
      feedback_id: 'F0002',
      issue_id: 'I0002',
      source_claim_id: 'CL0001',
      status: 'confirmed',
      feedback_type: 'confusing_explanation',
      feedback_label: '혼동 설명',
      claim_text: '프로세스는 실행 중인 프로그램이다.',
      resolved_claim: '프로세스는 실행 중인 프로그램의 실행 단위다.',
      location: { slide_number: 3, start_time: 42 },
      severity_score: 0.48,
      severity_score_percent: 48,
      problem: {
        problematic_content: '프로세스는 실행 중인 프로그램이다.',
        summary: '표현 보강 권장',
        issue_basis: '보충 설명',
        recommendation: '실행 상태와 자원 보유 관점을 함께 설명',
      },
      professor_feedback: {
        summary: '정의 자체는 맞지만 실행 단위 관점을 보강하면 좋음',
      },
      classified_issue_verifier: {
        final_severity_score: 0.48,
        final_severity_percent: 48,
        average_is_valid_issue: 0.62,
        average_category_severity: 0.4,
        average_context_resolution: 0.75,
        model_disagreement: 0.14,
        needs_manual_review: false,
      },
    },
  ],
  slide_errors: [
    {
      slide_error_id: 'S0001',
      slide_number: 5,
      error_type_label: '용어 확인',
      problematic_text: 'Thread has own address space',
      reason: '강의 맥락상 process의 주소 공간 설명과 혼동 가능',
      suggested_fix: 'Thread shares process address space',
    },
  ],
  classified_issue_artifacts: {
    claims_json: 'fallback',
    issue_judge: 'fallback',
    issue_types: 'fallback',
    classified_issue_verifier: 'fallback',
    slide_errors: 'fallback',
  },
}

const EMPTY_VERIFIER_PREVIEW = {
  lecture: {
    id: DEFAULT_PREVIEW_ID,
    title: DEFAULT_PREVIEW_VIDEO_TITLE,
    video_url: DEFAULT_PREVIEW_VIDEO_URL,
  },
  verifier: FALLBACK_VERIFIER_RESULT,
  verifierArtifacts: {},
}

function createStageGroups(flowNodes) {
  return flowNodes
    .map(node => node.stages?.map(stage => stage.key) ?? [])
    .filter(group => group.length > 0)
}

const VERIFY_PROGRESS_STAGE_GROUPS = createStageGroups(VERIFY_PROGRESS_PIPELINE_FLOW_NODES)
const UPLOAD_STAGE_GROUPS = createStageGroups(UPLOAD_PIPELINE_FLOW_NODES)
const VERIFY_TO_UPLOAD_PRIOR_STAGE_KEYS = [
  'preprocess_extract_media',
  'preprocess_textualize_transcribe',
]
const VERIFY_TO_UPLOAD_DONE_STAGE_KEYS = ['preprocess_enrich_audio_annotation']
const VERIFY_TO_UPLOAD_PRIOR_NODE_IDS = [
  'data_extract',
  'content_extract',
]
const PREVIEW_DEFAULT_STAGE_DELAY_MS = 2000
const VERIFY_PREPROCESS_STAGE_DELAY_MS = 3000
const VERIFY_REVIEW_STAGE_DELAY_MS = 6000
const VERIFY_PREPROCESS_STAGE_KEYS = new Set([
  'preprocess_extract_media',
  'preprocess_textualize_transcribe',
  'preprocess_enrich_audio_annotation',
])

function getPreviewStageGroups(phase, verifyEnabled = true) {
  if (phase === PHASES.PIPELINE1) return VERIFY_PROGRESS_STAGE_GROUPS
  if (phase === PHASES.PIPELINE2) return UPLOAD_STAGE_GROUPS
  return []
}

function getPipelinePriorStageKeys(phase, verifyEnabled) {
  return (phase === PHASES.PIPELINE2 || phase === PHASES.DONE) && verifyEnabled
    ? VERIFY_TO_UPLOAD_PRIOR_STAGE_KEYS
    : []
}

function getPreviewInitialStageGroupIndex(phase, verifyEnabled) {
  const stageGroups = getPreviewStageGroups(phase, verifyEnabled)
  const priorStageKeySet = new Set(getPipelinePriorStageKeys(phase, verifyEnabled))

  if (phase === PHASES.PIPELINE2 && verifyEnabled) {
    const nextIndex = stageGroups.findIndex(group => group.some(stage => !priorStageKeySet.has(stage)))
    return nextIndex >= 0 ? nextIndex : 0
  }

  return 0
}

function getPreviewStageDelayMs(phase, stageGroup) {
  if (phase !== PHASES.PIPELINE1) return PREVIEW_DEFAULT_STAGE_DELAY_MS

  const isPreprocessStage = stageGroup.some(stage => VERIFY_PREPROCESS_STAGE_KEYS.has(stage))
  return isPreprocessStage ? VERIFY_PREPROCESS_STAGE_DELAY_MS : VERIFY_REVIEW_STAGE_DELAY_MS
}

function getPipelinePriorNodeIds(phase, verifyEnabled) {
  return (phase === PHASES.PIPELINE2 || phase === PHASES.DONE) && verifyEnabled
    ? VERIFY_TO_UPLOAD_PRIOR_NODE_IDS
    : []
}

function createPreviewStages(phase, stageGroupIndex = -1, verifyEnabled = true) {
  const stageGroups = getPreviewStageGroups(phase, verifyEnabled)
  const priorStageKeys = getPipelinePriorStageKeys(phase, verifyEnabled)
  const priorStageKeySet = new Set(priorStageKeys)
  const doneHandoffStageSet = new Set(
    (phase === PHASES.PIPELINE2 || phase === PHASES.DONE) && verifyEnabled
      ? VERIFY_TO_UPLOAD_DONE_STAGE_KEYS
      : []
  )
  const doneHandoffStageKeys = Array.from(doneHandoffStageSet)

  if (stageGroupIndex >= 0 && stageGroups[stageGroupIndex]) {
    const doneStageKeys = stageGroups
      .slice(0, stageGroupIndex)
      .flat()
      .filter(stage => !priorStageKeySet.has(stage) && !doneHandoffStageSet.has(stage))
    const activeStageKeys = stageGroups[stageGroupIndex]
      .filter(stage => !priorStageKeySet.has(stage) && !doneHandoffStageSet.has(stage))

    return normalizePipelineStages([
      ...priorStageKeys.map(stage => ({ stage, status: 'prior' })),
      ...doneHandoffStageKeys.map(stage => ({ stage, status: 'done' })),
      ...doneStageKeys.map(stage => ({ stage, status: 'done' })),
      ...activeStageKeys.map(stage => ({ stage, status: 'run' })),
    ])
  }

  if (phase === PHASES.PIPELINE1) {
    return normalizePipelineStages(VERIFY_PROGRESS_STAGE_GROUPS[0].map(stage => ({ stage, status: 'run' })))
  }

  if (phase === PHASES.VERIFY_READY || phase === PHASES.REVIEWED) {
    return normalizePipelineStages(VERIFY_PROGRESS_STAGE_KEYS.map(stage => ({ stage, status: 'done' })))
  }

  if (phase === PHASES.PIPELINE2) {
    const initialIndex = getPreviewInitialStageGroupIndex(phase, verifyEnabled)
    const activeStageKeys = stageGroups[initialIndex]
      .filter(stage => !priorStageKeySet.has(stage) && !doneHandoffStageSet.has(stage))
    return normalizePipelineStages([
      ...priorStageKeys.map(stage => ({ stage, status: 'prior' })),
      ...doneHandoffStageKeys.map(stage => ({ stage, status: 'done' })),
      ...activeStageKeys.map(stage => ({ stage, status: 'run' })),
    ])
  }

  if (phase === PHASES.DONE) {
    const completedStageKeys = verifyEnabled
      ? UPLOAD_STAGE_KEYS.filter(stage => !priorStageKeySet.has(stage) && !doneHandoffStageSet.has(stage))
      : UPLOAD_STAGE_KEYS
    return normalizePipelineStages([
      ...priorStageKeys.map(stage => ({ stage, status: 'prior' })),
      ...doneHandoffStageKeys.map(stage => ({ stage, status: 'done' })),
      ...completedStageKeys.map(stage => ({ stage, status: 'done' })),
    ])
  }

  return normalizePipelineStages()
}

function getPipelineFlowNodes(phase, verifyEnabled) {
  if (phase === PHASES.PIPELINE2 || phase === PHASES.DONE) return UPLOAD_PIPELINE_FLOW_NODES
  if (!verifyEnabled) return UPLOAD_PIPELINE_FLOW_NODES
  if (phase === PHASES.PIPELINE1 || phase === PHASES.VERIFY_READY || phase === PHASES.REVIEWED) {
    return VERIFY_PROGRESS_PIPELINE_FLOW_NODES
  }
  return PIPELINE_FLOW_NODES
}

function getPipelineLabel(phase, verifyEnabled) {
  if (phase === PHASES.PIPELINE2 || phase === PHASES.DONE || !verifyEnabled) return '일반 파이프라인'
  return '검증 파이프라인'
}

function findStageInFlowNodes(flowNodes, stageKey) {
  for (const node of flowNodes) {
    const stage = node.stages?.find(item => item.key === stageKey)
    if (stage) return { ...stage, groupId: node.id, groupLabel: node.label }
  }
  return null
}

function getCurrentStageMessage(phase, stageGroupIndex, verifyEnabled = true) {
  const selectedGroup = getPreviewStageGroups(phase, verifyEnabled)[stageGroupIndex] ?? []
  const selectedStage = selectedGroup.length > 0
    ? findStageInFlowNodes(getPipelineFlowNodes(phase, verifyEnabled), selectedGroup[0]) ||
      PIPELINE_LOG_STAGES.find(stage => stage.key === selectedGroup[0])
    : null

  if (selectedStage) return `${selectedStage.groupLabel} 진행 중`
  if (phase === PHASES.PIPELINE1) return '검증 파이프라인 진행 중'
  if (phase === PHASES.VERIFY_READY) return '검증 결과가 준비되었습니다.'
  if (phase === PHASES.PIPELINE2) return '일반 파이프라인 진행 중'
  return ''
}

function getPreviewPipelineView(phase, stageGroupIndex, verifyEnabled = true) {
  return {
    flowNodes: getPipelineFlowNodes(phase, verifyEnabled),
    priorNodeIds: getPipelinePriorNodeIds(phase, verifyEnabled),
    label: getPipelineLabel(phase, verifyEnabled),
    currentStage: getCurrentStageMessage(phase, stageGroupIndex, verifyEnabled),
  }
}

function createPreviewResult(result, scenes = [], verifierArtifacts = {}) {
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
    verifierArtifacts,
  }
}

function asArray(value) {
  return Array.isArray(value) ? value : []
}

function mergeSlideErrorArtifact(result, slideErrorResult) {
  const slideErrors = asArray(slideErrorResult?.slide_errors)
  if (slideErrors.length === 0) return result

  const existingSlideErrors = asArray(result?.slide_errors)
  const mergedSlideErrors = existingSlideErrors.length > 0 ? existingSlideErrors : slideErrors

  return {
    ...result,
    slide_errors: mergedSlideErrors,
    slide_error_summary: result?.slide_error_summary || slideErrorResult?.summary || {},
    slide_error_status: result?.slide_error_status || slideErrorResult?.schema_version || '',
    summary: {
      ...(result?.summary || {}),
      slide_error_count: mergedSlideErrors.length,
    },
    counts: {
      ...(result?.counts || {}),
      slide_errors: mergedSlideErrors.length,
    },
  }
}

async function loadPreviewSlideErrors() {
  const response = await fetch(DEFAULT_PREVIEW_SLIDE_ERROR_URL).catch(() => null)
  if (!response?.ok) return null
  return response.json()
}

async function loadJsonArtifact(url) {
  const response = await fetch(url).catch(() => null)
  if (!response?.ok) return null
  return response.json()
}

async function loadJsonlArtifact(url) {
  const response = await fetch(url).catch(() => null)
  if (!response?.ok) return []
  const text = await response.text()
  return text
    .split(/\r?\n/)
    .map(line => line.trim())
    .filter(Boolean)
    .map(line => {
      try {
        return JSON.parse(line)
      } catch {
        return null
      }
    })
    .filter(Boolean)
}

async function loadPreviewArtifacts() {
  const [
    mergedClean,
    claims,
    claimsJsonl,
    issueJudge,
    issueJudgeSummary,
    issueJudgeCompare,
    issueTypes,
    classifiedIssues,
    classifiedIssueVerifier,
    slideErrors,
    slideClassified,
  ] = await Promise.all([
    loadJsonArtifact(DEFAULT_PREVIEW_ARTIFACT_URLS.mergedClean),
    loadJsonArtifact(DEFAULT_PREVIEW_ARTIFACT_URLS.claims),
    loadJsonlArtifact(DEFAULT_PREVIEW_ARTIFACT_URLS.claimsJsonl),
    loadJsonArtifact(DEFAULT_PREVIEW_ARTIFACT_URLS.issueJudge),
    loadJsonArtifact(DEFAULT_PREVIEW_ARTIFACT_URLS.issueJudgeSummary),
    loadJsonArtifact(DEFAULT_PREVIEW_ARTIFACT_URLS.issueJudgeCompare),
    loadJsonArtifact(DEFAULT_PREVIEW_ARTIFACT_URLS.issueTypes),
    loadJsonArtifact(DEFAULT_PREVIEW_ARTIFACT_URLS.classifiedIssues),
    loadJsonArtifact(DEFAULT_PREVIEW_ARTIFACT_URLS.classifiedIssueVerifier),
    loadJsonArtifact(DEFAULT_PREVIEW_ARTIFACT_URLS.slideErrors),
    loadJsonArtifact(DEFAULT_PREVIEW_ARTIFACT_URLS.slideClassified),
  ])

  return {
    urls: DEFAULT_PREVIEW_ARTIFACT_URLS,
    mergedClean,
    claims,
    claimsJsonl,
    issueJudge,
    issueJudgeSummary,
    issueJudgeCompare,
    issueTypes,
    classifiedIssues,
    classifiedIssueVerifier,
    slideErrors,
    slideClassified,
  }
}

function toPreviewFileUrl(value) {
  const raw = String(value ?? '').trim()
  if (!raw) return ''
  if (/^(https?:|data:|blob:)/.test(raw)) return raw

  const normalized = raw.replace(/\\/g, '/')
  const resultMatch = normalized.match(/(?:local_storage|\/?files)\/results\/[^/]+\/(.+)$/)
  if (resultMatch) return fileUrl(`/files/results/${DEFAULT_PREVIEW_ID}/${resultMatch[1]}`)
  if (normalized.startsWith('/files/')) return fileUrl(normalized)

  const marker = 'local_storage/'
  const markerIndex = normalized.indexOf(marker)
  return markerIndex >= 0 ? fileUrl(`/files/${normalized.slice(markerIndex + marker.length)}`) : raw
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
  const [title, setTitle] = useState('')
  const [file, setFile] = useState(null)

  const [phase, setPhase] = useState(PHASES.VERIFY_CHOICE)
  const [stageGroupIndex, setStageGroupIndex] = useState(-1)
  const [verifyEnabled, setVerifyEnabled] = useState(true)
  const [errorMessage, setErrorMessage] = useState('')

  const [expandedClaimKey, setExpandedClaimKey] = useState('')
  const [isVideoMode, setIsVideoMode] = useState(false)
  const [seekToSeconds, setSeekToSeconds] = useState(null)

  const pipelineStages = useMemo(
    () => createPreviewStages(phase, stageGroupIndex, verifyEnabled),
    [phase, stageGroupIndex, verifyEnabled]
  )
  const pipelineView = useMemo(
    () => getPreviewPipelineView(phase, stageGroupIndex, verifyEnabled),
    [phase, stageGroupIndex, verifyEnabled]
  )

  useEffect(() => {
    let active = true

    async function loadVerificationResult() {
      try {
        const response = await fetch(DEFAULT_PREVIEW_RESULT_URL)
        if (!response.ok) throw new Error('verification result load failed')
        const result = await response.json()
        const artifacts = await loadPreviewArtifacts()
        const verifierResult = mergeSlideErrorArtifact(result, artifacts.slideErrors)
        if (!active) return

        const nextPreview = createPreviewResult(verifierResult, [], artifacts)
        setPreview(nextPreview)

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
        if (!active) return
        const slideErrorResult = await loadPreviewSlideErrors()
        const fallbackResult = mergeSlideErrorArtifact(FALLBACK_VERIFIER_RESULT, slideErrorResult)
        const nextPreview = createPreviewResult(fallbackResult)
        setPreview(nextPreview)
      }
    }

    loadVerificationResult()

    return () => {
      active = false
    }
  }, [])

  useEffect(() => {
    const stageGroups = getPreviewStageGroups(phase, verifyEnabled)
    if (stageGroups.length === 0) return

    if (stageGroupIndex < 0 || stageGroupIndex >= stageGroups.length) {
      setStageGroupIndex(getPreviewInitialStageGroupIndex(phase, verifyEnabled))
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
    }, getPreviewStageDelayMs(phase, stageGroups[stageGroupIndex]))

    return () => clearTimeout(timer)
  }, [phase, stageGroupIndex, verifyEnabled])

  function selectFile(nextFile) {
    if (!nextFile) return
    setFile(nextFile)
    setTitle(prev => prev.trim() ? prev : nextFile.name.replace(/\.[^.]+$/, ''))
  }

  function upload() {
    if (!file) return
    setErrorMessage('')
    setStageGroupIndex(-1)
    setPhase(verifyEnabled ? PHASES.PIPELINE1 : PHASES.PIPELINE2)
  }

  function startVerify() {
    setErrorMessage('')
    setVerifyEnabled(true)
    setStageGroupIndex(-1)
    setPhase(PHASES.UPLOAD)
  }

  function startDirectUpload() {
    setErrorMessage('')
    setVerifyEnabled(false)
    setStageGroupIndex(-1)
    setPhase(PHASES.UPLOAD)
  }

  function backToChoice() {
    setErrorMessage('')
    setStageGroupIndex(-1)
    setPhase(PHASES.VERIFY_CHOICE)
  }

  function reset() {
    setPhase(PHASES.VERIFY_CHOICE)
    setErrorMessage('')
    setVerifyEnabled(true)
    setStageGroupIndex(-1)
    setExpandedClaimKey('')
    setIsVideoMode(false)
    setSeekToSeconds(null)
    setFile(null)
    setTitle('')
  }

  function retry() {
    setErrorMessage('')
    setStageGroupIndex(-1)
    setPhase(verifyEnabled ? PHASES.PIPELINE1 : PHASES.PIPELINE2)
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
    verifierArtifacts: preview.verifierArtifacts,
    title,
    file,
    selectedWorkflowMode: verifyEnabled ? 'verify' : 'publish',
    pipelineStages,
    pipelineFlowNodes: pipelineView.flowNodes,
    pipelinePriorNodeIds: pipelineView.priorNodeIds,
    pipelineLabel: pipelineView.label,
    currentStage: pipelineView.currentStage,
    errorMessage,
    expandedClaimKey,
    isVideoMode,
    seekToSeconds,
    actions: {
      setTitle,
      selectFile,
      upload,
      startVerify,
      startDirectUpload,
      backToChoice,
      openReview: () => setPhase(PHASES.REVIEWED),
      backToVerifyReady,
      confirmReview: reset,
      retry,
      reset,
      toggleClaim: key => setExpandedClaimKey(prev => prev === key ? '' : key),
      watchClaim: startTime => {
        setIsVideoMode(true)
        setSeekToSeconds({ seconds: Number(startTime) || 0, time: Date.now() })
      },
      exitVideo: () => setIsVideoMode(false),
    },
    stageGroupIndex,
  }
}
