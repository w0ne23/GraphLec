import { useEffect, useMemo, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  confirmLectureVerification,
  deleteLecture,
  getLectureDetail,
  getLectureVerifier,
  retryUploadPublish as retryUploadPublishRequest,
} from '../lib/api'
import {
  PHASES,
  UPLOAD_STAGE_KEYS,
  VERIFY_PROGRESS_STAGE_KEYS,
  VERIFY_STAGE_KEYS,
  normalizePipelineStages,
} from '../components/verifier/verifierConstants'

const EMPTY_LECTURE = {
  id: '',
  title: '',
  video_url: '',
  scenes: [],
}

function verifierArtifactsFromResult(verifier) {
  const artifacts = verifier?.verifier_artifacts || {}
  return {
    ...artifacts,
    classifiedIssueVerifier: artifacts.classifiedIssueVerifier || verifier?.classified_issue_verifier,
    slideErrors: artifacts.slideErrors || { slide_errors: verifier?.slide_errors || [] },
  }
}

function getKnownStageKeys(phase, routeKind = '') {
  if (
    phase === PHASES.PIPELINE2 ||
    phase === PHASES.DONE ||
    routeKind === 'finalize' ||
    routeKind === 'direct'
  ) {
    return UPLOAD_STAGE_KEYS
  }
  return [...VERIFY_STAGE_KEYS, ...VERIFY_PROGRESS_STAGE_KEYS]
}

function mergeStageStatus(current, incoming, phase, routeKind = '') {
  const known = new Set(getKnownStageKeys(phase, routeKind))
  const byStage = new Map(current.map(item => [item.stage, item.status]))

  incoming.forEach(item => {
    if (!item?.stage || !known.has(item.stage)) return
    byStage.set(item.stage, item.status)
  })

  return normalizePipelineStages(Array.from(byStage, ([stage, status]) => ({ stage, status })))
}

function markKnownStages(phase, status, routeKind = '') {
  return normalizePipelineStages(getKnownStageKeys(phase, routeKind).map(stage => ({ stage, status })))
}

function createUploadRetryStages() {
  return normalizePipelineStages(UPLOAD_STAGE_KEYS.map(stage => ({
    stage,
    status: stage.startsWith('preprocess_') ? 'done' : 'wait',
  })))
}

function normalizeJobType(jobType) {
  const token = String(jobType || '').trim().toLowerCase().replaceAll('-', '_')
  if (['verify', 'verified', 'verified_upload'].includes(token)) return 'verify'
  if (['publish', 'publication', 'upload', 'direct', 'direct_upload', 'graph', 'graph_upload'].includes(token)) return 'publish'
  return token || 'legacy_full'
}

function phaseFromStatus(status, jobType, routeKind) {
  const normalizedJobType = normalizeJobType(jobType)
  if (status === 'error') return PHASES.ERROR
  if (routeKind === 'result') return PHASES.REVIEWED
  if (normalizedJobType === 'verify' && (status === 'done' || status === 'waiting_approval')) {
    return PHASES.VERIFY_READY
  }
  if (status === 'done') return PHASES.DONE
  if (routeKind === 'finalize' || routeKind === 'direct' || normalizedJobType === 'publish') {
    return PHASES.PIPELINE2
  }
  if (status === 'waiting_approval') return PHASES.VERIFY_READY
  return PHASES.PIPELINE1
}

function stageStatus(stages, key) {
  return stages.find(stage => stage.stage === key)?.status ?? 'wait'
}

export function useVerifierRouteFlow(lectureId, routeKind = 'progress') {
  const navigate = useNavigate()
  const eventSourceRef = useRef(null)

  const [lecture, setLecture] = useState(EMPTY_LECTURE)
  const [verifier, setVerifier] = useState(null)
  const [pipelineStages, setPipelineStages] = useState(() => normalizePipelineStages())
  const [phase, setPhase] = useState(PHASES.PIPELINE1)
  const [currentStage, setCurrentStage] = useState('')
  const [errorMessage, setErrorMessage] = useState('')
  const [isLoading, setIsLoading] = useState(true)
  const [isBusy, setIsBusy] = useState(false)
  const [expandedClaimKey, setExpandedClaimKey] = useState('')
  const [isVideoMode, setIsVideoMode] = useState(false)
  const [seekToSeconds, setSeekToSeconds] = useState(null)

  const verifierArtifacts = useMemo(() => verifierArtifactsFromResult(verifier), [verifier])

  function closeEventSource() {
    if (eventSourceRef.current) {
      eventSourceRef.current.close()
      eventSourceRef.current = null
    }
  }

  function connectJob(jobId = '') {
    if (!lectureId) return

    closeEventSource()
    const query = jobId ? `?job_id=${encodeURIComponent(jobId)}` : ''
    const eventSource = new EventSource(`/api/jobs/${lectureId}/stream${query}`)
    eventSourceRef.current = eventSource

    eventSource.onmessage = event => {
      try {
        const payload = JSON.parse(event.data)
        if (payload.error) throw new Error(payload.error)

        const nextPhase = phaseFromStatus(payload.lecture_status, payload.job_type, routeKind)
        setPhase(nextPhase)
        setCurrentStage(payload.current_stage || '')
        setErrorMessage(payload.error_message || '')
        setLecture(prev => ({
          ...prev,
          id: prev.id || lectureId,
          job_id: payload.job_id || prev.job_id,
          job_type: payload.job_type || prev.job_type,
          status: payload.lecture_status || prev.status,
        }))
        setPipelineStages(prev => mergeStageStatus(prev, payload.pipeline_stages || [], nextPhase, routeKind))

        if (
          payload.lecture_status === 'waiting_approval' ||
          (payload.lecture_status === 'done' && normalizeJobType(payload.job_type) === 'verify')
        ) {
          getLectureVerifier(lectureId).then(setVerifier).catch(() => {})
        }
        if (['done', 'error', 'waiting_approval', 'rejected'].includes(payload.lecture_status)) {
          setIsBusy(false)
          closeEventSource()
        }
      } catch (error) {
        closeEventSource()
        setIsBusy(false)
        setErrorMessage(String(error?.message || error))
        setPhase(PHASES.ERROR)
      }
    }

    eventSource.onerror = () => {
      closeEventSource()
      setIsBusy(false)
      setErrorMessage('서버와의 연결이 끊어졌습니다.')
      setPhase(PHASES.ERROR)
    }
  }

  useEffect(() => closeEventSource, [])

  useEffect(() => {
    if (!lectureId) return undefined

    let cancelled = false
    setIsLoading(true)
    setErrorMessage('')

    async function loadInitialState() {
      try {
        const [detailResult, verifierResult] = await Promise.allSettled([
          getLectureDetail(lectureId),
          getLectureVerifier(lectureId),
        ])

        if (cancelled) return

        if (detailResult.status === 'rejected') throw detailResult.reason

        const detail = detailResult.value || EMPTY_LECTURE
        const verifierData = verifierResult.status === 'fulfilled' ? verifierResult.value : null
        const nextPhase = phaseFromStatus(detail.status, detail.job_type, routeKind)

        setLecture({ ...EMPTY_LECTURE, ...detail })
        setVerifier(verifierData)
        setPhase(nextPhase)
        setCurrentStage(detail.current_stage || '')
        setPipelineStages(markKnownStages(nextPhase, detail.status === 'done' ? 'done' : 'wait', routeKind))
      } catch (error) {
        if (!cancelled) {
          setErrorMessage(String(error?.message || error))
          setPhase(PHASES.ERROR)
        }
      } finally {
        if (!cancelled) setIsLoading(false)
      }
    }

    loadInitialState()
    return () => {
      cancelled = true
    }
  }, [lectureId, routeKind])

  useEffect(() => {
    if (!lectureId) return undefined

    connectJob()
    return closeEventSource
  }, [lectureId, routeKind])

  async function retryUploadPublish() {
    if (!lectureId || isBusy) return
    setIsBusy(true)
    setErrorMessage('')
    setPhase(PHASES.PIPELINE2)
    setCurrentStage('업로드 파이프라인을 다시 시작합니다.')
    setPipelineStages(createUploadRetryStages())

    try {
      const result = await retryUploadPublishRequest(lectureId)
      setLecture(prev => ({
        ...prev,
        id: prev.id || lectureId,
        job_id: result.job_id || prev.job_id,
        job_type: result.job_type || 'publish',
        status: 'pending',
      }))
      connectJob(result.job_id)
    } catch (error) {
      setIsBusy(false)
      setErrorMessage(String(error?.message || error))
      setPhase(PHASES.ERROR)
    }
  }

  async function cancelUpload() {
    if (!lectureId || isBusy) return
    if (!window.confirm('이 작업과 생성된 파일을 삭제할까요?')) return

    closeEventSource()
    setIsBusy(true)
    setErrorMessage('')

    try {
      await deleteLecture(lectureId)
      navigate('/upload')
    } catch (error) {
      setIsBusy(false)
      setErrorMessage(String(error?.message || error))
      setPhase(PHASES.ERROR)
    }
  }

  async function confirmReview() {
    if (!lectureId || isBusy) return
    setIsBusy(true)
    setErrorMessage('')
    try {
      await confirmLectureVerification(lectureId)
      navigate('/upload')
    } catch (error) {
      setIsBusy(false)
      setErrorMessage(String(error?.message || error))
      setPhase(PHASES.ERROR)
    }
  }

  return {
    phase,
    lecture,
    verifier,
    verifierArtifacts,
    pipelineStages,
    currentStage,
    errorMessage,
    expandedClaimKey,
    isVideoMode,
    seekToSeconds,
    isBusy,
    isLoading,
    actions: {
      confirmReview,
      retryUploadPublish,
      cancelUpload,
      backToVerifyReady: () => navigate(`/upload/${lectureId}/progress`),
      reset: () => navigate('/upload'),
      toggleClaim: key => setExpandedClaimKey(prev => prev === key ? '' : key),
      watchClaim: startTime => {
        setIsVideoMode(true)
        setSeekToSeconds({ seconds: Number(startTime) || 0, time: Date.now() })
      },
      exitVideo: () => setIsVideoMode(false),
    },
  }
}
