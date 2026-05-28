import { useEffect, useMemo, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  approveLectureUpload,
  getLectureDetail,
  getLectureVerifier,
  rejectLectureUpload,
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

function getKnownStageKeys(phase) {
  if (phase === PHASES.PIPELINE2 || phase === PHASES.DONE) return UPLOAD_STAGE_KEYS
  return [...VERIFY_STAGE_KEYS, ...VERIFY_PROGRESS_STAGE_KEYS]
}

function mergeStageStatus(current, incoming, phase) {
  const known = new Set(getKnownStageKeys(phase))
  const byStage = new Map(current.map(item => [item.stage, item.status]))

  incoming.forEach(item => {
    if (!item?.stage || !known.has(item.stage)) return
    byStage.set(item.stage, item.status)
  })

  return normalizePipelineStages(Array.from(byStage, ([stage, status]) => ({ stage, status })))
}

function markKnownStages(phase, status) {
  return normalizePipelineStages(getKnownStageKeys(phase).map(stage => ({ stage, status })))
}

function phaseFromStatus(status, jobType, routeKind) {
  if (status === 'error') return PHASES.ERROR
  if (status === 'done') return PHASES.DONE
  if (routeKind === 'result') return PHASES.REVIEWED
  if (routeKind === 'finalize' || routeKind === 'upload-progress' || jobType === 'direct_upload' || jobType === 'graph_upload') {
    return PHASES.PIPELINE2
  }
  if (status === 'waiting_approval') return PHASES.VERIFY_READY
  return PHASES.PIPELINE1
}

const VERIFIER_RUN_VISUAL_STAGE_KEYS = [
  'verifier_claim_extraction',
  'verifier_issue_judge',
  'verifier_issue_classification',
  'verifier_final_verification',
]

function stageStatus(stages, key) {
  return stages.find(stage => stage.stage === key)?.status ?? 'wait'
}

function expandVerifierRunStages(stages, phase, visualIndex) {
  const rows = [...stages]
  const verifierRunStatus = stageStatus(rows, 'verifier_run')
  const verifierReady = phase === PHASES.VERIFY_READY || phase === PHASES.REVIEWED || phase === PHASES.UPLOAD_RESUME

  VERIFIER_RUN_VISUAL_STAGE_KEYS.forEach((stage, index) => {
    let status = 'wait'
    if (verifierReady || verifierRunStatus === 'done') {
      status = 'done'
    } else if (verifierRunStatus === 'run') {
      const activeIndex = Math.max(0, Math.min(VERIFIER_RUN_VISUAL_STAGE_KEYS.length - 1, visualIndex))
      if (index < activeIndex) status = 'done'
      if (index === activeIndex) status = 'run'
    }
    rows.push({ stage, status })
  })

  return normalizePipelineStages(rows)
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
  const [verifierRunVisualIndex, setVerifierRunVisualIndex] = useState(0)

  const verifierArtifacts = useMemo(() => verifierArtifactsFromResult(verifier), [verifier])
  const visiblePipelineStages = useMemo(
    () => expandVerifierRunStages(pipelineStages, phase, verifierRunVisualIndex),
    [phase, pipelineStages, verifierRunVisualIndex]
  )

  function closeEventSource() {
    if (eventSourceRef.current) {
      eventSourceRef.current.close()
      eventSourceRef.current = null
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
        setPipelineStages(markKnownStages(nextPhase, detail.status === 'done' ? 'done' : 'wait'))
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

    closeEventSource()
    const eventSource = new EventSource(`/api/jobs/${lectureId}/stream`)
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
        setPipelineStages(prev => mergeStageStatus(prev, payload.pipeline_stages || [], nextPhase))

        if (payload.lecture_status === 'waiting_approval') {
          getLectureVerifier(lectureId).then(setVerifier).catch(() => {})
        }
        if (['done', 'error', 'waiting_approval', 'rejected'].includes(payload.lecture_status)) {
          closeEventSource()
        }
      } catch (error) {
        closeEventSource()
        setErrorMessage(String(error?.message || error))
        setPhase(PHASES.ERROR)
      }
    }

    eventSource.onerror = () => {
      closeEventSource()
      setErrorMessage('서버와의 연결이 끊어졌습니다.')
      setPhase(PHASES.ERROR)
    }

    return closeEventSource
  }, [lectureId, routeKind])

  useEffect(() => {
    if (phase !== PHASES.PIPELINE1 || stageStatus(pipelineStages, 'verifier_run') !== 'run') {
      setVerifierRunVisualIndex(0)
      return undefined
    }

    const timer = window.setTimeout(() => {
      setVerifierRunVisualIndex(index => Math.min(VERIFIER_RUN_VISUAL_STAGE_KEYS.length - 1, index + 1))
    }, 2000)
    return () => window.clearTimeout(timer)
  }, [phase, pipelineStages, verifierRunVisualIndex])

  async function continueUpload() {
    if (!lectureId || isBusy) return
    setIsBusy(true)
    setErrorMessage('')
    try {
      await approveLectureUpload(lectureId)
      navigate(`/verify/${lectureId}/finalize`)
    } catch (error) {
      setErrorMessage(String(error?.message || error))
      setPhase(PHASES.ERROR)
    } finally {
      setIsBusy(false)
    }
  }

  async function rejectUpload() {
    if (!lectureId || isBusy) return
    if (!window.confirm('검증 결과를 거절하고 업로드 파일을 삭제할까요?')) return
    setIsBusy(true)
    setErrorMessage('')
    try {
      await rejectLectureUpload(lectureId)
      navigate('/verify')
    } catch (error) {
      setErrorMessage(String(error?.message || error))
      setPhase(PHASES.ERROR)
    } finally {
      setIsBusy(false)
    }
  }

  return {
    phase,
    lecture,
    verifier,
    verifierArtifacts,
    pipelineStages: visiblePipelineStages,
    currentStage,
    errorMessage,
    expandedClaimKey,
    isVideoMode,
    seekToSeconds,
    isBusy,
    isLoading,
    actions: {
      continueUpload,
      confirmReview: continueUpload,
      rejectUpload,
      backToVerifyReady: () => navigate(`/verify/${lectureId}/progress`),
      reset: () => navigate('/verify'),
      toggleClaim: key => setExpandedClaimKey(prev => prev === key ? '' : key),
      watchClaim: startTime => {
        setIsVideoMode(true)
        setSeekToSeconds({ seconds: Number(startTime) || 0, time: Date.now() })
      },
      exitVideo: () => setIsVideoMode(false),
    },
  }
}
