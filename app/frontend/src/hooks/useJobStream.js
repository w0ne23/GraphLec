import { useEffect, useMemo, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { PHASES, normalizePipelineStages } from '../components/verifier/verifierConstants'
import {
  confirmLectureVerification,
  deleteLecture,
  getLectureDetail,
  getLectureVerifier,
  retryLecture as retryJobRequest,
} from '../lib/api'
import {
  createInitialStages,
  getPipelineFlowNodes,
  getPipelineLabel,
  getPipelinePriorNodeIds,
  markKnownStages,
  mergeStageStatus,
  normalizeJobType,
  normalizeMode,
  phaseFromStatus,
} from '../lib/jobStreamUtils'

const EMPTY_LECTURE = {
  id: '',
  title: '',
  video_url: '',
  scenes: [],
  is_verified: false,
  is_published: false,
}

function verifierArtifactsFromResult(verifier) {
  const artifacts = verifier?.verifier_artifacts || {}
  return {
    ...artifacts,
    classifiedIssueVerifier: artifacts.classifiedIssueVerifier || verifier?.classified_issue_verifier,
    slideErrors: artifacts.slideErrors || { slide_errors: verifier?.slide_errors || [] },
  }
}

export function useJobStream(lectureId, mode = 'verify') {
  const navigate = useNavigate()
  const normalizedMode = normalizeMode(mode)

  const eventSourceRef = useRef(null)
  const activeJobIdRef = useRef(null)
  const isVerifiedRef = useRef(false)

  const [lecture, setLecture] = useState(EMPTY_LECTURE)
  const [verifier, setVerifier] = useState(null)
  const [pipelineStages, setPipelineStages] = useState(() => normalizePipelineStages())
  const [phase, setPhase] = useState(PHASES.PIPELINE1)
  const [currentStage, setCurrentStage] = useState('')
  const [errorMessage, setErrorMessage] = useState('')
  const [isLoading, setIsLoading] = useState(true)
  const [isRestarting, setIsRestarting] = useState(false)
  const [isMutating, setIsMutating] = useState(false)
  const [expandedClaimKey, setExpandedClaimKey] = useState('')
  const [isVideoMode, setIsVideoMode] = useState(false)
  const [seekToSeconds, setSeekToSeconds] = useState(null)

  const verifierArtifacts = useMemo(() => verifierArtifactsFromResult(verifier), [verifier])
  const isVerified = Boolean(lecture.is_verified)

  const pipelineFlowNodes = useMemo(
    () => getPipelineFlowNodes(normalizedMode, isVerified),
    [normalizedMode, isVerified]
  )
  const pipelinePriorNodeIds = useMemo(
    () => getPipelinePriorNodeIds(normalizedMode, isVerified),
    [normalizedMode, isVerified]
  )
  const pipelineLabel = getPipelineLabel(normalizedMode)

  function closeEventSource() {
    if (eventSourceRef.current) {
      eventSourceRef.current.close()
      eventSourceRef.current = null
    }
  }

  function connectJob(jobId = '') {
    if (!lectureId) return

    closeEventSource()
    activeJobIdRef.current = jobId || null

    const params = new URLSearchParams()
    if (jobId) params.set('job_id', jobId)
    params.set('mode', normalizedMode)

    const url = `/api/jobs/${lectureId}/stream?${params.toString()}`
    console.log(`--- [SSE] Connecting to stream for lecture ${lectureId} (job ${jobId || 'latest'}, mode ${normalizedMode}) ---`)
    const eventSource = new EventSource(url)
    eventSourceRef.current = eventSource

    eventSource.onmessage = event => {
      try {
        const payload = JSON.parse(event.data)
        if (payload.error) {
          console.error('SSE Error:', payload.error)
          throw new Error(payload.error)
        }
        if (activeJobIdRef.current && payload.job_id && payload.job_id !== activeJobIdRef.current) return

        const nextPhase = phaseFromStatus(payload.lecture_status, payload.job_type, normalizedMode)
        const currentIsVerified = isVerifiedRef.current

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
        setPipelineStages(prev =>
          mergeStageStatus(prev, payload.pipeline_stages || [], nextPhase, normalizedMode, currentIsVerified)
        )

        if (
          payload.lecture_status === 'waiting_approval' ||
          (payload.lecture_status === 'done' && normalizeJobType(payload.job_type) === 'verify')
        ) {
          getLectureVerifier(lectureId).then(setVerifier).catch(() => {})
        }

        if (['done', 'error', 'waiting_approval', 'rejected'].includes(payload.lecture_status)) {
          closeEventSource()
        }
      } catch (error) {
        console.error('SSE Parse error:', error)
        closeEventSource()
        setErrorMessage(String(error?.message || error))
        setPhase(PHASES.ERROR)
      }
    }

    eventSource.onerror = () => {
      console.error('SSE connection error')
      closeEventSource()
      setErrorMessage('서버와의 연결이 끊어졌습니다.')
      setPhase(PHASES.ERROR)
    }
  }

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
        const detailIsVerified = Boolean(detail.is_verified)
        const nextPhase = phaseFromStatus(detail.status, detail.job_type, normalizedMode)

        isVerifiedRef.current = detailIsVerified
        setLecture({ ...EMPTY_LECTURE, ...detail })
        setVerifier(verifierData)
        setPhase(nextPhase)
        setCurrentStage(detail.current_stage || '')

        if (Array.isArray(detail.pipeline_stages) && detail.pipeline_stages.length > 0) {
          setPipelineStages(
            mergeStageStatus(
              markKnownStages(nextPhase, 'wait', normalizedMode, detailIsVerified),
              detail.pipeline_stages,
              nextPhase,
              normalizedMode,
              detailIsVerified
            )
          )
        } else {
          setPipelineStages(
            markKnownStages(nextPhase, detail.status === 'done' ? 'done' : 'wait', normalizedMode, detailIsVerified)
          )
        }

        connectJob()
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
      closeEventSource()
    }
  }, [lectureId, normalizedMode])

  async function restart(targetMode) {
    if (!lectureId || isRestarting || isMutating) return
    const resolvedMode = normalizeMode(targetMode)
    const currentIsVerified = isVerifiedRef.current

    setIsRestarting(true)
    setErrorMessage('')
    if (resolvedMode !== 'publish') setVerifier(null)
    setPhase(resolvedMode === 'publish' ? PHASES.PIPELINE2 : PHASES.PIPELINE1)
    setCurrentStage(resolvedMode === 'publish' ? '업로드 파이프라인을 다시 시작합니다.' : '검증 파이프라인을 다시 시작합니다.')
    setPipelineStages(createInitialStages(resolvedMode, currentIsVerified))

    try {
      const result = await retryJobRequest(lectureId, { mode: resolvedMode })
      setLecture(prev => ({
        ...prev,
        id: prev.id || lectureId,
        job_id: result.job_id || prev.job_id,
        job_type: result.job_type || resolvedMode,
        status: 'pending',
      }))
      setIsRestarting(false)
      connectJob(result.job_id)
    } catch (error) {
      setIsRestarting(false)
      setErrorMessage(String(error?.message || error))
      setPhase(PHASES.ERROR)
    }
  }

  async function cancelUpload() {
    if (!lectureId || isRestarting || isMutating) return
    if (!window.confirm('이 작업과 생성된 파일을 삭제할까요?')) return

    closeEventSource()
    setIsMutating(true)
    setErrorMessage('')

    try {
      await deleteLecture(lectureId)
      navigate('/upload')
    } catch (error) {
      setIsMutating(false)
      setErrorMessage(String(error?.message || error))
      setPhase(PHASES.ERROR)
    }
  }

  async function confirmReview() {
    if (!lectureId || isRestarting || isMutating) return
    setIsMutating(true)
    setErrorMessage('')

    try {
      await confirmLectureVerification(lectureId)
      navigate('/upload')
    } catch (error) {
      setIsMutating(false)
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
    pipelineFlowNodes,
    pipelinePriorNodeIds,
    pipelineLabel,
    currentStage,
    errorMessage,
    expandedClaimKey,
    isVideoMode,
    seekToSeconds,
    isLoading,
    isRestarting,
    isMutating,
    actions: {
      restart,
      cancelUpload,
      confirmReview,
      backToVerifyReady: () => navigate(`/verify/${lectureId}`),
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
