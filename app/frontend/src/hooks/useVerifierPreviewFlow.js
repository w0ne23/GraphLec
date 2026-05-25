import { useEffect, useMemo, useRef, useState } from 'react'
import {
  approveLectureUpload,
  getLectureDetail,
  getLectureVerifier,
  rejectLectureUpload,
  uploadLecture,
} from '../lib/api'
import {
  PHASES,
  UPLOAD_STAGE_KEYS,
  VERIFY_STAGE_KEYS,
  normalizePipelineStages,
} from '../components/verifier/verifierConstants'

const DEFAULT_CATEGORY = 'Default Category'

const EMPTY_FLOW = {
  lecture: {
    id: '',
    title: '',
    video_url: '',
    scenes: [],
  },
  file: null,
  verifier: null,
  verifierArtifacts: {},
}

function fileTitle(file) {
  return file?.name ? file.name.replace(/\.[^.]+$/, '') : ''
}

function getKnownStageKeys(phase) {
  if (phase === PHASES.PIPELINE2 || phase === PHASES.DONE) return UPLOAD_STAGE_KEYS
  return VERIFY_STAGE_KEYS
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

function markStages(stageKeys, status) {
  return normalizePipelineStages(stageKeys.map(stage => ({ stage, status })))
}

function lectureFromCreated(created, fallbackTitle) {
  return {
    id: created.id,
    job_id: created.job_id,
    job_type: created.job_type,
    title: created.title || fallbackTitle || 'Untitled',
    category: created.category || DEFAULT_CATEGORY,
    video_url: created.video_url || '',
    scenes: [],
  }
}

function verifierArtifactsFromResult(verifier) {
  return {
    classifiedIssueVerifier: verifier?.classified_issue_verifier,
    slideErrors: verifier?.slide_errors,
  }
}

export function useVerifierPreviewFlow() {
  const eventSourceRef = useRef(null)
  const activeJobRef = useRef(null)

  const [preview, setPreview] = useState(EMPTY_FLOW)
  const [title, setTitle] = useState('')
  const [file, setFile] = useState(null)

  const [phase, setPhase] = useState(PHASES.UPLOAD)
  const [pipelineStages, setPipelineStages] = useState(() => normalizePipelineStages())
  const [currentStage, setCurrentStage] = useState('')
  const [errorMessage, setErrorMessage] = useState('')
  const [isBusy, setIsBusy] = useState(false)

  const [expandedClaimKey, setExpandedClaimKey] = useState('')
  const [isVideoMode, setIsVideoMode] = useState(false)
  const [seekToSeconds, setSeekToSeconds] = useState(null)

  const lecture = preview.lecture
  const verifier = preview.verifier

  const visibleStages = useMemo(() => {
    if (phase === PHASES.PIPELINE2 || phase === PHASES.DONE) {
      return mergeStageStatus(markStages(UPLOAD_STAGE_KEYS, 'wait'), pipelineStages, PHASES.PIPELINE2)
    }
    return mergeStageStatus(markStages(VERIFY_STAGE_KEYS, 'wait'), pipelineStages, PHASES.PIPELINE1)
  }, [phase, pipelineStages])

  function closeEventSource() {
    if (eventSourceRef.current) {
      eventSourceRef.current.close()
      eventSourceRef.current = null
    }
    activeJobRef.current = null
  }

  useEffect(() => closeEventSource, [])

  function selectFile(nextFile) {
    if (!nextFile) return
    setFile(nextFile)
    setTitle(prev => prev.trim() ? prev : fileTitle(nextFile))
    setPreview(prev => ({
      ...prev,
      lecture: {
        ...prev.lecture,
        title: prev.lecture.title || fileTitle(nextFile),
      },
      file: nextFile,
    }))
  }

  function setRunningState(nextPhase, message) {
    setPhase(nextPhase)
    setCurrentStage(message)
    setErrorMessage('')
    setIsBusy(true)
    setPipelineStages(normalizePipelineStages())
  }

  async function loadVerifierResult(lectureId, fallbackLecture) {
    const [detailResult, verifierResult] = await Promise.allSettled([
      getLectureDetail(lectureId),
      getLectureVerifier(lectureId),
    ])

    const detail = detailResult.status === 'fulfilled' ? detailResult.value : null
    const verifierData = verifierResult.status === 'fulfilled' ? verifierResult.value : null

    setPreview(prev => ({
      lecture: {
        ...(fallbackLecture || prev.lecture),
        ...(detail || {}),
        scenes: detail?.scenes || prev.lecture.scenes || [],
      },
      file: prev.file,
      verifier: verifierData,
      verifierArtifacts: verifierArtifactsFromResult(verifierData),
    }))
  }

  function handleTerminalPayload(payload, runningPhase, fallbackLecture) {
    const status = payload.lecture_status
    if (status === 'error') {
      closeEventSource()
      setIsBusy(false)
      setErrorMessage(payload.error_message || '분석 중 오류가 발생했습니다.')
      setPhase(PHASES.ERROR)
      return true
    }

    if (status === 'waiting_approval') {
      closeEventSource()
      setIsBusy(false)
      setCurrentStage('검증 결과가 준비되었습니다.')
      setPipelineStages(markStages(VERIFY_STAGE_KEYS, 'done'))
      loadVerifierResult(fallbackLecture.id, fallbackLecture).finally(() => {
        setPhase(PHASES.VERIFY_READY)
      })
      return true
    }

    if (status === 'done') {
      closeEventSource()
      setIsBusy(false)
      setCurrentStage('분석이 완료되었습니다.')
      setPipelineStages(markStages(UPLOAD_STAGE_KEYS, 'done'))
      setPhase(PHASES.DONE)
      return true
    }

    if (status === 'rejected') {
      closeEventSource()
      setIsBusy(false)
      setCurrentStage('업로드가 거절되었습니다.')
      setPhase(PHASES.UPLOAD)
      return true
    }

    if (runningPhase === PHASES.PIPELINE2) setPhase(PHASES.PIPELINE2)
    if (runningPhase === PHASES.PIPELINE1) setPhase(PHASES.PIPELINE1)
    return false
  }

  function connectJob(lectureId, jobId, runningPhase, fallbackLecture) {
    closeEventSource()
    activeJobRef.current = { lectureId, jobId }

    const query = jobId ? `?job_id=${encodeURIComponent(jobId)}` : ''
    const eventSource = new EventSource(`/api/jobs/${lectureId}/stream${query}`)
    eventSourceRef.current = eventSource

    eventSource.onmessage = event => {
      try {
        const payload = JSON.parse(event.data)
        if (payload.error) {
          throw new Error(payload.error)
        }

        setCurrentStage(payload.current_stage || '')
        setPreview(prev => ({
          ...prev,
          lecture: {
            ...(fallbackLecture || prev.lecture),
            job_id: payload.job_id || fallbackLecture?.job_id,
            job_type: payload.job_type || fallbackLecture?.job_type,
          },
        }))
        setPipelineStages(prev => mergeStageStatus(prev, payload.pipeline_stages || [], runningPhase))
        handleTerminalPayload(payload, runningPhase, fallbackLecture)
      } catch (error) {
        closeEventSource()
        setIsBusy(false)
        setErrorMessage(String(error.message || error))
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

  async function startUpload(workflowMode, runningPhase) {
    if (!file || isBusy) return
    const uploadTitle = title.trim() || fileTitle(file)
    setRunningState(
      runningPhase,
      workflowMode === 'verified_upload' ? '검증 파이프라인을 시작합니다.' : '업로드 파이프라인을 시작합니다.'
    )

    try {
      const created = await uploadLecture({
        title: uploadTitle,
        category: DEFAULT_CATEGORY,
        description: '',
        file,
        workflowMode,
      })
      const createdLecture = lectureFromCreated(created, uploadTitle)
      setPreview(prev => ({
        ...prev,
        lecture: createdLecture,
        file,
      }))
      connectJob(created.id, created.job_id, runningPhase, createdLecture)
    } catch (error) {
      setIsBusy(false)
      setErrorMessage(String(error.message || error))
      setPhase(PHASES.ERROR)
    }
  }

  function upload() {
    if (!file) return
    setErrorMessage('')
    setPhase(PHASES.VERIFY_CHOICE)
  }

  function startVerify() {
    startUpload('verified_upload', PHASES.PIPELINE1)
  }

  function skipVerify() {
    startUpload('direct_upload', PHASES.PIPELINE2)
  }

  async function continueUpload() {
    if (!lecture.id || isBusy) return
    setRunningState(PHASES.PIPELINE2, '업로드 파이프라인을 이어서 시작합니다.')
    setPipelineStages(prev => mergeStageStatus(prev, [
      { stage: 'preprocess_extract_media', status: 'done' },
      { stage: 'preprocess_textualize_transcribe', status: 'done' },
      { stage: 'preprocess_enrich_audio_annotation', status: 'done' },
      { stage: 'preprocess_classify_scene', status: 'done' },
      { stage: 'preprocess_fusion', status: 'done' },
    ], PHASES.PIPELINE2))

    try {
      const approved = await approveLectureUpload(lecture.id)
      const nextLecture = {
        ...lecture,
        job_id: approved.job_id,
        job_type: approved.job_type,
      }
      setPreview(prev => ({ ...prev, lecture: nextLecture }))
      connectJob(lecture.id, approved.job_id, PHASES.PIPELINE2, nextLecture)
    } catch (error) {
      setIsBusy(false)
      setErrorMessage(String(error.message || error))
      setPhase(PHASES.ERROR)
    }
  }

  async function rejectUpload() {
    if (!lecture.id || isBusy) return
    if (!window.confirm('검증 결과를 거절하고 업로드 파일을 삭제할까요?')) return
    setIsBusy(true)
    try {
      await rejectLectureUpload(lecture.id)
      reset()
    } catch (error) {
      setIsBusy(false)
      setErrorMessage(String(error.message || error))
      setPhase(PHASES.ERROR)
    }
  }

  function reset() {
    closeEventSource()
    setPreview(EMPTY_FLOW)
    setPhase(PHASES.UPLOAD)
    setErrorMessage('')
    setPipelineStages(normalizePipelineStages())
    setCurrentStage('')
    setExpandedClaimKey('')
    setIsVideoMode(false)
    setSeekToSeconds(null)
    setIsBusy(false)
    setFile(null)
    setTitle('')
  }

  function retry() {
    if (!file) {
      reset()
      return
    }
    setErrorMessage('')
    setPhase(PHASES.VERIFY_CHOICE)
  }

  function backToVerifyReady() {
    setExpandedClaimKey('')
    setIsVideoMode(false)
    setSeekToSeconds(null)
    setPhase(PHASES.VERIFY_READY)
  }

  return {
    phase,
    lecture,
    verifier,
    verifierArtifacts: preview.verifierArtifacts,
    title,
    file,
    pipelineStages: visibleStages,
    currentStage,
    errorMessage,
    expandedClaimKey,
    isVideoMode,
    seekToSeconds,
    isBusy,
    actions: {
      setTitle,
      selectFile,
      upload,
      startVerify,
      skipVerify,
      continueUpload,
      rejectUpload,
      openReview: () => setPhase(PHASES.REVIEWED),
      backToVerifyReady,
      confirmReview: () => setPhase(PHASES.UPLOAD_RESUME),
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
