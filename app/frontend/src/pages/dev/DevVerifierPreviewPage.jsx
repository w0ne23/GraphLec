import { useEffect, useMemo, useState } from 'react'
import PipelineProgress from '../../components/verifier/PipelineProgress'
import VerifyReportPanels from '../../components/verifier/VerifyReportPanels'
import VerifierReviewPanel from '../../components/verifier/review/VerifierReviewPanel'
import {
  PHASES,
  FINALIZE_PIPELINE_FLOW_NODES,
  UPLOAD_PIPELINE_FLOW_NODES,
  UPLOAD_STAGE_KEYS,
  normalizePipelineStages,
} from '../../components/verifier/verifierConstants'

import '../../styles/verifier.css'

const PREVIEW_TICK_MS = 2000

const PREPROCESS_STAGE_KEYS = [
  'preprocess_extract_media',
  'preprocess_textualize_transcribe',
  'preprocess_enrich_audio_annotation',
  'preprocess_classify_scene',
  'preprocess_fusion',
]

const VERIFY_RUNTIME_STAGE_KEYS = [
  'verifier_build_analyzer_input',
  'verifier_run',
]
const FINALIZE_STAGE_KEYS = UPLOAD_STAGE_KEYS.filter(stage => !PREPROCESS_STAGE_KEYS.includes(stage))

const VERIFY_PREVIEW_STEP_COUNT = PREPROCESS_STAGE_KEYS.length + 5

const PREVIEW_LECTURE = {
  id: 'dev-verifier-preview',
  title: '운영체제 강의 영상',
  category: 'Default Category',
  video_url: '',
  scenes: [],
}

const PREVIEW_FILE = {
  name: '운영체제 강의 영상.mp4',
  size: null,
}

const PREVIEW_VERIFIER = {
  schema_version: 'content_verification.v2',
  mode: 'classified_issue_verifier',
  models: ['gpt', 'claude', 'grok'],
  counts: {
    final_confirmed: 1,
    needs_review: 1,
    rejected: 0,
    slide_errors: 1,
  },
  summary: {
    total_feedback_count: 2,
    confirmed_feedback_count: 1,
    review_needed_feedback_count: 1,
    rejected_feedback_count: 0,
  },
  claims: [
    {
      claim_id: 'CL0001',
      claim_text: '프로세스는 실행 중인 프로그램이다.',
      resolved_claim: '프로세스는 실행 중인 프로그램의 실행 단위다.',
      start_time: 42,
      slide_number: 3,
    },
    {
      claim_id: 'CL0002',
      claim_text: '스레드는 항상 독립된 주소 공간을 가진다.',
      resolved_claim: '스레드는 일반적으로 같은 프로세스의 주소 공간을 공유한다.',
      start_time: 128,
      slide_number: 5,
    },
  ],
  feedback_items: [
    {
      feedback_id: 'F0001',
      source_claim_id: 'CL0002',
      status: 'professor_check',
      feedback_type: 'factual_error',
      feedback_label: '사실 오류',
      claim_text: '스레드는 항상 독립된 주소 공간을 가진다.',
      resolved_claim: '스레드는 일반적으로 같은 프로세스의 주소 공간을 공유한다.',
      location: { slide_number: 5, start_time: 128 },
      severity_score: 0.82,
      problem: {
        summary: '스레드의 주소 공간 설명 확인 필요',
        problematic_content: '스레드는 항상 독립된 주소 공간을 가진다.',
        correct_info: '같은 프로세스의 스레드는 주소 공간을 공유한다.',
        recommendation: '프로세스와 스레드의 메모리 관계를 구분해 설명',
      },
      professor_feedback: {
        summary: '주소 공간 공유 여부가 반대로 전달될 수 있음',
      },
    },
    {
      feedback_id: 'F0002',
      source_claim_id: 'CL0001',
      status: 'confirmed',
      feedback_type: 'confusing_explanation',
      feedback_label: '혼동 설명',
      claim_text: '프로세스는 실행 중인 프로그램이다.',
      resolved_claim: '프로세스는 실행 중인 프로그램의 실행 단위다.',
      location: { slide_number: 3, start_time: 42 },
      severity_score: 0.48,
      problem: {
        summary: '표현 보강 권장',
        recommendation: '실행 상태와 자원 보유 관점을 함께 설명',
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
}

function stageStatusRows(stageKeys, cursor) {
  return stageKeys.map((stage, index) => {
    if (index < cursor) return { stage, status: 'done' }
    if (index === cursor) return { stage, status: 'run' }
    return { stage, status: 'wait' }
  })
}

function createVerifyPreviewStages(cursor) {
  const preprocessCursor = Math.min(cursor, PREPROCESS_STAGE_KEYS.length)
  const rows = stageStatusRows(PREPROCESS_STAGE_KEYS, preprocessCursor)

  if (cursor >= PREPROCESS_STAGE_KEYS.length) {
    const runtimeCursor = Math.min(cursor - PREPROCESS_STAGE_KEYS.length, VERIFY_RUNTIME_STAGE_KEYS.length)
    rows.push(...stageStatusRows(VERIFY_RUNTIME_STAGE_KEYS, runtimeCursor))
  }

  return normalizePipelineStages(rows)
}

function createUploadPreviewStages(cursor) {
  return normalizePipelineStages(stageStatusRows(UPLOAD_STAGE_KEYS, cursor))
}

function useDevVerifierPreviewFlow() {
  const [phase, setPhase] = useState(PHASES.UPLOAD)
  const [cursor, setCursor] = useState(-1)
  const [mode, setMode] = useState('verify')
  const [expandedClaimKey, setExpandedClaimKey] = useState('')
  const [isVideoMode, setIsVideoMode] = useState(false)
  const [seekToSeconds, setSeekToSeconds] = useState(null)

  useEffect(() => {
    if (phase !== PHASES.PIPELINE1 && phase !== PHASES.PIPELINE2) return undefined

    const limit = phase === PHASES.PIPELINE1
      ? VERIFY_PREVIEW_STEP_COUNT
      : mode === 'finalize'
        ? FINALIZE_STAGE_KEYS.length
        : UPLOAD_STAGE_KEYS.length
    const timer = window.setTimeout(() => {
      setCursor(current => {
        const next = current + 1
        if (next < limit) return next
        setPhase(phase === PHASES.PIPELINE1 ? PHASES.VERIFY_READY : PHASES.DONE)
        return -1
      })
    }, PREVIEW_TICK_MS)

    return () => window.clearTimeout(timer)
  }, [phase, cursor])

  const stageGroupIndex = phase === PHASES.PIPELINE1
    ? Math.max(0, Math.min(4, cursor - PREPROCESS_STAGE_KEYS.length))
    : -1

  const pipelineStages = useMemo(() => {
    if (phase === PHASES.PIPELINE1 || phase === PHASES.VERIFY_READY || phase === PHASES.REVIEWED || phase === PHASES.UPLOAD_RESUME) {
      if (phase !== PHASES.PIPELINE1) {
        return normalizePipelineStages([
          ...PREPROCESS_STAGE_KEYS.map(stage => ({ stage, status: 'done' })),
          ...VERIFY_RUNTIME_STAGE_KEYS.map(stage => ({ stage, status: 'done' })),
        ])
      }
      return createVerifyPreviewStages(Math.max(0, cursor))
    }
    if (phase === PHASES.DONE) {
      return normalizePipelineStages(UPLOAD_STAGE_KEYS.map(stage => ({ stage, status: 'done' })))
    }
    return normalizePipelineStages(stageStatusRows(mode === 'finalize' ? FINALIZE_STAGE_KEYS : UPLOAD_STAGE_KEYS, Math.max(0, cursor)))
  }, [cursor, mode, phase])

  function startPipeline(nextPhase, nextMode = mode) {
    setMode(nextMode)
    setCursor(0)
    setPhase(nextPhase)
  }

  function reset() {
    setPhase(PHASES.UPLOAD)
    setMode('verify')
    setCursor(-1)
    setExpandedClaimKey('')
    setIsVideoMode(false)
    setSeekToSeconds(null)
  }

  return {
    phase,
    stageGroupIndex,
    mode,
    lecture: PREVIEW_LECTURE,
    verifier: PREVIEW_VERIFIER,
    verifierArtifacts: {},
    title: PREVIEW_LECTURE.title,
    file: PREVIEW_FILE,
    pipelineStages,
    currentStage: '',
    errorMessage: '',
    expandedClaimKey,
    isVideoMode,
    seekToSeconds,
    actions: {
      setTitle: () => {},
      selectFile: () => {},
      upload: () => setPhase(PHASES.VERIFY_CHOICE),
      startVerify: () => startPipeline(PHASES.PIPELINE1, 'verify'),
      skipVerify: () => startPipeline(PHASES.PIPELINE2, 'direct'),
      continueUpload: () => startPipeline(PHASES.PIPELINE2, 'finalize'),
      openReview: () => setPhase(PHASES.REVIEWED),
      backToVerifyReady: () => setPhase(PHASES.VERIFY_READY),
      confirmReview: () => setPhase(PHASES.UPLOAD_RESUME),
      rejectUpload: reset,
      retry: () => startPipeline(PHASES.PIPELINE1, 'verify'),
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

function PreviewUploadStep({ flow }) {
  return (
    <div className="vf-upload-wrap">
      <div className="vf-upload-inner">
        <p className="vf-upload-sub">영상 업로드 없이 verifier 흐름을 확인합니다.</p>
        <div className="vf-dropzone vf-dropzone--file">
          <div className="vf-dropzone-icon">✓</div>
          <p className="vf-dropzone-text"><strong>{flow.file.name}</strong></p>
          <p className="vf-dropzone-sub">Preview data</p>
        </div>
        <div className="vf-field-group">
          <div className="vf-field">
            <label className="vf-field-label">강의 제목</label>
            <input className="vf-field-input" value={flow.title} readOnly />
          </div>
        </div>
        <button className="vf-submit-btn" onClick={flow.actions.upload}>
          다음
        </button>
      </div>
    </div>
  )
}

function PreviewChoiceStep({ flow }) {
  return (
    <div className="vf-choice-wrap">
      <div className="vf-choice-inner">
        <div className="vf-choice-head">
          <span>검증 파이프라인 Preview</span>
          <h1>{flow.title}</h1>
        </div>
        <div className="vf-choice-grid">
          <button className="vf-choice-card vf-choice-card--primary" onClick={flow.actions.startVerify}>
            <span>Preview</span>
            <strong>검증하기</strong>
            <em>2초마다 전처리와 검증 단계가 진행됩니다.</em>
          </button>
          <button className="vf-choice-card" onClick={flow.actions.skipVerify}>
            <span>Preview</span>
            <strong>검증 건너뛰기</strong>
            <em>검증 없이 업로드 파이프라인이 진행됩니다.</em>
          </button>
        </div>
        <button className="vf-cancel-btn" onClick={flow.actions.reset}>이전</button>
      </div>
    </div>
  )
}

function PreviewVerifyPipelineStep({ flow }) {
  const headerActions = (
    <div className="vf-flow-actions">
      <button className="vf-cancel-btn" onClick={flow.actions.reset}>Preview 종료</button>
      {flow.phase === PHASES.VERIFY_READY ? (
        <button className="vf-confirm-btn" onClick={flow.actions.openReview}>결과 보기</button>
      ) : (
        <button className="vf-confirm-btn" disabled>검증 진행 중</button>
      )}
    </div>
  )

  return (
    <div className="vf-flow-screen">
      <div className="vf-flow-screen-inner">
        <VerifyReportPanels flow={flow} headerActions={headerActions} />
      </div>
    </div>
  )
}

function PreviewUploadPipelineStep({ flow }) {
  const isFinalizePreview = flow.mode === 'finalize'

  return (
    <div className="vf-status-wrap">
      <div className="vf-status-inner">
        <div className="vf-status-title">{flow.lecture.title}</div>
        <div className="vf-status-label">{isFinalizePreview ? '검증 이후 업로드 파이프라인 Preview' : '업로드 파이프라인 Preview'}</div>
        <PipelineProgress
          stages={flow.pipelineStages}
          phase={flow.phase}
          errorMessage={flow.errorMessage}
          statusMessage="2초마다 다음 단계로 넘어갑니다."
          flowNodes={isFinalizePreview ? FINALIZE_PIPELINE_FLOW_NODES : UPLOAD_PIPELINE_FLOW_NODES}
        />
        <div className="vf-status-actions">
          <button className="vf-cancel-btn" onClick={flow.actions.reset}>Preview 종료</button>
        </div>
      </div>
    </div>
  )
}

function PreviewResumeStep({ flow }) {
  return (
    <div className="vf-choice-wrap">
      <div className="vf-choice-inner vf-choice-inner--resume">
        <div className="vf-choice-head">
          <span>검증 완료</span>
          <h1>{flow.lecture.title}</h1>
        </div>
        <div className="vf-resume-summary">
          <div>
            <span>검토 필요</span>
            <strong>{flow.verifier.counts.needs_review}</strong>
          </div>
          <div>
            <span>슬라이드 검토</span>
            <strong>{flow.verifier.counts.slide_errors}</strong>
          </div>
        </div>
        <button className="vf-submit-btn" onClick={flow.actions.continueUpload}>
          업로드 계속
        </button>
        <button className="vf-cancel-btn" onClick={flow.actions.rejectUpload}>
          Preview 종료
        </button>
      </div>
    </div>
  )
}

function PreviewDoneStep({ flow }) {
  const isFinalizePreview = flow.mode === 'finalize'

  return (
    <div className="vf-status-wrap">
      <div className="vf-status-inner">
        <div className="vf-done-icon">✓</div>
        <div className="vf-status-title">{flow.lecture.title}</div>
        <PipelineProgress
          stages={flow.pipelineStages}
          phase={PHASES.DONE}
          errorMessage={flow.errorMessage}
          statusMessage="Preview가 완료되었습니다."
          flowNodes={isFinalizePreview ? FINALIZE_PIPELINE_FLOW_NODES : UPLOAD_PIPELINE_FLOW_NODES}
        />
        <button className="vf-reset-btn" onClick={flow.actions.reset}>다시 보기</button>
      </div>
    </div>
  )
}

export default function DevVerifierPreviewPage() {
  const flow = useDevVerifierPreviewFlow()
  const isVerifyPipelinePhase = flow.phase === PHASES.PIPELINE1 || flow.phase === PHASES.VERIFY_READY

  return (
    <div className="vf-page">
      {flow.phase === PHASES.UPLOAD && <PreviewUploadStep flow={flow} />}
      {flow.phase === PHASES.VERIFY_CHOICE && <PreviewChoiceStep flow={flow} />}
      {isVerifyPipelinePhase && <PreviewVerifyPipelineStep flow={flow} />}
      {flow.phase === PHASES.REVIEWED && <VerifierReviewPanel flow={flow} />}
      {flow.phase === PHASES.UPLOAD_RESUME && <PreviewResumeStep flow={flow} />}
      {flow.phase === PHASES.PIPELINE2 && <PreviewUploadPipelineStep flow={flow} />}
      {flow.phase === PHASES.DONE && <PreviewDoneStep flow={flow} />}
    </div>
  )
}
