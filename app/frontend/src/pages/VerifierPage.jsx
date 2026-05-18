import { useRef, useState } from 'react'
import PipelineProgress from '../components/verifier/PipelineProgress'
import VerifierReviewPanel from '../components/verifier/review/VerifierReviewPanel'
import { PHASES } from '../components/verifier/verifierConstants'
import { useVerifierPreviewFlow } from '../hooks/useVerifierPreviewFlow'

import '../styles/verifier.css'

function VerifierUploadStep({ flow }) {
  const fileRef = useRef()
  const [dragOver, setDragOver] = useState(false)
  const { actions } = flow

  return (
    <div className="vf-upload-wrap">
      <div className="vf-upload-inner">
        <p className="vf-upload-sub">강의 영상을 업로드하여 분석을 시작합니다.</p>
        {flow.errorMessage && <p className="vf-upload-error">{flow.errorMessage}</p>}

        <div
          className={`vf-dropzone${dragOver ? ' vf-dropzone--drag' : ''}${flow.file ? ' vf-dropzone--file' : ''}`}
          onClick={() => fileRef.current?.click()}
          onDragOver={e => { e.preventDefault(); setDragOver(true) }}
          onDragLeave={() => setDragOver(false)}
          onDrop={e => {
            e.preventDefault()
            setDragOver(false)
            actions.selectFile(e.dataTransfer.files[0])
          }}
        >
          <div className="vf-dropzone-icon">{flow.file ? '✅' : '🎬'}</div>
          {flow.file ? (
            <>
              <p className="vf-dropzone-text"><strong>{flow.file.name}</strong></p>
              <p className="vf-dropzone-sub">
                {flow.file.size ? `${(flow.file.size / 1024 / 1024).toFixed(1)} MB` : '실제 verification 결과 기반'}
              </p>
            </>
          ) : (
            <>
              <p className="vf-dropzone-text"><strong>드래그하거나 클릭</strong></p>
              <p className="vf-dropzone-sub">MP4, MOV, AVI · 최대 2GB</p>
            </>
          )}
          <input
            ref={fileRef}
            type="file"
            accept="video/*"
            style={{ display: 'none' }}
            onChange={e => actions.selectFile(e.target.files[0])}
          />
        </div>

        <div className="vf-field-group">
          <div className="vf-field">
            <label className="vf-field-label">강의 제목</label>
            <input
              className="vf-field-input"
              value={flow.title}
              onChange={e => actions.setTitle(e.target.value)}
              placeholder="제목을 입력하세요"
            />
          </div>
        </div>

        <button className="vf-submit-btn" onClick={actions.upload} disabled={!flow.file}>
          업로드 시작
        </button>
      </div>
    </div>
  )
}

function VerifierPipelineStep({ flow }) {
  const { actions } = flow

  return (
    <div className="vf-status-wrap">
      <div className="vf-status-inner">
        <div className="vf-status-title">{flow.lecture.title}</div>
        <div className="vf-status-label">
          {flow.phase === PHASES.PIPELINE2
            ? '2단계 분석 중'
            : flow.phase === PHASES.VERIFY_READY
              ? '1단계 분석 완료'
              : '1단계 분석 중'}
        </div>
        <PipelineProgress
          stages={flow.pipelineStages}
          phase={flow.phase}
          errorMessage={flow.errorMessage}
          statusMessage={flow.currentStage}
        />
        <div className="vf-status-actions">
          {flow.phase === PHASES.VERIFY_READY && (
            <button className="vf-confirm-btn" onClick={actions.openReview}>
              검증 결과 확인하기
            </button>
          )}
          <button className="vf-cancel-btn" onClick={actions.reset}>업로드 취소</button>
        </div>
      </div>
    </div>
  )
}

function VerifierDoneStep({ flow }) {
  return (
    <div className="vf-status-wrap">
      <div className="vf-status-inner">
        <div className="vf-done-icon">✅</div>
        <div className="vf-status-title">{flow.lecture.title}</div>
        <PipelineProgress
          stages={flow.pipelineStages}
          phase={PHASES.DONE}
          errorMessage={flow.errorMessage}
          statusMessage="분석이 완료되었습니다."
        />
        <button className="vf-reset-btn" onClick={flow.actions.reset}>새 강의 업로드</button>
      </div>
    </div>
  )
}

function VerifierErrorStep({ flow }) {
  const { actions } = flow

  return (
    <div className="vf-status-wrap">
      <div className="vf-status-inner">
        <div className="vf-status-label vf-status-label--err">오류 발생</div>
        <div className="vf-status-title">{flow.lecture.title}</div>
        <PipelineProgress
          stages={flow.pipelineStages}
          phase={PHASES.ERROR}
          errorMessage={flow.errorMessage}
          statusMessage="분석 중 오류가 발생했습니다."
        />
        <div className="vf-error-actions">
          <button className="vf-retry-btn" onClick={actions.retry}>재시도</button>
          <button className="vf-cancel-btn" onClick={actions.reset}>업로드 취소</button>
        </div>
      </div>
    </div>
  )
}

export default function VerifierPage() {
  const flow = useVerifierPreviewFlow()
  const isPipelinePhase =
    flow.phase === PHASES.PIPELINE1 ||
    flow.phase === PHASES.VERIFY_READY ||
    flow.phase === PHASES.PIPELINE2

  return (
    <div className="vf-page">
      {flow.phase === PHASES.UPLOAD && <VerifierUploadStep flow={flow} />}
      {isPipelinePhase && <VerifierPipelineStep flow={flow} />}
      {flow.phase === PHASES.REVIEWED && <VerifierReviewPanel flow={flow} />}
      {flow.phase === PHASES.DONE && <VerifierDoneStep flow={flow} />}
      {flow.phase === PHASES.ERROR && <VerifierErrorStep flow={flow} />}
    </div>
  )
}