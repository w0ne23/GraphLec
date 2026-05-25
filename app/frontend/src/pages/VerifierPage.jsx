import { useRef, useState } from 'react'
import PipelineProgress from '../components/verifier/PipelineProgress'
import VerifyReportPanels from '../components/verifier/VerifyReportPanels'
import VerifierReviewPanel from '../components/verifier/review/VerifierReviewPanel'
import { PHASES, UPLOAD_PIPELINE_FLOW_NODES } from '../components/verifier/verifierConstants'
import { useVerifierPreviewFlow } from '../hooks/useVerifierPreviewFlow'

import '../styles/verifier.css'

function VerifierUploadStep({ flow }) {
  const fileRef = useRef()
  const [dragOver, setDragOver] = useState(false)
  const { actions } = flow

  return (
    <div className="vf-upload-wrap">
      <div className="vf-upload-inner">
        <p className="vf-upload-sub">강의 영상을 검증 파이프라인에 올립니다.</p>
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
          다음
        </button>
      </div>
    </div>
  )
}

function VerifyChoiceStep({ flow }) {
  const { actions } = flow

  return (
    <div className="vf-choice-wrap">
      <div className="vf-choice-inner">
        <div className="vf-choice-head">
          <span>검증 파이프라인</span>
          <h1>{flow.title || flow.lecture.title}</h1>
        </div>
        <div className="vf-choice-grid">
          <button className="vf-choice-card vf-choice-card--primary" onClick={actions.startVerify}>
            <span>권장</span>
            <strong>검증하기</strong>
            <em>검증 보고서를 생성한 뒤 결과 확인</em>
          </button>
          <button className="vf-choice-card" onClick={actions.skipVerify}>
            <span>선택</span>
            <strong>검증 건너뛰기</strong>
            <em>검증 없이 다음 단계로 진행</em>
          </button>
        </div>
        <button className="vf-cancel-btn" onClick={actions.reset}>이전</button>
      </div>
    </div>
  )
}

function VerifyPipelineStep({ flow }) {
  const { actions } = flow
  const flowActions = (
    <div className="vf-flow-actions">
      <button className="vf-cancel-btn" onClick={actions.reset}>업로드 취소</button>
      {flow.phase === PHASES.VERIFY_READY ? (
        <button className="vf-confirm-btn" onClick={actions.openReview}>
          결과 보기
        </button>
      ) : (
        <button className="vf-confirm-btn" disabled>
          검증 진행 중
        </button>
      )}
    </div>
  )

  return (
    <div className="vf-flow-screen">
      <div className="vf-flow-screen-inner">
        <VerifyReportPanels flow={flow} headerActions={flowActions} />
      </div>
    </div>
  )
}

function UploadPipelineStep({ flow }) {
  const { actions } = flow

  return (
    <div className="vf-status-wrap">
      <div className="vf-status-inner">
        <div className="vf-status-title">{flow.lecture.title}</div>
        <div className="vf-status-label">업로드 파이프라인</div>
        <PipelineProgress
          stages={flow.pipelineStages}
          phase={flow.phase}
          errorMessage={flow.errorMessage}
          statusMessage={flow.currentStage}
          flowNodes={UPLOAD_PIPELINE_FLOW_NODES}
        />
        <div className="vf-status-actions">
          <button className="vf-cancel-btn" onClick={actions.reset}>업로드 취소</button>
        </div>
      </div>
    </div>
  )
}

function UploadResumeStep({ flow }) {
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
            <strong>{flow.verifier?.counts?.needs_review ?? flow.verifier?.summary?.review_needed_feedback_count ?? '-'}</strong>
          </div>
          <div>
            <span>슬라이드 검토</span>
            <strong>{flow.verifier?.counts?.slide_errors ?? '-'}</strong>
          </div>
        </div>
        <button className="vf-submit-btn" onClick={flow.actions.continueUpload}>
          업로드 계속
        </button>
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
          flowNodes={UPLOAD_PIPELINE_FLOW_NODES}
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
  const isVerifyPipelinePhase =
    flow.phase === PHASES.PIPELINE1 ||
    flow.phase === PHASES.VERIFY_READY

  return (
    <div className="vf-page">
      {flow.phase === PHASES.UPLOAD && <VerifierUploadStep flow={flow} />}
      {flow.phase === PHASES.VERIFY_CHOICE && <VerifyChoiceStep flow={flow} />}
      {isVerifyPipelinePhase && <VerifyPipelineStep flow={flow} />}
      {flow.phase === PHASES.REVIEWED && <VerifierReviewPanel flow={flow} />}
      {flow.phase === PHASES.UPLOAD_RESUME && <UploadResumeStep flow={flow} />}
      {flow.phase === PHASES.PIPELINE2 && <UploadPipelineStep flow={flow} />}
      {flow.phase === PHASES.DONE && <VerifierDoneStep flow={flow} />}
      {flow.phase === PHASES.ERROR && <VerifierErrorStep flow={flow} />}
    </div>
  )
}