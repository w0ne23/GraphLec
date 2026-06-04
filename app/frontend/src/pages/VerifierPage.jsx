import { useEffect, useRef, useState } from 'react'
import PipelineProgress from '../components/verifier/PipelineProgress'
import PipelineStatusHeader from '../components/verifier/PipelineStatusHeader'
import VerifyReportPanels from '../components/verifier/VerifyReportPanels'
import VerifierReviewPanel from '../components/verifier/review/VerifierReviewPanel'
import { PHASES } from '../components/verifier/verifierConstants'
import { useVerifierPreviewFlow } from '../hooks/useVerifierPreviewFlow'
import { usePageTitle } from '../hooks/usePageTitle'

import '../styles/verifier.css'

function VerifierUploadStep({ flow }) {
  const fileRef = useRef()
  const [dragOver, setDragOver] = useState(false)
  const { actions } = flow
  const isVerify = flow.selectedWorkflowMode === 'verify'

  return (
    <div className="vf-upload-wrap">
      <div className="vf-upload-inner">
        <p className="vf-upload-sub">
          {isVerify ? '검증할 강의 영상' : '공개 업로드할 강의 영상'}
        </p>
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
          {isVerify ? '검증 시작' : '업로드 시작'}
        </button>
        <button className="vf-cancel-btn" onClick={actions.backToChoice}>이전</button>
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
          <h1>무엇을 할까요?</h1>
          <span>원하시는 작업을 선택해 주세요.</span>
        </div>
        <div className="vf-choice-grid">
          <button className="vf-choice-card vf-choice-card--primary" onClick={actions.startVerify}>
            <span>Verify</span>
            <strong>검증하기</strong>
          </button>
          <button className="vf-choice-card" onClick={actions.startDirectUpload}>
            <span>Publish</span>
            <strong>업로드하기</strong>
          </button>
        </div>
      </div>
    </div>
  )
}

function VerifyPipelineStep({ flow }) {
  const { actions } = flow

  return (
    <div className="vf-status-wrap">
      <div className="vf-status-inner">
        <PipelineStatusHeader title={flow.lecture.title} pipelineLabel={flow.pipelineLabel} />
        <PipelineProgress
          stages={flow.pipelineStages}
          phase={flow.phase}
          errorMessage={flow.errorMessage}
          statusMessage={flow.currentStage}
          flowNodes={flow.pipelineFlowNodes}
          priorNodeIds={flow.pipelinePriorNodeIds}
        />
        <div className="vf-status-actions">
          {flow.phase === PHASES.VERIFY_READY && (
            <button className="vf-confirm-btn" onClick={actions.openReview}>
              결과 보기
            </button>
          )}
          <button className="vf-cancel-btn" onClick={actions.reset}>검증 중단</button>
        </div>
      </div>
    </div>
  )
}

function PipelineRunStep({ flow }) {
  const { actions } = flow

  return (
    <div className="vf-status-wrap">
      <div className="vf-status-inner">
        <PipelineStatusHeader title={flow.lecture.title} pipelineLabel={flow.pipelineLabel} />
        <PipelineProgress
          stages={flow.pipelineStages}
          phase={flow.phase}
          errorMessage={flow.errorMessage}
          statusMessage={flow.currentStage}
          flowNodes={flow.pipelineFlowNodes}
          priorNodeIds={flow.pipelinePriorNodeIds}
        />
        <div className="vf-status-actions">
          <button className="vf-cancel-btn" onClick={actions.reset}>작업 취소</button>
        </div>
      </div>
    </div>
  )
}

function VerifierDoneStep({ flow }) {
  return (
    <div className="vf-status-wrap">
      <div className="vf-status-inner">
        <PipelineStatusHeader title={flow.lecture.title} pipelineLabel={flow.pipelineLabel} />
        <PipelineProgress
          stages={flow.pipelineStages}
          phase={PHASES.DONE}
          errorMessage={flow.errorMessage}
          statusMessage="업로드가 완료되었습니다."
          flowNodes={flow.pipelineFlowNodes}
          priorNodeIds={flow.pipelinePriorNodeIds}
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
        <PipelineStatusHeader title={flow.lecture.title} pipelineLabel={flow.pipelineLabel} />
        <PipelineProgress
          stages={flow.pipelineStages}
          phase={PHASES.ERROR}
          errorMessage={flow.errorMessage}
          statusMessage="분석 중 오류가 발생했습니다."
          flowNodes={flow.pipelineFlowNodes}
          priorNodeIds={flow.pipelinePriorNodeIds}
        />
        <div className="vf-error-actions">
          <button className="vf-retry-btn" onClick={actions.retry}>재시도</button>
          <button className="vf-cancel-btn" onClick={actions.reset}>작업 취소</button>
        </div>
      </div>
    </div>
  )
}

function VerifyDetailStep({ flow, onBackReview }) {
  const headerActions = (
    <div className="vf-flow-actions">
      <button className="vf-cancel-btn" onClick={onBackReview}>
        검토 결과
      </button>
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

export default function VerifierPage() {
  const flow = useVerifierPreviewFlow()
  usePageTitle(flow.phase === PHASES.VERIFY_CHOICE ? 'Upload' : flow.selectedWorkflowMode === 'verify' ? 'Verify' : 'Publish')
  const [reviewView, setReviewView] = useState('review')
  const isVerifyPipelinePhase =
    flow.phase === PHASES.PIPELINE1 ||
    flow.phase === PHASES.VERIFY_READY

  useEffect(() => {
    if (flow.phase !== PHASES.REVIEWED) setReviewView('review')
  }, [flow.phase])

  return (
    <div className="vf-page">
      {flow.phase === PHASES.UPLOAD && <VerifierUploadStep flow={flow} />}
      {flow.phase === PHASES.VERIFY_CHOICE && <VerifyChoiceStep flow={flow} />}
      {isVerifyPipelinePhase && <VerifyPipelineStep flow={flow} />}
      {flow.phase === PHASES.REVIEWED && reviewView === 'review' && (
        <VerifierReviewPanel flow={flow} onOpenDetail={() => setReviewView('detail')} />
      )}
      {flow.phase === PHASES.REVIEWED && reviewView === 'detail' && (
        <VerifyDetailStep flow={flow} onBackReview={() => setReviewView('review')} />
      )}
      {flow.phase === PHASES.PIPELINE2 && <PipelineRunStep flow={flow} />}
      {flow.phase === PHASES.DONE && <VerifierDoneStep flow={flow} />}
      {flow.phase === PHASES.ERROR && <VerifierErrorStep flow={flow} />}
    </div>
  )
}
