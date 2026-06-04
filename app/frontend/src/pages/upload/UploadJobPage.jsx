import { useEffect, useState } from 'react'
import { useParams } from 'react-router-dom'
import PipelineProgress from '../../components/verifier/PipelineProgress'
import PipelineStatusHeader from '../../components/verifier/PipelineStatusHeader'
import VerifyReportPanels from '../../components/verifier/VerifyReportPanels'
import VerifierReviewPanel from '../../components/verifier/review/VerifierReviewPanel'
import { PHASES } from '../../components/verifier/verifierConstants'
import { useJobStream } from '../../hooks/useJobStream'
import { usePageTitle } from '../../hooks/usePageTitle'
import { normalizeMode } from '../../lib/jobStreamUtils'

import '../../styles/verifier.css'

function LoadingState({ mode }) {
  return (
    <div className="vf-page">
      <div className="vf-status-wrap">
        <div className="vf-status-inner">
          <div className="vf-status-title">
            {mode === 'publish' ? '업로드 상태를 불러오는 중입니다.' : '검증 상태를 불러오는 중입니다.'}
          </div>
        </div>
      </div>
    </div>
  )
}

function VerifyDetailStep({ flow, onBackReview }) {
  const headerActions = (
    <div className="vf-flow-actions">
      <button className="vf-flow-close-btn" onClick={onBackReview} aria-label="검토 결과로 돌아가기">
        ×
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

function VerifyModePage({ flow }) {
  const [reviewView, setReviewView] = useState('status')
  const isVerifyReady = flow.phase === PHASES.VERIFY_READY
  const canOpenResult = isVerifyReady && flow.verifier
  const isError = flow.phase === PHASES.ERROR
  const actionDisabled = flow.isRestarting || flow.isMutating
  const reviewFlow = {
    ...flow,
    actions: {
      ...flow.actions,
      backToVerifyReady: () => setReviewView('status'),
    },
  }

  useEffect(() => {
    if (!isVerifyReady) setReviewView('status')
  }, [isVerifyReady])

  if (reviewView === 'review' && canOpenResult) {
    return (
      <div className="vf-page">
        <VerifierReviewPanel
          flow={reviewFlow}
          onOpenDetail={() => setReviewView('detail')}
        />
      </div>
    )
  }

  if (reviewView === 'detail' && canOpenResult) {
    return (
      <div className="vf-page">
        <VerifyDetailStep flow={reviewFlow} onBackReview={() => setReviewView('review')} />
      </div>
    )
  }

  return (
    <div className="vf-page">
      <div className="vf-status-wrap">
        <div className="vf-status-inner">
          <PipelineStatusHeader title={flow.lecture.title || '강의 영상'} pipelineLabel={flow.pipelineLabel} />
          <PipelineProgress
            stages={flow.pipelineStages}
            phase={isVerifyReady ? PHASES.VERIFY_READY : flow.phase === PHASES.ERROR ? PHASES.ERROR : PHASES.PIPELINE1}
            errorMessage={flow.errorMessage}
            statusMessage={flow.currentStage || '검증 파이프라인을 진행 중입니다.'}
            flowNodes={flow.pipelineFlowNodes}
            priorNodeIds={flow.pipelinePriorNodeIds}
          />
          <div className="vf-status-actions vf-status-actions--inline">
            {!isVerifyReady && (
              <button className="vf-cancel-btn" onClick={flow.actions.cancelUpload} disabled={actionDisabled}>
                검증 중단
              </button>
            )}
            {isError && (
              <button
                className="vf-confirm-btn"
                disabled={actionDisabled}
                onClick={() => flow.actions.restart('verify')}
              >
                {flow.isRestarting ? '재시작 중' : '재시작'}
              </button>
            )}
            {isVerifyReady && (
              <button
                className="vf-confirm-btn"
                disabled={actionDisabled || !canOpenResult}
                onClick={() => {
                  if (canOpenResult) setReviewView('review')
                }}
              >
                결과 보기
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

function PublishModePage({ flow }) {
  const isDone = flow.phase === PHASES.DONE
  const isError = flow.phase === PHASES.ERROR
  const actionDisabled = flow.isRestarting || flow.isMutating
  const progressPhase = flow.phase === PHASES.DONE
    ? PHASES.DONE
    : flow.phase === PHASES.ERROR
      ? PHASES.ERROR
      : PHASES.PIPELINE2

  return (
    <div className="vf-page">
      <div className="vf-status-wrap">
        <div className="vf-status-inner">
          <PipelineStatusHeader title={flow.lecture.title || '강의 영상'} pipelineLabel={flow.pipelineLabel} />
          <PipelineProgress
            stages={flow.pipelineStages}
            phase={progressPhase}
            errorMessage={flow.errorMessage}
            statusMessage={flow.currentStage || '업로드 파이프라인을 진행 중입니다.'}
            flowNodes={flow.pipelineFlowNodes}
            priorNodeIds={flow.pipelinePriorNodeIds}
          />
          <div className="vf-status-actions vf-status-actions--inline">
            {!isDone && (
              <button
                className="vf-cancel-btn"
                onClick={flow.actions.cancelUpload}
                disabled={actionDisabled}
              >
                업로드 취소
              </button>
            )}
            {isError && (
              <button
                className="vf-confirm-btn"
                disabled={actionDisabled}
                onClick={() => flow.actions.restart('publish')}
              >
                {flow.isRestarting ? '재시작 중' : '재시작'}
              </button>
            )}
            {isDone && (
              <button className="vf-reset-btn" onClick={flow.actions.reset} disabled={actionDisabled}>
                처음으로
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

export default function UploadJobPage({ mode: routeMode = 'verify' }) {
  const { lectureId } = useParams()
  const mode = normalizeMode(routeMode)
  const flow = useJobStream(lectureId, mode)
  const pageLabel = (mode === 'publish') ? 'Publish' : 'Verify'
  usePageTitle(flow.lecture.title ? `${flow.lecture.title} - ${pageLabel}` : pageLabel)

  if (flow.isLoading) return <LoadingState mode={mode} />
  if (mode === 'publish') return <PublishModePage flow={flow} />
  return <VerifyModePage flow={flow} />
}
