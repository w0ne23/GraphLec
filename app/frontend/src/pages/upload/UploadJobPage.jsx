import { useState } from 'react'
import { useParams } from 'react-router-dom'
import PipelineProgress from '../../components/verifier/PipelineProgress'
import VerifierReviewPanel from '../../components/verifier/review/VerifierReviewPanel'
import { PHASES } from '../../components/verifier/verifierConstants'
import { useJobStream } from '../../hooks/useJobStream'
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

function VerifyModePage({ flow }) {
  const [isReviewOpen, setIsReviewOpen] = useState(false)
  const canOpenResult = flow.phase === PHASES.VERIFY_READY && flow.verifier
  const isError = flow.phase === PHASES.ERROR
  const actionDisabled = flow.isRestarting || flow.isMutating

  if (isReviewOpen) {
    return (
      <div className="vf-page">
        <VerifierReviewPanel
          flow={{
            ...flow,
            actions: {
              ...flow.actions,
              backToVerifyReady: () => setIsReviewOpen(false),
            },
          }}
        />
      </div>
    )
  }

  return (
    <div className="vf-page">
      <div className="vf-status-wrap">
        <div className="vf-status-inner vf-status-inner--wide">
          <div className="vf-status-title">{flow.lecture.title || '강의 영상'}</div>
          <div className="vf-status-label">{flow.pipelineLabel}</div>
          <PipelineProgress
            stages={flow.pipelineStages}
            phase={canOpenResult ? PHASES.VERIFY_READY : flow.phase === PHASES.ERROR ? PHASES.ERROR : PHASES.PIPELINE1}
            errorMessage={flow.errorMessage}
            statusMessage={flow.currentStage || '검증 파이프라인을 진행 중입니다.'}
            flowNodes={flow.pipelineFlowNodes}
            priorNodeIds={flow.pipelinePriorNodeIds}
          />
          <div className="vf-status-actions vf-status-actions--inline">
            <button className="vf-cancel-btn" onClick={flow.actions.cancelUpload} disabled={actionDisabled}>
              작업 삭제
            </button>
            {(canOpenResult || isError) && (
              <button
                className="vf-confirm-btn"
                disabled={actionDisabled || (canOpenResult && !flow.verifier)}
                onClick={canOpenResult ? () => setIsReviewOpen(true) : () => flow.actions.restart('verify')}
              >
                {canOpenResult ? '결과 보기' : flow.isRestarting ? '재시작 중' : '재시작'}
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
          <div className="vf-status-title">{flow.lecture.title || '강의 영상'}</div>
          <div className="vf-status-label">{flow.pipelineLabel}</div>
          <PipelineProgress
            stages={flow.pipelineStages}
            phase={progressPhase}
            errorMessage={flow.errorMessage}
            statusMessage={flow.currentStage || '업로드 파이프라인을 진행 중입니다.'}
            flowNodes={flow.pipelineFlowNodes}
            priorNodeIds={flow.pipelinePriorNodeIds}
          />
          <div className="vf-status-actions vf-status-actions--inline">
            {isError && (
              <button
                className="vf-confirm-btn"
                disabled={actionDisabled}
                onClick={() => flow.actions.restart('publish')}
              >
                {flow.isRestarting ? '재시작 중' : '재시작'}
              </button>
            )}
            <button
              className={isDone ? 'vf-reset-btn' : 'vf-cancel-btn'}
              onClick={isDone ? flow.actions.reset : flow.actions.cancelUpload}
              disabled={actionDisabled}
            >
              {isDone ? '새 강의 업로드' : '업로드 취소'}
            </button>
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

  if (flow.isLoading) return <LoadingState mode={mode} />
  if (mode === 'publish') return <PublishModePage flow={flow} />
  return <VerifyModePage flow={flow} />
}
