import { useState } from 'react'
import { useParams } from 'react-router-dom'
import PipelineProgress from '../../components/verifier/PipelineProgress'
import VerifierReviewPanel from '../../components/verifier/review/VerifierReviewPanel'
import {
  FINALIZE_PIPELINE_FLOW_NODES,
  PHASES,
  UPLOAD_PIPELINE_FLOW_NODES,
  VERIFY_PROGRESS_PIPELINE_FLOW_NODES,
} from '../../components/verifier/verifierConstants'
import { useVerifierRouteFlow } from '../../hooks/useVerifierRouteFlow'

import '../../styles/verifier.css'

function normalizeMode(value) {
  const token = String(value || '').trim().toLowerCase().replaceAll('-', '_')
  if (['publish', 'publication', 'upload', 'direct', 'direct_upload', 'graph', 'graph_upload'].includes(token)) {
    return 'publish'
  }
  return 'verify'
}

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
          <div className="vf-status-label">검증 파이프라인</div>
          <PipelineProgress
            stages={flow.pipelineStages}
            phase={canOpenResult ? PHASES.VERIFY_READY : flow.phase === PHASES.ERROR ? PHASES.ERROR : PHASES.PIPELINE1}
            errorMessage={flow.errorMessage}
            statusMessage={flow.currentStage || '검증 파이프라인을 진행 중입니다.'}
            flowNodes={VERIFY_PROGRESS_PIPELINE_FLOW_NODES}
          />
          <div className="vf-status-actions vf-status-actions--inline">
            <button className="vf-cancel-btn" onClick={flow.actions.cancelUpload} disabled={flow.isBusy}>
              작업 삭제
            </button>
            <button
              className="vf-confirm-btn"
              disabled={!canOpenResult}
              onClick={() => setIsReviewOpen(true)}
            >
              {canOpenResult ? '결과 보기' : '검증 진행 중'}
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}

function PublishModePage({ flow }) {
  const isDone = flow.phase === PHASES.DONE
  const progressPhase = flow.phase === PHASES.DONE
    ? PHASES.DONE
    : flow.phase === PHASES.ERROR
      ? PHASES.ERROR
      : PHASES.PIPELINE2
  const flowNodes = flow.lecture.is_verified
    ? FINALIZE_PIPELINE_FLOW_NODES
    : UPLOAD_PIPELINE_FLOW_NODES

  return (
    <div className="vf-page">
      <div className="vf-status-wrap">
        <div className="vf-status-inner">
          <div className="vf-status-title">{flow.lecture.title || '강의 영상'}</div>
          <div className="vf-status-label">업로드 파이프라인</div>
          <PipelineProgress
            stages={flow.pipelineStages}
            phase={progressPhase}
            errorMessage={flow.errorMessage}
            statusMessage={flow.currentStage || '업로드 파이프라인을 진행 중입니다.'}
            flowNodes={flowNodes}
          />
          <div className="vf-status-actions vf-status-actions--inline">
            {flow.phase === PHASES.ERROR && (
              <button
                className="vf-confirm-btn"
                disabled={flow.isBusy}
                onClick={flow.actions.retryUploadPublish}
              >
                {flow.isBusy ? '재시도 중' : '업로드 재시도'}
              </button>
            )}
            <button
              className={isDone ? 'vf-reset-btn' : 'vf-cancel-btn'}
              onClick={isDone ? flow.actions.reset : flow.actions.cancelUpload}
              disabled={flow.isBusy}
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
  const flow = useVerifierRouteFlow(lectureId, mode)

  if (flow.isLoading) return <LoadingState mode={mode} />
  if (mode === 'publish') return <PublishModePage flow={flow} />
  return <VerifyModePage flow={flow} />
}
