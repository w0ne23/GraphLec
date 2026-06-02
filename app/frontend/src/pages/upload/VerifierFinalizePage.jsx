import { useParams } from 'react-router-dom'
import PipelineProgress from '../../components/verifier/PipelineProgress'
import { FINALIZE_PIPELINE_FLOW_NODES, PHASES } from '../../components/verifier/verifierConstants'
import { useVerifierRouteFlow } from '../../hooks/useVerifierRouteFlow'

import '../../styles/verifier.css'

export default function VerifierFinalizePage() {
  const { lectureId } = useParams()
  const flow = useVerifierRouteFlow(lectureId, 'finalize')
  const isDone = flow.phase === PHASES.DONE
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
          <div className="vf-status-label">업로드 파이프라인</div>
          <PipelineProgress
            stages={flow.pipelineStages}
            phase={progressPhase}
            errorMessage={flow.errorMessage}
            statusMessage={flow.currentStage || '업로드 파이프라인을 진행 중입니다.'}
            flowNodes={FINALIZE_PIPELINE_FLOW_NODES}
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
