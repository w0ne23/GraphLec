import { useParams } from 'react-router-dom'
import PipelineProgress from '../../components/verifier/PipelineProgress'
import { FINALIZE_PIPELINE_FLOW_NODES, PHASES } from '../../components/verifier/verifierConstants'
import { useVerifierRouteFlow } from '../../hooks/useVerifierRouteFlow'

import '../../styles/verifier.css'

export default function VerifierFinalizePage() {
  const { lectureId } = useParams()
  const flow = useVerifierRouteFlow(lectureId, 'finalize')
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
          <div className="vf-status-label">검증 이후 업로드 파이프라인</div>
          <PipelineProgress
            stages={flow.pipelineStages}
            phase={progressPhase}
            errorMessage={flow.errorMessage}
            statusMessage={flow.currentStage || '나머지 파이프라인을 진행 중입니다.'}
            flowNodes={FINALIZE_PIPELINE_FLOW_NODES}
          />
          <div className="vf-status-actions">
            {flow.phase === PHASES.ERROR && (
              <button
                className="vf-confirm-btn"
                disabled={flow.isBusy}
                onClick={flow.actions.retryGraphGeneration}
              >
                {flow.isBusy ? '재시도 중' : '그래프 생성 재시도'}
              </button>
            )}
            <button className="vf-cancel-btn" onClick={flow.actions.reset}>새 강의 업로드</button>
          </div>
        </div>
      </div>
    </div>
  )
}
