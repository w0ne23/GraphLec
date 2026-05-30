import { useNavigate, useParams } from 'react-router-dom'
import PipelineProgress from '../../components/verifier/PipelineProgress'
import { PHASES, VERIFY_PROGRESS_PIPELINE_FLOW_NODES } from '../../components/verifier/verifierConstants'
import { useVerifierRouteFlow } from '../../hooks/useVerifierRouteFlow'

import '../../styles/verifier.css'

export default function VerifierProgressPage() {
  const { lectureId } = useParams()
  const navigate = useNavigate()
  const flow = useVerifierRouteFlow(lectureId, 'progress')
  const canOpenResult = flow.phase === PHASES.VERIFY_READY && flow.verifier

  if (flow.isLoading) {
    return (
      <div className="vf-page">
        <div className="vf-status-wrap">
          <div className="vf-status-inner">
            <div className="vf-status-title">검증 상태를 불러오는 중입니다.</div>
          </div>
        </div>
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
            phase={canOpenResult ? PHASES.VERIFY_READY : PHASES.PIPELINE1}
            errorMessage={flow.errorMessage}
            statusMessage={flow.currentStage || '검증 파이프라인을 진행 중입니다.'}
            flowNodes={VERIFY_PROGRESS_PIPELINE_FLOW_NODES}
          />
          <div className="vf-status-actions vf-status-actions--inline">
            <button className="vf-cancel-btn" onClick={flow.actions.reset}>업로드 취소</button>
            <button
              className="vf-confirm-btn"
              disabled={!canOpenResult}
              onClick={() => navigate(`/upload/${lectureId}/result`)}
            >
              {canOpenResult ? '결과 보기' : '검증 진행 중'}
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
