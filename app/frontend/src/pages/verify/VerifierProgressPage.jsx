import { useEffect } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import VerifyReportPanels from '../../components/verifier/VerifyReportPanels'
import { PHASES } from '../../components/verifier/verifierConstants'
import { useVerifierRouteFlow } from '../../hooks/useVerifierRouteFlow'

import '../../styles/verifier.css'

export default function VerifierProgressPage() {
  const { lectureId } = useParams()
  const navigate = useNavigate()
  const flow = useVerifierRouteFlow(lectureId, 'progress')

  useEffect(() => {
    if (flow.phase === PHASES.VERIFY_READY && flow.verifier) {
      navigate(`/verify/${lectureId}/result`, { replace: true })
    }
  }, [flow.phase, flow.verifier, lectureId, navigate])

  const headerActions = (
    <div className="vf-flow-actions">
      <button className="vf-cancel-btn" onClick={flow.actions.reset}>업로드 취소</button>
      <button className="vf-confirm-btn" disabled>
        {flow.phase === PHASES.VERIFY_READY ? '결과 준비 완료' : '검증 진행 중'}
      </button>
    </div>
  )

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
      <div className="vf-flow-screen">
        <div className="vf-flow-screen-inner">
          <VerifyReportPanels flow={flow} headerActions={headerActions} />
        </div>
      </div>
    </div>
  )
}
