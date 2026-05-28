import { useParams } from 'react-router-dom'
import VerifierReviewPanel from '../../components/verifier/review/VerifierReviewPanel'
import { useVerifierRouteFlow } from '../../hooks/useVerifierRouteFlow'

import '../../styles/verifier.css'

export default function VerifierResultPage() {
  const { lectureId } = useParams()
  const flow = useVerifierRouteFlow(lectureId, 'result')

  return (
    <div className="vf-page">
      <VerifierReviewPanel flow={flow} />
    </div>
  )
}
