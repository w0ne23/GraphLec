import { 
  asArray,
  asObject, 
  getScore,
  statusLabel,
  statusFromSeverityScore,
  FinalScoreSummary,
  ModelEvidenceAccordion,
  ContextPreview,
  EmptyFiltered,
  MetricStrip,
  MouseTooltip
} from '../VerifyReportPanels'
import ClaimFlowRecord from './shared/ClaimFlowRecord'

/**
 * Stage 4: Final Verification
 * 모든 모델의 의견을 종합하여 최종 오류 여부와 심각도를 평가합니다.
 * 강의자 검토가 필요한 항목들을 식별하는 핵심 단계입니다.
 */
export default function FinalVerificationStage({ model, rows, resultId, activeFilter, status }) {
  const isDone = status === 'done'
  const summary = asObject(model.issueVerifier.summary)

  return (
    <div className="vf-stage-final">
      {rows.length > 0 ? (
        <div className="vf-record-list">
          {rows.map((row, rowIndex) => (
            <ClaimFlowRecord
              key={row.claim_id}
              row={row}
              activeTab="final_verification"
              resultId={resultId}
              displayId={row.claim_id}
            />
          ))}
        </div>
      ) : (
        <EmptyFiltered activeFilter={activeFilter} />
      )}
    </div>
  )
}

// --- Stage 4 전용 로직 ---

export function finalReviewReasonLabels(severity) {
  const FINAL_REVIEW_SCORE_THRESHOLD = 0.2
  const MANUAL_REVIEW_DISAGREEMENT_THRESHOLD = 0.35
  const LOW_MARGIN_THRESHOLD = 0.1

  if (!severity) return []
  const score = Number(getScore(severity))
  const status = severity.status || statusFromSeverityScore(score)
  const labels = []
  
  if (
    status === 'confirmed' ||
    status === 'professor_check' ||
    status === 'review_needed' ||
    (Number.isFinite(score) && score > FINAL_REVIEW_SCORE_THRESHOLD)
  ) {
    labels.push('문제 기준 초과')
  }

  const disagreement = Number(severity.model_disagreement ?? severity.classified_issue_verifier?.model_disagreement)
  if (Number.isFinite(disagreement) && disagreement >= MANUAL_REVIEW_DISAGREEMENT_THRESHOLD) {
    labels.push('모델 의견 불합치')
  }

  const margin = Number(severity.previous_classification?.margin)
  if (
    severity.previous_classification?.low_margin ||
    (Number.isFinite(margin) && margin < LOW_MARGIN_THRESHOLD)
  ) {
    labels.push('분류 모호함')
  }

  return labels
}

export function finalProblemThresholdCount(items) {
  const FINAL_REVIEW_SCORE_THRESHOLD = 0.2
  return asArray(items).filter(item => {
    const score = Number(getScore(item))
    const status = item?.status || statusFromSeverityScore(score)
    return (
      status === 'confirmed' ||
      status === 'professor_check' ||
      status === 'review_needed' ||
      (Number.isFinite(score) && score > FINAL_REVIEW_SCORE_THRESHOLD)
    )
  }).length
}

export function finalModelDisagreementCount(items) {
  const MANUAL_REVIEW_DISAGREEMENT_THRESHOLD = 0.35
  return asArray(items).filter(item => {
    const disagreement = Number(item?.model_disagreement ?? item?.classified_issue_verifier?.model_disagreement)
    return Number.isFinite(disagreement) && disagreement >= MANUAL_REVIEW_DISAGREEMENT_THRESHOLD
  }).length
}

export function finalLowMarginCount(items) {
  const LOW_MARGIN_THRESHOLD = 0.1
  return asArray(items).filter(item => {
    const margin = Number(item?.previous_classification?.margin)
    return item?.previous_classification?.low_margin || (Number.isFinite(margin) && margin < LOW_MARGIN_THRESHOLD)
  }).length
}

export function finalReviewConditionTooltip() {
  return (
    <span className="vf-final-condition-tooltip">
      <span>강의자 검토가 필요한 경우</span>
      <span>문제 기준 초과: 최종 점수 &gt; 0.20</span>
      <span>모델 의견 불합치: 모델 불일치 &gt;= 0.35</span>
      <span>분류 모호함: 유형 margin &lt; 0.10</span>
    </span>
  )
}
