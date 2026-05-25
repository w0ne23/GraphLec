import { 
  compactText, 
  claimRowDomId, 
  getClaimFlowSource, 
  hasIssueInComparison, 
  isFinalReviewTarget, 
  getIssueType, 
  issueTypeTone, 
  typeLabel, 
  statusLabel, 
  getScore, 
  formatUnitValue, 
  IssueJudgeStatusBadge, 
  RowMeta, 
  ModelDecisionStrip, 
  ModelEvidenceSection, 
  DistributionScores, 
  ModelEvidenceAccordion, 
  FinalScoreSummary, 
  TextBlock, 
  ChipList,
  InlineBlock
} from '../../VerifyReportPanels'

export default function ClaimFlowRecord({ row, activeTab, resultId, displayId }) {
  const source = getClaimFlowSource(row)
  const severity = row.severity
  const type = row.type || row.severity
  const showClaim = activeTab === 'claim_extraction'
  const showJudge = activeTab === 'issue_judge'
  const showType = activeTab === 'issue_classification'
  const showFinal = activeTab === 'final_verification'
  const isIssueCandidate = showJudge && Boolean(row.issue || hasIssueInComparison(row))
  const isOkRow = (showJudge || showFinal) && !isIssueCandidate && !(showFinal && isFinalReviewTarget(severity))
  const claimText = source.resolved_claim || source.claim_text
  const originalClaimText = source.claim_text || source.resolved_claim
  const issueText = showJudge
    ? row.issue?.issue
    : showType
      ? type?.resolved_claim || type?.claim_text
      : showFinal
        ? type?.resolved_claim || type?.claim_text
        : ''
  const summaryText = showClaim ? originalClaimText : showJudge ? claimText : issueText || claimText
  const metaItems = showClaim
    ? []
    : showJudge
      ? []
      : showType
        ? [
            { value: typeLabel(getIssueType(type)), tone: issueTypeTone(getIssueType(type)) },
          ]
        : showFinal
          ? []
          : [
            { label: '결론', value: statusLabel(severity?.status) },
            { label: '점수', value: formatUnitValue(getScore(severity)) },
            { label: '수동 검토', value: severity?.needs_manual_review ? '필요' : '불필요', tone: severity?.needs_manual_review ? 'warn' : '' },
            ]

  return (
    <details
      id={claimRowDomId(row.claim_id)}
      className={`vf-record vf-claim-flow-record vf-claim-row ${showClaim ? 'vf-claim-row--claim-extraction' : ''} ${showJudge ? 'vf-claim-row--issue-judge' : ''} ${showType ? 'vf-claim-row--issue-classification' : ''} ${showFinal ? 'vf-claim-row--final-verification' : ''}`}
    >
      <summary className="vf-claim-row-summary">
        <div className="vf-claim-row-summary-inner">
          <span className={`vf-bold vf-claim-row-id ${isIssueCandidate ? 'vf-claim-row-id--issue' : ''}`}>{compactText(displayId || row.claim_id)}</span>
          <div className="vf-claim-row-content">
            {showClaim ? (
              <span className="vf-claim-row-text">{compactText(summaryText)}</span>
            ) : (
              <span className={`vf-claim-row-text ${isOkRow ? 'vf-claim-row-text--ok' : ''}`}>{compactText(summaryText)}</span>
            )}
            {showClaim ? (
              <span className="vf-claim-row-visual-placeholder" aria-hidden="true" />
            ) : showJudge || showFinal ? (
              <IssueJudgeStatusBadge row={row} severity={showFinal ? severity : null} />
            ) : (
              <RowMeta items={metaItems} />
            )}
          </div>
          <span className="vf-claim-row-toggle" aria-hidden="true" />
        </div>
      </summary>
      <div className="vf-claim-accordion-body">
          {showClaim && (
            <>
              <TextBlock label="추출된 주장">{source.resolved_claim || source.claim_text}</TextBlock>
              <ChipList items={[source.is_approximate ? 'approx' : '', source.claim_fingerprint]} />
            </>
          )}
          {showJudge && (
            <>
              <InlineBlock label="판단 결과">
                {row.comparison && <ModelDecisionStrip models={row.comparison.models} />}
              </InlineBlock>
              {row.issue ? (
                <ModelEvidenceSection items={row.issue.source_model_issues} />
              ) : (
                <div className="vf-report-note">이 단계에서 이슈 후보로 남지 않은 클레임입니다.</div>
              )}
            </>
          )}
          {showType && type && (
            <>
              <span className="vf-score-section-label">유형별 점수</span>
              <DistributionScores scores={type.weighted_scores} valueFormat="unit" />
              <ModelEvidenceAccordion items={type.model_classifications} valueFormat="unit" />
            </>
          )}
          {showFinal && severity && (
            <>
              <FinalScoreSummary severity={severity} />
              <ModelEvidenceAccordion items={severity.model_judgments} valueFormat="unit" />
            </>
          )}
      </div>
    </details>
  )
}
