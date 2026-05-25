import { 
  asArray, 
  asObject, 
  compactText, 
  agreementLabel, 
  formatScore, 
  locationText, 
  ModelDecisionStrip, 
  ModelEvidenceSection, 
  EmptyFiltered, 
  ChipList,
  MetricStrip,
  MouseTooltip
} from '../VerifyReportPanels'
import ClaimFlowRecord from './shared/ClaimFlowRecord'

/**
 * Stage 2: Issue Candidate Judging
 * 추출된 클레임들에 대해 여러 모델의 판단 결과를 비교하고 이슈 후보를 보여줍니다.
 */
export default function IssueJudgeStage({ model, rows, resultId, activeFilter }) {
  const summary = model.issueJudgeSummary

  return (
    <div className="vf-stage-judge">
      <div className="vf-stage-summary">
        <ChipList items={[
          `모델: ${asArray(model.issueJudge.models).join(', ')}`,
          `모델별 이슈 수: ${JSON.stringify(summary.issue_counts_by_model || {})}`,
        ]} />
      </div>

      <IssueComparePanel 
        model={model} 
        rows={rows} 
        activeFilter={activeFilter} 
      />

      {rows.length > 0 ? (
        <div className="vf-record-list">
          {rows.map((row, rowIndex) => (
            <ClaimFlowRecord
              key={row.claim_id}
              row={row}
              activeTab="issue_judge"
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

// --- Stage 2 전용 컴포넌트 ---

export function IssueComparePanel({ model, rows, activeFilter }) {
  const compare = model.issueJudgeCompare
  const summary = asObject(compare.summary)
  const exclusiveByModel = asObject(compare.exclusive_by_model)
  
  if (!rows.length && !Object.keys(summary).length) return null

  return (
    <div className="vf-compare-panel">
      <div className="vf-compare-board">
        {Object.entries(asObject(summary.issue_counts_by_model)).map(([modelName, count]) => (
          <div key={modelName}>
            <span>{modelName}</span>
            <span className="vf-bold">{count}</span>
          </div>
        ))}
      </div>
      <div className="vf-exclusive-board">
        {Object.entries(exclusiveByModel).map(([modelName, claimIds]) => (
          <div key={modelName}>
            <span className="vf-bold">{modelName} 단독</span>
            <ChipList items={asArray(claimIds)} />
          </div>
        ))}
      </div>
      <div className="vf-compare-list">
        {rows.length ? rows.map((item, index) => {
          // compare list 전용 간단한 뷰
          const isIssue = item.issue || (Object.values(asObject(item.comparison?.models)).some(m => m?.has_issue))
          if (!isIssue) return null
          
          return (
            <article key={item.claim_id || index} className="vf-compare-row">
              <div>
                <span className="vf-bold">{compactText(item.claim_id || `CL${index + 1}`)}</span>
                <span className="vf-compare-row-status">{agreementLabel(item.comparison?.agreement?.status)}</span>
              </div>
              <p>{compactText(item.comparison?.resolved_claim || item.comparison?.claim_text || item.claim_id)}</p>
              <ModelDecisionStrip models={item.comparison?.models} />
            </article>
          )
        }) : <EmptyFiltered activeFilter={activeFilter} />}
      </div>
    </div>
  )
}

// --- Stage 2 전용 로직 ---

export function issueJudgeFailedClaimCount(model) {
  return model.issueComparisonRows.filter(item => item.agreement?.status === 'all_models_failed').length
}

export function issueJudgeClaimCheckMetrics(model, isReady) {
  return [
    { label: 'all_models_agreed', value: isReady ? model.issueJudgeSummary.all_models_agreed_count ?? '-' : '-' },
    { label: 'partial_agreement', value: isReady ? model.issueJudgeSummary.partial_agreement_count ?? '-' : '-' },
    { label: 'single_model_only', value: isReady ? model.issueJudgeSummary.single_model_only_count ?? '-' : '-' },
    { label: 'no_issue', value: isReady ? model.issueJudgeSummary.no_issue_claim_count ?? '-' : '-' },
    { label: 'all_models_failed', value: isReady ? issueJudgeFailedClaimCount(model) : '-' },
  ]
}

export function issueJudgeModelCount(row) {
  const issueModels = asArray(row?.comparison?.agreement?.issue_models)
  if (issueModels.length) return Array.from(new Set(issueModels.map(m => String(m)))).length

  const detectedModels = asArray(row?.issue?.detected_by_models)
  if (detectedModels.length) return Array.from(new Set(detectedModels.map(m => String(m)))).length

  const sourceModels = asArray(row?.issue?.source_model_issues)
    .map(item => item?.model || item?.source_model || item?.resolved_model)
    .filter(Boolean)
  if (sourceModels.length) return Array.from(new Set(sourceModels.map(m => String(m)))).length

  return Object.values(asObject(row?.comparison?.models)).filter(item => item?.has_issue).length
}
