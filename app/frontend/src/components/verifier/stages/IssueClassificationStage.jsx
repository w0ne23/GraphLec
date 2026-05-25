import { 
  asObject, 
  typeLabel, 
  DistributionScores, 
  EmptyFiltered,
  MetricStrip
} from '../VerifyReportPanels'
import ClaimFlowRecord from './shared/ClaimFlowRecord'

/**
 * Stage 3: Issue Classification
 * 감지된 이슈 후보들을 유형별(factual_error, temporal_error 등)로 분류하고 확신도를 표시합니다.
 */
export default function IssueClassificationStage({ model, rows, resultId, activeFilter, status }) {
  const isDone = status === 'done'
  const typeRows = issueTypeCountRows(model.issuesByType, isDone)

  return (
    <div className="vf-stage-classification">
      <div className="vf-type-board">
        {typeRows.map(row => (
          <div key={row.label}>
            <span className="vf-bold">{row.label}</span>
            <span>{row.value}</span>
          </div>
        ))}
      </div>

      {rows.length > 0 ? (
        <div className="vf-record-list">
          {rows.map((row, rowIndex) => (
            <ClaimFlowRecord
              key={row.claim_id}
              row={row}
              activeTab="issue_classification"
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

// --- Stage 3 전용 로직 ---

export function issueTypeCountRows(issuesByType, isReady) {
  // ISSUE_TYPE_KEYS는 메인에 상수로 존재하므로 직접 문자열 배열 사용 혹은 import 고려
  const ISSUE_TYPE_KEYS = [
    'factual_error',
    'temporal_error',
    'scope_overclaim',
    'confusing_explanation',
  ]
  
  return ISSUE_TYPE_KEYS.map(type => ({
    label: typeLabel(type),
    value: isReady ? (asObject(issuesByType)[type]?.length || 0) : '-',
    type,
  }))
}
