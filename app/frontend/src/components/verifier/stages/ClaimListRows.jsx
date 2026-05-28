import { Fragment } from 'react'
import {
  asArray,
  compactText,
  claimFlowContextId,
  claimFlowSceneId,
  claimRowDomId,
  claimValueFromId,
  contextValueFromId,
  getClaimFlowSource,
  getIssueType,
  isFinalReviewTarget,
  issueTypeTone,
  sceneValueFromId,
  typeLabel,
} from '../verifierUtils'
import {
  ChipList,
  DistributionScores,
  FinalScoreSummary,
  InlineBlock,
  IssueJudgeStatusBadge,
  ModelDecisionStrip,
  ModelEvidenceAccordion,
  ModelEvidenceSection,
  TextBlock,
} from '../VerifyReportParts'

function ClaimExtractionRow({ row, displayId }) {
  const source = getClaimFlowSource(row)
  const claimText = source.resolved_claim || source.claim_text
  const originalClaimText = source.claim_text || source.resolved_claim

  return (
    <details
      id={claimRowDomId(row.claim_id)}
      className="vf-record vf-claim-flow-record vf-claim-row vf-claim-row--claim-extraction"
    >
      <summary className="vf-claim-row-summary">
        <div className="vf-claim-row-summary-inner">
          <span className="vf-bold vf-claim-row-id">{compactText(displayId || row.claim_id)}</span>
          <div className="vf-claim-row-content">
            <span className="vf-claim-row-text">{compactText(originalClaimText)}</span>
            <span className="vf-claim-row-visual-placeholder" aria-hidden="true" />
          </div>
          <span className="vf-claim-row-toggle" aria-hidden="true" />
        </div>
      </summary>
      <div className="vf-claim-accordion-body">
        <TextBlock label="추출된 주장">{compactText(claimText)}</TextBlock>
        <ChipList items={[source.is_approximate ? 'approx' : '', source.claim_fingerprint]} />
      </div>
    </details>
  )
}

function groupClaimRowsBySceneContext(rows) {
  const sceneGroups = []
  const sceneMap = new Map()

  asArray(rows).forEach(row => {
    const sceneId = claimFlowSceneId(row)
    const contextId = claimFlowContextId(row)
    if (!sceneMap.has(sceneId)) {
      const sceneGroup = { id: sceneId, contexts: [], contextMap: new Map() }
      sceneMap.set(sceneId, sceneGroup)
      sceneGroups.push(sceneGroup)
    }
    const sceneGroup = sceneMap.get(sceneId)
    if (!sceneGroup.contextMap.has(contextId)) {
      const contextGroup = { id: contextId, rows: [] }
      sceneGroup.contextMap.set(contextId, contextGroup)
      sceneGroup.contexts.push(contextGroup)
    }
    sceneGroup.contextMap.get(contextId).rows.push(row)
  })

  return sceneGroups
}

export function renderClaimExtractionRows(rows) {
  const groups = groupClaimRowsBySceneContext(rows)
  return (
    <div className="vf-claim-table">
      <div className="vf-claim-table-head">장면</div>
      <div className="vf-claim-table-head">발화 문맥</div>
      <div className="vf-claim-table-head">주장</div>
      {groups.map((scene, sceneIndex) => (
        <Fragment key={scene.id}>
          <div
            className="vf-claim-table-cell vf-claim-table-cell--scene"
            style={{ gridRow: `span ${Math.max(1, scene.contexts.reduce((total, context) => total + context.rows.length, 0))}` }}
          >
            {sceneValueFromId(scene.id, sceneIndex)}
          </div>
          {scene.contexts.map((context, contextIndex) => (
            <Fragment key={context.id}>
              <div
                className="vf-claim-table-cell vf-claim-table-cell--context"
                style={{ gridRow: `span ${Math.max(1, context.rows.length)}` }}
              >
                {contextValueFromId(context.id, contextIndex)}
              </div>
              {context.rows.map((row, rowIndex) => (
                <div key={row.claim_id} className="vf-claim-table-record">
                  <ClaimExtractionRow
                    row={row}
                    displayId={claimValueFromId(row.claim_id, rowIndex)}
                  />
                </div>
              ))}
            </Fragment>
          ))}
        </Fragment>
      ))}
    </div>
  )
}

function IssueJudgeRow({ row, displayId }) {
  const source = getClaimFlowSource(row)
  const claimText = source.resolved_claim || source.claim_text

  return (
    <details
      id={claimRowDomId(row.claim_id)}
      className="vf-record vf-claim-flow-record vf-claim-row vf-claim-row--issue-judge"
    >
      <summary className="vf-claim-row-summary">
        <div className="vf-claim-row-summary-inner">
          <span className="vf-bold vf-claim-row-id vf-claim-row-id--issue">
            {compactText(displayId || row.claim_id)}
          </span>
          <div className="vf-claim-row-content">
            <span className="vf-claim-row-text">{compactText(claimText)}</span>
            <IssueJudgeStatusBadge row={row} />
          </div>
          <span className="vf-claim-row-toggle" aria-hidden="true" />
        </div>
      </summary>
      <div className="vf-claim-accordion-body">
        <InlineBlock label="판단 결과">
          {row.comparison && <ModelDecisionStrip models={row.comparison.models} />}
        </InlineBlock>
        <ModelEvidenceSection items={row.issue?.source_model_issues} />
      </div>
    </details>
  )
}

export function renderIssueJudgeRows(rows) {
  return (
    <div className="vf-record-list">
      {rows.map((row, rowIndex) => (
        <IssueJudgeRow
          key={row.claim_id}
          row={row}
          displayId={claimValueFromId(row.claim_id, rowIndex)}
        />
      ))}
    </div>
  )
}

function IssueClassificationRow({ row, displayId }) {
  const source = getClaimFlowSource(row)
  const type = row.type || row.severity
  const claimText = source.resolved_claim || source.claim_text
  const issueType = getIssueType(type)
  const metaItems = [
    { value: typeLabel(issueType), tone: issueTypeTone(issueType) },
  ]

  return (
    <details
      id={claimRowDomId(row.claim_id)}
      className="vf-record vf-claim-flow-record vf-claim-row vf-claim-row--issue-classification"
    >
      <summary className="vf-claim-row-summary">
        <div className="vf-claim-row-summary-inner">
          <span className="vf-bold vf-claim-row-id">{compactText(displayId || row.claim_id)}</span>
          <div className="vf-claim-row-content">
            <span className="vf-claim-row-text">{compactText(claimText)}</span>
            <div className="vf-claim-row-meta">
              {metaItems.map((item, idx) => (
                <span key={idx} className={`vf-claim-row-meta-item ${item.tone ? `vf-claim-row-meta-item--${item.tone}` : ''}`}>
                  <span className="vf-bold">{item.value}</span>
                </span>
              ))}
            </div>
          </div>
          <span className="vf-claim-row-toggle" aria-hidden="true" />
        </div>
      </summary>
      <div className="vf-claim-accordion-body">
        {type && (
          <>
            <span className="vf-score-section-label">유형별 점수</span>
            <DistributionScores scores={type.weighted_scores} valueFormat="unit" />
            <ModelEvidenceAccordion items={type.model_classifications} valueFormat="unit" />
          </>
        )}
      </div>
    </details>
  )
}

export function renderIssueClassificationRows(rows) {
  return (
    <div className="vf-record-list">
      {rows.map((row, rowIndex) => (
        <IssueClassificationRow
          key={row.claim_id}
          row={row}
          displayId={claimValueFromId(row.claim_id, rowIndex)}
        />
      ))}
    </div>
  )
}

function FinalVerificationRow({ row, displayId }) {
  const source = getClaimFlowSource(row)
  const severity = row.severity
  const claimText = source.resolved_claim || source.claim_text
  const isOkRow = !isFinalReviewTarget(severity)

  return (
    <details
      id={claimRowDomId(row.claim_id)}
      className="vf-record vf-claim-flow-record vf-claim-row vf-claim-row--final-verification"
    >
      <summary className="vf-claim-row-summary">
        <div className="vf-claim-row-summary-inner">
          <span className="vf-bold vf-claim-row-id">{compactText(displayId || row.claim_id)}</span>
          <div className="vf-claim-row-content">
            <span className={`vf-claim-row-text ${isOkRow ? 'vf-claim-row-text--ok' : ''}`}>
              {compactText(claimText)}
            </span>
            <IssueJudgeStatusBadge row={row} severity={severity} />
          </div>
          <span className="vf-claim-row-toggle" aria-hidden="true" />
        </div>
      </summary>
      <div className="vf-claim-accordion-body">
        {severity ? (
          <>
            <FinalScoreSummary severity={severity} />
            <ModelEvidenceAccordion items={severity.model_judgments} valueFormat="unit" />
          </>
        ) : (
          <div className="vf-report-note">최종 평가 데이터가 없습니다.</div>
        )}
      </div>
    </details>
  )
}

export function renderFinalVerificationRows(rows) {
  return (
    <div className="vf-record-list">
      {rows.map((row, rowIndex) => (
        <FinalVerificationRow
          key={row.claim_id}
          row={row}
          displayId={claimValueFromId(row.claim_id, rowIndex)}
        />
      ))}
    </div>
  )
}

export function FinalReviewConditionTooltip() {
  return (
    <span className="vf-final-condition-tooltip">
      <span>강의자 검토가 필요한 경우</span>
      <span>문제 기준 초과: 최종 점수 &gt; 0.20</span>
      <span>모델 의견 불합치: 모델 불일치 &gt;= 0.35</span>
      <span>분류 모호함: 유형 margin &lt; 0.10</span>
    </span>
  )
}
