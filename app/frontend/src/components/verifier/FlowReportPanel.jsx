import { useEffect, useRef, useState } from 'react'
import {
  asArray,
  asObject,
  compactText,
  finalReviewReasonFlags,
  formatUnitValue,
  getIssueType,
  isFinalReviewTarget,
  typeLabel,
  uniqueTexts,
  hasDataForTab,
} from './verifierUtils'

const ISSUE_TYPE_KEYS = [
  'factual_error',
  'temporal_error',
  'scope_overclaim',
  'confusing_explanation',
]

const UNKNOWN_MODEL_LABEL = '알 수 없음'

function issueJudgeFailedClaimCount(model) {
  return model.issueComparisonRows.filter(item => item.agreement?.status === 'all_models_failed').length
}

function issueJudgeClaimCheckMetrics(model, isReady) {
  return [
    { label: 'all_models_agreed', value: isReady ? model.issueJudgeSummary.all_models_agreed_count ?? '-' : '-' },
    { label: 'partial_agreement', value: isReady ? model.issueJudgeSummary.partial_agreement_count ?? '-' : '-' },
    { label: 'single_model_only', value: isReady ? model.issueJudgeSummary.single_model_only_count ?? '-' : '-' },
    { label: 'no_issue', value: isReady ? model.issueJudgeSummary.no_issue_claim_count ?? '-' : '-' },
    { label: 'all_models_failed', value: isReady ? issueJudgeFailedClaimCount(model) : '-' },
  ]
}

function finalProblemThresholdCount(items) {
  return asArray(items).filter(item => finalReviewReasonFlags(item).problemThreshold).length
}

function finalModelDisagreementCount(items) {
  return asArray(items).filter(item => finalReviewReasonFlags(item).modelDisagreement).length
}

function finalLowMarginCount(items) {
  return asArray(items).filter(item => finalReviewReasonFlags(item).lowMargin).length
}

function issueTypeCountRows(issuesByType, isReady) {
  return ISSUE_TYPE_KEYS.map(type => ({
    label: typeLabel(type),
    value: isReady ? asArray(asObject(issuesByType)[type]).length : '-',
    type,
  }))
}

function finalIssueTypeCountRows(model, isReady) {
  const counts = new Map(ISSUE_TYPE_KEYS.map(type => [type, 0]))
  if (isReady) {
    model.severityItems
      .filter(item => isFinalReviewTarget(item))
      .forEach(item => {
        const type = getIssueType(item)
        if (counts.has(type)) counts.set(type, counts.get(type) + 1)
      })
  }
  return ISSUE_TYPE_KEYS.map(type => ({
    label: typeLabel(type),
    value: isReady ? counts.get(type) || 0 : '-',
    type,
  }))
}

function finalReviewNeededCount(model) {
  if (model.severityItems.length) {
    return model.severityItems.filter(item => isFinalReviewTarget(item)).length
  }

  const combinedStatusCount = Number(model.counts.confirmed || 0) + Number(model.counts.review || 0)
  if (combinedStatusCount > 0) return combinedStatusCount

  const summaryRaw = asObject(model.issueVerifier.summary).needs_manual_review_count
  const summaryValue = Number(summaryRaw)
  return Number.isFinite(summaryValue) ? summaryValue : 0
}

function modelName(item, fallback = '') {
  const model = asObject(item)
  return compactText(model.resolved_model || model.model || model.provider || fallback, '')
}

function stageModelList(model, tabKey) {
  if (tabKey === 'claim_extraction') {
    return uniqueTexts(model.claimExtractionModels)
  }
  if (tabKey === 'issue_judge') {
    const issueCounts = asObject(model.issueJudgeSummary.issue_counts_by_model)
    const summaryModels = Object.keys(issueCounts)
    return uniqueTexts(summaryModels.length ? summaryModels : asArray(model.issueJudge.models))
  }
  if (tabKey === 'issue_classification') {
    const weightModels = Object.keys(asObject(model.issueTypes.model_weights))
    if (weightModels.length) return uniqueTexts(weightModels)
    const breakdown = asObject(asObject(model.issueTypes.summary || model.classifiedIssues.summary).model_breakdown_by_type)
    const breakdownModels = Object.entries(breakdown).map(([fallback, item]) => modelName(item, fallback))
    const rowModels = asArray(model.issueTypeRows[0]?.model_classifications).map(item => modelName(item))
    return uniqueTexts(breakdownModels.length ? breakdownModels : rowModels)
  }
  if (tabKey === 'final_verification') {
    const weightModels = uniqueTexts([
      ...Object.keys(asObject(model.finalVerifierModelWeights)),
      ...Object.keys(asObject(model.issueVerifier.model_weights)),
    ])
    if (weightModels.length) return weightModels
    return uniqueTexts(asArray(model.severityItems[0]?.model_judgments).map(item => modelName(item)))
  }
  return []
}

function modelWeightRows(weights, fallbackModels = []) {
  const configuredWeights = asObject(weights)
  const rows = Object.entries(configuredWeights).map(([label, value]) => ({
    label,
    value: Number.isFinite(Number(value)) ? formatUnitValue(value) : value,
  }))
  if (rows.length) return rows
  return uniqueTexts(fallbackModels).map(label => ({ label, value: '' }))
}

function modelCountValue(count, isReady) {
  return isReady ? `${Math.max(0, Number(count) || 0)}개` : '-'
}

function FlowLabel({ label, labelCount }) {
  if (labelCount === undefined) return compactText(label)

  return (
    <span className="vf-flow-label-group">
      <span className="vf-flow-label-text">{compactText(label)}</span>
      <span className="vf-bold">{compactText(labelCount)}</span>
    </span>
  )
}

function FlowDetailRows({ rows, indent = false }) {
  const visibleRows = asArray(rows).filter(item => compactText(item?.label, '') || item?.value !== undefined)
  if (!visibleRows.length) return null

  return (
    <>
      {visibleRows.map(item => (
        <div
          key={item.key || item.label}
          className={`vf-flow-detail-row ${indent ? 'vf-flow-detail-row--indent' : ''} ${item.compact ? 'vf-flow-detail-row--compact' : ''}`}
        >
          <div className="vf-flow-detail-label">
            <FlowLabel label={item.label} labelCount={item.labelCount} />
          </div>
          {item.value !== undefined ? <span className="vf-bold">{compactText(item.value)}</span> : null}
          {item.aside ? <div className="vf-flow-detail-aside">{compactText(item.aside)}</div> : null}
        </div>
      ))}
    </>
  )
}

function FlowDetailDivider({ indent = false }) {
  return <div className={`vf-flow-detail-divider ${indent ? 'vf-flow-detail-divider--indent' : ''}`} aria-hidden="true" />
}

function FlowBranchRows({ rows }) {
  const visibleRows = asArray(rows)
  if (!visibleRows.length) return null

  return (
    <>
      {visibleRows.map(row => (
        <div
          key={row.type || row.label}
          className="vf-flow-summary-branch-item"
        >
          <div>{row.label}</div>
          <span className="vf-bold">{row.value}</span>
        </div>
      ))}
    </>
  )
}

function FlowDetailExpansion({ children, detail, className = '' }) {
  const hasDetail = Boolean(detail)
  const [isOpen, setIsOpen] = useState(false)
  const rootRef = useRef(null)

  useEffect(() => {
    if (!isOpen) return undefined

    function closeOnOutsidePointer(event) {
      const rootNode = rootRef.current
      if (rootNode?.contains(event.target)) return
      setIsOpen(false)
    }

    function closeOnEscape(event) {
      if (event.key === 'Escape') setIsOpen(false)
    }

    document.addEventListener('pointerdown', closeOnOutsidePointer)
    document.addEventListener('keydown', closeOnEscape)
    return () => {
      document.removeEventListener('pointerdown', closeOnOutsidePointer)
      document.removeEventListener('keydown', closeOnEscape)
    }
  }, [isOpen])

  function toggleDetail(event) {
    event.stopPropagation()
    if (!hasDetail) return
    setIsOpen(prev => !prev)
  }

  const detailContent = hasDetail && isOpen ? detail : null

  return (
    <div
      ref={rootRef}
      className={`${className} ${isOpen ? `${className}--open` : ''}`}
    >
      {children({ isOpen, toggleDetail, detailContent })}
    </div>
  )
}

function StageSummaryGrid({ model }) {
  const claimReady = hasDataForTab(model, 'claim_extraction')
  const issueReady = hasDataForTab(model, 'issue_judge')
  const typeReady = hasDataForTab(model, 'issue_classification')
  const finalReady = hasDataForTab(model, 'final_verification')
  const issueTypeRows = issueTypeCountRows(model.issuesByType, typeReady)
  const finalTypeRows = finalIssueTypeCountRows(model, finalReady)
  const finalCount = finalReady ? finalReviewNeededCount(model) : '-'
  const claimDetail = <ClaimExtractionDetailPanel model={model} isReady={claimReady} />
  const issueDetail = <IssueJudgeDetailPanel model={model} isReady={issueReady} />
  const typeDetail = <IssueTypeDetailPanel model={model} isReady={typeReady} typeRows={issueTypeRows} />
  const finalDetail = <FinalVerificationDetailPanel model={model} isReady={finalReady} />

  return (
    <section className="vf-flow-summary-block" aria-label="단계별 요약">
      <div className="vf-flow-summary-content">
        <div className="vf-flow-summary-main">
          <div className="vf-flow-summary-column">
            <div className="vf-flow-summary-block-item">
              <FlowDetailExpansion className="vf-flow-detail-trigger" detail={claimDetail}>
                {({ isOpen, toggleDetail, detailContent }) => (
                  <>
                    <div className="vf-flow-summary-column-head-row">
                      <div className="vf-flow-summary-column-head">주장 추출</div>
                      <button type="button" className="vf-flow-summary-detail-hint" aria-expanded={isOpen} aria-label="주장 추출 상세정보" onClick={toggleDetail}>
                        상세
                      </button>
                    </div>
                    <div className="vf-flow-summary-node">
                      <div className="vf-flow-summary-frame">
                        <div className="vf-flow-summary-list">
                          <div className="vf-flow-summary-node-row">
                            <div className="vf-flow-summary-node-label">claims</div>
                            <span className="vf-bold">{claimReady ? model.claims.length : '-'}</span>
                          </div>
                        </div>
                      </div>
                      {isOpen ? (
                        <div className="vf-flow-summary-overlay">
                          <div className="vf-flow-summary-list">
                            <div className="vf-flow-summary-node-row">
                              <div className="vf-flow-summary-node-label">claims</div>
                              <span className="vf-bold">{claimReady ? model.claims.length : '-'}</span>
                            </div>
                            {detailContent}
                          </div>
                        </div>
                      ) : null}
                    </div>
                  </>
                )}
              </FlowDetailExpansion>
            </div>
          </div>
          <div className="vf-flow-summary-arrow" aria-hidden="true" />
          <div className="vf-flow-summary-column">
            <div className="vf-flow-summary-block-item">
              <FlowDetailExpansion className="vf-flow-detail-trigger" detail={issueDetail}>
                {({ isOpen, toggleDetail, detailContent }) => (
                  <>
                    <div className="vf-flow-summary-column-head-row">
                      <div className="vf-flow-summary-column-head">이슈 후보 판단</div>
                      <button type="button" className="vf-flow-summary-detail-hint" aria-expanded={isOpen} aria-label="이슈 후보 판단 상세정보" onClick={toggleDetail}>
                        상세
                      </button>
                    </div>
                    <div className="vf-flow-summary-node">
                      <div className="vf-flow-summary-frame">
                        <div className="vf-flow-summary-list">
                          <div className="vf-flow-summary-node-row">
                            <div className="vf-flow-summary-node-label">issues</div>
                            <span className="vf-bold">{issueReady ? model.issueCandidates.length : '-'}</span>
                          </div>
                        </div>
                      </div>
                      {isOpen ? (
                        <div className="vf-flow-summary-overlay">
                          <div className="vf-flow-summary-list">
                            <div className="vf-flow-summary-node-row">
                              <div className="vf-flow-summary-node-label">issues</div>
                              <span className="vf-bold">{issueReady ? model.issueCandidates.length : '-'}</span>
                            </div>
                            {detailContent}
                          </div>
                        </div>
                      ) : null}
                    </div>
                  </>
                )}
              </FlowDetailExpansion>
            </div>
            <div className="vf-flow-summary-arrow" aria-hidden="true" />
            <div className="vf-flow-summary-block-item">
              <FlowDetailExpansion className="vf-flow-detail-trigger" detail={typeDetail}>
                {({ isOpen, toggleDetail, detailContent }) => (
                  <>
                    <div className="vf-flow-summary-column-head-row">
                      <div className="vf-flow-summary-column-head">이슈 유형 분류</div>
                      <button type="button" className="vf-flow-summary-detail-hint" aria-expanded={isOpen} aria-label="이슈 유형 분류 상세정보" onClick={toggleDetail}>
                        상세
                      </button>
                    </div>
                    <div className="vf-flow-summary-branch">
                      <div className="vf-flow-summary-frame">
                        <div className="vf-flow-summary-list">
                          <FlowBranchRows rows={issueTypeRows} />
                        </div>
                      </div>
                      {isOpen ? (
                        <div className="vf-flow-summary-overlay">
                          <div className="vf-flow-summary-list">
                            {detailContent}
                          </div>
                        </div>
                      ) : null}
                    </div>
                  </>
                )}
              </FlowDetailExpansion>
            </div>
          </div>
          <div className="vf-flow-summary-arrow" aria-hidden="true" />
          <div className="vf-flow-summary-column">
            <div className="vf-flow-summary-block-item">
              <FlowDetailExpansion className="vf-flow-detail-trigger" detail={finalDetail}>
                {({ isOpen, toggleDetail, detailContent }) => (
                  <>
                    <div className="vf-flow-summary-column-head-row">
                      <div className="vf-flow-summary-column-head">최종 평가</div>
                      <button type="button" className="vf-flow-summary-detail-hint" aria-expanded={isOpen} aria-label="최종 평가 상세정보" onClick={toggleDetail}>
                        상세
                      </button>
                    </div>
                    <div className="vf-flow-summary-node">
                      <div className="vf-flow-summary-frame">
                        <div className="vf-flow-summary-list">
                          <div className="vf-flow-summary-node-row">
                            <div className="vf-flow-summary-node-label">final</div>
                            <span className="vf-bold">{finalCount}</span>
                          </div>
                        </div>
                      </div>
                      {isOpen ? (
                        <div className="vf-flow-summary-overlay">
                          <div className="vf-flow-summary-list">
                            <div className="vf-flow-summary-node-row">
                              <div className="vf-flow-summary-node-label">final</div>
                              <span className="vf-bold">{finalCount}</span>
                            </div>
                            {detailContent}
                          </div>
                        </div>
                      ) : null}
                    </div>
                    <div className="vf-flow-summary-branch">
                      <div className="vf-flow-summary-frame">
                        <div className="vf-flow-summary-list">
                          <FlowBranchRows rows={finalTypeRows} />
                        </div>
                      </div>
                    </div>
                  </>
                )}
              </FlowDetailExpansion>
            </div>
            <div className="vf-flow-summary-block-item">
              <div className="vf-flow-detail-trigger">
                <div className="vf-flow-summary-column-head-row">
                  <div className="vf-flow-summary-column-head" aria-hidden="true">&nbsp;</div>
                </div>
                <div className="vf-flow-summary-branch">
                  <div className="vf-flow-summary-frame">
                    <div className="vf-flow-summary-list">
                      <FlowBranchRows rows={finalTypeRows} />
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}

function ClaimExtractionDetailPanel({ model, isReady }) {
  const resolvedClaims = model.claims.filter(claim => claim.resolution_status === 'resolved').length
  const contextNeeded = model.claims.filter(claim => claim.needs_context).length
  const approximate = model.claims.filter(claim => claim.is_approximate).length
  const claimModel = stageModelList(model, 'claim_extraction')[0] || UNKNOWN_MODEL_LABEL
  const claimTypes = model.claims.reduce((counts, claim) => {
    const key = compactText(claim.claim_type || 'claim')
    counts[key] = (counts[key] || 0) + 1
    return counts
  }, {})
  const claimTypeRows = Object.entries(claimTypes).map(([label, value]) => ({ label: typeLabel(label), value }))

  return (
    <>
      <FlowDetailDivider />
      <FlowDetailRows
        indent
        rows={[
          { label: 'resolved', value: isReady ? resolvedClaims : '-' },
          { label: 'needs_context', value: isReady ? contextNeeded : '-' },
        ]}
      />
      <FlowDetailDivider indent />
      <FlowDetailRows indent rows={[{ label: 'approx', value: isReady ? approximate : '-' }]} />
      <FlowDetailDivider indent />
      <FlowDetailRows indent rows={isReady ? claimTypeRows : [{ label: 'claim_type', value: '-' }]} />
      <FlowDetailDivider />
      <FlowDetailRows rows={[{ label: '사용 모델', value: isReady ? claimModel : '-' }]} />
    </>
  )
}

function IssueJudgeDetailPanel({ model, isReady }) {
  const compareSummary = asObject(model.issueJudgeCompare.summary)
  const failedModels = asArray(compareSummary.failed_models)
  const issueCountsByModel = asObject(compareSummary.issue_counts_by_model)
  const allModels = uniqueTexts([
    ...asArray(model.issueJudgeCompare.models),
    ...asArray(model.issueJudge.models),
    ...Object.keys(issueCountsByModel),
    ...failedModels,
  ])
  const fallbackModelTotal = Number(compareSummary.evaluated_model_count || 0) + failedModels.length
  const totalModelCount = allModels.length || fallbackModelTotal
  const modelRows = isReady ? allModels.map(modelName => ({
    label: modelName,
    value: failedModels.includes(modelName) ? '실패' : issueCountsByModel[modelName] ?? 0,
  })) : []

  return (
    <>
      <FlowDetailDivider />
      <FlowDetailRows indent rows={issueJudgeClaimCheckMetrics(model, isReady)} />
      <FlowDetailDivider />
      <FlowDetailRows rows={[{ label: '평가 모델', labelCount: modelCountValue(totalModelCount, isReady), compact: true }]} />
      <FlowDetailDivider />
      <FlowDetailRows indent rows={modelRows} />
    </>
  )
}

function IssueTypeDetailPanel({ model, isReady, typeRows = [] }) {
  const summary = asObject(model.issueTypes.summary || model.classifiedIssues.summary)
  const configuredWeights = {
    ...asObject(model.issueTypes.model_weights),
    ...asObject(summary.model_weights),
  }
  const fallbackModels = stageModelList(model, 'issue_classification')
  const modelRows = modelWeightRows(configuredWeights, fallbackModels)
  const modelCount = fallbackModels.length || modelRows.length

  return (
    <>
      <FlowDetailRows rows={isReady ? typeRows : [{ label: '이슈 유형', value: '-' }]} />
      <FlowDetailDivider />
      <FlowDetailRows rows={[{ label: '평가 모델', labelCount: modelCountValue(modelCount, isReady), aside: isReady ? '가중치' : '', compact: true }]} />
      <FlowDetailDivider />
      <FlowDetailRows indent rows={isReady ? modelRows : []} />
    </>
  )
}

function FinalVerificationDetailPanel({ model, isReady }) {
  const problemThresholdCount = finalProblemThresholdCount(model.severityItems)
  const modelDisagreementCount = finalModelDisagreementCount(model.severityItems)
  const lowMarginCount = finalLowMarginCount(model.severityItems)
  const configuredWeights = {
    ...asObject(model.finalVerifierModelWeights),
    ...asObject(model.issueVerifier.model_weights),
    ...asObject(model.issueVerifier.summary?.model_weights),
  }
  const fallbackModels = stageModelList(model, 'final_verification')
  const modelRows = modelWeightRows(configuredWeights, fallbackModels)
  const modelCount = fallbackModels.length || modelRows.length

  return (
    <>
      <FlowDetailDivider />
      <FlowDetailRows
        indent
        rows={[
          { label: '문제 기준 초과', value: isReady ? problemThresholdCount : '-' },
          { label: '모델 의견 불합치', value: isReady ? modelDisagreementCount : '-' },
          { label: '분류 모호함', value: isReady ? lowMarginCount : '-' },
        ]}
      />
      <FlowDetailDivider />
      <FlowDetailRows rows={[{ label: '평가 모델', labelCount: modelCountValue(modelCount, isReady), aside: isReady ? '가중치' : '', compact: true }]} />
      <FlowDetailDivider />
      <FlowDetailRows indent rows={isReady ? modelRows : []} />
    </>
  )
}

export default function FlowReportPanel({ model }) {
  return (
    <section className="vf-overview" aria-label="흐름 보고서">
      <div className="vf-stage-data-head">
        <span className="vf-bold">흐름 보고서</span>
      </div>
      <StageSummaryGrid model={model} />
    </section>
  )
}
