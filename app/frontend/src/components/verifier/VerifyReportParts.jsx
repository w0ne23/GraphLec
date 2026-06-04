import { useLayoutEffect, useRef, useState } from 'react'
import {
  asArray,
  asObject,
  compactText,
  finalReviewReasonLabels,
  formatScore,
  formatUnitValue,
  getScore,
  hasIssueInComparison,
  hideMissingImage,
  resultFileUrl,
  typeLabel,
  uniqueTexts,
} from './verifierUtils'

const SCORE_LABELS = {
  valid: '유효성',
  severity: '심각도',
  context: '문맥 해소',
  disagreement: '모델 불일치',
  final: '최종 점수',
}

const DISTRIBUTION_COLORS = ['var(--chart-blue)', 'var(--chart-green)', 'var(--chart-amber)', 'var(--chart-red)']
const ISSUE_TYPE_SCORE_KEYS = ['factual_error', 'temporal_error', 'confusing_explanation', 'scope_overclaim']

const ROW_META_TONE_CLASS = {
  warn: 'vf-claim-row-meta-item--warn',
  'type-1': 'vf-claim-row-meta-item--type-1',
  'type-2': 'vf-claim-row-meta-item--type-2',
  'type-3': 'vf-claim-row-meta-item--type-3',
  'type-4': 'vf-claim-row-meta-item--type-4',
}

const ISSUE_STATUS_CLASS = {
  issue: 'vf-issue-judge-status--issue',
  review: 'vf-issue-judge-status--review',
}

function cx(...classes) {
  return classes.filter(Boolean).join(' ')
}

function scoreLabel(type) {
  return SCORE_LABELS[type] || typeLabel(type)
}

function issueResultLabel(value) {
  const text = compactText(value, '')
  return text ? typeLabel(text) : ''
}

function issueJudgeModelCount(row) {
  const issueModels = asArray(row?.comparison?.agreement?.issue_models)
  if (issueModels.length) return uniqueTexts(issueModels).length

  const detectedModels = asArray(row?.issue?.detected_by_models)
  if (detectedModels.length) return uniqueTexts(detectedModels).length

  const sourceModels = asArray(row?.issue?.source_model_issues)
    .map(item => item?.model || item?.source_model || item?.resolved_model)
    .filter(Boolean)
  if (sourceModels.length) return uniqueTexts(sourceModels).length

  return Object.values(asObject(row?.comparison?.models)).filter(item => item?.has_issue).length
}

export function RowMeta({ items }) {
  const visibleItems = items.filter(item => compactText(item.value, ''))
  if (!visibleItems.length) return null

  return (
    <div className="vf-claim-row-meta">
      {visibleItems.map(item => (
        <span
          key={`${item.label || 'flag'}-${item.value}`}
          className={cx('vf-claim-row-meta-item', ROW_META_TONE_CLASS[item.tone])}
        >
          {item.label && <span className="vf-claim-row-meta-label">{item.label}</span>}
          <span className="vf-bold">{item.value}</span>
        </span>
      ))}
    </div>
  )
}

export function IssueJudgeStatusBadge({ row, severity }) {
  const hasFinalDecision = Boolean(severity)
  const finalReasonLabels = hasFinalDecision ? finalReviewReasonLabels(severity) : []
  const needsReview = hasFinalDecision
    ? finalReasonLabels.length > 0
    : Boolean(severity?.needs_manual_review || severity?.status === 'professor_check' || severity?.status === 'review_needed')
  const hasIssue = hasFinalDecision ? finalReasonLabels.length > 0 : Boolean(row?.issue || hasIssueInComparison(row))
  const modelCount = !hasFinalDecision ? issueJudgeModelCount(row) : 0
  const showModelCount = !hasFinalDecision && hasIssue && modelCount > 0
  const showFinalReasons = finalReasonLabels.length > 0
  const reserveModelCount = !hasFinalDecision
  if (!hasIssue) {
    return (
      <span className={cx('vf-issue-judge-status', 'vf-issue-judge-status--empty', reserveModelCount && 'vf-issue-judge-status--with-count')} aria-hidden="true">
        <span className="vf-issue-judge-status-count" />
        <span className="vf-issue-judge-status-dot" />
      </span>
    )
  }
  const label = needsReview ? 'review needed' : 'issue'
  const tone = needsReview ? 'review' : 'issue'

  return (
    <span className={cx('vf-issue-judge-status', ISSUE_STATUS_CLASS[tone], showModelCount && 'vf-issue-judge-status--with-count', showFinalReasons && 'vf-issue-judge-status--with-reasons')} aria-label={label}>
      {showModelCount && <span className="vf-issue-judge-status-count" aria-label={`${modelCount} models`}>{modelCount}</span>}
      {showFinalReasons && (
        <span className="vf-final-review-reasons">
          {finalReasonLabels.map(reason => (
            <span key={reason} className="vf-final-review-reason-chip">{reason}</span>
          ))}
        </span>
      )}
      <span className="vf-issue-judge-status-dot" aria-hidden="true" />
    </span>
  )
}

export function EmptyFiltered({ activeFilter }) {
  return (
    <div className="vf-report-empty">
      {activeFilter ? `${activeFilter.label} 조건에 해당하는 항목이 없습니다.` : '표시할 항목이 없습니다.'}
    </div>
  )
}

export function MetricStrip({ items }) {
  return (
    <div className="vf-report-metrics">
      {items.map(item => {
        const subItems = asArray(item.subItems).filter(subItem => compactText(subItem.value, ''))
        if (subItems.length) {
          return (
            <div key={item.label} className="vf-report-metric vf-report-metric--stack">
              <div className="vf-report-metric-main">
                <span>{item.label}</span>
                <span className="vf-bold">{item.value}</span>
              </div>
              <div className="vf-report-metric-subrows">
                {subItems.map(subItem => (
                  <span
                    key={subItem.label}
                    className={subItem.divider ? 'vf-report-metric-subrow--divider' : ''}
                  >
                    <span className="vf-report-metric-subrow-label">{subItem.label}</span>
                    <b>{subItem.value}</b>
                  </span>
                ))}
              </div>
            </div>
          )
        }

        return (
          <div key={item.label} className="vf-report-metric">
            <span>{item.label}</span>
            <span className="vf-bold">{item.value}</span>
          </div>
        )
      })}
    </div>
  )
}

export function ChipList({ items }) {
  const chips = items.map(item => compactText(item, '')).filter(Boolean)
  if (!chips.length) return null
  return (
    <div className="vf-chip-list" data-chip-list="true">
      {chips.map(item => <span key={item}>{item}</span>)}
    </div>
  )
}

export function TextBlock({ label, children }) {
  if (!compactText(children, '')) return null
  return (
    <div className="vf-text-block" data-text-block="true">
      <span>{label}</span>
      <p>{children}</p>
    </div>
  )
}

export function InlineBlock({ label, children }) {
  if (!children) return null
  return (
    <div className="vf-text-block vf-text-block--inline" data-text-block="true" data-text-block-inline="true">
      <span>{label}</span>
      {children}
    </div>
  )
}

export function ModelEvidenceSection({ items, valueFormat, title = '모델별 판단' }) {
  if (!asArray(items).length) return null
  return (
    <div className="vf-model-evidence-section" data-model-evidence-section="true">
      <span>{title}</span>
      <ModelEvidence items={items} valueFormat={valueFormat} />
    </div>
  )
}

function ScoreBars({ scores, valueFormat = 'percent' }) {
  const entries = Object.entries(scores || {}).filter(([, value]) => Number.isFinite(Number(value)))
  if (!entries.length) return null
  return (
    <div className="vf-score-list">
      {entries.map(([label, value]) => {
        const score = Math.max(0, Math.min(1, Number(value)))
        return (
          <div key={label} className="vf-score-item">
            <span>{scoreLabel(label)}</span>
            <span className="vf-bold">{valueFormat === 'unit' ? formatUnitValue(score) : formatScore(score)}</span>
          </div>
        )
      })}
    </div>
  )
}

export function FinalScoreSummary({ severity }) {
  const verifier = asObject(severity?.classified_issue_verifier)
  const metrics = [
    ['valid', verifier.average_is_valid_issue ?? severity?.average_is_valid_issue],
    ['severity', verifier.average_category_severity ?? severity?.average_category_severity],
    ['context', verifier.average_context_resolution ?? severity?.average_context_resolution],
  ].filter(([, value]) => Number.isFinite(Number(value)))
  const disagreement = verifier.model_disagreement ?? severity?.model_disagreement
  const finalScore = getScore(severity)
  const hasDisagreement = Number.isFinite(Number(disagreement))
  const hasFinalScore = Number.isFinite(Number(finalScore))
  if (!metrics.length && !hasDisagreement && !hasFinalScore) return null

  return (
    <div className="vf-final-score-summary">
      <div className="vf-final-score-formula">
        <div className="vf-final-score-card vf-final-score-card--metrics">
          {metrics.map(([label, value]) => (
            <div key={label} className="vf-final-score-row">
              <span>{scoreLabel(label)}</span>
              <span className="vf-bold">{formatUnitValue(value)}</span>
            </div>
          ))}
        </div>
        {hasFinalScore && (
          <div className="vf-final-score-card vf-final-score-card--single">
            <span>{scoreLabel('final')}</span>
            <div>
              <span className="vf-bold">{formatUnitValue(finalScore)}</span>
            </div>
          </div>
        )}
      </div>
      {hasDisagreement && (
        <div className="vf-final-score-signal vf-final-score-card vf-final-score-card--single">
          <span>{scoreLabel('disagreement')}</span>
          <div>
            <span className="vf-bold">{formatUnitValue(disagreement)}</span>
          </div>
        </div>
      )}
    </div>
  )
}

function FinalModelScoreSummary({ item }) {
  const metrics = [
    ['valid', item?.is_valid_issue],
    ['severity', item?.category_severity],
    ['context', item?.context_resolution],
  ].filter(([, value]) => Number.isFinite(Number(value)))
  const finalScore = item?.final_model_score
  const hasFinalScore = Number.isFinite(Number(finalScore))
  if (!metrics.length && !hasFinalScore) return null

  return (
    <div className="vf-final-score-summary vf-final-score-summary--model">
      <div className="vf-final-score-formula">
        <div className="vf-final-score-card vf-final-score-card--metrics">
          {metrics.map(([label, value]) => (
            <div key={label} className="vf-final-score-row">
              <span>{scoreLabel(label)}</span>
              <span className="vf-bold">{formatUnitValue(value)}</span>
            </div>
          ))}
        </div>
        {hasFinalScore && (
          <div className="vf-final-score-card vf-final-score-card--single">
            <span>{scoreLabel('final')}</span>
            <div>
              <span className="vf-bold">{formatUnitValue(finalScore)}</span>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

export function DistributionScores({ scores, valueFormat = 'unit' }) {
  const source = asObject(scores)
  const labels = ISSUE_TYPE_SCORE_KEYS.some(key => Object.prototype.hasOwnProperty.call(source, key))
    ? ISSUE_TYPE_SCORE_KEYS
    : Object.keys(source)
  const entries = labels
    .map((label, index) => {
      const value = Number(source[label] ?? 0)
      return {
        label,
        value: Number.isFinite(value) ? Math.max(0, Math.min(1, value)) : 0,
        color: DISTRIBUTION_COLORS[index % DISTRIBUTION_COLORS.length],
      }
    })
  if (!entries.length) return null

  let cursor = 0
  const visibleEntries = entries.filter(item => item.value > 0)
  const segments = visibleEntries.map(item => {
    const start = cursor
    cursor += item.value * 100
    return `${item.color} ${start}% ${cursor}%`
  })
  if (cursor < 100) segments.push(`var(--b1) ${cursor}% 100%`)
  const chartStyle = { background: `conic-gradient(${segments.join(', ')})` }

  return (
    <div className="vf-score-distribution" data-score-distribution="true">
      <div className="vf-score-distribution-visual">
        <div className="vf-score-donut" style={chartStyle} aria-hidden="true" />
      </div>
      <div className="vf-score-distribution-items">
        {entries.map(item => (
          <div key={item.label} className="vf-score-distribution-item">
            <div className="vf-score-distribution-main">
              <div className="vf-score-distribution-bar" aria-hidden="true">
                <i
                  style={{
                    width: `${item.value * 100}%`,
                    background: item.color,
                  }}
                />
              </div>
              <span>
                <i style={{ background: item.color }} aria-hidden="true" />
                <span>{scoreLabel(item.label)}</span>
              </span>
            </div>
            <span className="vf-bold">{valueFormat === 'unit' ? formatUnitValue(item.value) : formatScore(item.value)}</span>
          </div>
        ))}
      </div>
    </div>
  )
}

export function MouseTooltip({ children, tooltip, className = '', tooltipClassName = '', tabIndex, ariaLabel, onClick, matchAnchorWidth = false, offset = 6 }) {
  const hasTooltip = Boolean(tooltip)
  const [position, setPosition] = useState(null)
  const tooltipRef = useRef(null)

  useLayoutEffect(() => {
    if (!position) return
    const tooltipNode = tooltipRef.current
    if (!tooltipNode) return

    const tooltipRect = tooltipNode.getBoundingClientRect()
    const contentRect = document.querySelector('.main-layout-content')?.getBoundingClientRect()
    const bounds = contentRect || {
      left: 0,
      top: 0,
      right: window.innerWidth || document.documentElement.clientWidth,
      bottom: window.innerHeight || document.documentElement.clientHeight,
    }
    const minLeft = bounds.left + 8
    const maxLeft = bounds.right - tooltipRect.width - 8
    const minTop = bounds.top + 8
    const maxTop = bounds.bottom - tooltipRect.height - 8
    const nextLeft = Math.min(Math.max(position.left, minLeft), Math.max(minLeft, maxLeft))
    const nextTop = Math.min(Math.max(position.top, minTop), Math.max(minTop, maxTop))

    if (
      Math.abs(nextLeft - position.left) < 0.5 &&
      Math.abs(nextTop - position.top) < 0.5
    ) return
    setPosition(prev => prev ? { ...prev, left: nextLeft, top: nextTop } : prev)
  }, [position])

  function showTooltip(event) {
    if (!hasTooltip) return
    const target = event?.currentTarget
    const anchor = target?.querySelector?.('[data-tooltip-anchor]') || target
    const rect = anchor?.getBoundingClientRect?.()
    setPosition({
      left: rect?.left ?? 0,
      top: rect?.bottom ? rect.bottom + offset : 0,
      width: matchAnchorWidth ? rect?.width : undefined,
    })
  }

  return (
    <span
      className={`${className} ${hasTooltip ? 'vf-mouse-tooltip-trigger' : ''}`}
      tabIndex={hasTooltip ? tabIndex : undefined}
      aria-label={ariaLabel}
      onPointerEnter={showTooltip}
      onPointerLeave={() => setPosition(null)}
      onFocus={showTooltip}
      onBlur={() => setPosition(null)}
      onClick={onClick}
    >
      {children}
      {hasTooltip && position ? (
        <span
          ref={tooltipRef}
          className={`vf-mouse-tooltip ${tooltipClassName}`}
          role="tooltip"
          style={{
            left: `${position.left}px`,
            top: `${position.top}px`,
            width: position.width ? `${position.width}px` : undefined,
          }}
        >
          {tooltip}
        </span>
      ) : null}
    </span>
  )
}

function ModelEvidence({ items, valueFormat }) {
  const rows = asArray(items)
  if (!rows.length) return null
  return (
    <div className="vf-model-evidence">
      {rows.map((item, index) => {
        const isTypeDistribution = Boolean(item.probabilities)
        const isFinalModelScore = !isTypeDistribution && item.final_model_score != null
        const scores = item.probabilities || {
          valid: item.is_valid_issue,
          severity: item.category_severity,
          context: item.context_resolution,
          final: item.final_model_score,
        }
        const topLabel = issueResultLabel(item.judgment || item.top_issue_type_label || item.top_issue_type || item.final_model_score)
        const confidenceLabel = item.confidence != null ? `신뢰도 ${formatScore(item.confidence)}` : ''
        return (
          <div key={`${item.model || item.provider || 'model'}-${index}`} className="vf-model-card">
            <div className="vf-model-card-head">
              <span className="vf-bold">{compactText(item.model || item.provider || item.resolved_model)}</span>
              {(topLabel || confidenceLabel) && (
                <div className="vf-model-card-result">
                  {topLabel ? <span className="vf-model-card-result-label">{topLabel}</span> : null}
                  {confidenceLabel ? <ChipList items={[confidenceLabel]} /> : null}
                </div>
              )}
            </div>
            {isTypeDistribution ? (
              <DistributionScores scores={scores} valueFormat="unit" />
            ) : isFinalModelScore ? (
              <FinalModelScoreSummary item={item} />
            ) : (
              <ScoreBars scores={scores} valueFormat={valueFormat || (item.final_model_score != null ? 'unit' : 'percent')} />
            )}
            <TextBlock label="판단 근거">{item.reason || item.candidate_reason || item.issue}</TextBlock>
            <TextBlock label="수정 제안">{item.minimal_fix}</TextBlock>
            {!isFinalModelScore && item.final_model_score != null && <ChipList items={[`모델 점수 ${formatUnitValue(item.final_model_score)}`]} />}
          </div>
        )
      })}
    </div>
  )
}

export function ModelEvidenceAccordion({ items, valueFormat, title = '모델별 결과' }) {
  if (!asArray(items).length) return null
  return (
    <details className="vf-model-evidence-accordion">
      <summary>
        <span>{title}</span>
        <i aria-hidden="true" />
      </summary>
      <ModelEvidence items={items} valueFormat={valueFormat} />
    </details>
  )
}

export function ModelDecisionStrip({ models }) {
  const entries = Object.entries(asObject(models))
  if (!entries.length) return null
  return (
    <div className="vf-model-decision-strip">
      {entries.map(([model, item]) => {
        const hasIssue = Boolean(item?.has_issue)
        return (
          <span
            key={model}
            className={`vf-model-decision ${hasIssue ? 'vf-model-decision--issue' : 'vf-model-decision--clear'}`}
          >
            <span className="vf-bold">{model}</span>
          </span>
        )
      })}
    </div>
  )
}

export function ContextPreview({ item, resultId }) {
  const judgeContext = asObject(item.judge_context)
  const slide = asObject(judgeContext.slide)
  const bundle = asObject(judgeContext.context_bundle)
  const contexts = asArray(bundle.target_contexts)
  const imageUrl = resultFileUrl(slide.image_path, resultId)
  if (!Object.keys(slide).length && !contexts.length) return null

  return (
    <div className="vf-context-panel" data-image-missing-container="true">
      {imageUrl && (
        <div className="vf-context-media" data-missing-target="true">
          <img src={imageUrl} alt={compactText(slide.title || '슬라이드 이미지')} onError={hideMissingImage} />
        </div>
      )}
      <div className="vf-context-body">
        <ChipList items={[
          slide.slide_number ? `슬라이드 ${slide.slide_number}` : '',
          slide.title,
          slide.role,
          slide.slide_type,
          slide.time_range,
        ]} />
        <TextBlock label="슬라이드 텍스트">{slide.t1 || slide.slide_text}</TextBlock>
        {contexts.map((context, index) => (
          <TextBlock key={context.context_id || index} label={context.context_id || `컨텍스트 ${index + 1}`}>
            {context.text}
          </TextBlock>
        ))}
      </div>
    </div>
  )
}
