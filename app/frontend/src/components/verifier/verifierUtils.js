/**
 * Verifier display utilities
 * 검증 화면 전용 표시, 판정, 파일 URL, DOM 보조 함수를 담고 있습니다.
 */
import { toResultFileUrl } from './review/verifierReviewUtils'

export function asArray(value) {
  return Array.isArray(value) ? value : []
}

export function asObject(value) {
  return value && typeof value === 'object' && !Array.isArray(value) ? value : {}
}

export function compactText(value, fallback = '-') {
  const text = String(value ?? '').replace(/\s+/g, ' ').trim()
  return text || fallback
}

export function formatScore(value) {
  if (value === undefined || value === null || value === '') return '-'
  const score = Number(value)
  if (!Number.isFinite(score)) return '-'
  return score > 1 ? `${score.toFixed(1)}%` : `${(score * 100).toFixed(1)}%`
}

export function formatUnitValue(value) {
  if (value === undefined || value === null || value === '') return '-'
  const number = Number(value)
  if (!Number.isFinite(number)) return '-'
  return number.toFixed(2)
}

export function formatTime(value) {
  const seconds = Number(value)
  if (!Number.isFinite(seconds)) return ''
  const safe = Math.max(0, seconds)
  const m = Math.floor(safe / 60)
  const s = Math.floor(safe % 60)
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`
}

export function idNumber(value, prefix) {
  const escaped = String(prefix).replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
  const match = compactText(value, '').match(new RegExp(`(?:^|-)${escaped}(\\d+)(?=-|$)`, 'i'))
  if (!match) return ''
  const number = Number(match[1])
  return Number.isFinite(number) ? String(number) : match[1]
}

export function sceneIdFromIndex(value) {
  const text = compactText(value, '')
  if (!text) return ''
  const existing = text.match(/SC\d+/i)
  if (existing) return existing[0].toUpperCase()
  const number = Number(text)
  return Number.isFinite(number) ? `SC${String(number).padStart(4, '0')}` : text
}

export function typeLabel(type) {
  // ISSUE_TYPE_LABELS는 상수로 관리되거나 이 함수 내에서 직접 처리
  const labels = {
    factual_error: 'factual_error',
    temporal_error: 'temporal_error',
    scope_overclaim: 'scope_overclaim',
    confusing_explanation: 'confusing_explanation',
    unknown: 'unknown',
  }
  return labels[type] || compactText(type)
}

export function agreementLabel(type) {
  const labels = {
    all_models_agreed: 'all_models_agreed',
    single_model_only: 'single_model_only',
    partial_agreement: 'partial_agreement',
    no_issue: 'no_issue',
    all_models_failed: 'all_models_failed',
  }
  return labels[type] || compactText(type)
}

export function statusLabel(status) {
  if (status === 'confirmed') return '확정'
  if (status === 'professor_check' || status === 'review_needed') return '검토'
  if (status === 'rejected') return '기각'
  return compactText(status)
}

export function firstFilled(...values) {
  return values.find(value => String(value ?? '').trim()) ?? ''
}

export function locationText(item) {
  const location = asObject(item?.location)
  const slide = item?.slide_number ?? location.slide_number
  const start = item?.start_time ?? location.start_time
  const end = item?.end_time ?? location.end_time
  const parts = []
  if (slide !== undefined && slide !== null && slide !== '') parts.push(`슬라이드 ${slide}`)
  if (start !== undefined && start !== null && start !== '') {
    const range = end !== undefined && end !== null && end !== ''
      ? `${formatTime(start)}-${formatTime(end)}`
      : formatTime(start)
    if (range) parts.push(range)
  }
  return parts.join(' · ')
}

export function getIssueType(item) {
  return item?.feedback_type || item?.category || item?.final_issue_type || item?.issue_type || item?.type || ''
}

export function getScore(item) {
  return (
    item?.severity_score ??
    item?.classified_issue_verifier?.final_severity_score ??
    item?.final_severity_score ??
    item?.crosscheck_score ??
    item?.confidence
  )
}

export function statusFromSeverityScore(score) {
  const value = Number(score)
  if (!Number.isFinite(value)) return 'professor_check'
  if (value >= 0.5) return 'confirmed'
  if (value <= 0.2) return 'rejected'
  return 'professor_check'
}

export function claimKey(item) {
  return item?.claim_id || item?.source_claim_id || item?.source_claim_key || ''
}

export function safeDomId(value) {
  return compactText(value, '')
    .replace(/[^A-Za-z0-9_-]+/g, '-')
    .replace(/^-+|-+$/g, '') || 'item'
}

export function claimRowDomId(claimId) {
  return `vf-claim-row-${safeDomId(claimId)}`
}

export function sceneValueFromId(id, fallbackIndex) {
  return idNumber(id, 'SC') || (fallbackIndex + 1)
}

export function contextValueFromId(id, fallbackIndex) {
  return idNumber(id, 'C') || (fallbackIndex + 1)
}

export function claimValueFromId(id, fallbackIndex) {
  return idNumber(id, 'CL') || (fallbackIndex + 1)
}

export function uniqueTexts(values) {
  return Array.from(new Set(asArray(values).map(item => compactText(item, '')).filter(Boolean)))
}

export function getClaimFlowSource(row) {
  return row.claim || row.comparison || row.issue || row.type || row.severity || {}
}

export function claimFlowContextId(row) {
  const source = getClaimFlowSource(row)
  return compactText(source.context_id || asArray(source.context_ids)[0], '-')
}

export function claimFlowSceneId(row) {
  const source = getClaimFlowSource(row)
  const explicit = compactText(source.scene_id || source.scene_key, '')
  if (explicit) return explicit
  const contextId = claimFlowContextId(row)
  return sceneIdFromContextId(contextId) || compactText(source.scene_index || source.scene_number, '-')
}

export function sceneIdFromContextId(contextId) {
  const match = compactText(contextId, '').match(/(?:^|-)SC\d+/i)
  return match ? match[0].replace(/^-/, '') : ''
}

export function hasIssueInComparison(row) {
  return Object.values(asObject(row.comparison?.models)).some(item => item?.has_issue)
}

export function issueTypeTone(type) {
  const ISSUE_TYPE_SCORE_KEYS = ['factual_error', 'temporal_error', 'confusing_explanation', 'scope_overclaim']
  const index = ISSUE_TYPE_SCORE_KEYS.indexOf(compactText(type, ''))
  return index >= 0 ? `type-${index + 1}` : ''
}

export function jumpToClaimRow(claimId, retries = 2) {
  const node = document.getElementById(claimRowDomId(claimId))
  if (!node) {
    if (retries > 0 && typeof window !== 'undefined') {
      window.requestAnimationFrame(() => jumpToClaimRow(claimId, retries - 1))
    }
    return
  }
  if (node.tagName === 'DETAILS') node.open = true
  node.scrollIntoView({ behavior: 'smooth', block: 'center' })
  node.querySelector('summary')?.focus?.({ preventScroll: true })
}

export function pendingText(status, runningText, waitingText) {
  return status === 'run' ? runningText : waitingText
}

export function statusText(status) {
  if (status === 'done') return '완료'
  if (status === 'run') return '진행 중'
  return '대기'
}

export function hasDataForTab(model, tabKey) {
  if (tabKey === 'claim_extraction') return model.claims.length > 0 || model.transcriptEntries.length > 0 || model.transcriptScenes.length > 0
  if (tabKey === 'issue_judge') return model.issueComparisonRows.length > 0 || model.issueCandidates.length > 0
  if (tabKey === 'issue_classification') return model.issueTypeRows.length > 0
  if (tabKey === 'final_verification') return model.severityItems.length > 0
  if (tabKey === 'slide_review') return model.slideFindings.length > 0
  return false
}

export function displayStatusForTab(model, tabKey, pipelineStatus) {
  return hasDataForTab(model, tabKey) ? 'done' : pipelineStatus
}

export function resultFileUrl(value, resultId) {
  const DEV_FILE_BASE = typeof window !== 'undefined' && window.location?.hostname
    ? `http://${window.location.hostname}:8000`
    : ''
  const FILE_BASE = (import.meta.env.VITE_API_BASE_URL || DEV_FILE_BASE || '').replace(/\/$/, '')
  const url = toResultFileUrl(value, resultId)
  return url.startsWith('/files/') ? `${FILE_BASE}${url}` : url
}

export function hideMissingImage(event) {
  event.currentTarget.closest('.vf-context-media, .vf-slide-thumb, .vf-source-transcript-thumb')?.setAttribute('data-missing', 'true')
  event.currentTarget.closest('.vf-context-panel, .vf-slide-report, .vf-source-transcript-scene')?.setAttribute('data-image-missing', 'true')
}

const FINAL_REVIEW_SCORE_THRESHOLD = 0.2
const MANUAL_REVIEW_DISAGREEMENT_THRESHOLD = 0.35
const LOW_MARGIN_THRESHOLD = 0.1

export function finalReviewReasonFlags(severity) {
  if (!severity) {
    return {
      problemThreshold: false,
      modelDisagreement: false,
      lowMargin: false,
    }
  }
  const score = Number(getScore(severity))
  const status = severity.status || statusFromSeverityScore(score)
  const disagreement = Number(severity.model_disagreement ?? severity.classified_issue_verifier?.model_disagreement)
  const margin = Number(severity.previous_classification?.margin)
  return {
    problemThreshold: (
      status === 'confirmed' ||
      status === 'professor_check' ||
      status === 'review_needed' ||
      (Number.isFinite(score) && score > FINAL_REVIEW_SCORE_THRESHOLD)
    ),
    modelDisagreement: Number.isFinite(disagreement) && disagreement >= MANUAL_REVIEW_DISAGREEMENT_THRESHOLD,
    lowMargin: Boolean(
      severity.previous_classification?.low_margin ||
      (Number.isFinite(margin) && margin < LOW_MARGIN_THRESHOLD)
    ),
  }
}

export function finalReviewReasonLabels(severity) {
  const flags = finalReviewReasonFlags(severity)
  const labels = []
  if (flags.problemThreshold) labels.push('문제 기준 초과')
  if (flags.modelDisagreement) labels.push('모델 의견 불합치')
  if (flags.lowMargin) labels.push('분류 모호함')
  return labels
}

export function isFinalReviewTarget(severity) {
  return finalReviewReasonLabels(severity).length > 0
}

