import { useEffect, useMemo, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { getLectureDetail, getLectureVerifier } from '../lib/api'

import VideoPlayer from '../components/watch/VideoPlayer'

import '../styles/verifier.css'

const VERIFIER_POLL_MS = 5000
const PROFESSOR_CHECK_MIN_SCORE = 0.4
const ISSUE_FILTERS = [
  { key: 'all', label: '전체' },
  { key: 'factual_error', code: 'A', label: '발언 자체 오류' },
  { key: 'temporal_error', code: 'B', label: '시간적 오류' },
  { key: 'scope_overclaim', code: 'C', label: '범위 과잉 단정' },
  { key: 'confusing_explanation', code: 'D', label: '혼동 가능 설명' },
]
const ISSUE_FILTER_DESCRIPTIONS = {
  factual_error: '문장 자체의 개념, 인과관계, 용어 연결, 수치가 강의 문맥을 봐도 틀린 경우입니다.',
  temporal_error: '현재성, 최신성, 지원 여부, 사용 여부, 시점 의존 수치나 상태가 기준 시점에서 틀리거나 확인이 필요한 경우입니다.',
  scope_overclaim: '반례나 예외가 있는데도 항상, 모든, 오직, 반드시처럼 범위를 과하게 닫아 말한 경우입니다.',
  confusing_explanation: '발언 자체가 명백히 틀렸다고 단정하기보다, 학생이 핵심 개념이나 주체/과정을 잘못 외울 가능성이 큰 설명입니다.',
}

function asArray(value) {
  return Array.isArray(value) ? value : []
}

function formatTime(seconds) {
  const value = Number(seconds)
  const safe = Number.isFinite(value) ? Math.max(0, value) : 0
  const m = Math.floor(safe / 60)
  const s = Math.floor(safe % 60)
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`
}

function formatPercent(value) {
  const number = Number(value)
  if (!Number.isFinite(number)) return '-'
  return `${Math.round(number * 100)}%`
}

function toNumberOrUndefined(value) {
  const number = Number(value)
  return Number.isFinite(number) ? number : undefined
}

function compactText(value, fallback = '-') {
  const text = String(value ?? '').trim()
  return text || fallback
}

function normalizeComparableText(value) {
  return String(value ?? '')
    .replace(/`/g, '')
    .replace(/[^\p{L}\p{N}]+/gu, ' ')
    .replace(/\s+/g, ' ')
    .trim()
    .toLowerCase()
}

function uniqueTextParts(...values) {
  const parts = []
  for (const value of values.flat()) {
    const text = String(value ?? '').trim()
    if (!text) continue
    const normalized = normalizeComparableText(text)
    if (!normalized) continue
    const duplicate = parts.some((existing) => {
      const current = normalizeComparableText(existing)
      return current === normalized || current.includes(normalized) || normalized.includes(current)
    })
    if (!duplicate) parts.push(text)
  }
  return parts
}

function joinTextParts(...values) {
  return uniqueTextParts(...values).join(' / ')
}

function labeledText(label, value) {
  const text = String(value ?? '').trim()
  return text ? `${label}: ${text}` : ''
}

function labelForIssueType(type) {
  const labels = {
    factual_error: '발언 자체 오류',
    temporal_error: '시간적 오류',
    scope_overclaim: '범위 과잉 단정',
    confusing_explanation: '혼동 가능 설명',
    outdated: '시간적 오류',
    simple_factual_error: '단순 사실 오류',
    scope_error: '범위 오류',
  }
  return labels[type] || compactText(type)
}

function codeForIssueType(type) {
  const codes = {
    factual_error: 'A',
    temporal_error: 'B',
    scope_overclaim: 'C',
    confusing_explanation: 'D',
    outdated: 'B',
    simple_factual_error: 'A',
    scope_error: 'C',
  }
  return codes[type] || ''
}

function labelWithIssueCode(type, explicitCode) {
  const label = labelForIssueSubtype(type) || labelForIssueType(type)
  const code = explicitCode || codeForIssueType(type)
  return code ? `${code}. ${label}` : label
}

function labelForIssueSubtype(type) {
  const labels = {
    factual_error: '발언 자체 오류',
    temporal_error: '시간적 오류',
    scope_overclaim: '범위 과잉 단정',
    confusing_explanation: '혼동 가능 설명',
    simple_factual_error: '단순 사실 오류',
    scope_error: '범위 오류',
    outdated: '시간적 오류',
  }
  return labels[type] || compactText(type)
}

function getIssueSubtype(item) {
  const direct = item.feedback_type || item.issue_type || item.type || item.issue_subtype
  if (direct === 'temporal_error') return 'temporal_error'
  if (direct === 'outdated') return 'temporal_error'
  if (direct === 'scope_overclaim') return 'scope_overclaim'
  if (direct === 'scope_error') return 'scope_overclaim'
  if (direct === 'confusing_explanation') return 'confusing_explanation'
  if (direct === 'factual_error') return 'factual_error'
  if (direct === 'simple_factual_error') return 'factual_error'
  if (item.issue_pattern === 'scope_overstatement') return 'scope_overclaim'
  return ''
}

function countIssueFilters(items) {
  return items.reduce((acc, item) => {
    const key = getIssueSubtype(item) || 'unknown'
    acc.all += 1
    acc[key] = (acc[key] || 0) + 1
    return acc
  }, { all: 0, factual_error: 0, temporal_error: 0, scope_overclaim: 0, confusing_explanation: 0 })
}

function matchesIssueFilter(item, filter) {
  if (!filter || filter === 'all') return true
  return getIssueSubtype(item) === filter
}

function claimDisplayIssueKey(claim) {
  return getIssueSubtype(claim) || claim.feedback_type || claim.issue_type || claim.type
}

function labelForClaimIssue(claim) {
  if (claim.issue_type_code_label) return claim.issue_type_code_label
  const key = claimDisplayIssueKey(claim)
  return labelWithIssueCode(key, claim.issue_type_code)
}

function formatIssueTypeScores(scores) {
  if (!scores || typeof scores !== 'object') return ''
  return ISSUE_FILTERS
    .filter((item) => item.key !== 'all')
    .map((item) => {
      const value = scores[item.key] ?? scores[item.code] ?? scores[item.code?.toLowerCase?.()]
      const number = Number(value)
      return Number.isFinite(number) ? `${item.code}.${item.label} ${Math.round(number * 100)}%` : ''
    })
    .filter(Boolean)
    .join(' / ')
}

function formatTypeJudgments(judgments) {
  if (!judgments || typeof judgments !== 'object') return ''
  return ISSUE_FILTERS
    .filter((item) => item.key !== 'all')
    .map((item) => {
      const row = judgments[item.code] || judgments[item.key] || judgments[item.code?.toLowerCase?.()]
      if (!row || typeof row !== 'object') return ''
      const score = Number(row.score)
      const reason = compactText(row.reason, '')
      if (!Number.isFinite(score) && !reason) return ''
      const scoreText = Number.isFinite(score) ? `${Math.round(score * 100)}%` : ''
      return `${item.code}.${item.label} ${scoreText}${reason ? ` - ${reason}` : ''}`.trim()
    })
    .filter(Boolean)
    .join('\n')
}

function ModelEvidenceRow({ label, value }) {
  if (value === undefined || value === null || value === '') return null
  return (
    <div className="vf-model-evidence-row">
      <span>{label}</span>
      <p>{value}</p>
    </div>
  )
}

function buildModelIssueBasis(verdict) {
  return joinTextParts(
    verdict?.reason,
  )
}

function buildModelContextBasis(verdict) {
  return joinTextParts(
    verdict?.context_resolution,
    verdict?.context_resolution_reason,
  )
}

function buildModelCorrection(verdict) {
  return joinTextParts(
    verdict?.correction_hint,
  )
}

function buildClaimCoreIssue(claim) {
  return joinTextParts(
    claim.issue,
  )
}

function buildClaimBasis(claim) {
  return joinTextParts(
    claim.context_resolution_reason,
  )
}

function buildClaimCorrection(claim) {
  return joinTextParts(
    claim.correction_hint,
  )
}

function groupTyposBySlide(items) {
  const groups = new Map()
  asArray(items).forEach((typo, idx) => {
    const slideNumber = typo.slide_number ?? 'unknown'
    const key = String(slideNumber)
    if (!groups.has(key)) {
      groups.set(key, {
        key,
        slideNumber,
        title: typo.slide_title || '',
        imageUrl: typo.slide_image_url || typo.image_url || '',
        items: [],
      })
    }

    const group = groups.get(key)
    if (!group.imageUrl && (typo.slide_image_url || typo.image_url)) {
      group.imageUrl = typo.slide_image_url || typo.image_url
    }
    if (!group.title && typo.slide_title) {
      group.title = typo.slide_title
    }
    group.items.push({ ...typo, _typoIndex: idx })
  })

  return Array.from(groups.values()).sort((a, b) => {
    const aNumber = Number(a.slideNumber)
    const bNumber = Number(b.slideNumber)
    if (Number.isFinite(aNumber) && Number.isFinite(bNumber)) return aNumber - bNumber
    return String(a.slideNumber).localeCompare(String(b.slideNumber))
  })
}

function uniqueStrings(values) {
  return [...new Set(asArray(values).map((value) => String(value || '').trim()).filter(Boolean))]
}

function sortUtteranceIds(values) {
  return uniqueStrings(values).sort((a, b) => {
    const left = /^U\d+$/.test(a) ? Number(a.slice(1)) : Number.MAX_SAFE_INTEGER
    const right = /^U\d+$/.test(b) ? Number(b.slice(1)) : Number.MAX_SAFE_INTEGER
    return left - right || a.localeCompare(b)
  })
}

function extractUtteranceIds(text) {
  if (!text) return []
  return uniqueStrings(String(text).match(/\bU\d{4,}\b/g) || [])
}

function getItemLocation(item, sourceClaim = {}) {
  return item.location || sourceClaim.location || {}
}

function getItemStartTime(item, sourceClaim = {}) {
  const location = getItemLocation(item, sourceClaim)
  const value = Number(location.start_time ?? item.start_time ?? sourceClaim.start_time)
  return Number.isFinite(value) ? value : undefined
}

function getFeedbackUtteranceIds(item, sourceClaim = {}) {
  const evidence = item.evidence || {}
  const sourceIssues = asArray(evidence.source_issues)
  return sortUtteranceIds([
    ...asArray(item.utterance_ids),
    ...asArray(item.related_utterance_ids),
    ...asArray(evidence.related_utterance_ids),
    item.utterance_id,
    sourceClaim.utterance_id,
    ...sourceIssues.map((issue) => issue?.utterance_id),
    ...extractUtteranceIds(item.confirmation_reason || evidence.confirmation_reason),
  ])
}

function getConfirmationReason(item) {
  const direct =
    item.confirmation_reason ||
    item.evidence?.confirmation_reason ||
    item.cross_recheck_reason
  if (direct) return direct

  const modelResults = item.checks?.crosscheck?.model_results
  if (!Array.isArray(modelResults)) return ''
  return modelResults
    .filter((row) => row?.verdict === 'agree' && row?.reason)
    .map((row) => `[${row.model || 'model'}] ${row.reason}`)
    .join(' / ')
}

function getRejectionReason(item) {
  return (
    item.rejection_reason ||
    item.professor_check_reason ||
    item.review_reason ||
    item.evidence?.rejection_reason ||
    item.checks?.crosscheck?.reason ||
    ''
  )
}

function getModelVerdicts(item) {
  const rows = asArray(item.checks?.crosscheck?.model_results)
  return rows.reduce((acc, row) => {
    const model = row?.model || row?.resolved_model || row?.source_model
    if (model) acc[model] = row
    return acc
  }, {})
}

function getCrosscheckScoreFromModels(modelResults) {
  const rows = asArray(modelResults)
  let weightedSum = 0
  let totalWeight = 0

  rows.forEach((row) => {
    const score = toNumberOrUndefined(row?.confidence ?? row?.vote_score ?? row?.score)
    const weight = toNumberOrUndefined(row?.model_weight) ?? 1
    if (score === undefined || weight <= 0) return
    weightedSum += score * weight
    totalWeight += weight
  })

  if (totalWeight <= 0) return undefined
  return Math.max(0, Math.min(1, weightedSum / totalWeight))
}

function getCrosscheckScore(item, crosscheck = {}) {
  return (
    toNumberOrUndefined(item.crosscheck_score) ??
    toNumberOrUndefined(item.score) ??
    toNumberOrUndefined(crosscheck.score) ??
    toNumberOrUndefined(crosscheck.scoring?.score) ??
    getCrosscheckScoreFromModels(crosscheck.model_results)
  )
}

function getCrosscheckScorePercent(item, crosscheck = {}, score) {
  const direct = toNumberOrUndefined(item.crosscheck_score_percent ?? crosscheck.score_percent ?? crosscheck.scoring?.score_percent)
  if (direct !== undefined) return direct
  return score !== undefined ? score * 100 : undefined
}

function statusFromScore(score) {
  if (score === undefined) return ''
  if (score >= 0.8) return 'confirmed'
  if (score >= PROFESSOR_CHECK_MIN_SCORE) return 'professor_check'
  return 'rejected'
}

function displayStageFromScore(score, fallbackStatus = '') {
  if (score !== undefined) {
    return score >= PROFESSOR_CHECK_MIN_SCORE ? 'professor_check' : 'rejected'
  }
  const status = fallbackStatus === 'review_needed' ? 'professor_check' : fallbackStatus
  if (status === 'confirmed') return 'professor_check'
  return status || 'professor_check'
}

function scoreLabel(score) {
  return score !== undefined ? formatPercent(score) : ''
}

function sourceIssueUtterance(issue) {
  return (
    issue?.display_text ||
    issue?.display_source_text ||
    issue?.source_text ||
    issue?.claim_context_text ||
    issue?.claim_text ||
    issue?.raw_claim_text ||
    ''
  )
}

function splitUtteranceText(text, fallbackUid = '') {
  const raw = String(text || '').trim()
  const match = raw.match(/^([A-Z]\d+(?:-C\d+)?):\s*(.*)$/)
  if (match) {
    return { uid: match[1], body: match[2], raw }
  }
  return { uid: fallbackUid, body: raw, raw }
}

function normalizeSourceText(text) {
  const { body } = splitUtteranceText(text)
  return body
    .replace(/`/g, '')
    .replace(/[^\p{L}\p{N}]+/gu, ' ')
    .replace(/\s+/g, ' ')
    .trim()
    .toLowerCase()
}

function sourceTextTokens(text) {
  return normalizeSourceText(text)
    .split(' ')
    .map((token) => token.trim())
    .filter((token) => token.length >= 2)
}

function sourceTextSimilarity(left, right) {
  const leftTokens = new Set(sourceTextTokens(left))
  const rightTokens = new Set(sourceTextTokens(right))
  if (!leftTokens.size || !rightTokens.size) return 0
  let intersection = 0
  for (const token of leftTokens) {
    if (rightTokens.has(token)) intersection += 1
  }
  return intersection / Math.min(leftTokens.size, rightTokens.size)
}

function areSimilarSourceIssues(left, right) {
  const leftText = sourceIssueUtterance(left)
  const rightText = sourceIssueUtterance(right)
  if (!leftText || !rightText) return false
  const leftNorm = normalizeSourceText(leftText)
  const rightNorm = normalizeSourceText(rightText)
  if (!leftNorm || !rightNorm) return false
  if (leftNorm === rightNorm) return true
  if (leftNorm.includes(rightNorm) || rightNorm.includes(leftNorm)) return true
  return sourceTextSimilarity(leftText, rightText) >= 0.78
}

function mergeSourceIssueDisplay(group) {
  const first = group[0] || {}
  const firstText = sourceIssueUtterance(first)
  const firstSplit = splitUtteranceText(firstText, first?.utterance_id)
  const ids = Array.from(
    new Set(
      group
        .map((issue) => issue?.utterance_id || splitUtteranceText(sourceIssueUtterance(issue)).uid)
        .map((uid) => String(uid || '').trim())
        .filter(Boolean),
    ),
  )
  const mergedUid = ids.join(', ') || firstSplit.uid
  return {
    ...first,
    utterance_id: mergedUid,
    merged_utterance_ids: ids,
    display_text: mergedUid ? `${mergedUid}: ${firstSplit.body || firstText}` : firstText,
  }
}

function dedupeSourceIssues(issues) {
  const groups = []
  for (const issue of asArray(issues)) {
    const existing = groups.find((group) => areSimilarSourceIssues(group[0], issue))
    if (existing) {
      existing.push(issue)
    } else {
      groups.push([issue])
    }
  }
  return groups.map(mergeSourceIssueDisplay)
}

function sourceIssueTitle(issue, index) {
  const text = sourceIssueUtterance(issue)
  if (text) return compactText(text, '', 180)
  const uid = issue?.utterance_id || `근거 ${index + 1}`
  return uid
}

function buildSourceUtteranceTitle(sourceIssues, fallback = '') {
  const utterances = dedupeSourceIssues(sourceIssues)
    .map(sourceIssueUtterance)
    .map((text) => String(text || '').trim())
    .filter(Boolean)

  if (utterances.length === 1) return utterances[0]
  if (utterances.length > 1) return utterances.join(' / ')
  return fallback
}

function buildMergeExplanation(evidence) {
  const sourceIssues = asArray(evidence?.source_issues)
  if (sourceIssues.length <= 1) return ''
  return evidence?.merge_rationale || ''
}

function firstModelField(crosscheck, field) {
  const rows = asArray(crosscheck?.model_results)
    .map((row) => {
      const value = String(row?.[field] || '').trim()
      const verdict = String(row?.verdict || '').toLowerCase()
      const status = String(row?.status || '').toLowerCase()
      const confidence = Number(row?.confidence ?? row?.vote_score ?? 0)
      const isRejectedSummary =
        field === 'context_issue_summary' &&
        value === '제공 문맥상 독립적으로 남는 잘못된 명제 없음'
      const rank = isRejectedSummary
        ? -1
        : verdict === 'agree' || status === 'kept'
          ? 2
          : verdict === 'inconclusive'
            ? 1
            : 0
      return { row, value, rank, confidence: Number.isFinite(confidence) ? confidence : 0 }
    })
    .filter((item) => item.value && item.rank >= 0)
    .sort((a, b) => b.rank - a.rank || b.confidence - a.confidence)

  return rows[0]?.value || ''
}

function buildContextTitle(sourceIssues, fallback = '') {
  const rows = dedupeSourceIssues(sourceIssues)
  if (rows.length > 1) return buildSourceUtteranceTitle(rows, fallback)
  return buildSourceUtteranceTitle(sourceIssues, fallback)
}

function feedbackItemToClaim(item, claimById) {
  const sourceClaim = claimById.get(item.source_claim_id) || {}
  const problem = item.problem || {}
  const feedback = item.professor_feedback || {}
  const evidence = item.evidence || {}
  const crosscheck = item.checks?.crosscheck || {}
  const crosscheckScore = getCrosscheckScore(item, crosscheck)
  const crosscheckScorePercent = getCrosscheckScorePercent(item, crosscheck, crosscheckScore)
  const crosscheckWeightedStatus = displayStageFromScore(
    crosscheckScore,
    item.crosscheck_weighted_status ||
      crosscheck.status_by_score ||
      crosscheck.scoring?.status ||
      statusFromScore(crosscheckScore),
  )
  const location = getItemLocation(item, sourceClaim)
  const utteranceIds = getFeedbackUtteranceIds(item, sourceClaim)
  const status = displayStageFromScore(crosscheckScore, item.status)
  const sourceIssues = asArray(evidence.source_issues)
  const canonicalWrongProposition = evidence.canonical_wrong_proposition || item.canonical_wrong_proposition || ''
  const mergeExplanation = buildMergeExplanation(evidence)
  const sourceUtteranceTitle = buildContextTitle(
    sourceIssues,
    problem.source_text ||
      evidence.source_text ||
      item.source_text ||
      item.claim_text ||
      sourceClaim.claim_text ||
      '',
  )
  const title =
    sourceUtteranceTitle ||
    '-'

  return {
    ...item,
    stage: status || 'professor_check',
    utterance_id: utteranceIds.join(', ') || item.utterance_id || sourceClaim.utterance_id,
    utterance_ids: utteranceIds,
    start_time: getItemStartTime(item, sourceClaim),
    slide_number: location.slide_number ?? evidence.slide_number,
    claim_text: title,
    raw_claim_text: item.claim_text || sourceClaim.claim_text,
    source_text: sourceUtteranceTitle || problem.source_text || evidence.source_text || item.source_text,
    issue: problem.summary || feedback.summary,
    issue_type: item.feedback_type || item.issue_type || item.type,
    issue_type_code: item.issue_type_code,
    issue_type_code_label: item.issue_type_code_label,
    issue_type_scores: item.issue_type_scores || crosscheck.scoring?.issue_type_scores,
    issue_classification_scores: item.issue_classification_scores,
    issue_classification_by_model: item.issue_classification_by_model,
    issue_classification_primary_code: item.issue_classification_primary_code,
    issue_classification_primary_type: item.issue_classification_primary_type,
    issue_classification_rationale: item.issue_classification_rationale,
    primary_issue_type: item.primary_issue_type || crosscheck.scoring?.primary_issue_type,
    secondary_issue_types: item.secondary_issue_types || crosscheck.scoring?.secondary_issue_types,
    issue_type_rationale: item.issue_type_rationale || item.issue_classification_rationale,
    issue_category_label: item.feedback_label || item.issue_category_label,
    context_issue_summary:
      problem.context_issue_summary ||
      item.context_issue_summary ||
      firstModelField(crosscheck, 'context_issue_summary'),
    context_resolution: problem.context_resolution || evidence.context_resolution || feedback.context_resolution,
    context_resolution_reason:
      problem.context_resolution_reason ||
      evidence.context_resolution_reason ||
      feedback.context_resolution_reason,
    correction_hint: problem.correction_hint || feedback.correction_hint,
    crosscheck_score: crosscheckScore,
    crosscheck_score_percent: crosscheckScorePercent,
    crosscheck_score_verdict: item.crosscheck_score_verdict ?? crosscheck.verdict,
    crosscheck_weighted_status: crosscheckWeightedStatus,
    model_verdicts: getModelVerdicts(item),
    issue_detection: buildIssueDetection(item),
    confirmation_reason: status === 'confirmed' ? getConfirmationReason(item) : '',
    rejection_reason: status === 'rejected' ? getRejectionReason(item) : item.professor_check_reason || item.review_reason,
    evidence_sources: evidence.evidence_sources || item.evidence_sources,
    canonical_wrong_proposition: canonicalWrongProposition,
    merge_rationale: evidence.merge_rationale,
    merge_explanation: mergeExplanation,
    source_issues: sourceIssues,
    crosscheck_context_text: evidence.crosscheck_context_text || item.crosscheck_context_text,
  }
}

function SummaryMetric({ label, value, tone = '', active = false, onClick }) {
  return (
    <button
      className={`vf-metric ${tone ? `vf-metric--${tone}` : ''} ${active ? 'vf-metric--active' : ''}`}
      onClick={onClick}
    >
      <div className="vf-metric-value">{value}</div>
      <div className="vf-metric-label">{label}</div>
    </button>
  )
}

function Section({ title, count, tone = '', empty, children }) {
  return (
    <section className={`vf-section ${tone ? `vf-section--${tone}` : ''}`}>
      <div className="vf-section-head">
        <h2>{title}</h2>
        <span>{count}</span>
      </div>
      {count > 0 ? children : <div className="vf-empty">{empty}</div>}
    </section>
  )
}

function IssueTypeBreakdown({ items, activeFilter = 'all', onFilterChange }) {
  const counts = countIssueFilters(items)
  return (
    <div className="vf-type-breakdown">
      {ISSUE_FILTERS.map((filter) => {
        const count = counts[filter.key] || 0
        return (
          <button
            className={`vf-type-pill ${activeFilter === filter.key ? 'vf-type-pill--active' : ''} ${count ? '' : 'vf-type-pill--empty'}`}
            key={filter.key}
            onClick={() => onFilterChange?.(activeFilter === filter.key ? 'all' : filter.key)}
          >
            <span>{filter.code ? `${filter.code}. ${filter.label}` : filter.label}</span>
            <strong>{count || '없음'}</strong>
          </button>
        )
      })}
    </div>
  )
}

function IssueFilterDescription({ filter }) {
  const description = ISSUE_FILTER_DESCRIPTIONS[filter]
  if (!description) return null
  const filterItem = ISSUE_FILTERS.find((item) => item.key === filter)
  const label = filterItem ? `${filterItem.code ? `${filterItem.code}. ` : ''}${filterItem.label}` : filter
  return (
    <div className="vf-filter-description">
      <strong>{label}</strong>
      <span>{description}</span>
    </div>
  )
}

function SortControls({ value, onChange }) {
  return (
    <div className="vf-sort-controls" aria-label="정렬 방식">
      <button
        className={value === 'utterance' ? 'vf-sort-btn vf-sort-btn--active' : 'vf-sort-btn'}
        onClick={() => onChange('utterance')}
      >
        발화순
      </button>
      <button
        className={value === 'score' ? 'vf-sort-btn vf-sort-btn--active' : 'vf-sort-btn'}
        onClick={() => onChange('score')}
      >
        점수순
      </button>
    </div>
  )
}

function DetailRow({ label, value }) {
  if (value === undefined || value === null || value === '') return null
  return (
    <div className="vf-detail-row">
      <dt>{label}</dt>
      <dd>{value}</dd>
    </div>
  )
}

function SourceIssueBreakdown({ issues }) {
  const rows = dedupeSourceIssues(issues)
  if (!rows.length) return null
  return (
    <div className="vf-evidence-block">
      <div className="vf-evidence-title">해당 문맥</div>
      <div className="vf-source-issue-list">
        {rows.map((issue, idx) => (
          <div className="vf-source-issue" key={`${issue?.utterance_id || 'issue'}-${idx}`}>
            <strong>{sourceIssueTitle(issue, idx)}</strong>
          </div>
        ))}
      </div>
    </div>
  )
}

function ContextTextBlock({ text }) {
  const value = String(text || '').trim()
  if (!value) return null
  return (
    <div className="vf-evidence-block">
      <div className="vf-evidence-title">검증에 사용한 문맥</div>
      <pre className="vf-context-text">{value}</pre>
    </div>
  )
}

function ModelVerdicts({ verdicts }) {
  const entries = Object.entries(verdicts || {})
  if (!entries.length) return null
  return (
    <div className="vf-evidence-block">
      <div className="vf-evidence-title">모델별 판정 근거</div>
      <div className="vf-verdict-grid">
        {entries.map(([model, verdict]) => {
          return (
            <div className="vf-verdict" key={model}>
              <div className="vf-verdict-head">
                <span>{model}</span>
                <strong>점수 {formatPercent(verdict?.confidence ?? verdict?.vote_score)}</strong>
                {(verdict?.decision || verdict?.verdict || verdict?.status) && (
                  <em>{compactText(verdict?.decision || verdict?.verdict || verdict?.status)}</em>
                )}
                {verdict?.model_weight !== undefined && <em>가중치 {Number(verdict.model_weight).toFixed(2)}</em>}
                {verdict?.weighted_score !== undefined && <em>반영점수 {Number(verdict.weighted_score).toFixed(2)}</em>}
              </div>
              <div className="vf-model-evidence-list">
                <ModelEvidenceRow label="판정 근거" value={buildModelIssueBasis(verdict)} />
                <ModelEvidenceRow label="유형 점수" value={formatIssueTypeScores(verdict?.issue_type_scores)} />
                <ModelEvidenceRow label="유형별 독립 판단" value={formatTypeJudgments(verdict?.type_judgments)} />
                <ModelEvidenceRow label="유형 근거" value={verdict?.issue_type_rationale} />
                <ModelEvidenceRow label="문맥 판단" value={buildModelContextBasis(verdict)} />
                <ModelEvidenceRow label="수정 방향" value={buildModelCorrection(verdict)} />
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}

function formatDetectorAgreementLabel(label) {
  if (label === 'all_models') return '전체 공통'
  if (label === 'partial_overlap') return '부분 공통'
  if (label === 'single_model') return '단독 탐지'
  return label || ''
}

function buildIssueDetection(item) {
  const issueDetection = item.checks?.issue_detection || {}
  const detected = asArray(issueDetection.detected_by_models || item.detected_by_models)
    .map((model) => String(model || '').trim())
    .filter(Boolean)
  return {
    detected_by_models: detected,
    model_count: issueDetection.model_count ?? item.detector_model_count ?? detected.length,
    model_total: issueDetection.model_total ?? item.detector_model_total,
    agreement_ratio: issueDetection.agreement_ratio ?? item.detector_model_agreement_ratio,
    agreement_label: issueDetection.agreement_label ?? item.detector_agreement_label,
    model_votes: asArray(issueDetection.model_votes || item.detector_model_votes),
  }
}

function formatDetectorModels(issueDetection) {
  const detected = asArray(issueDetection?.detected_by_models).filter(Boolean)
  if (!detected.length) return ''
  const count = issueDetection?.model_count ?? detected.length
  const total = issueDetection?.model_total
  const suffix = total ? ` (${count}/${total})` : ''
  const label = formatDetectorAgreementLabel(issueDetection?.agreement_label)
  return `${detected.join(', ')}${suffix}${label ? ` · ${label}` : ''}`
}

function DetectorVotes({ issueDetection }) {
  const rows = asArray(issueDetection?.model_votes)
  const detected = asArray(issueDetection?.detected_by_models)
  if (!rows.length && !detected.length) return null
  return (
    <div className="vf-evidence-block">
      <div className="vf-evidence-title">1차 탐지 모델</div>
      <div className="vf-detector-vote-list">
        {rows.length > 0
          ? rows.map((row, idx) => (
            <div className="vf-detector-vote" key={`${row?.model || 'model'}-${idx}`}>
              <span>{row?.model || '-'}</span>
              {row?.confidence !== undefined && <strong>{formatPercent(row.confidence)}</strong>}
              <p>{compactText(row?.claim_text || row?.source_text, '', 180)}</p>
            </div>
          ))
          : detected.map((model) => (
            <div className="vf-detector-vote" key={model}>
              <span>{model}</span>
            </div>
          ))}
      </div>
    </div>
  )
}

function IssueDetectionStats({ stats }) {
  if (!stats || !Object.keys(stats).length) return null
  const rawPerModel = stats.raw_detections_per_model || {}
  const unionPerModel = stats.union_detections_per_model || {}
  const models = asArray(stats.models).filter(Boolean)
  const pairwise = asArray(stats.pairwise_overlap)
  return (
    <div className="vf-detector-stats">
      <div className="vf-detector-stats-head">
        <strong>Issue detection union</strong>
        <span>
          합집합 {stats.union_issue_count ?? 0}건 · 전체 공통 {stats.common_all_model_count ?? 0}건 ·
          부분 공통 {stats.partial_overlap_count ?? 0}건 · 단독 {stats.single_model_count ?? 0}건
        </span>
      </div>
      <div className="vf-detector-model-grid">
        {models.map((model) => (
          <div className="vf-detector-model" key={model}>
            <span>{model}</span>
            <strong>{rawPerModel[model] ?? 0}건</strong>
            <em>union {unionPerModel[model] ?? 0}건</em>
          </div>
        ))}
      </div>
      {pairwise.length > 0 && (
        <div className="vf-detector-pairwise">
          {pairwise.map((row, idx) => (
            <span key={`${asArray(row?.models).join('-') || 'pair'}-${idx}`}>
              {asArray(row?.models).join('↔')} {row?.overlap_count ?? 0}/{row?.either_count ?? 0}
              {row?.jaccard !== undefined ? ` (${Math.round(Number(row.jaccard) * 100)}%)` : ''}
            </span>
          ))}
        </div>
      )}
    </div>
  )
}

function EvidenceSources({ sources }) {
  const items = asArray(sources).slice(0, 5)
  if (!items.length) return null
  return (
    <div className="vf-evidence-block">
      <div className="vf-evidence-title">근거 링크</div>
      <div className="vf-source-list">
        {items.map((source, idx) => {
          const url = typeof source === 'string' ? source : source?.url
          const label = source?.title || source?.domain || url || `source ${idx + 1}`
          if (!url) return null
          return (
            <a key={`${url}-${idx}`} href={url} target="_blank" rel="noreferrer">
              {label}
            </a>
          )
        })}
      </div>
    </div>
  )
}

function ClaimCard({ claim, section, expanded, onToggle, onWatch }) {
  const title = claim.claim_text || claim.source_text || '-'
  const subtitle = []
  const startTime = Number(claim.start_time)
  const canWatch = Number.isFinite(startTime)
  const grounding = claim.grounding || {}
  const displayIssueKey = claimDisplayIssueKey(claim)
  const displayIssueLabel = labelForClaimIssue(claim)
  const sources = asArray(grounding.evidence_sources).length
    ? grounding.evidence_sources
    : asArray(claim.evidence_sources)
  const hasCrosscheckScore = claim.crosscheck_score !== undefined && claim.crosscheck_score !== null
  const crosscheckStatus = claim.crosscheck_weighted_status || claim.crosscheck_score_verdict
  const detectorLabel = formatDetectorModels(claim.issue_detection)

  return (
    <article className={`vf-claim-card ${expanded ? 'vf-claim-card--expanded' : ''}`}>
      <button className="vf-claim-main" onClick={onToggle}>
        <div className="vf-claim-copy">
          <div className="vf-claim-title">{title}</div>
          {subtitle.map((line) => (
            <div className="vf-claim-subtitle" key={line}>{line}</div>
          ))}
          <div className="vf-chip-row">
            {displayIssueKey && <span className={`vf-chip vf-chip--${displayIssueKey}`}>{displayIssueLabel}</span>}
            {hasCrosscheckScore && (
              <span className={`vf-chip vf-chip--score vf-chip--score-${crosscheckStatus || 'unknown'}`}>
                점수 {scoreLabel(claim.crosscheck_score)}
              </span>
            )}
            {detectorLabel && <span className="vf-chip vf-chip--detector">1차 {detectorLabel}</span>}
          </div>
        </div>
        <div className="vf-claim-meta">
          <span>{claim.utterance_id || '-'}</span>
          <span>{canWatch ? formatTime(startTime) : '-'}</span>
          {claim.slide_number !== undefined && <span>slide {claim.slide_number}</span>}
        </div>
      </button>

      {expanded && (
        <div className="vf-claim-detail">
          <dl>
            <DetailRow label="등장 시각" value={canWatch ? `${formatTime(startTime)} (${startTime.toFixed(2)}s)` : '-'} />
            <DetailRow label="발화 ID" value={claim.utterance_ids?.length ? claim.utterance_ids.join(', ') : claim.utterance_id} />
            <DetailRow
              label="검증 점수"
              value={
                hasCrosscheckScore
                  ? formatPercent(claim.crosscheck_score)
                  : ''
              }
            />
            <DetailRow label="검증 대상 문맥" value={claim.source_text || claim.claim_text} />
            <DetailRow label="원문 발화" value={claim.source_text || claim.raw_claim_text} />
            <DetailRow label="묶은 이유" value={claim.merge_explanation} />
            <DetailRow label="문제 유형" value={displayIssueLabel} />
            <DetailRow label="유형 코드" value={claim.issue_type_code || codeForIssueType(displayIssueKey)} />
            <DetailRow label="유형 confidence" value={formatIssueTypeScores(claim.issue_classification_scores || claim.issue_type_scores)} />
            <DetailRow label="유형 근거" value={claim.issue_type_rationale} />
            <DetailRow label="1차 탐지 모델" value={detectorLabel} />
            <DetailRow label="문맥 내 오류" value={claim.context_issue_summary} />
            <DetailRow label="핵심 문제" value={buildClaimCoreIssue(claim)} />
            <DetailRow label="판단 근거" value={buildClaimBasis(claim)} />
            <DetailRow label="문맥 해소 여부" value={claim.context_resolution} />
            <DetailRow label="문맥 해소 판단" value={claim.context_resolution_reason} />
            <DetailRow label="수정 방향" value={buildClaimCorrection(claim)} />
            <DetailRow label="기각/검토 사유" value={claim.rejection_reason || claim.review_reason_code || claim.rejection_reason_code} />
            <DetailRow label="기각 단계" value={claim.rejection_stage} />
            <DetailRow label="분류" value={[claim.claim_type, claim.issue_category_label].filter(Boolean).join(' / ')} />
            <DetailRow label="Grounding" value={grounding.status || grounding.reason || claim.grounding_status} />
          </dl>
          <SourceIssueBreakdown issues={claim.source_issues} />
          <DetectorVotes issueDetection={claim.issue_detection} />
          <ContextTextBlock text={claim.crosscheck_context_text} />
          <ModelVerdicts verdicts={claim.model_verdicts} />
          <EvidenceSources sources={sources} />
        </div>
      )}

      {canWatch && (
        <div className="vf-claim-actions">
          <button className="vf-watch-btn" onClick={onWatch}>영상 보기</button>
        </div>
      )}
    </article>
  )
}

function TypoItem({ typo }) {
  const candidates = asArray(typo.correction_candidates)
  const runCount = Number(typo.run_count || 0)
  return (
    <div className="vf-typo-item">
      <div className="vf-typo-main">
        <div>
          <div className="vf-typo-title">
            {compactText(typo.problematic_text)} <span>→</span> {compactText(typo.corrected_text)}
          </div>
          <div className="vf-typo-reason">{compactText(typo.reason, '')}</div>
        </div>
        <div className="vf-typo-meta">
          {typo.confidence !== undefined && <span>{formatPercent(typo.confidence)}</span>}
          {runCount > 1 && <span>{typo.support_count || 0}/{runCount}</span>}
        </div>
      </div>
      {candidates.length > 1 && (
        <div className="vf-candidate-list">
          {candidates.map((candidate, idx) => (
            <span key={`${candidate.corrected_text}-${idx}`}>
              {candidate.corrected_text} ({candidate.support_count || 0})
            </span>
          ))}
        </div>
      )}
    </div>
  )
}

function SlideTypoCard({ group, review = false }) {
  return (
    <article className={`vf-typo-slide-card ${review ? 'vf-typo-slide-card--review' : ''}`}>
      <div className={`vf-typo-slide-layout ${group.imageUrl ? '' : 'vf-typo-slide-layout--no-image'}`}>
        {group.imageUrl && (
          <div className="vf-typo-slide-image">
            <img src={group.imageUrl} alt={`Slide ${group.slideNumber}`} loading="lazy" />
          </div>
        )}
        <div className="vf-typo-slide-panel">
          <div className="vf-typo-slide-head">
            <strong>slide {group.slideNumber}</strong>
            <span>{group.items.length}건</span>
          </div>
          <div className="vf-typo-items">
            {group.items.map((typo) => (
              <TypoItem
                key={`${group.key}-${typo.problematic_text}-${typo.corrected_text}-${typo._typoIndex}`}
                typo={typo}
              />
            ))}
          </div>
        </div>
      </div>
    </article>
  )
}

export default function VerifierPage() {
  const { id } = useParams()
  const navigate = useNavigate()

  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [waitingForVerifier, setWaitingForVerifier] = useState(false)
  const [lecture, setLecture] = useState(null)
  const [verifier, setVerifier] = useState(null)
  const [expandedClaimKey, setExpandedClaimKey] = useState('')
  const [activeTab, setActiveTab] = useState('review')
  const [activeIssueFilter, setActiveIssueFilter] = useState('all')
  const [sortMode, setSortMode] = useState('utterance')
  const [isVideoMode, setIsVideoMode] = useState(false)
  const [seekToSeconds, setSeekToSeconds] = useState(null)

  useEffect(() => {
    if (!id) return

    let cancelled = false
    let timerId = null
    let firstLoad = true

    async function loadVerifier() {
      if (firstLoad) setLoading(true)
      setError('')

      try {
        const [detail, verifierResult] = await Promise.all([
          getLectureDetail(id),
          getLectureVerifier(id),
        ])
        if (cancelled) return

        setLecture(detail)
        setVerifier(verifierResult)
        setWaitingForVerifier(!verifierResult)

        if (!verifierResult) {
          timerId = window.setTimeout(loadVerifier, VERIFIER_POLL_MS)
        }
      } catch (err) {
        if (cancelled) return
        console.error('Verifier fetch error:', err)
        setWaitingForVerifier(false)
        setError(String(err?.message || err))
      } finally {
        if (firstLoad && !cancelled) {
          setLoading(false)
          firstLoad = false
        }
      }
    }

    loadVerifier()

    return () => {
      cancelled = true
      if (timerId) window.clearTimeout(timerId)
    }
  }, [id])

  const claimById = useMemo(() => {
    const map = new Map()
    for (const claim of asArray(verifier?.claims)) {
      if (claim?.claim_id) map.set(claim.claim_id, claim)
    }
    return map
  }, [verifier])

  const sections = useMemo(() => {
    const feedbackItems = asArray(verifier?.feedback_items)
    if (feedbackItems.length > 0) {
      const normalized = feedbackItems.map((item) => feedbackItemToClaim(item, claimById))
      const finalClaims = []
      const needsReview = normalized.filter((item) => item.stage === 'professor_check' || item.stage === 'review_needed')
      const rejected = normalized.filter((item) => item.stage === 'rejected')
      return {
        finalClaims,
        needsReview,
        slideTypos: asArray(verifier?.slide_typos),
        crossRejected: rejected,
        inconclusive: [],
        groundingRejected: [],
        filtered: rejected,
        firstStageRejected: [],
        usesFeedbackItems: true,
      }
    }

    const finalClaims = asArray(verifier?.final_confirmed_claims)
    const needsReview = [
      ...finalClaims,
      ...asArray(verifier?.needs_review_claims),
    ]
    const crossRejected = asArray(verifier?.crosscheck_rejected_claims)
    const inconclusive = asArray(verifier?.crosscheck_inconclusive_claims)
    const groundingRejected = asArray(verifier?.grounding_rejected_claims)
    const firstStageRejected = asArray(verifier?.first_stage_rejected_claims)
    return {
      finalClaims,
      needsReview,
      slideTypos: asArray(verifier?.slide_typos),
      crossRejected,
      inconclusive,
      groundingRejected,
      filtered: [
        ...crossRejected,
        ...inconclusive,
        ...groundingRejected,
      ],
      firstStageRejected,
      usesFeedbackItems: false,
    }
  }, [claimById, verifier])

  const counts = verifier?.counts || {}
  const reviewCount = sections.needsReview.length
  const typoCount = counts.slide_typos ?? sections.slideTypos.length
  const filteredCount = sections.filtered.length + sections.firstStageRejected.length

  function selectTab(tab) {
    setActiveTab(tab)
    setActiveIssueFilter('all')
    setExpandedClaimKey('')
  }

  function toggleClaim(claimKey) {
    setExpandedClaimKey((prev) => (prev === claimKey ? '' : claimKey))
  }

  function handleWatchClaim(startTime) {
    setIsVideoMode(true)
    setSeekToSeconds(Number(startTime) || 0)
  }

  function renderClaimList(items, section) {
    return (
      <div className="vf-claim-list">
        {items.map((claim, idx) => {
          const claimKey = `${section}-${claim.utterance_id || claim.source_claim_key || 'claim'}-${idx}`
          return (
            <ClaimCard
              key={claimKey}
              claim={claim}
              section={section}
              expanded={expandedClaimKey === claimKey}
              onToggle={() => toggleClaim(claimKey)}
              onWatch={() => handleWatchClaim(claim.start_time)}
            />
          )
        })}
      </div>
    )
  }

  function filterIssueClaims(items) {
    return items.filter((item) => matchesIssueFilter(item, activeIssueFilter))
  }

  function utteranceSortValue(item) {
    const start = Number(item.start_time)
    if (Number.isFinite(start)) return start
    const firstId = asArray(item.utterance_ids)[0] || item.utterance_id || ''
    const match = String(firstId).match(/U(\d+)/)
    return match ? Number(match[1]) : Number.MAX_SAFE_INTEGER
  }

  function sortClaims(items) {
    return [...items].sort((a, b) => {
      if (sortMode === 'score') {
        const scoreDelta = (Number(b.crosscheck_score) || 0) - (Number(a.crosscheck_score) || 0)
        if (scoreDelta !== 0) return scoreDelta
      }
      return utteranceSortValue(a) - utteranceSortValue(b)
    })
  }

  function renderTypoGroups(items, review = false) {
    return (
      <div className="vf-typo-list">
        {groupTyposBySlide(items).map((group) => (
          <SlideTypoCard key={`${review ? 'review' : 'typo'}-${group.key}`} group={group} review={review} />
        ))}
      </div>
    )
  }

  function renderFilteredSection({ title, items, section, empty, tone = '' }) {
    if (!items.length) return null
    const sortedItems = sortClaims(items)
    return (
      <Section key={section} title={title} count={items.length} tone={tone} empty={empty}>
        <SortControls value={sortMode} onChange={setSortMode} />
        {renderClaimList(sortedItems, section)}
      </Section>
    )
  }

  function renderActivePanel() {
    if (activeTab === 'review') {
      const filteredReview = sortClaims(filterIssueClaims(sections.needsReview))
      return (
        <Section
          title="강의자 확인이 필요한 내용 이슈"
          count={sections.needsReview.length}
          tone="review"
          empty="강의자 확인이 필요한 내용 이슈가 없습니다."
        >
          <IssueTypeBreakdown
            items={sections.needsReview}
            section="needs_review"
            activeFilter={activeIssueFilter}
            onFilterChange={setActiveIssueFilter}
          />
          <IssueFilterDescription filter={activeIssueFilter} />
          <SortControls value={sortMode} onChange={setSortMode} />
          {filteredReview.length > 0
            ? renderClaimList(filteredReview, 'needs_review')
            : <div className="vf-empty">선택한 유형의 강의자 확인 이슈가 없습니다.</div>}
        </Section>
      )
    }

    if (activeTab === 'typos') {
      return (
        <Section
          title="슬라이드 오타"
          count={sections.slideTypos.length}
          tone="typo"
          empty="슬라이드 오타가 없습니다."
        >
          {renderTypoGroups(sections.slideTypos)}
        </Section>
      )
    }

    if (activeTab === 'filtered') {
      if (sections.usesFeedbackItems) {
        const sortedRejected = sortClaims(sections.filtered)
        return (
          <Section
            title="기각된 내용 후보"
            count={sections.filtered.length}
            empty="기각된 내용 후보가 없습니다."
          >
            <SortControls value={sortMode} onChange={setSortMode} />
            {renderClaimList(sortedRejected, 'rejected')}
          </Section>
        )
      }

      const filteredGroups = [
        {
          title: '교차검증 기각',
          items: sections.crossRejected,
          section: 'crosscheck_rejected',
          empty: '두 모델 모두 검토 가치가 낮다고 본 후보가 없습니다.',
        },
        {
          title: '교차검증 불확실',
          items: sections.inconclusive,
          section: 'crosscheck_inconclusive',
          empty: '교차검증에서 불확실로 남은 후보가 없습니다.',
          tone: 'review',
        },
        {
          title: '근거 기각',
          items: sections.groundingRejected,
          section: 'grounding_rejected',
          empty: '외부 근거로 기각된 후보가 없습니다.',
        },
      ]
      const hasDetailedFiltered = filteredGroups.some((group) => group.items.length > 0)
      return (
        <>
          {hasDetailedFiltered
            ? filteredGroups
              .filter((group) => group.items.length > 0)
              .map((group) => renderFilteredSection(group))
            : (
              <Section
                title="필터링된 내용 후보"
                count={0}
                empty="문맥상 맞음, 교차검증 기각, 근거 기각 후보가 없습니다."
              />
            )}

          <Section
            title="1차 판정에서 제외된 claim"
            count={sections.firstStageRejected.length}
            empty="1차 판정에서 제외된 claim이 없습니다."
          >
            {renderClaimList(sections.firstStageRejected, 'first_stage_rejected')}
          </Section>
        </>
      )
    }

    return null
  }

  if (loading) return <div className="lp-loading">불러오는 중...</div>

  if (error) {
    return (
      <div className="vf-shell">
        <div className="vf-topbar">
          <button className="lp-back" onClick={() => navigate(-1)}>← 이전으로</button>
          <span className="lp-title">Verifier</span>
        </div>
        <div className="vf-error">{error}</div>
      </div>
    )
  }

  if (waitingForVerifier) {
    return <div className="lp-loading">Verifier 결과 생성 중...</div>
  }

  if (!lecture || !verifier) {
    return <div className="lp-loading">Verifier 결과를 찾을 수 없습니다</div>
  }

  return (
    <div className="vf-shell">
      <div className="vf-topbar">
        <div className="vf-topbar-left">
          <button className="lp-back" onClick={() => navigate(-1)}>← 이전으로</button>
          <span className="lp-title">{lecture.title} · Verifier</span>
        </div>
        {isVideoMode && (
          <button className="vf-exit-video-btn" onClick={() => setIsVideoMode(false)}>
            영상 닫기
          </button>
        )}
      </div>

      <div className={`vf-body ${isVideoMode ? 'vf-body--video' : ''}`}>
        {isVideoMode && (
          <section className="vf-video-pane">
            <VideoPlayer
              lecture={lecture}
              scenes={[]}
              currentScene={0}
              onSceneChange={() => {}}
              seekTo={null}
              seekToSeconds={seekToSeconds}
            />
          </section>
        )}

        <section className="vf-list-pane">
          <div className="vf-summary-card">
            <SummaryMetric
              label="강의자 확인"
              value={reviewCount}
              tone="review"
              active={activeTab === 'review'}
              onClick={() => selectTab('review')}
            />
            <SummaryMetric
              label="슬라이드 오타"
              value={typoCount}
              tone="typo"
              active={activeTab === 'typos'}
              onClick={() => selectTab('typos')}
            />
            <SummaryMetric
              label="기각"
              value={filteredCount}
              active={activeTab === 'filtered'}
              onClick={() => selectTab('filtered')}
            />
          </div>
          <IssueDetectionStats stats={verifier?.summary?.issue_detection_stats} />

          {renderActivePanel()}
        </section>
      </div>
    </div>
  )
}
