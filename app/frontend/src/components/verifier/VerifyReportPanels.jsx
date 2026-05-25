import { Fragment, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import { toResultFileUrl } from './review/verifierReviewUtils'
import { PHASES } from './verifierConstants'

const DEV_FILE_BASE = typeof window !== 'undefined' && window.location?.hostname
  ? `http://${window.location.hostname}:8000`
  : ''

const FILE_BASE = (import.meta.env.VITE_API_BASE_URL || DEV_FILE_BASE || '').replace(/\/$/, '')

const VERIFY_STEPS = [
  { key: 'claim_extraction', label: '주장 추출' },
  { key: 'issue_judge', label: '이슈 후보 판단' },
  { key: 'issue_classification', label: '이슈 유형 분류' },
  { key: 'final_verification', label: '최종 평가' },
  { key: 'slide_review', label: '슬라이드 오류' },
]

const REPORT_TAB_ORDER = [
  'claim_extraction',
  'issue_judge',
  'issue_classification',
  'final_verification',
  'slide_review',
]

const ISSUE_TYPE_LABELS = {
  factual_error: 'factual_error',
  temporal_error: 'temporal_error',
  scope_overclaim: 'scope_overclaim',
  confusing_explanation: 'confusing_explanation',
  unknown: 'unknown',
}

const ISSUE_TYPE_KEYS = [
  'factual_error',
  'temporal_error',
  'scope_overclaim',
  'confusing_explanation',
]

const SCORE_LABELS = {
  valid: '유효성',
  severity: '심각도',
  context: '문맥 해소',
  disagreement: '모델 불일치',
  final: '최종 점수',
}

const UNKNOWN_MODEL_LABEL = '알 수 없음'

const MANUAL_REVIEW_DISAGREEMENT_THRESHOLD = 0.35

const FINAL_REVIEW_SCORE_THRESHOLD = 0.2

const LOW_MARGIN_THRESHOLD = 0.1

const AGREEMENT_LABELS = {
  all_models_agreed: 'all_models_agreed',
  single_model_only: 'single_model_only',
  partial_agreement: 'partial_agreement',
  no_issue: 'no_issue',
  all_models_failed: 'all_models_failed',
}

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
  return ISSUE_TYPE_LABELS[type] || compactText(type)
}

function scoreLabel(type) {
  return SCORE_LABELS[type] || typeLabel(type)
}

export function agreementLabel(type) {
  return AGREEMENT_LABELS[type] || compactText(type)
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

export function resultFileUrl(value, resultId) {
  const url = toResultFileUrl(value, resultId)
  return url.startsWith('/files/') ? `${FILE_BASE}${url}` : url
}

export function hideMissingImage(event) {
  event.currentTarget.closest('.vf-context-media, .vf-slide-thumb, .vf-source-transcript-thumb')?.setAttribute('data-missing', 'true')
  event.currentTarget.closest('.vf-context-panel, .vf-slide-report, .vf-source-transcript-scene')?.setAttribute('data-image-missing', 'true')
}

export function statusLabel(status) {
  if (status === 'confirmed') return '확정'
  if (status === 'professor_check' || status === 'review_needed') return '검토'
  if (status === 'rejected') return '기각'
  return compactText(status)
}

function firstFilled(...values) {
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

function normalizeFeedbackItems(verifier) {
  const feedbackItems = asArray(verifier?.feedback_items)
  if (feedbackItems.length) return feedbackItems

  const decisionItems = [
    ...asArray(verifier?.final_confirmed_claims).map(item => ({ ...item, status: 'confirmed' })),
    ...asArray(verifier?.needs_review_claims).map(item => ({ ...item, status: 'professor_check' })),
    ...asArray(verifier?.verifier_rejected_claims).map(item => ({ ...item, status: 'rejected' })),
    ...asArray(verifier?.crosscheck_inconclusive_claims).map(item => ({ ...item, status: 'professor_check' })),
    ...asArray(verifier?.crosscheck_rejected_claims).map(item => ({ ...item, status: 'rejected' })),
    ...asArray(verifier?.first_stage_rejected_claims).map(item => ({ ...item, status: 'rejected' })),
    ...asArray(verifier?.grounding_rejected_claims).map(item => ({ ...item, status: 'rejected' })),
    ...asArray(verifier?.slide_rejected_claims).map(item => ({ ...item, status: 'rejected' })),
  ]
  if (decisionItems.length) return decisionItems

  return []
}

export function statusFromSeverityScore(score) {
  const value = Number(score)
  if (!Number.isFinite(value)) return 'professor_check'
  if (value >= 0.5) return 'confirmed'
  if (value <= 0.2) return 'rejected'
  return 'professor_check'
}

function getArtifactClaims(verifier, artifacts) {
  const claimsPayload = asObject(artifacts?.claims)
  const artifactClaims = asArray(claimsPayload.claims)
  if (artifactClaims.length) return artifactClaims

  const claimsJsonl = asArray(artifacts?.claimsJsonl)
  if (claimsJsonl.length) return claimsJsonl

  return asArray(verifier?.claims || verifier?.merged_claims || verifier?.extracted_claims)
}

function flattenIssuesByType(issuesByType) {
  return Object.entries(asObject(issuesByType)).flatMap(([type, rows]) => (
    asArray(rows).map(item => ({
      ...item,
      feedback_type: item.feedback_type || item.category || item.final_issue_type || type,
    }))
  ))
}

function getIssuesByType(artifacts, fallbackItems) {
  const classifiedIssues = asObject(artifacts?.classifiedIssues?.issues_by_type)
  if (Object.keys(classifiedIssues).length) return classifiedIssues

  const verifierIssues = asObject(artifacts?.classifiedIssueVerifier?.issues_by_type)
  if (Object.keys(verifierIssues).length) return verifierIssues

  return fallbackItems.reduce((groups, item) => {
    const type = getIssueType(item) || 'unknown'
    groups[type] = [...(groups[type] || []), item]
    return groups
  }, {})
}

function getVerifierItems(artifacts, feedbackItems) {
  const verifierRows = asArray(artifacts?.classifiedIssueVerifier?.all_issues)
  const typedRows = verifierRows.length
    ? verifierRows
    : flattenIssuesByType(artifacts?.classifiedIssueVerifier?.issues_by_type)
  if (!typedRows.length) return feedbackItems

  const feedbackByIssue = new Map(feedbackItems.map(item => [item.issue_id || item.feedback_id, item]))
  const judgeByIssue = new Map(asArray(artifacts?.issueJudge?.issues).map(item => [item.issue_id || item.id, item]))
  const typeByIssue = new Map(asArray(artifacts?.issueTypes?.classifications).map(item => [item.issue_id || item.id, item]))
  return typedRows.map(row => {
    const feedback = feedbackByIssue.get(row.issue_id || row.id) || {}
    const judged = judgeByIssue.get(row.issue_id || row.id) || {}
    const typed = typeByIssue.get(row.issue_id || row.id) || {}
    return {
      ...judged,
      ...typed,
      ...row,
      ...feedback,
      feedback_type: feedback.feedback_type || row.category || row.feedback_type,
      feedback_label: feedback.feedback_label || row.category_label || row.feedback_label,
      status: feedback.status || statusFromSeverityScore(row.final_severity_score),
      model_judgments: row.model_judgments,
      needs_manual_review: row.needs_manual_review,
      severity_score: feedback.severity_score ?? row.final_severity_score,
      classified_issue_verifier: {
        ...(feedback.classified_issue_verifier || {}),
        final_severity_score: row.final_severity_score,
        final_severity_percent: row.final_severity_percent,
        average_is_valid_issue: row.average_is_valid_issue,
        average_category_severity: row.average_category_severity,
        average_context_resolution: row.average_context_resolution,
        model_disagreement: row.model_disagreement,
        needs_manual_review: row.needs_manual_review,
      },
    }
  })
}

function getSlideFindings(verifier, artifacts) {
  const artifactSlides = asArray(artifacts?.slideErrors?.slide_errors)
  if (artifactSlides.length) return artifactSlides
  return asArray(verifier?.slide_errors)
}

function transcriptTextForContext(context) {
  return firstFilled(context?.text, context?.text_corrected, context?.text_original)
}

function transcriptRowsForSlide(slide) {
  const contextRows = asArray(slide?.contexts)
  return contextRows.length ? contextRows : asArray(slide?.transcript_segments)
}

function transcriptEntryFromContext(slide, context, index) {
  const slideNumber = slide?.slide_number
  const text = transcriptTextForContext(context)
  if (!text) return null
  const contextId = compactText(context?.context_id || `S${slideNumber || '0'}-T${index + 1}`, '')
  const occurrences = asArray(slide?.occurrences)
  const sceneIndex = firstFilled(context?.scene_index, context?.scene_number, idNumber(contextId, 'SC'))
  const occurrence = occurrences.find(item => compactText(item?.scene_index, '') === compactText(sceneIndex, ''))
    || (occurrences.length === 1 ? occurrences[0] : {})
  const resolvedSceneIndex = firstFilled(sceneIndex, occurrence?.scene_index, slide?.scene_index)
  const start = firstFilled(context?.start_time, context?.start, occurrence?.start_sec)
  const end = firstFilled(context?.end_time, context?.end, occurrence?.end_sec)
  return {
    ...context,
    context_id: contextId,
    slide_number: slideNumber ?? context?.slide_number,
    scene_id: sceneIdFromIndex(resolvedSceneIndex),
    scene_index: resolvedSceneIndex,
    slide_title: slide?.title || context?.slide_title || '',
    slide_image_path: firstFilled(slide?.image_path, slide?.slide_image_path, slide?.thumbnail_path, context?.image_path, context?.slide_image_path),
    slide_image_url: firstFilled(slide?.image_url, slide?.slide_image_url, slide?.thumbnail_url, context?.image_url, context?.slide_image_url),
    start_time: start,
    end_time: end,
    text,
  }
}

function getTranscriptScenes(artifacts) {
  const merged = asObject(artifacts?.mergedClean)
  const scenes = []
  const sceneMap = new Map()

  const ensureScene = (sceneIndex, slide, occurrence, fallbackKey = '') => {
    const resolvedSceneIndex = firstFilled(sceneIndex, occurrence?.scene_index, slide?.scene_index)
    const sceneId = sceneIdFromIndex(resolvedSceneIndex)
    const key = sceneId || compactText(fallbackKey, '') || `unknown-scene-${sceneMap.size + 1}`
    if (!sceneMap.has(key)) {
      const timeRange = asArray(slide?.time_range_seconds)
      const scene = {
        key,
        scene_id: sceneId,
        scene_index: resolvedSceneIndex,
        slide_number: slide?.slide_number,
        start_time: firstFilled(occurrence?.start_sec, timeRange[0], slide?.start_time),
        end_time: firstFilled(occurrence?.end_sec, timeRange[1], slide?.end_time),
        slide_title: slide?.title || '',
        slide_image_path: firstFilled(slide?.image_path, slide?.slide_image_path, slide?.thumbnail_path),
        slide_image_url: firstFilled(slide?.image_url, slide?.slide_image_url, slide?.thumbnail_url),
        entries: [],
      }
      sceneMap.set(key, scene)
      scenes.push(scene)
    }

    const scene = sceneMap.get(key)
    scene.scene_id = scene.scene_id || sceneId
    scene.scene_index = firstFilled(scene.scene_index, resolvedSceneIndex)
    scene.slide_number = firstFilled(scene.slide_number, slide?.slide_number)
    scene.start_time = firstFilled(scene.start_time, occurrence?.start_sec, slide?.start_time)
    scene.end_time = firstFilled(scene.end_time, occurrence?.end_sec, slide?.end_time)
    scene.slide_title = scene.slide_title || slide?.title || ''
    scene.slide_image_path = firstFilled(scene.slide_image_path, slide?.image_path, slide?.slide_image_path, slide?.thumbnail_path)
    scene.slide_image_url = firstFilled(scene.slide_image_url, slide?.image_url, slide?.slide_image_url, slide?.thumbnail_url)
    return scene
  }

  asArray(merged.slides).forEach((slide, slideIndex) => {
    const occurrences = asArray(slide?.occurrences)
    occurrences.forEach((occurrence, occurrenceIndex) => {
      const fallbackKey = compactText(occurrence?.scene_id || occurrence?.id, '') || `unknown-occurrence-${slideIndex + 1}-${occurrenceIndex + 1}`
      ensureScene(occurrence?.scene_index, slide, occurrence, fallbackKey)
    })

    const entries = transcriptRowsForSlide(slide)
      .map((context, index) => transcriptEntryFromContext(slide, context, index))
      .filter(Boolean)
      .sort((left, right) => {
        const leftStart = Number(left.start_time)
        const rightStart = Number(right.start_time)
        if (Number.isFinite(leftStart) && Number.isFinite(rightStart) && leftStart !== rightStart) {
          return leftStart - rightStart
        }
        return compactText(left.context_id, '').localeCompare(compactText(right.context_id, ''))
      })

    entries.forEach(entry => {
      const occurrence = occurrences.find(item => compactText(item?.scene_index, '') === compactText(entry.scene_index, ''))
      const scene = ensureScene(entry.scene_index, slide, occurrence, entry.context_id)
      scene.entries.push(entry)
      scene.start_time = firstFilled(scene.start_time, entry.start_time)
      scene.end_time = firstFilled(scene.end_time, entry.end_time)
    })
  })

  return scenes.sort((left, right) => {
    const leftStart = Number(left.start_time)
    const rightStart = Number(right.start_time)
    if (Number.isFinite(leftStart) && Number.isFinite(rightStart) && leftStart !== rightStart) {
      return leftStart - rightStart
    }
    return compactText(left.key, '').localeCompare(compactText(right.key, ''))
  })
}

function sortClaimRows(left, right) {
  const leftKey = compactText(left.claim_id, '')
  const rightKey = compactText(right.claim_id, '')
  const leftNumber = Number(leftKey.replace(/\D/g, ''))
  const rightNumber = Number(rightKey.replace(/\D/g, ''))
  if (Number.isFinite(leftNumber) && Number.isFinite(rightNumber) && leftNumber !== rightNumber) {
    return leftNumber - rightNumber
  }
  return leftKey.localeCompare(rightKey)
}

function ensureClaimFlowRow(rows, claimId) {
  const safeId = compactText(claimId, '')
  if (!safeId) return null
  if (!rows.has(safeId)) {
    rows.set(safeId, {
      claim_id: safeId,
      claim: null,
      comparison: null,
      issue: null,
      type: null,
      severity: null,
    })
  }
  return rows.get(safeId)
}

function buildClaimFlowRows({ claims, issueComparisonRows, issueCandidates, issueTypeRows, verifierItems }) {
  const rows = new Map()

  asArray(claims).forEach(claim => {
    const row = ensureClaimFlowRow(rows, claimKey(claim))
    if (row) row.claim = claim
  })

  asArray(issueComparisonRows).forEach(item => {
    const row = ensureClaimFlowRow(rows, claimKey(item))
    if (row) row.comparison = item
  })

  asArray(issueCandidates).forEach(item => {
    const row = ensureClaimFlowRow(rows, claimKey(item))
    if (row) row.issue = item
  })

  asArray(issueTypeRows).forEach(item => {
    const row = ensureClaimFlowRow(rows, claimKey(item))
    if (row) row.type = item
  })

  asArray(verifierItems).forEach(item => {
    const row = ensureClaimFlowRow(rows, claimKey(item))
    if (row) row.severity = item
  })

  return [...rows.values()].sort(sortClaimRows)
}

function buildReportModel(verifier, artifacts = {}) {
  const feedbackItems = normalizeFeedbackItems(verifier)
  const issueJudge = asObject(artifacts.issueJudge)
  const issueTypes = asObject(artifacts.issueTypes)
  const classifiedIssues = asObject(artifacts.classifiedIssues)
  const issueVerifier = asObject(artifacts.classifiedIssueVerifier)
  const slideErrors = asObject(artifacts.slideErrors)
  const issuesByType = getIssuesByType(artifacts, feedbackItems)
  const verifierItems = getVerifierItems(artifacts, feedbackItems)
  const slideFindings = getSlideFindings(verifier, artifacts)
  const claims = getArtifactClaims(verifier, artifacts)
  const transcriptScenes = getTranscriptScenes(artifacts)
  const transcriptEntries = transcriptScenes.flatMap(scene => asArray(scene.entries))
  const issueCandidates = asArray(issueJudge.issues).length ? asArray(issueJudge.issues) : feedbackItems
  const issueComparisonRows = asArray(artifacts.issueJudgeCompare?.by_claim)
  const issueTypeRows = asArray(issueTypes.classifications)
  const hasVerifierArtifact = Boolean(Object.keys(issueVerifier).length)
  const decisionItems = feedbackItems.length && !hasVerifierArtifact ? feedbackItems : verifierItems
  const claimFlowRows = buildClaimFlowRows({
    claims,
    issueComparisonRows,
    issueCandidates,
    issueTypeRows,
    verifierItems,
  })

  return {
    claims,
    transcriptScenes,
    transcriptEntries,
    contextCount: transcriptEntries.length,
    claimsSummary: asObject(artifacts.claims),
    issueCandidates,
    issueJudge,
    issueJudgeSummary: asObject(artifacts.issueJudgeSummary?.summary),
    issueJudgeCompare: asObject(artifacts.issueJudgeCompare),
    issueComparisonRows,
    issueTypes,
    issueTypeRows,
    classifiedIssues,
    issuesByType,
    severityItems: verifierItems,
    verifierItems,
    claimFlowRows,
    issueVerifier,
    feedbackItems,
    slideFindings,
    slideErrors,
    claimExtractionModels: [UNKNOWN_MODEL_LABEL],
    slideReviewModels: uniqueTexts(asArray(slideErrors.models)),
    finalVerifierModelWeights: asObject(verifier?.verifier_model_weights),
    finalSummary: asObject(verifier?.summary),
    finalCounts: asObject(verifier?.counts),
    counts: {
      confirmed: decisionItems.filter(item => item?.status === 'confirmed').length,
      review: decisionItems.filter(item => item?.status === 'professor_check' || item?.status === 'review_needed').length,
      rejected: decisionItems.filter(item => item?.status === 'rejected').length,
    },
  }
}

function getVerifyStepStatus(flow, index) {
  if (flow.phase === PHASES.VERIFY_READY || flow.phase === PHASES.REVIEWED || flow.phase === PHASES.UPLOAD_RESUME) {
    return 'done'
  }
  if (flow.phase !== PHASES.PIPELINE1) return 'wait'

  const activeIndex = Math.min(VERIFY_STEPS.length - 1, Math.max(0, Number(flow.stageGroupIndex) || 0))
  if (index < activeIndex) return 'done'
  if (index === activeIndex) return 'run'
  return 'wait'
}

function pendingText(status, runningText, waitingText) {
  return status === 'run' ? runningText : waitingText
}

function statusText(status) {
  if (status === 'done') return '완료'
  if (status === 'run') return '진행 중'
  return '대기'
}

function hasDataForTab(model, tabKey) {
  if (tabKey === 'claim_extraction') return model.claims.length > 0 || model.transcriptEntries.length > 0 || model.transcriptScenes.length > 0
  if (tabKey === 'issue_judge') return model.issueComparisonRows.length > 0 || model.issueCandidates.length > 0
  if (tabKey === 'issue_classification') return model.issueTypeRows.length > 0
  if (tabKey === 'final_verification') return model.severityItems.length > 0
  if (tabKey === 'slide_review') return model.slideFindings.length > 0
  return false
}

function displayStatusForTab(model, tabKey, pipelineStatus) {
  return hasDataForTab(model, tabKey) ? 'done' : pipelineStatus
}

function getCompletedVerifyStageCount(flow) {
  if (flow.phase === PHASES.VERIFY_READY || flow.phase === PHASES.REVIEWED || flow.phase === PHASES.UPLOAD_RESUME) {
    return VERIFY_STEPS.length
  }
  if (flow.phase !== PHASES.PIPELINE1) return 0
  const stageIndex = Number(flow.stageGroupIndex)
  return Math.max(0, Math.min(VERIFY_STEPS.length, Number.isFinite(stageIndex) ? stageIndex : 0))
}

function getCompletedDetailKey(completedStageCount) {
  const completedIndex = Math.min(
    VERIFY_STEPS.length - 1,
    Math.max(0, Number(completedStageCount) - 1)
  )
  return VERIFY_STEPS[completedIndex]?.key || 'claim_extraction'
}

function getVisibleVerifier(verifier, completedStageCount) {
  if (completedStageCount >= VERIFY_STEPS.length) return verifier

  return {
    schema_version: verifier?.schema_version,
    mode: verifier?.mode,
    models: verifier?.models,
    claims: completedStageCount >= 1 ? verifier?.claims : [],
    merged_claims: completedStageCount >= 1 ? verifier?.merged_claims : [],
    extracted_claims: completedStageCount >= 1 ? verifier?.extracted_claims : [],
    issues: completedStageCount >= 4 ? verifier?.issues : [],
    feedback_items: [],
    final_confirmed_claims: [],
    needs_review_claims: [],
    verifier_rejected_claims: [],
    crosscheck_inconclusive_claims: [],
    crosscheck_rejected_claims: [],
    first_stage_rejected_claims: [],
    grounding_rejected_claims: [],
    slide_rejected_claims: [],
    slide_errors: [],
    summary: {},
    counts: {},
    classified_issue_artifacts: verifier?.classified_issue_artifacts,
  }
}

function getVisibleArtifacts(artifacts, completedStageCount) {
  return {
    urls: artifacts?.urls,
    mergedClean: completedStageCount >= 1 ? artifacts?.mergedClean : null,
    claims: completedStageCount >= 1 ? artifacts?.claims : null,
    claimsJsonl: completedStageCount >= 1 ? artifacts?.claimsJsonl : [],
    issueJudge: completedStageCount >= 2 ? artifacts?.issueJudge : null,
    issueJudgeSummary: completedStageCount >= 2 ? artifacts?.issueJudgeSummary : null,
    issueJudgeCompare: completedStageCount >= 2 ? artifacts?.issueJudgeCompare : null,
    issueTypes: completedStageCount >= 3 ? artifacts?.issueTypes : null,
    classifiedIssues: completedStageCount >= 3 ? artifacts?.classifiedIssues : null,
    classifiedIssueVerifier: completedStageCount >= 4 ? artifacts?.classifiedIssueVerifier : null,
    slideErrors: completedStageCount >= 5 ? artifacts?.slideErrors : null,
    verification: completedStageCount >= 4 ? artifacts?.verification : null,
  }
}

function isSameFilter(left, right) {
  if (!left || !right) return false
  return left.type === right.type && left.value === right.value
}

function issueKey(item) {
  return item?.issue_id || item?.id || item?.feedback_id || ''
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

function sceneIdFromContextId(contextId) {
  const match = compactText(contextId, '').match(/(?:^|-)SC\d+/i)
  return match ? match[0].replace(/^-/, '') : ''
}

function idValueFromId(id, prefix, fallback) {
  return idNumber(id, prefix) || fallback
}

export function sceneValueFromId(id, fallbackIndex) {
  return idValueFromId(id, 'SC', fallbackIndex + 1)
}

export function contextValueFromId(id, fallbackIndex) {
  return idValueFromId(id, 'C', fallbackIndex + 1)
}

export function claimValueFromId(id, fallbackIndex) {
  return idValueFromId(id, 'CL', fallbackIndex + 1)
}

function addIssueToScope(scope, item) {
  const issue = issueKey(item)
  const claim = claimKey(item)
  if (issue) scope.issueIds.add(issue)
  if (claim) scope.claimIds.add(claim)
}

function getHighSeverityRows(model) {
  const count = Number(model.issueVerifier.summary?.high_severity_count)
  const rows = [...model.severityItems]
    .filter(item => Number.isFinite(Number(item.final_severity_score)))
    .sort((a, b) => Number(b.final_severity_score) - Number(a.final_severity_score))

  if (Number.isFinite(count) && count > 0) return rows.slice(0, count)
  return rows.filter(item => Number(item.final_severity_score) >= 0.7)
}

function buildFilterScope(model, filter) {
  const scope = {
    active: Boolean(filter),
    claimIds: new Set(),
    issueIds: new Set(),
    slideErrorIds: new Set(),
  }

  if (!filter) return scope

  if (filter.type === 'agreement') {
    model.issueComparisonRows
      .filter(item => item.agreement?.status === filter.value)
      .forEach(item => {
        if (item.claim_id) scope.claimIds.add(item.claim_id)
      })
    return scope
  }

  if (filter.type === 'model_only') {
    asArray(model.issueJudgeCompare.exclusive_by_model?.[filter.value]).forEach(id => scope.claimIds.add(id))
    model.issueComparisonRows
      .filter(item => asArray(item.agreement?.issue_models).length === 1 && item.agreement?.issue_models?.[0] === filter.value)
      .forEach(item => {
        if (item.claim_id) scope.claimIds.add(item.claim_id)
      })
    return scope
  }

  if (filter.type === 'issue_type') {
    model.issueTypeRows
      .filter(item => item.final_issue_type === filter.value || getIssueType(item) === filter.value)
      .forEach(item => addIssueToScope(scope, item))
    return scope
  }

  if (filter.type === 'final_issue_type') {
    model.severityItems
      .filter(item => (
        (item.needs_manual_review || item.status === 'confirmed' || item.status === 'professor_check') &&
        getIssueType(item) === filter.value
      ))
      .forEach(item => addIssueToScope(scope, item))
    return scope
  }

  if (filter.type === 'high_severity') {
    getHighSeverityRows(model).forEach(item => addIssueToScope(scope, item))
    return scope
  }

  if (filter.type === 'manual_review') {
    model.severityItems.filter(item => item.needs_manual_review).forEach(item => addIssueToScope(scope, item))
    return scope
  }

  if (filter.type === 'final_status') {
    model.severityItems
      .filter(item => (item.status || statusFromSeverityScore(item.final_severity_score)) === filter.value)
      .forEach(item => addIssueToScope(scope, item))
    return scope
  }

  if (filter.type === 'slide_reportable') {
    model.slideFindings
      .filter(item => item.is_reportable !== false)
      .forEach(item => {
        if (item.slide_error_id) scope.slideErrorIds.add(item.slide_error_id)
      })
  }

  return scope
}

function rowMatchesScope(item, scope, kind = 'issue') {
  if (!scope.active) return true
  if (kind === 'slide') {
    return scope.slideErrorIds.has(item.slide_error_id)
  }
  if (kind === 'claim_flow') {
    const issue = issueKey(item.severity) || issueKey(item.type) || issueKey(item.issue)
    const claim = item.claim_id || claimKey(item.claim) || claimKey(item.comparison) || claimKey(item.issue)
    return Boolean((issue && scope.issueIds.has(issue)) || (claim && scope.claimIds.has(claim)))
  }
  const issue = issueKey(item)
  const claim = claimKey(item)
  return Boolean((issue && scope.issueIds.has(issue)) || (claim && scope.claimIds.has(claim)))
}

function filterRows(rows, scope, kind) {
  if (!scope.active) return rows
  return rows.filter(item => rowMatchesScope(item, scope, kind))
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
    <div className="vf-chip-list">
      {chips.map(item => <span key={item}>{item}</span>)}
    </div>
  )
}

export function TextBlock({ label, children }) {
  if (!compactText(children, '')) return null
  return (
    <div className="vf-text-block">
      <span>{label}</span>
      <p>{children}</p>
    </div>
  )
}

export function InlineBlock({ label, children }) {
  if (!children) return null
  return (
    <div className="vf-text-block vf-text-block--inline">
      <span>{label}</span>
      {children}
    </div>
  )
}

export function ModelEvidenceSection({ items, valueFormat, title = '모델별 판단' }) {
  if (!asArray(items).length) return null
  return (
    <div className="vf-model-evidence-section">
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

const DISTRIBUTION_COLORS = ['var(--blue-fill)', 'var(--green-fill)', 'var(--amber-fill)', 'var(--red-fill)']

const ISSUE_TYPE_SCORE_KEYS = ['factual_error', 'temporal_error', 'confusing_explanation', 'scope_overclaim']

export function issueTypeTone(type) {
  const index = ISSUE_TYPE_SCORE_KEYS.indexOf(compactText(type, ''))
  return index >= 0 ? `type-${index + 1}` : ''
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
    <div className="vf-score-distribution">
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

function normalizeHeaderChip(chip) {
  if (chip && typeof chip === 'object') {
    return {
      label: compactText(chip.label, ''),
      models: asArray(chip.models).map(item => compactText(item, '')).filter(Boolean),
      tooltipTitle: compactText(chip.tooltipTitle, ''),
      tooltipVariant: compactText(chip.tooltipVariant, ''),
    }
  }
  return { label: compactText(chip, ''), models: [], tooltipTitle: '', tooltipVariant: '' }
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

function HeaderChips({ chips }) {
  const visibleChips = asArray(chips).map(normalizeHeaderChip).filter(chip => chip.label)
  if (!visibleChips.length) return null

  function headerChipTooltip(chip) {
    if (!chip.models.length) return null
    return (
      <>
        <span className="vf-bold">{chip.tooltipTitle || (chip.label === '모델' ? '사용 모델' : '사용 모델 목록')}</span>
        {chip.tooltipVariant === 'value' ? (
          <span className="vf-header-chip-tooltip-value">{chip.models[0]}</span>
        ) : (
          <span className="vf-header-chip-tooltip-list">
            {chip.models.map(item => <span key={item}>{item}</span>)}
          </span>
        )}
      </>
    )
  }

  return (
    <div className="vf-header-chip-list">
      {visibleChips.map((chip, index) => {
        const chipKey = `${chip.label}-${index}`
        return (
          <MouseTooltip
            key={chipKey}
            className={`vf-header-chip ${chip.models.length ? 'vf-header-chip--tooltip' : ''}`}
            tabIndex={0}
            tooltip={headerChipTooltip(chip)}
            tooltipClassName="vf-header-chip-tooltip"
          >
            {chip.label}
          </MouseTooltip>
        )
      })}
    </div>
  )
}

function uniqueTexts(values) {
  return Array.from(new Set(values.map(item => compactText(item, '')).filter(Boolean)))
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
  if (tabKey === 'slide_review') {
    return uniqueTexts(model.slideReviewModels)
  }
  return []
}

function stageModelChip(model, tabKey) {
  const models = stageModelList(model, tabKey)
  if (tabKey === 'claim_extraction' || tabKey === 'slide_review') {
    return {
      label: '모델',
      models: models.length ? models : [UNKNOWN_MODEL_LABEL],
      tooltipTitle: '사용 모델',
      tooltipVariant: 'value',
    }
  }
  return models.length ? { label: `${models.length}개 모델`, models } : ''
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

function CountBoard({ items }) {
  const rows = asArray(items).filter(item => compactText(item?.label, '') || item?.value !== undefined)
  if (!rows.length) return null

  return (
    <div className="vf-stage-data-board">
      {rows.map(item => (
        <div key={item.label}>
          <span>{compactText(item.label)}</span>
          <span className="vf-bold">{compactText(item.value)}</span>
        </div>
      ))}
    </div>
  )
}

function BreakdownRows({ rows, rowClassName = '' }) {
  const visibleRows = asArray(rows).filter(item => compactText(item?.label, '') || item?.value !== undefined)
  if (!visibleRows.length) return null

  return (
    <>
      {visibleRows.map(item => (
        <div key={item.label} className={`vf-breakdown-row ${item.compact ? 'vf-breakdown-row--compact' : ''} ${rowClassName}`}>
          <span className="vf-breakdown-label">
            <span className="vf-breakdown-label-text">
              <FlowLabel label={item.label} labelCount={item.labelCount} />
            </span>
            {item.help ? (
              <MouseTooltip
                className="vf-help-tooltip-wrap"
                tabIndex={0}
                ariaLabel={item.help}
                tooltip={item.help}
                tooltipClassName="vf-help-tooltip"
              >
                <span className="vf-help-icon" aria-hidden="true">?</span>
              </MouseTooltip>
            ) : null}
          </span>
          {item.value !== undefined ? <span className="vf-bold">{compactText(item.value)}</span> : null}
        </div>
      ))}
    </>
  )
}

function BreakdownList({ groups, rowClassName = '' }) {
  const visibleGroups = asArray(groups)
    .map(group => ({
      ...group,
      rows: asArray(group?.rows).filter(item => compactText(item?.label, '') || item?.value !== undefined),
    }))
    .filter(group => group.rows.length)

  if (!visibleGroups.length) return null

  return (
    <div className="vf-breakdown-list">
      {visibleGroups.map((group, groupIndex) => (
        <div
          key={group.key || groupIndex}
          className={`vf-breakdown-group ${groupIndex > 0 ? 'vf-breakdown-group--divider' : ''} ${group.indent ? 'vf-breakdown-group--indent' : ''} ${group.className || ''}`}
        >
          <BreakdownRows rows={group.rows} rowClassName={rowClassName} />
        </div>
      ))}
    </div>
  )
}

function FlowDetailRows({ rows, indent = false }) {
  const visibleRows = asArray(rows).filter(item => compactText(item?.label, '') || item?.value !== undefined)
  if (!visibleRows.length) return null

  return (
    <>
      {visibleRows.map(item => {
        return (
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
        )
      })}
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

function StageSummaryGrid({ model, statuses }) {
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
              <FlowDetailExpansion
                className="vf-flow-detail-trigger"
                detail={claimDetail}
              >
                {({ isOpen, toggleDetail, detailContent }) => (
                  <>
                    <div className="vf-flow-summary-column-head-row">
                      <div className="vf-flow-summary-column-head">주장 추출</div>
                      <button
                        type="button"
                        className="vf-flow-summary-detail-hint"
                        aria-expanded={isOpen}
                        aria-label="주장 추출 상세정보"
                        onClick={toggleDetail}
                      >
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
              <FlowDetailExpansion
                className="vf-flow-detail-trigger"
                detail={issueDetail}
              >
                {({ isOpen, toggleDetail, detailContent }) => (
                  <>
                    <div className="vf-flow-summary-column-head-row">
                      <div className="vf-flow-summary-column-head">이슈 후보 판단</div>
                      <button
                        type="button"
                        className="vf-flow-summary-detail-hint"
                        aria-expanded={isOpen}
                        aria-label="이슈 후보 판단 상세정보"
                        onClick={toggleDetail}
                      >
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
              <FlowDetailExpansion
                className="vf-flow-detail-trigger"
                detail={typeDetail}
              >
                {({ isOpen, toggleDetail, detailContent }) => (
                  <>
                    <div className="vf-flow-summary-column-head-row">
                      <div className="vf-flow-summary-column-head">이슈 유형 분류</div>
                      <button
                        type="button"
                        className="vf-flow-summary-detail-hint"
                        aria-expanded={isOpen}
                        aria-label="이슈 유형 분류 상세정보"
                        onClick={toggleDetail}
                    >
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
              <FlowDetailExpansion
                className="vf-flow-detail-trigger"
                detail={finalDetail}
              >
                {({ isOpen, toggleDetail, detailContent }) => (
                  <>
                    <div className="vf-flow-summary-column-head-row">
                      <div className="vf-flow-summary-column-head">최종 평가</div>
                      <button
                        type="button"
                        className="vf-flow-summary-detail-hint"
                        aria-expanded={isOpen}
                        aria-label="최종 평가 상세정보"
                        onClick={toggleDetail}
                      >
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

function FlowReportSection({ model, statuses, activeFilter, onSelectFilter, onSelectTab }) {
  return (
    <section className="vf-overview" aria-label="흐름 보고서">
      <div className="vf-stage-data-head">
        <span className="vf-bold">흐름 보고서</span>
      </div>
      <StageSummaryGrid
        model={model}
        statuses={statuses}
      />
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

function StageDataDetail({ activeTab, model, statuses, slideStatus }) {
  if (activeTab === 'slide_review') {
    const isReady = hasDataForTab(model, 'slide_review')
    const summary = asObject(model.slideErrors.summary)
    return (
      <div className="vf-stage-data-panel">
        <div className="vf-stage-data-panel-head">
          <div>
            <span>슬라이드</span>
            <span className="vf-bold">슬라이드 오류</span>
          </div>
          <HeaderChips chips={[isReady ? stageModelChip(model, 'slide_review') : '']} />
        </div>
        <MetricStrip items={[
          { label: '슬라이드', value: isReady ? summary.total_slide_count ?? '-' : '-' },
          { label: '오류', value: isReady ? model.slideFindings.length : '-' },
          { label: '보고 대상', value: isReady ? summary.reportable_error_count ?? '-' : '-' },
          { label: '유형', value: isReady ? Object.keys(summary.breakdown_by_type || {}).length : '-' },
        ]} />
        <CountBoard items={Object.entries(asObject(summary.breakdown_by_type)).map(([label, value]) => ({ label: typeLabel(label), value }))} />
      </div>
    )
  }

  const status = getFilterTabStatus(activeTab, statuses)
  const isReady = hasDataForTab(model, activeTab)
  const step = VERIFY_STEPS.find(item => item.key === activeTab)

  if (activeTab === 'issue_judge') {
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
      <div className={`vf-stage-data-panel vf-stage-data-panel--${status}`}>
        <div className="vf-stage-data-panel-head">
          <div>
            <span>02</span>
            <span className="vf-bold">{step?.label}</span>
          </div>
        </div>
        <div className="vf-breakdown-list vf-issue-judge-breakdown">
          <div className="vf-breakdown-group">
            <BreakdownRows rows={[
              { label: '평가 모델', labelCount: modelCountValue(totalModelCount, isReady), compact: true },
            ]} />
          </div>
          <div className="vf-breakdown-group vf-breakdown-group--divider vf-breakdown-group--indent">
            <BreakdownRows rows={modelRows} />
          </div>
          <div className="vf-breakdown-group vf-breakdown-group--divider vf-breakdown-group--indent">
            <BreakdownRows rows={issueJudgeClaimCheckMetrics(model, isReady)} />
          </div>
        </div>
      </div>
    )
  }

  if (activeTab === 'issue_classification') {
    return null
  }

  if (activeTab === 'final_verification') {
    const finalIssueCount = finalReviewNeededCount(model)
    const problemThresholdCount = finalProblemThresholdCount(model.severityItems)
    const modelDisagreementCount = finalModelDisagreementCount(model.severityItems)
    const lowMarginCount = finalLowMarginCount(model.severityItems)
    return (
      <div className={`vf-stage-data-panel vf-stage-data-panel--${status}`}>
        <div className="vf-stage-data-panel-head">
          <div>
            <span>04</span>
            <span className="vf-bold">{step?.label}</span>
          </div>
        </div>
        <div className="vf-breakdown-list">
          <div className="vf-breakdown-group">
            <BreakdownRows rows={[
              {
                label: '최종 이슈',
                value: isReady ? finalIssueCount : '-',
                help: '최종 이슈는 문제 기준 초과, 모델 의견 불합치, 분류 모호함 조건에 해당하는 항목의 합집합입니다. 조건이 함께 적용될 수 있어 아래 값의 합과 다를 수 있습니다.',
              },
            ]} />
          </div>
          <div className="vf-breakdown-group vf-breakdown-group--divider vf-breakdown-group--indent">
            <BreakdownRows rows={[
              { label: '문제 기준 초과', value: isReady ? problemThresholdCount : '-' },
              { label: '모델 의견 불합치', value: isReady ? modelDisagreementCount : '-' },
              { label: '분류 모호함', value: isReady ? lowMarginCount : '-' },
            ]} />
          </div>
        </div>
      </div>
    )
  }

  const resolvedClaims = model.claims.filter(claim => claim.resolution_status === 'resolved').length
  const contextNeeded = model.claims.filter(claim => claim.needs_context).length
  const approximate = model.claims.filter(claim => claim.is_approximate).length
  const claimTypes = model.claims.reduce((counts, claim) => {
    const key = compactText(claim.claim_type || 'claim')
    counts[key] = (counts[key] || 0) + 1
    return counts
  }, {})
  const claimTypeRows = Object.entries(claimTypes).map(([label, value]) => ({ label: typeLabel(label), value }))

  return (
    <div className={`vf-stage-data-panel vf-stage-data-panel--${status}`}>
      <div className="vf-stage-data-panel-head">
        <div>
          <span>01</span>
          <span className="vf-bold">{step?.label}</span>
        </div>
      </div>
      <BreakdownList
        groups={[
          {
            key: 'claims',
            rows: [
              { label: 'claims', value: isReady ? model.claims.length : '-' },
            ],
          },
          {
            key: 'resolution',
            indent: true,
            rows: [
              { label: 'resolved', value: isReady ? resolvedClaims : '-' },
              { label: 'needs_context', value: isReady ? contextNeeded : '-' },
            ],
          },
          {
            key: 'approx',
            indent: true,
            rows: [
              { label: 'approx', value: isReady ? approximate : '-' },
            ],
          },
          {
            key: 'claim_type',
            indent: true,
            rows: isReady ? claimTypeRows : [{ label: 'claim_type', value: '-' }],
          },
        ]}
      />
    </div>
  )
}

function StageDataSection({ activeTab, model, statuses, slideStatus }) {
  return (
    <section className="vf-stage-data" aria-label="단계별 데이터">
      <div className="vf-stage-data-head">
        <span className="vf-bold">단계별 데이터</span>
      </div>
      <StageDataDetail
        activeTab={activeTab}
        model={model}
        statuses={statuses}
        slideStatus={slideStatus}
      />
    </section>
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
        const topLabel = compactText(item.judgment || item.top_issue_type_label || item.top_issue_type || item.final_model_score, '')
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
    <div className="vf-context-panel">
      {imageUrl && (
        <div className="vf-context-media">
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

function StageTimeline({ flow, statuses = [], activeTab, onSelectTab }) {
  return (
    <div className="vf-report-rail vf-report-rail--tabs">
      {VERIFY_STEPS.map((step, index) => {
        const status = statuses[index] || getVerifyStepStatus(flow, index)
        const isActive = activeTab === step.key
        const isContentStep = index < 4
        const linkStatus = status === 'done' ? 'done' : 'idle'
        return (
          <Fragment key={step.key}>
            {index === 4 && <span className="vf-report-step-divider" aria-hidden="true" />}
            <button
              type="button"
              className={`vf-report-step vf-report-step--${status} ${isActive ? 'vf-report-step--active' : ''} ${isContentStep ? 'vf-report-step--content' : ''}`}
              onClick={() => onSelectTab(step.key)}
              aria-pressed={isActive}
            >
              <div>
                <span className="vf-bold">{step.label}</span>
                <span className="vf-report-step-status">{statusText(status)}</span>
              </div>
            </button>
            {index < 3 && <span className={`vf-report-step-link vf-report-step-link--${linkStatus}`} aria-hidden="true" />}
          </Fragment>
        )
      })}
    </div>
  )
}

function getFilterTabStatus(tabKey, statuses) {
  if (tabKey === 'all_claims') return statuses[0]
  if (tabKey === 'slide_review') return statuses[4]
  const index = VERIFY_STEPS.findIndex(step => step.key === tabKey)
  return index >= 0 ? statuses[index] : 'wait'
}

function getClaimFlowRowsForTab(model, activeTab) {
  if (activeTab === 'issue_classification') {
    return model.claimFlowRows.filter(row => row.type)
  }
  if (activeTab === 'final_verification') {
    return model.claimFlowRows.filter(row => row.severity)
  }
  return model.claimFlowRows
}

export function getClaimFlowSource(row) {
  return row.claim || row.comparison || row.issue || row.type || row.severity || {}
}

function transcriptClaimToneForRow(row, activeTab) {
  if (!row) return ''
  if (activeTab === 'claim_extraction') return 'claim-extraction'
  if (activeTab === 'issue_judge') {
    return row.issue || hasIssueInComparison(row) ? 'issue-judge' : 'muted'
  }
  if (activeTab === 'final_verification') {
    if (isFinalReviewTarget(row.severity)) return 'final-review'
    return 'muted'
  }
  return issueTypeTone(getIssueType(row.type || row.severity || row.issue))
}

function transcriptClaimFromRow(row, activeTab) {
  return {
    ...getClaimFlowSource(row),
    _transcriptTone: transcriptClaimToneForRow(row, activeTab),
  }
}

function claimFlowContextId(row) {
  const source = getClaimFlowSource(row)
  return compactText(source.context_id || asArray(source.context_ids)[0], '-')
}

function claimFlowSceneId(row) {
  const source = getClaimFlowSource(row)
  const explicit = compactText(source.scene_id || source.scene_key, '')
  if (explicit) return explicit
  const contextId = claimFlowContextId(row)
  return sceneIdFromContextId(contextId) || compactText(source.scene_index || source.scene_number, '-')
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

export function hasIssueInComparison(row) {
  return Object.values(asObject(row.comparison?.models)).some(item => item?.has_issue)
}

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

function finalReviewReasonLabels(severity) {
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

export function isFinalReviewTarget(severity) {
  return finalReviewReasonLabels(severity).length > 0
}

function finalReviewConditionTooltip() {
  return (
    <span className="vf-final-condition-tooltip">
      <span>강의자 검토가 필요한 경우</span>
      <span>문제 기준 초과: 최종 점수 &gt; 0.20</span>
      <span>모델 의견 불합치: 모델 불일치 &gt;= 0.35</span>
      <span>분류 모호함: 유형 margin &lt; 0.10</span>
    </span>
  )
}

function finalProblemThresholdCount(items) {
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

function finalModelDisagreementCount(items) {
  return asArray(items).filter(item => {
    const disagreement = Number(item?.model_disagreement ?? item?.classified_issue_verifier?.model_disagreement)
    return Number.isFinite(disagreement) && disagreement >= MANUAL_REVIEW_DISAGREEMENT_THRESHOLD
  }).length
}

function finalLowMarginCount(items) {
  return asArray(items).filter(item => {
    const margin = Number(item?.previous_classification?.margin)
    return item?.previous_classification?.low_margin || (Number.isFinite(margin) && margin < LOW_MARGIN_THRESHOLD)
  }).length
}

export function RowMeta({ items }) {
  const visibleItems = items.filter(item => compactText(item.value, ''))
  if (!visibleItems.length) return null

  return (
    <div className="vf-claim-row-meta">
      {visibleItems.map(item => (
        <span
          key={`${item.label || 'flag'}-${item.value}`}
          className={`vf-claim-row-meta-item ${item.tone ? `vf-claim-row-meta-item--${item.tone}` : ''}`}
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
      <span className={`vf-issue-judge-status vf-issue-judge-status--empty ${reserveModelCount ? 'vf-issue-judge-status--with-count' : ''}`} aria-hidden="true">
        <span className="vf-issue-judge-status-count" />
        <span className="vf-issue-judge-status-dot" />
      </span>
    )
  }
  const label = needsReview ? 'review needed' : 'issue'
  const tone = needsReview ? 'review' : 'issue'

  return (
    <span className={`vf-issue-judge-status vf-issue-judge-status--${tone} ${showModelCount ? 'vf-issue-judge-status--with-count' : ''} ${showFinalReasons ? 'vf-issue-judge-status--with-reasons' : ''}`} aria-label={label}>
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

function claimContextIds(claim) {
  return uniqueTexts([
    claim?.context_id,
    ...asArray(claim?.context_ids),
  ])
}

function claimsByContextId(claims) {
  const byContext = new Map()
  asArray(claims).forEach(claim => {
    claimContextIds(claim).forEach(contextId => {
      if (!byContext.has(contextId)) byContext.set(contextId, [])
      byContext.get(contextId).push(claim)
    })
  })
  return byContext
}

function claimTranscriptCandidates(claim) {
  return uniqueTexts([
    claim?.claim_text,
    claim?.resolved_claim,
  ])
}

function transcriptSceneKey(entry) {
  const fallbackKey = compactText(entry?.context_id, 'scene')
  return compactText(
    entry?.scene_id || sceneIdFromIndex(firstFilled(entry?.scene_index, entry?.scene_number, idNumber(entry?.context_id, 'SC'))),
    fallbackKey,
  )
}

function transcriptSceneNumber(scene, index) {
  const first = asArray(scene?.entries)[0] || {}
  const fromId = idNumber(first.context_id, 'SC')
  const value = idNumber(scene?.scene_id, 'SC')
    || firstFilled(scene?.scene_index, first.scene_index, scene?.scene_number, first.scene_number, fromId)
  return value || index + 1
}

function transcriptSceneTitle(scene, index) {
  return `장면 ${transcriptSceneNumber(scene, index)}`
}

function transcriptSceneMeta(scene, claimCount) {
  return `발화 문맥 ${asArray(scene?.entries).length}개 · 주장 ${claimCount}개`
}

function transcriptTimeRange(start, end) {
  const startText = formatTime(start)
  const endText = formatTime(end)
  if (startText && endText) return `${startText}~${endText}`
  return startText || endText
}

function transcriptContextLabel(entry, index) {
  const idValue = idNumber(entry?.context_id, 'C')
  if (idValue) return idValue
  const contextIndex = Number(entry?.context_index)
  const number = Number.isFinite(contextIndex) ? contextIndex + 1 : index + 1
  return number
}

function groupTranscriptEntriesByScene(entries) {
  const groups = []
  const byKey = new Map()
  asArray(entries).forEach(entry => {
    const key = transcriptSceneKey(entry)
    if (!byKey.has(key)) {
      const group = {
        key,
        scene_id: entry?.scene_id || sceneIdFromIndex(firstFilled(entry?.scene_index, entry?.scene_number, idNumber(entry?.context_id, 'SC'))),
        slide_number: entry?.slide_number,
        scene_index: entry?.scene_index ?? entry?.scene_number,
        slide_title: entry?.slide_title || '',
        slide_image_path: entry?.slide_image_path,
        slide_image_url: entry?.slide_image_url,
        start_time: entry?.start_time,
        end_time: entry?.end_time,
        entries: [],
      }
      byKey.set(key, group)
      groups.push(group)
    }
    byKey.get(key).entries.push(entry)
  })
  return groups
}

function transcriptSegmentsForEntry(entry, claims) {
  const text = compactText(entry?.text, '')
  const matches = []
  asArray(claims).forEach(claim => {
    const claimText = claimTranscriptCandidates(claim).find(candidate => text.includes(candidate))
    if (!claimText) return
    const index = text.indexOf(claimText)
    if (index < 0) return
    matches.push({
      claim,
      text: claimText,
      start: index,
      end: index + claimText.length,
    })
  })
  matches.sort((left, right) => left.start - right.start || left.end - right.end)

  const segments = []
  let cursor = 0
  matches.forEach(match => {
    if (match.start < cursor) return
    const before = text.slice(cursor, match.start)
    if (before) segments.push({ type: 'text', text: before })
    segments.push({ type: 'claim', text: match.text, claim: match.claim })
    cursor = match.end
  })
  const after = text.slice(cursor)
  if (after) segments.push({ type: 'text', text: after })

  return segments.length ? segments : [{ type: 'text', text }]
}

function sceneClaimCount(scene, byContext) {
  return uniqueTexts(asArray(scene?.entries).flatMap(entry => (
    asArray(byContext.get(entry.context_id)).map(claim => claimKey(claim))
  ))).length
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

function TranscriptSourcePanel({ model, resultId, rows, activeTab, onClaimClick }) {
  const sceneGroups = asArray(model.transcriptScenes).length
    ? asArray(model.transcriptScenes)
    : groupTranscriptEntriesByScene(model.transcriptEntries)
  if (!sceneGroups.length) return null

  const byContext = claimsByContextId(asArray(rows).length ? rows.map(row => transcriptClaimFromRow(row, activeTab)) : model.claims)

  return (
    <div className="vf-record-list vf-source-transcript-list">
      {sceneGroups.map((group, groupIndex) => {
        const entries = asArray(group.entries)
        const first = entries[0] || {}
        const imageUrl = resultFileUrl(group.slide_image_url || group.slide_image_path || first.slide_image_url || first.slide_image_path, resultId)
        const claimCount = sceneClaimCount(group, byContext)
        const title = transcriptSceneTitle(group, groupIndex)
        const meta = transcriptSceneMeta(group, claimCount)
        return (
          <article key={group.key} className="vf-record vf-source-transcript-scene">
            <div className="vf-source-transcript-scene-head">
              <span className="vf-bold">{title}</span>
              <span>| {meta}</span>
            </div>
            <div className="vf-source-transcript-scene-body">
              {imageUrl && (
                <aside className="vf-source-transcript-side">
                  <div className="vf-source-transcript-thumb">
                    <img src={imageUrl} alt={compactText(group.slide_title || first.slide_title || title || 'scene thumbnail')} onError={hideMissingImage} />
                  </div>
                </aside>
              )}
              <div className="vf-source-transcript-main">
                {entries.length ? entries.map((entry, entryIndex) => {
                  const claims = byContext.get(entry.context_id) || []
                  const segments = transcriptSegmentsForEntry(entry, claims)
                  let claimSegmentIndex = 0
                  const timeRange = transcriptTimeRange(entry.start_time, entry.end_time)
                  return (
                    <div key={`${entry.context_id || 'context'}-${entryIndex}`} className="vf-source-transcript-context-block">
                      <span className="vf-bold vf-source-transcript-context-id">{transcriptContextLabel(entry, entryIndex)}</span>
                      <p className="vf-source-transcript-context">
                        {segments.map((segment, segmentIndex) => {
                          if (segment.type === 'claim') {
                            claimSegmentIndex += 1
                            const claimIndexLabel = idNumber(claimKey(segment.claim), 'CL') || claimSegmentIndex
                            const claimTooltip = compactText(segment.claim?.resolved_claim || segment.claim?.claim_text || segment.text)
                            const claimTooltipContent = (
                              <>
                                <div>추출된 주장</div>
                                <p>{claimTooltip}</p>
                              </>
                            )
                            return (
                              <MouseTooltip
                                key={`${claimKey(segment.claim)}-${segmentIndex}`}
                                className={`vf-source-transcript-claim ${segment.claim?._transcriptTone ? `vf-source-transcript-claim--${safeDomId(segment.claim._transcriptTone)}` : ''}`}
                                tooltip={claimTooltipContent}
                                tooltipClassName="vf-source-transcript-claim-tooltip"
                                tabIndex={0}
                                ariaLabel={claimTooltip}
                                onClick={() => onClaimClick ? onClaimClick(claimKey(segment.claim)) : jumpToClaimRow(claimKey(segment.claim))}
                              >
                                <span className="vf-source-transcript-claim-index">{claimIndexLabel}</span>
                                {segment.text}
                              </MouseTooltip>
                            )
                          }
                          return (
                            <Fragment key={`text-${segmentIndex}`}>{segment.text}</Fragment>
                          )
                        })}
                        {timeRange && <span className="vf-source-transcript-time"> ({timeRange})</span>}
                      </p>
                    </div>
                  )
                }) : (
                  <div className="vf-source-transcript-empty">발화 없음</div>
                )}
              </div>
            </div>
          </article>
        )
      })}
    </div>
  )
}

function ClaimFlowRecord({ row, activeTab, resultId, displayId }) {
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

function ClaimExtractionGroupedList({ rows, resultId }) {
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
                  <ClaimFlowRecord
                    row={row}
                    activeTab="claim_extraction"
                    resultId={resultId}
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

function ClaimFlowFilterPanel({ model, activeTab, statuses, resultId, filterScope, activeFilter }) {
  const [claimListView, setClaimListView] = useState('list')
  const [issueJudgeListFilter, setIssueJudgeListFilter] = useState('all')
  const [finalListFilter, setFinalListFilter] = useState('all')
  const sourceRows = getClaimFlowRowsForTab(model, activeTab)
  const scopedRows = filterRows(sourceRows, filterScope, 'claim_flow')
  const isIssueJudge = activeTab === 'issue_judge'
  const isFinalVerification = activeTab === 'final_verification'
  const listRows = isIssueJudge && issueJudgeListFilter === 'candidate'
    ? scopedRows.filter(row => row.issue)
    : isFinalVerification && finalListFilter === 'needs_review'
      ? scopedRows.filter(row => isFinalReviewTarget(row.severity))
      : scopedRows
  const rows = claimListView === 'transcript' ? scopedRows : listRows
  const finalSourceCount = isFinalVerification
    ? sourceRows.filter(row => isFinalReviewTarget(row.severity)).length
    : sourceRows.length
  const finalRowsCount = isFinalVerification
    ? rows.filter(row => isFinalReviewTarget(row.severity)).length
    : rows.length
  const displayedCount = claimListView === 'transcript' ? rows.length : finalRowsCount
  const totalCount = claimListView === 'transcript' ? sourceRows.length : finalSourceCount
  const tab = VERIFY_STEPS.find(item => item.key === activeTab)
  const status = getFilterTabStatus(activeTab, statuses)
  const hasRows = sourceRows.length > 0
  const hasLocalListFilter = claimListView === 'list' && (
    (isIssueJudge && issueJudgeListFilter !== 'all') ||
    (isFinalVerification && finalListFilter !== 'all')
  )
  const countLabel = activeFilter || hasLocalListFilter || displayedCount !== totalCount
    ? `${displayedCount} / ${totalCount}`
    : `${displayedCount}`
  const canShowTranscript = asArray(model.transcriptScenes).length > 0 || asArray(model.transcriptEntries).length > 0
  const handleTranscriptClaimClick = claimId => {
    setClaimListView('list')
    if (typeof window !== 'undefined') window.setTimeout(() => jumpToClaimRow(claimId), 0)
  }
  const viewSwitchControl = canShowTranscript ? (
    <div className="vf-claim-view-switch vf-claim-view-switch--inline" aria-label="클레임 목록 보기 방식">
      <button
        type="button"
        className={claimListView === 'transcript' ? 'vf-claim-view-switch-btn--active' : ''}
        aria-pressed={claimListView === 'transcript'}
        onClick={() => setClaimListView(prev => prev === 'transcript' ? 'list' : 'transcript')}
      >
        {claimListView === 'transcript' ? '목록에서 보기' : '내용에서 보기'}
      </button>
    </div>
  ) : null
  const listFilterControl = claimListView === 'list' && isIssueJudge ? (
    <div className="vf-claim-view-switch vf-claim-view-switch--inline" aria-label="이슈 후보 판단 필터">
      <button
        type="button"
        className={issueJudgeListFilter === 'all' ? 'vf-claim-view-switch-btn--active' : ''}
        aria-pressed={issueJudgeListFilter === 'all'}
        onClick={() => setIssueJudgeListFilter('all')}
      >
        전체
      </button>
      <button
        type="button"
        className={issueJudgeListFilter === 'candidate' ? 'vf-claim-view-switch-btn--active' : ''}
        aria-pressed={issueJudgeListFilter === 'candidate'}
        onClick={() => setIssueJudgeListFilter('candidate')}
      >
        이슈 후보
      </button>
    </div>
  ) : claimListView === 'list' && isFinalVerification ? (
    <div className="vf-claim-view-switch vf-claim-view-switch--inline" aria-label="최종 평가 필터">
      <button
        type="button"
        className={finalListFilter === 'all' ? 'vf-claim-view-switch-btn--active' : ''}
        aria-pressed={finalListFilter === 'all'}
        onClick={() => setFinalListFilter('all')}
      >
        전체
      </button>
      <button
        type="button"
        className={finalListFilter === 'needs_review' ? 'vf-claim-view-switch-btn--active' : ''}
        aria-pressed={finalListFilter === 'needs_review'}
        onClick={() => setFinalListFilter('needs_review')}
      >
        검토 필요
      </button>
    </div>
  ) : null

  if (!hasRows) {
    return (
      <section className={`vf-claim-flow vf-claim-flow--${status}`}>
        <div className="vf-claim-flow-head">
          <div>
            <span>클레임 목록</span>
            <div className="vf-claim-flow-title-row">
              <h2>{tab?.label || '검증 항목'}</h2>
            </div>
          </div>
          <span className="vf-bold">{statusText(status)}</span>
        </div>
        <div className="vf-stage-pending">{pendingText(status, '이 단계의 결과가 생성되는 중입니다.', '이 단계 결과는 아직 없습니다.')}</div>
      </section>
    )
  }

  return (
    <>
      <section className={`vf-claim-flow vf-claim-flow--${status}`}>
        <div className="vf-claim-flow-head">
          <div className="vf-claim-flow-title">
            <span>클레임 목록</span>
            <div className="vf-claim-flow-title-row">
              <h2>{tab?.label || '검증 항목'}</h2>
              {isFinalVerification && (
                <MouseTooltip
                  className="vf-claim-flow-help-wrap"
                  tabIndex={0}
                  ariaLabel="최종 평가 조건"
                  tooltip={finalReviewConditionTooltip()}
                  tooltipClassName="vf-claim-flow-help-tooltip"
                >
                  <span className="vf-claim-flow-help-icon" aria-hidden="true">?</span>
                </MouseTooltip>
              )}
            </div>
          </div>
          <div className="vf-claim-flow-head-actions">
            {viewSwitchControl}
            {listFilterControl}
            <span className="vf-bold vf-claim-flow-count">{countLabel}</span>
          </div>
        </div>
        {rows.length ? (
          claimListView === 'transcript' && canShowTranscript ? (
            <TranscriptSourcePanel model={model} resultId={resultId} rows={rows} activeTab={activeTab} onClaimClick={handleTranscriptClaimClick} />
          ) : activeTab === 'claim_extraction' ? (
            <ClaimExtractionGroupedList rows={rows} resultId={resultId} />
          ) : (
            <div className="vf-record-list">
              {rows.map((row, rowIndex) => (
                  <ClaimFlowRecord
                    key={row.claim_id}
                    row={row}
                    activeTab={activeTab}
                    resultId={resultId}
                    displayId={claimValueFromId(row.claim_id, rowIndex)}
                  />
              ))}
            </div>
          )
        ) : <EmptyFiltered activeFilter={activeFilter} />}
      </section>
    </>
  )
}

function SlideErrorFilterPanel({ model, status, resultId, filterScope, activeFilter }) {
  const rows = filterRows(model.slideFindings, filterScope, 'slide')
  const headerChips = [model.slideFindings.length ? stageModelChip(model, 'slide_review') : '']

  if (!model.slideFindings.length) {
    return (
      <section className={`vf-claim-flow vf-claim-flow--${status}`}>
        <div className="vf-claim-flow-head">
          <div>
            <span>슬라이드 오류 목록</span>
            <div className="vf-claim-flow-title-row">
              <h2>슬라이드 오류</h2>
              <HeaderChips chips={headerChips} />
            </div>
          </div>
          <span className="vf-bold">{statusText(status)}</span>
        </div>
        <div className="vf-stage-pending">{pendingText(status, '슬라이드 오류 결과가 생성되는 중입니다.', '슬라이드 오류 결과는 아직 없습니다.')}</div>
      </section>
    )
  }

  return (
    <section className="vf-claim-flow vf-claim-flow--slide">
      <div className="vf-claim-flow-head">
        <div>
          <span>슬라이드 오류 목록</span>
          <div className="vf-claim-flow-title-row">
            <h2>슬라이드 오류</h2>
            <HeaderChips chips={headerChips} />
          </div>
        </div>
        <span className="vf-bold">{rows.length}</span>
      </div>
      {rows.length ? (
        <div className="vf-record-list">
          {rows.map((item, index) => {
            const imageUrl = resultFileUrl(item.slide_image_path, resultId)
            return (
              <article key={item.slide_error_id || index} className="vf-record vf-slide-report">
                {imageUrl && (
                  <div className="vf-slide-thumb">
                    <img src={imageUrl} alt={compactText(item.slide_title || '슬라이드 이미지')} onError={hideMissingImage} />
                  </div>
                )}
                <div className="vf-slide-detail">
                  <div className="vf-record-head">
                    <span className="vf-bold">{compactText(item.slide_error_id || `S${index + 1}`)}</span>
                    <ChipList items={[
                      item.slide_number ? `슬라이드 ${item.slide_number}` : '',
                      item.slide_title,
                      item.error_type_label || item.error_type,
                      `신뢰도 ${formatScore(item.confidence)}`,
                      `심각도 ${formatScore(item.severity_score)}`,
                      item.source,
                      item.model,
                    ]} />
                  </div>
                  <div className="vf-diff-row">
                    <div><span>문제 표기</span><span className="vf-bold">{compactText(item.problematic_text)}</span></div>
                    <div><span>수정 제안</span><span className="vf-bold">{compactText(item.corrected_text || item.suggested_fix)}</span></div>
                  </div>
                  <TextBlock label="근거">{item.reason}</TextBlock>
                  <TextBlock label="슬라이드 이미지">{item.slide_image_path}</TextBlock>
                </div>
              </article>
            )
          })}
        </div>
      ) : <EmptyFiltered activeFilter={activeFilter} />}
    </section>
  )
}

function renderActiveDetail({ activeTab, model, statuses, slideStatus, resultId, filterScope, activeFilter }) {
  if (activeTab === 'slide_review') {
    return <SlideErrorFilterPanel model={model} status={slideStatus} resultId={resultId} filterScope={filterScope} activeFilter={activeFilter} />
  }
  return <ClaimFlowFilterPanel model={model} activeTab={activeTab} statuses={statuses} resultId={resultId} filterScope={filterScope} activeFilter={activeFilter} />
}

export default function VerifyReportPanels({ flow, headerActions = null }) {
  const progressRef = useRef(null)
  const completedVerifyStageCount = getCompletedVerifyStageCount(flow)
  const visibleVerifier = useMemo(
    () => getVisibleVerifier(flow.verifier, completedVerifyStageCount),
    [flow.verifier, completedVerifyStageCount]
  )
  const visibleArtifacts = useMemo(
    () => getVisibleArtifacts(flow.verifierArtifacts, completedVerifyStageCount),
    [flow.verifierArtifacts, completedVerifyStageCount]
  )
  const model = useMemo(
    () => buildReportModel(visibleVerifier, visibleArtifacts),
    [visibleVerifier, visibleArtifacts]
  )
  const resultId = flow.lecture?.id || flow.verifier?.lecture_id || flow.verifier?.id || ''
  const pipelineStatuses = useMemo(() => VERIFY_STEPS.map((_, index) => getVerifyStepStatus(flow, index)), [flow])
  const statuses = useMemo(() => (
    VERIFY_STEPS.map((step, index) => displayStatusForTab(model, step.key, pipelineStatuses[index]))
  ), [model, pipelineStatuses])
  const slideStatus = displayStatusForTab(model, 'slide_review', pipelineStatuses[4])
  const completedDetailKey = getCompletedDetailKey(completedVerifyStageCount)
  const [activeTab, setActiveTab] = useState(completedDetailKey)
  const [activeFilter, setActiveFilter] = useState(null)
  const [isProgressDocked, setIsProgressDocked] = useState(false)
  const filterScope = useMemo(() => buildFilterScope(model, activeFilter), [model, activeFilter])

  useEffect(() => {
    setActiveTab(completedDetailKey)
  }, [completedDetailKey])

  useEffect(() => {
    if (activeFilter?.tab && activeFilter.tab !== activeTab) {
      setActiveFilter(null)
    }
  }, [activeFilter, activeTab])

  useEffect(() => {
    const progress = progressRef.current
    const scrollRoot = progress?.closest('.vf-flow-screen')
    if (!progress || !scrollRoot) return

    function updateDocked() {
      const progressRect = progress.getBoundingClientRect()
      const rootRect = scrollRoot.getBoundingClientRect()
      setIsProgressDocked(progressRect.bottom <= rootRect.top)
    }

    updateDocked()
    scrollRoot.addEventListener('scroll', updateDocked, { passive: true })
    window.addEventListener('resize', updateDocked)
    return () => {
      scrollRoot.removeEventListener('scroll', updateDocked)
      window.removeEventListener('resize', updateDocked)
    }
  }, [])

  function selectFilter(filter) {
    setActiveFilter(prev => isSameFilter(prev, filter) ? null : filter)
    if (filter?.tab) setActiveTab(filter.tab)
  }

  function selectTab(tab) {
    const tabIndex = REPORT_TAB_ORDER.indexOf(tab)
    if (tabIndex >= 0 && statuses[tabIndex] !== 'done') return
    setActiveTab(tab)
    setActiveFilter(null)
  }

  return (
    <div className="vf-report">
      <div className="vf-report-head">
        <div>
          <span>검증 파이프라인</span>
          <h1>{flow.lecture?.title || '강의 영상'}</h1>
        </div>
        {headerActions}
      </div>
      <section ref={progressRef} className="vf-progress-only" aria-label="검증 진행 상태">
        <div className="vf-progress-block-head">
          <span className="vf-bold">검증 진행 과정</span>
        </div>
        <StageTimeline flow={flow} statuses={statuses} activeTab={activeTab} onSelectTab={selectTab} />
      </section>
      {isProgressDocked && (
        <section className="vf-progress-dock" aria-label="검증 진행 상태">
          <div className="vf-progress-block-head">
            <span className="vf-bold">검증 진행 과정</span>
          </div>
          <StageTimeline flow={flow} statuses={statuses} activeTab={activeTab} onSelectTab={selectTab} />
        </section>
      )}
      {activeTab !== 'slide_review' && (
        <FlowReportSection
          model={model}
          statuses={statuses}
          activeFilter={activeFilter}
          onSelectFilter={selectFilter}
          onSelectTab={selectTab}
        />
      )}
      {activeTab === 'slide_review' && (
        <StageDataSection
          activeTab={activeTab}
          model={model}
          statuses={statuses}
          slideStatus={slideStatus}
        />
      )}
      <div className="vf-detail">
        {activeFilter && (
          <div className="vf-active-filter">
            <span>현재 보기</span>
            <span className="vf-bold">{activeFilter.label}</span>
            <button type="button" onClick={() => setActiveFilter(null)}>필터 해제</button>
          </div>
        )}
        <div className="vf-detail-body">
          {renderActiveDetail({ activeTab, model, statuses, slideStatus, resultId, filterScope, activeFilter })}
        </div>
      </div>
    </div>
  )
}
