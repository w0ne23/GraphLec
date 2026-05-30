import { useEffect, useMemo, useRef, useState } from 'react'
import { VERIFY_STEPS } from './verifierConstants'
import SlideReviewReportPanel from './SlideReviewReportPanel'
import VerifyStageTimeline from './VerifyStageTimeline'
import ClaimListPanel from './stages/ClaimListPanel'
import {
  FinalReviewConditionTooltip,
  renderClaimExtractionRows,
  renderFinalVerificationRows,
  renderIssueClassificationRows,
  renderIssueJudgeRows,
} from './stages/ClaimListRows'
import SlideReviewStage from './stages/SlideReviewStage'
import {
  MouseTooltip,
} from './VerifyReportParts'

import { 
  asArray, asObject, compactText, 
  idNumber, sceneIdFromIndex,
  displayStatusForTab, firstFilled, getIssueType, 
  isFinalReviewTarget, claimKey, statusFromSeverityScore, uniqueTexts,
} from './verifierUtils'

const REPORT_TAB_ORDER = [
  'claim_extraction',
  'issue_judge',
  'issue_classification',
  'final_verification',
  'slide_review',
]

const UNKNOWN_MODEL_LABEL = '알 수 없음'

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

function ClaimListFilterActions({ options, value, onChange }) {
  return (
    <div className="vf-claim-view-switch vf-claim-view-switch--inline">
      {options.map(option => (
        <button
          key={option.key}
          type="button"
          className={value === option.key ? 'vf-claim-view-switch-btn--active' : ''}
          aria-pressed={value === option.key}
          onClick={() => onChange(option.key)}
        >
          {option.label}
        </button>
      ))}
    </div>
  )
}

function renderClaimListDetail({
  activeTab,
  model,
  status,
  resultId,
  claimListView,
  claimListFilters,
  onClaimListFilterChange,
  onClaimListViewChange,
}) {
  if (activeTab === 'claim_extraction') {
    const sourceRows = model.claimFlowRows
    const rows = sourceRows
    const displayedCount = rows.length
    const totalCount = sourceRows.length

    return (
      <ClaimListPanel
        activeTab={activeTab}
        countLabel={displayedCount !== totalCount ? `${displayedCount} / ${totalCount}` : totalCount}
        model={model}
        resultId={resultId}
        rows={rows}
        sourceRows={sourceRows}
        status={status}
        viewMode={claimListView}
        onViewModeChange={onClaimListViewChange}
      >
        {renderClaimExtractionRows(rows)}
      </ClaimListPanel>
    )
  }

  if (activeTab === 'issue_judge') {
    const sourceRows = model.claimFlowRows.filter(row => row.issue)
    const rows = sourceRows
    const sourceCount = sourceRows.length
    const displayedCount = rows.length

    return (
      <ClaimListPanel
        activeTab={activeTab}
        countLabel={displayedCount !== sourceCount
          ? `${displayedCount} / ${sourceCount}`
          : sourceCount}
        model={model}
        resultId={resultId}
        rows={rows}
        sourceRows={sourceRows}
        status={status}
        viewMode={claimListView}
        onViewModeChange={onClaimListViewChange}
      >
        {renderIssueJudgeRows(rows)}
      </ClaimListPanel>
    )
  }

  if (activeTab === 'issue_classification') {
    const sourceRows = model.claimFlowRows.filter(row => row.type)
    const rows = sourceRows
    const sourceCount = sourceRows.length
    const displayedCount = rows.length

    return (
      <ClaimListPanel
        activeTab={activeTab}
        countLabel={displayedCount !== sourceCount
          ? `${displayedCount} / ${sourceCount}`
          : sourceCount}
        model={model}
        resultId={resultId}
        rows={rows}
        sourceRows={sourceRows}
        status={status}
        viewMode={claimListView}
        onViewModeChange={onClaimListViewChange}
      >
        {renderIssueClassificationRows(rows)}
      </ClaimListPanel>
    )
  }

  if (activeTab === 'final_verification') {
    const listFilter = claimListFilters.final_verification || 'all'
    const sourceRows = model.claimFlowRows.filter(row => row.severity)
    const listRows = listFilter === 'needs_review'
      ? sourceRows.filter(row => isFinalReviewTarget(row.severity))
      : sourceRows
    const rows = claimListView === 'transcript' ? sourceRows : listRows
    const sourceCount = sourceRows.filter(row => isFinalReviewTarget(row.severity)).length
    const displayedCount = claimListView === 'transcript'
      ? rows.length
      : rows.filter(row => isFinalReviewTarget(row.severity)).length
    const totalCount = claimListView === 'transcript' ? sourceRows.length : sourceCount

    return (
      <ClaimListPanel
        activeTab={activeTab}
        actions={claimListView === 'list' ? (
          <ClaimListFilterActions
            options={[
              { key: 'all', label: '전체' },
              { key: 'needs_review', label: '검토 필요' },
            ]}
            value={listFilter}
            onChange={value => onClaimListFilterChange('final_verification', value)}
          />
        ) : null}
        countLabel={(claimListView === 'list' && listFilter !== 'all') || (claimListView === 'list' && displayedCount !== sourceCount)
          ? `${displayedCount} / ${totalCount}`
          : totalCount}
        model={model}
        resultId={resultId}
        rows={rows}
        sourceRows={sourceRows}
        status={status}
        titleAddon={(
          <MouseTooltip
            className="vf-claim-flow-help-wrap"
            tabIndex={0}
            ariaLabel="최종 평가 조건"
            tooltip={<FinalReviewConditionTooltip />}
            tooltipClassName="vf-claim-flow-help-tooltip"
          >
            <span className="vf-claim-flow-help-icon" aria-hidden="true">?</span>
          </MouseTooltip>
        )}
        viewMode={claimListView}
        onViewModeChange={onClaimListViewChange}
      >
        {renderFinalVerificationRows(rows)}
      </ClaimListPanel>
    )
  }

  return <div className="vf-report-empty">정의되지 않은 스테이지입니다.</div>
}

function renderActiveDetail({ activeTab, model, status, resultId, claimListView, claimListFilters, onClaimListFilterChange, onClaimListViewChange }) {
  if (activeTab === 'slide_review') {
    return <SlideReviewStage model={model} status={status} resultId={resultId} />
  }
  return renderClaimListDetail({
    activeTab,
    model,
    status,
    resultId,
    claimListView,
    claimListFilters,
    onClaimListFilterChange,
    onClaimListViewChange,
  })
}

export default function VerifyReportPanels({ flow, headerActions = null }) {
  const progressRef = useRef(null)
  const model = useMemo(
    () => buildReportModel(flow.verifier, flow.verifierArtifacts),
    [flow.verifier, flow.verifierArtifacts]
  )
  const resultId = flow.lecture?.id || flow.verifier?.lecture_id || flow.verifier?.id || ''
  const statuses = useMemo(() => (
    VERIFY_STEPS.map(step => displayStatusForTab(model, step.key, 'wait'))
  ), [model])
  const slideStatus = displayStatusForTab(model, 'slide_review', 'wait')
  const [activeTab, setActiveTab] = useState('claim_extraction')
  const [claimListView, setClaimListView] = useState('list')
  const [claimListFilters, setClaimListFilters] = useState({
    final_verification: 'all',
  })
  const [isProgressDocked, setIsProgressDocked] = useState(false)

  useEffect(() => {
    setClaimListFilters({
      final_verification: 'all',
    })
  }, [activeTab])

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

  function selectTab(tab) {
    const tabIndex = REPORT_TAB_ORDER.indexOf(tab)
    if (tabIndex >= 0 && statuses[tabIndex] !== 'done') return
    setActiveTab(tab)
  }

  function setClaimListFilter(tab, value) {
    setClaimListFilters(prev => ({ ...prev, [tab]: value }))
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
        <VerifyStageTimeline statuses={statuses} activeTab={activeTab} onSelectTab={selectTab} compact />
      </section>
      {isProgressDocked && (
        <section className="vf-progress-dock" aria-label="검증 진행 상태">
          <div className="vf-progress-block-head">
            <span className="vf-bold">검증 진행 과정</span>
          </div>
          <VerifyStageTimeline statuses={statuses} activeTab={activeTab} onSelectTab={selectTab} compact />
        </section>
      )}
      {activeTab === 'slide_review' && (
        <SlideReviewReportPanel model={model} />
      )}
      <div className="vf-detail">
        <div className="vf-detail-body">
          {renderActiveDetail({ 
            activeTab, 
            model, 
            status: activeTab === 'slide_review' ? slideStatus : statuses[REPORT_TAB_ORDER.indexOf(activeTab)], 
            resultId, 
            claimListView,
            claimListFilters,
            onClaimListFilterChange: setClaimListFilter,
            onClaimListViewChange: setClaimListView,
          })}
        </div>
      </div>
    </div>
  )
}
