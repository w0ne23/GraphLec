import { useEffect, useMemo, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { getLectureDetail, getLectureVerifier } from '../lib/api'

import VideoPlayer from '../components/watch/VideoPlayer'

import '../styles/verifier.css'

const VERIFIER_POLL_MS = 5000
const ISSUE_FILTERS = [
  { key: 'all', label: '전체' },
  { key: 'factual_error', label: '발언 자체 오류' },
  { key: 'temporal_error', label: '시대적 오류' },
  { key: 'scope_overclaim', label: '범위 과잉 단정' },
  { key: 'confusing_explanation', label: '혼동 가능 설명' },
]
const ISSUE_FILTER_DESCRIPTIONS = {
  factual_error: '문장 자체의 개념, 인과관계, 용어 연결, 수치가 강의 문맥을 봐도 틀린 경우입니다.',
  temporal_error: '현재 시점 기준으로 더 이상 유효하지 않거나 시대착오적인 정보를 현재도 맞는 것처럼 설명한 경우입니다.',
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

function labelForStage(stage) {
  const labels = {
    confirmed: '확정',
    professor_check: '교수 확인',
    review_needed: '교수 확인',
    rejected: '기각',
    final_confirmed: '확정',
    needs_review: '교수 확인',
    verifier_rejected: '검증 기각',
    agree: '확정',
    inconclusive: '교수 확인',
    disagree: '기각',
  }
  return labels[stage] || compactText(stage)
}

function labelForIssueType(type) {
  const labels = {
    factual_error: '발언 자체 오류',
    temporal_error: '시대적 오류',
    scope_overclaim: '범위 과잉 단정',
    confusing_explanation: '혼동 가능 설명',
  }
  return labels[type] || compactText(type)
}

function labelForIssueSubtype(type) {
  const labels = {
    factual_error: '발언 자체 오류',
    temporal_error: '시대적 오류',
    scope_overclaim: '범위 과잉 단정',
    confusing_explanation: '혼동 가능 설명',
  }
  return labels[type] || compactText(type)
}

function getIssueSubtype(item) {
  const direct = item.feedback_type || item.issue_type || item.type || item.issue_subtype
  if (direct === 'temporal_error') return 'temporal_error'
  if (direct === 'scope_overclaim') return 'scope_overclaim'
  if (direct === 'confusing_explanation') return 'confusing_explanation'
  if (direct === 'factual_error') return 'factual_error'
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
  const key = claimDisplayIssueKey(claim)
  return labelForIssueSubtype(key) || labelForIssueType(key)
}

function groupSlideErrorsBySlide(items) {
  const groups = new Map()
  asArray(items).forEach((error, idx) => {
    const slideNumber = error.slide_number ?? 'unknown'
    const key = String(slideNumber)
    if (!groups.has(key)) {
      groups.set(key, {
        key,
        slideNumber,
        title: error.slide_title || '',
        imageUrl: error.slide_image_url || error.image_url || '',
        items: [],
      })
    }

    const group = groups.get(key)
    if (!group.imageUrl && (error.slide_image_url || error.image_url)) {
      group.imageUrl = error.slide_image_url || error.image_url
    }
    if (!group.title && error.slide_title) {
      group.title = error.slide_title
    }
    group.items.push({ ...error, _slideErrorIndex: idx })
  })

  return Array.from(groups.values()).sort((a, b) => {
    const aNumber = Number(a.slideNumber)
    const bNumber = Number(b.slideNumber)
    if (Number.isFinite(aNumber) && Number.isFinite(bNumber)) return aNumber - bNumber
    return String(a.slideNumber).localeCompare(String(b.slideNumber))
  })
}

function getItemLocation(item, sourceClaim = {}) {
  return item.location || sourceClaim.location || {}
}

function getItemStartTime(item, sourceClaim = {}) {
  const location = getItemLocation(item, sourceClaim)
  const value = Number(location.start_time ?? item.start_time ?? sourceClaim.start_time)
  return Number.isFinite(value) ? value : undefined
}

function getConfirmationReason(item) {
  const direct =
    item.confirmation_reason ||
    item.evidence?.confirmation_reason
  if (direct) return direct

  const modelResults = item.checks?.severity?.model_results
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
    item.checks?.severity?.reason ||
    ''
  )
}

function getModelVerdicts(item) {
  const rows = asArray(item.checks?.severity?.model_results)
  return rows.reduce((acc, row) => {
    const model = row?.model || row?.resolved_model || row?.source_model
    if (model) acc[model] = row
    return acc
  }, {})
}

function getSeverityScoreFromModels(modelResults) {
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

function getSeverityScore(item, severity = {}) {
  return (
    toNumberOrUndefined(item.severity_score) ??
    toNumberOrUndefined(item.score) ??
    toNumberOrUndefined(severity.score) ??
    toNumberOrUndefined(severity.scoring?.score) ??
    getSeverityScoreFromModels(severity.model_results)
  )
}

function getSeverityScorePercent(item, severity = {}, score) {
  const direct = toNumberOrUndefined(item.severity_score_percent ?? severity.score_percent ?? severity.scoring?.score_percent)
  if (direct !== undefined) return direct
  return score !== undefined ? score * 100 : undefined
}

function statusFromScore(score) {
  if (score === undefined) return ''
  if (score >= 0.8) return 'confirmed'
  if (score >= 0.45) return 'professor_check'
  return 'rejected'
}

function scoreLabel(score) {
  return score !== undefined ? formatPercent(score) : ''
}

function feedbackItemToClaim(item, claimById) {
  const sourceClaim = claimById.get(item.source_claim_id) || {}
  const problem = item.problem || {}
  const feedback = item.professor_feedback || {}
  const evidence = item.evidence || {}
  const severity = item.checks?.severity || {}
  const severityScore = getSeverityScore(item, severity)
  const severityScorePercent = getSeverityScorePercent(item, severity, severityScore)
  const severityStatus =
    item.severity_status ||
    severity.status_by_score ||
    severity.scoring?.status ||
    statusFromScore(severityScore)
  const location = getItemLocation(item, sourceClaim)
  const status = item.status === 'review_needed' ? 'professor_check' : item.status
  const title =
    problem.problematic_content ||
    item.claim_text ||
    sourceClaim.claim_text ||
    item.resolved_claim ||
    sourceClaim.resolved_claim ||
    problem.summary ||
    '-'

  return {
    ...item,
    stage: status || 'professor_check',
    start_time: getItemStartTime(item, sourceClaim),
    slide_number: location.slide_number ?? evidence.slide_number,
    original_claim_text: item.claim_text || sourceClaim.claim_text,
    claim_text: title,
    resolved_claim: item.resolved_claim || sourceClaim.resolved_claim,
    issue: problem.summary || feedback.summary,
    correct_info: problem.correct_info,
    issue_type: item.feedback_type || item.issue_type || item.type,
    issue_category_label: item.feedback_label || item.issue_category_label,
    student_misunderstanding: feedback.student_misunderstanding,
    why_it_matters: feedback.why_it_matters,
    suggested_rephrase: feedback.suggested_rephrase,
    teaching_note: feedback.teaching_note,
    why_wrong: problem.why_wrong || feedback.why_wrong,
    issue_basis: problem.issue_basis || feedback.issue_basis,
    student_error: problem.student_error || feedback.student_error,
    counterexample_or_condition:
      problem.counterexample_or_condition ||
      evidence.counterexample_or_condition ||
      feedback.counterexample_or_condition,
    context_resolution: problem.context_resolution || evidence.context_resolution || feedback.context_resolution,
    recommendation: problem.recommendation || feedback.teaching_note,
    evidence_in_context: evidence.evidence_in_context || feedback.evidence_in_context,
    severity_score: severityScore,
    severity_score_percent: severityScorePercent,
    severity_verdict: item.severity_verdict ?? severity.verdict,
    severity_status: severityStatus,
    model_verdicts: getModelVerdicts(item),
    confirmation_reason: status === 'confirmed' ? getConfirmationReason(item) : '',
    rejection_reason: status === 'rejected' ? getRejectionReason(item) : item.professor_check_reason || item.review_reason,
    evidence_sources: evidence.evidence_sources || item.evidence_sources,
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
            <span>{filter.label}</span>
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
  const label = ISSUE_FILTERS.find((item) => item.key === filter)?.label || filter
  return (
    <div className="vf-filter-description">
      <strong>{label}</strong>
      <span>{description}</span>
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

function ModelVerdicts({ verdicts }) {
  const entries = Object.entries(verdicts || {})
  if (!entries.length) return null
  return (
    <div className="vf-evidence-block">
      <div className="vf-evidence-title">모델 판정</div>
      <div className="vf-verdict-grid">
        {entries.map(([model, verdict]) => (
          <div className="vf-verdict" key={model}>
            <span>{model}</span>
            <strong>점수 {formatPercent(verdict?.confidence ?? verdict?.vote_score)}</strong>
            {(verdict?.decision || verdict?.verdict || verdict?.status) && (
              <em>{compactText(verdict?.decision || verdict?.verdict || verdict?.status)}</em>
            )}
            {verdict?.model_weight !== undefined && <em>가중치 {Number(verdict.model_weight).toFixed(2)}</em>}
            {verdict?.weighted_score !== undefined && <em>반영점수 {Number(verdict.weighted_score).toFixed(2)}</em>}
          </div>
        ))}
      </div>
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
  const title = claim.original_claim_text || claim.claim_text || claim.resolved_claim || claim.problematic_content || '-'
  const startTime = Number(claim.start_time)
  const canWatch = Number.isFinite(startTime)
  const grounding = claim.grounding || {}
  const displayIssueKey = claimDisplayIssueKey(claim)
  const displayIssueLabel = labelForClaimIssue(claim)
  const sources = asArray(grounding.evidence_sources).length
    ? grounding.evidence_sources
    : asArray(claim.evidence_sources)
  const hasSeverityScore = claim.severity_score !== undefined && claim.severity_score !== null
  const severityStatus = claim.severity_status || claim.severity_verdict

  return (
    <article className={`vf-claim-card ${expanded ? 'vf-claim-card--expanded' : ''}`}>
      <button className="vf-claim-main" onClick={onToggle}>
        <div className="vf-claim-copy">
          <div className="vf-claim-title">{title}</div>
          <div className="vf-chip-row">
            <span className="vf-chip vf-chip--stage">{labelForStage(claim.stage || section)}</span>
            {displayIssueKey && <span className={`vf-chip vf-chip--${displayIssueKey}`}>{displayIssueLabel}</span>}
            {hasSeverityScore && (
              <span className={`vf-chip vf-chip--score vf-chip--score-${severityStatus || 'unknown'}`}>
                점수 {scoreLabel(claim.severity_score)}
              </span>
            )}
          </div>
        </div>
        <div className="vf-claim-meta">
          <span>{canWatch ? formatTime(startTime) : '-'}</span>
          {claim.slide_number !== undefined && <span>slide {claim.slide_number}</span>}
        </div>
      </button>

      {expanded && (
        <div className="vf-claim-detail">
          <dl>
            <DetailRow label="등장 시각" value={canWatch ? `${formatTime(startTime)} (${startTime.toFixed(2)}s)` : '-'} />
            <DetailRow label="상태" value={labelForStage(claim.stage || section)} />
            <DetailRow
              label="검증 점수"
              value={
                hasSeverityScore
                  ? `${formatPercent(claim.severity_score)} (${labelForStage(severityStatus) || '-'})`
                  : ''
              }
            />
            <DetailRow label="Claim" value={claim.resolved_claim || claim.claim_text} />
            <DetailRow label="문제 유형" value={displayIssueLabel} />
            <DetailRow label="문제점" value={claim.issue} />
            <DetailRow label="학생이 잘못 외울 수 있는 명제" value={claim.student_error} />
            <DetailRow label="반례/조건" value={claim.counterexample_or_condition || claim.counterexample} />
            <DetailRow label="문맥 해소 여부" value={claim.context_resolution} />
            <DetailRow label="학생 오해 가능성" value={claim.student_misunderstanding} />
            <DetailRow label="왜 중요한가" value={claim.why_it_matters} />
            <DetailRow label="권장 수정" value={claim.recommendation || claim.teaching_note} />
            <DetailRow label="문맥 근거" value={claim.evidence_in_context} />
            <DetailRow label="확정 사유" value={claim.confirmation_reason} />
            <DetailRow label="기각/검토 사유" value={claim.rejection_reason || claim.review_reason_code || claim.rejection_reason_code} />
            <DetailRow label="기각 단계" value={claim.rejection_stage} />
            <DetailRow label="Grounding" value={grounding.status || grounding.reason || claim.grounding_status} />
          </dl>
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

function SlideErrorItem({ error }) {
  const candidates = asArray(error.correction_candidates)
  const runCount = Number(error.run_count || 0)
  return (
    <div className="vf-error-item">
      <div className="vf-error-main">
        <div className="vf-error-fields">
          <div className="vf-error-correction">
            <span>수정</span>
            <strong>
              {compactText(error.problematic_text)}
              <em>→</em>
              {compactText(error.corrected_text)}
            </strong>
          </div>
          {error.error_type_label && (
            <div className="vf-error-field">
              <span>오류 유형</span>
              <p>{compactText(error.error_type_label, '')}</p>
            </div>
          )}
          <div className="vf-error-field">
            <span>이유</span>
            <p>{compactText(error.reason, '')}</p>
          </div>
        </div>
        <div className="vf-error-meta">
          {error.confidence !== undefined && <span>신뢰도 {formatPercent(error.confidence)}</span>}
          {runCount > 1 && <span>지지 {error.support_count || 0}/{runCount}</span>}
        </div>
      </div>
      {candidates.length > 1 && (
        <div className="vf-candidate-list">
          <b>후보 수정안</b>
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

function SlideErrorCard({ group, review = false }) {
  return (
    <article className={`vf-error-slide-card ${review ? 'vf-error-slide-card--review' : ''}`}>
      <div className={`vf-error-slide-layout ${group.imageUrl ? '' : 'vf-error-slide-layout--no-image'}`}>
        {group.imageUrl && (
          <div className="vf-error-slide-image">
            <img src={group.imageUrl} alt={`Slide ${group.slideNumber}`} loading="lazy" />
          </div>
        )}
        <div className="vf-error-slide-panel">
          <div className="vf-error-slide-head">
            <strong>slide {group.slideNumber}</strong>
            <span>{group.items.length}건</span>
          </div>
          <div className="vf-error-items">
            {group.items.map((error) => (
              <SlideErrorItem
                key={`${group.key}-${error.problematic_text}-${error.corrected_text}-${error._slideErrorIndex}`}
                error={error}
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
  const [activeTab, setActiveTab] = useState('confirmed')
  const [activeIssueFilter, setActiveIssueFilter] = useState('all')
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
      const finalClaims = normalized.filter((item) => item.stage === 'confirmed')
      const needsReview = normalized.filter((item) => item.stage === 'professor_check' || item.stage === 'review_needed')
      const rejected = normalized.filter((item) => item.stage === 'rejected')
      return {
        finalClaims,
        needsReview,
        slideErrors: asArray(verifier?.slide_errors),
        verifierRejected: rejected,
        filtered: rejected,
        usesFeedbackItems: true,
      }
    }

    const finalClaims = asArray(verifier?.final_confirmed_claims)
    const needsReview = asArray(verifier?.needs_review_claims)
    const verifierRejected = asArray(verifier?.verifier_rejected_claims)
    return {
      finalClaims,
      needsReview,
      slideErrors: asArray(verifier?.slide_errors),
      verifierRejected,
      filtered: verifierRejected,
      usesFeedbackItems: false,
    }
  }, [claimById, verifier])

  const counts = verifier?.counts || {}
  const finalCount = counts.final_confirmed ?? verifier?.final_confirmed_claim_count ?? sections.finalClaims.length
  const reviewCount = counts.needs_review ?? counts.professor_check ?? sections.needsReview.length
  const slideErrorCount = counts.slide_errors ?? sections.slideErrors.length
  const filteredCount = counts.rejected ?? counts.verifier_rejected ?? sections.filtered.length

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
          const claimKey = `${section}-${claim.feedback_id || claim.issue_id || claim.source_claim_id || claim.claim_id || claim.source_claim_key || 'claim'}-${idx}`
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

  function renderSlideErrorGroups(items, review = false) {
    return (
      <div className="vf-error-list">
        {groupSlideErrorsBySlide(items).map((group) => (
          <SlideErrorCard key={`${review ? 'review' : 'slide-error'}-${group.key}`} group={group} review={review} />
        ))}
      </div>
    )
  }

  function renderActivePanel() {
    if (activeTab === 'review') {
      const filteredReview = filterIssueClaims(sections.needsReview)
      return (
        <Section
          title="교수 확인이 필요한 내용 이슈"
          count={sections.needsReview.length}
          tone="review"
          empty="교수 확인이 필요한 내용 이슈가 없습니다."
        >
          <IssueTypeBreakdown
            items={sections.needsReview}
            section="needs_review"
            activeFilter={activeIssueFilter}
            onFilterChange={setActiveIssueFilter}
          />
          <IssueFilterDescription filter={activeIssueFilter} />
          {filteredReview.length > 0
            ? renderClaimList(filteredReview, 'needs_review')
            : <div className="vf-empty">선택한 유형의 교수 확인 이슈가 없습니다.</div>}
        </Section>
      )
    }

    if (activeTab === 'slideErrors') {
      return (
        <Section
          title="슬라이드 오류"
          count={sections.slideErrors.length}
          tone="error"
          empty="확정된 슬라이드 오류가 없습니다."
        >
          {renderSlideErrorGroups(sections.slideErrors)}
        </Section>
      )
    }

    if (activeTab === 'filtered') {
      if (sections.usesFeedbackItems) {
        return (
          <Section
            title="기각된 내용 후보"
            count={sections.filtered.length}
            empty="기각된 내용 후보가 없습니다."
          >
            {renderClaimList(sections.filtered, 'rejected')}
          </Section>
        )
      }

      return (
        <Section
          title="기각된 내용 후보"
          count={sections.filtered.length}
          empty="검증에서 기각된 내용 후보가 없습니다."
        >
          {renderClaimList(sections.filtered, 'verifier_rejected')}
        </Section>
      )
    }

    const filteredFinal = filterIssueClaims(sections.finalClaims)
    return (
      <Section
        title="확정된 내용 이슈"
        count={sections.finalClaims.length}
        tone="danger"
        empty="확정된 내용 이슈가 없습니다."
      >
        <IssueTypeBreakdown
          items={sections.finalClaims}
          section="final_confirmed"
          activeFilter={activeIssueFilter}
          onFilterChange={setActiveIssueFilter}
        />
        <IssueFilterDescription filter={activeIssueFilter} />
        {filteredFinal.length > 0
          ? renderClaimList(filteredFinal, 'final_confirmed')
          : <div className="vf-empty">선택한 유형의 확정 이슈가 없습니다.</div>}
      </Section>
    )
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
              label="확정 이슈"
              value={finalCount}
              tone="danger"
              active={activeTab === 'confirmed'}
              onClick={() => selectTab('confirmed')}
            />
            <SummaryMetric
              label="교수 확인"
              value={reviewCount}
              tone="review"
              active={activeTab === 'review'}
              onClick={() => selectTab('review')}
            />
            <SummaryMetric
              label="슬라이드 오류"
              value={slideErrorCount}
              tone="error"
              active={activeTab === 'slideErrors'}
              onClick={() => selectTab('slideErrors')}
            />
            <SummaryMetric
              label="기각"
              value={filteredCount}
              active={activeTab === 'filtered'}
              onClick={() => selectTab('filtered')}
            />
          </div>

          {renderActivePanel()}
        </section>
      </div>
    </div>
  )
}
