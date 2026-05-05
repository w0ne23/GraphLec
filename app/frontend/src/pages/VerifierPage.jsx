import { useEffect, useMemo, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { getLectureDetail, getLectureVerifier } from '../lib/api'

import VideoPlayer from '../components/watch/VideoPlayer'

import '../styles/verifier.css'

const VERIFIER_POLL_MS = 5000
const ISSUE_FILTERS = [
  { key: 'all', label: '전체' },
  { key: 'simple_factual_error', label: '단순 사실 오류' },
  { key: 'scope_error', label: '범위 오류' },
  { key: 'outdated', label: '현행성 오류' },
]
const ISSUE_FILTER_DESCRIPTIONS = {
  simple_factual_error: '문장 자체의 개념, 인과관계, 용어 연결이 부정확한 경우입니다. 범위 표현의 과장보다는 핵심 사실 관계가 틀린 항목을 모았습니다.',
  scope_error: '항상, 모든, 오직, 반드시 같은 표현 때문에 예외나 조건이 사라져 오해될 수 있는 경우입니다. 강의 맥락상 맞는 설명이어도 범위가 과하게 들리면 여기에 포함됩니다.',
  outdated: '현재 날짜 기준으로 더 이상 유효하지 않거나 폐기된 정보를 현재도 맞는 것처럼 설명한 경우입니다.',
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

function compactText(value, fallback = '-') {
  const text = String(value ?? '').trim()
  return text || fallback
}

function labelForStage(stage) {
  const labels = {
    final_confirmed: '확정',
    needs_review: '검토 필요',
    crosscheck_rejected: '교차검증 기각',
    crosscheck_inconclusive: '교차검증 불확실',
    slide_rejected: '슬라이드 기각',
    grounding_rejected: '근거 기각',
    first_stage_rejected: '1차 제외',
  }
  return labels[stage] || compactText(stage)
}

function labelForIssueType(type) {
  const labels = {
    factual_error: '사실 오류',
    outdated: '현행성 오류',
  }
  return labels[type] || compactText(type)
}

function labelForIssueSubtype(type) {
  const labels = {
    simple_factual_error: '단순 사실 오류',
    scope_error: '범위 오류',
    outdated: '현행성 오류',
  }
  return labels[type] || compactText(type)
}

function getIssueSubtype(item) {
  if (item.issue_subtype) return item.issue_subtype
  if (item.issue_pattern === 'scope_overstatement') return 'scope_error'
  if ((item.issue_type || item.type) === 'outdated') return 'outdated'
  if ((item.issue_type || item.type) === 'factual_error') return 'simple_factual_error'
  return ''
}

function countIssueTypes(items) {
  return items.reduce(
    (acc, item) => {
      const type = item.issue_type || item.type || 'unknown'
      acc[type] = (acc[type] || 0) + 1
      return acc
    },
    { factual_error: 0, outdated: 0 }
  )
}

function countIssueSubtypes(items) {
  return items.reduce((acc, item) => {
    const subtype = getIssueSubtype(item)
    if (!subtype || subtype === 'outdated') return acc
    acc[subtype] = (acc[subtype] || 0) + 1
    return acc
  }, { simple_factual_error: 0, scope_error: 0 })
}

function countIssueFilters(items) {
  const typeCounts = countIssueTypes(items)
  const subtypeCounts = countIssueSubtypes(items)
  return {
    all: items.length,
    factual_error: typeCounts.factual_error || 0,
    outdated: typeCounts.outdated || 0,
    scope_error: subtypeCounts.scope_error || 0,
    simple_factual_error: subtypeCounts.simple_factual_error || 0,
  }
}

function matchesIssueFilter(item, filter) {
  if (!filter || filter === 'all') return true
  if (filter === 'factual_error' || filter === 'outdated') {
    return (item.issue_type || item.type) === filter
  }
  return getIssueSubtype(item) === filter
}

function claimDisplayIssueKey(claim) {
  const issueType = claim.issue_type || claim.type
  if (issueType === 'outdated') return 'outdated'
  return getIssueSubtype(claim) || issueType
}

function labelForClaimIssue(claim) {
  const key = claimDisplayIssueKey(claim)
  if (key === 'outdated') return labelForIssueType('outdated')
  return labelForIssueSubtype(key) || labelForIssueType(key)
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
        imageUrl: typo.slide_image_url || '',
        items: [],
      })
    }

    const group = groups.get(key)
    if (!group.imageUrl && typo.slide_image_url) {
      group.imageUrl = typo.slide_image_url
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
            <strong>{compactText(verdict?.verdict || verdict?.decision || verdict?.status)}</strong>
            {verdict?.confidence !== undefined && (
              <em>{formatPercent(verdict.confidence)}</em>
            )}
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
  const title = claim.claim_text || claim.resolved_claim || claim.problematic_content || '-'
  const startTime = Number(claim.start_time)
  const canWatch = Number.isFinite(startTime)
  const grounding = claim.grounding || {}
  const slideRecheck = claim.slide_recheck || {}
  const displayIssueKey = claimDisplayIssueKey(claim)
  const displayIssueLabel = labelForClaimIssue(claim)
  const sources = asArray(grounding.evidence_sources).length
    ? grounding.evidence_sources
    : asArray(claim.evidence_sources)

  return (
    <article className={`vf-claim-card ${expanded ? 'vf-claim-card--expanded' : ''}`}>
      <button className="vf-claim-main" onClick={onToggle}>
        <div className="vf-claim-copy">
          <div className="vf-claim-title">{title}</div>
          <div className="vf-chip-row">
            <span className="vf-chip vf-chip--stage">{labelForStage(claim.stage || section)}</span>
            {displayIssueKey && <span className={`vf-chip vf-chip--${displayIssueKey}`}>{displayIssueLabel}</span>}
            {claim.issue_pattern && (
              <span className="vf-chip vf-chip--pattern">{claim.issue_pattern}</span>
            )}
            {claim.severity && <span className="vf-chip">{claim.severity}</span>}
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
            <DetailRow label="Claim" value={claim.resolved_claim || claim.claim_text} />
            <DetailRow label="Issue" value={claim.issue} />
            <DetailRow label="Correct Info" value={claim.correct_info} />
            <DetailRow label="분류" value={[claim.claim_type, claim.issue_category_label, claim.issue_pattern].filter(Boolean).join(' / ')} />
            <DetailRow label="기각/검토 사유" value={claim.rejection_reason || claim.review_reason_code || claim.rejection_reason_code} />
            <DetailRow label="Slide Recheck" value={slideRecheck.status || slideRecheck.reason || claim.slide_recheck_status} />
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

  const sections = useMemo(() => {
    const finalClaims = asArray(verifier?.final_confirmed_claims)
    const needsReview = asArray(verifier?.needs_review_claims)
    const crossRejected = asArray(verifier?.crosscheck_rejected_claims)
    const inconclusive = asArray(verifier?.crosscheck_inconclusive_claims)
    const slideRejected = asArray(verifier?.slide_rejected_claims)
    const groundingRejected = asArray(verifier?.grounding_rejected_claims)
    const firstStageRejected = asArray(verifier?.first_stage_rejected_claims)
    return {
      finalClaims,
      needsReview,
      slideTypos: asArray(verifier?.slide_typos),
      filtered: [
        ...crossRejected,
        ...inconclusive,
        ...slideRejected,
        ...groundingRejected,
      ],
      firstStageRejected,
    }
  }, [verifier])

  const counts = verifier?.counts || {}
  const finalCount = counts.final_confirmed ?? verifier?.final_confirmed_claim_count ?? sections.finalClaims.length
  const reviewCount = counts.needs_review ?? sections.needsReview.length
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

  function renderTypoGroups(items, review = false) {
    return (
      <div className="vf-typo-list">
        {groupTyposBySlide(items).map((group) => (
          <SlideTypoCard key={`${review ? 'review' : 'typo'}-${group.key}`} group={group} review={review} />
        ))}
      </div>
    )
  }

  function renderActivePanel() {
    if (activeTab === 'review') {
      const filteredReview = filterIssueClaims(sections.needsReview)
      return (
        <Section
          title="검토가 필요한 내용 이슈"
          count={sections.needsReview.length}
          tone="review"
          empty="검토가 필요한 내용 이슈가 없습니다."
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
            : <div className="vf-empty">선택한 유형의 검토 필요 이슈가 없습니다.</div>}
        </Section>
      )
    }

    if (activeTab === 'typos') {
      return (
        <Section
          title="슬라이드 오타"
          count={sections.slideTypos.length}
          tone="typo"
          empty="확정된 슬라이드 오타가 없습니다."
        >
          {renderTypoGroups(sections.slideTypos)}
        </Section>
      )
    }

    if (activeTab === 'filtered') {
      return (
        <>
          <Section
            title="필터링된 내용 후보"
            count={sections.filtered.length}
            empty="필터링된 내용 후보가 없습니다."
          >
            {renderClaimList(sections.filtered, 'filtered')}
          </Section>

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
              label="검토 필요"
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
              label="필터링됨"
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
