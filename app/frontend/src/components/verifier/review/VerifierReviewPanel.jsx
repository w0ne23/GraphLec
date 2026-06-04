import { useMemo, useState } from 'react'
import VideoPlayer from '../../watch/VideoPlayer'
import ClaimCard from './ClaimCard'
import SlideTypoCard from './SlideTypoCard'
import {
  ISSUE_FILTERS,
  ISSUE_FILTER_DESCRIPTIONS,
  asArray,
  buildVerifierScenes,
  buildVerifierSlideMap,
  countIssueFilters,
  feedbackItemToClaim,
  groupTyposBySlide,
  matchesIssueFilter,
} from './verifierReviewUtils'

const SECTION_TONE_CLASS = {
  review: 'vf-section--review',
  typo: 'vf-section--typo',
}

const REVIEW_TAB_CLASS = {
  review: 'vf-review-tab--review',
  typos: 'vf-review-tab--typos',
}

function cx(...classNames) {
  return classNames.filter(Boolean).join(' ')
}

function ReviewSection({ title, count, tone = '', empty, stickyContent = null, children }) {
  return (
    <section className={cx('vf-section', SECTION_TONE_CLASS[tone])}>
      <div className={cx('vf-section-sticky', stickyContent && 'vf-section-sticky--with-controls')}>
        <div className="vf-section-head">
          <h2>{title}</h2>
          <span>{count}</span>
        </div>
        {stickyContent}
      </div>
      {count > 0 ? children : <div className="vf-empty">{empty}</div>}
    </section>
  )
}

function IssueTypeBreakdown({ items, activeFilter = 'all', onFilterChange }) {
  const counts = countIssueFilters(items)
  return (
    <div className="vf-type-breakdown">
      {ISSUE_FILTERS.map(filter => {
        const count = counts[filter.key] || 0
        return (
          <button
            className={cx(
              'vf-type-pill',
              activeFilter === filter.key && 'vf-type-pill--active',
              !count && 'vf-type-pill--empty',
            )}
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
  const filterItem = ISSUE_FILTERS.find(item => item.key === filter)
  const label = filterItem ? filterItem.label : filter
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
        className={cx('vf-sort-btn', value === 'utterance' && 'vf-sort-btn--active')}
        onClick={() => onChange('utterance')}
      >
        발화순
      </button>
      <button
        className={cx('vf-sort-btn', value === 'score' && 'vf-sort-btn--active')}
        onClick={() => onChange('score')}
      >
        신뢰도순
      </button>
    </div>
  )
}

function reviewTabClassName(tab, activeTab) {
  return cx('vf-review-tab', REVIEW_TAB_CLASS[tab], activeTab === tab && 'vf-review-tab--active')
}

function VerifierReviewHeader({
  onBack,
  activeTab,
  lectureTitle,
  reviewCount,
  typoCount,
  onSelectTab,
  onOpenDetail,
  onCancelUpload,
  isCancelling,
  onComplete,
}) {
  return (
    <div className="vf-review-header">
      <div className="vf-review-header-main">
        <div className="vf-review-heading">
          {onBack && <button className="vf-topbar-back" onClick={onBack}>◀</button>}
          <strong>{lectureTitle}</strong>
          <span>검토 결과</span>
        </div>
        <div className="vf-review-header-actions">
          {onOpenDetail && (
            <button className="vf-review-detail-btn" onClick={onOpenDetail}>
              상세보기
            </button>
          )}
          <nav className="vf-review-tabs" aria-label="검토 항목">
            <button className={reviewTabClassName('review', activeTab)} onClick={() => onSelectTab('review')}>
              <span>검토 필요</span>
              <strong>{reviewCount}</strong>
            </button>
            <button className={reviewTabClassName('typos', activeTab)} onClick={() => onSelectTab('typos')}>
              <span>슬라이드 오타</span>
              <strong>{typoCount}</strong>
            </button>
          </nav>
          {onCancelUpload && (
            <button className="vf-cancel-btn vf-review-cancel-btn" onClick={onCancelUpload} disabled={isCancelling}>
              작업 삭제
            </button>
          )}
          <button className="vf-confirm-btn vf-review-complete-btn" onClick={onComplete}>
            검토 완료
          </button>
        </div>
      </div>
    </div>
  )
}

function VerifierVideoPane({
  lecture,
  scenes,
  currentScene,
  seekToSeconds,
  onSceneChange,
  onClose,
}) {
  return (
    <section className="vf-video-pane">
      <button className="vf-video-close-btn" onClick={onClose} title="영상 닫기" aria-label="영상 닫기">
        영상 닫기
      </button>
      <VideoPlayer
        lecture={lecture}
        scenes={scenes}
        currentScene={currentScene}
        seekTo={null}
        seekToSeconds={seekToSeconds}
        onSceneChange={onSceneChange}
      />
    </section>
  )
}

function ReviewConfirmModal({ onCancel, onConfirm }) {
  return (
    <div className="vf-confirm-backdrop" role="presentation" onClick={onCancel}>
      <div
        className="vf-confirm-modal"
        role="dialog"
        aria-modal="true"
        aria-labelledby="vf-next-confirm-title"
        onClick={e => e.stopPropagation()}
      >
        <h2 id="vf-next-confirm-title">검토를 완료할까요?</h2>
        <p>검토 결과를 확정하고 현재 작업을 마칩니다.</p>
        <div className="vf-confirm-modal-actions">
          <button className="vf-modal-secondary-btn" onClick={onCancel}>
            계속 검토
          </button>
          <button className="vf-modal-primary-btn" onClick={onConfirm}>
            완료
          </button>
        </div>
      </div>
    </div>
  )
}

export default function VerifierReviewPanel({ flow, onOpenDetail }) {
  const { actions } = flow
  const verifier = flow.verifier
  const lecture = flow.lecture
  const [activeTab, setActiveTab] = useState('review')
  const [activeIssueFilter, setActiveIssueFilter] = useState('all')
  const [sortMode, setSortMode] = useState('utterance')
  const [showNextConfirm, setShowNextConfirm] = useState(false)
  const [currentScene, setCurrentScene] = useState(0)
  const resultId = lecture?.id || verifier?.lecture_id || verifier?.id || ''

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
      const normalized = feedbackItems
        .map(item => feedbackItemToClaim(item, claimById))
      const needsReview = normalized.filter(item => item.stage === 'confirmed' || item.stage === 'professor_check' || item.stage === 'review_needed')
      const rejected = normalized.filter(item => item.stage === 'rejected')
      return {
        needsReview,
        slideTypos: asArray(verifier?.slide_errors),
        crossRejected: rejected,
        inconclusive: [],
        groundingRejected: [],
        filtered: rejected,
        firstStageRejected: [],
        usesFeedbackItems: true,
      }
    }

    const needsReview = [
      ...asArray(verifier?.final_confirmed_claims),
      ...asArray(verifier?.needs_review_claims),
    ]
    const crossRejected = asArray(verifier?.verifier_rejected_claims)
    const inconclusive = asArray(verifier?.crosscheck_inconclusive_claims)
    const groundingRejected = asArray(verifier?.grounding_rejected_claims)
    const firstStageRejected = asArray(verifier?.first_stage_rejected_claims)
    return {
      needsReview,
      slideTypos: asArray(verifier?.slide_errors),
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

  const scenes = useMemo(
    () => buildVerifierScenes(verifier, lecture?.scenes, resultId),
    [lecture?.scenes, resultId, verifier]
  )

  const lectureWithScenes = useMemo(
    () => lecture ? { ...lecture, scenes } : lecture,
    [lecture, scenes]
  )

  const slideMap = useMemo(
    () => buildVerifierSlideMap(verifier, scenes, resultId),
    [resultId, scenes, verifier]
  )

  const counts = verifier?.counts || {}
  const reviewCount = sections.needsReview.length
  const typoCount = counts.slide_errors ?? sections.slideTypos.length

  function selectTab(tab) {
    setActiveTab(tab)
    setActiveIssueFilter('all')
    actions.toggleClaim('')
  }

  function confirmNextStep() {
    setShowNextConfirm(false)
    actions.confirmReview()
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
              expanded={flow.expandedClaimKey === claimKey}
              onToggle={() => actions.toggleClaim(claimKey)}
              onWatch={() => actions.watchClaim(claim.start_time)}
            />
          )
        })}
      </div>
    )
  }

  function filterIssueClaims(items) {
    return items.filter(item => matchesIssueFilter(item, activeIssueFilter))
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
        const scoreDelta = (Number(b.severity_score ?? b.crosscheck_score) || 0) - (Number(a.severity_score ?? a.crosscheck_score) || 0)
        if (scoreDelta !== 0) return scoreDelta
      }
      return utteranceSortValue(a) - utteranceSortValue(b)
    })
  }

  function renderTypoGroups(items, review = false) {
    return (
      <div className="vf-typo-list">
        {groupTyposBySlide(items, slideMap, resultId).map(group => (
          <SlideTypoCard key={`${review ? 'review' : 'typo'}-${group.key}`} group={group} review={review} />
        ))}
      </div>
    )
  }

  function renderActivePanel() {
    if (activeTab === 'review') {
      const filteredReview = sortClaims(filterIssueClaims(sections.needsReview))
      return (
        <ReviewSection
          title="검토가 필요한 내용"
          count={sections.needsReview.length}
          tone="review"
          empty="검토가 필요한 내용이 없습니다."
          stickyContent={(
            <div className="vf-review-control-row">
              <IssueTypeBreakdown
                items={sections.needsReview}
                activeFilter={activeIssueFilter}
                onFilterChange={setActiveIssueFilter}
              />
              <SortControls value={sortMode} onChange={setSortMode} />
            </div>
          )}
        >
          <IssueFilterDescription filter={activeIssueFilter} />
          {filteredReview.length > 0
            ? renderClaimList(filteredReview, 'needs_review')
            : <div className="vf-empty vf-empty--filter">선택한 유형의 검토 필요 이슈가 없습니다.</div>}
        </ReviewSection>
      )
    }

    if (activeTab === 'typos') {
      return (
        <ReviewSection
          title="슬라이드 오타"
          count={sections.slideTypos.length}
          tone="typo"
          empty="슬라이드 오타가 없습니다."
        >
          {renderTypoGroups(sections.slideTypos)}
        </ReviewSection>
      )
    }

    return null
  }

  if (!verifier) {
    return (
      <div className="vf-shell">
        <div className="vf-error">Verifier 결과를 불러오는 중입니다.</div>
      </div>
    )
  }

  return (
    <div className="vf-shell">
        <section className="vf-review-panel">
          <VerifierReviewHeader
            onBack={actions.backToVerifyReady}
            activeTab={activeTab}
            lectureTitle={lecture.title}
            reviewCount={reviewCount}
            typoCount={typoCount}
            onSelectTab={selectTab}
            onOpenDetail={onOpenDetail}
            onCancelUpload={actions.cancelUpload}
            isCancelling={Boolean(flow.isMutating || flow.isRestarting || flow.isBusy)}
            onComplete={() => setShowNextConfirm(true)}
          />
          <div className="vf-review-content">
            {flow.isVideoMode && (
              <VerifierVideoPane
                lecture={lectureWithScenes}
                scenes={scenes}
                currentScene={currentScene}
                seekToSeconds={flow.seekToSeconds}
                onSceneChange={setCurrentScene}
                onClose={actions.exitVideo}
              />
            )}
            <div className="vf-review-scroll">
              {renderActivePanel()}
            </div>
          </div>
        </section>

      {showNextConfirm && (
        <ReviewConfirmModal
          onCancel={() => setShowNextConfirm(false)}
          onConfirm={confirmNextStep}
        />
      )}
    </div>
  )
}
