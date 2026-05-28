import { useState } from 'react'
import {
  asArray,
  claimDisplayIssueKey,
  compactText,
  formatModelName,
  formatPercent,
  formatTime,
  labelForClaimIssue,
  scoreLabel,
  uniqueDetailValue,
} from './verifierReviewUtils'

const ISSUE_CHIP_CLASS = {
  confusing_explanation: 'vf-chip--confusing_explanation',
  factual_error: 'vf-chip--factual_error',
  outdated: 'vf-chip--outdated',
  scope_error: 'vf-chip--scope_error',
  scope_overclaim: 'vf-chip--scope_overclaim',
  simple_factual_error: 'vf-chip--simple_factual_error',
  temporal_error: 'vf-chip--temporal_error',
}

const SCORE_CHIP_CLASS = {
  agree: 'vf-chip--score-agree',
  disagree: 'vf-chip--score-disagree',
  inconclusive: 'vf-chip--score-inconclusive',
  professor_check: 'vf-chip--score-professor_check',
  rejected: 'vf-chip--score-rejected',
  review_needed: 'vf-chip--score-review_needed',
}

function cx(...classNames) {
  return classNames.filter(Boolean).join(' ')
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

function TranscriptText({ text, highlightText }) {
  const value = String(text || '')
  const target = String(highlightText || '').trim()
  const start = target ? value.indexOf(target) : -1

  if (start < 0) return value

  const end = start + target.length
  return (
    <>
      {value.slice(0, start)}
      <strong className="vf-transcript-claim">{value.slice(start, end)}</strong>
      {value.slice(end)}
    </>
  )
}

function WatchLocationRow({ canWatch, onWatch }) {
  if (!canWatch) return null
  return (
    <div className="vf-detail-row">
      <dt>영상 위치</dt>
      <dd>
        <button className="vf-transcript-toggle vf-transcript-toggle--button" type="button" onClick={onWatch}>
          동영상 보기
        </button>
      </dd>
    </div>
  )
}

function TranscriptRow({ contexts, highlightText }) {
  const [open, setOpen] = useState(false)
  const items = asArray(contexts).filter(context => context?.text)
  if (!items.length) return null

  return (
    <div className="vf-detail-row vf-transcript-row">
      <dt>원문</dt>
      <dd>
        <button
          type="button"
          className="vf-transcript-toggle vf-transcript-toggle--button"
          aria-expanded={open}
          onClick={() => setOpen(prev => !prev)}
        >
          {open ? '접기 ▲' : '원문 보기 ▼'}
        </button>
        {open && (
          <div className="vf-transcript-body">
            <div className="vf-transcript-list">
              {items.map((context, idx) => (
                <p key={`${context.context_id || context.slide_number || 'transcript'}-${idx}`}>
                  <span className="vf-transcript-text">
                    <TranscriptText text={context.text} highlightText={highlightText} />
                  </span>
                </p>
              ))}
            </div>
          </div>
        )}
      </dd>
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
            <div className="vf-verdict-model">
              <span>{formatModelName(model)}</span>
              {verdict?.model_weight !== undefined && <span className="vf-verdict-meta">가중치 {Number(verdict.model_weight).toFixed(2)}</span>}
            </div>
            <strong>{formatPercent(verdict?.confidence ?? verdict?.vote_score)}</strong>
            {(verdict?.decision || verdict?.verdict || verdict?.status) && (
              <span className="vf-verdict-meta">판정 유형: {compactText(verdict?.decision || verdict?.verdict || verdict?.status)}</span>
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

export default function ClaimCard({ claim, expanded, onToggle, onWatch }) {
  const title = claim.claim_text || claim.resolved_claim || claim.problematic_content || '-'
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
  const whyWrong = uniqueDetailValue(claim.why_wrong, [claim.issue])
  const recommendation = uniqueDetailValue(claim.recommendation || claim.teaching_note, [claim.correct_info])
  const suggestedRephrase = uniqueDetailValue(claim.suggested_rephrase, [claim.correct_info, recommendation])
  const evidenceInContext = uniqueDetailValue(claim.evidence_in_context, [claim.issue, whyWrong])
  const hasClaimChips = displayIssueKey || hasCrosscheckScore
  const locationLabel = [claim.scene_label, claim.slide_title].filter(Boolean).join(' ')
  const issueChipClass = ISSUE_CHIP_CLASS[displayIssueKey]
  const scoreChipClass = SCORE_CHIP_CLASS[crosscheckStatus]

  return (
    <article className={cx('vf-claim-card', expanded && 'vf-claim-card--expanded')}>
      <div className="vf-claim-summary">
        <button className="vf-claim-main" onClick={onToggle}>
          {hasClaimChips && (
            <div className="vf-chip-row">
              {displayIssueKey && <span className={cx('vf-chip', issueChipClass)}>{displayIssueLabel}</span>}
              {hasCrosscheckScore && (
                <span className={cx('vf-chip', 'vf-chip--score', scoreChipClass)}>
                  {scoreLabel(claim.crosscheck_score)}
                </span>
              )}
            </div>
          )}
          <div className="vf-claim-copy">
            <div className="vf-claim-headline">
              <span className="vf-claim-time">{canWatch ? formatTime(startTime) : '-'}</span>
              <span className="vf-claim-title">{title}</span>
            </div>
            {(claim.scene_label || claim.slide_title) && (
              <div className="vf-claim-location" aria-label={locationLabel}>
                {claim.scene_label && <span>{claim.scene_label}</span>}
                {claim.slide_title && <span>{claim.slide_title}</span>}
              </div>
            )}
          </div>
          <span
            className={cx('vf-claim-toggle', expanded && 'vf-claim-toggle--open')}
            aria-hidden="true"
          />
        </button>
      </div>

      {expanded && (
        <div className="vf-claim-detail">
          <dl>
            <WatchLocationRow canWatch={canWatch} onWatch={onWatch} />
            <TranscriptRow contexts={claim.transcript_contexts} highlightText={claim.transcript_claim_text} />
            <DetailRow label="발화 ID" value={claim.utterance_ids?.length ? claim.utterance_ids.join(', ') : claim.utterance_id} />
            <DetailRow label="유형 근거" value={claim.issue_type_rationale} />
            <DetailRow label="문제점" value={claim.issue} />
            <DetailRow label="학생이 잘못 외울 수 있는 명제" value={claim.student_error} />
            <DetailRow label="왜 문제인가" value={whyWrong} />
            <DetailRow label="반례/조건" value={claim.counterexample_or_condition || claim.counterexample} />
            <DetailRow label="학생 오해 가능성" value={claim.student_misunderstanding} />
            <DetailRow label="올바른 정보/보충 조건" value={claim.correct_info} />
            <DetailRow label="왜 중요한가" value={claim.why_it_matters} />
            <DetailRow label="권장 수정" value={recommendation} />
            <DetailRow label="대체 표현" value={suggestedRephrase} />
            <DetailRow label="문맥 근거" value={evidenceInContext} />
            <DetailRow label="기각/검토 사유" value={claim.rejection_reason || claim.review_reason_code || claim.rejection_reason_code} />
            <DetailRow label="기각 단계" value={claim.rejection_stage} />
            <DetailRow label="Grounding" value={grounding.status || grounding.reason || claim.grounding_status} />
          </dl>
          <ModelVerdicts verdicts={claim.model_verdicts} />
          <EvidenceSources sources={sources} />
        </div>
      )}
    </article>
  )
}
