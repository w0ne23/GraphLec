import { useState } from 'react'
import {
  asArray,
  claimDisplayIssueKey,
  compactText,
  formatModelName,
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

const SCORE_TEXT_CLASS = {
  agree: 'vf-claim-score-text--agree',
  confirmed: 'vf-claim-score-text--confirmed',
  disagree: 'vf-claim-score-text--disagree',
  inconclusive: 'vf-claim-score-text--inconclusive',
  professor_check: 'vf-claim-score-text--professor_check',
  rejected: 'vf-claim-score-text--rejected',
  review_needed: 'vf-claim-score-text--review_needed',
}

function cx(...classNames) {
  return classNames.filter(Boolean).join(' ')
}

function DetailRow({ label, value, className = '' }) {
  if (value === undefined || value === null || value === '') return null
  return (
    <div className={cx('vf-detail-row', className)}>
      <dt>{label}</dt>
      <dd>{value}</dd>
    </div>
  )
}

function textFromVerdict(verdict) {
  return compactText(
    verdict?.reason ||
      verdict?.problem ||
      verdict?.summary ||
      verdict?.why_wrong ||
      verdict?.rationale ||
      verdict?.explanation,
    ''
  )
}

function modelProblemEntries(verdicts) {
  return Object.entries(verdicts || {})
    .map(([model, verdict]) => ({
      model,
      text: textFromVerdict(verdict),
    }))
    .filter(item => item.text)
}

function ModelProblemList({ entries }) {
  if (!entries.length) return null

  return (
    <div className="vf-model-problem-list">
      {entries.map(entry => (
        <div className="vf-model-problem" key={entry.model}>
          <div className="vf-model-problem-head">
            <strong>{entry.model === 'web_grounding' ? 'web_grounding' : formatModelName(entry.model)}</strong>
          </div>
          <p>{entry.text}</p>
        </div>
      ))}
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

function WatchLocationRow({ canWatch, timestamp, onWatch }) {
  if (!canWatch) return null
  return (
    <div className="vf-detail-row">
      <dt>영상 위치</dt>
      <dd className="vf-watch-location">
        <button className="vf-transcript-toggle vf-transcript-toggle--button" type="button" onClick={onWatch}>
          동영상 보기
        </button>
        <span className="vf-claim-time">({timestamp})</span>
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
  const grounding = claim.web_grounding || claim.grounding || {}
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
  const locationLabel = [claim.scene_label, claim.slide_title].filter(Boolean).join(' ')
  const issueChipClass = ISSUE_CHIP_CLASS[displayIssueKey]
  const scoreTextClass = SCORE_TEXT_CLASS[crosscheckStatus]
  const problemsByModel = modelProblemEntries(claim.model_verdicts)

  return (
    <article className={cx('vf-claim-card', expanded && 'vf-claim-card--expanded')}>
      <div className="vf-claim-summary">
        <button className="vf-claim-main" onClick={onToggle}>
          <div className="vf-claim-copy">
            <div className="vf-claim-headline">
              {displayIssueKey && <span className={cx('vf-chip', issueChipClass)}>{displayIssueLabel}</span>}
              <span className="vf-claim-title">{title}</span>
            </div>
            {(claim.scene_label || claim.slide_title) && (
              <div className="vf-claim-location" aria-label={locationLabel}>
                {claim.scene_label && <span>{claim.scene_label}</span>}
                {claim.slide_title && <span>{claim.slide_title}</span>}
              </div>
            )}
          </div>
          {hasCrosscheckScore && (
            <div className="vf-claim-meta">
              <span className={cx('vf-claim-score-text', scoreTextClass)}>{scoreLabel(claim.crosscheck_score)}</span>
            </div>
          )}
          <span
            className={cx('vf-claim-toggle', expanded && 'vf-claim-toggle--open')}
            aria-hidden="true"
          />
        </button>
      </div>

      {expanded && (
        <div className="vf-claim-detail">
          <dl>
            <WatchLocationRow canWatch={canWatch} timestamp={formatTime(startTime)} onWatch={onWatch} />
            <TranscriptRow contexts={claim.transcript_contexts} highlightText={claim.transcript_claim_text} />
            <DetailRow label="발화 ID" value={claim.utterance_ids?.length ? claim.utterance_ids.join(', ') : claim.utterance_id} />
            <DetailRow label="유형 근거" value={claim.issue_type_rationale} />
            <DetailRow label="문제점" value={problemsByModel.length ? <ModelProblemList entries={problemsByModel} /> : claim.issue} />
            <DetailRow label="수정 제안" value={claim.correct_info} className="vf-detail-row--separated" />
            <DetailRow label="학생이 잘못 외울 수 있는 명제" value={claim.student_error} />
            <DetailRow label="왜 문제인가" value={whyWrong} />
            <DetailRow label="반례/조건" value={claim.counterexample_or_condition || claim.counterexample} />
            <DetailRow label="학생 오해 가능성" value={claim.student_misunderstanding} />
            <DetailRow label="왜 중요한가" value={claim.why_it_matters} />
            <DetailRow label="권장 수정" value={recommendation} />
            <DetailRow label="대체 표현" value={suggestedRephrase} />
            <DetailRow label="문맥 근거" value={evidenceInContext} />
            <DetailRow label="기각/검토 사유" value={claim.rejection_reason || claim.review_reason_code || claim.rejection_reason_code} />
            <DetailRow label="기각 단계" value={claim.rejection_stage} />
            <DetailRow label="web_grounding" value={grounding.status || grounding.reason || claim.grounding_status} />
            <DetailRow label="web 근거 요약" value={grounding.evidence_summary} />
          </dl>
          <EvidenceSources sources={sources} />
        </div>
      )}
    </article>
  )
}
