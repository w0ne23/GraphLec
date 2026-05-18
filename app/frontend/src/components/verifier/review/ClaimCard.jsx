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

function TranscriptRow({ contexts, highlightText }) {
  const [open, setOpen] = useState(false)
  const items = asArray(contexts).filter(context => context?.text)
  if (!items.length) return null

  function handleToggleKeyDown(event, nextOpen) {
    if (event.key !== 'Enter' && event.key !== ' ') return
    event.preventDefault()
    setOpen(nextOpen)
  }

  return (
    <div className="vf-detail-row vf-transcript-row">
      <dt>원문</dt>
      <dd>
        {!open && (
          <span
            className="vf-transcript-toggle"
            role="button"
            tabIndex={0}
            onClick={() => setOpen(true)}
            onKeyDown={event => handleToggleKeyDown(event, true)}
          >
            원문 보기 ▼
          </span>
        )}
        {open && (
          <div className="vf-transcript-body">
            <div className="vf-transcript-list">
              {items.map((context, idx) => {
                const isLast = idx === items.length - 1
                return (
                  <p key={`${context.context_id || context.slide_number || 'transcript'}-${idx}`}>
                    <span className="vf-transcript-text">
                      <TranscriptText text={context.text} highlightText={highlightText} />
                    </span>
                    {isLast && (
                      <span
                        className="vf-transcript-toggle vf-transcript-toggle--collapse"
                        role="button"
                        tabIndex={0}
                        onClick={() => setOpen(false)}
                        onKeyDown={event => handleToggleKeyDown(event, false)}
                      >
                        접기 ▲
                      </span>
                    )}
                  </p>
                )
              })}
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
              {verdict?.model_weight !== undefined && <em>가중치 {Number(verdict.model_weight).toFixed(2)}</em>}
            </div>
            <strong>{formatPercent(verdict?.confidence ?? verdict?.vote_score)}</strong>
            {(verdict?.decision || verdict?.verdict || verdict?.status) && (
              <em>판정 유형: {compactText(verdict?.decision || verdict?.verdict || verdict?.status)}</em>
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

  return (
    <article className={`vf-claim-card ${expanded ? 'vf-claim-card--expanded' : ''}`}>
      <div className={`vf-claim-summary ${canWatch ? 'vf-claim-summary--watch' : ''}`}>
        {canWatch && (
          <button className="vf-watch-btn" onClick={onWatch} title="영상 보기" aria-label="영상 보기">
            <span aria-hidden="true">▶</span>
          </button>
        )}
        <button className="vf-claim-main" onClick={onToggle}>
          {hasClaimChips && (
            <div className="vf-chip-row">
              {displayIssueKey && <span className={`vf-chip vf-chip--${displayIssueKey}`}>{displayIssueLabel}</span>}
              {hasCrosscheckScore && (
                <span className={`vf-chip vf-chip--score vf-chip--score-${crosscheckStatus || 'unknown'}`}>
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
            className={`vf-claim-toggle ${expanded ? 'vf-claim-toggle--open' : ''}`}
            aria-hidden="true"
          />
        </button>
      </div>

      {expanded && (
        <div className="vf-claim-detail">
          <dl>
            <TranscriptRow contexts={claim.transcript_contexts} highlightText={claim.transcript_claim_text} />
            <DetailRow label="발화 ID" value={claim.utterance_ids?.length ? claim.utterance_ids.join(', ') : claim.utterance_id} />
            <DetailRow label="유형 근거" value={claim.issue_type_rationale} />
            <DetailRow label="문제점" value={claim.issue} />
            <DetailRow label="학생이 잘못 외울 수 있는 명제" value={claim.student_error} />
            <DetailRow label="왜 문제인가" value={whyWrong} />
            <DetailRow label="반례/조건" value={claim.counterexample_or_condition || claim.counterexample} />
            <DetailRow label="문맥 해소 여부" value={claim.context_resolution} />
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
