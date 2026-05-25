import {
  asArray,
  compactText,
  formatPercent,
} from './verifierReviewUtils'

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

export default function SlideTypoCard({ group, review = false }) {
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
            {group.items.map(typo => (
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
