import { useState } from 'react'

/**
 * 추천 강의 목록 아이템
 * 레이아웃: [썸네일] [태그·제목·강의자] [추천 점수 + 간접관련배지 / 자세히▼]
 */
export default function RecommendListItem({ lecture, onPlay, queryText = '' }) {
  const [isExpanded, setIsExpanded] = useState(false)
  const detail        = lecture.score_detail || {}
  const overallScore  = lecture.display_score ?? null
  const durationLabel = formatDuration(lecture.duration_sec)

  // tier 판단: API 필드 우선, 없으면 reason 텍스트로 fallback
  const isBackground = lecture.tier === 'background'
  const isRelated = lecture.tier === 'related'
    || (typeof lecture.reason === 'string' && lecture.reason.includes('직접 일치하지 않지만'))
  const hasSupportTier = isRelated || isBackground
  const showScore = overallScore != null

  // keywords 배열 우선 — reason 텍스트가 태그로 뽑히는 버그 방지
  const relatedTags = deriveRelatedTags(lecture, queryText, 4)

  const durationMin      = lecture.duration_sec ? Math.round(lecture.duration_sec / 60) : null
  const durationMismatch = detail.duration_mismatch === true

  return (
    <div className={`rec-item${isExpanded ? ' rec-item--expanded' : ''}${hasSupportTier ? ' rec-item--related' : ''}${isBackground ? ' rec-item--background' : ''}`}>

      {/* ── 메인 행 ── */}
      <div className="rec-item-main" onClick={() => setIsExpanded(v => !v)}>

        {/* 썸네일 */}
        <div className="rec-col">
          <div className="rec-thumb">
            <span className="rec-thumb-icon">🎬</span>
            {durationLabel && <span className="rec-duration">{durationLabel}</span>}
          </div>
        </div>

        {/* 메타 */}
        <div className="rec-col rec-col-meta">
          {relatedTags.length > 0 && (
            <div className="rec-col-tags">
              {relatedTags.map((tag, i) => (
                <span key={i} className="rec-tag">#{tag}</span>
              ))}
            </div>
          )}
          <div className="rec-title-row">
            <span className="rec-title">{lecture.title}</span>
            {durationLabel && <span className="rec-duration-inline">· {durationLabel}</span>}
          </div>
          <div className="rec-sub">
            {lecture.instructor || '강사 미상'}
            {lecture.video_id && ` | ${lecture.video_id}`}
          </div>
          {durationMismatch && durationMin && (
            <span className="rec-duration-warn">⚠ {durationMin}분 · 시간 범위 초과</span>
          )}
        </div>

        {/* 점수 + 티어 배지 + 토글 */}
        <div className="rec-col rec-col-score">
          {showScore && (
            <div className={`rec-overall-score${hasSupportTier ? ' rec-overall-score--related' : ''}${isBackground ? ' rec-overall-score--background' : ''}`}>
              {overallScore}점
            </div>
          )}
          {hasSupportTier && (
            <span className={`rec-tier-badge${isBackground ? ' rec-tier-badge--background' : ''}`}>
              {isBackground ? '배경 강의' : '간접 관련'}
            </span>
          )}
          <button className="rec-toggle-btn">
            {isExpanded ? '접기 ▲' : '자세히 ▼'}
          </button>
        </div>
      </div>

      {/* ── 아코디언 ── */}
      {isExpanded && (
        <div className="rec-item-details">
          <div className="rec-details-content">
            {hasSupportTier && (
              <p className={`rec-related-notice${isBackground ? ' rec-related-notice--background' : ''}`}>
                {isBackground
                  ? '이 강의는 직접 답변 강의는 아니지만, 주제를 이해하는 데 도움이 되는 배경 개념을 다룹니다.'
                  : '이 강의는 질의 주제와 직접 일치하지 않을 수 있습니다. 관련 개념을 포함하고 있어 함께 참고할 수 있습니다.'}
              </p>
            )}
            {lecture.reason && (
              <p className="rec-sub" style={{ marginBottom: 8 }}>{lecture.reason}</p>
            )}
            {lecture.summary && (
              <p className="rec-sub" style={{ marginBottom: 10 }}>{lecture.summary}</p>
            )}
            {lecture.score_detail && (
              <>
                <h4 className="rec-details-title">AI 세부 분석 지표</h4>
                <div className="rec-scores-grid">
                  <DetailScoreBar label="내용 관련도"     score={detail.content_pct ?? 0} color="#3b82f6" />
                  <DetailScoreBar label="개념 그래프"     score={pct(detail.graph_score)} color="#10b981" />
                  {detail.community_score > 0 && (
                    <DetailScoreBar label="커뮤니티 맥락" score={pct(detail.community_score)} color="#14b8a6" />
                  )}
                  <DetailScoreBar label="키워드 직접 매칭" score={pct(detail.dm_keyword)}  color="#f59e0b" />
                  <DetailScoreBar label="벡터 유사도"     score={pct(detail.vec_score)}   color="#8b5cf6" />
                </div>
              </>
            )}
            <div className="rec-details-action">
              <button
                className="rec-play-btn"
                onClick={e => { e.stopPropagation(); onPlay?.(lecture.video_id) }}
              >
                ▶ 이 강의 시청하기
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

function DetailScoreBar({ label, score, color }) {
  return (
    <div className="rec-score-row">
      <span className="rec-score-label">{label}</span>
      <div className="rec-score-track">
        <div className="rec-score-fill" style={{ width: `${score}%`, backgroundColor: color }} />
      </div>
      <span className="rec-score-num">{score}</span>
    </div>
  )
}

function pct(value) {
  return Math.round((value || 0) * 100)
}

function formatDuration(durationSec) {
  if (!durationSec || Number.isNaN(Number(durationSec))) return ''
  const total   = Math.max(0, Math.floor(Number(durationSec)))
  const minutes = Math.floor(total / 60)
  const seconds = total % 60
  return `${minutes}:${String(seconds).padStart(2, '0')}`
}

function deriveRelatedTags(lecture, queryText, topN = 4) {
  if (Array.isArray(lecture.tags) && lecture.tags.length > 0) {
    return lecture.tags.slice(0, topN)
  }

  // keywords 배열 우선 — reason/notice 텍스트 오염 방지
  if (Array.isArray(lecture.keywords) && lecture.keywords.length > 0) {
    return lecture.keywords.slice(0, topN).map(k => k.keyword ?? k)
  }

  // fallback: reason 제외하고 title + summary + query만 사용
  const source = [queryText, lecture.title, lecture.summary].filter(Boolean).join(' ')
  const tokens = tokenize(source).filter(t => t.length >= 2)
  const counts = new Map()
  for (const t of tokens) counts.set(t, (counts.get(t) || 0) + 1)

  const stop = new Set([
    '강의', '관련', '추천', '내용', '직접', '매칭', '유사도', '설명', '요약', '기초', '심화',
    '포함', '일치', '개념', '주제', '분석', '지표', '학습', '이해', '방식', '방법',
    'the', 'and', 'for', 'with', 'that', 'this',
  ])

  const ranked = [...counts.entries()]
    .filter(([w]) => !stop.has(w))
    .sort((a, b) => b[1] - a[1])
    .map(([w]) => w)
    .slice(0, topN)

  return ranked.length > 0 ? ranked : (lecture.domain ? [lecture.domain] : [])
}

function tokenize(text) {
  return String(text || '')
    .toLowerCase()
    .match(/[a-z0-9가-힣]+/g) || []
}
