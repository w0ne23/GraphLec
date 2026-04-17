import { useState } from 'react'

/**
 * 추천 강의 목록 아이템
 * 레이아웃: [썸네일] [태그·제목·강의자] [점수% / 자세히 보기▼]
 *
 * @param {object}   lecture  - 강의 및 AI 추천 분석 데이터
 * @param {function} onPlay   - 시청 버튼 클릭 시 호출 (lectureId 전달)
 */
export default function RecommendListItem({ lecture, onPlay, queryText = '' }) {
  const [isExpanded, setIsExpanded] = useState(false)
  const detail = lecture.score_detail || {}
  const overallScore = Math.round((lecture.score || 0) * 100)
  const durationLabel = formatDuration(lecture.duration_sec)
  const relatedTags = deriveRelatedTags(lecture, queryText, 4)

  return (
    <div className={`rec-item${isExpanded ? ' rec-item--expanded' : ''}`}>

      {/* ── 메인 행 ── */}
      <div className="rec-item-main" onClick={() => setIsExpanded(v => !v)}>

        {/* 썸네일 */}
        <div className="rec-col">
          <div className="rec-thumb">
            <span className="rec-thumb-icon">🎬</span>
            {durationLabel && (
              <span className="rec-duration">{durationLabel}</span>
            )}
          </div>
        </div>

        {/* 메타: 태그 → 제목+시간 → 강의자·날짜 */}
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
            {durationLabel && (
              <span className="rec-duration-inline">· {durationLabel}</span>
            )}
          </div>
          <div className="rec-sub">
            {lecture.instructor || '강사 미상'}
            {lecture.video_id && ` | ${lecture.video_id}`}
          </div>
        </div>

        {/* 점수 + 토글 */}
        <div className="rec-col rec-col-score">
          {lecture.score != null && (
            <div className="rec-overall-score">{overallScore}%</div>
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
                  <DetailScoreBar label="내용 관련도" score={detail.content_pct ?? 0} color="#3b82f6" />
                  <DetailScoreBar label="키워드 직접 매칭" score={pct(detail.dm_keyword)} color="#f59e0b" />
                  <DetailScoreBar label="벡터 유사도" score={pct(detail.vec_score)} color="#8b5cf6" />
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
  const total = Math.max(0, Math.floor(Number(durationSec)))
  const minutes = Math.floor(total / 60)
  const seconds = total % 60
  return `${minutes}:${String(seconds).padStart(2, '0')}`
}

function deriveRelatedTags(lecture, queryText, topN = 4) {
  if (Array.isArray(lecture.tags) && lecture.tags.length > 0) {
    return lecture.tags.slice(0, topN)
  }

  const source = [queryText, lecture.title, lecture.summary, lecture.reason].filter(Boolean).join(' ')
  const tokens = tokenize(source).filter(t => t.length >= 2)
  const counts = new Map()

  for (const t of tokens) {
    counts.set(t, (counts.get(t) || 0) + 1)
  }

  const stop = new Set([
    '강의', '관련', '추천', '내용', '직접', '매칭', '유사도', '설명', '요약', '기초', '심화',
    'the', 'and', 'for', 'with', 'that', 'this',
  ])

  const ranked = [...counts.entries()]
    .filter(([w]) => !stop.has(w))
    .sort((a, b) => b[1] - a[1])
    .map(([w]) => w)
    .slice(0, topN)

  if (ranked.length > 0) return ranked
  return lecture.domain ? [lecture.domain] : []
}

function tokenize(text) {
  return String(text || '')
    .toLowerCase()
    .match(/[a-z0-9가-힣]+/g) || []
}