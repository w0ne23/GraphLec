import { useState } from 'react'

/**
 * 추천 강의 목록 아이템
 * 레이아웃: [썸네일] [태그·제목] [추천 점수 / 자세히▼]
 */
export default function RecommendListItem({ lecture, onPlay, queryText = '' }) {
  const [isExpanded, setIsExpanded] = useState(false)
  const detail        = lecture.score_detail || {}
  const overallScore  = lecture.display_score ?? null
  const durationLabel = formatDuration(lecture.duration_sec)
  const scoreParts    = getScoreParts(detail, overallScore)

  const showScore = overallScore != null

  // keywords 배열 우선 — reason 텍스트가 태그로 뽑히는 버그 방지
  const relatedTags = deriveRelatedTags(lecture, queryText, 4)

  const durationMin      = lecture.duration_sec ? Math.round(lecture.duration_sec / 60) : null
  const durationMismatch = detail.duration_mismatch === true

  return (
    <div className={`rec-item${isExpanded ? ' rec-item--expanded' : ''}`}>

      {/* ── 메인 행 ── */}
      <div className="rec-item-main" onClick={() => setIsExpanded(v => !v)}>

        {/* 썸네일 */}
        <div className="rec-col">
          <div className="rec-thumb">
            {lecture.thumbnail_url ? (
              <img
                className="rec-thumb-image"
                src={lecture.thumbnail_url}
                alt=""
                loading="lazy"
                onError={(event) => {
                  event.currentTarget.hidden = true
                  event.currentTarget.nextElementSibling?.removeAttribute('hidden')
                }}
              />
            ) : null}
            <span className="rec-thumb-icon" hidden={Boolean(lecture.thumbnail_url)}>🎬</span>
            {durationLabel && <span className="rec-duration">{durationLabel}</span>}
          </div>
        </div>

        {/* 메타 */}
        <div className="rec-col rec-col-meta">
          <div className="rec-title-row">
            <span className="rec-title">{lecture.title}</span>
            {durationLabel && <span className="rec-duration-inline">· {durationLabel}</span>}
          </div>
          {durationMismatch && durationMin && (
            <span className="rec-condition-warning rec-condition-warning--inline">
              {durationMin}분 · 시간 범위 초과
            </span>
          )}
        </div>

        {/* 점수 + 티어 배지 + 토글 */}
        <div className="rec-col rec-col-score">
          {showScore && (
            <ScorePills parts={scoreParts} />
          )}
          {showScore && (
            <div className="rec-overall-score">
              {overallScore}점
            </div>
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
            <div className="rec-details-grid">
              <div className="rec-details-copy">
                <section className="rec-detail-section">
                  <h4 className="rec-details-title">핵심 키워드</h4>
                  <div className="rec-detail-tags">
                    {(relatedTags.length > 0 ? relatedTags : ['추천 강의']).map((tag, i) => (
                      <span key={i} className="rec-tag">#{tag}</span>
                    ))}
                  </div>
                </section>

                <section className="rec-detail-section rec-detail-section--summary">
                  <h4 className="rec-details-title">요약 설명</h4>
                  <p className="rec-detail-summary">
                    {lecture.summary || lecture.reason || '이 강의는 검색한 질의와 관련된 핵심 개념을 포함하고 있어 학습 흐름에 맞춰 참고하기 좋습니다.'}
                  </p>
                  <button
                    className="rec-play-btn rec-play-btn--summary"
                    onClick={e => { e.stopPropagation(); onPlay?.(lecture.video_id) }}
                  >
                    ▶ 이 강의 시청하기
                  </button>
                </section>
              </div>

              <div className="rec-details-chart">
                <RadialScoreChart parts={scoreParts} />
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

function ScorePills({ parts }) {
  return (
    <div className="rec-score-pills" aria-label="세부 추천 점수">
      {parts.map(part => (
        <span key={part.key} className={`rec-score-pill rec-score-pill--${part.key}`}>
          {part.label} {part.value}
        </span>
      ))}
    </div>
  )
}

function RadialScoreChart({ parts }) {
  const total = parts.reduce((sum, part) => sum + part.value, 0)
  const maxPartValue = Math.max(...parts.map(part => part.value), 1)
  let cursor = -90
  const sectors = parts.map(part => {
    const angle = total > 0 ? (part.value / total) * 360 : 360 / parts.length
    const sector = {
      ...part,
      startAngle: cursor,
      endAngle: cursor + angle,
      radius: 56 * Math.max(0.18, Math.min(1, part.value / maxPartValue)),
    }
    cursor += angle
    return sector
  })

  return (
      <div className="rec-radial-card">
        <div className="rec-radial-head">
          <span>점수 구성</span>
        </div>
      <div className="rec-radial-chart" aria-label="내용, 의미, 조건 점수 원형 차트">
        <svg viewBox="0 0 132 132" role="img">
          <circle className="rec-radial-boundary" cx="66" cy="66" r="56" />
          {sectors.map(sector => (
            <path
              key={sector.key}
              className={`rec-radial-sector rec-radial-sector--${sector.key}`}
              d={describeSector(66, 66, sector.radius, sector.startAngle, sector.endAngle)}
            />
          ))}
        </svg>
      </div>
      <div className="rec-radial-legend">
        {parts.map(part => (
          <div key={part.key} className={`rec-radial-legend-item rec-radial-legend-item--${part.key}`}>
            <span>{part.shortLabel}</span>
            <div className="rec-radial-meter" aria-label={`${part.label} ${part.value} / 100`}>
              <i style={{ width: `${part.value}%` }} />
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

function describeSector(cx, cy, radius, startAngle, endAngle) {
  if (radius <= 0) return ''
  const start = polarToCartesian(cx, cy, radius, endAngle)
  const end = polarToCartesian(cx, cy, radius, startAngle)
  const largeArcFlag = endAngle - startAngle <= 180 ? '0' : '1'

  return [
    `M ${cx} ${cy}`,
    `L ${start.x} ${start.y}`,
    `A ${radius} ${radius} 0 ${largeArcFlag} 0 ${end.x} ${end.y}`,
    'Z',
  ].join(' ')
}

function polarToCartesian(cx, cy, radius, angleInDegrees) {
  const angleInRadians = (angleInDegrees * Math.PI) / 180
  return {
    x: cx + radius * Math.cos(angleInRadians),
    y: cy + radius * Math.sin(angleInRadians),
  }
}

function getScoreParts(detail, overallScore) {
  const total = clampScore(overallScore ?? detail.score * 100 ?? 0)
  const rawParts = [
    {
      key: 'content',
      label: '내용 일치',
      shortLabel: '내용',
      raw: toRatio(detail.content_score ?? ((detail.content_pct ?? 0) / 100)),
    },
    {
      key: 'meaning',
      label: '의미 유사',
      shortLabel: '의미',
      raw: toRatio(detail.vec_score ?? detail.graph_score ?? 0),
    },
    {
      key: 'condition',
      label: '조건 적합',
      shortLabel: '조건',
      raw: toRatio(detail.combined_boost ?? detail.duration_score ?? detail.dm_keyword ?? 0),
    },
  ]

  const rawTotal = rawParts.reduce((sum, part) => sum + part.raw, 0)
  if (!total || rawTotal <= 0) {
    return rawParts.map(part => ({ ...part, value: 0 }))
  }

  const values = rawParts.map(part => Math.max(0, Math.round((part.raw / rawTotal) * total)))
  const diff = total - values.reduce((sum, value) => sum + value, 0)
  if (values.length > 0) values[0] += diff

  return rawParts.map((part, index) => ({
    key: part.key,
    label: part.label,
    shortLabel: part.shortLabel,
    value: clampScore(values[index]),
    score10: Math.max(0, Math.min(10, Number((part.raw * 10).toFixed(1)))),
  }))
}

function toRatio(value) {
  const numeric = Number(value) || 0
  return numeric > 1 ? numeric / 100 : numeric
}

function clampScore(value) {
  const numeric = Number(value) || 0
  return Math.max(0, Math.min(100, Math.round(numeric)))
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

  if (Array.isArray(lecture.keywords) && lecture.keywords.length > 0) {
    return lecture.keywords
      .map(k => k.keyword ?? k)
      .filter(Boolean)
      .slice(0, topN)
  }

  // fallback: 추천 질의 문구가 태그로 섞이지 않도록 강의 자체 텍스트만 사용
  const source = [lecture.title, lecture.summary].filter(Boolean).join(' ')
  const tokens = tokenize(source).filter(t => t.length >= 2)
  const counts = new Map()
  for (const t of tokens) counts.set(t, (counts.get(t) || 0) + 1)

  const stop = new Set([
    '강의', '관련', '추천', '내용', '직접', '매칭', '유사도', '설명', '요약', '기초', '심화',
    '포함', '일치', '개념', '주제', '분석', '지표', '학습', '이해', '방식', '방법',
    '추천해줘', '추천해', '해줘', '알려줘', '보여줘',
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
