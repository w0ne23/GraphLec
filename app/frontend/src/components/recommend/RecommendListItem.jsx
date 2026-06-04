import { useState } from 'react'

const DISPLAY_SCORE_FULL_RATIO = 0.8
const CONDITION_CHART_SHARE = 20
const CONTENT_MEANING_CHART_SHARE = 100 - CONDITION_CHART_SHARE

/**
 * 추천 강의 목록 아이템
 * 레이아웃: [썸네일] [태그·제목] [추천 점수 / 자세히▼]
 */
export default function RecommendListItem({ lecture, onPlay, queryText = '' }) {
  const [isExpanded, setIsExpanded] = useState(false)
  const detail        = lecture.score_detail || {}
  const overallScore  = lecture.display_score ?? null
  const durationLabel = formatDuration(lecture.duration_sec)
  const scoreParts    = getScoreParts(detail)

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
  const total = parts.reduce((sum, part) => sum + chartValue(part), 0)
  let cursor = -90
  const sectors = parts.map(part => {
    const composition = chartValue(part)
    const score = clampScore(part.value)
    const angle = total > 0 ? (composition / total) * 360 : 360 / parts.length
    const sector = {
      ...part,
      startAngle: cursor,
      endAngle: cursor + angle,
      radius: 56 * Math.max(0.18, Math.min(1, score / 100)),
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
            <div className="rec-radial-meter" aria-label={`${part.label} 점수 ${part.value} / 100, 구성 비중 ${chartValue(part)} / 100`}>
              <i style={{ width: `${part.value}%` }} />
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

function chartValue(part) {
  return clampScore(part.chartValue ?? part.value)
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

function getScoreParts(detail) {
  const contentRaw = toRatio(detail.content_score ?? ((detail.content_pct ?? 0) / 100))
  const meaningRaw = Math.max(
    toRatio(detail.vec_score ?? 0),
    toRatio(detail.graph_score ?? 0),
    toRatio(detail.sim_keyword ?? 0),
  )
  const conditionRaw = getConditionScoreRatio(detail)
  const rawParts = [
    {
      key: 'content',
      label: '내용 일치',
      shortLabel: '내용',
      raw: contentRaw,
    },
    {
      key: 'meaning',
      label: '의미 유사도',
      shortLabel: '의미',
      raw: meaningRaw,
    },
    {
      key: 'condition',
      label: '조건 적합도',
      shortLabel: '조건',
      raw: conditionRaw,
    },
  ]

  const contentChartValue = getContentChartValue(contentRaw, meaningRaw)
  const meaningChartValue = CONTENT_MEANING_CHART_SHARE - contentChartValue

  return rawParts.map(part => ({
    key: part.key,
    label: part.label,
    shortLabel: part.shortLabel,
    value: displayRatioScore(part.raw),
    chartValue: part.key === 'condition'
      ? CONDITION_CHART_SHARE
      : part.key === 'content'
        ? contentChartValue
        : meaningChartValue,
    score10: Math.max(0, Math.min(10, Number((displayRatioScore(part.raw) / 10).toFixed(1)))),
  }))
}

function getContentChartValue(contentRaw, meaningRaw) {
  const content = Math.max(0, toRatio(contentRaw))
  const meaning = Math.max(0, toRatio(meaningRaw))
  const total = content + meaning
  if (total <= 0) return CONTENT_MEANING_CHART_SHARE / 2
  return Math.round((content / total) * CONTENT_MEANING_CHART_SHARE)
}

function displayRatioScore(value) {
  return clampScore((toRatio(value) / DISPLAY_SCORE_FULL_RATIO) * 100)
}

function getConditionScoreRatio(detail) {
  const explicitConditionScores = []
  const hasDurationCondition = detail.duration_score > 0 || detail.duration_mismatch === true
  const conditionWarnings = Array.isArray(detail.condition_warnings) ? detail.condition_warnings : []

  if (hasDurationCondition) {
    explicitConditionScores.push(toRatio(detail.duration_score ?? 0))
  }
  if (detail.visual_preference) {
    explicitConditionScores.push(Math.max(
      toRatio(detail.visual_score ?? 0),
      toRatio(detail.visual_density_score ?? 0),
      toRatio(detail.visual_concept_score ?? 0),
    ))
  }
  if (detail.application_preference) {
    explicitConditionScores.push(toRatio(detail.application_score ?? 0))
  }
  if (detail.listenability_preference) {
    explicitConditionScores.push(toRatio(detail.listenability_score ?? 0))
  }
  if (detail.slow_speech_preference) {
    explicitConditionScores.push(toRatio(detail.speech_rate_score ?? 0))
  }
  if (detail.recency_preference) {
    explicitConditionScores.push(toRatio(detail.recency_score ?? 0))
  }

  if (explicitConditionScores.length > 0) {
    const combined = toRatio(detail.combined_boost ?? 0)
    if (combined > 0) return combined
    return explicitConditionScores.reduce((sum, value) => sum + value, 0) / explicitConditionScores.length
  }

  return conditionWarnings.length > 0 ? toRatio(detail.combined_boost ?? 0) : 1
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
