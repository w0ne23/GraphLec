import { useEffect, useRef, useState } from 'react'

const BADGE = {
  slide:    { cls: 'tl-badge--slide',    label: '슬라이드', emoji: '📋' },
  emphasis: { cls: 'tl-badge--emphasis', label: '⚡ 강조',  emoji: '⚡' },
  demo:     { cls: 'tl-badge--demo',     label: '💻 데모',  emoji: '💻' },
}

const VERTICAL_THRESHOLD = 300
const TOPBAR_H = 48

/**
 * VideoTimeline — 뷰어 하단/측면 타임라인
 * lp-left 크기를 감지하여 남는 공간에 따라 가로/세로 모드 자동 전환
 * 모든 모드에서 'tl-' 접두사 및 통합 스타일 시스템 사용
 */
export default function VideoTimeline({
  scenes = [],
  currentScene = 0,
  onSceneChange,
  leftRef,
  isOpen = true,
  onToggle,
  isCinemaMode = false,
}) {
  const listRef = useRef(null)
  const activeRef = useRef(null)
  const [isVertical, setIsVertical] = useState(false)
  const [timelineHeight, setTimelineHeight] = useState(0)

  // lp-left 크기 감지 → 남는 공간 계산
  useEffect(() => {
    const el = leftRef?.current
    if (!el) return
    const ro = new ResizeObserver(([entry]) => {
      const { width, height } = entry.contentRect
      const videoHeight = width / 16 * 9
      const remaining   = height - TOPBAR_H - videoHeight
      setIsVertical(remaining > VERTICAL_THRESHOLD)
      setTimelineHeight(Math.max(0, remaining))
    })
    ro.observe(el)
    return () => ro.disconnect()
  }, [leftRef])

  // 오토스크롤
  useEffect(() => {
    if (!activeRef.current || !listRef.current || !isOpen || isCinemaMode) return
    const list = listRef.current
    const card = activeRef.current

    if (isVertical) {
      // 세로: 현재 요소 높이의 50%만큼 상단 여유를 둠
      const target = card.offsetTop - (card.offsetHeight * 0.5)
      list.scrollTo({ top: Math.max(0, target), behavior: 'smooth' })
    } else {
      // 가로: 현재 요소 너비의 50%만큼 좌측 여유를 둠
      const target = card.offsetLeft - (card.offsetWidth * 0.5)
      list.scrollTo({ left: Math.max(0, target), behavior: 'smooth' })
    }
  }, [currentScene, isVertical, isOpen, isCinemaMode])

  // 상하 휠 → 가로 스크롤 변환 (가로 모드)
  useEffect(() => {
    if (isVertical || !listRef.current || !isOpen || isCinemaMode) return
    const el = listRef.current
    const onWheel = (e) => {
      if (Math.abs(e.deltaY) > Math.abs(e.deltaX)) {
        e.preventDefault()
        el.scrollLeft += e.deltaY
      }
    }
    el.addEventListener('wheel', onWheel, { passive: false })
    return () => el.removeEventListener('wheel', onWheel)
  }, [isVertical, isOpen, isCinemaMode])

  if (!scenes.length) {
    return (
      <div className="tl-empty" style={isVertical ? { height: timelineHeight } : {}}>
        타임라인 데이터가 없습니다.
      </div>
    )
  }

  // ── 렌더링 아이템 생성 ──
  const items = scenes.map((s, i) => {
    const badge    = BADGE[s.type] ?? BADGE.slide
    const isActive = i === currentScene

    // 세로 모드일 때는 tl-row, 가로 모드일 때는 tl-card
    if (isVertical) {
      return (
        <div
          key={i}
          ref={isActive ? activeRef : null}
          className={`tl-row ${isActive ? 'tl-row--active' : ''} ${s.type === 'emphasis' ? 'tl-row--emphasis' : ''}`}
          onClick={() => onSceneChange(i)}
        >
          <div className="tl-thumb">
            {s.image_url
              ? <img src={s.image_url} alt={`Slide ${s.slide_number}`} className="tl-thumb-img" />
              : badge.emoji
            }
          </div>
          <div className="tl-info">
            <div className="tl-header">
              <div className="tl-ts">{s.timestamp}</div>
              <span className={`tl-badge ${badge.cls}`}>{badge.label}</span>
            </div>
            <div className="tl-text">{s.text}</div>
          </div>
        </div>
      )
    }

    // 가로 모드 (Cards)
    return (
      <div
        key={i}
        ref={isActive ? activeRef : null}
        className={`tl-card ${isActive ? 'tl-card--active' : ''}`}
        onClick={() => onSceneChange(i)}
      >
        <div className="tl-thumb">
          {s.image_url
            ? <img src={s.image_url} alt={`slide ${s.slide_number}`} className="tl-thumb-img" />
            : <span className="tl-thumb-icon">{badge.emoji}</span>
          }
        </div>
        <div className="tl-info">
          <div className="tl-ts">{s.timestamp}</div>
          <div className="tl-title">{s.text || `씬 ${i + 1}`}</div>
          <span className={`tl-badge ${badge.cls}`}>{badge.label}</span>
        </div>
      </div>
    )
  })

  // ── 스타일 계산 ──
  const rootStyle = {
    height: isCinemaMode 
      ? '0px' 
      : (isOpen 
          ? (isVertical ? 'var(--tl-v-height)' : 'var(--tl-horz-h)') 
          : 'var(--tl-header-h)'
        ),
    opacity: isCinemaMode ? 0 : 1,
    pointerEvents: isCinemaMode ? 'none' : 'auto',
    borderTop: isCinemaMode ? 'none' : '1px solid var(--b1)'
  }

  return (
    <div 
      className={`${isVertical ? "tl-vertical" : "tl-wrap"} ${isOpen ? 'tl--open' : ''} ${isCinemaMode ? 'tl--cinema' : ''}`} 
      style={rootStyle}
    >
      <div className="tl-panel-header" onClick={onToggle}>
        <span className="tl-panel-title">타임라인</span>
        <div style={{ flex: 1 }} />
        <button className="tl-toggle-btn">{isOpen ? '▼' : '▲'}</button>
      </div>
      <div className="tl-body">
        <div 
          className={isVertical ? "tl-list--vert" : "tl-list--horz"} 
          ref={listRef}
        >
          {items}
        </div>
      </div>
    </div>
  )
}