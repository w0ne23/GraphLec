import { useCallback, useEffect, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import HeroSection from '../components/landing/HeroSection'
import VerifySection from '../components/landing/VerifySection'
import LearnerSection from '../components/landing/LearnerSection'
import WorkflowSection from '../components/landing/WorkflowSection'
import EntrySection from '../components/landing/EntrySection'
import RaspberryIcon from '../components/common/RaspberryIcon'
import '../styles/landing.css'

const TOTAL = 5
const DOT_THEMES = ['on-dark', 'on-light', 'on-light', 'on-light', 'on-light']

export default function LandingPage() {
  const navigate = useNavigate()
  const [current, setCurrent] = useState(0)
  const animating = useRef(false)

  const goTo = useCallback((idx) => {
    if (animating.current || idx === current || idx < 0 || idx >= TOTAL) return
    animating.current = true
    setCurrent(idx)
    setTimeout(() => { animating.current = false }, 850)
  }, [current])

  const goNext = useCallback(() => goTo(current + 1), [current, goTo])
  const goPrev = useCallback(() => goTo(current - 1), [current, goTo])

  // Wheel
  useEffect(() => {
    let accum = 0
    let timer = null
    const onWheel = (e) => {
      e.preventDefault()
      accum += e.deltaY
      clearTimeout(timer)
      timer = setTimeout(() => {
        if (accum > 60)       goNext()
        else if (accum < -60) goPrev()
        accum = 0
      }, 80)
    }
    window.addEventListener('wheel', onWheel, { passive: false })
    return () => window.removeEventListener('wheel', onWheel)
  }, [goNext, goPrev])

  // Keyboard
  useEffect(() => {
    const onKey = (e) => {
      if (['ArrowDown', 'ArrowRight', 'PageDown'].includes(e.key)) goNext()
      if (['ArrowUp',   'ArrowLeft',  'PageUp'  ].includes(e.key)) goPrev()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [goNext, goPrev])

  // Touch
  useEffect(() => {
    let y0 = 0
    const onStart = (e) => { y0 = e.touches[0].clientY }
    const onEnd   = (e) => {
      const dy = y0 - e.changedTouches[0].clientY
      if (dy >  50) goNext()
      if (dy < -50) goPrev()
    }
    window.addEventListener('touchstart', onStart, { passive: true })
    window.addEventListener('touchend',   onEnd,   { passive: true })
    return () => {
      window.removeEventListener('touchstart', onStart)
      window.removeEventListener('touchend',   onEnd)
    }
  }, [goNext, goPrev])

  const slideClass = (i) => {
    if (i < current)  return 'slide gone'
    if (i === current) return 'slide active'
    return 'slide'
  }

  const dotClass = (i) => {
    const base = `pdot ${DOT_THEMES[i]}`
    return i === current ? `${base} active` : base
  }

  const isEntrySlide = current === 4

  return (
    <div className="ld-landing">
      {/* NAV */}
      <nav className={`ld-nav${isEntrySlide ? ' ld-nav-entry' : ''}`}>
        <button type="button" className="nav-logo" onClick={() => goTo(0)} aria-label="첫 번째 랜딩 페이지로 이동">
          <div className="nav-logo-mark">
            <RaspberryIcon className="raspberry-icon" />
          </div>
          <div className="nav-brand">Graph<span>Lec</span></div>
        </button>
        <ul className="nav-links">
          {isEntrySlide ? (
            <>
              <li><a href="/verify" onClick={(e) => { e.preventDefault(); navigate('/verify') }}>Verify</a></li>
              <li><a href="/recommend" onClick={(e) => { e.preventDefault(); navigate('/recommend') }}>Recommend</a></li>
              <li><a href="/lectures" onClick={(e) => { e.preventDefault(); navigate('/lectures') }}>QnA</a></li>
            </>
          ) : (
            <>
              <li><a href="#" onClick={(e) => { e.preventDefault(); goTo(1) }}>강의자 기능</a></li>
              <li><a href="#" onClick={(e) => { e.preventDefault(); goTo(2) }}>학습자 기능</a></li>
              <li><a href="#" onClick={(e) => { e.preventDefault(); goTo(3) }}>작동 방식</a></li>
              <li><a href="#" className="nav-cta" onClick={(e) => { e.preventDefault(); goTo(4) }}>시작하기</a></li>
            </>
          )}
        </ul>
      </nav>

      {/* SLIDE STACK */}
      <div id="stack">
        <div className={slideClass(0)} id="s1">
          <HeroSection current={current} onNext={goNext} />
        </div>
        <div className={slideClass(1)} id="s2">
          <VerifySection />
        </div>
        <div className={slideClass(2)} id="s3">
          <LearnerSection />
        </div>
        <div className={slideClass(3)} id="s4">
          <WorkflowSection active={current === 3} />
        </div>
        <div className={slideClass(4)} id="s5">
          <EntrySection onNavigate={navigate} />
        </div>
      </div>

      {/* FOOTER */}
      <footer className="slide-footer">
        <div>GraphLEC · ©2025 캡스톤 프로젝트</div>
        <div className="slide-footer-team">
          황베리 — 정다원, 김지민, 정규민, 김동석, 신지현
        </div>
      </footer>

      {/* PROGRESS DOTS */}
      <div className="progress-dots">
        {DOT_THEMES.map((_, i) => (
          <div key={i} className={dotClass(i)} onClick={() => goTo(i)} />
        ))}
      </div>
    </div>
  )
}