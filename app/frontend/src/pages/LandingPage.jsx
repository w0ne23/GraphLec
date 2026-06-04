import { useCallback, useEffect, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import HeroSection from '../components/landing/HeroSection'
import VerifySection from '../components/landing/VerifySection'
import LearnerSection from '../components/landing/LearnerSection'
import WorkflowSection from '../components/landing/WorkflowSection'
import EntrySection from '../components/landing/EntrySection'
import AppHeader from '../components/layout/AppHeader'
import { usePageTitle } from '../hooks/usePageTitle'
import '../styles/landing.css'

export default function LandingPage() {
  const navigate = useNavigate()
  usePageTitle()
  const [exiting, setExiting] = useState(false)
  const animating = useRef(false)
  const exitTimer = useRef(null)

  const goMain = useCallback(() => {
    if (animating.current) return
    animating.current = true
    setExiting(true)
    exitTimer.current = setTimeout(() => {
      navigate('/', { replace: true })
    }, 860)
  }, [navigate])

  useEffect(() => {
    return () => clearTimeout(exitTimer.current)
  }, [])

  // Wheel
  useEffect(() => {
    let accum = 0
    let timer = null
    const onWheel = (e) => {
      e.preventDefault()
      accum += e.deltaY
      clearTimeout(timer)
      timer = setTimeout(() => {
        if (accum > 60) goMain()
        accum = 0
      }, 80)
    }
    window.addEventListener('wheel', onWheel, { passive: false })
    return () => window.removeEventListener('wheel', onWheel)
  }, [goMain])

  // Keyboard
  useEffect(() => {
    const onKey = (e) => {
      if (['ArrowDown', 'ArrowRight', 'PageDown', 'Enter', ' '].includes(e.key)) goMain()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [goMain])

  // Touch
  useEffect(() => {
    let y0 = 0
    const onStart = (e) => { y0 = e.touches[0].clientY }
    const onEnd   = (e) => {
      const dy = y0 - e.changedTouches[0].clientY
      if (dy > 50) goMain()
    }
    window.addEventListener('touchstart', onStart, { passive: true })
    window.addEventListener('touchend',   onEnd,   { passive: true })
    return () => {
      window.removeEventListener('touchstart', onStart)
      window.removeEventListener('touchend',   onEnd)
    }
  }, [goMain])

  return (
    <div className={`ld-landing${exiting ? ' ld-landing-exit' : ''}`}>
      <div className="ld-main-underlay" aria-hidden="true">
        <AppHeader logoTo="/" inert />
        <div id="stack">
          <div className="slide active" id="s5">
            <EntrySection onNavigate={() => {}} />
          </div>
        </div>
        <footer className="slide-footer">
          <div>GraphLEC · ©2026 캡스톤 프로젝트</div>
          <div className="slide-footer-team">
            황베리 — 정다원, 김지민, 정규민, 김동석, 신지현
          </div>
        </footer>
      </div>

      <div className="ld-hero-overlay">
      {/* NAV */}
      <AppHeader logoTo="/landing" />

      {/* SLIDE STACK */}
      <div id="stack">
        <div className="slide active" id="s1">
          <HeroSection current={0} onNext={goMain} />
        </div>
      </div>

      {/* FOOTER */}
      <footer className="slide-footer">
        <div>GraphLEC · ©2026 캡스톤 프로젝트</div>
        <div className="slide-footer-team">
          황베리 — 정다원, 김지민, 정규민, 김동석, 신지현
        </div>
      </footer>

      </div>
    </div>
  )
}

void VerifySection
void LearnerSection
void WorkflowSection
