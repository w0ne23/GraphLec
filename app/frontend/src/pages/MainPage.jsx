import { useNavigate } from 'react-router-dom'
import EntrySection from '../components/landing/EntrySection'
import AppHeader from '../components/layout/AppHeader'
import { usePageTitle } from '../hooks/usePageTitle'
import '../styles/landing.css'

export default function MainPage() {
  const navigate = useNavigate()
  usePageTitle()

  return (
    <div className="ld-landing">
      <AppHeader logoTo="/" />

      <div id="stack">
        <div className="slide active" id="s5">
          <EntrySection onNavigate={navigate} />
        </div>
      </div>

      <footer className="slide-footer">
        <div>GraphLEC · ©2026 캡스톤 프로젝트</div>
        <div className="slide-footer-team">
          황베리 — 정다원, 김지민, 정규민, 김동석, 신지현
        </div>
      </footer>
    </div>
  )
}
