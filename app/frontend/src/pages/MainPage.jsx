import { useNavigate } from 'react-router-dom'
import EntrySection from '../components/landing/EntrySection'
import RaspberryIcon from '../components/common/RaspberryIcon'
import '../styles/landing.css'

export default function MainPage() {
  const navigate = useNavigate()

  return (
    <div className="ld-landing">
      <nav className="ld-nav ld-nav-entry">
        <button type="button" className="nav-logo" onClick={() => navigate('/')} aria-label="메인 페이지로 이동">
          <div className="nav-logo-mark">
            <RaspberryIcon className="raspberry-icon" />
          </div>
          <div className="nav-brand">Graph<span>Lec</span></div>
        </button>
        <ul className="nav-links">
          <li><a href="/verify" onClick={(e) => { e.preventDefault(); navigate('/verify') }}>Verify</a></li>
          <li><a href="/recommend" onClick={(e) => { e.preventDefault(); navigate('/recommend') }}>Recommend</a></li>
          <li><a href="/lectures" onClick={(e) => { e.preventDefault(); navigate('/lectures') }}>QnA</a></li>
        </ul>
      </nav>

      <div id="stack">
        <div className="slide active" id="s5">
          <EntrySection onNavigate={navigate} />
        </div>
      </div>

      <footer className="slide-footer">
        <div>GraphLEC · ©2025 캡스톤 프로젝트</div>
        <div className="slide-footer-team">
          황베리 — 정다원, 김지민, 정규민, 김동석, 신지현
        </div>
      </footer>
    </div>
  )
}