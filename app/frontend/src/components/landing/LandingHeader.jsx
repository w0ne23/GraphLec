import { useNavigate } from 'react-router-dom'
import RaspberryIcon from '../common/RaspberryIcon'

export default function LandingHeader({ logoTo = '/', inert = false }) {
  const navigate = useNavigate()

  const go = (path) => {
    if (inert) return
    navigate(path)
  }

  const linkProps = (path) => ({
    href: path,
    tabIndex: inert ? -1 : undefined,
    onClick: (event) => {
      event.preventDefault()
      go(path)
    },
  })

  return (
    <nav className="ld-nav ld-nav-entry">
      <button
        type="button"
        className="nav-logo"
        onClick={() => go(logoTo)}
        aria-label="GraphLec 페이지로 이동"
        tabIndex={inert ? -1 : undefined}
      >
        <div className="nav-logo-mark">
          <RaspberryIcon className="raspberry-icon" />
        </div>
        <div className="nav-brand">Graph<span>Lec</span></div>
        <div className="nav-hero-badge">
          <span className="badge-dot" />
          멀티모달 강의 분석 솔루션
        </div>
      </button>
      <ul className="nav-links">
        <li><a {...linkProps('/upload')}>Verify</a></li>
        <li><a {...linkProps('/recommend')}>Recommend</a></li>
        <li><a {...linkProps('/lectures')}>QnA</a></li>
      </ul>
    </nav>
  )
}
