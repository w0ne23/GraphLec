import { useNavigate } from 'react-router-dom'
import RaspberryIcon from '../common/RaspberryIcon'

export default function AppHeader({ logoTo = '/', inert = false, className = '' }) {
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
    <nav className={`app-header app-header-entry${className ? ` ${className}` : ''}`}>
      <button
        type="button"
        className="app-header-logo"
        onClick={() => go(logoTo)}
        aria-label="GraphLec 페이지로 이동"
        tabIndex={inert ? -1 : undefined}
      >
        <div className="app-header-logo-mark">
          <RaspberryIcon className="raspberry-icon" />
        </div>
        <div className="app-header-brand">Graph<span>Lec</span></div>
        <div className="app-header-badge">
          <span className="app-header-badge-dot" />
          멀티모달 강의 분석 솔루션
        </div>
      </button>
      <ul className="app-header-links">
        <li><a {...linkProps('/verify')}>Verify</a></li>
        <li><a {...linkProps('/recommend')}>Recommend</a></li>
        <li><a {...linkProps('/lectures')}>QnA</a></li>
      </ul>
    </nav>
  )
}
