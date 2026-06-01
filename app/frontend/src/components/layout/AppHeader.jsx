import { NavLink } from 'react-router-dom'

export default function AppHeader() {
  return (
    <header className="app-header">
      <NavLink to="/" className="app-header-logo">
        <span className="app-header-logo-mark">GL</span>
        <span className="app-header-brand">Graph<span>Lec</span></span>
      </NavLink>
      <nav className="app-header-nav">
        <NavLink to="/verify" className={({ isActive }) => 'app-header-tab' + (isActive ? ' app-header-tab--active' : '')}>Verify</NavLink>
        <NavLink to="/recommend" className={({ isActive }) => 'app-header-tab' + (isActive ? ' app-header-tab--active' : '')}>Recommend</NavLink>
        <NavLink to="/lectures" className={({ isActive }) => 'app-header-tab' + (isActive ? ' app-header-tab--active' : '')}>QnA</NavLink>
      </nav>
    </header>
  )
}
