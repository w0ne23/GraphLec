import { NavLink } from 'react-router-dom'

export default function AppHeader() {
  return (
    <header className="app-header">
      <NavLink to="/" className="app-header-logo">
        Graph<span>Lec</span>
      </NavLink>
      <nav className="app-header-nav">
        <NavLink to="/upload" className={({ isActive }) => 'app-header-tab' + (isActive ? ' app-header-tab--active' : '')}>Upload</NavLink>
        <NavLink to="/recommend" className={({ isActive }) => 'app-header-tab' + (isActive ? ' app-header-tab--active' : '')}>Recommend</NavLink>
        <NavLink to="/lectures" className={({ isActive }) => 'app-header-tab' + (isActive ? ' app-header-tab--active' : '')}>Browse</NavLink>
      </nav>
    </header>
  )
}
