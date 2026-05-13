import { NavLink } from 'react-router-dom'

export default function AppTabBar() {
  return (
    <nav className="app-tabbar">
      <NavLink to="/upload" className={({ isActive }) => 'app-tabbar-tab' + (isActive ? ' app-tabbar-tab--active' : '')}>
        <i className="ti ti-upload" aria-hidden="true" />
        <span>Upload</span>
      </NavLink>
      <NavLink to="/recommend" className={({ isActive }) => 'app-tabbar-tab' + (isActive ? ' app-tabbar-tab--active' : '')}>
        <i className="ti ti-bulb" aria-hidden="true" />
        <span>Recommend</span>
      </NavLink>
      <NavLink to="/lectures" className={({ isActive }) => 'app-tabbar-tab' + (isActive ? ' app-tabbar-tab--active' : '')}>
        <i className="ti ti-books" aria-hidden="true" />
        <span>Browse</span>
      </NavLink>
    </nav>
  )
}
