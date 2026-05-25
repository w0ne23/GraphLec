import { NavLink } from 'react-router-dom'

export default function AppTabBar() {
  return (
    <nav className="app-tabbar">
      <NavLink to="/verify" className={({ isActive }) => 'app-tabbar-tab' + (isActive ? ' app-tabbar-tab--active' : '')}>
        <span>Verify</span>
      </NavLink>
      <NavLink to="/recommend" className={({ isActive }) => 'app-tabbar-tab' + (isActive ? ' app-tabbar-tab--active' : '')}>
        <span>Recommend</span>
      </NavLink>
      <NavLink to="/lectures" className={({ isActive }) => 'app-tabbar-tab' + (isActive ? ' app-tabbar-tab--active' : '')}>
        <span>QnA</span>
      </NavLink>
    </nav>
  )
}
