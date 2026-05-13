import { Outlet } from 'react-router-dom'
import AppHeader from './AppHeader'
import AppTabBar from './AppTabBar'

export default function MainLayout() {
  return (
    <div className="main-layout">
      <AppHeader />
      <main className="main-layout-content">
        <Outlet />
      </main>
      <AppTabBar />
    </div>
  )
}
