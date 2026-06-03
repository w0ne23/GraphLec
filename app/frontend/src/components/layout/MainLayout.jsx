import { Outlet, useMatch } from 'react-router-dom'
import AppTabBar from './AppTabBar'
import AppFooter from './AppFooter'
import AppHeader from './AppHeader'

export default function MainLayout() {
  const isLectureViewer = Boolean(useMatch('/lectures/:id'))

  return (
    <div className="main-layout">
      <AppHeader logoTo="/" className="app-site-header" />
      <main className="main-layout-content">
        <Outlet />
      </main>
      {!isLectureViewer && <AppFooter />}
      <AppTabBar />
    </div>
  )
}
