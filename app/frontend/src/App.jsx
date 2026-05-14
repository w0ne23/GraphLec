import { useCallback, useState } from 'react'
import { BrowserRouter, Routes, Route, useNavigate, useLocation, NavLink } from 'react-router-dom'
import LandingPage   from './pages/LandingPage'
import UploadPage    from './pages/UploadPage'
import RecommendPage from './pages/RecommendPage'
import LectureListPage from './pages/LectureListPage'
import LecturePage   from './pages/LecturePage'
import VerifierPage  from './pages/VerifierPage'
import SplashScreen from './components/common/SplashScreen'

const HIDE_HEADER_PATHS = ['/']
const HIDE_HEADER_PATTERNS = [/^\/lectures\/[^/]+(\/verifier)?$/]

function AppHeader() {
  const location = useLocation()
  const hide =
    HIDE_HEADER_PATHS.includes(location.pathname) ||
    HIDE_HEADER_PATTERNS.some(p => p.test(location.pathname))
  if (hide) return null

  return (
    <header className="app-header">
      <NavLink to="/" className="app-header-logo">
        Graph<span>Lec</span>
      </NavLink>
      <nav className="app-header-nav">
        <NavLink to="/upload"    className={({ isActive }) => 'app-header-tab' + (isActive ? ' app-header-tab--active' : '')}>Upload</NavLink>
        <NavLink to="/recommend" className={({ isActive }) => 'app-header-tab' + (isActive ? ' app-header-tab--active' : '')}>Recommend</NavLink>
        <NavLink to="/lectures"  className={({ isActive }) => 'app-header-tab' + (isActive ? ' app-header-tab--active' : '')}>Browse</NavLink>
      </nav>
    </header>
  )
}

function AppTabBar() {
  const location = useLocation()
  const hide =
    HIDE_HEADER_PATHS.includes(location.pathname) ||
    HIDE_HEADER_PATTERNS.some(p => p.test(location.pathname))
  if (hide) return null

  return (
    <nav className="app-tabbar">
      <NavLink to="/upload"    className={({ isActive }) => 'app-tabbar-tab' + (isActive ? ' app-tabbar-tab--active' : '')}>
        <i className="ti ti-upload" aria-hidden="true" />
        <span>Upload</span>
      </NavLink>
      <NavLink to="/recommend" className={({ isActive }) => 'app-tabbar-tab' + (isActive ? ' app-tabbar-tab--active' : '')}>
        <i className="ti ti-bulb" aria-hidden="true" />
        <span>Recommend</span>
      </NavLink>
      <NavLink to="/lectures"  className={({ isActive }) => 'app-tabbar-tab' + (isActive ? ' app-tabbar-tab--active' : '')}>
        <i className="ti ti-books" aria-hidden="true" />
        <span>Browse</span>
      </NavLink>
    </nav>
  )
}

function MainLayout() {
  const navigate = useNavigate()

  const handleNavigate = useCallback(({ page, lectureId = null }) => {
    if (page === 'lecture' && lectureId != null) {
      navigate(`/lectures/${lectureId}`)
      return
    }
    if (page === 'verifier' && lectureId != null) {
      navigate(`/lectures/${lectureId}/verifier`)
      return
    }
    navigate(`/${page}`)
  }, [navigate])

  return (
    <div className="app-shell">
      <AppHeader />
      <div className="app-main">
        <Routes>
          <Route path="/"                      element={<LandingPage />} />
          <Route path="/lectures"              element={<LectureListPage onNavigate={handleNavigate} />} />
          <Route path="/upload"                element={<UploadPage onNavigate={handleNavigate} />} />
          <Route path="/lectures/:id"          element={<LecturePage onNavigate={handleNavigate} />} />
          <Route path="/lectures/:id/verifier" element={<VerifierPage />} />
          <Route path="/recommend"             element={<RecommendPage onNavigate={handleNavigate} />} />
          <Route path="*"                      element={<LandingPage />} />
        </Routes>
      </div>
      <AppTabBar />
    </div>
  )
}

export default function App() {
  const [showSplash, setShowSplash] = useState(true)

  return (
    <BrowserRouter>
      <MainLayout />
      {showSplash && <SplashScreen onDone={() => setShowSplash(false)} />}
    </BrowserRouter>
  )
}
