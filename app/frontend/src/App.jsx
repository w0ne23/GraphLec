import { useState } from 'react'
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import LandingPage   from './pages/LandingPage'
import DevUploadPage from './pages/dev/DevUploadPage'
import RecommendPage from './pages/RecommendPage'
import LectureListPage from './pages/LectureListPage'
import LecturePage   from './pages/LecturePage'
import VerifierPage  from './pages/VerifierPage'
import DevVerifierPage from './pages/dev/DevVerifierPage'
import MainLayout    from './components/layout/MainLayout'
import SplashScreen from './components/common/SplashScreen'

export default function App() {
  const [showSplash, setShowSplash] = useState(true)

  return (
    <BrowserRouter>
      <div className="app-shell">
        <div className="app-main">
          <Routes>
            {/* 헤더/탭바 X */}
            <Route path="/"                      element={<LandingPage />} />
            <Route path="/lectures/:id"          element={<LecturePage />} />
            <Route path="/dev/lectures/:id/verifier" element={<DevVerifierPage />} />

            {/* 헤더/탭바 O */}
            <Route element={<MainLayout />}>
              <Route path="/lectures"   element={<LectureListPage />} />
              <Route path="/verify" element={<VerifierPage />} />
              <Route path="/dev/upload" element={<DevUploadPage />} />
              <Route path="/recommend"  element={<RecommendPage />} />
            </Route>

            {/* Fallback */}
            <Route path="*" element={<LandingPage />} />
          </Routes>
        </div>
      </div>
      {showSplash && <SplashScreen onDone={() => setShowSplash(false)} />}
    </BrowserRouter>
  )
}
