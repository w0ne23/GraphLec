import { BrowserRouter, Routes, Route } from 'react-router-dom'
import LandingPage   from './pages/LandingPage'
import UploadPage    from './pages/UploadPage'
import RecommendPage from './pages/RecommendPage'
import LectureListPage from './pages/LectureListPage'
import LecturePage   from './pages/LecturePage'
import VerifierPage  from './pages/VerifierPage'
import MainLayout    from './components/layout/MainLayout'

export default function App() {
  return (
    <BrowserRouter>
      <div className="app-shell">
        <div className="app-main">
          <Routes>
            {/* 헤더/탭바 X */}
            <Route path="/"                      element={<LandingPage />} />
            <Route path="/lectures/:id"          element={<LecturePage />} />
            <Route path="/lectures/:id/verifier" element={<VerifierPage />} />

            {/* 헤더/탭바 O */}
            <Route element={<MainLayout />}>
              <Route path="/lectures"   element={<LectureListPage />} />
              <Route path="/upload"     element={<UploadPage />} />
              <Route path="/recommend"  element={<RecommendPage />} />
            </Route>

            {/* Fallback */}
            <Route path="*" element={<LandingPage />} />
          </Routes>
        </div>
      </div>
    </BrowserRouter>
  )
}
