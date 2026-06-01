import { BrowserRouter, Routes, Route } from 'react-router-dom'
import LandingPage   from './pages/LandingPage'
import MainPage      from './pages/MainPage'
import DevUploadPage from './pages/dev/DevUploadPage'
import RecommendPage from './pages/RecommendPage'
import LectureListPage from './pages/LectureListPage'
import LecturePage   from './pages/LecturePage'
import VerifierPage  from './pages/VerifierPage'
import DevVerifierPage from './pages/dev/DevVerifierPage'
import UploadPage from './pages/UploadPage'
import VerifierProgressPage from './pages/upload/VerifierProgressPage'
import VerifierResultPage from './pages/upload/VerifierResultPage'
import VerifierFinalizePage from './pages/upload/VerifierFinalizePage'
import VerifierDirectPage from './pages/upload/VerifierDirectPage'
import MainLayout    from './components/layout/MainLayout'

export default function App() {
  return (
    <BrowserRouter>
      <div className="app-shell">
        <div className="app-main">
          <Routes>
            {/* 헤더/탭바 X */}
            <Route path="/"                      element={<MainPage />} />
            <Route path="/landing"               element={<LandingPage />} />
            <Route path="/lectures/:id"          element={<LecturePage />} />
            <Route path="/dev/lectures/:id/verifier" element={<DevVerifierPage />} />

            {/* 헤더/탭바 O */}
            <Route element={<MainLayout />}>
              <Route path="/lectures"   element={<LectureListPage />} />
              <Route path="/verify" element={<VerifierPage />} />
              <Route path="/upload" element={<UploadPage />} />
              <Route path="/upload/:lectureId/progress" element={<VerifierProgressPage />} />
              <Route path="/upload/:lectureId/result" element={<VerifierResultPage />} />
              <Route path="/upload/:lectureId/finalize" element={<VerifierFinalizePage />} />
              <Route path="/upload/:lectureId/direct" element={<VerifierDirectPage />} />
              <Route path="/dev/upload" element={<DevUploadPage />} />
              <Route path="/recommend"  element={<RecommendPage />} />
            </Route>

            {/* Fallback */}
            <Route path="*" element={<MainPage />} />
          </Routes>
        </div>
      </div>
    </BrowserRouter>
  )
}
