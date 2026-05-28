import { BrowserRouter, Routes, Route } from 'react-router-dom'
import LandingPage   from './pages/LandingPage'
import DevUploadPage from './pages/dev/DevUploadPage'
import RecommendPage from './pages/RecommendPage'
import LectureListPage from './pages/LectureListPage'
import LecturePage   from './pages/LecturePage'
import VerifierPage  from './pages/VerifierPage'
import DevVerifierPage from './pages/dev/DevVerifierPage'
import DevVerifierPreviewPage from './pages/dev/DevVerifierPreviewPage'
import VerifierFinalizePage from './pages/verify/VerifierFinalizePage'
import VerifierProgressPage from './pages/verify/VerifierProgressPage'
import VerifierResultPage from './pages/verify/VerifierResultPage'
import VerifierUploadProgressPage from './pages/verify/VerifierUploadProgressPage'
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
            <Route path="/dev/lectures/:id/verifier" element={<DevVerifierPage />} />

            {/* 헤더/탭바 O */}
            <Route element={<MainLayout />}>
              <Route path="/lectures"   element={<LectureListPage />} />
              <Route path="/verify" element={<VerifierPage />} />
              <Route path="/verify/:lectureId/progress" element={<VerifierProgressPage />} />
              <Route path="/verify/:lectureId/result" element={<VerifierResultPage />} />
              <Route path="/verify/:lectureId/finalize" element={<VerifierFinalizePage />} />
              <Route path="/verify/:lectureId/upload-progress" element={<VerifierUploadProgressPage />} />
              <Route path="/dev/upload" element={<DevUploadPage />} />
              <Route path="/dev/verifier-preview" element={<DevVerifierPreviewPage />} />
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
