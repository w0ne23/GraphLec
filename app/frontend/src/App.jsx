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
import UploadJobPage from './pages/upload/UploadJobPage'
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
            <Route path="/dev/lectures/:id/verifier" element={<DevVerifierPage />} />

            {/* 헤더/탭바 O */}
            <Route element={<MainLayout />}>
              <Route path="/lectures"   element={<LectureListPage />} />
              <Route path="/lectures/:id" element={<LecturePage />} />
              <Route path="/verify" element={<VerifierPage />} />
              <Route path="/upload" element={<UploadPage />} />
              <Route path="/verify/:lectureId" element={<UploadJobPage mode="verify" />} />
              <Route path="/publish/:lectureId" element={<UploadJobPage mode="publish" />} />
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
