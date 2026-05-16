import { useNavigate } from 'react-router-dom'
// useRef는 스크롤 기능 제거로 더 이상 필요하지 않습니다.

import '../styles/landing.css'

export default function LandingPage() {
  const navigate = useNavigate()

  return (
    // 전체 페이지를 중앙 정렬하는 컨테이너
    <div className="ld-page">
      
      {/* global.css의 content-max 유틸리티 활용 */}
      <div className="ld-content content-max">
        
        {/* ── 헤더 영역: 로고 및 설명 ── */}
        <header className="ld-header">
          <div className="ld-logo">
            Graph<span>Lec</span>
          </div>
          <p className="ld-subtitle">영상 강의 토탈 솔루션</p>
        </header>

        {/* ── 메인 영역: 3개 선택지 카드 ── */}
        <div className="ld-cards">
          
          <button 
            className="ld-card"
            onClick={() => navigate('/upload')}
          >
            <div className="ld-card-icon">✅</div>
            <h3 className="ld-card-title">Verify</h3>
            <p className="ld-card-desc">
              강의를 업로드하여 강의 내용을 검토해 보세요
            </p>
          </button>

          <button 
            className="ld-card"
            onClick={() => navigate('/recommend')}
          >
            <div className="ld-card-icon">💡</div>
            <h3 className="ld-card-title">Recommend</h3>
            <p className="ld-card-desc">
              지금 내가 원하는 강의를 추천받으세요
            </p>
          </button>

          <button 
            className="ld-card"
            onClick={() => navigate('/lectures')}
          >
            <div className="ld-card-icon">💬</div>
            <h3 className="ld-card-title">QnA</h3>
            <p className="ld-card-desc">
              영상 강의를 보며 궁금한 점을 AI에게 바로 질문해 보세요
            </p>
          </button>

        </div>
      </div>
    </div>
  )
}
