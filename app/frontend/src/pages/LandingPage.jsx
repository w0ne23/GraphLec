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
          <p className="ld-desc">
            Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod
            tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim
            veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea
            commodo consequat.
          </p>
        </header>

        {/* ── 메인 영역: 3개 선택지 카드 ── */}
        <div className="ld-cards">
          
          <button 
            className="ld-card"
            onClick={() => navigate('/upload')}
          >
            <div className="ld-card-icon">📤</div>
            <h3 className="ld-card-title">강의 업로드</h3>
            <p className="ld-card-desc">
              강의를 업로드하고 AI의 분석을 통해 검토해 보세요
            </p>
          </button>

          <button 
            className="ld-card"
            onClick={() => navigate('/recommend')}
          >
            <div className="ld-card-icon">💡</div>
            <h3 className="ld-card-title">원하는 강의 추천받기</h3>
            <p className="ld-card-desc">
              지금 내가 원하는 강의를 추천받으세요
            </p>
          </button>

          <button 
            className="ld-card"
            onClick={() => navigate('/lectures')}
          >
            <div className="ld-card-icon">📚</div>
            <h3 className="ld-card-title">모든 강의 둘러보기</h3>
            <p className="ld-card-desc">
              어떤 강의가 있는지 살펴보세요
            </p>
          </button>

        </div>
      </div>
    </div>
  )
}