export default function EntrySection({ onNavigate }) {
  return (
    <div className="ld-page">
      <div className="ld-content">
        <header className="ld-header">
          <div className="ld-logo">
            Graph<span>Lec</span>
          </div>
          <p className="ld-subtitle">영상 강의 토탈 솔루션</p>
        </header>

        <div className="ld-cards">
          <button
            className="ld-card"
            onClick={() => onNavigate('/upload')}
          >
            <div className="feature-icon-wrap ld-entry-icon">
              <svg className="feature-icon" viewBox="0 0 24 24">
                <path d="M9 12l2 2 4-4" />
                <circle cx="12" cy="12" r="9" />
              </svg>
            </div>
            <h3 className="ld-card-title">Verify</h3>
            <p className="ld-card-desc">
              강의를 업로드하여<br />강의 내용을 검토해 보세요
            </p>
          </button>

          <button
            className="ld-card"
            onClick={() => onNavigate('/recommend')}
          >
            <div className="feature-icon-wrap ld-entry-icon">
              <svg className="feature-icon" viewBox="0 0 24 24">
                <circle cx="12" cy="12" r="3" />
                <path d="M12 2v3M12 19v3M4.22 4.22l2.12 2.12M17.66 17.66l2.12 2.12M2 12h3M19 12h3M4.22 19.78l2.12-2.12M17.66 6.34l2.12-2.12" />
              </svg>
            </div>
            <h3 className="ld-card-title">Recommend</h3>
            <p className="ld-card-desc">
              지금 내가 원하는 강의를 추천받으세요
            </p>
          </button>

          <button
            className="ld-card"
            onClick={() => onNavigate('/lectures')}
          >
            <div className="feature-icon-wrap ld-entry-icon">
              <svg className="feature-icon" viewBox="0 0 24 24">
                <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
              </svg>
            </div>
            <h3 className="ld-card-title">QnA</h3>
            <p className="ld-card-desc">
              영상 강의를 보며 궁금한 점을<br />AI에게 바로 질문해 보세요
            </p>
          </button>
        </div>
      </div>
    </div>
  )
}
