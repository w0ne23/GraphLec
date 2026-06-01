export default function EntrySection({ onNavigate }) {
  return (
    <div className="ld-page">
      <div className="ld-content content-max">
        <header className="ld-header">
          <div className="ld-logo">
            Graph<span>Lec</span>
          </div>
          <p className="ld-subtitle">영상 강의 토탈 솔루션</p>
        </header>

        <div className="ld-cards">
          <button
            className="ld-card"
            onClick={() => onNavigate('/verify')}
          >
            <div className="ld-card-icon">✓</div>
            <h3 className="ld-card-title">Verify</h3>
            <p className="ld-card-desc">
              강의를 업로드하여 강의 내용을 검토해 보세요
            </p>
          </button>

          <button
            className="ld-card"
            onClick={() => onNavigate('/recommend')}
          >
            <div className="ld-card-icon">✦</div>
            <h3 className="ld-card-title">Recommend</h3>
            <p className="ld-card-desc">
              지금 내가 원하는 강의를 추천받으세요
            </p>
          </button>

          <button
            className="ld-card"
            onClick={() => onNavigate('/lectures')}
          >
            <div className="ld-card-icon">?</div>
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