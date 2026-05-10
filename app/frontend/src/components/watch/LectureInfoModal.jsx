/**
 * LectureInfoModal — 강의 상세 정보 모달
 * topbar ℹ️ 버튼으로 열기/닫기
 *
 * @param {Object} lecture — 강의 객체
 * @param {Array}  scenes — 씬 배열
 * @param {Function} onClose — 닫기 핸들러
 */
export default function LectureInfoModal({ lecture, scenes = [], onClose }) {
  if (!lecture) return null

  const keywords = lecture.keywords ?? []
  const emphasis = scenes.filter(s => s.type === 'emphasis')

  return (
    <div className="lim-backdrop" onClick={onClose}>
      <div className="lim-panel" onClick={e => e.stopPropagation()}>
        <div className="lim-header">
          <span className="lim-title">{lecture.title}</span>
          <button className="lim-close" onClick={onClose}>✕</button>
        </div>

        <div className="lim-body">
          {/* 요약 */}
          <div className="lim-block">
            <h4>📌 핵심 요약</h4>
            <p>{lecture.summary || '강의 분석이 완료되면 요약이 표시됩니다.'}</p>
          </div>

          {/* 강조 구간 */}
          {emphasis.length > 0 && (
            <div className="lim-block">
              <h4>⚡ 강조 구간</h4>
              <p>
                {emphasis.map((s, i) => (
                  <span key={i}>
                    <strong style={{ color: 'var(--amber)' }}>{s.timestamp}</strong>
                    {' '}{s.text}
                    {i < emphasis.length - 1 ? ' · ' : ''}
                  </span>
                ))}
              </p>
            </div>
          )}

          {/* 키워드 */}
          <div className="lim-block">
            <h4>🔑 주요 키워드</h4>
            <div className="lim-kw-cloud">
              {keywords.length === 0
                ? <span className="lim-empty">분석 완료 후 표시됩니다.</span>
                : keywords.map((kw, i) => <span key={i} className="lim-kw">{kw}</span>)
              }
            </div>
          </div>

          {/* 통계 */}
          <div className="lim-block">
            <h4>ℹ️ 강의 상세</h4>
            <div className="lim-stat-row">
              <div className="lim-stat-box">
                <div className="lim-stat-num" style={{ color: 'var(--blue)' }}>{scenes.length}</div>
                <div className="lim-stat-lbl">장면 전환</div>
              </div>
              <div className="lim-stat-box">
                <div className="lim-stat-num" style={{ color: 'var(--amber)' }}>{emphasis.length}</div>
                <div className="lim-stat-lbl">강조 구간</div>
              </div>
              <div className="lim-stat-box">
                <div className="lim-stat-num" style={{ color: 'var(--green)' }}>94%</div>
                <div className="lim-stat-lbl">STT 신뢰도</div>
              </div>
            </div>
            <div className="lim-meta">
              <strong>강의자:</strong> {lecture.instructor_name}<br />
              <strong>카테고리:</strong> {lecture.category}<br />
              <strong>업로드:</strong> {new Date(lecture.created_at).toLocaleDateString('ko-KR')}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

