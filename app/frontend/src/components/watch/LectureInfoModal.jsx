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

  const info = lecture.info || {}
  const stats = info.stats || {}
  const keywords = info.keywords ?? lecture.keywords ?? []
  const highlights = info.highlights ?? []
  const summary = info.summary || lecture.summary || '강의 분석이 완료되면 요약이 표시됩니다.'
  const domain = info.domain || lecture.domain || lecture.category || '기타'
  const sceneCount = Number.isFinite(Number(stats.scene_count))
    ? Number(stats.scene_count)
    : scenes.length
  const emphasisCount = Number.isFinite(Number(stats.emphasis_contexts))
    ? Number(stats.emphasis_contexts)
    : scenes.filter(s => s.type === 'emphasis').length
  const visualPercent = Number.isFinite(Number(stats.visual_asset_percent ?? info.visual_asset_percent))
    ? `${Math.round(Number(stats.visual_asset_percent ?? info.visual_asset_percent))}%`
    : '0%'
  const uploadedAt = lecture.created_at
    ? new Date(lecture.created_at).toLocaleDateString('ko-KR')
    : '알 수 없음'

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
            <p>{summary}</p>
          </div>

          {/* 강조 구간 */}
          {highlights.length > 0 && (
            <div className="lim-block">
              <h4>⚡ 강조 구간</h4>
              <div className="lim-highlight-list">
                {highlights.map((s, i) => (
                  <div key={i} className="lim-highlight-row">
                    <strong style={{ color: 'var(--amber)' }}>{s.timestamp}</strong>
                    {' '}{s.text}
                  </div>
                ))}
              </div>
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
                <div className="lim-stat-num" style={{ color: 'var(--blue)' }}>{sceneCount}</div>
                <div className="lim-stat-lbl">장면</div>
              </div>
              <div className="lim-stat-box">
                <div className="lim-stat-num" style={{ color: 'var(--amber)' }}>{emphasisCount}</div>
                <div className="lim-stat-lbl">강조 구간</div>
              </div>
              <div className="lim-stat-box">
                <div className="lim-stat-num" style={{ color: 'var(--green)' }}>{visualPercent}</div>
                <div className="lim-stat-lbl">시각자료</div>
              </div>
            </div>
            <div className="lim-meta">
              <strong>카테고리:</strong> {domain}<br />
              <strong>업로드:</strong> {uploadedAt}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
