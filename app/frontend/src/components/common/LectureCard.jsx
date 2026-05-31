const THUMB_COLORS = {
  engineering: '#1e3a5f',
  natural_science: '#3d2a1a',
  humanities: '#4b3426',
  social_science: '#314155',
  arts: '#5a2d54',
  health_sciences: '#1f4c3b',
  sports: '#45512a',
  education: '#3f3b67',
  etc: '#4b5563',
}

const THUMB_ICONS = {
  engineering: '🧠',
  natural_science: '📐',
  humanities: '📚',
  social_science: '🏛️',
  arts: '🎨',
  health_sciences: '⚕️',
  sports: '🏃',
  education: '🎓',
  etc: '🎬',
}

const STATUS_MAP = {
  done:       { label: '분석 완료', cls: 'sDone' },
  processing: { label: '분석 중',   cls: 'sProc' },
  pending:    { label: '대기',       cls: 'sWait' },
  error:      { label: '오류',       cls: 'sErr'  },
}

/**
 * @param {object}   lecture
 * @param {boolean}  showStatus
 * @param {function} onSelect  - 카드 클릭 시 호출. QueryPage 추천 목록 등에서 전달.
 */
export default function LectureCard({ lecture, showStatus = false, onSelect }) {
  const status  = STATUS_MAP[lecture.status] ?? STATUS_MAP.pending
  const thumbBg = THUMB_COLORS[lecture.category] ?? '#1e2333'

  function handleClick() {
    onSelect?.(lecture)
  }

  return (
    <div className="card" onClick={handleClick}>
      <div className="lecture-card-thumb" style={{ background: thumbBg }}>
        <div className="lecture-card-thumb-icon">
          {THUMB_ICONS[lecture.category] ?? THUMB_ICONS.etc}
        </div>
        {showStatus && (
          <div className={`lecture-card-status-badge ${status.cls}`}>
            {status.label}
          </div>
        )}
      </div>

      <div className="lecture-card-info">
        <div className="lecture-card-category">{lecture.category}</div>
        <div className="lecture-card-title">{lecture.title}</div>
        {showStatus && (
          <div className="lecture-card-meta">
            <span className="date">{new Date(lecture.created_at).toLocaleDateString('ko-KR')}</span>
          </div>
        )}
        {lecture.tags?.length > 0 && (
          <div className="lecture-card-tags">
            {lecture.tags.map(tag => (
              <span key={tag} className="lecture-card-tag">{tag}</span>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
