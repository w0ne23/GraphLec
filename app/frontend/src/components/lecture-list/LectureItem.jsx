/**
 * 강의 목록 아이템 컴포넌트
 * LectureListPage의 그리드/리스트 뷰를 모두 지원합니다.
 * 
 * @param {object}   lecture    - 강의 데이터 객체
 * @param {string}   viewMode   - 'grid' | 'list' (기본값 'grid')
 * @param {function} onClick    - 카드 클릭 시 호출되는 핸들러
 */
export default function LectureItem({ lecture, viewMode = 'grid', onClick }) {
  const isDone = lecture.status === 'done';
  
  return (
    <article 
      className={`lecture-card ${viewMode === 'list' ? 'lecture-card--list' : ''}`}
      onClick={() => isDone && onClick?.(lecture)}
      style={{ cursor: isDone ? 'pointer' : 'default' }}
    >
      <div className="lecture-card-thumb" style={{ background: 'var(--card)' }}>
        <span className="lecture-card-thumb-icon">
          {lecture.category === '수학' ? '📐' : '🎬'}
        </span>
        {!isDone && (
          <span className={`lecture-card-status-badge status-${lecture.status}`}>
            {lecture.status === 'error' ? '오류' : '분석 중'}
          </span>
        )}
      </div>
      
      <div className="lecture-card-info">
        <div className="lecture-card-category">{lecture.category}</div>
        <h3 className="lecture-card-title">{lecture.title}</h3>
        <div className="lecture-card-meta">
          {new Date(lecture.created_at).toLocaleDateString()}
        </div>
        {lecture.tags?.length > 0 && (
          <div className="lecture-card-tags">
            {lecture.tags.slice(0, 3).map(tag => (
              <span key={tag} className="lecture-card-tag">{tag}</span>
            ))}
          </div>
        )}
      </div>
    </article>
  );
}
