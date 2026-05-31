/**
 * 강의 목록 아이템 컴포넌트
 * LectureListPage의 그리드/리스트 뷰를 모두 지원합니다.
 * 
 * @param {object}   lecture    - 강의 데이터 객체
 * @param {string}   viewMode   - 'grid' | 'list' (기본값 'grid')
 * @param {function} onClick    - 카드 클릭 시 호출되는 핸들러
 */
const CATEGORY_LABELS = {
  engineering: '공학',
  natural_science: '자연과학',
  humanities: '인문학',
  social_science: '사회과학',
  arts: '예술',
  health_sciences: '보건의료',
  sports: '스포츠',
  education: '교육',
  etc: '기타',
}

export default function LectureItem({ lecture, viewMode = 'grid', onClick }) {
  const isDone = lecture.status === 'done';
  const isDummy = lecture.is_dummy || lecture.source === 'metadata';
  const isPlayable = isDone && !isDummy;
  const metaLabel = lecture.created_at ? new Date(lecture.created_at).toLocaleDateString() : '';
  const categoryLabel = CATEGORY_LABELS[lecture.category] ?? lecture.category;
  
  return (
    <article 
      className={`lecture-card ${viewMode === 'list' ? 'lecture-card--list' : ''}`}
      onClick={() => isPlayable && onClick?.(lecture)}
      style={{ cursor: isPlayable ? 'pointer' : 'default' }}
    >
      <div className="lecture-card-thumb" style={{ background: 'var(--card)' }}>
        {lecture.thumbnail_url ? (
          <img
            className="lecture-card-thumb-image"
            src={lecture.thumbnail_url}
            alt=""
            loading="lazy"
            onError={(event) => {
              event.currentTarget.hidden = true
              event.currentTarget.nextElementSibling?.removeAttribute('hidden')
            }}
          />
        ) : null}
        <span className="lecture-card-thumb-icon" hidden={Boolean(lecture.thumbnail_url)}>
          {lecture.category === 'natural_science' ? '📐' : '🎬'}
        </span>
        {!isDummy && !isDone && (
          <span className={`lecture-card-status-badge status-${lecture.status}`}>
            {lecture.status === 'error' ? '오류' : '분석 중'}
          </span>
        )}
      </div>
      
      <div className="lecture-card-info">
        <div className="lecture-card-category">{categoryLabel}</div>
        <h3 className="lecture-card-title">{lecture.title}</h3>
        <div className="lecture-card-meta">
          {metaLabel}
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
