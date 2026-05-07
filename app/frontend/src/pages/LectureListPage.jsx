import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { listLectures } from '../lib/api'

import '../styles/lecture-list.css'

const CATEGORIES = ['전체', '컴퓨터 과학', '수학', '데이터 사이언스', '소프트웨어 공학']
const ITEMS_PER_PAGE = 12

export default function LectureListPage() {
  const navigate = useNavigate()
  
  const [lectures, setLectures] = useState([])
  const [loading, setLoading] = useState(true)
  const [viewMode, setViewMode] = useState('grid') 
  
  // ── 1. 조건 상태 (All, 카테고리, 검색) ──
  const [activeCategory, setActiveCategory] = useState('전체')
  const [searchInput, setSearchInput] = useState('') // 타이핑 중인 값
  const [activeSearch, setActiveSearch] = useState('') // 실제 API에 요청할 검색어
  
  // ── 2. 페이지네이션 상태 ──
  const [currentPage, setCurrentPage] = useState(1)
  const [totalPages, setTotalPages] = useState(1)

  // 조건이나 페이지가 바뀔 때마다 API 단일 호출
  useEffect(() => {
    setLoading(true)
    listLectures({
      page: currentPage,
      limit: ITEMS_PER_PAGE,
      category: activeCategory,
      search: activeSearch // 타이핑 중인 값이 아닌, 확정된 검색어 사용
    })
      .then(res => {
        setLectures(res.items)
        setTotalPages(res.totalPages)
      })
      .finally(() => setLoading(false))
  }, [currentPage, activeCategory, activeSearch])

  // 카테고리 변경 핸들러
  const handleCategoryChange = (cat) => {
    if (activeCategory === cat) return
    setActiveCategory(cat)
    setCurrentPage(1) // 조건 변경 시 1페이지로 리셋
  }

  // 검색 실행 핸들러 (엔터 키 또는 버튼 클릭)
  const handleSearchSubmit = (e) => {
    e?.preventDefault() // form 제출 새로고침 방지
    if (activeSearch === searchInput.trim()) return
    setActiveSearch(searchInput.trim())
    setCurrentPage(1) // 조건 변경 시 1페이지로 리셋
  }

  return (
    <div className="ll-page">
      <div className="ll-content content-max">
        
        <header className="ll-header">
          <h1 className="ll-title">모든 강의 둘러보기</h1>
          
          {/* ── 검색바 추가 ── */}
          <form className="ll-search-form" onSubmit={handleSearchSubmit}>
            <input 
              type="text" 
              className="ll-search-input"
              placeholder="강의 제목을 검색하세요" 
              value={searchInput}
              onChange={(e) => setSearchInput(e.target.value)}
            />
            <button type="submit" className="ll-search-btn">검색</button>
          </form>
          
          <div className="ll-toolbar">
            <div className="ll-filter-bar">
              {CATEGORIES.map(cat => (
                <button
                  key={cat}
                  className={`ll-filter-btn ${activeCategory === cat ? 'll-filter-btn--active' : ''}`}
                  onClick={() => handleCategoryChange(cat)}
                >
                  {cat}
                </button>
              ))}
            </div>

            <div className="ll-view-toggles">
              <button 
                className={`ll-icon-btn ${viewMode === 'grid' ? 'll-icon-btn--active' : ''}`}
                onClick={() => setViewMode('grid')}
                title="그리드 뷰"
              >
                ⊞
              </button>
              <button 
                className={`ll-icon-btn ${viewMode === 'list' ? 'll-icon-btn--active' : ''}`}
                onClick={() => setViewMode('list')}
                title="리스트 뷰"
              >
                ☰
              </button>
            </div>
          </div>
        </header>

        {loading ? (
          <div className="ll-loading">강의 목록을 불러오는 중...</div>
        ) : lectures.length === 0 ? (
          <div className="ll-empty">
            조건에 맞는 강의가 없습니다. 다른 검색어나 카테고리를 선택해 보세요.
          </div>
        ) : (
          <>
            <div className={viewMode === 'grid' ? 'll-grid' : 'll-list'}>
              {lectures.map(lec => (
                <article 
                  key={lec.id || lec.job_id} 
                  className={`lecture-card ${viewMode === 'list' ? 'lecture-card--list' : ''}`}
                  onClick={() => lec.status === 'done' && navigate(`/lectures/${lec.id}`)}
                >
                  <div className="lecture-card-thumb" style={{ background: 'var(--card)' }}>
                    <span className="lecture-card-thumb-icon">
                      {lec.category === '수학' ? '📐' : '🎬'}
                    </span>
                    {lec.status !== 'done' && (
                      <span className={`lecture-card-status-badge status-${lec.status}`}>
                        {lec.status === 'error' ? '오류' : '분석 중'}
                      </span>
                    )}
                  </div>
                  <div className="lecture-card-info">
                    <div className="lecture-card-category">{lec.category}</div>
                    <h3 className="lecture-card-title">{lec.title}</h3>
                    <div className="lecture-card-meta">
                      {new Date(lec.created_at).toLocaleDateString()}
                    </div>
                    <div className="lecture-card-tags">
                      {lec.tags?.slice(0, 3).map(tag => (
                        <span key={tag} className="lecture-card-tag">{tag}</span>
                      ))}
                    </div>
                  </div>
                </article>
              ))}
            </div>

            {/* ── 페이지네이션 UI ── */}
            {totalPages > 0 && (
              <div className="ll-pagination">
                <button
                  className="ll-page-btn"
                  disabled={currentPage === 1}
                  onClick={() => setCurrentPage(p => Math.max(1, p - 1))}
                >
                  &lt;
                </button>
                
                {Array.from({ length: totalPages }, (_, i) => i + 1).map(pageNum => (
                  <button
                    key={pageNum}
                    className={`ll-page-btn ${currentPage === pageNum ? 'll-page-btn--active' : ''}`}
                    onClick={() => setCurrentPage(pageNum)}
                  >
                    {pageNum}
                  </button>
                ))}

                <button
                  className="ll-page-btn"
                  disabled={currentPage === totalPages}
                  onClick={() => setCurrentPage(p => Math.min(totalPages, p + 1))}
                >
                  &gt;
                </button>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  )
}