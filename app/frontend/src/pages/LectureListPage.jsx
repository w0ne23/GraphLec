import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { listLectures } from '../lib/api'
import LectureItem from '../components/lecture-list/LectureItem'

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
  
  // ── 2. 상세검색 패널 상태 ──
  const [showFilterPanel, setShowFilterPanel] = useState(false)
  const [pendingCategory, setPendingCategory] = useState('전체')

  // ── 3. 페이지네이션 상태 ──
  const [currentPage, setCurrentPage] = useState(1)
  const [totalPages, setTotalPages] = useState(1)
  const [totalCount, setTotalCount] = useState(0)

  // 조건이나 페이지가 바뀔 때마다 API 단일 호출
  useEffect(() => {
    setLoading(true)
    listLectures({
      page: currentPage,
      limit: ITEMS_PER_PAGE,
      category: activeCategory === '전체' ? null : activeCategory,
      search: activeSearch 
    })
      .then(res => {
        setLectures(res.items)
        setTotalPages(res.totalPages)
        setTotalCount(res.totalItems || res.items.length)
      })
      .finally(() => setLoading(false))
  }, [currentPage, activeCategory, activeSearch])

  // 검색 실행 핸들러 (엔터 키 또는 버튼 클릭)
  const handleSearchSubmit = (e) => {
    e?.preventDefault() 
    if (activeSearch === searchInput.trim()) return
    setActiveSearch(searchInput.trim())
    setCurrentPage(1) 
  }

  // 필터 적용 핸들러
  const applyFilters = () => {
    setActiveCategory(pendingCategory)
    setCurrentPage(1)
    setShowFilterPanel(false)
  }

  // 필터 초기화 핸들러
  const resetFilters = () => {
    setPendingCategory('전체')
  }

  return (
    <div className="ll-page">
      <div className="ll-content content-max">
        
        <header className="ll-header">
          <h1 className="ll-title">모든 강의 둘러보기</h1>
          
          <form className="ll-search-form" onSubmit={handleSearchSubmit}>
            <div className="ll-search-main">
              <input 
                type="text" 
                className="ll-search-input"
                placeholder="강의 제목을 검색하세요" 
                value={searchInput}
                onChange={(e) => setSearchInput(e.target.value)}
              />
              <button type="submit" className="ll-search-btn">검색</button>
            </div>
            <button 
              type="button"
              className={`ll-detail-toggle-btn ${showFilterPanel ? 'active' : ''}`}
              onClick={() => setShowFilterPanel(!showFilterPanel)}
            >
              상세검색 {showFilterPanel ? '▴' : '▾'}
            </button>
          </form>

          {/* ── 상세검색 패널 ── */}
          {showFilterPanel && (
            <div className="ll-filter-panel">
              <div className="ll-filter-section">
                <h4 className="ll-filter-label">카테고리</h4>
                {/* 전체 선택 칩 영역 */}
                <div className="ll-filter-chips">
                  <button
                    className={`ll-chip ${pendingCategory === '전체' ? 'active' : ''}`}
                    onClick={() => setPendingCategory('전체')}
                  >
                    전체
                  </button>
                </div>

                <div className="ll-chip-divider" />

                {/* 서브그룹: Engineering */}
                <div className="ll-filter-subgroup">
                  <h5 className="ll-filter-sublabel">Engineering</h5>
                  <div className="ll-filter-chips">
                    {CATEGORIES.filter(cat => cat !== '전체').map(cat => (
                      <button
                        key={cat}
                        className={`ll-chip ${pendingCategory === cat ? 'active' : ''}`}
                        onClick={() => setPendingCategory(cat)}
                      >
                        {cat}
                      </button>
                    ))}
                  </div>
                </div>
              </div>              <div className="ll-filter-actions">
                <button className="ll-filter-reset" onClick={resetFilters}>초기화</button>
                <button className="ll-filter-apply" onClick={applyFilters}>적용</button>
              </div>
            </div>
          )}
          
          <div className="ll-toolbar">
            <div className="ll-toolbar-left">
              <span className="ll-total-count">전체 <strong>{totalCount}</strong>개</span>
              {activeCategory !== '전체' && (
                <div className="ll-active-filters">
                  <span className="ll-active-chip">
                    {activeCategory}
                    <button className="ll-active-remove" onClick={() => {
                      setActiveCategory('전체')
                      setPendingCategory('전체')
                    }}>✕</button>
                  </span>
                </div>
              )}
            </div>

            <div className="ll-toolbar-right">
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
                <LectureItem 
                  key={lec.id || lec.job_id}
                  lecture={lec}
                  viewMode={viewMode}
                  onClick={() => navigate(`/lectures/${lec.id}`)}
                />
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