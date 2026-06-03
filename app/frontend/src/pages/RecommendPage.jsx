import { useState, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { recommendLectures } from '../lib/api'

import RecommendListItem from '../components/recommend/RecommendListItem'

import '../styles/recommend.css'

const SUGGESTIONS = [
  '선형대수 행렬 강의 뭐 있어?',
  '그림 위주로 설명는 30분 내외 파이썬 강의 알려줘',
  'DNA 복제 단계 설명해주는 음질 좋은 강의 추천해줘',
  '딥러닝 CNN 실습 위주 강의 보여줘',
]

export default function RecommendPage() {
  const navigate = useNavigate()
  const [query,       setQuery]       = useState('')
  const [submitted,   setSubmitted]   = useState(false)
  const [loading,     setLoading]     = useState(false)
  const [results,     setResults]     = useState(null)
  const [searchLabel, setSearchLabel] = useState('')
  const [error,       setError]       = useState('')
  const inputRef = useRef(null)

  async function handleSearch(overrideQuery) {
    const text = (overrideQuery ?? query).trim()
    if (!text || loading) return

    setQuery(text)
    setSearchLabel(text)
    setSubmitted(true)   // 클래스 토글 → CSS transition 시작
    setLoading(true)
    setError('')
    setResults(null)

    try {
      const res = await recommendLectures(text, 5)
      setResults(res?.results ?? [])
    } catch (e) {
      console.error('Recommend search failed:', e)
      setError('추천 서버가 잠시 불안정합니다. 잠시 후 다시 검색해 주세요.')
      setResults(null)
    } finally {
      setLoading(false)
    }
  }

  // 로고 클릭 → 초기화면으로 복귀
  function handleReset() {
    setSubmitted(false)
    setQuery('')
    setResults(null)
    setError('')
    setSearchLabel('')
    setTimeout(() => inputRef.current?.focus(), 350) // transition 끝난 뒤 포커스
  }

  function handlePlay(lectureId) {
    navigate(`/lectures/${lectureId}`)
  }

  return (
    <div className={`rec-page${submitted ? ' rec-page--searched' : ''}`}>

      {/* 히어로: 로고(클릭 시 초기화) + 문구 */}
      <div className="rec-hero">
        <div className="rec-logo" onClick={handleReset}>
          Graph<span>Lec</span>
        </div>
        <p className="rec-headline">어떤 강의 영상을 보고 싶은지 설명해 주세요!</p>
      </div>

      {/* 검색바 — 항상 동일한 디자인 */}
      <div className="rec-bar-wrap content-max">
        <div className="rec-searchbar">
          <span className="rec-searchbar-icon">🔍</span>
          <input
            ref={inputRef}
            className="rec-searchbar-input"
            placeholder="강의 내용, 주제, 조건 등을 자유롭게 입력하세요"
            value={query}
            onChange={e => setQuery(e.target.value)}
            onKeyDown={e => {
              if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSearch() }
            }}
            disabled={loading}
            autoFocus
          />
          {query && (
            <button className="rec-searchbar-clear" onClick={() => setQuery('')} tabIndex={-1}>
              ✕
            </button>
          )}
          <button
            className="rec-searchbar-btn"
            onClick={() => handleSearch()}
            disabled={loading || !query.trim()}
          >
            {loading ? '…' : '검색'}
          </button>
        </div>
      </div>

      {/* 예시 칩 */}
      <div className="rec-chips content-max">
        {SUGGESTIONS.map((s, i) => (
          <button
            key={i}
            className="rec-chip"
            onClick={() => { setQuery(s); handleSearch(s) }}
          >
            {s}
          </button>
        ))}
      </div>

      {/* 결과 목록 */}
      <div className="rec-results">
        <div className="content-max" style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
          {loading ? (
            <div className="rec-loading">
              <span className="rec-loading-dot" />
              <span className="rec-loading-dot" />
              <span className="rec-loading-dot" />
            </div>
          ) : error ? (
            <div className="rec-error-state">
              <p className="rec-empty-title">추천을 불러오지 못했습니다</p>
              <p className="rec-empty-sub">{error}</p>
              <button
                className="rec-retry-btn"
                onClick={() => handleSearch(searchLabel || query)}
                disabled={loading}
              >
                다시 검색
              </button>
            </div>
          ) : results === null ? null : results.length === 0 ? (
            <div className="rec-empty">
              <p className="rec-empty-title">검색 결과가 없습니다</p>
              <p className="rec-empty-sub">다른 키워드로 다시 시도해 보세요</p>
            </div>
          ) : (
            <>
              <div className="rec-result-label">
                <strong>"{searchLabel}"</strong>에 대한 추천 강의 {results.length}개
              </div>
              <div className="rec-list">
                {results.map(lec => (
                  <RecommendListItem
                    key={lec.video_id}
                    lecture={lec}
                    onPlay={handlePlay}
                    queryText={searchLabel}
                  />
                ))}
              </div>
            </>
          )}
        </div>
      </div>

    </div>
  )
}
