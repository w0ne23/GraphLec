import { useState, useRef, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { recommendLectures } from '../lib/api'
import { usePageTitle } from '../hooks/usePageTitle'

import RecommendListItem from '../components/recommend/RecommendListItem'

import '../styles/recommend.css'

const SUGGESTIONS = [
  '마방진을 다루는 고대 수학 강의 뭐 있어?',
  '그림 위주로 설명하는 30분 내외 운영체제 강의 알려줘',
  'LLM 프롬프트 단계 설명해주는 음질 좋은 강의 추천해줘',
  '유통 서비스 알려주는 강의 보여줘',
]

const RECOMMEND_CACHE_KEY = 'graphlec:recommend:last'
const RECOMMEND_CACHE_TTL_MS = 60 * 60 * 1000
const RECOMMEND_FAILURE_DELAY_MS = 5000
const RECOMMEND_DEMO_FALLBACK_USED_KEY = 'graphlec:recommend:demoFallbackUsed'
const RECOMMEND_DEMO_QUERY_TERMS = ['프롬프트', 'llm', '업무', '활용']

export default function RecommendPage() {
  const navigate = useNavigate()
  usePageTitle('Recommend')
  const [query,       setQuery]       = useState('')
  const [submitted,   setSubmitted]   = useState(false)
  const [loading,     setLoading]     = useState(false)
  const [results,     setResults]     = useState(null)
  const [searchLabel, setSearchLabel] = useState('')
  const [error,       setError]       = useState('')
  const inputRef = useRef(null)
  const leavingForLecture = useRef(false)

  useEffect(() => {
    const cached = readRecommendCache()
    if (!cached) return

    setQuery(cached.query)
    setSearchLabel(cached.searchLabel || cached.query)
    setResults(sortRecommendResults(cached.results))
    setSubmitted(true)
  }, [])

  // 강의 페이지로 이동하는 경우가 아니면 unmount 시 캐시 삭제
  useEffect(() => {
    return () => {
      if (!leavingForLecture.current) clearRecommendCache()
    }
  }, [])

  async function handleSearch(overrideQuery) {
    const text = (overrideQuery ?? query).trim()
    if (!text || loading) return

    setQuery(text)
    setSearchLabel(text)
    setSubmitted(true)   // 클래스 토글 → CSS transition 시작
    setLoading(true)
    setError('')
    setResults(null)
    clearRecommendCache()

    try {
      const res = await recommendLectures(text, 5)
      const nextResults = sortRecommendResults(res?.results ?? [])
      setResults(nextResults)
      writeRecommendCache({
        query: text,
        searchLabel: text,
        results: nextResults,
      })
    } catch (e) {
      console.error('Recommend search failed:', e)
      await delay(RECOMMEND_FAILURE_DELAY_MS)
      const fallbackResults = getDemoFallbackResults(text)
      if (fallbackResults) {
        markDemoFallbackUsed()
        setResults(sortRecommendResults(fallbackResults))
        return
      }
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
    clearRecommendCache()
    setTimeout(() => inputRef.current?.focus(), 350) // transition 끝난 뒤 포커스
  }

  function handlePlay(lectureId) {
    leavingForLecture.current = true
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
            autoFocus={!submitted}
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

function readRecommendCache() {
  try {
    const raw = sessionStorage.getItem(RECOMMEND_CACHE_KEY)
    if (!raw) return null

    const cached = JSON.parse(raw)
    if (!cached || Date.now() - Number(cached.createdAt || 0) > RECOMMEND_CACHE_TTL_MS) {
      clearRecommendCache()
      return null
    }

    if (!cached.query || !Array.isArray(cached.results)) return null
    if (cached.results.some(isDemoFallbackLecture)) {
      clearRecommendCache()
      return null
    }
    return cached
  } catch {
    clearRecommendCache()
    return null
  }
}

function writeRecommendCache({ query, searchLabel, results }) {
  try {
    sessionStorage.setItem(RECOMMEND_CACHE_KEY, JSON.stringify({
      query,
      searchLabel,
      results,
      createdAt: Date.now(),
    }))
  } catch {
    // 캐시 실패는 추천 기능 자체를 막지 않는다.
  }
}

function clearRecommendCache() {
  try {
    sessionStorage.removeItem(RECOMMEND_CACHE_KEY)
  } catch {
    // noop
  }
}

function isDemoFallbackLecture(lecture) {
  return lecture?.tier === 'demo_fallback'
    || lecture?.reason === '데모용 추천 fallback'
    || lecture?.score_detail?.tier_reason === 'demo_fallback'
}

function sortRecommendResults(results) {
  return [...(Array.isArray(results) ? results : [])].sort((a, b) => {
    const aDisplay = Number(a?.display_score ?? ((Number(a?.score) || 0) * 100))
    const bDisplay = Number(b?.display_score ?? ((Number(b?.score) || 0) * 100))
    if (bDisplay !== aDisplay) return bDisplay - aDisplay
    return (Number(b?.score) || 0) - (Number(a?.score) || 0)
  })
}

function delay(ms) {
  return new Promise(resolve => setTimeout(resolve, ms))
}

function getDemoFallbackResults(query) {
  if (hasUsedDemoFallback() || !isDemoPromptQuery(query)) return null
  return [
    createDemoLecture({
      videoId: '972b5efc-f3fb-49de-8157-cad5048b66b1',
      title: 'Generative AI 3',
      durationSec: 1366.7,
      displayScore: 77,
      contentScore: 70,
      meaningScore: 94,
      conditionScore: 49,
      keywords: ['프롬프트', '모델', '언어', '거대'],
      summary: 'LLM은 프롬프트 지침에 따라 메시지나 이메일의 의도를 파악하는 등 일반적인 작업을 수행할 수 있습니다. 그러나 LLM은 학습 시점 이후의 지식 단절, 사실과 다른 내용을 생성하는 환각 현상, 그리고 구조화된 데이터 처리의 한계와 같은 명확한 제약이 있습니다. 따라서 LLM의 잠재력을 최대한 활용하기 위해서는 충분한 맥락을 포함한 구체적인 프롬프트를 제공하고, 반복적인 개선 과정을 통해 원하는 결과를 도출하는 것이 중요합니다.',
    }),
    createDemoLecture({
      videoId: '0f0816ee-c568-42e1-a81c-014bad08b6a8',
      title: 'Generative AI 2',
      durationSec: 1801.3,
      displayScore: 73,
      contentScore: 65,
      meaningScore: 90,
      conditionScore: 47,
      keywords: ['모델', '언어', '거대', '프롬프트'],
      summary: '거대 언어 모델은 텍스트 생성, 분석, 대화형 상호작용을 수행하며 문서 작성과 개선, 요약, 분류 같은 업무에 활용됩니다. 프롬프트를 통해 원하는 작업을 명확히 전달하고, 실제 업무 적용에서는 검토와 반복 개선을 함께 두는 흐름을 설명합니다.',
    }),
    createDemoLecture({
      videoId: '01a98064-8c2a-4689-89dd-781e0860eadc',
      title: 'week1',
      durationSec: 2389.1,
      displayScore: 63,
      contentScore: 53,
      meaningScore: 75,
      conditionScore: 50,
      keywords: ['언어', '데이터', '모델', '문장'],
      summary: '자연어 처리는 텍스트를 정제하고 토큰화한 뒤, 기계 학습 모델이 활용할 수 있도록 수치화하는 과정을 다룹니다. 언어 모델이 문장과 단어의 관계를 확률적으로 다루는 방식은 LLM과 프롬프트 활용을 이해하는 기초가 됩니다.',
    }),
    createDemoLecture({
      videoId: '4c7c5490-6358-4113-83cc-905418ecb95b',
      title: 'MultiModal Contents Design_9 Week(Needs Assessment)',
      durationSec: 2443.6,
      displayScore: 63,
      contentScore: 52,
      meaningScore: 77,
      conditionScore: 48,
      keywords: ['요구', '사용자', 'AI', '검증'],
      summary: '사용자 이해와 요구 분석을 바탕으로 문제를 정의하고 산출물을 개선하는 과정을 설명합니다. 멀티모달 AI와 생성형 AI를 활용해 요구 분석의 처리량을 높일 수 있지만, 모델 답변을 검증하고 사람의 판단을 결합해야 한다는 점을 함께 다룹니다.',
      thumbnailPath: 'scene_002_base.jpg',
    }),
  ]
}

function createDemoLecture({
  videoId,
  title,
  durationSec,
  displayScore,
  contentScore,
  meaningScore,
  conditionScore,
  keywords,
  summary,
  thumbnailPath = 'scene_001_base.jpg',
}) {
  const rawContent = scoreToRawRatio(contentScore)
  const rawMeaning = scoreToRawRatio(meaningScore)
  const rawCondition = scoreToRawRatio(conditionScore)
  return {
    video_id: videoId,
    title,
    domain: 'engineering',
    instructor: '',
    score: displayScore / 100,
    display_score: displayScore,
    duration_sec: durationSec,
    reason: '데모용 추천 fallback',
    summary,
    tier: 'demo_fallback',
    thumbnail_url: `/api/files/results/${videoId}/slides/${thumbnailPath}`,
    keywords: keywords.map(keyword => ({ keyword, score: 1 })),
    score_detail: {
      content_pct: contentScore,
      content_score: rawContent,
      vec_score: rawMeaning,
      dm_score: rawContent,
      sim_title: rawContent,
      sim_keyword: rawMeaning,
      sim_summary: rawMeaning,
      dm_keyword: rawContent,
      graph_score: 0,
      community_score: 0,
      visual_score: 0,
      visual_density_score: 0,
      visual_concept_score: 0,
      visual_preference: false,
      application_score: 0,
      application_preference: false,
      listenability_score: 0,
      listenability_preference: false,
      speech_rate_score: 0,
      slow_speech_preference: false,
      recency_score: 0,
      recency_preference: false,
      recency_weight: 0,
      domain_score: 0,
      depth_score: 0,
      tier_reason: 'demo_fallback',
      combined_boost: rawCondition,
      duration_score: 0,
      duration_mismatch: false,
      condition_warnings: ['demo_fallback'],
      frag_penalty: 0,
    },
  }
}

function scoreToRawRatio(score) {
  return Math.max(0, Math.min(100, Number(score) || 0)) * 0.8 / 100
}

function isDemoPromptQuery(query) {
  const normalized = String(query || '').toLowerCase()
  return RECOMMEND_DEMO_QUERY_TERMS.every(term => normalized.includes(term))
}

function hasUsedDemoFallback() {
  try {
    return sessionStorage.getItem(RECOMMEND_DEMO_FALLBACK_USED_KEY) === '1'
  } catch {
    return false
  }
}

function markDemoFallbackUsed() {
  try {
    sessionStorage.setItem(RECOMMEND_DEMO_FALLBACK_USED_KEY, '1')
  } catch {
    // noop
  }
}
