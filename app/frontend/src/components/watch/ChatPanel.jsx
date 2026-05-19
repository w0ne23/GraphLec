import { useState, useRef, useEffect } from 'react'
import { askQa } from '../../lib/api'

/**
 * src/components/chat/ChatPanel.jsx
 * LecturePage 전용 — 영상 질의 사이드패널
 *
 * lecture        — 현재 강의 객체
 * onJumpToScene  — 타임스탬프 클릭 시 VideoPlayer + SceneList 동기화
 */

export default function ChatPanel({ 
  lecture, 
  currentSceneIndex = 0,
  onJumpToScene, 
  onClose,
  messages,
  setMessages,
  input,
  setInput,
  loading,
  setLoading
}) {
  const bottomRef = useRef(null)
  // 인덱스별 펼침 상태를 관리하는 배열
  const [expandedIndices, setExpandedIndices] = useState([])

  const toggleExpand = (idx) => {
    setExpandedIndices(prev => {
      const next = [...prev]
      next[idx] = !next[idx]
      return next
    })
  }

  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: 'smooth' }) }, [messages, loading])

  function formatTime(seconds) {
    if (seconds == null || Number.isNaN(Number(seconds))) return null
    const total = Math.max(0, Math.floor(Number(seconds)))
    const m = Math.floor(total / 60).toString().padStart(2, '0')
    const s = Math.floor(total % 60).toString().padStart(2, '0')
    return `${m}:${s}`
  }

  function parseTimestamp(value) {
    if (Number.isFinite(Number(value))) return Number(value)
    if (value == null) return null
    const parts = String(value).split(':').map(v => Number(v))
    if (parts.some(Number.isNaN)) return null
    if (parts.length === 1) return parts[0]
    if (parts.length === 2) return parts[0] * 60 + parts[1]
    return parts[0] * 3600 + parts[1] * 60 + parts[2]
  }

  function sourceLabel(text) {
    const compact = String(text || '').replace(/\s+/g, ' ').trim()
    if (!compact || compact.startsWith('{') || compact.startsWith('[')) return '구간 근거'
    return compact.length > 42 ? `${compact.slice(0, 42)}...` : compact
  }

  function refDedupeKey(ref) {
    if (Number.isFinite(Number(ref.startSec))) return `t:${Math.round(Number(ref.startSec))}`
    if (ref.slideNumber != null) return `s:${ref.slideNumber}`
    return `l:${ref.label || ''}`
  }

  function refsFromResponse(res) {
    if (['visual_location', 'scene_location', 'overview'].includes(res.source_mode)) return []
    const chunks = Array.isArray(res.retrieved_chunks) ? res.retrieved_chunks : []
    const chunkRefs = chunks
        .filter(chunk => chunk.start_sec != null && chunk.chunk_type !== 'slide')
        .map(chunk => ({
          timestamp: formatTime(chunk.start_sec),
          startSec: chunk.start_sec,
          score: Number.isFinite(Number(chunk.score)) ? Number(chunk.score) : null,
          label: sourceLabel(chunk.text || chunk.chunk_type),
          text: chunk.text || '',
        }))
    const rawRefs = chunkRefs.length > 0
      ? chunkRefs
      : (res.timestamps || []).map(t => ({
          timestamp: formatTime(t.start),
          startSec: t.start,
          slideNumber: t.slide_number,
          score: null,
          label: sourceLabel(t.label),
          text: t.label || '',
        }))

    const bestByKey = new Map()
    rawRefs
      .filter(ref => ref.timestamp || ref.slideNumber != null || ref.label)
      .forEach(ref => {
        const key = refDedupeKey(ref)
        const prev = bestByKey.get(key)
        const prevScore = prev?.score ?? -Infinity
        const score = ref.score ?? -Infinity
        if (!prev || score > prevScore || String(ref.label).length > String(prev.label || '').length) {
          bestByKey.set(key, ref)
        }
      })

    return Array.from(bestByKey.values())
      .sort((a, b) => {
        const scoreGap = (b.score ?? -Infinity) - (a.score ?? -Infinity)
        if (Number.isFinite(scoreGap) && Math.abs(scoreGap) > 1e-9) return scoreGap
        return (Number(a.startSec ?? Infinity) - Number(b.startSec ?? Infinity))
      })
      .slice(0, 8)
  }

  function scenesFromSlideResponse(res) {
    const slides = Array.isArray(res.related_slides) ? res.related_slides : []
    const fallbackChunks = Array.isArray(res.retrieved_chunks) ? res.retrieved_chunks : []
    const rawSlides = slides.length > 0
      ? slides
      : fallbackChunks
        .filter(chunk => chunk.chunk_type === 'slide' && chunk.slide_number != null)
        .map(chunk => ({
          slide_number: chunk.slide_number,
          start_sec: chunk.start_sec,
          score: Number.isFinite(Number(chunk.score)) ? Number(chunk.score) : null,
        }))

    const bestBySlide = new Map()
    rawSlides.forEach(item => {
      const slideNumber = item.slide_number ?? item.slideNumber
      if (slideNumber == null) return
      const key = Number(slideNumber)
      const prev = bestBySlide.get(key)
      const score = Number.isFinite(Number(item.score)) ? Number(item.score) : -Infinity
      const prevScore = prev?.score ?? -Infinity
      if (!prev || score > prevScore) {
        bestBySlide.set(key, {
          slideNumber: key,
          startSec: item.start_sec ?? item.startSec ?? null,
          score,
          label: item.label || `슬라이드 ${key}`,
        })
      }
    })

    return Array.from(bestBySlide.values())
      .sort((a, b) => {
        const scoreGap = (b.score ?? -Infinity) - (a.score ?? -Infinity)
        if (Number.isFinite(scoreGap) && Math.abs(scoreGap) > 1e-9) return scoreGap
        return a.slideNumber - b.slideNumber
      })
      .slice(0, 4)
  }

  function findRefSceneIndex(ref) {
    const scenes = lecture?.scenes || []
    if (!scenes.length) return -1

    if (ref.startSec != null) {
      const targetSec = Number(ref.startSec)
      let bestIdx = -1
      let bestSec = -1
      scenes.forEach((scene, idx) => {
        if (ref.slideNumber != null && Number(scene.slide_number) !== Number(ref.slideNumber)) return
        const sec = parseTimestamp(scene.timestamp_sec ?? scene.timestamp)
        if (sec == null) return
        if (sec <= targetSec && sec > bestSec) {
          bestSec = sec
          bestIdx = idx
        }
      })
      if (bestIdx >= 0) return bestIdx
    }

    if (ref.slideNumber != null) {
      const bySlide = scenes.findIndex(s => Number(s.slide_number) === Number(ref.slideNumber))
      if (bySlide >= 0) return bySlide
    }

    return scenes.findIndex(s => s.timestamp === ref.timestamp)
  }

  function canOpenRef(ref, idx) {
    return idx >= 0 || Number.isFinite(Number(ref.startSec))
  }

  function sceneLabel(ref) {
    const idx = findRefSceneIndex(ref)
    if (idx < 0) return null
    const scene = idx >= 0 ? lecture?.scenes?.[idx] : null
    const n = scene?.scene_number ?? idx + 1
    if (ref.slideNumber != null) return `슬라이드 ${ref.slideNumber} · Scene${n}`
    return `Scene${n}`
  }

  function renderAnswerContent(content) {
    const lines = String(content || '').split('\n')
    const nodes = []
    let i = 0

    const isTableLine = line => /^\s*\|.*\|\s*$/.test(line)
    const isSeparatorLine = line => /^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$/.test(line)

    while (i < lines.length) {
      if (isTableLine(lines[i]) && i + 1 < lines.length && isSeparatorLine(lines[i + 1])) {
        const header = lines[i].trim().slice(1, -1).split('|').map(cell => cell.trim())
        i += 2
        const rows = []
        while (i < lines.length && isTableLine(lines[i])) {
          rows.push(lines[i].trim().slice(1, -1).split('|').map(cell => cell.trim()))
          i += 1
        }
        nodes.push(
          <div key={`tbl-${nodes.length}`} className="chat-table-wrap">
            <table className="chat-answer-table">
              <thead>
                <tr>{header.map((cell, idx) => <th key={idx}>{cell}</th>)}</tr>
              </thead>
              <tbody>
                {rows.map((row, rIdx) => (
                  <tr key={rIdx}>{header.map((_, cIdx) => <td key={cIdx}>{row[cIdx] || ''}</td>)}</tr>
                ))}
              </tbody>
            </table>
          </div>
        )
        continue
      }

      if (!lines[i].trim()) {
        nodes.push(<br key={`br-${nodes.length}`} />)
        i += 1
        continue
      }

      nodes.push(<div key={`ln-${nodes.length}`}>{lines[i]}</div>)
      i += 1
    }

    return nodes
  }

  async function send() {
    const question = input.trim()
    if (!question || loading || !lecture?.id) return
    setInput('')
    // id를 제거하고 간결하게 변경
    setMessages(prev => [...prev, { role: 'user', content: question, refs: [] }])
    setLoading(true)
    
    try {
      const currentScene = lecture?.scenes?.[currentSceneIndex] || null
      const res = await askQa(lecture.id, question, {
        current_scene_number: currentScene?.scene_number ?? null,
        current_slide_number: currentScene?.slide_number ?? null,
      })

      setMessages(prev => [...prev, {
        role: 'assistant',
        content: res.answer || '답변을 생성하지 못했습니다.',
        refs: refsFromResponse(res),
        scenes: scenesFromSlideResponse(res),
        sourceMode: res.source_mode || 'default',
      }])
    } catch (e) {
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: `오류: ${e.message}`, refs: [],
      }])
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="chat-panel">
      <div className="chat-header">
        <span className="chat-header-title">Chat</span>
        <button className="chat-close-btn" onClick={onClose} title="채팅창 닫기">
          ✕
        </button>
      </div>

      <div className="chat-messages">
        {messages.map((msg, msgIdx) =>
          msg.role === 'user' ? (
            <div key={msgIdx} className="chat-msg-user">
              <div className="chat-bubble-user">{msg.content}</div>
            </div>
          ) : (
            <div key={msgIdx} className="chat-msg-ai">
              <div className="chat-bubble-ai">
                <div className="chat-answer-text">{renderAnswerContent(msg.content)}</div>
                {!['visual_location', 'scene_location'].includes(msg.sourceMode) && msg.refs?.length > 0 && (
                  <div className="chat-refs-container">
                    <div className="chat-refs-header">영상 구간</div>
                    
                    {/* 첫 번째 출처 (단독 행) */}
                    <div className="chat-refs-first-row">
                      {(() => {
                        const ref = msg.refs[0]
                        const idx = findRefSceneIndex(ref)
                        const canOpen = canOpenRef(ref, idx)
                        return (
                          <button className="chat-ref-btn"
                            title={ref.text || ref.label || '출처'}
                            disabled={!canOpen}
                            onClick={() => canOpen && onJumpToScene?.(idx, ref.startSec)}
                          >
                            ▶ {ref.timestamp || (ref.slideNumber != null ? `S${ref.slideNumber}` : '출처')}{ref.label ? ` ${ref.label}` : ''}
                          </button>
                        )
                      })()}
                    </div>

                    {/* 추가 출처 목록 (펼쳐졌을 때만 노출) */}
                    {expandedIndices[msgIdx] && msg.refs.length > 1 && (
                      <div className="chat-refs-list">
                        {msg.refs.slice(1).map((ref, i) => {
                          const idx = findRefSceneIndex(ref)
                          const canOpen = canOpenRef(ref, idx)
                          return (
                            <button key={i} className="chat-ref-btn"
                              title={ref.text || ref.label || '출처'}
                              disabled={!canOpen}
                              onClick={() => canOpen && onJumpToScene?.(idx, ref.startSec)}
                            >
                              ▶ {ref.timestamp || (ref.slideNumber != null ? `S${ref.slideNumber}` : '출처')}{ref.label ? ` ${ref.label}` : ''}
                            </button>
                          )
                        })}
                      </div>
                    )}

                    {/* 더보기 / 접기 컨트롤 (우측 정렬) */}
                    {msg.refs.length > 1 && (
                      <div className="chat-refs-control">
                        <button 
                          className="chat-refs-toggle-btn"
                          onClick={() => toggleExpand(msgIdx)}
                        >
                          {expandedIndices[msgIdx] ? '접기' : '더보기'}
                        </button>
                      </div>
                    )}
                  </div>
                )}
                {msg.scenes?.length > 0 && (
                  <div className="chat-refs-container">
                    <div className="chat-refs-header">
                      {msg.sourceMode === 'visual_location'
                        ? '확인 위치'
                        : msg.sourceMode === 'overview'
                          ? '관련 슬라이드'
                          : '관련 장면'}
                    </div>
                    <div className="chat-refs-list">
                      {msg.scenes.map((scene, i) => {
                        const idx = findRefSceneIndex(scene)
                        const label = sceneLabel(scene)
                        if (!label) return null
                        return (
                          <button key={i} className="chat-scene-btn"
                            title={`슬라이드 ${scene.slideNumber} 관련 장면`}
                            onClick={() => onJumpToScene?.(idx, null, { autoPlay: false, offsetSec: 0.7 })}
                          >
                            {label}
                          </button>
                        )
                      })}
                    </div>
                  </div>
                )}
              </div>
            </div>
          )
        )}
        {loading && <div className="chat-loading">답변 생성 중...</div>}
        <div ref={bottomRef} />
      </div>

      <div className="chat-input-row">
        <input
          className="chat-input"
          placeholder="강의 내용에 대해 질문해보세요..."
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send() } }}
          disabled={loading}
        />
        <button className="chat-send-btn" onClick={send} disabled={loading || !input.trim()}>↑</button>
      </div>
    </div>
  )
}
