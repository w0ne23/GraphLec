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

  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: 'smooth' }) }, [messages, loading])

  function formatTime(seconds) {
    if (seconds == null || Number.isNaN(Number(seconds))) return null
    const total = Math.max(0, Math.floor(Number(seconds)))
    const m = Math.floor(total / 60).toString().padStart(2, '0')
    const s = Math.floor(total % 60).toString().padStart(2, '0')
    return `${m}:${s}`
  }

  function parseTimestamp(value) {
    if (value == null) return null
    const parts = String(value).split(':').map(v => Number(v))
    if (parts.some(Number.isNaN)) return null
    if (parts.length === 1) return parts[0]
    if (parts.length === 2) return parts[0] * 60 + parts[1]
    return parts[0] * 3600 + parts[1] * 60 + parts[2]
  }

  function sourceLabel(text) {
    const compact = String(text || '').replace(/\s+/g, ' ').trim()
    return compact.length > 42 ? `${compact.slice(0, 42)}...` : compact
  }

  function refDedupeKey(ref) {
    if (Number.isFinite(Number(ref.startSec))) return `t:${Math.round(Number(ref.startSec))}`
    if (ref.slideNumber != null) return `s:${ref.slideNumber}`
    return `l:${ref.label || ''}`
  }

  function refsFromResponse(res) {
    const chunks = Array.isArray(res.retrieved_chunks) ? res.retrieved_chunks : []
    const rawRefs = chunks.length > 0
      ? chunks.map(chunk => ({
          timestamp: formatTime(chunk.start_sec),
          startSec: chunk.start_sec,
          slideNumber: chunk.slide_number,
          score: Number.isFinite(Number(chunk.score)) ? Number(chunk.score) : null,
          label: sourceLabel(chunk.text || chunk.chunk_type),
          text: chunk.text || '',
        }))
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

  function findRefSceneIndex(ref) {
    const scenes = lecture?.scenes || []
    if (!scenes.length) return -1

    if (ref.slideNumber != null) {
      const bySlide = scenes.findIndex(s => Number(s.slide_number) === Number(ref.slideNumber))
      if (bySlide >= 0) return bySlide
    }

    if (ref.startSec != null) {
      const targetSec = Number(ref.startSec)
      let bestIdx = -1
      let bestSec = -1
      scenes.forEach((scene, idx) => {
        const sec = parseTimestamp(scene.timestamp)
        if (sec == null) return
        if (sec <= targetSec && sec > bestSec) {
          bestSec = sec
          bestIdx = idx
        }
      })
      if (bestIdx >= 0) return bestIdx
    }

    return scenes.findIndex(s => s.timestamp === ref.timestamp)
  }

  function canOpenRef(ref, idx) {
    return idx >= 0 || Number.isFinite(Number(ref.startSec))
  }

  async function send() {
    const question = input.trim()
    if (!question || loading || !lecture?.id) return
    setInput('')
    setMessages(prev => [...prev, { id: Date.now(), role: 'user', content: question, refs: [] }])
    setLoading(true)
    
    try {
      const res = await askQa(lecture.id, question)

      setMessages(prev => [...prev, {
        id: Date.now() + 1, role: 'assistant',
        content: res.answer || '답변을 생성하지 못했습니다.',
        refs: refsFromResponse(res),
      }])
    } catch (e) {
      setMessages(prev => [...prev, {
        id: Date.now() + 1, role: 'assistant',
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
        {messages.map(msg =>
          msg.role === 'user' ? (
            <div key={msg.id} className="chat-msg-user">
              <div className="chat-bubble-user">{msg.content}</div>
            </div>
          ) : (
            <div key={msg.id} className="chat-msg-ai">
              <div className="chat-bubble-ai">
                <div className="chat-answer-text">{msg.content}</div>
                {msg.refs?.length > 0 && (
                  <div className="chat-refs">
                    {msg.refs.map((ref, i) => {
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
