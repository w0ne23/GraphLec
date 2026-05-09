import { useEffect, useState, useRef, useCallback } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import {
  getLectureDetail,
  getLectureTimeline,
  enterLectureGraphSession,
  heartbeatLectureGraphSession,
  leaveLectureGraphSession,
} from '../lib/api'

import VideoPlayer  from '../components/watch/VideoPlayer'
import LecturePanel from '../components/watch/LecturePanel'
import ChatPanel    from '../components/chat/ChatPanel'

import '../styles/lecture.css'

const INIT_MSG = { id: 0, role: 'assistant', content: '강의에 대해 질문해보세요.', refs: [] }

/**
 * LecturePage — /lectures/:id
 * 좌: VideoPlayer + LecturePanel (가변 너비)
 * 우: ChatPanel 사이드패널 (리사이저로 너비 조절)
 *
 * @param {function} onNavigate - 페이지 전환 콜백
 */
export default function LecturePage({ onNavigate }) {
  const { id } = useParams()
  const navigate = useNavigate()
  const graphSessionIdRef = useRef(
    (window.crypto && window.crypto.randomUUID)
      ? window.crypto.randomUUID()
      : (`sess-${Date.now()}-${Math.random().toString(36).slice(2)}`)
  )

  const [lecture,      setLecture]      = useState(null)
  const [loading,      setLoading]      = useState(false)
  const [currentScene, setCurrentScene] = useState(0)
  const [seekTo,       setSeekTo]       = useState(null)
  const [seekToSeconds, setSeekToSeconds] = useState(null)
  const [chatWidth,    setChatWidth]    = useState(300)
  const [isChatOpen,   setIsChatOpen]   = useState(true)
  const [isFocusMode,  setIsFocusMode]  = useState(false) // 타임라인 집중 모드 추가
  const isResizing = useRef(false)

  // 채팅 상태를 부모로 끌어올림 (레이아웃 전환 시 컨텍스트 유지)
  const [chatMessages, setChatMessages] = useState([INIT_MSG])
  const [chatInput, setChatInput] = useState('')
  const [chatLoading, setChatLoading] = useState(false)

  const toggleChat = () => setIsChatOpen(v => !v)
  const toggleFocusMode = () => setIsFocusMode(v => !v)

  // 강의(id)가 변경될 때만 채팅 상태 초기화
  useEffect(() => {
    setChatMessages([INIT_MSG])
    setChatInput('')
    setChatLoading(false)
    setSeekToSeconds(null)
  }, [id])

  const handleMouseDown = useCallback(() => {
    isResizing.current = true
    document.body.style.cursor = 'col-resize'
    document.body.style.userSelect = 'none'
  }, [])

  useEffect(() => {
    const handleMouseMove = (e) => {
      if (!isResizing.current) return
      let newWidth = window.innerWidth - e.clientX
      if (newWidth < 300) newWidth = 300
      if (window.innerWidth - newWidth < 300) newWidth = window.innerWidth - 300
      setChatWidth(newWidth)
    }
    const handleMouseUp = () => {
      if (!isResizing.current) return
      isResizing.current = false
      document.body.style.cursor = 'default'
      document.body.style.userSelect = 'auto'
    }
    document.addEventListener('mousemove', handleMouseMove)
    document.addEventListener('mouseup',   handleMouseUp)
    return () => {
      document.removeEventListener('mousemove', handleMouseMove)
      document.removeEventListener('mouseup',   handleMouseUp)
    }
  }, [])

  useEffect(() => {
    if (!id) return
    setLoading(true)
    
    Promise.all([
      getLectureDetail(id),
      getLectureTimeline(id)
    ])
      .then(([detail, timeline]) => {
        // detail에 scenes가 없으면 timeline 데이터를 주입
        setLecture({
          ...detail,
          scenes: timeline || []
        })
        setCurrentScene(0)
        setSeekTo({ index: 0, time: Date.now() })
      })
      .catch((err) => {
        console.error("Lecture fetch error:", err)
        setLecture(null)
      })
      .finally(() => setLoading(false))
  }, [id])

  useEffect(() => {
    if (!id) return

    let alive = true
    let timer = null
    const sessionId = graphSessionIdRef.current

    const startGraphSession = async () => {
      try {
        await enterLectureGraphSession(id, sessionId)
        if (!alive) return
        timer = setInterval(async () => {
          try {
            await heartbeatLectureGraphSession(id, sessionId)
          } catch (e) {
            // heartbeat 실패는 다음 주기에서 재시도
          }
        }, 25000)
      } catch (e) {
        // 질의 API에서 로드 fallback이 있어 여기 실패해도 페이지는 계속 사용 가능
        console.error('graph enter failed:', e)
      }
    }

    const sendLeave = () => {
      leaveLectureGraphSession(id, sessionId).catch(() => {})
    }

    const handleBeforeUnload = () => {
      sendLeave()
    }

    startGraphSession()
    window.addEventListener('beforeunload', handleBeforeUnload)

    return () => {
      alive = false
      if (timer) clearInterval(timer)
      window.removeEventListener('beforeunload', handleBeforeUnload)
      sendLeave()
    }
  }, [id])

  const scenes      = lecture?.scenes ?? []
  const chatContext = { type: 'watch', lecture_id: id }

  function handleJumpToScene(idx, seconds = null) {
    if (Number.isInteger(idx) && idx >= 0) {
      setCurrentScene(idx)
      setSeekTo({ index: idx, time: Date.now() })
    }
    if (Number.isFinite(Number(seconds))) {
      setSeekToSeconds({ seconds: Number(seconds), time: Date.now() })
    }
  }

  if (loading) return <div className="lp-loading">불러오는 중...</div>
  if (!lecture) return <div className="lp-loading">강의를 찾을 수 없습니다</div>

  return (
    <div className="lp-shell" style={{ '--chat-width': isChatOpen ? `${chatWidth}px` : '0px' }}>
      {/* ── 본문: 전체 좌우 분할 ── */}
      <div className="lp-body">

        {/* 좌: 상단바 + 뷰어 */}
        <div className={`lp-left ${isFocusMode ? 'lp-left--focus' : ''}`}>
          {/* ── 상단바 ── */}
          <div className="lp-topbar">
            <div className="lp-topbar-left">
              <button className="lp-back" onClick={() => navigate(-1)}>
                ← 이전으로
              </button>
              <span className="lp-title">{lecture.title}</span>
            </div>
            <div className="lp-topbar-right">
              <button 
                className={`lp-focus-btn ${isFocusMode ? 'lp-focus-btn--active' : ''}`}
                onClick={toggleFocusMode}
                title={isFocusMode ? "일반 모드로 전환" : "타임라인 포커스 모드 (목록을 화면 가득히)"}
              >
                {isFocusMode ? '📺 일반 뷰' : '📜 타임라인 집중'}
              </button>
              <button className="lp-chat-open-btn" onClick={toggleChat} title="Chat 토글">
                💬 Chat
              </button>
            </div>
          </div>

          <div className="lp-viewer-container">
            <VideoPlayer
              lecture={lecture}
              scenes={scenes}
              currentScene={currentScene}
              seekTo={seekTo}
              seekToSeconds={seekToSeconds}
              onSceneChange={setCurrentScene}
              isMini={isFocusMode}
            />
            <LecturePanel
              lecture={lecture}
              scenes={scenes}
              currentScene={currentScene}
              onSceneChange={handleJumpToScene}
            />
          </div>
          {/* 모바일: 채팅이 여기 아래로 붙음 */}
          {!isFocusMode && (
            <div className="lp-mobile-chat">
              <ChatPanel
                context={chatContext}
                lecture={lecture}
                onJumpToScene={handleJumpToScene}
                mode="sidebar"
                messages={chatMessages}
                setMessages={setChatMessages}
                input={chatInput}
                setInput={setChatInput}
                loading={chatLoading}
                setLoading={setChatLoading}
              />
            </div>
          )}
        </div>

        {/* 우: 채팅 (PC만) */}
        {isChatOpen && (
          <div className="lp-right" style={{ width: chatWidth }}>
            <div className="lp-resizer" onMouseDown={handleMouseDown} />
            <ChatPanel
              context={chatContext}
              lecture={lecture}
              onJumpToScene={handleJumpToScene}
              onClose={toggleChat}
              mode="sidebar"
              messages={chatMessages}
              setMessages={setChatMessages}
              input={chatInput}
              setInput={setChatInput}
              loading={chatLoading}
              setLoading={setChatLoading}
            />
          </div>
        )}
      </div>
    </div>
  )
}
