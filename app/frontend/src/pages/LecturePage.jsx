import { useEffect, useState, useRef } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import {
  getLectureDetail,
  getLectureTimeline,
} from '../lib/api'

import { useChatSession } from '../hooks/useChatSession'
import { useGraphSession } from '../hooks/useGraphSession'
import { useResizer } from '../hooks/useResizer'

import VideoPlayer from '../components/watch/VideoPlayer'
import VideoTimeline from '../components/watch/VideoTimeline'
import GraphViewer from '../components/watch/GraphViewer'
import ChatPanel from '../components/watch/ChatPanel'
import LectureInfoModal from '../components/watch/LectureInfoModal'

import '../styles/lecture.css'

export default function LecturePage() {
  const { id } = useParams()
  const navigate = useNavigate()

  const { chatWidth, handleResizerMouseDown } = useResizer(300)
  const { messages: chatMessages, setMessages: setChatMessages, input: chatInput, setInput: setChatInput, loading: chatLoading, setLoading: setChatLoading, sessionId: chatSessionId } = useChatSession(id)
  useGraphSession(id)

  const [lecture, setLecture] = useState(null)
  const [loading, setLoading] = useState(false)
  const [currentScene, setCurrentScene] = useState(0)
  const [seekTo, setSeekTo] = useState(null)
  const [seekToSeconds, setSeekToSeconds] = useState(null)

  // 우측 패널
  const [isChatOpen, setIsChatOpen] = useState(true)
  const [isGraphPanelOpen, setIsGraphPanelOpen] = useState(false)

  // 타임라인 패널
  const [isTimelineOpen, setIsTimelineOpen] = useState(true)

  // 영화관 모드: 타임라인 + 우측 패널 전부 숨김
  const [isCinemaMode, setIsCinemaMode] = useState(false)

  // 강의 정보 모달
  const [isInfoOpen, setIsInfoOpen] = useState(false)

  const leftRef = useRef(null)

  const toggleChat = () => setIsChatOpen(v => !v)
  const toggleGraphPanel = () => setIsGraphPanelOpen(v => !v)
  const toggleTimeline = () => setIsTimelineOpen(v => !v)
  const toggleCinemaMode = () => setIsCinemaMode(v => !v)

  // ESC 키로 영화관 모드 종료
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === 'Escape' && isCinemaMode) {
        setIsCinemaMode(false)
      }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [isCinemaMode])

  // 강의 데이터 로딩
  useEffect(() => {
    if (!id) return
    setLoading(true)
    setSeekToSeconds(null)

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

  const scenes = lecture?.scenes ?? []
  const chatContext = { type: 'watch', lecture_id: id }

  function handleJumpToScene(idx, seconds = null, options = {}) {
    const autoPlay = options.autoPlay ?? true
    const offsetSec = Number.isFinite(Number(options.offsetSec)) ? Number(options.offsetSec) : 0
    if (Number.isInteger(idx) && idx >= 0) {
      setCurrentScene(idx)
      setSeekTo({ index: idx, time: Date.now(), autoPlay, offsetSec })
    }
    if (seconds !== null && Number.isFinite(Number(seconds))) {
      setSeekToSeconds({ seconds: Number(seconds), time: Date.now() })
    }
  }

  if (loading) return <div className="lp-loading">불러오는 중...</div>
  if (!lecture) return <div className="lp-loading">강의를 찾을 수 없습니다</div>

  return (
    <div
      className={`lp-shell ${isCinemaMode ? 'lp-shell--cinema' : ''}`}
      style={{ '--chat-width': isChatOpen && !isCinemaMode ? `${chatWidth}px` : '0px' }}
    >
      <div className="lp-body">

        {/* 좌: 상단바 + 뷰어 + 타임라인 */}
        <div className="lp-left" ref={leftRef}>

          {/* 상단바 래퍼 (공간 점유 및 애니메이션 주체) */}
          <div className={`lp-topbar-wrap ${isCinemaMode ? 'lp-topbar-wrap--cinema' : ''}`}>
            <div className={`lp-topbar ${isCinemaMode ? 'lp-topbar--cinema' : ''}`}>
              <div className="lp-topbar-left">
                <button className="lp-back" onClick={() => navigate(-1)}>
                  ← 이전으로
                </button>
                <span className="lp-title">{lecture.title}</span>
                <button
                  className="lp-icon-btn"
                  onClick={() => setIsInfoOpen(true)}
                  title="강의 정보"
                >ℹ️</button>
              </div>
              <div className="lp-topbar-right">
                {!isCinemaMode && (
                  <button 
                    className="lp-chat-open-btn"
                    onClick={toggleChat}
                    title="Chat 토글"
                  >
                    💬 Chat
                  </button>
                )}
              </div>
            </div>
          </div>

          {/* 영상 뷰어 */}
          <div className="lp-video-area">
            <VideoPlayer
              lecture={lecture}
              scenes={scenes}
              currentScene={currentScene}
              seekTo={seekTo}
              seekToSeconds={seekToSeconds}
              onSceneChange={setCurrentScene}
              isCinemaMode={isCinemaMode}
              onToggleCinemaMode={toggleCinemaMode}
            />
          </div>

          {/* 타임라인 (가로/세로 자동 전환) */}
          <VideoTimeline
            scenes={scenes}
            currentScene={currentScene}
            onSceneChange={handleJumpToScene}
            leftRef={leftRef}
            isOpen={isTimelineOpen}
            onToggle={toggleTimeline}
            isCinemaMode={isCinemaMode}
          />
        </div>

        {/* 우: 그래프 + 채팅 */}
        <div 
          className={`lp-right ${!isChatOpen || isCinemaMode ? 'lp-right--closed' : ''}`}
          style={{ width: isChatOpen && !isCinemaMode ? chatWidth : 0 }}
        >
          <div className="lp-resizer" onMouseDown={handleResizerMouseDown} />

          {/* 그래프 */}
          <div className={`mg-wrap ${isGraphPanelOpen ? 'mg-wrap--open' : ''}`}>
            <div className="mg-header" onClick={toggleGraphPanel}>
              <span className="mg-title">그래프</span>
              <button className="mg-toggle-btn">{isGraphPanelOpen ? '▲' : '▼'}</button>
            </div>
            <div className="mg-body">
              <GraphViewer lectureId={id} />
            </div>
          </div>

          {/* 채팅 */}
          <div className="lp-chat-area">
          <ChatPanel
            context={chatContext}
            lecture={lecture}
            currentSceneIndex={currentScene}
            onJumpToScene={handleJumpToScene}
            onClose={toggleChat}
            mode="sidebar"
            messages={chatMessages}
            setMessages={setChatMessages}
            input={chatInput}
            setInput={setChatInput}
            loading={chatLoading}
            setLoading={setChatLoading}
            chatSessionId={chatSessionId}
          />
          </div>
        </div>
      </div>

      {/* 강의 정보 모달 */}
      {isInfoOpen && (
        <LectureInfoModal
          lecture={lecture}
          scenes={scenes}
          onClose={() => setIsInfoOpen(false)}
        />
      )}
    </div>
  )
}
