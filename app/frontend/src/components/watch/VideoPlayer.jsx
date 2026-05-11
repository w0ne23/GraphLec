import { useState, useRef, useEffect, useCallback } from 'react'

function secToTs(s) {
  const m = Math.floor(s / 60)
  const sec = Math.floor(s % 60)
  return `${String(m).padStart(2,'0')}:${String(sec).padStart(2,'0')}`
}

function tsToSec(ts = '00:00') {
  if (typeof ts === 'number') return ts
  const parts = String(ts).split(':').map(Number)
  if (parts.some(Number.isNaN)) return 0
  if (parts.length === 1) return parts[0]
  if (parts.length === 2) return parts[0] * 60 + parts[1]
  return parts[0] * 3600 + parts[1] * 60 + parts[2]
}

export default function VideoPlayer({
  lecture,
  scenes = [],
  currentScene,
  seekTo,
  seekToSeconds = null,
  onSceneChange,
  isCinemaMode = false,
  onToggleCinemaMode,
}) {
  const videoRef = useRef(null)
  const trackRef = useRef(null)
  
  const [playing, setPlaying] = useState(false)
  const [progress, setProgress] = useState(0)
  const [duration, setDuration] = useState(0)
  const [currentTime, setCurrentTime] = useState(0)
  const [isDragging, setIsDragging] = useState(false)

  // 씬 변경 감지 시 불필요한 함수 재생성을 막기 위해 Ref 사용
  const currentSceneRef = useRef(currentScene)
  useEffect(() => {
    currentSceneRef.current = currentScene
  }, [currentScene])

  // 공통 플레이어 상태 업데이트 로직 (재생/드래그 공용)
  const updatePlayerState = useCallback((sec) => {
    const dur = videoRef.current?.duration || duration
    if (!dur) return

    setCurrentTime(sec)
    setProgress((sec / dur) * 100 || 0)

    // 장면(Scene) 동기화
    if (scenes.length > 0) {
      const activeSceneIndex = [...scenes].reverse().findIndex(s => tsToSec(s.timestamp) <= sec)
      if (activeSceneIndex !== -1) {
        const actualIndex = scenes.length - 1 - activeSceneIndex
        // Ref를 참조하여 의존성 배열에서 currentScene 제거
        if (actualIndex !== currentSceneRef.current) {
          onSceneChange(actualIndex)
        }
      }
    }
  }, [scenes, onSceneChange, duration])

  // 모든 버튼 클릭 시 포커스 해제 (컨트롤 패널 숨김 방해 방지)
  const handleBtnClick = (e, callback) => {
    e.currentTarget.blur()
    if (callback) callback()
  }

  const togglePlay = (e) => {
    if (e && e.currentTarget) e.currentTarget.blur()
    if (!videoRef.current) return
    if (playing) videoRef.current.pause()
    else videoRef.current.play()
  }

  const handleTimeUpdate = () => {
    if (!videoRef.current || isDragging) return 
    updatePlayerState(videoRef.current.currentTime)
  }

  const handleLoadedMetadata = () => {
    if (videoRef.current) setDuration(videoRef.current.duration)
  }

  // 드래그 로직
  const calculateProgress = useCallback((clientX) => {
    if (!trackRef.current || !duration) return null
    const rect = trackRef.current.getBoundingClientRect()
    let pct = (clientX - rect.left) / rect.width
    pct = Math.max(0, Math.min(1, pct))
    return pct
  }, [duration])

  const handleMouseDown = (e) => {
    e.preventDefault()
    setIsDragging(true)
    const pct = calculateProgress(e.clientX)
    if (pct !== null) {
      updatePlayerState(pct * duration)
    }
  }

  useEffect(() => {
    const handleMouseMove = (e) => {
      if (!isDragging) return
      e.preventDefault()
      const pct = calculateProgress(e.clientX)
      if (pct !== null) {
        updatePlayerState(pct * duration)
      }
    }

    const handleMouseUp = (e) => {
      if (isDragging) {
        setIsDragging(false)
        const pct = calculateProgress(e.clientX)
        if (pct !== null && videoRef.current) {
          videoRef.current.currentTime = pct * duration
        }
      }
    }

    if (isDragging) {
      document.body.style.userSelect = 'none'
      window.addEventListener('mousemove', handleMouseMove, { passive: false })
      window.addEventListener('mouseup', handleMouseUp)
    } else {
      document.body.style.userSelect = ''
      window.removeEventListener('mousemove', handleMouseMove)
      window.removeEventListener('mouseup', handleMouseUp)
    }

    return () => {
      document.body.style.userSelect = ''
      window.removeEventListener('mousemove', handleMouseMove)
      window.removeEventListener('mouseup', handleMouseUp)
    }
  }, [isDragging, calculateProgress, duration, updatePlayerState])

  // 공통 시점 이동 로직
  const jumpToTime = useCallback((targetSec, autoPlay = false) => {
    if (!videoRef.current) return
    videoRef.current.currentTime = targetSec
    updatePlayerState(targetSec)
    if (autoPlay) videoRef.current.play()
  }, [updatePlayerState])

  // 외부(목록 클릭 등)에서 명시적인 점프 요청이 왔을 때 영상 이동
  useEffect(() => {
    if (!seekTo || !scenes[seekTo.index]) return
    const targetSec = tsToSec(scenes[seekTo.index].timestamp)
    jumpToTime(targetSec, true)
  }, [seekTo, scenes, jumpToTime])

  useEffect(() => {
    const target = typeof seekToSeconds === 'object' && seekToSeconds !== null
      ? seekToSeconds.seconds
      : seekToSeconds
    if (!Number.isFinite(Number(target))) return
    jumpToTime(Math.max(0, Number(target)))
  }, [seekToSeconds, jumpToTime])

  // 스페이스바 재생/일시정지
  useEffect(() => {
    const handleKeyDown = (e) => {
      // 입력창, 텍스트 영역 예외 처리
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return
      
      if (e.code === 'Space') {
        e.preventDefault()
        togglePlay()
      }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [playing])

  if (!lecture) {
    return (
      <div className="vp-placeholder">
        <div className="vp-placeholder-icon">🎬</div>
        <p>강의 데이터를 불러오는 중입니다...</p>
      </div>
    )
  }

  return (
    <div className={`vp-wrap`}>
      <div className="vp-video-container">
        <video
          ref={videoRef}
          src={lecture.video_url}
          className="vp-video-element"
          onPlay={() => setPlaying(true)}
          onPause={() => setPlaying(false)}
          onTimeUpdate={handleTimeUpdate}
          onLoadedMetadata={handleLoadedMetadata}
          onClick={togglePlay}
        />
        {!playing && !isDragging && (
          <div className="vp-overlay" onClick={togglePlay}>
            <div className="vp-play-icon">▶</div>
          </div>
        )}

        <div className="vp-controls">
          <div 
            className="vp-track" 
            ref={trackRef}
            onMouseDown={handleMouseDown}
            style={{ cursor: isDragging ? 'grabbing' : 'pointer' }}
          >
            <div className="vp-fill" style={{ width: `${progress}%`, transition: isDragging ? 'none' : 'width 0.1s linear' }}>
              <div className="vp-thumb" style={{ transform: isDragging ? 'scale(1.2)' : 'scale(1)' }} />
            </div>
            {scenes.map((s, i) => {
              if (!duration) return null
              const pct = (tsToSec(s.timestamp) / duration) * 100
              const cls = { slide: 'vp-marker--slide', emphasis: 'vp-marker--emphasis', demo: 'vp-marker--demo' }[s.type] ?? 'vp-marker--slide'
              return (
                <div
                  key={i}
                  className={`vp-marker ${cls} ${i === currentScene ? 'vp-marker--active' : ''}`}
                  style={{ left: `${pct}%` }}
                  title={s.text}
                />
              )
            })}
          </div>
          <div className="vp-ctrl-row">
            <button className="vp-cbtn" onClick={(e) => handleBtnClick(e, () => { if(videoRef.current) videoRef.current.currentTime -= 10 })}>↺</button>
            <button className="vp-cbtn" onClick={togglePlay}>{playing ? '⏸' : '▶'}</button>
            <button className="vp-cbtn" onClick={(e) => handleBtnClick(e, () => { if(videoRef.current) videoRef.current.currentTime += 10 })}>↻</button>
            
            <span className="vp-mono">{secToTs(currentTime)}</span>
            <span className="vp-muted">/ {secToTs(duration)}</span>

            <span className="vp-scene-no">
              Scene {currentScene + 1}
            </span>

            <span className="vp-scene-title">
              {scenes[currentScene]?.text || ''}
            </span>

            <div style={{ flex: 1 }} />
            <button
              className={`vp-cbtn ${isCinemaMode ? 'vp-cbtn--active' : ''}`}
              onClick={(e) => handleBtnClick(e, onToggleCinemaMode)}
              title={isCinemaMode ? '일반 모드' : '영화관 모드'}
              style={{ marginLeft: '8px', fontSize: '14px' }}
            >
              🎬
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
