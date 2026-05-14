import { useEffect, useRef } from 'react'
import {
  enterLectureGraphSession,
  heartbeatLectureGraphSession,
  leaveLectureGraphSession,
} from '../lib/api'

export function useGraphSession(lectureId) {
  const sessionIdRef = useRef(
    window.crypto?.randomUUID?.()
    ?? `sess-${Date.now()}-${Math.random().toString(36).slice(2)}`
  )

  useEffect(() => {
    if (!lectureId) return

    let alive = true
    let timer = null
    const sessionId = sessionIdRef.current

    const startGraphSession = async () => {
      try {
        await enterLectureGraphSession(lectureId, sessionId)
        if (!alive) return
        timer = setInterval(async () => {
          try {
            await heartbeatLectureGraphSession(lectureId, sessionId)
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
      leaveLectureGraphSession(lectureId, sessionId).catch(() => {})
    }

    const handleBeforeUnload = () => sendLeave()

    startGraphSession()
    window.addEventListener('beforeunload', handleBeforeUnload)

    return () => {
      alive = false
      if (timer) clearInterval(timer)
      window.removeEventListener('beforeunload', handleBeforeUnload)
      sendLeave()
    }
  }, [lectureId])
}
