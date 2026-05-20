import { useState, useEffect } from 'react'

const INIT_MSG = { role: 'assistant', content: '강의에 대해 질문해보세요.', refs: [] }

export function useChatSession(lectureId) {
  const [messages, setMessages] = useState([INIT_MSG])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [sessionId, setSessionId] = useState('')

  useEffect(() => {
    setMessages([INIT_MSG])
    setInput('')
    setLoading(false)
    if (!lectureId) {
      setSessionId('')
      return
    }
    const storageKey = `graphlec.chatSession.${lectureId}`
    const existing = window.localStorage.getItem(storageKey)
    if (existing) {
      setSessionId(existing)
      return
    }
    const next = window.crypto?.randomUUID?.() || `${Date.now()}-${Math.random().toString(36).slice(2)}`
    window.localStorage.setItem(storageKey, next)
    setSessionId(next)
  }, [lectureId])

  return { messages, setMessages, input, setInput, loading, setLoading, sessionId }
}
