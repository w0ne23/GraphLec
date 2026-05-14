import { useState, useEffect } from 'react'

const INIT_MSG = { role: 'assistant', content: '강의에 대해 질문해보세요.', refs: [] }

export function useChatSession(lectureId) {
  const [messages, setMessages] = useState([INIT_MSG])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    setMessages([INIT_MSG])
    setInput('')
    setLoading(false)
  }, [lectureId])

  return { messages, setMessages, input, setInput, loading, setLoading }
}
