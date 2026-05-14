import { useState, useRef, useCallback, useEffect } from 'react'

export function useResizer(initialWidth = 300) {
  const [chatWidth, setChatWidth] = useState(initialWidth)
  const isResizing = useRef(false)

  const handleResizerMouseDown = useCallback(() => {
    isResizing.current = true
    document.body.style.cursor = 'col-resize'
    document.body.style.userSelect = 'none'
  }, [])

  useEffect(() => {
    const handleMouseMove = (e) => {
      if (!isResizing.current) return
      let w = window.innerWidth - e.clientX
      if (w < 260) w = 260
      if (window.innerWidth - w < 300) w = window.innerWidth - 300
      setChatWidth(w)
    }
    const handleMouseUp = () => {
      if (!isResizing.current) return
      isResizing.current = false
      document.body.style.cursor = 'default'
      document.body.style.userSelect = 'auto'
    }
    document.addEventListener('mousemove', handleMouseMove)
    document.addEventListener('mouseup', handleMouseUp)
    return () => {
      document.removeEventListener('mousemove', handleMouseMove)
      document.removeEventListener('mouseup', handleMouseUp)
    }
  }, [])

  return { chatWidth, handleResizerMouseDown }
}
