import { useEffect, useRef, useState } from 'react'
import { listActiveJobs, listUploadedLectures, uploadLecture, deleteLecture, retryLecture } from '../lib/api'

import '../styles/upload.css'

const STAGE_LABELS = ['장면 감지', '음성 분석', 'STT 전사', '분석 통합', '그래프/DB', '요약 색인', '메타데이터']
const STAGE_KEYS   = ['scene', 'voice', 'stt', 'integrate', 'graph', 'summarize', 'metadata']

const STATUS_MAP = {
  done:       { label: '분석 완료', cls: 'status-done' },
  processing: { label: '분석 중',   cls: 'status-proc' },
  pending:    { label: '대기',       cls: 'status-wait' },
  error:      { label: '오류',       cls: 'status-err'  },
}

const THUMB_COLOR = {
  '컴퓨터 과학': '#1e3a5f', '데이터 사이언스': '#1a3d2b',
  '소프트웨어 공학': '#2d1f3d', '수학': '#3d2a1a',
}
const THUMB_ICON = {
  '컴퓨터 과학': '🧠', '데이터 사이언스': '📊',
  '소프트웨어 공학': '⚙️', '수학': '📐',
}

export default function UploadPage() {
  const [lectures,    setLectures]    = useState([])
  const [title,       setTitle]       = useState('')
  const [category,    setCategory]    = useState('컴퓨터 과학')
  const [description, setDescription] = useState('')
  const [file,        setFile]        = useState(null)
  const [dragOver,    setDragOver]    = useState(false)
  const [uploading,   setUploading]   = useState(null)
  const [error,       setError]       = useState('')
  const [loadingLectures, setLoadingLectures] = useState(true)

  const [totalPages, setTotalPages] = useState(1)
  const [currentPage, setCurrentPage] = useState(1)

  const fileRef    = useRef()
  const scrollRef  = useRef()   // 페이지 스크롤 컨테이너
  const listRef    = useRef()   // 강의 목록 섹션
  const eventSources = useRef({}) // { job_id: EventSource }
  const healthTimer  = useRef(null)

  // 서버 다운 감지 시 진행 중인 모든 SSE 연결을 일괄 에러 처리
  const handleServerDown = () => {
    console.error('--- [Health] Server down detected ---')
    const activeIds = Object.keys(eventSources.current)
    if (activeIds.length === 0) return

    activeIds.forEach(id => {
      eventSources.current[id]?.close()
      delete eventSources.current[id]
    })

    setLectures(prev => prev.map(lec =>
      (lec.status === 'running' || lec.status === 'pending')
        ? { ...lec, status: 'error', error_message: '서버와의 연결이 끊어졌습니다.' }
        : lec
    ))

    if (healthTimer.current) {
      clearInterval(healthTimer.current)
      healthTimer.current = null
    }
  }

  // 진행 중인 작업이 있을 때만 헬스체크 폴링 동작
  const startHealthCheck = () => {
    if (healthTimer.current) return
    console.log('--- [Health] Starting health check polling ---')
    healthTimer.current = setInterval(async () => {
      if (Object.keys(eventSources.current).length === 0) {
        clearInterval(healthTimer.current)
        healthTimer.current = null
        console.log('--- [Health] No active jobs, stopping health check ---')
        return
      }
      try {
        const res = await fetch('/api/health')
        if (!res.ok) throw new Error(`status ${res.status}`)
      } catch (e) {
        handleServerDown()
      }
    }, 3000)
  }

  // 주기적으로(또는 처음 로드 시) 진행 중인 작업에 대해 SSE 연결을 맺는 함수
  const setupSSEForJob = (lecture_id, job_id) => {
    if (eventSources.current[lecture_id]) return // 이미 연결되어 있음

    const url = job_id
      ? `/api/jobs/${lecture_id}/stream?job_id=${job_id}`
      : `/api/jobs/${lecture_id}/stream`
    console.log(`--- [SSE] Connecting to stream for lecture ${lecture_id} (job ${job_id ?? 'latest'}) ---`)
    const eventSource = new EventSource(url)
    eventSources.current[lecture_id] = eventSource
    startHealthCheck()

    let closed = false
    const closeSSE = () => {
      if (closed) return
      closed = true
      eventSource.close()
      delete eventSources.current[lecture_id]
    }

    eventSource.onmessage = (event) => {
      if (closed) return
      try {
        const data = JSON.parse(event.data)
        if (data.error) {
          console.error("SSE Error:", data.error)
          setLectures(prev => prev.map(lec => 
             lec.id === lecture_id
               ? { ...lec, status: 'error', error_message: data.error } 
               : lec
           ))
          closeSSE()
          return
        }

        setLectures(prev => prev.map(lec => {
          if (lec.id === lecture_id) {
            let stages = data.pipeline_stages
            if (!stages || stages.length === 0) {
               stages = STAGE_KEYS.map((key) => ({ stage: key, status: 'wait' }))
            }
            return { 
              ...lec, 
              job_id: data.job_id, // 혹시 바뀌었을 경우를 대비
              status: data.lecture_status,
              current_stage: data.current_stage,
              error_message: data.error_message || lec.error_message,
              pipeline_stages: stages
            }
          }
          return lec
        }))

        if (data.lecture_status === 'done' || data.lecture_status === 'error') {
           closeSSE()
        }
      } catch (err) { console.error("SSE Parse error:", err) }
    }

    eventSource.onerror = (err) => {
      if (closed) return
      closeSSE()
      setLectures(prev => prev.map(lec => 
        lec.id === lecture_id ? { ...lec, status: 'error', error_message: '연결이 끊어졌습니다.' } : lec
      ))
    }
  }

  // 처음 로드 시
  useEffect(() => {
    Promise.all([
      listActiveJobs(),
      listUploadedLectures({ page: 1, limit: 12 }),
    ]).then(([activeJobs, result]) => {
      setLectures([...activeJobs, ...result.items])
      setTotalPages(result.totalPages)
      activeJobs.forEach(lec => {
        setupSSEForJob(lec.id, lec.job_id)
      })
    })
      .catch(e => setError(String(e.message || e)))
      .finally(() => setLoadingLectures(false))

    return () => {
      Object.values(eventSources.current).forEach(s => s.close())
      eventSources.current = {}
      if (healthTimer.current) clearInterval(healthTimer.current)
    }
  }, [])

  function setFileWithAutoTitle(f) {
    if (!f) return
    setFile(f)
    setTitle(prev => prev.trim() ? prev : f.name.replace(/\.[^.]+$/, ''))
  }

  async function handleSubmit() {
    if (uploading || !file) return
    setError('')
    listRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' })

    try {
      const created = await uploadLecture({ title: title || file.name, category, description, file })
      // created에는 id(lecture_id)와 job_id가 모두 있음
      setLectures(prev => [created, ...prev])
      setupSSEForJob(created.id, created.job_id)
    } catch (e) {
      setError(String(e.message || e))
    } finally {
      setTitle(''); setCategory('컴퓨터 과학'); setDescription(''); setFile(null)
    }
  }

  async function handleDelete(lectureId, e) {
    e.stopPropagation()
    if (!confirm('이 강의를 삭제하시겠습니까?')) return
    try {
      await deleteLecture(lectureId)
      setLectures(prev => prev.filter(l => l.id !== lectureId))
      if (eventSources.current[lectureId]) {
        eventSources.current[lectureId].close()
        delete eventSources.current[lectureId]
      }
    } catch (err) { alert(`삭제 실패: ${err.message}`) }
  }

  async function handleRetry(lectureId, e) {
    e.stopPropagation()
    if (!confirm('분석을 다시 시도하시겠습니까?')) return
    try {
      const { job_id } = await retryLecture(lectureId)

      // 이전 SSE 정리 후 새 job_id로 재연결
      if (eventSources.current[lectureId]) {
        eventSources.current[lectureId].close()
        delete eventSources.current[lectureId]
      }
      setLectures(prev => prev.map(l =>
        l.id === lectureId
          ? { ...l, job_id, status: 'pending', pipeline_stages: [], current_stage: 'Resuming pipeline...' }
          : l
      ))
      setupSSEForJob(lectureId, job_id)
    } catch (err) { alert(`재시도 실패: ${err.message}`) }
  }

  return (
    <div className="up-page" ref={scrollRef}>

      {/* ── 섹션 1: 업로드 폼 ── */}
      <section className="up-form-section">
        <div className="up-form-inner">
          <div className="up-form-logo">Graph<span>Lec</span></div>
          <p className="up-form-sub">강의 영상을 업로드하면 자동으로 분석합니다.</p>
          {error && <p className="up-error">{error}</p>}
          <div
            className={`up-dropzone${dragOver ? ' up-dropzone--drag' : ''}${file ? ' up-dropzone--file' : ''}`}
            onClick={() => fileRef.current.click()}
            onDragOver={e => { e.preventDefault(); setDragOver(true) }}
            onDragLeave={() => setDragOver(false)}
            onDrop={e => { e.preventDefault(); setDragOver(false); setFileWithAutoTitle(e.dataTransfer.files[0]) }}
          >
            <div className="up-dz-icon">{file ? '✅' : '🎬'}</div>
            {file
              ? <><p className="up-dz-text"><strong>{file.name}</strong></p><p className="up-dz-sub">{(file.size/1024/1024).toFixed(1)} MB</p></>
              : <><p className="up-dz-text"><strong>드래그하거나 클릭</strong></p><p className="up-dz-sub">MP4, MOV, AVI · 최대 2GB</p></>
            }
            <input ref={fileRef} type="file" accept="video/*" style={{ display: 'none' }} onChange={e => setFileWithAutoTitle(e.target.files[0])} />
          </div>
          <div className="up-field-group">
            <div className="up-field">
              <label className="up-label">강의 제목</label>
              <input className="up-input" value={title} onChange={e => setTitle(e.target.value)} placeholder="제목을 입력하세요" />
            </div>
            <div className="up-field">
              <label className="up-label">카테고리</label>
              <select className="up-select" value={category} onChange={e => setCategory(e.target.value)}>
                <option>컴퓨터 과학</option><option>수학</option><option>데이터 사이언스</option><option>소프트웨어 공학</option>
              </select>
            </div>
            <div className="up-field">
              <label className="up-label">강의 설명 (선택)</label>
              <textarea className="up-textarea" value={description} onChange={e => setDescription(e.target.value)} placeholder="강의 내용을 간략히 설명하세요..." />
            </div>
          </div>
          <button className="up-submit-btn" onClick={handleSubmit} disabled={!!uploading || !file}>업로드 시작</button>
          <button className="up-scroll-hint" onClick={() => listRef.current?.scrollIntoView({ behavior: 'smooth' })}>강의 목록 보기 ↓</button>
        </div>
      </section>

      {/* ── 섹션 2: 강의 목록 ── */}
      <section className="up-list-section" ref={listRef}>
        <div className="up-list-header content-max">
          <div className="up-list-title">강의 목록</div>
          <span className="up-list-count">{lectures.length}개</span>
        </div>
        <div className="up-list content-max">
          {loadingLectures && lectures.length === 0 && <div className="up-empty">강의 목록을 불러오는 중입니다</div>}
          {!loadingLectures && lectures.length === 0 && <div className="up-empty">업로드된 강의가 없습니다</div>}
          {lectures.map(lec => {
            const st = STATUS_MAP[lec.status] ?? STATUS_MAP.pending
            const thumbBg = THUMB_COLOR[lec.category] ?? '#1e2333'
            const thumbIcon = THUMB_ICON[lec.category] ?? '🎬'

            return (
              <div key={lec.id}>
                <div className={`upload-row${lec.status === 'done' ? ' upload-row--done' : ''}`} onClick={() => lec.status === 'done' && navigate(`/lectures/${lec.id}`)}>
                  <div className="upload-row-thumb" style={{ background: thumbBg }}><span className="upload-row-thumb-icon">{thumbIcon}</span></div>
                  <div className="upload-row-main">
                    <div className="upload-row-title">{lec.title}</div>
                    <div className="upload-row-meta"><span className="upload-row-cat">{lec.category}</span></div>
                  </div>
                  <div className="upload-row-date">{new Date(lec.created_at).toLocaleDateString('ko-KR')}</div>
                  <div className="upload-row-status"><span className={`upload-status-badge ${st.cls}`}>{st.label}</span></div>
                  <div className="upload-row-actions">
                    {lec.status === 'done' && <button className="upload-btn-verifier" onClick={(e) => { e.stopPropagation(); navigate(`/lectures/${lec.id}/verifier`) }}>Verifier</button>}
                    {lec.status === 'error' && <button className="upload-btn-retry" onClick={e => handleRetry(lec.id, e)}>재시도</button>}
                    {lec.status !== 'done' && <button className="upload-btn-delete" onClick={e => handleDelete(lec.id, e)}>삭제</button>}
                    {lec.status === 'done' && <span className="upload-row-arrow">→</span>}
                  </div>
                </div>
                {(lec.status === 'running' || lec.status === 'pending' || lec.status === 'error') && lec.pipeline_stages && (
                  <div className="upload-pipe" style={{ margin: '-4px 0 4px', borderTopLeftRadius: 0, borderTopRightRadius: 0 }}>
                    <div className="upload-pipe-track">
                      <div className="upload-pipe-fill" style={{ width: lec.status === 'done' ? '100%' : lec.status === 'error' ? '0%' : `${Math.max(0, lec.pipeline_stages.filter(s => s.status === 'done').length * (100 / STAGE_KEYS.length) + (lec.pipeline_stages.some(s => s.status === 'run') ? (100 / STAGE_KEYS.length) / 2 : 0))}%` }} />
                    </div>
                    <div className="upload-pipe-stages">
                      {STAGE_LABELS.map((lbl, i) => {
                        const stageStatus = lec.pipeline_stages.find(s => s.stage === STAGE_KEYS[i])?.status ?? 'wait'
                        return <span key={i} className={`upload-chip${stageStatus === 'done' ? ' upload-chip--done' : stageStatus === 'run' ? ' upload-chip--run' : ''}`}>{stageStatus === 'done' ? '✓ ' : stageStatus === 'run' ? '↻ ' : ''}{lbl}</span>
                      })}
                    </div>
                    {lec.status === 'error' && <div style={{ marginTop: '8px', fontSize: '11px', color: 'var(--red)' }}>오류: {lec.error_message}</div>}
                  </div>
                )}
              </div>
            )
          })}
        </div>
      </section>
    </div>
  )
}