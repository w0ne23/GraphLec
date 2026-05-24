import { useEffect, useMemo, useRef, useState } from 'react'
import VideoPlayer from '../components/watch/VideoPlayer'

import '../styles/preview.css'

const PHASES = {
  UPLOAD: 'upload',
  PREPROCESS: 'preprocess',
  VERIFY_READY: 'verifyReady',
  REVIEW: 'review',
  GRAPH: 'graph',
  DONE: 'done',
}

const SAMPLE_FILE = {
  name: 'lecture2.mp4',
  size: 13436881,
  url: '/sample-videos/lecture2.mp4',
}

const DEFAULT_TITLE = '운영체제 강의 영상'
const STEP_DELAY_MS = 3000

const PREPROCESS_STAGES = [
  { key: 'scene', label: '슬라이드 프레임 추출' },
  { key: 'audio', label: '오디오 품질 분석' },
  { key: 'stt', label: '강의 음성 텍스트 전사' },
  { key: 'merge', label: '전처리 결과 통합' },
]

const VERIFIER_STAGES = [
  { key: 'verifier_input', label: '검증 입력 데이터 구성' },
  { key: 'run_verifier', label: '강의 내용 검증 실행' },
]

const GRAPH_STAGES = [
  { key: 'classify', label: '슬라이드 유형 분류' },
  { key: 'fusion', label: '전체 데이터 통합' },
  { key: 'graph', label: '그래프 데이터 생성' },
  { key: 'index', label: '검색 인덱스 생성' },
  { key: 'metadata', label: '강의 메타데이터 생성' },
]

const FLOW_NODES = {
  direct_upload: [
    { id: 'upload', label: '업로드', type: 'major' },
    { id: 'preprocess', label: '전처리', type: 'minor', stages: PREPROCESS_STAGES },
    { id: 'graph', label: '그래프 생성', type: 'minor', stages: GRAPH_STAGES },
    { id: 'done', label: '완료', type: 'major' },
  ],
  verified_upload: [
    { id: 'upload', label: '업로드', type: 'major' },
    { id: 'preprocess', label: '전처리', type: 'minor', stages: PREPROCESS_STAGES },
    { id: 'verifier', label: '강의 내용 검증', type: 'minor', stages: VERIFIER_STAGES },
    { id: 'approval', label: '검증 결과 확인', type: 'major' },
    { id: 'graph', label: '그래프 생성', type: 'minor', stages: GRAPH_STAGES },
    { id: 'done', label: '완료', type: 'major' },
  ],
}

const REVIEW_CLAIMS = [
  {
    id: 'claim-1',
    time: 192,
    title: '운영체제는 항상 CPU를 하나의 프로세스에만 할당한다.',
    type: 'scope_overclaim',
    typeLabel: '범위 과잉 단정',
    detail: '스케줄링 정책, 멀티코어, 스레드 모델에 따라 여러 실행 단위를 동시에 처리할 수 있습니다.',
  },
  {
    id: 'claim-2',
    time: 521,
    title: '가상 메모리는 실제 메모리 크기와 항상 동일하다.',
    type: 'factual_error',
    typeLabel: '발언 자체 오류',
    detail: '가상 메모리는 프로세스가 보는 주소 공간이며 실제 물리 메모리와 일대일로 고정되지 않습니다.',
  },
  {
    id: 'claim-3',
    time: 728,
    title: '페이지 교체는 보통 LRU만 사용한다고 설명한 부분',
    type: 'confusing_explanation',
    typeLabel: '혼동 가능 설명',
    detail: 'LRU는 대표 예시지만 실제 시스템에서는 Clock, 근사 LRU 등 다양한 정책을 사용할 수 있습니다.',
  },
]

const SLIDE_TYPOS = [
  {
    id: 'typo-1',
    slide: 7,
    from: 'Interupt',
    to: 'Interrupt',
    reason: '슬라이드 내 운영체제 용어 오탈자',
  },
]

function formatSize(bytes) {
  const value = Number(bytes)
  return Number.isFinite(value) ? `${(value / 1024 / 1024).toFixed(1)} MB` : '실제 verification 결과 기반'
}

function formatTime(seconds) {
  const safe = Math.max(0, Number(seconds) || 0)
  const min = Math.floor(safe / 60)
  const sec = Math.floor(safe % 60)
  return `${String(min).padStart(2, '0')}:${String(sec).padStart(2, '0')}`
}

function stagesForPhase(mode, phase) {
  if (phase === PHASES.PREPROCESS) {
    return mode === 'verified_upload'
      ? [...PREPROCESS_STAGES, ...VERIFIER_STAGES]
      : [...PREPROCESS_STAGES, ...GRAPH_STAGES]
  }
  if (phase === PHASES.GRAPH) return GRAPH_STAGES
  return []
}

function nodeStatus(node, mode, phase, activeStageKey, finishedStageKeys) {
  if (node.id === 'upload') return phase === PHASES.UPLOAD ? 'run' : 'done'
  if (node.id === 'approval') {
    if (phase === PHASES.VERIFY_READY || phase === PHASES.REVIEW) return 'run'
    if (phase === PHASES.GRAPH || phase === PHASES.DONE) return 'done'
    return 'wait'
  }
  if (node.id === 'done') return phase === PHASES.DONE ? 'run' : 'wait'
  if (!node.stages?.length) return 'wait'

  const keys = node.stages.map(stage => stage.key)
  if (keys.includes(activeStageKey)) return 'run'
  if (keys.every(key => finishedStageKeys.has(key))) return 'done'
  if (mode === 'direct_upload' && node.id === 'graph' && phase === PHASES.DONE) return 'done'
  if (mode === 'verified_upload' && node.id === 'graph' && phase === PHASES.DONE) return 'done'
  return 'wait'
}

function stageStatus(stage, activeStageKey, finishedStageKeys) {
  if (finishedStageKeys.has(stage.key)) return 'done'
  if (stage.key === activeStageKey) return 'run'
  return 'wait'
}

function PipelineProgress({ mode, phase, stageIndex }) {
  const nodes = FLOW_NODES[mode]
  const activeStages = stagesForPhase(mode, phase)
  const activeStage = activeStages[stageIndex]
  const finishedStageKeys = new Set(activeStages.slice(0, stageIndex).map(stage => stage.key))

  if (phase === PHASES.VERIFY_READY || phase === PHASES.REVIEW) {
    PREPROCESS_STAGES.forEach(stage => finishedStageKeys.add(stage.key))
    VERIFIER_STAGES.forEach(stage => finishedStageKeys.add(stage.key))
  }
  if (phase === PHASES.GRAPH || phase === PHASES.DONE) {
    PREPROCESS_STAGES.forEach(stage => finishedStageKeys.add(stage.key))
    VERIFIER_STAGES.forEach(stage => finishedStageKeys.add(stage.key))
  }
  if (phase === PHASES.DONE) {
    GRAPH_STAGES.forEach(stage => finishedStageKeys.add(stage.key))
  }

  const activeNode = nodes.find(node => nodeStatus(node, mode, phase, activeStage?.key, finishedStageKeys) === 'run')
  const visibleStages = activeNode?.stages?.length ? activeNode.stages : activeStages
  const statusMessage = activeStage
    ? `${activeNode?.label || '파이프라인'} 진행 중`
    : phase === PHASES.VERIFY_READY
      ? 'Verifier 결과가 준비되었습니다.'
      : phase === PHASES.DONE
        ? '분석이 완료되었습니다.'
        : '작업 대기 중'

  return (
    <div className="pv-vf-pipe">
      <div className="pv-vf-progress-head">
        <div className="pv-vf-progress-message">{statusMessage}</div>
      </div>
      <div className="pv-vf-work-log">
        <div className="pv-vf-work-log-lines">
          {visibleStages.map(stage => {
            const status = stageStatus(stage, activeStage?.key, finishedStageKeys)
            return (
              <div key={stage.key} className={`pv-vf-work-log-line pv-vf-work-log-line--${status}`}>
                <span>{stage.label}</span>
                <em>{status === 'done' ? '완료!' : status === 'run' ? '진행 중...' : '대기 중'}</em>
              </div>
            )
          })}
        </div>
      </div>
      <div className="pv-vf-flow" style={{ '--flow-count': nodes.length }}>
        {nodes.map(node => {
          const status = nodeStatus(node, mode, phase, activeStage?.key, finishedStageKeys)
          return (
            <div key={node.id} className={`pv-vf-flow-item pv-vf-flow-item--${node.type} pv-vf-flow-item--${status}`}>
              <div className="pv-vf-flow-node-slot">
                <div className="pv-vf-flow-node" aria-label={`${node.label} ${status}`} />
              </div>
              <div className="pv-vf-flow-label">{node.label}</div>
            </div>
          )
        })}
      </div>
    </div>
  )
}

function UploadStep({ file, title, dragOver, onDrop, onPickFile, onTitleChange, onStart, onResetSample, setDragOver }) {
  const fileRef = useRef(null)

  return (
    <div className="pv-vf-upload-wrap">
      <div className="pv-vf-upload-inner">
        <div className="pv-vf-preview-header">
          <strong>Preview</strong>
          <button className="pv-vf-cancel-btn" onClick={onResetSample}>샘플 영상 다시 넣기</button>
        </div>
        <p className="pv-vf-upload-sub">실제 파일 생성, DB 저장, API 호출 없이 파이프라인 흐름만 확인합니다.</p>

        <div
          className={`pv-vf-dropzone${dragOver ? ' pv-vf-dropzone--drag' : ''}${file ? ' pv-vf-dropzone--file' : ''}`}
          onClick={() => fileRef.current?.click()}
          onDragOver={event => { event.preventDefault(); setDragOver(true) }}
          onDragLeave={() => setDragOver(false)}
          onDrop={event => {
            event.preventDefault()
            setDragOver(false)
            onDrop(event.dataTransfer.files[0])
          }}
        >
          <div className="pv-vf-dropzone-icon">{file ? '✅' : '🎬'}</div>
          <p className="pv-vf-dropzone-text"><strong>{file?.name || '드래그하거나 클릭'}</strong></p>
          <p className="pv-vf-dropzone-sub">{file ? `${formatSize(file.size)} · 서버로 전송하지 않음` : 'MP4, MOV, AVI · preview only'}</p>
          <input ref={fileRef} type="file" accept="video/*" hidden onChange={event => onPickFile(event.target.files[0])} />
        </div>

        <div className="pv-vf-field-group">
          <div className="pv-vf-field">
            <label className="pv-vf-field-label">강의 제목</label>
            <input className="pv-vf-field-input" value={title} onChange={event => onTitleChange(event.target.value)} />
          </div>
        </div>

        <div className="pv-vf-preview-submit-row">
          <button className="pv-vf-submit-btn pv-vf-submit-btn--dark" onClick={() => onStart('direct_upload')} disabled={!file}>
            바로 업로드
          </button>
          <button className="pv-vf-submit-btn" onClick={() => onStart('verified_upload')} disabled={!file}>
            검증 후 업로드
          </button>
        </div>
      </div>
    </div>
  )
}

function PipelineStep({ title, mode, phase, stageIndex, onCancel, onOpenReview }) {
  return (
    <div className="pv-vf-status-wrap">
      <div className="pv-vf-status-inner">
        <div className="pv-vf-status-title">{title}</div>
        <div className="pv-vf-status-label">
          {phase === PHASES.GRAPH ? 'graph pipeline' : mode === 'direct_upload' ? 'direct upload pipeline' : 'verifier pipeline'}
        </div>
        <PipelineProgress mode={mode} phase={phase} stageIndex={stageIndex} />
        <div className="pv-vf-status-actions">
          {phase === PHASES.VERIFY_READY && (
            <button className="pv-vf-confirm-btn" onClick={onOpenReview}>
              검증 결과 확인하기
            </button>
          )}
          <button className="pv-vf-cancel-btn" onClick={onCancel}>업로드 취소</button>
        </div>
      </div>
    </div>
  )
}

function ReviewClaim({ claim, expanded, onToggle, onWatch }) {
  return (
    <article className={`pv-vf-claim-card ${expanded ? 'pv-vf-claim-card--expanded' : ''}`}>
      <div className="pv-vf-claim-summary pv-vf-claim-summary--watch">
        <button className="pv-vf-watch-btn" onClick={onWatch} title="영상 보기" aria-label="영상 보기">
          <span aria-hidden="true">▶</span>
        </button>
        <button className="pv-vf-claim-main" onClick={onToggle}>
          <div className="pv-vf-chip-row">
            <span className={`pv-vf-chip pv-vf-chip--${claim.type}`}>{claim.typeLabel}</span>
            <span className="pv-vf-chip pv-vf-chip--score pv-vf-chip--score-inconclusive">검토 필요</span>
          </div>
          <div className="pv-vf-claim-copy">
            <div className="pv-vf-claim-headline">
              <span className="pv-vf-claim-time">{formatTime(claim.time)}</span>
              <span className="pv-vf-claim-title">{claim.title}</span>
            </div>
          </div>
          <span className={`pv-vf-claim-toggle ${expanded ? 'pv-vf-claim-toggle--open' : ''}`} aria-hidden="true" />
        </button>
      </div>
      {expanded && (
        <div className="pv-vf-claim-detail">
          <dl>
            <div className="pv-vf-detail-row">
              <dt>검증 내용</dt>
              <dd>{claim.detail}</dd>
            </div>
            <div className="pv-vf-detail-row">
              <dt>Preview</dt>
              <dd>로컬 상태만 사용하는 가짜 verifier 결과입니다.</dd>
            </div>
          </dl>
        </div>
      )}
    </article>
  )
}

function ReviewPanel({ lecture, onBack, onApprove, onReject }) {
  const [activeTab, setActiveTab] = useState('review')
  const [expandedClaim, setExpandedClaim] = useState('')
  const [videoOpen, setVideoOpen] = useState(false)
  const [seekToSeconds, setSeekToSeconds] = useState(null)

  return (
    <div className="pv-vf-shell pv-vf-shell--preview-review">
      <div className="pv-vf-topbar">
        <div className="pv-vf-topbar-main">
          <button className="pv-vf-topbar-back" onClick={onBack}>← 이전으로</button>
          <div className="pv-vf-topbar-title">
            <strong>Verifier</strong>
          </div>
        </div>
      </div>

      <div className="pv-vf-body">
        <section className="pv-vf-review-panel">
          <div className="pv-vf-review-header">
            <div className="pv-vf-review-header-main">
              <div className="pv-vf-review-heading">
                <strong>{lecture.title}</strong>
                <span>검토 결과</span>
              </div>
              <div className="pv-vf-review-header-actions">
                <nav className="pv-vf-review-tabs" aria-label="검토 항목">
                  <button className={`pv-vf-review-tab pv-vf-review-tab--review ${activeTab === 'review' ? 'pv-vf-review-tab--active' : ''}`} onClick={() => setActiveTab('review')}>
                    <span>검토 필요</span>
                    <strong>{REVIEW_CLAIMS.length}</strong>
                  </button>
                  <button className={`pv-vf-review-tab pv-vf-review-tab--typos ${activeTab === 'typos' ? 'pv-vf-review-tab--active' : ''}`} onClick={() => setActiveTab('typos')}>
                    <span>슬라이드 오타</span>
                    <strong>{SLIDE_TYPOS.length}</strong>
                  </button>
                </nav>
                <button className="pv-vf-modal-secondary-btn" onClick={onReject}>업로드 거절</button>
                <button className="pv-vf-confirm-btn pv-vf-review-complete-btn" onClick={onApprove}>업로드 승인</button>
              </div>
            </div>
          </div>

          <div className="pv-vf-review-content">
            {videoOpen && (
              <section className="pv-vf-video-pane">
                <button className="pv-vf-video-close-btn" onClick={() => setVideoOpen(false)}>영상 닫기</button>
                <VideoPlayer
                  lecture={lecture}
                  scenes={[]}
                  currentScene={0}
                  seekTo={null}
                  seekToSeconds={seekToSeconds}
                  onSceneChange={() => {}}
                />
              </section>
            )}

            <div className="pv-vf-review-scroll">
              {activeTab === 'review' ? (
                <section className="pv-vf-section pv-vf-section--review">
                  <div className="pv-vf-section-head">
                    <h2>검토가 필요한 내용</h2>
                    <span>{REVIEW_CLAIMS.length}</span>
                  </div>
                  <div className="pv-vf-claim-list">
                    {REVIEW_CLAIMS.map(claim => (
                      <ReviewClaim
                        key={claim.id}
                        claim={claim}
                        expanded={expandedClaim === claim.id}
                        onToggle={() => setExpandedClaim(prev => prev === claim.id ? '' : claim.id)}
                        onWatch={() => {
                          setVideoOpen(true)
                          setSeekToSeconds({ seconds: claim.time, time: Date.now() })
                        }}
                      />
                    ))}
                  </div>
                </section>
              ) : (
                <section className="pv-vf-section pv-vf-section--typo">
                  <div className="pv-vf-section-head">
                    <h2>슬라이드 오타</h2>
                    <span>{SLIDE_TYPOS.length}</span>
                  </div>
                  <div className="pv-vf-typo-list">
                    {SLIDE_TYPOS.map(typo => (
                      <article className="pv-vf-typo-card" key={typo.id}>
                        <div className="pv-vf-typo-card-head">
                          <strong>slide {typo.slide}</strong>
                        </div>
                        <div className="pv-vf-typo-row">
                          <span>{typo.from}</span>
                          <em>→</em>
                          <strong>{typo.to}</strong>
                        </div>
                        <p>{typo.reason}</p>
                      </article>
                    ))}
                  </div>
                </section>
              )}
            </div>
          </div>
        </section>
      </div>
    </div>
  )
}

function DoneStep({ title, mode, onReset }) {
  return (
    <div className="pv-vf-status-wrap">
      <div className="pv-vf-status-inner">
        <div className="pv-vf-done-icon">✅</div>
        <div className="pv-vf-status-title">{title}</div>
        <PipelineProgress mode={mode} phase={PHASES.DONE} stageIndex={GRAPH_STAGES.length} />
        <button className="pv-vf-reset-btn" onClick={onReset}>새 강의 업로드</button>
      </div>
    </div>
  )
}

export default function PreviewPage() {
  const objectUrlRef = useRef('')
  const [file, setFile] = useState(SAMPLE_FILE)
  const [title, setTitle] = useState(DEFAULT_TITLE)
  const [dragOver, setDragOver] = useState(false)
  const [mode, setMode] = useState('verified_upload')
  const [phase, setPhase] = useState(PHASES.UPLOAD)
  const [stageIndex, setStageIndex] = useState(0)

  const lecture = useMemo(() => ({
    id: 'preview-lecture',
    title,
    video_url: file.url,
  }), [file.url, title])

  const isTimedPhase = phase === PHASES.PREPROCESS || phase === PHASES.GRAPH

  useEffect(() => {
    if (!isTimedPhase) return undefined
    const stages = stagesForPhase(mode, phase)
    const timer = window.setTimeout(() => {
      setStageIndex(current => {
        const next = current + 1
        if (next < stages.length) return next
        if (phase === PHASES.PREPROCESS) {
          setPhase(mode === 'verified_upload' ? PHASES.VERIFY_READY : PHASES.DONE)
          return 0
        }
        setPhase(PHASES.DONE)
        return 0
      })
    }, STEP_DELAY_MS)

    return () => window.clearTimeout(timer)
  }, [isTimedPhase, mode, phase, stageIndex])

  useEffect(() => {
    return () => {
      if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current)
    }
  }, [])

  function selectFile(nextFile) {
    if (!nextFile) return
    if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current)
    const url = URL.createObjectURL(nextFile)
    objectUrlRef.current = url
    setFile({ name: nextFile.name, size: nextFile.size, url })
    setTitle(prev => prev.trim() ? prev : nextFile.name.replace(/\.[^.]+$/, ''))
  }

  function resetSample() {
    if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current)
    objectUrlRef.current = ''
    setFile(SAMPLE_FILE)
    setTitle(DEFAULT_TITLE)
  }

  function start(nextMode) {
    setMode(nextMode)
    setStageIndex(0)
    setPhase(PHASES.PREPROCESS)
  }

  function reset() {
    setStageIndex(0)
    setPhase(PHASES.UPLOAD)
  }

  return (
    <div className="pv-vf-page">
      {phase === PHASES.UPLOAD && (
        <UploadStep
          file={file}
          title={title}
          dragOver={dragOver}
          onDrop={selectFile}
          onPickFile={selectFile}
          onTitleChange={setTitle}
          onStart={start}
          onResetSample={resetSample}
          setDragOver={setDragOver}
        />
      )}

      {(phase === PHASES.PREPROCESS || phase === PHASES.VERIFY_READY || phase === PHASES.GRAPH) && (
        <PipelineStep
          title={title}
          mode={mode}
          phase={phase}
          stageIndex={stageIndex}
          onCancel={reset}
          onOpenReview={() => setPhase(PHASES.REVIEW)}
        />
      )}

      {phase === PHASES.REVIEW && (
        <ReviewPanel
          lecture={lecture}
          onBack={() => setPhase(PHASES.VERIFY_READY)}
          onApprove={() => {
            setStageIndex(0)
            setPhase(PHASES.GRAPH)
          }}
          onReject={reset}
        />
      )}

      {phase === PHASES.DONE && (
        <DoneStep title={title} mode={mode} onReset={reset} />
      )}
    </div>
  )
}
