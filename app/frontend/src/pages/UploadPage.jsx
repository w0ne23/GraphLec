import { useRef, useState } from 'react'
import { PHASES } from '../components/verifier/verifierConstants'
import { useUploadForm } from '../hooks/useUploadForm'
import { usePageTitle } from '../hooks/usePageTitle'

import '../styles/verifier.css'

function VerifierUploadStep({ flow }) {
  const fileRef = useRef()
  const [dragOver, setDragOver] = useState(false)
  const { actions } = flow
  const isVerify = flow.mode === 'verify'

  return (
    <div className="vf-upload-wrap">
      <div className="vf-upload-inner">
        <p className="vf-upload-sub">
          {isVerify ? '검증할 강의 영상을 업로드합니다.' : '공개 업로드할 강의 영상을 업로드합니다.'}
        </p>
        {flow.errorMessage && <p className="vf-upload-error">{flow.errorMessage}</p>}

        <div
          className={`vf-dropzone${dragOver ? ' vf-dropzone--drag' : ''}${flow.file ? ' vf-dropzone--file' : ''}`}
          onClick={() => fileRef.current?.click()}
          onDragOver={event => {
            event.preventDefault()
            setDragOver(true)
          }}
          onDragLeave={() => setDragOver(false)}
          onDrop={event => {
            event.preventDefault()
            setDragOver(false)
            actions.selectFile(event.dataTransfer.files[0])
          }}
        >
          <div className="vf-dropzone-icon">{flow.file ? '✅' : '🎬'}</div>
          {flow.file ? (
            <>
              <p className="vf-dropzone-text"><strong>{flow.file.name}</strong></p>
              <p className="vf-dropzone-sub">
                {flow.file.size ? `${(flow.file.size / 1024 / 1024).toFixed(1)} MB` : '실제 verification 결과 기반'}
              </p>
            </>
          ) : (
            <>
              <p className="vf-dropzone-text"><strong>드래그하거나 클릭</strong></p>
              <p className="vf-dropzone-sub">MP4, MOV, AVI · 최대 2GB</p>
            </>
          )}
          <input
            ref={fileRef}
            type="file"
            accept="video/*"
            style={{ display: 'none' }}
            onChange={event => actions.selectFile(event.target.files[0])}
          />
        </div>

        <div className="vf-field-group">
          <div className="vf-field">
            <label className="vf-field-label">강의 제목</label>
            <input
              className="vf-field-input"
              value={flow.title}
              onChange={event => actions.setTitle(event.target.value)}
              placeholder="제목을 입력하세요"
            />
          </div>
        </div>

        <button className="vf-submit-btn" onClick={actions.submit} disabled={!flow.file || flow.isSubmitting}>
          {flow.isSubmitting ? '처리 중' : isVerify ? '검증 시작' : '업로드 시작'}
        </button>
        <button className="vf-cancel-btn" onClick={actions.backToChoice} disabled={flow.isSubmitting}>
          이전
        </button>
      </div>
    </div>
  )
}

function VerifyChoiceStep({ flow }) {
  const { actions } = flow

  return (
    <div className="vf-choice-wrap">
      <div className="vf-choice-inner">
        <div className="vf-choice-head">
          <span>작업 선택</span>
          <h1>강의 처리 방식 선택</h1>
        </div>
        <div className="vf-choice-grid">
          <button className="vf-choice-card vf-choice-card--primary" onClick={actions.startVerify} disabled={flow.isSubmitting}>
            <span>검증</span>
            <strong>{flow.isSubmitting ? '검증 준비 중' : '검증하기'}</strong>
            <em>{flow.isSubmitting ? '검증 작업을 생성하고 있습니다' : '공개 업로드 없이 검증 보고서 생성'}</em>
          </button>
          <button className="vf-choice-card" onClick={actions.startPublish} disabled={flow.isSubmitting}>
            <span>업로드</span>
            <strong>업로드하기</strong>
            <em>검증 없이 공개 업로드 파이프라인 실행</em>
          </button>
        </div>
      </div>
    </div>
  )
}

function VerifierErrorStep({ flow }) {
  const { actions } = flow

  return (
    <div className="vf-status-wrap">
      <div className="vf-status-inner">
        <div className="vf-status-label vf-status-label--err">오류 발생</div>
        <div className="vf-status-title">작업을 시작하지 못했습니다.</div>
        {flow.errorMessage && <div className="vf-pipe-error">오류: {flow.errorMessage}</div>}
        <div className="vf-error-actions">
          <button className="vf-retry-btn" onClick={actions.retry}>다시 시도</button>
          <button className="vf-cancel-btn" onClick={actions.reset}>처음으로</button>
        </div>
      </div>
    </div>
  )
}

export default function UploadPage() {
  const flow = useUploadForm()
  usePageTitle(flow.phase === PHASES.VERIFY_CHOICE ? 'Upload' : flow.mode === 'verify' ? 'Verify' : 'Publish')

  return (
    <div className="vf-page">
      {flow.phase === PHASES.VERIFY_CHOICE && <VerifyChoiceStep flow={flow} />}
      {flow.phase === PHASES.UPLOAD && <VerifierUploadStep flow={flow} />}
      {flow.phase === PHASES.ERROR && <VerifierErrorStep flow={flow} />}
    </div>
  )
}
