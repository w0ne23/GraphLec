import { useEffect, useMemo, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { getLectureDetail, getLectureVerifier } from '../lib/api'
import VideoPlayer from '../components/watch/VideoPlayer'

function formatTime(seconds) {
  const safe = Number.isFinite(seconds) ? Math.max(0, seconds) : 0
  const m = Math.floor(safe / 60)
  const s = Math.floor(safe % 60)
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`
}

export default function VerifierPage() {
  const { id } = useParams()
  const navigate = useNavigate()

  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [lecture, setLecture] = useState(null)
  const [verifier, setVerifier] = useState(null)
  const [expandedClaimKey, setExpandedClaimKey] = useState('')
  const [isVideoMode, setIsVideoMode] = useState(false)
  const [seekToSeconds, setSeekToSeconds] = useState(null)

  useEffect(() => {
    if (!id) return

    setLoading(true)
    setError('')

    Promise.all([getLectureDetail(id), getLectureVerifier(id)])
      .then(([detail, verifierResult]) => {
        setLecture(detail)
        setVerifier(verifierResult)
      })
      .catch((err) => {
        console.error('Verifier fetch error:', err)
        setError(String(err?.message || err))
      })
      .finally(() => setLoading(false))
  }, [id])

  const finalClaims = useMemo(
    () => verifier?.final_confirmed_claims ?? [],
    [verifier]
  )

  const finalCount = verifier?.final_confirmed_claim_count ?? finalClaims.length

  function toggleClaim(claimKey) {
    setExpandedClaimKey((prev) => (prev === claimKey ? '' : claimKey))
  }

  function handleWatchClaim(startTime) {
    setIsVideoMode(true)
    setSeekToSeconds(Number(startTime) || 0)
  }

  if (loading) return <div className="lp-loading">불러오는 중...</div>

  if (error) {
    return (
      <div className="vf-shell">
        <div className="vf-topbar">
          <button className="lp-back" onClick={() => navigate(-1)}>← 이전으로</button>
          <span className="lp-title">Verifier</span>
        </div>
        <div className="vf-error">{error}</div>
      </div>
    )
  }

  if (!lecture || !verifier) {
    return <div className="lp-loading">Verifier 결과를 찾을 수 없습니다</div>
  }

  return (
    <div className="vf-shell">
      <div className="vf-topbar">
        <div className="vf-topbar-left">
          <button className="lp-back" onClick={() => navigate(-1)}>← 이전으로</button>
          <span className="lp-title">{lecture.title} · Verifier</span>
        </div>
        {isVideoMode && (
          <button className="vf-exit-video-btn" onClick={() => setIsVideoMode(false)}>
            영상 닫기
          </button>
        )}
      </div>

      <div className={`vf-body ${isVideoMode ? 'vf-body--video' : ''}`}>
        {isVideoMode && (
          <section className="vf-video-pane">
            <VideoPlayer
              lecture={lecture}
              scenes={[]}
              currentScene={0}
              onSceneChange={() => {}}
              seekTo={null}
              seekToSeconds={seekToSeconds}
            />
          </section>
        )}

        <section className="vf-list-pane">
          <div className="vf-summary-card">
            <div className="vf-summary-label">최종 확정된 클레임 수</div>
            <div className="vf-summary-value">{finalCount}</div>
          </div>

          <div className="vf-claim-list">
            {finalClaims.length === 0 && (
              <div className="vf-empty">최종 확정된 클레임이 없습니다.</div>
            )}

            {finalClaims.map((claim, idx) => {
              const claimKey = `${claim.utterance_id || 'claim'}-${idx}`
              const expanded = expandedClaimKey === claimKey
              return (
                <article
                  key={claimKey}
                  className={`vf-claim-card ${expanded ? 'vf-claim-card--expanded' : ''}`}
                >
                  <button
                    className="vf-claim-main"
                    onClick={() => toggleClaim(claimKey)}
                  >
                    <div className="vf-claim-title">{claim.claim_text || claim.resolved_claim || '-'}</div>
                    <div className="vf-claim-meta">
                      <span>{claim.utterance_id || '-'}</span>
                      <span>{formatTime(claim.start_time)}</span>
                    </div>
                  </button>

                  {expanded && (
                    <div className="vf-claim-detail">
                      <div><strong>등장 시각:</strong> {formatTime(claim.start_time)} ({Number(claim.start_time || 0).toFixed(2)}s)</div>
                      <div><strong>Issue:</strong> {claim.issue || '-'}</div>
                      <div><strong>Correct Info:</strong> {claim.correct_info || '-'}</div>
                      <div><strong>Slide:</strong> {claim.slide_number ?? '-'}</div>
                    </div>
                  )}

                  <div className="vf-claim-actions">
                    <button
                      className="vf-watch-btn"
                      onClick={() => handleWatchClaim(claim.start_time)}
                    >
                      영상 보기
                    </button>
                  </div>
                </article>
              )
            })}
          </div>
        </section>
      </div>
    </div>
  )
}
