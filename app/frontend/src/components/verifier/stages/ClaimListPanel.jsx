import {
  statusText,
  pendingText,
  asArray,
  jumpToClaimRow,
} from '../verifierUtils'
import {
  EmptyFiltered,
} from '../VerifyReportParts'
import TranscriptSourcePanel from './TranscriptSourcePanel'

const CLAIM_LIST_STAGE_COPY = {
  claim_extraction: {
    title: '주장 추출',
    pendingRunningText: '주장 추출 결과가 생성되는 중입니다.',
    pendingWaitingText: '추출된 주장이 없습니다.',
  },
  issue_judge: {
    title: '이슈 후보 판단',
    pendingRunningText: '이슈 판단 결과가 생성되는 중입니다.',
    pendingWaitingText: '판단된 이슈 후보가 없습니다.',
  },
  issue_classification: {
    title: '이슈 유형 분류',
    pendingRunningText: '이슈 분류 결과가 생성되는 중입니다.',
    pendingWaitingText: '분류된 이슈 유형이 없습니다.',
  },
  final_verification: {
    title: '이슈 확정',
    pendingRunningText: '멀티 LLM 검증 결과가 생성되는 중입니다.',
    pendingWaitingText: '진행된 멀티 LLM 검증 결과가 없습니다.',
  },
}

export default function ClaimListPanel({
  activeTab,
  actions = null,
  children,
  countLabel,
  model,
  pendingRunningText,
  pendingWaitingText,
  resultId,
  rows,
  sourceRows,
  status,
  title,
  titleAddon = null,
  viewMode = 'list',
  onViewModeChange,
}) {
  const hasSourceRows = asArray(sourceRows).length > 0
  const canShowTranscript = asArray(model.transcriptScenes).length > 0 || asArray(model.transcriptEntries).length > 0
  const stageCopy = CLAIM_LIST_STAGE_COPY[activeTab] || {}
  const panelTitle = title || stageCopy.title || '클레임 목록'
  const runningText = pendingRunningText || stageCopy.pendingRunningText || '결과가 생성되는 중입니다.'
  const waitingText = pendingWaitingText || stageCopy.pendingWaitingText || '표시할 클레임이 없습니다.'

  const handleTranscriptClaimClick = claimId => {
    onViewModeChange?.('list')
    if (typeof window !== 'undefined') window.setTimeout(() => jumpToClaimRow(claimId), 0)
  }

  if (!hasSourceRows) {
    return (
      <section className={`vf-claim-flow vf-claim-flow--${status}`}>
        <div className="vf-claim-flow-head">
          <div>
            <span>클레임 목록</span>
            <div className="vf-claim-flow-title-row">
              <h2>{panelTitle}</h2>
              {titleAddon}
            </div>
          </div>
          <span className="vf-bold">{statusText(status)}</span>
        </div>
        <div className="vf-stage-pending">
          {pendingText(status, runningText, waitingText)}
        </div>
      </section>
    )
  }

  return (
    <section className={`vf-claim-flow vf-claim-flow--${status}`}>
      <div className="vf-claim-flow-head">
        <div className="vf-claim-flow-title">
          <span>클레임 목록</span>
          <div className="vf-claim-flow-title-row">
            <h2>{panelTitle}</h2>
            {titleAddon}
          </div>
        </div>
        <div className="vf-claim-flow-head-actions">
          {canShowTranscript && (
            <div className="vf-claim-view-switch vf-claim-view-switch--inline">
              <button
                type="button"
                className={viewMode === 'transcript' ? 'vf-claim-view-switch-btn--active' : ''}
                aria-pressed={viewMode === 'transcript'}
                onClick={() => onViewModeChange?.(viewMode === 'transcript' ? 'list' : 'transcript')}
              >
                {viewMode === 'transcript' ? '목록에서 보기' : '내용에서 보기'}
              </button>
            </div>
          )}
          {actions}
          <span className="vf-bold vf-claim-flow-count">{countLabel}</span>
        </div>
      </div>

      {viewMode === 'transcript' && canShowTranscript ? (
        <TranscriptSourcePanel
          model={model}
          resultId={resultId}
          rows={rows}
          activeTab={activeTab}
          onClaimClick={handleTranscriptClaimClick}
        />
      ) : asArray(rows).length ? (
        children
      ) : (
        <EmptyFiltered />
      )}
    </section>
  )
}
