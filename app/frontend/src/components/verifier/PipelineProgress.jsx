import {
  PHASES,
  PIPELINE_FLOW_NODES,
} from './verifierConstants'

const NODE_TYPE_CLASS = {
  major: 'vf-flow-item--major',
  minor: '',
}

const NODE_STATUS_CLASS = {
  done: 'vf-flow-item--done',
  prior: 'vf-flow-item--prior',
  run: 'vf-flow-item--run',
  wait: '',
}

const CONNECTOR_COMPLETE_STATUSES = new Set(['done', 'prior', 'run'])

const WORK_LOG_STATUS_CLASS = {
  done: '',
  prior: '',
  run: 'vf-work-log-line--run',
  wait: '',
}

function cx(...classNames) {
  return classNames.filter(Boolean).join(' ')
}

function getStageStatus(stages, key) {
  return stages.find(s => s.stage === key)?.status ?? 'wait'
}

function summarizeStatuses(statuses) {
  if (statuses.some(status => status === 'run')) return 'run'
  if (statuses.length > 0 && statuses.every(status => status === 'done')) return 'done'
  if (statuses.length > 0 && statuses.every(status => status === 'prior')) return 'prior'
  return 'wait'
}

function getMajorStatus(nodeId, phase, stages) {
  if (nodeId === 'upload') return phase === PHASES.UPLOAD ? 'run' : 'done'
  if (nodeId === 'verify_start') {
    return phase === PHASES.PIPELINE1 || phase === PHASES.VERIFY_READY || phase === PHASES.REVIEWED
      ? 'done'
      : 'wait'
  }
  if (nodeId === 'verified') {
    if (phase === PHASES.VERIFY_READY) return 'run'
    if (phase === PHASES.REVIEWED || phase === PHASES.PIPELINE2 || phase === PHASES.DONE) return 'done'
    if (getStageStatus(stages, 'verify_slide_errors') === 'done') return 'done'
    if (getStageStatus(stages, 'verifier_run') === 'done') return 'done'
    return 'wait'
  }
  if (nodeId === 'done') return phase === PHASES.DONE ? 'run' : 'wait'
  return 'wait'
}

function getNodeStatus(node, stages, phase, priorNodeIds = new Set()) {
  if (priorNodeIds.has(node.id)) return 'prior'
  if (node.type === 'major') return getMajorStatus(node.id, phase, stages)
  return summarizeStatuses((node.stages ?? []).map(stage => getStageStatus(stages, stage.key)))
}

function getActiveNode(flowNodes, stages, phase, priorNodeIds) {
  return flowNodes.find(node => getNodeStatus(node, stages, phase, priorNodeIds) === 'run')
}

function getRenderedNodeStatus(node, nodeIndex, activeNodeIndex, stages, phase, priorNodeIds) {
  const rawNodeStatus = getNodeStatus(node, stages, phase, priorNodeIds)
  if (activeNodeIndex >= 0 && nodeIndex > activeNodeIndex) return 'wait'
  return rawNodeStatus
}

function getLogNode(activeNode, flowNodes) {
  if (activeNode?.stages?.length) return activeNode
  if (activeNode?.id === 'verified') {
    return flowNodes.find(node => node.id === 'slide_review')
  }
  if (activeNode?.id === 'done') {
    return flowNodes.find(node => node.id === 'metadata')
  }
  return activeNode
}

function getStageText(status) {
  if (status === 'done') return '완료!'
  if (status === 'prior') return '이전 완료'
  if (status === 'run') return '진행 중...'
  return '대기 중'
}

function getStageLabel(flowNodes, stageKey) {
  for (const node of flowNodes) {
    const stage = node.stages?.find(item => item.key === stageKey)
    if (stage) return stage.label
  }
  return stageKey
}

function getStatusMessageText(message, flowNodes) {
  const raw = String(message || '').trim()
  if (!raw) return '분석 준비 중...'

  if (raw === 'Starting pipeline') return '파이프라인을 시작합니다.'
  if (raw === 'Finished') return '분석이 완료되었습니다.'
  if (raw === 'Waiting approval' || raw === 'Waiting for approval') return '검증 결과 확인 대기 중'

  const processing = raw.match(/^Processing\s+(.+?)(?:\.\.\.)?$/)
  if (processing) return `${getStageLabel(flowNodes, processing[1])} 진행 중`

  const finished = raw.match(/^Finished\s+(.+?)$/)
  if (finished) return `${getStageLabel(flowNodes, finished[1])} 완료`

  return raw
}

function getNodeStatusText(status) {
  if (status === 'done') return '완료'
  if (status === 'prior') return '이전 완료'
  if (status === 'run') return '현재 단계'
  return '대기'
}

function getVisibleLogStages(node, stages) {
  const nodeStages = node?.stages ?? []
  const lastRunningIndex = nodeStages.reduce((lastIndex, stage, index) => (
    getStageStatus(stages, stage.key) === 'run' ? index : lastIndex
  ), -1)
  if (lastRunningIndex >= 0) return nodeStages.slice(0, lastRunningIndex + 1)

  const currentIndex = nodeStages.findIndex(stage => getStageStatus(stages, stage.key) !== 'done')
  if (currentIndex < 0) return nodeStages
  return nodeStages.slice(0, currentIndex + 1)
}

export default function PipelineProgress({
  stages,
  phase,
  errorMessage,
  statusMessage,
  flowNodes = PIPELINE_FLOW_NODES,
  priorNodeIds = [],
}) {
  const priorNodeIdSet = new Set(priorNodeIds)
  const activeNode = getActiveNode(flowNodes, stages, phase, priorNodeIdSet)
  const activeNodeIndex = flowNodes.findIndex(node => node.id === activeNode?.id)
  const logNode = getLogNode(activeNode, flowNodes)
  const visibleStages = getVisibleLogStages(logNode, stages)
  const showErrorInLog = phase === PHASES.ERROR && errorMessage
  const displayStatusMessage = getStatusMessageText(statusMessage, flowNodes)

  function renderNode(node, nodeIndex) {
    const nodeStatus = getRenderedNodeStatus(node, nodeIndex, activeNodeIndex, stages, phase, priorNodeIdSet)
    const nextNode = flowNodes[nodeIndex + 1]
    const nextNodeStatus = nextNode
      ? getRenderedNodeStatus(nextNode, nodeIndex + 1, activeNodeIndex, stages, phase, priorNodeIdSet)
      : 'wait'
    const connectorComplete =
      CONNECTOR_COMPLETE_STATUSES.has(nodeStatus) &&
      CONNECTOR_COMPLETE_STATUSES.has(nextNodeStatus)
    const classes = cx(
      'vf-flow-item',
      NODE_TYPE_CLASS[node.type],
      NODE_STATUS_CLASS[nodeStatus],
      connectorComplete && 'vf-flow-item--connector-complete',
    )

    return (
      <li
        key={node.id}
        className={classes}
        aria-current={nodeStatus === 'run' ? 'step' : undefined}
        aria-label={`${nodeIndex + 1}단계 ${node.label}: ${getNodeStatusText(nodeStatus)}`}
      >
        <div className="vf-flow-node-slot">
          <span className="vf-flow-node" aria-hidden="true" />
        </div>
        <div className="vf-flow-label">{node.label}</div>
      </li>
    )
  }

  return (
    <div className="vf-pipe">
      <div className="vf-progress-head">
        <div className="vf-progress-message">{displayStatusMessage}</div>
      </div>
      <div className="vf-work-log">
        {logNode ? (
          <div className="vf-work-log-lines">
            {visibleStages.map(stage => {
              const status = getStageStatus(stages, stage.key)
              return (
                <div key={stage.key} className={cx('vf-work-log-line', WORK_LOG_STATUS_CLASS[status])}>
                  <span>{stage.label}</span>
                  <span className="vf-work-log-line-status">{getStageText(status)}</span>
                </div>
              )
            })}
          </div>
        ) : (
          !showErrorInLog && (
            <div className="vf-work-log-empty">
              {activeNode?.id === 'verified' ? '검토 대기 중' : '작업 대기 중'}
            </div>
          )
        )}
        {showErrorInLog && (
          <div className="vf-pipe-error">오류: {errorMessage}</div>
        )}
      </div>
      <ol
        className="vf-flow"
        style={{ '--flow-count': flowNodes.length }}
        aria-label="파이프라인 단계"
      >
        {flowNodes.map(renderNode)}
      </ol>
    </div>
  )
}
