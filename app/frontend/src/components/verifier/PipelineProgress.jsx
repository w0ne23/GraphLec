import {
  PHASES,
  PIPELINE_FLOW_NODES,
} from './verifierConstants'

function getStageStatus(stages, key) {
  return stages.find(s => s.stage === key)?.status ?? 'wait'
}

function summarizeStatuses(statuses) {
  if (statuses.some(status => status === 'run')) return 'run'
  if (statuses.length > 0 && statuses.every(status => status === 'done')) return 'done'
  return 'wait'
}

function getMajorStatus(nodeId, phase, stages) {
  if (nodeId === 'upload') return phase === PHASES.UPLOAD ? 'run' : 'done'
  if (nodeId === 'verified') {
    if (phase === PHASES.VERIFY_READY) return 'run'
    if (phase === PHASES.REVIEWED || phase === PHASES.PIPELINE2 || phase === PHASES.DONE) return 'done'
    if (getStageStatus(stages, 'stage10_run_analyzers') === 'done') return 'done'
    return 'wait'
  }
  if (nodeId === 'done') return phase === PHASES.DONE ? 'run' : 'wait'
  return 'wait'
}

function getNodeStatus(node, stages, phase) {
  if (node.type === 'major') return getMajorStatus(node.id, phase, stages)
  return summarizeStatuses((node.stages ?? []).map(stage => getStageStatus(stages, stage.key)))
}

function getActiveNode(stages, phase) {
  return PIPELINE_FLOW_NODES.find(node => getNodeStatus(node, stages, phase) === 'run')
}

function getLogNode(activeNode) {
  if (activeNode?.stages?.length) return activeNode
  if (activeNode?.id === 'verified') {
    return PIPELINE_FLOW_NODES.find(node => node.id === 'content_verify')
  }
  if (activeNode?.id === 'done') {
    return PIPELINE_FLOW_NODES.find(node => node.id === 'metadata')
  }
  return activeNode
}

function getStageText(status) {
  if (status === 'done') return '완료!'
  if (status === 'run') return '진행 중...'
  return '대기 중'
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

export default function PipelineProgress({ stages, phase, errorMessage, statusMessage }) {
  const activeNode = getActiveNode(stages, phase)
  const activeNodeIndex = PIPELINE_FLOW_NODES.findIndex(node => node.id === activeNode?.id)
  const logNode = getLogNode(activeNode)
  const visibleStages = getVisibleLogStages(logNode, stages)
  const showErrorInLog = phase === PHASES.ERROR && errorMessage

  function renderNode(node, nodeIndex) {
    const rawNodeStatus = getNodeStatus(node, stages, phase)
    const nodeStatus = activeNodeIndex >= 0 && nodeIndex > activeNodeIndex ? 'wait' : rawNodeStatus
    const classes = [
      'vf-flow-item',
      `vf-flow-item--${node.type}`,
      `vf-flow-item--${nodeStatus}`,
    ].join(' ')

    return (
      <div key={node.id} className={classes}>
        <div className="vf-flow-node-slot">
          <div className="vf-flow-node" aria-label={`${node.label} ${nodeStatus}`} />
        </div>
        <div className="vf-flow-label">{node.label}</div>
      </div>
    )
  }

  return (
    <div className="vf-pipe">
      <div className="vf-progress-head">
        <div className="vf-progress-message">{statusMessage || '분석 준비 중...'}</div>
      </div>
      <div className="vf-work-log">
        {logNode ? (
          <div className="vf-work-log-lines">
            {visibleStages.map(stage => {
              const status = getStageStatus(stages, stage.key)
              return (
                <div key={stage.key} className={`vf-work-log-line vf-work-log-line--${status}`}>
                  <span>{stage.label}</span>
                  <em>{getStageText(status)}</em>
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
      <div className="vf-flow" style={{ '--flow-count': PIPELINE_FLOW_NODES.length }}>
        {PIPELINE_FLOW_NODES.map(renderNode)}
      </div>
    </div>
  )
}
