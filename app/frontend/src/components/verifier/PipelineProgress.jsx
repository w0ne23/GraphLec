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
  if (nodeId === 'verify_start') {
    return phase === PHASES.PIPELINE1 || phase === PHASES.VERIFY_READY || phase === PHASES.REVIEWED || phase === PHASES.UPLOAD_RESUME || phase === PHASES.PIPELINE2 || phase === PHASES.DONE
      ? 'done'
      : 'wait'
  }
  if (nodeId === 'verified') {
    if (phase === PHASES.VERIFY_READY) return 'run'
    if (phase === PHASES.REVIEWED || phase === PHASES.PIPELINE2 || phase === PHASES.DONE) return 'done'
    if (getStageStatus(stages, 'verifier_run') === 'done') return 'done'
    return 'wait'
  }
  if (nodeId === 'done') return phase === PHASES.DONE ? 'run' : 'wait'
  return 'wait'
}

function getNodeStatus(node, stages, phase) {
  if (node.type === 'major') return getMajorStatus(node.id, phase, stages)
  return summarizeStatuses((node.stages ?? []).map(stage => getStageStatus(stages, stage.key)))
}

function getActiveNode(flowNodes, stages, phase) {
  return flowNodes.find(node => getNodeStatus(node, stages, phase) === 'run')
}

function getLogNode(activeNode, flowNodes) {
  if (activeNode?.stages?.length) return activeNode
  if (activeNode?.id === 'verified') {
    return flowNodes.find(node => node.id === 'verifier_run')
  }
  if (activeNode?.id === 'done') {
    return flowNodes.find(node => node.id === 'recommender_index') || flowNodes.find(node => node.id === 'metadata')
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

export default function PipelineProgress({
  stages,
  phase,
  errorMessage,
  statusMessage,
  flowNodes = PIPELINE_FLOW_NODES,
  showDetails = true,
  compact = false,
}) {
  const activeNode = getActiveNode(flowNodes, stages, phase)
  const activeNodeIndex = flowNodes.findIndex(node => node.id === activeNode?.id)
  const logNode = getLogNode(activeNode, flowNodes)
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
    <div className={`vf-pipe${compact ? ' vf-pipe--compact' : ''}`}>
      {showDetails && (
        <>
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
        </>
      )}
      <div className="vf-flow" style={{ '--flow-count': flowNodes.length }}>
        {flowNodes.map(renderNode)}
      </div>
    </div>
  )
}
