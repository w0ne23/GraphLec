import {
  PIPELINE_FLOW_NODES,
} from './verifierConstants'

const NODE_STATUS_CLASS = {
  done: 'vf-flow-item--done',
  prior: 'vf-flow-item--prior',
  run: 'vf-flow-item--run',
  wait: '',
}

const CONNECTOR_COMPLETE_STATUSES = new Set(['done', 'prior', 'run'])
const NODE_POSITION_PRECISION = 4
const ACTIVE_NODE_WEIGHT = 1.75
const DEFAULT_NODE_WEIGHT = 1
const MIN_CUT_SPACER_WEIGHT = 1.5
const CUT_LINE_RATIO = 2 / 3
const FLOW_ROW_BREAK_AFTER_NODE_IDS = ['verifier_data', 'fusion']
const FLOW_UNIT_LABELS = {
  verifier_data: {
    firstRow: '전처리',
    secondRow: '검증',
  },
  fusion: {
    firstRow: '전처리',
    secondRow: '그래프화',
  },
}
const FLOW_NODE_UNIT = {
  data_extract: 'preprocess',
  content_extract: 'preprocess',
  context_analysis: 'preprocess',
  verifier_data: 'preprocess',
  structure: 'preprocess',
  fusion: 'preprocess',
  claim_extraction: 'verify',
  issue_judge: 'verify',
  issue_classification: 'verify',
  final_verification: 'verify',
  slide_review: 'verify',
  graph_build: 'graph',
  graph_lance_index: 'graph',
  graph_graphrag_index: 'graph',
  graph_metadata: 'graph',
  graph_recommender_index: 'graph',
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

function getNodeStatus(node, stages, priorNodeIds = new Set()) {
  if (priorNodeIds.has(node.id)) return 'prior'
  return summarizeStatuses((node.stages ?? []).map(stage => getStageStatus(stages, stage.key)))
}

function getActiveNode(flowNodes, stages, priorNodeIds) {
  return flowNodes.find(node => getNodeStatus(node, stages, priorNodeIds) === 'run')
}

function getRenderedNodeStatus(node, nodeIndex, activeNodeIndex, stages, priorNodeIds) {
  const rawNodeStatus = getNodeStatus(node, stages, priorNodeIds)
  if (activeNodeIndex >= 0 && nodeIndex > activeNodeIndex) return 'wait'
  return rawNodeStatus
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

function getFlowBreakNodeId(flowNodes) {
  const breakIndex = flowNodes.findIndex((node, index) => (
    index < flowNodes.length - 1 && FLOW_ROW_BREAK_AFTER_NODE_IDS.includes(node.id)
  ))
  return breakIndex < 0 ? null : flowNodes[breakIndex].id
}

function splitFlowRows(flowNodes, breakNodeId) {
  const breakIndex = flowNodes.findIndex(node => node.id === breakNodeId)
  if (breakIndex < 0) return [flowNodes]
  return [
    flowNodes.slice(0, breakIndex + 1),
    flowNodes.slice(breakIndex + 1),
  ]
}

function toPercent(value) {
  return `${(value * 100).toFixed(NODE_POSITION_PRECISION)}%`
}

function getConnectorStatus(connector, nodeIndexById, flowNodes, stages, activeNodeIndex, priorNodeIds) {
  const fromIndex = nodeIndexById.get(connector.from)
  const toIndex = nodeIndexById.get(connector.to)
  const fromStatus = fromIndex == null
    ? null
    : getRenderedNodeStatus(flowNodes[fromIndex], fromIndex, activeNodeIndex, stages, priorNodeIds)
  const toStatus = toIndex == null
    ? null
    : getRenderedNodeStatus(flowNodes[toIndex], toIndex, activeNodeIndex, stages, priorNodeIds)
  return {
    fromOn: toStatus != null && CONNECTOR_COMPLETE_STATUSES.has(toStatus),
    toOn: toStatus != null && CONNECTOR_COMPLETE_STATUSES.has(toStatus),
    unit: getConnectorUnit(connector),
  }
}

function createFlowRows(flowNodes, breakNodeId) {
  if (!breakNodeId) {
    return [{
      id: 'single',
      label: '',
      nodes: flowNodes,
      cut: null,
    }]
  }

  const [firstRowNodes, secondRowNodes] = splitFlowRows(flowNodes, breakNodeId)
  const unitLabels = FLOW_UNIT_LABELS[breakNodeId] ?? {}
  const firstSecondRowNode = secondRowNodes[0]

  return [
    {
      id: `${breakNodeId}-first-row`,
      label: unitLabels.firstRow ?? '',
      nodes: firstRowNodes,
      cut: 'end',
      cutTarget: firstSecondRowNode?.id ?? null,
    },
    {
      id: `${breakNodeId}-second-row`,
      label: unitLabels.secondRow ?? '',
      nodes: secondRowNodes,
      cut: 'start',
    },
  ]
}

function getConnectorUnit(connector) {
  if (connector.variant === 'cut-start') return FLOW_NODE_UNIT[connector.to]
  return FLOW_NODE_UNIT[connector.from]
}

function getSlotWeight(slot, activeNodeId) {
  if (slot.type === 'spacer') return slot.weight
  const { node } = slot
  return node.id === activeNodeId ? ACTIVE_NODE_WEIGHT : DEFAULT_NODE_WEIGHT
}

function getTargetBaseRowWeight(rows) {
  return Math.max(...rows.map(row => (
    row.nodes.length + (row.cut ? MIN_CUT_SPACER_WEIGHT : 0)
  )))
}

function createRowSlots(row, targetBaseRowWeight) {
  const nodeSlots = row.nodes.map(node => ({
    id: node.id,
    type: 'node',
    node,
  }))
  const spacerWeight = row.cut
    ? Math.max(MIN_CUT_SPACER_WEIGHT, targetBaseRowWeight - row.nodes.length)
    : 0

  if (row.cut === 'start') {
    return [
      { id: `${row.id}-cut-spacer-start`, type: 'spacer', weight: spacerWeight },
      ...nodeSlots,
    ]
  }

  if (row.cut === 'end') {
    return [
      ...nodeSlots,
      { id: `${row.id}-cut-spacer-end`, type: 'spacer', weight: spacerWeight },
    ]
  }

  return nodeSlots
}

function getSlotCenterPositions(slots, activeNodeId) {
  const weights = slots.map(slot => getSlotWeight(slot, activeNodeId))
  const totalWeight = weights.reduce((sum, weight) => sum + weight, 0)
  let cursor = 0

  return slots.map((slot, index) => {
    const weight = weights[index]
    const center = (cursor + (weight / 2)) / totalWeight
    cursor += weight
    return center
  })
}

function getNodeCenterPositions(nodes, activeNodeId, start, end) {
  if (nodes.length <= 1) return [start]

  const nodeSlots = nodes.map(node => ({
    id: node.id,
    type: 'node',
    node,
  }))
  const positions = getSlotCenterPositions(nodeSlots, activeNodeId)
  const first = positions[0]
  const last = positions[positions.length - 1]

  return positions.map(position => (
    start + ((position - first) / (last - first)) * (end - start)
  ))
}

function getNodeWidthPositions(positions) {
  if (positions.length <= 1) return [null]

  return positions.map((position, index) => {
    const prevPosition = positions[index - 1]
    const nextPosition = positions[index + 1]
    if (prevPosition == null) return nextPosition - position
    if (nextPosition == null) return position - prevPosition
    const left = (prevPosition + position) / 2
    const right = (position + nextPosition) / 2
    return Math.max(right - left, 0)
  })
}

function getRowLayout(row, activeNodeId, targetBaseRowWeight) {
  const slots = createRowSlots(row, targetBaseRowWeight)
  const basePositions = getSlotCenterPositions(slots, null)
  const nodePositionsById = new Map()
  const baseNodePositionsById = new Map()
  const nodeWidthsById = new Map()
  const baseActualNodePositions = []

  slots.forEach((slot, index) => {
    if (slot.type !== 'node') return
    baseActualNodePositions.push(basePositions[index])
    baseNodePositionsById.set(slot.node.id, basePositions[index])
  })

  const firstBasePosition = baseActualNodePositions[0] ?? 0
  const lastBasePosition = baseActualNodePositions[baseActualNodePositions.length - 1] ?? firstBasePosition
  const actualNodePositions = getNodeCenterPositions(
    row.nodes,
    activeNodeId,
    firstBasePosition,
    lastBasePosition,
  )
  const nodeWidths = getNodeWidthPositions(actualNodePositions)

  row.nodes.forEach((node, index) => {
    nodePositionsById.set(node.id, actualNodePositions[index])
    nodeWidthsById.set(node.id, nodeWidths[index])
  })

  return { slots, nodePositionsById, baseNodePositionsById, nodeWidthsById }
}

function createRowConnectors(row, layout) {
  const connectors = row.nodes.slice(0, -1).map((node, index) => ({
    id: `${node.id}-${row.nodes[index + 1].id}`,
    from: node.id,
    to: row.nodes[index + 1].id,
    left: layout.nodePositionsById.get(node.id),
    width: layout.nodePositionsById.get(row.nodes[index + 1].id) - layout.nodePositionsById.get(node.id),
    variant: 'normal',
  }))

  if (row.cut === 'end') {
    const lastNode = row.nodes[row.nodes.length - 1]
    const prevNode = row.nodes[row.nodes.length - 2]
    const nodePosition = layout.nodePositionsById.get(lastNode.id)
    const baseNodePosition = layout.baseNodePositionsById.get(lastNode.id)
    const prevNodePosition = prevNode ? layout.baseNodePositionsById.get(prevNode.id) : null
    const cutWidth = prevNodePosition == null
      ? 0
      : (baseNodePosition - prevNodePosition) * CUT_LINE_RATIO

    connectors.push({
      id: `${lastNode.id}-cut-end`,
      from: lastNode.id,
      to: row.cutTarget,
      left: nodePosition,
      width: cutWidth,
      variant: 'cut-end',
    })
  }

  if (row.cut === 'start') {
    const firstNode = row.nodes[0]
    const nextNode = row.nodes[1]
    const nodePosition = layout.nodePositionsById.get(firstNode.id)
    const baseNodePosition = layout.baseNodePositionsById.get(firstNode.id)
    const nextNodePosition = nextNode ? layout.baseNodePositionsById.get(nextNode.id) : null
    const cutWidth = nextNodePosition == null
      ? 0
      : (nextNodePosition - baseNodePosition) * CUT_LINE_RATIO

    connectors.unshift({
      id: `${firstNode.id}-cut-start`,
      from: null,
      to: firstNode.id,
      left: nodePosition - cutWidth,
      width: cutWidth,
      variant: 'cut-start',
    })
  }

  return connectors
}

export default function PipelineProgress({
  stages,
  statusMessage,
  flowNodes = PIPELINE_FLOW_NODES,
  priorNodeIds = [],
}) {
  const priorNodeIdSet = new Set(priorNodeIds)
  const activeNode = getActiveNode(flowNodes, stages, priorNodeIdSet)
  const activeNodeIndex = flowNodes.findIndex(node => node.id === activeNode?.id)
  const displayStatusMessage = getStatusMessageText(statusMessage, flowNodes)
  const breakNodeId = getFlowBreakNodeId(flowNodes)
  const flowRows = createFlowRows(flowNodes, breakNodeId)
  const targetBaseRowWeight = getTargetBaseRowWeight(flowRows)
  const nodeIndexById = new Map(flowNodes.map((node, nodeIndex) => [node.id, nodeIndex]))

  function renderUnitLabel(row) {
    return (
      <div
        key={`${row.id}-label`}
        className="vf-flow-unit-label"
        aria-hidden="true"
      >
        {row.label}
      </div>
    )
  }

  function renderConnector(connector) {
    const connectorStatus = getConnectorStatus(
      connector,
      nodeIndexById,
      flowNodes,
      stages,
      activeNodeIndex,
      priorNodeIdSet,
    )
    const classes = cx(
      'vf-flow-connector',
      connector.variant === 'cut-end' && 'vf-flow-connector--cut-end',
      connector.variant === 'cut-start' && 'vf-flow-connector--cut-start',
      connectorStatus?.unit && `vf-flow-connector--${connectorStatus.unit}`,
      connectorStatus?.fromOn && 'vf-flow-connector--from-on',
      connectorStatus?.toOn && 'vf-flow-connector--to-on',
    )

    return (
      <span
        key={connector.id}
        className={classes}
        style={{
          left: toPercent(connector.left),
          width: toPercent(connector.width),
        }}
        aria-hidden="true"
      />
    )
  }

  function renderNode(node, layout) {
    const nodeIndex = nodeIndexById.get(node.id)
    const nodeStatus = getRenderedNodeStatus(node, nodeIndex, activeNodeIndex, stages, priorNodeIdSet)
    const classes = cx(
      'vf-flow-item',
      NODE_STATUS_CLASS[nodeStatus],
    )

    return (
      <div
        key={node.id}
        className={classes}
        style={{
          left: toPercent(layout.nodePositionsById.get(node.id)),
          width: toPercent(layout.nodeWidthsById.get(node.id)),
        }}
        role="listitem"
        aria-current={nodeStatus === 'run' ? 'step' : undefined}
        aria-label={`${nodeIndex + 1}단계 ${node.label}: ${getNodeStatusText(nodeStatus)}`}
      >
        <div className="vf-flow-node-slot">
          <span className="vf-flow-node" aria-hidden="true" />
        </div>
        <div className="vf-flow-label">{node.label}</div>
      </div>
    )
  }

  function renderFlowRow(row) {
    const layout = getRowLayout(row, activeNode?.id, targetBaseRowWeight)
    const connectors = createRowConnectors(row, layout)

    return (
      <div key={row.id} className="vf-flow-row">
        {renderUnitLabel(row)}
        <div className="vf-flow-row-body">
          <div className="vf-flow-row-track">
            {connectors.map(renderConnector)}
            <div className="vf-flow-items">
              {row.nodes.map(node => renderNode(node, layout))}
            </div>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="vf-pipe">
      <div className="vf-progress-head">
        <div className="vf-progress-message">{displayStatusMessage}</div>
      </div>
      <div className="vf-flow" role="list" aria-label="파이프라인 단계">
        {flowRows.map(renderFlowRow)}
      </div>
    </div>
  )
}
