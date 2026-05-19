import { useEffect, useMemo, useRef, useState } from 'react'
import { Network } from 'vis-network'

const NODE_LIMIT = 12
const EDGE_LIMIT = 16
const STRUCTURE_VISIBLE_TYPES = new Set(['Slide', 'Scene', 'VisualAsset'])
const GRAPH_COLORS = {
  GraphRAGEntity: '#FF6B6B',
  GraphRAGCommunity: '#FF9F43',
  Concept: '#FF6B6B',
  Slide: '#4ECDC4',
  Scene: '#A29BFE',
  Context: '#81ECEC',
  Segment: '#7ED957',
  VisualAsset: '#F59E0B',
  AnnotationEmphasis: '#E879F9',
  Video: '#60A5FA',
  Lecture: '#60A5FA',
}
const STRUCTURE_TYPES = new Set(['Slide', 'Scene', 'Context', 'Segment'])
const GENERIC_CONCEPT_LABELS = new Set(['concept', 'entity', 'node', 'graphragentity', 'graphrag entity'])

function nodeProps(node) {
  if (node?.props && typeof node.props === 'object') return node.props
  if (node?.properties && typeof node.properties === 'object') return node.properties
  const raw = typeof node?.title === 'string' ? node.title.trim() : ''
  if (!raw.startsWith('{')) return {}
  try {
    const parsed = JSON.parse(raw)
    return parsed && typeof parsed === 'object' ? parsed : {}
  } catch {
    return {}
  }
}

function normalizedType(node) {
  const props = nodeProps(node)
  const id = String(node?.id || '').toLowerCase()
  const label = String(node?.label || '').toLowerCase()
  const raw = String(node?.type || props.type || '').replace(/[\s_-]+/g, '').toLowerCase()
  if (raw === 'slide') return 'Slide'
  if (raw === 'scene') return 'Scene'
  if (raw === 'context' || raw === 'ctx') return 'Context'
  if (raw === 'segment') return 'Segment'
  if (raw === 'visualasset' || raw === 'visual') return 'VisualAsset'
  if (raw === 'video' || raw === 'lecture' || raw === 'lecturevideo') return 'Video'
  if (raw === 'graphragentity' || raw === 'concept' || raw === 'conceptgraph' || raw === 'entity') return 'GraphRAGEntity'
  if (raw === 'graphragcommunity' || raw === 'community') return 'GraphRAGCommunity'
  if (raw === 'annotation' || raw === 'annotationemphasis' || raw === 'annot') return 'AnnotationEmphasis'
  if (/slide[_/-]?\d+/.test(id) || /^s\d+$/.test(label)) return 'Slide'
  if (/scene[_/-]?\d+/.test(id)) return 'Scene'
  if (/(context|ctx)[_/-]?\d+/.test(id)) return 'Context'
  if (/segment[_/-]?\d+/.test(id)) return 'Segment'
  if (/visual/.test(id) || label === 'visualasset') return 'VisualAsset'
  if (/lecture|video/.test(id) || label === 'video' || label === 'lecture') return 'Video'
  if (/annotation|annot/.test(id) || /annotation|annot/.test(label)) return 'AnnotationEmphasis'
  if (/concept|graphrag/.test(id) || /concept|entity/.test(label)) return 'GraphRAGEntity'
  return node?.type || 'node'
}

function graphColor(node) {
  const type = normalizedType(node)
  if (GRAPH_COLORS[type]) return GRAPH_COLORS[type]
  return type === 'node' || type === 'orphan' ? '#94A3B8' : GRAPH_COLORS.GraphRAGEntity
}

function conceptLabel(node) {
  const props = nodeProps(node)
  const name = String(node?.name || props.name || props.title || props.target_content || '').trim()
  if (name) return name.slice(0, 34)
  const label = String(node?.label || '').trim()
  const title = String(props.title || '').trim()
  const id = String(node?.id || '').trim()
  if (label && !GENERIC_CONCEPT_LABELS.has(label.toLowerCase())) return label.slice(0, 34)
  if (title && !GENERIC_CONCEPT_LABELS.has(title.toLowerCase())) return title.slice(0, 34)
  return id.slice(0, 34)
}

function visualAssetLabel(node, fallback) {
  const props = nodeProps(node)
  const slideNo = props.slide_number ?? String(node?.id || '').match(/slide[_/-]?0*(\d+)/i)?.[1]
  return slideNo != null ? `visualAsset${Number(slideNo)}` : fallback
}

function firstSentence(text) {
  const compact = String(text || '').replace(/\s+/g, ' ').trim()
  if (!compact) return ''
  const match = compact.match(/^(.+?[.!?。！？]|.+?(?:다|요)\.)\s+/)
  return (match?.[1] || compact).slice(0, 180)
}

function edgeLabel(edge) {
  return String(edge.label || edge.rel_type || '')
}

function slideNumberFromNode(node) {
  const label = String(node?.label || '')
  const id = String(node?.id || '')
  const labelMatch = label.match(/^S(\d+)$/i)
  const idMatch = id.match(/slide[_/-]?0*(\d+)/i)
  const num = labelMatch?.[1] || idMatch?.[1]
  return num == null ? null : Number(num)
}

function slideNumbersFrom(items) {
  const out = new Set()
  ;(items || []).forEach(item => {
    const value = item?.slideNumber ?? item?.slide_number
    if (value == null) return
    const n = Number(value)
    if (Number.isFinite(n)) out.add(n)
  })
  return out
}

function compactGraph(graph, options = {}) {
  const rawNodes = Array.isArray(graph?.nodes) ? graph.nodes : []
  const rawEdges = Array.isArray(graph?.edges) ? graph.edges : []
  if (!rawNodes.length) return { nodes: [], edges: [] }

  const sourceMode = options.sourceMode || 'default'
  const relatedSlideNumbers = slideNumbersFrom(options.relatedSlides)
  slideNumbersFrom(options.refs).forEach(n => relatedSlideNumbers.add(n))
  const nodesById = new Map(rawNodes.map(node => [String(node.id), node]))
  const validEdges = rawEdges
    .map(edge => ({
      ...edge,
      _from: String(edge.from ?? edge.src_id ?? ''),
      _to: String(edge.to ?? edge.tgt_id ?? ''),
    }))
    .filter(edge => edge._from && edge._to && nodesById.has(edge._from) && nodesById.has(edge._to))

  const degree = new Map()
  validEdges.forEach(edge => {
    degree.set(edge._from, (degree.get(edge._from) || 0) + 1)
    degree.set(edge._to, (degree.get(edge._to) || 0) + 1)
  })

  const adjacentById = new Map()
  validEdges.forEach(edge => {
    if (!adjacentById.has(edge._from)) adjacentById.set(edge._from, [])
    if (!adjacentById.has(edge._to)) adjacentById.set(edge._to, [])
    adjacentById.get(edge._from).push(edge)
    adjacentById.get(edge._to).push(edge)
  })

  function isRelatedSlide(id) {
    const n = slideNumberFromNode(nodesById.get(id))
    return n != null && relatedSlideNumbers.has(n)
  }

  function connectedToRelatedSlide(edge) {
    return isRelatedSlide(edge._from) || isRelatedSlide(edge._to)
  }

  function nodePriority(node) {
    const id = String(node.id)
    const type = normalizedType(node)
    const relatedSlide = isRelatedSlide(id)
    if (sourceMode === 'visual_location') {
      if (type === 'Slide' && relatedSlide) return 120
      if (type === 'VisualAsset' && (relatedSlide || (adjacentById.get(id) || []).some(connectedToRelatedSlide))) return 110
      if (type === 'Scene' && (adjacentById.get(id) || []).some(connectedToRelatedSlide)) return 95
      if (type === 'GraphRAGEntity' && (adjacentById.get(id) || []).some(connectedToRelatedSlide)) return 55
      if (type === 'GraphRAGEntity') return 35 + (degree.get(id) || 0)
      return 10 + (degree.get(id) || 0)
    }
    if (sourceMode === 'scene_location') {
      if (type === 'Scene') return 110
      if (type === 'Context') return 95
      if (type === 'Segment') return 90
      if (type === 'Slide' && relatedSlide) return 85
      if (type === 'GraphRAGEntity') return 35
      return 10 + (degree.get(id) || 0)
    }
    if (sourceMode === 'overview') {
      if (type === 'GraphRAGEntity') return 110 + (degree.get(id) || 0)
      if (type === 'Slide' && relatedSlide) return 70
      if (type === 'VisualAsset' && (adjacentById.get(id) || []).some(connectedToRelatedSlide)) return 45
      return 10 + (degree.get(id) || 0)
    }
    if (type === 'GraphRAGEntity') return 105 + (degree.get(id) || 0)
    if (type === 'Slide' && relatedSlide) return 65
    if (type === 'Scene' && (adjacentById.get(id) || []).some(connectedToRelatedSlide)) return 35
    return 10 + (degree.get(id) || 0)
  }

  function edgePriority(edge) {
    const label = edgeLabel(edge)
    const fromType = normalizedType(nodesById.get(edge._from))
    const toType = normalizedType(nodesById.get(edge._to))
    if (sourceMode === 'visual_location') {
      if (label === 'HAS_VISUAL_ASSET' && connectedToRelatedSlide(edge)) return 130
      if (label === 'USES_SLIDE' && connectedToRelatedSlide(edge)) return 120
      if (label === 'GRAPHRAG_APPEARS_IN' && connectedToRelatedSlide(edge)) return 65
      if (label === 'GRAPHRAG_APPEARS_IN_SCENE' && (fromType === 'GraphRAGEntity' || toType === 'GraphRAGEntity')) return 50
      if (label === 'GRAPHRAG_RELATES_TO') return 45
      if (label === 'HAS_CONTEXT' || label === 'HAS_SEGMENT') return -100
      return 10
    }
    if (sourceMode === 'scene_location') {
      if (label === 'HAS_CONTEXT') return 120
      if (label === 'HAS_SEGMENT') return 115
      if (label === 'USES_SLIDE' && connectedToRelatedSlide(edge)) return 100
      if (label === 'GRAPHRAG_APPEARS_IN_SCENE') return 55
      if (label === 'GRAPHRAG_RELATES_TO') return 25
      return 10
    }
    if (sourceMode === 'overview') {
      if (label === 'GRAPHRAG_RELATES_TO') return 120
      if (label === 'GRAPHRAG_APPEARS_IN' && connectedToRelatedSlide(edge)) return 75
      if (label === 'USES_SLIDE' && connectedToRelatedSlide(edge)) return 45
      return 10
    }
    if (label === 'GRAPHRAG_RELATES_TO') return 120
    if (label === 'GRAPHRAG_APPEARS_IN' && connectedToRelatedSlide(edge)) return 70
    if (label === 'GRAPHRAG_APPEARS_IN_SCENE') return 45
    return 10
  }

  function edgeAllowed(edge) {
    if (sourceMode !== 'visual_location') return true
    const label = edgeLabel(edge)
    const fromType = nodesById.get(edge._from)?.type
    const toType = nodesById.get(edge._to)?.type
    if (label === 'HAS_CONTEXT' || label === 'HAS_SEGMENT') return false
    if (STRUCTURE_VISIBLE_TYPES.has(fromType) && STRUCTURE_VISIBLE_TYPES.has(toType)) return true
    if (fromType === 'GraphRAGEntity' || toType === 'GraphRAGEntity') return true
    return false
  }

  const rankedEdges = validEdges
    .filter(edgeAllowed)
    .slice()
    .sort((a, b) => {
      const priorityGap = edgePriority(b) - edgePriority(a)
      if (priorityGap !== 0) return priorityGap
      const degreeGap = (degree.get(b._from) || 0) + (degree.get(b._to) || 0) - (degree.get(a._from) || 0) - (degree.get(a._to) || 0)
      return degreeGap
    })

  function nodeAllowed(node) {
    if (sourceMode !== 'visual_location') return true
    const type = normalizedType(node)
    return STRUCTURE_VISIBLE_TYPES.has(type) || type === 'GraphRAGEntity'
  }

  const selectedIds = new Set()
  rawNodes
    .filter(nodeAllowed)
    .slice()
    .sort((a, b) => nodePriority(b) - nodePriority(a))
    .forEach(node => {
      if (selectedIds.size < Math.min(4, NODE_LIMIT)) selectedIds.add(String(node.id))
    })

  const edges = []
  const edgeKeys = new Set()
  const structureCounts = new Map()
  rankedEdges.forEach(edge => {
    if (edges.length >= EDGE_LIMIT) return
    const label = edgeLabel(edge)
    const structureKey = `${label}:${edge._to}`
    if (sourceMode === 'visual_location' && label !== 'GRAPHRAG_RELATES_TO') {
      const count = structureCounts.get(structureKey) || 0
      if (count >= 2) return
      structureCounts.set(structureKey, count + 1)
    }
    const edgeKey = `${edge._from}|${label}|${edge._to}`
    if (edgeKeys.has(edgeKey)) return
    const wouldAdd = Number(!selectedIds.has(edge._from)) + Number(!selectedIds.has(edge._to))
    if (selectedIds.size + wouldAdd > NODE_LIMIT && wouldAdd > 0) return
    edges.push(edge)
    edgeKeys.add(edgeKey)
    selectedIds.add(edge._from)
    selectedIds.add(edge._to)
  })

  rawNodes
    .filter(nodeAllowed)
    .slice()
    .sort((a, b) => nodePriority(b) - nodePriority(a))
    .forEach(node => {
      if (selectedIds.size < NODE_LIMIT) selectedIds.add(String(node.id))
    })

  const nodes = Array.from(selectedIds)
    .slice(0, NODE_LIMIT)
    .map(id => nodesById.get(id))
    .filter(Boolean)
  const nodeIds = new Set(nodes.map(node => String(node.id)))
  const visibleEdges = edges.filter(edge => nodeIds.has(edge._from) && nodeIds.has(edge._to))

  return { nodes, edges: visibleEdges }
}

export function graphStats(graph) {
  const nodes = Array.isArray(graph?.nodes) ? graph.nodes.length : 0
  const edges = Array.isArray(graph?.edges) ? graph.edges.length : 0
  return { nodes, edges, hasGraph: nodes > 0 }
}

export default function ChatGraphPreview({ graph, sourceMode = 'default', relatedSlides = [], refs = [] }) {
  const containerRef = useRef(null)
  const networkRef = useRef(null)
  const [hoverInfo, setHoverInfo] = useState(null)
  const [selectedEdge, setSelectedEdge] = useState(null)
  const graphData = useMemo(
    () => compactGraph(graph, { sourceMode, relatedSlides, refs }),
    [graph, refs, relatedSlides, sourceMode]
  )
  const displayById = useMemo(() => {
    const counters = {}
    const labels = new Map()

    function nextLabel(type) {
      counters[type] = (counters[type] || 0) + 1
      return `${type}${counters[type]}`
    }

    graphData.nodes.forEach(node => {
      const id = String(node.id)
      const label = String(node.label || '')
      const type = normalizedType(node)
      if (type === 'Slide') {
        const slideNo = label.match(/^S(\d+)$/i)?.[1] || id.match(/slide[_/-]?0*(\d+)/i)?.[1]
        labels.set(id, slideNo ? `slide${Number(slideNo)}` : nextLabel('slide'))
      } else if (type === 'VisualAsset') {
        labels.set(id, visualAssetLabel(node, nextLabel('visualAsset')))
      } else if (type === 'AnnotationEmphasis') {
        labels.set(id, id.slice(0, 34))
      } else if (type === 'Scene') {
        const sceneNo = id.match(/scene[_/-]?0*(\d+)/i)?.[1]
        labels.set(id, sceneNo ? `scene${Number(sceneNo)}` : nextLabel('scene'))
      } else if (type === 'Context') {
        const contextNo = id.match(/context[_/-]?0*(\d+)/i)?.[1] || id.match(/ctx[_/-]?0*(\d+)/i)?.[1]
        labels.set(id, contextNo ? `context${Number(contextNo)}` : nextLabel('context'))
      } else if (type === 'Segment') {
        const segmentNo = id.match(/segment[_/-]?0*(\d+)/i)?.[1]
        labels.set(id, segmentNo ? `segment${Number(segmentNo)}` : nextLabel('segment'))
      } else {
        labels.set(id, conceptLabel(node))
      }
    })

    return labels
  }, [graphData])

  function nodeDetail(node) {
    const type = normalizedType(node)
    const props = nodeProps(node)
    if (type === 'VisualAsset') {
      const assetType = String(node?.asset_type || props.asset_type || '').trim()
      return assetType ? `asset_type: ${assetType}` : 'asset_type: visual'
    }
    if (type === 'Segment' || type === 'Context') {
      const text = firstSentence(node?.text || props.text)
      return text
    }
    if (type === 'AnnotationEmphasis') {
      return `type: ${String(node?.type || props.type || type).trim()}`
    }
    return ''
  }

  function relationLabel(edge) {
    return String(edge.label || edge.rel_type || 'RELATED')
      .replace(/^GRAPHRAG_/, '')
      .replaceAll('_', ' ')
      .trim()
  }

  useEffect(() => {
    if (!containerRef.current || !graphData.nodes.length) return undefined

    const nodesById = new Map(graphData.nodes.map(node => [String(node.id), node]))
    const edgeInfoById = new Map()

    const nodes = graphData.nodes.map(node => {
      const color = graphColor(node)
      const type = normalizedType(node)
      const isConcept = type === 'GraphRAGEntity' || type === 'GraphRAGCommunity'
      const isVisual = type === 'VisualAsset'
      return {
        id: String(node.id),
        label: displayById.get(String(node.id)),
        title: nodeDetail(node),
        shape: 'dot',
        size: isConcept ? 18 : (isVisual ? 16 : 14),
        color: {
          background: color,
          border: '#334155',
          highlight: { background: color, border: '#0f172a' },
        },
        font: { color: '#111827', size: 11 },
      }
    })

    const edges = graphData.edges.map((edge, idx) => {
      const edgeId = `qa-edge-${idx}`
      const from = String(edge._from ?? edge.from ?? edge.src_id)
      const to = String(edge._to ?? edge.to ?? edge.tgt_id)
      const relation = relationLabel(edge)
      edgeInfoById.set(edgeId, {
        relation,
        from: displayById.get(from) || from,
        to: displayById.get(to) || to,
        source: nodesById.get(from),
        target: nodesById.get(to),
      })
      return {
        id: edgeId,
        from,
        to,
        title: relation,
        arrows: 'to',
      }
    })

    networkRef.current?.destroy()
    networkRef.current = new Network(containerRef.current, { nodes, edges }, {
      nodes: { borderWidth: 1 },
      edges: {
        color: '#94a3b8',
        font: { size: 0 },
        smooth: { type: 'dynamic' },
        width: 1.2,
        arrows: { to: { enabled: true, scaleFactor: 0.45 } },
        selectionWidth: 2.5,
      },
      physics: {
        enabled: true,
        stabilization: { iterations: 90, updateInterval: 20 },
        barnesHut: {
          gravitationalConstant: -1600,
          springLength: 95,
          springConstant: 0.045,
          damping: 0.18,
        },
      },
      interaction: {
        hover: true,
        dragNodes: true,
        hideEdgesOnDrag: true,
        selectConnectedEdges: false,
      },
    })

    networkRef.current.on('hoverNode', params => {
      const node = nodesById.get(String(params.node))
      if (!node) return
      const body = nodeDetail(node)
      if (!body) return
      setHoverInfo({
        x: params.pointer.DOM.x,
        y: params.pointer.DOM.y,
        label: displayById.get(String(params.node)) || String(params.node),
        body,
      })
    })
    networkRef.current.on('blurNode', () => setHoverInfo(null))
    networkRef.current.on('click', params => {
      if (!params.edges?.length) {
        setSelectedEdge(null)
        return
      }
      setSelectedEdge(edgeInfoById.get(params.edges[0]) || null)
    })

    const fitTimer = setTimeout(() => {
      networkRef.current?.fit({ animation: { duration: 350, easingFunction: 'easeInOutQuad' } })
    }, 120)

    return () => {
      clearTimeout(fitTimer)
      networkRef.current?.destroy()
      networkRef.current = null
    }
  }, [displayById, graphData])

  if (!graphData.nodes.length) return null

  return (
    <div className="chat-graph-preview">
      <div className="chat-graph-network" ref={containerRef} />
      {hoverInfo && (
        <div className="chat-graph-tooltip" style={{ left: hoverInfo.x + 10, top: hoverInfo.y + 10 }}>
          <div className="chat-graph-tooltip-title">{hoverInfo.label}</div>
          <div>{hoverInfo.body}</div>
        </div>
      )}
      {selectedEdge && (
        <div className="chat-graph-edge-info">
          <strong>{selectedEdge.from}</strong>
          <span>{selectedEdge.relation}</span>
          <strong>{selectedEdge.to}</strong>
        </div>
      )}
    </div>
  )
}
