import React, { useEffect, useMemo, useRef, useState } from 'react';
import { Network } from 'vis-network';
import { getLectureGraph } from '../../lib/api';

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
};

const GRAPH_VIEW_MODES = [
  { id: 'all', label: '전체 보기' },
  { id: 'structure', label: '구조 그래프 보기' },
  { id: 'concept', label: '개념 그래프 보기' },
];

const GENERIC_CONCEPT_LABELS = new Set(['concept', 'entity', 'node', 'graphragentity', 'graphrag entity']);
const STRUCTURE_TYPES = new Set(['Video', 'Slide', 'Scene', 'Context', 'Segment', 'VisualAsset', 'AnnotationEmphasis']);
const CONCEPT_TYPES = new Set(['GraphRAGEntity', 'GraphRAGCommunity']);

function nodeProps(node) {
  if (node?.props && typeof node.props === 'object') return node.props;
  if (node?.properties && typeof node.properties === 'object') return node.properties;
  const raw = typeof node?.title === 'string' ? node.title.trim() : '';
  if (!raw.startsWith('{')) return {};
  try {
    const parsed = JSON.parse(raw);
    return parsed && typeof parsed === 'object' ? parsed : {};
  } catch {
    return {};
  }
}

function normalizedType(node) {
  const props = nodeProps(node);
  const id = String(node?.id || '').toLowerCase();
  const label = String(node?.label || '').toLowerCase();
  const raw = String(node?.type || props.type || '').replace(/[\s_-]+/g, '').toLowerCase();
  if (raw === 'slide') return 'Slide';
  if (raw === 'scene') return 'Scene';
  if (raw === 'context' || raw === 'ctx') return 'Context';
  if (raw === 'segment') return 'Segment';
  if (raw === 'visualasset' || raw === 'visual') return 'VisualAsset';
  if (raw === 'video' || raw === 'lecture' || raw === 'lecturevideo') return 'Video';
  if (raw === 'graphragentity' || raw === 'concept' || raw === 'conceptgraph' || raw === 'entity') return 'GraphRAGEntity';
  if (raw === 'graphragcommunity' || raw === 'community') return 'GraphRAGCommunity';
  if (raw === 'annotation' || raw === 'annotationemphasis' || raw === 'annot') return 'AnnotationEmphasis';
  if (/slide[_/-]?\d+/.test(id) || /^s\d+$/.test(label)) return 'Slide';
  if (/scene[_/-]?\d+/.test(id)) return 'Scene';
  if (/(context|ctx)[_/-]?\d+/.test(id)) return 'Context';
  if (/segment[_/-]?\d+/.test(id)) return 'Segment';
  if (/visual/.test(id) || label === 'visualasset') return 'VisualAsset';
  if (/lecture|video/.test(id) || label === 'video' || label === 'lecture') return 'Video';
  if (/annotation|annot/.test(id) || /annotation|annot/.test(label)) return 'AnnotationEmphasis';
  if (/concept|graphrag/.test(id) || /concept|entity/.test(label)) return 'GraphRAGEntity';
  return node?.type || 'node';
}

function graphColor(node) {
  const type = normalizedType(node);
  if (GRAPH_COLORS[type]) return GRAPH_COLORS[type];
  return type === 'node' || type === 'orphan' ? '#94A3B8' : GRAPH_COLORS.GraphRAGEntity;
}

function graphGroup(type) {
  if (STRUCTURE_TYPES.has(type)) return 'structure';
  if (CONCEPT_TYPES.has(type)) return 'concept';
  return 'other';
}

function isModeFocused(group, mode) {
  return mode === 'all' || group === mode;
}

function styledNodeColor(baseColor, focused) {
  const background = focused ? baseColor : '#CBD5E1';
  const border = focused ? '#334155' : '#94A3B8';
  return {
    background,
    border,
    highlight: {
      background: focused ? baseColor : '#CBD5E1',
      border: focused ? '#0f172a' : '#64748B',
    },
  };
}

function styledEdge(edge, source, target, mode) {
  const fromGroup = graphGroup(normalizedType(source));
  const toGroup = graphGroup(normalizedType(target));
  const relation = relationLabel(edge);
  const isStructureEdge = fromGroup === 'structure' && toGroup === 'structure';
  const isConceptEdge = fromGroup === 'concept' && toGroup === 'concept';
  const focused = mode === 'all'
    || (mode === 'structure' && isStructureEdge)
    || (mode === 'concept' && isConceptEdge);
  const color = mode === 'all'
    ? '#94A3B8'
    : (focused ? (mode === 'structure' ? '#38BDF8' : '#FB7185') : 'rgba(148, 163, 184, 0.22)');
  return {
    relation,
    focused,
    color,
    width: focused ? 1.5 : 0.45,
  };
}

function conceptLabel(node) {
  const props = nodeProps(node);
  const name = String(node?.name || props.name || props.title || props.target_content || '').trim();
  if (name) return name.slice(0, 34);
  const label = String(node?.label || '').trim();
  const title = String(props.title || '').trim();
  const id = String(node?.id || '').trim();
  if (label && !GENERIC_CONCEPT_LABELS.has(label.toLowerCase())) return label.slice(0, 34);
  if (title && !GENERIC_CONCEPT_LABELS.has(title.toLowerCase())) return title.slice(0, 34);
  return id.slice(0, 34);
}

function visualAssetLabel(node, fallback) {
  const props = nodeProps(node);
  const slideNo = props.slide_number ?? String(node?.id || '').match(/slide[_/-]?0*(\d+)/i)?.[1];
  return slideNo != null ? `visualAsset${Number(slideNo)}` : fallback;
}

function firstSentence(text) {
  const compact = String(text || '').replace(/\s+/g, ' ').trim();
  if (!compact) return '';
  const match = compact.match(/^(.+?[.!?。！？]|.+?(?:다|요)\.)\s+/);
  return (match?.[1] || compact).slice(0, 180);
}

function displayLabelsFor(nodes) {
  const counters = {};
  const labels = new Map();

  function nextLabel(type) {
    counters[type] = (counters[type] || 0) + 1;
    return `${type}${counters[type]}`;
  }

  nodes.forEach((node) => {
    const id = String(node.id);
    const label = String(node.label || '');
    const type = normalizedType(node);
    if (type === 'Slide') {
      const slideNo = label.match(/^S(\d+)$/i)?.[1] || id.match(/slide[_/-]?0*(\d+)/i)?.[1];
      labels.set(id, slideNo ? `slide${Number(slideNo)}` : nextLabel('slide'));
    } else if (type === 'VisualAsset') {
      labels.set(id, visualAssetLabel(node, nextLabel('visualAsset')));
    } else if (type === 'AnnotationEmphasis') {
      labels.set(id, id.slice(0, 34));
    } else if (type === 'Scene') {
      const sceneNo = id.match(/scene[_/-]?0*(\d+)/i)?.[1];
      labels.set(id, sceneNo ? `scene${Number(sceneNo)}` : nextLabel('scene'));
    } else if (type === 'Context') {
      const contextNo = id.match(/context[_/-]?0*(\d+)/i)?.[1] || id.match(/ctx[_/-]?0*(\d+)/i)?.[1];
      labels.set(id, contextNo ? `context${Number(contextNo)}` : nextLabel('context'));
    } else if (type === 'Segment') {
      const segmentNo = id.match(/segment[_/-]?0*(\d+)/i)?.[1];
      labels.set(id, segmentNo ? `segment${Number(segmentNo)}` : nextLabel('segment'));
    } else {
      labels.set(id, conceptLabel(node));
    }
  });

  return labels;
}

function nodeDetail(node) {
  const type = normalizedType(node);
  const props = nodeProps(node);
  if (type === 'VisualAsset') {
    const assetType = String(node?.asset_type || props.asset_type || '').trim();
    return assetType ? `asset_type: ${assetType}` : 'asset_type: visual';
  }
  if (type === 'Segment' || type === 'Context') {
    const text = firstSentence(node?.text || props.text);
    return text;
  }
  if (type === 'AnnotationEmphasis') {
    return `type: ${String(node?.type || props.type || type).trim()}`;
  }
  return '';
}

function relationLabel(edge) {
  return String(edge.label || edge.rel_type || 'RELATED')
    .replace(/^GRAPHRAG_/, '')
    .replaceAll('_', ' ')
    .trim();
}

function buildStyledGraph(rawGraph, displayById, mode) {
  const rawNodes = Array.isArray(rawGraph?.nodes) ? rawGraph.nodes : [];
  const rawEdges = Array.isArray(rawGraph?.edges) ? rawGraph.edges : [];
  const nCount = rawNodes.length;
  const heavy = nCount > 150;
  const nodesById = new Map(rawNodes.map(node => [String(node.id), node]));
  const edgeInfoById = new Map();

  const nodes = rawNodes.map((n) => {
    const baseColor = graphColor(n);
    const type = normalizedType(n);
    const group = graphGroup(type);
    const focused = isModeFocused(group, mode);
    const isConcept = type === 'GraphRAGEntity' || type === 'GraphRAGCommunity';
    const isVisual = type === 'VisualAsset';
    return {
      id: n.id != null ? String(n.id) : 'n',
      label: displayById.get(String(n.id)) || conceptLabel(n),
      title: nodeDetail(n),
      shape: 'dot',
      size: isConcept ? (nCount > 400 ? 13 : 18) : (isVisual ? (nCount > 400 ? 12 : 16) : (nCount > 400 ? 11 : 14)),
      color: styledNodeColor(baseColor, focused),
      borderWidth: focused ? 2 : 1,
      opacity: focused ? 1 : 0.28,
      font: {
        color: focused ? '#111827' : 'rgba(100, 116, 139, 0.48)',
        size: nCount > 400 ? 10 : 11,
      },
    };
  });

  const edges = rawEdges.map((e, i) => {
    const edgeId = `gv-edge-${i}`;
    const from = String(e.from != null ? e.from : e.src_id);
    const to = String(e.to != null ? e.to : e.tgt_id);
    const source = nodesById.get(from);
    const target = nodesById.get(to);
    const edgeStyle = styledEdge(e, source, target, mode);
    edgeInfoById.set(edgeId, {
      relation: edgeStyle.relation,
      from: displayById.get(from) || from,
      to: displayById.get(to) || to,
      source,
      target,
      focused: edgeStyle.focused,
    });
    return {
      id: edgeId,
      from,
      to,
      title: edgeStyle.relation,
      arrows: 'to',
      color: edgeStyle.color,
      width: heavy ? Math.max(0.35, edgeStyle.width * 0.8) : edgeStyle.width,
    };
  }).filter(e => e.from && e.to);

  return { nodes, edges, nodesById, edgeInfoById, heavy, nCount };
}

function GraphViewer({ lectureId }) {
  const containerRef = useRef(null);
  const networkRef = useRef(null);
  const graphCacheRef = useRef(null);
  const lastStyledModeRef = useRef(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [graphData, setGraphData] = useState(null);
  const [hoverInfo, setHoverInfo] = useState(null);
  const [selectedEdge, setSelectedEdge] = useState(null);
  const [viewMode, setViewMode] = useState('all');

  const rawGraph = graphData?.graph || { nodes: [], edges: [] };
  const displayById = useMemo(
    () => displayLabelsFor(Array.isArray(rawGraph.nodes) ? rawGraph.nodes : []),
    [rawGraph]
  );

  useEffect(() => {
    if (!lectureId) return;
    setLoading(true);
    setError(null);

    getLectureGraph(lectureId)
      .then((data) => {
        if (!data || !data.graph) {
          setError('그래프 데이터가 없습니다.');
          return;
        }
        setGraphData(data);
      })
      .catch((err) => {
        console.error('Graph fetch error:', err);
        setError('그래프 데이터를 불러오는 중 오류가 발생했습니다.');
      })
      .finally(() => {
        setLoading(false);
      });
  }, [lectureId]);

  useEffect(() => {
    if (!containerRef.current || !graphData || !graphData.graph) return;

    const {
      nodes,
      edges,
      nodesById,
      edgeInfoById,
      heavy,
      nCount,
    } = buildStyledGraph(graphData.graph, displayById, viewMode);
    graphCacheRef.current = { nodesById, edgeInfoById };
    lastStyledModeRef.current = viewMode;

    const data = { nodes, edges };

    const options = {
      nodes: { borderWidth: 2 },
      edges: {
        color: '#94a3b8',
        arrows: { to: { enabled: true, scaleFactor: 0.45 } },
        smooth: heavy ? false : { type: 'dynamic' },
        font: { size: 0 },
        width: heavy ? 0.8 : 1.2,
        selectionWidth: 2.5,
      },
      physics: {
        enabled: true,
        stabilization: { enabled: false },
        barnesHut: {
          gravitationalConstant: heavy ? -4000 : -2000,
          springLength: heavy ? 150 : 100,
          springConstant: 0.04,
          damping: 0.09,
        },
      },
      interaction: { 
        hover: true,
        tooltipDelay: 0,
        hideEdgesOnDrag: heavy,
        dragNodes: true,
        selectConnectedEdges: false,
      },
    };

    if (networkRef.current) {
      networkRef.current.destroy();
    }

    const network = new Network(containerRef.current, data, options);
    networkRef.current = network;

    const handleMouseMove = (params) => {
      setHoverInfo(prev => prev ? {
        ...prev,
        x: params.pointer.DOM.x,
        y: params.pointer.DOM.y,
      } : null);
    };

    network.on('hoverNode', (params) => {
      const nodeId = params.node;
      const nodeData = nodesById.get(String(nodeId));
      const body = nodeData ? nodeDetail(nodeData) : '';
      if (body) {
        setHoverInfo({
          id: nodeId,
          label: displayById.get(String(nodeId)) || String(nodeId),
          body,
          type: nodeData.type,
          x: params.pointer.DOM.x,
          y: params.pointer.DOM.y
        });
      }
    });

    network.on('blurNode', () => {
      setHoverInfo(null);
    });
    
    network.on('mousemove', handleMouseMove);
    network.on('click', (params) => {
      if (!params.edges?.length) {
        setSelectedEdge(null);
        return;
      }
      setSelectedEdge(graphCacheRef.current?.edgeInfoById?.get(params.edges[0]) || null);
    });

    // [개선] 초기 로딩 시 그래프 맞춤 (안정화 시 재정렬은 사용자 요청으로 제거)
    const handleFit = () => {
      if (networkRef.current) {
        networkRef.current.fit({
          animation: { duration: 1000, easingFunction: 'easeInOutQuad' }
        });
        // 너무 과하게 확대되는 것 방지 (노드가 적을 때)
        const currentScale = networkRef.current.getScale();
        if (currentScale > 1.2) {
          networkRef.current.moveTo({ scale: 1.0, animation: { duration: 1000 } });
        }
      }
    };

    // 패널이 열리면서 크기가 변할 때를 대비해 약간의 지연 후 fit 실행
    const initialFitTimer = setTimeout(handleFit, 600);
    const physicsStopTimer = setTimeout(() => {
      if (networkRef.current) {
        networkRef.current.setOptions({ physics: false });
      }
    }, heavy ? 2600 : 1800);

    // [개선] 컨테이너 크기 변화 감지 (ResizeObserver)
    const resizeObserver = new ResizeObserver(() => {
      if (networkRef.current) {
        networkRef.current.fit();
        const currentScale = networkRef.current.getScale();
        if (currentScale > 1.2) {
          networkRef.current.moveTo({ scale: 1.0 });
        }
      }
    });
    resizeObserver.observe(containerRef.current);

    return () => {
      if (initialFitTimer) clearTimeout(initialFitTimer);
      if (physicsStopTimer) clearTimeout(physicsStopTimer);
      resizeObserver.disconnect();
      
      if (networkRef.current) {
        networkRef.current.destroy();
        networkRef.current = null;
      }
    };
  }, [displayById, graphData]);

  useEffect(() => {
    if (!networkRef.current || !graphData?.graph) return;
    if (lastStyledModeRef.current === viewMode) return;
    const { nodes, edges, nodesById, edgeInfoById } = buildStyledGraph(graphData.graph, displayById, viewMode);
    graphCacheRef.current = { nodesById, edgeInfoById };
    networkRef.current.body.data.nodes.update(nodes);
    networkRef.current.body.data.edges.update(edges);
    networkRef.current.setOptions({ physics: false });
    lastStyledModeRef.current = viewMode;
  }, [displayById, graphData, viewMode]);

  return (
    <div className="gv-container" style={{ position: 'relative' }}>
      <div className="gv-header">
        <span className="gv-stats">
          {loading ? '데이터 로딩 중...' : `노드: ${graphData?.node_count || 0}개 · 엣지: ${graphData?.edge_count || 0}개`}
        </span>
        <div className="gv-mode-tabs" aria-label="그래프 보기 모드">
          {GRAPH_VIEW_MODES.map(mode => (
            <button
              key={mode.id}
              type="button"
              className={`gv-mode-tab ${viewMode === mode.id ? 'is-active' : ''}`}
              onClick={() => {
                setViewMode(mode.id);
                setSelectedEdge(null);
              }}
            >
              {mode.label}
            </button>
          ))}
        </div>
      </div>
      
      <div className="gv-network-wrapper" style={{ position: 'relative', flex: 1, minHeight: 0 }}>
        {error && (
          <div className="gv-status">
            <p className="gv-error">{error}</p>
          </div>
        )}
        
        <div 
          className="gv-network" 
          ref={containerRef} 
          style={{ width: '100%', height: '100%', visibility: loading || error ? 'hidden' : 'visible' }}
        />

        {hoverInfo && !loading && (
          <div className="gv-tooltip" style={{ left: hoverInfo.x + 10, top: hoverInfo.y + 10 }}>
            <div className="gv-tooltip-title">{hoverInfo.label}</div>
            <div>{hoverInfo.body}</div>
          </div>
        )}

        {selectedEdge && !loading && (
          <div className="gv-edge-info">
            <strong>{selectedEdge.from}</strong>
            <span>{selectedEdge.relation}</span>
            <strong>{selectedEdge.to}</strong>
          </div>
        )}
        
        {loading && (
          <div className="gv-status" style={{ position: 'absolute', inset: 0, background: 'var(--bg-soft)', zIndex: 10, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <p>전체 지식 그래프 데이터를 불러오는 중...</p>
          </div>
        )}
      </div>
    </div>
  );
}

export default GraphViewer;
