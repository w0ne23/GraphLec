import {
  FINALIZE_PIPELINE_FLOW_NODES,
  FINALIZE_STAGE_KEYS,
  PHASES,
  UPLOAD_PIPELINE_FLOW_NODES,
  UPLOAD_STAGE_KEYS,
  VERIFY_PROGRESS_PIPELINE_FLOW_NODES,
  VERIFY_STAGE_KEYS,
  normalizePipelineStages,
} from '../components/verifier/verifierConstants'

export function normalizeJobType(jobType) {
  const token = String(jobType || '').trim().toLowerCase().replaceAll('-', '_')
  if (['verify', 'verified', 'verified_upload'].includes(token)) return 'verify'
  if (['publish', 'publication', 'upload', 'direct', 'direct_upload', 'graph', 'graph_upload'].includes(token)) return 'publish'
  return token || 'legacy_full'
}

export function normalizeMode(mode, jobType = '') {
  const token = String(mode || '').trim().toLowerCase().replaceAll('-', '_')
  if (['publish', 'publication', 'upload', 'direct', 'direct_upload', 'graph', 'graph_upload'].includes(token)) return 'publish'
  if (['verify', 'verification', 'verified', 'verified_upload'].includes(token)) return 'verify'
  if (!jobType) return 'verify'

  const normalizedJobType = normalizeJobType(jobType)
  if (normalizedJobType === 'publish' || normalizedJobType === 'legacy_full') return 'publish'
  return 'verify'
}

export function phaseFromStatus(status, jobType, mode) {
  const normalizedJobType = normalizeJobType(jobType)
  const normalizedMode = normalizeMode(mode, jobType)

  if (status === 'error') return PHASES.ERROR
  if (normalizedMode === 'publish') {
    if (status === 'done') return PHASES.DONE
    return PHASES.PIPELINE2
  }
  if (normalizedJobType === 'verify' && (status === 'done' || status === 'waiting_approval')) {
    return PHASES.VERIFY_READY
  }
  if (status === 'done') return PHASES.DONE
  if (normalizedJobType === 'publish') return PHASES.PIPELINE2
  if (status === 'waiting_approval') return PHASES.VERIFY_READY
  return PHASES.PIPELINE1
}

export function getKnownStageKeys(phase, mode, isVerified = false) {
  const normalizedMode = normalizeMode(mode)
  if (normalizedMode === 'publish') {
    return isVerified ? FINALIZE_STAGE_KEYS : UPLOAD_STAGE_KEYS
  }
  if (phase === PHASES.PIPELINE2 || phase === PHASES.DONE) {
    return isVerified ? FINALIZE_STAGE_KEYS : UPLOAD_STAGE_KEYS
  }
  return VERIFY_STAGE_KEYS
}

export function mergeStageStatus(current, incoming, phase, mode, isVerified = false) {
  const known = new Set(getKnownStageKeys(phase, mode, isVerified))
  const byStage = new Map(current.map(item => [item.stage, item.status]))

  incoming.forEach(item => {
    if (!item?.stage || !known.has(item.stage)) return
    byStage.set(item.stage, item.status)
  })

  return normalizePipelineStages(Array.from(byStage, ([stage, status]) => ({ stage, status })))
}

export function markKnownStages(phase, status, mode, isVerified = false) {
  return normalizePipelineStages(
    getKnownStageKeys(phase, mode, isVerified).map(stage => ({ stage, status }))
  )
}

export function createInitialStages(mode, isVerified = false) {
  const normalizedMode = normalizeMode(mode)

  if (normalizedMode === 'publish' && isVerified) {
    return normalizePipelineStages(FINALIZE_STAGE_KEYS.map(stage => ({ stage, status: 'wait' })))
  }

  if (normalizedMode === 'publish') {
    return normalizePipelineStages(UPLOAD_STAGE_KEYS.map(stage => ({ stage, status: 'wait' })))
  }

  return normalizePipelineStages(VERIFY_STAGE_KEYS.map(stage => ({ stage, status: 'wait' })))
}

export function getPipelineFlowNodes(mode, isVerified = false) {
  const normalizedMode = normalizeMode(mode)
  if (normalizedMode === 'publish' && isVerified) return FINALIZE_PIPELINE_FLOW_NODES
  if (normalizedMode === 'publish') return UPLOAD_PIPELINE_FLOW_NODES
  return VERIFY_PROGRESS_PIPELINE_FLOW_NODES
}

export function getPipelinePriorNodeIds(mode, isVerified = false) {
  const normalizedMode = normalizeMode(mode)
  if (normalizedMode === 'publish' && isVerified) return ['verified']
  return []
}

export function getPipelineLabel(mode) {
  return normalizeMode(mode) === 'publish' ? '업로드 파이프라인' : '검증 파이프라인'
}
