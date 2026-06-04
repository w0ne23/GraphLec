function formatPipelineName(label) {
  const name = String(label || '').replace('파이프라인', '').trim()
  if (name === '업로드') return '그래프화'
  return name
}

export default function PipelineStatusHeader({ title, pipelineLabel }) {
  const pipelineName = formatPipelineName(pipelineLabel)

  return (
    <div className="vf-status-header">
      <div className="vf-status-title">{title || '강의 영상'}</div>
      {pipelineName && <div className="vf-status-label">{pipelineName}</div>}
    </div>
  )
}
