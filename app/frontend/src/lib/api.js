const API_BASE = '/api'
const RECOMMENDER_BASE = import.meta.env.VITE_RECOMMENDER_API_BASE || '/recommender-api'

export async function uploadLecture({ file, title, category, description }) {
  const formData = new FormData();
  formData.append('video', file);
  formData.append('title', title);
  formData.append('category', category);
  formData.append('description', description);

  const res = await fetch(`${API_BASE}/jobs`, {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    const msg = await res.text();
    throw new Error(msg || 'Upload failed');
  }
  
  const data = await res.json();
  const lectureId = data.lecture_id || data.id || data.job_id;
  return {
    id: lectureId,
    job_id: data.job_id,
    lecture_id: data.lecture_id || lectureId,
    title,
    category,
    description,
    status: 'pending',
    created_at: data.created_at,  // ← 이것만 추가
  };
}

export async function getLectureStatus(jobId) {
  const res = await fetch(`${API_BASE}/jobs/${jobId}`);
  if (!res.ok) throw new Error('Status check failed');
  
  const data = await res.json();
  
  const STAGE_KEYS = ['stt', 'voice', 'scene', 'integrate', 'summarize'];
  const stages = STAGE_KEYS.map((key) => ({
    stage: key,
    status: data.status === 'done' ? 'done' : (data.status === 'running' ? 'run' : 'wait')
  }));

  return {
    lecture_status: data.status,
    stages: stages,
    error_message: data.error_message
  };
}

export async function listLectures() {
  try {
    const res = await fetch(`${API_BASE}/results`);
    if (!res.ok) throw new Error('Failed to fetch lectures');

    const data = await res.json();
    return data.map(lecture => ({
      id: lecture.id,          // lecture_id — 라우팅 등 범용 식별자
      job_id: lecture.job_id,  // job 제어용 (상태조회, 삭제, 재시도)
      lecture_id: lecture.id,  // 결과 조회용 (그래프, 질의)
      title: lecture.title || 'Untitled',
      category: lecture.category || '기타',
      status: lecture.status,
      created_at: lecture.created_at,
      error_message: lecture.error_message,
      pipeline_stages: lecture.pipeline_stages || [],
    }));
  } catch (error) {
    console.error("listLectures error:", error);
    return [];
  }
}

export async function getLectureDetail(lectureId) {
  const res = await fetch(`${API_BASE}/results/${lectureId}`);
  if (!res.ok) throw new Error('Detail fetch failed');
  return res.json();
}

export async function getLectureTimeline(lectureId) {
  const res = await fetch(`${API_BASE}/results/${lectureId}/timeline`);
  if (!res.ok) {
    if (res.status === 404) return []; // No timeline yet
    throw new Error('Timeline fetch failed');
  }
  return res.json();
}

export async function activateLectureGraphRag(lectureId) {
  const res = await fetch(`${API_BASE}/results/${lectureId}/graph/activate`, {
    method: 'POST',
  });
  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || 'GraphRAG activate failed');
  }
  return res.json();
}

export async function unloadLectureGraphRag(lectureId) {
  return fetch(`${API_BASE}/results/${lectureId}/graph/unload`, {
    method: 'POST',
    keepalive: true,
  }).catch(() => null);
}

export async function deleteLecture(jobId) {
  const res = await fetch(`${API_BASE}/jobs/${jobId}`, {
    method: 'DELETE',
  });
  if (!res.ok) {
    const msg = await res.text();
    throw new Error(msg || 'Delete failed');
  }
  return res.json();
}

export async function retryLecture(jobId) {
  const res = await fetch(`${API_BASE}/jobs/${jobId}/retry`, {
    method: 'POST',
  });
  if (!res.ok) {
    const msg = await res.text();
    throw new Error(msg || 'Retry failed');
  }
  return res.json();
}

export async function getLectureGraph(lectureId) {
  const res = await fetch(`${API_BASE}/results/${lectureId}/graph`);
  if (!res.ok) {
    if (res.status === 404) return null; // No graph yet
    throw new Error('Graph fetch failed');
  }
  return res.json();
}

export async function getLectureVerifier(lectureId) {
  const res = await fetch(`${API_BASE}/results/${lectureId}/verifier`);
  if (!res.ok) {
    if (res.status === 404) return null;
    throw new Error('Verifier fetch failed');
  }
  return res.json();
}

export async function askQa(lectureId, question) {
  const res = await fetch(`${API_BASE}/results/${lectureId}/query`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question })
  });
  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || 'QA request failed');
  }
  return res.json();
}

async function postGraphLifecycle(lectureId, action, sessionId) {
  const res = await fetch(`${API_BASE}/results/${lectureId}/graph/${action}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: sessionId }),
  });
  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || `Graph ${action} failed`);
  }
  return res.json();
}

export async function enterLectureGraphSession(lectureId, sessionId) {
  return postGraphLifecycle(lectureId, 'enter', sessionId);
}

export async function heartbeatLectureGraphSession(lectureId, sessionId) {
  return postGraphLifecycle(lectureId, 'heartbeat', sessionId);
}

export async function leaveLectureGraphSession(lectureId, sessionId) {
  return postGraphLifecycle(lectureId, 'leave', sessionId);
}

export async function getLectureGraphSessionStatus(lectureId) {
  const res = await fetch(`${API_BASE}/results/${lectureId}/graph/status`);
  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || 'Graph status fetch failed');
  }
  return res.json();
}

export async function getQaHistory(lectureId) {
  return [];
}

export async function recommendLectures(query, topK = 3) {
  const res = await fetch(`${RECOMMENDER_BASE}/recommend`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, top_k: topK }),
  });
  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || 'Recommend request failed');
  }
  return res.json();
}
