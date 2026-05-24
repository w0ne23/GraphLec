const API_BASE = '/api'
const RECOMMENDER_BASE = import.meta.env.VITE_RECOMMENDER_API_BASE || '/recommender-api'

export async function uploadLecture({ file, title, category, description, workflowMode = 'legacy_full' }) {
  const formData = new FormData();
  formData.append('video', file);
  formData.append('title', title);
  formData.append('category', category);
  formData.append('description', description);
  formData.append('workflow_mode', workflowMode);

  const res = await fetch(`${API_BASE}/jobs`, {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    const msg = await res.text();
    throw new Error(msg || 'Upload failed');
  }
  
  const data = await res.json();
  return {
    id: data.id,
    job_id: data.job_id,
    job_type: data.job_type,
    title: data.title || title,
    category: data.category || category,
    description: data.description || description,
    status: data.status || 'pending',
    created_at: data.created_at,
  };
}

// 1. SSE 영역 - pending/running
export async function listActiveJobs() {
  const res = await fetch(`${API_BASE}/jobs?status=active`)
  if (!res.ok) throw new Error('Failed to fetch active jobs')
  return res.json() // []
}

// 2. UploadPage 완료 목록 - error/done/기타
export async function listUploadedLectures(params = {}) {
  return _fetchResults({ ...params, scope: 'upload' })
}

// 3. LectureListPage - done만
export async function listLectures(params = {}) {
  return _fetchResults({ ...params, scope: 'browse' })
}

async function _fetchResults(params) {
  try {
    const query = new URLSearchParams()
    const limit = params.limit || 12
    query.append('limit', limit)
    query.append('scope', params.scope)
    if (params.page) query.append('page', params.page)
    if (params.category && params.category !== '전체') query.append('category', params.category)
    if (params.search) query.append('search', params.search)

    const res = await fetch(`${API_BASE}/results?${query.toString()}`)
    if (!res.ok) throw new Error('Failed to fetch lectures')
    const data = await res.json()

    const items = Array.isArray(data) ? data : (data.items || [])
    const totalItems = data.total_items || items.length
    const totalPages = Math.max(1, Math.ceil(totalItems / limit))

    return {
      items: items.map(lec => ({
        id: lec.id,
        job_id: lec.job_id,
        job_type: lec.job_type,
        title: lec.title || 'Untitled',
        category: lec.category || '기타',
        status: lec.status,
        created_at: lec.created_at,
        error_message: lec.error_message,
        pipeline_stages: lec.pipeline_stages || [],
        tags: lec.tags || [],
        is_dummy: lec.is_dummy || false,
        source: lec.source || 'database',
      })),
      totalPages,
      totalItems,
    }
  } catch (error) {
    console.error('_fetchResults error:', error)
    return { items: [], totalPages: 1 }
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

export async function deleteLecture(lectureId) {
  const res = await fetch(`${API_BASE}/jobs/${lectureId}`, {
    method: 'DELETE',
  });
  if (!res.ok) {
    const msg = await res.text();
    throw new Error(msg || 'Delete failed');
  }
  return res.json();
}

export async function retryLecture(lectureId) {
  const res = await fetch(`${API_BASE}/jobs/${lectureId}/retry`, {
    method: 'POST',
  });
  if (!res.ok) {
    const msg = await res.text();
    throw new Error(msg || 'Retry failed');
  }
  const data = await res.json();
  // 새로 생성된 job_id를 반환 — 프론트에서 SSE 재연결에 사용
  return { job_id: data.job_id };
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

export async function askQa(lectureId, question, context = {}) {
  const res = await fetch(`${API_BASE}/results/${lectureId}/query`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, ...context })
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

export async function recommendLectures(query, topK = 3) {
  let res
  try {
    res = await fetch(`${RECOMMENDER_BASE}/recommend`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, top_k: topK }),
    });
  } catch (error) {
    const err = new Error('RECOMMENDER_NETWORK_ERROR')
    err.cause = error
    throw err
  }
  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    const err = new Error(errorData.detail || 'RECOMMENDER_SERVER_ERROR');
    err.status = res.status;
    throw err;
  }
  return res.json();
}
