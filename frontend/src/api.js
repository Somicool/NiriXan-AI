// Thin API client for the FastAPI backend. In dev, Vite proxies /api and /media
// to :8000 (see vite.config.js); in prod the backend serves both.
import axios from 'axios'

const api = axios.create({ baseURL: '/api' })

export async function getHealth() {
  const { data } = await api.get('/health')
  return data
}

export async function getCameras() {
  const { data } = await api.get('/cameras')
  return data
}

export async function getVideos() {
  const { data } = await api.get('/videos')
  return data
}

export async function getLibrary() {
  const { data } = await api.get('/library')
  return data
}

export async function ingestAll() {
  const { data } = await api.post('/ingest/all')
  return data
}

export async function getIngestJob(jobId) {
  const { data } = await api.get(`/ingest/job/${jobId}`)
  return data
}

// Upload a video from the user's device, save it server-side, and start
// ingestion. onProgress reports upload % (0-100) while bytes transfer.
export async function uploadVideo({ file, cameraId, onProgress }) {
  const form = new FormData()
  form.append('file', file)
  if (cameraId) form.append('camera_id', cameraId)
  const { data } = await api.post('/ingest/upload', form, {
    onUploadProgress: (e) => {
      if (onProgress && e.total) onProgress(Math.round((e.loaded / e.total) * 100))
    },
  })
  return data
}

export async function stopIngest() {
  const { data } = await api.post('/ingest/stop')
  return data
}

// Permanently delete a video: file on disk + DB rows + FAISS vectors + crops.
export async function deleteVideo(filename) {
  const { data } = await api.post('/videos/delete', { filename })
  return data
}

export async function addCamera({ cameraId, name, location, lat, lon }) {
  const num = (v) => (v === '' || v == null ? null : Number(v))
  const { data } = await api.post('/cameras', {
    camera_id: cameraId, name: name || null, location: location || null,
    lat: num(lat), lon: num(lon),
  })
  return data
}

export async function searchText({ query, language = 'en', topK = 60,
                                   includeScenes = false, filters = {} }) {
  const { data } = await api.post('/search/text', {
    query, language, top_k: topK, include_scenes: includeScenes, filters,
  })
  return data
}

// Describe-and-filter search: parses the plain-language query into structured
// constraints, filters exactly on them, and annotates each result with matches.
export async function describeSearch({ query, topK = 60, filters = {} }) {
  const { data } = await api.post('/search/describe', {
    query, top_k: topK, filters,
  })
  return data
}

export async function searchImage({ file, topK = 60, useReid = false }) {
  const form = new FormData()
  form.append('file', file)
  form.append('top_k', String(topK))
  form.append('use_reid', String(useReid))
  const { data } = await api.post('/search/image', form)
  return data
}

export async function searchPlate({ plate, filters = {} }) {
  const { data } = await api.post('/search/plate', { plate, filters })
  return data
}

export async function trackDetection(detectionId) {
  const { data } = await api.get(`/track/${detectionId}`)
  return data
}

// Per-frame ByteTrack trajectory of a detection's track within its own clip
// (interactive tracking viewer). Reads stored metadata only - no AI re-run.
export async function getTrackPath(detectionId) {
  const { data } = await api.get(`/track/${detectionId}/path`)
  return data
}

export async function getAudit(limit = 50) {
  const { data } = await api.get(`/audit?limit=${limit}`)
  return data
}

// Demo Vehicle Registry (offline, synthetic). Keyed by number plate; the backend
// provider is swappable (demo now, real police API later) with no UI change.
export async function getVehicleRegistry(plate) {
  const { data } = await api.get(`/vehicle-registry/${encodeURIComponent(plate)}`)
  return data
}

// All permanent registry records.
export async function listVehicleRegistry() {
  const { data } = await api.get('/vehicle-registry')
  return data
}

// Investigation activity history (persons + vehicles searched / found / tracked).
export async function getActivity(limit = 300) {
  const { data } = await api.get(`/history?limit=${limit}`)
  return data
}

export function logActivity(entry) {
  // fire-and-forget; never block or break the UI on a logging hiccup
  api.post('/history', entry).catch(() => {})
}

export async function clearActivity() {
  const { data } = await api.delete('/history')
  return data
}

// ---- Face Gallery ----
// deep=false -> instant preview (no AI). deep=true -> full track scan.
export async function getFaceForDetection(detectionId, deep = true) {
  const { data } = await api.get(`/face/for-detection/${detectionId}?deep=${deep}`)
  return data
}

export async function saveFace({ detectionId, investigation }) {
  const { data } = await api.post('/faces/save', { detection_id: detectionId, investigation })
  return data
}

export async function listSavedFaces() {
  const { data } = await api.get('/faces/saved')
  return data
}

export async function deleteSavedFace(savedId) {
  const { data } = await api.delete(`/faces/saved/${savedId}`)
  return data
}

export async function findSimilarFaces(savedId, topK = 60) {
  const { data } = await api.get(`/faces/saved/${savedId}/similar?top_k=${topK}`)
  return data
}

// ---- Face Enhancement (on-demand derived visualisation) ----
// The result is EXTRA evidence: the original saved face is never modified.
export async function getEnhancedFace(savedId) {
  const { data } = await api.get(`/faces/saved/${savedId}/enhanced`)
  return data
}

// Returns { ok: true, data } or { ok: false, detail } — a refusal ("insufficient
// facial evidence") is a legitimate 422 outcome, not a crash, so it is handed
// back to the caller to display rather than thrown.
export async function enhanceFace(savedId, force = false) {
  try {
    const { data } = await api.post(`/faces/saved/${savedId}/enhance`, { force })
    return { ok: true, data }
  } catch (e) {
    const d = e?.response?.data?.detail
    return { ok: false, detail: typeof d === 'object' ? d : { error: d || 'Enhancement failed.' } }
  }
}

// ---- Camera Registry ----
export async function listCameraRegistry() {
  const { data } = await api.get('/camera-registry')
  return data
}

export async function getRegistryStatus() {
  const { data } = await api.get('/camera-registry/status')
  return data
}

export async function getRegistryCamera(cameraId) {
  const { data } = await api.get(`/camera-registry/${encodeURIComponent(cameraId)}`)
  return data
}

export async function saveRegistryCamera(cam) {
  const { data } = await api.post('/camera-registry', cam)
  return data
}

export async function deleteRegistryCamera(cameraId, force = false) {
  const { data } = await api.delete(`/camera-registry/${encodeURIComponent(cameraId)}?force=${force}`)
  return data
}

export async function exportCameraRegistry() {
  const { data } = await api.get('/camera-registry/export')
  return data
}

export async function importCameraRegistry(cameras, replace = false) {
  const { data } = await api.post('/camera-registry/import', { cameras, replace })
  return data
}

export async function probeVideoGps(filename) {
  const { data } = await api.get(`/video-gps/${encodeURIComponent(filename)}`)
  return data
}

export async function getRouteEngine() {
  const { data } = await api.get('/route-engine')
  return data
}

// ---- Journey Reconstruction ----
export async function reconstructJourney({ detectionId, cameras, investigation }) {
  const { data } = await api.post('/journey/reconstruct', {
    detection_id: detectionId, cameras: cameras || null, investigation,
  })
  return data
}

export async function listJourneys(investigation) {
  const { data } = await api.get('/journeys' + (investigation ? `?investigation=${encodeURIComponent(investigation)}` : ''))
  return data
}

export async function getJourney(journeyId) {
  const { data } = await api.get(`/journeys/${journeyId}`)
  return data
}

export async function deleteJourney(journeyId) {
  const { data } = await api.delete(`/journeys/${journeyId}`)
  return data
}

// fmt: 'json' (full) | 'summary' (report) | 'geojson' (map data)
export async function exportJourneyApi(journeyId, fmt = 'json') {
  const { data } = await api.get(`/journeys/${journeyId}/export?fmt=${encodeURIComponent(fmt)}`)
  return data
}

export async function journeyToCaseFile(journeyId, investigation, note) {
  const { data } = await api.post(`/journeys/${journeyId}/case-file`, { investigation, note })
  return data
}

export async function getRouteProviders() {
  const { data } = await api.get('/journey/route-providers')
  return data
}

// Re-run the full pipeline over clips that were ALREADY processed. Their previous
// analysis is discarded first, and each clip keeps its original camera, start time
// and fps. `plates` / `faces` switch on number-plate reading and face recognition,
// which Fast mode leaves off.
export async function reprocessVideos(filenames, { mode, plates, faces } = {}) {
  const { data } = await api.post('/ingest/reprocess', {
    filenames: Array.isArray(filenames) ? filenames : [filenames],
    mode, plates, faces,
  })
  return data
}

// ---- System / Settings ----
export async function getSystemInfo() {
  const { data } = await api.get('/system/info')
  return data
}

export async function updateSettings(payload) {
  const { data } = await api.post('/system/settings', payload)
  return data
}

export async function recomputePlates(videoId) {
  const { data } = await api.post('/recompute-plates' + (videoId ? `?video_id=${videoId}` : ''))
  return data
}

export async function recomputeColors(videoId) {
  const { data } = await api.post('/recompute-colors' + (videoId ? `?video_id=${videoId}` : ''))
  return data
}

export async function backfillRegistry() {
  const { data } = await api.post('/vehicle-registry/backfill')
  return data
}

export async function updateVehicleRegistry(plate, updates) {
  const { data } = await api.put(`/vehicle-registry/${encodeURIComponent(plate)}`, updates)
  return data
}

export async function createExport({ detectionIds, caseNumber, officer, notes }) {
  const { data } = await api.post('/export', {
    detection_ids: detectionIds, case_number: caseNumber, officer, notes,
  })
  return data
}

export async function getExports() {
  const { data } = await api.get('/exports')
  return data
}

// ---------------------------------------------------------------- Case File
// The evidence set and case details are stored server-side in SQLite so a case
// survives a page refresh and a backend restart (they used to be React-only).
export async function loadCase(caseKey) {
  const { data } = await api.get('/case', { params: caseKey ? { case_key: caseKey } : {} })
  return data
}

export async function saveCaseEvidence(items, caseKey) {
  const { data } = await api.put('/case/evidence', { items, case_key: caseKey })
  return data
}

export async function saveCaseInfo(caseInfo, caseKey) {
  const { data } = await api.put('/case/info', { case_info: caseInfo, case_key: caseKey })
  return data
}

// Start choosing a person's best face in the background as soon as their result
// is opened. Picking it needs ~19s of frame decoding + face detection, so doing
// it here means "Save Face" is instant instead of blocking on the same work.
// Fire-and-forget: failures are harmless, Save Face just does the work itself.
export async function prepareFace(detectionId) {
  try {
    const { data } = await api.post(`/face/prepare/${detectionId}`)
    return data
  } catch {
    return null
  }
}

// ---- Case File evidence report (PDF, Gemini-narrated) ----
// Builds the report server-side: original frame with the subject outlined, the
// matched close-up, a situational description per exhibit, and every recorded
// particular. Can take a while - a vision call is made per exhibit.
// `detectionIds` limits the report to specific exhibits (one saved evidence item,
// for example). Omit it to report on the whole saved case.
export async function createCaseReport({ caseKey, caseInfo, detectionIds, useGemini = true } = {}) {
  const { data } = await api.post('/case/report', {
    case_key: caseKey, case_info: caseInfo, detection_ids: detectionIds,
    use_gemini: useGemini,
  }, { timeout: 600000 })
  return data
}

// Report for a case that was already sealed as an export. Uses that export's own
// case number, officer and filed detections, so any past case can be pulled as a PDF.
export async function createReportForExport(exportId, useGemini = true) {
  const { data } = await api.post(`/case/report/for-export/${exportId}`,
    { use_gemini: useGemini }, { timeout: 600000 })
  return data
}

export async function getCaseReports(caseKey) {
  const { data } = await api.get('/case/reports', {
    params: caseKey ? { case_key: caseKey } : {},
  })
  return data
}
