// Interactive Object Tracking Viewer.
//
// Replays a single detected person/vehicle inside its ORIGINAL CCTV clip and
// draws a bounding box that follows the object across its tracked lifetime.
// It reuses the ByteTrack track that was computed during ingestion - the box
// path is loaded from indexed metadata via GET /track/{id}/path. No detection,
// tracking, ReID, OCR or CLIP runs here; playback is pure metadata + <video>.
//
// The box is interpolated between stored sample frames for smoothness and is
// hidden automatically when the object isn't present (gaps / outside lifetime).
//
// Designed to be extended for cross-camera replay later: the component takes a
// single track "path" object, so a future multi-camera timeline can feed it a
// stitched path (or swap clips) without changing the overlay/controls core.
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { getTargetTrack, getTrackPath } from '../api'

const SPEEDS = [0.25, 0.5, 1, 2, 4]

export default function TrackingViewer({ detection, onClose, onAddEvidence, inEvidence }) {
  const [path, setPath] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [playing, setPlaying] = useState(false)
  const [speed, setSpeed] = useState(1)
  const [showBox, setShowBox] = useState(true)
  const [curTime, setCurTime] = useState(0)
  const [verified, setVerified] = useState(null)   // metrics from the target pass
  const [status, setStatus] = useState(null)       // live per-frame lock state

  const videoRef = useRef(null)
  const boxRef = useRef(null)
  const seekRef = useRef(null)
  const draggingRef = useRef(false)
  const rafRef = useRef(0)
  const showBoxRef = useRef(true)
  showBoxRef.current = showBox
  // read inside the rAF paint loop, which must not re-subscribe on every render
  const verifiedRef = useRef(null)
  const statusRef = useRef(null)

  const detId = detection?.detection_id

  // ---- load the track ----
  // Preferred source is the QUERY-TIME target pass: it re-detects people in the
  // original recording at ~5 FPS and only emits a box where the candidate is
  // verified as the selected person. The stored ~2 FPS ingestion path is kept as a
  // fallback for anything the target pass cannot handle (no usable identity
  // embedding, source video missing), so the viewer never comes up empty.
  useEffect(() => {
    let alive = true
    setLoading(true); setError(null); setPath(null); setVerified(null); setStatus(null)
    if (detId == null) { setLoading(false); return }
    verifiedRef.current = null; statusRef.current = null
    ;(async () => {
      // Stage 1: the indexed track, immediately. This is metadata only, so the clip
      // is playable in well under a second. Waiting for the verification pass before
      // showing anything meant staring at a spinner for ~50 s with no video.
      let meta = null
      try {
        meta = await getTrackPath(detId)
      } catch (e) {
        if (alive) {
          setError(e?.response?.data?.detail || e.message || 'Could not load track')
          setLoading(false)
        }
        return
      }
      if (!alive) return
      setPath(meta)
      setVerified({ source: 'stored', refining: true })
      setLoading(false)

      // Stage 2: upgrade to the verified target pass in the background, then swap
      // the boxes in. The video keeps playing throughout.
      const r = await getTargetTrack(detId)
      if (!alive) return
      if (r.ok && (r.data?.points || []).length) {
        const d = r.data
        const pts = d.points.map((p) => ({
          detection_id: detId,
          offset_seconds: p.offset_seconds,
          frame_number: p.frame_number,
          timestamp: null,
          bbox: p.bbox,
          confidence: p.confidence,
          predicted: !!p.predicted,
          status: p.status,
          identity: p.identity,
        }))
        setPath({
          ...meta,
          points: pts,
          start_offset: d.start_offset,
          end_offset: d.end_offset,
          duration: (d.end_offset != null && d.start_offset != null)
            ? Number((d.end_offset - d.start_offset).toFixed(3)) : meta.duration,
          detected_points: (d.metrics || {}).frames_with_target,
          predicted_points: (d.metrics || {}).predicted_boxes,
          lost_spans: (d.metrics || {}).lost_spans || [],
          identity_confidence: (d.metrics || {}).identity_mean,
        })
        const vm = { ...(d.metrics || {}), cached: d.cached, source: 'target' }
        setVerified(vm); verifiedRef.current = vm
        return
      }
      // verification unavailable for this person: keep the indexed track, and say so
      setVerified({ source: 'stored', refining: false,
                    reason: r.detail?.error || null })
    })()
    return () => { alive = false }
  }, [detId])

  // typical sampling interval between stored boxes (used for gap detection)
  const stride = useMemo(() => {
    const pts = path?.points || []
    const diffs = []
    for (let i = 1; i < pts.length; i++) {
      const d = pts[i].offset_seconds - pts[i - 1].offset_seconds
      if (d > 0.001) diffs.push(d)
    }
    if (!diffs.length) return 1.0
    diffs.sort((a, b) => a - b)
    return diffs[Math.floor(diffs.length / 2)] || 1.0   // median
  }, [path])

  const start = path?.start_offset ?? 0
  const end = path?.end_offset ?? 0
  const hasBoxes = (path?.points?.length || 0) > 0
  const playable = !!path?.video_url && /\.(mp4|webm|ogg)(\?|#|$)/i.test(path?.video_url || '')

  // ---- interpolate the box at time t (or null when object not in frame) ----
  const boxAt = useCallback((t) => {
    const pts = path?.points || []
    if (!pts.length) return null
    if (t < pts[0].offset_seconds - stride * 1.5 || t > pts[pts.length - 1].offset_seconds + stride * 1.5) return null
    let idx = 0
    for (let i = 0; i < pts.length; i++) { if (pts[i].offset_seconds <= t + 1e-6) idx = i; else break }
    const p0 = pts[idx]
    const p1 = pts[idx + 1]
    if (!p1) return (t - p0.offset_seconds <= stride * 1.5) ? p0.bbox : null
    const gap = p1.offset_seconds - p0.offset_seconds
    if (gap > stride * 2.5) {                          // object left the frame in this interval
      if (t - p0.offset_seconds <= stride * 1.5) return p0.bbox
      if (p1.offset_seconds - t <= stride * 1.5) return p1.bbox
      return null
    }
    const f = gap > 1e-6 ? (t - p0.offset_seconds) / gap : 0
    return [
      p0.bbox[0] + (p1.bbox[0] - p0.bbox[0]) * f,
      p0.bbox[1] + (p1.bbox[1] - p0.bbox[1]) * f,
      p0.bbox[2] + (p1.bbox[2] - p0.bbox[2]) * f,
      p0.bbox[3] + (p1.bbox[3] - p0.bbox[3]) * f,
    ]
  }, [path, stride])

  // ---- paint the box directly on the DOM (smooth, no per-frame React render) ----
  const paint = useCallback((t) => {
    const el = boxRef.current
    if (!el) return
    const fw = path?.frame_width, fh = path?.frame_height
    const b = (showBoxRef.current && fw && fh) ? boxAt(t) : null
    if (!b) { el.style.display = 'none'; return }
    el.style.display = 'block'
    el.style.left = `${(b[0] / fw) * 100}%`
    el.style.top = `${(b[1] / fh) * 100}%`
    el.style.width = `${(b[2] / fw) * 100}%`
    el.style.height = `${(b[3] / fh) * 100}%`
  }, [boxAt, path])

  // ---- rAF loop keeps box + timeline in sync with playback ----
  useEffect(() => {
    const v = videoRef.current
    if (!v || !playable) return
    const tick = () => {
      const t = v.currentTime
      if (t >= end && !v.paused) { v.pause(); v.currentTime = end }   // stop at track end
      paint(t)
      setCurTime(t)
      if (seekRef.current && !draggingRef.current) seekRef.current.value = String(t)
      rafRef.current = requestAnimationFrame(tick)
    }
    rafRef.current = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(rafRef.current)
  }, [playable, end, paint])

  // ---- seek to track start once the clip is ready ----
  useEffect(() => {
    const v = videoRef.current
    if (!v || !playable) return
    const onReady = () => { try { v.currentTime = start; v.pause() } catch { /* noop */ } paint(start); setCurTime(start) }
    const onPlay = () => setPlaying(true)
    const onPause = () => setPlaying(false)
    v.addEventListener('loadedmetadata', onReady)
    v.addEventListener('play', onPlay)
    v.addEventListener('pause', onPause)
    if (v.readyState >= 1) onReady()
    return () => { v.removeEventListener('loadedmetadata', onReady); v.removeEventListener('play', onPlay); v.removeEventListener('pause', onPause) }
  }, [playable, start, paint])

  useEffect(() => { const v = videoRef.current; if (v) v.playbackRate = speed }, [speed, path])
  useEffect(() => { const esc = (e) => { if (e.key === 'Escape') onClose() }; window.addEventListener('keydown', esc); return () => window.removeEventListener('keydown', esc) }, [onClose])

  // ---- controls ----
  function togglePlay() {
    const v = videoRef.current; if (!v) return
    if (v.paused) { if (v.currentTime < start || v.currentTime >= end - 0.02) v.currentTime = start; v.play() }
    else v.pause()
  }
  function seekTo(t) {
    const v = videoRef.current; if (!v) return
    t = Math.max(start, Math.min(end, t))
    v.currentTime = t; paint(t); setCurTime(t)
  }
  function stepFrame(dir) {
    // step through the stored track samples (the frames the object was tracked in)
    const v = videoRef.current, pts = path?.points || []
    if (!v || !pts.length) return
    v.pause()
    const t = v.currentTime
    let target = null
    if (dir > 0) { for (const p of pts) if (p.offset_seconds > t + 1e-3) { target = p.offset_seconds; break } }
    else { for (let i = pts.length - 1; i >= 0; i--) if (pts[i].offset_seconds < t - 1e-3) { target = pts[i].offset_seconds; break } }
    if (target != null) seekTo(target)
  }

  const attrs = path?.attributes || {}
  const isPerson = (path?.class_label || '') === 'person'

  return (
    <div className="tv-overlay" onMouseDown={onClose}>
      <div className="tv-modal" onMouseDown={(e) => e.stopPropagation()}>
        <div className="tv-head">
          <div className="tv-title">
            <span className="tv-dot" /> Object Tracking
            <span className="tv-sub">{path?.class_label || detection?.class_label || 'object'} · track #{path?.track_id ?? '—'}</span>
          </div>
          <button className="tv-x" onClick={onClose} aria-label="Close">×</button>
        </div>

        <div className="tv-body">
          {/* ---- stage + overlay ---- */}
          <div className="tv-stagewrap">
            {loading ? <div className="tv-msg">Loading tracking data…</div>
              : error ? <div className="tv-msg err">{error}</div>
                : !playable ? <div className="tv-msg">This recording isn’t playable in the browser.</div>
                  : (
                    <>
                      <div className="tv-stage" style={{ aspectRatio: (path.frame_width && path.frame_height) ? `${path.frame_width} / ${path.frame_height}` : '16 / 9' }}>
                        <video ref={videoRef} src={path.video_url} className="tv-video" preload="metadata" playsInline />
                        <div ref={boxRef} className="tv-box" style={{ display: 'none' }}>
                          <span className="tv-box-tag">#{path.track_id} · {path.class_label}</span>
                        </div>
                        {!hasBoxes && <div className="tv-msg abs">No stored boxes for this track.</div>}
                      </div>

                      {/* ---- transport ---- */}
                      <div className="tv-controls">
                        <button className="tv-btn" onClick={() => stepFrame(-1)} title="Previous tracked frame">⏮</button>
                        <button className="tv-btn play" onClick={togglePlay} title="Play / Pause">{playing ? '❚❚' : '►'}</button>
                        <button className="tv-btn" onClick={() => stepFrame(1)} title="Next tracked frame">⏭</button>
                        <label className="tv-chk"><input type="checkbox" checked={showBox} onChange={(e) => setShowBox(e.target.checked)} /> box</label>
                        <div className="tv-speed">
                          {SPEEDS.map((s) => <button key={s} className={speed === s ? 'on' : ''} onClick={() => setSpeed(s)}>{s}×</button>)}
                        </div>
                      </div>

                      {/* ---- timeline (constrained to the tracked segment) ---- */}
                      <div className="tv-timeline">
                        <span className="tv-t start" title="Track start">{fmt(start)}</span>
                        <input ref={seekRef} className="tv-seek" type="range" min={start} max={end} step="0.01" defaultValue={start}
                          onMouseDown={() => { draggingRef.current = true }}
                          onMouseUp={() => { draggingRef.current = false }}
                          onChange={(e) => seekTo(parseFloat(e.target.value))} />
                        <span className="tv-t end" title="Track end">{fmt(end)}</span>
                      </div>
                      <div className="tv-nowline"><span className="tv-now">{fmt(curTime)}</span> / {fmt(end)} <span className="tv-muted">· {path.points.length} tracked frames</span></div>
                    </>
                  )}
          </div>

          {/* ---- track information ---- */}
          <aside className="tv-info">
            <div className="tv-info-h">Track information</div>
            <InfoRow label="Camera" value={path?.camera_name && path.camera_name !== path.camera_id ? `${path.camera_name}` : (path?.camera_id || '—')} sub={path?.camera_name && path.camera_name !== path.camera_id ? path.camera_id : null} />
            <InfoRow label="Track ID" value={path?.track_id != null ? `#${path.track_id}` : '—'} />
            <InfoRow label="Object" value={path?.class_label || '—'} />
            <InfoRow label="Duration" value={path?.duration != null ? `${path.duration.toFixed(1)}s` : '—'} sub={hasBoxes ? `${fmt(start)} → ${fmt(end)}` : null} />
            <InfoRow label="Confidence" value={path?.max_confidence != null ? `${Math.round(path.max_confidence * 100)}% peak` : '—'} sub={path?.avg_confidence != null ? `${Math.round(path.avg_confidence * 100)}% avg` : null} />
            <InfoRow label="Tracked frames" value={hasBoxes ? String(path.points.length) : '0'} />

            {(attrs && Object.keys(attrs).length > 0) && (
              <>
                <div className="tv-info-h">{isPerson ? 'Person attributes' : 'Vehicle attributes'}</div>
                <div className="tv-attrs">
                  {attrChips(attrs).map((a) => <span key={a} className="tv-chip">{a}</span>)}
                  {attrChips(attrs).length === 0 && <span className="tv-muted">No attributes recorded.</span>}
                </div>
              </>
            )}

            {onAddEvidence && detection && (
              <button className={'tv-ev ' + (inEvidence?.(detId) ? 'on' : '')} onClick={() => onAddEvidence(detection)}>
                {inEvidence?.(detId) ? '✓ In evidence' : '＋ Add to evidence'}
              </button>
            )}
            <div className="tv-note">Replayed from stored ByteTrack data — no AI re-run.</div>
          </aside>
        </div>
      </div>
    </div>
  )
}

function InfoRow({ label, value, sub }) {
  return (
    <div className="tv-row">
      <div className="tv-row-l">{label}</div>
      <div className="tv-row-v">{value}{sub ? <span className="tv-row-sub">{sub}</span> : null}</div>
    </div>
  )
}

function attrChips(a) {
  const out = []
  const push = (v) => { if (v && typeof v === 'string' && v !== 'unknown' && v !== 'kind') out.push(v) }
  if (a.upper_color) push(`${a.upper_color} top`)
  if (a.lower_color) push(`${a.lower_color} bottom`)
  if (a.color) push(a.color)
  if (a.gender) push(a.gender)
  if (a.age != null) push(`age ~${a.age}`)
  if (a.vehicle_type) push(a.vehicle_type)
  if (a.carrying) push(a.carrying)
  return out
}

function fmt(s) {
  s = Math.max(0, s || 0)
  const m = Math.floor(s / 60)
  const ss = Math.floor(s % 60)
  const ds = Math.floor((s * 10) % 10)
  return `${m}:${String(ss).padStart(2, '0')}.${ds}`
}
