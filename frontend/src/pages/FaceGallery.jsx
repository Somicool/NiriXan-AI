// Face Gallery - saved best faces + find the same individual across footage.
// Reuses the existing InsightFace face index (no re-detection). Saved faces are
// permanent (server-side) and only removed on explicit delete.
import { useEffect, useMemo, useState } from 'react'
import { listSavedFaces, deleteSavedFace, findSimilarFaces, enhanceFace, getEnhancedFace } from '../api'
import VideoPlayer from '../components/VideoPlayer'
import TrackingViewer from '../components/TrackingViewer'
import { IcFace, IcSearch } from '../components/icons'

const fmtTs = (t) => (t ? t.replace('T', ' ').slice(0, 19) : '—')
const fmtDate = (t) => { try { return new Date(t).toLocaleString() } catch { return t } }

export default function FaceGallery() {
  const [faces, setFaces] = useState(null)
  const [q, setQ] = useState('')
  const [detail, setDetail] = useState(null)     // saved face open in the viewer

  async function load() { setFaces(await listSavedFaces().catch(() => [])) }
  useEffect(() => { load() }, [])

  const filtered = useMemo(() => {
    const list = faces || []
    const s = q.trim().toLowerCase()
    if (!s) return list
    return list.filter((f) => [f.investigation, f.camera_id, f.gender].some((x) => (x || '').toLowerCase().includes(s)))
  }, [faces, q])

  async function onDelete(id) {
    if (!window.confirm('Delete this saved face? This cannot be undone.')) return
    await deleteSavedFace(id)
    if (detail?.saved_id === id) setDetail(null)
    load()
  }

  return (
    <div className="fp-page">
      <div className="fp-page-head">
        <div>
          <h1 className="fp-page-title">Face Gallery</h1>
          <p className="fp-page-desc">Saved faces from your investigations. Find the same individual across all indexed footage.</p>
        </div>
      </div>

      <div className="fp-quicksearch">
        <IcSearch size={20} />
        <input placeholder="Search saved faces — investigation, camera…" value={q} onChange={(e) => setQ(e.target.value)} />
      </div>

      {faces === null ? <div className="dash-empty">Loading saved faces…</div>
        : faces.length === 0 ? (
          <div className="dash-empty">
            No saved faces yet. In the <b>Investigation Workspace</b>, search for people and click
            <b> Save Face</b> on a person result — the clearest face is stored here permanently.
          </div>)
          : filtered.length === 0 ? <div className="dash-empty">No saved faces match “{q}”.</div>
            : (
              <div className="fg-grid">
                {filtered.map((f) => (
                  <div key={f.saved_id} className="fg-card" onClick={() => setDetail(f)} title="Open face">
                    <div className="fg-thumb">
                      {(f.preview_crop_url || f.face_crop_url)
                        ? <img src={f.preview_crop_url || f.face_crop_url} alt="face" loading="lazy" />
                        : <div className="ph"><IcFace size={34} /></div>}
                      {f.confidence != null && <span className="fg-q">Q {Math.round(f.confidence * 100)}</span>}
                      {f.low_quality && <span className="fg-lowq">LOW QUALITY</span>}
                    </div>
                    <div className="fg-body">
                      <div className="fg-inv">{f.investigation || 'Unassigned case'}</div>
                      <div className="fg-meta">{f.camera_id || '—'}</div>
                      <div className="fg-meta mono">{fmtTs(f.timestamp)}</div>
                      <div className="fg-saved">Saved {fmtDate(f.created_at)}</div>
                    </div>
                  </div>
                ))}
              </div>)}

      {detail && <FaceViewer face={detail} onClose={() => setDetail(null)} onDelete={onDelete} />}
    </div>
  )
}

/* ---------------------------- saved-face viewer ---------------------------- */
function FaceViewer({ face, onClose, onDelete }) {
  const [tab, setTab] = useState('view')          // view | enhance | similar
  const [sim, setSim] = useState(null)
  const [loading, setLoading] = useState(false)
  const [jump, setJump] = useState(null)          // similar result to play
  const [track, setTrack] = useState(null)        // detection to track

  useEffect(() => { const esc = (e) => { if (e.key === 'Escape') onClose() }; window.addEventListener('keydown', esc); return () => window.removeEventListener('keydown', esc) }, [onClose])

  async function findSimilar() {
    setTab('similar'); setLoading(true)
    try { setSim(await findSimilarFaces(face.saved_id, 60)) } catch { setSim({ results: [] }) } finally { setLoading(false) }
  }

  function exportFace() {
    // client-side export of the saved-face evidence (image link + JSON record)
    const rec = { ...face }; delete rec.embedding
    const blob = new Blob([JSON.stringify(rec, null, 2)], { type: 'application/json' })
    const a = document.createElement('a')
    a.href = URL.createObjectURL(blob); a.download = `face_${face.saved_id}.json`; a.click()
    if (face.face_crop_url) window.open(face.face_crop_url, '_blank')
  }

  return (
    <div className="vi-overlay" onMouseDown={onClose}>
      <div className="vi-modal fg-modal" onMouseDown={(e) => e.stopPropagation()}>
        <div className="vi-head">
          <div className="vi-title"><span className="vi-badge">FACE</span> Saved Face
            <span className="vi-plate" style={{ background: 'var(--fp-accent)', color: '#04121a' }}>#{face.saved_id}</span>
          </div>
          <div className="vi-head-actions">
            <button className={'fp-btn sm ' + (tab === 'view' ? 'primary' : '')} onClick={() => setTab('view')}>View Face</button>
            <button className={'fp-btn sm ' + (tab === 'enhance' ? 'primary' : '')} onClick={() => setTab('enhance')}>Enhance Face</button>
            <button className={'fp-btn sm ' + (tab === 'similar' ? 'primary' : '')} onClick={findSimilar}>Find Similar Person</button>
            <button className="fp-btn sm" onClick={exportFace}>Export</button>
            <button className="fp-btn sm" onClick={() => onDelete(face.saved_id)} style={{ borderColor: 'var(--fp-danger)', color: '#ffb3bb' }}>Delete</button>
            <button className="vi-x" onClick={onClose}>×</button>
          </div>
        </div>

        <div className="vi-body">
          {tab === 'view' ? (
            <div className="fg-view">
              <div>
                <img className="fg-view-face" src={face.preview_crop_url || face.face_crop_url || face.person_crop_url} alt="face" />
                {face.low_quality && <div className="fg-lowq-note">⚠ Low-quality face — best available in this track</div>}
              </div>
              <div className="fg-view-info">
                <InfoRow k="Investigation" v={face.investigation || '—'} />
                <InfoRow k="Camera" v={face.camera_id || '—'} />
                <InfoRow k="Timestamp" v={fmtTs(face.timestamp)} />
                <InfoRow k="Age (est.)" v={face.age ?? '—'} />
                <InfoRow k="Quality score" v={face.confidence != null ? `${Math.round(face.confidence * 100)}%` : '—'} />
                <InfoRow k="Saved" v={fmtDate(face.created_at)} />
                {face.metrics && (
                  <>
                    <div className="vi-group-h" style={{ marginTop: 14 }}>Quality breakdown</div>
                    <div className="fg-metrics">
                      {[['det_score', 'Confidence'], ['sharpness', 'Sharpness'], ['frontal', 'Frontal pose'],
                        ['eyes', 'Eyes visible'], ['brightness', 'Brightness'], ['occlusion', 'Visibility'],
                        ['noise', 'Low noise']].map(([k, l]) => (
                          face.metrics[k] != null && <Meter key={k} label={l} v={face.metrics[k]} />))}
                      <div className="fg-mrow"><span>Face size</span><b>{face.metrics.face_size ?? '—'} px</b></div>
                      <div className="fg-mrow"><span>Resolution</span><b>{face.metrics.resolution ?? '—'} px²</b></div>
                      <div className="fg-mrow"><span>Frames inspected</span><b>{face.metrics.frames_seen ?? '—'}</b></div>
                      <div className="fg-mrow"><span>Faces ranked</span><b>{face.metrics.faces_seen ?? '—'}</b></div>
                    </div>
                  </>
                )}
                {face.person_crop_url && <img className="fg-person" src={face.person_crop_url} alt="person profile" title="Person profile image" />}
              </div>
            </div>
          ) : tab === 'enhance' ? (
            <EnhancePanel face={face} />
          ) : (
            <div className="fg-similar">
              {loading ? <div className="vi-msg">Searching all indexed faces…</div>
                : !sim || !sim.results?.length ? <div className="vi-msg">No similar person found in indexed footage.</div>
                  : (
                    <>
                      <div className="fg-sim-h">{sim.total} match{sim.total === 1 ? '' : 'es'} · sorted by similarity</div>
                      <div className="fg-sim-list">
                        {sim.results.map((r) => (
                          <div className="fg-sim" key={r.face_id}>
                            <div className="fg-sim-imgs">
                              {r.face_crop_url && <img src={r.face_crop_url} alt="" />}
                            </div>
                            <div className="fg-sim-info">
                              <div className="fg-sim-top">
                                <span className="cam">{r.camera_name || r.camera_id}</span>
                                <span className="sim">{Math.round(r.similarity * 100)}%</span>
                              </div>
                              <div className="fg-sim-ts mono">{fmtTs(r.timestamp)}</div>
                              <div className="fg-sim-actions">
                                {r.video_url && <button className="ws-btn-sm" onClick={() => setJump(r)}>⤿ Jump to Video</button>}
                                {r.detection_id != null && <button className="ws-btn-sm" onClick={() => setTrack(r)}>⤳ Track Person</button>}
                              </div>
                            </div>
                          </div>
                        ))}
                      </div>
                    </>)}
            </div>
          )}
        </div>
      </div>

      {jump && (
        <div className="vi-overlay" onMouseDown={() => setJump(null)} style={{ zIndex: 1100 }}>
          <div className="vi-modal" style={{ width: 'min(880px,96vw)' }} onMouseDown={(e) => e.stopPropagation()}>
            <div className="vi-head"><div className="vi-title">Jump to Video · {jump.camera_name || jump.camera_id}</div>
              <button className="vi-x" onClick={() => setJump(null)}>×</button></div>
            <div className="vi-body">
              <VideoPlayer key={jump.detection_id} src={jump.video_url} offset={jump.offset_seconds}
                           bbox={jump.bbox} frameW={jump.frame_width} frameH={jump.frame_height} autoPlay />
            </div>
          </div>
        </div>
      )}
      {track && <TrackingViewer detection={{ detection_id: track.detection_id, class_label: 'person', camera_id: track.camera_id, attributes: {} }} onClose={() => setTrack(null)} />}
    </div>
  )
}

/* ------------------------- Face Enhancement (derived) -------------------------
   The original saved crop stays on the left, untouched, at all times. The right
   side is explicitly labelled a DERIVED visualisation and never presented as the
   subject's real face. A refusal is shown as a result, not as an error. */
function EnhancePanel({ face }) {
  const [enh, setEnh] = useState(null)          // stored/produced result
  const [busy, setBusy] = useState(false)
  const [refusal, setRefusal] = useState(null)  // 422 payload from the backend
  const [mode, setMode] = useState('side')      // side | slider
  const [split, setSplit] = useState(50)
  const [openFrames, setOpenFrames] = useState(false)
  const [loaded, setLoaded] = useState(false)

  const original = face.preview_crop_url || face.face_crop_url || face.person_crop_url

  // Opening the tab only LOOKS for an existing result. Nothing is processed until
  // the officer asks for it (on-demand requirement).
  useEffect(() => {
    let alive = true
    getEnhancedFace(face.saved_id)
      .then((r) => { if (alive) setEnh(r?.available ? r : null) })
      .catch(() => {})
      .finally(() => { if (alive) setLoaded(true) })
    return () => { alive = false }
  }, [face.saved_id])

  async function run(force) {
    setBusy(true); setRefusal(null)
    const r = await enhanceFace(face.saved_id, force)
    if (r.ok) { setEnh(r.data); setOpenFrames(false) } else { setRefusal(r.detail); setEnh(null) }
    setBusy(false)
  }

  function saveEnhanced() {
    // The result is already stored server-side; this takes a local copy of the
    // derived image plus its full provenance record.
    const rec = { ...enh }; delete rec._aligned
    const blob = new Blob([JSON.stringify(rec, null, 2)], { type: 'application/json' })
    const a = document.createElement('a')
    a.href = URL.createObjectURL(blob); a.download = `enhanced_face_${face.saved_id}.json`; a.click()
    if (enh.enhanced_url) window.open(enh.enhanced_url, '_blank')
  }

  const m = enh?.metrics || {}
  const t = m.track_gain || {}
  const chosen = (m.gate_report || []).find((r) => r.passed && r.name === enh?.model_name)

  return (
    <div className="fe-wrap">
      <div className="fe-actions">
        <button className="fp-btn sm primary" onClick={() => run(false)} disabled={busy}>
          {busy ? 'Analysing source footage…' : enh ? 'Re-run Enhancement' : 'Enhance Face'}
        </button>
        {enh && <button className="fp-btn sm" onClick={() => run(true)} disabled={busy}>Force Re-enhance</button>}
        {enh && <button className="fp-btn sm" onClick={saveEnhanced}>Save Enhanced Face</button>}
        {original && <button className="fp-btn sm" onClick={() => window.open(original, '_blank')}>View Source</button>}
        {enh && (
          <button className="fp-btn sm" onClick={() => setMode(mode === 'side' ? 'slider' : 'side')}>
            {mode === 'side' ? 'Compare with slider' : 'Side by side'}
          </button>
        )}
      </div>

      {busy && (
        <div className="fe-busy">
          Re-reading the original recording, collecting every face on this person's track,
          re-sampling between the indexed frames at native rate, verifying each appearance
          against the saved identity and testing whether any enhancement actually helps.
          This takes a few seconds and is never run during ingestion.
        </div>
      )}

      {refusal && (
        <div className="fe-refuse">
          <b>{refusal.error}</b>
          <div className="fe-refuse-d">
            {refusal.reason ? <>Reason: {refusal.reason}. </> : null}
            Frames analysed {refusal.frames_analysed ?? 0}, faces found {refusal.faces_found ?? 0},
            passed verification {refusal.frames_selected ?? 0}.
            {refusal.rejected && (
              <> Rejected — too small {refusal.rejected.size ?? 0}, occluded {refusal.rejected.occluded ?? 0},
                low quality {refusal.rejected.quality ?? 0}, identity mismatch {refusal.rejected.identity ?? 0}.</>
            )}
            <div style={{ marginTop: 6 }}>No image is produced when the footage does not contain
              enough facial information — a fabricated face would not be evidence.</div>
          </div>
        </div>
      )}

      {!enh && !busy && !refusal && (
        <div className="fe-intro">
          {loaded ? <>This does not sharpen the stored crop. It re-reads the original
            recording, finds every appearance of this tracked person — including the frames
            between the indexed samples, which nothing has examined before — verifies each
            against the saved identity and picks the <b>clearest real frame</b>. Restoration
            is only applied if it measurably beats that frame.</>
            : 'Checking for an existing result…'}
        </div>
      )}

      {enh && (
        <>
          {mode === 'side' ? (
            <div className="fe-three">
              <figure>
                <figcaption>1 · ORIGINAL SAVED FACE</figcaption>
                <img src={original} alt="original saved face" />
                <span className="fe-tag orig">Unmodified evidence</span>
              </figure>
              <figure>
                <figcaption>2 · BEST SOURCE FRAME FOUND</figcaption>
                <img src={enh.best_source_url || original} alt="clearest frame found in the track" />
                <span className="fe-tag natural">
                  Real frame {enh.best_source_frame}{m.best_from_refined ? ' · unindexed' : ''}
                </span>
              </figure>
              <figure>
                <figcaption>3 · AI-ENHANCED</figcaption>
                {enh.enhancement_applied ? (
                  <>
                    <img src={enh.enhanced_url} alt="enhanced derived visualisation" />
                    <span className="fe-tag derived">{enh.label}</span>
                  </>
                ) : (
                  <div className="fe-declined">
                    <b>Not applied</b>
                    <span>AI enhancement did not improve the verified source image,
                      so the real frame is shown instead.</span>
                  </div>
                )}
              </figure>
            </div>
          ) : (
            <div className="fe-slider-wrap">
              <div className="fe-slider" style={{ '--split': split + '%' }}>
                <img className="a" src={original} alt="original saved face" />
                <img className="b" src={enh.enhanced_url} alt="result of the analysis" />
                <span className="fe-handle" />
                <span className="fe-lab l">SAVED</span>
                <span className="fe-lab r">{enh.enhancement_applied ? 'AI-ENHANCED' : 'BEST SOURCE'}</span>
              </div>
              <input type="range" min="0" max="100" value={split} aria-label="Compare original and enhanced"
                     onChange={(e) => setSplit(Number(e.target.value))} />
              <div className="fe-tag derived" style={{ position: 'static', marginTop: 8 }}>{enh.label}</div>
            </div>
          )}

          <div className={'fe-verdict ' + (enh.enhancement_applied ? 'ok' : 'natural')}>
            <div className="fe-verdict-h">
              Status: {enh.gate_status || enh.label}
              <span className="fe-applied">Enhancement applied: {enh.enhancement_applied ? 'Yes' : 'No'}</span>
            </div>
            {!enh.enhancement_applied && (
              <div className="fe-verdict-r">
                Reason: AI enhancement did not improve the verified source image.
                The clearest real frame from this person's track is shown instead.
              </div>
            )}
          </div>

          <div className="fe-panel">
            <div className="vi-group-h">Track analysis</div>
            <InfoRow k="Faces analysed" v={m.faces_found ?? '—'} />
            <InfoRow k="Verified as this person" v={m.faces_verified ?? '—'} />
            <InfoRow k="Frames examined" v={m.frames_analysed != null
              ? `${m.frames_analysed} (${m.frames_sampled_pass} indexed + ${m.frames_refined_pass} re-read at native rate)`
              : '—'} />
            <InfoRow k="Best source frame" v={enh.best_source_frame != null
              ? `frame ${enh.best_source_frame}${m.best_from_refined ? ' — never indexed until now' : ''}` : '—'} />
            <InfoRow k="Best source timestamp" v={fmtTs(enh.best_source_timestamp)} />
            <InfoRow k="Best source quality score" v={enh.best_source_quality != null
              ? `${enh.best_source_quality} (${Math.round(enh.best_source_quality * 100)}%)` : '—'} />
            <InfoRow k="Source quality" v={<QualityPill q={enh.source_quality} px={m.source_face_px} />} />
            <InfoRow k="Versus the saved frame" v={!t.better_frame_found
              ? 'Same frame — the saved one is already the best view in this track'
              : <span>
                  <b style={{ color: 'var(--fp-success)' }}>Different frame selected</b>
                  {' — '}face {t.saved_frame_px} px → {m.source_face_px} px,
                  quality {t.saved_frame_quality} → {m.quality_best}
                  {t.px_gain > 0 && t.quality_gain < 0 &&
                    <span className="fe-dim"> (larger face preferred over a marginally
                      higher composite score)</span>}
                </span>} />
            <InfoRow k="Native crop" v={m.natural_px || '—'} />
            <InfoRow k="Enhancement method" v={enh.model_name || '—'} />
            {enh.enhancement_applied && (
              <InfoRow k="Enhancement type" v={m.enhancement_kind === 'generative'
                ? <span style={{ color: 'var(--fp-warn)' }}>Generative reconstruction — pixels are
                    synthesised by a trained model, not recovered from the footage</span>
                : 'Multi-frame — built only from real pixels across several views'} />
            )}
            {enh.enhancement_applied && chosen && (
              <InfoRow k="Reconstruction vs saved face" v={<span>
                <b style={{ color: chosen.identity >= 0.7 ? 'var(--fp-success)' : 'var(--fp-warn)' }}>
                  {chosen.identity} cosine
                </b>
                {chosen.identity_drop != null &&
                  <span className="fe-dim"> — moved {chosen.identity_drop} from the real frame</span>}
              </span>} />
            )}
            <InfoRow k="Identity agreement" v={m.identity_min != null
              ? `${m.identity_min.toFixed(2)} – ${m.identity_max.toFixed(2)} cosine` : '—'} />
            <InfoRow k="Processing time" v={m.elapsed_s != null ? `${m.elapsed_s} s` : '—'} />

            {(m.gate_report || []).length > 0 && (
              <>
                <div className="vi-group-h" style={{ marginTop: 14 }}>Quality gate</div>
                <div className="fe-note" style={{ marginTop: 0, marginBottom: 8 }}>
                  Every candidate is measured against the best verified source frame at
                  identical size. Ratios below 100% mean the candidate lost detail.
                </div>
                <table className="fe-gate">
                  <thead><tr><th>Candidate</th><th>Edge</th><th>Sharp</th><th>Identity</th><th /></tr></thead>
                  <tbody>
                    {m.gate_report.map((r, i) => (
                      <tr key={i} className={r.passed ? 'pass' : 'fail'}>
                        <td>{r.name}{r.kind === 'generative' && <span className="fe-dim"> · generative</span>}</td>
                        <td>{Math.round((r.edge_ratio || 0) * 100)}%</td>
                        <td>{Math.round((r.sharpness_ratio || 0) * 100)}%</td>
                        <td>{r.identity != null ? r.identity.toFixed(2) : '—'}</td>
                        <td>{r.passed ? 'accepted' : 'rejected'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                {m.gate_report.filter((r) => !r.passed).map((r, i) => (
                  <div className="fe-note" key={'w' + i}>{r.name}: {r.failed_because.join('; ')}</div>
                ))}
              </>
            )}
            {/* Only shown when fusion was actually ACCEPTED. Reporting a noise
                reduction from a fusion the gate threw away implies a benefit that
                was never taken. */}
            {enh.enhancement_applied && m.noise_reduction_pct != null && (
              <>
                <div className="vi-group-h" style={{ marginTop: 14 }}>Multi-frame measurement</div>
                <InfoRow k="Noise, single frame" v={`${m.noise_single_frame} grey levels`} />
                <InfoRow k="Noise after fusion" v={`${m.noise_fused_halfsplit} grey levels`} />
                <InfoRow k="Noise reduction" v={`${m.noise_reduction_pct}%`} />
              </>
            )}
            {m.note && <div className="fe-note">{m.note}</div>}

            <div className="vi-group-h" style={{ marginTop: 14 }}>Chain of custody</div>
            <InfoRow k="Source video" v={enh.source_video_id != null ? `video #${enh.source_video_id}` : '—'} />
            <InfoRow k="Source track" v={enh.source_track_id != null ? `track ${enh.source_track_id}` : '—'} />
            <InfoRow k="Source camera" v={enh.source_camera_id || '—'} />
            <InfoRow k="Original SHA-256" v={<code className="fe-hash">{enh.original_hash || 'original file not on disk'}</code>} />
            <InfoRow k="Best source SHA-256" v={<code className="fe-hash">{enh.best_source_hash || '—'}</code>} />
            <InfoRow k="Result SHA-256" v={<code className="fe-hash">{enh.enhanced_hash || '—'}</code>} />
            <InfoRow k="Enhanced at" v={fmtDate(enh.created_at)} />

            <button className="fe-frames-h" onClick={() => setOpenFrames(!openFrames)}>
              {openFrames ? '▾' : '▸'} Source Frames Used ({(enh.source_frames || []).length})
            </button>
            {openFrames && (
              <div className="fe-frames">
                {(enh.source_frames || []).map((f, i) => (
                  <div className="fe-frame" key={f.detection_id + '-' + i}>
                    <img src={f.url} alt={`source frame ${f.frame_number}`} />
                    <div className="fe-frame-m">
                      <b>Frame {f.frame_number}</b>
                      <span className="mono">{fmtTs(f.timestamp)}</span>
                      <span>Quality {Math.round((f.quality || 0) * 100)} · {f.face_size} px</span>
                      <span>Identity {f.identity != null ? f.identity.toFixed(2) : '—'} · conf {f.det_score}</span>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>

          <div className="fe-warn">
            {enh.enhancement_applied ? (
              m.enhancement_kind === 'generative' ? (
                <>⚠ The third image is a <b>machine reconstruction</b>, not a photograph.
                  A model trained on sharp faces has drawn a plausible face consistent with
                  a {m.source_face_px}&nbsp;px source — so most of the fine detail you see
                  was <b>generated, not captured</b>. It scores {chosen?.identity} against the
                  saved face, above the {m.gate_base ? '0.50' : '0.50'} same-person floor but
                  clearly moved from it. Use it to guide a search, never as identification.
                  The middle image is the real evidence.</>
              ) : (
                <>⚠ The third image is a <b>derived visualisation</b> built from
                  {' '}{enh.frames_selected} verified view{enh.frames_selected === 1 ? '' : 's'} of
                  a {m.source_face_px}&nbsp;px face. Every pixel comes from real frames, but
                  treat it as an investigative aid rather than identification on its own.</>
              )
            ) : (
              <>✓ No AI reconstruction was used. The result is a <b>real frame</b> from the
                recording at {m.natural_px}, chosen from {m.faces_verified} appearances
                verified as this person. Restoration was measured against it and rejected
                for losing detail, so nothing here is invented.</>
            )}
            {' '}The original saved crop is unaltered.
          </div>
        </>
      )}
    </div>
  )
}

function QualityPill({ q, px }) {
  const col = q === 'High' ? 'var(--fp-success)' : q === 'Medium' ? 'var(--fp-warn)' : '#ff8a94'
  return <span><b style={{ color: col }}>{q || '—'}</b>{px ? <span className="fe-dim"> · {px} px face in source</span> : null}</span>
}

function InfoRow({ k, v }) {
  return <div className="vi-row"><div className="vi-k">{k}</div><div className="vi-v">{v}</div></div>
}

function Meter({ label, v }) {
  const pct = Math.max(0, Math.min(100, Math.round((v || 0) * 100)))
  const col = pct >= 70 ? 'var(--fp-success)' : pct >= 40 ? 'var(--fp-warn)' : '#ff8a94'
  return (
    <div className="fg-mrow">
      <span>{label}</span>
      <span className="fg-bar"><i style={{ width: pct + '%', background: col }} /></span>
      <b>{pct}</b>
    </div>
  )
}
