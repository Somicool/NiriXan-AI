"""Track Person - query-time, target-specific re-tracking of ONE selected identity.

WHY THIS EXISTS
---------------
Ingestion still owns detection, tracking, embeddings, crops and search. Nothing
here changes any of that. But the stored ByteTrack track is not good enough to
drive a bounding-box overlay, and the reasons were measured on this project's own
footage rather than assumed:

  1. TEMPORAL SPARSITY - the dominant cause. Detections are stored at ~2 FPS
     (median stride 0.50 s) while the clips play at 20 FPS. So 9 of every 10
     rendered frames had no real box, and the viewer drew a straight line between
     two samples half a second apart. A walking person covers real ground in
     0.5 s, so that line drifts off the target and passes through whoever else is
     in between. This looks exactly like an identity switch but is a rendering
     artefact of sparse data.
  2. A CLOSED CANDIDATE POOL - only detections carrying the stored track_id were
     ever considered. When the identity check correctly rejected a frame (the
     stored track was holding someone else there), there was no way to find the
     target's REAL box in that frame. A hole appeared, and the hole got
     interpolated across.
  3. ID REUSE - one stored track_id is not always one person. Track 89 on video
     122 spans a 132.5 s internal gap; track 100070 spans 49.0 s.

Two hypotheses were tested and REJECTED, so they are not what this fixes:
malformed ReID vectors (measured 0% malformed, 94.4% usable across 8,988
detections) and track fragmentation (1 likely pair across 733 tracks).

WHAT THIS DOES INSTEAD
----------------------
    stored result -> multi-view reference identity -> re-detect people in the
    ORIGINAL video at TRACK_TARGET_FPS -> score every candidate against the
    reference -> lock, hold, or declare temporarily lost

The guiding rule throughout: identity outranks motion. Motion continuity may
SUPPORT a match but may never create one, so "this is the closest box, therefore
it is the target" can never happen. When the evidence is thin the overlay shows
nothing and says so, because a missing box is honest and a box on the wrong
person is not.
"""
from __future__ import annotations

import json
import math
import time
from datetime import datetime, timezone

import cv2
import numpy as np

from . import config, database, faces_gallery
from .search import track_path, vector_store


# --------------------------------------------------------------------- helpers
def _unit(v):
    if v is None:
        return None
    v = np.asarray(v, dtype="float32").ravel()
    n = float(np.linalg.norm(v))
    return None if n < 1e-6 else v / n


def _centre(b):
    return (b[0] + b[2] / 2.0, b[1] + b[3] / 2.0)


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a[0], a[1], a[0] + a[2], a[1] + a[3]
    bx1, by1, bx2, by2 = b[0], b[1], b[0] + b[2], b[1] + b[3]
    ix, iy = max(0.0, min(ax2, bx2) - max(ax1, bx1)), max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    return float(inter / (a[2] * a[3] + b[2] * b[3] - inter + 1e-9))


# ------------------------------------------------------------------- reference
def build_reference(detection_id: int) -> dict:
    """Multi-view identity for the SELECTED person.

    Reuses track_path.reference_views, which already picks up to five sharp,
    well-sized, visually varied views of the chosen track and anchors them on the
    detection the investigator actually clicked. Those stored ReID embeddings are
    the identity.

    Clothing colour is deliberately NOT part of this. Auditing the colour data
    found 69% of detections labelled grey/white/silver and 74% of tops equal to
    their bottoms, so a colour term would inject noise, not evidence. ReID is the
    primary signal; a face embedding is added as corroboration when the gallery
    already holds one for this track.
    """
    refs = database.get_detections([detection_id])
    if not refs:
        return {}
    ref = refs[0]
    vid, tid = ref.get("video_id"), ref.get("track_id")
    rows = database.get_track_detections(vid, tid) or [ref]

    view_ids, _diag = track_path.reference_views(rows, detection_id)
    vecs = []
    for did in view_ids:
        u = _unit(vector_store.get_vector("reid", did))
        if u is not None and u.shape[0] == config.REID_DIM:
            vecs.append(u)
    if not vecs:                     # fall back to the whole track's usable vectors
        for r in rows:
            u = _unit(vector_store.get_vector("reid", r["detection_id"]))
            if u is not None and u.shape[0] == config.REID_DIM:
                vecs.append(u)
            if len(vecs) >= config.REID_DIM:
                break
    if not vecs:
        return {}

    M = np.vstack(vecs).astype("float32")
    centroid = _unit(M.mean(axis=0))
    # a face embedding for this track, if Save Face already computed one
    face = None
    try:
        with database.get_conn() as conn:
            row = conn.execute(
                "SELECT embedding FROM saved_faces WHERE detection_id IN "
                "(SELECT detection_id FROM detections WHERE video_id=? AND track_id=?) "
                "AND embedding IS NOT NULL LIMIT 1", (vid, tid)).fetchone()
        if row:
            import base64
            face = _unit(np.frombuffer(base64.b64decode(row["embedding"]), dtype="float32"))
    except Exception:
        face = None

    # Where to look. Split the stored detections into APPEARANCES at long gaps and
    # keep only the one the officer clicked - a stored track_id can cover two
    # separate appearances (or two people) minutes apart, and analysing that whole
    # span forces the sampling stride back down to the stored density.
    vindex = track_path._video_index()
    stamped = []
    for r in rows:
        pb = track_path.playback_fields(r, vindex)
        if pb.get("offset_seconds") is not None:
            stamped.append((pb["offset_seconds"], r["detection_id"]))
    stamped.sort()
    seg_from = seg_to = None
    segments = []
    if stamped:
        cur = [stamped[0]]
        for prev, nxt in zip(stamped, stamped[1:]):
            if nxt[0] - prev[0] > config.TRACK_TARGET_SEGMENT_GAP_S:
                segments.append(cur); cur = [nxt]
            else:
                cur.append(nxt)
        segments.append(cur)
        chosen_seg = next((s for s in segments
                           if any(d == detection_id for _o, d in s)), segments[0])
        seg_from, seg_to = chosen_seg[0][0], chosen_seg[-1][0]

    # On-screen height range of the target across the clicked appearance, used to
    # skip candidates that are obviously at a different distance (see config).
    seg_ids = {d for _o, d in (chosen_seg if stamped else [])}
    heights = [r["bbox_h"] for r in rows
               if r.get("bbox_h") and r["detection_id"] in seg_ids]
    if not heights:
        heights = [r["bbox_h"] for r in rows if r.get("bbox_h")]
    h_lo = h_hi = None
    if heights:
        h_lo = min(heights) * config.TRACK_TARGET_H_TOL_LO
        h_hi = max(heights) * config.TRACK_TARGET_H_TOL_HI

    return {"views": M, "centroid": centroid, "view_ids": view_ids, "face": face,
            "ref": ref, "video_id": vid, "track_id": tid,
            "known_from": seg_from, "known_to": seg_to,
            "h_lo": h_lo, "h_hi": h_hi,
            "appearances": len(segments),
            "track_span": [stamped[0][0], stamped[-1][0]] if stamped else None}


def _similarity(reference: dict, emb) -> float:
    """Best-of-set similarity, softened by the set mean.

    Best-of-set alone lets one lucky reference view carry a poor match; the mean
    alone punishes legitimate pose changes. 0.7/0.3 is the same blend the stored
    path already used, kept so the two agree."""
    u = _unit(emb)
    if u is None:
        return 0.0
    s = reference["views"] @ u
    return float(0.7 * s.max() + 0.3 * s.mean()) if s.size > 1 else float(s.max())


# ----------------------------------------------------------------- the pass
def iter_window(cap, start: int, stop: int, step: int):
    """Yield (frame_number, frame) across a window, decoding SEQUENTIALLY.

    One seek to the window start, then plain reads with cheap grab() skips. This
    matters enormously: faces_gallery._FrameReader seeks per frame, which costs
    ~128 ms on these 1080p clips because the decoder re-finds the preceding
    keyframe every time. At 6 FPS over a 100 s window that is ~78 s of pure
    seeking, which made the first version of this pass unusably slow.

    Sequential reading is also MORE frame-exact here, not less. The known problem
    with decode-skipping is deriving a skip count from a post-seek position; this
    counts every frame it consumes itself, so the numbering cannot drift."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(max(0, start)))
    try:                                  # a seek may land on a nearby keyframe
        pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    except Exception:
        pos = int(max(0, start))
    while pos <= stop:
        got, frame = cap.read()
        if not got or frame is None:
            return
        yield pos, frame
        pos += 1
        for _ in range(step - 1):         # skip without decoding to BGR
            if not cap.grab():
                return
            pos += 1


def _analyse_frames(cap, start, stop, step, reference, fw, fh, native_fps):
    """Detect people on each sampled frame and score them against the target.

    Returns one record per analysed frame holding EVERY candidate's similarity, so
    the decision that follows can be audited rather than trusted.

    Frames are processed in BATCHES. One predict() per frame was the second big
    cost after seeking - the GPU sat idle between calls - so a chunk of frames goes
    through YOLO in a single call and every crop from that chunk goes through ReID
    in one more. Same results, far fewer round-trips."""
    from .ingestion import detector, reid_embedder

    model = detector.get_model()
    wanted = set(config.PERSON_CLASSES)
    batch = max(1, int(config.TRACK_TARGET_BATCH))
    out = []
    _t0 = time.time()
    pending: list[tuple[int, "np.ndarray"]] = []

    def flush():
        if not pending:
            return
        frames = [f for _n, f in pending]
        results = model.predict(frames, conf=config.TRACK_TARGET_DET_CONF,
                                device=config.DEVICE, imgsz=config.TRACK_TARGET_IMGSZ,
                                classes=sorted(wanted), verbose=False)
        crops, owner = [], []
        h_lo, h_hi = reference.get("h_lo"), reference.get("h_hi")
        for k, (r, (fno, frame)) in enumerate(zip(results, pending)):
            H, W = frame.shape[:2]
            if r.boxes is None:
                continue
            boxes = []
            for b in r.boxes:
                if int(b.cls[0]) not in wanted:
                    continue
                x1, y1, x2, y2 = (float(v) for v in b.xyxy[0].tolist())
                cx1, cy1 = max(0, int(x1)), max(0, int(y1))
                cx2, cy2 = min(W, int(x2)), min(H, int(y2))
                if cx2 - cx1 < 12 or cy2 - cy1 < 24:
                    continue
                boxes.append((cx1, cy1, cx2, cy2, round(float(b.conf[0]), 4)))
            # SIZE gate: skip people at an implausible distance before paying for
            # their embedding. Falls back to the largest few if it rejects all.
            if h_lo is not None and h_hi is not None and boxes:
                keep = [bx for bx in boxes if h_lo <= (bx[3] - bx[1]) <= h_hi]
                if len(keep) < config.TRACK_TARGET_MIN_CANDS:
                    extra = sorted((bx for bx in boxes if bx not in keep),
                                   key=lambda bx: bx[3] - bx[1], reverse=True)
                    keep += extra[:config.TRACK_TARGET_MIN_CANDS - len(keep)]
                boxes = keep
            for cx1, cy1, cx2, cy2, conf in boxes:
                c = frame[cy1:cy2, cx1:cx2]
                if c is None or not c.size:
                    continue
                crops.append(c)
                owner.append((k, [float(cx1), float(cy1),
                                  float(cx2 - cx1), float(cy2 - cy1)], conf))
        embs = reid_embedder.embed_persons(crops) if crops else []
        per_frame: dict[int, list] = {k: [] for k in range(len(pending))}
        for (k, box, conf), emb in zip(owner, embs):
            per_frame[k].append({"bbox": box, "det_conf": conf,
                                 "sim": round(_similarity(reference, emb), 4)})
        for k, (fno, _f) in enumerate(pending):
            cands = sorted(per_frame.get(k, []), key=lambda c: c["sim"], reverse=True)
            out.append({"frame": fno,
                        "offset": round(fno / native_fps, 3) if native_fps else None,
                        "candidates": cands})
        pending.clear()

    for fno, frame in iter_window(cap, start, stop, step):
        pending.append((fno, frame))
        if len(pending) >= batch:
            flush()
            if config.TRACK_TARGET_DEBUG and len(out) % 120 < batch:
                print(f"[track_target] {len(out)} frames analysed "
                      f"({time.time() - _t0:.0f}s)", flush=True)
    flush()
    out.sort(key=lambda r: r["frame"])
    return out


def _decide(records, fw, fh, stride_s):
    """Target-lock state machine over the scored candidates.

    Per frame the best candidate is accepted only if the identity evidence
    supports it at the level the current state demands:

      CONFIRMED    sim >= CONFIRM_SIM and it beats the runner-up by MIN_MARGIN
      CONTINUED    sim >= CONTINUE_SIM, motion is plausible from the last accepted
                   box, and the margin holds - medium confidence, held not claimed
      REACQUIRED   the target was lost; needs REACQUIRE_SIM, which is strictly
                   higher than continuation
      LOST         nothing cleared the bar, or two candidates were too close to
                   separate. No box is emitted.

    Motion is only ever a veto or a supporting condition. There is no branch in
    which proximity alone produces a box, so a passer-by cannot inherit the
    overlay. Where the margin is too small the previous target is kept ONLY if
    identity still supports it; otherwise the frame is marked lost.
    """
    diag = math.hypot(fw or 1920, fh or 1080)
    max_step = config.TRACK_TARGET_MAX_SPEED_DIAG * diag * max(stride_s, 1e-3)

    points, debug = [], []
    last_box = last_off = None
    misses = 0
    locked = False
    reacquisitions = 0

    for rec in records:
        cands = rec["candidates"]
        best = cands[0] if cands else None
        runner = cands[1] if len(cands) > 1 else None
        margin = round((best["sim"] - runner["sim"]), 4) if (best and runner) else None
        off = rec["offset"]

        status = "lost"
        reason = "no person detected" if not cands else None
        chosen = None

        if best is not None:
            motion_ok = True
            travelled = None
            if last_box is not None and last_off is not None and off is not None:
                dt = max(off - last_off, 1e-3)
                ca, cb = _centre(last_box), _centre(best["bbox"])
                travelled = math.hypot(cb[0] - ca[0], cb[1] - ca[1])
                motion_ok = travelled <= max_step * (dt / max(stride_s, 1e-3))
            # Ambiguity: two candidates too close to separate on identity alone.
            # Rather than veto the frame outright (a flat veto cost ~half of all
            # frames, since the measured median margin is only 0.065), let spatial
            # continuity with the last CONFIRMED box settle which of the two is the
            # target. This is the one place overlap is consulted, and it only ever
            # chooses BETWEEN candidates that already cleared an identity bar - it
            # can never conjure a target where identity says there is none.
            tight = (margin is not None and margin < config.TRACK_TARGET_MIN_MARGIN)
            tie_ok = False
            if tight and last_box is not None:
                tie_ok = _iou(last_box, best["bbox"]) >= config.TRACK_TARGET_TIE_IOU
                if tie_ok and runner is not None:
                    # if the runner-up overlaps the previous box MORE, the best
                    # candidate is probably not the one we were following
                    tie_ok = _iou(last_box, best["bbox"]) >= _iou(last_box, runner["bbox"])
            tight = tight and not tie_ok

            if not locked:
                # first lock, or reclaiming after being lost: strict
                need = (config.TRACK_TARGET_REACQUIRE_SIM if last_box is not None
                        else config.TRACK_TARGET_CONFIRM_SIM)
                if best["sim"] >= need and not tight:
                    chosen, status = best, ("reacquired" if last_box is not None else "confirmed")
                    if last_box is not None:
                        reacquisitions += 1
                else:
                    reason = (f"below {'reacquire' if last_box is not None else 'confirm'} "
                              f"threshold {need}" if best["sim"] < need
                              else f"runner-up too close (margin {margin})")
            elif best["sim"] >= config.TRACK_TARGET_CONFIRM_SIM and not tight:
                chosen, status = best, "confirmed"
            elif (best["sim"] >= config.TRACK_TARGET_CONTINUE_SIM and motion_ok and not tight):
                chosen, status = best, "continued"
            else:
                if tight:
                    reason = f"two candidates within {margin} - refusing to guess"
                elif not motion_ok:
                    reason = f"identity ok ({best['sim']}) but moved {travelled:.0f}px, implausible"
                else:
                    reason = f"identity too weak ({best['sim']})"

        if chosen is not None:
            points.append({"offset_seconds": off, "frame_number": rec["frame"],
                           "bbox": [round(v, 2) for v in chosen["bbox"]],
                           "confidence": chosen["det_conf"],
                           "identity": chosen["sim"], "status": status,
                           "predicted": False})
            last_box, last_off = chosen["bbox"], off
            locked = True
            misses = 0
        else:
            misses += 1
            if misses >= config.TRACK_TARGET_LOST_AFTER:
                locked = False               # lock dropped; re-acquisition needed

        if config.TRACK_TARGET_DEBUG:
            debug.append({"frame": rec["frame"], "offset": off, "status": status,
                          "target_detected": chosen is not None,
                          "best_sim": best["sim"] if best else None,
                          "runner_sim": runner["sim"] if runner else None,
                          "margin": margin, "candidates": len(cands),
                          "reason": reason})

    return points, debug, reacquisitions


def _bridge(points, stride_s, fw, fh):
    """Fill only single-sample holes where the person barely moved.

    At TRACK_TARGET_FPS a one-sample hole is a fraction of a second, so filling it
    keeps the overlay steady through a momentary miss. Anything longer is left
    empty: a longer guess is what used to slide the box across other people."""
    if len(points) < 2 or stride_s <= 0:
        return points
    diag = math.hypot(fw or 1920, fh or 1080)
    limit = 0.06 * diag
    out = [points[0]]
    for a, b in zip(points, points[1:]):
        gap = (b["offset_seconds"] or 0) - (a["offset_seconds"] or 0)
        steps = int(round(gap / stride_s)) - 1
        if steps == 1 and gap <= config.TRACK_TARGET_BRIDGE_S:
            ca, cb = _centre(a["bbox"]), _centre(b["bbox"])
            if math.hypot(cb[0] - ca[0], cb[1] - ca[1]) <= limit:
                out.append({"offset_seconds": round((a["offset_seconds"] + b["offset_seconds"]) / 2, 3),
                            "frame_number": None,
                            "bbox": [round((a["bbox"][i] + b["bbox"][i]) / 2, 2) for i in range(4)],
                            "confidence": None,
                            "identity": min(a["identity"], b["identity"]),
                            "status": "continued", "predicted": True})
        out.append(b)
    return out


# ------------------------------------------------------------------ public API
def _cache_get(video_id, ref_id):
    with database.get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM target_tracks WHERE video_id=? AND ref_detection_id=? "
            "AND version=?", (video_id, ref_id, config.TRACK_TARGET_VERSION)).fetchone()
    return dict(row) if row else None


def _out(row: dict, cached: bool) -> dict:
    for k in ("points", "debug", "metrics", "reference_views"):
        if row.get(k):
            try:
                row[k] = json.loads(row[k])
            except (TypeError, ValueError):
                row[k] = None
    row["cached"] = cached
    return row
