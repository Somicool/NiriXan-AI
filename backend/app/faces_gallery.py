"""Face Gallery - save the best face of a person and find the same individual
across all indexed footage.

Additive module. It reuses the faces already detected during ingestion (the
`faces` table + the InsightFace `face` FAISS index) and the existing
result/playback helpers - it does NOT re-run person detection or redesign any
backend. Saved faces are stored permanently in the `saved_faces` table (with a
copy of the embedding) and only removed on explicit delete.
"""
from __future__ import annotations

import base64
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from . import config, database, gpu_lock
from .search import vector_store
from .search.text_search import media_url, _camera_names, _video_index, playback_fields, to_result_item

_PERSON_LABELS = {config.DETECT_CLASSES[c] for c in config.PERSON_CLASSES}


# --------------------------------------------------------------- helpers
def _sharpness(path) -> float:
    img = cv2.imread(str(path)) if path else None
    if img is None or not img.size:
        return 0.0
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def _face_rows_for_track(video_id, track_id) -> list[dict]:
    if video_id is None or track_id is None:
        return []
    with database.get_conn() as conn:
        rows = conn.execute(
            "SELECT f.* FROM faces f JOIN detections d ON d.detection_id = f.detection_id "
            "WHERE d.video_id=? AND d.track_id=?", (video_id, track_id)).fetchall()
    return [dict(r) for r in rows]


# ------------------------------------------------- original frame + expanded box
def _frame_path(video_id, frame_number) -> str | None:
    """Original full frame for a (video, frame). Ingestion stores every sampled
    frame as a 'scene' row whose crop_path IS the frame path - so we can recover
    the ORIGINAL frame for any detection instead of relying on its tight crop."""
    if video_id is None or frame_number is None:
        return None
    with database.get_conn() as conn:
        row = conn.execute(
            "SELECT crop_path FROM detections WHERE video_id=? AND class_label='scene' "
            "AND frame_number=? LIMIT 1", (video_id, frame_number)).fetchone()
    p = row["crop_path"] if row else None
    return p if (p and Path(p).exists()) else None


def _expand_box(bbox, W, H, frac=None):
    """Expand a person bbox by ~frac (default config.FACE_BOX_EXPAND), clamped to
    the image. Gives the face the context a tight body crop is missing."""
    frac = config.FACE_BOX_EXPAND if frac is None else frac
    x, y, w, h = (float(v) for v in bbox)
    px, py = w * frac, h * frac
    x1, y1 = max(0, int(x - px)), max(0, int(y - py))
    x2, y2 = min(int(W), int(x + w + px)), min(int(H), int(y + h + py))
    return (x1, y1, x2, y2) if (x2 > x1 and y2 > y1) else None


def source_video_path(video_id):
    """Absolute path of the ORIGINAL source video for a video_id (or None)."""
    if video_id is None:
        return None
    with database.get_conn() as conn:
        row = conn.execute("SELECT filename FROM videos WHERE video_id=?", (video_id,)).fetchone()
    if not row or not row["filename"]:
        return None
    p = config.VIDEO_DIR / row["filename"]
    return p if p.exists() else None


class _FrameReader:
    """Seeks and reads FULL-RESOLUTION frames straight from the source video.

    Face extraction must never reuse the tiny stored person crops (they are only
    ~90px wide), so every candidate frame is decoded from the original video.

    Reading strategy: one explicit seek per frame. This costs ~128 ms per frame on
    1080p CCTV because the decoder re-finds the preceding keyframe every time, and
    decode-skipping forward with grab() between nearby candidates measured 3.7x
    faster. It is NOT used: on these recordings cv2 seeks are not reliably
    frame-exact, so a skip count derived from the post-seek position drifts, and a
    measured 15-63% of frames came back as a DIFFERENT frame than the seek path
    returned. A faster reader that quietly examines different pixels would change
    which face is chosen, so the exact-but-slower seek is kept. Repeat saves avoid
    the scan entirely instead (see _save_from_cache)."""

    def __init__(self, video_path):
        self.cap = cv2.VideoCapture(str(video_path)) if video_path else None
        self.ok = bool(self.cap and self.cap.isOpened())

    def read(self, frame_index):
        if not self.ok or frame_index is None:
            return None
        try:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            got, frame = self.cap.read()
            return frame if (got and frame is not None and frame.size) else None
        except Exception:
            return None

    def close(self):
        try:
            if self.cap is not None:
                self.cap.release()
        except Exception:
            pass


def _expanded_from_full_frame(frame, det: dict):
    """(expanded person crop, offset (x1,y1)) taken from a FULL-RES video frame.
    The offset lets us map a face box back to original-frame coordinates."""
    if frame is None or not frame.size or det.get("bbox_x") is None:
        return None, (0, 0)
    H, W = frame.shape[:2]
    box = _expand_box((det["bbox_x"], det["bbox_y"], det["bbox_w"], det["bbox_h"]), W, H)
    if not box:
        return None, (0, 0)
    x1, y1, x2, y2 = box
    crop = frame[y1:y2, x1:x2]
    return (crop if (crop is not None and crop.size) else None), (x1, y1)


def expanded_crop_url(detection_id: int, generate: bool = True) -> str | None:
    """Cache + return a /media URL for the EXPANDED person crop (display).

    No AI, but building one DECODES a frame from the source video, which is far
    from free. `generate=False` serves the cached image if it exists and otherwise
    falls straight back to the stored crop: a search returns up to 60 results
    whose thumbnails all render at once, and opening/seeking the source file 60
    times over starved the video element's own range requests, which is what made
    the clip take seconds to appear. The selected result still gets the full
    expanded crop (generate=True)."""
    refs = database.get_detections([detection_id])
    if not refs:
        return None
    det = refs[0]
    out = config.EXPANDED_CROP_DIR / f"exp_{detection_id}.jpg"
    if out.exists():
        return media_url(str(out))
    if not generate:
        return media_url(det.get("crop_path"))
    # decode the ORIGINAL frame from the source video (stored crops are tiny)
    crop = None
    vpath = source_video_path(det.get("video_id"))
    if vpath:
        reader = _FrameReader(vpath)
        try:
            frame = reader.read(det.get("frame_number"))
            if frame is not None:
                crop, _off = _expanded_from_full_frame(frame, det)
        finally:
            reader.close()
    if crop is None or not crop.size:
        return media_url(det.get("crop_path"))
    try:
        cv2.imwrite(str(out), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        return media_url(str(out))
    except Exception:
        return media_url(det.get("crop_path"))


# ------------------------------------------------- face quality scoring
def _detect_faces_kps(img):
    """InsightFace faces WITH landmarks (kps), in the ORIGINAL `img` coordinate
    space. Small crops are upscaled for detection, so bbox/kps are scaled back -
    otherwise the face region would slice out of bounds. Face detection only;
    person detection is never re-run."""
    try:
        from .ingestion import face_recognizer
        if img is None or not img.size:
            return []
        up = face_recognizer._upscale(img, 320)
        faces = list(face_recognizer.get_face_app().get(up))
        if not faces:
            return []
        s = up.shape[1] / float(img.shape[1] or 1)          # upscale factor
        if abs(s - 1.0) > 1e-6:
            for f in faces:                                  # map back to `img` space
                try:
                    f.bbox = np.asarray(f.bbox, dtype="float32") / s
                    if getattr(f, "kps", None) is not None:
                        f.kps = np.asarray(f.kps, dtype="float32") / s
                except Exception:
                    pass
        return faces
    except Exception:
        return []


def _frontal_score(f) -> float:
    """0-1 frontal-ness from the 5 landmarks: eye/nose horizontal symmetry plus
    both eyes being visible. A profile face scores low."""
    try:
        kps = np.asarray(f.kps, dtype="float32")
        if kps.shape[0] < 3:
            return 0.5
        le, re, nose = kps[0], kps[1], kps[2]
        eye_dx = abs(float(re[0] - le[0]))
        if eye_dx < 1e-3:
            return 0.0
        # nose should sit near the midpoint of the eyes when frontal
        mid = (le[0] + re[0]) / 2.0
        off = abs(float(nose[0]) - mid) / eye_dx           # 0 = perfectly centred
        return float(max(0.0, min(1.0, 1.0 - off * 1.6)))
    except Exception:
        return 0.5


def _eyes_score(f, fw) -> float:
    """0-1: are both eyes clearly visible/separated (vs squashed by profile)?"""
    try:
        kps = np.asarray(f.kps, dtype="float32")
        if kps.shape[0] < 2 or fw <= 0:
            return 0.5
        eye_dx = abs(float(kps[1][0] - kps[0][0]))
        # a frontal face has eye separation ~35-45% of face width
        return float(max(0.0, min(1.0, (eye_dx / fw) / 0.38)))
    except Exception:
        return 0.5


def _brightness_score(reg) -> float:
    """0-1: well-exposed face (peaks around mid brightness, penalises dark/blown)."""
    try:
        g = cv2.cvtColor(reg, cv2.COLOR_BGR2GRAY) if reg.ndim == 3 else reg
        m = float(g.mean()) / 255.0
        return float(max(0.0, 1.0 - abs(m - 0.55) / 0.45))
    except Exception:
        return 0.5


def _occlusion_score(f, crop_shape) -> float:
    """0-1 visibility: how much of the face box lies inside the frame (a face cut
    off by the crop/frame edge is likely occluded or truncated)."""
    try:
        H, W = crop_shape[:2]
        x1, y1, x2, y2 = (float(v) for v in f.bbox.tolist())
        area = max(1.0, (x2 - x1) * (y2 - y1))
        vis_w = max(0.0, min(x2, W) - max(x1, 0.0))
        vis_h = max(0.0, min(y2, H) - max(y1, 0.0))
        return float(max(0.0, min(1.0, (vis_w * vis_h) / area)))
    except Exception:
        return 0.5


def _noise_score(reg) -> float:
    """0-1: low sensor/compression noise scores high (median-residual estimate)."""
    try:
        g = cv2.cvtColor(reg, cv2.COLOR_BGR2GRAY) if reg.ndim == 3 else reg
        if g.size < 64:
            return 0.5
        resid = cv2.absdiff(g, cv2.medianBlur(g, 3))
        return float(max(0.0, min(1.0, 1.0 - (float(resid.std()) / 18.0))))
    except Exception:
        return 0.5


def _face_quality(f, crop) -> dict:
    """Composite forensic face quality over 9 factors: detection confidence,
    sharpness/blur, resolution, face size, frontal pose, brightness, eyes visible,
    occlusion and image noise."""
    x1, y1, x2, y2 = (float(v) for v in f.bbox.tolist())
    fw, fh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    det = float(f.det_score)
    size = min(1.0, (min(fw, fh) / 90.0))                  # ~90px face = full marks
    resolution = int(fw * fh)                              # face pixel area

    reg = None
    sharp = 0.0
    try:
        H, W = crop.shape[:2]
        fx1, fy1 = max(0, int(x1)), max(0, int(y1))
        fx2, fy2 = min(W, int(x2)), min(H, int(y2))
        reg = crop[fy1:fy2, fx1:fx2]
        if reg is not None and reg.size:
            g = cv2.cvtColor(reg, cv2.COLOR_BGR2GRAY) if reg.ndim == 3 else reg
            sharp = min(1.0, float(cv2.Laplacian(g, cv2.CV_64F).var()) / 220.0)
        else:
            reg = None
    except Exception:
        reg = None

    pose = _frontal_score(f)
    eyes = _eyes_score(f, fw)
    bright = _brightness_score(reg) if reg is not None else 0.5
    occl = _occlusion_score(f, crop.shape)
    noise = _noise_score(reg) if reg is not None else 0.5

    q = (config.FACE_Q_W_DET * det + config.FACE_Q_W_SIZE * size
         + config.FACE_Q_W_SHARP * sharp + config.FACE_Q_W_POSE * pose
         + config.FACE_Q_W_EYES * eyes + config.FACE_Q_W_BRIGHT * bright
         + config.FACE_Q_W_OCCL * occl + config.FACE_Q_W_NOISE * noise)
    return {"quality": round(float(q), 4), "det_score": round(det, 4),
            "face_size": int(min(fw, fh)), "resolution": resolution,
            "sharpness": round(sharp, 3), "frontal": round(pose, 3),
            "eyes": round(eyes, 3), "brightness": round(bright, 3),
            "occlusion": round(occl, 3), "noise": round(noise, 3),
            "bbox": [x1, y1, x2, y2]}


def _scan_order(dets: list[dict], max_frames: int) -> list[dict]:
    """Frames to examine: the ENTIRE track. Short tracks are scanned fully; long
    tracks are sampled EVENLY across their whole span (never front-biased), with
    the largest boxes always included since they carry the best face detail."""
    by_frame = sorted(dets, key=lambda d: (d.get("frame_number") or 0))
    if len(by_frame) <= max_frames:
        return by_frame
    idx = np.linspace(0, len(by_frame) - 1, max_frames).astype(int)
    out, seen = [], set()
    for i in idx:                                          # even coverage of the full track
        d = by_frame[int(i)]
        if d["detection_id"] not in seen:
            seen.add(d["detection_id"]); out.append(d)
    big = sorted(dets, key=lambda d: (d.get("bbox_w") or 0) * (d.get("bbox_h") or 0), reverse=True)
    for d in big[:max(4, max_frames // 4)]:
        if d["detection_id"] not in seen:
            seen.add(d["detection_id"]); out.append(d)
    return out


def rank_faces_in_track(video_id, track_id, max_frames=None) -> dict:
    """Forensic best-face pipeline over a whole ByteTrack person track.

    For EVERY candidate frame of the track it reloads the ORIGINAL full-resolution
    frame from the source video, expands the original person bbox by ~20%, runs
    face detection on that fresh crop, and scores the face on 9 factors. Stored
    person crops are never reused. The running best is replaced whenever a better
    face appears later, so the winner represents the entire track.

    Returns {"best", "ranked", "frames_seen", "faces_seen", "reason",
             "video_path", "diagnostics"}. `best` is None when no face in the whole
    track clears the forensic acceptance bar."""
    out = {"best": None, "ranked": [], "frames_seen": 0, "faces_seen": 0,
           "reason": None, "video_path": None, "diagnostics": []}
    if video_id is None or track_id is None:
        out["reason"] = "missing video/track id"
        return out
    max_frames = max_frames or config.FACE_TRACK_SCAN_FRAMES

    with database.get_conn() as conn:
        dets = [dict(r) for r in conn.execute(
            "SELECT * FROM detections WHERE video_id=? AND track_id=? AND class_label!='scene'",
            (video_id, track_id)).fetchall()]
    if not dets:
        out["reason"] = "no detections for this track"
        return out

    vpath = source_video_path(video_id)
    if vpath is None:
        out["reason"] = "source video not available"
        return out
    out["video_path"] = str(vpath)

    reader = _FrameReader(vpath)
    if not reader.ok:
        reader.close()
        out["reason"] = "could not open source video"
        return out

    # candidates ordered by frame so seeking moves mostly forward
    cands = sorted(_scan_order(dets, max_frames), key=lambda d: (d.get("frame_number") or 0))
    best, ranked, frames_seen, faces_seen = None, [], 0, 0
    try:
        for d in cands:
            # Step aside while the officer is waiting on a GPU pass of their own
            # (Track Person). This only WAITS between frames - no frame is skipped
            # and nothing about the scoring below changes, so the winning face is
            # identical; it just may take longer to arrive. Measured: without this,
            # an overlapping Track Person pass ran 12x slower.
            # Only the speculative background scan steps aside; a scan the officer
            # is waiting on (Save Face) must not pause behind anything.
            if getattr(_speculative, "on", False):
                gpu_lock.yield_to_priority()
            frame = reader.read(d.get("frame_number"))          # FULL-RES original frame
            if frame is None:
                continue
            crop, (ox, oy) = _expanded_from_full_frame(frame, d)
            if crop is None or not crop.size:
                continue
            frames_seen += 1
            for f in _detect_faces_kps(crop):
                if float(f.det_score) < config.FACE_MIN_DET_SCORE:
                    continue
                faces_seen += 1
                m = _face_quality(f, crop)
                ranked.append({**{k: v for k, v in m.items() if k != "bbox"},
                               "detection_id": d["detection_id"],
                               "frame_number": d.get("frame_number")})
                if config.FACE_DIAG_LOG:
                    out["diagnostics"].append(
                        f"frame {d.get('frame_number')}: q={m['quality']:.3f} "
                        f"conf={m['det_score']:.2f} sharp={m['sharpness']:.2f} "
                        f"size={m['face_size']}px res={m['resolution']}px2 "
                        f"frontal={m['frontal']:.2f} eyes={m['eyes']:.2f}")
                # progressive replacement: a better face later in the track wins
                if best is None or m["quality"] > best["quality"]:
                    fx1, fy1, fx2, fy2 = m["bbox"]
                    best = {**m, "detection_id": d["detection_id"],
                            "camera_id": d.get("camera_id"), "timestamp": d.get("timestamp"),
                            "frame_number": d.get("frame_number"),
                            # face box in ORIGINAL-FRAME coordinates (for full-res re-crop)
                            "frame_bbox": [fx1 + ox, fy1 + oy, fx2 + ox, fy2 + oy],
                            "insight": f}
    finally:
        reader.close()

    ranked.sort(key=lambda x: x["quality"], reverse=True)
    out.update(ranked=ranked, frames_seen=frames_seen, faces_seen=faces_seen)

    # forensic acceptance: never fall back to a poor crop
    if best is None:
        out["reason"] = ("No usable face found in this track."
                         if frames_seen else "no frames could be decoded")
        return out
    if best["quality"] < config.FACE_ACCEPT_QUALITY or best["face_size"] < config.FACE_ACCEPT_MIN_PX:
        out["reason"] = (f"No usable face found in this track. "
                         f"(best q={best['quality']:.2f} < {config.FACE_ACCEPT_QUALITY}, "
                         f"size={best['face_size']}px)")
        out["rejected_best"] = {k: best[k] for k in ("quality", "face_size", "sharpness",
                                                     "det_score", "frame_number")}
        return out

    best["low_quality"] = bool(best["quality"] < config.FACE_LOW_QUALITY_THRESHOLD)
    best["reason"] = (f"highest quality of {faces_seen} face(s) across {frames_seen} "
                      f"full-resolution frames of track {track_id}")
    out["best"] = best
    return out


def _recrop_face_full_res(video_id, frame_number, frame_bbox, pad_x=0.30, pad_y=0.40):
    """Re-crop the winning face DIRECTLY from the original frame at full resolution
    (not from the working crop), so the saved image keeps maximum detail."""
    vpath = source_video_path(video_id)
    if vpath is None or frame_bbox is None:
        return None
    reader = _FrameReader(vpath)
    try:
        frame = reader.read(frame_number)
        if frame is None:
            return None
        H, W = frame.shape[:2]
        x1, y1, x2, y2 = (float(v) for v in frame_bbox)
        fw, fh = max(1.0, x2 - x1), max(1.0, y2 - y1)
        cx1 = max(0, int(x1 - fw * pad_x)); cy1 = max(0, int(y1 - fh * pad_y))
        cx2 = min(W, int(x2 + fw * pad_x)); cy2 = min(H, int(y2 + fh * pad_y))
        if cx2 <= cx1 or cy2 <= cy1:
            return None
        return frame[cy1:cy2, cx1:cx2].copy()
    finally:
        reader.close()


def best_face_in_track(video_id, track_id, ref_frame=None, max_frames=None) -> dict | None:
    """Best representative face of the whole track (compatibility wrapper)."""
    return rank_faces_in_track(video_id, track_id, max_frames=max_frames)["best"]


# ------------------------------------------------- artifacts + per-track cache
def _write_face_artifacts(video_id, found: dict, tag: str) -> dict:
    """Write the three images for a winning face, all re-cropped from the ORIGINAL
    full-resolution frame (never from a stored crop):
      face    - tight best-face crop        (Face Gallery image)
      preview - normalised 256px square     (Preview image)
      profile - expanded person crop        (Person profile image)
    Returns absolute paths (any may be None on failure)."""
    out = {"face": None, "preview": None, "profile": None, "face_px": None}
    vpath = source_video_path(video_id)
    reader = _FrameReader(vpath) if vpath else None
    try:
        frame = reader.read(found.get("frame_number")) if (reader and reader.ok) else None
        if frame is None:
            return out
        H, W = frame.shape[:2]
        # 1) tight face crop at FULL resolution
        x1, y1, x2, y2 = (float(v) for v in found["frame_bbox"])
        fw, fh = max(1.0, x2 - x1), max(1.0, y2 - y1)
        cx1, cy1 = max(0, int(x1 - fw * 0.30)), max(0, int(y1 - fh * 0.40))
        cx2, cy2 = min(W, int(x2 + fw * 0.30)), min(H, int(y2 + fh * 0.40))
        reg = frame[cy1:cy2, cx1:cx2] if (cx2 > cx1 and cy2 > cy1) else None
        if reg is not None and reg.size:
            fp = config.SAVED_FACE_DIR / f"face_{tag}.jpg"
            cv2.imwrite(str(fp), reg, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            out["face"] = str(fp)
            out["face_px"] = f"{reg.shape[1]}x{reg.shape[0]}"
            # 2) 256px square preview (upscaled with a good kernel for small faces)
            prev = cv2.resize(reg, (256, 256), interpolation=cv2.INTER_CUBIC)
            pp = config.SAVED_FACE_DIR / f"preview_{tag}.jpg"
            cv2.imwrite(str(pp), prev, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            out["preview"] = str(pp)
        # 3) person profile image: expanded person box from the same original frame
        pdet = (database.get_detections([found["detection_id"]]) or [None])[0]
        if pdet is not None:
            pcrop, _off = _expanded_from_full_frame(frame, pdet)
            if pcrop is not None and pcrop.size:
                prof = config.SAVED_FACE_DIR / f"profile_{tag}.jpg"
                cv2.imwrite(str(prof), pcrop, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
                out["profile"] = str(prof)
    except Exception:
        pass
    finally:
        if reader is not None:
            reader.close()
    return out


def _cache_get(video_id, track_id) -> dict | None:
    if video_id is None or track_id is None:
        return None
    with database.get_conn() as conn:
        row = conn.execute("SELECT * FROM track_best_face WHERE video_id=? AND track_id=?",
                           (video_id, track_id)).fetchone()
    return dict(row) if row else None


def _scan_fingerprint() -> str:
    """Identifies the settings the best-face scan ran under. A cached winner is
    only reused while this matches, so re-tuning any face threshold or weight
    automatically invalidates the cache instead of serving a stale choice."""
    import hashlib
    parts = (config.FACE_TRACK_SCAN_FRAMES, config.FACE_MIN_DET_SCORE,
             config.FACE_ACCEPT_QUALITY, config.FACE_ACCEPT_MIN_PX,
             config.FACE_BOX_EXPAND, config.FACE_LOW_QUALITY_THRESHOLD,
             config.FACE_Q_W_DET, config.FACE_Q_W_SIZE, config.FACE_Q_W_SHARP,
             config.FACE_Q_W_POSE, config.FACE_Q_W_EYES, config.FACE_Q_W_BRIGHT,
             config.FACE_Q_W_OCCL, config.FACE_Q_W_NOISE)
    return hashlib.sha1(repr(parts).encode()).hexdigest()[:12]


def _cache_put(video_id, track_id, found, arts, scan) -> None:
    import json
    metrics = {k: found.get(k) for k in
               ("det_score", "face_size", "resolution", "sharpness", "frontal",
                "eyes", "brightness", "occlusion", "noise")}
    # Everything a later SAVE needs, so it never has to repeat the scan that has
    # already picked this face. Not exposed to the UI.
    payload = None
    try:
        f = found.get("insight")
        if f is not None and getattr(f, "normed_embedding", None) is not None:
            payload = json.dumps({
                "scan_fp": _scan_fingerprint(),
                "embedding": _encode_emb(np.asarray(f.normed_embedding, dtype="float32")),
                "age": int(f.age) if getattr(f, "age", None) is not None else None,
                "camera_id": found.get("camera_id"), "timestamp": found.get("timestamp"),
                "frame_number": found.get("frame_number"), "reason": found.get("reason"),
                "face_px": arts.get("face_px"),
                "frames_seen": scan.get("frames_seen"), "faces_seen": scan.get("faces_seen"),
            })
    except Exception:
        payload = None
    with database.get_conn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO track_best_face (video_id, track_id, detection_id, quality, "
            " low_quality, face_crop, preview_crop, person_crop, metrics, frames_seen, faces_seen, "
            " created_at, save_payload) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (video_id, track_id, found["detection_id"], found["quality"],
             1 if found.get("low_quality") else 0, arts.get("face"), arts.get("preview"),
             arts.get("profile"), json.dumps(metrics), scan.get("frames_seen"),
             scan.get("faces_seen"), database._now(), payload))


def _log_selection(video_id, track_id, found, arts, scan) -> None:
    """Diagnostic log for every saved/selected face: which frame won and why."""
    if not config.FACE_DIAG_LOG:
        return
    try:
        print(f"[face] track {track_id} (video {video_id}) -> SELECTED frame "
              f"{found.get('frame_number')}")
        print(f"[face]   quality={found['quality']:.3f} blur/sharpness={found['sharpness']:.3f} "
              f"confidence={found['det_score']:.3f} face={found['face_size']}px "
              f"res={found['resolution']}px2 frontal={found['frontal']:.2f} "
              f"eyes={found['eyes']:.2f} brightness={found['brightness']:.2f} "
              f"occlusion={found['occlusion']:.2f} noise={found['noise']:.2f}")
        print(f"[face]   reason: {found.get('reason')}")
        print(f"[face]   scanned {scan.get('frames_seen')} full-res frames, "
              f"ranked {scan.get('faces_seen')} faces; saved face image "
              f"{arts.get('face_px')} from source video")
        runner = (scan.get("ranked") or [{}])[1] if len(scan.get("ranked") or []) > 1 else None
        if runner:
            print(f"[face]   runner-up: frame {runner.get('frame_number')} q={runner.get('quality')}")
    except Exception:
        pass


def _preview_from_cache(cached: dict, detection_id: int, generate: bool = True) -> dict:
    import json
    try:
        metrics = json.loads(cached.get("metrics") or "{}")
    except Exception:
        metrics = {}
    return {"available": True, "source": "track-best (cached)",
            "face_id": None, "detection_id": cached.get("detection_id"),
            "camera_id": None, "timestamp": None,
            "person_crop_url": media_url(cached.get("person_crop"))
                               or expanded_crop_url(detection_id, generate=generate),
            "face_crop_url": media_url(cached.get("face_crop")),
            "preview_crop_url": media_url(cached.get("preview_crop")),
            "quality": cached.get("quality"),
            "low_quality": bool(cached.get("low_quality")),
            "frames_seen": cached.get("frames_seen"), "faces_seen": cached.get("faces_seen"),
            **metrics}


def best_face_for_detection(detection_id: int, deep: bool = True,
                            generate: bool | None = None) -> dict | None:
    """Best face for the PERSON that `detection_id` belongs to.

    deep=True  -> scan the whole track from the ORIGINAL frames (recovers faces the
                  tight stored crops miss).
    deep=False -> fast preview only from faces already stored at ingest time. This
                  is the search-result thumbnail path, called once per result, so
                  it must not decode video (see expanded_crop_url).
    Returns a preview dict, or None when the entire track has no usable face."""
    if generate is None:
        generate = deep
    refs = database.get_detections([detection_id])
    if not refs:
        return None
    ref = refs[0]
    vid, tid = ref.get("video_id"), ref.get("track_id")

    # fast path: a face already stored for this track at ingest time
    stored = _face_rows_for_track(vid, tid)
    if not stored:
        with database.get_conn() as conn:
            stored = [dict(r) for r in conn.execute(
                "SELECT * FROM faces WHERE detection_id=?", (detection_id,)).fetchall()]
    # Best representative face already computed for this track -> serve instantly
    # (this is what lets search results show the BEST face, not the first one).
    cached = _cache_get(vid, tid)
    if cached and cached.get("face_crop"):
        return _preview_from_cache(cached, detection_id, generate=generate)

    if not deep:
        # preview only: never scan. No stored face -> report unavailable.
        if not stored:
            return None
        best = max(stored, key=lambda r: _sharpness(r.get("crop_path")))
        return {"available": True, "source": "stored", "face_id": best["face_id"],
                "detection_id": best["detection_id"], "camera_id": best.get("camera_id"),
                "timestamp": best.get("timestamp"), "gender": best.get("gender"),
                "age": best.get("age"),
                "person_crop_url": expanded_crop_url(detection_id, generate=generate),
                "face_crop_url": media_url(best.get("crop_path")),
                "quality": round(min(1.0, _sharpness(best.get("crop_path")) / 300.0), 3)}

    # deep: rank every face across the WHOLE track and take the best
    scan = rank_faces_in_track(vid, tid)
    found = scan["best"]
    if found is None:
        if stored:                                          # fall back to the ingest face
            best = max(stored, key=lambda r: _sharpness(r.get("crop_path")))
            return {"available": True, "source": "stored", "face_id": best["face_id"],
                    "detection_id": best["detection_id"], "camera_id": best.get("camera_id"),
                    "timestamp": best.get("timestamp"), "gender": best.get("gender"),
                    "age": best.get("age"),
                    "person_crop_url": expanded_crop_url(detection_id, generate=generate),
                    "face_crop_url": media_url(best.get("crop_path")),
                    "quality": round(min(1.0, _sharpness(best.get("crop_path")) / 300.0), 3)}
        return None

    # write the face / preview / profile images (all re-cropped from the ORIGINAL
    # full-res frame) and cache the winner for this track
    arts = _write_face_artifacts(vid, found, f"t{vid}_{tid}")
    try:
        _cache_put(vid, tid, found, arts, scan)
    except Exception as exc:                    # cache is optional, but never silent:
        print(f"[face] WARNING: best-face cache write failed for "
              f"video {vid} track {tid}: {exc}")
    _log_selection(vid, tid, found, arts, scan)

    f = found["insight"]
    return {"available": True, "source": "track-best",
            "selected_frame": found.get("frame_number"),
            "selection_reason": found.get("reason"),
            "face_image_px": arts.get("face_px"),
            "face_id": None, "detection_id": found["detection_id"],
            "camera_id": found.get("camera_id"), "timestamp": found.get("timestamp"),
            "gender": None, "age": int(f.age) if getattr(f, "age", None) is not None else None,
            "person_crop_url": media_url(arts.get("profile")) or expanded_crop_url(found["detection_id"]),
            "face_crop_url": media_url(arts.get("face")),
            "preview_crop_url": media_url(arts.get("preview")),
            "quality": found["quality"], "low_quality": bool(found.get("low_quality")),
            "det_score": found["det_score"], "face_size": found["face_size"],
            "resolution": found["resolution"], "sharpness": found["sharpness"],
            "frontal": found["frontal"], "eyes": found["eyes"],
            "brightness": found["brightness"], "occlusion": found["occlusion"],
            "noise": found["noise"], "frame_number": found.get("frame_number"),
            "frames_seen": scan["frames_seen"], "faces_seen": scan["faces_seen"],
            "candidates": scan["ranked"][:8]}


def _tight_face_crop(person_crop_path, saved_id) -> str | None:
    """Crop the tight face region out of a person crop (InsightFace) and save it.
    Face DETECTION only (never person detection). Falls back to None on failure."""
    try:
        img = cv2.imread(str(person_crop_path))
        if img is None or not img.size:
            return None
        from .ingestion import face_recognizer
        faces = face_recognizer.get_face_app().get(face_recognizer._upscale(img))
        if not faces:
            return None
        f = max(faces, key=lambda x: float(x.det_score))
        x1, y1, x2, y2 = (int(v) for v in f.bbox.tolist())
        H, W = img.shape[:2]
        # NOTE: bbox is in the upscaled space; re-run on the original for correct coords
        faces2 = face_recognizer.get_face_app().get(img)
        if faces2:
            f = max(faces2, key=lambda x: float(x.det_score))
            x1, y1, x2, y2 = (int(v) for v in f.bbox.tolist())
        pad_x, pad_y = int((x2 - x1) * 0.25), int((y2 - y1) * 0.35)
        x1, y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
        x2, y2 = min(W, x2 + pad_x), min(H, y2 + pad_y)
        crop = img[y1:y2, x1:x2]
        if crop is None or not crop.size:
            return None
        out = config.SAVED_FACE_DIR / f"face_{saved_id}.jpg"
        cv2.imwrite(str(out), crop)
        return str(out)
    except Exception:
        return None


def _encode_emb(v: np.ndarray) -> str:
    return base64.b64encode(np.asarray(v, dtype="float32").tobytes()).decode("ascii")


def _decode_emb(s: str) -> np.ndarray | None:
    try:
        return np.frombuffer(base64.b64decode(s), dtype="float32")
    except Exception:
        return None


def _row_urls(row: dict) -> dict:
    import json
    d = dict(row)
    d["face_crop_url"] = media_url(row.get("face_crop"))
    d["person_crop_url"] = media_url(row.get("person_crop"))
    d["preview_crop_url"] = media_url(row.get("preview_crop"))
    d["low_quality"] = bool(row.get("low_quality"))
    if row.get("quality_metrics"):
        try:
            d["metrics"] = json.loads(row["quality_metrics"])
        except Exception:
            pass
    d.pop("embedding", None)                    # don't ship the raw vector to the UI
    d.pop("quality_metrics", None)
    return d


def _save_from_cache(video_id, track_id, investigation) -> dict | None:
    """Write a saved_faces row from the cached best-face decision for this track.

    Returns None (so the caller performs the normal scan) unless the cache holds a
    winner picked under the CURRENT settings, with its embedding and its face image
    still on disk."""
    import json
    cached = _cache_get(video_id, track_id)
    if not cached or not cached.get("save_payload") or not cached.get("face_crop"):
        return None
    if not Path(cached["face_crop"]).exists():
        return None
    try:
        pl = json.loads(cached["save_payload"])
    except Exception:
        return None
    if pl.get("scan_fp") != _scan_fingerprint() or not pl.get("embedding"):
        return None

    src_det = cached["detection_id"]
    profile = cached.get("person_crop")
    if not profile or not Path(profile).exists():
        expanded_crop_url(src_det)                               # build it once
        candidate = config.EXPANDED_CROP_DIR / f"exp_{src_det}.jpg"
        profile = (str(candidate) if candidate.exists()
                   else (database.get_detections([src_det]) or [{}])[0].get("crop_path"))

    try:
        metrics = json.loads(cached.get("metrics") or "{}")
    except Exception:
        metrics = {}
    metrics.update(frames_seen=pl.get("frames_seen"), faces_seen=pl.get("faces_seen"),
                   selected_frame=pl.get("frame_number"), selection_reason=pl.get("reason"),
                   face_image_px=pl.get("face_px"), source="original-video",
                   reused_track_scan=True)
    with database.get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO saved_faces (face_id, detection_id, investigation, camera_id, "
            " timestamp, confidence, face_crop, person_crop, embedding, gender, age, "
            " created_at, preview_crop, low_quality, quality_metrics) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (None, src_det, investigation, pl.get("camera_id"),
             pl.get("timestamp"), cached.get("quality"), cached.get("face_crop"), profile,
             pl["embedding"], None, pl.get("age"), database._now(),
             cached.get("preview_crop"), 1 if cached.get("low_quality") else 0,
             json.dumps(metrics)))
        saved_id = cur.lastrowid
        row = dict(conn.execute("SELECT * FROM saved_faces WHERE saved_id=?", (saved_id,)).fetchone())
    if config.FACE_DIAG_LOG:
        print(f"[face] track {track_id} (video {video_id}) -> saved #{saved_id} from the "
              f"cached best-face decision (frame {pl.get('frame_number')}, "
              f"q={cached.get('quality')}); no rescan needed")
    return _row_urls(row)


# ------------------------------------------------- warm the best-face decision early
# Choosing the best face costs ~19 s per track (measured: 45% seeking/decoding 60
# full-resolution frames, 54% InsightFace). That is inherent to the method and is
# not something to trim by looking at fewer or smaller frames. Instead the SAME
# work is started as soon as the investigator opens a person, so by the time they
# press "Save Face" the decision is already cached and saving is instant.
#
# One worker only: the scan is GPU work on a 6 GB card and must not contend with
# ingestion or with another scan.
_prep_lock = threading.Lock()
_prep_inflight: set[tuple] = set()
# Tracks already scanned in this process. Needed because a track with NO usable
# face caches nothing (by design - a poor face is never stored), so without this
# every reopen would pay for the same fruitless 19 s scan again.
_prep_done: set[tuple] = set()
# Tracks whose full scan COMPLETED and found no face clearing the forensic bar.
# Separate from _prep_done because a scan that crashed proved nothing - only a
# finished scan may be reused as a "no face" answer. Valid for this process,
# whose face settings cannot change while it runs.
_prep_noface: set[tuple] = set()
_prep_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="face-prep")
# Marks the background worker's scan as speculative, so only IT yields to
# user-facing GPU work (see rank_faces_in_track).
_speculative = threading.local()
# One lock per track: at most one scan of a given track at a time, whichever path
# (background prepare or Save Face) gets there first. The other then reuses the
# cached decision. Previously Save Face waited for a QUEUED prepare (behind every
# other person the officer had opened - measured 47 s), and a prepare clicked
# during a save scanned the same track a second time.
_key_locks: dict[tuple, threading.Lock] = {}


def _key_lock(key) -> threading.Lock:
    with _prep_lock:
        lk = _key_locks.get(key)
        if lk is None:
            lk = _key_locks[key] = threading.Lock()
        return lk


def _prep_worker(video_id, track_id, detection_id) -> None:
    key = (video_id, track_id)
    try:
        with _key_lock(key):
            with _prep_lock:
                known_noface = key in _prep_noface
            cached = _cache_get(video_id, track_id)
            if known_noface or (cached and cached.get("face_crop")):
                return                           # Save Face already did this scan
            # This speculative scan yields to user-facing GPU work between frames
            # (see gpu_lock and the check inside rank_faces_in_track). Measured:
            # the two running together made the Track Person pass 12x slower.
            # Same frames, same scoring, same winner - only the timing changes.
            _speculative.on = True
            try:
                best_face_for_detection(detection_id, deep=True)
            finally:
                _speculative.on = False
            cached = _cache_get(video_id, track_id)
            if not (cached and cached.get("face_crop")):
                with _prep_lock:
                    _prep_noface.add(key)
    except Exception as exc:                     # never take the server down
        print(f"[face] prepare failed for video {video_id} track {track_id}: {exc}")
    finally:
        with _prep_lock:
            _prep_inflight.discard(key)
            _prep_done.add(key)


def prepare_best_face(detection_id: int) -> dict:
    """Start (or report) the best-face scan for this person's track.

    Returns immediately. Identical work to the on-click path - same frames, same
    scoring, same winner - just started earlier and de-duplicated so two clicks
    can never scan the same track twice at once."""
    refs = database.get_detections([detection_id])
    if not refs:
        return {"status": "unknown-detection"}
    ref = refs[0]
    vid, tid = ref.get("video_id"), ref.get("track_id")
    if vid is None or tid is None:
        return {"status": "no-track"}
    if ref.get("class_label") not in _PERSON_LABELS:
        return {"status": "not-a-person"}

    cached = _cache_get(vid, tid)
    if cached and cached.get("face_crop"):
        return {"status": "ready", "video_id": vid, "track_id": tid}

    key = (vid, tid)
    with _prep_lock:
        if key in _prep_inflight:
            return {"status": "running", "video_id": vid, "track_id": tid}
        if key in _prep_done:
            # already scanned and nothing cleared the forensic quality bar
            return {"status": "no-face", "video_id": vid, "track_id": tid}
        _prep_inflight.add(key)
    _prep_pool.submit(_prep_worker, vid, tid, detection_id)
    return {"status": "started", "video_id": vid, "track_id": tid}


# --------------------------------------------------------------- public API
def save_face(detection_id: int, investigation: str | None = None) -> dict | None:
    """Permanently save the BEST face found anywhere in the person's track.

    Recovers the face from the original frames + expanded boxes (scanning the whole
    track), so tight body-only search crops no longer cause a failure. Returns None
    only when the entire track genuinely has no usable face."""
    refs = database.get_detections([detection_id])
    if not refs:
        return None
    import json
    ref = refs[0]
    vid, tid = ref.get("video_id"), ref.get("track_id")

    # Fast path: this track's best face has ALREADY been chosen (e.g. the user
    # opened the face preview first), and it was chosen under the same settings.
    # Reuse that decision instead of decoding 60 full-resolution frames and
    # re-running detection to arrive at the identical answer. Pure memoisation -
    # the selection method, the frame examined and the stored metrics are the same.
    reused = _save_from_cache(vid, tid, investigation)
    if reused is not None:
        return reused

    # Blocks only while a scan of THIS track is actually running (then reuses its
    # decision). A prepare that is merely queued behind other people no longer
    # makes the officer wait - this request scans now and the queued job then
    # finds the cached answer.
    key = (vid, tid)
    with _key_lock(key):
        reused = _save_from_cache(vid, tid, investigation)
        if reused is not None:
            return reused
        return _save_face_scan(vid, tid, key, investigation)


def _save_face_scan(vid, tid, key, investigation) -> dict:
    import json
    # This track was already fully scanned in this process and nothing cleared the
    # forensic bar. Same scan, same settings -> same answer; don't pay ~35 s again.
    with _prep_lock:
        known_noface = key in _prep_noface
    if known_noface:
        return {"error": "No usable face found in this track.",
                "reason": "the whole track was already scanned and no face cleared "
                          "the forensic quality bar"}

    # Rank EVERY face in the track and take the best representative one. Holding
    # priority pauses background prepare scans of OTHER people between frames, so
    # they don't share the GPU with the scan the officer is waiting on.
    with gpu_lock.hold(priority=True):
        scan = rank_faces_in_track(vid, tid)
    found = scan["best"]

    if found is not None:
        f = found["insight"]
        src_det = found["detection_id"]
        # embedding straight from the winning face (no re-detection elsewhere)
        emb_b64 = _encode_emb(np.asarray(f.normed_embedding, dtype="float32"))
        arts = _write_face_artifacts(vid, found, f"t{vid}_{tid}")
        try:
            _cache_put(vid, tid, found, arts, scan)
        except Exception as exc:                # cache is optional, but never silent:
            print(f"[face] WARNING: best-face cache write failed for "
                  f"video {vid} track {tid}: {exc}")
        _log_selection(vid, tid, found, arts, scan)
        expanded_crop_url(src_det)                       # ensure the expanded crop exists
        profile = arts.get("profile") or str(config.EXPANDED_CROP_DIR / f"exp_{src_det}.jpg")
        if not Path(profile).exists():
            profile = (database.get_detections([src_det]) or [{}])[0].get("crop_path")
        metrics = {k: found.get(k) for k in
                   ("det_score", "face_size", "resolution", "sharpness", "frontal",
                    "eyes", "brightness", "occlusion", "noise")}
        metrics.update(frames_seen=scan["frames_seen"], faces_seen=scan["faces_seen"],
                       selected_frame=found.get("frame_number"),
                       selection_reason=found.get("reason"),
                       face_image_px=arts.get("face_px"), source="original-video")
        with database.get_conn() as conn:
            cur = conn.execute(
                "INSERT INTO saved_faces (face_id, detection_id, investigation, camera_id, "
                " timestamp, confidence, face_crop, person_crop, embedding, gender, age, "
                " created_at, preview_crop, low_quality, quality_metrics) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (None, src_det, investigation, found.get("camera_id"), found.get("timestamp"),
                 found["quality"], arts.get("face") or profile, profile, emb_b64, None,
                 int(f.age) if getattr(f, "age", None) is not None else None, database._now(),
                 arts.get("preview"), 1 if found.get("low_quality") else 0, json.dumps(metrics)))
            saved_id = cur.lastrowid
            row = dict(conn.execute("SELECT * FROM saved_faces WHERE saved_id=?", (saved_id,)).fetchone())
        return _row_urls(row)

    # Forensic rule: never save a poor crop. If no face in the whole track clears
    # the acceptance bar, report it instead of storing a low-quality image.
    if scan.get("frames_seen"):                  # a real, completed scan - remember it
        with _prep_lock:
            _prep_noface.add(key)
    if config.FACE_DIAG_LOG:
        print(f"[face] track {tid} (video {vid}) -> REJECTED: {scan.get('reason')} "
              f"(scanned {scan.get('frames_seen')} full-res frames, "
              f"{scan.get('faces_seen')} face(s) seen)")
    return {"error": "No usable face found in this track.",
            "reason": scan.get("reason"),
            "frames_seen": scan.get("frames_seen"), "faces_seen": scan.get("faces_seen"),
            "rejected_best": scan.get("rejected_best")}


def list_saved() -> list[dict]:
    with database.get_conn() as conn:
        rows = conn.execute("SELECT * FROM saved_faces ORDER BY created_at DESC").fetchall()
    return [_row_urls(dict(r)) for r in rows]


def get_saved(saved_id: int) -> dict | None:
    with database.get_conn() as conn:
        row = conn.execute("SELECT * FROM saved_faces WHERE saved_id=?", (saved_id,)).fetchone()
    return _row_urls(dict(row)) if row else None


def delete_saved(saved_id: int) -> dict:
    with database.get_conn() as conn:
        row = conn.execute("SELECT face_crop FROM saved_faces WHERE saved_id=?", (saved_id,)).fetchone()
        conn.execute("DELETE FROM saved_faces WHERE saved_id=?", (saved_id,))
    # remove the saved tight-crop file (leave the shared person crop alone)
    try:
        if row and row["face_crop"] and str(config.SAVED_FACE_DIR) in str(row["face_crop"]):
            Path(row["face_crop"]).unlink(missing_ok=True)
    except Exception:
        pass
    return {"deleted": saved_id}


def find_similar(saved_id: int, top_k: int | None = None) -> dict:
    """Find the same individual across all indexed footage using the STORED face
    embedding (no person detection re-run). Returns matches sorted by similarity."""
    top_k = top_k or config.DEFAULT_TOP_K
    with database.get_conn() as conn:
        row = conn.execute("SELECT * FROM saved_faces WHERE saved_id=?", (saved_id,)).fetchone()
    if not row:
        return {"saved_id": saved_id, "results": []}
    row = dict(row)
    emb = _decode_emb(row.get("embedding")) if row.get("embedding") else None
    if emb is None and row.get("face_id") is not None:
        emb = vector_store.get_vector("face", row["face_id"])
    if emb is None:
        return {"saved_id": saved_id, "results": []}

    ids, scores = vector_store.search("face", emb, top_k=top_k * 3)
    score_by_face = dict(zip(ids, scores))
    face_rows = database.get_faces(ids)
    dets = {d["detection_id"]: d for d in database.get_detections([r["detection_id"] for r in face_rows])}
    cam_names = _camera_names()
    vindex = _video_index()

    results = []
    for r in face_rows:
        sim = float(score_by_face.get(r["face_id"], 0.0))
        if sim < config.FACE_SIMILAR_MIN:
            continue
        d = dets.get(r["detection_id"])
        if not d:
            continue
        item = to_result_item(d, sim, cam_names, vindex).model_dump()
        pb = playback_fields(d, vindex)
        results.append({
            "face_id": r["face_id"],
            "detection_id": r["detection_id"],
            "similarity": round(sim, 4),
            "camera_id": r.get("camera_id"),
            "camera_name": cam_names.get(r.get("camera_id")),
            "timestamp": r.get("timestamp"),
            "gender": r.get("gender"), "age": r.get("age"),
            "face_crop_url": media_url(r.get("crop_path")),
            "person_crop_url": media_url(d.get("crop_path")),
            "video_url": pb.get("video_url"),
            "offset_seconds": pb.get("offset_seconds"),
            "bbox": item.get("bbox"),
            "frame_width": item.get("frame_width"),
            "frame_height": item.get("frame_height"),
            "video_id": pb.get("video_id"),
        })
        if len(results) >= top_k:
            break
    results.sort(key=lambda x: x["similarity"], reverse=True)
    database.log_audit("search", query_type="face-similar", result_count=len(results))
    return {"saved_id": saved_id, "reference": _row_urls(row), "total": len(results), "results": results}
