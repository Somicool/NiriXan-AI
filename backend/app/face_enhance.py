"""Face Enhancement - derived visualisation from multiple verified views.

WHAT THIS IS
------------
Given a saved face, this does NOT sharpen the stored crop. It goes back to the
ORIGINAL recording, collects every face belonging to that same tracked person,
verifies each one against the saved identity, keeps only the best views, aligns
them onto a common facial geometry and fuses them into one image.

Multi-frame fusion is the point. A 12-pixel CCTV face contains almost no detail in
any single frame, but the same face across 8 frames contains slightly different
sub-pixel samples of the same subject. Combining them recovers real detail and
suppresses sensor noise.

WHY NOT GFPGAN / CodeFormer
---------------------------
Two reasons, one practical and one forensic.

Practical: GFPGAN and CodeFormer both depend on `basicsr`, which imports
`torchvision.transforms.functional_tensor`. That module was removed in
torchvision 0.17, and this environment runs 0.20.1, so the import fails. Making it
work means pinning older torch/torchvision, which would put the working YOLO, CLIP,
OSNet, InsightFace and Paddle stack at risk for one feature.

Forensic: those models are GENERATIVE. They restore a face by sampling a learned
prior of what faces look like. On the faces actually in this footage - measured at
10 to 25 pixels - almost every pixel of the output would be invented by the model
rather than derived from the evidence. Median fusion of aligned real frames cannot
invent a feature that was not photographed; it can only average what was.

So the default engine is multi-frame fusion. A GFPGAN backend is still wired in and
is used automatically IF the package is ever installed (see _gfpgan_restore), and
the model actually used is recorded on every result.

HONESTY
-------
The output is labelled "AI-Enhanced - Derived Visualisation" everywhere it appears.
The original saved face is never touched. Both images are hashed. When the evidence
is too thin the pipeline refuses and says so instead of producing something
confident-looking.
"""
from __future__ import annotations

import base64
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from . import config, database, faces_gallery


# --------------------------------------------------------------- small helpers
def _sha256(path) -> str | None:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def _decode_emb(b64: str | None):
    if not b64:
        return None
    try:
        v = np.frombuffer(base64.b64decode(b64), dtype="float32")
        n = float(np.linalg.norm(v))
        return (v / n) if n > 0 else None
    except Exception:
        return None


def _cosine(a, b) -> float:
    try:
        a = np.asarray(a, dtype="float32").ravel()
        b = np.asarray(b, dtype="float32").ravel()
        if a.size != b.size or a.size == 0:
            return 0.0
        na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))
    except Exception:
        return 0.0


def _sharpness(img) -> float:
    """Variance of Laplacian - only compared between images of the SAME size."""
    try:
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        return float(cv2.Laplacian(g, cv2.CV_64F).var())
    except Exception:
        return 0.0


# --------------------------------------------------------- candidate gathering
def _align(crop, kps, size: int):
    """Warp a face onto the canonical ArcFace geometry.

    Uses insightface's own landmark transform (already a dependency) so every
    candidate lands in the same coordinate frame - that is what makes fusing them
    meaningful. LANCZOS4 instead of the library default because these faces are
    being scaled UP by a large factor and the interpolation quality matters."""
    try:
        from insightface.utils import face_align
        kps = np.asarray(kps, dtype="float32")
        if kps.shape != (5, 2):
            return None
        # insightface changed this return value between versions: older builds
        # return (matrix, pose_index), 1.0.1 returns the 2x3 matrix on its own.
        # Unpacking blindly silently produced two 1-D rows and every warp failed.
        est = face_align.estimate_norm(kps, size)
        M = np.asarray(est[0] if isinstance(est, tuple) else est, dtype="float32")
        if M.shape != (2, 3):
            return None
        return cv2.warpAffine(crop, M, (size, size), flags=cv2.INTER_LANCZOS4,
                              borderMode=cv2.BORDER_REPLICATE)
    except Exception:
        return None


def collect_candidates(video_id, track_id, ref_emb, max_frames=None) -> dict:
    """Every usable face of this tracked person, scored and identity-checked.

    Reuses the EXISTING scan order, frame reader, face detector and 9-factor
    quality scorer from faces_gallery - this adds only the identity check and the
    canonical alignment. Nothing about best-face selection or saving is changed."""
    out = {"candidates": [], "frames_seen": 0, "faces_seen": 0,
           "rejected": {"quality": 0, "size": 0, "occluded": 0, "identity": 0, "align": 0},
           "reason": None, "video_path": None}
    if video_id is None or track_id is None:
        out["reason"] = "the saved face is not linked to a tracked person"
        return out

    max_frames = max_frames or config.FACE_ENH_SCAN_FRAMES
    with database.get_conn() as conn:
        dets = [dict(r) for r in conn.execute(
            "SELECT * FROM detections WHERE video_id=? AND track_id=? AND class_label!='scene'",
            (video_id, track_id)).fetchall()]
    if not dets:
        out["reason"] = "no stored detections for this person's track"
        return out

    vpath = faces_gallery.source_video_path(video_id)
    if vpath is None:
        out["reason"] = "the original recording is no longer on disk"
        return out
    out["video_path"] = str(vpath)

    # even coverage across the whole track, largest boxes always included
    cands = sorted(faces_gallery._scan_order(dets, max_frames),
                   key=lambda d: (d.get("frame_number") or 0))
    reader = faces_gallery._FrameReader(vpath)
    if not reader.ok:
        reader.close()
        out["reason"] = "the original recording could not be opened"
        return out

    try:
        for d in cands:
            frame = reader.read(d.get("frame_number"))
            if frame is None:
                continue
            crop, _off = faces_gallery._expanded_from_full_frame(frame, d)
            if crop is None or not crop.size:
                continue
            out["frames_seen"] += 1
            for f in faces_gallery._detect_faces_kps(crop):
                if float(f.det_score) < config.FACE_MIN_DET_SCORE:
                    continue
                out["faces_seen"] += 1
                m = faces_gallery._face_quality(f, crop)

                # --- quality gates (reject poor views) ---
                if m["face_size"] < config.FACE_ENH_MIN_PX:
                    out["rejected"]["size"] += 1
                    continue
                if m["occlusion"] < config.FACE_ENH_MIN_VISIBLE:
                    out["rejected"]["occluded"] += 1
                    continue
                if m["quality"] < config.FACE_ENH_MIN_QUALITY:
                    out["rejected"]["quality"] += 1
                    continue

                # --- identity gate: this must be the SAME person ---
                emb = getattr(f, "normed_embedding", None)
                sim = _cosine(ref_emb, emb) if (ref_emb is not None and emb is not None) else None
                if ref_emb is not None:
                    if sim is None or sim < config.FACE_ENH_MIN_IDENTITY:
                        out["rejected"]["identity"] += 1
                        continue

                aligned = _align(crop, f.kps, config.FACE_ENH_ALIGN_SIZE)
                if aligned is None or not aligned.size:
                    out["rejected"]["align"] += 1
                    continue

                out["candidates"].append({
                    "detection_id": d["detection_id"],
                    "frame_number": d.get("frame_number"),
                    "timestamp": d.get("timestamp"),
                    "quality": m["quality"], "face_size": m["face_size"],
                    "sharpness": m["sharpness"], "frontal": m["frontal"],
                    "brightness": m["brightness"], "occlusion": m["occlusion"],
                    "det_score": m["det_score"], "resolution": m["resolution"],
                    "identity": round(sim, 4) if sim is not None else None,
                    "_aligned": aligned,
                })
    finally:
        reader.close()

    # best first: quality, then how much it looks like the saved identity
    out["candidates"].sort(key=lambda c: (c["quality"], c["identity"] or 0), reverse=True)
    return out
