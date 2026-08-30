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


# ------------------------------------------------------------------- fusion
def _register(ref_gray, img):
    """Sub-pixel align one candidate onto the reference view.

    The landmark warp in _align is NOT accurate enough on its own here, and that
    was measured rather than assumed: with landmark alignment alone the median
    came out 9-22% NOISIER than the single best frame on every one of the 13 real
    saved faces. The reason is scale - InsightFace's 5 landmarks are off by a pixel
    or two on a 10-25 px face, and _align blows that face up to 224 px, so a 1.5 px
    landmark error becomes a 10-20 px misalignment. Taking a median across
    misaligned faces ghosts edges instead of averaging noise away.

    So each candidate is refined against the reference by intensity-based
    registration (ECC, euclidean: rotation + translation). It also doubles as a
    third verification stage: a view that cannot be registered to the reference is
    not describing the same thing and is dropped."""
    warp = np.eye(2, 3, dtype="float32")
    try:
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype("float32")
        cc, warp = cv2.findTransformECC(
            ref_gray, g, warp, cv2.MOTION_EUCLIDEAN,
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 80, 1e-6), None, 5)
    except cv2.error:
        return None, 0.0
    if not np.isfinite(cc) or cc < config.FACE_ENH_MIN_ECC:
        return None, float(cc if np.isfinite(cc) else 0.0)
    h, w = img.shape[:2]
    out = cv2.warpAffine(img, warp, (w, h),
                         flags=cv2.INTER_LANCZOS4 | cv2.WARP_INVERSE_MAP,
                         borderMode=cv2.BORDER_REPLICATE)
    return out, float(cc)


def _fuse(aligned: list) -> tuple:
    """Combine the verified views into one image.

    Per-pixel MEDIAN, not mean: a median ignores a frame where a limb, another
    head or a compression artefact crossed the face, where an average would smear
    it in. Returns (image, mode, indices actually used, ECC correlations)."""
    ref = aligned[0]
    if len(aligned) == 1:
        return ref.copy(), "single-frame", [0], [], None

    ref_gray = cv2.GaussianBlur(
        cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY).astype("float32"), (0, 0), 1.0)
    stack, used, ccs = [ref.astype("float32")], [0], []
    for i, img in enumerate(aligned[1:], start=1):
        reg, cc = _register(ref_gray, img)
        ccs.append(round(cc, 4))
        if reg is None:
            continue
        stack.append(reg.astype("float32"))
        used.append(i)

    if len(stack) == 1:
        return ref.copy(), "single-frame", used, ccs, None
    arr = np.stack(stack, axis=0)
    med = np.median(arr, axis=0)
    return np.clip(med, 0, 255).astype("uint8"), "median", used, ccs, arr


def _noise_stats(arr) -> dict:
    """Measure what the multi-frame step actually did to the noise.

    Deliberately NOT a median-residual reading of the output image. That measures
    high-frequency CONTENT, and both recovered detail and alignment ghosting raise
    it, so it cannot answer the question - it was tried first and reported fusion
    as worse on every track while sharpness simultaneously improved, which is the
    signature of a metric measuring the wrong thing.

    This is a half-split estimate instead, which needs no ground truth:
      - two frames differ by (noise_i - noise_j), so std(difference)/sqrt(2)
        estimates the noise in ONE frame
      - split the stack in half, median each half, and the same construction on the
        two half-medians estimates the noise left in a median of k/2 frames
    The output actually uses all k frames, so the reported reduction is a
    conservative floor. Residual misalignment inflates both terms, which keeps this
    an honest upper bound on the noise rather than a flattering one."""
    k = arr.shape[0]
    if k < 4:
        return {}
    g = arr.mean(axis=3) if arr.ndim == 4 else arr
    pairs = [float(np.std(g[i] - g[i + 1]) / np.sqrt(2.0)) for i in range(k - 1)]
    single = float(np.mean(pairs))
    a = np.median(g[0::2], axis=0)
    b = np.median(g[1::2], axis=0)
    fused = float(np.std(a - b) / np.sqrt(2.0))
    out = {"temporal_sigma": round(float(np.mean(np.std(g, axis=0))), 3),
           "noise_single_frame": round(single, 3),
           "noise_fused_halfsplit": round(fused, 3),
           "noise_halves": k // 2}
    if single > 0:
        out["noise_reduction_pct"] = round(100.0 * (single - fused) / single, 1)
    return out


def _upscale_sharpen(img, scale: int):
    """Upscale then apply a mild unsharp mask.

    LANCZOS4 for the resize. The unsharp is deliberately gentle: it raises local
    contrast on detail that is already present and cannot add features."""
    h, w = img.shape[:2]
    up = cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_LANCZOS4)
    blur = cv2.GaussianBlur(up, (0, 0), sigmaX=1.1)
    sharp = cv2.addWeighted(up, 1.0 + config.FACE_ENH_UNSHARP,
                            blur, -config.FACE_ENH_UNSHARP, 0)
    return np.clip(sharp, 0, 255).astype("uint8")


def _gfpgan_restore(img):
    """Optional generative restoration, used ONLY if the package is importable.

    Not installed here (basicsr needs torchvision.transforms.functional_tensor,
    removed in torchvision 0.17+; this environment runs 0.20.1). Left wired so the
    engine is a configuration choice rather than a rewrite. Returns None when
    unavailable, and the caller falls back to fusion and records which ran."""
    if config.FACE_ENH_MODEL not in ("gfpgan", "auto"):
        return None, None
    try:
        from gfpgan import GFPGANer
        weights = Path(config.FACE_ENH_GFPGAN_WEIGHTS)
        if not weights.exists():
            return None, None
        restorer = GFPGANer(model_path=str(weights), upscale=config.FACE_ENH_SCALE,
                            arch="clean", channel_multiplier=2, bg_upsampler=None)
        _c, _r, out = restorer.enhance(img, has_aligned=True, only_center_face=True,
                                       paste_back=False)
        if out is None:
            return None, None
        return out, f"GFPGAN v1.4 (upscale x{config.FACE_ENH_SCALE})"
    except Exception:
        return None, None


# ------------------------------------------------------------------ public API
def get_enhanced(saved_id: int) -> dict | None:
    """The stored enhancement for a saved face, if one was already produced."""
    with database.get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM enhanced_faces WHERE saved_face_id=? "
            "ORDER BY id DESC LIMIT 1", (int(saved_id),)).fetchone()
    return _row_out(dict(row)) if row else None


def _row_out(r: dict) -> dict:
    from .search.text_search import media_url
    for key in ("source_frames", "metrics"):
        if r.get(key):
            try:
                r[key] = json.loads(r[key])
            except (TypeError, ValueError):
                r[key] = None
    r["enhanced_url"] = media_url(r.get("file_path"))
    r["available"] = bool(r.get("file_path") and Path(r["file_path"]).exists())
    for fr in (r.get("source_frames") or []):
        if fr.get("path"):
            fr["url"] = media_url(fr["path"])
    return r
