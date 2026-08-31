"""Face Enhancement - find the clearest real view of a person, then only improve
it if the improvement can be proved.

WHAT THIS IS
------------
Given a saved face, this does NOT sharpen the stored crop. It goes back to the
ORIGINAL recording, collects every face belonging to that same tracked person,
verifies each one against the saved identity, and scores them all. The primary
output is the best NATURAL frame - a real photograph from somewhere else in the
person's track, which is very often far clearer than the frame that happened to
get saved. Restoration is secondary and optional.

    saved face -> analyse the whole track -> all verified appearances
    -> rank by real image quality -> BEST NATURAL FRAME
    -> optional light enhancement, accepted only if it measurably wins

WHY THAT ORDER (this was measured, and it reversed the original design)
----------------------------------------------------------------------
The first version fused the top 8 views onto a 224 px canvas and sharpened. On all
seven faces checked it made things worse in the way that matters most: mean
gradient magnitude, which collapses when an image is over-smoothed, fell by
roughly half against the best natural crop - e.g. 48.8 -> 22.8 and 40.3 -> 18.7.

Two causes, both fixed here:
  1. Every face was warped to a 224 px canvas. On a 12-25 px face that is a 10-20x
     upscale before any fusion, so the input to the median was already soft. The
     canvas is now chosen from the actual face size (112 for small faces).
  2. The 8 best-quality views were fused unconditionally, including distant small
     ones. Averaging a 10 px view into a 40 px view destroys the detail the good
     view had. Fusion now requires several views of comparable scale that register
     tightly, and is skipped entirely otherwise.

THE QUALITY GATE
----------------
Nothing is presented as enhanced on trust. Every candidate enhancement is compared
against the best aligned source frame at identical size on edge energy, sharpness,
clipped pixels, and - the important one - identity: the output is re-embedded with
ArcFace and must not drift away from the saved reference. An enhancement that
smooths detail away, clips, or moves the face towards someone else is REJECTED and
the best natural frame is returned instead, labelled
"Best Available Original Evidence". Saying the footage was already as good as it
gets is a valid, honest answer.

WHY NOT GFPGAN / CodeFormer AS THE DEFAULT
------------------------------------------
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

A GFPGAN backend is still wired in and is used automatically IF the package is ever
installed (see _gfpgan_restore), at a deliberately conservative restoration weight,
and it has to pass the same gate as everything else. Whichever engine won is
recorded on every result.

HONESTY
-------
An accepted enhancement is labelled "AI-Enhanced - Derived Visualisation"; a
rejected one is labelled "Best Available Original Evidence" with the reason it was
rejected. The original saved face is never touched. Both images are hashed. When
the track holds no usable facial evidence at all the pipeline refuses outright
instead of producing something confident-looking.
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


def _edge_energy(img) -> float:
    """Mean Sobel gradient magnitude.

    This is the over-smoothing detector. Variance of Laplacian can be pushed up by
    an unsharp mask even while real structure is being lost, but mean gradient
    magnitude falls when edges are washed out. It is what caught the original
    pipeline degrading every one of the faces it was tested on."""
    try:
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        return float(np.mean(cv2.magnitude(gx, gy)))
    except Exception:
        return 0.0


def _saturated_frac(img) -> float:
    """Fraction of pixels crushed to pure black or blown to pure white.

    Sharpening overshoots into clipping, which reads as 'crisp' on a sharpness
    metric while actually destroying information."""
    try:
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        return float(np.mean((g <= 2) | (g >= 253)))
    except Exception:
        return 0.0


def _embed_aligned(img):
    """ArcFace embedding of an ALREADY-ALIGNED face crop.

    Uses the recognition model directly rather than running the detector first.
    Detection fails on these images for a reason that has nothing to do with
    quality - the face fills the entire frame with no surrounding context, and
    SCRFD needs some margin (verified: 0 faces found on a 448 px aligned crop, 1
    face at det=0.82 on the same image with 50% padding added). Since the crop is
    already in canonical ArcFace geometry, get_feat is the correct entry point and
    gives a comparable embedding for any candidate image."""
    try:
        from .ingestion import face_recognizer
        rec = getattr(face_recognizer.get_face_app(), "models", {}).get("recognition")
        if rec is None or img is None or not img.size:
            return None
        a = cv2.resize(img, (112, 112), interpolation=cv2.INTER_AREA)
        v = np.asarray(rec.get_feat(a), dtype="float32").ravel()
        n = float(np.linalg.norm(v))
        return (v / n) if n > 0 else None
    except Exception:
        return None


# --------------------------------------------------------- candidate gathering
def align_size_for(face_px: int) -> int:
    """Alignment canvas chosen from the ACTUAL face size.

    A fixed 224 canvas was the single biggest cause of soft output: it upscales a
    12 px face by 18x before anything else runs. Small faces stay on the 112 px
    canonical canvas, which is still an upscale but a far smaller one."""
    return (config.FACE_ENH_ALIGN_LARGE if (face_px or 0) >= config.FACE_ENH_ALIGN_SWITCH
            else config.FACE_ENH_ALIGN_SMALL)


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
    quality scorer from faces_gallery - this adds only the identity check.
    Nothing about best-face selection or saving is changed.

    Alignment deliberately does NOT happen here. The canvas size depends on the
    winning face's size, which is not known until every candidate has been scored,
    so each candidate keeps its source crop and landmarks and is warped later. That
    also means only the handful of views actually used get warped."""
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
            crop, (ox, oy) = faces_gallery._expanded_from_full_frame(frame, d)
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

                if getattr(f, "kps", None) is None:
                    out["rejected"]["align"] += 1
                    continue

                # face box in ORIGINAL-FRAME coordinates, so the winning view can be
                # re-cropped from the source video at its true native resolution
                fx1, fy1, fx2, fy2 = m["bbox"]
                out["candidates"].append({
                    "detection_id": d["detection_id"],
                    "frame_number": d.get("frame_number"),
                    "timestamp": d.get("timestamp"),
                    "quality": m["quality"], "face_size": m["face_size"],
                    "sharpness": m["sharpness"], "frontal": m["frontal"],
                    "brightness": m["brightness"], "occlusion": m["occlusion"],
                    "det_score": m["det_score"], "resolution": m["resolution"],
                    "identity": round(sim, 4) if sim is not None else None,
                    "frame_bbox": [fx1 + ox, fy1 + oy, fx2 + ox, fy2 + oy],
                    "_crop": crop, "_kps": np.asarray(f.kps, dtype="float32"),
                })
    finally:
        reader.close()

    # best first: quality, then how much it looks like the saved identity
    out["candidates"].sort(key=lambda c: (c["quality"], c["identity"] or 0), reverse=True)
    out["detections"] = dets
    return out


def _interp_box(dets_by_frame: list, frame: int):
    """Person box at an UNINDEXED native frame, interpolated between the stored
    detections either side. A person's bounding box moves smoothly between two
    samples a fraction of a second apart, so linear interpolation is sound."""
    prev = nxt = None
    for d in dets_by_frame:
        fn = d.get("frame_number")
        if fn is None:
            continue
        if fn <= frame:
            prev = d
        if fn >= frame:
            nxt = d
            break
    if prev is None and nxt is None:
        return None, None
    if prev is None or nxt is None:
        d = prev or nxt
        return {**d, "frame_number": frame}, d
    f0, f1 = prev["frame_number"], nxt["frame_number"]
    if f1 == f0:
        return {**prev, "frame_number": frame}, prev
    t = (frame - f0) / float(f1 - f0)
    box = {}
    for k in ("bbox_x", "bbox_y", "bbox_w", "bbox_h"):
        a, b = prev.get(k), nxt.get(k)
        if a is None or b is None:
            return None, None
        box[k] = a + (b - a) * t
    near = prev if t < 0.5 else nxt
    return {**near, **box, "frame_number": frame}, near


def refine_around_best(video_id, got: dict, ref_emb) -> dict:
    """Second pass: look at the native frames NOBODY has ever examined.

    The sparse pass can only see frames that ingestion happened to sample, and
    Save Face already chose the best of those with the same scorer - so the sparse
    pass alone almost never beats the saved face. This re-reads the source video at
    native frame rate around the best few views, interpolating the person box, and
    is where a genuinely clearer natural appearance actually turns up.

    Adds any better views to got["candidates"] and returns the pass statistics."""
    stats = {"frames": 0, "faces": 0, "added": 0, "seeds": []}
    if not config.FACE_ENH_REFINE or not got.get("candidates"):
        return stats
    dets = sorted([d for d in (got.get("detections") or [])
                   if d.get("frame_number") is not None],
                  key=lambda d: d["frame_number"])
    if not dets:
        return stats
    lo, hi = dets[0]["frame_number"], dets[-1]["frame_number"]
    vpath = faces_gallery.source_video_path(video_id)
    if vpath is None:
        return stats

    seen = {c["frame_number"] for c in got["candidates"]}
    targets: list[int] = []
    for seed in got["candidates"][:config.FACE_ENH_REFINE_SEEDS]:
        sf = seed["frame_number"]
        stats["seeds"].append(sf)
        for off in range(-config.FACE_ENH_REFINE_RADIUS,
                         config.FACE_ENH_REFINE_RADIUS + 1,
                         config.FACE_ENH_REFINE_STRIDE):
            f = int(sf) + off
            if lo <= f <= hi and f not in seen:
                seen.add(f)
                targets.append(f)
    targets = sorted(targets)[:config.FACE_ENH_REFINE_MAX]
    if not targets:
        return stats

    reader = faces_gallery._FrameReader(vpath)
    if not reader.ok:
        reader.close()
        return stats
    try:
        for f in targets:
            synth, near = _interp_box(dets, f)
            if synth is None:
                continue
            frame = reader.read(f)
            if frame is None:
                continue
            crop, (ox, oy) = faces_gallery._expanded_from_full_frame(frame, synth)
            if crop is None or not crop.size:
                continue
            stats["frames"] += 1
            for face in faces_gallery._detect_faces_kps(crop):
                if float(face.det_score) < config.FACE_MIN_DET_SCORE:
                    continue
                stats["faces"] += 1
                m = faces_gallery._face_quality(face, crop)
                if (m["face_size"] < config.FACE_ENH_MIN_PX
                        or m["occlusion"] < config.FACE_ENH_MIN_VISIBLE
                        or m["quality"] < config.FACE_ENH_MIN_QUALITY):
                    continue
                emb = getattr(face, "normed_embedding", None)
                sim = _cosine(ref_emb, emb) if (ref_emb is not None and emb is not None) else None
                if ref_emb is not None and (sim is None or sim < config.FACE_ENH_MIN_IDENTITY):
                    continue
                if getattr(face, "kps", None) is None:
                    continue
                fx1, fy1, fx2, fy2 = m["bbox"]
                got["candidates"].append({
                    "detection_id": near["detection_id"],
                    "frame_number": f, "timestamp": near.get("timestamp"),
                    "quality": m["quality"], "face_size": m["face_size"],
                    "sharpness": m["sharpness"], "frontal": m["frontal"],
                    "brightness": m["brightness"], "occlusion": m["occlusion"],
                    "det_score": m["det_score"], "resolution": m["resolution"],
                    "identity": round(sim, 4) if sim is not None else None,
                    "frame_bbox": [fx1 + ox, fy1 + oy, fx2 + ox, fy2 + oy],
                    "refined": True,
                    "_crop": crop, "_kps": np.asarray(face.kps, dtype="float32"),
                })
                stats["added"] += 1
    finally:
        reader.close()

    got["candidates"].sort(key=lambda c: (c["quality"], c["identity"] or 0), reverse=True)
    return stats


# ------------------------------------------------------------------- fusion
def _register(ref_gray, img, min_ecc: float | None = None):
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
    floor = config.FACE_ENH_MIN_ECC if min_ecc is None else min_ecc
    warp = np.eye(2, 3, dtype="float32")
    try:
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype("float32")
        cc, warp = cv2.findTransformECC(
            ref_gray, g, warp, cv2.MOTION_EUCLIDEAN,
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 80, 1e-6), None, 5)
    except cv2.error:
        return None, 0.0
    if not np.isfinite(cc) or cc < floor:
        return None, float(cc if np.isfinite(cc) else 0.0)
    h, w = img.shape[:2]
    out = cv2.warpAffine(img, warp, (w, h),
                         flags=cv2.INTER_LANCZOS4 | cv2.WARP_INVERSE_MAP,
                         borderMode=cv2.BORDER_REPLICATE)
    return out, float(cc)


def _fuse(aligned: list, min_ecc: float | None = None) -> tuple:
    """Combine the verified views into one image.

    Per-pixel MEDIAN, not mean: a median ignores a frame where a limb, another
    head or a compression artefact crossed the face, where an average would smear
    it in. `aligned[0]` is the reference and is always kept.

    Returns (image, mode, indices used, ECC correlations, registered stack)."""
    ref = aligned[0]
    if len(aligned) == 1:
        return ref.copy(), "single-frame", [0], [], None

    ref_gray = cv2.GaussianBlur(
        cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY).astype("float32"), (0, 0), 1.0)
    stack, used, ccs = [ref.astype("float32")], [0], []
    for i, img in enumerate(aligned[1:], start=1):
        reg, cc = _register(ref_gray, img, min_ecc)
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


def enhance(saved_id: int, force: bool = False) -> dict:
    """Produce (or return the cached) enhanced visualisation for a saved face.

    On-demand only - nothing here runs during ingestion. Cached by default so
    reopening a face is instant; `force=True` re-runs it."""
    if not force:
        cached = get_enhanced(saved_id)
        if cached and cached.get("available"):
            cached["cached"] = True
            return cached

    with database.get_conn() as conn:
        row = conn.execute("SELECT * FROM saved_faces WHERE saved_id=?",
                           (int(saved_id),)).fetchone()
    if row is None:
        return {"error": "Saved face not found."}
    saved = dict(row)

    refs = database.get_detections([saved["detection_id"]])
    det = refs[0] if refs else {}
    vid, tid = det.get("video_id"), det.get("track_id")
    ref_emb = _decode_emb(saved.get("embedding"))

    t0 = datetime.now()
    got = collect_candidates(vid, tid, ref_emb)
    keep = got["candidates"][:config.FACE_ENH_MAX_FRAMES]

    # Refuse rather than fabricate.
    if len(keep) < config.FACE_ENH_MIN_FRAMES:
        return {
            "error": "Insufficient high-quality facial evidence available for "
                     "reliable enhancement.",
            "reason": got.get("reason"),
            "frames_analysed": got["frames_seen"], "faces_found": got["faces_seen"],
            "frames_selected": len(keep), "rejected": got["rejected"],
            "identity_threshold": config.FACE_ENH_MIN_IDENTITY,
            "min_face_px": config.FACE_ENH_MIN_PX,
        }

    fused, fusion, used, ccs, stack = _fuse([c["_aligned"] for c in keep])
    dropped_reg = len(keep) - len(used)
    keep = [keep[i] for i in used]          # only the views actually in the result

    restored, model = _gfpgan_restore(fused)
    if restored is not None:
        final = restored
    else:
        final = _upscale_sharpen(fused, config.FACE_ENH_SCALE)
        stage = (f"multi-frame median fusion of {len(keep)} verified views"
                 if fusion == "median" else
                 "single verified view (no second view available to fuse)")
        model = (f"{stage} + LANCZOS x{config.FACE_ENH_SCALE} + unsharp "
                 f"(local, non-generative)")

    # --------------------------------------------------- write derived evidence
    config.ENHANCED_FACE_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = config.ENHANCED_FACE_DIR / f"enh_{saved_id}_{stamp}.jpg"
    cv2.imwrite(str(out_path), final, [int(cv2.IMWRITE_JPEG_QUALITY), 96])

    # the exact frames the result was built from, so an investigator can audit it
    frames_dir = config.ENHANCED_FACE_DIR / f"src_{saved_id}_{stamp}"
    frames_dir.mkdir(parents=True, exist_ok=True)
    source_frames = []
    for i, c in enumerate(keep, 1):
        p = frames_dir / f"f{i:02d}_frame{c['frame_number']}.jpg"
        cv2.imwrite(str(p), c["_aligned"], [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        source_frames.append({k: c[k] for k in
                              ("detection_id", "frame_number", "timestamp", "quality",
                               "face_size", "sharpness", "frontal", "brightness",
                               "occlusion", "det_score", "identity")} | {"path": str(p)})

    # ------------------------------------------------- measurable before/after
    # Every image is brought to the SAME output resolution first, otherwise a
    # variance-of-Laplacian comparison is meaningless. Three stages are measured
    # so the two effects stay separable:
    #   best_up  - the single best real frame, upscaled          (the "before")
    #   fused_up - the median of the verified views, upscaled    (what FUSION did)
    #   final    - after the unsharp mask                        (the "after")
    # Reporting only before/after would let the unsharp mask take credit for the
    # fusion, and would make fusion's noise reduction invisible.
    size = final.shape[1]
    best_up = cv2.resize(keep[0]["_aligned"], (size, size), interpolation=cv2.INTER_LANCZOS4)
    fused_up = cv2.resize(fused, (size, size), interpolation=cv2.INTER_LANCZOS4)

    def _noise(img) -> float:
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return round(float(cv2.absdiff(g, cv2.medianBlur(g, 3)).std()), 3)

    src_px = int(max(c["face_size"] for c in keep))
    metrics = {
        "frames_analysed": got["frames_seen"],
        "faces_found": got["faces_seen"],
        "frames_selected": len(keep),
        "rejected": got["rejected"],
        "source_face_px": src_px,
        "output_px": int(size),
        "align_size": config.FACE_ENH_ALIGN_SIZE,
        "fusion": fusion,
        "registration_cc": ccs,             # ECC correlation per extra view
        "dropped_registration": dropped_reg,
        "sharpness_best_frame": round(_sharpness(best_up), 2),
        "sharpness_fused": round(_sharpness(fused_up), 2),
        "sharpness_enhanced": round(_sharpness(final), 2),
        # median-residual reading of each stage. Kept, but named for what it
        # actually measures - high-frequency content, not noise. Rising here is
        # expected once an unsharp mask is applied and must not be read as a
        # quality loss (see _noise_stats for the real noise measurement).
        "detail_best_frame": _noise(best_up),
        "detail_fused": _noise(fused_up),
        "detail_enhanced": _noise(final),
        "identity_min": min([c["identity"] for c in keep if c["identity"] is not None] or [None]),
        "identity_max": max([c["identity"] for c in keep if c["identity"] is not None] or [None]),
        "quality_best": round(keep[0]["quality"], 4),
        "elapsed_s": round((datetime.now() - t0).total_seconds(), 2),
        "note": ("Sharpness rises partly from the unsharp mask, so the fused column "
                 "shows what the multi-frame step contributed on its own. Noise "
                 "figures are a half-split measurement and are a conservative floor."),
        **(_noise_stats(stack) if stack is not None else {}),
    }

    # Source quality must reflect how much real facial information the FOOTAGE
    # held, which is dominated by how many pixels the face occupied. The composite
    # face score alone rated a 25 px face "High" because size carries only 0.18 of
    # its weight - misleading on an evidence report, so size leads here.
    src_q = ("High" if (src_px >= 60 and keep[0]["quality"] >= 0.58) else
             "Medium" if (src_px >= 30 and keep[0]["quality"] >= 0.48) else "Low")

    rec = {
        "saved_face_id": int(saved_id),
        "source_video_id": vid, "source_track_id": tid,
        "source_camera_id": saved.get("camera_id"),
        "original_detection_ids": json.dumps([c["detection_id"] for c in keep]),
        "source_timestamps": json.dumps([c["timestamp"] for c in keep]),
        "best_source_timestamp": keep[0]["timestamp"],
        "model_name": model,
        "frames_analysed": got["frames_seen"], "frames_selected": len(keep),
        "quality_score": round(keep[0]["quality"], 4),
        "source_quality": src_q,
        "original_hash": _sha256(saved.get("face_crop")),
        "enhanced_hash": _sha256(out_path),
        "file_path": str(out_path),
        "source_frames": json.dumps(source_frames),
        "metrics": json.dumps(metrics),
        "label": "AI-Enhanced - Derived Visualisation",
        "created_at": _now(),
    }
    with database.get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO enhanced_faces (saved_face_id, source_video_id, source_track_id, "
            " source_camera_id, original_detection_ids, source_timestamps, "
            " best_source_timestamp, model_name, frames_analysed, frames_selected, "
            " quality_score, source_quality, original_hash, enhanced_hash, file_path, "
            " source_frames, metrics, label, created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            tuple(rec[k] for k in (
                "saved_face_id", "source_video_id", "source_track_id", "source_camera_id",
                "original_detection_ids", "source_timestamps", "best_source_timestamp",
                "model_name", "frames_analysed", "frames_selected", "quality_score",
                "source_quality", "original_hash", "enhanced_hash", "file_path",
                "source_frames", "metrics", "label", "created_at")))
        stored = dict(conn.execute("SELECT * FROM enhanced_faces WHERE id=?",
                                   (cur.lastrowid,)).fetchone())

    database.log_audit("face_enhance", query_type="derived-visualisation",
                       result_count=len(keep),
                       details={"saved_face_id": saved_id, "model": model,
                                "enhanced_hash": rec["enhanced_hash"]})
    out = _row_out(stored)
    out["cached"] = False
    return out


def list_enhanced(saved_id: int | None = None) -> list[dict]:
    q = "SELECT * FROM enhanced_faces"
    params: tuple = ()
    if saved_id is not None:
        q += " WHERE saved_face_id=?"
        params = (int(saved_id),)
    q += " ORDER BY id DESC"
    with database.get_conn() as conn:
        return [_row_out(dict(r)) for r in conn.execute(q, params).fetchall()]


def delete_enhanced(enh_id: int) -> dict:
    """Remove a derived visualisation. The saved face is never affected."""
    with database.get_conn() as conn:
        row = conn.execute("SELECT file_path FROM enhanced_faces WHERE id=?",
                           (int(enh_id),)).fetchone()
        conn.execute("DELETE FROM enhanced_faces WHERE id=?", (int(enh_id),))
    try:
        if row and row["file_path"] and str(config.ENHANCED_FACE_DIR) in str(row["file_path"]):
            Path(row["file_path"]).unlink(missing_ok=True)
    except Exception:
        pass
    return {"deleted": enh_id}
