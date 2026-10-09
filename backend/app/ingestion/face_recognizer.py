"""Face detection + embedding (InsightFace buffalo_l). Bonus feature, gated by
config.FACE_RECOGNITION_ENABLED for ethics.

detect_faces(image) -> [{embedding(512 normed float32), bbox, age, gender, det_score}]
Runs on GPU (CUDAExecutionProvider) when config.DEVICE == "cuda", else CPU.
Requires an onnxruntime-gpu build matching the installed CUDA/cuDNN (CUDA 12 + cuDNN 9
for torch cu121); otherwise it silently falls back to CPU.
"""
from __future__ import annotations

import cv2

from .. import config

import threading

_app = None
# Save Face, the background prepare scan and the startup warm-up can all reach
# this first; without a lock two threads each built (and held) a full copy.
_app_lock = threading.Lock()


def get_face_app():
    global _app
    if _app is not None:
        return _app
    with _app_lock:
        return _load_face_app()


def _load_face_app():
    global _app
    if _app is None:
        from insightface.app import FaceAnalysis
        providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                     if config.DEVICE == "cuda" else ["CPUExecutionProvider"])
        # Only the three models whose outputs are used: detection (bbox, 5-point
        # kps, det_score), recognition (embedding) and genderage. buffalo_l also
        # ships 3D-68 and 2D-106 landmark models that ran on every face for
        # nothing - no code reads their output. Verified on 36 real faces: bbox,
        # kps, det_score, embedding, age and gender are identical without them,
        # and per-face analysis is ~33% faster (41 -> 28 ms per crop).
        _app = FaceAnalysis(name=config.FACE_MODEL, providers=providers,
                            allowed_modules=["detection", "recognition", "genderage"])
        _app.prepare(ctx_id=0 if config.DEVICE == "cuda" else -1, det_size=(640, 640))
    return _app


def _gender(face):
    g = getattr(face, "sex", None)
    if g in ("M", "F"):
        return "male" if g == "M" else "female"
    gi = getattr(face, "gender", None)
    if gi is None:
        return None
    try:
        return "male" if int(gi) == 1 else "female"
    except (TypeError, ValueError):
        return None


def _upscale(img, min_side: int = 320):
    """Enlarge a small person crop so InsightFace can find/analyse the face."""
    if img is None or not img.size:
        return img
    h, w = img.shape[:2]
    m = min(h, w)
    if m and m < min_side:
        f = min_side / m
        img = cv2.resize(img, (int(w * f), int(h * f)), interpolation=cv2.INTER_CUBIC)
    return img


def detect_faces(image) -> list[dict]:
    if isinstance(image, str):
        image = cv2.imread(image)
    if image is None:
        return []
    image = _upscale(image)
    results = []
    for f in get_face_app().get(image):
        results.append({
            "embedding": f.normed_embedding.astype("float32"),
            "bbox": [float(x) for x in f.bbox.tolist()],
            "age": int(f.age) if getattr(f, "age", None) is not None else None,
            "gender": _gender(f),
            "det_score": float(f.det_score),
        })
    return results


def detect_faces_voted(crop_paths, max_frames: int | None = None) -> dict | None:
    """Aggregate gender/age across several crops of the SAME person track.

    Runs detection on up to `max_frames` crops, keeps only faces above
    config.FACE_DET_MIN, majority-votes gender (weighted by det_score) and takes a
    det_score-weighted mean age. Returns one face dict (embedding = the single
    highest-quality face, for the face index) with the aggregated gender/age, or
    None if no confident face was found. Reduces the single-frame errors that
    dominate CCTV gender estimation.
    """
    if max_frames is None:
        max_frames = config.FACE_VOTE_FRAMES
    best = None
    gender_w: dict[str, float] = {}
    age_acc = age_w = 0.0
    for cp in list(crop_paths)[:max_frames]:
        for f in detect_faces(cp):
            if f["det_score"] < config.FACE_DET_MIN:      # quality gate
                continue
            w = float(f["det_score"])
            if f["gender"]:
                gender_w[f["gender"]] = gender_w.get(f["gender"], 0.0) + w
            if f["age"] is not None:
                age_acc += f["age"] * w
                age_w += w
            if best is None or f["det_score"] > best["det_score"]:
                best = f
    if best is None:
        return None
    gender = max(gender_w, key=gender_w.get) if gender_w else best["gender"]
    age = int(round(age_acc / age_w)) if age_w > 0 else best["age"]
    return {**best, "gender": gender, "age": age}


if __name__ == "__main__":
    from ultralytics.utils import ASSETS
    faces = detect_faces(str(ASSETS / "bus.jpg"))
    print(f"detected {len(faces)} faces in bus.jpg")
    for f in faces:
        print(f"  age={f['age']} gender={f['gender']} det={f['det_score']:.2f} "
              f"bbox={[round(x) for x in f['bbox']]}")
