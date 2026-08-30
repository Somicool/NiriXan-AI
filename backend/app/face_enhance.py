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
