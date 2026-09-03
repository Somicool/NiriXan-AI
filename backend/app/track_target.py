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
