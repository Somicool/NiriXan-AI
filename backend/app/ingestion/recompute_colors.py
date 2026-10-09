"""Recompute person upper/lower clothing colours for detections already stored,
using the region-split + HSV method (attribute_extractor). Fixes existing data
without re-running the whole detection/tracking pipeline - it only re-reads each
person crop and updates the colour attributes.

Batches the CLIP embedding of the upper/lower halves for speed. Optionally scoped
to one video_id. Run via POST /api/recompute-colors.
"""
from __future__ import annotations

import json
import os

import cv2

from .. import config, database, ingest_jobs, ingest_progress
from . import embedder, attribute_extractor as ax

_PERSON_LABELS = [config.DETECT_CLASSES[c] for c in config.PERSON_CLASSES]
_UPPER_PROMPTS = [f"a person wearing a {c} colored top" for c in config.COLORS]
_LOWER_PROMPTS = [f"a person wearing {c} colored trousers" for c in config.COLORS]


def _color_of(emb, region, which: str):
    """CLIP colour of a (prepped) region, fused with HSV using the same
    confidence-weighted rule as live ingest. `region` is the centre-sampled,
    illumination-normalised half that `emb` was computed from."""
    key = "person_upper" if which == "upper" else "person_lower"
    prompts = _UPPER_PROMPTS if which == "upper" else _LOWER_PROMPTS
    clip_color, score = ax._best(emb, key, prompts, config.COLORS)
    return ax.fuse_clip_hsv(clip_color, score, region)


def recompute_colors(video_id=None, job_id=None, batch: int = 128) -> dict:
    ph = ",".join("?" * len(_PERSON_LABELS))
    q = (f"SELECT detection_id, video_id, crop_path, attributes FROM detections "
         f"WHERE class_label IN ({ph})")
    params = list(_PERSON_LABELS)
    if video_id is not None:
        q += " AND video_id=?"
        params.append(video_id)
    with database.get_conn() as conn:
        rows = [dict(r) for r in conn.execute(q, params).fetchall()]

    total, updated = len(rows), 0
    colorless_ids = database.colorless_video_ids()
    for i in range(0, total, batch):
        chunk = rows[i:i + batch]
        uppers, lowers, keep = [], [], []
        for r in chunk:
            cp = r.get("crop_path")
            img = cv2.imread(cp) if cp and os.path.exists(cp) else None
            if img is None or not img.size:
                continue
            if r.get("video_id") in colorless_ids:
                # Night-vision / B&W clip: black/white/unknown only, same reading
                # as ingestion, so a recompute can never reintroduce a hue.
                uc, ucs, lc, lcs = ax.bw_person_colors(img)
                try:
                    attrs = json.loads(r["attributes"]) if r["attributes"] else {}
                except (TypeError, ValueError):
                    attrs = {}
                attrs.update(upper_color=uc, upper_color_score=ucs, lower_color=lc,
                             lower_color_score=lcs, color_mode="bw")
                with database.get_conn() as conn:
                    conn.execute("UPDATE detections SET attributes=? WHERE detection_id=?",
                                 (json.dumps(attrs), r["detection_id"]))
                updated += 1
                continue
            h = img.shape[0]
            cut = max(1, int(round(0.40 * h)))
            up, lo = img[:cut, :], img[cut:, :]
            if not up.size or not lo.size:
                continue
            up, lo = ax.prep_region(up), ax.prep_region(lo)   # centre + illumination (match ingest)
            if up is None or lo is None or not up.size or not lo.size:
                continue
            uppers.append(up)
            lowers.append(lo)
            keep.append(r)
        if not keep:
            _progress(job_id, min(i + batch, total), total)
            continue

        uemb = embedder.embed_images(uppers)
        lemb = embedder.embed_images(lowers)
        with database.get_conn() as conn:
            for r, ureg, lreg, ue, le in zip(keep, uppers, lowers, uemb, lemb):
                uc, ucs = _color_of(ue, ureg, "upper")
                lc, lcs = _color_of(le, lreg, "lower")
                try:
                    attrs = json.loads(r["attributes"]) if r["attributes"] else {}
                except (TypeError, ValueError):
                    attrs = {}
                attrs.update(upper_color=uc, upper_color_score=ucs,
                             lower_color=lc, lower_color_score=lcs)
                conn.execute("UPDATE detections SET attributes=? WHERE detection_id=?",
                             (json.dumps(attrs), r["detection_id"]))
                updated += 1
        _progress(job_id, min(i + batch, total), total)

    if job_id:
        ingest_jobs.update(job_id, status="done", done=total, total=total, current=None)
    ingest_progress.reset()
    return {"persons": total, "updated": updated, "video_id": video_id}


def _progress(job_id, done, total):
    ingest_progress.set_progress(done, total)
    if job_id:
        ingest_jobs.update(job_id, status="processing", done=done, total=total,
                           current="recomputing colours")


if __name__ == "__main__":
    print(recompute_colors())
