"""Benchmark Track Person: stored ingestion replay vs query-time target pass.

    python scripts/benchmark_track_target.py [n_tracks]

HONESTY NOTE
------------
There are no hand-labelled identity ground truths for this footage, so the two
kinds of number are kept apart:

  MEASURED (technical, no labels needed)
    frames analysed, frames with a verified box, coverage %, temporarily-lost
    segments, re-acquisitions, interpolated boxes, span covered, seconds elapsed,
    GPU memory.

  PROXY (depends on ReID being right)
    "genuine" similarity = best candidate per frame; "impostor" similarity =
    runner-up in the same frame. On a populated street the runner-up is almost
    always a different person, which makes it a fair impostor sample - but it is
    still a proxy. Identity-switch counts below are therefore reported as
    "boxes that changed identity beyond the continuity limit", which is an
    observable event, NOT a claim of ground-truth accuracy.
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import config, database, track_target                      # noqa: E402
from app.search import track_path                                   # noqa: E402


def gpu_mb():
    try:
        import torch
        if torch.cuda.is_available():
            return round(torch.cuda.memory_allocated() / 1e6, 1)
    except Exception:
        pass
    return None


def jumps(points, fw, fh, limit_frac=0.25):
    """Boxes that moved further between consecutive samples than a person plausibly
    could. An observable discontinuity - the visible form of an identity switch."""
    diag = math.hypot(fw or 1920, fh or 1080)
    n = 0
    for a, b in zip(points, points[1:]):
        dt = max((b["offset_seconds"] or 0) - (a["offset_seconds"] or 0), 1e-3)
        ca = (a["bbox"][0] + a["bbox"][2] / 2, a["bbox"][1] + a["bbox"][3] / 2)
        cb = (b["bbox"][0] + b["bbox"][2] / 2, b["bbox"][1] + b["bbox"][3] / 2)
        if math.hypot(cb[0] - ca[0], cb[1] - ca[1]) > limit_frac * diag * max(dt, 0.2) / 0.2:
            n += 1
    return n


def main(n_tracks: int = 5) -> int:
    database.init_db()
    person = [config.DETECT_CLASSES[c] for c in config.PERSON_CLASSES]
    with database.get_conn() as c:
        tracks = [dict(r) for r in c.execute(
            "SELECT video_id, track_id, COUNT(*) n FROM detections WHERE class_label IN "
            "({}) AND track_id IS NOT NULL GROUP BY video_id, track_id HAVING n >= 25 "
            "ORDER BY n DESC LIMIT ?".format(",".join("?" * len(person))),
            (*person, n_tracks)).fetchall()]
        vids = {r["video_id"]: dict(r) for r in c.execute("SELECT * FROM videos")}

    base_gpu = gpu_mb()
    rows, gen, imp = [], [], []
    for t in tracks:
        det_rows = database.get_track_detections(t["video_id"], t["track_id"])
        if not det_rows:
            continue
        did = det_rows[len(det_rows) // 2]["detection_id"]
        v = vids.get(t["video_id"], {})
        fw, fh = v.get("width"), v.get("height")

        old = track_path.get_track_path(did)
        old_pts = [{"offset_seconds": p.offset_seconds, "bbox": p.bbox} for p in old.points]

        t0 = time.time()
        new = track_target.retrack(did, force=True)
        el = round(time.time() - t0, 1)
        if new.get("error"):
            print(f"video {t['video_id']} track {t['track_id']}: {new['error'][:70]}")
            continue
        m = new["metrics"]
        for d in (new.get("debug") or []):
            if d.get("best_sim") is not None:
                gen.append(d["best_sim"])
            if d.get("runner_sim") is not None:
                imp.append(d["runner_sim"])

        rows.append({
            "vid": t["video_id"], "trk": t["track_id"],
            "old_boxes": len(old.points), "old_interp": old.predicted_points or 0,
            "old_span": round((old.end_offset or 0) - (old.start_offset or 0), 1),
            "old_jumps": jumps(old_pts, fw, fh),
            "new_boxes": m["frames_with_target"], "new_interp": m["predicted_boxes"],
            "new_span": round((new["end_offset"] or 0) - (new["start_offset"] or 0), 1),
            "new_jumps": jumps(new["points"], fw, fh),
            "analysed": m["frames_analysed"], "cover": m["coverage_pct"],
            "fps": m["analysis_fps"], "lost": m["lost_segments"],
            "reacq": m["reacquisitions"], "idmean": m["identity_mean"], "sec": el,
        })

    if not rows:
        print("no tracks benchmarked")
        return 1

    print("\n=== MEASURED: stored ingestion replay vs query-time target pass ===")
    hdr = (f"{'vid':>4} {'track':>7} | {'old box':>7} {'old int':>7} {'old jmp':>7} "
           f"{'old span':>8} | {'new box':>7} {'new int':>7} {'new jmp':>7} "
           f"{'new span':>8} {'cover%':>7} {'fps':>5} {'lost':>5} {'reacq':>6} {'sec':>6}")
    print(hdr); print("-" * len(hdr))
    for r in rows:
        print(f"{r['vid']:>4} {r['trk']:>7} | {r['old_boxes']:>7} {r['old_interp']:>7} "
              f"{r['old_jumps']:>7} {r['old_span']:>8} | {r['new_boxes']:>7} "
              f"{r['new_interp']:>7} {r['new_jumps']:>7} {r['new_span']:>8} "
              f"{r['cover']:>7} {r['fps']:>5} {r['lost']:>5} {r['reacq']:>6} {r['sec']:>6}")

    def avg(k):
        return round(sum(r[k] for r in rows) / len(rows), 1)
    print(f"\nmean: old interpolated {avg('old_interp')} boxes/track, "
          f"implausible jumps {avg('old_jumps')}  ->  new interpolated {avg('new_interp')}, "
          f"jumps {avg('new_jumps')}")
    print(f"mean coverage {avg('cover')}% of analysed frames; "
          f"{avg('lost')} temporarily-lost gap(s); {avg('reacq')} re-acquisition(s); "
          f"{avg('sec')} s per track")
    print(f"analysis rate {rows[0]['fps']} fps vs stored ~2 fps "
          f"({round(rows[0]['fps'] / 2.0, 1)}x denser)")
    g = gpu_mb()
    print(f"GPU allocated: {base_gpu} MB before -> {g} MB after "
          f"(delta {round((g or 0) - (base_gpu or 0), 1)} MB)")

    print("\n=== PROXY (ReID-based, not ground truth) ===")
    if gen and imp:
        gp, ip = np.asarray(gen), np.asarray(imp)
        print(f"genuine proxy  (best candidate): median {np.median(gp):.3f}  "
              f"p05 {np.percentile(gp,5):.3f}  p90 {np.percentile(gp,90):.3f}")
        print(f"impostor proxy (runner-up)     : median {np.median(ip):.3f}  "
              f"p90 {np.percentile(ip,90):.3f}  p99 {np.percentile(ip,99):.3f}")
        th = config.TRACK_TARGET_CONFIRM_SIM
        print(f"at confirm={th}: {float((gp>=th).mean())*100:.1f}% of frames keep a box, "
              f"{float((ip>=th).mean())*100:.1f}% of impostor samples would also clear it")
        print("The overlap is real: OSNet does not separate these people crisply, so the")
        print("threshold is a chosen trade-off, not a clean decision boundary.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 5))
