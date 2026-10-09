"""Ingestion pipeline orchestrator.

One call ingests a whole video end-to-end:
  detect+track -> CLIP-embed crops -> attributes -> OSNet re-ID (persons)
  -> whole-frame 'scene' embeddings -> store in SQLite + FAISS.

Also handles dynamic camera registration (unknown camera on upload) and prints
a timing/benchmark breakdown so we know how long ingest takes per clip.

    python -m app.ingestion.pipeline <video> --camera CAM-02
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np

from .. import config, database, ingest_progress, registry
from ..search import vector_store
from . import tracker, embedder, attribute_extractor, reid_embedder, recording_meta, transcode

_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".wmv", ".flv"}


def _infer_camera_id(video_path) -> str:
    return recording_meta.parse_camera_id(video_path)


def _resolve_preset(mode):
    """Return the processing-mode preset dict (Fast is the default)."""
    mode = (mode or config.PROCESSING_MODE or "fast").lower()
    if mode not in config.MODE_PRESETS:
        mode = "fast"
    return mode, dict(config.MODE_PRESETS[mode])


def _adaptive_fps(video_path, preset) -> float:
    """Fast-mode adaptive sampling: probe a dozen frames for motion and pick the
    lower FPS for quiet scenes, the higher FPS for busy ones (1-2 FPS)."""
    lo, hi = preset.get("fps_min", 1), preset.get("fps_max", 2)
    try:
        cap = cv2.VideoCapture(str(video_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total <= 1:
            cap.release()
            return lo
        idxs = np.linspace(0, total - 1, num=min(12, total), dtype=int)
        prev, diffs = None, []
        for fi in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
            ok, frame = cap.read()
            if not ok:
                continue
            g = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (160, 90))
            if prev is not None:
                diffs.append(float(np.mean(cv2.absdiff(g, prev))))
            prev = g
        cap.release()
        activity = float(np.mean(diffs)) if diffs else 0.0
        return hi if activity >= config.FAST_ACTIVITY_THRESHOLD else lo
    except Exception:
        return lo


def _blur_var(crop_path) -> float:
    """Variance of Laplacian (sharpness) of a crop; higher = sharper."""
    img = cv2.imread(str(crop_path)) if crop_path else None
    if img is None or not img.size:
        return 0.0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def ingest_video(video_path, camera_id=None, start_time=None, fps=None,
                 camera_meta=None, save_indexes: bool = True, progress_cb=None,
                 do_faces=None, do_plates=None, mode=None) -> dict:
    """Ingest one video into SQLite + FAISS. Returns a stats dict.

    progress_cb, if given, is called as progress_cb(stage, pct, message) after
    each stage - used by the API to stream ingest progress over a WebSocket.

    mode selects a processing preset ("fast" default, or "accurate"); every knob
    that differs between the two lives in config.MODE_PRESETS so this one function
    serves both with no duplicated pipeline. do_faces / do_plates / fps still
    override the preset for a single call when given.
    """
    mode, preset = _resolve_preset(mode)
    faces_on = preset["do_faces"] if do_faces is None else bool(do_faces)
    plates_on = preset["do_plates"] if do_plates is None else bool(do_plates)
    imgsz = preset["imgsz"]
    clip_batch = preset["clip_batch"]
    region_split = preset["region_split"]
    voting = preset["voting"]
    index_chunk = preset["index_chunk"] if preset["incremental"] else 0
    def _emit(stage, pct, message=""):
        ingest_progress.set_stage(pct, stage, message)   # advance the per-video bar past tracking
        if progress_cb:
            try:
                progress_cb(stage, pct, message)
            except Exception:
                pass

    video_path = Path(video_path)
    video_path = transcode.ensure_mp4(video_path)   # AVI/MOV/MKV -> H.264 MP4 for browser playback
    camera_id = camera_id or _infer_camera_id(video_path)
    meta = camera_meta or {}
    # Dynamic camera registration: INSERT OR IGNORE keeps known cameras' config
    # but registers an unknown camera (e.g. the judge's uploaded footage).
    database.register_camera(camera_id, name=meta.get("name"), location=meta.get("location"),
                             lat=meta.get("lat"), lon=meta.get("lon"))

    # Probe the source recording: native fps + frame size let results seek to
    # the exact moment and overlay the box; duration/end_time index the clip on
    # a timeline (how real VMS/NVR software stores footage).
    cap = cv2.VideoCapture(str(video_path))
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    v_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    v_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    if native_fps <= 0:
        native_fps = 30.0

    # Real-world recording time. Without this every detection timestamp fell back
    # to "now" (the tracker's default), so evidence and saved faces were stamped
    # with the INGESTION time instead of when the footage was actually recorded -
    # and clips ingested on different days sat hours apart on the journey
    # timeline. recording_meta prefers a timestamp in the filename, then the
    # camera's own creation time inside the container, then the file mtime.
    if start_time is None:
        start_time = recording_meta.parse_start_time(video_path).isoformat()
        print(f"[ingest] {video_path.name}: recording start {start_time} "
              f"(source: {recording_meta.start_time_source(video_path)})")

    duration = round(total_frames / native_fps, 3) if total_frames else None
    end_time = None
    if start_time and duration is not None:
        try:
            end_time = (datetime.fromisoformat(str(start_time)) + timedelta(seconds=duration)).isoformat()
        except ValueError:
            end_time = None

    # Resolve the sampling FPS: explicit arg wins; else the preset value; else
    # (Fast mode) an adaptive 1-2 FPS chosen from a quick motion probe.
    if fps is not None:
        target_fps = fps
    elif preset.get("fps") is not None:
        target_fps = preset["fps"]
    elif preset.get("adaptive_fps"):
        target_fps = _adaptive_fps(video_path, preset)
    else:
        target_fps = config.DEFAULT_FPS

    t0 = time.time()
    video_id = database.add_video(
        camera_id, video_path.name, fps=target_fps,
        start_time=str(start_time) if start_time else None, status="processing",
        duration=duration, end_time=end_time, native_fps=native_fps, width=v_w, height=v_h,
    )
    ingest_progress.set_meta(indexed=0, searchable=False, video_id=video_id)
    _emit("start", 2, f"registered video (mode={mode}, fps={target_fps}, imgsz={imgsz})")

    # Night-vision / B&W check (a dozen downscaled frames). Only changes how
    # colour attributes are read; detection, tracking, embeddings are unaffected.
    try:
        colorless, med_sat = attribute_extractor.probe_colorless(video_path)
    except Exception:
        colorless, med_sat = False, -1.0
    database.set_video_colorless(video_id, colorless)
    if colorless:
        print(f"[ingest] {video_path.name}: low-colour footage (median saturation "
              f"{med_sat}) - clothing colour limited to black/white")

    # ------------------------------------------------------------------
    # PROGRESSIVE / CHUNKED INGEST
    # ------------------------------------------------------------------
    # ByteTrack runs as ONE continuous stream over the whole clip (identical
    # track_ids -> identical final index), but detections are yielded in chunks
    # of ~PROGRESSIVE_CHUNK_FRAMES sampled frames. Each chunk is embedded, stored
    # in SQLite and pushed into FAISS *immediately*, so early portions of a long
    # video become searchable within seconds instead of waiting for the whole
    # clip. The expensive secondary analysis (faces / plates) is deferred to the
    # end - it doesn't affect describe/image/text search and is the slowest work.
    #
    # Accurate mode sets incremental=False -> chunk_frames=None -> the tracker
    # yields the whole clip as one final chunk, i.e. byte-identical to the old
    # monolithic path but through the same code.
    chunk_frames = config.PROGRESSIVE_CHUNK_FRAMES if preset.get("incremental") else None

    # size/quality gates for the deferred face + plate stages (applied while we
    # collect candidates per chunk so we don't have to keep every detection around)
    faces_g = (preset["face_min_w"], preset["face_min_h"], preset["face_min_det"]) if faces_on else (0, 0, 0)
    plates_g = (preset["plate_min_w"], preset["plate_min_h"], preset["plate_blur_min"]) if plates_on else (0, 0, 0.0)
    fmw, fmh, fdet = faces_g
    pmw, pmh, pblur = plates_g

    n_dets = 0
    scene_count = 0
    reid_count = 0
    t_embed = t_reid = t_store = 0.0
    track_meta: dict = {}                 # tid -> {label, fmin, fmax, tmin, tmax}
    face_cands: dict = {}                 # (cam, tid) -> [(area, det_id, crop_path, ts, cam), ...]
    plate_cands: dict = {}                # (cam, tid) -> [(area, det_id, crop_path, ts, cam), ...]

    loop_t0 = time.time()
    for chunk in tracker.iter_track_chunks(video_path, camera_id, start_time=start_time,
                                           fps=target_fps, imgsz=imgsz, chunk_frames=chunk_frames):
        cdets = chunk["dets"]
        cframes = chunk["frames"]

        clip_ids: list = []
        clip_vecs: list = []
        reid_ids: list = []
        reid_vecs: list = []

        if cdets:
            # CLIP-embed this chunk's crops (batched, in-memory - no JPEG re-decode)
            t = time.time()
            clip_embs = embedder.embed_crops([d.crop_img for d in cdets], batch_size=clip_batch)
            attrs_list = attribute_extractor.extract_batch(
                [d.crop_img for d in cdets], [d.class_id for d in cdets], clip_embs,
                region_split=region_split, colorless=colorless)
            t_embed += time.time() - t

            # OSNet re-ID for the person crops in this chunk.
            # The tracker already embedded these people to verify their track
            # associations (identity_guard), so REUSE those vectors - re-embedding
            # here would double the OSNet work for an identical result. Only crops
            # the tracker did not embed fall back to a fresh pass.
            t = time.time()
            person_idx = [i for i, d in enumerate(cdets) if d.class_id in config.PERSON_CLASSES]
            reid_by_det = {i: cdets[i].reid_vec for i in person_idx
                           if getattr(cdets[i], "reid_vec", None) is not None}
            missing = [i for i in person_idx if i not in reid_by_det]
            if missing:
                fresh = reid_embedder.embed_persons([cdets[i].crop_path for i in missing])
                reid_by_det.update({missing[k]: fresh[k] for k in range(len(missing))})
            t_reid += time.time() - t

            # store detections + attributes for this chunk in one bulk transaction
            t = time.time()
            obj_rows = [{
                "video_id": video_id, "camera_id": d.camera_id, "track_id": d.track_id,
                "frame_number": d.frame_number, "timestamp": d.timestamp,
                "class_label": d.class_label, "confidence": d.confidence,
                "bbox_x": d.bbox[0], "bbox_y": d.bbox[1], "bbox_w": d.bbox[2], "bbox_h": d.bbox[3],
                "crop_path": d.crop_path, "attributes": attrs_list[i],
            } for i, d in enumerate(cdets)]
            det_ids = database.insert_detections_bulk(obj_rows)   # aligned with cdets
            t_store += time.time() - t

            clip_ids = list(det_ids)
            clip_vecs = [clip_embs[i] for i in range(len(cdets))]
            reid_ids = [det_ids[i] for i in reid_by_det]
            reid_vecs = [reid_by_det[i] for i in reid_by_det]
            reid_count += len(reid_by_det)
            n_dets += len(cdets)

            # accumulate per-track summaries + deferred face/plate candidates
            for i, d in enumerate(cdets):
                tid = d.track_id
                m = track_meta.get(tid)
                if m is None:
                    track_meta[tid] = {"label": d.class_label, "fmin": d.frame_number,
                                       "fmax": d.frame_number, "tmin": d.timestamp, "tmax": d.timestamp}
                else:
                    if d.frame_number < m["fmin"]:
                        m["fmin"] = d.frame_number
                    if d.frame_number > m["fmax"]:
                        m["fmax"] = d.frame_number
                    if d.timestamp < m["tmin"]:
                        m["tmin"] = d.timestamp
                    if d.timestamp > m["tmax"]:
                        m["tmax"] = d.timestamp
                if faces_on and d.class_id in config.PERSON_CLASSES and d.bbox[2] >= fmw and d.bbox[3] >= fmh:
                    face_cands.setdefault((d.camera_id, tid), []).append(
                        (d.bbox[2] * d.bbox[3], det_ids[i], d.crop_path, d.timestamp, d.camera_id))
                if plates_on and d.class_id in config.VEHICLE_CLASSES and d.bbox[2] >= pmw and d.bbox[3] >= pmh:
                    plate_cands.setdefault((d.camera_id, tid), []).append({
                        "area": d.bbox[2] * d.bbox[3], "det_id": det_ids[i],
                        "crop_path": d.crop_path, "ts": d.timestamp, "cam": d.camera_id,
                        "frame_number": d.frame_number, "bbox": tuple(d.bbox),
                        "confidence": d.confidence, "cls": d.class_id})

        # whole-frame 'scene' embeddings for this chunk (scene-level search)
        if cframes:
            t = time.time()
            frame_embs = embedder.embed_images([f["frame_path"] for f in cframes], batch_size=clip_batch)
            scene_rows = [{
                "video_id": video_id, "camera_id": camera_id, "track_id": None,
                "frame_number": f["frame_number"], "timestamp": f["timestamp"],
                "class_label": "scene", "confidence": 1.0,
                "bbox_x": None, "bbox_y": None, "bbox_w": None, "bbox_h": None,
                "crop_path": f["frame_path"], "attributes": {"kind": "scene"},
            } for f in cframes]
            scene_ids = database.insert_detections_bulk(scene_rows)
            t_store += time.time() - t
            clip_ids += scene_ids
            clip_vecs += [frame_embs[k] for k in range(len(cframes))]
            scene_count += len(scene_ids)

        # push this chunk's vectors into FAISS immediately -> searchable now
        t = time.time()
        if clip_vecs:
            vector_store.add("clip", np.stack(clip_vecs), clip_ids)
        if reid_vecs:
            vector_store.add("reid", np.stack(reid_vecs), reid_ids)
        t_store += time.time() - t

        # progressive UI: "Indexed N detections / Search Ready (Partial)"
        sampled = chunk.get("sampled", 0)
        total = chunk.get("total_sampled", 0) or 1
        pct = int(5 + 88 * (sampled / max(total, 1)))
        pct = max(5, min(pct, 93))
        ingest_progress.set_meta(indexed=n_dets, searchable=(n_dets > 0), video_id=video_id)
        _emit("indexing", pct, f"indexed {n_dets} detections")

    # detect+track time = wall time of the loop minus the embed/reid/store we
    # measured inside it (tracking is interleaved with those in the stream)
    t_track = max(0.0, (time.time() - loop_t0) - t_embed - t_reid - t_store)

    # per-track summaries (from accumulated meta) ------------------------
    for tid, m in track_meta.items():
        database.upsert_track(f"{video_id}:{tid}", video_id, camera_id, tid, m["label"],
                              m["fmin"], m["fmax"], m["tmin"], m["tmax"])

    # ------------------------------------------------------------------
    # DEFERRED secondary analysis (faces + plates) - runs after the clip is
    # already searchable, so it never blocks first results.
    # ------------------------------------------------------------------
    # faces (bonus, ethics-gated). Accurate mode votes gender/age over several
    # frames per person-track; Fast mode reads the single largest crop.
    t = time.time()
    face_count = 0
    if faces_on and face_cands:
        from . import face_recognizer
        face_ids, face_vecs = [], []
        for grp in face_cands.values():
            grp.sort(key=lambda g: g[0], reverse=True)     # largest (closest) crops first
            top = grp[:config.FACE_VOTE_FRAMES]
            paths = [g[2] for g in top]
            if voting:
                voted = face_recognizer.detect_faces_voted(paths)
            else:                                          # Fast: single best frame + quality gate
                faces = face_recognizer.detect_faces(paths[0])
                voted = faces[0] if faces and faces[0]["det_score"] >= fdet else None
            if not voted:
                continue
            _area, rep_det, rep_crop, rep_ts, rep_cam = top[0]   # representative (largest)
            face_id = database.insert_face({
                "detection_id": rep_det, "camera_id": rep_cam,
                "timestamp": rep_ts, "age": voted["age"],
                "gender": voted["gender"], "crop_path": rep_crop,
            })
            face_ids.append(face_id)
            face_vecs.append(voted["embedding"])
            face_count += 1
        if face_vecs:
            vector_store.add("face", np.stack(face_vecs), face_ids)
    t_face = time.time() - t
    _emit("faces", 96, f"{face_count} faces")

    # licence plates (bonus). Accurate mode votes OCR across frames; Fast mode
    # reads only the largest crop of vehicles with a big, sharp plate region.
    t = time.time()
    plate_count = 0
    budget = skipped_tracks = 0
    if plates_on and plate_cands:
        from . import plate_reader, anpr
        # Spend the ANPR budget on the CLOSEST vehicles first and cap how many
        # tracks are attempted. OCR is CPU-bound here, so an unbounded loop over
        # every vehicle track in a busy clip runs for hours while the distant ones
        # contribute nothing readable.
        groups = sorted(plate_cands.values(),
                        key=lambda g: max((x["area"] for x in g), default=0),
                        reverse=True)
        budget = max(1, int(getattr(config, "ANPR_MAX_TRACKS_PER_VIDEO", 60)))
        skipped_tracks = max(0, len(groups) - budget)
        for grp in groups[:budget]:
            grp.sort(key=lambda g: g["area"], reverse=True)   # largest (closest) crops first
            top = grp[:config.PLATE_VOTE_FRAMES]
            rep = top[0]
            rep_det, rep_crop, rep_ts, rep_cam = rep["det_id"], rep["crop_path"], rep["ts"], rep["cam"]
            is_tw = rep["cls"] in config.ANPR_TWOWHEELER_CLASSES
            if not is_tw and pblur > 0 and _blur_var(rep_crop) < pblur:  # Fast: skip blurry car plates
                continue
            paths = [g["crop_path"] for g in top]
            tag = f"{video_id}_{rep_det}"
            # ANPR. Two-wheelers / autos: ADAPTIVE high-FPS re-sampling of the source
            # video around the track (recovers small/blurred plates); other vehicles:
            # crop-based multi-frame voting. Falls back to crop-based if re-sampling
            # yields nothing. Old OCR path used only when ANPR is disabled.
            if config.ANPR_ENABLED and config.ANPR_ADAPTIVE_ENABLED and is_tw:
                cands = anpr.read_plate_track_adaptive(
                    str(video_path), grp, native_fps,
                    save_dir=str(config.PLATE_CROP_DIR), tag=tag)
                if not cands:
                    cands = anpr.read_plate_track(paths, save_dir=str(config.PLATE_CROP_DIR), tag=tag)
            elif config.ANPR_ENABLED:
                cands = anpr.read_plate_track(paths, save_dir=str(config.PLATE_CROP_DIR), tag=tag)
            else:
                cands = (plate_reader.read_plates_voted(paths)
                         if voting else plate_reader.read_plates(rep_crop))
            # Store up to PLATE_MAX_CANDIDATES reads per vehicle track (not just the
            # top one) so a confident misread can't hide the correct plate.
            stored = 0
            for p in cands:
                # trust a plate agreed across frames, or a confident single read
                if p.get("votes", 1) >= config.PLATE_MIN_VOTES or p["conf"] >= config.PLATE_SINGLE_CONF:
                    database.insert_plate({
                        "detection_id": rep_det, "camera_id": rep_cam,
                        "timestamp": rep_ts, "plate_text": p["text"],
                        "confidence": p["conf"], "crop_path": rep_crop,
                        "votes": p.get("votes"), "source": p.get("source"),
                        "plate_crop": p.get("plate_crop"),
                    })
                    # Demo Vehicle Registry: one permanent synthetic record per
                    # unique recognised plate (never regenerated). No effect on OCR.
                    registry.ensure(p["text"], hint=config.DETECT_CLASSES.get(rep["cls"]))
                    plate_count += 1
                    stored += 1
                    if stored >= config.PLATE_MAX_CANDIDATES:
                        break
    t_plate = time.time() - t
    _emit("plates", 99, f"{plate_count} plates")
    if plates_on and skipped_tracks:
        print(f"[plates] ANPR budget reached: read the {budget} closest vehicle tracks, "
              f"skipped {skipped_tracks} more distant ones "
              f"(raise config.ANPR_MAX_TRACKS_PER_VIDEO to cover them)")

    if save_indexes:
        vector_store.save()
    database.set_video_status(video_id, "done")

    stats = {
        "video_id": video_id, "camera": camera_id, "video": video_path.name,
        "mode": mode, "fps": target_fps, "imgsz": imgsz,
        # Make the two optional stages explicit. Fast mode leaves both OFF, and a
        # silent skip here reads downstream as "plate search is broken".
        "plates_enabled": plates_on, "faces_enabled": faces_on,
        "notes": ([] if plates_on else
                  ["number-plate reading was OFF for this run: no plates were "
                   "indexed, so plate search will not match this clip. Re-run with "
                   "plates=true (or mode=accurate) to read them."])
                 + ([] if faces_on else
                    ["face recognition was OFF for this run: no faces were indexed."]),
        "object_detections": n_dets, "scene_frames": scene_count,
        "tracks": len(track_meta), "person_reid": reid_count, "faces": face_count,
        "plates": plate_count,
        "total_seconds": round(time.time() - t0, 1),
        "timing_s": {"track": round(t_track, 1), "clip": round(t_embed, 1),
                     "reid": round(t_reid, 1), "store": round(t_store, 1),
                     "face": round(t_face, 1), "plate": round(t_plate, 1)},
        "faiss": vector_store.stats(),
    }
    print(f"[ingest] {stats}")
    _emit("done", 100, "done")
    return stats


def ingest_directory(video_dir=None, fps=None, start_time=None, mode=None) -> list[dict]:
    video_dir = Path(video_dir or config.VIDEO_DIR)
    vids = (sorted(p for p in video_dir.iterdir() if p.suffix.lower() in _VIDEO_EXTS)
            if video_dir.exists() else [])
    return [ingest_video(v, fps=fps, start_time=start_time, mode=mode) for v in vids]


def reset_all() -> None:
    """Clear ingested data (detections/tracks/videos/faces/plates) + FAISS."""
    with database.get_conn() as conn:
        for tbl in ("detections", "tracks", "videos", "faces", "plates"):
            conn.execute(f"DELETE FROM {tbl}")
    vector_store.reset()
    print("reset: cleared detections/tracks/videos/faces/plates + FAISS indexes")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Ingest one video end-to-end")
    ap.add_argument("video")
    ap.add_argument("--camera", default=None)
    ap.add_argument("--fps", type=float, default=None)
    ap.add_argument("--start-time", default=None)
    ap.add_argument("--mode", default=None, choices=[None, "fast", "accurate"])
    ap.add_argument("--reset", action="store_true")
    args = ap.parse_args()
    database.init_db()
    if args.reset:
        reset_all()
    ingest_video(args.video, camera_id=args.camera, fps=args.fps,
                 start_time=args.start_time, mode=args.mode)
