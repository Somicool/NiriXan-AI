"""REST API routes.

Thin handlers over the search / ingestion modules. Heavy imports (torch, the
models) are done lazily *inside* handlers so importing this module (and booting
the app) stays fast and model loading only happens on first real use.
"""
from __future__ import annotations

import threading

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from . import config, database, ingest_jobs
from .models.schemas import (Camera, ExportRequest, ExportResponse, IngestRequest,
                             PlateSearchRequest, SearchResponse, TextSearchRequest,
                             TrackPathResponse, TrackResponse)

router = APIRouter()


def _decode_upload(data: bytes):
    """Bytes from an UploadFile -> BGR ndarray, or 400 if it isn't an image."""
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="could not decode uploaded image")
    return img


@router.get("/cameras", response_model=list[Camera])
def get_cameras():
    return database.list_cameras()


@router.get("/videos")
def get_videos():
    """Ingested source clips, with a /media URL the frontend video player can use."""
    vids = database.list_videos()
    for v in vids:
        v["url"] = f"/media/videos/{v['filename']}" if v.get("filename") else None
    return vids


@router.post("/search/text", response_model=SearchResponse)
def search_text_route(req: TextSearchRequest):
    from .search import text_search, translate
    translated = None
    if req.language and req.language != "en":
        translated, _method = translate.translate_query(req.query, req.language)
    return text_search.search_text(req.query, req.filters, top_k=req.top_k,
                                   include_scenes=req.include_scenes,
                                   translated_query=translated)


@router.post("/search/image", response_model=SearchResponse)
async def search_image_route(file: UploadFile = File(...),
                             top_k: int = Form(60),
                             use_reid: bool = Form(False)):
    from .search import image_search
    img = _decode_upload(await file.read())
    return image_search.search_by_image(img, top_k=top_k, use_reid=use_reid)


@router.post("/search/face", response_model=SearchResponse)
async def search_face_route(file: UploadFile = File(...), top_k: int = Form(60)):
    from .search import face_search
    img = _decode_upload(await file.read())
    return face_search.search_by_face(img, top_k=top_k)


@router.post("/search/plate", response_model=SearchResponse)
def search_plate_route(req: PlateSearchRequest):
    from .search import plate_search
    return plate_search.search_by_plate(req.plate, req.filters)


@router.get("/track/{detection_id}", response_model=TrackResponse)
def track_route(detection_id: int):
    from .search import cross_camera
    return cross_camera.track_across_cameras(detection_id)


@router.get("/track/{detection_id}/path", response_model=TrackPathResponse)
def track_path_route(detection_id: int):
    """Per-frame ByteTrack trajectory of a detection's track WITHIN its own clip,
    for the interactive single-camera tracking viewer. Reads indexed metadata
    only (stored boxes / timestamps / track ids) - no AI model is re-run."""
    from .search import track_path
    return track_path.get_track_path(detection_id)


@router.get("/track/{detection_id}/target")
def track_target_route(detection_id: int, force: bool = False):
    """Track Person: re-analyse the ORIGINAL video for this one selected identity.

    Unlike /track/{id}/path - which replays the boxes stored at ingest time at
    ~2 FPS - this re-detects people in the source recording at a much denser rate
    and only emits a box where the candidate is verified as the selected person.
    Runs on demand only, and the result is cached per (video, reference detection,
    logic version). The existing /path route is unchanged.
    """
    from . import track_target
    out = track_target.retrack(detection_id, force=force)
    if out.get("error") and not out.get("points"):
        raise HTTPException(status_code=422, detail=out)
    return out


@router.get("/audit")
def audit_route(limit: int = 50):
    with database.get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM audit_log ORDER BY log_id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


@router.post("/export", response_model=ExportResponse)
def export_route(req: ExportRequest):
    from .forensics import create_export
    if not req.detection_ids:
        raise HTTPException(status_code=400, detail="no detections selected for export")
    return create_export(req)


@router.get("/exports")
def exports_route():
    rows = database.list_exports()
    for r in rows:
        r["download_url"] = f"/media/exports/{r['export_id']}.zip"
    return rows


@router.post("/ingest")
def ingest_route(req: IngestRequest):
    """Kick off ingestion of a video already present in the server's video dir.
    Runs in a background thread; progress streams over /ws/ingest/{job_id}."""
    from .ingestion import pipeline
    video_path = config.VIDEO_DIR / req.video
    if not video_path.exists():
        raise HTTPException(status_code=404,
                            detail=f"video not found in videos dir: {req.video}")

    job_id = ingest_jobs.new_job(req.video)

    def run():
        try:
            def cb(stage, pct, message=""):
                ingest_jobs.update(job_id, status="processing", stage=stage,
                                   pct=pct, message=message)
            stats = pipeline.ingest_video(video_path, camera_id=req.camera_id,
                                          start_time=req.start_time, fps=req.fps,
                                          mode=req.mode, progress_cb=cb)
            ingest_jobs.update(job_id, status="done", pct=100, message="done", stats=stats)
        except Exception as exc:  # noqa: BLE001 - report any failure to the client
            ingest_jobs.update(job_id, status="error", message=str(exc))

    threading.Thread(target=run, daemon=True).start()
    return {"job_id": job_id, "video": req.video,
            "camera_id": req.camera_id, "status": "processing"}
