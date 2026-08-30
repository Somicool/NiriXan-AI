"""Face Gallery endpoints (save best face + find the same individual).

Additive router. Reuses the existing InsightFace `face` FAISS index; does not
change OCR / tracking / ReID / search / export.
"""
from __future__ import annotations

from fastapi import APIRouter, Body, HTTPException

from . import face_enhance, faces_gallery

router = APIRouter()


@router.get("/face/for-detection/{detection_id}")
def face_for_detection(detection_id: int, deep: bool = True):
    """Best available face for the person a search result belongs to.

    deep=True scans the whole ByteTrack track from the ORIGINAL frames (expanded
    boxes), recovering faces that the tight stored crops miss.

    deep=False is the search-result thumbnail path: it is called once per result,
    so it must stay off the source video entirely (no frame decoding), otherwise
    60 thumbnails compete with the video player for the same files."""
    best = faces_gallery.best_face_for_detection(detection_id, deep=deep)
    if best:
        return best
    return {"available": False,
            "reason": "No usable face found in this track.",
            "person_crop_url": faces_gallery.expanded_crop_url(detection_id, generate=deep)}


@router.get("/face/expanded-crop/{detection_id}")
def expanded_crop(detection_id: int):
    """Context-padded person crop from the ORIGINAL frame (display). No AI."""
    return {"detection_id": detection_id,
            "person_crop_url": faces_gallery.expanded_crop_url(detection_id)}


@router.post("/face/prepare/{detection_id}")
def prepare_face(detection_id: int):
    """Start choosing this person's best face now, in the background.

    Picking the best face means decoding 60 full-resolution frames and running
    face detection on each (~19 s). Called when a person result is opened, so the
    work is already done when "Save Face" is pressed and saving is immediate. The
    face chosen is exactly the same either way."""
    return faces_gallery.prepare_best_face(detection_id)


@router.post("/faces/save")
def save_face(payload: dict = Body(...)):
    """Permanently save the best face of a person from a search result."""
    det = payload.get("detection_id")
    if det is None:
        raise HTTPException(status_code=400, detail="detection_id required")
    rec = faces_gallery.save_face(int(det), payload.get("investigation"))
    if rec is None:
        raise HTTPException(status_code=404, detail="No usable face found in this track.")
    if rec.get("error"):                      # no face cleared the forensic quality bar
        raise HTTPException(status_code=404, detail=rec["error"])
    return rec


@router.get("/faces/saved")
def list_saved_faces():
    return faces_gallery.list_saved()


@router.get("/faces/saved/{saved_id}")
def get_saved_face(saved_id: int):
    rec = faces_gallery.get_saved(saved_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="not found")
    return rec


@router.delete("/faces/saved/{saved_id}")
def delete_saved_face(saved_id: int):
    return faces_gallery.delete_saved(saved_id)


@router.get("/faces/saved/{saved_id}/similar")
def similar_faces(saved_id: int, top_k: int = 60):
    """Find the same individual across all indexed footage (stored embedding)."""
    return faces_gallery.find_similar(saved_id, top_k=top_k)


# --------------------------------------------------------- Face Enhancement
# On-demand only. Nothing below is ever called during ingestion, and none of it
# touches the saved_faces row - the result is separate derived evidence.
@router.get("/faces/saved/{saved_id}/enhanced")
def get_enhanced_face(saved_id: int):
    """The stored derived visualisation for this saved face, if one exists."""
    rec = face_enhance.get_enhanced(saved_id)
    return rec or {"available": False}


@router.post("/faces/saved/{saved_id}/enhance")
def enhance_face(saved_id: int, payload: dict = Body(default={})):
    """Build (or return the cached) AI-enhanced derived visualisation.

    Goes back to the ORIGINAL recording, collects every face of that tracked
    person, verifies each against the saved identity, keeps the best views and
    fuses them. Refuses with 422 when the footage does not contain enough usable
    facial evidence, rather than inventing a face.

    `force: true` re-runs the pipeline instead of serving the cached result."""
    rec = face_enhance.enhance(int(saved_id), force=bool((payload or {}).get("force")))
    if rec.get("error"):
        status = 404 if "not found" in rec["error"].lower() else 422
        raise HTTPException(status_code=status, detail=rec)
    return rec


@router.delete("/faces/enhanced/{enh_id}")
def delete_enhanced_face(enh_id: int):
    """Discard a derived visualisation. The original saved face is untouched."""
    return face_enhance.delete_enhanced(enh_id)
