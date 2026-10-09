"""Metadata filtering for search results.

Applied AFTER the vector search narrows candidates: keeps only detections that
match the requested camera / time window / object type / colour / vehicle type
/ minimum confidence.
"""
from __future__ import annotations

from .. import config, database
from ..models.schemas import SearchFilters

_VEHICLE_LABELS = {config.DETECT_CLASSES[c] for c in config.VEHICLE_CLASSES}
_PERSON_LABELS = {config.DETECT_CLASSES[c] for c in config.PERSON_CLASSES}


def _object_type_labels(object_type):
    if object_type == "person":
        return _PERSON_LABELS
    if object_type == "vehicle":
        return _VEHICLE_LABELS
    return None


def _colour_ok(det: dict, wanted: set, colorless_ids: set) -> bool:
    """Explicit colour-filter test for one detection.

    Normal (colour) footage: unchanged - any wanted colour must be present.

    Night-vision / B&W footage (video flagged colorless): only black and white
    are measurable there, so a hue such as "blue" is UNRELIABLE, not absent.
    Rejecting on it would silently zero out every result from that camera, so a
    hue-only request skips the colour constraint for that video. Black / white
    selections still filter normally."""
    attrs = det.get("attributes") or {}
    present = {attrs.get("color"), attrs.get("upper_color"), attrs.get("lower_color")}
    present = {c.lower() for c in present if c}
    if det.get("video_id") in colorless_ids:
        bw = wanted & set(config.BW_COLORS)
        if not bw:
            return True                   # hue on B&W footage: unreliable, don't filter
        return bool(bw & present)
    return bool(wanted & present)


def match(det: dict, f: SearchFilters, colorless_ids: set | None = None) -> bool:
    if f.cameras and det.get("camera_id") not in f.cameras:
        return False

    if f.video_id is not None and det.get("video_id") != f.video_id:
        return False

    ts = det.get("timestamp")
    if f.start_time and ts and ts < f.start_time:
        return False
    if f.end_time and ts and ts > f.end_time:
        return False

    if f.min_confidence and (det.get("confidence") or 0.0) < f.min_confidence:
        return False

    labels = _object_type_labels(f.object_type)
    if labels is not None and det.get("class_label") not in labels:
        return False

    attrs = det.get("attributes") or {}

    if f.colors:
        wanted = {c.lower() for c in f.colors}
        if not _colour_ok(det, wanted, colorless_ids or set()):
            return False

    if f.vehicle_type and attrs.get("vehicle_type") != f.vehicle_type:
        return False

    return True


def apply_filters(detections: list[dict], f: SearchFilters | None) -> list[dict]:
    if f is None:
        return detections
    # one small query, and only when a colour filter is actually in use
    colorless_ids = database.colorless_video_ids() if f.colors else set()
    return [d for d in detections if match(d, f, colorless_ids)]
