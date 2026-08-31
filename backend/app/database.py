"""SQLite metadata store.

Holds the *facts* about everything detected (timestamps, cameras, colours,
plate text, ...). Vector fingerprints live in FAISS; each row here keeps the
FAISS row-id so the two can be joined at search time.
"""
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS cameras (
    camera_id   TEXT PRIMARY KEY,
    name        TEXT,
    location    TEXT,
    lat         REAL,
    lon         REAL,
    created_at  TEXT
);

CREATE TABLE IF NOT EXISTS videos (
    video_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    camera_id   TEXT,
    filename    TEXT,
    fps         REAL,          -- sampled (analysis) fps
    start_time  TEXT,          -- real-world time of first frame (ISO)
    duration    REAL,          -- seconds
    end_time    TEXT,          -- real-world time of last frame (ISO)
    native_fps  REAL,          -- source video fps (for exact seek offset)
    width       INTEGER,       -- native frame width  (for bbox overlay)
    height      INTEGER,       -- native frame height
    status      TEXT DEFAULT 'pending',
    created_at  TEXT
);

CREATE TABLE IF NOT EXISTS detections (
    detection_id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id     INTEGER,
    camera_id    TEXT,
    track_id     INTEGER,
    frame_number INTEGER,
    timestamp    TEXT,         -- real-world time of this detection (ISO)
    class_label  TEXT,
    confidence   REAL,
    bbox_x       REAL,
    bbox_y       REAL,
    bbox_w       REAL,
    bbox_h       REAL,
    crop_path    TEXT,
    clip_vec_id  INTEGER,      -- row id in the CLIP FAISS index
    reid_vec_id  INTEGER,      -- row id in the re-ID FAISS index (persons)
    attributes   TEXT          -- JSON: colour, type, clothing, accessories...
);

CREATE TABLE IF NOT EXISTS tracks (
    track_key    TEXT PRIMARY KEY,   -- "{video_id}:{track_id}"
    video_id     INTEGER,
    camera_id    TEXT,
    track_id     INTEGER,
    class_label  TEXT,
    start_frame  INTEGER,
    end_frame    INTEGER,
    start_time   TEXT,
    end_time     TEXT,
    direction    TEXT
);

CREATE TABLE IF NOT EXISTS faces (
    face_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    detection_id INTEGER,
    camera_id    TEXT,
    timestamp    TEXT,
    face_vec_id  INTEGER,      -- row id in the face FAISS index
    age          INTEGER,
    gender       TEXT,
    crop_path    TEXT
);

CREATE TABLE IF NOT EXISTS plates (
    plate_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    detection_id INTEGER,
    camera_id    TEXT,
    timestamp    TEXT,
    plate_text   TEXT,
    confidence   REAL,
    crop_path    TEXT
);

CREATE TABLE IF NOT EXISTS audit_log (
    log_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp    TEXT,
    action       TEXT,
    query_text   TEXT,
    query_type   TEXT,
    result_count INTEGER,
    user         TEXT,
    details      TEXT
);

CREATE TABLE IF NOT EXISTS exports (
    export_id     TEXT PRIMARY KEY,
    case_number   TEXT,
    officer       TEXT,
    created_at    TEXT,
    manifest_hash TEXT,
    file_path     TEXT,
    detection_ids TEXT
);

-- Demo Vehicle Registry (synthetic, permanent). One JSON record per unique
-- plate; never regenerated, only edited. See app/registry.py.
CREATE TABLE IF NOT EXISTS vehicle_registry (
    plate         TEXT PRIMARY KEY,
    data          TEXT,
    created_at    TEXT,
    updated_at    TEXT
);

-- Face Gallery: permanently saved best-faces of persons. Reuses the existing
-- 'face' FAISS index for "find similar". See app/faces_gallery.py.
CREATE TABLE IF NOT EXISTS saved_faces (
    saved_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    face_id       INTEGER,
    detection_id  INTEGER,
    investigation TEXT,
    camera_id     TEXT,
    timestamp     TEXT,
    confidence    REAL,
    face_crop     TEXT,
    person_crop   TEXT,
    embedding     TEXT,       -- base64 float32 face embedding (permanent copy)
    gender        TEXT,
    age           INTEGER,
    created_at    TEXT
);

-- Cached BEST representative face per ByteTrack person track (so search results
-- can show it cheaply and repeat scans are free). Rebuilt on demand.
CREATE TABLE IF NOT EXISTS track_best_face (
    video_id      INTEGER,
    track_id      INTEGER,
    detection_id  INTEGER,
    quality       REAL,
    low_quality   INTEGER,
    face_crop     TEXT,
    preview_crop  TEXT,
    person_crop   TEXT,
    metrics       TEXT,
    frames_seen   INTEGER,
    faces_seen    INTEGER,
    created_at    TEXT,
    PRIMARY KEY (video_id, track_id)
);

-- Track Identity Descriptors: one permanent multi-view identity record per
-- completed ByteTrack person track (see app/track_identity.py).
CREATE TABLE IF NOT EXISTS track_identity (
    video_id      INTEGER,
    track_id      INTEGER,
    camera_id     TEXT,
    n_detections  INTEGER,
    n_views       INTEGER,
    first_seen    TEXT,
    last_seen     TEXT,
    meta          TEXT,
    reid_vecs     TEXT,
    clip_vecs     TEXT,
    face_vecs     TEXT,
    created_at    TEXT,
    PRIMARY KEY (video_id, track_id)
);

-- OSRM road-route cache, keyed on the exact camera coordinate sequence. Routing
-- is the only network call in the journey path, so caching it makes a repeated
-- investigation instant. Failures are never cached (see app/routing.py).
CREATE TABLE IF NOT EXISTS route_cache (
    cache_key   TEXT PRIMARY KEY,
    provider    TEXT,
    profile     TEXT,
    n_points    INTEGER,
    payload     TEXT,
    created_at  TEXT
);

-- Reconstructed cross-camera journeys, stored permanently per investigation.
-- Full result (nodes/legs/alternatives) is kept as JSON; see app/journey.py.
CREATE TABLE IF NOT EXISTS journeys (
    journey_id             INTEGER PRIMARY KEY AUTOINCREMENT,
    investigation          TEXT,
    reference_detection_id INTEGER,
    scope                  TEXT,
    confidence             REAL,
    camera_count           INTEGER,
    distance_km            REAL,
    span_seconds           REAL,
    data                   TEXT,
    created_at             TEXT
);

-- Investigation activity history (persons + vehicles searched / found / tracked).
-- Persistent dashboard history; see app/history.py.
CREATE TABLE IF NOT EXISTS activity_history (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    kind        TEXT,      -- 'person' | 'vehicle'
    action      TEXT,      -- 'searched' | 'found' | 'tracked'
    ref         TEXT,      -- detection id / plate / query key (for de-dup)
    label       TEXT,
    camera_id   TEXT,
    timestamp   TEXT,      -- event time (detection ts or now)
    crop_url    TEXT,
    plate       TEXT,
    query       TEXT,
    created_at  TEXT
);

-- Case File persistence. The bookmarked evidence set and the case metadata used
-- to live ONLY in React state (context/investigation.jsx), so every saved item
-- was lost on a page refresh or a backend restart. Stored here so a case is as
-- permanent as every other forensic record. See app/case_store.py.
CREATE TABLE IF NOT EXISTS case_evidence (
    case_key     TEXT    NOT NULL,
    detection_id INTEGER NOT NULL,
    position     INTEGER,           -- preserves the order the officer added them
    snapshot     TEXT,              -- full result item as saved (JSON)
    added_at     TEXT,
    PRIMARY KEY (case_key, detection_id)
);

CREATE TABLE IF NOT EXISTS case_meta (
    case_key     TEXT PRIMARY KEY,
    title        TEXT,
    case_number  TEXT,
    officer      TEXT,
    notes        TEXT,
    updated_at   TEXT
);

-- Generated PDF evidence reports (see app/case_report.py). Kept so previously
-- issued reports stay listable and their SHA-256 remains on record.
CREATE TABLE IF NOT EXISTS case_reports (
    report_id     TEXT PRIMARY KEY,
    case_key      TEXT,
    export_id     TEXT,          -- set when the report covers a sealed past case
    detection_ids TEXT,          -- JSON list of the exhibits covered
    case_number   TEXT,
    case_title    TEXT,
    officer       TEXT,
    created_at    TEXT,
    exhibit_count INTEGER,
    gemini_used   INTEGER,
    gemini_model  TEXT,
    sha256        TEXT,
    file_path     TEXT
);

-- Face Enhancement: DERIVED visualisations built on demand from several verified
-- views of the same tracked person (see app/face_enhance.py). Purely additive -
-- the saved_faces row it points at is never modified, and both the original and
-- the derived image are hashed so the chain of custody stays checkable.
CREATE TABLE IF NOT EXISTS enhanced_faces (
    id                     INTEGER PRIMARY KEY AUTOINCREMENT,
    saved_face_id          INTEGER,
    source_video_id        INTEGER,
    source_track_id        INTEGER,
    source_camera_id       TEXT,
    original_detection_ids TEXT,   -- JSON list of the detections fused
    source_timestamps      TEXT,   -- JSON list, one per fused frame
    best_source_timestamp  TEXT,
    model_name             TEXT,   -- exactly which engine produced this
    frames_analysed        INTEGER,
    frames_selected        INTEGER,
    quality_score          REAL,
    source_quality         TEXT,   -- High | Medium | Low
    original_hash          TEXT,   -- SHA-256 of the untouched saved face crop
    enhanced_hash          TEXT,   -- SHA-256 of the derived image
    file_path              TEXT,
    source_frames          TEXT,   -- JSON: per-frame evidence behind the result
    metrics                TEXT,   -- JSON: before/after measurements
    label                  TEXT,
    created_at             TEXT,
    -- The clearest REAL frame found anywhere in the person's track. This is the
    -- primary result: restoration is only accepted when it measurably beats it.
    best_source_path       TEXT,
    best_source_frame      INTEGER,
    best_source_quality    REAL,
    best_source_hash       TEXT,
    -- Outcome of the quality gate. 0 means the enhancement was REJECTED and
    -- file_path holds the best natural frame instead.
    enhancement_applied    INTEGER,
    gate_status            TEXT,
    gate_reason            TEXT
);

CREATE INDEX IF NOT EXISTS idx_enh_saved ON enhanced_faces(saved_face_id);
CREATE INDEX IF NOT EXISTS idx_case_ev ON case_evidence(case_key, position);
CREATE INDEX IF NOT EXISTS idx_det_camera ON detections(camera_id);
CREATE INDEX IF NOT EXISTS idx_det_time   ON detections(timestamp);
CREATE INDEX IF NOT EXISTS idx_det_class  ON detections(class_label);
CREATE INDEX IF NOT EXISTS idx_plate_text ON plates(plate_text);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def get_conn():
    """Context-managed connection with dict-like rows."""
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _migrate(conn) -> None:
    """Add columns introduced after a DB was first created (safe, idempotent)."""
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(videos)").fetchall()}
    for col, decl in (("duration", "REAL"), ("end_time", "TEXT"),
                      ("native_fps", "REAL"), ("width", "INTEGER"), ("height", "INTEGER")):
        if col not in cols:
            conn.execute(f"ALTER TABLE videos ADD COLUMN {col} {decl}")
    # ANPR: supporting-frame count, reading engine, and the best PLATE crop path.
    pcols = {r["name"] for r in conn.execute("PRAGMA table_info(plates)").fetchall()}
    for col, decl in (("votes", "INTEGER"), ("source", "TEXT"), ("plate_crop", "TEXT")):
        if col not in pcols:
            conn.execute(f"ALTER TABLE plates ADD COLUMN {col} {decl}")
    # Face Gallery: preview image, low-quality flag, and the full metric breakdown.
    scols = {r["name"] for r in conn.execute("PRAGMA table_info(saved_faces)").fetchall()}
    for col, decl in (("preview_crop", "TEXT"), ("low_quality", "INTEGER"),
                      ("quality_metrics", "TEXT")):
        if col not in scols:
            conn.execute(f"ALTER TABLE saved_faces ADD COLUMN {col} {decl}")
    # Best-face cache: everything needed to SAVE the already-chosen face without
    # rescanning the track (embedding + the config fingerprint it was chosen
    # under). Kept out of `metrics`, which is sent to the UI.
    tcols = {r["name"] for r in conn.execute("PRAGMA table_info(track_best_face)").fetchall()}
    if "save_payload" not in tcols:
        conn.execute("ALTER TABLE track_best_face ADD COLUMN save_payload TEXT")
    # Evidence reports can now be produced per case, including cases already sealed
    # as exports - that link is stored so the UI can offer the existing PDF.
    rcols = {r["name"] for r in conn.execute("PRAGMA table_info(case_reports)").fetchall()}
    if rcols:
        for col, decl in (("export_id", "TEXT"), ("detection_ids", "TEXT")):
            if col not in rcols:
                conn.execute(f"ALTER TABLE case_reports ADD COLUMN {col} {decl}")
    # Face Enhancement: the pipeline now prefers the clearest REAL frame from the
    # person's track and only accepts restoration that passes a quality gate, so
    # the chosen natural frame and the gate's verdict are recorded alongside.
    ecols = {r["name"] for r in conn.execute("PRAGMA table_info(enhanced_faces)").fetchall()}
    if ecols:
        for col, decl in (("best_source_path", "TEXT"), ("best_source_frame", "INTEGER"),
                          ("best_source_quality", "REAL"), ("best_source_hash", "TEXT"),
                          ("enhancement_applied", "INTEGER"), ("gate_status", "TEXT"),
                          ("gate_reason", "TEXT")):
            if col not in ecols:
                conn.execute(f"ALTER TABLE enhanced_faces ADD COLUMN {col} {decl}")
    # Camera Registry: siting details needed for real journey reconstruction.
    ccols = {r["name"] for r in conn.execute("PRAGMA table_info(cameras)").fetchall()}
    for col, decl in (("address", "TEXT"), ("road_name", "TEXT"),
                      ("facing_deg", "REAL"), ("fov_deg", "REAL"),
                      ("coverage_m", "REAL"), ("description", "TEXT"),
                      ("active", "INTEGER DEFAULT 1"), ("source", "TEXT"),
                      ("updated_at", "TEXT")):
        if col not in ccols:
            conn.execute(f"ALTER TABLE cameras ADD COLUMN {col} {decl}")


def init_db() -> None:
    """Create tables and seed cameras from camera_config.json (once)."""
    with get_conn() as conn:
        conn.executescript(SCHEMA)
        _migrate(conn)
        # Sync configured cameras from camera_config.json on every init, so the
        # camera list is user-owned (edit the file -> names/locations update).
        # Cameras that self-register from footage (not in the file) are untouched.
        if config.CAMERA_CONFIG_PATH.exists():
            try:
                cams = json.loads(config.CAMERA_CONFIG_PATH.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                cams = []
            for c in cams:
                conn.execute(
                    "INSERT OR IGNORE INTO cameras "
                    "(camera_id, name, location, lat, lon, created_at) VALUES (?,?,?,?,?,?)",
                    (c["camera_id"], c.get("name"), c.get("location"),
                     c.get("lat"), c.get("lon"), _now()),
                )
                conn.execute(
                    "UPDATE cameras SET name=?, location=?, lat=?, lon=? WHERE camera_id=?",
                    (c.get("name"), c.get("location"), c.get("lat"), c.get("lon"), c["camera_id"]),
                )
            # Drop stale cameras that aren't configured and have no footage
            # (e.g. removed demo entries), so the UI never shows empty cameras.
            # Cameras created/edited through the Camera Registry (source='registry')
            # are PERMANENT and never dropped, even before any video is uploaded.
            config_ids = [c["camera_id"] for c in cams]
            if config_ids:
                ph = ",".join("?" * len(config_ids))
                conn.execute(
                    f"DELETE FROM cameras WHERE camera_id NOT IN ({ph}) AND camera_id NOT IN "
                    "(SELECT DISTINCT camera_id FROM videos WHERE camera_id IS NOT NULL) "
                    "AND COALESCE(source,'') <> 'registry'",
                    config_ids,
                )


def register_camera(camera_id, name=None, location=None, lat=None, lon=None) -> None:
    """Add a camera on the fly (used when the judge uploads unfamiliar footage)."""
    with get_conn() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO cameras "
            "(camera_id, name, location, lat, lon, created_at) VALUES (?,?,?,?,?,?)",
            (camera_id, name or camera_id, location, lat, lon, _now()),
        )


def list_cameras() -> list[dict]:
    with get_conn() as conn:
        return [dict(r) for r in conn.execute("SELECT * FROM cameras").fetchall()]


def list_videos() -> list[dict]:
    with get_conn() as conn:
        return [dict(r) for r in conn.execute(
            "SELECT video_id, camera_id, filename, fps, start_time, duration, end_time, "
            " native_fps, width, height, status FROM videos ORDER BY camera_id, start_time"
        ).fetchall()]


def video_index() -> dict:
    """{video_id: {filename, camera_id, start_time, duration, native_fps, width, height}}
    - used to resolve a detection to its recording clip + seek offset."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT video_id, camera_id, filename, start_time, duration, native_fps, width, height "
            "FROM videos"
        ).fetchall()
    return {r["video_id"]: dict(r) for r in rows}


def log_audit(action, query_text=None, query_type=None,
              result_count=None, user="officer", details=None) -> None:
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO audit_log "
            "(timestamp, action, query_text, query_type, result_count, user, details) "
            "VALUES (?,?,?,?,?,?,?)",
            (_now(), action, query_text, query_type, result_count, user,
             json.dumps(details) if details is not None else None),
        )


def add_video(camera_id, filename, fps=None, start_time=None, duration=None, status="pending",
              end_time=None, native_fps=None, width=None, height=None) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO videos (camera_id, filename, fps, start_time, duration, end_time, "
            " native_fps, width, height, status, created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (camera_id, filename, fps, start_time, duration, end_time,
             native_fps, width, height, status, _now()),
        )
        return cur.lastrowid


def set_video_status(video_id, status) -> None:
    with get_conn() as conn:
        conn.execute("UPDATE videos SET status=? WHERE video_id=?", (status, video_id))


_DET_COLS = ("video_id, camera_id, track_id, frame_number, timestamp, class_label, confidence, "
             "bbox_x, bbox_y, bbox_w, bbox_h, crop_path, clip_vec_id, reid_vec_id, attributes")
_DET_QS = "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"


def _det_values(d: dict) -> tuple:
    return (d.get("video_id"), d.get("camera_id"), d.get("track_id"), d.get("frame_number"),
            d.get("timestamp"), d.get("class_label"), d.get("confidence"),
            d.get("bbox_x"), d.get("bbox_y"), d.get("bbox_w"), d.get("bbox_h"),
            d.get("crop_path"), d.get("clip_vec_id"), d.get("reid_vec_id"),
            json.dumps(d["attributes"]) if d.get("attributes") is not None else None)


def insert_detection(d: dict) -> int:
    """Insert one detection row (dict of column values). Returns detection_id."""
    with get_conn() as conn:
        cur = conn.execute(
            f"INSERT INTO detections ({_DET_COLS}) VALUES {_DET_QS}", _det_values(d))
        return cur.lastrowid


def insert_detections_bulk(rows: list[dict]) -> list[int]:
    """Insert many detection rows in ONE connection/transaction and return their
    detection_ids in order. Identical rows to insert_detection, but avoids the
    per-row connect+commit overhead - a big speed-up on video ingest."""
    if not rows:
        return []
    ids = []
    with get_conn() as conn:
        cur = conn.cursor()
        for d in rows:
            cur.execute(f"INSERT INTO detections ({_DET_COLS}) VALUES {_DET_QS}", _det_values(d))
            ids.append(cur.lastrowid)
    return ids


def upsert_track(track_key, video_id, camera_id, track_id, class_label,
                 start_frame, end_frame, start_time, end_time, direction=None) -> None:
    with get_conn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO tracks "
            "(track_key, video_id, camera_id, track_id, class_label, start_frame, end_frame, "
            " start_time, end_time, direction) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (track_key, video_id, camera_id, track_id, class_label,
             start_frame, end_frame, start_time, end_time, direction),
        )


def _row_to_detection(row) -> dict:
    d = dict(row)
    if d.get("attributes"):
        try:
            d["attributes"] = json.loads(d["attributes"])
        except (TypeError, ValueError):
            d["attributes"] = {}
    else:
        d["attributes"] = {}
    return d


def get_detections(ids) -> list[dict]:
    """Fetch detections by id, preserving the given order (for ranked results)."""
    ids = list(ids)
    if not ids:
        return []
    placeholders = ",".join("?" * len(ids))
    with get_conn() as conn:
        rows = conn.execute(
            f"SELECT * FROM detections WHERE detection_id IN ({placeholders})", ids
        ).fetchall()
    by_id = {r["detection_id"]: _row_to_detection(r) for r in rows}
    return [by_id[i] for i in ids if i in by_id]


def get_track_detections(video_id, track_id) -> list[dict]:
    """All detection rows for ONE ByteTrack track within a video, time-ordered.

    Used by the single-camera tracking viewer to replay a track's per-frame
    bounding boxes. Reads stored metadata only - no detection/tracking is re-run.
    Scene (whole-frame) rows are excluded."""
    if video_id is None or track_id is None:
        return []
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM detections WHERE video_id=? AND track_id=? "
            "AND class_label != 'scene' ORDER BY frame_number",
            (video_id, track_id),
        ).fetchall()
    return [_row_to_detection(r) for r in rows]


def query_detections(camera_ids=None, start_time=None, end_time=None,
                     class_labels=None, limit=2000) -> list[dict]:
    """Metadata-only filtered query (time / camera / class)."""
    clauses, params = [], []
    if camera_ids:
        clauses.append(f"camera_id IN ({','.join('?' * len(camera_ids))})")
        params += list(camera_ids)
    if start_time:
        clauses.append("timestamp >= ?")
        params.append(start_time)
    if end_time:
        clauses.append("timestamp <= ?")
        params.append(end_time)
    if class_labels:
        clauses.append(f"class_label IN ({','.join('?' * len(class_labels))})")
        params += list(class_labels)
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    with get_conn() as conn:
        rows = conn.execute(
            f"SELECT * FROM detections {where} ORDER BY timestamp LIMIT ?", params + [limit]
        ).fetchall()
    return [_row_to_detection(r) for r in rows]


def count_detections() -> int:
    with get_conn() as conn:
        return conn.execute("SELECT COUNT(*) AS n FROM detections").fetchone()["n"]


def insert_face(f: dict) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO faces (detection_id, camera_id, timestamp, face_vec_id, age, gender, crop_path) "
            "VALUES (?,?,?,?,?,?,?)",
            (f.get("detection_id"), f.get("camera_id"), f.get("timestamp"),
             f.get("face_vec_id"), f.get("age"), f.get("gender"), f.get("crop_path")),
        )
        return cur.lastrowid


def get_faces(ids) -> list[dict]:
    ids = list(ids)
    if not ids:
        return []
    placeholders = ",".join("?" * len(ids))
    with get_conn() as conn:
        rows = conn.execute(
            f"SELECT * FROM faces WHERE face_id IN ({placeholders})", ids
        ).fetchall()
    by_id = {r["face_id"]: dict(r) for r in rows}
    return [by_id[i] for i in ids if i in by_id]


def count_faces() -> int:
    with get_conn() as conn:
        return conn.execute("SELECT COUNT(*) AS n FROM faces").fetchone()["n"]


def insert_plate(p: dict) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO plates (detection_id, camera_id, timestamp, plate_text, "
            " confidence, crop_path, votes, source, plate_crop) VALUES (?,?,?,?,?,?,?,?,?)",
            (p.get("detection_id"), p.get("camera_id"), p.get("timestamp"),
             p.get("plate_text"), p.get("confidence"), p.get("crop_path"),
             p.get("votes"), p.get("source"), p.get("plate_crop")),
        )
        return cur.lastrowid


def get_plates(ids) -> list[dict]:
    ids = list(ids)
    if not ids:
        return []
    placeholders = ",".join("?" * len(ids))
    with get_conn() as conn:
        rows = conn.execute(
            f"SELECT * FROM plates WHERE plate_id IN ({placeholders})", ids
        ).fetchall()
    by_id = {r["plate_id"]: dict(r) for r in rows}
    return [by_id[i] for i in ids if i in by_id]


def search_plates(text, camera_ids=None, start_time=None, end_time=None, limit=200) -> list[dict]:
    """Partial/full match on plate_text. Both sides are normalised (uppercased,
    spaces/hyphens stripped) so 'AB1234' matches a stored 'GJ 05 AB 1234'."""
    needle = "%" + "".join(ch for ch in (text or "").upper() if ch.isalnum()) + "%"
    clauses = ["REPLACE(REPLACE(UPPER(plate_text), ' ', ''), '-', '') LIKE ?"]
    params = [needle]
    if camera_ids:
        clauses.append(f"camera_id IN ({','.join('?' * len(camera_ids))})")
        params += list(camera_ids)
    if start_time:
        clauses.append("timestamp >= ?")
        params.append(start_time)
    if end_time:
        clauses.append("timestamp <= ?")
        params.append(end_time)
    where = " AND ".join(clauses)
    with get_conn() as conn:
        rows = conn.execute(
            f"SELECT * FROM plates WHERE {where} ORDER BY confidence DESC LIMIT ?",
            params + [limit],
        ).fetchall()
    return [dict(r) for r in rows]


def count_plates() -> int:
    with get_conn() as conn:
        return conn.execute("SELECT COUNT(*) AS n FROM plates").fetchone()["n"]


def insert_export(export_id, case_number, officer, created_at, manifest_hash,
                  file_path, detection_ids) -> None:
    with get_conn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO exports "
            "(export_id, case_number, officer, created_at, manifest_hash, file_path, detection_ids) "
            "VALUES (?,?,?,?,?,?,?)",
            (export_id, case_number, officer, created_at, manifest_hash, file_path,
             json.dumps(list(detection_ids))),
        )


def list_exports() -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM exports ORDER BY created_at DESC").fetchall()
    out = []
    for r in rows:
        d = dict(r)
        try:
            d["detection_ids"] = json.loads(d["detection_ids"]) if d.get("detection_ids") else []
        except (TypeError, ValueError):
            d["detection_ids"] = []
        out.append(d)
    return out


if __name__ == "__main__":
    init_db()
    print(f"Initialised DB at {config.DB_PATH}")
    print(f"Cameras: {len(list_cameras())}")
    print(f"Detections: {count_detections()}")
    print(f"Faces: {count_faces()}")
    print(f"Plates: {count_plates()}")
