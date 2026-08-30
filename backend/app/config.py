"""Central configuration: paths, model settings, thresholds, vocabularies.

Everything tunable lives here so the rest of the code stays clean.
"""
from pathlib import Path

try:
    import torch
    _CUDA = torch.cuda.is_available()
except Exception:  # torch not installed yet (e.g. before env setup)
    _CUDA = False

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent          # .../backend

# Load backend/.env (if present) so secrets like GEMINI_API_KEY are available as
# environment variables without hardcoding them. Safe no-op if python-dotenv or
# the file is missing.
try:
    from dotenv import load_dotenv
    load_dotenv(BASE_DIR / ".env")
except Exception:
    pass
DATA_DIR = BASE_DIR / "data"
VIDEO_DIR = DATA_DIR / "videos"                            # source footage
FRAME_DIR = DATA_DIR / "frames"                            # extracted frames
CROP_DIR = DATA_DIR / "crops"                              # detected object crops
FACE_DIR = DATA_DIR / "faces"                              # face crops
FAISS_DIR = DATA_DIR / "faiss_indexes"                     # saved vector indexes
EXPORT_DIR = DATA_DIR / "exports"                          # forensic exports
DB_PATH = DATA_DIR / "cctv.db"
CAMERA_CONFIG_PATH = DATA_DIR / "camera_config.json"

for _d in (VIDEO_DIR, FRAME_DIR, CROP_DIR, FACE_DIR, FAISS_DIR, EXPORT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# Device / VRAM
# ------------------------------------------------------------------
DEVICE = "cuda" if _CUDA else "cpu"
# On a 4GB GPU we load models one stage at a time and clear the cache
# between stages during ingestion. Set False on bigger cards to keep
# models resident for speed.
LOW_VRAM = True

# ------------------------------------------------------------------
# Frame extraction
# ------------------------------------------------------------------
DEFAULT_FPS = 2            # frames sampled per second of source video (denser
                          # sampling = better attribute coverage for describe-
                          # search; applies to newly processed videos)
UPLOAD_FAST_FPS = 1        # sparser sampling for "analyse this upload now" so the
                          # clip is searchable quickly; skips the CPU-heavy face
                          # + plate bonus stages too (see routes_library upload)

# ------------------------------------------------------------------
# Detection (YOLO) + tracking (ByteTrack)
# ------------------------------------------------------------------
YOLO_MODEL = "yolov10b.pt"                 # auto-downloaded by ultralytics
# COCO class ids we index. Person + vehicles are the core (they match the
# problem statement); a few carried-object classes broaden search coverage
# (e.g. "person with a backpack", "abandoned bag").
DETECT_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
    24: "backpack",
    25: "umbrella",
    26: "handbag",
    28: "suitcase",
}
# Class ids the PRIMARY (COCO) YOLOv10 is asked to detect. Snapshotted BEFORE any
# secondary/plugin classes are merged into DETECT_CLASSES below, so the primary
# model is never asked for a non-COCO id.
PRIMARY_CLASSES = set(DETECT_CLASSES)
# Groupings the attribute extractor uses to decide which attributes apply.
PERSON_CLASSES = {0}
VEHICLE_CLASSES = {1, 2, 3, 5, 7}
DETECT_CONF = 0.4                          # confidence floor for a STORED detection

# ------------------------------------------------------------------
# Tracking stability (CCTV-tuned)
# ------------------------------------------------------------------
# ByteTrack config tuned for sampled CCTV (see the yaml for the reasoning).
# Falls back to the ultralytics built-in if the file is ever missing.
_CCTV_TRACKER = Path(__file__).resolve().parent / "ingestion" / "trackers" / "bytetrack_cctv.yaml"
TRACKER_CFG = str(_CCTV_TRACKER) if _CCTV_TRACKER.exists() else "bytetrack.yaml"

# Confidence handed to YOLO for the TRACKING pass. Lower than DETECT_CONF on
# purpose: ByteTrack's second association stage needs low-score boxes to carry a
# track through occlusion. Detections are still filtered to DETECT_CONF before
# anything is stored, so the database and search results keep the same quality
# floor as before.
TRACK_INPUT_CONF = 0.15

# --- Appearance-assisted association (identity_guard) ---
TRACK_APPEARANCE_GUARD = True     # verify every ByteTrack association with ReID
# A detection may only extend a track if it looks like that track's stored
# appearance at least this much. Below it the association is REFUSED rather than
# allowed to overwrite the identity. Swept in scripts/benchmark_tracking.py.
TRACK_REID_MIN_SIM = 0.62
# The detection must ALSO still look like the track's most recent confirmed view.
# The view set alone is too forgiving: after a track has accumulated several poses
# it can contain a decent match for two different people, and an identity hand-over
# only shows clearly frame-to-frame. Set to 0.0 to disable this second test.
TRACK_REID_RECENT_MIN = 0.55
# Re-acquiring a track that was lost needs stronger evidence than continuing one.
TRACK_REACQUIRE_MIN_SIM = 0.72
# How long (in SAMPLED frames) a lost track stays eligible for re-acquisition.
# 90 frames = ~45 s at the default 2 fps, matching track_buffer in the yaml.
TRACK_LOST_WINDOW = 90
# Representative embeddings kept per live track (Part 6). More views = more
# posture/lighting coverage when re-acquiring or matching across cameras.
TRACK_MAX_EMBEDDINGS = 8
# A track must be seen this many times before its appearance is trusted enough to
# reject an association; one noisy first crop should not lock a track down.
TRACK_GUARD_WARMUP = 2

# --- Detection responsibility split (India-vehicle redesign) ---
# COCO YOLOv10 is redesigned to own only PEOPLE + GENERAL OBJECTS (person, bicycle,
# backpack, umbrella, handbag, suitcase). The COCO *motorised* vehicle classes
# (car / motorcycle / bus / truck) are handed to the dedicated India Vehicle
# Detector, which becomes the SOLE source of vehicle detections.
#
# Safety fallback: this hand-off only happens when the India detector is actually
# loaded (weights present). If it isn't, the primary model keeps detecting these
# COCO vehicles so vehicle detection never disappears (the tracker selects the
# class set at runtime - see tracker.iter_track_chunks).
PRIMARY_VEHICLE_CLASSES = {2, 3, 5, 7}                       # car, motorcycle, bus, truck
PRIMARY_NONVEHICLE_CLASSES = set(PRIMARY_CLASSES) - PRIMARY_VEHICLE_CLASSES  # people + objects (+bicycle)

# ------------------------------------------------------------------
# India-specific SECONDARY detectors (extensible, multi-class plugin system)
# ------------------------------------------------------------------
# A secondary detector (a YOLO11n trained on an Indian driving dataset) runs
# ALONGSIDE YOLOv10 and its boxes are merged into ONE unified stream via
# class-aware NMS. Downstream (ByteTrack, CLIP, attributes, plate OCR, ReID,
# FAISS, search, export) is untouched - it just sees more, correctly-labelled
# boxes and never knows which detector produced them.
#
# Canonical India vehicle taxonomy. FIXED global ids keep the downstream label
# sets (search / filters / attributes) stable no matter which dataset the model
# was trained on. COCO-equivalent classes (car/bus/truck/motorcycle/bicycle) are
# mapped onto the EXISTING COCO ids so a secondary "car" never duplicates the
# primary "car" - only genuinely India-specific classes get new ids (100+).
INDIA_VEHICLE_CLASSES = {          # global_id: (label, india_specific)
    100: ("auto-rickshaw", True),  # also "three-wheeler"
    101: ("tractor", True),
    102: ("tempo", True),          # small goods LCV (matador / tempo)
    103: ("mini-truck", True),     # Tata Ace / chhota hathi
    104: ("hcv", True),            # heavy commercial vehicle (multi-axle / trailer)
    105: ("lcv", True),            # light commercial vehicle
    106: ("scooter", True),        # distinct from motorcycle (COCO conflates them)
    107: ("pickup", True),         # pickup / goods carrier
}
INDIA_SPECIFIC_IDS = {g for g, (_l, s) in INDIA_VEHICLE_CLASSES.items() if s}

# Dataset class-name (normalised: lowercase, alphanumerics only) -> global id.
# Lets a plugin auto-build its class map from ANY dataset's class names, so the
# detector can support as many Indian vehicle classes as the dataset provides.
VEHICLE_NAME_ALIASES = {
    # India-specific (new global ids)
    "autorickshaw": 100, "auto": 100, "rickshaw": 100, "autorick": 100,
    "tuktuk": 100, "threewheeler": 100, "3wheeler": 100,
    "tractor": 101,
    "tempo": 102, "tempotraveller": 102, "matador": 102,
    "minitruck": 103, "tataace": 103, "ace": 103, "chotahathi": 103,
    "chhotahathi": 103,
    "hcv": 104, "heavyvehicle": 104, "heavycommercialvehicle": 104,
    "trailer": 104, "multiaxle": 104, "trucktrailer": 104,
    "lcv": 105, "lightcommercialvehicle": 105,
    "scooter": 106, "moped": 106, "activa": 106,
    "pickup": 107, "pickuptruck": 107, "goodscarrier": 107,
    # generic -> EXISTING COCO ids (the India detector now owns these too)
    "car": 2, "sedan": 2, "hatchback": 2, "suv": 2, "jeep": 2, "taxi": 2, "van": 2,
    "bus": 5, "minibus": 5,
    "truck": 7, "lorry": 7,
    "motorcycle": 3, "motorbike": 3, "bike": 3, "twowheeler": 3,
    "bicycle": 1, "cycle": 1,
    "person": 0, "pedestrian": 0, "rider": 0,
}

SECONDARY_DETECTOR_SPECS = [
    {
        "name": "india_vehicles",
        "weights": str(BASE_DIR.parent / "auto_rickshaw_detector" / "weights" / "india_vehicles.pt"),
        # class_map omitted -> built AUTOMATICALLY from the model's own class names
        # via VEHICLE_NAME_ALIASES at load time (dataset-agnostic, multi-class).
        "conf": 0.35,
        "imgsz": 640,
        "enabled": True,          # no-op until the weights file is present
    },
    # Add another India-specific detector later with ZERO downstream changes:
    # {"name": "special_vehicles", "weights": ".../special.pt", "conf": 0.35, "imgsz": 640, "enabled": True},
]
SECONDARY_NMS_IOU = 0.55          # overlap above which the more specific Indian box
                                  # replaces a generic (COCO) box for the same object

# Register the canonical India vehicle classes so the WHOLE downstream (attributes,
# plate OCR, query parser, filters, search, export) treats them as first-class
# searchable vehicles - with no per-stage changes.
for _gid, (_lbl, _spec) in INDIA_VEHICLE_CLASSES.items():
    DETECT_CLASSES[_gid] = _lbl
    VEHICLE_CLASSES.add(_gid)


def _pick_imgsz() -> int:
    """Adaptive YOLO inference resolution. Wide HD/4K CCTV loses small/distant
    objects at the default 640, so we run larger where the GPU can afford it.
    Chosen once from the GPU's total VRAM (RTX 3050 6GB -> 960)."""
    try:
        if _CUDA:
            import torch
            gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            if gb >= 10:
                return 1280
            if gb >= 5:
                return 960          # RTX 3050 6GB
            return 768
    except Exception:
        pass
    return 640                      # CPU / unknown -> keep the default


YOLO_IMGSZ = _pick_imgsz()          # detection inference resolution (see _pick_imgsz)

# ------------------------------------------------------------------
# Object crops (padding + quality gating before embedding)
# ------------------------------------------------------------------
CROP_PAD_FRAC = 0.12               # padding added around each detection bbox for
                                   # the saved crop (context helps CLIP / re-ID);
                                   # the stored bbox stays tight for overlay accuracy
CROP_MIN_SIDE = 24                 # reject crops whose shorter side is under this (px)
CROP_MIN_VISIBLE = 0.5             # reject if <50% of the detection box is on-screen
CROP_BLUR_VAR_MIN = 8.0            # reject near-flat / severely blurred crops
                                   # (variance-of-Laplacian floor)
CLIP_PAD_SQUARE = True             # letterbox crops to square before CLIP preprocess
                                   # so the whole object survives the 224 centre-crop

# ------------------------------------------------------------------
# CLIP (descriptive text-image search + zero-shot attributes)
# ------------------------------------------------------------------
CLIP_MODEL = "ViT-B-16"
CLIP_PRETRAINED = "laion2b_s34b_b88k"
CLIP_DIM = 512

# ------------------------------------------------------------------
# Person re-ID (OSNet)
# ------------------------------------------------------------------
REID_MODEL = "osnet_x1_0"
REID_DIM = 512
REID_SIM_THRESHOLD = 0.75

# ------------------------------------------------------------------
# Face recognition (InsightFace) - bonus, ethics-gated
# ------------------------------------------------------------------
FACE_MODEL = "buffalo_l"
FACE_DIM = 512
FACE_SIM_THRESHOLD = 0.5
FACE_RECOGNITION_ENABLED = True            # master on/off switch
FACE_DET_MIN = 0.5                         # min InsightFace det_score to trust a face
FACE_VOTE_FRAMES = 5                        # frames per track aggregated for gender/age

# ------------------------------------------------------------------
# License plate OCR (EasyOCR) - bonus
# ------------------------------------------------------------------
OCR_LANGS = ["en"]                         # add "hi" for Hindi script if needed
# Loose Indian plate pattern, e.g. GJ 05 AB 1234
PLATE_REGEX = r"[A-Z]{2}[ -]?\d{1,2}[ -]?[A-Z]{1,3}[ -]?\d{3,4}"
OCR_USE_GPU = True                         # EasyOCR (fallback engine) on GPU
PLATE_RECOGNITION_ENABLED = True           # master on/off switch
PLATE_VOTE_FRAMES = 6                       # frames per vehicle track aggregated for OCR
PLATE_MIN_VOTES = 2                         # a plate seen in >=2 frames is trusted;
                                            # a single-frame read needs high confidence
PLATE_SINGLE_CONF = 0.38                    # confidence floor for a 1-frame plate
                                            # (lowered for recall on small vehicles -
                                            # motorcycles / autos / distant buses)
PLATE_MAX_CANDIDATES = 3                     # store up to N plate reads per vehicle track
                                            # so a top misread never hides the correct
                                            # plate (recall for all vehicle types)
PLATE_FUZZY_THRESHOLD = 0.66                 # plate SEARCH similarity floor: exact
                                            # substrings score 1.0, near-misses (OCR
                                            # noise / partial input) still surface as
                                            # probable results above this (~1 error / 3 chars)

# --- Hybrid OCR engine selection (swappable) ---
OCR_ENGINE = "paddle"                       # "paddle" (PP-OCRv4) | "easyocr"; auto-falls
                                            # back to easyocr if PaddleOCR can't load
PADDLE_USE_GPU = False                      # CPU keeps the 6GB VRAM for torch/CLIP/re-ID
PADDLE_LANG = "en"

# --- Gemini Vision fallback (optional, difficult cases only) ---
GEMINI_ENABLED = True                       # master switch; still a no-op without a key
GEMINI_API_KEY_ENV = "GEMINI_API_KEY"       # env var holding the API key
GEMINI_MODEL = "gemini-flash-lite-latest"

# --- Gemini narration for the Case File evidence report (app/gemini_report.py) ---
# Separate model + switch from the plate fallback above: the report needs written
# scene description rather than a few OCR characters, so it uses the fuller flash
# model. Empty string -> fall back to GEMINI_MODEL. Turning this off only removes
# the narrative; the report still builds from stored attributes.
GEMINI_REPORT_ENABLED = True
GEMINI_REPORT_MODEL = "gemini-flash-latest"
GEMINI_REPORT_TIMEOUT = 60                   # seconds per request
GEMINI_REPORT_WORKERS = 4                    # exhibits described concurrently
PLATE_GEMINI_CONF = 0.55                     # PaddleOCR best-conf below this -> try Gemini
                                            # (only once per vehicle track, key permitting)

# ------------------------------------------------------------------
# High-accuracy ANPR pipeline (plate-region detection -> enhance -> multi-frame
# temporal voting). Layered ON TOP of the existing OCR (does not replace it); the
# old plate_reader path stays intact and is used when ANPR_ENABLED is False.
# ------------------------------------------------------------------
ANPR_ENABLED = True                         # route plate reads through the ANPR pipeline
ANPR_MAX_FRAMES = 8                         # OCR at most this many SHARPEST frames per track
ANPR_BLUR_MIN = 55.0                        # variance-of-Laplacian floor; blurrier frames skipped
ANPR_SR_ENABLED = True                      # super-resolve small plate crops before OCR
ANPR_SR_SCALE = 2                           # super-resolution upscale factor (2 or 4)
ANPR_SR_MODEL = ""                          # optional cv2.dnn_superres model (FSRCNN/EDSR .pb);
                                            # empty -> high-quality LANCZOS (fully offline)
ANPR_PLATE_MIN_SIDE = 26                    # plate ROI shorter side (px) below which SR is forced
ANPR_DENOISE = True                         # edge-preserving denoise on the plate ROI
# Optional dedicated plate DETECTOR (YOLO). If the weights exist it localises the
# plate inside each vehicle; otherwise ANPR falls back to the classical
# morphological plate-region proposer (still OCRs the plate, not the whole vehicle).
PLATE_DETECTOR_WEIGHTS = str(BASE_DIR.parent / "auto_rickshaw_detector" / "weights" / "plate_detector.pt")
PLATE_DETECTOR_CONF = 0.25

# --- Adaptive frame sampling for hard-to-read plates (two-wheelers / autos) ---
# Root-cause fix from the ANPR diagnostic: bike/auto plates are tiny/blurred in the
# sparse 2 FPS samples. For these tracks we re-open the source video and re-sample
# DENSELY within the track's active window, score every candidate frame, and OCR
# only the sharpest/largest plate crops. Normal detection sampling is unchanged.
# --- ANPR cost control -------------------------------------------------------
# OCR is by far the most expensive stage: PaddleOCR here is a CPU-only build
# (paddle 2.6.2, compiled_with_cuda=False), so one OCR call costs ~140 ms and
# read_plates makes several per region. Unbounded, a 300-vehicle clip needs hours.
# These limits keep it finite; every one of them is a time/recall trade-off.
#
# Pre-OCR gate: regions smaller or flatter than this have never yielded a valid
# plate, so they are rejected before an OCR call is spent on them.
ANPR_ROI_MIN_SIDE = 16
ANPR_ROI_BLUR_MIN = 18.0
# Stop refining one region once a confident, well-formed plate has been read.
PLATE_EARLY_EXIT_CONF = 0.80
# Cap the vehicle tracks ANPR is attempted on per video, largest (closest) first.
# Distant vehicles do not produce legible plates, so spending the budget on the
# biggest crops is what maximises plates-per-second.
ANPR_MAX_TRACKS_PER_VIDEO = 60

ANPR_ADAPTIVE_ENABLED = True
ANPR_TWOWHEELER_CLASSES = {3, 100, 106}     # motorcycle, auto-rickshaw, scooter (three-wheeler=100)
ANPR_ADAPTIVE_FPS = 8                        # dense re-sampling FPS inside a track window
# Was 30 frames / top-6. With the early exit in anpr._vote_over_frames a settled
# plate stops the loop regardless, so these ceilings now only bound the tracks
# whose plate never resolves - which is exactly where the hours were going.
ANPR_ADAPTIVE_MAX_FRAMES = 12                # cap candidate frames read per track (cost guard)
ANPR_ADAPTIVE_TOPK = 4                       # sharpest/largest candidates actually OCR'd
ANPR_SCORE_W_BLUR = 0.50                     # frame-quality score weights
ANPR_SCORE_W_SIZE = 0.35
ANPR_SCORE_W_CONF = 0.15
PLATE_CROP_DIR = DATA_DIR / "plate_crops"    # saved best plate crops (evidence)
PLATE_CROP_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# Demo Vehicle Registry (OFFLINE, SYNTHETIC - not a real police database)
# ------------------------------------------------------------------
# One permanent synthetic RC-style record is generated per unique recognised
# plate and stored in SQLite + a JSON mirror. Records are NEVER regenerated and
# only change on manual edit. The provider is swappable so a real police-database
# API can replace the demo later without any frontend change.
REGISTRY_PROVIDER = "demo"                   # "demo" | (future) "police_api"
VEHICLE_REGISTRY_JSON = DATA_DIR / "vehicle_registry.json"

# ------------------------------------------------------------------
# Face Gallery - permanently saved "best face" of a person + face search.
# Reuses the existing InsightFace 'face' FAISS index; nothing else changes.
# ------------------------------------------------------------------
SAVED_FACE_DIR = DATA_DIR / "saved_faces"    # tight best-face crops for saved faces
SAVED_FACE_DIR.mkdir(parents=True, exist_ok=True)
EXPANDED_CROP_DIR = DATA_DIR / "expanded_crops"   # context-padded person crops (display)
EXPANDED_CROP_DIR.mkdir(parents=True, exist_ok=True)
FACE_SIMILAR_MIN = 0.30                      # min cosine similarity for a "similar person" hit

# --- Track-wide face recovery (fixes "No clear face available") ---
# Stored search crops are often too tight (body only). We instead go back to the
# ORIGINAL frame + bbox, expand the box for context, and - if that frame has no
# face - scan neighbouring frames of the SAME ByteTrack track (both directions),
# scoring every candidate to pick the best face anywhere in the track.
FACE_BOX_EXPAND = 0.20                       # expand person bbox by ~20% (15-25%)
FACE_MIN_DET_SCORE = 0.35                    # accept a face above this det score
# Forensic acceptance bar. If the best face in the WHOLE track is below these,
# nothing is saved and the caller reports "No usable face found in this track."
# (a poor crop is worse than none for forensic use).
FACE_ACCEPT_QUALITY = 0.40                   # min composite quality to accept a face
FACE_ACCEPT_MIN_PX = 22                      # min face side in ORIGINAL frame pixels
FACE_DIAG_LOG = True                         # print per-candidate face diagnostics

# --- BEST representative face per track (forensic-grade selection) ---
# Every frame of a ByteTrack person track is inspected; each detected face is
# scored on 9 factors and the single highest-quality face in the whole track wins
# (a later, better face automatically replaces the running best). Very long tracks
# are sampled EVENLY across their full span so coverage is never front-biased.
FACE_TRACK_SCAN_FRAMES = 60                  # frames examined per track (even spread)
FACE_LOW_QUALITY_THRESHOLD = 0.45            # below this the saved face is flagged low-quality
# Composite quality weights (sum ~1.0)
FACE_Q_W_DET = 0.22                          # detection confidence
FACE_Q_W_SIZE = 0.18                         # face size / resolution
FACE_Q_W_SHARP = 0.20                        # sharpness (blur)
FACE_Q_W_POSE = 0.16                         # frontal pose
FACE_Q_W_EYES = 0.08                         # eyes visible
FACE_Q_W_BRIGHT = 0.06                       # brightness (well-exposed)
FACE_Q_W_OCCL = 0.06                         # occlusion / truncation
FACE_Q_W_NOISE = 0.04                        # image noise

# ------------------------------------------------------------------
# Face Enhancement (app/face_enhance.py) - ON-DEMAND derived visualisation.
# Never runs during ingestion. The original saved face is never modified; the
# result is written here as SEPARATE, additional evidence and labelled
# "AI-Enhanced - Derived Visualisation".
# ------------------------------------------------------------------
ENHANCED_FACE_DIR = DATA_DIR / "enhanced_faces"
ENHANCED_FACE_DIR.mkdir(parents=True, exist_ok=True)

# How much of the person's track to re-examine. Same budget as the best-face
# scan, because it re-uses that exact scan order and frame reader.
FACE_ENH_SCAN_FRAMES = 60
# Frames actually fused. Beyond ~8 the median stops improving and only costs time.
FACE_ENH_MAX_FRAMES = 8
# Below this many surviving candidates the pipeline REFUSES. At 1 it will still
# work from a single verified view (upscale + unsharp, no fusion) and says so; it
# only refuses when the footage yields NO usable, identity-verified face at all.
FACE_ENH_MIN_FRAMES = 1

# ------------------------------------------------------------------
# Processing modes: Fast (default, quick indexing/demos) vs Accurate
# (full forensic pipeline). Every knob that differs between the two lives here,
# so the single ingest_video() reads a preset instead of duplicating code.
# ------------------------------------------------------------------
PROCESSING_MODE = "fast"                     # default mode for all ingestion
FAST_ACTIVITY_THRESHOLD = 9.0                # mean inter-frame diff -> 2 FPS if busier, else 1
# Progressive processing: index the video in chunks of this many SAMPLED frames
# so early portions become searchable within seconds instead of waiting for the
# whole clip. Final index is identical - only the processing ORDER changes.
PROGRESSIVE_CHUNK_FRAMES = 80

MODE_PRESETS = {
    "fast": {
        "adaptive_fps": True, "fps": None, "fps_min": 1, "fps_max": 2,
        "imgsz": 736,                        # lower detection resolution (faster)
        "clip_batch": 48,                    # bigger CLIP batches
        "region_split": False,               # cheaper CLIP-only attributes
        "voting": False,                     # single best frame for OCR / face
        # Face recognition (GPU) + licence-plate OCR (CPU) are the slow "bonus"
        # stages and do NOT affect describe / image / text search. Fast mode skips
        # them so uploads index quickly; use Accurate mode for face/plate search.
        "do_faces": False, "do_plates": False,
        "face_min_w": 56, "face_min_h": 112, "face_min_det": 0.60,  # skip tiny/poor faces
        "plate_min_w": 120, "plate_min_h": 70, "plate_blur_min": 40.0,  # only big, sharp plates
        "incremental": True, "index_chunk": 256,   # searchable before the clip finishes
    },
    "accurate": {
        "adaptive_fps": False, "fps": 3,     # higher, fixed sampling
        "imgsz": YOLO_IMGSZ,                 # full adaptive resolution (960 on RTX 3050)
        "clip_batch": 32,
        "region_split": True,                # full attribute extraction (upper/lower + HSV)
        "voting": True,                      # temporal OCR + gender voting
        "do_faces": True, "do_plates": True,
        "face_min_w": 40, "face_min_h": 80, "face_min_det": FACE_DET_MIN,
        "plate_min_w": 44, "plate_min_h": 28, "plate_blur_min": 0.0,  # attempt small vehicles too
        "incremental": False, "index_chunk": 0,
    },
}
# If False, accept general (non-Indian) plate-like strings too, not only the
# strict LL DD L DDDD Indian format - needed for foreign/stock footage.
PLATE_STRICT_INDIAN = False
PLATE_MIN_CONF = 0.10                       # OCR confidence floor for a plate candidate

# ------------------------------------------------------------------
# Search
# ------------------------------------------------------------------
DEFAULT_TOP_K = 60
# CLIP text-image cosine sims sit ~0.18-0.32 even for correct matches, so raw
# scores read as misleadingly low. Map them to a readable 0-1 "relevance" for
# display, and treat anything below the floor as a weak / no-strong match.
CLIP_REL_LOW = 0.18            # maps to 0% relevance
CLIP_REL_HIGH = 0.30           # maps to 100% relevance
CLIP_MIN_RELEVANT = 0.225      # top raw sim below this -> "no strong match"

# ------------------------------------------------------------------
# Attribute vocabularies (used for CLIP zero-shot classification)
# ------------------------------------------------------------------
COLORS = ["red", "blue", "white", "black", "silver", "grey",
          "yellow", "green", "brown", "orange", "maroon", "purple", "pink"]
# Confidence weight applied to the HSV colour reading when it disagrees with the
# CLIP zero-shot colour. 1.0 = balanced (HSV wins only when its pixel support
# beats CLIP's probability) instead of the old unconditional HSV priority.
HSV_COLOR_WEIGHT = 1.0
VEHICLE_TYPES = ["sedan", "hatchback", "SUV", "pickup truck", "van",
                 "auto-rickshaw", "bus", "truck", "motorcycle", "bicycle",
                 "tractor", "tempo", "mini truck", "scooter", "pickup"]
ACCESSORIES = ["cap", "helmet", "backpack", "handbag", "sunglasses", "face mask"]

# ------------------------------------------------------------------
# API
# ------------------------------------------------------------------
CORS_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]
