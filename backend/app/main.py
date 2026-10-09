"""FastAPI entry point.

Run from the backend/ directory:
    uvicorn app.main:app --reload --port 8000

Interactive API docs at http://localhost:8000/docs
"""
import asyncio
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from . import config, database, ingest_jobs
from .routes import router
from .routes_library import router as library_router
from .routes_registry import router as registry_router
from .routes_history import router as history_router
from .routes_faces import router as faces_router
from .routes_system import router as system_router
from .routes_journey import router as journey_router
from .routes_camera_registry import router as camera_registry_router
from .routes_case import router as case_router



def _warm_search_models() -> None:
    """Load the SEARCH models once, in the background, at boot.

    They were already loaded only once per process (module-level singletons, never
    reloaded per request) - but lazily, so the very FIRST search of a session paid
    the cost: measured 4.60 s for CLIP, after which every call was 0 ms. Doing it
    at startup moves that wait off the officer's first query.

    Only CLIP and the FAISS indexes are warmed - together a small, predictable
    footprint. YOLO / OSNet are deliberately NOT preloaded (InsightFace is, below): on a 6 GB
    card holding all of them resident risks an out-of-memory failure during a real
    pass, which would be a worse problem than a slow first click.
    """
    try:
        from .ingestion import embedder
        from .search import vector_store
        embedder.get_clip()
        for name in ("clip", "reid", "face"):
            vector_store.get_index(name)
        print("[startup] search models + FAISS indexes warm")
    except Exception as exc:                      # never block the server from starting
        print(f"[startup] model warm-up skipped: {exc}")
    # InsightFace too: measured 30.5 s to load on first use, which landed on the
    # officer's first Save Face / face preview after every restart. It does not
    # raise peak VRAM - ingestion and every face feature load the same singleton
    # and keep it resident for the life of the process anyway.
    try:
        from . import config as _cfg
        if getattr(_cfg, "FACE_RECOGNITION_ENABLED", True):
            from .ingestion import face_recognizer
            face_recognizer.get_face_app()
            print("[startup] face model warm")
    except Exception as exc:
        print(f"[startup] face model warm-up skipped: {exc}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    database.init_db()
    # Background thread so the port opens immediately and health checks don't wait.
    threading.Thread(target=_warm_search_models, name="warmup", daemon=True).start()
    yield
    # Shutdown (nothing to clean up yet)


app = FastAPI(
    title="NiriXan AI - Forensic Investigation API",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    """Lightweight health + environment check the frontend can ping."""
    info = {
        "status": "ok",
        "device": config.DEVICE,
        "low_vram_mode": config.LOW_VRAM,
        "cameras": len(database.list_cameras()),
    }
    try:
        import torch
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            total = torch.cuda.get_device_properties(0).total_memory
            info["gpu_vram_gb"] = round(total / (1024 ** 3), 1)
    except Exception:
        pass
    return info


@app.get("/")
def root():
    return {"name": "NiriXan AI - Forensic Investigation API",
            "docs": "/docs", "health": "/api/health"}


# REST routes (search / track / cameras / audit / ingest).
app.include_router(router, prefix="/api")
app.include_router(library_router, prefix="/api")
app.include_router(registry_router, prefix="/api")
app.include_router(history_router, prefix="/api")
app.include_router(faces_router, prefix="/api")
app.include_router(system_router, prefix="/api")
app.include_router(journey_router, prefix="/api")
app.include_router(camera_registry_router, prefix="/api")
app.include_router(case_router, prefix="/api")


# Serve crop/frame images the search results point to. media_url() emits
# "/media/<path relative to DATA_DIR>", so mount DATA_DIR at /media.
app.mount("/media", StaticFiles(directory=str(config.DATA_DIR)), name="media")


@app.websocket("/ws/ingest/{job_id}")
async def ws_ingest(websocket: WebSocket, job_id: str):
    """Stream ingestion progress for a job until it finishes (or is unknown)."""
    await websocket.accept()
    try:
        while True:
            job = ingest_jobs.get(job_id)
            if job is None:
                await websocket.send_json({"job_id": job_id, "status": "unknown"})
                break
            await websocket.send_json(job)
            if job.get("status") in ("done", "error"):
                break
            await asyncio.sleep(0.3)
    except WebSocketDisconnect:
        pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
