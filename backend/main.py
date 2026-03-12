"""
FastAPI application entry point for the HITL AI Multimedia Pipeline.
Start with: uvicorn backend.main:app --reload
"""

import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.config import settings
from backend.database import init_db
from backend.routers import runs, review, export, providers

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="HITL AI Multimedia Pipeline",
    description=(
        "Adaptive Human-in-the-Loop AI Multimedia Pipeline with "
        "ML-Based Intervention Prediction. "
        "MSc IT Dissertation — Shyam Kishor Pandit, UWS, B01804169."
    ),
    version="1.0.0",
)

# CORS for React dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount routers
app.include_router(runs.router)
app.include_router(review.router)
app.include_router(export.router)
app.include_router(providers.router)


@app.on_event("startup")
async def startup_event() -> None:
    """Initialise the database tables on startup."""
    init_db()
    logger.info("HITL Pipeline API started.")
    logger.info(f"Output directory: {settings.OUTPUT_DIR}")


@app.get("/", tags=["health"])
def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": "HITL AI Multimedia Pipeline",
        "version": "1.0.0",
    }


@app.get("/prompts", tags=["prompts"])
def list_prompts():
    """Return all experiment prompts."""
    import json
    prompts = json.loads(settings.PROMPTS_FILE.read_text(encoding="utf-8"))
    return prompts


@app.get("/stages/status", tags=["stages"])
def get_stages_status():
    """
    Return which stage slots are currently occupied and by which run_id.
    Used by the frontend to render the assembly-line slot indicator.
    Returns: { "script": run_id|null, "audio": run_id|null, ... }
    """
    from backend.pipeline.stage_locks import get_stage_status
    return get_stage_status()


@app.get("/api/file", tags=["files"])
def serve_json_file(path: str):
    """Serve a JSON output file for the ReviewStage preview (script.json)."""
    import json
    from pathlib import Path as _Path
    from fastapi import HTTPException
    file_path = _Path(path)
    # Only allow files inside the configured output directory for safety
    try:
        file_path.relative_to(settings.OUTPUT_DIR)
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied")
    if not file_path.exists() or file_path.suffix != ".json":
        raise HTTPException(status_code=404, detail="File not found")
    return json.loads(file_path.read_text(encoding="utf-8"))


@app.get("/api/media", tags=["files"])
def serve_media_file(path: str):
    """Serve an audio/video/image file for browser preview in ReviewStage."""
    from pathlib import Path as _Path
    from fastapi import HTTPException
    from fastapi.responses import FileResponse
    file_path = _Path(path)
    try:
        file_path.relative_to(settings.OUTPUT_DIR)
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied")
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    media_types = {
        ".mp3": "audio/mpeg",
        ".wav": "audio/wav",
        ".mp4": "video/mp4",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
    }
    media_type = media_types.get(file_path.suffix.lower(), "application/octet-stream")
    return FileResponse(str(file_path), media_type=media_type)
