"""
Router: /runs — Start, monitor, and manage pipeline runs.
"""

import json
import logging
import shutil
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.config import settings
from backend.database import get_db
from backend.models import PipelineRun, StageAttempt
from backend.pipeline.orchestrator import PipelineOrchestrator
from backend.pipeline.stage_locks import get_stage_status

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/runs", tags=["runs"])


# ── Pydantic schemas ───────────────────────────────────────────────────────

class StartRunRequest(BaseModel):
    """Request body for starting a new pipeline run."""
    prompt_id: int
    mode: str  # 'A', 'B', or 'C'
    video_provider_choice: str = "ken_burns"  # 'modal' or 'ken_burns'


class StageAttemptResponse(BaseModel):
    """Full stage attempt data including all extracted ML features."""
    id: str
    stage_name: str
    attempt_number: int
    status: str
    output_path: Optional[str] = None
    human_decision: Optional[str] = None
    reviewer_notes: Optional[str] = None
    created_at: Optional[datetime] = None
    # Script features
    readability_score: Optional[float] = None
    lexical_diversity: Optional[float] = None
    prompt_coverage: Optional[float] = None
    sentence_redundancy: Optional[float] = None
    entity_consistency: Optional[float] = None
    topic_coherence: Optional[float] = None
    factual_conflict_flag: Optional[int] = None
    prompt_ambiguity: Optional[float] = None
    # Audio features
    phoneme_error_rate: Optional[float] = None
    silence_ratio: Optional[float] = None
    speaking_rate_variance: Optional[float] = None
    energy_variance: Optional[float] = None
    tts_word_count_match: Optional[float] = None
    # Visual features
    clip_similarity: Optional[float] = None
    aesthetic_score: Optional[float] = None
    blur_score: Optional[float] = None
    object_match_score: Optional[float] = None
    colour_tone_match: Optional[float] = None
    visual_provider: Optional[str] = None
    # Video features
    av_sync_error_ms: Optional[float] = None
    transition_smoothness: Optional[float] = None
    duration_deviation_s: Optional[float] = None
    # Cross-stage
    prior_stage_corrections: Optional[int] = None
    cumulative_risk_score: Optional[float] = None
    api_retry_count: Optional[int] = None
    is_fallback_video: Optional[bool] = None

    class Config:
        from_attributes = True


class RunStatusResponse(BaseModel):
    """Full status of a pipeline run including all stage attempts."""
    id: str
    prompt_id: int
    prompt_text: str
    mode: str
    status: str
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    final_video_path: Optional[str] = None
    has_final_video: bool = False
    total_corrections: int = 0
    video_provider_used: Optional[str] = None
    video_provider_choice: Optional[str] = None
    stages: List[StageAttemptResponse] = []

    class Config:
        from_attributes = True


class RunSummaryResponse(BaseModel):
    """Lightweight summary for the runs list."""
    id: str
    prompt_id: int
    prompt_text: str
    mode: str
    status: str
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    total_corrections: int = 0
    video_provider_used: Optional[str] = None
    video_provider_choice: Optional[str] = None

    class Config:
        from_attributes = True


# ── Route handlers ─────────────────────────────────────────────────────────

@router.post("/start", response_model=dict, summary="Start a new pipeline run")
def start_run(
    body: StartRunRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """
    Start a new pipeline run in a background thread.
    Returns the run_id immediately; poll /runs/{run_id}/status for progress.
    """
    if body.mode not in ("A", "B", "C"):
        raise HTTPException(status_code=400, detail="mode must be 'A', 'B', or 'C'")

    if not (1 <= body.prompt_id <= 50):
        raise HTTPException(status_code=400, detail="prompt_id must be 1-50")

    if body.video_provider_choice not in ("modal", "modal_ai", "ken_burns"):
        raise HTTPException(status_code=400, detail="video_provider_choice must be 'modal', 'modal_ai', or 'ken_burns'")

    # Normalise "modal_ai" → "modal" so both frontend labels map to the same provider
    _vpc = "modal" if body.video_provider_choice in ("modal", "modal_ai") else "ken_burns"

    # Pre-create the run record so we can return run_id immediately
    run_id = str(uuid.uuid4())
    run = PipelineRun(
        id=run_id,
        prompt_id=body.prompt_id,
        prompt_text="",           # orchestrator will fill this in
        mode=body.mode,
        status="running",
        started_at=datetime.utcnow(),
        video_provider_choice=_vpc,
    )
    db.add(run)
    db.commit()

    orchestrator = PipelineOrchestrator()
    _pid = body.prompt_id
    _mode = body.mode

    def _run():
        try:
            orchestrator.run_pipeline(_pid, _mode, _vpc, run_id=run_id)
        except Exception as exc:
            logger.error(f"Background pipeline run failed: {exc}", exc_info=True)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    return {
        "message": "Pipeline run started.",
        "run_id": run_id,
        "prompt_id": body.prompt_id,
        "mode": body.mode,
    }


@router.get("/{run_id}/status", response_model=RunStatusResponse, summary="Get run status")
def get_run_status(run_id: str, db: Session = Depends(get_db)):
    """Return the current status of a pipeline run including all stage attempts."""
    run = db.query(PipelineRun).filter(PipelineRun.id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    stages = (
        db.query(StageAttempt)
        .filter(StageAttempt.run_id == run_id)
        .order_by(StageAttempt.created_at)
        .all()
    )

    return RunStatusResponse(
        id=run.id,
        prompt_id=run.prompt_id,
        prompt_text=run.prompt_text,
        mode=run.mode,
        status=run.status,
        started_at=run.started_at,
        completed_at=run.completed_at,
        final_video_path=run.final_video_path,
        has_final_video=bool(run.final_video_path and Path(run.final_video_path).exists()),
        total_corrections=run.total_corrections,
        video_provider_used=run.video_provider_used,
        video_provider_choice=run.video_provider_choice,
        stages=[
            StageAttemptResponse(
                id=s.id,
                stage_name=s.stage_name,
                attempt_number=s.attempt_number,
                status=s.status,
                output_path=s.output_path,
                human_decision=s.human_decision,
                reviewer_notes=s.reviewer_notes,
                created_at=s.created_at,
                readability_score=s.readability_score,
                lexical_diversity=s.lexical_diversity,
                prompt_coverage=s.prompt_coverage,
                sentence_redundancy=s.sentence_redundancy,
                entity_consistency=s.entity_consistency,
                topic_coherence=s.topic_coherence,
                factual_conflict_flag=s.factual_conflict_flag,
                prompt_ambiguity=s.prompt_ambiguity,
                phoneme_error_rate=s.phoneme_error_rate,
                silence_ratio=s.silence_ratio,
                speaking_rate_variance=s.speaking_rate_variance,
                energy_variance=s.energy_variance,
                tts_word_count_match=s.tts_word_count_match,
                clip_similarity=s.clip_similarity,
                aesthetic_score=s.aesthetic_score,
                blur_score=s.blur_score,
                object_match_score=s.object_match_score,
                colour_tone_match=s.colour_tone_match,
                visual_provider=s.visual_provider,
                av_sync_error_ms=s.av_sync_error_ms,
                transition_smoothness=s.transition_smoothness,
                duration_deviation_s=s.duration_deviation_s,
                prior_stage_corrections=s.prior_stage_corrections,
                cumulative_risk_score=s.cumulative_risk_score,
                api_retry_count=s.api_retry_count,
                is_fallback_video=s.is_fallback_video,
            )
            for s in stages
        ],
    )


@router.get("", response_model=List[RunSummaryResponse], summary="List all runs")
def list_runs(
    mode: Optional[str] = None,
    status: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """Return a summary list of all pipeline runs, optionally filtered by mode/status."""
    query = db.query(PipelineRun).order_by(PipelineRun.started_at.desc())
    if mode:
        query = query.filter(PipelineRun.mode == mode)
    if status:
        query = query.filter(PipelineRun.status == status)
    runs = query.all()
    return [
        RunSummaryResponse(
            id=r.id,
            prompt_id=r.prompt_id,
            prompt_text=r.prompt_text,
            mode=r.mode,
            status=r.status,
            started_at=r.started_at,
            completed_at=r.completed_at,
            total_corrections=r.total_corrections,
            video_provider_used=r.video_provider_used,
            video_provider_choice=r.video_provider_choice,
        )
        for r in runs
    ]


_ACTIVE_STATUSES = {"running", "pending", "queued", "generating", "waiting_for_review"}
_TERMINAL_STATUSES = {"completed", "failed", "stopped"}


@router.post("/{run_id}/stop", response_model=dict, summary="Stop a running pipeline")
def stop_run(run_id: str, db: Session = Depends(get_db)):
    """
    Stop an active pipeline run. Sets run status to 'stopped' and unblocks
    any pending_review stage so the orchestrator's poll loop exits cleanly.
    """
    run = db.query(PipelineRun).filter(PipelineRun.id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    if run.status in _TERMINAL_STATUSES:
        raise HTTPException(
            status_code=400,
            detail=f"Run is already in a terminal state (status: {run.status})",
        )

    # ── Determine which stage is ACTUALLY running right now ───────────────
    # Read the live lock state first (most accurate: this is the stage currently
    # holding a slot).  Fall back to the DB value set at stage-start if the
    # run is queued but hasn't entered the slot yet.
    locks = get_stage_status()
    active_stage: Optional[str] = None
    for stage in ("script", "audio", "video"):
        if locks.get(stage) == run_id:
            active_stage = stage
            break
    if active_stage is None and run_id in locks.get("visual", []):
        active_stage = "visual"
    if active_stage is None:
        # Fallback: use what the orchestrator last committed before entering the stage
        active_stage = run.current_stage

    # Persist the exact stage so resume_pipeline knows where to restart
    if active_stage:
        run.current_stage = active_stage

    run.status = "stopped"
    run.completed_at = datetime.utcnow()

    # Unblock any stage waiting for human review so the orchestrator poll loop exits
    pending = (
        db.query(StageAttempt)
        .filter(StageAttempt.run_id == run_id, StageAttempt.status == "pending_review")
        .first()
    )
    if pending:
        pending.human_decision = "accept"
        pending.reviewer_notes = "Stopped by user"
        pending.decision_timestamp = datetime.utcnow()

    db.commit()
    logger.info(
        f"Run {run_id} stopped by user (active_stage={active_stage!r}, "
        f"from_locks={active_stage is not None!r})."
    )
    return {"message": f"Run {run_id} stopped.", "stopped_at_stage": active_stage}


@router.post("/{run_id}/resume", response_model=dict, summary="Resume a stopped pipeline")
def resume_run(
    run_id: str,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """
    Resume a stopped pipeline from the last incomplete stage.
    Completed stages are skipped — their on-disk outputs are reused.
    The run status is reset to 'running' and execution continues in a
    background thread exactly as a normal run.
    """
    run = db.query(PipelineRun).filter(PipelineRun.id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    if run.status != "stopped":
        raise HTTPException(
            status_code=400,
            detail=f"Only stopped runs can be resumed (current status: {run.status})",
        )

    _VALID_STAGES = {"script", "audio", "visual", "video"}
    from_stage = run.current_stage if run.current_stage in _VALID_STAGES else "script"

    orchestrator = PipelineOrchestrator()
    _fs = from_stage  # capture for thread closure

    def _resume():
        try:
            orchestrator.resume_pipeline(run_id, from_stage=_fs)
        except Exception as exc:
            logger.error(f"Background resume of {run_id} failed: {exc}", exc_info=True)

    thread = threading.Thread(target=_resume, daemon=True)
    thread.start()

    logger.info(f"Run {run_id} resume started from stage '{from_stage}'.")
    return {"message": f"Run {run_id} resuming from stage '{from_stage}'.", "run_id": run_id, "from_stage": from_stage}


@router.get("/{run_id}/output/images", response_model=list, summary="List generated images for a run")
def list_run_images(run_id: str, db: Session = Depends(get_db)):
    """Return a list of image filenames available in data/outputs/{run_id}/images/."""
    run = db.query(PipelineRun).filter(PipelineRun.id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    images_dir = settings.OUTPUT_DIR / run_id / "images"
    if not images_dir.exists():
        return []

    return sorted(
        f.name for f in images_dir.iterdir()
        if f.suffix.lower() in (".png", ".jpg", ".jpeg") and f.stat().st_size > 0
    )


@router.get("/{run_id}/output/images/{filename}", summary="Serve a generated image")
def get_run_image(run_id: str, filename: str, db: Session = Depends(get_db)):
    """Serve an individual image from data/outputs/{run_id}/images/{filename}."""
    from fastapi.responses import FileResponse
    run = db.query(PipelineRun).filter(PipelineRun.id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    image_path = settings.OUTPUT_DIR / run_id / "images" / filename
    if not image_path.exists():
        raise HTTPException(status_code=404, detail=f"Image {filename} not found")

    media_types = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}
    media_type = media_types.get(image_path.suffix.lower(), "image/png")
    return FileResponse(str(image_path), media_type=media_type)


@router.get("/{run_id}/output/audio", summary="Stream audio file for a run")
def get_run_audio(run_id: str, db: Session = Depends(get_db)):
    """
    Stream the audio file for a run so the frontend review panel can play it.
    Tries audio.wav first (the clean converted file), then audio.mp3 as fallback.
    """
    from fastapi.responses import FileResponse
    run = db.query(PipelineRun).filter(PipelineRun.id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    for filename, media_type in [("audio.wav", "audio/wav"), ("audio.mp3", "audio/mpeg")]:
        audio_path = settings.OUTPUT_DIR / run_id / filename
        if audio_path.exists():
            return FileResponse(
                str(audio_path),
                media_type=media_type,
                headers={"Accept-Ranges": "bytes"},
            )

    raise HTTPException(status_code=404, detail="No audio file found for this run")


@router.get("/{run_id}/video", summary="Serve final video for a run")
def get_run_video(run_id: str, db: Session = Depends(get_db)):
    """
    Serve the final video for a completed run.
    Looks in order:
      1. final_video_path stored in DB  (data/final_videos/{run_id}.mp4)
      2. data/outputs/{run_id}/final_video.mp4  (legacy / in-progress runs)
    """
    from fastapi.responses import FileResponse
    run = db.query(PipelineRun).filter(PipelineRun.id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    # 1. DB-stored path (data/final_videos/{run_id}.mp4)
    video_path: Optional[Path] = None
    if run.final_video_path and Path(run.final_video_path).exists():
        video_path = Path(run.final_video_path)

    # 2. Legacy fallback: data/outputs/{run_id}/final_video.mp4
    if video_path is None:
        legacy = settings.OUTPUT_DIR / run_id / "final_video.mp4"
        if legacy.exists():
            video_path = legacy

    if video_path is None:
        raise HTTPException(status_code=404, detail="Final video not available for this run")

    return FileResponse(
        str(video_path),
        media_type="video/mp4",
        headers={
            "Accept-Ranges": "bytes",
            "Content-Disposition": "inline",
            "Cache-Control": "no-cache",
        },
    )


@router.get("/{run_id}/output/video", summary="Stream final video for a run (legacy)")
def get_run_video_legacy(run_id: str, db: Session = Depends(get_db)):
    """Legacy endpoint — redirects to /runs/{run_id}/video."""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url=f"/runs/{run_id}/video")


@router.get("/{run_id}/output/final-video", summary="Serve final video file for browser playback (legacy)")
def get_run_final_video_legacy(run_id: str, db: Session = Depends(get_db)):
    """Legacy endpoint — redirects to /runs/{run_id}/video."""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url=f"/runs/{run_id}/video")


@router.get("/{run_id}/output/script", response_model=dict, summary="Get script JSON for a run")
def get_run_script(run_id: str, db: Session = Depends(get_db)):
    """Return the script.json file for a run so the frontend can preview it."""
    run = db.query(PipelineRun).filter(PipelineRun.id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    script_path = settings.OUTPUT_DIR / run_id / "script.json"
    if not script_path.exists():
        raise HTTPException(status_code=404, detail="script.json not found for this run")

    with open(script_path, encoding="utf-8") as f:
        return json.load(f)


@router.get("/{run_id}/output/scenes", response_model=list, summary="List generated scene clips for a run")
def list_run_scenes(run_id: str, db: Session = Depends(get_db)):
    """Return a list of scene clip filenames available in data/outputs/{run_id}/scenes/."""
    run = db.query(PipelineRun).filter(PipelineRun.id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    scenes_dir = settings.OUTPUT_DIR / run_id / "scenes"
    if not scenes_dir.exists():
        return []

    return sorted(
        f.name for f in scenes_dir.iterdir()
        if f.suffix.lower() == ".mp4" and f.stat().st_size > 0
    )


@router.get("/{run_id}/output/scenes/{filename}", summary="Serve a generated scene clip")
def get_run_scene(run_id: str, filename: str, db: Session = Depends(get_db)):
    """Serve an individual scene clip from data/outputs/{run_id}/scenes/{filename}."""
    from fastapi.responses import FileResponse
    run = db.query(PipelineRun).filter(PipelineRun.id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    scene_path = settings.OUTPUT_DIR / run_id / "scenes" / filename
    if not scene_path.exists():
        raise HTTPException(status_code=404, detail=f"Scene {filename} not found")

    return FileResponse(
        str(scene_path),
        media_type="video/mp4",
        headers={"Accept-Ranges": "bytes"},
    )


@router.delete("/{run_id}", response_model=dict, summary="Delete a pipeline run")
def delete_run(run_id: str, db: Session = Depends(get_db)):
    """
    Permanently delete a pipeline run: removes all stage attempts from the database
    (cascade), deletes the PipelineRun row, and removes the output folder on disk.
    """
    run = db.query(PipelineRun).filter(PipelineRun.id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    # Delete DB rows (cascade removes stage_attempts automatically)
    db.delete(run)
    db.commit()

    # Delete output folder from disk (best-effort)
    output_folder = settings.OUTPUT_DIR / run_id
    if output_folder.exists():
        try:
            shutil.rmtree(output_folder)
        except Exception as exc:
            logger.warning(f"Could not delete output folder {output_folder}: {exc}")

    return {"message": f"Run {run_id} deleted."}
