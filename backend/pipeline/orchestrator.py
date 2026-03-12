"""
Pipeline Orchestrator — Runs a full end-to-end pipeline (all 4 stages).
Implements Mode A (automated), Mode B (human-in-the-loop), and
Mode C (ML-predicted intervention) as specified in Section 7.
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from sqlalchemy.orm import Session

from backend.config import settings
from backend.database import SessionLocal
from backend.models import PipelineRun, StageAttempt
from backend.pipeline.stage1_script import Stage1ScriptGenerator
from backend.pipeline.stage2_audio import Stage2AudioSynthesizer
from backend.pipeline.stage3_visual import Stage3VisualGenerator
from backend.pipeline.stage4_assembly import Stage4VideoAssembler
from backend.pipeline.stage_locks import acquire_stage, release_stage
from backend.pipeline.feature_logger import (
    save_script_features,
    save_audio_features,
    save_visual_features,
    save_video_features,
    update_cross_stage_features,
)

logger = logging.getLogger(__name__)

STAGE_NAMES = ["script", "audio", "visual", "video"]
POLL_INTERVAL = 5   # seconds between review status polls
REVIEW_TIMEOUT = settings.HUMAN_REVIEW_TIMEOUT_SECONDS
MAX_RETRIES = settings.MAX_RETRIES_PER_STAGE

# Sentinel raised internally to exit the pipeline cleanly when stopped.
_STOPPED = "__PIPELINE_STOPPED__"


class PipelineOrchestrator:
    """
    Orchestrates the 4-stage multimedia pipeline.
    Handles Modes A, B, and C with full database persistence.
    """

    def __init__(self) -> None:
        self.script_gen = Stage1ScriptGenerator()
        self.audio_synth = Stage2AudioSynthesizer()
        self.visual_gen = Stage3VisualGenerator()
        self.video_asm = Stage4VideoAssembler()

    # ── Public entry point ─────────────────────────────────────────────────

    def run_pipeline(
        self,
        prompt_id: int,
        mode: str,
        video_provider_choice: str = "ken_burns",
        run_id: Optional[str] = None,
    ) -> str:
        """
        Execute a full pipeline run for the given prompt.

        Args:
            prompt_id: Integer 1-20 mapping to data/prompts.json.
            mode: 'A' (automated), 'B' (human review), or 'C' (ML-predicted).
            video_provider_choice: 'modal' or 'ken_burns'.
            run_id: If provided, update the existing PipelineRun record instead
                    of creating a new one (used when the API pre-creates the row).

        Returns:
            The run_id UUID string.
        """
        prompt_text, topic = self._load_prompt(prompt_id)
        db = SessionLocal()

        try:
            if run_id:
                # Record was pre-created by the API endpoint; just fill in prompt_text
                run = db.query(PipelineRun).filter(PipelineRun.id == run_id).first()
                if run:
                    run.prompt_text = prompt_text
                    db.commit()
                else:
                    run_id = None  # fall through to create below

            if not run_id:
                run_id = str(uuid.uuid4())
                run = PipelineRun(
                    id=run_id,
                    prompt_id=prompt_id,
                    prompt_text=prompt_text,
                    mode=mode,
                    status="running",
                    started_at=datetime.utcnow(),
                    video_provider_choice=video_provider_choice,
                )
                db.add(run)
                db.commit()

            logger.info(f"Pipeline run {run_id} started (mode={mode}).")

            total_corrections = 0
            cumulative_risk = 0.0
            script_data = None
            providers_used = []

            # ── STAGE 1: Script ────────────────────────────────────────────
            run.current_stage = "script"
            db.commit()
            attempt_1, script_data, corrections = self._run_stage(
                db=db,
                run_id=run_id,
                stage_name="script",
                mode=mode,
                attempt_fn=lambda: self.script_gen.run(prompt_text, run_id),
                save_features_fn=save_script_features,
                prior_corrections=total_corrections,
                cumulative_risk=cumulative_risk,
                topic=topic,
            )
            total_corrections += corrections
            cumulative_risk = self._update_risk(attempt_1, cumulative_risk)

            if script_data is None:
                raise RuntimeError("Script generation failed after all retries.")

            # ── STAGE 2: Audio ─────────────────────────────────────────────
            run.current_stage = "audio"
            db.commit()
            attempt_2, _, corrections = self._run_stage(
                db=db,
                run_id=run_id,
                stage_name="audio",
                mode=mode,
                attempt_fn=lambda: self.audio_synth.run(script_data, run_id),
                save_features_fn=save_audio_features,
                prior_corrections=total_corrections,
                cumulative_risk=cumulative_risk,
                topic=topic,
            )
            total_corrections += corrections
            cumulative_risk = self._update_risk(attempt_2, cumulative_risk)

            # ── STAGE 3: Visual — generates scene clips (both paths) ──────
            # ken_burns: DreamShaper image -> Ken Burns FFmpeg clip per scene
            # modal:     Modal endpoint clip per scene (Ken Burns fallback per scene)
            _vpc = video_provider_choice  # capture for lambda closure
            _sd = script_data             # capture for lambda closure

            run.current_stage = "visual"
            db.commit()
            attempt_3, visual_result, corrections = self._run_stage(
                db=db,
                run_id=run_id,
                stage_name="visual",
                mode=mode,
                attempt_fn=lambda: self.visual_gen.run(_sd, run_id, _vpc),
                save_features_fn=save_visual_features,
                prior_corrections=total_corrections,
                cumulative_risk=cumulative_risk,
                topic=topic,
                provider=_vpc,  # modal → semaphore(5), ken_burns → semaphore(1)
            )
            total_corrections += corrections
            cumulative_risk = self._update_risk(attempt_3, cumulative_risk)

            # ── STAGE 4: Final Assembly — FFmpeg concat + audio mix ────────
            run.current_stage = "video"
            db.commit()
            attempt_4, asm_result, corrections = self._run_stage(
                db=db,
                run_id=run_id,
                stage_name="video",
                mode=mode,
                attempt_fn=lambda: self.video_asm.run(run_id, video_provider_choice=video_provider_choice),
                save_features_fn=save_video_features,
                prior_corrections=total_corrections,
                cumulative_risk=cumulative_risk,
                topic=topic,
            )
            total_corrections += corrections

            # ── Finalise run ───────────────────────────────────────────────
            final_video_path = asm_result.get("output_path") if asm_result else None

            # Copy final video to data/final_videos/{run_id}.mp4
            stored_video_path = None
            if final_video_path and Path(final_video_path).exists():
                final_videos_dir = settings.DATA_DIR / "final_videos"
                final_videos_dir.mkdir(parents=True, exist_ok=True)
                dest = final_videos_dir / f"{run_id}.mp4"
                import shutil as _shutil
                _shutil.copy2(final_video_path, dest)
                stored_video_path = str(dest)
                logger.info(f"Final video stored at: {stored_video_path}")

            run.status = "completed"
            run.completed_at = datetime.utcnow()
            run.final_video_path = stored_video_path
            run.total_corrections = total_corrections
            run.video_provider_used = ",".join(p for p in providers_used if p)
            db.commit()
            logger.info(f"Pipeline run {run_id} completed. Video: {stored_video_path}")
            return run_id

        except Exception as exc:
            stopped = _STOPPED in str(exc) or (
                hasattr(db, "is_active") and self._check_stopped(db, run_id)
            )
            if not stopped:
                logger.error(f"Pipeline run {run_id} failed: {exc}", exc_info=True)
            try:
                run = db.query(PipelineRun).get(run_id)
                if run and run.status not in ("stopped",):
                    run.status = "failed"
                    run.completed_at = datetime.utcnow()
                    db.commit()
            except Exception:
                pass
            if not stopped:
                raise
        finally:
            db.close()

    # ── Resume ─────────────────────────────────────────────────────────────

    def resume_pipeline(self, run_id: str, from_stage: str = "script") -> str:
        """
        Resume a stopped pipeline run from `from_stage` onwards.

        Logic per stage (in order):
          1. If the stage's output file already exists on disk → skip it
             (protects against partial outputs; also reuses good prior work).
          2. Otherwise → update current_stage in DB, then run the stage normally
             (locks, retries, and human review gates all apply).

        Args:
            run_id:     UUID of the stopped PipelineRun.
            from_stage: First stage to (attempt to) run — one of
                        "script" | "audio" | "visual" | "video".
                        Stages before this index are never re-run.

        Returns:
            The run_id on completion.
        """
        _STAGE_ORDER = ["script", "audio", "visual", "video"]

        if from_stage not in _STAGE_ORDER:
            logger.warning(
                f"[resume] Invalid from_stage '{from_stage}' — defaulting to 'script'."
            )
            from_stage = "script"

        start_index = _STAGE_ORDER.index(from_stage)

        db = SessionLocal()
        try:
            run = db.query(PipelineRun).get(run_id)
            if not run:
                raise ValueError(f"Run {run_id} not found")

            prompt_text = run.prompt_text
            mode = run.mode
            vpc = run.video_provider_choice or "ken_burns"
            _, topic = self._load_prompt(run.prompt_id)

            # Reset run to active state
            run.status = "running"
            run.completed_at = None
            db.commit()

            total_corrections = run.total_corrections or 0
            cumulative_risk = 0.0
            script_data: Optional[dict] = None
            asm_result: Optional[dict] = None

            logger.info(
                f"[resume] Run {run_id}: starting from stage '{from_stage}' "
                f"(mode={mode}, vpc={vpc})"
            )

            # ── Pre-load prior stage outputs ───────────────────────────────
            # The loop starts at from_stage, so stages before it are never
            # visited and their outputs are never loaded into memory.
            # Audio and visual stages both require script_data; load it now
            # so the lambda closures don't receive None.
            if start_index >= 1:  # audio, visual, or video
                script_data = self._load_script_from_disk(run_id)
                if script_data is None and start_index <= 2:
                    # Audio (1) and visual (2) both call script_data.get(...)
                    raise RuntimeError(
                        f"Cannot resume from '{from_stage}': script.json not found at "
                        f"{settings.OUTPUT_DIR / run_id / 'script.json'}. "
                        "Stage 1 must have completed before resuming from audio or visual."
                    )
                if script_data is not None:
                    logger.info(f"[resume] Pre-loaded script.json for run {run_id}")

            for stage_name in _STAGE_ORDER[start_index:]:

                # ── Check if this stage's output is already on disk ────────
                if self._stage_output_exists(run_id, stage_name):
                    logger.info(
                        f"[resume] Stage '{stage_name}' — output file exists on disk, "
                        "skipping re-generation."
                    )
                    if stage_name == "script":
                        # Load script_data so subsequent stages can use it
                        script_data = self._load_script_from_disk(run_id)
                    continue   # move to next stage without running

                # ── Update current_stage BEFORE doing any work ────────────
                run.current_stage = stage_name
                db.commit()

                # Snapshot mutable closures for lambda safety
                _pt  = prompt_text
                _sd  = script_data
                _vpc = vpc

                if stage_name == "script":
                    attempt, result, corrections = self._run_stage(
                        db=db,
                        run_id=run_id,
                        stage_name="script",
                        mode=mode,
                        attempt_fn=lambda: self.script_gen.run(_pt, run_id),
                        save_features_fn=save_script_features,
                        prior_corrections=total_corrections,
                        cumulative_risk=cumulative_risk,
                        topic=topic,
                    )
                    script_data = result
                    if script_data is None:
                        raise RuntimeError("Script generation failed after all retries.")

                elif stage_name == "audio":
                    attempt, result, corrections = self._run_stage(
                        db=db,
                        run_id=run_id,
                        stage_name="audio",
                        mode=mode,
                        attempt_fn=lambda: self.audio_synth.run(_sd, run_id),
                        save_features_fn=save_audio_features,
                        prior_corrections=total_corrections,
                        cumulative_risk=cumulative_risk,
                        topic=topic,
                    )

                elif stage_name == "visual":
                    attempt, result, corrections = self._run_stage(
                        db=db,
                        run_id=run_id,
                        stage_name="visual",
                        mode=mode,
                        attempt_fn=lambda: self.visual_gen.run(_sd, run_id, _vpc),
                        save_features_fn=save_visual_features,
                        prior_corrections=total_corrections,
                        cumulative_risk=cumulative_risk,
                        topic=topic,
                        provider=_vpc,
                    )

                elif stage_name == "video":
                    _vpc_video = vpc  # capture current value for lambda
                    attempt, result, corrections = self._run_stage(
                        db=db,
                        run_id=run_id,
                        stage_name="video",
                        mode=mode,
                        attempt_fn=lambda: self.video_asm.run(run_id, video_provider_choice=_vpc_video),
                        save_features_fn=save_video_features,
                        prior_corrections=total_corrections,
                        cumulative_risk=cumulative_risk,
                        topic=topic,
                    )
                    asm_result = result

                else:
                    continue

                total_corrections += corrections
                cumulative_risk = self._update_risk(attempt, cumulative_risk)

            # ── Finalise ──────────────────────────────────────────────────
            final_video_path = asm_result.get("output_path") if asm_result else None
            stored_video_path = run.final_video_path  # keep existing if video was skipped

            if final_video_path and Path(final_video_path).exists():
                import shutil as _shutil
                final_videos_dir = settings.DATA_DIR / "final_videos"
                final_videos_dir.mkdir(parents=True, exist_ok=True)
                dest = final_videos_dir / f"{run_id}.mp4"
                _shutil.copy2(final_video_path, dest)
                stored_video_path = str(dest)
                logger.info(f"[resume] Final video stored at: {stored_video_path}")

            run.status = "completed"
            run.completed_at = datetime.utcnow()
            run.final_video_path = stored_video_path
            run.total_corrections = total_corrections
            db.commit()
            logger.info(f"[resume] Run {run_id} completed.")
            return run_id

        except Exception as exc:
            stopped = _STOPPED in str(exc)
            if not stopped:
                logger.error(f"[resume] Run {run_id} failed: {exc}", exc_info=True)
            try:
                run = db.query(PipelineRun).get(run_id)
                if run and run.status not in ("stopped",):
                    run.status = "failed"
                    run.completed_at = datetime.utcnow()
                    db.commit()
            except Exception:
                pass
            if not stopped:
                raise
        finally:
            db.close()

    # ── Stage runner ───────────────────────────────────────────────────────

    def _run_stage(
        self,
        db: Session,
        run_id: str,
        stage_name: str,
        mode: str,
        attempt_fn,
        save_features_fn,
        prior_corrections: int,
        cumulative_risk: float,
        topic: str = "",
        provider: str = "ken_burns",
    ) -> tuple:
        """
        Run a single stage with retry logic and mode-based review routing.

        Returns:
            (latest_attempt, last_result, corrections_count)
        """
        last_attempt = None
        last_result = None
        corrections = 0

        for attempt_num in range(1, MAX_RETRIES + 1):
            attempt_id = str(uuid.uuid4())

            # ── Mark as queued until the stage slot is free ───────────────
            attempt = StageAttempt(
                id=attempt_id,
                run_id=run_id,
                stage_name=stage_name,
                attempt_number=attempt_num,
                mode=mode,
                status="queued",
            )
            db.add(attempt)
            db.commit()

            # ── Acquire stage slot (blocks until prior run exits) ─────────
            # ASSEMBLY LINE: only one run in each stage at a time.
            # Visual stage uses provider-aware semaphore (modal=5, ken_burns=1).
            acquire_stage(stage_name, run_id, provider=provider)
            attempt.status = "pending"
            db.commit()

            lock_held = True

            # ── Abort immediately if stopped while waiting for the slot ───
            if self._check_stopped(db, run_id):
                release_stage(stage_name, run_id, provider=provider)
                lock_held = False
                attempt.status = "failed"
                db.commit()
                raise RuntimeError(_STOPPED)

            try:
                # Execute the stage function
                result = attempt_fn()
                last_result = result

                # Save output path
                attempt.output_path = result.get("output_path")
                attempt.status = "completed"
                db.commit()

                # Save features
                features = result.get("features", {})
                try:
                    save_features_fn(db, attempt, features)
                except Exception as fe:
                    logger.warning(f"Feature save failed for {stage_name}: {fe}")

                # Update cross-stage features
                update_cross_stage_features(
                    db=db,
                    attempt=attempt,
                    prior_corrections=prior_corrections,
                    cumulative_risk=cumulative_risk,
                    is_fallback=(features.get("visual_provider") == "ken_burns"),
                )

                # ── Release slot BEFORE human review wait ─────────────────
                # CRITICAL: free the stage slot so the next run can enter
                # while this run waits for a human decision.
                release_stage(stage_name, run_id, provider=provider)
                lock_held = False

            except Exception as exc:
                if lock_held:
                    release_stage(stage_name, run_id, provider=provider)
                    lock_held = False
                logger.error(f"Stage {stage_name} attempt {attempt_num} failed: {exc}")
                attempt.status = "failed"
                db.commit()
                last_attempt = attempt
                # Permanent quota errors: no point retrying further attempts
                if "daily" in str(exc).lower() or "quota exhausted" in str(exc).lower():
                    raise
                continue

            last_attempt = attempt

            # ── Mode-based decision (lock already released) ───────────────
            decision = self._make_decision(db, attempt, mode, stage_name, topic)

            if decision == "accept":
                return last_attempt, last_result, corrections

            elif decision == "reject":
                corrections += 1
                attempt.status = "rejected"
                db.commit()
                logger.info(
                    f"Stage {stage_name} attempt {attempt_num} rejected. "
                    f"Retrying ({attempt_num}/{MAX_RETRIES})."
                )
                continue

        # All retries exhausted — if every attempt raised an exception
        # last_result is still None; abort the pipeline instead of silently
        # continuing to the next stage with no output.
        if last_result is None:
            raise RuntimeError(
                f"Stage '{stage_name}' failed on all {MAX_RETRIES} attempts. "
                "Aborting pipeline."
            )

        logger.warning(
            f"Stage {stage_name}: all {MAX_RETRIES} attempts exhausted after rejections. "
            "Using last accepted result."
        )
        return last_attempt, last_result, corrections

    # ── Decision routing ───────────────────────────────────────────────────

    def _make_decision(
        self,
        db: Session,
        attempt: StageAttempt,
        mode: str,
        stage_name: str,
        topic: str,
    ) -> str:
        """
        Route the accept/reject decision based on pipeline mode.

        Returns:
            'accept' or 'reject'
        """
        if mode == "A":
            return "accept"

        elif mode == "B":
            return self._wait_for_human_decision(db, attempt)

        elif mode == "C":
            return self._mode_c_decision(db, attempt, stage_name)

        return "accept"

    def _check_stopped(self, db: Session, run_id: str) -> bool:
        """Return True if the run has been stopped by the user."""
        run = db.query(PipelineRun).filter(PipelineRun.id == run_id).first()
        return bool(run and run.status == "stopped")

    def _wait_for_human_decision(
        self, db: Session, attempt: StageAttempt
    ) -> str:
        """
        Set attempt status to 'pending_review' and poll until human submits
        a decision or the timeout expires.

        Returns:
            'accept' or 'reject'
        """
        attempt.status = "pending_review"
        db.commit()
        logger.info(f"Waiting for human review of attempt {attempt.id}...")

        elapsed = 0
        while elapsed < REVIEW_TIMEOUT:
            time.sleep(POLL_INTERVAL)
            elapsed += POLL_INTERVAL

            db.refresh(attempt)
            if attempt.human_decision in ("accept", "reject"):
                logger.info(
                    f"Human decision received: {attempt.human_decision} "
                    f"for attempt {attempt.id}"
                )
                return attempt.human_decision

            # Check if run has been stopped by the user
            run_check = db.query(PipelineRun).filter(
                PipelineRun.id == attempt.run_id
            ).first()
            if run_check and run_check.status == "stopped":
                logger.info(f"Run {attempt.run_id} stopped by user during review.")
                raise RuntimeError("__PIPELINE_STOPPED__")

        # Timeout: auto-accept
        logger.warning(
            f"Human review timed out after {REVIEW_TIMEOUT}s for attempt {attempt.id}. "
            "Auto-accepting."
        )
        attempt.human_decision = "accept"
        attempt.reviewer_notes = "Auto-accepted: reviewer timeout"
        attempt.decision_timestamp = datetime.utcnow()
        db.commit()
        return "accept"

    def _mode_c_decision(
        self, db: Session, attempt: StageAttempt, stage_name: str
    ) -> str:
        """
        Apply per-stage Mode C logic via the ML predictor.

        Calls should_request_human_review() from backend.ml.predictor, which
        reads classifier_config.json and runs the appropriate pkl model.
        The resulting rejection_probability is persisted to the DB for every
        Mode C attempt regardless of the final routing decision.
        """
        from backend.ml.predictor import should_request_human_review

        # Build the feature dict from the ORM attempt object
        features = self._attempt_to_feature_row(attempt, stage_name)

        needs_review, probability = should_request_human_review(stage_name, features)

        # Persist predicted probability for dissertation analysis
        attempt.rejection_probability = probability
        db.commit()

        if needs_review:
            logger.info(
                f"Mode C: stage={stage_name}  probability={probability:.3f}"
                "  -> requesting human review"
            )
            return self._wait_for_human_decision(db, attempt)
        else:
            logger.info(
                f"Mode C: stage={stage_name}  probability={probability:.3f}"
                "  -> auto-accept"
            )
            attempt.human_decision = "accept"
            attempt.decision_timestamp = datetime.utcnow()
            db.commit()
            return "accept"

    @staticmethod
    def _attempt_to_feature_row(attempt: StageAttempt, stage_name: str) -> dict:
        """Convert a StageAttempt ORM object to a feature dict for ML inference."""
        feature_map = {
            "script": [
                "readability_score", "lexical_diversity", "prompt_coverage",
                "sentence_redundancy", "entity_consistency", "topic_coherence",
                "factual_conflict_flag", "prompt_ambiguity",
                "prior_stage_corrections", "cumulative_risk_score",
                "api_retry_count",
            ],
            "audio": [
                "phoneme_error_rate", "silence_ratio", "speaking_rate_variance",
                "energy_variance", "tts_word_count_match",
                "prior_stage_corrections", "cumulative_risk_score",
                "api_retry_count",
            ],
            "visual": [
                "clip_similarity", "aesthetic_score", "blur_score",
                "object_match_score", "colour_tone_match",
                "prior_stage_corrections", "cumulative_risk_score",
                "api_retry_count",
            ],
            "video": [
                "av_sync_error_ms", "transition_smoothness", "duration_deviation_s",
                "prior_stage_corrections", "cumulative_risk_score",
                "api_retry_count",
            ],
        }
        cols = feature_map.get(stage_name, [])
        return {col: getattr(attempt, col, None) for col in cols}

    # ── Utilities ──────────────────────────────────────────────────────────

    @staticmethod
    def _stage_output_exists(run_id: str, stage_name: str) -> bool:
        """
        Return True when a previous attempt already left a usable output on disk.
        Used by resume_pipeline to skip stages whose outputs survived the stop.
        """
        out_dir = settings.OUTPUT_DIR / run_id
        if stage_name == "script":
            return (out_dir / "script.json").exists()
        elif stage_name == "audio":
            return (out_dir / "audio.mp3").exists() or (out_dir / "audio.wav").exists()
        elif stage_name == "visual":
            scenes_dir = out_dir / "scenes"
            return scenes_dir.is_dir() and any(scenes_dir.glob("*.mp4"))
        elif stage_name == "video":
            # Video assembly is cheap and always re-runs to ensure a clean merge.
            return False
        return False

    @staticmethod
    def _load_script_from_disk(run_id: str) -> Optional[dict]:
        """Load script.json from disk and wrap it in the same dict shape as Stage 1 returns."""
        script_path = settings.OUTPUT_DIR / run_id / "script.json"
        if not script_path.exists():
            return None
        raw = json.loads(script_path.read_text(encoding="utf-8"))
        return {
            "output_path": str(script_path),
            "scenes": raw.get("scenes", []),
            "features": {},
        }

    @staticmethod
    def _load_prompt(prompt_id: int) -> tuple:
        """Load prompt text and topic from data/prompts.json."""
        prompts_file = settings.PROMPTS_FILE
        prompts = json.loads(prompts_file.read_text(encoding="utf-8"))
        for p in prompts:
            if p["id"] == prompt_id:
                return p["prompt"], p["topic"]
        raise ValueError(f"Prompt ID {prompt_id} not found in prompts.json")

    @staticmethod
    def _update_risk(attempt: Optional[StageAttempt], current_risk: float) -> float:
        """
        Add the latest stage's cumulative_risk_score to the running total.
        Returns updated cumulative risk.
        """
        if attempt and attempt.cumulative_risk_score is not None:
            return current_risk + attempt.cumulative_risk_score
        return current_risk
