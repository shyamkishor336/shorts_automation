"""
Kaggle Provider — Async job queue pattern for Kaggle GPU notebook execution.
Kaggle is NOT a real-time API. Calling generate() enqueues a job in
data/kaggle_queue.json. The actual inference happens when the user manually
runs kaggle_cogvideox_notebook.py on Kaggle with GPU P100.
"""

import json
import logging
import uuid
from pathlib import Path
from datetime import datetime
from typing import Optional

from backend.config import settings
from backend.providers.exceptions import ProviderUnavailableError

logger = logging.getLogger(__name__)


class KaggleProvider:
    """
    Adds video generation jobs to the Kaggle queue.
    Jobs are processed asynchronously by the Kaggle notebook.
    """

    def __init__(self) -> None:
        self.name = "kaggle"
        self.queue_file: Path = settings.KAGGLE_QUEUE_FILE

    def is_available(self) -> bool:
        """Kaggle is always available for queuing (no credentials needed to queue)."""
        return True

    def generate(
        self,
        prompt: str,
        output_path: Path,
        run_id: str = "",
        scene_number: int = 0,
        duration: int = settings.SCENE_DURATION_SECONDS,
    ) -> dict:
        """
        Enqueue a video generation job for Kaggle batch processing.

        Args:
            prompt: Visual description for the video.
            output_path: Where the result should be saved when complete.
            run_id: Pipeline run ID for tracking.
            scene_number: Scene index.
            duration: Clip duration in seconds.

        Returns:
            Dict with job_id and status='queued'.
        """
        output_path = Path(output_path)
        job_id = str(uuid.uuid4())

        job = {
            "job_id": job_id,
            "prompt": prompt,
            "output_filename": output_path.name,
            "output_path": str(output_path),
            "run_id": run_id,
            "scene_number": scene_number,
            "duration": duration,
            "status": "queued",
            "queued_at": datetime.utcnow().isoformat(),
            "completed_at": None,
        }

        self._append_to_queue(job)
        logger.info(f"Kaggle: job {job_id} queued for scene {scene_number}.")

        return {"job_id": job_id, "status": "queued", "provider": "kaggle"}

    def mark_complete(self, job_id: str, result_path: Optional[str] = None) -> bool:
        """
        Mark a queued job as complete (called when batch results are returned).

        Args:
            job_id: The job ID to mark complete.
            result_path: Optional path to the generated video file.

        Returns:
            True if the job was found and updated.
        """
        queue = self._load_queue()
        updated = False

        for job in queue:
            if job["job_id"] == job_id:
                job["status"] = "completed"
                job["completed_at"] = datetime.utcnow().isoformat()
                if result_path:
                    job["result_path"] = result_path
                updated = True
                break

        if updated:
            self._save_queue(queue)
            logger.info(f"Kaggle: job {job_id} marked complete.")
        else:
            logger.warning(f"Kaggle: job {job_id} not found in queue.")

        return updated

    def get_pending_jobs(self) -> list:
        """Return all jobs with status='queued'."""
        return [j for j in self._load_queue() if j.get("status") == "queued"]

    # ── Private helpers ────────────────────────────────────────────────────

    def _load_queue(self) -> list:
        """Load the queue JSON file."""
        try:
            if self.queue_file.exists():
                return json.loads(self.queue_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.error(f"Failed to load Kaggle queue: {exc}")
        return []

    def _save_queue(self, queue: list) -> None:
        """Save the queue back to JSON."""
        try:
            self.queue_file.write_text(
                json.dumps(queue, indent=2), encoding="utf-8"
            )
        except OSError as exc:
            raise ProviderUnavailableError(f"Cannot write Kaggle queue: {exc}") from exc

    def _append_to_queue(self, job: dict) -> None:
        """Append a new job to the queue file."""
        queue = self._load_queue()
        queue.append(job)
        self._save_queue(queue)
