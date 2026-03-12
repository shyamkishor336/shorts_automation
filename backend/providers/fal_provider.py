"""
fal.ai Provider — CogVideoX-5b video generation via fal_client library.
Uses the fal-ai/cogvideox-5b model on the fal.ai platform.
"""

import logging
import time
from pathlib import Path

import httpx

from backend.config import settings
from backend.providers.exceptions import (
    CreditExhaustedError,
    QuotaExceededError,
    ProviderUnavailableError,
)

logger = logging.getLogger(__name__)

FAL_MODEL = "fal-ai/cogvideox-5b"
MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 5


class FalProvider:
    """
    Generates video clips using fal.ai's CogVideoX-5b model.
    Requires FAL_KEY environment variable.
    """

    def __init__(self) -> None:
        self.name = "fal"
        self._configure_fal()

    def _configure_fal(self) -> None:
        """Set fal API key from environment."""
        import os
        os.environ["FAL_KEY"] = settings.FAL_KEY

    def is_available(self) -> bool:
        """Return True if the fal API key is configured."""
        return bool(settings.FAL_KEY)

    def generate(
        self,
        prompt: str,
        output_path: Path,
        duration: int = settings.SCENE_DURATION_SECONDS,
    ) -> Path:
        """
        Generate a video clip using fal.ai CogVideoX-5b.

        Args:
            prompt: Visual description for the video.
            output_path: Where to save the .mp4 file.
            duration: Clip duration in seconds.

        Returns:
            Path to the saved .mp4 file.

        Raises:
            CreditExhaustedError: If fal returns a credit/quota exhaustion error.
            QuotaExceededError: If rate-limited after retries.
            ProviderUnavailableError: On network/server errors.
        """
        import fal_client  # imported here to avoid hard dependency at startup

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                logger.info(
                    f"fal.ai: attempt {attempt}/{MAX_RETRIES} for prompt: {prompt[:60]}..."
                )
                result = fal_client.run(
                    FAL_MODEL,
                    arguments={
                        "prompt": prompt,
                        "num_frames": duration * 8,  # 8fps
                        "guidance_scale": 6.0,
                        "num_inference_steps": 25,
                    },
                )

                video_url: str = result["video"]["url"]
                self._download_video(video_url, output_path)
                logger.info(f"fal.ai: video saved to {output_path}")
                return output_path

            except Exception as exc:
                exc_str = str(exc).lower()

                if any(kw in exc_str for kw in ("credit", "quota", "402")):
                    raise CreditExhaustedError(
                        f"fal.ai credits exhausted: {exc}"
                    ) from exc

                if "429" in exc_str or "rate limit" in exc_str:
                    if attempt < MAX_RETRIES:
                        wait = RETRY_BACKOFF_BASE * (2 ** (attempt - 1))
                        logger.warning(f"fal.ai rate-limited. Waiting {wait}s.")
                        time.sleep(wait)
                        continue
                    raise QuotaExceededError(
                        f"fal.ai rate limit after {MAX_RETRIES} retries."
                    ) from exc

                if attempt < MAX_RETRIES:
                    wait = RETRY_BACKOFF_BASE * (2 ** (attempt - 1))
                    logger.warning(f"fal.ai error: {exc}. Retrying in {wait}s.")
                    time.sleep(wait)
                else:
                    raise ProviderUnavailableError(
                        f"fal.ai failed after {MAX_RETRIES} attempts: {exc}"
                    ) from exc

        raise ProviderUnavailableError("fal.ai: all retries exhausted.")

    @staticmethod
    def _download_video(url: str, output_path: Path) -> None:
        """Download video bytes from a URL to a local file."""
        with httpx.Client(timeout=300) as client:
            response = client.get(url)
            response.raise_for_status()
            output_path.write_bytes(response.content)
