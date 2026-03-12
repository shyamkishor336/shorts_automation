"""
Hugging Face Inference Provider — CogVideoX-5b via huggingface_hub InferenceClient.
Used as overflow provider; unlimited but slow.
"""

import logging
import time
from pathlib import Path
from typing import Optional

from backend.config import settings
from backend.providers.exceptions import (
    QuotaExceededError,
    ProviderUnavailableError,
)

logger = logging.getLogger(__name__)

HF_MODEL = "THUDM/CogVideoX-5b"
MAX_RETRIES = 2
RATE_LIMIT_WAIT = 60  # seconds to wait when rate-limited


class HuggingFaceProvider:
    """
    Generates video clips via Hugging Face Inference API using CogVideoX-5b.
    Rate-limited requests are retried once after a 60-second wait.
    """

    def __init__(self) -> None:
        self.name = "hf"
        self._client: Optional[object] = None

    def _get_client(self):
        """Lazily initialise the HF InferenceClient."""
        if self._client is None:
            from huggingface_hub import InferenceClient
            self._client = InferenceClient(token=settings.HUGGINGFACE_TOKEN)
        return self._client

    def is_available(self) -> bool:
        """Return True if the HF token is configured."""
        return bool(settings.HUGGINGFACE_TOKEN)

    def generate(
        self,
        prompt: str,
        output_path: Path,
        duration: int = settings.SCENE_DURATION_SECONDS,
    ) -> Path:
        """
        Generate a video clip using HF Inference API.

        Args:
            prompt: Visual description for the video.
            output_path: Where to save the .mp4 file.
            duration: Clip duration in seconds.

        Returns:
            Path to the saved .mp4 file.

        Raises:
            QuotaExceededError: If rate-limited after one retry.
            ProviderUnavailableError: On unrecoverable errors.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        client = self._get_client()

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                logger.info(
                    f"HuggingFace: attempt {attempt}/{MAX_RETRIES} "
                    f"for prompt: {prompt[:60]}..."
                )
                # HF InferenceClient video generation
                video_bytes = client.text_to_video(  # type: ignore[attr-defined]
                    prompt,
                    model=HF_MODEL,
                )

                # Handle both bytes and file-like objects
                if hasattr(video_bytes, "read"):
                    data = video_bytes.read()
                else:
                    data = bytes(video_bytes)

                output_path.write_bytes(data)
                logger.info(f"HuggingFace: video saved to {output_path}")
                return output_path

            except Exception as exc:
                exc_str = str(exc).lower()

                if "429" in exc_str or "rate limit" in exc_str or "too many" in exc_str:
                    if attempt < MAX_RETRIES:
                        logger.warning(
                            f"HuggingFace rate-limited (429). "
                            f"Waiting {RATE_LIMIT_WAIT}s then retrying once."
                        )
                        time.sleep(RATE_LIMIT_WAIT)
                        continue
                    # Do NOT mark exhausted — just skip for now
                    raise QuotaExceededError(
                        f"HuggingFace rate-limited after retry: {exc}"
                    ) from exc

                raise ProviderUnavailableError(
                    f"HuggingFace generation failed: {exc}"
                ) from exc

        raise ProviderUnavailableError("HuggingFace: all retries exhausted.")
