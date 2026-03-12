"""
Modal.com Provider — Real-time CogVideoX-2b video generation via Modal web endpoint.
Calls the deployed modal_cogvideox_endpoint.py web endpoint.
"""

import json
import logging
import subprocess
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

MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 5  # seconds
REQUEST_TIMEOUT = 180   # 3 minutes per scene


class ModalProvider:
    """
    Calls the Modal.com web endpoint to generate video clips using CogVideoX-2b.
    The endpoint is deployed separately via: modal deploy modal_cogvideox_endpoint.py
    """

    def __init__(self) -> None:
        self.name = "modal"
        self.endpoint_url = settings.MODAL_ENDPOINT_URL

    def is_available(self) -> bool:
        """Return True if the Modal endpoint URL is configured."""
        return bool(self.endpoint_url)

    def generate(
        self,
        prompt: str,
        output_path: Path,
        duration: int = settings.SCENE_DURATION_SECONDS,
    ) -> Path:
        """
        Generate a video clip via the Modal web endpoint.

        Args:
            prompt: Visual description for the video.
            output_path: Where to save the resulting .mp4 file.
            duration: Clip duration in seconds.

        Returns:
            Path to the saved .mp4 file.

        Raises:
            CreditExhaustedError: If the endpoint returns HTTP 402.
            QuotaExceededError: If rate-limited after all retries.
            ProviderUnavailableError: If the endpoint is unreachable.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {"prompt": prompt, "duration": duration}

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                logger.info(
                    f"Modal: attempt {attempt}/{MAX_RETRIES} for prompt: {prompt[:60]}..."
                )
                with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
                    response = client.post(self.endpoint_url, json=payload)

                if response.status_code == 200:
                    output_path.write_bytes(response.content)
                    logger.info(f"Modal: clip saved to {output_path}")
                    self._log_clip_dimensions(output_path)
                    return output_path

                elif response.status_code == 402:
                    raise CreditExhaustedError(
                        f"Modal credits exhausted (HTTP 402). Response: {response.text}"
                    )

                elif response.status_code == 429:
                    if attempt < MAX_RETRIES:
                        wait = RETRY_BACKOFF_BASE * (2 ** (attempt - 1))
                        logger.warning(
                            f"Modal rate-limited (429). Waiting {wait}s before retry."
                        )
                        time.sleep(wait)
                    else:
                        raise QuotaExceededError(
                            f"Modal rate limit hit after {MAX_RETRIES} retries."
                        )

                else:
                    error_msg = (
                        f"Modal returned unexpected status {response.status_code}: "
                        f"{response.text[:500]}"
                    )
                    print(f"[MODAL ERROR] {error_msg}", flush=True)
                    raise ProviderUnavailableError(error_msg)

            except (httpx.TimeoutException, httpx.ConnectError) as exc:
                error_msg = f"Modal network error (attempt {attempt}/{MAX_RETRIES}): {exc}"
                print(f"[MODAL ERROR] {error_msg}", flush=True)
                if attempt < MAX_RETRIES:
                    wait = RETRY_BACKOFF_BASE * (2 ** (attempt - 1))
                    logger.warning(f"{error_msg}. Retrying in {wait}s.")
                    time.sleep(wait)
                else:
                    raise ProviderUnavailableError(
                        f"Modal unreachable after {MAX_RETRIES} attempts: {exc}"
                    ) from exc

        raise ProviderUnavailableError("Modal: all retries exhausted.")

    @staticmethod
    def _log_clip_dimensions(clip_path: Path) -> None:
        """
        Use ffprobe to read the actual width/height of the saved clip and log it.
        This helps diagnose aspect-ratio or cropping issues in Stage 4.
        """
        try:
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v", "quiet",
                    "-print_format", "json",
                    "-show_streams",
                    str(clip_path),
                ],
                capture_output=True,
                timeout=30,
            )
            if result.returncode != 0:
                logger.warning(
                    f"Modal ffprobe failed for {clip_path.name}: "
                    f"{result.stderr.decode(errors='replace')[:200]}"
                )
                return

            info = json.loads(result.stdout)
            for stream in info.get("streams", []):
                if stream.get("codec_type") == "video":
                    w = stream.get("width", "?")
                    h = stream.get("height", "?")
                    dur = stream.get("duration", "?")
                    codec = stream.get("codec_name", "?")
                    logger.info(
                        f"Modal clip dimensions: {w}x{h}  codec={codec}  duration={dur}s"
                        f"  file={clip_path.name}"
                    )
                    return
            logger.warning(f"Modal ffprobe: no video stream found in {clip_path.name}")
        except Exception as exc:
            logger.warning(f"Modal _log_clip_dimensions failed: {exc}")
