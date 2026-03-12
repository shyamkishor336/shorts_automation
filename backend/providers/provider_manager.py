"""
Provider Manager — Two-provider system: Modal (AI video) or Ken Burns (local FFmpeg).
The caller specifies which provider to use via video_provider_choice.
"""

import logging
from pathlib import Path
from typing import Optional

from backend.config import settings
from backend.providers.ken_burns_provider import KenBurnsProvider
from backend.providers.modal_provider import ModalProvider

logger = logging.getLogger(__name__)


class ProviderManager:
    """
    Manages Modal and Ken Burns video providers.
    The provider is selected explicitly via video_provider_choice — no fallback chain.
    Ken Burns is always used as the final safety net if Modal fails.
    """

    def __init__(self) -> None:
        self.providers = {
            "modal": ModalProvider(),
            "ken_burns": KenBurnsProvider(),
        }

    # ── Public API ─────────────────────────────────────────────────────────

    def generate_video(
        self,
        prompt: str,
        output_path: Path,
        video_provider_choice: str = "ken_burns",
        run_id: str = "",
        scene_number: int = 0,
        source_image_path: Optional[Path] = None,
        topic: str = "",
        duration: int = settings.SCENE_DURATION_SECONDS,
    ) -> dict:
        """
        Generate a video clip using the specified provider.

        Args:
            prompt: Visual description for the video.
            output_path: Where to save the resulting .mp4 file.
            video_provider_choice: 'modal' or 'ken_burns'.
            run_id: Pipeline run ID (for logging).
            scene_number: Scene number (0-based).
            source_image_path: FLUX still image path (for Ken Burns).
            topic: Topic text for Ken Burns placeholder.
            duration: Clip duration in seconds.

        Returns:
            Dict with 'provider', 'output_path', and 'status'.
        """
        # Normalise "modal_ai" → "modal"; anything unrecognised → "ken_burns"
        if video_provider_choice in ("modal", "modal_ai"):
            choice = "modal"
        elif video_provider_choice == "ken_burns":
            choice = "ken_burns"
        else:
            logger.error(
                f"PROVIDER FALLBACK: unrecognised video_provider_choice={video_provider_choice!r}. "
                "Falling back to Ken Burns."
            )
            choice = "ken_burns"

        if choice == "modal":
            modal = self.providers["modal"]
            if modal.is_available():
                try:
                    saved_path = modal.generate(
                        prompt=prompt,
                        output_path=output_path,
                        duration=duration,
                    )
                    logger.info(f"Modal generated scene {scene_number}: {saved_path}")
                    return {
                        "provider": "modal",
                        "status": "completed",
                        "output_path": str(saved_path),
                    }
                except Exception as exc:
                    logger.error(
                        f"PROVIDER FALLBACK: Modal failed for scene {scene_number} "
                        f"(run {run_id}): {exc}. Falling back to Ken Burns."
                    )
            else:
                logger.error(
                    f"PROVIDER FALLBACK: Modal not available — MODAL_ENDPOINT_URL is not set or empty. "
                    f"Falling back to Ken Burns for scene {scene_number} (run {run_id})."
                )

        # Ken Burns — either by explicit choice or as fallback after Modal failure
        if choice == "modal":
            logger.error(
                f"PROVIDER FALLBACK: Using Ken Burns as fallback for scene {scene_number} (run {run_id})."
            )
        return self._call_ken_burns(prompt, output_path, source_image_path, topic, duration)

    def get_budget_status(self) -> dict:
        """Return a simple status dict for Modal and Ken Burns."""
        modal_available = self.providers["modal"].is_available()
        return {
            "modal": {
                "allocated": 9999,
                "used": 0,
                "exhausted": not modal_available,
            },
            "ken_burns": {
                "allocated": 9999,
                "used": 0,
                "exhausted": False,
            },
        }

    def reset_provider(self, provider_name: str) -> bool:
        """No-op for compatibility — providers don't track exhausted state here."""
        return provider_name in self.providers

    # ── Private helpers ────────────────────────────────────────────────────

    def _call_ken_burns(
        self,
        prompt: str,
        output_path: Path,
        source_image_path: Optional[Path],
        topic: str,
        duration: int,
    ) -> dict:
        """Generate a Ken Burns video clip."""
        kb = self.providers["ken_burns"]
        saved_path = kb.generate(
            prompt=prompt,
            output_path=output_path,
            source_image_path=source_image_path,
            topic=topic,
            duration=duration,
        )
        return {
            "provider": "ken_burns",
            "status": "completed",
            "output_path": str(saved_path),
        }
