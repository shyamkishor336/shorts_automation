"""
Stage 1 — Script Generation using Gemini 2.5 Flash API.
Generates an educational YouTube Shorts script from a text prompt.
Scene count is read from SCENE_COUNT env var (default 6).
"""

import json
import logging
import time
import warnings
from pathlib import Path
from typing import Dict, Any

with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    import google.generativeai as genai

from backend.config import settings
from backend.features import script_features

logger = logging.getLogger(__name__)


def _build_system_prompt(scene_count: int) -> str:
    """Build the Gemini system prompt for the configured scene count."""
    total_seconds = scene_count * settings.SCENE_DURATION_SECONDS
    return (
        "You are an educational YouTube Shorts scriptwriter. "
        f"Write a script for a {total_seconds}-second educational video on the given topic. "
        f"The script must have exactly {scene_count} scenes. "
        "For each scene provide:\n"
        f"  - scene_number (1-{scene_count})\n"
        "  - narration (the spoken text, max 20 words)\n"
        "  - visual_prompt (description of what to show visually, max 30 words, "
        "detailed enough for an AI video generator)\n"
        "Return valid JSON only. No markdown. No explanation.\n"
        '{"scenes": [{"scene_number": 1, "narration": "...", "visual_prompt": "..."}, ...]}'
    )

MAX_RETRIES = 3
RETRY_BACKOFF = 5


class Stage1ScriptGenerator:
    """
    Generates an educational script using the Gemini 2.5 Flash API.
    Scene count is controlled by SCENE_COUNT env var.
    """

    def __init__(self) -> None:
        genai.configure(api_key=settings.GEMINI_API_KEY)
        self.scene_count = settings.SCENE_COUNT
        self.model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            system_instruction=_build_system_prompt(self.scene_count),
        )

    def run(self, prompt: str, run_id: str) -> Dict[str, Any]:
        """
        Generate a script for the given prompt and save to disk.

        Args:
            prompt: The user-facing educational topic prompt.
            run_id: Pipeline run ID (used for output directory).

        Returns:
            Dict with 'script' (parsed JSON), 'output_path', and 'features'.

        Raises:
            RuntimeError: If Gemini fails after all retries.
        """
        output_dir = settings.OUTPUT_DIR / run_id
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "script.json"

        script_data = self._call_gemini_with_retry(prompt)

        # Add timing estimates based on character count
        self._add_timing(script_data)

        # Save to disk
        output_path.write_text(
            json.dumps(script_data, indent=2), encoding="utf-8"
        )
        logger.info(f"Script saved to {output_path}")

        # Extract features
        scenes = script_data.get("scenes", [])
        features = script_features.extract_all(prompt, scenes)
        logger.info(f"Script features extracted for run {run_id}")

        return {
            "script": script_data,
            "output_path": str(output_path),
            "features": features,
        }

    # ── Private helpers ────────────────────────────────────────────────────

    def _call_gemini_with_retry(self, prompt: str) -> dict:
        """Call Gemini API with exponential backoff, parse JSON response."""
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                logger.info(
                    f"Gemini: attempt {attempt}/{MAX_RETRIES} for: {prompt[:60]}..."
                )
                response = self.model.generate_content(prompt)
                raw_text = response.text.strip()

                # Strip markdown code fences if present
                if raw_text.startswith("```"):
                    lines = raw_text.splitlines()
                    raw_text = "\n".join(
                        line for line in lines
                        if not line.startswith("```")
                    )

                script_data = json.loads(raw_text)

                # Validate structure
                actual = len(script_data.get("scenes", []))
                if "scenes" not in script_data or actual == 0:
                    raise ValueError(f"No scenes found in Gemini response")
                if actual != self.scene_count:
                    logger.warning(
                        f"Expected {self.scene_count} scenes, got {actual}. "
                        "Truncating/padding is handled downstream."
                    )
                    raise ValueError(
                        f"Expected {self.scene_count} scenes, got {actual}"
                    )

                return script_data

            except (json.JSONDecodeError, ValueError) as exc:
                logger.warning(f"Gemini: parse error on attempt {attempt}: {exc}")
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_BACKOFF)
                else:
                    raise RuntimeError(
                        f"Gemini failed to produce valid JSON after {MAX_RETRIES} attempts."
                    ) from exc

            except Exception as exc:
                exc_str = str(exc)
                logger.warning(f"Gemini: API error on attempt {attempt}: {exc_str[:200]}")

                # Respect the retry_delay hint in 429 responses
                wait = RETRY_BACKOFF * (2 ** (attempt - 1))
                if "429" in exc_str or "quota" in exc_str.lower():
                    # Parse suggested retry delay from the error message if present
                    import re
                    match = re.search(r"retry_delay\s*\{\s*seconds:\s*(\d+)", exc_str)
                    if match:
                        wait = int(match.group(1)) + 5  # small buffer
                    # Daily quota exhausted — no point retrying immediately
                    if "PerDay" in exc_str:
                        raise RuntimeError(
                            "Gemini daily free-tier quota exhausted. "
                            "Wait until tomorrow or enable billing at "
                            "https://ai.dev/rate-limit"
                        ) from exc

                if attempt < MAX_RETRIES:
                    logger.info(f"Gemini: waiting {wait}s before retry...")
                    time.sleep(wait)
                else:
                    raise RuntimeError(
                        f"Gemini API failed after {MAX_RETRIES} attempts: {exc}"
                    ) from exc

        raise RuntimeError("Gemini: unreachable code reached.")

    @staticmethod
    def _add_timing(script_data: dict) -> None:
        """
        Estimate per-scene audio timing based on character count.
        Assumes approx. 15 characters per second of speech.
        Adds 'estimated_start_s' and 'estimated_duration_s' to each scene.
        """
        chars_per_second = 15.0
        cursor = 0.0

        for scene in script_data.get("scenes", []):
            narration = scene.get("narration", "")
            duration = max(len(narration) / chars_per_second, 1.0)
            scene["estimated_start_s"] = round(cursor, 2)
            scene["estimated_duration_s"] = round(duration, 2)
            cursor += duration
