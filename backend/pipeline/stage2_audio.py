"""
Stage 2 — Audio Synthesis using edge-tts (Microsoft, offline).
Generates narration audio for all 8 scenes and saves as audio.wav.

Runs `sys.executable -m edge_tts` so we always use the correct Python
environment. Audio is captured from stdout as raw bytes (no --write-media
flag, no file-path issues). The MP3 bytes are then piped through FFmpeg
stdin/stdout for conversion to 16 kHz mono WAV.
"""

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, List

from backend.config import settings
from backend.features import audio_features

SCENE_COUNT = settings.SCENE_COUNT

logger = logging.getLogger(__name__)


class Stage2AudioSynthesizer:
    """
    Synthesises speech audio from the script narrations using edge-tts.
    No API key required — uses Microsoft Edge TTS offline.
    """

    def __init__(self) -> None:
        self.voice = settings.TTS_VOICE

    def run(self, script_data: dict, run_id: str) -> Dict[str, Any]:
        """
        Generate audio for the full script and extract quality features.

        Args:
            script_data: Parsed script JSON with 'scenes' list.
            run_id: Pipeline run ID (for output directory).

        Returns:
            Dict with 'output_path' (WAV) and 'features'.

        Raises:
            RuntimeError: If audio generation or conversion fails.
        """
        output_dir = settings.OUTPUT_DIR / run_id
        output_dir.mkdir(parents=True, exist_ok=True)
        mp3_path = output_dir / "audio.mp3"
        wav_path = output_dir / "audio.wav"

        # script_data is the full return value from Stage1ScriptGenerator.run():
        # {"script": {"scenes": [...]}, "output_path": "...", "features": {...}}
        # The scenes live under the nested "script" key.
        inner = script_data.get("script", script_data)  # fall back to root if already unwrapped
        scenes = inner.get("scenes", [])
        if len(scenes) > SCENE_COUNT:
            logger.warning(
                f"Script has {len(scenes)} scenes but SCENE_COUNT={SCENE_COUNT}. "
                f"Truncating to {SCENE_COUNT} for audio."
            )
            scenes = scenes[:SCENE_COUNT]

        logger.info(f"script_data keys: {list(script_data.keys())}")
        logger.info(f"scene count: {len(scenes)}")
        for i, sc in enumerate(scenes):
            logger.info(f"  scene[{i}] keys={list(sc.keys())}  narration_len={len(sc.get('narration', ''))}")

        full_narration = self._build_narration(scenes)
        logger.info(f"full_narration length: {len(full_narration)}")
        logger.info(f"full_narration preview: {full_narration[:200]!r}")

        if not full_narration or len(full_narration.strip()) < 10:
            raise RuntimeError(
                f"Script text is empty or too short for audio generation "
                f"(length={len(full_narration)}). "
                f"script_data had {len(scenes)} scene(s)."
            )

        # Synthesise — audio captured from stdout, written to mp3_path
        mp3_bytes = self._synthesise(full_narration)
        logger.info(f"edge-tts returned {len(mp3_bytes):,} bytes")

        if len(mp3_bytes) < 5120:
            raise RuntimeError(
                f"Synthesised audio is too small ({len(mp3_bytes)} bytes) — "
                "edge-tts produced no audio."
            )
        mp3_path.write_bytes(mp3_bytes)

        # Convert MP3 bytes → WAV via FFmpeg stdin/stdout pipes
        self._convert_to_wav(mp3_bytes, wav_path)
        logger.info(f"Converted to WAV: {wav_path}")

        # Validate the WAV (size, ffprobe readability, duration)
        self._validate_audio(wav_path)
        logger.info(f"Audio validated: {wav_path}")

        # Remove leading/trailing silence and normalise loudness
        wav_path = self._clean_audio(wav_path)
        logger.info(f"Audio cleaned (silence removed + loudnorm): {wav_path}")

        # Extract features from the clean WAV
        features = audio_features.extract_all(
            audio_path=wav_path,
            original_text=full_narration,
            scenes=scenes,
        )
        logger.info(f"Audio features extracted for run {run_id}")

        return {
            "output_path": str(wav_path),
            "features": features,
        }

    # ── Private helpers ────────────────────────────────────────────────────

    @staticmethod
    def _build_narration(scenes: List[dict]) -> str:
        """Concatenate all scene narrations with a pause between each."""
        parts = []
        for scene in scenes:
            narration = scene.get("narration", "").strip()
            if narration:
                parts.append(narration)
        return " ... ".join(parts)

    def _synthesise(self, text: str) -> bytes:
        """
        Run edge-tts via `python -m edge_tts` and capture MP3 bytes from stdout.

        Using sys.executable ensures we use the same Python environment as the
        server. Not passing --write-media avoids all Windows file-path issues —
        edge-tts writes audio to stdout when no output file is specified.
        """
        result = subprocess.run(
            [
                sys.executable, "-m", "edge_tts",
                "--voice", self.voice,
                "--text", text,
            ],
            capture_output=True,   # stdout = MP3 bytes, stderr = log/subtitles
            timeout=120,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"edge-tts failed (rc={result.returncode}): "
                f"{result.stderr.decode(errors='replace')[:300]}"
            )
        return result.stdout

    @staticmethod
    def _convert_to_wav(mp3_bytes: bytes, wav_path: Path) -> None:
        """
        Convert MP3 bytes to 16 kHz mono WAV using FFmpeg stdin/stdout pipes.
        No file paths appear in the command, avoiding Windows space-in-path issues.
        """
        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", "pipe:0",
                "-ar", "16000",
                "-ac", "1",
                "-f", "wav",
                "pipe:1",
            ],
            input=mp3_bytes,
            capture_output=True,
            timeout=60,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"FFmpeg MP3→WAV conversion failed (rc={result.returncode}): "
                f"{result.stderr.decode(errors='replace')[:300]}"
            )
        if not result.stdout:
            raise RuntimeError("FFmpeg produced 0 WAV bytes — conversion failed silently.")
        wav_path.write_bytes(result.stdout)

    @staticmethod
    def _get_duration(audio_path: Path) -> float:
        """Return audio duration in seconds via ffprobe, or 0.0 on failure."""
        try:
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v", "quiet",
                    "-print_format", "json",
                    "-show_format",
                    str(audio_path),
                ],
                capture_output=True,
                timeout=30,
            )
            if result.returncode == 0:
                info = json.loads(result.stdout)
                return float(info.get("format", {}).get("duration", 0))
        except Exception:
            pass
        return 0.0

    @staticmethod
    def _clean_audio(wav_path: Path) -> Path:
        """
        Remove leading/trailing silence and normalise loudness with FFmpeg.

        Uses:
          - silenceremove: strips silence at start (and end via areverse trick)
          - loudnorm: EBU R128 loudness normalisation (-23 LUFS target)

        Falls back to the original WAV if FFmpeg fails for any reason.
        Returns the path to the cleaned WAV (same path, overwritten in-place).
        """
        original_dur = Stage2AudioSynthesizer._get_duration(wav_path)
        logger.info(f"Audio before cleaning: duration={original_dur:.2f}s  path={wav_path.name}")

        cleaned_path = wav_path.parent / "audio_cleaned.wav"
        try:
            result = subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-i", str(wav_path),
                    "-af", (
                        # Remove silence at start (stop_periods=-1 = also remove internal gaps)
                        "silenceremove=start_periods=1:start_silence=0.03:start_threshold=-50dB"
                        ":stop_periods=-1:stop_silence=0.03:stop_threshold=-50dB,"
                        # Loudness normalisation to -23 LUFS (EBU R128)
                        "loudnorm=I=-23:TP=-2:LRA=11"
                    ),
                    "-ar", "16000",
                    "-ac", "1",
                    str(cleaned_path),
                ],
                capture_output=True,
                timeout=120,
            )

            if result.returncode != 0:
                logger.warning(
                    f"Audio cleaning FFmpeg failed (rc={result.returncode}): "
                    f"{result.stderr.decode(errors='replace')[:300]}"
                    f" — using original audio."
                )
                cleaned_path.unlink(missing_ok=True)
                return wav_path

            cleaned_dur = Stage2AudioSynthesizer._get_duration(cleaned_path)
            logger.info(
                f"Audio after cleaning: duration={cleaned_dur:.2f}s"
                f"  (was {original_dur:.2f}s, delta={cleaned_dur - original_dur:+.2f}s)"
            )

            # Sanity check — reject if cleaning produced a very short file
            if cleaned_dur < 0.5:
                logger.warning(
                    f"Cleaned audio too short ({cleaned_dur:.2f}s) — using original audio."
                )
                cleaned_path.unlink(missing_ok=True)
                return wav_path

            # Replace original with cleaned
            cleaned_path.replace(wav_path)
            return wav_path

        except Exception as exc:
            logger.warning(f"Audio cleaning failed unexpectedly: {exc} — using original audio.")
            cleaned_path.unlink(missing_ok=True)
            return wav_path

    @staticmethod
    def _validate_audio(audio_path: Path) -> None:
        """
        Validate the WAV file before triggering human review.
        Raises RuntimeError if missing, < 5 KB, unreadable by ffprobe,
        or duration < 0.5 s.
        """
        if not audio_path.exists():
            raise RuntimeError(f"Audio file was not created at {audio_path}")

        size = audio_path.stat().st_size
        if size < 5120:  # 5 KB
            raise RuntimeError(
                f"Audio file is too small ({size} bytes) — likely empty or corrupt: {audio_path}"
            )

        result = subprocess.run(
            [
                "ffprobe",
                "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                str(audio_path),
            ],
            capture_output=True,
            timeout=30,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"ffprobe could not read audio file {audio_path}: "
                f"{result.stderr.decode(errors='replace')[:200]}"
            )

        try:
            info = json.loads(result.stdout)
            duration = float(info.get("format", {}).get("duration", 0))
        except (json.JSONDecodeError, ValueError):
            duration = 0

        if duration < 0.5:
            raise RuntimeError(
                f"Audio duration too short ({duration:.2f}s) — synthesis may have failed: {audio_path}"
            )
