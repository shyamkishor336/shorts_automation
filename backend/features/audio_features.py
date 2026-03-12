"""
Stage 2 Feature Extraction — Audio quality metrics.
All functions are wrapped in try/except to prevent pipeline crashes.
"""

import logging
import subprocess
import tempfile
import warnings
from pathlib import Path
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)


def _convert_to_wav(audio_path: Path) -> Optional[Path]:
    """
    Convert any audio file (e.g. edge-tts MP3) to a 16 kHz mono WAV using FFmpeg.
    Returns the Path to the temporary WAV file, or None if conversion fails.
    The caller is responsible for deleting the file when done.
    """
    try:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        wav_path = Path(tmp.name)

        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", str(audio_path),
                "-ar", "16000",
                "-ac", "1",
                str(wav_path),
            ],
            capture_output=True,
            timeout=60,
        )

        if result.returncode != 0:
            logger.warning(
                f"FFmpeg WAV conversion failed (rc={result.returncode}): "
                f"{result.stderr.decode(errors='replace')[:200]}"
            )
            wav_path.unlink(missing_ok=True)
            return None

        return wav_path
    except Exception as exc:
        logger.warning(f"_convert_to_wav failed: {exc}")
        return None


def extract_phoneme_error_rate(
    audio_path: Path, original_text: str
) -> Optional[float]:
    """
    Transcribe audio with Whisper, compare to original text using edit distance.
    Rate = edit_distance / len(original_words). Range: 0–∞ (lower = better).
    """
    try:
        import whisper
        import editdistance  # may need: pip install editdistance

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = whisper.load_model("base")
        result = model.transcribe(str(audio_path))
        transcription = result.get("text", "")

        original_words = original_text.lower().split()
        transcribed_words = transcription.lower().split()

        if not original_words:
            return None

        dist = editdistance.eval(original_words, transcribed_words)
        rate = dist / len(original_words)
        return round(float(rate), 4)
    except ImportError:
        # editdistance not available; fall back to basic ratio
        try:
            import whisper
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = whisper.load_model("base")
            result = model.transcribe(str(audio_path))
            transcription = result.get("text", "")

            original_words = set(original_text.lower().split())
            transcribed_words = set(transcription.lower().split())

            if not original_words:
                return None

            missing = len(original_words - transcribed_words)
            return round(missing / len(original_words), 4)
        except Exception as exc2:
            logger.warning(f"phoneme_error_rate (fallback) failed: {exc2}")
            return None
    except Exception as exc:
        logger.warning(f"phoneme_error_rate extraction failed: {exc}")
        return None


def extract_silence_ratio(audio_path: Path) -> Optional[float]:
    """
    Ratio of silent frames (below energy threshold) using librosa.
    Range: 0–1. Higher = more silence.
    """
    try:
        import librosa
        import numpy as np

        y, sr = librosa.load(str(audio_path), sr=None)
        # Compute RMS energy
        rms = librosa.feature.rms(y=y)[0]
        threshold = 0.01  # silence threshold
        silent_frames = np.sum(rms < threshold)
        return round(float(silent_frames / len(rms)), 4)
    except Exception as exc:
        logger.warning(f"silence_ratio extraction failed: {exc}")
        return None


def extract_speaking_rate_variance(
    audio_path: Path, scenes: List[Dict]
) -> Optional[float]:
    """
    Variance in words-per-second across 8 scene segments.
    Higher variance = uneven pacing.
    """
    try:
        import librosa
        import numpy as np

        y, sr = librosa.load(str(audio_path), sr=None)
        total_duration = librosa.get_duration(y=y, sr=sr)
        segment_duration = total_duration / max(len(scenes), 1)

        wps_values = []
        for scene in scenes:
            narration = scene.get("narration", "")
            word_count = len(narration.split())
            if segment_duration > 0:
                wps_values.append(word_count / segment_duration)

        if len(wps_values) < 2:
            return None

        variance = float(np.var(wps_values))
        return round(variance, 4)
    except Exception as exc:
        logger.warning(f"speaking_rate_variance extraction failed: {exc}")
        return None


def extract_energy_variance(audio_path: Path) -> Optional[float]:
    """
    Variance in RMS energy across audio frames using librosa.
    Higher = more dynamic audio amplitude.
    """
    try:
        import librosa
        import numpy as np

        y, sr = librosa.load(str(audio_path), sr=None)
        rms = librosa.feature.rms(y=y)[0]
        variance = float(np.var(rms))
        return round(variance, 8)
    except Exception as exc:
        logger.warning(f"energy_variance extraction failed: {exc}")
        return None


def extract_tts_word_count_match(
    audio_path: Path, original_text: str
) -> Optional[float]:
    """
    Word count of Whisper transcription / original word count.
    Close to 1.0 = good match. Less than 1.0 = words dropped.
    """
    try:
        import whisper

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = whisper.load_model("base")
        result = model.transcribe(str(audio_path))
        transcription = result.get("text", "")

        original_count = len(original_text.split())
        transcribed_count = len(transcription.split())

        if original_count == 0:
            return None

        ratio = transcribed_count / original_count
        return round(float(ratio), 4)
    except Exception as exc:
        logger.warning(f"tts_word_count_match extraction failed: {exc}")
        return None


def extract_all(
    audio_path: Path,
    original_text: str,
    scenes: List[Dict],
) -> Dict[str, Any]:
    """
    Run all Stage 2 feature extractions.

    Args:
        audio_path: Path to the generated audio file (e.g. .mp3 from edge-tts).
        original_text: Full narration text (all scenes concatenated).
        scenes: List of scene dicts for per-scene analysis.

    Returns:
        Dict mapping feature name to value (None if extraction failed).
    """
    # stage2_audio now provides a clean 16 kHz WAV directly; skip conversion.
    # For any other caller that still passes an MP3, convert on the fly.
    if audio_path.suffix.lower() == ".wav":
        wav_path = None
        work_path = audio_path
    else:
        wav_path = _convert_to_wav(audio_path)
        work_path = wav_path if wav_path is not None else audio_path

    try:
        return {
            "phoneme_error_rate": extract_phoneme_error_rate(work_path, original_text),
            "silence_ratio": extract_silence_ratio(work_path),
            "speaking_rate_variance": extract_speaking_rate_variance(work_path, scenes),
            "energy_variance": extract_energy_variance(work_path),
            "tts_word_count_match": extract_tts_word_count_match(work_path, original_text),
        }
    finally:
        if wav_path is not None:
            wav_path.unlink(missing_ok=True)
