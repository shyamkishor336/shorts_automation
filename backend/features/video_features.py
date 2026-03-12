"""
Stage 4 Feature Extraction — Final video assembly quality metrics.
All functions are wrapped in try/except to prevent pipeline crashes.
"""

import logging
import subprocess
import json
from pathlib import Path
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)


def extract_av_sync_error_ms(video_path: Path) -> Optional[float]:
    """
    Use ffprobe to get audio and video stream start times.
    Returns |audio_start - video_start| in milliseconds.
    Lower = better synchronised.
    """
    try:
        cmd = [
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_streams",
            str(video_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        data = json.loads(result.stdout)

        audio_start = None
        video_start = None

        for stream in data.get("streams", []):
            if stream.get("codec_type") == "video":
                video_start = float(stream.get("start_time", 0))
            elif stream.get("codec_type") == "audio":
                audio_start = float(stream.get("start_time", 0))

        if audio_start is None or video_start is None:
            return None

        error_ms = abs(audio_start - video_start) * 1000
        return round(float(error_ms), 4)
    except Exception as exc:
        logger.warning(f"av_sync_error_ms extraction failed: {exc}")
        return None


def extract_transition_smoothness(scene_paths: List[Path]) -> Optional[float]:
    """
    Mean absolute pixel difference between last frame of scene N and
    first frame of scene N+1. Averaged across all transitions.
    Normalised to 0–1. Lower = smoother transitions.
    """
    try:
        import cv2
        import numpy as np

        diffs = []
        for i in range(len(scene_paths) - 1):
            path_a = scene_paths[i]
            path_b = scene_paths[i + 1]

            if not path_a.exists() or not path_b.exists():
                continue

            cap_a = cv2.VideoCapture(str(path_a))
            cap_b = cv2.VideoCapture(str(path_b))

            # Get last frame of scene A
            total_a = int(cap_a.get(cv2.CAP_PROP_FRAME_COUNT))
            cap_a.set(cv2.CAP_PROP_POS_FRAMES, max(total_a - 1, 0))
            ret_a, frame_a = cap_a.read()
            cap_a.release()

            # Get first frame of scene B
            ret_b, frame_b = cap_b.read()
            cap_b.release()

            if not ret_a or not ret_b:
                continue

            # Resize to same size for comparison
            frame_b_resized = cv2.resize(
                frame_b, (frame_a.shape[1], frame_a.shape[0])
            )

            diff = np.mean(
                np.abs(
                    frame_a.astype(np.float32) - frame_b_resized.astype(np.float32)
                )
            )
            # Normalise: max possible diff is 255
            diffs.append(diff / 255.0)

        if not diffs:
            return None

        return round(float(np.mean(diffs)), 4)
    except Exception as exc:
        logger.warning(f"transition_smoothness extraction failed: {exc}")
        return None


def extract_duration_deviation(
    video_path: Path,
    target_duration: float,
) -> Optional[float]:
    """
    Absolute difference between actual video duration and target duration.
    |actual_duration - target_duration| in seconds.
    """
    try:
        cmd = [
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            str(video_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        data = json.loads(result.stdout)

        actual_duration = float(data["format"]["duration"])
        deviation = abs(actual_duration - target_duration)
        return round(float(deviation), 4)
    except Exception as exc:
        logger.warning(f"duration_deviation extraction failed: {exc}")
        return None


def extract_all(
    final_video_path: Path,
    scene_paths: List[Path],
    target_duration: float = 48.0,
) -> Dict[str, Any]:
    """
    Run all Stage 4 feature extractions.

    Args:
        final_video_path: Path to the assembled final_video.mp4.
        scene_paths: List of paths to individual scene clips (for transitions).
        target_duration: Expected total video duration (default 8×6=48s).

    Returns:
        Dict mapping feature name to value (None if extraction failed).
    """
    return {
        "av_sync_error_ms": extract_av_sync_error_ms(final_video_path),
        "transition_smoothness": extract_transition_smoothness(scene_paths),
        "duration_deviation_s": extract_duration_deviation(
            final_video_path, target_duration
        ),
    }
