"""
Stage 4 — Final Video Assembly using FFmpeg.
Reads scene clips from Stage 3 (scenes/*.mp4) and audio from Stage 2 (audio.mp3).
Concatenates clips, overlays audio, outputs 1080x1920 final_video.mp4.
No clip generation here — clips must already exist in scenes/.
"""

import json
import logging
import subprocess
from pathlib import Path
from typing import Dict, Any, List

from backend.config import settings
from backend.features import video_features

logger = logging.getLogger(__name__)


class Stage4VideoAssembler:
    """
    Assembles final_video.mp4 from pre-generated scene clips and narration audio.
    """

    def run(self, run_id: str, video_provider_choice: str = "ken_burns") -> Dict[str, Any]:
        """
        Concatenate scene clips and mix narration audio into final_video.mp4.

        Args:
            run_id: Pipeline run ID (used to locate input/output files).
            video_provider_choice: 'modal' → 1280×720 (16:9); 'ken_burns' → 1080×1920 (9:16).

        Returns:
            Dict with 'output_path' and 'features'.

        Raises:
            RuntimeError: If no clips are found or FFmpeg fails.
        """
        output_dir = settings.OUTPUT_DIR / run_id
        scenes_dir = output_dir / "scenes"
        audio_path = output_dir / "audio.mp3"
        final_output = output_dir / "final_video.mp4"

        # Collect scene clips written by Stage 3, capped at SCENE_COUNT
        scene_count = settings.SCENE_COUNT
        scene_paths = sorted(scenes_dir.glob("scene_*.mp4"))[:scene_count]

        if not scene_paths:
            raise RuntimeError(
                f"No scene clips found in {scenes_dir}. "
                "Stage 3 must run before Stage 4."
            )
        if len(scene_paths) < scene_count:
            logger.warning(
                f"Expected {scene_count} clips, found {len(scene_paths)}. "
                "Assembling with available clips."
            )

        logger.info(f"Assembling {len(scene_paths)} clips for run {run_id}")

        # Diagnostic: log actual dimensions of every input clip
        self._log_scene_dimensions(scene_paths)

        # Step 1: Write FFmpeg concat list
        concat_list = self._write_concat_list(scene_paths, output_dir)

        # Step 2: Concatenate clips + mix audio -> final_video.mp4
        self._ffmpeg_assemble(concat_list, audio_path, final_output, video_provider_choice)
        logger.info(f"Final video assembled: {final_output}")

        # Step 3: Extract quality features
        target_duration = float(settings.SCENE_COUNT * settings.SCENE_DURATION_SECONDS)
        features = video_features.extract_all(
            final_video_path=final_output,
            scene_paths=scene_paths,
            target_duration=target_duration,
        )
        logger.info(f"Video features extracted for run {run_id}")

        return {
            "output_path": str(final_output),
            "features": features,
        }

    # ── Private helpers ────────────────────────────────────────────────────

    @staticmethod
    def _log_scene_dimensions(scene_paths: List[Path]) -> None:
        """
        Run ffprobe on every input scene clip and log width/height/duration.
        Helps diagnose cropping or dimension mismatch issues in the final video.
        """
        for clip in scene_paths:
            try:
                result = subprocess.run(
                    [
                        "ffprobe",
                        "-v", "quiet",
                        "-print_format", "json",
                        "-show_streams",
                        str(clip),
                    ],
                    capture_output=True,
                    timeout=30,
                )
                if result.returncode != 0:
                    logger.warning(
                        f"Stage4 ffprobe failed for {clip.name}: "
                        f"{result.stderr.decode(errors='replace')[:200]}"
                    )
                    continue

                info = json.loads(result.stdout)
                for stream in info.get("streams", []):
                    if stream.get("codec_type") == "video":
                        w = stream.get("width", "?")
                        h = stream.get("height", "?")
                        dur = stream.get("duration", "?")
                        codec = stream.get("codec_name", "?")
                        logger.info(
                            f"Stage4 input clip: {clip.name}  {w}x{h}"
                            f"  codec={codec}  duration={dur}s"
                        )
                        break
            except Exception as exc:
                logger.warning(f"Stage4 _log_scene_dimensions failed for {clip.name}: {exc}")

    @staticmethod
    def _write_concat_list(scene_paths: List[Path], output_dir: Path) -> Path:
        """Write an FFmpeg concat demuxer list file."""
        concat_path = output_dir / "concat_list.txt"
        lines = [f"file '{p.resolve()}'\n" for p in scene_paths]
        concat_path.write_text("".join(lines), encoding="utf-8")
        return concat_path

    def _ffmpeg_assemble(
        self,
        concat_list: Path,
        audio_path: Path,
        output_path: Path,
        video_provider_choice: str = "ken_burns",
    ) -> None:
        """
        Step 1: Concatenate scene clips into a silent video at the correct resolution.
        Step 2: Mix in narration audio with -shortest to trim to the shorter track.

        Modal runs → 1280×720 (16:9 landscape).
        Ken Burns runs → 1080×1920 (9:16 portrait).
        """
        silent_video = output_path.parent / "silent_concat.mp4"

        # Choose output dimensions based on provider
        if video_provider_choice == "modal":
            out_w, out_h = 1280, 720
        else:
            out_w, out_h = settings.VIDEO_WIDTH, settings.VIDEO_HEIGHT  # 1080×1920

        # Scale/pad every clip to the target resolution (handles any input AR)
        vf = (
            f"scale={out_w}:{out_h}"
            f":force_original_aspect_ratio=decrease,"
            f"pad={out_w}:{out_h}"
            f":(ow-iw)/2:(oh-ih)/2:color=black"
        )
        concat_cmd = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_list),
            "-vf", vf,
            "-c:v", settings.VIDEO_CODEC,
            "-pix_fmt", "yuv420p",
            "-r", str(settings.VIDEO_FPS),
            str(silent_video),
        ]
        self._run_ffmpeg(concat_cmd, "scene concatenation")
        logger.info(
            f"Stage4 concat complete: target={out_w}x{out_h}  provider={video_provider_choice}"
        )

        # Mix narration audio
        if audio_path.exists():
            mix_cmd = [
                "ffmpeg", "-y",
                "-i", str(silent_video),
                "-i", str(audio_path),
                "-c:v", "copy",
                "-c:a", settings.AUDIO_CODEC,
                "-shortest",
                str(output_path),
            ]
            self._run_ffmpeg(mix_cmd, "audio mixing")
        else:
            logger.warning("audio.mp3 not found — outputting silent video.")
            silent_video.rename(output_path)

        if silent_video.exists():
            silent_video.unlink()

    @staticmethod
    def _run_ffmpeg(cmd: List[str], stage_name: str) -> None:
        """Run an FFmpeg command; raise RuntimeError on non-zero exit."""
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"FFmpeg {stage_name} failed (code {result.returncode}): "
                    f"{result.stderr[-500:]}"
                )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"FFmpeg {stage_name} timed out after 300s."
            ) from exc
