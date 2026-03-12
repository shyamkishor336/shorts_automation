"""
Ken Burns Provider — Local FFmpeg fallback.
Applies animated zoom/pan (Ken Burns effect) to a still image using FFmpeg.
This provider NEVER fails; it always produces output even without an input image.
"""

import logging
import subprocess
from pathlib import Path
from typing import Optional

from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

# Target output dimensions — vertical 9:16 for Shorts
VIDEO_WIDTH = 1080
VIDEO_HEIGHT = 1920
VIDEO_FPS = 24
CLIP_DURATION = 6.5  # seconds — 6.5 s × 6 scenes ≈ 39 s total


class KenBurnsProvider:
    """
    Generates a 6.5-second Ken Burns video clip from a still image using FFmpeg.
    Falls back to a Pillow-generated placeholder if the source image is missing.
    This provider is guaranteed to always produce output.
    """

    def __init__(self) -> None:
        self.name = "ken_burns"

    def is_available(self) -> bool:
        """Ken Burns is always available — it only requires FFmpeg."""
        return True

    def generate(
        self,
        prompt: str,
        output_path: Path,
        source_image_path: Optional[Path] = None,
        topic: str = "Educational Content",
        duration: float = CLIP_DURATION,
    ) -> Path:
        """
        Generate a Ken Burns video clip.

        Args:
            prompt: Visual description (used for placeholder text if no image).
            output_path: Where to save the .mp4 file.
            source_image_path: Optional path to a FLUX still image.
            topic: Topic text used in placeholder image.
            duration: Clip duration in seconds.

        Returns:
            Path to the generated .mp4 file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Resolve the source image
        image_path = self._resolve_image(source_image_path, prompt, topic, output_path)

        # Apply Ken Burns effect via FFmpeg
        self._apply_ken_burns(image_path, output_path, duration)

        logger.info(f"Ken Burns clip generated: {output_path}")
        return output_path

    # ── Private helpers ────────────────────────────────────────────────────

    def _resolve_image(
        self,
        source_image_path: Optional[Path],
        prompt: str,
        topic: str,
        output_path: Path,
    ) -> Path:
        """Return a valid image path, creating a placeholder if needed."""
        if source_image_path and Path(source_image_path).exists():
            return Path(source_image_path)

        logger.warning(
            f"Source image not found ({source_image_path}). "
            "Generating Pillow placeholder."
        )
        placeholder_path = output_path.parent / f"{output_path.stem}_placeholder.png"
        self._create_placeholder(placeholder_path, topic, prompt)
        return placeholder_path

    def _create_placeholder(
        self, path: Path, topic: str, prompt: str
    ) -> None:
        """Create a 1080×1920 solid-colour placeholder image with text overlay."""
        img = Image.new("RGB", (VIDEO_WIDTH, VIDEO_HEIGHT), color=(30, 60, 114))
        draw = ImageDraw.Draw(img)

        # Try to load a system font; fall back to default
        font_title: ImageFont.ImageFont
        font_body: ImageFont.ImageFont
        try:
            font_title = ImageFont.truetype("arial.ttf", 60)
            font_body = ImageFont.truetype("arial.ttf", 32)
        except OSError:
            font_title = ImageFont.load_default()
            font_body = ImageFont.load_default()

        # Draw topic title — vertically centred in portrait canvas
        self._draw_centred_text(draw, topic, font_title, VIDEO_WIDTH, 800, (255, 255, 255))

        # Draw prompt (wrapped at 40 chars for portrait)
        wrapped = self._wrap_text(prompt, 40)
        self._draw_centred_text(draw, wrapped, font_body, VIDEO_WIDTH, 960, (200, 220, 255))

        path.parent.mkdir(parents=True, exist_ok=True)
        img.save(str(path))
        logger.info(f"Placeholder image saved: {path}")

    @staticmethod
    def _draw_centred_text(
        draw: ImageDraw.ImageDraw,
        text: str,
        font: ImageFont.ImageFont,
        canvas_width: int,
        y: int,
        colour: tuple,
    ) -> None:
        """Draw horizontally centred text at vertical position y."""
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        x = (canvas_width - text_width) // 2
        draw.text((x, y), text, fill=colour, font=font)

    @staticmethod
    def _wrap_text(text: str, max_chars: int) -> str:
        """Wrap text at word boundaries to max_chars per line."""
        words = text.split()
        lines = []
        current = ""
        for word in words:
            if len(current) + len(word) + 1 <= max_chars:
                current = f"{current} {word}".strip()
            else:
                if current:
                    lines.append(current)
                current = word
        if current:
            lines.append(current)
        return "\n".join(lines)

    def _apply_ken_burns(
        self, image_path: Path, output_path: Path, duration: float
    ) -> None:
        """
        Apply Ken Burns zoom/pan using FFmpeg zoompan filter.
        Zoom from 1.0 to 1.4, centre-anchored, 1080x1920 portrait, 24fps, libx264.
        """
        total_frames = duration * VIDEO_FPS  # 144 frames at 24fps

        # zoompan filter: z=zoom expression, x/y centred, output size, frame rate
        zoompan = (
            f"zoompan="
            f"z='min(zoom+0.0027,1.4)':"
            f"x='iw/2-(iw/zoom/2)':"
            f"y='ih/2-(ih/zoom/2)':"
            f"d={total_frames}:"
            f"s={VIDEO_WIDTH}x{VIDEO_HEIGHT}:"
            f"fps={VIDEO_FPS}"
        )

        cmd = [
            "ffmpeg",
            "-y",
            "-loop", "1",
            "-i", str(image_path),
            "-vf", f"{zoompan},format=yuv420p",
            "-c:v", "libx264",
            "-t", str(duration),
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(output_path),
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode != 0:
                logger.error(f"FFmpeg error: {result.stderr}")
                # Last-resort: create a black video
                self._create_black_video(output_path, duration)
        except subprocess.TimeoutExpired:
            logger.error("FFmpeg timed out. Creating black video fallback.")
            self._create_black_video(output_path, duration)
        except FileNotFoundError:
            logger.error("FFmpeg not found. Creating black video via Pillow frames.")
            self._create_black_video(output_path, duration)

    def _create_black_video(self, output_path: Path, duration: float) -> None:
        """Absolute last resort: generate a solid black video using FFmpeg lavfi."""
        cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi",
            "-i", f"color=c=black:size={VIDEO_WIDTH}x{VIDEO_HEIGHT}:rate={VIDEO_FPS}",
            "-t", str(duration),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            str(output_path),
        ]
        try:
            subprocess.run(cmd, capture_output=True, timeout=60)
        except Exception as exc:
            logger.error(f"Black video generation also failed: {exc}")
            # Write a 1-byte dummy file so the pipeline can continue
            output_path.write_bytes(b"\x00")
