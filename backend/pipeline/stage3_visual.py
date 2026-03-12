"""
Stage 3 — Visual Generation.
Generates one video clip per scene using either:
  PATH A (ken_burns): DreamShaper image -> Ken Burns FFmpeg animation
  PATH B (modal):     Modal CogVideoX endpoint (falls back to Ken Burns per-scene on failure)

Output: data/outputs/{run_id}/scenes/scene_01.mp4 ... scene_06.mp4
"""

import logging
import re
from pathlib import Path
from typing import Dict, Any, List

from backend.config import settings
from backend.features import visual_features

logger = logging.getLogger(__name__)

# ── Lazy DreamShaper loader ────────────────────────────────────────────────
# DreamShaper is only loaded on the first ken_burns image request.
# Modal runs never touch this — sd_pipe stays None for the entire process.

_dreamshaper_pipeline = None


def _get_dreamshaper():
    global _dreamshaper_pipeline
    if _dreamshaper_pipeline is None:
        import torch
        from diffusers import StableDiffusionPipeline
        logger.info("Lazy-loading DreamShaper onto CUDA (first ken_burns run)…")
        _dreamshaper_pipeline = StableDiffusionPipeline.from_pretrained(
            "Lykon/DreamShaper",
            torch_dtype=torch.float16,
            safety_checker=None,
        ).to("cuda")
        logger.info("DreamShaper ready.")
    return _dreamshaper_pipeline


class Stage3VisualGenerator:
    """
    Generates one video clip per scene.
    ken_burns path: DreamShaper image -> Ken Burns FFmpeg clip.
    modal path:     Modal endpoint clip, falls back to Ken Burns on failure.
    """

    def run(
        self,
        script_data: dict,
        run_id: str,
        video_provider_choice: str = "ken_burns",
    ) -> Dict[str, Any]:
        """
        Generate one video clip per scene and save to scenes/.

        Args:
            script_data: Stage 1 result envelope {"script": {"scenes": [...]}, ...}.
            run_id: Pipeline run ID.
            video_provider_choice: 'ken_burns' or 'modal'.

        Returns:
            Dict with 'output_path', 'scene_paths', 'image_paths', and 'features'.
        """
        output_dir = settings.OUTPUT_DIR / run_id
        images_dir = output_dir / "images"
        scenes_dir = output_dir / "scenes"
        images_dir.mkdir(parents=True, exist_ok=True)
        scenes_dir.mkdir(parents=True, exist_ok=True)

        # Unwrap Stage 1 envelope
        inner = script_data.get("script", script_data)
        scenes = inner.get("scenes", [])
        topic = inner.get("topic", script_data.get("topic", "Educational Content"))

        scene_count = settings.SCENE_COUNT
        if len(scenes) > scene_count:
            logger.warning(
                f"Script has {len(scenes)} scenes but SCENE_COUNT={scene_count}. "
                f"Truncating to {scene_count}."
            )
            scenes = scenes[:scene_count]

        image_paths: List[Path] = []
        scene_paths: List[Path] = []
        provider_used = video_provider_choice

        for i, scene in enumerate(scenes):
            scene_num = str(i + 1).zfill(2)
            visual_prompt = scene.get("visual_prompt", "")
            video_path = scenes_dir / f"scene_{scene_num}.mp4"

            if video_provider_choice == "modal":
                # PATH B: Modal endpoint, per-scene Ken Burns fallback on failure
                success, fail_reason = self._generate_modal_clip(visual_prompt, video_path, scene_num)
                if not success:
                    print(
                        f"[MODAL FALLBACK] scene {scene_num} — Modal failed, reason: {fail_reason}. "
                        "Falling back to Ken Burns.",
                        flush=True,
                    )
                    logger.error(
                        f"PROVIDER FALLBACK: Modal failed for scene {scene_num}. "
                        f"Reason: {fail_reason}. Falling back to Ken Burns."
                    )
                    image_path = images_dir / f"scene_{scene_num}.png"
                    self._generate_image(visual_prompt, image_path)
                    image_paths.append(image_path)
                    self._generate_ken_burns_clip(
                        image_path=image_path if image_path.exists() else None,
                        prompt=visual_prompt,
                        topic=topic,
                        output_path=video_path,
                    )
                    # provider_used stays as "modal" — Modal was the attempted provider
            else:
                # PATH A: DreamShaper image -> Ken Burns clip
                image_path = images_dir / f"scene_{scene_num}.png"
                self._generate_image(visual_prompt, image_path)
                image_paths.append(image_path)
                self._generate_ken_burns_clip(
                    image_path=image_path if image_path.exists() else None,
                    prompt=visual_prompt,
                    topic=topic,
                    output_path=video_path,
                )

            scene_paths.append(video_path)
            logger.info(
                f"Scene {scene_num}: clip={'ok' if video_path.exists() else 'MISSING'} "
                f"({video_path.stat().st_size if video_path.exists() else 0:,} bytes)"
            )

        # Validate clips
        valid_clips = [p for p in scene_paths if p.exists() and p.stat().st_size > 1024]
        if not valid_clips:
            raise RuntimeError(
                f"Stage 3 produced no valid video clips for run {run_id}. "
                f"Expected {len(scenes)} clips in {scenes_dir}."
            )
        logger.info(f"Clip validation passed: {len(valid_clips)}/{len(scenes)} valid clips")

        # Extract visual features — non-fatal
        try:
            features = visual_features.extract_all(
                scenes=scenes,
                video_paths=scene_paths,
                image_paths=image_paths,
                provider=provider_used,
            )
            logger.info(f"Visual features extracted for run {run_id}")
        except Exception as fe:
            logger.warning(f"Visual feature extraction failed (non-fatal): {fe}")
            features = {"visual_provider": provider_used}

        return {
            "output_path": str(scenes_dir),
            "scene_paths": [str(p) for p in scene_paths],
            "image_paths": [str(p) for p in image_paths],
            "features": features,
        }

    # ── Private helpers ────────────────────────────────────────────────────

    def _generate_image(self, prompt: str, output_path: Path) -> bool:
        """Generate a DreamShaper still image. Returns True on success."""
        try:
            cleaned = re.sub(r'Text overlay[^.!?]*[.!?]?', '', prompt, flags=re.IGNORECASE)
            cleaned = cleaned.replace("'", "").replace('"', "").replace(":", "").replace("!", "")
            cleaned = " ".join(cleaned.split())
            cleaned_prompt = cleaned[:200]

            image = _get_dreamshaper()(
                prompt=cleaned_prompt,
                negative_prompt="blurry, low quality, watermark, text, ugly, deformed",
                width=384,
                height=672,
                num_inference_steps=15,
                guidance_scale=7.5,
            ).images[0]

            output_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(str(output_path))
            logger.info(f"DreamShaper image saved: {output_path}")
            return True

        except Exception as exc:
            logger.warning(f"DreamShaper image generation failed: {exc}")
            return False

    def _generate_ken_burns_clip(
        self,
        image_path: Path,
        prompt: str,
        topic: str,
        output_path: Path,
    ) -> Path:
        """Animate a still image into a Ken Burns video clip via FFmpeg."""
        from backend.providers.ken_burns_provider import KenBurnsProvider
        kb = KenBurnsProvider()
        return kb.generate(
            prompt=prompt,
            output_path=output_path,
            source_image_path=image_path if image_path and image_path.exists() else None,
            topic=topic,
            duration=settings.SCENE_DURATION_SECONDS,
        )

    def _generate_modal_clip(
        self,
        prompt: str,
        output_path: Path,
        scene_num: str,
    ) -> tuple:
        """
        Call the Modal endpoint to generate one video clip.
        Returns (True, "") on success, (False, reason_str) on any failure.
        """
        from backend.providers.modal_provider import ModalProvider
        modal = ModalProvider()

        if not modal.is_available():
            reason = (
                f"MODAL_ENDPOINT_URL is empty or not set in .env. "
                f"Current value: {settings.MODAL_ENDPOINT_URL!r}"
            )
            print(f"[MODAL ERROR] scene {scene_num}: {reason}", flush=True)
            logger.error(f"Modal not available for scene {scene_num}: {reason}")
            return False, reason

        print(
            f"[MODAL] scene {scene_num}: calling {modal.endpoint_url} …",
            flush=True,
        )
        try:
            modal.generate(
                prompt=prompt,
                output_path=output_path,
                duration=settings.SCENE_DURATION_SECONDS,
            )
            print(f"[MODAL] scene {scene_num}: clip saved to {output_path}", flush=True)
            return True, ""
        except Exception as exc:
            reason = str(exc)
            print(f"[MODAL ERROR] scene {scene_num}: {reason}", flush=True)
            logger.error(f"Modal clip generation failed for scene {scene_num}: {reason}")
            return False, reason
