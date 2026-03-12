"""
Stage 3 Feature Extraction — Visual quality metrics for generated video clips.
All functions are wrapped in try/except to prevent pipeline crashes.
Scores are averaged across all 8 scenes.
"""

import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

# ── CLIP singleton — loaded once per backend process ──────────────────────
_clip_model = None
_clip_processor = None


def _get_clip():
    global _clip_model, _clip_processor
    if _clip_model is None:
        from transformers import CLIPModel, CLIPProcessor
        logger.info("Loading CLIP model (openai/clip-vit-base-patch32) for first time…")
        _clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        _clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        _clip_model.eval()
        logger.info("CLIP model loaded and cached — will not reload again this session.")
    return _clip_model, _clip_processor


def _extract_middle_frame(video_path: Path):
    """Extract the middle frame of a video as a numpy array (BGR)."""
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise ValueError(f"Cannot extract frame from {video_path}")
    return frame


def extract_clip_similarity(
    visual_prompt: str, video_path: Path
) -> Optional[float]:
    """
    CLIP cosine similarity between visual_prompt text and the video middle frame.
    Range: -1 to 1. Higher = better visual-text alignment.
    """
    try:
        import torch
        from PIL import Image
        import cv2
        import numpy as np

        frame_bgr = _extract_middle_frame(video_path)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)

        model, processor = _get_clip()

        inputs = processor(
            text=[visual_prompt],
            images=[pil_image],
            return_tensors="pt",
            padding=True,
        )
        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits_per_image
        similarity = torch.sigmoid(logits).item()
        return round(float(similarity), 4)
    except Exception as exc:
        logger.warning(f"clip_similarity extraction failed: {exc}")
        return None


def extract_aesthetic_score(video_path: Path) -> Optional[float]:
    """
    Proxy aesthetic score: sharpness × colour_variance, normalised to 0–1.
    """
    try:
        import cv2
        import numpy as np

        frame = _extract_middle_frame(video_path)

        # Sharpness via Laplacian variance (normalised)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness = min(laplacian_var / 1000.0, 1.0)  # normalise

        # Colour variance across channels
        colour_var = float(np.var(frame.astype(np.float32))) / (255.0 ** 2)
        colour_var = min(colour_var, 1.0)

        aesthetic = sharpness * colour_var
        return round(float(aesthetic), 4)
    except Exception as exc:
        logger.warning(f"aesthetic_score extraction failed: {exc}")
        return None


def extract_blur_score(video_path: Path) -> Optional[float]:
    """
    OpenCV Laplacian variance on middle frame.
    Higher = sharper (less blurry).
    """
    try:
        import cv2

        frame = _extract_middle_frame(video_path)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        return round(blur, 4)
    except Exception as exc:
        logger.warning(f"blur_score extraction failed: {exc}")
        return None


def extract_object_match_score(
    visual_prompt: str, video_path: Path
) -> Optional[float]:
    """
    YOLOv8n object detection on middle frame.
    Score = matched_objects / total_expected_objects (min 0).
    Checks if any detected class name appears in the visual prompt.
    """
    try:
        from ultralytics import YOLO
        import cv2

        frame = _extract_middle_frame(video_path)
        model = YOLO("yolov8n.pt")
        results = model(frame, verbose=False)

        prompt_lower = visual_prompt.lower()
        detected_classes = set()
        for r in results:
            for cls_id in r.boxes.cls.tolist():
                class_name = model.names[int(cls_id)].lower()
                detected_classes.add(class_name)

        # Count words in prompt that match a detected class
        prompt_words = set(prompt_lower.split())
        matched = len(detected_classes & prompt_words)

        # Use number of detected objects as denominator (min 1)
        total_detected = max(len(detected_classes), 1)
        score = matched / total_detected
        return round(float(score), 4)
    except Exception as exc:
        logger.warning(f"object_match_score extraction failed: {exc}")
        return None


def extract_colour_tone_match(
    image_path: Path, video_path: Path
) -> float:
    """
    HSV histogram cosine similarity between source image and video middle frame.
    Range: 0–1. Higher = colour palettes match well.
    Returns 0.5 (neutral) on any failure so the feature is never NULL.
    """
    try:
        import cv2
        import numpy as np

        # Read source image
        img = cv2.imread(str(image_path))
        if img is None:
            return 0.5

        # Extract middle frame from video
        frame = _extract_middle_frame(video_path)

        def hsv_histogram(img_bgr):
            hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
            cv2.normalize(hist, hist)
            return hist.flatten()

        hist1 = hsv_histogram(img)
        hist2 = hsv_histogram(frame)

        # Cosine similarity
        dot = float(np.dot(hist1, hist2))
        norm = float(np.linalg.norm(hist1) * np.linalg.norm(hist2))
        if norm == 0:
            return 0.5
        return round(dot / norm, 4)
    except Exception as exc:
        logger.warning(f"colour_tone_match extraction failed: {exc}")
        return 0.5


def extract_all(
    scenes: List[Dict],
    video_paths: List[Path],
    image_paths: List[Path],
    provider: str = "",
) -> Dict[str, Any]:
    """
    Run all Stage 3 feature extractions and average across all 8 scenes.

    Args:
        scenes: List of scene dicts with 'visual_prompt' key.
        video_paths: List of paths to generated scene .mp4 files.
        image_paths: List of paths to FLUX still images.
        provider: Provider name used for generation.

    Returns:
        Dict mapping feature name to averaged value (None if all failed).
    """
    def _avg(values: List) -> Optional[float]:
        valid = [v for v in values if v is not None]
        if not valid:
            return None
        return round(sum(valid) / len(valid), 4)

    clip_scores = []
    aesthetic_scores = []
    blur_scores = []
    object_scores = []
    colour_scores = []

    for i, (scene, video_path) in enumerate(zip(scenes, video_paths)):
        visual_prompt = scene.get("visual_prompt", "")
        image_path = image_paths[i] if i < len(image_paths) else None

        if not video_path.exists():
            logger.warning(f"Video path missing: {video_path}")
            continue

        clip_scores.append(extract_clip_similarity(visual_prompt, video_path))
        aesthetic_scores.append(extract_aesthetic_score(video_path))
        blur_scores.append(extract_blur_score(video_path))
        object_scores.append(extract_object_match_score(visual_prompt, video_path))

        if image_path and image_path.exists():
            colour_scores.append(extract_colour_tone_match(image_path, video_path))
        else:
            # No source image (Modal provider) — use neutral default so
            # colour_tone_match is never NULL in the database.
            colour_scores.append(0.5)

    return {
        "clip_similarity": _avg(clip_scores),
        "aesthetic_score": _avg(aesthetic_scores),
        "blur_score": _avg(blur_scores),
        "object_match_score": _avg(object_scores),
        "colour_tone_match": _avg(colour_scores),
        "visual_provider": provider,
    }
