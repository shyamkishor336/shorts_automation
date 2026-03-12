"""
Modal.com Deployment — CogVideoX-2b Web Endpoint.
This file is deployed ONCE to Modal.com and runs on GPU A10G.

Deploy with:
    pip install modal
    modal setup
    modal deploy modal_cogvideox_endpoint.py

The deployed URL goes into .env as MODAL_ENDPOINT_URL.
"""

# NOTE: This file is STANDALONE. Do NOT import from backend/.
# It runs in Modal's cloud environment, not locally.

import os

import modal

# ── Modal App Definition ───────────────────────────────────────────────────

app = modal.App("cogvideox-hitl")


def _download_model():
    """
    Pre-download CogVideoX-2b model weights into the container image.
    This runs at build time so inference starts without download delay.
    Must be defined BEFORE it is passed to .run_function().
    """
    from huggingface_hub import snapshot_download

    snapshot_download(
        "THUDM/CogVideoX-2b",
        ignore_patterns=["*.bin"],  # prefer safetensors
    )


# Build the container image with all required dependencies.
# _download_model is referenced here, so it must be defined above.
cogvideox_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "fastapi[standard]",   # required explicitly for Modal web endpoints
        "torch==2.3.0",
        "torchvision==0.18.0",
        "diffusers==0.30.3",
        "transformers==4.44.2",
        "accelerate",
        "imageio",
        "imageio-ffmpeg",
        "opencv-python",
        "sentencepiece",
        "huggingface_hub",
        "numpy",
    )
    .run_function(_download_model)
)


# ── Web Endpoint ───────────────────────────────────────────────────────────

@app.function(
    image=cogvideox_image,
    gpu="A10G",
    timeout=600,
    memory=32768,
)
@modal.fastapi_endpoint(method="POST")
def generate(request: dict):
    """
    Generate a video clip using CogVideoX-2b.

    Request body (JSON):
        prompt (str): Visual description for the video.
        duration (int): Clip duration in seconds (default 6).

    Returns:
        Raw .mp4 bytes with Content-Type: video/mp4.
    """
    # fastapi is available inside Modal's container at runtime.
    # Import here (not at module level) so local `modal deploy` parsing
    # does not require fastapi to be installed on the developer's machine.
    import tempfile
    import torch
    from diffusers import CogVideoXPipeline
    from diffusers.utils import export_to_video
    from fastapi.responses import Response

    prompt: str = request.get("prompt", "")
    duration: int = int(request.get("duration", 6))

    if not prompt:
        return Response(
            content=b"prompt is required",
            status_code=400,
            media_type="text/plain",
        )

    # Load pipeline
    pipe = CogVideoXPipeline.from_pretrained(
        "THUDM/CogVideoX-2b",
        torch_dtype=torch.float16,
    )
    pipe.to("cuda")
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    # Generate at native 16:9 (480x272). 40 frames @ 6 fps = 6.67 s per clip.
    # Six scenes → ~40 s total. Stage 4 scales to 1280x720 for Modal runs.
    num_frames = 40
    video_frames = pipe(
        prompt=prompt,
        num_inference_steps=25,
        guidance_scale=6.0,
        num_frames=num_frames,
        width=480,
        height=272,
    ).frames[0]

    # Export to mp4 bytes — must match num_frames / fps = 6.67 s
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name

    export_to_video(video_frames, tmp_path, fps=6)

    with open(tmp_path, "rb") as f:
        video_bytes = f.read()

    os.unlink(tmp_path)

    return Response(
        content=video_bytes,
        status_code=200,
        media_type="video/mp4",
    )
