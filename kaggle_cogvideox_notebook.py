#!/usr/bin/env python3
"""
Kaggle CogVideoX-2b Batch Generation Notebook.
Run this MANUALLY on Kaggle with a P100 GPU.

# HOW TO RUN ON KAGGLE:
# 1. Go to kaggle.com/code → New Notebook
# 2. Upload this file as a script
# 3. Enable GPU: Settings → Accelerator → GPU T4 x2 or P100
# 4. Upload data/kaggle_queue.json as a dataset named "hitl-kaggle-queue"
#    (or modify QUEUE_PATH below to match your dataset path)
# 5. Run all cells
# 6. Download results:
#    - /kaggle/working/kaggle_results.json  (job completion manifest)
#    - /kaggle/working/*.mp4               (generated video clips)
# 7. Copy .mp4 files into data/outputs/[run_id]/scenes/ on your machine
# 8. Call POST /providers/queue/complete for each completed job_id
#
# Kaggle free GPU quota: ~30 hours/week
# Expected throughput: ~13 clips/hour on P100 = ~390 clips per 30-hr week
"""

import json
import os
import subprocess
import sys
from pathlib import Path

# ── Install dependencies ────────────────────────────────────────────────────
print("Installing dependencies...")
subprocess.run([
    sys.executable, "-m", "pip", "install", "-q",
    "diffusers>=0.30.1",
    "transformers>=4.44.2",
    "accelerate",
    "imageio-ffmpeg",
    "sentencepiece",
], check=True)

import torch
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video

# ── Configuration ───────────────────────────────────────────────────────────
QUEUE_PATH = "/kaggle/input/hitl-kaggle-queue/kaggle_queue.json"
OUTPUT_DIR = Path("/kaggle/working")
RESULTS_FILE = OUTPUT_DIR / "kaggle_results.json"

NUM_INFERENCE_STEPS = 25
GUIDANCE_SCALE = 6.0
FPS = 8


def load_queue(queue_path: str) -> list:
    """Load pending jobs from the queue JSON file."""
    if not Path(queue_path).exists():
        print(f"Queue file not found: {queue_path}")
        return []
    with open(queue_path) as f:
        queue = json.load(f)
    pending = [j for j in queue if j.get("status") == "queued"]
    print(f"Found {len(pending)} pending jobs.")
    return pending


def generate_video(
    pipe: CogVideoXPipeline,
    prompt: str,
    output_path: Path,
    duration: int = 6,
) -> bool:
    """
    Generate a single video clip.

    Args:
        pipe: Loaded CogVideoX pipeline.
        prompt: Visual description.
        output_path: Where to save the .mp4.
        duration: Clip duration in seconds.

    Returns:
        True if successful.
    """
    try:
        num_frames = duration * FPS
        result = pipe(
            prompt=prompt,
            num_inference_steps=NUM_INFERENCE_STEPS,
            guidance_scale=GUIDANCE_SCALE,
            num_frames=num_frames,
        )
        frames = result.frames[0]
        export_to_video(frames, str(output_path), fps=FPS)
        print(f"  Saved: {output_path}")
        return True
    except Exception as exc:
        print(f"  ERROR generating video: {exc}")
        return False


def main():
    """Main entry point: load queue, generate videos, save results."""
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load the pipeline
    print("\nLoading CogVideoX-2b pipeline...")
    pipe = CogVideoXPipeline.from_pretrained(
        "THUDM/CogVideoX-2b",
        torch_dtype=torch.float16,
    )
    pipe.to("cuda")
    pipe.enable_model_cpu_offload()
    print("Pipeline loaded.")

    # Load queue
    jobs = load_queue(QUEUE_PATH)
    if not jobs:
        print("No jobs to process. Exiting.")
        return

    results = []
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for i, job in enumerate(jobs, 1):
        job_id = job.get("job_id", f"job_{i}")
        prompt = job.get("prompt", "")
        output_filename = job.get("output_filename", f"scene_{i:02d}.mp4")
        duration = int(job.get("duration", 6))
        output_path = OUTPUT_DIR / output_filename

        print(f"\n[{i}/{len(jobs)}] Job: {job_id}")
        print(f"  Prompt: {prompt[:80]}...")
        print(f"  Output: {output_filename}")

        success = generate_video(pipe, prompt, output_path, duration)

        results.append({
            "job_id": job_id,
            "output_filename": output_filename,
            "output_path": str(output_path),
            "status": "completed" if success else "failed",
            "run_id": job.get("run_id", ""),
            "scene_number": job.get("scene_number", 0),
        })

    # Save results manifest
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {RESULTS_FILE}")

    completed = sum(1 for r in results if r["status"] == "completed")
    failed = len(results) - completed
    print(f"\nSummary: {completed} completed, {failed} failed out of {len(results)} jobs.")
    print("\nNext steps:")
    print("1. Download all .mp4 files from /kaggle/working/")
    print("2. Download kaggle_results.json")
    print("3. Copy .mp4 files to data/outputs/[run_id]/scenes/")
    print("4. For each job_id, call: POST /providers/queue/complete")


if __name__ == "__main__":
    main()
