#!/usr/bin/env python3
"""
GCP CogVideoX-2b Batch Generation Script.
Run this MANUALLY on a GCP VM with T4 GPU using the $300 free credit.

# HOW TO USE GCP $300 CREDITS:
# 1. Go to console.cloud.google.com and sign up (new account = $300 free credit)
# 2. Create a VM instance:
#    - Compute Engine → VM Instances → Create Instance
#    - Name: hitl-cogvideox-vm
#    - Machine type: n1-standard-4 (4 vCPU, 15GB RAM)
#    - GPU: Add GPU → NVIDIA T4 (1x)
#    - Boot disk: Deep Learning on Linux, 50GB SSD
#    - Region/Zone: us-central1-a (cheapest with T4)
#    - Allow HTTP/HTTPS traffic
# 3. SSH into the VM:
#    gcloud compute ssh hitl-cogvideox-vm --zone=us-central1-a
# 4. Install dependencies:
#    pip install diffusers>=0.30.1 transformers>=4.44.2 accelerate imageio-ffmpeg sentencepiece
# 5. Upload this script and gcp_queue.json to the VM:
#    gcloud compute scp gcp_cogvideox_batch.py gcp_queue.json hitl-cogvideox-vm:~/ --zone=us-central1-a
# 6. Run the script:
#    python gcp_cogvideox_batch.py
# 7. Download results:
#    gcloud compute scp hitl-cogvideox-vm:~/gcp_outputs/* data/outputs/ --zone=us-central1-a --recurse
# 8. IMPORTANT: Stop the VM immediately after to avoid charges:
#    gcloud compute instances stop hitl-cogvideox-vm --zone=us-central1-a
#
# Cost estimate: n1-standard-4 + T4 GPU ≈ $0.40/hour
# 500 clips at 5 clips/hr = 100 hours × $0.40 = $40 (well within $300 credit)
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime

# ── Install dependencies (run once on fresh VM) ─────────────────────────────
def install_dependencies():
    """Install required packages if not already installed."""
    try:
        import diffusers
    except ImportError:
        print("Installing dependencies...")
        subprocess.run([
            sys.executable, "-m", "pip", "install",
            "diffusers>=0.30.1",
            "transformers>=4.44.2",
            "accelerate",
            "imageio-ffmpeg",
            "sentencepiece",
            "torch",
        ], check=True)


install_dependencies()

import torch
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video

# ── Configuration ────────────────────────────────────────────────────────────
QUEUE_FILE = Path("gcp_queue.json")
OUTPUT_DIR = Path("gcp_outputs")
RESULTS_FILE = Path("gcp_results.json")

NUM_INFERENCE_STEPS = 25
GUIDANCE_SCALE = 6.0
FPS = 8


def load_queue() -> list:
    """Load pending jobs from gcp_queue.json."""
    if not QUEUE_FILE.exists():
        print(f"Queue file not found: {QUEUE_FILE}")
        return []

    with open(QUEUE_FILE) as f:
        queue = json.load(f)

    pending = [j for j in queue if j.get("status") == "queued"]
    print(f"Found {len(pending)} pending jobs out of {len(queue)} total.")
    return pending


def load_pipeline() -> CogVideoXPipeline:
    """Load CogVideoX-2b pipeline onto GPU."""
    print("Loading CogVideoX-2b pipeline (this may take a few minutes)...")
    pipe = CogVideoXPipeline.from_pretrained(
        "THUDM/CogVideoX-2b",
        torch_dtype=torch.float16,
    )
    pipe.to("cuda")
    pipe.enable_model_cpu_offload()
    print("Pipeline loaded successfully.")
    return pipe


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
        file_size = output_path.stat().st_size / (1024 * 1024)
        print(f"  Saved: {output_path.name} ({file_size:.1f} MB)")
        return True
    except torch.cuda.OutOfMemoryError:
        print("  ERROR: GPU out of memory. Clearing cache and skipping.")
        torch.cuda.empty_cache()
        return False
    except Exception as exc:
        print(f"  ERROR: {exc}")
        return False


def main():
    """Main entry point."""
    print("="*60)
    print("GCP CogVideoX-2b Batch Generation")
    print("="*60)
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU Memory: {mem_gb:.1f} GB")
    print()

    jobs = load_queue()
    if not jobs:
        print("No pending jobs. Exiting.")
        return

    pipe = load_pipeline()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    start_time = datetime.now()

    for i, job in enumerate(jobs, 1):
        job_id = job.get("job_id", f"job_{i}")
        prompt = job.get("prompt", "")
        output_filename = job.get("output_filename", f"scene_{i:02d}.mp4")
        duration = int(job.get("duration", 6))
        output_path = OUTPUT_DIR / output_filename

        elapsed = (datetime.now() - start_time).total_seconds()
        rate = (i - 1) / elapsed * 3600 if elapsed > 0 else 0
        print(f"\n[{i}/{len(jobs)}] Job: {job_id} | Rate: {rate:.1f} clips/hr")
        print(f"  Prompt: {prompt[:80]}...")

        success = generate_video(pipe, prompt, output_path, duration)
        results.append({
            "job_id": job_id,
            "output_filename": output_filename,
            "output_path": str(output_path),
            "status": "completed" if success else "failed",
            "run_id": job.get("run_id", ""),
            "scene_number": job.get("scene_number", 0),
            "completed_at": datetime.now().isoformat(),
        })

        # Save results after each job (in case of interruption)
        with open(RESULTS_FILE, "w") as f:
            json.dump(results, f, indent=2)

    # Final summary
    total_time = (datetime.now() - start_time).total_seconds() / 3600
    completed = sum(1 for r in results if r["status"] == "completed")
    failed = len(results) - completed

    print("\n" + "="*60)
    print(f"BATCH COMPLETE")
    print(f"  Completed: {completed}/{len(results)}")
    print(f"  Failed:    {failed}/{len(results)}")
    print(f"  Duration:  {total_time:.2f} hours")
    print(f"  Avg speed: {len(results)/total_time:.1f} clips/hour")
    print(f"\nResults saved to: {RESULTS_FILE}")
    print(f"Videos saved to:  {OUTPUT_DIR}/")
    print()
    print("Next steps:")
    print("1. Run: gcloud compute scp hitl-cogvideox-vm:~/gcp_outputs/* data/outputs/ \\")
    print("         --zone=us-central1-a --recurse")
    print("2. Download gcp_results.json")
    print("3. Copy videos to their run directories: data/outputs/[run_id]/scenes/")
    print("4. For each completed job, POST /providers/queue/complete")
    print("5. STOP THE VM: gcloud compute instances stop hitl-cogvideox-vm --zone=us-central1-a")


if __name__ == "__main__":
    main()
