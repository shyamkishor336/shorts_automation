"""
Mode A Batch Runner — 120 runs (24 prompts × 5 repetitions).

Usage:
    python run_mode_a_batch.py

Requires the backend to be running:
    uvicorn backend.main:app --reload
"""

import time
import datetime
import requests

API_BASE = "http://localhost:8000"
POLL_INTERVAL = 15          # seconds between status polls
INTER_RUN_GAP = 5           # seconds between runs (GPU cooldown)
PROMPTS = list(range(1, 25))  # prompt_id 1 – 24
REPS = 5                    # repetitions per prompt
TOTAL = len(PROMPTS) * REPS
LOG_FILE = "batch_log.txt"


def log(msg: str) -> None:
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def start_run(prompt_id: int) -> str | None:
    """POST /runs/start and return the run_id, or None on failure."""
    try:
        resp = requests.post(
            f"{API_BASE}/runs/start",
            json={"prompt_id": prompt_id, "mode": "A", "video_provider_choice": "ken_burns"},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("run_id")
    except Exception as exc:
        log(f"ERROR starting run for prompt {prompt_id}: {exc}")
        return None


def poll_until_done(run_id: str, run_num: int, prompt_id: int, attempt: int) -> str:
    """Poll /runs/{run_id}/status every POLL_INTERVAL seconds until terminal state."""
    while True:
        try:
            resp = requests.get(f"{API_BASE}/runs/{run_id}/status", timeout=15)
            resp.raise_for_status()
            status = resp.json().get("status", "unknown")
        except Exception as exc:
            log(f"  Poll error for run {run_id}: {exc}")
            status = "unknown"

        print(
            f"\rRun {run_num}/{TOTAL} | Prompt {prompt_id} | "
            f"Attempt {attempt} | Status: {status}   ",
            end="",
            flush=True,
        )

        if status in ("completed", "failed", "stopped"):
            print()  # newline after the carriage-return line
            return status

        time.sleep(POLL_INTERVAL)


def main() -> None:
    log(f"=== Mode A Batch started — {TOTAL} runs ({len(PROMPTS)} prompts × {REPS} reps) ===")

    run_num = 0
    for prompt_id in PROMPTS:
        for rep in range(1, REPS + 1):
            run_num += 1

            log(f"Starting run {run_num}/{TOTAL} | Prompt {prompt_id} | Rep {rep}")

            run_id = start_run(prompt_id)
            if run_id is None:
                log(f"FAILED to start run {run_num}/{TOTAL} (prompt {prompt_id} rep {rep}) — skipping")
                continue

            log(f"  run_id={run_id}")

            final_status = poll_until_done(run_id, run_num, prompt_id, rep)

            log(
                f"DONE run {run_num}/{TOTAL} | prompt_id={prompt_id} | rep={rep} "
                f"| run_id={run_id} | status={final_status}"
            )

            if run_num < TOTAL:
                log(f"  Waiting {INTER_RUN_GAP}s before next run…")
                time.sleep(INTER_RUN_GAP)

    log(f"=== Batch complete — {run_num} runs processed ===")


if __name__ == "__main__":
    main()
