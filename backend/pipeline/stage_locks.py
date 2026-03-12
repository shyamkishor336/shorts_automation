"""
Pipeline Stage Locks — Assembly-line concurrency control.

Stages 1, 2, 4 (script / audio / video): threading.Lock — one run at a time.
Stage  3 (visual):
    Modal provider   → threading.Semaphore(5)  — up to 5 concurrent cloud calls
    Ken Burns        → threading.Semaphore(1)  — 1 run at a time (local GPU)

All primitives are threading-based (not asyncio) because the orchestrator
runs in background daemon threads.

Usage in orchestrator:
    acquire_stage("visual", run_id, provider="modal")  # blocks if 5 slots full
    ... generate ...
    release_stage("visual", run_id, provider="modal")  # BEFORE human review wait
"""

import logging
import threading
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Stage 1 / 2 / 4 — single-slot locks ───────────────────────────────────

_locks: Dict[str, threading.Lock] = {
    "script": threading.Lock(),
    "audio":  threading.Lock(),
    "video":  threading.Lock(),
}

_holders: Dict[str, Optional[str]] = {
    "script": None,
    "audio":  None,
    "video":  None,
}
_holders_guard = threading.Lock()

# ── Stage 3 (visual) — provider-aware semaphores ───────────────────────────
# MODAL_CONCURRENT_RUNS=5  (documented in .env)

_visual_semaphores: Dict[str, threading.Semaphore] = {
    "modal":     threading.Semaphore(5),   # up to 5 concurrent Modal cloud calls
    "ken_burns": threading.Semaphore(1),   # 1 local GPU run at a time
}

_visual_holders: List[str] = []           # all run_ids currently in visual stage
_visual_holders_guard = threading.Lock()


# ── Public API ─────────────────────────────────────────────────────────────

def acquire_stage(stage_name: str, run_id: str, provider: str = "ken_burns") -> None:
    """
    Block until a slot is available for this stage, then claim it.

    For the visual stage, `provider` selects which semaphore to use:
        'modal'     → up to 5 concurrent
        'ken_burns' → 1 at a time (local GPU)
    For all other stages, `provider` is ignored.
    """
    logger.info(f"[LOCK] {run_id[:8]} queued  for  '{stage_name}' slot (provider={provider})")

    if stage_name == "visual":
        sem = _visual_semaphores.get(provider, _visual_semaphores["ken_burns"])
        sem.acquire()
        with _visual_holders_guard:
            _visual_holders.append(run_id)
    else:
        _locks[stage_name].acquire()
        with _holders_guard:
            _holders[stage_name] = run_id

    logger.info(f"[LOCK] {run_id[:8]} entered '{stage_name}' slot (provider={provider})")


def release_stage(stage_name: str, run_id: str, provider: str = "ken_burns") -> None:
    """
    Release the stage slot so the next queued run can proceed.
    Safe to call multiple times (extra calls are silently ignored).
    """
    if stage_name == "visual":
        with _visual_holders_guard:
            if run_id not in _visual_holders:
                return   # already released
            _visual_holders.remove(run_id)
        sem = _visual_semaphores.get(provider, _visual_semaphores["ken_burns"])
        try:
            sem.release()
            logger.info(f"[LOCK] {run_id[:8]} exited  '{stage_name}' slot (provider={provider})")
        except ValueError:
            pass  # semaphore already at max — ignore
    else:
        with _holders_guard:
            if _holders.get(stage_name) != run_id:
                return   # already released or held by another run
            _holders[stage_name] = None
        try:
            _locks[stage_name].release()
            logger.info(f"[LOCK] {run_id[:8]} exited  '{stage_name}' slot")
        except RuntimeError:
            pass  # lock was not locked — already released


def get_stage_status() -> dict:
    """
    Return a snapshot of which stage slots are currently occupied.

    Returns:
        {
          "script":  run_id | None,
          "audio":   run_id | None,
          "visual":  [run_id, ...],   # list (up to 5 for modal)
          "video":   run_id | None,
        }
    """
    with _holders_guard:
        status: dict = dict(_holders)
    with _visual_holders_guard:
        status["visual"] = list(_visual_holders)   # [] when none active
    return status
