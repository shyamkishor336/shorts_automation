"""
Router: /providers — Provider status for Modal and Ken Burns.
"""

import logging

from fastapi import APIRouter, HTTPException

from backend.providers.provider_manager import ProviderManager

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/providers", tags=["providers"])

provider_manager = ProviderManager()


# ── Route handlers ─────────────────────────────────────────────────────────

@router.get("/status", summary="Get provider status")
def get_provider_status():
    """Return the current status of Modal and Ken Burns providers."""
    return provider_manager.get_budget_status()


@router.post("/{name}/reset", response_model=dict,
             summary="Reset provider exhausted flag")
def reset_provider(name: str):
    """
    Reset a provider's exhausted flag.
    Only 'modal' and 'ken_burns' are valid.
    """
    valid_providers = ["modal", "ken_burns"]
    if name not in valid_providers:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown provider: {name}. Valid: {valid_providers}",
        )

    success = provider_manager.reset_provider(name)
    if not success:
        raise HTTPException(status_code=404, detail=f"Provider {name} not found.")

    return {"message": f"Provider '{name}' reset."}
