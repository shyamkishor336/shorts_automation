"""
Router: /review — Human review submission for Mode B (and Mode C) stages.
"""

import logging
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models import StageAttempt

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/review", tags=["review"])


# ── Pydantic schemas ───────────────────────────────────────────────────────

class ReviewSubmitRequest(BaseModel):
    """Request body for submitting a human review decision."""
    decision: str       # 'accept' or 'reject'
    notes: Optional[str] = None


class PendingReviewResponse(BaseModel):
    """Details of the next stage attempt awaiting human review."""
    id: str
    run_id: str
    stage_name: str
    attempt_number: int
    output_path: Optional[str] = None
    prompt_used: Optional[str] = None
    created_at: Optional[datetime] = None

    class Config:
        from_attributes = True


# ── Route handlers ─────────────────────────────────────────────────────────

@router.get("/pending", response_model=Optional[PendingReviewResponse],
            summary="Get next stage awaiting human review")
def get_pending_review(db: Session = Depends(get_db)):
    """
    Return the oldest stage attempt with status='pending_review'.
    Returns null if nothing is pending.
    """
    attempt = (
        db.query(StageAttempt)
        .filter(StageAttempt.status == "pending_review")
        .order_by(StageAttempt.created_at)
        .first()
    )
    if not attempt:
        return None

    return PendingReviewResponse(
        id=attempt.id,
        run_id=attempt.run_id,
        stage_name=attempt.stage_name,
        attempt_number=attempt.attempt_number,
        output_path=attempt.output_path,
        prompt_used=attempt.prompt_used,
        created_at=attempt.created_at,
    )


@router.post("/{attempt_id}", response_model=dict,
             summary="Submit a human review decision")
def submit_review(
    attempt_id: str,
    body: ReviewSubmitRequest,
    db: Session = Depends(get_db),
):
    """
    Submit an accept or reject decision for a stage attempt.
    The orchestrator polls this and picks it up automatically.
    """
    if body.decision not in ("accept", "reject"):
        raise HTTPException(
            status_code=400, detail="decision must be 'accept' or 'reject'"
        )

    attempt = (
        db.query(StageAttempt)
        .filter(StageAttempt.id == attempt_id)
        .first()
    )
    if not attempt:
        raise HTTPException(
            status_code=404, detail=f"Attempt {attempt_id} not found"
        )

    if attempt.status != "pending_review":
        raise HTTPException(
            status_code=400,
            detail=f"Attempt is not pending review (status: {attempt.status})",
        )

    attempt.human_decision = body.decision
    attempt.reviewer_notes = body.notes
    attempt.decision_timestamp = datetime.utcnow()
    # Update status immediately so the frontend can distinguish decided stages
    # from genuinely-pending ones. The orchestrator reads human_decision (not status)
    # to make its accept/reject routing decision, so this is safe.
    attempt.status = "accepted" if body.decision == "accept" else "rejected"
    db.commit()

    logger.info(
        f"Review submitted: attempt={attempt_id} decision={body.decision}"
    )
    return {
        "message": f"Decision '{body.decision}' recorded for attempt {attempt_id}."
    }


@router.post("/{attempt_id}/decide", response_model=dict,
             summary="Submit a human review decision (explicit /decide path)")
def submit_review_decide(
    attempt_id: str,
    body: ReviewSubmitRequest,
    db: Session = Depends(get_db),
):
    """Alias for POST /review/{attempt_id} — accepts decisions from the dashboard panel."""
    return submit_review(attempt_id, body, db)
