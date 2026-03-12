"""
Router: /export — Export ML training data as CSV.
"""

import io
import logging

import pandas as pd
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models import StageAttempt

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/export", tags=["export"])

# All feature columns to include in the export
FEATURE_COLUMNS = [
    "id", "run_id", "stage_name", "attempt_number", "mode",
    # Script features
    "readability_score", "lexical_diversity", "prompt_coverage",
    "sentence_redundancy", "entity_consistency", "topic_coherence",
    "factual_conflict_flag", "prompt_ambiguity",
    # Audio features
    "phoneme_error_rate", "silence_ratio", "speaking_rate_variance",
    "energy_variance", "tts_word_count_match",
    # Visual features
    "clip_similarity", "aesthetic_score", "blur_score",
    "object_match_score", "colour_tone_match", "visual_provider",
    # Video features
    "av_sync_error_ms", "transition_smoothness", "duration_deviation_s",
    # Cross-stage
    "prior_stage_corrections", "cumulative_risk_score",
    "api_retry_count", "is_fallback_video",
    # Label
    "human_decision",
    "created_at",
]


@router.get("/csv", summary="Export Mode B training data as CSV")
def export_csv(db: Session = Depends(get_db)):
    """
    Export all stage_attempts rows where human_decision is not null
    (Mode B data only) as a CSV file for ML training.
    """
    attempts = (
        db.query(StageAttempt)
        .filter(StageAttempt.human_decision.isnot(None))
        .all()
    )

    if not attempts:
        # Return empty CSV with headers
        df = pd.DataFrame(columns=FEATURE_COLUMNS)
    else:
        rows = []
        for a in attempts:
            row = {col: getattr(a, col, None) for col in FEATURE_COLUMNS}
            rows.append(row)
        df = pd.DataFrame(rows)

    # Stream as CSV response
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)

    logger.info(f"Exported {len(df)} rows of ML training data.")

    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={
            "Content-Disposition": "attachment; filename=hitl_training_data.csv"
        },
    )
