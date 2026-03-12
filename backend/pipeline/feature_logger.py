"""
Feature Logger — Extracts stage features and persists them to the database.
Central utility used by all pipeline stages.
"""

import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

from sqlalchemy.orm import Session

from backend.models import StageAttempt

logger = logging.getLogger(__name__)


def save_script_features(
    db: Session,
    attempt: StageAttempt,
    features: Dict[str, Any],
) -> None:
    """
    Persist Stage 1 (script) features to a StageAttempt record.

    Args:
        db: Active SQLAlchemy session.
        attempt: The StageAttempt ORM object to update.
        features: Dict from backend.features.script_features.extract_all().
    """
    try:
        attempt.readability_score = features.get("readability_score")
        attempt.lexical_diversity = features.get("lexical_diversity")
        attempt.prompt_coverage = features.get("prompt_coverage")
        attempt.sentence_redundancy = features.get("sentence_redundancy")
        attempt.entity_consistency = features.get("entity_consistency")
        attempt.topic_coherence = features.get("topic_coherence")
        attempt.factual_conflict_flag = features.get("factual_conflict_flag")
        attempt.prompt_ambiguity = features.get("prompt_ambiguity")
        db.commit()
        logger.info(f"Script features saved for attempt {attempt.id}")
    except Exception as exc:
        db.rollback()
        logger.error(f"Failed to save script features for attempt {attempt.id}: {exc}")


def save_audio_features(
    db: Session,
    attempt: StageAttempt,
    features: Dict[str, Any],
) -> None:
    """
    Persist Stage 2 (audio) features to a StageAttempt record.

    Args:
        db: Active SQLAlchemy session.
        attempt: The StageAttempt ORM object to update.
        features: Dict from backend.features.audio_features.extract_all().
    """
    try:
        attempt.phoneme_error_rate = features.get("phoneme_error_rate")
        attempt.silence_ratio = features.get("silence_ratio")
        attempt.speaking_rate_variance = features.get("speaking_rate_variance")
        attempt.energy_variance = features.get("energy_variance")
        attempt.tts_word_count_match = features.get("tts_word_count_match")
        db.commit()
        logger.info(f"Audio features saved for attempt {attempt.id}")
    except Exception as exc:
        db.rollback()
        logger.error(f"Failed to save audio features for attempt {attempt.id}: {exc}")


def save_visual_features(
    db: Session,
    attempt: StageAttempt,
    features: Dict[str, Any],
) -> None:
    """
    Persist Stage 3 (visual) features to a StageAttempt record.

    Args:
        db: Active SQLAlchemy session.
        attempt: The StageAttempt ORM object to update.
        features: Dict from backend.features.visual_features.extract_all().
    """
    try:
        attempt.clip_similarity = features.get("clip_similarity")
        attempt.aesthetic_score = features.get("aesthetic_score")
        attempt.blur_score = features.get("blur_score")
        attempt.object_match_score = features.get("object_match_score")
        attempt.colour_tone_match = features.get("colour_tone_match")
        attempt.visual_provider = features.get("visual_provider")
        db.commit()
        logger.info(f"Visual features saved for attempt {attempt.id}")
    except Exception as exc:
        db.rollback()
        logger.error(f"Failed to save visual features for attempt {attempt.id}: {exc}")


def save_video_features(
    db: Session,
    attempt: StageAttempt,
    features: Dict[str, Any],
) -> None:
    """
    Persist Stage 4 (video) features to a StageAttempt record.

    Args:
        db: Active SQLAlchemy session.
        attempt: The StageAttempt ORM object to update.
        features: Dict from backend.features.video_features.extract_all().
    """
    try:
        attempt.av_sync_error_ms = features.get("av_sync_error_ms")
        attempt.transition_smoothness = features.get("transition_smoothness")
        attempt.duration_deviation_s = features.get("duration_deviation_s")
        db.commit()
        logger.info(f"Video features saved for attempt {attempt.id}")
    except Exception as exc:
        db.rollback()
        logger.error(f"Failed to save video features for attempt {attempt.id}: {exc}")


def update_cross_stage_features(
    db: Session,
    attempt: StageAttempt,
    prior_corrections: int,
    cumulative_risk: Optional[float],
    api_retry_count: int = 0,
    is_fallback: bool = False,
) -> None:
    """
    Update cross-stage features on a StageAttempt.

    Args:
        db: Active SQLAlchemy session.
        attempt: The StageAttempt ORM object to update.
        prior_corrections: Number of corrections made in prior stages.
        cumulative_risk: Sum of rejection probabilities from prior stages.
        api_retry_count: Number of API retries used for this attempt.
        is_fallback: Whether a fallback provider (Ken Burns) was used.
    """
    try:
        attempt.prior_stage_corrections = prior_corrections
        attempt.cumulative_risk_score = cumulative_risk
        attempt.api_retry_count = api_retry_count
        attempt.is_fallback_video = is_fallback
        db.commit()
    except Exception as exc:
        db.rollback()
        logger.error(
            f"Failed to update cross-stage features for attempt {attempt.id}: {exc}"
        )
