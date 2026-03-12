"""
SQLAlchemy ORM models for the HITL AI Multimedia Pipeline.
These models map directly to the PostgreSQL schema specified in Section 3.
"""

import uuid
from datetime import datetime
from sqlalchemy import (
    Boolean,
    Column,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    DateTime,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from backend.database import Base


def _uuid() -> str:
    """Generate a new UUID string."""
    return str(uuid.uuid4())


class PipelineRun(Base):
    """Represents a single end-to-end pipeline execution."""

    __tablename__ = "pipeline_runs"

    id = Column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    prompt_id = Column(Integer, nullable=False)
    prompt_text = Column(Text, nullable=False)
    mode = Column(String(1), nullable=False)            # 'A', 'B', or 'C'
    status = Column(String(20), nullable=False, default="running")
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    final_video_path = Column(Text, nullable=True)
    total_corrections = Column(Integer, default=0)
    video_provider_used = Column(String(30), nullable=True)
    video_provider_choice = Column(String(20), nullable=True)  # 'modal' or 'ken_burns'
    gdrive_file_id = Column(Text, nullable=True)
    current_stage = Column(String(20), nullable=True)   # last stage initiated (script/audio/visual/video)

    # Relationship
    stage_attempts = relationship(
        "StageAttempt", back_populates="run", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<PipelineRun id={self.id} prompt_id={self.prompt_id} mode={self.mode} status={self.status}>"


class StageAttempt(Base):
    """Represents one attempt at a single pipeline stage within a run."""

    __tablename__ = "stage_attempts"

    id = Column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    run_id = Column(
        UUID(as_uuid=False), ForeignKey("pipeline_runs.id"), nullable=False
    )
    stage_name = Column(String(20), nullable=False)     # 'script', 'audio', 'visual', 'video'
    attempt_number = Column(Integer, default=1)
    mode = Column(String(1), nullable=False)
    status = Column(String(20), nullable=False, default="pending")
    output_path = Column(Text, nullable=True)
    prompt_used = Column(Text, nullable=True)
    human_decision = Column(String(10), nullable=True)  # 'accept' or 'reject'
    decision_timestamp = Column(DateTime, nullable=True)
    reviewer_notes = Column(Text, nullable=True)

    # ── SCRIPT FEATURES (Stage 1) ─────────────────────────────────────────
    readability_score = Column(Float, nullable=True)
    lexical_diversity = Column(Float, nullable=True)
    prompt_coverage = Column(Float, nullable=True)
    sentence_redundancy = Column(Float, nullable=True)
    entity_consistency = Column(Float, nullable=True)
    topic_coherence = Column(Float, nullable=True)
    factual_conflict_flag = Column(Integer, nullable=True)
    prompt_ambiguity = Column(Float, nullable=True)

    # ── AUDIO FEATURES (Stage 2) ──────────────────────────────────────────
    phoneme_error_rate = Column(Float, nullable=True)
    silence_ratio = Column(Float, nullable=True)
    speaking_rate_variance = Column(Float, nullable=True)
    energy_variance = Column(Float, nullable=True)
    tts_word_count_match = Column(Float, nullable=True)

    # ── VISUAL FEATURES (Stage 3) ─────────────────────────────────────────
    clip_similarity = Column(Float, nullable=True)
    aesthetic_score = Column(Float, nullable=True)
    blur_score = Column(Float, nullable=True)
    object_match_score = Column(Float, nullable=True)
    colour_tone_match = Column(Float, nullable=True)
    visual_provider = Column(String(30), nullable=True)

    # ── VIDEO FEATURES (Stage 4) ──────────────────────────────────────────
    av_sync_error_ms = Column(Float, nullable=True)
    transition_smoothness = Column(Float, nullable=True)
    duration_deviation_s = Column(Float, nullable=True)

    # ── CROSS-STAGE FEATURES ──────────────────────────────────────────────
    prior_stage_corrections = Column(Integer, nullable=True)
    cumulative_risk_score = Column(Float, nullable=True)
    api_retry_count = Column(Integer, default=0)
    is_fallback_video = Column(Boolean, default=False)

    # ── MODE C ML OUTPUT ──────────────────────────────────────────────────
    rejection_probability = Column(Float, nullable=True)  # predicted P(reject) from classifier

    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationship
    run = relationship("PipelineRun", back_populates="stage_attempts")

    def __repr__(self) -> str:
        return (
            f"<StageAttempt id={self.id} stage={self.stage_name} "
            f"attempt={self.attempt_number} status={self.status}>"
        )
