"""
Database setup for the HITL AI Multimedia Pipeline.
Uses SQLAlchemy with PostgreSQL. Run this file directly to create all tables.

Usage (from the shorts_automation/ directory):
    python -m backend.database          # recommended
    python backend/database.py          # also works
"""

import logging
import sys
from pathlib import Path

# When run directly as a script, ensure the project root is on sys.path
# so that `from backend.config import settings` resolves correctly.
if __name__ == "__main__" or "backend.database" not in sys.modules:
    _root = Path(__file__).parent.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from backend.config import settings

logger = logging.getLogger(__name__)

engine = create_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    """FastAPI dependency: yields a database session and ensures it is closed."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Create all tables defined in models.py. Safe to call multiple times."""
    from backend import models  # noqa: F401 — import triggers table registration
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created / verified.")

    # ── Schema migrations: add new columns to existing tables ────────────
    # Uses IF NOT EXISTS so these are safe to run on every startup.
    from sqlalchemy import text
    _migrations = [
        "ALTER TABLE pipeline_runs ADD COLUMN IF NOT EXISTS current_stage VARCHAR(20)",
        "ALTER TABLE stage_attempts ADD COLUMN IF NOT EXISTS rejection_probability FLOAT",
    ]
    with engine.begin() as conn:
        for sql in _migrations:
            try:
                conn.execute(text(sql))
            except Exception as exc:
                logger.warning(f"Migration skipped ({sql[:60]}…): {exc}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    init_db()
    print("Database initialised successfully.")
