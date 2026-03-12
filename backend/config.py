"""
Configuration module for the HITL AI Multimedia Pipeline.
Loads all settings from the .env file using python-dotenv.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")


class Settings:
    """Central settings object populated from environment variables."""

    # API Keys
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    HUGGINGFACE_TOKEN: str = os.getenv("HUGGINGFACE_TOKEN", "")
    MODAL_ENDPOINT_URL: str = os.getenv("MODAL_ENDPOINT_URL", "")
    FAL_KEY: str = os.getenv("FAL_KEY", "")

    # Database
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        "postgresql://postgres:password@localhost:5432/hitl_pipeline",
    )

    # File paths
    BASE_DIR: Path = Path(__file__).parent.parent
    OUTPUT_DIR: Path = Path(os.getenv("OUTPUT_DIR", str(BASE_DIR / "data" / "outputs")))
    DATA_DIR: Path = BASE_DIR / "data"
    PROMPTS_FILE: Path = DATA_DIR / "prompts.json"
    PROVIDER_BUDGET_FILE: Path = DATA_DIR / "provider_budget.json"
    PROVIDER_STATE_FILE: Path = DATA_DIR / "provider_state.json"
    GCP_QUEUE_FILE: Path = DATA_DIR / "gcp_queue.json"
    KAGGLE_QUEUE_FILE: Path = DATA_DIR / "kaggle_queue.json"
    ML_MODEL_DIR: Path = BASE_DIR / "backend" / "ml"

    # Pipeline settings
    DEFAULT_MODE: str = os.getenv("DEFAULT_MODE", "B")
    SCENE_COUNT: int = int(os.getenv("SCENE_COUNT", "8"))
    SCENE_DURATION_SECONDS: float = float(os.getenv("SCENE_DURATION_SECONDS", "6.67"))
    MAX_RETRIES_PER_STAGE: int = int(os.getenv("MAX_RETRIES_PER_STAGE", "3"))
    HUMAN_REVIEW_TIMEOUT_SECONDS: int = int(
        os.getenv("HUMAN_REVIEW_TIMEOUT_SECONDS", "300")
    )
    ML_REJECTION_THRESHOLD: float = float(
        os.getenv("ML_REJECTION_THRESHOLD", "0.65")
    )

    # Prompts that use Modal for video generation (all others use Ken Burns)
    DEMO_PROMPT_IDS: list = [1, 2, 3, 4, 5]

    # TTS settings
    TTS_VOICE: str = "en-US-AriaNeural"

    # Video settings — vertical 9:16 for Shorts
    VIDEO_WIDTH: int = 1080
    VIDEO_HEIGHT: int = 1920
    VIDEO_FPS: int = 24
    VIDEO_CODEC: str = "libx264"
    AUDIO_CODEC: str = "aac"

    # HF model IDs
    HF_FLUX_MODEL: str = "black-forest-labs/FLUX.1-schnell"
    HF_COGVIDEOX_MODEL: str = "THUDM/CogVideoX-5b"
    CLIP_MODEL: str = "openai/clip-vit-base-patch32"
    SENTENCE_TRANSFORMER_MODEL: str = "all-MiniLM-L6-v2"
    NLI_MODEL: str = "cross-encoder/nli-deberta-v3-small"


settings = Settings()
