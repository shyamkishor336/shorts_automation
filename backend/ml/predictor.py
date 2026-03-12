"""
ML Predictor — singleton loader for all per-stage classifiers.

Called by the orchestrator (Mode C) to decide whether human review is needed.
Models and config are loaded once on first call and cached for the process lifetime.
"""

import json
import logging
import warnings
from pathlib import Path
from typing import Tuple

logger = logging.getLogger(__name__)

_MODEL_DIR = Path(__file__).parent / "models"

# Module-level caches — populated lazily on first use
_models: dict = {}        # stage_name -> loaded sklearn model (or None if missing)
_feature_cols: dict = {}  # stage_name -> list[str] from feature_columns.json
_config: dict | None = None  # classifier_config.json, loaded once


# ── Private loaders ───────────────────────────────────────────────────────────

def _load_config() -> dict:
    global _config
    if _config is None:
        config_path = _MODEL_DIR / "classifier_config.json"
        try:
            _config = json.loads(config_path.read_text(encoding="utf-8"))
            logger.info(f"Predictor: loaded classifier_config.json from {config_path}")
        except Exception as exc:
            logger.warning(f"Predictor: could not load classifier_config.json: {exc}")
            _config = {}
    return _config


def _load_model(stage_name: str):
    """Lazy-load and cache the sklearn model for a stage. Returns None if missing."""
    if stage_name not in _models:
        model_path = _MODEL_DIR / f"{stage_name}_classifier.pkl"
        if not model_path.exists():
            logger.warning(f"Predictor: model not found at {model_path}")
            _models[stage_name] = None
        else:
            try:
                import joblib
                _models[stage_name] = joblib.load(str(model_path))
                logger.info(f"Predictor: loaded {stage_name} classifier from {model_path}")
            except Exception as exc:
                logger.warning(f"Predictor: failed to load {model_path}: {exc}")
                _models[stage_name] = None
    return _models[stage_name]


def _load_feature_cols(stage_name: str) -> list:
    """Lazy-load and cache the feature column list for a stage."""
    if stage_name not in _feature_cols:
        fc_path = _MODEL_DIR / "feature_columns.json"
        try:
            all_cols = json.loads(fc_path.read_text(encoding="utf-8"))
            _feature_cols[stage_name] = all_cols.get(stage_name, [])
        except Exception as exc:
            logger.warning(f"Predictor: could not load feature_columns.json: {exc}")
            _feature_cols[stage_name] = []
    return _feature_cols[stage_name]


# ── Public API ────────────────────────────────────────────────────────────────

def should_request_human_review(
    stage_name: str,
    features: dict,
) -> Tuple[bool, float]:
    """
    Decide whether human review is needed for a stage output.

    Decision logic (checked in order):
      1. always_review=true in classifier_config  → (True,  1.0)  [e.g. video stage]
      2. use_classifier=false                     → (False, 0.0)  [auto-accept]
      3. Model missing                            → (True,  0.5)  [safe fallback]
      4. prob >= threshold                        → (True,  prob) [classifier says reject]
      5. prob <  threshold                        → (False, prob) [classifier says accept]

    Args:
        stage_name: One of 'script', 'audio', 'visual', 'video'.
        features:   Flat dict of feature_name -> value for this attempt.
                    Missing columns are filled with 0.5 (neutral / uncertain).

    Returns:
        (needs_review, rejection_probability)
    """
    config = _load_config()
    stage_cfg = config.get(stage_name, {})

    # ── Rule 1: always_review overrides everything ────────────────────────
    if stage_cfg.get("always_review", False):
        logger.info(
            f"Predictor [{stage_name}]: always_review=true "
            "-> human review required (classifier bypassed)"
        )
        return True, 1.0

    # ── Rule 2: classifier explicitly disabled ────────────────────────────
    if not stage_cfg.get("use_classifier", False):
        logger.info(f"Predictor [{stage_name}]: use_classifier=false -> auto-accept")
        return False, 0.0

    threshold = float(stage_cfg.get("threshold", 0.65))
    model = _load_model(stage_name)

    # ── Rule 3: model file missing ────────────────────────────────────────
    if model is None:
        logger.warning(
            f"Predictor [{stage_name}]: model unavailable, "
            "defaulting to human review (prob=0.5)"
        )
        return True, 0.5

    # ── Rules 4 & 5: run the classifier ──────────────────────────────────
    try:
        import pandas as pd

        feature_cols = _load_feature_cols(stage_name)
        # Fill missing feature values with 0.5 (neutral / mid-range)
        row = {col: features.get(col, 0.5) for col in feature_cols}
        df = pd.DataFrame([row])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            prob = float(model.predict_proba(df)[0][1])
        needs_review = prob >= threshold
        return needs_review, prob

    except Exception as exc:
        logger.warning(
            f"Predictor [{stage_name}]: prediction failed ({exc}), "
            "defaulting to human review (prob=0.5)"
        )
        return True, 0.5


def reset_cache() -> None:
    """Clear all in-memory caches. Useful for testing or after model retraining."""
    global _config
    _models.clear()
    _feature_cols.clear()
    _config = None
    logger.info("Predictor: model cache cleared.")
