"""
ML Classifier Evaluation — Loads saved models and evaluates on test data.
Generates confusion matrices, ROC curves, and a full classification report.

Usage:
    python backend/ml/evaluate_classifier.py
"""

import json
import logging
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from backend.config import settings
from backend.ml.train_classifier import STAGE_FEATURES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_stage_model(
    df: pd.DataFrame,
    stage_name: str,
    model_dir: Path,
) -> dict:
    """
    Load the saved model for a stage, evaluate on held-out test set,
    and generate visual reports.

    Args:
        df: Full Mode B dataframe.
        stage_name: One of 'script', 'audio', 'visual', 'video'.
        model_dir: Directory containing model_{stage}.pkl files.

    Returns:
        Dict with evaluation metrics and confusion matrix data.
    """
    model_path = model_dir / f"model_{stage_name}.pkl"
    if not model_path.exists():
        logger.warning(f"Model not found: {model_path}. Skipping.")
        return {}

    model = joblib.load(str(model_path))
    stage_df = df[df["stage_name"] == stage_name].copy()
    feature_cols = STAGE_FEATURES[stage_name]

    if len(stage_df) < 10:
        logger.warning(f"Insufficient data for stage '{stage_name}'.")
        return {}

    X = stage_df[feature_cols].copy()
    y = (stage_df["human_decision"] == "reject").astype(int)
    valid_cols = [c for c in feature_cols if X[c].notna().sum() > 0]
    X = X[valid_cols].fillna(X[valid_cols].median())

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if y.sum() > 1 else None
    )

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print(f"\n{'='*60}")
    print(f"Evaluation: {stage_name}")
    print("="*60)
    print(classification_report(y_test, y_pred, target_names=["accept", "reject"]))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=["accept", "reject"]).plot(ax=ax)
    ax.set_title(f"Confusion Matrix — {stage_name}")
    cm_path = model_dir / f"confusion_matrix_{stage_name}.png"
    fig.savefig(str(cm_path), dpi=150)
    plt.close(fig)
    logger.info(f"Confusion matrix saved: {cm_path}")

    # ROC curve
    roc_path = None
    if y_test.sum() > 0:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
        ax.plot([0, 1], [0, 1], "k--")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC Curve — {stage_name}")
        ax.legend()
        roc_path = model_dir / f"roc_curve_{stage_name}.png"
        fig.savefig(str(roc_path), dpi=150)
        plt.close(fig)
        logger.info(f"ROC curve saved: {roc_path}")
    else:
        auc = 0.5

    return {
        "stage": stage_name,
        "test_samples": int(len(y_test)),
        "roc_auc": round(float(auc), 4),
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_path": str(cm_path),
        "roc_curve_path": str(roc_path) if roc_path else None,
    }


def main() -> None:
    """Main evaluation entry point."""
    model_dir = settings.ML_MODEL_DIR
    engine = create_engine(settings.DATABASE_URL)

    query = """
        SELECT *
        FROM stage_attempts
        WHERE mode = 'B'
          AND human_decision IS NOT NULL
    """
    df = pd.read_sql(query, engine)
    logger.info(f"Loaded {len(df)} rows for evaluation.")

    if len(df) == 0:
        logger.error("No Mode B data found.")
        sys.exit(1)

    all_results = {}

    for stage in ["script", "audio", "visual", "video"]:
        result = evaluate_stage_model(df, stage, model_dir)
        if result:
            all_results[stage] = result

    # Save evaluation results
    results_path = model_dir / "evaluation_results.json"
    results_path.write_text(
        json.dumps(all_results, indent=2), encoding="utf-8"
    )
    logger.info(f"Evaluation results saved to {results_path}")

    print("\nEvaluation complete. Results:")
    print(json.dumps(all_results, indent=2))


if __name__ == "__main__":
    main()
