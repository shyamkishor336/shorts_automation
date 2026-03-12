"""
ML Classifier Training — Trains intervention prediction models on Mode B data.
Reads training data from mode_b_export.csv (not the live database).

Usage:
    python -m backend.ml.train_classifier

Outputs (per stage, saved to backend/ml/models/):
    {stage}_classifier.pkl            — Best trained classifier
    feature_importance_{stage}.png    — Feature importance chart
    training_results.json             — All metrics
"""

import json
import logging
import sys
from pathlib import Path

# Force UTF-8 stdout so Unicode chars don't crash on Windows cp1252 terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import joblib
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe on Windows servers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from backend.config import settings

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logging.warning("XGBoost not available — will train LR and RF only.")

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# ── CSV path ────────────────────────────────────────────────────────────────
CSV_PATH = Path(__file__).parent.parent.parent / "mode_b_export.csv"

# ── Feature sets per stage (stage-specific + cross-stage) ───────────────────
STAGE_FEATURES = {
    "script": [
        "readability_score", "lexical_diversity", "prompt_coverage",
        "sentence_redundancy", "entity_consistency", "topic_coherence",
        "factual_conflict_flag", "prompt_ambiguity",
        # cross-stage
        "prior_stage_corrections", "cumulative_risk_score", "api_retry_count",
    ],
    "audio": [
        "phoneme_error_rate", "silence_ratio", "speaking_rate_variance",
        "energy_variance", "tts_word_count_match",
        # cross-stage
        "prior_stage_corrections", "cumulative_risk_score", "api_retry_count",
    ],
    "visual": [
        "clip_similarity", "aesthetic_score", "blur_score",
        "object_match_score", "colour_tone_match",
        # cross-stage
        "prior_stage_corrections", "cumulative_risk_score", "api_retry_count",
    ],
    "video": [
        "av_sync_error_ms", "transition_smoothness", "duration_deviation_s",
        # cross-stage
        "prior_stage_corrections", "cumulative_risk_score", "api_retry_count",
    ],
}


# ── Data loading ────────────────────────────────────────────────────────────

def load_mode_b_data() -> pd.DataFrame:
    """
    Load Mode B stage attempts from the CSV export file.
    Filters to rows with a recorded human decision.
    Label encoding: accept → 1, reject → 0.
    """
    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"CSV not found at {CSV_PATH}. "
            "Export Mode B data first with: python -m backend.scripts.export_mode_b"
        )

    df = pd.read_csv(str(CSV_PATH))

    print("\n" + "=" * 60)
    print(f"CSV loaded from: {CSV_PATH}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print("\nFirst 3 rows:")
    print(df.head(3).to_string())
    print("=" * 60 + "\n")

    # Filter Mode B only (safety net — CSV should already be all B)
    df = df[df["mode"] == "B"].copy()
    logger.info(f"After Mode B filter: {len(df)} rows")

    # Keep only rows with a human decision
    df = df[df["human_decision"].notna()].copy()
    logger.info(f"Rows with human decision: {len(df)}")

    # Encode label: reject → 1, accept → 0
    df["label"] = (df["human_decision"] == "reject").astype(int)

    logger.info(f"Label distribution:\n{df['label'].value_counts().rename({1: 'reject(1)', 0: 'accept(0)'}).to_string()}")
    logger.info(f"Stage distribution:\n{df['stage_name'].value_counts().to_string()}")

    return df


# ── Per-stage training ───────────────────────────────────────────────────────

def train_stage_models(
    df: pd.DataFrame,
    stage_name: str,
    output_dir: Path,
) -> tuple:
    """
    Train and evaluate LR, Random Forest, and XGBoost classifiers for one stage.
    Saves the best model (by F1) to output_dir/{stage_name}_classifier.pkl.
    Returns (results_dict, valid_feature_cols).
    """
    stage_df = df[df["stage_name"] == stage_name].copy()
    feature_cols = STAGE_FEATURES[stage_name]

    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  STAGE: {stage_name.upper()}  —  {len(stage_df)} rows")
    print(sep)

    if len(stage_df) < 10:
        logger.warning(f"Insufficient data for stage '{stage_name}' (need ≥10). Skipping.")
        return {}, []

    # ── Features ──────────────────────────────────────────────────────────
    # Only include columns that actually exist in the DataFrame
    available_cols = [c for c in feature_cols if c in stage_df.columns]
    X = stage_df[available_cols].copy()

    # Drop columns that are entirely NaN (no information)
    valid_cols = [c for c in available_cols if X[c].notna().sum() > 0]
    X = X[valid_cols].copy()

    # Fill remaining NaN with column MEAN (not median) — keeps all rows
    col_means = X.mean()
    X = X.fillna(col_means)

    y = stage_df["label"].values

    print(f"Features ({len(valid_cols)}): {valid_cols}")
    n_reject = int((y == 1).sum())
    n_accept = int((y == 0).sum())
    print(f"Class distribution: reject(1)={n_reject}  accept(0)={n_accept}")

    if len(valid_cols) == 0:
        logger.warning("No valid feature columns. Skipping.")
        return {}, []

    # ── Train/test split — stratified so both classes are proportional ─────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
        stratify=y if n_reject > 1 else None
    )

    # ── Scale for Logistic Regression ──────────────────────────────────────
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # ── scale_pos_weight for XGBoost: ratio of negatives to positives ──────
    # reject=1 (positive), accept=0 (negative)
    n_neg_train = int((y_train == 0).sum())
    n_pos_train = int((y_train == 1).sum())
    spw = n_neg_train / max(n_pos_train, 1)
    print(f"XGBoost scale_pos_weight = {spw:.3f}  (accept_train={n_neg_train}, reject_train={n_pos_train})")

    # ── Build classifiers ──────────────────────────────────────────────────
    models: dict = {
        "LogisticRegression": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=42
        ),
        "RandomForest": RandomForestClassifier(
            class_weight="balanced", n_estimators=100, random_state=42
        ),
    }
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBClassifier(
            n_estimators=100, random_state=42,
            scale_pos_weight=spw,
            eval_metric="logloss",
        )

    results: dict = {}
    best_f1 = -1.0
    best_model_name: str = ""
    best_model = None

    for name, model in models.items():
        # ── Fit ────────────────────────────────────────────────────────────
        if name == "LogisticRegression":
            model.fit(X_train_sc, y_train)
            y_pred = model.predict(X_test_sc)
            y_prob = model.predict_proba(X_test_sc)[:, 1]
        elif name == "XGBoost":
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

        # ── Metrics ────────────────────────────────────────────────────────
        acc   = round(float(accuracy_score(y_test, y_pred)), 4)
        prec  = round(float(precision_score(y_test, y_pred, zero_division=0)), 4)
        rec   = round(float(recall_score(y_test, y_pred, zero_division=0)), 4)
        f1    = round(float(f1_score(y_test, y_pred, zero_division=0)), 4)
        auc   = round(float(roc_auc_score(y_test, y_prob)) if y_test.sum() > 0 else 0.5, 4)
        cm    = confusion_matrix(y_test, y_pred)

        metrics = {
            "accuracy": acc, "precision": prec,
            "recall": rec,   "f1": f1,  "roc_auc": auc,
        }
        results[name] = metrics

        print(f"\n  -- {name} --")
        print(f"  Accuracy:  {acc:.4f}    Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}    F1:        {f1:.4f}    ROC-AUC: {auc:.4f}")
        print(f"  Confusion matrix (rows=actual, cols=predicted):")
        print(f"               Pred accept(0)  Pred reject(1)")
        print(f"  Act accept(0)    {cm[0,0]:>5}           {cm[0,1]:>5}")
        print(f"  Act reject(1)    {cm[1,0]:>5}           {cm[1,1]:>5}")

        # ── Feature importance (RF / XGBoost) — all + top-5 callout ───────
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            pairs = sorted(zip(valid_cols, importances), key=lambda x: x[1], reverse=True)
            print(f"\n  Feature importance ({name}) — all features:")
            for feat, imp in pairs:
                bar = "#" * int(imp * 40)
                print(f"    {feat:<35} {imp:.4f}  {bar}")
            print(f"\n  Top-5 features ({name}):")
            for feat, imp in pairs[:5]:
                print(f"    {feat:<35} {imp:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name
            best_model = model

    # ── Save best model ────────────────────────────────────────────────────
    if best_model is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = output_dir / f"{stage_name}_classifier.pkl"
        joblib.dump(best_model, str(model_path))
        print(f"\n  [BEST] {best_model_name} (F1={best_f1:.4f})")
        print(f"    Saved → {model_path}")

        # Feature importance chart
        if hasattr(best_model, "feature_importances_"):
            _save_importance_chart(
                best_model.feature_importances_,
                valid_cols,
                stage_name,
                best_model_name,
                output_dir,
            )

    return results, valid_cols


# ── Feature importance chart ─────────────────────────────────────────────────

def _save_importance_chart(
    importances: np.ndarray,
    feature_names: list,
    stage_name: str,
    model_name: str,
    output_dir: Path,
) -> None:
    indices = np.argsort(importances)
    sorted_features    = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]

    fig, ax = plt.subplots(figsize=(10, max(4, len(feature_names) * 0.4)))
    ax.barh(sorted_features, sorted_importances, color="steelblue")
    ax.set_xlabel("Importance")
    ax.set_title(f"Feature Importance — {stage_name} ({model_name})")
    plt.tight_layout()

    chart_path = output_dir / f"feature_importance_{stage_name}.png"
    fig.savefig(str(chart_path), dpi=150)
    plt.close(fig)
    logger.info(f"Feature importance chart saved: {chart_path}")


# ── Comparison table ─────────────────────────────────────────────────────────

def print_comparison_table(all_results: dict) -> None:
    print("\n" + "=" * 80)
    print("FINAL MODEL COMPARISON TABLE")
    print("=" * 80)
    hdr = f"{'Stage':<10} {'Model':<22} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'AUC':>6}"
    print(hdr)
    print("-" * 80)
    for stage, models in all_results.items():
        for model_name, m in models.items():
            print(
                f"{stage:<10} {model_name:<22} "
                f"{m.get('accuracy',0):>6.3f} "
                f"{m.get('precision',0):>6.3f} "
                f"{m.get('recall',0):>6.3f} "
                f"{m.get('f1',0):>6.3f} "
                f"{m.get('roc_auc',0):>6.3f}"
            )
        print()
    print("=" * 80)


# ── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    output_dir = settings.ML_MODEL_DIR / "models"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading Mode B training data from CSV...")
    df = load_mode_b_data()

    if len(df) == 0:
        logger.error("No Mode B data found in CSV. Exiting.")
        sys.exit(1)

    all_results: dict = {}
    feature_columns: dict = {}
    for stage in ["script", "audio", "visual", "video"]:
        results, valid_cols = train_stage_models(df, stage, output_dir)
        if results:
            all_results[stage] = results
            feature_columns[stage] = valid_cols

    print_comparison_table(all_results)

    results_path = output_dir / "training_results.json"
    results_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\nTraining results saved to {results_path}")

    # Save feature column names — needed by the production classifier
    fc_path = output_dir / "feature_columns.json"
    fc_path.write_text(json.dumps(feature_columns, indent=2), encoding="utf-8")
    print(f"Feature columns saved to {fc_path}")


if __name__ == "__main__":
    main()
