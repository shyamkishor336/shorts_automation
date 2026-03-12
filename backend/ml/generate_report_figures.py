"""
Generate dissertation evaluation figures from trained ML models + mode_b_export.csv.

Usage (from project root):
    python -m backend.ml.generate_report_figures

Outputs to: report_figures/
    fig_confusion_matrices.png
    fig_roc_curves.png
    fig_model_comparison_f1.png
    fig_feature_importance.png
    fig_class_distribution.png
    fig_metrics_summary_table.png
    fig_mode_c_probability_distribution.png  (skipped if mode_c_export.csv absent)
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent.parent
CSV_PATH = ROOT / "mode_b_export.csv"
MODE_C_CSV = ROOT / "mode_c_export.csv"
MODELS_DIR = Path(__file__).parent / "models"
OUT_DIR = ROOT / "report_figures"

STAGES = ["script", "audio", "visual", "video"]

STAGE_FEATURES = {
    "script": [
        "readability_score", "lexical_diversity", "prompt_coverage",
        "sentence_redundancy", "entity_consistency", "topic_coherence",
        "factual_conflict_flag", "prompt_ambiguity",
        "prior_stage_corrections", "cumulative_risk_score", "api_retry_count",
    ],
    "audio": [
        "phoneme_error_rate", "silence_ratio", "speaking_rate_variance",
        "energy_variance", "tts_word_count_match",
        "prior_stage_corrections", "cumulative_risk_score", "api_retry_count",
    ],
    "visual": [
        "clip_similarity", "aesthetic_score", "blur_score",
        "object_match_score", "colour_tone_match",
        "prior_stage_corrections", "cumulative_risk_score", "api_retry_count",
    ],
    "video": [
        "av_sync_error_ms", "transition_smoothness", "duration_deviation_s",
        "prior_stage_corrections", "cumulative_risk_score", "api_retry_count",
    ],
}

MODEL_COLOURS = {
    "LogisticRegression": "#4C72B0",
    "RandomForest":       "#DD8452",
    "XGBoost":            "#55A868",
}
MODEL_LABELS = {
    "LogisticRegression": "Logistic Regression",
    "RandomForest":       "Random Forest",
    "XGBoost":            "XGBoost",
}

sns.set_style("whitegrid")
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "figure.dpi": 300,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
})


# ── Data helpers ──────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)
    df = df[df["mode"] == "B"].copy()
    df = df[df["human_decision"].notna()].copy()
    df["label"] = (df["human_decision"] == "reject").astype(int)
    return df


def prep_stage(df: pd.DataFrame, stage: str):
    """Return (X_train_raw, X_train_sc, X_test_raw, X_test_sc, y_train, y_test, valid_cols)."""
    sdf = df[df["stage_name"] == stage].copy()
    feat_cols = STAGE_FEATURES[stage]
    avail = [c for c in feat_cols if c in sdf.columns]
    X = sdf[avail].copy()
    valid_cols = [c for c in avail if X[c].notna().sum() > 0]
    X = X[valid_cols].copy().fillna(X[valid_cols].mean())
    y = sdf["label"].values

    n_rej = int((y == 1).sum())
    strat = y if n_rej > 1 else None
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=strat
    )

    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_te_sc = scaler.transform(X_te)
    return X_tr, X_tr_sc, X_te, X_te_sc, y_tr, y_te, valid_cols


def fit_all_models(X_tr, X_tr_sc, X_te, X_te_sc, y_tr, y_te):
    """Fit LR, RF, XGB; return dict of {name: (model, y_pred, y_prob)}."""
    n_neg = int((y_tr == 0).sum())
    n_pos = max(int((y_tr == 1).sum()), 1)
    spw = n_neg / n_pos

    classifiers = {
        "LogisticRegression": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=42
        ),
        "RandomForest": RandomForestClassifier(
            class_weight="balanced", n_estimators=100, random_state=42
        ),
    }
    if XGBOOST_AVAILABLE:
        classifiers["XGBoost"] = XGBClassifier(
            n_estimators=100, random_state=42,
            scale_pos_weight=spw, eval_metric="logloss",
        )

    results = {}
    for name, clf in classifiers.items():
        if name == "LogisticRegression":
            clf.fit(X_tr_sc, y_tr)
            y_pred = clf.predict(X_te_sc)
            y_prob = clf.predict_proba(X_te_sc)[:, 1]
        else:
            clf.fit(X_tr, y_tr)
            y_pred = clf.predict(X_te)
            y_prob = clf.predict_proba(X_te)[:, 1]
        results[name] = (clf, y_pred, y_prob)
    return results


# ── Figure 1 — Confusion Matrices ────────────────────────────────────────────

def fig_confusion_matrices(df: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle("Confusion Matrices — Best Classifier per Stage (Mode B Test Set)",
                 fontsize=13, fontweight="bold", y=1.01)

    for ax, stage in zip(axes.flat, STAGES):
        model_path = MODELS_DIR / f"{stage}_classifier.pkl"
        if not model_path.exists():
            ax.set_title(f"{stage.title()} — model not found")
            ax.axis("off")
            continue

        model = joblib.load(str(model_path))
        X_tr, X_tr_sc, X_te, X_te_sc, y_tr, y_te, valid_cols = prep_stage(df, stage)

        # Determine if this is a scaled model (LR) by checking type
        if isinstance(model, LogisticRegression):
            y_pred = model.predict(X_te_sc)
        else:
            y_pred = model.predict(X_te)

        cm = confusion_matrix(y_te, y_pred)
        labels = ["Accept", "Reject"]
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels,
            ax=ax, cbar=False, linewidths=0.5,
        )
        f1 = f1_score(y_te, y_pred, zero_division=0)
        mname = type(model).__name__
        ax.set_title(f"{stage.title()}  ({mname}, F1={f1:.3f})")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    plt.tight_layout()
    path = out / "fig_confusion_matrices.png"
    fig.savefig(str(path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {path.name}  ({path.stat().st_size // 1024} KB)")


# ── Figure 2 — ROC Curves ────────────────────────────────────────────────────

def fig_roc_curves(df: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    fig.suptitle("ROC Curves — All Classifiers per Stage (Mode B Test Set)",
                 fontsize=13, fontweight="bold", y=1.01)

    for ax, stage in zip(axes.flat, STAGES):
        X_tr, X_tr_sc, X_te, X_te_sc, y_tr, y_te, _ = prep_stage(df, stage)
        fit = fit_all_models(X_tr, X_tr_sc, X_te, X_te_sc, y_tr, y_te)

        ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Random (AUC=0.50)")

        for name, (_, _, y_prob) in fit.items():
            if y_te.sum() == 0:
                continue
            fpr, tpr, _ = roc_curve(y_te, y_prob)
            roc_auc = auc(fpr, tpr)
            ax.plot(
                fpr, tpr,
                color=MODEL_COLOURS[name], lw=1.8,
                label=f"{MODEL_LABELS[name]} (AUC={roc_auc:.3f})",
            )

        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.05])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"{stage.title()} Stage")
        ax.legend(loc="lower right", framealpha=0.85)

    plt.tight_layout()
    path = out / "fig_roc_curves.png"
    fig.savefig(str(path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {path.name}  ({path.stat().st_size // 1024} KB)")


# ── Figure 3 — Model Comparison F1 ───────────────────────────────────────────

def fig_model_comparison_f1(out: Path) -> None:
    results_path = MODELS_DIR / "training_results.json"
    if not results_path.exists():
        print(f"  ✗  training_results.json not found, skipping fig_model_comparison_f1")
        return

    results = json.loads(results_path.read_text())
    model_names = ["LogisticRegression", "RandomForest", "XGBoost"]
    x = np.arange(len(STAGES))
    width = 0.25

    fig, ax = plt.subplots(figsize=(11, 5))
    for i, mname in enumerate(model_names):
        f1_vals = [results.get(s, {}).get(mname, {}).get("f1", 0) for s in STAGES]
        bars = ax.bar(
            x + i * width, f1_vals, width,
            label=MODEL_LABELS[mname],
            color=MODEL_COLOURS[mname], alpha=0.88, edgecolor="white", linewidth=0.5,
        )
        for bar, v in zip(bars, f1_vals):
            if v > 0.02:
                ax.text(
                    bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=7.5,
                )

    ax.set_xticks(x + width)
    ax.set_xticklabels([s.title() for s in STAGES])
    ax.set_ylabel("F1 Score")
    ax.set_title("Model Comparison — F1 Score per Stage", fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.legend(framealpha=0.85)
    ax.axhline(0.5, color="gray", lw=0.8, linestyle="--", alpha=0.5)

    plt.tight_layout()
    path = out / "fig_model_comparison_f1.png"
    fig.savefig(str(path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {path.name}  ({path.stat().st_size // 1024} KB)")


# ── Figure 4 — Feature Importance ────────────────────────────────────────────

def fig_feature_importance(df: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle("Feature Importance — Best Model per Stage",
                 fontsize=13, fontweight="bold", y=1.01)

    for ax, stage in zip(axes.flat, STAGES):
        model_path = MODELS_DIR / f"{stage}_classifier.pkl"
        if not model_path.exists():
            ax.set_title(f"{stage.title()} — model not found")
            ax.axis("off")
            continue

        model = joblib.load(str(model_path))
        _, _, _, _, _, _, valid_cols = prep_stage(df, stage)
        mname = type(model).__name__

        # Get importances
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_[0])
            if len(importances) != len(valid_cols):
                ax.set_title(f"{stage.title()} — coef mismatch")
                ax.axis("off")
                continue
        else:
            ax.set_title(f"{stage.title()} — importance N/A")
            ax.axis("off")
            continue

        # Top 8 features
        n_show = min(8, len(valid_cols))
        pairs = sorted(zip(valid_cols, importances), key=lambda x: x[1], reverse=True)[:n_show]
        feats, vals = zip(*pairs) if pairs else ([], [])
        feats = list(reversed(feats))
        vals = list(reversed(vals))

        # Readable labels
        readable = [f.replace("_", " ").title() for f in feats]

        ax.barh(readable, vals, color="#4C72B0", alpha=0.85, edgecolor="white", linewidth=0.4)
        ax.set_title(f"{stage.title()}  ({mname})")
        ax.set_xlabel("Importance" if hasattr(model, "feature_importances_") else "|Coefficient|")
        ax.tick_params(axis="y", labelsize=8)

    plt.tight_layout()
    path = out / "fig_feature_importance.png"
    fig.savefig(str(path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {path.name}  ({path.stat().st_size // 1024} KB)")


# ── Figure 5 — Class Distribution ────────────────────────────────────────────

def fig_class_distribution(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))

    accept_counts = [int((df[df["stage_name"] == s]["label"] == 0).sum()) for s in STAGES]
    reject_counts = [int((df[df["stage_name"] == s]["label"] == 1).sum()) for s in STAGES]

    x = np.arange(len(STAGES))
    w = 0.35
    bars_a = ax.bar(x - w / 2, accept_counts, w, label="Accept", color="#55A868", alpha=0.88, edgecolor="white")
    bars_r = ax.bar(x + w / 2, reject_counts, w, label="Reject", color="#C44E52", alpha=0.88, edgecolor="white")

    for bar in list(bars_a) + list(bars_r):
        v = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.4,
                str(int(v)), ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([s.title() for s in STAGES])
    ax.set_ylabel("Number of Stage Attempts")
    ax.set_title("Class Distribution — Accept vs Reject per Stage (Mode B)", fontweight="bold")
    ax.legend(framealpha=0.85)

    plt.tight_layout()
    path = out / "fig_class_distribution.png"
    fig.savefig(str(path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {path.name}  ({path.stat().st_size // 1024} KB)")


# ── Figure 6 — Metrics Summary Table ─────────────────────────────────────────

def fig_metrics_summary_table(out: Path) -> None:
    results_path = MODELS_DIR / "training_results.json"
    if not results_path.exists():
        print(f"  ✗  training_results.json not found, skipping fig_metrics_summary_table")
        return

    results = json.loads(results_path.read_text())
    model_names = ["LogisticRegression", "RandomForest", "XGBoost"]
    metric_keys = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]

    rows = []
    for stage in STAGES:
        for mname in model_names:
            m = results.get(stage, {}).get(mname, {})
            row = [stage.title(), MODEL_LABELS[mname]] + [
                f"{m.get(k, 0):.3f}" for k in metric_keys
            ]
            rows.append(row)

    col_labels = ["Stage", "Model"] + metric_labels
    fig, ax = plt.subplots(figsize=(14, len(rows) * 0.42 + 1.2))
    ax.axis("off")

    tbl = ax.table(
        cellText=rows,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1, 1.45)

    # Header style
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor("#2C3E50")
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    # Row banding + highlight best F1
    stage_colours = {
        "Script": "#EBF5FB", "Audio": "#FDFEFE",
        "Visual": "#EBF5FB", "Video": "#FDFEFE",
    }
    for i, row in enumerate(rows, start=1):
        bg = stage_colours.get(row[0], "white")
        for j in range(len(col_labels)):
            tbl[i, j].set_facecolor(bg)
        # Highlight F1 column (index 5) if >= 0.5
        try:
            if float(row[5]) >= 0.5:
                tbl[i, 5].set_facecolor("#D5F5E3")
        except ValueError:
            pass

    ax.set_title(
        "Classifier Metrics Summary — All Models × All Stages (Mode B Test Set)",
        fontsize=11, fontweight="bold", pad=12,
    )
    plt.tight_layout()
    path = out / "fig_metrics_summary_table.png"
    fig.savefig(str(path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {path.name}  ({path.stat().st_size // 1024} KB)")


# ── Figure 7 — Mode C Probability Distribution ───────────────────────────────

def fig_mode_c_prob_distribution(out: Path) -> None:
    if not MODE_C_CSV.exists():
        print(f"  –  mode_c_export.csv not found — skipping fig_mode_c_probability_distribution")
        return

    df_c = pd.read_csv(MODE_C_CSV)
    prob_col = next(
        (c for c in df_c.columns if "prob" in c.lower() or "confidence" in c.lower() or "score" in c.lower()),
        None,
    )
    if prob_col is None:
        print(f"  –  No probability column found in mode_c_export.csv — skipping")
        return

    probs = df_c[prob_col].dropna().values
    threshold = 0.65

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(probs, bins=30, color="#4C72B0", alpha=0.8, edgecolor="white", linewidth=0.4)
    ax.axvline(threshold, color="#C44E52", lw=2, linestyle="--",
               label=f"Threshold = {threshold}")
    ax.set_xlabel(f"Predicted Probability ({prob_col})")
    ax.set_ylabel("Count")
    ax.set_title("Mode C — Predicted Probability Distribution", fontweight="bold")
    ax.legend(framealpha=0.85)

    n_auto = int((probs >= threshold).sum())
    n_human = int((probs < threshold).sum())
    ax.text(0.97, 0.95,
            f"Auto: {n_auto} ({100*n_auto/len(probs):.1f}%)\nHuman: {n_human} ({100*n_human/len(probs):.1f}%)",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    plt.tight_layout()
    path = out / "fig_mode_c_probability_distribution.png"
    fig.savefig(str(path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {path.name}  ({path.stat().st_size // 1024} KB)")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if not CSV_PATH.exists():
        sys.exit(f"ERROR: mode_b_export.csv not found at {CSV_PATH}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nGenerating dissertation figures → {OUT_DIR}\n{'─'*55}")

    df = load_data()
    print(f"Data loaded: {len(df)} Mode B rows across {df['stage_name'].nunique()} stages\n")

    fig_confusion_matrices(df, OUT_DIR)
    fig_roc_curves(df, OUT_DIR)
    fig_model_comparison_f1(OUT_DIR)
    fig_feature_importance(df, OUT_DIR)
    fig_class_distribution(df, OUT_DIR)
    fig_metrics_summary_table(OUT_DIR)
    fig_mode_c_prob_distribution(OUT_DIR)

    # Summary
    print(f"\n{'─'*55}")
    print(f"Output directory: {OUT_DIR}")
    total_kb = sum(p.stat().st_size for p in OUT_DIR.glob("fig_*.png")) // 1024
    print(f"Total size: {total_kb} KB  ({len(list(OUT_DIR.glob('fig_*.png')))} files)")


if __name__ == "__main__":
    main()
