"""
evaluate_model.py
-----------------
Trains MediGraph's Random Forest on real ChEMBL/DrugBank data
and produces ROC AUC curve + confusion matrix — both saved as PNGs
and printed to console.

Usage:
    python evaluate_model.py --data data/training.csv
    python evaluate_model.py --data data/training.csv --model-path models/rf_model.pkl
"""

import argparse
import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    confusion_matrix, classification_report,
    average_precision_score,
)
from sklearn.preprocessing import label_binarize

try:
    from explainability import generate_global_shap_plots
except ImportError:
    generate_global_shap_plots = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# Columns that are metadata, not features
META_COLS = {"drug_a_id", "drug_b_id", "drug_a_name", "drug_b_name", "label"}


def load_data(csv_path: str):
    log.info(f"Loading dataset from {csv_path}")
    df = pd.read_csv(csv_path)
    log.info(f"Shape: {df.shape} | Label distribution:\n{df['label'].value_counts()}")

    feature_cols = [c for c in df.columns if c not in META_COLS]
    X = df[feature_cols].fillna(0).astype(float)
    y = df["label"].astype(int)
    meta = df[[c for c in META_COLS - {"label"} if c in df.columns]]
    return X, y, meta, feature_cols


def train_model(X_train, y_train) -> RandomForestClassifier:
    log.info("Training Random Forest...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=4,
        class_weight="balanced",   # handles imbalanced labels
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def plot_roc_curve(y_test, y_prob, output_dir: Path):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, color="#1D9E75", lw=2, label=f"ROC curve (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], color="#B4B2A9", lw=1, linestyle="--", label="Random classifier")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve — Drug-Drug Interaction Prediction", fontsize=13)
    ax.legend(loc="lower right", fontsize=11)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out = output_dir / "roc_curve.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    log.info(f"ROC curve saved to {out}")
    return auc


def plot_confusion_matrix(y_test, y_pred, output_dir: Path):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="YlOrRd",
        xticklabels=["No DDI", "DDI"],
        yticklabels=["No DDI", "DDI"],
        ax=ax, linewidths=0.5,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title("Confusion Matrix — Drug-Drug Interaction", fontsize=13)
    fig.tight_layout()

    out = output_dir / "confusion_matrix.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    log.info(f"Confusion matrix saved to {out}")


def plot_feature_importance(model, feature_cols, output_dir: Path, top_n: int = 20):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(
        range(top_n),
        importances[indices],
        color="#534AB7", alpha=0.85,
    )
    ax.set_xticks(range(top_n))
    ax.set_xticklabels(
        [feature_cols[i] for i in indices],
        rotation=45, ha="right", fontsize=9,
    )
    ax.set_title(f"Top {top_n} Feature Importances", fontsize=13)
    ax.set_ylabel("Importance", fontsize=12)
    fig.tight_layout()

    out = output_dir / "feature_importance.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    log.info(f"Feature importance chart saved to {out}")


def save_metrics(metrics: dict, output_dir: Path):
    out = output_dir / "metrics.json"
    with open(out, "w") as f:
        json.dump(metrics, f, indent=2)
    log.info(f"Metrics saved to {out}")


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate MediGraph model")
    parser.add_argument("--data", required=True, help="Path to training CSV")
    parser.add_argument("--model-path", default="models/rf_model.pkl", help="Where to save the model")
    parser.add_argument("--output-dir", default="evaluation", help="Where to save evaluation charts")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction for test split")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    Path(args.model_path).parent.mkdir(parents=True, exist_ok=True)

    # ── Load ──────────────────────────────────────────────────────────────────
    X, y, meta, feature_cols = load_data(args.data)

    # ── Split ─────────────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )
    log.info(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows")

    # ── Cross-validation ──────────────────────────────────────────────────────
    log.info("Running 5-fold cross-validation...")
    cv_model = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1)
    cv_scores = cross_val_score(cv_model, X, y, cv=StratifiedKFold(5), scoring="roc_auc")
    log.info(f"CV ROC AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # ── Train ─────────────────────────────────────────────────────────────────
    model = train_model(X_train, y_train)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    auc = plot_roc_curve(y_test, y_prob, output_dir)
    plot_confusion_matrix(y_test, y_pred, output_dir)
    plot_feature_importance(model, feature_cols, output_dir)

    report = classification_report(y_test, y_pred, target_names=["No DDI", "DDI"], output_dict=True)
    avg_precision = average_precision_score(y_test, y_prob)

    metrics = {
        "roc_auc": round(auc, 4),
        "avg_precision": round(avg_precision, 4),
        "cv_roc_auc_mean": round(float(cv_scores.mean()), 4),
        "cv_roc_auc_std": round(float(cv_scores.std()), 4),
        "classification_report": report,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_features": len(feature_cols),
    }
    save_metrics(metrics, output_dir)
    
    if generate_global_shap_plots:
        try:
            generate_global_shap_plots(
                model_input=model, 
                X_test=X_test, 
                output_dir=output_dir, 
                json_path=output_dir / "metrics.json",
                is_dti=True
            )
        except Exception as e:
            log.error(f"SHAP generation failed: {e}")

    # ── Save model ────────────────────────────────────────────────────────────
    with open(args.model_path, "wb") as f:
        pickle.dump({"model": model, "feature_cols": feature_cols}, f)
    log.info(f"Model saved to {args.model_path}")

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  MediGraph Model Evaluation Results")
    print(f"{'='*55}")
    print(f"  ROC AUC Score:        {auc:.4f}")
    print(f"  Avg Precision Score:  {avg_precision:.4f}")
    print(f"  CV ROC AUC (5-fold):  {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["No DDI", "DDI"]))
    print(f"  Charts saved in: {output_dir}/")
    print(f"  Model saved to:  {args.model_path}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
