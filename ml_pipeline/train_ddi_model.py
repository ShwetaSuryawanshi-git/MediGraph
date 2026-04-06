# File: MediGraph/ml_pipeline/train_ddi_model.py
"""
DDI Model Trainer
=================
Trains a drug-drug interaction (DDI) classifier using real ChEMBL/DrugBank data.
Uses your existing DTIModel class from model_training.py — zero code duplication.

Produces:
  models/ddi_model.pkl         ← trained model (same format as your DTI model)
  evaluation/roc_curve.png     ← ROC AUC curve
  evaluation/confusion_matrix.png
  evaluation/feature_importance.png
  evaluation/ddi_metrics.json  ← all scores

Usage:
    python train_ddi_model.py --data data/chembl_training.csv
    python train_ddi_model.py --data data/chembl_training.csv --model svm
"""

import argparse
import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    confusion_matrix, classification_report,
    average_precision_score,
)

try:
    from explainability import generate_global_shap_plots
except ImportError:
    generate_global_shap_plots = None

# Your existing classes — used as-is
from model_training import DTIModel
from config import Config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

META_COLS = {"drug_a_id", "drug_b_id", "drug_a_name", "drug_b_name", "label"}


def load_dataset(csv_path: str):
    log.info(f"Loading dataset: {csv_path}")
    df = pd.read_csv(csv_path)
    log.info(f"Shape: {df.shape}")
    log.info(f"Label distribution:\n{df['label'].value_counts().to_string()}")

    feature_cols = [c for c in df.columns if c not in META_COLS]
    X = df[feature_cols].fillna(0).astype(float).values
    y = df["label"].astype(int).values
    return X, y, feature_cols, df


def plot_roc(y_test, y_prob, output_dir: Path) -> float:
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, color="#1D9E75", lw=2, label=f"DDI model (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], color="#B4B2A9", lw=1, ls="--", label="Random")
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate",
           title="ROC Curve — Drug-Drug Interaction (ChEMBL + DrugBank)",
           xlim=[0, 1], ylim=[0, 1.02])
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "roc_curve.png", dpi=150)
    plt.close(fig)
    log.info(f"ROC curve saved → {output_dir}/roc_curve.png  (AUC={auc:.4f})")
    return auc


def plot_confusion(y_test, y_pred, output_dir: Path):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlOrRd",
                xticklabels=["No DDI", "DDI"],
                yticklabels=["No DDI", "DDI"],
                ax=ax, linewidths=0.5)
    ax.set(xlabel="Predicted", ylabel="Actual",
           title="Confusion Matrix — Drug-Drug Interaction")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)
    log.info(f"Confusion matrix saved → {output_dir}/confusion_matrix.png")


def plot_feature_importance(model: DTIModel, feature_cols: list, output_dir: Path, top_n: int = 20):
    if not hasattr(model.model, "feature_importances_"):
        log.info("Model doesn't support feature_importances_ — skipping plot.")
        return

    importances = model.model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(range(top_n), importances[indices], color="#534AB7", alpha=0.85)
    ax.set_xticks(range(top_n))
    ax.set_xticklabels(
        [feature_cols[i] for i in indices],
        rotation=45, ha="right", fontsize=9
    )
    ax.set(title=f"Top {top_n} Feature Importances (DDI Model)",
           ylabel="Importance")
    fig.tight_layout()
    fig.savefig(output_dir / "feature_importance.png", dpi=150)
    plt.close(fig)
    log.info(f"Feature importance saved → {output_dir}/feature_importance.png")


def main():
    parser = argparse.ArgumentParser(description="Train DDI model for MediGraph")
    parser.add_argument("--data", required=True, help="Path to chembl_training.csv")
    parser.add_argument(
        "--model", default="random_forest",
        choices=["random_forest", "logistic_regression", "svm"],
        help="Model type (default: random_forest)"
    )
    parser.add_argument("--model-path", default="models/ddi_model.pkl")
    parser.add_argument("--output-dir", default="evaluation")
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    Path(args.model_path).parent.mkdir(parents=True, exist_ok=True)

    # ── Load ──────────────────────────────────────────────────────────────────
    X, y, feature_cols, df = load_dataset(args.data)

    # ── Split ─────────────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )
    log.info(f"Train: {len(X_train)} | Test: {len(X_test)}")

    # ── Cross-validation ──────────────────────────────────────────────────────
    log.info("Running 5-fold cross-validation...")
    cv_model = DTIModel(model_type=args.model)
    # Temporarily fit for CV (cross_val_score needs a fresh estimator)
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    cv_scores = cross_val_score(
        cv_model.model, X, y,
        cv=StratifiedKFold(5, shuffle=True, random_state=42),
        scoring="roc_auc"
    )
    log.info(f"CV ROC AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # ── Train using your existing DTIModel.train() ─────────────────────────────
    log.info(f"Training {args.model} on full train split...")
    dti_model = DTIModel(model_type=args.model)
    train_metrics = dti_model.train(X_train, y_train, test_size=0.1)
    # train() does its own internal split — we retrain on full X_train here
    dti_model.model.fit(X_train, y_train)
    dti_model.is_trained = True

    # ── Evaluate ──────────────────────────────────────────────────────────────
    y_pred = dti_model.model.predict(X_test)
    y_prob = dti_model.model.predict_proba(X_test)[:, 1]

    auc = plot_roc(y_test, y_prob, output_dir)
    plot_confusion(y_test, y_pred, output_dir)
    plot_feature_importance(dti_model, feature_cols, output_dir)

    report = classification_report(
        y_test, y_pred, target_names=["No DDI", "DDI"], output_dict=True
    )
    avg_prec = average_precision_score(y_test, y_prob)

    metrics = {
        "model_type": args.model,
        "roc_auc": round(auc, 4),
        "avg_precision": round(avg_prec, 4),
        "cv_roc_auc_mean": round(float(cv_scores.mean()), 4),
        "cv_roc_auc_std": round(float(cv_scores.std()), 4),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_features": len(feature_cols),
        "n_positive_test": int(y_test.sum()),
        "n_negative_test": int((y_test == 0).sum()),
        "classification_report": report,
    }

    with open(output_dir / "ddi_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    if generate_global_shap_plots:
        try:
            generate_global_shap_plots(
                model_input=dti_model, 
                X_test=X_test, 
                output_dir=output_dir, 
                json_path=output_dir / "ddi_metrics.json",
                is_dti=False
            )
        except Exception as e:
            log.error(f"SHAP generation failed: {e}")

    # ── Save model in your existing format ─────────────────────────────────────
    # DTIModel.save() stores: model, model_type, is_trained, metrics
    dti_model.metrics = {k: v for k, v in metrics.items()
                         if k in ("roc_auc", "avg_precision", "cv_roc_auc_mean")}
    dti_model.save(args.model_path)
    log.info(f"Model saved → {args.model_path}")

    # ── Print summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*58}")
    print(f"  MediGraph DDI Model — ChEMBL + DrugBank")
    print(f"{'='*58}")
    print(f"  Model:              {args.model}")
    print(f"  ROC AUC:            {auc:.4f}")
    print(f"  Avg Precision:      {avg_prec:.4f}")
    print(f"  CV ROC AUC:         {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['No DDI', 'DDI'])}")
    print(f"  Charts:  {output_dir}/")
    print(f"  Model:   {args.model_path}")
    print(f"{'='*58}\n")
    print("  Load this model in run_pipeline.py with:")
    print(f"  DTIModel.load('{args.model_path}')")


if __name__ == "__main__":
    main()
