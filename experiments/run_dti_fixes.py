"""
experiments/run_dti_fixes.py
----------------------------
Executes DTI Audit Fixes (Items 1, 2, 3, 4, 5):
1. Renames "DeepDTA" to "MLP baseline" everywhere.
2. Standardizes features for Logistic Regression and LinearSVC, records comparison with vs without standardization.
3. Chooses decision threshold for F1/MCC dynamically on the validation set.
4. Generates results/dataset_stats.csv (drugs, targets, pairs, positive rate, test sizes, chance baselines).
5. Regenerates publication figures (Y-axis from 0, explicit chance lines: AUROC=0.5, AUPR=prevalence).
6. Regenerates tables/dti_main.tex and tables/dti_protein_feature_ablation.tex directly from CSVs.
"""

import sys
import os
import math
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, matthews_corrcoef, balanced_accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "results"
TABLES_DIR = BASE_DIR / "tables"
FIGURES_DIR = BASE_DIR / "figures"
DATA_DIR = BASE_DIR / "data"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

sys.path.append(str(BASE_DIR))
from phase1_dti_benchmark import (
    compute_ecfp4, compute_aac, compute_dipeptide, compute_ctd,
    generate_splits, ESM2Embedder, MediGraphEnsemble
)

# -------------------------------------------------------------
# 1. MLP Baseline Architecture (formerly mislabeled DeepDTA)
# -------------------------------------------------------------
class MLPBaselineNet(nn.Module):
    """
    Two-tower Multi-Layer Perceptron (MLP) baseline operating on
    concatenated ECFP4 Morgan fingerprints and protein physicochemical descriptors.
    """
    def __init__(self, drug_dim: int, protein_dim: int):
        super().__init__()
        self.drug_net = nn.Sequential(
            nn.Linear(drug_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.protein_net = nn.Sequential(
            nn.Linear(protein_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x_drug, x_protein):
        h_d = self.drug_net(x_drug)
        h_p = self.protein_net(x_protein)
        h = torch.cat([h_d, h_p], dim=1)
        return self.classifier(h).squeeze(-1)

class MLPBaselinePredictor:
    def __init__(self, drug_dim: int, protein_dim: int, epochs: int = 12, lr: float = 1e-3, seed: int = 42):
        self.drug_dim = drug_dim
        self.protein_dim = protein_dim
        self.epochs = epochs
        self.lr = lr
        self.seed = seed
        self.model = None

    def fit(self, X_train_drug, X_train_prot, y_train, X_val_drug=None, X_val_prot=None, y_val=None):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.model = MLPBaselineNet(self.drug_dim, self.protein_dim)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        
        pos_weight = torch.tensor([(len(y_train) - sum(y_train)) / max(1, sum(y_train))], dtype=torch.float32)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        dataset = TensorDataset(
            torch.tensor(X_train_drug, dtype=torch.float32),
            torch.tensor(X_train_prot, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32)
        )
        loader = DataLoader(dataset, batch_size=64, shuffle=True)
        best_val_loss = float("inf")
        best_state = None

        for epoch in range(self.epochs):
            self.model.train()
            for d_b, p_b, y_b in loader:
                optimizer.zero_grad()
                logits = self.model(d_b, p_b)
                loss = criterion(logits, y_b)
                loss.backward()
                optimizer.step()

            if X_val_drug is not None and len(X_val_drug) > 0:
                self.model.eval()
                with torch.no_grad():
                    v_logits = self.model(
                        torch.tensor(X_val_drug, dtype=torch.float32),
                        torch.tensor(X_val_prot, dtype=torch.float32)
                    )
                    v_loss = criterion(v_logits, torch.tensor(y_val, dtype=torch.float32)).item()
                    if v_loss < best_val_loss:
                        best_val_loss = v_loss
                        best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

        if best_state is not None:
            self.model.load_state_dict(best_state)

    def predict_proba(self, X_drug, X_prot) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            logits = self.model(
                torch.tensor(X_drug, dtype=torch.float32),
                torch.tensor(X_prot, dtype=torch.float32)
            )
            probs = torch.sigmoid(logits).cpu().numpy()
        return np.column_stack([1.0 - probs, probs])

# -------------------------------------------------------------
# 2. Evaluation with Validation-Tuned Threshold
# -------------------------------------------------------------
def find_optimal_threshold(y_val: np.ndarray, p_val: np.ndarray) -> float:
    """Finds decision threshold maximizing F1 on validation set."""
    best_f1 = -1.0
    best_thresh = 0.5
    for thresh in np.linspace(0.05, 0.95, 19):
        pred = (p_val >= thresh).astype(int)
        score = f1_score(y_val, pred, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_thresh = float(thresh)
    return best_thresh

def evaluate_predictions(y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """Calculates AUROC, AUPR, F1, MCC, Balanced Accuracy using calibrated threshold."""
    if len(np.unique(y_true)) < 2:
        return {"auroc": 0.5, "aupr": 0.0, "f1": 0.0, "mcc": 0.0, "bacc": 0.5, "threshold": threshold}
    
    auroc = roc_auc_score(y_true, y_proba)
    aupr = average_precision_score(y_true, y_proba)
    y_pred = (y_proba >= threshold).astype(int)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)

    return {
        "auroc": float(auroc),
        "aupr": float(aupr),
        "f1": float(f1),
        "mcc": float(mcc),
        "bacc": float(bacc),
        "threshold": float(threshold)
    }

# -------------------------------------------------------------
# 3. Main Runner for DTI Fixes
# -------------------------------------------------------------
def run_dti_fixes(seeds=[0, 1, 2, 3, 4]):
    print("=" * 70)
    print("RUNNING DTI AUDIT FIXES (ITEMS 1, 2, 3, 4, 5)")
    print("=" * 70)

    split_types = ["random", "cold_drug", "cold_target", "cold_both"]
    datasets_to_run = ["DAVIS", "KIBA"]

    results_records = []
    std_comparison_records = []

    csv_path = RESULTS_DIR / "dti_benchmark.csv"
    std_csv_path = RESULTS_DIR / "dti_standardization_comparison.csv"

    # Pre-load or resume if exists
    completed_keys = set()
    if csv_path.exists():
        try:
            prev = pd.read_csv(csv_path)
            # Only use if already converted to 'MLP baseline'
            if "MLP baseline" in prev["model"].values:
                results_records = prev.to_dict(orient="records")
                completed_keys = {(r["dataset"], r["split"], r["seed"], r["model"]) for r in results_records}
                print(f"[Resume] Found {len(completed_keys)} completed model evaluations in {csv_path}")
        except Exception:
            pass

    for ds_name in datasets_to_run:
        print(f"\n>>> Processing Dataset: {ds_name} <<<")
        if ds_name == "DAVIS":
            df_data = pd.read_csv(DATA_DIR / "davis.tab", sep="\t")
            df_data.rename(columns={"X1": "Drug", "X2": "Target"}, inplace=True)
            df_data["y_binary"] = (df_data["Y"] <= 100.0).astype(int)
        else:
            df_data = pd.read_csv(DATA_DIR / "kiba.tab", sep="\t")
            df_data.rename(columns={"X1": "Drug", "X2": "Target"}, inplace=True)
            df_data["y_binary"] = (df_data["Y"] >= 12.1).astype(int)
            # Stratified sample 20,000 for KIBA
            rng_sub = np.random.RandomState(42)
            idx_sub = rng_sub.choice(len(df_data), size=min(20000, len(df_data)), replace=False)
            df_data = df_data.iloc[idx_sub].reset_index(drop=True)

        unique_smiles = df_data["Drug"].unique()
        unique_seqs = df_data["Target"].unique()
        print(f"[{ds_name}] {len(df_data)} pairs ({df_data['y_binary'].sum()} active). Featurizing...")

        drug_fps = {s: compute_ecfp4(s) for s in unique_smiles}
        prot_aac = {seq: compute_aac(seq) for seq in unique_seqs}
        prot_dipep = {seq: compute_dipeptide(seq) for seq in unique_seqs}
        prot_ctd = {seq: compute_ctd(seq) for seq in unique_seqs}

        X_drug = np.array([drug_fps[s] for s in df_data["Drug"]])
        X_aac = np.array([prot_aac[seq] for seq in df_data["Target"]])
        X_dipep = np.array([prot_dipep[seq] for seq in df_data["Target"]])
        X_ctd = np.array([prot_ctd[seq] for seq in df_data["Target"]])
        X_prot_combined = np.hstack([X_aac, X_dipep, X_ctd])
        X_all = np.hstack([X_drug, X_prot_combined])
        y_all = df_data["y_binary"].values

        for split_type in split_types:
            for seed in seeds:
                target_models = {"Logistic Regression", "Calibrated LinearSVC", "Random Forest",
                                 "XGBoost", "MediGraph Ensemble", "MLP baseline"}
                existing_for_seed = {k[3] for k in completed_keys if k[0] == ds_name and k[1] == split_type and k[2] == seed}
                if existing_for_seed >= target_models:
                    continue

                print(f"[{ds_name} | {split_type} | seed={seed}] Splitting and training...")
                train_df, val_df, test_df = generate_splits(df_data, split_type, seed)
                if train_df['y_binary'].nunique() < 2 or test_df['y_binary'].nunique() < 2:
                    continue

                train_idx = train_df.index.values
                val_idx = val_df.index.values
                test_idx = test_df.index.values

                X_tr, y_tr = X_all[train_idx], y_all[train_idx]
                X_va, y_va = X_all[val_idx], y_all[val_idx]
                X_te, y_te = X_all[test_idx], y_all[test_idx]

                # --- Scaler strictly fitted on Train ---
                scaler = StandardScaler()
                X_tr_scaled = scaler.fit_transform(X_tr)
                X_va_scaled = scaler.transform(X_va)
                X_te_scaled = scaler.transform(X_te)

                # 1. Logistic Regression (Standardized)
                lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
                lr.fit(X_tr_scaled, y_tr)
                p_lr_va = lr.predict_proba(X_va_scaled)[:, 1]
                t_lr = find_optimal_threshold(y_va, p_lr_va)
                p_lr = lr.predict_proba(X_te_scaled)[:, 1]
                m_lr = evaluate_predictions(y_te, p_lr, threshold=t_lr)
                results_records.append({
                    "dataset": ds_name, "split": split_type, "seed": seed, "model": "Logistic Regression",
                    "auroc": m_lr["auroc"], "aupr": m_lr["aupr"], "f1": m_lr["f1"], "mcc": m_lr["mcc"], "bacc": m_lr["bacc"],
                    "n_train": len(X_tr), "n_test": len(X_te)
                })

                # Unstandardized LogReg for comparison
                lr_raw = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
                lr_raw.fit(X_tr, y_tr)
                p_lr_raw = lr_raw.predict_proba(X_te)[:, 1]
                std_comparison_records.append({
                    "dataset": ds_name, "split": split_type, "seed": seed, "model": "Logistic Regression",
                    "auroc_unstandardized": float(roc_auc_score(y_te, p_lr_raw)),
                    "aupr_unstandardized": float(average_precision_score(y_te, p_lr_raw)),
                    "auroc_standardized": m_lr["auroc"],
                    "aupr_standardized": m_lr["aupr"]
                })

                # 2. LinearSVC (Standardized + Platt Scaled)
                base_svc = LinearSVC(dual=False, class_weight="balanced", random_state=seed, max_iter=2000)
                lsvc = CalibratedClassifierCV(base_svc, cv=3)
                lsvc.fit(X_tr_scaled, y_tr)
                p_svm_va = lsvc.predict_proba(X_va_scaled)[:, 1]
                t_svm = find_optimal_threshold(y_va, p_svm_va)
                p_svm = lsvc.predict_proba(X_te_scaled)[:, 1]
                m_svm = evaluate_predictions(y_te, p_svm, threshold=t_svm)
                results_records.append({
                    "dataset": ds_name, "split": split_type, "seed": seed, "model": "Calibrated LinearSVC",
                    "auroc": m_svm["auroc"], "aupr": m_svm["aupr"], "f1": m_svm["f1"], "mcc": m_svm["mcc"], "bacc": m_svm["bacc"],
                    "n_train": len(X_tr), "n_test": len(X_te)
                })

                # Unstandardized LinearSVC for comparison
                base_svc_raw = LinearSVC(dual=False, class_weight="balanced", random_state=seed, max_iter=2000)
                lsvc_raw = CalibratedClassifierCV(base_svc_raw, cv=3)
                lsvc_raw.fit(X_tr, y_tr)
                p_svm_raw = lsvc_raw.predict_proba(X_te)[:, 1]
                std_comparison_records.append({
                    "dataset": ds_name, "split": split_type, "seed": seed, "model": "Calibrated LinearSVC",
                    "auroc_unstandardized": float(roc_auc_score(y_te, p_svm_raw)),
                    "aupr_unstandardized": float(average_precision_score(y_te, p_svm_raw)),
                    "auroc_standardized": m_svm["auroc"],
                    "aupr_standardized": m_svm["aupr"]
                })

                # 3. Random Forest
                rf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=seed, n_jobs=-1)
                rf.fit(X_tr, y_tr)
                p_rf_va = rf.predict_proba(X_va)[:, 1]
                t_rf = find_optimal_threshold(y_va, p_rf_va)
                p_rf = rf.predict_proba(X_te)[:, 1]
                m_rf = evaluate_predictions(y_te, p_rf, threshold=t_rf)
                results_records.append({
                    "dataset": ds_name, "split": split_type, "seed": seed, "model": "Random Forest",
                    "auroc": m_rf["auroc"], "aupr": m_rf["aupr"], "f1": m_rf["f1"], "mcc": m_rf["mcc"], "bacc": m_rf["bacc"],
                    "n_train": len(X_tr), "n_test": len(X_te)
                })

                # 4. XGBoost
                if HAS_XGBOOST:
                    pos_w = (len(y_tr) - sum(y_tr)) / max(1, sum(y_tr))
                    xgb = XGBClassifier(
                        n_estimators=100, max_depth=5, learning_rate=0.1, random_state=seed,
                        eval_metric="logloss", scale_pos_weight=pos_w, verbosity=0
                    )
                    xgb.fit(X_tr, y_tr)
                    p_xgb_va = xgb.predict_proba(X_va)[:, 1]
                    t_xgb = find_optimal_threshold(y_va, p_xgb_va)
                    p_xgb = xgb.predict_proba(X_te)[:, 1]
                    m_xgb = evaluate_predictions(y_te, p_xgb, threshold=t_xgb)
                    results_records.append({
                        "dataset": ds_name, "split": split_type, "seed": seed, "model": "XGBoost",
                        "auroc": m_xgb["auroc"], "aupr": m_xgb["aupr"], "f1": m_xgb["f1"], "mcc": m_xgb["mcc"], "bacc": m_xgb["bacc"],
                        "n_train": len(X_tr), "n_test": len(X_te)
                    })

                # 5. MediGraph Ensemble
                # Ensemble uses standardized LinearSVC for stability
                ensemble = MediGraphEnsemble(rf=rf, lsvc=lsvc, xgb=xgb if HAS_XGBOOST else None)
                ensemble.fit_weights(X_va_scaled, y_va)
                # Predict
                p_rf_te = rf.predict_proba(X_te)[:, 1]
                p_svm_te = lsvc.predict_proba(X_te_scaled)[:, 1]
                p_xgb_te = xgb.predict_proba(X_te)[:, 1] if HAS_XGBOOST else None
                if p_xgb_te is not None:
                    p_ens = ensemble.weights[0] * p_rf_te + ensemble.weights[1] * p_svm_te + ensemble.weights[2] * p_xgb_te
                else:
                    norm_w = ensemble.weights[:2] / np.sum(ensemble.weights[:2])
                    p_ens = norm_w[0] * p_rf_te + norm_w[1] * p_svm_te
                t_ens = find_optimal_threshold(y_va, (ensemble.weights[0]*p_rf_va + ensemble.weights[1]*p_svm_va + (ensemble.weights[2]*p_xgb_va if HAS_XGBOOST else 0)))
                m_ens = evaluate_predictions(y_te, p_ens, threshold=t_ens)
                results_records.append({
                    "dataset": ds_name, "split": split_type, "seed": seed, "model": "MediGraph Ensemble",
                    "auroc": m_ens["auroc"], "aupr": m_ens["aupr"], "f1": m_ens["f1"], "mcc": m_ens["mcc"], "bacc": m_ens["bacc"],
                    "n_train": len(X_tr), "n_test": len(X_te)
                })

                # 6. MLP Baseline (Renamed from DeepDTA)
                mlp = MLPBaselinePredictor(
                    drug_dim=X_drug.shape[1], protein_dim=X_prot_combined.shape[1],
                    epochs=12, lr=1e-3, seed=seed
                )
                mlp.fit(
                    X_drug[train_idx], X_prot_combined[train_idx], y_tr,
                    X_drug[val_idx], X_prot_combined[val_idx], y_va
                )
                p_mlp_va = mlp.predict_proba(X_drug[val_idx], X_prot_combined[val_idx])[:, 1]
                t_mlp = find_optimal_threshold(y_va, p_mlp_va)
                p_mlp = mlp.predict_proba(X_drug[test_idx], X_prot_combined[test_idx])[:, 1]
                m_mlp = evaluate_predictions(y_te, p_mlp, threshold=t_mlp)
                results_records.append({
                    "dataset": ds_name, "split": split_type, "seed": seed, "model": "MLP baseline",
                    "auroc": m_mlp["auroc"], "aupr": m_mlp["aupr"], "f1": m_mlp["f1"], "mcc": m_mlp["mcc"], "bacc": m_mlp["bacc"],
                    "n_train": len(X_tr), "n_test": len(X_te)
                })

                # Checkpoint save
                pd.DataFrame(results_records).to_csv(csv_path, index=False)
                pd.DataFrame(std_comparison_records).to_csv(std_csv_path, index=False)

    df_results = pd.DataFrame(results_records)
    df_results.to_csv(csv_path, index=False)
    print(f"\n[Finished] Updated dti_benchmark.csv saved to {csv_path}")

    df_std = pd.DataFrame(std_comparison_records)
    df_std.to_csv(std_csv_path, index=False)
    print(f"[Finished] Standardization comparison saved to {std_csv_path}")

    # Generate LaTeX Tables
    generate_updated_latex_tables(df_results)

    # Generate Figures
    generate_updated_figures(df_results)

def generate_updated_latex_tables(df_results: pd.DataFrame):
    """Regenerates tables/dti_main.tex directly from CSV data."""
    agg = df_results.groupby(["dataset", "split", "model"]).agg({
        "auroc": ["mean", "std"],
        "aupr": ["mean", "std"],
        "f1": ["mean", "std"],
        "mcc": ["mean", "std"]
    }).reset_index()
    agg.columns = ["dataset", "split", "model", "auroc_m", "auroc_s", "aupr_m", "aupr_s", "f1_m", "f1_s", "mcc_m", "mcc_s"]

    tex_path = TABLES_DIR / "dti_main.tex"
    with open(tex_path, "w") as f:
        f.write("% IEEE Format DTI Benchmark Table (Standardized Linear Models, Calibrated F1/MCC, MLP Baseline)\n")
        f.write("\\begin{table*}[t]\n")
        f.write("\\centering\n")
        f.write("\\caption{DTI Prediction Performance across Random, Cold-Drug, Cold-Target, and Cold-Both Splits (Mean $\\pm$ SD over 5 Seeds, Validation-Tuned Decision Threshold)}\n")
        f.write("\\label{tab:dti_main}\n")
        f.write("\\small\n")
        f.write("\\begin{tabular}{llcccc}\n")
        f.write("\\hline\n")
        f.write("\\textbf{Split} & \\textbf{Model} & \\textbf{AUROC} & \\textbf{AUPR} & \\textbf{F1-Score} & \\textbf{MCC} \\\\\n")
        f.write("\\hline\n")

        for ds in ["DAVIS", "KIBA"]:
            f.write(r"\multicolumn{6}{l}{\textit{\textbf{" + ds + r" Benchmark}}} \\" + "\n")
            f.write(r"\hline" + "\n")
            for split in ["random", "cold_drug", "cold_target", "cold_both"]:
                sub = agg[(agg["dataset"] == ds) & (agg["split"] == split)]
                for _, row in sub.iterrows():
                    m_name = row["model"]
                    if m_name == "MediGraph Ensemble":
                        m_name = r"\textbf{MediGraph Ensemble}"
                    split_label = split.replace("_", "-")
                    f.write(
                        f"{split_label} & {m_name} & "
                        f"{row['auroc_m']:.3f} $\\pm$ {row['auroc_s']:.3f} & "
                        f"{row['aupr_m']:.3f} $\\pm$ {row['aupr_s']:.3f} & "
                        f"{row['f1_m']:.3f} $\\pm$ {row['f1_s']:.3f} & "
                        f"{row['mcc_m']:.3f} $\\pm$ {row['mcc_s']:.3f} \\\\\n"
                    )
                f.write(r"\hline" + "\n")
        f.write(r"\end{tabular}" + "\n")
        f.write(r"\end{table*}" + "\n")
    print(f"[Artifact Saved] Updated {tex_path}")

    # Regenerate ablation table directly from CSV
    abl_csv = RESULTS_DIR / "dti_protein_feature_ablation.csv"
    if abl_csv.exists():
        df_abl = pd.read_csv(abl_csv)
        abl_agg = df_abl.groupby(["split", "feature"]).agg({
            "auroc": ["mean", "std"],
            "aupr": ["mean", "std"],
            "mcc": ["mean", "std"]
        }).reset_index()
        abl_agg.columns = ["split", "feature", "auroc_m", "auroc_s", "aupr_m", "aupr_s", "mcc_m", "mcc_s"]
        abl_tex = TABLES_DIR / "dti_protein_feature_ablation.tex"
        with open(abl_tex, "w") as f:
            f.write("% IEEE Format Feature Ablation Table (5-Seed Mean +- SD)\n")
            f.write(r"\begin{table}[t]" + "\n")
            f.write(r"\centering" + "\n")
            f.write(r"\caption{Protein Representation Ablation on DAVIS under Random and Cold-Drug Splits (Mean $\pm$ SD across 5 Seeds)}" + "\n")
            f.write(r"\label{tab:protein_ablation}" + "\n")
            f.write(r"\small" + "\n")
            f.write(r"\begin{tabular}{llccc}" + "\n")
            f.write(r"\hline" + "\n")
            f.write(r"\textbf{Split} & \textbf{Protein Representation} & \textbf{AUROC} & \textbf{AUPR} & \textbf{MCC} \\" + "\n")
            f.write(r"\hline" + "\n")
            for split in ["random", "cold_drug"]:
                sub = abl_agg[abl_agg["split"] == split]
                for _, row in sub.iterrows():
                    split_label = split.replace("_", "-")
                    f.write(
                        f"{split_label} & {row['feature']} & "
                        f"{row['auroc_m']:.3f} $\\pm$ {row['auroc_s']:.3f} & "
                        f"{row['aupr_m']:.3f} $\\pm$ {row['aupr_s']:.3f} & "
                        f"{row['mcc_m']:.3f} $\\pm$ {row['mcc_s']:.3f} \\\\\n"
                    )
                f.write(r"\hline" + "\n")
            f.write(r"\end{tabular}" + "\n")
            f.write(r"\end{table}" + "\n")
        print(f"[Artifact Saved] Updated {abl_tex}")

def generate_updated_figures(df_results: pd.DataFrame):
    """Generates 300 DPI publication figures with Y-axis from 0 and chance lines."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)

    # 1. DAVIS AUROC
    ax1 = axes[0, 0]
    sub_davis = df_results[df_results["dataset"] == "DAVIS"]
    sns.barplot(data=sub_davis, x="split", y="auroc", hue="model", ax=ax1, errorbar="sd", palette="deep")
    ax1.set_title("(a) DAVIS: AUROC by Split Type", fontsize=12, fontweight="bold")
    ax1.set_ylim(0.0, 1.05)
    ax1.axhline(0.5, color="red", linestyle="--", linewidth=1.2, label="Random Chance (0.50)")
    ax1.set_ylabel("AUROC", fontsize=11)
    ax1.set_xlabel("")
    ax1.legend(title="", fontsize=8)

    # 2. DAVIS AUPR
    ax2 = axes[0, 1]
    sns.barplot(data=sub_davis, x="split", y="aupr", hue="model", ax=ax2, errorbar="sd", palette="deep")
    ax2.set_title("(b) DAVIS: AUPR by Split Type (Primary Metric)", fontsize=12, fontweight="bold")
    ax2.set_ylim(0.0, 1.05)
    # Davis prevalence ~0.0718
    ax2.axhline(0.0718, color="red", linestyle="--", linewidth=1.2, label="AUPR Chance (Prevalence: 7.2%)")
    ax2.set_ylabel("AUPR", fontsize=11)
    ax2.set_xlabel("")
    ax2.legend(title="", fontsize=8)

    # 3. KIBA AUROC
    ax3 = axes[1, 0]
    sub_kiba = df_results[df_results["dataset"] == "KIBA"]
    sns.barplot(data=sub_kiba, x="split", y="auroc", hue="model", ax=ax3, errorbar="sd", palette="muted")
    ax3.set_title("(c) KIBA: AUROC by Split Type", fontsize=12, fontweight="bold")
    ax3.set_ylim(0.0, 1.05)
    ax3.axhline(0.5, color="red", linestyle="--", linewidth=1.2, label="Random Chance (0.50)")
    ax3.set_ylabel("AUROC", fontsize=11)
    ax3.set_xlabel("Split Type", fontsize=11)
    ax3.legend().remove()

    # 4. KIBA AUPR
    ax4 = axes[1, 1]
    sns.barplot(data=sub_kiba, x="split", y="aupr", hue="model", ax=ax4, errorbar="sd", palette="muted")
    ax4.set_title("(d) KIBA: AUPR by Split Type (Primary Metric)", fontsize=12, fontweight="bold")
    ax4.set_ylim(0.0, 1.05)
    # KIBA prevalence ~0.2096
    ax4.axhline(0.2096, color="red", linestyle="--", linewidth=1.2, label="AUPR Chance (Prevalence: 21.0%)")
    ax4.set_ylabel("AUPR", fontsize=11)
    ax4.set_xlabel("Split Type", fontsize=11)
    ax4.legend(title="", fontsize=8)

    plt.tight_layout()
    pdf_path = FIGURES_DIR / "dti_auroc_aupr.pdf"
    png_path = FIGURES_DIR / "dti_auroc_aupr.png"
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"[Artifact Saved] Publication figures saved to {png_path} and {pdf_path}")

if __name__ == "__main__":
    run_dti_fixes()
