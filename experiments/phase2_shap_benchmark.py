"""
experiments/phase2_shap_benchmark.py
------------------------------------
Rigorous empirical benchmark for SHAP Explanation Validity & Faithfulness (Phase 2 - RQ3).
Methodological Audit Updates (Audit Issue 2 & Issue 4):
1. Progressive feature masking evaluated with DUAL masking variants:
   - Variant A ('zero'): features set to 0.0.
   - Variant B ('train_mean'): features replaced with training-set column mean.
2. Global feature rankings derived strictly from training data (subsample of 300 training rows),
   NOT from test rows being evaluated.
3. Random baseline averaged across 20 independent random feature sets per seed, eliminating noisy single draws.
4. Evaluates on both COLD_DRUG and RANDOM splits across all 5 seeds (0..4) at k in {0, 5, 10, 20, 50}.
5. Per-seed paired comparisons: reports number of seeds (out of 5) where SHAP degradation beats Random.
6. Attribution stability evaluated via bootstrap-resampling of the training set (B=5).
7. Figure shap_faithfulness_curves.png regenerated with y-axis from 0, error bars (SD over seeds),
   and 20-repetition averaged random baseline.
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem, Draw

from phase1_dti_benchmark import compute_ecfp4, compute_aac, compute_dipeptide, compute_ctd, generate_splits

BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "results"
TABLES_DIR = BASE_DIR / "tables"
FIGURES_DIR = BASE_DIR / "figures"
DATA_DIR = BASE_DIR / "data"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------
# 1. Progressive Masking Faithfulness Benchmark (Dual Masking)
# -------------------------------------------------------------
def run_progressive_masking_benchmark(
    df_data: pd.DataFrame,
    X_all: np.ndarray,
    y_all: np.ndarray,
    splits: List[str] = ["cold_drug", "random"],
    seeds: List[int] = [0, 1, 2, 3, 4],
    k_values: List[int] = [0, 5, 10, 20, 50],
    n_random_reps: int = 20
) -> pd.DataFrame:
    """
    Evaluates progressive feature masking across splits and seeds.
    Compares SHAP vs. MDI vs. 20-rep averaged Random baseline.
    Evaluates both 'zero' and 'train_mean' masking variants.
    Features ranked by mean |SHAP| on training data.
    """
    print(f"\n--- Running Progressive Masking Benchmark across Splits: {splits} (Seeds: {seeds}) ---", flush=True)
    masking_records = []
    n_feats = X_all.shape[1]

    for split in splits:
        for seed in seeds:
            print(f"[{split.upper()} | Seed {seed}] Training RF and calculating SHAP on training data...", flush=True)
            train_df, val_df, test_df = generate_splits(df_data, split, seed)
            train_idx = train_df.index.values
            test_idx = test_df.index.values

            X_tr, y_tr = X_all[train_idx], y_all[train_idx]
            X_te, y_te = X_all[test_idx], y_all[test_idx]

            rf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=seed, n_jobs=-1)
            rf.fit(X_tr, y_tr)

            # Feature ranking derived from TRAINING SET subsample (N=300), preserving test set independence
            rng_rank = np.random.RandomState(seed + 100)
            tr_sub_size = min(300, len(X_tr))
            tr_sub_idx = rng_rank.choice(len(X_tr), size=tr_sub_size, replace=False)
            X_tr_sub = X_tr[tr_sub_idx]

            explainer = shap.TreeExplainer(rf)
            sv_tr = explainer.shap_values(X_tr_sub)
            if isinstance(sv_tr, list):
                sv_tr_pos = sv_tr[1]
            elif len(sv_tr.shape) == 3:
                sv_tr_pos = sv_tr[:, :, 1]
            else:
                sv_tr_pos = sv_tr

            shap_global = np.abs(sv_tr_pos).mean(axis=0)
            shap_ranking = np.argsort(shap_global)[::-1]

            mdi_global = rf.feature_importances_
            mdi_ranking = np.argsort(mdi_global)[::-1]

            # Test evaluation subsample (size 300)
            rng_eval = np.random.RandomState(seed)
            te_sub_size = min(300, len(X_te))
            te_sub_idx = rng_eval.choice(len(X_te), size=te_sub_size, replace=False)
            X_sub = X_te[te_sub_idx]
            y_sub = y_te[te_sub_idx]
            p_sub_orig = rf.predict_proba(X_sub)[:, 1]

            base_auroc = roc_auc_score(y_sub, p_sub_orig) if len(np.unique(y_sub)) > 1 else 0.5
            base_aupr = average_precision_score(y_sub, p_sub_orig) if len(np.unique(y_sub)) > 1 else 0.0

            # Training set feature mean for train_mean masking variant
            tr_feat_means = X_tr.mean(axis=0)

            masking_variants = ["zero", "train_mean"]

            for mask_var in masking_variants:
                # k = 0 baseline (unmasked)
                for method_name in ["SHAP (MediGraph)", "MDI (Gini)", "Random Baseline (20 reps avg)"]:
                    masking_records.append({
                        "split": split,
                        "seed": seed,
                        "masking_type": mask_var,
                        "k_masked": 0,
                        "method": method_name,
                        "auroc": float(base_auroc),
                        "aupr": float(base_aupr),
                        "mean_prediction_drop": 0.0,
                        "mean_abs_drop": 0.0
                    })

                for k in [k for k in k_values if k > 0]:
                    # 1. SHAP masking
                    top_k_shap = shap_ranking[:k]
                    X_masked_shap = X_sub.copy()
                    if mask_var == "zero":
                        X_masked_shap[:, top_k_shap] = 0.0
                    else:
                        X_masked_shap[:, top_k_shap] = tr_feat_means[top_k_shap]

                    p_shap = rf.predict_proba(X_masked_shap)[:, 1]
                    shap_auroc = roc_auc_score(y_sub, p_shap) if len(np.unique(y_sub)) > 1 else 0.5
                    shap_aupr = average_precision_score(y_sub, p_shap) if len(np.unique(y_sub)) > 1 else 0.0
                    shap_pred_drop = float(np.mean(p_sub_orig - p_shap))
                    shap_abs_drop = float(np.mean(np.abs(p_sub_orig - p_shap)))

                    masking_records.append({
                        "split": split,
                        "seed": seed,
                        "masking_type": mask_var,
                        "k_masked": k,
                        "method": "SHAP (MediGraph)",
                        "auroc": float(shap_auroc),
                        "aupr": float(shap_aupr),
                        "mean_prediction_drop": shap_pred_drop,
                        "mean_abs_drop": shap_abs_drop
                    })

                    # 2. MDI masking
                    top_k_mdi = mdi_ranking[:k]
                    X_masked_mdi = X_sub.copy()
                    if mask_var == "zero":
                        X_masked_mdi[:, top_k_mdi] = 0.0
                    else:
                        X_masked_mdi[:, top_k_mdi] = tr_feat_means[top_k_mdi]

                    p_mdi = rf.predict_proba(X_masked_mdi)[:, 1]
                    mdi_auroc = roc_auc_score(y_sub, p_mdi) if len(np.unique(y_sub)) > 1 else 0.5
                    mdi_aupr = average_precision_score(y_sub, p_mdi) if len(np.unique(y_sub)) > 1 else 0.0
                    mdi_pred_drop = float(np.mean(p_sub_orig - p_mdi))
                    mdi_abs_drop = float(np.mean(np.abs(p_sub_orig - p_mdi)))

                    masking_records.append({
                        "split": split,
                        "seed": seed,
                        "masking_type": mask_var,
                        "k_masked": k,
                        "method": "MDI (Gini)",
                        "auroc": float(mdi_auroc),
                        "aupr": float(mdi_aupr),
                        "mean_prediction_drop": mdi_pred_drop,
                        "mean_abs_drop": mdi_abs_drop
                    })

                    # 3. Random Baseline (averaged across 20 repetitions)
                    rnd_aurocs, rnd_auprs, rnd_pred_drops, rnd_abs_drops = [], [], [], []
                    rng_reps = np.random.RandomState(seed * 1000 + k)
                    for _ in range(n_random_reps):
                        rand_feats = rng_reps.choice(n_feats, size=k, replace=False)
                        X_masked_rnd = X_sub.copy()
                        if mask_var == "zero":
                            X_masked_rnd[:, rand_feats] = 0.0
                        else:
                            X_masked_rnd[:, rand_feats] = tr_feat_means[rand_feats]

                        p_rnd = rf.predict_proba(X_masked_rnd)[:, 1]
                        r_auc = roc_auc_score(y_sub, p_rnd) if len(np.unique(y_sub)) > 1 else 0.5
                        r_apr = average_precision_score(y_sub, p_rnd) if len(np.unique(y_sub)) > 1 else 0.0
                        rnd_aurocs.append(r_auc)
                        rnd_auprs.append(r_apr)
                        rnd_pred_drops.append(float(np.mean(p_sub_orig - p_rnd)))
                        rnd_abs_drops.append(float(np.mean(np.abs(p_sub_orig - p_rnd))))

                    masking_records.append({
                        "split": split,
                        "seed": seed,
                        "masking_type": mask_var,
                        "k_masked": k,
                        "method": "Random Baseline (20 reps avg)",
                        "auroc": float(np.mean(rnd_aurocs)),
                        "aupr": float(np.mean(rnd_auprs)),
                        "mean_prediction_drop": float(np.mean(rnd_pred_drops)),
                        "mean_abs_drop": float(np.mean(rnd_abs_drops))
                    })

    df_results = pd.DataFrame(masking_records)
    csv_path = RESULTS_DIR / "shap_faithfulness.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"[Artifact Saved] Faithfulness results saved to {csv_path}", flush=True)
    return df_results

# -------------------------------------------------------------
# 2. Attribution Stability Benchmark
# -------------------------------------------------------------
def run_bootstrap_stability_benchmark(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    B: int = 5,
    top_k_jaccard_list: List[int] = [20, 50],
    top_k_spearman: int = 100
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    print(f"\n--- Running Training Data Bootstrap Resampling Stability (B={B}) ---", flush=True)
    n_train = len(X_train)
    rng = np.random.RandomState(42)

    X_eval = X_test[:250]
    bootstrap_shap_vectors = []
    bootstrap_top20 = []
    bootstrap_top50 = []

    for b in range(B):
        print(f"[Bootstrap {b+1}/{B}] Fitting model on resampled training set...", flush=True)
        boot_idx = rng.choice(n_train, size=n_train, replace=True)
        X_b, y_b = X_train[boot_idx], y_train[boot_idx]

        rf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1)
        rf.fit(X_b, y_b)

        explainer = shap.TreeExplainer(rf)
        sv = explainer.shap_values(X_eval)
        if isinstance(sv, list):
            sv_pos = sv[1]
        elif len(sv.shape) == 3:
            sv_pos = sv[:, :, 1]
        else:
            sv_pos = sv

        global_imp = np.abs(sv_pos).mean(axis=0)
        bootstrap_shap_vectors.append(global_imp)

        ranked = np.argsort(global_imp)[::-1]
        bootstrap_top20.append(set(ranked[:20]))
        bootstrap_top50.append(set(ranked[:50]))

    pairwise_records = []
    jaccards_20, jaccards_50, spearmans_100 = [], [], []

    for i in range(B):
        for j in range(i + 1, B):
            j20 = len(bootstrap_top20[i] & bootstrap_top20[j]) / len(bootstrap_top20[i] | bootstrap_top20[j])
            jaccards_20.append(j20)

            j50 = len(bootstrap_top50[i] & bootstrap_top50[j]) / len(bootstrap_top50[i] | bootstrap_top50[j])
            jaccards_50.append(j50)

            top_union = list(set(np.argsort(bootstrap_shap_vectors[i])[::-1][:top_k_spearman]) |
                             set(np.argsort(bootstrap_shap_vectors[j])[::-1][:top_k_spearman]))[:top_k_spearman]
            v1 = bootstrap_shap_vectors[i][top_union]
            v2 = bootstrap_shap_vectors[j][top_union]
            rho, _ = spearmanr(v1, v2)
            spearmans_100.append(float(rho))

            pairwise_records.append({
                "bootstrap_pair": f"{i}-{j}",
                "jaccard_top20": float(j20),
                "jaccard_top50": float(j50),
                "spearman_rho_top100": float(rho)
            })

    df_stab = pd.DataFrame(pairwise_records)
    csv_path = RESULTS_DIR / "shap_stability.csv"
    df_stab.to_csv(csv_path, index=False)
    print(f"[Artifact Saved] Bootstrap stability results saved to {csv_path}", flush=True)

    summary_stats = {
        "jaccard_top20_mean": float(np.mean(jaccards_20)),
        "jaccard_top20_std": float(np.std(jaccards_20)),
        "jaccard_top50_mean": float(np.mean(jaccards_50)),
        "jaccard_top50_std": float(np.std(jaccards_50)),
        "spearman_top100_mean": float(np.mean(spearmans_100)),
        "spearman_top100_std": float(np.std(spearmans_100))
    }
    return df_stab, summary_stats

# -------------------------------------------------------------
# 3. LaTeX Table Generation
# -------------------------------------------------------------
def generate_shap_latex_tables(df_faith: pd.DataFrame, stab_stats: Dict[str, float]):
    cd_zero = df_faith[(df_faith["split"] == "cold_drug") & (df_faith["masking_type"] == "zero")]
    agg = cd_zero.groupby(["k_masked", "method"]).agg({
        "aupr": ["mean", "std"],
        "mean_prediction_drop": ["mean", "std"]
    }).reset_index()
    agg.columns = ["k_masked", "method", "aupr_mean", "aupr_std", "drop_mean", "drop_std"]

    tex_faith = TABLES_DIR / "shap_faithfulness.tex"
    with open(tex_faith, "w") as f:
        f.write("% IEEE Format SHAP Progressive Masking Faithfulness Table (Cold-Drug Split, Zero Masking)\n")
        f.write(r"\begin{table}[t]" + "\n")
        f.write(r"\centering" + "\n")
        f.write(r"\caption{Progressive Feature Masking on TDC DAVIS Cold-Drug Split (Zero Masking; Mean $\pm$ SD across 5 Seeds)}" + "\n")
        f.write(r"\label{tab:shap_faithfulness}" + "\n")
        f.write(r"\small" + "\n")
        f.write(r"\begin{tabular}{lcccc}" + "\n")
        f.write(r"\hline" + "\n")
        f.write(r"\textbf{Method} & \textbf{$k=0$ (Baseline)} & \textbf{$k=5$} & \textbf{$k=20$} & \textbf{$k=50$} \\" + "\n")
        f.write(r"\hline" + "\n")

        for m_name in ["SHAP (MediGraph)", "MDI (Gini)", "Random Baseline (20 reps avg)"]:
            sub = agg[agg["method"] == m_name]
            vals = {}
            for _, r in sub.iterrows():
                vals[int(r["k_masked"])] = f"{r['aupr_mean']:.3f} $\\pm$ {r['aupr_std']:.3f}"
            disp_m = f"\\textbf{{{m_name}}}" if "SHAP" in m_name else m_name
            f.write(
                f"{disp_m} & {vals.get(0, 'N/A')} & {vals.get(5, 'N/A')} & {vals.get(20, 'N/A')} & {vals.get(50, 'N/A')} \\\\\n"
            )

        f.write(r"\hline" + "\n")
        f.write(r"\end{tabular}" + "\n")
        f.write(r"\end{table}" + "\n")

    tex_stab = TABLES_DIR / "shap_stability.tex"
    with open(tex_stab, "w") as f:
        f.write("% IEEE Format Attribution Stability Table (Training Bootstrap Resampling)\n")
        f.write(r"\begin{table}[t]" + "\n")
        f.write(r"\centering" + "\n")
        f.write(r"\caption{Attribution Stability Under Training Data Bootstrap Resampling ($B=5$)}" + "\n")
        f.write(r"\label{tab:shap_stability}" + "\n")
        f.write(r"\small" + "\n")
        f.write(r"\begin{tabular}{lc}" + "\n")
        f.write(r"\hline" + "\n")
        f.write(r"\textbf{Metric} & \textbf{Mean $\pm$ SD across Resample Pairs} \\" + "\n")
        f.write(r"\hline" + "\n")
        f.write(f"Jaccard Overlap (Top-20 Features) & {stab_stats['jaccard_top20_mean']:.3f} $\\pm$ {stab_stats['jaccard_top20_std']:.3f} \\\\\n")
        f.write(f"Jaccard Overlap (Top-50 Features) & {stab_stats['jaccard_top50_mean']:.3f} $\\pm$ {stab_stats['jaccard_top50_std']:.3f} \\\\\n")
        f.write(f"Spearman Rank Correlation (Top-100 Non-Zero Features) & {stab_stats['spearman_top100_mean']:.3f} $\\pm$ {stab_stats['spearman_top100_std']:.3f} \\\\\n")
        f.write(r"\hline" + "\n")
        f.write(r"\end{tabular}" + "\n")
        f.write(r"\end{table}" + "\n")

    print(f"[Artifact Saved] Faithfulness & stability LaTeX tables written to {TABLES_DIR}", flush=True)

# -------------------------------------------------------------
# 4. Publication Figure Generation
# -------------------------------------------------------------
def generate_shap_figures(df_faith: pd.DataFrame):
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

    # Filter to cold_drug split, zero masking
    df_cd = df_faith[(df_faith["split"] == "cold_drug") & (df_faith["masking_type"] == "zero")]

    # 1. AUPR Degradation under progressive masking
    ax1 = axes[0]
    sns.lineplot(
        data=df_cd, x="k_masked", y="aupr", hue="method",
        marker="o", linewidth=2.0, ax=ax1, errorbar="sd",
        palette=["#d95f02", "#7570b3", "#1b9e77"]
    )
    ax1.set_title("(a) Cold-Drug AUPR under Progressive Masking (Zero Mask)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Number of Top Features Masked ($k$)", fontsize=10)
    ax1.set_ylabel("AUPR on Held-Out Test Drugs", fontsize=10)
    ax1.axhline(0.0716, color="red", linestyle="--", linewidth=1.2, label="DAVIS Base Positive Rate (7.2%)")
    ax1.set_ylim(0.0, 0.45)  # y-axis starts strictly from 0
    ax1.legend(fontsize=8)

    # 2. Prediction Score Drop (Faithfulness)
    ax2 = axes[1]
    df_cd_drops = df_cd[df_cd["k_masked"] > 0]
    sns.lineplot(
        data=df_cd_drops, x="k_masked", y="mean_prediction_drop", hue="method",
        marker="s", linewidth=2.0, ax=ax2, errorbar="sd",
        palette=["#d95f02", "#7570b3", "#1b9e77"]
    )
    ax2.set_title("(b) Model Prediction Probability Drop (Zero Mask)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Number of Top Features Masked ($k$)", fontsize=10)
    ax2.set_ylabel("Mean Probability Drop ($p_{orig} - p_{masked}$)", fontsize=10)
    ax2.set_ylim(0.0, None)  # y-axis starts strictly from 0
    ax2.legend(fontsize=8)

    plt.tight_layout()
    png_path = FIGURES_DIR / "shap_faithfulness_curves.png"
    pdf_path = FIGURES_DIR / "shap_faithfulness_curves.pdf"
    plt.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()
    print(f"[Artifact Saved] Faithfulness publication figures saved to {png_path} and {pdf_path}", flush=True)

# -------------------------------------------------------------
# 5. Main Runner
# -------------------------------------------------------------
def run_all_shap_benchmarks():
    print("=" * 70)
    print("STARTING REDESIGNED SHAP FAITHFULNESS & STABILITY BENCHMARK")
    print("=" * 70, flush=True)

    df_davis = pd.read_csv(DATA_DIR / "davis.tab", sep="\t")
    df_davis.rename(columns={"X1": "Drug", "X2": "Target"}, inplace=True)
    df_davis["y_binary"] = (df_davis["Y"] <= 100.0).astype(int)

    unique_smiles = df_davis["Drug"].unique()
    unique_seqs = df_davis["Target"].unique()

    print("Computing molecular fingerprints and protein descriptors...", flush=True)
    drug_fps = {s: compute_ecfp4(s) for s in unique_smiles}
    prot_aac = {seq: compute_aac(seq) for seq in unique_seqs}
    prot_dipep = {seq: compute_dipeptide(seq) for seq in unique_seqs}
    prot_ctd = {seq: compute_ctd(seq) for seq in unique_seqs}

    X_tr_drug = np.array([drug_fps[s] for s in df_davis["Drug"]])
    X_tr_prot = np.array([np.hstack([prot_aac[seq], prot_dipep[seq], prot_ctd[seq]]) for seq in df_davis["Target"]])
    X_all = np.hstack([X_tr_drug, X_tr_prot])
    y_all = df_davis["y_binary"].values

    # 1. Progressive Masking Benchmark on cold_drug and random split
    df_faith = run_progressive_masking_benchmark(
        df_davis, X_all, y_all,
        splits=["cold_drug", "random"],
        seeds=[0, 1, 2, 3, 4],
        k_values=[0, 5, 10, 20, 50],
        n_random_reps=20
    )

    # 2. Bootstrap Stability Benchmark
    train_df, _, test_df = generate_splits(df_davis, "cold_drug", 0)
    X_tr_stab = X_all[train_df.index.values]
    y_tr_stab = y_all[train_df.index.values]
    X_te_stab = X_all[test_df.index.values]

    df_stab, stab_stats = run_bootstrap_stability_benchmark(
        X_tr_stab, y_tr_stab, X_te_stab, B=5
    )

    # 3. LaTeX Tables
    generate_shap_latex_tables(df_faith, stab_stats)

    # 4. Publication Figures
    generate_shap_figures(df_faith)

    # 5. Compute Paired Comparisons Summary
    print("\n--- Paired Comparison Summary: SHAP vs. Random Baseline (20 reps) ---")
    for split in ["cold_drug", "random"]:
        for m_type in ["zero", "train_mean"]:
            sub = df_faith[(df_faith["split"] == split) & (df_faith["masking_type"] == m_type)]
            print(f"\n[{split} | {m_type}]")
            for k in [5, 10, 20, 50]:
                shap_sub = sub[(sub["k_masked"] == k) & (sub["method"] == "SHAP (MediGraph)")].set_index("seed")
                rnd_sub = sub[(sub["k_masked"] == k) & (sub["method"] == "Random Baseline (20 reps avg)")].set_index("seed")
                
                # SHAP-minus-Random AUPR
                diff_aupr = shap_sub["aupr"] - rnd_sub["aupr"]
                # In faithfulness, higher prediction drop or lower remaining AUPR indicates greater effect
                # Seeds where SHAP has greater prediction drop than random:
                drop_beats = (shap_sub["mean_prediction_drop"] > rnd_sub["mean_prediction_drop"]).sum()
                # Seeds where SHAP remaining AUPR is lower than random (more degradation):
                aupr_degraded = (shap_sub["aupr"] < rnd_sub["aupr"]).sum()
                
                print(f"  k={k}: Mean SHAP-Random AUPR = {diff_aupr.mean():.4f} (SD={diff_aupr.std():.4f}) | "
                      f"Seeds SHAP drop > Random drop: {drop_beats}/5 | "
                      f"Seeds SHAP AUPR < Random AUPR: {aupr_degraded}/5")

    print("\n[Finished] SHAP Benchmarks Completed Successfully!", flush=True)

if __name__ == "__main__":
    run_all_shap_benchmarks()
