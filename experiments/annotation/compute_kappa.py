"""
experiments/annotation/compute_kappa.py
---------------------------------------
Computes Cohen's Kappa (kappa) and observed agreement between two independent
human annotators or between human and automated LLM judge.
"""

import sys
import argparse
import pandas as pd
from sklearn.metrics import cohen_kappa_score

def compute_inter_annotator_agreement(file_1: str, file_2: str):
    df1 = pd.read_csv(file_1)
    df2 = pd.read_csv(file_2)

    # Ensure aligned claim_ids
    merged = pd.merge(df1, df2, on="claim_id", suffixes=("_annotator1", "_annotator2"))
    print(f"Total overlapping annotated claims: {len(merged)}")

    y1 = merged["label_annotator1"].astype(str).str.strip().str.upper()
    y2 = merged["label_annotator2"].astype(str).str.strip().str.upper()

    kappa = cohen_kappa_score(y1, y2)
    obs_agreement = (y1 == y2).mean() * 100.0

    print("=" * 50)
    print("INTER-ANNOTATOR AGREEMENT REPORT")
    print("=" * 50)
    print(f"Observed Agreement: {obs_agreement:.2f}%")
    print(f"Cohen's Kappa (kappa): {kappa:.4f}")
    
    # Interpretation according to Landis & Koch (1977)
    if kappa < 0:
        interpretation = "Poor (Less than chance)"
    elif kappa <= 0.20:
        interpretation = "Slight agreement"
    elif kappa <= 0.40:
        interpretation = "Fair agreement"
    elif kappa <= 0.60:
        interpretation = "Moderate agreement"
    elif kappa <= 0.80:
        interpretation = "Substantial agreement"
    else:
        interpretation = "Almost perfect agreement"

    print(f"Interpretation: {interpretation}")
    print("=" * 50)
    return kappa, obs_agreement

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute Cohen's Kappa between two annotation sets.")
    parser.add_argument("file_1", type=str, help="Path to first annotator CSV")
    parser.add_argument("file_2", type=str, help="Path to second annotator CSV")
    args = parser.parse_args()

    compute_inter_annotator_agreement(args.file_1, args.file_2)
