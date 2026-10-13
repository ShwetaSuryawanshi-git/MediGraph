# MediGraph: Figures & Tables Index

All artifacts are publication-ready for IEEE-format papers.

---

## LaTeX Tables

| Table | File | Description | Paper Section |
|---|---|---|---|
| Table I | [`dti_main.tex`](file:///Users/shweta/Desktop/vit%20chennai/major%20project%20fy/MediGraph/experiments/tables/dti_main.tex) | DTI benchmark: AUROC, AUPR, F1, MCC across all models, splits, datasets (mean ± SD, 5 seeds) | RQ1 Results |
| Table II | [`dti_protein_feature_ablation.tex`](file:///Users/shweta/Desktop/vit%20chennai/major%20project%20fy/MediGraph/experiments/tables/dti_protein_feature_ablation.tex) | Protein feature ablation: AAC vs Dipeptide vs CTD vs ESM-2 | RQ1 Discussion |
| Table III | [`rag_comparison.tex`](file:///Users/shweta/Desktop/vit%20chennai/major%20project%20fy/MediGraph/experiments/tables/rag_comparison.tex) | RAG evaluation: Total claims, Support Rate (%), Hallucination (%), and Negative Control Abstention (%) | RQ2 Results |
| Table IV | [`shap_faithfulness.tex`](file:///Users/shweta/Desktop/vit%20chennai/major%20project%20fy/MediGraph/experiments/tables/shap_faithfulness.tex) | SHAP perturbation faithfulness test results | RQ3 Results |
| Table V | [`shap_stability.tex`](file:///Users/shweta/Desktop/vit%20chennai/major%20project%20fy/MediGraph/experiments/tables/shap_stability.tex) | SHAP cross-seed stability (Jaccard, Spearman) | RQ3 Results |
| Table VI | [`repurposing_recovery.tex`](file:///Users/shweta/Desktop/vit%20chennai/major%20project%20fy/MediGraph/experiments/tables/repurposing_recovery.tex) | Retrospective repurposing recovery: MRR, Hits@k | RQ4 Results |

---

## Publication Figures (300 DPI PNG + Vector PDF)

| Figure | Files | Description | Paper Section |
|---|---|---|---|
| Fig. 1 | [`dti_auroc_aupr.png`](file:///Users/shweta/Desktop/vit%20chennai/major%20project%20fy/MediGraph/experiments/figures/dti_auroc_aupr.png) / [`.pdf`](file:///Users/shweta/Desktop/vit%20chennai/major%20project%20fy/MediGraph/experiments/figures/dti_auroc_aupr.pdf) | Grouped bar chart comparing AUROC/AUPR across models and splits | RQ1 Results |
| Fig. 2 | [`rag_hallucination_rate.png`](file:///Users/shweta/Desktop/vit%20chennai/major%20project%20fy/MediGraph/experiments/figures/rag_hallucination_rate.png) / [`.pdf`](file:///Users/shweta/Desktop/vit%20chennai/major%20project%20fy/MediGraph/experiments/figures/rag_hallucination_rate.pdf) | Two-panel bar chart: (a) Claim hallucination rate, (b) Literature support rate by category | RQ2 Results |
| Fig. 3 | [`shap_faithfulness_curves.png`](file:///Users/shweta/Desktop/vit%20chennai/major%20project%20fy/MediGraph/experiments/figures/shap_faithfulness_curves.png) / [`.pdf`](file:///Users/shweta/Desktop/vit%20chennai/major%20project%20fy/MediGraph/experiments/figures/shap_faithfulness_curves.pdf) | SHAP vs Random vs MDI feature masking curves | RQ3 Results |
| Fig. 4 | [`repurposing_mrr_hits.png`](file:///Users/shweta/Desktop/vit%20chennai/major%20project%20fy/MediGraph/experiments/figures/repurposing_mrr_hits.png) / [`.pdf`](file:///Users/shweta/Desktop/vit%20chennai/major%20project%20fy/MediGraph/experiments/figures/repurposing_mrr_hits.pdf) | Repurposing MRR and Hits@k comparison bar chart | RQ4 Results |

---

## Experimental Datasets & Artifacts

| File | Rows | Description |
|---|---|---|
| [`dti_benchmark.csv`](file:///Users/shweta/Desktop/vit%20chennai/major%20project%20fy/MediGraph/experiments/results/dti_benchmark.csv) | 240 | Full DTI benchmark results |
| [`dti_protein_feature_ablation.csv`](file:///Users/shweta/Desktop/vit%20chennai/major%20project%20fy/MediGraph/experiments/results/dti_protein_feature_ablation.csv) | 60 | Protein feature ablation |
| [`rag_evaluation.csv`](file:///Users/shweta/Desktop/vit%20chennai/major%20project%20fy/MediGraph/experiments/results/rag_evaluation.csv) | 180 | RAG condition evaluation across 60 triples (C0, C1, C2) |
| [`rag_claims_detailed.csv`](file:///Users/shweta/Desktop/vit%20chennai/major%20project%20fy/MediGraph/experiments/results/rag_claims_detailed.csv) | 1,138 | Decomposed atomic claims with verdicts and rationales |
| [`sample_for_annotation.csv`](file:///Users/shweta/Desktop/vit%20chennai/major%20project%20fy/MediGraph/experiments/annotation/sample_for_annotation.csv) | 30 | Stratified sample for two-annotator blind verification |
| [`shap_faithfulness.csv`](file:///Users/shweta/Desktop/vit%20chennai/major%20project%20fy/MediGraph/experiments/results/shap_faithfulness.csv) | 21 | Feature masking test |
| [`shap_stability.csv`](file:///Users/shweta/Desktop/vit%20chennai/major%20project%20fy/MediGraph/experiments/results/shap_stability.csv) | 10 | Seed-pair stability metrics |
| [`repurposing_benchmark.csv`](file:///Users/shweta/Desktop/vit%20chennai/major%20project%20fy/MediGraph/experiments/results/repurposing_benchmark.csv) | 10 | Per-drug repurposing rankings |
| [`repurposing_metrics.json`](file:///Users/shweta/Desktop/vit%20chennai/major%20project%20fy/MediGraph/experiments/results/repurposing_metrics.json) | — | Aggregate MRR/Hits@k |
