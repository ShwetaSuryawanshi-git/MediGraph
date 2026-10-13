# Experimental Progress Tracker

**Project**: MediGraph Empirical Study for IEEE Conference / NCSET  
**Last Updated**: October 9, 2026  
**Status**: **ALL PHASES COMPLETED (Phases 0–5 100% Complete)**

---

## Phase Status Summary

| Phase | Description | Status | Key Output Files | Blockers / Deviations |
|---|---|---|---|---|
| **Phase 0** | Setup and Audit | **DONE** ✅ | [`AUDIT.md`](file:///Users/shweta/Desktop/vit%20chennai/major%20project%20fy/MediGraph/experiments/AUDIT.md), [`requirements.txt`](file:///Users/shweta/Desktop/vit%20chennai/major%20project%20fy/MediGraph/experiments/requirements.txt), [`CHANGES.md`](file:///Users/shweta/Desktop/vit%20chennai/major%20project%20fy/MediGraph/experiments/CHANGES.md) | Original `ml_pipeline/venv` verified intact (all `.pkl` load). Isolated `experiments/venv` created. Version requirements frozen. |
| **Phase 1** | DTI Benchmark (RQ1) | **DONE** ✅ | [`results/dti_benchmark.csv`](file:///Users/shweta/Desktop/vit%20chennai/major%20project%20fy/MediGraph/experiments/results/dti_benchmark.csv) (240 rows), [`tables/dti_main.tex`](file:///Users/shweta/Desktop/vit%20chennai/major%20project%20fy/MediGraph/experiments/tables/dti_main.tex), [`figures/dti_auroc_aupr.png`](file:///Users/shweta/Desktop/vit%20chennai/major%20project%20fy/MediGraph/experiments/figures/dti_auroc_aupr.png) | DeepDTA in PyTorch; KIBA uses Calibrated LinearSVC / subsampling. MediGraph ensemble ranks 1st/2nd in 7 of 8 splits. |
| **Phase 2** | SHAP Validity (RQ3) | **DONE** ✅ | [`results/shap_faithfulness.csv`](file:///Users/shweta/Desktop/vit%20chennai/major%20project%20fy/MediGraph/experiments/results/shap_faithfulness.csv), [`results/shap_stability.csv`](file:///Users/shweta/Desktop/vit%20chennai/major%20project%20fy/MediGraph/experiments/results/shap_stability.csv), [`tables/shap_faithfulness.tex`](file:///Users/shweta/Desktop/vit%20chennai/major%20project%20fy/MediGraph/experiments/tables/shap_faithfulness.tex), [`figures/shap_faithfulness_curves.png`](file:///Users/shweta/Desktop/vit%20chennai/major%20project%20fy/MediGraph/experiments/figures/shap_faithfulness_curves.png) | 4.3× degradation vs random; rank stability ρ = 0.942; chemical fragment validation. |
| **Phase 3** | RAG Evaluation (RQ2) | **DONE** ✅ | [`results/rag_evaluation.csv`](file:///Users/shweta/Desktop/vit%20chennai/major%20project%20fy/MediGraph/experiments/results/rag_evaluation.csv), [`results/rag_claims_detailed.csv`](file:///Users/shweta/Desktop/vit%20chennai/major%20project%20fy/MediGraph/experiments/results/rag_claims_detailed.csv), [`tables/rag_comparison.tex`](file:///Users/shweta/Desktop/vit%20chennai/major%20project%20fy/MediGraph/experiments/tables/rag_comparison.tex), [`figures/rag_hallucination_rate.png`](file:///Users/shweta/Desktop/vit%20chennai/major%20project%20fy/MediGraph/experiments/figures/rag_hallucination_rate.png) | 60 stratified triples (25 supported, 20 plausible, 15 unsupported). 1,138 claims verified. Staged RAG achieves 100% support vs 8.8% for LLM-only. |
| **Phase 4** | Retrospective Repurposing (RQ4) | **DONE** ✅ | [`results/repurposing_benchmark.csv`](file:///Users/shweta/Desktop/vit%20chennai/major%20project%20fy/MediGraph/experiments/results/repurposing_benchmark.csv), [`results/repurposing_metrics.json`](file:///Users/shweta/Desktop/vit%20chennai/major%20project%20fy/MediGraph/experiments/results/repurposing_metrics.json), [`tables/repurposing_recovery.tex`](file:///Users/shweta/Desktop/vit%20chennai/major%20project%20fy/MediGraph/experiments/tables/repurposing_recovery.tex), [`figures/repurposing_mrr_hits.png`](file:///Users/shweta/Desktop/vit%20chennai/major%20project%20fy/MediGraph/experiments/figures/repurposing_mrr_hits.png) | MRR 0.043 vs 0.018 random (2.4× improvement); Aspirin (14), Metformin (15), Tamoxifen (16) recovered. |
| **Phase 5** | Report & Publication Artifacts | **DONE** ✅ | [`REPORT.md`](file:///Users/shweta/Desktop/vit%20chennai/major%20project%20fy/MediGraph/experiments/REPORT.md), [`FIGURES_TABLES_INDEX.md`](file:///Users/shweta/Desktop/vit%20chennai/major%20project%20fy/MediGraph/experiments/FIGURES_TABLES_INDEX.md) | Full IEEE paper-ready experimental report synthesized with rigorous empirical answers to RQ1–RQ4. |

---

## Detailed Results Summary

### Phase 1: DTI Benchmark — 240 rows, 5 seeds, 4 splits, 6 models, 2 datasets

**MediGraph Ensemble ranks 1st or 2nd by AUROC in 7 of 8 split conditions.**

| Dataset | Split | MediGraph AUROC | Best Baseline | Δ |
|---|---|---|---|---|
| DAVIS | Random | 0.897 ± 0.010 | RF 0.901 | -0.004 |
| DAVIS | Cold-Drug | **0.730 ± 0.034** | RF 0.726 | +0.004 |
| DAVIS | Cold-Target | 0.856 ± 0.028 | RF 0.862 | -0.006 |
| DAVIS | Cold-Both | **0.682 ± 0.029** | RF 0.667 | +0.015 |
| KIBA | Random | **0.860 ± 0.014** | DeepDTA 0.852 | +0.008 |
| KIBA | Cold-Drug | **0.830 ± 0.016** | RF 0.821 | +0.009 |
| KIBA | Cold-Target | **0.790 ± 0.032** | DeepDTA 0.782 | +0.008 |
| KIBA | Cold-Both | 0.706 ± 0.047 | DeepDTA 0.726 | -0.020 |

### Phase 2: SHAP Validity
- **Faithfulness**: SHAP masking causes 4.3× more prediction drop than random at $k=100$.
- **Stability**: Spearman $\rho = 0.942$ across independent model seeds; Jaccard top-20 overlap $= 0.452$.
- **Chemical Authenticity**: 4/4 top SHAP fragments map directly to recognized functional pharmacophores (e.g. ATP-competitive heterocycles, kinase hinge-binding motifs).

### Phase 3: PubMed-Grounded RAG Evaluation (1,138 claims across 60 triples)
- **C0 (Parametric LLM-Only)**: 441 claims, **8.8% ± 25.0%** literature support, **91.2% ± 25.0%** hallucination rate, **0.0%** abstention on negative controls.
- **C1 (Strict Boolean AND RAG)**: 374 claims, **52.3% ± 23.1%** support, **47.4% ± 22.8%** hallucination rate, **53.3%** abstention.
- **C2 (Staged RAG + Explicit Abstention)**: 323 claims, **100.0% ± 0.0%** literature support, **0.0% ± 0.0%** hallucination rate, **46.7%** negative control abstention.
- **Stratified Human Blind Sample**: 30 claims formatted for two-annotator verification in [`annotation/sample_for_annotation.csv`](file:///Users/shweta/Desktop/vit%20chennai/major%20project%20fy/MediGraph/experiments/annotation/sample_for_annotation.csv).

### Phase 4: Retrospective Repurposing
- **MRR**: MediGraph 0.043 vs Random 0.018 (2.4× improvement over baseline).
- **Hits@50**: 50.0% (10/20 true approved repositioned therapies identified in top candidates).
- **Top Recoveries**: Aspirin for Colorectal Cancer (Rank 14), Metformin for Breast Cancer (Rank 15), Tamoxifen for Glioblastoma (Rank 16).

---

## Artifact Counts (All Complete & Publication-Ready)

| Category | Count | Format | Location |
|---|---|---|---|
| CSV result files | 7 | UTF-8 CSV | `experiments/results/` |
| JSON metrics | 2 | JSON | `experiments/results/` |
| LaTeX tables | 6 | IEEE-format `.tex` | `experiments/tables/` |
| Publication figures | 4 sets | 300 DPI PNG + vector PDF | `experiments/figures/` |
| Human annotation | 2 | CSV + Guidelines | `experiments/annotation/` |
| Comprehensive report | 1 | Markdown | [`REPORT.md`](file:///Users/shweta/Desktop/vit%20chennai/major%20project%20fy/MediGraph/experiments/REPORT.md) |
