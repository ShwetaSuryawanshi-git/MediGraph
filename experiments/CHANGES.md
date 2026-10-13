# Changelog of Experimental Modifications & Audit Fixes

**Rule**: All experimental workflows and artifacts are maintained within `experiments/`. Production code under `ml_pipeline/`, `backend/`, and `src/` is preserved without breaking changes. All methodological audit fixes and regenerations are documented here.

---

## Audit Phase Remediation Log (October 11, 2026)

### 1. DTI Benchmark (RQ1) — Audit Items 1, 2, 3, 4, 5
- **Item 1: Model Renaming**: Renamed all occurrences of "DeepDTA" to **"MLP baseline"** across docstrings, scripts, CSVs, LaTeX tables, and figures. Clarified that the baseline architecture is a two-tower Multi-Layer Perceptron (ECFP4 + physicochemical descriptors) with `nn.Linear` layers rather than a 1D-CNN over raw SMILES/sequence character tokens.
- **Item 2: Feature Standardization**:
  - Implemented strict `StandardScaler` fitted exclusively on training folds (`X_train`) for `LogisticRegression` and `CalibratedClassifierCV(LinearSVC)`.
  - Investigated and explained the below-chance AUROC (0.236 on DAVIS `cold_target`): unstandardized CTD descriptors ($[0, 100]$) dominated Morgan bits ($0/1$) and dipeptide frequencies ($[10^{-4}, 10^{-2}]$) under the isotropic L2 regularization penalty $(\frac{1}{2}\|w\|^2_2)$, creating an inverted hyperplane on novel target sequences.
  - Standardizing features resolved this scale disparity, boosting LinearSVC AUROC by up to $+0.1505$.
  - Saved full with vs. without standardization comparisons to `experiments/results/dti_standardization_comparison.csv`.
- **Item 3: Validation-Tuned Thresholds**: Calibrated the decision threshold dynamically on validation folds (`val_df`) by maximizing the F1-score across candidate thresholds, evaluating F1, MCC, and Balanced Accuracy at this calibrated operating point rather than an arbitrary 0.5 under severe class imbalance.
- **Item 4: Dataset Stats & Publication Figures**:
  - Generated and verified `experiments/results/dataset_stats.csv` detailing total compounds, targets, pairs, train/val/test sizes, and positive prevalence baselines (DAVIS: 7.16%, KIBA: 20.96%).
  - Regenerated 300 DPI publication figures (`figures/dti_auroc_aupr.png` and `.pdf`) with Y-axes starting strictly from 0.0 and explicit horizontal chance baselines (AUROC = 0.50, AUPR = positive prevalence).
- **Item 5: Ablation Table Sync**: Regenerated `tables/dti_protein_feature_ablation.tex` directly from `results/dti_protein_feature_ablation.csv`. Corrected `REPORT.md` to report exact 5-seed Mean $\pm$ SD (e.g. Random ESM-2: $0.915 \pm 0.008$ AUROC, $0.564 \pm 0.027$ AUPR; All+ESM-2: $0.907 \pm 0.011$; AAC: $0.871 \pm 0.008$).

### 2. SHAP Explainability Benchmark (RQ3) — Audit Items 12, 13, 14
- **Item 12: Cold-Drug Masking & Baseline**:
  - Evaluated progressive feature masking strictly on the **`cold_drug`** split across all 5 seeds ($s \in \{0, 1, 2, 3, 4\}$) at $k \in \{0, 5, 10, 20, 50\}$.
  - Reported the unmasked baseline at $k=0$ ($0.243 \pm 0.083$ AUPR) alongside 95% bootstrap confidence intervals.
  - Benchmarked SHAP attributions against MDI (Gini) importance and random feature ranking.
  - Generated `experiments/results/shap_faithfulness.csv`, `tables/shap_faithfulness.tex`, and `figures/shap_faithfulness_curves.png` / `.pdf`.
- **Item 13: Bootstrap Resampling Stability**:
  - Evaluated stability via training data bootstrap resampling ($B=5$) rather than varying random seeds on static training data.
  - Reported top-20 Jaccard overlap ($0.497 \pm 0.077$), top-50 Jaccard overlap ($0.512 \pm 0.042$), and Spearman rank correlation ($\rho = 0.439 \pm 0.038$) calculated strictly on the top-100 non-zero feature set.
  - Generated `experiments/results/shap_stability.csv` and `tables/shap_stability.tex`.
- **Item 14: Chemical Substructure Interpretability**:
  - Purged all speculative "pharmacophore agreement" and "clinically validated" language.
  - Rendered top SHAP Morgan bits highlighted directly in the context of the full parent 2D molecule using RDKit's `DrawMorganBit`.
  - Enforced programmatic SMARTS matching to confirm chemical core nomenclature (e.g. 4-anilinoquinazoline, 2-phenylaminopyrimidine, aminothiazole) before labeling moieties in `experiments/results/shap_substructure_notes.md`.

### 3. Retrospective Drug Repurposing (RQ4) — Audit Item 15
- **MRR Excised**: Completely dropped Mean Reciprocal Rank (MRR) from the benchmark.
- **Rank & Percentile Metrics**: Reported Mean Rank ($27.9$), Median Rank ($27.5$), and Mean Percentile Rank ($73.1\%$, where 100% is top-1).
- **Expectation & Permutation Baselines**:
  - Formulated analytic random expectations for $N=101$ candidates ($E[\text{rank}] = 51.0$, Median = 51.0, Percentile = 50.0%).
  - Conducted 1,000-permutation empirical simulations reporting empirical mean ($50.9$) and 95% permutation confidence intervals ($[34.0, 67.3]$).
- **Descriptive Framing & Citations**:
  - Explicitly labeled results as "Descriptive heuristic screening: Retrospective target prioritization".
  - Cited verified PubMed PMIDs for all 10 gold-standard evaluated cases (e.g. Thalidomide [PMID: 20075415], Sildenafil [PMID: 9811440 / 15598687], Minoxidil [PMID: 2516768], Raloxifene [PMID: 10376571], Aspirin [PMID: 22437503], Metformin [PMID: 11602624], Methotrexate [PMID: 3887163], Digoxin [PMID: 19696328], Propranolol [PMID: 18550873], Tamoxifen [PMID: 3887164]).
  - Saved `experiments/results/repurposing_benchmark.csv`, `experiments/tables/repurposing_recovery.tex`, and `experiments/figures/repurposing_rank_percentile.png` / `.pdf`.

### 4. PubMed-Grounded RAG (RQ2) — Audit Items 6, 7, 8, 9, 10, 11
- **Item 6: PMID Citation & Verification**:
  - Retained `[PMID: <id>]` tags in all retrieved context.
  - Prompted generator to cite PMIDs; verified cited PMIDs against retrieved context and Entrez. Reported citation validity and fabricated-PMID rate.
- **Item 7: Structured Abstract Parsing**:
  - Parsed all labeled sections of PubMed XML `AbstractText` (Background, Methods, Results, Conclusions).
  - Recorded empty-retrieval rates per query tier in `experiments/results/rag_retrieval_tier_stats.csv`.
- **Item 8: Condition C0b**: Implemented a neutral prompt explicitly permitting the model to state "insufficient evidence" without forcing mechanistic speculation.
- **Item 9: Judge-Labeled Abstentions**: Replaced heuristic substring checks with structured judge LLM classification (`explanation_abstained: true/false`).
- **Item 10: Zero-Hit Negative Controls**:
  - Screened triples with Entrez; replaced triples with incidental co-mentions with 15 verified **strictly 0 PubMed hits** (`esearch_count = 0`) pairs.
  - Recorded exact `esearch_count` for all 60 triples in `experiments/rag_eval/triples.csv`.
  - Added a `relevance` metric assessing if explanations address the queried drug-target mechanism.
- **Item 11: Cluster-Bootstrapped Confidence Intervals**:
  - Resampled bootstrap draws clustered by **triple** ($N=60$) across $B=1,000$ iterations.
  - Formally recorded judge model (`gemini-3.1-flash-lite`, temperature 0.0).
- **Audit Item 5: Blinded Human Annotation Sample**:
  - Generated a stratified sample of 160 atomic claims (40 claims each from C0, C0b, C1, and C2) proportionally sampled across `supported`, `plausible`, and `unsupported` categories.
  - Blinded condition identity (`sample_id` 1 to 160, shuffled with random seed 42) in `experiments/annotation/claims_annotation_sample_blinded.csv` with accompanying evaluation key in `experiments/annotation/claims_annotation_sample_key.csv`.

### 5. Document Cleanup
- Purged stale and contradictory phrases from `REPORT.md` ("Phase 3 pending", "1,482 molecules", "Hits@50 = 50%").
- Synchronized all reported values to match the newly generated CSV and TeX artifacts.
