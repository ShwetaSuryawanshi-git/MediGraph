# MANIFEST.md: MediGraph External Audit Bundle v2

**Date of Generation**: 2026-10-18 16:06:23  
**Bundle Archive**: `MediGraph_Audit_Bundle_v2.zip`  
**Python Version**: Python 3.12.0  
**Environment Requirements**: See `requirements.txt`  

---

## PART A. Bundle File Index

All files listed below are placed directly at the top level (flat archive root) of `MediGraph_Audit_Bundle_v2.zip`.

| File | Lines / Rows | Category / Phase | Production Timestamp |
|---|---|---|---|
| `dti_benchmark.csv` | 241 | Phase 1 (DTI) | 2026-10-17 23:55:48 |
| `dataset_stats.csv` | 9 | Phase 1 (DTI) | 2026-10-10 01:23:54 |
| `dti_protein_feature_ablation.csv` | 61 | Phase 1 (DTI) | 2026-10-03 20:20:08 |
| `dti_standardization_comparison.csv` | 81 | Phase 1 (DTI) | 2026-10-17 15:04:33 |
| `shap_faithfulness.csv` | 301 | Phase 2 (SHAP) | 2026-10-17 16:04:54 |
| `shap_stability.csv` | 11 | Phase 2 (SHAP) | 2026-10-17 16:07:30 |
| `shap_substructure_notes.md` | 12 | Phase 2 (SHAP) | 2026-10-17 14:56:13 |
| `fragments_check.csv` | 5 | Phase 2 (SHAP) | 2026-10-17 15:58:51 |
| `repurposing_benchmark.csv` | 11 | Phase 4 (Repurposing) | 2026-10-17 15:54:42 |
| `repurposing_metrics.json` | 116 | Phase 4 (Repurposing) | 2026-10-17 15:54:42 |
| `repurposing_baselines.csv` | 5 | Phase 4 (Repurposing) | 2026-10-17 15:54:42 |
| `repurposing_permutations.csv` | 1001 | Phase 4 (Repurposing) | 2026-10-17 15:54:42 |
| `triples.csv` | 61 | Phase 3 (RAG) | 2026-10-11 14:44:04 |
| `rag_evaluation.csv` | 241 | Phase 3 (RAG) | 2026-10-17 16:00:35 |
| `rag_claims_detailed.csv` | 1335 | Phase 3 (RAG) | 2026-10-11 15:57:21 |
| `rag_evaluation.json` | 3242 | Phase 3 (RAG) | 2026-10-09 23:08:34 |
| `rag_rescore_unified.csv` | 1139 | Phase 3 (RAG) | 2026-10-10 00:33:27 |
| `rag_rescore_unified_summary.csv` | 4 | Phase 3 (RAG) | 2026-10-10 00:33:27 |
| `rag_retrieval_tier_stats.csv` | 6 | Phase 3 (RAG) | 2026-10-17 16:08:58 |
| `pmid_check.csv` | 7 | Phase 3 (RAG) | 2026-10-17 16:00:41 |
| `judge_agreement.csv` | 121 | Phase 3 (RAG) | 2026-10-17 16:08:58 |
| `claims_annotation_sample_blinded.csv` | 161 | Phase 3 (RAG) | 2026-10-17 16:36:53 |
| `claims_annotation_sample_key.csv` | 161 | Phase 3 (RAG) | 2026-10-11 16:15:20 |
| `rag_prompts.md` | 123 | Phase 3 (RAG) | 2026-10-17 14:46:20 |
| `reproduce_dti_lr.py` | 305 | Code | 2026-10-18 01:02:34 |
| `phase1_dti_benchmark.py` | 866 | Code | 2026-10-02 23:43:19 |
| `run_dti_audit_fixes.py` | 475 | Code | 2026-10-17 14:12:10 |
| `phase2_shap_benchmark.py` | 494 | Code | 2026-10-17 15:56:00 |
| `phase3_rag_benchmark.py` | 611 | Code | 2026-10-11 14:47:31 |
| `run_rag_audit_fixes.py` | 359 | Code | 2026-10-17 15:59:55 |
| `phase4_repurposing_benchmark.py` | 603 | Code | 2026-10-17 15:54:09 |
| `generate_fragment_figures.py` | 170 | Code | 2026-10-17 15:58:24 |
| `llm_wrapper.py` | 306 | Code | 2026-10-09 21:59:21 |
| `requirements.txt` | 16 | Environment | 2026-10-17 16:02:09 |
| `dti_auroc_aupr.png` | N/A (Binary/Image) | Figures | 2026-10-17 15:04:35 |
| `shap_faithfulness_curves.png` | N/A (Binary/Image) | Figures | 2026-10-17 16:07:32 |
| `repurposing_rank_percentile.png` | N/A (Binary/Image) | Figures | 2026-10-17 15:54:43 |
| `repurposing_mrr_hits.png` | N/A (Binary/Image) | Figures | 2026-10-17 15:54:44 |
| `rag_hallucination_rate.png` | N/A (Binary/Image) | Figures | 2026-10-11 15:57:24 |
| `dasatinib_bit_798.png` | N/A (Binary/Image) | Figures | 2026-10-17 15:58:51 |
| `dasatinib_bit_798_radius_2.png` | N/A (Binary/Image) | Figures | 2026-10-17 15:58:51 |
| `erlotinib_bit_974.png` | N/A (Binary/Image) | Figures | 2026-10-17 15:58:51 |
| `erlotinib_bit_974_radius_1.png` | N/A (Binary/Image) | Figures | 2026-10-17 15:58:51 |
| `gefitinib_bit_841.png` | N/A (Binary/Image) | Figures | 2026-10-17 15:58:51 |
| `gefitinib_bit_841_radius_1.png` | N/A (Binary/Image) | Figures | 2026-10-17 15:58:51 |
| `imatinib_bit_207.png` | N/A (Binary/Image) | Figures | 2026-10-17 15:58:51 |
| `imatinib_bit_207_radius_1.png` | N/A (Binary/Image) | Figures | 2026-10-17 15:58:51 |
| `shap_faithfulness.tex` | 16 | Tables | 2026-10-17 16:07:30 |
| `shap_stability.tex` | 16 | Tables | 2026-10-17 16:07:30 |
| `repurposing_recovery.tex` | 17 | Tables | 2026-10-17 15:54:42 |
| `CHANGES.md` | 33 | Documentation | 2026-10-18 01:03:47 |
| `REPORT.md` | 181 | Documentation | 2026-10-17 23:57:50 |
| `AUDIT.md` | 153 | Documentation | 2026-10-10 00:32:32 |
| `FIGURES_TABLES_INDEX.md` | 43 | Documentation | 2026-10-18 16:05:06 |
| `PROGRESS.md` | 67 | Documentation | 2026-10-18 16:05:33 |

---

## PART B. Not Done

1. **KIBA SHAP Progressive Masking**: Progressive feature masking was conducted across 5 seeds on TDC DAVIS for both `cold_drug` and `random` splits under dual masking variants ('zero' and 'train_mean') with 20-repetition averaged random baselines. Masking on KIBA was omitted due to the 118,254 pair size and memory/time constraints under the free tier budget.
2. **Entrez PMID Title/Year Mismatches in Repurposing**: In `repurposing_benchmark.csv`, 8 of the 10 literature citations cited in earlier retrospective repurposing benchmarks (Thalidomide, Sildenafil, Minoxidil, Aspirin, Methotrexate, Digoxin, Propranolol, Tamoxifen) failed automated Entrez efetch verification against the exact title and publication year of the cited paper. These mismatches are reported directly in the `pmid_verified` column (`False`) without silent alteration.

---

## PART C. Self-Check Verifications

### 1. DTI Benchmark AUROC across All Models, Datasets, and Splits (`dti_benchmark.csv`)
Evaluated across 5 random seeds (0..4) with `StandardScaler` fitted strictly on training folds for linear models and validation-tuned decision thresholds. Small test sets are visible via explicit `n_pos_test` and `n_neg_test` columns.

#### (a) TDC DAVIS Dataset (Mean $\pm$ SD AUROC across 5 Seeds)
| Model | cold_both | cold_drug | cold_target | random |
|---|---|---|---|---|
| **Calibrated LinearSVC** | 0.5617 | 0.6694 | 0.5321 | 0.8373 |
| **Logistic Regression** | 0.5646 | 0.6712 | 0.7202 | 0.8352 |
| **Two-tower MLP** | 0.6262 | 0.6977 | 0.8294 | 0.8875 |
| **XGBoost** | 0.6613 | 0.7183 | 0.8409 | 0.8823 |
| **Random Forest** | 0.6669 | 0.7256 | 0.8624 | 0.9005 |
| **MediGraph Ensemble** | 0.6797 | 0.7288 | 0.8558 | 0.8965 |

#### (b) TDC KIBA Dataset (Mean $\pm$ SD AUROC across 5 Seeds)
| Model | cold_both | cold_drug | cold_target | random |
|---|---|---|---|---|
| **Calibrated LinearSVC** | 0.6325 | 0.7014 | 0.7316 | 0.8141 |
| **Logistic Regression** | 0.6390 | 0.7412 | 0.7271 | 0.8153 |
| **Two-tower MLP** | 0.7263 | 0.7968 | 0.7818 | 0.8517 |
| **XGBoost** | 0.7307 | 0.8120 | 0.7495 | 0.8322 |
| **Random Forest** | 0.7218 | 0.8362 | 0.7671 | 0.8423 |
| **MediGraph Ensemble** | 0.7385 | 0.8416 | 0.7829 | 0.8611 |

---

### 2. SHAP Faithfulness: SHAP-minus-Random AUPR per $k$ (`shap_faithfulness.csv`)
Global features ranked strictly by mean $|SHAP|$ on training-set subsamples ($N=300$). Random baseline averaged across 20 independent repetitions per seed.

- **Status**: **No measurable faithfulness advantage** on the cold-drug split.

| Split | Masking Variant | $k=5$ | $k=10$ | $k=20$ | $k=50$ |
|---|---|---|---|---|---|
| **cold_drug** | zero | +0.0026 $\pm$ 0.0049 | +0.0061 $\pm$ 0.0065 | -0.0084 $\pm$ 0.0509 | -0.0163 $\pm$ 0.0860 |
| **cold_drug** | train_mean | +0.0029 $\pm$ 0.0144 | +0.0077 $\pm$ 0.0199 | +0.0017 $\pm$ 0.0444 | +0.0069 $\pm$ 0.0761 |
| **random** | zero | -0.0009 $\pm$ 0.0485 | -0.0098 $\pm$ 0.0555 | -0.0649 $\pm$ 0.0631 | -0.1787 $\pm$ 0.0804 |
| **random** | train_mean | -0.0027 $\pm$ 0.0483 | -0.0145 $\pm$ 0.0477 | -0.0625 $\pm$ 0.0772 | -0.1925 $\pm$ 0.0880 |

---

### 3. Retrospective Repurposing Target Prioritization (`repurposing_metrics.json`)
Evaluated across 10 gold-standard cases against 100 sequence-length and family-matched decoys ($N=101$ total candidates per drug). Labeled strictly as exploratory descriptive screening. Target Prior baseline deleted.

| Method | Mean Rank [95% CI] | Median Rank [95% CI] | MRR [95% CI] | Hits@1 (%) | Hits@5 (%) | Hits@10 (%) | Permutation $p$-value |
|---|---|---|---|---|---|---|---|
| **MediGraph Engine** | 36.6 [20.8, 53.3] | 29.5 [10.0, 64.0] | 0.056 [0.030, 0.085] | 0.0% | 0.0% | 30.0% | 0.0529 |
| **Sequence Length Heuristic** | 85.3 [80.7, 89.9] | 85.5 [82.0, 88.5] | 0.012 [0.011, 0.012] | 0.0% | 0.0% | 0.0% | 1.0000 |
| **Random Baseline (Empirical Draw)** | 58.0 [45.6, 70.2] | 60.0 [39.0, 77.0] | 0.020 [0.015, 0.026] | 0.0% | 0.0% | 0.0% | 0.5000 |
| **Random (1,000 Permutations)** | 50.9 [34.0, 67.3] | 51.0 | 0.051 | 1.0% | 4.9% | 9.6% | 1.0000 |

- **Permutation Array**: Stored in `repurposing_permutations.csv` ($N=1000$).
- **Permutation $p$-value formula**: `(sum(perm_means <= medigraph_mean) + 1) / (B + 1)` = `0.0529`.
- **Entrez PMID Verification**: 2 verified (Raloxifene PMID 10376571, Metformin PMID 11602624), 8 mismatched.

---

### 4. RAG Audit Numbers and Judge Agreement (`judge_agreement.csv`)
- **Citation Validity among Answers with Retrieval**: 100.0% for both C1 and C2.
- **Cited without Retrieval Rate**: 6.7% for C1 (4 triples: 44, 49, 53, 57); 0.0% for C0, C0b, and C2.
- **PMID Verification of C1 Unretrieved Citations (`pmid_check.csv`)**: 6 PMIDs checked via Entrez efetch; none discuss the queried drugs (`mentions_drug: False` for all 6).
- **Automated Independent Second-Judge Agreement Check (NOT human annotation)**: Evaluated on 30 randomly sampled claims per condition (seed 0, 120 total claims) using `gemini-3.5-flash-lite`:
  - Overall Percent Agreement: **85.8%**
  - Overall Cohen's Kappa: **0.717**
  - C0_LLM_Only: 93.3% agreement (Kappa: 0.474)
  - C0b_Neutral_Prompt: 73.3% agreement (Kappa: 0.375)
  - C1_Strict_RAG: 80.0% agreement (Kappa: 0.600)
  - C2_Staged_RAG: 96.7% agreement (Kappa: 0.000)
- **Claim-Level Support vs. Negative Control Abstention**:
  - Non-control triples (45 triples): 235 / 239 claims supported (**98.3%**)
  - Negative control triples (15 triples): 15 / 15 abstentions (**100.0%**, 95% Wilson CI: [79.6%, 100.0%])
- **Human Expert Annotation Sample (`claims_annotation_sample_blinded.csv`)**: 160 human-reviewed claims: 102 Supported, 39 Partially supported, 17 Contradicted, 2 Insufficient evidence. Detailed pharmacological notes provided for all 160 claims.
