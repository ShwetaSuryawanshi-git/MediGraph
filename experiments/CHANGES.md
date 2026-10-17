# CHANGES.md: Modifications Versus Previous Bundle

## Audit Fixes and Redesigns
- `dti_benchmark.csv`: Added `n_pos_test` and `n_neg_test` columns across all 240 benchmark rows without changing existing metric values, exposing small positive counts in held-out splits (e.g., DAVIS cold_both: 12-40 positives per seed).
- `phase4_repurposing_benchmark.py`: Deleted the Target Prior baseline oracle across code, CSVs, JSON, LaTeX tables, figures, and documentation to eliminate label leakage.
- `repurposing_benchmark.csv`: Rebuilt decoy set with length- and family-matched decoys (100 closest human non-kinases for non-kinase targets, 100 closest kinases for PRKAA1) to eliminate kinase confounding.
- `repurposing_baselines.csv`: Added second non-oracle baseline ranking targets by sequence length deviation from median human protein length (424 aa) with zero access to true target labels.
- `repurposing_permutations.csv`: Generated 1,000-permutation distribution from empirical null simulation and computed exact unbiased permutation p-value with +1 in numerator and denominator.
- `repurposing_metrics.json`: Evaluated mean rank, median rank, MRR, and Hits@1/5/10 with 95% bootstrap confidence intervals over the 10 drugs, labeled as exploratory descriptive screening.
- `repurposing_benchmark.csv`: Verified all 10 cited PMIDs against Entrez efetch, flagging title and publication year mismatches in column `pmid_verified`.
- `phase2_shap_benchmark.py`: Implemented dual feature masking (zero masking and training-set mean masking) to address inactive binary ECFP bits.
- `shap_faithfulness.csv`: Ranked features by mean |SHAP| on training-set subsample (N=300) rather than held-out test rows to maintain evaluation independence.
- `shap_faithfulness.csv`: Averaged random baseline over 20 independent feature sets per seed and evaluated on both cold_drug and random splits across all 5 seeds.
- `shap_faithfulness.csv`: Added per-seed paired comparisons reporting the number of seeds where SHAP degradation exceeds the 20-rep averaged random baseline.
- `figures/shap_faithfulness_curves.png`: Re-rendered curves with y-axis anchored at 0.0, error bars (SD across seeds), and 20-repetition averaged random baseline.
- `rag_evaluation.csv`: Added `n_retrieved_pmids`, `n_cited_pmids`, set `citation_validity` to NaN when nothing was retrieved or cited, and added `cited_without_retrieval`.
- `results/pmid_check.csv`: Checked cited PMIDs for triples 44, 49, 53, and 57 against Entrez efetch, verifying whether retrieved abstracts discuss the queried drugs.
- `results/judge_agreement.csv`: Executed automated independent second-judge agreement check on 30 randomly sampled claims per condition (seed 0, 120 total claims) using distinct prompt and model, reporting percent agreement and Cohen's kappa.
- `rag_retrieval_tier_stats.csv`: Added explanatory audit note clarifying that C2 0% empty retrieval reflects Tier 3 broad secondary disease fallback on negative controls, followed by model abstention.
- `figures/shap_fragments/`: Re-rendered Morgan bit PNGs with RDKit `DrawMorganBit` annotated with bit ID, Morgan radius, and parent drug SMILES.
- `results/fragments_check.csv`: Recorded central atom, radius, substructure environment SMILES, and training-set molecule prevalence for each highlighted Morgan bit.
- `requirements.txt`: Pinned exact package versions actively imported in the environment.
- `annotation/claims_annotation_sample_blinded.csv`: Updated with 160 human-expert-reviewed verdicts and notes (102 Supported, 39 Partially supported, 17 Contradicted, 2 Insufficient evidence) assessing pharmacological validity against established literature.

## Results that got worse or stayed null
- **Target Prior Oracle Deleted**: Removed the artificial rank 1.0 (100% Hits@1) baseline caused by hardcoded label leakage (`0.5 if i == 0 else 0.4`).
- **Repurposing Retrieval Performance**: Under unconfounded family- and length-matched decoys, MediGraph achieves mean rank 36.6 (95% CI: [20.8, 53.3]), median rank 29.5 (95% CI: [10.0, 64.0]), and MRR 0.056 (95% CI: [0.030, 0.085]) out of 101 candidates. Hits@1 is 0.0%, Hits@5 is 0.0%, and Hits@10 is 30.0%. Permutation p-value is 0.053 (0.0529), not 0.000999. The sequence-length heuristic baseline achieves mean rank 85.3 (median 85.5, MRR 0.012), while random permutation expectation is 50.8 (median 50.5).
- **Repurposing PMID Verification**: 8 out of 10 cited PMIDs failed Entrez efetch verification due to title and publication year mismatches (only Raloxifene PMID 10376571 and Metformin PMID 11602624 matched the cited publication).
- **SHAP Faithfulness on Cold-Drug Split**: Evaluated with dual masking and 20-repetition averaged random baseline. On the cold-drug split, SHAP shows no measurable faithfulness advantage over the 20-rep averaged random baseline: mean SHAP-minus-Random AUPR under zero masking is +0.0026 at k=5, +0.0061 at k=10, -0.0084 at k=20, and -0.0163 at k=50 (seed SD 0.05-0.10). SHAP prediction drop exceeds random drop in only 2-3 of 5 seeds at k=5, 10, 20 and 2 of 5 seeds at k=50. MDI degradation exceeds SHAP degradation across multiple seeds. Under training-mean masking, SHAP-minus-Random AUPR is +0.0029 at k=5, +0.0077 at k=10, +0.0017 at k=20, and +0.0069 at k=50.
- **C1 Hallucinated Citations with Zero Retrieval**: For triples 44, 49, 53, and 57 in C1, strict boolean queries returned zero abstracts, but the model generated citations to PMIDs from pre-training memory; Entrez verification confirms none of the 6 cited papers discuss the queried drugs (`mentions_drug: False`). Citation validity is set to NaN for these cases instead of 0.0.
- **RAG C2 Claim Support vs. Negative Control Abstention**: While claim-level support on non-control triples is 98.3% (235/239), the 52 claims on negative controls represent statements confirming lack of literature evidence, where the model correctly abstained (15/15 = 100.0%, 95% Wilson CI: [79.6%, 100.0%]). High support rate reflects factual fidelity to the negative context, not biological interaction.
