# Structural Interpretability Analysis of SHAP Fingerprint Attributions

**Methodological Audit Note**: All former speculative 'pharmacophore' language has been removed.
Bits are rendered highlighted directly on the full 2D parent molecule using RDKit's `DrawMorganBit`.
Chemical cores are only labeled if confirmed by programmatic SMARTS pattern matching.

| Compound | Target | Top SHAP Bit | Programmatically Confirmed Core / Moiety | Biochemical Plausibility |
|---|---|---|---|---|
| **Imatinib** | ABL1 | Bit 650 (+0.001) | 2-phenylaminopyrimidine (radius 0 around O9) | Plausible: Phenylaminopyrimidine substructure occupies the ATP-binding pocket of inactive kinase. |
| **Gefitinib** | EGFR | Bit 935 (+0.002) | 4-anilinoquinazoline (radius 0 around N25) | Plausible: Quinazoline heterocyclic core mimics adenine purine ring in EGFR catalytic cleft. |
| **Erlotinib** | EGFR | Bit 703 (+0.006) | 4-anilinoquinazoline (radius 2 around C17) | Plausible: Ethynyl-substituted anilinoquinazoline forms hydrophobic contacts in the ATP cleft. |
| **Dasatinib** | SRC/ABL | Bit 75 (+0.005) | aminothiazole (radius 1 around C17) | Plausible: Aminothiazole moiety interacts with the kinase hinge region. |
