# Human Annotation Guidelines: Evaluating Factuality & Hallucination in Drug-Target-Disease Explanations

**Project**: MediGraph Empirical Study for IEEE Conference / NCSET  
**Task**: Blind Claim-Level Factual Verification  

---

## 1. Objective
You are tasked with independently and blindly evaluating the factual truthfulness of machine-generated explanations describing drug-target-disease biological mechanisms. Your evaluation will be compared against automated LLM judge evaluations and another human annotator to establish Cohen's Kappa inter-annotator agreement.

---

## 2. Evaluation Unit: Atomic Claims
Each generated explanation has been decomposed into individual, atomic factual claims. An atomic claim is a single declarative statement asserting a biological fact (e.g., *"Imatinib binds to the ATP-binding site of BCR-ABL."* or *"EGFR activation promotes cell proliferation."*).

You will be given:
1. **Drug Name**: The compound under evaluation.
2. **Target Name**: The protein/gene target.
3. **Disease Name**: The pathological condition.
4. **Retrieved Literature Context**: The scientific abstracts retrieved from PubMed.
5. **Atomic Claim**: The specific statement to evaluate.

---

## 3. Label Categories

For each atomic claim, assign exactly one of the following three labels:

| Label | Definition | Operational Criterion |
|---|---|---|
| **SUPPORTED (S)** | The claim is directly stated, paraphrased, or strictly entailed by the provided PubMed literature context. | A reader can point to specific sentences in the provided abstract that confirm the claim. |
| **UNSUPPORTED / HALLUCINATED (U)** | The claim introduces factual assertions (e.g., binding affinities, specific residues, clinical outcomes) that are NOT present in or verifiable from the provided context. | The statement may sound biologically plausible to a layperson, but the provided text does not contain evidence for it. |
| **CONTRADICTED (C)** | The claim directly asserts facts that contradict the provided PubMed literature context. | The text states the drug does *not* bind or has no effect, but the claim asserts it does. |

---

## 4. Special Guidelines & Edge Cases

1. **Strict Context Adherence (No External Knowledge)**:
   - Base your decision **strictly** on whether the provided PubMed context supports the claim.
   - Do NOT use outside general knowledge or external web searches to "forgive" an ungrounded claim. If the claim is factually true in real-world pharmacology but *absent* from the retrieved text, label it **UNSUPPORTED (U)**. Grounding requires verifiable literature provenance in the retrieved context.

2. **Entity Synonyms**:
   - Standard gene symbol variants (e.g., `ABL1` vs `BCR-ABL`, `HER2` vs `ERBB2`, `VEGFR2` vs `KDR`) are considered equivalent if the text establishes the context.

3. **Abstention Protocols**:
   - If the system explicitly outputs an abstention token (e.g., `[NO_LITERATURE_EVIDENCE]`), and no positive binding interaction is asserted, this claim is considered **SUPPORTED (S)** as a correct abstention.

---

## 5. Annotation File Format
Annotations must be recorded in CSV format with columns:
`claim_id, condition, triple_id, claim_text, label, annotator_notes`
Where `label` is one of `SUPPORTED`, `UNSUPPORTED`, or `CONTRADICTED`.
