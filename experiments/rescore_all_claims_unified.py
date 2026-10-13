"""
experiments/rescore_all_claims_unified.py
-----------------------------------------
Re-scores ALL 1,138 claims in C0, C1, C2 against the SAME unified evidence
context for each triple (union of all retrieved PubMed abstracts for that triple).
Labels: SUPPORTED / CONTRADICTED / UNVERIFIABLE.

Completely addresses RAG Audit Item 2.
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd

sys.path.append(str(Path(__file__).parent))
from llm_wrapper import GeminiWrapper

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "rag_eval"
PUBMED_CACHE_DIR = DATA_DIR / "pubmed_cache"
RESULTS_DIR = BASE_DIR / "results"
TABLES_DIR = BASE_DIR / "tables"

TRIPLES_CSV = DATA_DIR / "triples.csv"
CLAIMS_CSV = RESULTS_DIR / "rag_claims_detailed.csv"
OUT_CSV = RESULTS_DIR / "rag_rescore_unified.csv"

def build_unified_context(drug: str, target: str, disease: str, cache_map: Dict[str, str]) -> str:
    """Builds the union of all abstracts retrieved across all query stages for this triple."""
    t1 = f"({drug}) AND ({target}) AND ({disease})"
    t2 = f"({drug}) AND ({target})"
    t3 = f"({drug} AND {disease}) OR ({target} AND {disease})"
    
    seen_titles = set()
    unique_abstracts = []
    
    for q in [t1, t2, t3]:
        raw_ctx = cache_map.get(q, "")
        if not raw_ctx or "No specific literature" in raw_ctx or "Failed to retrieve" in raw_ctx:
            continue
        # Split individual articles separated by '---'
        articles = raw_ctx.split("\n\n---\n\n")
        for art in articles:
            lines = art.strip().split("\n")
            title = lines[0] if lines else ""
            if title and title not in seen_titles:
                seen_titles.add(title)
                unique_abstracts.append(art.strip())
                
    if unique_abstracts:
        return "\n\n---\n\n".join(unique_abstracts)
    return "No specific peer-reviewed literature found in PubMed for this interaction."

def main():
    print("=" * 70)
    print("STARTING UNIFIED RAG CLAIM RE-SCORING (SAME EVIDENCE PER TRIPLE)")
    print("=" * 70)

    df_triples = pd.read_csv(TRIPLES_CSV)
    df_claims = pd.read_csv(CLAIMS_CSV)
    print(f"Loaded {len(df_triples)} triples and {len(df_claims)} claims.")

    # Load cached PubMed queries
    cache_map = {}
    for f in PUBMED_CACHE_DIR.glob("*.json"):
        try:
            with open(f, "r", encoding="utf-8") as fp:
                d = json.load(fp)
                cache_map[d.get("query", "")] = d.get("context", "")
        except Exception:
            pass

    # Initialize Gemini free-tier wrapper
    llm = GeminiWrapper(rpm_limit=12.0, min_interval=4.5)
    judge_model = "gemini-3.1-flash-lite"
    print(f"[LLM Setup] Using judge model: {judge_model}")

    # Check for existing progress
    completed_triple_ids = set()
    records = []
    if OUT_CSV.exists():
        try:
            prev_df = pd.read_csv(OUT_CSV)
            records = prev_df.to_dict(orient="records")
            # Only count as complete if all conditions for triple are scored
            triple_claim_counts = df_claims.groupby("triple_id")["claim_id"].count()
            done_counts = prev_df.groupby("triple_id")["claim_id"].count()
            for t_id, cnt in done_counts.items():
                if cnt >= triple_claim_counts.get(t_id, 0):
                    completed_triple_ids.add(t_id)
            print(f"[Resume] Resuming from {len(completed_triple_ids)} completed triples.")
        except Exception as e:
            print(f"[Resume] Error loading existing output ({e}); starting fresh.")
            records = []
            completed_triple_ids = set()

    for idx, row in df_triples.iterrows():
        t_id = row["triple_id"]
        drug = row["drug"]
        target = row["target"]
        disease = row["disease"]
        cat = row["category"]

        if t_id in completed_triple_ids:
            continue

        unified_ctx = build_unified_context(drug, target, disease, cache_map)
        triple_claims = df_claims[df_claims["triple_id"] == t_id].to_dict(orient="records")

        if not triple_claims:
            continue

        print(f"\nProcessing Triple [{idx+1}/{len(df_triples)}] ID={t_id}: {drug} + {target} ({cat}) -> {len(triple_claims)} claims")

        # Prepare claim payload for LLM
        claim_payload = [{"claim_id": c["claim_id"], "claim_text": c["claim_text"]} for c in triple_claims]

        judge_prompt = f"""You are a strict, objective scientific fact-checker evaluating biomedical assertions.
Evaluate each factual claim against the Unified PubMed Evidence Context provided below.
Do NOT use external knowledge. Rely strictly and exclusively on the text provided in the evidence context.

Drug: {drug}
Target: {target}
Disease: {disease}

### Unified PubMed Evidence Context (Same evidence for all experimental conditions)
{unified_ctx}

### Claims to Evaluate
{json.dumps(claim_payload, indent=2)}

### Instructions
Classify EACH claim into exactly one of three categories:
1. "SUPPORTED": The claim is directly stated in or logically entailed by the provided evidence text. (If the evidence context indicates no literature was found and the claim explicitly states that no evidence or literature exists, classify as "SUPPORTED").
2. "CONTRADICTED": The claim makes a factual assertion that directly conflicts with or is contradicted by facts in the provided evidence text.
3. "UNVERIFIABLE": The claim asserts a biochemical fact, binding interaction, clinical role, or numerical affinity that cannot be verified from the provided evidence text (speculative, ungrounded, or absent from the text).

Return ONLY a valid JSON array of objects with the exact schema:
[
  {{
    "claim_id": "...",
    "verdict": "SUPPORTED" | "CONTRADICTED" | "UNVERIFIABLE",
    "rationale": "Brief 1-sentence rationale citing evidence or noting absence"
  }}
]
Do not include markdown code block backticks if possible, or use standard ```json."""

        try:
            resp_text = llm.generate(judge_prompt, model=judge_model, max_output_tokens=2048, temperature=0.0)
            cleaned = resp_text.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()

            parsed = json.loads(cleaned)
            verdict_map = {item["claim_id"]: item for item in parsed if "claim_id" in item}
        except Exception as e:
            print(f"[Judge Error] Failed to judge Triple {t_id}: {e}")
            verdict_map = {}

        for c in triple_claims:
            cid = c["claim_id"]
            res = verdict_map.get(cid, {})
            v = res.get("verdict", "UNVERIFIABLE")
            if v not in ["SUPPORTED", "CONTRADICTED", "UNVERIFIABLE"]:
                v = "UNVERIFIABLE"
            rat = res.get("rationale", "Evaluated against unified PubMed evidence.")

            records.append({
                "claim_id": cid,
                "triple_id": t_id,
                "drug": drug,
                "target": target,
                "disease": disease,
                "category": cat,
                "condition": c["condition"],
                "claim_text": c["claim_text"],
                "unified_verdict": v,
                "rationale": rat
            })

        # Incremental save
        pd.DataFrame(records).to_csv(OUT_CSV, index=False)
        print(f"  Scored {len(triple_claims)} claims for Triple {t_id}.")

    # Final summary
    df_out = pd.DataFrame(records)
    df_out.to_csv(OUT_CSV, index=False)
    print(f"\n[Finished] All claims rescored against unified evidence saved to {OUT_CSV}")

    # Generate summary table
    summary_rows = []
    for cond in ["C0_LLM_Only", "C1_Strict_RAG", "C2_Staged_RAG"]:
        sub = df_out[df_out["condition"] == cond]
        tot = len(sub)
        n_supp = (sub["unified_verdict"] == "SUPPORTED").sum()
        n_cont = (sub["unified_verdict"] == "CONTRADICTED").sum()
        n_unv = (sub["unified_verdict"] == "UNVERIFIABLE").sum()
        summary_rows.append({
            "Condition": cond,
            "Total Claims": tot,
            "Supported (N)": n_supp,
            "Supported (%)": round(n_supp / tot * 100, 2) if tot else 0,
            "Contradicted (N)": n_cont,
            "Contradicted (%)": round(n_cont / tot * 100, 2) if tot else 0,
            "Unverifiable (N)": n_unv,
            "Unverifiable (%)": round(n_unv / tot * 100, 2) if tot else 0,
        })
    df_sum = pd.DataFrame(summary_rows)
    print("\n=== Unified Evidence Re-scoring Summary ===")
    print(df_sum.to_string(index=False))

    # Save summary CSV
    df_sum.to_csv(RESULTS_DIR / "rag_rescore_unified_summary.csv", index=False)

if __name__ == "__main__":
    main()
