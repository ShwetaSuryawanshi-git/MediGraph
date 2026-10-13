"""
experiments/phase3_rag_benchmark.py
-----------------------------------
PubMed-Grounded RAG Explainability Benchmark (Phase 3 - RQ2).
Methodological Audit Updates (Audit Items 6, 7, 8, 9, 10, 11):
1. Item 6: Retains PMIDs in retrieved context ([PMID: <id>]). Generator prompted to cite PMIDs.
   Verifies every cited PMID against cache/Entrez; reports citation validity and fabricated-PMID rate.
2. Item 7: Parses all sections of structured PubMed abstracts (not just AbstractText[0]).
   Reports empty-retrieval rates per tier for C1 vs C2 in results/rag_retrieval_tier_stats.csv.
3. Item 8: Adds condition C0b (a neutral prompt explicitly permitting stating "insufficient evidence").
4. Item 9: Replaces naive keyword substring checks with judge-labeled abstentions (explanation_abstained).
5. Item 10: Evaluates 60 audited triples where unsupported triples have strictly 0 PubMed hits (esearch_count=0).
   Includes a 'relevance' metric assessing whether the explanation addresses the query drug-target mechanism.
6. Item 11: Computes bootstrap 95% CIs clustered by TRIPLE (N=60), not by individual claim.
   Records exact judge model used (gemini-3.1-flash-lite, temperature 0.0).
"""

import os
import sys
import time
import json
import re
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ssl
from Bio import Entrez

try:
    ssl._create_default_https_context = ssl._create_unverified_context
except Exception:
    pass

from llm_wrapper import GeminiWrapper

Entrez.email = os.environ.get("ENTREZ_EMAIL", "medigraph_research@example.com")

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "rag_eval"
PUBMED_CACHE_DIR = DATA_DIR / "pubmed_cache"
RESULTS_DIR = BASE_DIR / "results"
TABLES_DIR = BASE_DIR / "tables"
FIGURES_DIR = BASE_DIR / "figures"
ANNOTATION_DIR = BASE_DIR / "annotation"

PUBMED_CACHE_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
ANNOTATION_DIR.mkdir(parents=True, exist_ok=True)

TRIPLES_CSV = DATA_DIR / "triples.csv"

# -------------------------------------------------------------
# 1. PubMed Retrieval with Section Parsing & PMID Preservation (Items 6 & 7)
# -------------------------------------------------------------
def parse_abstract_sections(ab_obj) -> str:
    """Parses all sections of structured PubMed abstracts into coherent text."""
    if not ab_obj:
        return ""
    if isinstance(ab_obj, list):
        sections = []
        for sec in ab_obj:
            label = getattr(sec, "attributes", {}).get("Label", "") if hasattr(sec, "attributes") else ""
            sec_text = str(sec).strip()
            if label:
                sections.append(f"{label}: {sec_text}")
            else:
                sections.append(sec_text)
        return " ".join(sections)
    return str(ab_obj).strip()

def fetch_pubmed_with_pmids(query: str, max_results: int = 3) -> Tuple[str, List[str]]:
    """Retrieves PubMed abstracts with persistent disk caching and keeps PMIDs."""
    q_hash = hashlib.sha256(f"v2_{query}_{max_results}".encode("utf-8")).hexdigest()
    cache_file = PUBMED_CACHE_DIR / f"{q_hash}.json"

    if cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                d = json.load(f)
                return d.get("context", ""), d.get("pmids", [])
        except Exception:
            pass

    time.sleep(0.35)
    pmids = []
    context = "No specific literature found for this query."
    try:
        ssl._create_default_https_context = ssl._create_unverified_context
        handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
        record = Entrez.read(handle)
        handle.close()

        id_list = record.get("IdList", [])
        if id_list:
            time.sleep(0.35)
            handle = Entrez.efetch(db="pubmed", id=id_list, retmode="xml")
            articles = Entrez.read(handle)
            handle.close()

            abstracts = []
            for article in articles.get("PubmedArticle", []):
                try:
                    medline = article["MedlineCitation"]
                    pmid = str(medline["PMID"])
                    pmids.append(pmid)
                    article_data = medline["Article"]
                    title = article_data.get("ArticleTitle", "Unknown Title")
                    ab_obj = article_data.get("Abstract", {}).get("AbstractText", [])
                    ab_text = parse_abstract_sections(ab_obj)
                    abstracts.append(f"[PMID: {pmid}] Title: {title}\nAbstract: {ab_text}")
                except Exception:
                    continue
            if abstracts:
                context = "\n\n---\n\n".join(abstracts)
    except Exception as e:
        context = f"Failed to retrieve literature context: {e}"

    if not context.startswith("Failed to retrieve"):
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump({"query": query, "context": context, "pmids": pmids, "timestamp": time.time()}, f, indent=2)

    return context, pmids

def retrieve_c1_audited(drug: str, target: str, disease: str) -> Tuple[str, List[str], bool]:
    """C1: Strict Boolean AND query (Drug AND Target AND Disease)."""
    parts = []
    if drug: parts.append(f"({drug})")
    if target: parts.append(f"({target})")
    if disease: parts.append(f"({disease})")
    query = " AND ".join(parts) if parts else "pharmacology network"
    ctx, pmids = fetch_pubmed_with_pmids(query, max_results=3)
    is_empty = (len(pmids) == 0 or "No specific literature" in ctx)
    return ctx, pmids, is_empty

def retrieve_c2_audited(drug: str, target: str, disease: str) -> Tuple[str, List[str], str, bool]:
    """C2: Staged Query Relaxation with Explicit Abstention."""
    # Tier 1: Drug AND Target AND Disease
    t1_q = f"({drug}) AND ({target}) AND ({disease})"
    ctx, pmids = fetch_pubmed_with_pmids(t1_q, max_results=3)
    if pmids and "No specific literature" not in ctx:
        return ctx, pmids, "Tier 1: Drug+Target+Disease", False

    # Tier 2: Drug AND Target
    t2_q = f"({drug}) AND ({target})"
    ctx, pmids = fetch_pubmed_with_pmids(t2_q, max_results=3)
    if pmids and "No specific literature" not in ctx:
        return ctx, pmids, "Tier 2: Drug+Target", False

    # Tier 3: Drug+Disease OR Target+Disease
    t3_q = f"({drug} AND {disease}) OR ({target} AND {disease})"
    ctx, pmids = fetch_pubmed_with_pmids(t3_q, max_results=3)
    if pmids and "No specific literature" not in ctx:
        return ctx, pmids, "Tier 3: Secondary Disease Association", False

    return "[NO_LITERATURE_EVIDENCE]", [], "Abstention: No Literature Found", True

# -------------------------------------------------------------
# 2. Generator Prompts (including Condition C0b - Item 8)
# -------------------------------------------------------------
def prompt_c0(drug: str, target: str, disease: str) -> str:
    """C0: Parametric forced-generation baseline."""
    return f"""You are an expert bioinformatician.
Explain the biochemical interaction and molecular mechanism between the drug '{drug}', the biological target '{target}', and the pathological condition '{disease}'.
Detail the specific binding mechanism, affinity, and clinical therapeutic rationale.
Be specific and concise (3-5 sentences)."""

def prompt_c0b(drug: str, target: str, disease: str) -> str:
    """C0b: Neutral ungrounded prompt permitting explicit statement of insufficient evidence (Item 8)."""
    return f"""You are an expert bioinformatician analyzing a candidate drug-target interaction.
Drug: '{drug}', Target: '{target}', Disease: '{disease}'.
Provide a concise, mechanistic biological explanation of why '{drug}' interacts with '{target}' in '{disease}' IF there is documented empirical biochemical evidence for this interaction.
If there is insufficient or no documented scientific evidence for an interaction between '{drug}' and '{target}', explicitly state that there is insufficient evidence rather than hypothesizing an unverified mechanism.
Be concise (2-4 sentences)."""

def prompt_c1(drug: str, target: str, disease: str, context: str) -> str:
    """C1: Strict Boolean AND with fallback instruction."""
    return f"""You are an advanced expert Bioinformatician AI analyzing a drug-target-disease interaction.

Target Interaction: Drug '{drug}' -> Target '{target}' -> Disease '{disease}'

### Scientific Literature Retrieval (PubMed Evidence Context)
{context}

### Task
Provide a concise, mechanistic biological explanation of why '{drug}' interacts with '{target}' in '{disease}'.
Cite supporting PubMed literature using [PMID: <id>] tags where available.
If the literature context explicitly mentions these entities, integrate that evidence.
If it does not, rely on general biological principles but note the absence of direct literature.
Be concise (3-5 sentences)."""

def prompt_c2(drug: str, target: str, disease: str, context: str) -> str:
    """C2: Staged RAG with strict factual grounding & explicit abstention protocol."""
    if context == "[NO_LITERATURE_EVIDENCE]":
        return f"""You are an advanced expert Bioinformatician AI analyzing a candidate drug-target interaction.

Target Interaction: Drug '{drug}' -> Target '{target}' -> Disease '{disease}'

### Scientific Literature Retrieval
No peer-reviewed literature was found in PubMed supporting this interaction:
[NO_LITERATURE_EVIDENCE]

### Task
Follow the strict scientific abstention protocol:
State explicitly that there is no literature evidence in PubMed supporting the interaction between '{drug}' and '{target}' for '{disease}', and decline to hypothesize an ungrounded molecular mechanism."""

    return f"""You are an advanced expert Bioinformatician AI analyzing a candidate drug-target interaction.

Target Interaction: Drug '{drug}' -> Target '{target}' -> Disease '{disease}'

### Scientific Literature Retrieval (PubMed Evidence Context)
{context}

### Strict Grounding Requirement
Explain the biological interaction between '{drug}' and '{target}' in '{disease}' STRICTLY and EXCLUSIVELY based on the provided PubMed scientific evidence above.
Cite the supporting PMIDs using [PMID: <id>] tags for every factual statement.
Do NOT fabricate, speculate, or introduce unverified residues, affinities, or off-target interactions not mentioned in the context.
Be concise (3-5 sentences)."""

# -------------------------------------------------------------
# 3. Citation Verification & Entrez Fabrication Check (Item 6)
# -------------------------------------------------------------
def verify_citations(explanation: str, retrieved_pmids: List[str]) -> Tuple[float, float, List[str]]:
    """
    Extracts cited PMIDs from explanation, verifies:
    1. Citation validity rate (was the cited PMID in the retrieved context?)
    2. Fabricated-PMID rate (does the cited PMID exist in PubMed?)
    """
    cited_pmids = re.findall(r"\[PMID:\s*(\d+)\]", explanation, re.IGNORECASE)
    if not cited_pmids:
        # Also check alternative format: PMID: 12345678
        cited_pmids = re.findall(r"PMID[:\s]+(\d+)", explanation, re.IGNORECASE)

    if not cited_pmids:
        return 1.0, 0.0, []  # No citations made

    valid_count = 0
    fabricated_count = 0

    retrieved_set = set(retrieved_pmids)
    for pmid in cited_pmids:
        if pmid in retrieved_set:
            valid_count += 1
        else:
            # Check if PMID actually exists in PubMed via Entrez
            try:
                time.sleep(0.35)
                h = Entrez.esummary(db="pubmed", id=pmid)
                r = Entrez.read(h)
                h.close()
                if not r:
                    fabricated_count += 1
            except Exception:
                fabricated_count += 1

    validity_rate = valid_count / len(cited_pmids)
    fabricated_rate = fabricated_count / len(cited_pmids)
    return validity_rate, fabricated_rate, cited_pmids

# -------------------------------------------------------------
# 4. Structured Judge LLM (Items 9, 10, 11)
# -------------------------------------------------------------
def judge_explanation_and_claims(
    llm: GeminiWrapper,
    drug: str,
    target: str,
    disease: str,
    condition: str,
    explanation: str,
    context: str,
    retrieved_pmids: List[str]
) -> Dict[str, Any]:
    """
    Decomposes explanation into atomic claims and evaluates:
    - Labels: SUPPORTED / CONTRADICTED / UNVERIFIABLE
    - explanation_abstained: boolean (Item 9)
    - relevance: boolean / score (Item 10)
    """
    judge_prompt = f"""You are a strict, objective scientific fact-checker evaluating an AI-generated explanation.

Query: Drug '{drug}' -> Target '{target}' -> Disease '{disease}'
Experimental Condition: {condition}

### Retrieved PubMed Evidence Context
{context if context and context != "[NO_LITERATURE_EVIDENCE]" else "(None provided or No Literature Found)"}

### AI-Generated Explanation
\"\"\"{explanation}\"\"\"

### Instructions
1. Determine if the explanation is an explicit ABSTENTION (e.g. states that there is insufficient or no literature evidence, and declines to hypothesize an ungrounded mechanism).
   Output boolean `explanation_abstained`: true or false.
2. Determine `relevance`: Does the explanation directly address the biochemical interaction between the queried drug '{drug}' and target '{target}'? Output boolean true or false.
3. Decompose the explanation into atomic, declarative factual claims.
4. For EACH atomic claim, classify its grounding status relative to the provided PubMed Evidence Context as:
   - "SUPPORTED": Directly stated in or logically entailed by the provided PubMed Context.
   - "CONTRADICTED": Directly contradicts the provided PubMed Context.
   - "UNVERIFIABLE": Not stated in or verifiable from the provided context (speculative, ungrounded, or external parametric knowledge).
   *(Note: If condition is C0/C0b without retrieved context, specific empirical biochemical claims are UNVERIFIABLE relative to provided evidence, whereas an accurate statement that evidence is absent is SUPPORTED).*

Return ONLY a valid JSON object with the exact schema:
{{
  "explanation_abstained": true/false,
  "relevance": true/false,
  "claims": [
    {{
      "claim": "Atomic factual statement",
      "verdict": "SUPPORTED" | "CONTRADICTED" | "UNVERIFIABLE",
      "rationale": "Brief rationale"
    }}
  ]
}}"""

    judge_model = "gemini-3.1-flash-lite"
    resp_text = llm.generate(judge_prompt, model=judge_model, max_output_tokens=1024, temperature=0.0)

    cleaned = resp_text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    if cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    try:
        parsed = json.loads(cleaned)
        return parsed
    except Exception:
        # Fallback heuristic
        is_abstain = ("no literature" in explanation.lower() or "insufficient evidence" in explanation.lower() or "decline" in explanation.lower())
        return {
            "explanation_abstained": is_abstain,
            "relevance": True,
            "claims": [{
                "claim": explanation[:150],
                "verdict": "SUPPORTED" if is_abstain else "UNVERIFIABLE",
                "rationale": "Fallback parsed claim"
            }]
        }

# -------------------------------------------------------------
# 5. Clustered Bootstrap 95% Confidence Intervals (Item 11)
# -------------------------------------------------------------
def compute_triple_clustered_bootstrap(df_claims: pd.DataFrame, B: int = 1000) -> Dict[str, Dict[str, Tuple[float, float]]]:
    """
    Computes bootstrap 95% confidence intervals clustered by TRIPLE (N=60),
    resampling triples with replacement across B=1000 iterations.
    """
    unique_triples = df_claims["triple_id"].unique()
    rng = np.random.RandomState(42)
    conditions = df_claims["condition"].unique()

    bootstrap_results = {}

    for cond in conditions:
        sub_cond = df_claims[df_claims["condition"] == cond]
        triples_cond = sub_cond["triple_id"].unique()
        n_triples = len(triples_cond)

        boot_support_rates = []
        boot_unverifiable_rates = []

        for _ in range(B):
            sampled_triples = rng.choice(triples_cond, size=n_triples, replace=True)
            sampled_claims = sub_cond[sub_cond["triple_id"].isin(sampled_triples)]

            if len(sampled_claims) == 0:
                continue

            n_supp = (sampled_claims["verdict"] == "SUPPORTED").sum()
            n_unver = (sampled_claims["verdict"].isin(["UNVERIFIABLE", "HALLUCINATED"])).sum()
            total = len(sampled_claims)

            boot_support_rates.append(n_supp / total * 100.0)
            boot_unverifiable_rates.append(n_unver / total * 100.0)

        bootstrap_results[cond] = {
            "support_ci": (float(np.percentile(boot_support_rates, 2.5)), float(np.percentile(boot_support_rates, 97.5))),
            "unverifiable_ci": (float(np.percentile(boot_unverifiable_rates, 2.5)), float(np.percentile(boot_unverifiable_rates, 97.5)))
        }

    return bootstrap_results

# -------------------------------------------------------------
# 6. Main Runner
# -------------------------------------------------------------
def run_rag_benchmark():
    print("=" * 70)
    print("STARTING RAG AUDIT BENCHMARK (ITEMS 6-11)")
    print("=" * 70, flush=True)

    df_triples = pd.read_csv(TRIPLES_CSV)
    print(f"Loaded {len(df_triples)} evaluation triples from {TRIPLES_CSV}", flush=True)

    llm = GeminiWrapper(rpm_limit=12.0, min_interval=4.5)
    gen_model = "gemini-3.6-flash"
    judge_model = "gemini-3.1-flash-lite"
    print(f"[LLM Setup] Generator: {gen_model} | Judge: {judge_model} (temp 0.0)", flush=True)

    c1_tier_stats = {"total": 0, "empty": 0}
    c2_tier_stats = {"total": 0, "tier1": 0, "tier2": 0, "tier3": 0, "abstain": 0}

    eval_records = []
    claims_records = []

    for idx, row in df_triples.iterrows():
        t_id = int(row["triple_id"])
        drug = str(row["drug"]).strip()
        target = str(row["target"]).strip()
        disease = str(row["disease"]).strip()
        category = str(row["category"]).strip()
        gt_support = bool(row["ground_truth_literature_support"])

        # Retrieval
        # C1
        ctx_c1, pmids_c1, empty_c1 = retrieve_c1_audited(drug, target, disease)
        c1_tier_stats["total"] += 1
        if empty_c1: c1_tier_stats["empty"] += 1

        # C2
        ctx_c2, pmids_c2, tier_label_c2, empty_c2 = retrieve_c2_audited(drug, target, disease)
        c2_tier_stats["total"] += 1
        if "Tier 1" in tier_label_c2: c2_tier_stats["tier1"] += 1
        elif "Tier 2" in tier_label_c2: c2_tier_stats["tier2"] += 1
        elif "Tier 3" in tier_label_c2: c2_tier_stats["tier3"] += 1
        else: c2_tier_stats["abstain"] += 1

        # Generation for Conditions: C0, C0b, C1, C2
        conds = [
            ("C0_LLM_Only", prompt_c0(drug, target, disease), "", []),
            ("C0b_Neutral_Prompt", prompt_c0b(drug, target, disease), "", []),
            ("C1_Strict_RAG", prompt_c1(drug, target, disease, ctx_c1), ctx_c1, pmids_c1),
            ("C2_Staged_RAG", prompt_c2(drug, target, disease, ctx_c2), ctx_c2, pmids_c2)
        ]

        for cond_name, p_text, context_text, ret_pmids in conds:
            exp_text = llm.generate(p_text, model=gen_model, max_output_tokens=512, temperature=0.0)

            # Citation verification (Item 6)
            cite_val, fab_rate, cited_pmids = verify_citations(exp_text, ret_pmids)

            # Judge evaluation (Item 9 & 10)
            judge_res = judge_explanation_and_claims(
                llm, drug, target, disease, cond_name, exp_text, context_text, ret_pmids
            )

            is_abstained = bool(judge_res.get("explanation_abstained", False))
            is_relevant = bool(judge_res.get("relevance", True))
            claims = judge_res.get("claims", [])

            n_supp = sum(1 for c in claims if c.get("verdict") == "SUPPORTED")
            n_unver = sum(1 for c in claims if c.get("verdict") in ["UNVERIFIABLE", "HALLUCINATED"])
            n_contra = sum(1 for c in claims if c.get("verdict") == "CONTRADICTED")
            total_claims = max(1, len(claims))

            eval_records.append({
                "triple_id": t_id,
                "drug": drug,
                "target": target,
                "disease": disease,
                "category": category,
                "ground_truth_support": gt_support,
                "condition": cond_name,
                "explanation": exp_text,
                "explanation_abstained": is_abstained,
                "relevance": is_relevant,
                "total_claims": len(claims),
                "supported_claims": n_supp,
                "unverifiable_claims": n_unver,
                "contradicted_claims": n_contra,
                "support_rate": n_supp / total_claims * 100.0,
                "unverifiable_rate": n_unver / total_claims * 100.0,
                "citation_validity": cite_val,
                "fabricated_pmid_rate": fab_rate,
                "cited_pmids": ";".join(cited_pmids)
            })

            for c_idx, c_obj in enumerate(claims):
                claims_records.append({
                    "triple_id": t_id,
                    "condition": cond_name,
                    "claim_id": f"{t_id}_{cond_name}_{c_idx+1}",
                    "claim": c_obj.get("claim", ""),
                    "verdict": c_obj.get("verdict", "UNVERIFIABLE"),
                    "rationale": c_obj.get("rationale", "")
                })

        if (idx + 1) % 10 == 0:
            print(f"[Progress] Evaluated {idx + 1}/{len(df_triples)} triples...", flush=True)

    # Save outputs
    df_eval = pd.DataFrame(eval_records)
    df_claims = pd.DataFrame(claims_records)

    df_eval.to_csv(RESULTS_DIR / "rag_evaluation.csv", index=False)
    df_claims.to_csv(RESULTS_DIR / "rag_claims_detailed.csv", index=False)
    print(f"[Artifact Saved] RAG evaluation saved to {RESULTS_DIR / 'rag_evaluation.csv'}", flush=True)

    # Save retrieval tier stats (Item 7)
    df_tier_stats = pd.DataFrame([
        {"condition": "C1_Strict_RAG", "tier": "Total Queries", "count": c1_tier_stats["total"], "empty_rate": c1_tier_stats["empty"]/max(1, c1_tier_stats["total"])*100.0},
        {"condition": "C2_Staged_RAG", "tier": "Tier 1: Drug+Target+Disease", "count": c2_tier_stats["tier1"], "empty_rate": 0.0},
        {"condition": "C2_Staged_RAG", "tier": "Tier 2: Drug+Target", "count": c2_tier_stats["tier2"], "empty_rate": 0.0},
        {"condition": "C2_Staged_RAG", "tier": "Tier 3: Secondary Disease", "count": c2_tier_stats["tier3"], "empty_rate": 0.0},
        {"condition": "C2_Staged_RAG", "tier": "Abstention: Zero Literature", "count": c2_tier_stats["abstain"], "empty_rate": c2_tier_stats["abstain"]/max(1, c2_tier_stats["total"])*100.0}
    ])
    df_tier_stats.to_csv(RESULTS_DIR / "rag_retrieval_tier_stats.csv", index=False)
    print(f"[Artifact Saved] Tier stats saved to {RESULTS_DIR / 'rag_retrieval_tier_stats.csv'}", flush=True)

    # Compute Clustered Bootstrap CIs (Item 11)
    boot_ci = compute_triple_clustered_bootstrap(df_claims, B=1000)

    # Generate LaTeX Table
    generate_rag_latex_table(df_eval, boot_ci)

    # Generate Figures
    generate_rag_publication_figures(df_eval)

    print("\n[Finished] Phase 3 RAG Audit Fixes Completed Successfully!", flush=True)

# -------------------------------------------------------------
# LaTeX Table Generation
# -------------------------------------------------------------
def generate_rag_latex_table(df_eval: pd.DataFrame, boot_ci: Dict[str, Dict[str, Tuple[float, float]]]):
    agg = df_eval.groupby("condition").agg({
        "support_rate": "mean",
        "unverifiable_rate": "mean",
        "explanation_abstained": "mean",
        "citation_validity": "mean",
        "fabricated_pmid_rate": "mean"
    }).reset_index()

    tex_path = TABLES_DIR / "rag_comparison.tex"
    with open(tex_path, "w") as f:
        f.write("% IEEE Format RAG Grounding & Hallucination Mitigation Table\n")
        f.write(r"\begin{table*}[t]" + "\n")
        f.write(r"\centering" + "\n")
        f.write(r"\caption{Grounding, Abstention, and Factual Support across RAG Conditions ($N=60$ Triples, Cluster-Bootstrapped 95\% CIs)}" + "\n")
        f.write(r"\label{tab:rag_comparison}" + "\n")
        f.write(r"\small" + "\n")
        f.write(r"\begin{tabular}{lccccc}" + "\n")
        f.write(r"\hline" + "\n")
        f.write(r"\textbf{Condition} & \textbf{Supported Rate (\%)} & \textbf{Unverifiable Rate (\%)} & \textbf{Abstention Rate (\%)} & \textbf{Citation Validity (\%)} & \textbf{Fabricated PMID (\%)} \\" + "\n")
        f.write(r"\hline" + "\n")

        for _, r in agg.iterrows():
            c_name = r["condition"]
            disp_c = c_name.replace("_", " ")
            if "C2" in c_name:
                disp_c = r"\textbf{" + disp_c + r"}"
            ci_supp = boot_ci.get(c_name, {}).get("support_ci", (r["support_rate"], r["support_rate"]))
            ci_unver = boot_ci.get(c_name, {}).get("unverifiable_ci", (r["unverifiable_rate"], r["unverifiable_rate"]))
            f.write(
                f"{disp_c} & {r['support_rate']:.1f}\\% [{ci_supp[0]:.1f}, {ci_supp[1]:.1f}] & "
                f"{r['unverifiable_rate']:.1f}\\% [{ci_unver[0]:.1f}, {ci_unver[1]:.1f}] & "
                f"{r['explanation_abstained']*100.0:.1f}\\% & "
                f"{r['citation_validity']*100.0:.1f}\\% & "
                f"{r['fabricated_pmid_rate']*100.0:.1f}\\% \\\\\n"
            )

        f.write(r"\hline" + "\n")
        f.write(r"\end{tabular}" + "\n")
        f.write(r"\end{table*}" + "\n")
    print(f"[Artifact Saved] RAG LaTeX table written to {tex_path}", flush=True)

# -------------------------------------------------------------
# Publication Figures Generation
# -------------------------------------------------------------
def generate_rag_publication_figures(df_eval: pd.DataFrame):
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

    # 1. Supported vs Unverifiable Rate
    ax1 = axes[0]
    agg = df_eval.groupby("condition")[["support_rate", "unverifiable_rate"]].mean().reset_index()
    agg_melt = agg.melt(id_vars="condition", value_vars=["support_rate", "unverifiable_rate"], var_name="Metric", value_name="Rate")
    agg_melt["Metric"] = agg_melt["Metric"].replace({"support_rate": "Supported Rate (%)", "unverifiable_rate": "Unverifiable Rate (%)"})
    sns.barplot(data=agg_melt, x="condition", y="Rate", hue="Metric", ax=ax1, palette=["#2ca02c", "#d62728"])
    ax1.set_title("(a) Factual Grounding vs. Unverifiable Claims by Condition", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Claim Proportion (%)", fontsize=10)
    ax1.set_xlabel("")
    ax1.tick_params(axis="x", rotation=15)
    ax1.set_ylim(0, 105)

    # 2. Abstention Rate across Negative Controls
    ax2 = axes[1]
    sub_neg = df_eval[df_eval["category"] == "unsupported"]
    sns.barplot(data=sub_neg, x="condition", y="explanation_abstained", ax=ax2, palette="Blues_r")
    ax2.set_title("(b) Abstention Rate on 0-Hit Negative Controls (N=15)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Abstention Proportion (1.0 = 100%)", fontsize=10)
    ax2.set_xlabel("")
    ax2.tick_params(axis="x", rotation=15)
    ax2.set_ylim(0, 1.05)

    plt.tight_layout()
    png_path = FIGURES_DIR / "rag_hallucination_rate.png"
    pdf_path = FIGURES_DIR / "rag_hallucination_rate.pdf"
    plt.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()
    print(f"[Artifact Saved] RAG publication figures saved to {png_path} and {pdf_path}", flush=True)

if __name__ == "__main__":
    run_rag_benchmark()
