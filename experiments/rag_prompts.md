# MediGraph: Exact RAG Prompts, Model Configurations, and Parameters

## Model Configuration & Hyperparameters
- **Generator Model**: `gemini-3.6-flash`
- **Judge Model**: `gemini-3.1-flash-lite` (distinct model family variant to reduce self-preference bias)
- **Temperature**: `0.0` (strictly deterministic greedy decoding across all conditions)
- **Thinking Configuration**: Minimal / Disabled
- **Max Output Tokens**: `512` (Generator), `1024` (Judge)

---

## Condition C0: Parametric LLM-Only Baseline (Forced Generation)
```text
You are an expert bioinformatician.
Explain the biochemical interaction and molecular mechanism between the drug '{drug}', the biological target '{target}', and the pathological condition '{disease}'.
Detail the specific binding mechanism, affinity, and clinical therapeutic rationale.
Be specific and concise (3-5 sentences).
```

---

## Condition C0b: Neutral Parametric Prompt (Abstention Permitted)
```text
You are an expert bioinformatician analyzing a candidate drug-target interaction.
Drug: '{drug}', Target: '{target}', Disease: '{disease}'.
Provide a concise, mechanistic biological explanation of why '{drug}' interacts with '{target}' in '{disease}' IF there is documented empirical biochemical evidence for this interaction.
If there is insufficient or no documented scientific evidence for an interaction between '{drug}' and '{target}', explicitly state that there is insufficient evidence rather than hypothesizing an unverified mechanism.
Be concise (2-4 sentences).
```

---

## Condition C1: Strict Boolean AND RAG (With Ungrounded Fallback)
```text
You are an advanced expert Bioinformatician AI analyzing a drug-target-disease interaction.

Target Interaction: Drug '{drug}' -> Target '{target}' -> Disease '{disease}'

### Scientific Literature Retrieval (PubMed Evidence Context)
{context}

### Task
Provide a concise, mechanistic biological explanation of why '{drug}' interacts with '{target}' in '{disease}'.
Cite supporting PubMed literature using [PMID: <id>] tags where available.
If the literature context explicitly mentions these entities, integrate that evidence.
If it does not, rely on general biological principles but note the absence of direct literature.
Be concise (3-5 sentences).
```

---

## Condition C2: Staged Relaxation RAG (Strict Factual Grounding & Explicit Abstention)

### Sub-condition C2a: Literature Evidence Retrieved (Tier 1, 2, or 3)
```text
You are an advanced expert Bioinformatician AI analyzing a candidate drug-target interaction.

Target Interaction: Drug '{drug}' -> Target '{target}' -> Disease '{disease}'

### Scientific Literature Retrieval (PubMed Evidence Context)
{context}

### Strict Grounding Requirement
Explain the biological interaction between '{drug}' and '{target}' in '{disease}' STRICTLY and EXCLUSIVELY based on the provided PubMed scientific evidence above.
Cite the supporting PMIDs using [PMID: <id>] tags for every factual statement.
Do NOT fabricate, speculate, or introduce unverified residues, affinities, or off-target interactions not mentioned in the context.
Be concise (3-5 sentences).
```

### Sub-condition C2b: Zero Literature Evidence (Empty Retrieval across all 3 Tiers)
```text
You are an advanced expert Bioinformatician AI analyzing a candidate drug-target interaction.

Target Interaction: Drug '{drug}' -> Target '{target}' -> Disease '{disease}'

### Scientific Literature Retrieval
No peer-reviewed literature was found in PubMed supporting this interaction:
[NO_LITERATURE_EVIDENCE]

### Task
Follow the strict scientific abstention protocol:
State explicitly that there is no literature evidence in PubMed supporting the interaction between '{drug}' and '{target}' for '{disease}', and decline to hypothesize an ungrounded molecular mechanism.
```

---

## LLM Fact-Checking Judge Prompt (Applied Across All Conditions)
```text
You are a strict, objective scientific fact-checker evaluating an AI-generated explanation.

Query: Drug '{drug}' -> Target '{target}' -> Disease '{disease}'
Experimental Condition: {condition}

### Retrieved PubMed Evidence Context
{context if context and context != "[NO_LITERATURE_EVIDENCE]" else "(None provided or No Literature Found)"}

### AI-Generated Explanation
"""{explanation}"""

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
{
  "explanation_abstained": true/false,
  "relevance": true/false,
  "claims": [
    {
      "claim": "Atomic factual statement",
      "verdict": "SUPPORTED" | "CONTRADICTED" | "UNVERIFIABLE",
      "rationale": "Brief rationale"
    }
  ]
}
```
