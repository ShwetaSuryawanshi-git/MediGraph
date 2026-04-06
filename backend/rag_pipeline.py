import os
import logging
from typing import List, Dict, Any
from Bio import Entrez
from openai import OpenAI

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

Entrez.email = os.environ.get("ENTREZ_EMAIL", "medigraph_research@example.com")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

def retrieve_pubmed_abstracts(query: str, max_results: int = 3) -> str:
    """
    Searches PubMed for a query and retrieves the abstracts.
    """
    try:
        log.info(f"Querying PubMed for: {query}")
        handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
        record = Entrez.read(handle)
        handle.close()
        
        id_list = record.get("IdList", [])
        if not id_list:
            log.info("No PubMed articles found.")
            return "No specific literature found for this query."
            
        handle = Entrez.efetch(db="pubmed", id=id_list, retmode="xml")
        articles = Entrez.read(handle)
        handle.close()
        
        abstracts = []
        for article in articles.get("PubmedArticle", []):
            try:
                article_data = article["MedlineCitation"]["Article"]
                title = article_data.get("ArticleTitle", "Unknown Title")
                abstract_text = article_data.get("Abstract", {}).get("AbstractText", [""])
                if isinstance(abstract_text, list) and len(abstract_text) > 0:
                    abstract = abstract_text[0]
                else:
                    abstract = str(abstract_text)
                    
                abstracts.append(f"Title: {title}\nAbstract: {abstract}")
            except Exception as e:
                log.warning(f"Failed to parse an article: {e}")
                continue
                
        return "\n\n---\n\n".join(abstracts)
    except Exception as e:
        log.error(f"PubMed retrieval failed: {e}")
        return "Failed to retrieve literature context due to an API error."

def explain_graph_rag(nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]) -> str:
    """
    Main RAG function to generate explainability report for a sub-graph.
    """
    if not client:
        return "Error: OPENAI_API_KEY is not set on the server."
        
    # Extract entities
    drugs = [n['label'] for n in nodes if n.get('type') == 'drug']
    targets = [n['label'] for n in nodes if n.get('type') == 'target']
    diseases = [n['label'] for n in nodes if n.get('type') == 'disease']
    
    # 1. Construct Search Query
    # A simple strategy: search for Drug + Disease, or if missing, Drug + Target etc.
    query_parts = []
    if drugs:
        query_parts.append("(" + " OR ".join(drugs) + ")")
    if targets:
        # Don't overload the query, just use top targets if there are many
        query_parts.append("(" + " OR ".join(targets[:3]) + ")")
    if diseases:
        query_parts.append("(" + " OR ".join(diseases) + ")")
        
    pubmed_query = " AND ".join(query_parts) if query_parts else "pharmacology network"
    
    # 2. Retrieve Context
    context = retrieve_pubmed_abstracts(pubmed_query, max_results=3)
    
    # 3. Construct LLM Prompt
    num_nodes = len(nodes)
    num_edges = len(edges)
    
    prompt = f"""You are an advanced expert Bioinformatician AI analyzing a drug-target-disease network graph.

### Graph Structural Metrics
- Total Entities: {num_nodes}
- Total Interactions (Edges): {num_edges}
- Drugs in view: {', '.join(drugs) if drugs else 'None'}
- Biological Targets in view: {', '.join(targets) if targets else 'None'}
- Pathological Diseases in view: {', '.join(diseases) if diseases else 'None'}

### Scientific Literature Retrieval (RAG Context)
The following abstracts were dynamically retrieved from PubMed based on the graph entities:
{context}

### Task
Synthesize a comprehensive, professional, 2-3 paragraph explainability report for this specific sub-graph. 
- Explain how these distinct entities interrelate biologically.
- Highlight any potential mechanisms of action or repurposing hypotheses suggested by the combination of these nodes and the retrieved literature.
- If the RAG Context explicitly mentions these drugs/targets/diseases, integrate that evidence. If it doesn't, rely on fundamental biological principles but state that direct literature was not pulled.
- Use professional Markdown formatting (bolding key entities, bullet points if necessary). Do not hallucinate novel unverified interactions as truth.
"""

    log.info("Sending RAG prompt to OpenAI...")
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert computational biologist and network pharmacologist."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=600
        )
        return response.choices[0].message.content
    except Exception as e:
        log.error(f"OpenAI Generation failed: {e}")
        return f"Failed to generate AI rationale. Error: {str(e)}"
