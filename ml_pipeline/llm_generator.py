import os
import logging
from google import genai
from google.genai import types

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

def generate_repurposing_rationale(drug_name, target_name, disease_name, top_features_list):
    """
    Calls Gemini API to generate a 2-3 sentence scientific rationale for repurposing
    a drug for a disease based on the target association.
    Falls back to a rule-based rationale if the API fails or key is missing.
    """
    fallback = (f"{drug_name} may treat {disease_name} via interaction with {target_name}. "
               f"This is based on structural feature alignments across known positive interactions.")
               
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        log.warning("GEMINI_API_KEY environment variable not set. Falling back to rule-based rationale.")
        return fallback

    try:
        client = genai.Client(api_key=api_key)
        
        # Format the top features for the prompt
        features_str = ", ".join([f"{f['name']} ({f['direction']} interaction probability)" for f in top_features_list])
        
        prompt = (
            f"Write a 2-3 sentence scientific biological rationale explaining how the drug {drug_name} "
            f"might be repurposed to treat {disease_name} by interacting with the target {target_name}.\n"
            f"The AI model identified these key chemical/amino-acid structural features driving the prediction: {features_str}.\n"
            f"Ensure the tone is scientific, concise, and do not use conversational filler."
        )

        response = client.models.generate_content(
            model='gemini-1.5-flash',
            contents=prompt,
        )
        if response.text:
             # strip newlines which might mess up markdown rendering
             return response.text.replace('\n', ' ').strip()
        else:
            return fallback

    except Exception as e:
        log.error(f"Gemini API call failed: {e}. Falling back to rule-based rationale.")
        return fallback

def generate_node_summary(entity_type, name, context=""):
    """
    Generates a 3-4 sentence biological summary for a graph node (Drug, Target, or Disease).
    Because frontend API calls directly to Gemini expose the API key, we could pre-generate
    or proxy this. For now, it provides the Python utility.
    """
    fallback = f"Detailed biological summary for the {entity_type.lower()} {name} is currently limited."
    
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        return fallback

    try:
        client = genai.Client(api_key=api_key)
        
        prompt = (
            f"Provide a 3-4 sentence biological and medical summary of the {entity_type}: {name}.\n"
        )
        if context:
            prompt += f"Context: {context}\n"
            
        prompt += "Keep the response purely informational and scientific. No conversational filler."

        response = client.models.generate_content(
            model='gemini-1.5-flash',
            contents=prompt,
        )
        if response.text:
             return response.text.replace('\n', ' ').strip()
        else:
            return fallback

    except Exception as e:
        log.error(f"Gemini API call failed: {e}")
        return fallback
