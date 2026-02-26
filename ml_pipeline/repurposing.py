# File: MediGraph/ml_pipeline/repurposing.py
# Path: /Users/shweta/Desktop/major project fy/MediGraph/ml_pipeline/repurposing.py

"""
Repurposing Derivation Module
Derives drug repurposing candidates from DTI predictions
Following the deterministic logic defined in the ML Pipeline Design
"""

import numpy as np
from config import Config
from firebase_client import firebase_client


class RepurposingDeriver:
    """Derives repurposing candidates from prediction results"""
    
    def __init__(self):
        """Initialize with Firebase client"""
        self.db = firebase_client
        self._disease_cache = {}
        self._target_cache = {}
    
    def _load_caches(self):
        """Pre-load diseases and targets for lookups"""
        diseases = self.db.get_all_diseases()
        for d in diseases:
            self._disease_cache[d['id']] = d
        
        targets = self.db.get_all_targets()
        for t in targets:
            self._target_cache[t['id']] = t
    
    def derive_candidates(self, positive_interactions, drugs_data):
        """
        Derive repurposing candidates from positive DTI predictions
        
        Logic:
        1. For each high-confidence positive prediction
        2. Find target's associated diseases
        3. Compare against drug's approved indication
        4. If different, create repurposing candidate
        
        Args:
            positive_interactions: List of positive interaction dicts
            drugs_data: Dict mapping drug_id to drug data
            
        Returns:
            List of repurposing candidate dicts
        """
        self._load_caches()
        candidates = []
        
        for interaction in positive_interactions:
            # Only consider high confidence predictions
            if interaction.get('confidence_score', 0) < Config.HIGH_CONFIDENCE_THRESHOLD:
                continue
            
            drug_id = interaction['drug_id']
            target_id = interaction['target_id']
            interaction_id = interaction['id']
            
            drug = drugs_data.get(drug_id)
            target = self._target_cache.get(target_id)
            
            if not drug or not target:
                continue
            
            # Get drug's current approved indication
            approved_indication = drug.get('approved_indication', '').lower()
            
            # Get target's associated diseases
            associated_disease_ids = target.get('associated_disease_ids', [])
            
            for disease_id in associated_disease_ids:
                disease = self._disease_cache.get(disease_id)
                if not disease:
                    continue
                
                disease_name = disease.get('disease_name', '').lower()
                
                # Skip if drug is already approved for this disease
                if approved_indication and disease_name in approved_indication:
                    continue
                
                # Calculate repurposing score
                repurposing_score = self._calculate_repurposing_score(
                    interaction, target, disease
                )
                
                if repurposing_score < Config.REPURPOSING_SCORE_THRESHOLD:
                    continue
                
                # Create candidate
                candidate = {
                    'drug_id': drug_id,
                    'target_id': target_id,
                    'proposed_disease_id': disease_id,
                    'original_disease_id': self._find_original_disease_id(approved_indication),
                    'repurposing_score': repurposing_score,
                    'pathway': [interaction_id],  # Traceability
                    'rationale': self._generate_rationale(drug, target, disease)
                }
                
                candidates.append(candidate)
        
        return candidates
    
    def _calculate_repurposing_score(self, interaction, target, disease):
        """
        Calculate repurposing confidence score
        
        Combines:
        - Prediction confidence
        - Target-disease association strength
        """
        prediction_confidence = interaction.get('confidence_score', 0)
        
        # Simple scoring - can be enhanced with more factors
        # Base score from ML confidence
        base_score = prediction_confidence
        
        # Bonus for validated targets
        if target.get('is_validated'):
            base_score *= 1.1
        
        return min(base_score, 1.0)  # Cap at 1.0
    
    def _generate_rationale(self, drug, target, disease):
        """Generate human-readable rationale"""
        drug_name = drug.get('drug_name', 'Unknown drug')
        target_name = target.get('target_name', 'Unknown target')
        disease_name = disease.get('disease_name', 'Unknown disease')
        
        return (
            f"{drug_name} is predicted to interact with {target_name}, "
            f"which is associated with {disease_name}. "
            f"This suggests potential therapeutic application."
        )
    
    def _find_original_disease_id(self, indication_text):
        """
        Try to find disease ID matching the approved indication
        Returns None if no match found
        """
        if not indication_text:
            return None
        
        indication_lower = indication_text.lower()
        for disease_id, disease in self._disease_cache.items():
            if disease.get('disease_name', '').lower() in indication_lower:
                return disease_id
        
        return None
    
    def write_candidates_to_firestore(self, candidates):
        """
        Write repurposing candidates to Firestore
        
        Args:
            candidates: List of candidate dicts
            
        Returns:
            List of created document IDs
        """
        if not candidates:
            return []
        
        return self.db.batch_create_repurposing_candidates(candidates)


# Export singleton
repurposing_deriver = RepurposingDeriver()
