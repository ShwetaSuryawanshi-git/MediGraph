# File: MediGraph/ml_pipeline/feature_extraction.py
# Path: /Users/shweta/Desktop/major project fy/MediGraph/ml_pipeline/feature_extraction.py

"""
Feature Extraction Module
Computes molecular fingerprints and protein features for DTI prediction
"""

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from config import Config


class FeatureExtractor:
    """Feature extraction for drugs and targets"""
    
    # Standard amino acids
    AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
    
    @staticmethod
    def compute_morgan_fingerprint(smiles):
        """
        Compute Morgan fingerprint from SMILES string
        
        Args:
            smiles: SMILES representation of the molecule
            
        Returns:
            numpy array of shape (1024,) or None if invalid SMILES
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            fp = AllChem.GetMorganFingerprintAsBitVect(
                mol, 
                radius=Config.MORGAN_FP_RADIUS, 
                nBits=Config.MORGAN_FP_BITS
            )
            return np.array(fp)
        except Exception as e:
            print(f"Error computing Morgan fingerprint: {e}")
            return None
    
    @staticmethod
    def compute_amino_acid_composition(sequence):
        """
        Compute amino acid composition from protein sequence
        
        Args:
            sequence: Protein amino acid sequence (string)
            
        Returns:
            numpy array of shape (20,) representing composition
        """
        if not sequence or len(sequence) == 0:
            return np.zeros(Config.AAC_LENGTH)
        
        # Clean sequence - keep only valid amino acids
        sequence = sequence.upper()
        sequence = ''.join(c for c in sequence if c in FeatureExtractor.AMINO_ACIDS)
        
        if len(sequence) == 0:
            return np.zeros(Config.AAC_LENGTH)
        
        # Count amino acids
        composition = np.zeros(Config.AAC_LENGTH)
        for i, aa in enumerate(FeatureExtractor.AMINO_ACIDS):
            composition[i] = sequence.count(aa) / len(sequence)
        
        return composition
    
    @staticmethod
    def extract_drug_features(drug_data):
        """
        Extract features from drug data
        
        Args:
            drug_data: Dict containing drug information with 'smiles' key
            
        Returns:
            numpy array of Morgan fingerprint or None
        """
        smiles = drug_data.get('smiles')
        if not smiles:
            return None
        
        return FeatureExtractor.compute_morgan_fingerprint(smiles)
    
    @staticmethod
    def extract_target_features(target_data):
        """
        Extract features from target data
        
        Args:
            target_data: Dict containing target information with 'protein_sequence' key
            
        Returns:
            numpy array of amino acid composition
        """
        sequence = target_data.get('protein_sequence', '')
        return FeatureExtractor.compute_amino_acid_composition(sequence)
    
    @staticmethod
    def create_pair_features(drug_features, target_features):
        """
        Create feature vector for a drug-target pair
        
        Args:
            drug_features: Morgan fingerprint (1024,)
            target_features: Amino acid composition (20,)
            
        Returns:
            Concatenated feature vector (1044,)
        """
        if drug_features is None or target_features is None:
            return None
        
        return np.concatenate([drug_features, target_features])


# Export singleton instance
feature_extractor = FeatureExtractor()
