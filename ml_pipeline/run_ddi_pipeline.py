# File: MediGraph/ml_pipeline/run_ddi_pipeline.py

"""
DDI Pipeline Runner
Orchestrates the complete DDI prediction workflow
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from itertools import combinations
import numpy as np

from config import Config
from firebase_client import firebase_client
from chembl_fetcher import build_drug_features, build_pair_feature_vector
from model_training import DTIModel


def load_data():
    """Load drugs from Firestore"""
    print("\n[1/4] Loading data from Firestore...")
    drugs = firebase_client.get_all_drugs()
    print(f"  → Loaded {len(drugs)} drugs")
    return drugs


def extract_features(drugs):
    """Extract features for all drugs"""
    print("\n[2/4] Extracting drug features...")
    drug_fps = {}
    valid_drugs = []
    
    for drug in drugs:
        fp = build_drug_features(drug)
        if fp is not None:
            drug_fps[drug['id']] = fp
            valid_drugs.append(drug)
            
    print(f"  → Extracted valid features for {len(valid_drugs)} out of {len(drugs)} drugs")
    return valid_drugs, drug_fps


def generate_pairs_and_predict(valid_drugs, drug_fps, model_type='random_forest', model_path=None):
    """Generate pairs and predict directly to save memory"""
    print(f"\n[3/4] Generating pairs and predicting with {model_type}...")
    
    if model_path and Path(model_path).exists():
        print(f"  → Loading trained model from {model_path}")
        model = DTIModel.load(model_path)
    else:
        raise ValueError(f"Model file not found at {model_path}. Train it first using train_ddi_model.py!")

    results = []
    
    all_pairs = list(combinations(valid_drugs, 2))
    print(f"  → Total pairs to evaluate: {len(all_pairs)}")
    
    # Process in batches to avoid huge memory usage
    batch_size = 5000
    for i in range(0, len(all_pairs), batch_size):
        batch_pairs = all_pairs[i:i+batch_size]
        X_batch = []
        meta_batch = []
        
        for drug_a, drug_b in batch_pairs:
            fp_a = drug_fps[drug_a['id']]
            fp_b = drug_fps[drug_b['id']]
            
            pair_vec = build_pair_feature_vector(drug_a, drug_b, fp_a, fp_b)
            if pair_vec is not None:
                X_batch.append(pair_vec)
                meta_batch.append((drug_a, drug_b))
                
        if not X_batch:
            continue
            
        X_batch = np.array(X_batch)
        predictions, probabilities = model.predict(X_batch)
        
        for j, (drug_a, drug_b) in enumerate(meta_batch):
            is_positive = bool(predictions[j])
            pred_score = float(probabilities[j])
            
            # Since the ChEMBL fallback logic trains to identify positive DDI
            if is_positive:
                results.append({
                    'drug_a_id': drug_a['id'],
                    'drug_b_id': drug_b['id'],
                    'drug_a_name': drug_a.get('drug_name', 'Unknown'),
                    'drug_b_name': drug_b.get('drug_name', 'Unknown'),
                    'prediction_score': pred_score,
                    'confidence_score': pred_score,
                    'model_used': model_type,
                    'interaction_type': 'predicted_ddi'
                })
                
    print(f"  → {len(results)} positive DDI predictions found out of {len(all_pairs)} pairs evaluated")
    return results


def write_interactions_to_firestore(interactions):
    """Write positive interactions to Firestore"""
    print("\n[4/4] Writing DDI interactions to Firestore...")
    
    if not interactions:
        print("  → No positive interactions to write")
        return []
    
    # Normally we would check existing interactions, but we'll clear and seed for the demo 
    # to avoid complex queries across large drug_drug_interactions collections.
    # In a full production app, you would check existence first.
    
    # First, let's clear existing DDI to avoid duplicates on re-runs
    print("  → Clearing old DDI predictions...")
    docs = firebase_client.db.collection(Config.COLLECTIONS['ddi_interactions']).stream()
    for doc in docs:
        doc.reference.delete()
        
    print("  → Writing new predictions...")
    created_ids = firebase_client.batch_create_ddi_interactions(interactions)
    
    for i, created_id in enumerate(created_ids):
        interactions[i]['id'] = created_id
        
    print(f"  → Created {len(created_ids)} new DDI document entries")
    return interactions


def run_full_pipeline(model_type='random_forest', model_path='./models/ddi_model.pkl'):
    start_time = datetime.now()
    print("=" * 60)
    print("MediGraph DDI Prediction Pipeline")
    print(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    try:
        drugs = load_data()
        valid_drugs, drug_fps = extract_features(drugs)
        interactions = generate_pairs_and_predict(valid_drugs, drug_fps, model_type, model_path)
        interactions_saved = write_interactions_to_firestore(interactions)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print("\n" + "=" * 60)
        print("DDI Pipeline Complete!")
        print(f"Duration: {duration:.1f} seconds")
        print(f"DDI Saved: {len(interactions_saved)}")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n[ERROR] Pipeline failed: {e}")
        return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MediGraph DDI Prediction Pipeline')
    parser.add_argument('--model', choices=['random_forest', 'logistic_regression', 'svm'], default='random_forest')
    parser.add_argument('--model-path', type=str, default='./models/ddi_model.pkl')
    args = parser.parse_args()
    
    success = run_full_pipeline(args.model, args.model_path)
    sys.exit(0 if success else 1)
