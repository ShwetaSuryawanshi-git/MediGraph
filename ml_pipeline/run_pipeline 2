# File: MediGraph/ml_pipeline/run_pipeline.py
# Path: /Users/shweta/Desktop/major project fy/MediGraph/ml_pipeline/run_pipeline.py

"""
Main Pipeline Runner
Orchestrates the complete DTI prediction and repurposing workflow
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
import numpy as np

from config import Config
from firebase_client import firebase_client
from feature_extraction import FeatureExtractor
from model_training import DTIModel, ModelEnsemble
from repurposing import repurposing_deriver


def load_data():
    """Load drugs and targets from Firestore"""
    print("\n[1/6] Loading data from Firestore...")
    
    drugs = firebase_client.get_all_drugs()
    targets = firebase_client.get_all_targets()
    
    print(f"  → Loaded {len(drugs)} drugs")
    print(f"  → Loaded {len(targets)} targets")
    
    return drugs, targets


def extract_features(drugs, targets):
    """Extract features for all drug-target pairs"""
    print("\n[2/6] Extracting features...")
    
    # Pre-compute drug features
    drug_features = {}
    for drug in drugs:
        features = FeatureExtractor.extract_drug_features(drug)
        if features is not None:
            drug_features[drug['id']] = {
                'features': features,
                'data': drug
            }
    print(f"  → Extracted features for {len(drug_features)} drugs")
    
    # Pre-compute target features
    target_features = {}
    for target in targets:
        features = FeatureExtractor.extract_target_features(target)
        target_features[target['id']] = {
            'features': features,
            'data': target
        }
    print(f"  → Extracted features for {len(target_features)} targets")
    
    return drug_features, target_features


def generate_pairs(drug_features, target_features):
    """Generate all drug-target pair features"""
    print("\n[3/6] Generating drug-target pairs...")
    
    pairs = []
    pair_labels = []
    
    for drug_id, drug_info in drug_features.items():
        for target_id, target_info in target_features.items():
            pair_features = FeatureExtractor.create_pair_features(
                drug_info['features'],
                target_info['features']
            )
            if pair_features is not None:
                pairs.append({
                    'drug_id': drug_id,
                    'target_id': target_id,
                    'features': pair_features
                })
    
    print(f"  → Generated {len(pairs)} drug-target pairs")
    return pairs


def make_predictions(pairs, model_type='random_forest', model_path=None):
    """Make DTI predictions for all pairs"""
    print(f"\n[4/6] Making predictions with {model_type}...")
    
    # Load or create model
    if model_path and Path(model_path).exists():
        print(f"  → Loading trained model from {model_path}")
        model = DTIModel.load(model_path)
    else:
        print("  → Note: No trained model provided. Running in inference-only mode.")
        print("         For training, use: python run_pipeline.py train --data <path>")
        # Create untrained model for demonstration
        # In production, you would train first or load a trained model
        raise ValueError("Model must be trained before prediction. Use --train flag first.")
    
    # Prepare feature matrix
    X = np.array([p['features'] for p in pairs])
    
    # Make predictions
    predictions, probabilities = model.predict(X)
    
    # Attach predictions to pairs
    results = []
    for i, pair in enumerate(pairs):
        is_positive = bool(predictions[i])
        pred_score = float(probabilities[i])
        
        results.append({
            'drug_id': pair['drug_id'],
            'target_id': pair['target_id'],
            'prediction_score': pred_score,
            'is_positive': is_positive,
            'confidence_score': pred_score if is_positive else 1 - pred_score,
            'model_used': model_type,
            'interaction_type': 'predicted'
        })
    
    positive_count = sum(1 for r in results if r['is_positive'])
    print(f"  → {positive_count} positive predictions out of {len(results)}")
    
    return results, model


def write_interactions_to_firestore(interactions):
    """Write positive interactions to Firestore"""
    print("\n[5/6] Writing interactions to Firestore...")
    
    positive_interactions = [i for i in interactions if i['is_positive']]
    
    if not positive_interactions:
        print("  → No positive interactions to write")
        return []
    
    # Check for existing interactions to avoid duplicates
    new_interactions = []
    for interaction in positive_interactions:
        existing = firebase_client.get_existing_interactions(
            interaction['drug_id'],
            interaction['target_id']
        )
        if not existing:
            new_interactions.append(interaction)
    
    if not new_interactions:
        print("  → All interactions already exist in Firestore")
        return []
    
    created_ids = firebase_client.batch_create_interactions(new_interactions)
    
    # Add IDs back to interactions for traceability
    for i, created_id in enumerate(created_ids):
        new_interactions[i]['id'] = created_id
    
    print(f"  → Created {len(created_ids)} new interaction documents")
    
    return new_interactions


def derive_and_write_repurposing(positive_interactions, drug_features):
    """Derive repurposing candidates and write to Firestore"""
    print("\n[6/6] Deriving repurposing candidates...")
    
    # Build drug data lookup
    drugs_data = {
        drug_id: info['data'] 
        for drug_id, info in drug_features.items()
    }
    
    # Derive candidates
    candidates = repurposing_deriver.derive_candidates(
        positive_interactions,
        drugs_data
    )
    
    if not candidates:
        print("  → No repurposing candidates derived")
        return []
    
    # Write to Firestore
    created_ids = repurposing_deriver.write_candidates_to_firestore(candidates)
    
    print(f"  → Created {len(created_ids)} repurposing candidates")
    
    return candidates


def run_full_pipeline(model_type='random_forest', model_path=None):
    """Run the complete prediction and repurposing pipeline"""
    start_time = datetime.now()
    print("=" * 60)
    print("MediGraph DTI Prediction Pipeline")
    print(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    try:
        # Step 1: Load data
        drugs, targets = load_data()
        
        # Step 2: Extract features
        drug_features, target_features = extract_features(drugs, targets)
        
        # Step 3: Generate pairs
        pairs = generate_pairs(drug_features, target_features)
        
        # Step 4: Make predictions
        interactions, model = make_predictions(pairs, model_type, model_path)
        
        # Step 5: Write to Firestore
        positive_interactions = write_interactions_to_firestore(interactions)
        
        # Step 6: Derive repurposing
        candidates = derive_and_write_repurposing(positive_interactions, drug_features)
        
        # Summary
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print("\n" + "=" * 60)
        print("Pipeline Complete!")
        print(f"Duration: {duration:.1f} seconds")
        print(f"Drugs processed: {len(drugs)}")
        print(f"Targets processed: {len(targets)}")
        print(f"Pairs evaluated: {len(pairs)}")
        print(f"Positive interactions: {len(positive_interactions)}")
        print(f"Repurposing candidates: {len(candidates)}")
        print("=" * 60)
        
        return {
            'success': True,
            'drugs': len(drugs),
            'targets': len(targets),
            'interactions': len(positive_interactions),
            'candidates': len(candidates),
            'duration': duration
        }
        
    except Exception as e:
        print(f"\n[ERROR] Pipeline failed: {e}")
        return {'success': False, 'error': str(e)}


def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser(description='MediGraph DTI Prediction Pipeline')
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Run command
    run_parser = subparsers.add_parser('run', help='Run the prediction pipeline')
    run_parser.add_argument(
        '--model', 
        choices=['random_forest', 'logistic_regression', 'svm'],
        default='random_forest',
        help='Model type to use'
    )
    run_parser.add_argument(
        '--model-path',
        type=str,
        help='Path to trained model file'
    )
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a new model')
    train_parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='Path to training data CSV'
    )
    train_parser.add_argument(
        '--model',
        choices=['random_forest', 'logistic_regression', 'svm', 'all'],
        default='random_forest',
        help='Model type to train'
    )
    train_parser.add_argument(
        '--output',
        type=str,
        default='./models',
        help='Output directory for trained models'
    )
    
    args = parser.parse_args()
    
    if args.command == 'run':
        result = run_full_pipeline(args.model, args.model_path)
        sys.exit(0 if result['success'] else 1)
    
    elif args.command == 'train':
        print("Training mode - implement with your training data")
        # Training would be implemented here with real labeled data
        print(f"Data path: {args.data}")
        print(f"Model: {args.model}")
        print(f"Output: {args.output}")
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
