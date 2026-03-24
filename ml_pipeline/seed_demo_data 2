# File: MediGraph/ml_pipeline/seed_demo_data.py
# Path: /Users/shweta/Desktop/major project fy/MediGraph/ml_pipeline/seed_demo_data.py
"""
Demo Data Seeder for MediGraph
Seeds the Firestore database with sample drugs, targets, diseases,
interactions, and repurposing candidates for demonstration purposes.
"""

import os
import sys
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import firebase_admin
from firebase_admin import credentials, firestore

# Sample data
SAMPLE_DRUGS = [
    {"id": "drug_001", "name": "Aspirin", "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O", "molecular_weight": 180.16, "approved_indications": ["pain", "inflammation", "fever"]},
    {"id": "drug_002", "name": "Metformin", "smiles": "CN(C)C(=N)NC(=N)N", "molecular_weight": 129.16, "approved_indications": ["type_2_diabetes"]},
    {"id": "drug_003", "name": "Ibuprofen", "smiles": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O", "molecular_weight": 206.29, "approved_indications": ["pain", "inflammation"]},
    {"id": "drug_004", "name": "Atorvastatin", "smiles": "CC(C)C1=C(C(=C(N1CCC(CC(CC(=O)O)O)O)C2=CC=C(C=C2)F)C3=CC=CC=C3)C(=O)NC4=CC=CC=C4", "molecular_weight": 558.64, "approved_indications": ["hypercholesterolemia"]},
    {"id": "drug_005", "name": "Omeprazole", "smiles": "CC1=CN=C(C(=C1OC)C)CS(=O)C2=NC3=C(N2)C=C(C=C3)OC", "molecular_weight": 345.42, "approved_indications": ["gerd", "ulcers"]},
]

SAMPLE_TARGETS = [
    {"id": "target_001", "name": "Cyclooxygenase-2 (COX-2)", "uniprot_id": "P35354", "gene_symbol": "PTGS2", "organism": "Homo sapiens"},
    {"id": "target_002", "name": "AMP-activated protein kinase", "uniprot_id": "Q13131", "gene_symbol": "PRKAA1", "organism": "Homo sapiens"},
    {"id": "target_003", "name": "HMG-CoA reductase", "uniprot_id": "P04035", "gene_symbol": "HMGCR", "organism": "Homo sapiens"},
    {"id": "target_004", "name": "H+/K+-ATPase", "uniprot_id": "P20648", "gene_symbol": "ATP4A", "organism": "Homo sapiens"},
    {"id": "target_005", "name": "Peroxisome proliferator-activated receptor gamma", "uniprot_id": "P37231", "gene_symbol": "PPARG", "organism": "Homo sapiens"},
]

SAMPLE_DISEASES = [
    {"id": "disease_001", "name": "Type 2 Diabetes", "mesh_id": "D003924", "category": "metabolic"},
    {"id": "disease_002", "name": "Alzheimer's Disease", "mesh_id": "D000544", "category": "neurological"},
    {"id": "disease_003", "name": "Cardiovascular Disease", "mesh_id": "D002318", "category": "cardiovascular"},
    {"id": "disease_004", "name": "Rheumatoid Arthritis", "mesh_id": "D001172", "category": "autoimmune"},
    {"id": "disease_005", "name": "Cancer", "mesh_id": "D009369", "category": "oncology"},
]

# Target-Disease associations
TARGET_DISEASE_ASSOCIATIONS = [
    ("target_001", "disease_004"),  # COX-2 - Rheumatoid Arthritis
    ("target_002", "disease_001"),  # AMPK - Type 2 Diabetes
    ("target_003", "disease_003"),  # HMGCR - Cardiovascular Disease
    ("target_005", "disease_001"),  # PPARG - Type 2 Diabetes
    ("target_005", "disease_002"),  # PPARG - Alzheimer's
]

# Sample interactions (drug-target pairs with predictions)
SAMPLE_INTERACTIONS = [
    {"drug_id": "drug_001", "target_id": "target_001", "is_positive": True, "confidence_score": 0.92, "model_type": "random_forest"},
    {"drug_id": "drug_002", "target_id": "target_002", "is_positive": True, "confidence_score": 0.88, "model_type": "random_forest"},
    {"drug_id": "drug_003", "target_id": "target_001", "is_positive": True, "confidence_score": 0.85, "model_type": "random_forest"},
    {"drug_id": "drug_004", "target_id": "target_003", "is_positive": True, "confidence_score": 0.94, "model_type": "random_forest"},
    {"drug_id": "drug_005", "target_id": "target_004", "is_positive": True, "confidence_score": 0.91, "model_type": "random_forest"},
    {"drug_id": "drug_002", "target_id": "target_005", "is_positive": True, "confidence_score": 0.72, "model_type": "random_forest"},  # Repurposing candidate!
    {"drug_id": "drug_001", "target_id": "target_005", "is_positive": True, "confidence_score": 0.68, "model_type": "random_forest"},  # Repurposing candidate!
]


def initialize_firebase():
    """Initialize Firebase Admin SDK"""
    service_account_path = os.getenv('FIREBASE_SERVICE_ACCOUNT_PATH', 'service-account.json')
    
    if not os.path.exists(service_account_path):
        print(f"❌ Service account file not found at: {service_account_path}")
        print("Please place your Firebase service account JSON file here.")
        sys.exit(1)
    
    if not firebase_admin._apps:
        cred = credentials.Certificate(service_account_path)
        firebase_admin.initialize_app(cred)
    
    return firestore.client()


def seed_collection(db, collection_name, data, id_field='id'):
    """Seed a collection with data"""
    print(f"\n📦 Seeding {collection_name}...")
    batch = db.batch()
    
    for item in data:
        doc_id = item.get(id_field, None)
        doc_ref = db.collection(collection_name).document(doc_id) if doc_id else db.collection(collection_name).document()
        
        # Add metadata
        item_with_meta = {
            **item,
            'created_at': datetime.utcnow(),
            'updated_at': datetime.utcnow()
        }
        batch.set(doc_ref, item_with_meta)
        print(f"  ✓ Added: {item.get('name', item.get('id', 'unknown'))}")
    
    batch.commit()
    print(f"✅ {collection_name} seeded with {len(data)} items")


def seed_interactions(db):
    """Seed drug-target interactions"""
    print(f"\n📦 Seeding drug_target_interactions...")
    batch = db.batch()
    
    for interaction in SAMPLE_INTERACTIONS:
        doc_id = f"{interaction['drug_id']}_{interaction['target_id']}"
        doc_ref = db.collection('drug_target_interactions').document(doc_id)
        
        interaction_data = {
            **interaction,
            'predicted_at': datetime.utcnow(),
            'created_at': datetime.utcnow()
        }
        batch.set(doc_ref, interaction_data)
        print(f"  ✓ Added: {interaction['drug_id']} -> {interaction['target_id']} (confidence: {interaction['confidence_score']})")
    
    batch.commit()
    print(f"✅ drug_target_interactions seeded with {len(SAMPLE_INTERACTIONS)} items")


def seed_repurposing_candidates(db):
    """Generate and seed repurposing candidates based on interactions"""
    print(f"\n📦 Generating repurposing candidates...")
    
    # Find interactions that could lead to repurposing
    candidates = []
    
    for interaction in SAMPLE_INTERACTIONS:
        if interaction['confidence_score'] >= 0.6:  # HIGH_CONFIDENCE_THRESHOLD
            # Check if this target is associated with diseases the drug isn't approved for
            drug = next((d for d in SAMPLE_DRUGS if d['id'] == interaction['drug_id']), None)
            
            for target_id, disease_id in TARGET_DISEASE_ASSOCIATIONS:
                if target_id == interaction['target_id']:
                    disease = next((d for d in SAMPLE_DISEASES if d['id'] == disease_id), None)
                    
                    if disease and disease['name'].lower().replace(' ', '_') not in [i.lower() for i in drug.get('approved_indications', [])]:
                        candidate = {
                            'drug_id': interaction['drug_id'],
                            'target_id': interaction['target_id'],
                            'proposed_disease_id': disease_id,
                            'repurposing_score': interaction['confidence_score'] * 0.9,  # Slightly lower than confidence
                            'pathway': [f"{interaction['drug_id']}_{interaction['target_id']}"],
                            'rationale': f"{drug['name']} may treat {disease['name']} via {next((t['name'] for t in SAMPLE_TARGETS if t['id'] == target_id), 'unknown target')}",
                            'status': 'pending',
                            'created_at': datetime.utcnow()
                        }
                        candidates.append(candidate)
    
    if candidates:
        batch = db.batch()
        for i, candidate in enumerate(candidates):
            doc_id = f"candidate_{i+1:03d}"
            doc_ref = db.collection('repurposing_candidates').document(doc_id)
            batch.set(doc_ref, candidate)
            print(f"  ✓ Candidate: {candidate['drug_id']} for {candidate['proposed_disease_id']} (score: {candidate['repurposing_score']:.2f})")
        
        batch.commit()
        print(f"✅ repurposing_candidates seeded with {len(candidates)} items")
    else:
        print("⚠️ No repurposing candidates generated")


def main():
    print("=" * 60)
    print("🧬 MediGraph Demo Data Seeder")
    print("=" * 60)
    
    # Initialize Firebase
    db = initialize_firebase()
    print("✅ Firebase initialized successfully")
    
    # Seed collections
    seed_collection(db, 'drugs', SAMPLE_DRUGS)
    seed_collection(db, 'targets', SAMPLE_TARGETS)
    seed_collection(db, 'diseases', SAMPLE_DISEASES)
    seed_interactions(db)
    seed_repurposing_candidates(db)
    
    print("\n" + "=" * 60)
    print("🎉 Demo data seeding complete!")
    print("=" * 60)
    print("\nYou can now:")
    print("1. Refresh the MediGraph app at http://localhost:3000")
    print("2. Log in and explore the Graph View")
    print("3. Check the Repurposing Candidates section")


if __name__ == "__main__":
    main()
