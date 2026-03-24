import os
import sys
import pandas as pd
from tdc.multi_pred import DTI
import firebase_admin
from firebase_admin import credentials, firestore
import datetime

def initialize_firebase():
    service_account_path = os.getenv('FIREBASE_SERVICE_ACCOUNT_PATH', 'service-account.json')
    if not os.path.exists(service_account_path):
        print(f"Service account file not found at: {service_account_path}")
        sys.exit(1)
    
    if not firebase_admin._apps:
        cred = credentials.Certificate(service_account_path)
        firebase_admin.initialize_app(cred)
    
    return firestore.client()

def seed_real_data():
    db = initialize_firebase()
    print("Fetching DAVIS dataset...")
    data = DTI(name='DAVIS')
    df = data.get_data()
    
    # Get 20 unique drugs and 20 unique targets
    unique_drugs = df.drop_duplicates(subset=['Drug']).head(25)
    unique_targets = df.drop_duplicates(subset=['Target']).head(25)
    
    print("Clearing old demo data from Firestore...")
    for col in ['drugs', 'targets', 'drug_target_interactions', 'repurposing_candidates', 'diseases']:
        docs = db.collection(col).stream()
        for doc in docs:
            doc.reference.delete()
            
    print(f"Uploading {len(unique_drugs)} real drugs from DAVIS...")
    for i, (_, row) in enumerate(unique_drugs.iterrows()):
        drug_id = f"davis_drug_{i}"
        db.collection('drugs').document(drug_id).set({
            'drug_name': f"DAVIS Compound {i+1}",
            'smiles': row['Drug'],
            'created_at': datetime.datetime.now(datetime.timezone.utc),
            'approved_indication': 'Candidate'
        })
        
    print(f"Uploading {len(unique_targets)} real targets from DAVIS...")
    for i, (_, row) in enumerate(unique_targets.iterrows()):
        target_id = f"davis_target_{i}"
        db.collection('targets').document(target_id).set({
            'target_name': f"DAVIS Kinase {i+1}",
            'protein_sequence': row['Target'],
            'gene_name': f"KIN{i+1}",
            'created_at': datetime.datetime.now(datetime.timezone.utc)
        })
        
    db.collection('diseases').document("davis_disease_1").set({
        'disease_name': "Kinase-related Disorder",
        'category': "Genetic",
        'created_at': datetime.datetime.now(datetime.timezone.utc)
    })
    
    print("Done seeding basic real data!")
    print("Now run `python run_pipeline.py run --model random_forest --model-path models/random_forest_dti.pkl` to generate interactions and repurpose!")

if __name__ == '__main__':
    seed_real_data()
