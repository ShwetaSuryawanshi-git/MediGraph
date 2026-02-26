# File: MediGraph/ml_pipeline/firebase_client.py
# Path: /Users/shweta/Desktop/major project fy/MediGraph/ml_pipeline/firebase_client.py

"""
Firebase Admin SDK Client
Handles Firestore read/write operations for the ML pipeline
"""

import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime
from config import Config


class FirebaseClient:
    """Firebase client for ML pipeline operations"""
    
    _instance = None
    _db = None
    
    def __new__(cls):
        """Singleton pattern to reuse Firebase connection"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._initialize()
        return cls._instance
    
    @classmethod
    def _initialize(cls):
        """Initialize Firebase Admin SDK"""
        Config.validate()
        cred = credentials.Certificate(Config.FIREBASE_SERVICE_ACCOUNT_PATH)
        firebase_admin.initialize_app(cred)
        cls._db = firestore.client()
    
    @property
    def db(self):
        """Get Firestore client"""
        return self._db
    
    # =========================================================================
    # READ OPERATIONS
    # =========================================================================
    
    def get_all_drugs(self):
        """Fetch all drugs from Firestore"""
        docs = self._db.collection(Config.COLLECTIONS['drugs']).stream()
        return [{'id': doc.id, **doc.to_dict()} for doc in docs]
    
    def get_all_targets(self):
        """Fetch all targets from Firestore"""
        docs = self._db.collection(Config.COLLECTIONS['targets']).stream()
        return [{'id': doc.id, **doc.to_dict()} for doc in docs]
    
    def get_all_diseases(self):
        """Fetch all diseases from Firestore"""
        docs = self._db.collection(Config.COLLECTIONS['diseases']).stream()
        return [{'id': doc.id, **doc.to_dict()} for doc in docs]
    
    def get_existing_interactions(self, drug_id, target_id):
        """Check for existing interactions between a drug-target pair"""
        query = (
            self._db.collection(Config.COLLECTIONS['interactions'])
            .where('drug_id', '==', drug_id)
            .where('target_id', '==', target_id)
        )
        return [{'id': doc.id, **doc.to_dict()} for doc in query.stream()]
    
    # =========================================================================
    # WRITE OPERATIONS
    # =========================================================================
    
    def create_drug(self, drug_data):
        """Create a new drug document"""
        drug_data['created_at'] = firestore.SERVER_TIMESTAMP
        drug_data['updated_at'] = firestore.SERVER_TIMESTAMP
        doc_ref = self._db.collection(Config.COLLECTIONS['drugs']).add(drug_data)
        return doc_ref[1].id
    
    def create_target(self, target_data):
        """Create a new target document"""
        target_data['created_at'] = firestore.SERVER_TIMESTAMP
        target_data['updated_at'] = firestore.SERVER_TIMESTAMP
        doc_ref = self._db.collection(Config.COLLECTIONS['targets']).add(target_data)
        return doc_ref[1].id
    
    def create_interaction(self, interaction_data):
        """Create a new drug-target interaction document"""
        interaction_data['created_at'] = firestore.SERVER_TIMESTAMP
        doc_ref = self._db.collection(Config.COLLECTIONS['interactions']).add(interaction_data)
        return doc_ref[1].id
    
    def create_repurposing_candidate(self, candidate_data):
        """Create a new repurposing candidate document"""
        candidate_data['created_at'] = firestore.SERVER_TIMESTAMP
        candidate_data['status'] = 'pending'
        doc_ref = self._db.collection(Config.COLLECTIONS['repurposing']).add(candidate_data)
        return doc_ref[1].id
    
    def update_target_disease_association(self, target_id, disease_id):
        """Add a disease association to a target"""
        target_ref = self._db.collection(Config.COLLECTIONS['targets']).document(target_id)
        target_ref.update({
            'associated_disease_ids': firestore.ArrayUnion([disease_id]),
            'updated_at': firestore.SERVER_TIMESTAMP
        })
    
    def update_disease_target_association(self, disease_id, target_id):
        """Add a target association to a disease"""
        disease_ref = self._db.collection(Config.COLLECTIONS['diseases']).document(disease_id)
        disease_ref.update({
            'associated_target_ids': firestore.ArrayUnion([target_id]),
            'updated_at': firestore.SERVER_TIMESTAMP
        })
    
    # =========================================================================
    # BATCH OPERATIONS
    # =========================================================================
    
    def batch_create_interactions(self, interactions_list):
        """Create multiple interactions in batches"""
        created_ids = []
        batch = self._db.batch()
        batch_count = 0
        
        for interaction in interactions_list:
            interaction['created_at'] = firestore.SERVER_TIMESTAMP
            doc_ref = self._db.collection(Config.COLLECTIONS['interactions']).document()
            batch.set(doc_ref, interaction)
            created_ids.append(doc_ref.id)
            batch_count += 1
            
            if batch_count >= Config.FIRESTORE_BATCH_SIZE:
                batch.commit()
                batch = self._db.batch()
                batch_count = 0
        
        if batch_count > 0:
            batch.commit()
        
        return created_ids
    
    def batch_create_repurposing_candidates(self, candidates_list):
        """Create multiple repurposing candidates in batches"""
        created_ids = []
        batch = self._db.batch()
        batch_count = 0
        
        for candidate in candidates_list:
            candidate['created_at'] = firestore.SERVER_TIMESTAMP
            candidate['status'] = 'pending'
            doc_ref = self._db.collection(Config.COLLECTIONS['repurposing']).document()
            batch.set(doc_ref, candidate)
            created_ids.append(doc_ref.id)
            batch_count += 1
            
            if batch_count >= Config.FIRESTORE_BATCH_SIZE:
                batch.commit()
                batch = self._db.batch()
                batch_count = 0
        
        if batch_count > 0:
            batch.commit()
        
        return created_ids


# Export singleton instance
firebase_client = FirebaseClient()
