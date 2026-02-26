# File: MediGraph/ml_pipeline/config.py
# Path: /Users/shweta/Desktop/major project fy/MediGraph/ml_pipeline/config.py

"""
ML Pipeline Configuration
Environment-aware configuration for Firebase and model parameters
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Configuration class for ML pipeline"""
    
    # Firebase Configuration
    FIREBASE_SERVICE_ACCOUNT_PATH = os.getenv('FIREBASE_SERVICE_ACCOUNT_PATH')
    FIREBASE_PROJECT_ID = os.getenv('FIREBASE_PROJECT_ID')
    
    # Firestore Collections
    COLLECTIONS = {
        'drugs': 'drugs',
        'targets': 'targets',
        'interactions': 'drug_target_interactions',
        'diseases': 'diseases',
        'repurposing': 'repurposing_candidates'
    }
    
    # Feature Engineering Parameters
    MORGAN_FP_RADIUS = 2
    MORGAN_FP_BITS = 1024
    AAC_LENGTH = 20  # 20 standard amino acids
    
    # Model Parameters
    MODELS = {
        'random_forest': {
            'n_estimators': 100,
            'max_depth': 10,
            'random_state': 42
        },
        'logistic_regression': {
            'max_iter': 1000,
            'random_state': 42
        },
        'svm': {
            'kernel': 'rbf',
            'probability': True,
            'random_state': 42
        }
    }
    
    # Thresholds
    POSITIVE_PREDICTION_THRESHOLD = 0.5
    HIGH_CONFIDENCE_THRESHOLD = 0.6
    REPURPOSING_SCORE_THRESHOLD = 0.7
    
    # Batch Settings
    FIRESTORE_BATCH_SIZE = 500
    
    @classmethod
    def validate(cls):
        """Validate required configuration"""
        if not cls.FIREBASE_SERVICE_ACCOUNT_PATH:
            raise ValueError("FIREBASE_SERVICE_ACCOUNT_PATH environment variable is required")
        if not os.path.exists(cls.FIREBASE_SERVICE_ACCOUNT_PATH):
            raise FileNotFoundError(f"Service account file not found: {cls.FIREBASE_SERVICE_ACCOUNT_PATH}")
        return True
