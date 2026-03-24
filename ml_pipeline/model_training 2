# File: MediGraph/ml_pipeline/model_training.py
# Path: /Users/shweta/Desktop/major project fy/MediGraph/ml_pipeline/model_training.py

"""
Model Training Module
Classical ML models for Drug-Target Interaction prediction
"""

import numpy as np
import pickle
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from config import Config


class DTIModel:
    """Drug-Target Interaction prediction model"""
    
    SUPPORTED_MODELS = ['random_forest', 'logistic_regression', 'svm']
    
    def __init__(self, model_type='random_forest'):
        """
        Initialize DTI model
        
        Args:
            model_type: One of 'random_forest', 'logistic_regression', 'svm'
        """
        if model_type not in self.SUPPORTED_MODELS:
            raise ValueError(f"Model type must be one of {self.SUPPORTED_MODELS}")
        
        self.model_type = model_type
        self.model = self._create_model(model_type)
        self.is_trained = False
        self.metrics = {}
    
    def _create_model(self, model_type):
        """Create sklearn model instance"""
        params = Config.MODELS.get(model_type, {})
        
        if model_type == 'random_forest':
            return RandomForestClassifier(**params)
        elif model_type == 'logistic_regression':
            return LogisticRegression(**params)
        elif model_type == 'svm':
            return SVC(**params)
    
    def train(self, X, y, test_size=0.2):
        """
        Train the model
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Labels (n_samples,)
            test_size: Fraction for test split
            
        Returns:
            Dict of evaluation metrics
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1] if hasattr(self.model, 'predict_proba') else y_pred
        
        self.metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_prob)
        }
        
        return self.metrics
    
    def cross_validate(self, X, y, cv=5):
        """
        Perform cross-validation
        
        Args:
            X: Feature matrix
            y: Labels
            cv: Number of folds
            
        Returns:
            Dict with mean and std of scores
        """
        scores = cross_val_score(self.model, X, y, cv=cv, scoring='roc_auc')
        return {
            'mean_roc_auc': scores.mean(),
            'std_roc_auc': scores.std()
        }
    
    def predict(self, X):
        """
        Make predictions
        
        Args:
            X: Feature matrix (n_samples, n_features)
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)[:, 1] if hasattr(self.model, 'predict_proba') else predictions
        
        return predictions, probabilities
    
    def predict_single(self, features):
        """
        Predict for a single sample
        
        Args:
            features: Feature vector (n_features,)
            
        Returns:
            Tuple of (prediction, probability)
        """
        X = features.reshape(1, -1)
        predictions, probabilities = self.predict(X)
        return predictions[0], probabilities[0]
    
    def save(self, path):
        """Save model to file"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'model_type': self.model_type,
                'is_trained': self.is_trained,
                'metrics': self.metrics
            }, f)
    
    @classmethod
    def load(cls, path):
        """Load model from file"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        instance = cls(model_type=data['model_type'])
        instance.model = data['model']
        instance.is_trained = data['is_trained']
        instance.metrics = data['metrics']
        
        return instance


class ModelEnsemble:
    """Ensemble of DTI models for robust predictions"""
    
    def __init__(self):
        """Initialize ensemble with all supported models"""
        self.models = {
            name: DTIModel(name) for name in DTIModel.SUPPORTED_MODELS
        }
        self.weights = {name: 1.0 for name in DTIModel.SUPPORTED_MODELS}
    
    def train_all(self, X, y):
        """Train all models in ensemble"""
        results = {}
        for name, model in self.models.items():
            metrics = model.train(X, y)
            results[name] = metrics
            # Weight by ROC-AUC
            self.weights[name] = metrics['roc_auc']
        
        # Normalize weights
        total = sum(self.weights.values())
        for name in self.weights:
            self.weights[name] /= total
        
        return results
    
    def predict_ensemble(self, X):
        """
        Make weighted ensemble predictions
        
        Returns:
            Array of weighted average probabilities
        """
        weighted_probs = np.zeros(len(X))
        
        for name, model in self.models.items():
            _, probs = model.predict(X)
            weighted_probs += self.weights[name] * probs
        
        return weighted_probs
