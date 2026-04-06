import os
import json
import logging
from pathlib import Path
import numpy as np
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from feature_extraction import FeatureExtractor
from model_training import DTIModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

def get_meaningful_feature_names():
    """
    Returns 1044 labels for DTI models:
    - 1024 Morgan fingerprint bits
    - 20 Amino Acid letters
    """
    fp_bits = [f"fp_bit_{i}" for i in range(1024)]
    aa_letters = list("ACDEFGHIKLMNPQRSTVWY")
    return fp_bits + aa_letters


def generate_global_shap_plots(model_input, X_test, output_dir: Path, json_path: Path = None, is_dti=True):
    """
    Computes SHAP values, generates Beeswarm, Bar, and Waterfall plots,
    and optionally returns top 20 features to append to metrics JSON.
    """
    log.info("Computing SHAP values for test set...")
    
    # Extract native sklearn model
    rf_model = getattr(model_input, 'model', model_input)
    explainer = shap.TreeExplainer(rf_model)
    
    # Test set might be a pandas df or numpy array. Ensure numpy array.
    if hasattr(X_test, 'values'):
        X_test_arr = X_test.values
    else:
        X_test_arr = X_test

    shap_values = explainer.shap_values(X_test_arr)
    
    # In scikit-learn random forests, shap_values is a list of arrays [negative_class, positive_class]
    if isinstance(shap_values, list):
        shap_pos = shap_values[1]
        expected_value = explainer.expected_value[1]
    else:
        shap_pos = shap_values
        expected_value = explainer.expected_value

    feature_names = get_meaningful_feature_names() if is_dti and X_test_arr.shape[1] == 1044 else [f"Feature {i}" for i in range(X_test_arr.shape[1])]

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Summary Plot (Beeswarm)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_pos, X_test_arr, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(output_dir / "shap_summary.png", dpi=150)
    plt.close()

    # 2. Bar Plot (Global feature importance)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_pos, X_test_arr, feature_names=feature_names, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(output_dir / "shap_bar.png", dpi=150)
    plt.close()

    # 3. Waterfall Plot for highest confidence positive prediction
    probs = rf_model.predict_proba(X_test_arr)[:, 1]
    best_idx = np.argmax(probs)
    
    # Create an Explanation object for the single prediction
    exp = shap.Explanation(
        values=shap_pos[best_idx], 
        base_values=expected_value, 
        data=X_test_arr[best_idx], 
        feature_names=feature_names
    )
    
    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(exp, show=False)
    plt.tight_layout()
    plt.savefig(output_dir / "shap_waterfall.png", dpi=150)
    plt.close()

    # Calculate mean absolute SHAP values for top 20
    mean_abs_shap = np.abs(shap_pos).mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[::-1][:20]
    
    top_features = {}
    for idx in top_indices:
        top_features[feature_names[idx]] = float(mean_abs_shap[idx])

    if json_path and json_path.exists():
        with open(json_path, 'r') as f:
            metrics = json.load(f)
        metrics["shap_top_features"] = top_features
        with open(json_path, 'w') as f:
            json.dump(metrics, f, indent=2)

    log.info("SHAP plots saved successfully.")
    return top_features


def explain_single_prediction(drug_smiles, target_sequence, model_path):
    """
    Takes a raw SMILES + protein sequence, extracts features, runs SHAP,
    and returns a structured dictionary of the top 5 contributing features.
    """
    # 1. Load model
    dti_model = DTIModel.load(model_path)
    rf_model = dti_model.model
    
    # 2. Extract features
    d_feat = FeatureExtractor.compute_morgan_fingerprint(drug_smiles)
    t_feat = FeatureExtractor.compute_amino_acid_composition(target_sequence)
    
    if d_feat is None or t_feat is None:
        raise ValueError("Failed to extract features from provided SMILES or Sequence.")
        
    pair_feat = FeatureExtractor.create_pair_features(d_feat, t_feat)
    pair_2d = np.array([pair_feat])
    
    # 3. Predict
    prob = float(rf_model.predict_proba(pair_2d)[0, 1])
    pred = int(rf_model.predict(pair_2d)[0])
    
    # 4. Explain
    explainer = shap.TreeExplainer(rf_model)
    shap_vals = explainer.shap_values(pair_2d)
    
    if isinstance(shap_vals, list):
        shap_pos = shap_vals[1][0]
    else:
        shap_pos = shap_vals[0]
        
    feature_names = get_meaningful_feature_names()
    if len(feature_names) != len(shap_pos):
        feature_names = [f"f_{i}" for i in range(len(shap_pos))]
    
    # 5. Get top 5 features by absolute value
    abs_shap = np.abs(shap_pos)
    top_indices = np.argsort(abs_shap)[::-1][:5]
    
    top_5 = []
    for idx in top_indices:
        val = shap_pos[idx]
        top_5.append({
            "name": feature_names[idx],
            "shap_value": float(val),
            "direction": "increases" if val > 0 else "decreases"
        })
        
    return {
        "prediction": pred,
        "confidence": prob,
        "top_5_features": top_5
    }
