"""
Feature importance analysis using SHAP and built-in importance methods
"""

import pandas as pd
import numpy as np
import joblib
import os
from config import Config
from preprocessing.encoder import encode_categorical_features
from preprocessing.scaler import scale_features, load_scaler

def get_feature_importance(df, target_column='Obesity Category', method='tree'):
    """
    Get feature importance scores
    
    Args:
        df: Input DataFrame
        target_column: Name of target column
        method: 'tree' (for tree-based models) or 'shap'
    
    Returns:
        Dictionary of feature importance scores
    """
    try:
        # Prepare data (use onehot to match training pipeline and saved scaler/model)
        df_encoded, encoders = encode_categorical_features(df, target_column, encode_type='onehot')
        X = df_encoded.drop(columns=[target_column])
        y = df_encoded[target_column]
        
        # Load scaler (fit on one-hot encoded features during training)
        scaler = load_scaler()
        # Ensure feature order matches scaler
        if hasattr(scaler, 'feature_names_in_'):
            for c in scaler.feature_names_in_:
                if c not in X.columns:
                    X[c] = 0
            X = X[scaler.feature_names_in_]
        X_scaled, _ = scale_features(X, scaler_type='standard', fit=False, scaler=scaler)
        
        if method == 'tree':
            # Use XGBoost for feature importance
            model_path = os.path.join(Config.MODELS_FOLDER, 'xgboost_model.pkl')
            if os.path.exists(model_path):
                model = joblib.load(model_path)
                importance = model.feature_importances_
                feature_names = X.columns.tolist()
                
                # Create importance dictionary
                importance_dict = dict(zip(feature_names, importance.tolist()))
                # Sort by importance
                importance_dict = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
                return importance_dict
        
        elif method == 'shap':
            try:
                import shap
                model_path = os.path.join(Config.MODELS_FOLDER, 'xgboost_model.pkl')
                if os.path.exists(model_path):
                    model = joblib.load(model_path)
                    
                    # Use TreeExplainer for XGBoost
                    explainer = shap.TreeExplainer(model)
                    X_sample = X_scaled.sample(min(100, len(X_scaled)))
                    shap_values = explainer.shap_values(X_sample)
                    
                    # Get mean absolute SHAP values
                    if isinstance(shap_values, list):
                        shap_values = np.abs(shap_values).mean(0)
                    else:
                        shap_values = np.abs(shap_values).mean(0)
                    
                    feature_names = X.columns.tolist()
                    importance_dict = dict(zip(feature_names, shap_values.tolist()))
                    importance_dict = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
                    return importance_dict
            except ImportError:
                return {'error': 'SHAP library not available'}
        
        return {'error': 'No feature importance available'}
    except Exception as e:
        return {'error': str(e)}

