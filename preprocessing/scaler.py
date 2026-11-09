"""
Feature scaling utilities (StandardScaler, MinMaxScaler, etc.)
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib
import os
from config import Config


def scale_features(X, scaler_type='standard', fit=True, scaler=None):
    """
    Scale numerical features
    
    Args:
        X: Features DataFrame
        scaler_type: 'standard' or 'minmax'
        fit: Whether to fit the scaler
        scaler: Pre-fitted scaler (if not fitting)
    
    Returns:
        Scaled features and scaler object
    """
    if scaler is None:
        if scaler_type == 'standard':
            scaler = StandardScaler()
        elif scaler_type == 'minmax':
            scaler = MinMaxScaler()
    
    if fit:
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)
    
    # Convert back to DataFrame with original column names
    if isinstance(X, pd.DataFrame):
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    
    return X_scaled, scaler


def save_scaler(scaler, filename='scaler.pkl'):
    """Save scaler to file"""
    os.makedirs(Config.MODELS_FOLDER, exist_ok=True)
    filepath = os.path.join(Config.MODELS_FOLDER, filename)
    joblib.dump(scaler, filepath)


def load_scaler(filename='scaler.pkl'):
    """Load scaler from file"""
    filepath = os.path.join(Config.MODELS_FOLDER, filename)
    return joblib.load(filepath)

