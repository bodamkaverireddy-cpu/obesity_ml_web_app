"""
Encoding utilities for categorical variables (One-hot and Label encoding)
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import joblib
import os
from config import Config


def encode_categorical_features(df, target_column='Obesity Category', encode_type='label'):
    """
    Encode categorical features in the dataframe
    
    Args:
        df: Input DataFrame
        target_column: Name of target column
        encode_type: 'label' or 'onehot'
    
    Returns:
        Encoded DataFrame and encoders
    """
    df_encoded = df.copy()
    encoders = {}
    
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    
    if target_column in categorical_columns:
        categorical_columns.remove(target_column)
    
    if encode_type == 'label':
        for col in categorical_columns:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            encoders[col] = le
    
    elif encode_type == 'onehot':
        for col in categorical_columns:
            ohe = OneHotEncoder(drop='first', sparse_output=False)
            encoded = ohe.fit_transform(df_encoded[[col]])
            encoded_df = pd.DataFrame(encoded, columns=[f'{col}_{i}' for i in range(encoded.shape[1])])
            df_encoded = pd.concat([df_encoded.drop(columns=[col]), encoded_df], axis=1)
            encoders[col] = ohe
    
    # Encode target column separately
    if target_column in df_encoded.columns:
        le_target = LabelEncoder()
        df_encoded[target_column] = le_target.fit_transform(df_encoded[target_column])
        encoders['target'] = le_target
    
    return df_encoded, encoders


def save_encoders(encoders, filename='encoders.pkl'):
    """Save encoders to file"""
    os.makedirs(Config.MODELS_FOLDER, exist_ok=True)
    filepath = os.path.join(Config.MODELS_FOLDER, filename)
    joblib.dump(encoders, filepath)


def load_encoders(filename='encoders.pkl'):
    """Load encoders from file"""
    filepath = os.path.join(Config.MODELS_FOLDER, filename)
    return joblib.load(filepath)

