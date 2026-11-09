"""
Data cleaning utilities for handling missing values and outliers
"""

import pandas as pd
import numpy as np


def handle_missing_values(df, strategy='mean'):
    """
    Handle missing values in the dataframe
    
    Args:
        df: Input DataFrame
        strategy: 'mean', 'median', 'mode', 'drop', or 'forward_fill'
    
    Returns:
        Cleaned DataFrame
    """
    df_cleaned = df.copy()
    
    if strategy == 'drop':
        df_cleaned = df_cleaned.dropna()
    elif strategy == 'forward_fill':
        df_cleaned = df_cleaned.ffill()
    else:
        for col in df_cleaned.columns:
            if df_cleaned[col].isnull().sum() > 0:
                if df_cleaned[col].dtype in ['int64', 'float64']:
                    if strategy == 'mean':
                        df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mean())
                    elif strategy == 'median':
                        df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())
                else:
                    # For categorical columns, use mode
                    mode_value = df_cleaned[col].mode()[0] if not df_cleaned[col].mode().empty else 'Unknown'
                    df_cleaned[col] = df_cleaned[col].fillna(mode_value)
    
    return df_cleaned


def remove_outliers(df, columns=None, method='iqr'):
    """
    Remove outliers from numerical columns
    
    Args:
        df: Input DataFrame
        columns: List of columns to process (None = all numerical)
        method: 'iqr' or 'zscore'
    
    Returns:
        DataFrame with outliers removed
    """
    df_cleaned = df.copy()
    
    if columns is None:
        columns = df_cleaned.select_dtypes(include=[np.number]).columns.tolist()
    
    if method == 'iqr':
        for col in columns:
            Q1 = df_cleaned[col].quantile(0.25)
            Q3 = df_cleaned[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df_cleaned = df_cleaned[(df_cleaned[col] >= lower_bound) & (df_cleaned[col] <= upper_bound)]
    
    elif method == 'zscore':
        from scipy import stats
        z_scores = np.abs(stats.zscore(df_cleaned[columns]))
        df_cleaned = df_cleaned[(z_scores < 3).all(axis=1)]
    
    return df_cleaned


def get_data_quality_report(df):
    """
    Generate data quality report
    
    Returns:
        Dictionary with quality metrics
    """
    report = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().to_dict(),
        'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
        'duplicate_rows': df.duplicated().sum(),
        'data_types': df.dtypes.to_dict()
    }
    
    return report

