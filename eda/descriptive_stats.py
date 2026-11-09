"""
Descriptive statistics and statistical tests
"""

import pandas as pd
import numpy as np
from scipy import stats


def get_descriptive_stats(df):
    """
    Get descriptive statistics for the dataset
    
    Returns:
        Dictionary with statistical summaries
    """
    stats_dict = {}
    
    # Overall statistics
    stats_dict['total_samples'] = len(df)
    stats_dict['total_features'] = len(df.columns)
    stats_dict['numerical_features'] = len(df.select_dtypes(include=[np.number]).columns)
    stats_dict['categorical_features'] = len(df.select_dtypes(include=['object']).columns)
    
    # Numerical column statistics
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numerical_cols:
        stats_dict['numerical_summary'] = {}
        for col in numerical_cols:
            stats_dict['numerical_summary'][col] = {
                'mean': float(df[col].mean()),
                'median': float(df[col].median()),
                'std': float(df[col].std()),
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'skewness': float(df[col].skew()),
                'kurtosis': float(df[col].kurtosis()),
                'variance': float(df[col].var())
            }
    
    # Categorical column statistics
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    if categorical_cols:
        stats_dict['categorical_summary'] = {}
        for col in categorical_cols:
            value_counts = df[col].value_counts()
            stats_dict['categorical_summary'][col] = {
                'unique_count': int(df[col].nunique()),
                'mode': str(df[col].mode()[0]) if not df[col].mode().empty else None,
                'mode_frequency': int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                'distribution': value_counts.head(10).to_dict()
            }
    
    # Check for normal distribution in numerical columns
    if numerical_cols:
        stats_dict['normality_test'] = {}
        for col in numerical_cols[:5]:  # Limit to first 5 for performance
            try:
                stat, p_value = stats.normaltest(df[col].dropna())
                stats_dict['normality_test'][col] = {
                    'statistic': float(stat),
                    'p_value': float(p_value),
                    'is_normal': p_value > 0.05
                }
            except:
                pass
    
    return stats_dict

