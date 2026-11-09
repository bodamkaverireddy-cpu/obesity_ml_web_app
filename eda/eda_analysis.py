"""
Exploratory Data Analysis computations
"""

import pandas as pd
import numpy as np


def perform_eda(df):
    """
    Perform comprehensive EDA on the dataset
    
    Returns:
        Dictionary containing EDA results
    """
    results = {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'data_types': df.dtypes.astype(str).to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'missing_percentage': (df.isnull().sum() / len(df) * 100).round(2).to_dict(),
        'duplicate_rows': df.duplicated().sum(),
        'memory_usage': df.memory_usage(deep=True).sum() / 1024**2,  # MB
    }
    
    # Numerical statistics
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numerical_cols:
        results['numerical_stats'] = df[numerical_cols].describe().to_dict()
        # Add skewness and kurtosis for numerical columns
        results['skewness'] = {}
        results['outlier_info'] = {}
        for col in numerical_cols:
            results['skewness'][col] = float(df[col].skew())
            # Calculate outliers using IQR method
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
            results['outlier_info'][col] = {
                'count': int(len(outliers)),
                'percentage': float((len(outliers) / len(df)) * 100),
                'lower_bound': float(lower_bound),
                'upper_bound': float(upper_bound)
            }
    
    # Categorical statistics
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    if categorical_cols:
        results['categorical_stats'] = {}
        for col in categorical_cols:
            results['categorical_stats'][col] = {
                'unique_count': df[col].nunique(),
                'top_values': df[col].value_counts().head(10).to_dict(),
                'null_count': df[col].isnull().sum()
            }
    
    # Missing values handling recommendations
    results['missing_handling'] = {}
    for col in df.columns:
        missing_pct = results['missing_percentage'][col]
        if missing_pct > 0:
            if missing_pct < 5:
                recommendation = 'Drop rows with missing values or use mean/median/mode imputation'
            elif missing_pct < 50:
                recommendation = 'Use mean/median/mode imputation or forward/backward fill'
            else:
                recommendation = 'Consider dropping the column or advanced imputation techniques'
            
            if col in numerical_cols:
                imputation_method = 'Mean or Median'
            else:
                imputation_method = 'Mode'
            
            results['missing_handling'][col] = {
                'recommendation': recommendation,
                'imputation_method': imputation_method,
                'missing_percentage': missing_pct
            }
    
    # Correlation matrix (for numerical columns only)
    if len(numerical_cols) > 1:
        corr_matrix = df[numerical_cols].corr()
        results['correlation'] = corr_matrix.to_dict()
        # Find highly correlated pairs
        results['high_correlation'] = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) > 0.7:  # High correlation threshold
                    results['high_correlation'].append({
                        'feature1': corr_matrix.columns[i],
                        'feature2': corr_matrix.columns[j],
                        'correlation': float(corr_value)
                    })
    
    return results

