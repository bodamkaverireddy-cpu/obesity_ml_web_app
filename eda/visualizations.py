"""
Data visualization functions for EDA
"""

import pandas as pd
import numpy as np
import base64
import io
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)


def fig_to_base64(fig):
    """Convert matplotlib figure to base64 string"""
    img = io.BytesIO()
    fig.savefig(img, format='png', bbox_inches='tight', dpi=100)
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()


def extract_insights_for_chart(chart_type, chart_name, df, **kwargs):
    """Extract insights and hidden facts from data for each chart"""
    insights = []
    
    if chart_type == 'target_distribution':
        target_col = kwargs.get('target_col', 'Obesity Category')
        if target_col in df.columns:
            value_counts = df[target_col].value_counts()
            total = len(df)
            most_common = value_counts.index[0]
            most_common_pct = (value_counts.iloc[0] / total) * 100
            least_common = value_counts.index[-1]
            least_common_pct = (value_counts.iloc[-1] / total) * 100
            
            insights.append(f"📊 <strong>Dominant Category:</strong> '{most_common}' represents {most_common_pct:.1f}% of the dataset")
            insights.append(f"📊 <strong>Minority Category:</strong> '{least_common}' is the least common at {least_common_pct:.1f}%")
            if most_common_pct > 50:
                insights.append("⚠️ <strong>Class Imbalance Detected:</strong> The dataset is imbalanced, which may affect model performance")
            insights.append(f"📈 <strong>Total Categories:</strong> {len(value_counts)} distinct obesity categories found")
    
    elif chart_type == 'pie_chart':
        col = kwargs.get('column')
        if col and col in df.columns:
            value_counts = df[col].value_counts()
            total = len(df)
            top_category = value_counts.index[0]
            top_pct = (value_counts.iloc[0] / total) * 100
            insights.append(f"🎯 <strong>Largest Segment:</strong> '{top_category}' accounts for {top_pct:.1f}% of the data")
            if top_pct > 70:
                insights.append("⚠️ <strong>High Concentration:</strong> One category dominates the distribution significantly")
    
    elif chart_type == 'bar_chart':
        col = kwargs.get('column')
        if col and col in df.columns:
            value_counts = df[col].value_counts()
            insights.append(f"📊 <strong>Most Frequent:</strong> '{value_counts.index[0]}' appears {value_counts.iloc[0]} times")
            insights.append(f"📊 <strong>Least Frequent:</strong> '{value_counts.index[-1]}' appears {value_counts.iloc[-1]} times")
            if value_counts.iloc[0] > value_counts.iloc[-1] * 3:
                insights.append("⚠️ <strong>Uneven Distribution:</strong> Significant variation in category frequencies")
    
    elif chart_type == 'histogram' or chart_type == 'distribution':
        col = kwargs.get('column')
        if col and col in df.columns:
            data = df[col].dropna()
            mean_val = data.mean()
            median_val = data.median()
            std_val = data.std()
            skewness = data.skew()
            
            insights.append(f"📈 <strong>Mean:</strong> {mean_val:.2f} | <strong>Median:</strong> {median_val:.2f}")
            insights.append(f"📊 <strong>Standard Deviation:</strong> {std_val:.2f}")
            if abs(mean_val - median_val) > std_val * 0.5:
                insights.append("⚠️ <strong>Skewness Detected:</strong> Mean and median differ significantly, indicating distribution skew")
            if skewness > 0.5:
                insights.append(f"↗️ <strong>Right Skewed:</strong> Skewness = {skewness:.2f} (tail extends to the right)")
            elif skewness < -0.5:
                insights.append(f"↖️ <strong>Left Skewed:</strong> Skewness = {skewness:.2f} (tail extends to the left)")
            else:
                insights.append(f"✓ <strong>Normal Distribution:</strong> Skewness = {skewness:.2f} (approximately symmetric)")
    
    elif chart_type == 'boxplot':
        col = kwargs.get('column')
        if col and col in df.columns:
            data = df[col].dropna()
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = data[(data < lower_bound) | (data > upper_bound)]
            outlier_pct = (len(outliers) / len(data)) * 100
            
            insights.append(f"📦 <strong>Quartiles:</strong> Q1={Q1:.2f}, Q3={Q3:.2f}, IQR={IQR:.2f}")
            insights.append(f"📊 <strong>Median:</strong> {data.median():.2f}")
            insights.append(f"🎯 <strong>Normal Range:</strong> {lower_bound:.2f} to {upper_bound:.2f}")
            if len(outliers) > 0:
                insights.append(f"⚠️ <strong>Outliers Detected:</strong> {len(outliers)} outliers ({outlier_pct:.1f}% of data) outside normal range")
                insights.append(f"🔍 <strong>Outlier Range:</strong> Values < {lower_bound:.2f} or > {upper_bound:.2f}")
            else:
                insights.append("✓ <strong>No Significant Outliers:</strong> Data points are within expected range")
    
    elif chart_type == 'scatter':
        x_col = kwargs.get('x_column')
        y_col = kwargs.get('y_column')
        if x_col and y_col and x_col in df.columns and y_col in df.columns:
            correlation = df[x_col].corr(df[y_col])
            insights.append(f"🔗 <strong>Correlation:</strong> {correlation:.3f} between {x_col} and {y_col}")
            if abs(correlation) > 0.7:
                insights.append(f"💪 <strong>Strong {'Positive' if correlation > 0 else 'Negative'} Relationship:</strong> High correlation suggests strong linear relationship")
            elif abs(correlation) > 0.4:
                insights.append(f"📊 <strong>Moderate {'Positive' if correlation > 0 else 'Negative'} Relationship:</strong> Moderate correlation detected")
            else:
                insights.append("📉 <strong>Weak Relationship:</strong> Low correlation suggests weak linear relationship")
    
    elif chart_type == 'correlation_heatmap':
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numerical_cols) > 1:
            corr_matrix = df[numerical_cols].corr()
            # Find strongest correlations
            max_corr = 0
            max_pair = None
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = abs(corr_matrix.iloc[i, j])
                    if corr_val > max_corr:
                        max_corr = corr_val
                        max_pair = (corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j])
            
            if max_pair:
                insights.append(f"🔗 <strong>Strongest Correlation:</strong> {max_pair[0]} ↔ {max_pair[1]} ({max_pair[2]:.3f})")
            insights.append(f"📊 <strong>Features Analyzed:</strong> {len(numerical_cols)} numerical features")
            # Count high correlations
            high_corr_count = sum(1 for i in range(len(corr_matrix.columns)) 
                                for j in range(i+1, len(corr_matrix.columns)) 
                                if abs(corr_matrix.iloc[i, j]) > 0.7)
            if high_corr_count > 0:
                insights.append(f"⚠️ <strong>Multicollinearity Warning:</strong> {high_corr_count} highly correlated pairs detected (|r| > 0.7)")
    
    elif chart_type == 'line_chart':
        col = kwargs.get('column')
        if col and col in df.columns:
            data = df[col].dropna()
            if len(data) > 1:
                trend = 'Increasing' if data.iloc[-1] > data.iloc[0] else 'Decreasing'
                change_pct = ((data.iloc[-1] - data.iloc[0]) / data.iloc[0]) * 100 if data.iloc[0] != 0 else 0
                insights.append(f"📈 <strong>Overall Trend:</strong> {trend} ({change_pct:.1f}% change)")
                insights.append(f"📊 <strong>Range:</strong> Min = {data.min():.2f}, Max = {data.max():.2f}")
    
    return insights


def generate_comprehensive_visualizations(df):
    """
    Generate comprehensive visualizations with insights for the dataset
    Returns charts and insights dictionary
    """
    charts_data = {}
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # 1. Target Distribution (Bar Chart)
    target_col = 'Obesity Category'
    if target_col in df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        value_counts = df[target_col].value_counts()
        bars = ax.bar(range(len(value_counts)), value_counts.values, color='steelblue', alpha=0.8, edgecolor='black')
        ax.set_title('Distribution of Obesity Categories', fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel('Obesity Category', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_xticks(range(len(value_counts)))
        ax.set_xticklabels(value_counts.index, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        for i, (idx, val) in enumerate(value_counts.items()):
            ax.text(i, val, str(val), ha='center', va='bottom', fontweight='bold')
        plt.tight_layout()
        charts_data['target_distribution'] = {
            'chart': fig_to_base64(fig),
            'type': 'target_distribution',
            'title': 'Obesity Category Distribution',
            'insights': extract_insights_for_chart('target_distribution', 'target_distribution', df, target_col=target_col)
        }
        plt.close(fig)
    
    # 2. Pie Chart for Categorical Features (if applicable)
    for col in categorical_cols[:3]:  # Limit to first 3 categorical columns
        if df[col].nunique() <= 10:  # Only for columns with limited unique values
            fig, ax = plt.subplots(figsize=(10, 8))
            value_counts = df[col].value_counts().head(10)
            colors = plt.cm.Set3(range(len(value_counts)))
            wedges, texts, autotexts = ax.pie(value_counts.values, labels=value_counts.index, autopct='%1.1f%%',
                                             startangle=90, colors=colors, textprops={'fontsize': 10})
            ax.set_title(f'{col} Distribution', fontsize=14, fontweight='bold', pad=20)
            plt.setp(autotexts, size=10, weight='bold', color='white')
            plt.tight_layout()
            charts_data[f'{col}_pie'] = {
                'chart': fig_to_base64(fig),
                'type': 'pie_chart',
                'title': f'{col} Distribution (Pie Chart)',
                'insights': extract_insights_for_chart('pie_chart', f'{col}_pie', df, column=col)
            }
            plt.close(fig)
    
    # 3. Histograms for Numerical Features
    for col in numerical_cols[:5]:  # Limit to first 5 numerical columns
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            data = df[col].dropna()
            n, bins, patches = ax.hist(data, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
            mean_val = data.mean()
            median_val = data.median()
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
            ax.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.2f}')
            ax.set_title(f'{col} Distribution', fontsize=14, fontweight='bold', pad=15)
            ax.set_xlabel(col, fontsize=12)
            ax.set_ylabel('Frequency', fontsize=12)
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3, linestyle='--')
            plt.tight_layout()
            charts_data[f'{col}_histogram'] = {
                'chart': fig_to_base64(fig),
                'type': 'histogram',
                'title': f'{col} Distribution (Histogram)',
                'insights': extract_insights_for_chart('histogram', f'{col}_histogram', df, column=col)
            }
            plt.close(fig)
        except:
            continue
    
    # 4. Box Plots for Numerical Features
    for col in numerical_cols[:5]:
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            data = df[col].dropna()
            bp = ax.boxplot(data, vert=True, patch_artist=True,
                          boxprops=dict(facecolor='lightblue', alpha=0.7),
                          medianprops=dict(color='red', linewidth=2),
                          whiskerprops=dict(color='black', linewidth=1.5),
                          capprops=dict(color='black', linewidth=1.5))
            ax.set_title(f'{col} Box Plot (Outlier Detection)', fontsize=14, fontweight='bold', pad=15)
            ax.set_ylabel(col, fontsize=12)
            ax.grid(True, alpha=0.3, linestyle='--', axis='y')
            plt.tight_layout()
            charts_data[f'{col}_boxplot'] = {
                'chart': fig_to_base64(fig),
                'type': 'boxplot',
                'title': f'{col} Box Plot',
                'insights': extract_insights_for_chart('boxplot', f'{col}_boxplot', df, column=col)
            }
            plt.close(fig)
        except:
            continue
    
    # 5. Scatter Plots for Numerical Feature Pairs
    if len(numerical_cols) >= 2:
        for i in range(min(3, len(numerical_cols))):
            for j in range(i+1, min(i+2, len(numerical_cols))):
                try:
                    col1, col2 = numerical_cols[i], numerical_cols[j]
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.scatter(df[col1], df[col2], alpha=0.6, color='coral', edgecolors='black', linewidth=0.5)
                    ax.set_title(f'{col1} vs {col2}', fontsize=14, fontweight='bold', pad=15)
                    ax.set_xlabel(col1, fontsize=12)
                    ax.set_ylabel(col2, fontsize=12)
                    ax.grid(True, alpha=0.3)
                    plt.tight_layout()
                    charts_data[f'{col1}_vs_{col2}_scatter'] = {
                        'chart': fig_to_base64(fig),
                        'type': 'scatter',
                        'title': f'{col1} vs {col2} (Scatter Plot)',
                        'insights': extract_insights_for_chart('scatter', f'{col1}_vs_{col2}_scatter', df, 
                                                               x_column=col1, y_column=col2)
                    }
                    plt.close(fig)
                    break  # Only one scatter per first column
                except:
                    continue
    
    # 6. Correlation Heatmap
    if len(numerical_cols) > 1:
        fig, ax = plt.subplots(figsize=(12, 10))
        corr = df[numerical_cols].corr()
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                   square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax, vmin=-1, vmax=1)
        ax.set_title('Correlation Heatmap - Numerical Features', fontsize=14, fontweight='bold', pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        charts_data['correlation_heatmap'] = {
            'chart': fig_to_base64(fig),
            'type': 'correlation_heatmap',
            'title': 'Feature Correlation Heatmap',
            'insights': extract_insights_for_chart('correlation_heatmap', 'correlation_heatmap', df)
        }
        plt.close(fig)
    
    # 7. Bar Charts for Top Values in Categorical Features
    for col in categorical_cols[:3]:
        try:
            if df[col].nunique() > 1:
                fig, ax = plt.subplots(figsize=(10, 6))
                value_counts = df[col].value_counts().head(10)
                bars = ax.bar(range(len(value_counts)), value_counts.values, color='lightgreen', alpha=0.8, edgecolor='black')
                ax.set_title(f'Top Values in {col}', fontsize=14, fontweight='bold', pad=15)
                ax.set_xlabel(col, fontsize=12)
                ax.set_ylabel('Count', fontsize=12)
                ax.set_xticks(range(len(value_counts)))
                ax.set_xticklabels(value_counts.index, rotation=45, ha='right')
                ax.grid(True, alpha=0.3, axis='y')
                for i, (idx, val) in enumerate(value_counts.items()):
                    ax.text(i, val, str(val), ha='center', va='bottom', fontweight='bold')
                plt.tight_layout()
                charts_data[f'{col}_bar'] = {
                    'chart': fig_to_base64(fig),
                    'type': 'bar_chart',
                    'title': f'{col} Top Values (Bar Chart)',
                    'insights': extract_insights_for_chart('bar_chart', f'{col}_bar', df, column=col)
                }
                plt.close(fig)
        except:
            continue
    
    return charts_data


def generate_visualizations(df):
    """
    Legacy function - redirects to comprehensive visualizations
    """
    charts_data = generate_comprehensive_visualizations(df)
    # Return simple format for backward compatibility
    return {key: val['chart'] for key, val in charts_data.items()}


def generate_numerical_distributions(df):
    """
    Generate histograms for all numerical features to identify skewness
    
    Returns:
        Dictionary with base64 encoded histogram images for each numerical column
    """
    charts = {}
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in numerical_cols:
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            data = df[col].dropna()
            
            # Create histogram
            n, bins, patches = ax.hist(data, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
            
            # Add vertical lines for mean and median
            mean_val = data.mean()
            median_val = data.median()
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
            ax.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.2f}')
            
            # Calculate and display skewness
            skewness = data.skew()
            skewness_label = 'Right Skewed' if skewness > 0.5 else 'Left Skewed' if skewness < -0.5 else 'Approximately Normal'
            
            ax.set_title(f'{col} Distribution (Skewness: {skewness:.2f} - {skewness_label})', 
                        fontsize=12, fontweight='bold')
            ax.set_xlabel(col, fontsize=11)
            ax.set_ylabel('Frequency', fontsize=11)
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3, linestyle='--')
            
            charts[f'{col}_histogram'] = fig_to_base64(fig)
            plt.close(fig)
        except Exception as e:
            continue
    
    return charts


def generate_boxplots(df):
    """
    Generate box plots for all numerical features to identify outliers
    
    Returns:
        Dictionary with base64 encoded box plot images for each numerical column
    """
    charts = {}
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in numerical_cols:
        try:
            fig, ax = plt.subplots(figsize=(8, 6))
            data = df[col].dropna()
            
            # Create box plot
            bp = ax.boxplot(data, vert=True, patch_artist=True, 
                           boxprops=dict(facecolor='lightblue', alpha=0.7),
                           medianprops=dict(color='red', linewidth=2),
                           whiskerprops=dict(color='black', linewidth=1.5),
                           capprops=dict(color='black', linewidth=1.5))
            
            # Calculate outliers
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = data[(data < lower_bound) | (data > upper_bound)]
            outlier_count = len(outliers)
            outlier_pct = (outlier_count / len(data)) * 100
            
            ax.set_title(f'{col} Box Plot (Outliers: {outlier_count} ({outlier_pct:.1f}%))', 
                        fontsize=12, fontweight='bold')
            ax.set_ylabel(col, fontsize=11)
            ax.grid(True, alpha=0.3, linestyle='--', axis='y')
            
            # Add text annotation for quartiles
            ax.text(1.15, Q1, f'Q1: {Q1:.2f}', verticalalignment='center', fontsize=9)
            ax.text(1.15, Q3, f'Q3: {Q3:.2f}', verticalalignment='center', fontsize=9)
            ax.text(1.15, data.median(), f'Median: {data.median():.2f}', 
                   verticalalignment='center', fontsize=9, fontweight='bold')
            
            charts[f'{col}_boxplot'] = fig_to_base64(fig)
            plt.close(fig)
        except Exception as e:
            continue
    
    return charts


def generate_all_eda_visualizations(df):
    """
    Generate all EDA visualizations including distributions, box plots, and correlation
    
    Returns:
        Dictionary with all base64 encoded chart images
    """
    charts = {}
    
    # Basic visualizations
    charts.update(generate_visualizations(df))
    
    # Numerical feature distributions (histograms)
    charts.update(generate_numerical_distributions(df))
    
    # Box plots for outlier detection
    charts.update(generate_boxplots(df))
    
    return charts

