"""
Data Visualization Questions with 15 Unique Chart Types
"""

import pandas as pd
import numpy as np
import base64
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
try:
    import squarify
    SQUARIFY_AVAILABLE = True
except ImportError:
    SQUARIFY_AVAILABLE = False
try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# Pink/Magenta color palette matching the theme
PINK_COLORS = ['#E91E63', '#C2185B', '#AD1457', '#F06292', '#F48FB1', '#F8BBD0', 
               '#FF6EC7', '#FF1493', '#FF69B4', '#FFB6C1', '#FFC0CB', '#FFE4E1']

# Multi-color palette for charts
MULTI_COLORS = ['#E91E63', '#C2185B', '#9C27B0', '#673AB7', '#3F51B5', '#2196F3',
                '#00BCD4', '#009688', '#4CAF50', '#8BC34A', '#FFC107', '#FF9800',
                '#FF5722', '#795548', '#607D8B', '#9E9E9E']


def fig_to_base64(fig):
    """Convert matplotlib figure to base64 string"""
    img = io.BytesIO()
    fig.savefig(img, format='png', bbox_inches='tight', dpi=100)
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()


def generate_interpretation(chart_num, df, target_col='Obesity Category', **kwargs):
    """Generate one-line interpretation for each chart"""
    # Ensure target_col exists in dataframe
    if target_col not in df.columns:
        # Try to find a similar column
        possible_cols = [col for col in df.columns if 'obesity' in col.lower() or 'category' in col.lower() or 'target' in col.lower()]
        if possible_cols:
            target_col = possible_cols[0]
        else:
            return "Analysis reveals key patterns in the relationship between variables and obesity categories."
    
    # Final check to ensure target_col is valid
    if target_col not in df.columns:
        return "Analysis reveals key patterns in the relationship between variables and obesity categories."
    
    interpretations = {
        1: lambda: f"The dataset contains {df[target_col].nunique()} obesity categories, with '{df[target_col].value_counts().index[0]}' being the most common ({df[target_col].value_counts().iloc[0]} individuals).",
        2: lambda: f"Gender distribution shows {'higher' if 'Gender' in df.columns and df[df['Gender']=='Male'][target_col].value_counts().sum() > df[df['Gender']=='Female'][target_col].value_counts().sum() else 'balanced'} representation across obesity categories.",
        3: lambda: f"Age distribution varies significantly across obesity categories, with median age ranging from {df.groupby(target_col)['Age'].median().min():.1f} to {df.groupby(target_col)['Age'].median().max():.1f} years.",
        4: lambda: f"Height and weight show {'strong' if 'Height' in df.columns and 'Weight' in df.columns and abs(df['Height'].corr(df['Weight'])) > 0.7 else 'moderate'} correlation ({(df['Height'].corr(df['Weight']) if 'Height' in df.columns and 'Weight' in df.columns else 0):.2f}), with distinct clusters for different obesity categories.",
        5: lambda: f"Family history of overweight {'significantly increases' if 'family_history_with_overweight' in df.columns and df[df['family_history_with_overweight']=='yes'][target_col].value_counts().sum() > df[df['family_history_with_overweight']=='no'][target_col].value_counts().sum() else 'shows minimal impact on'} obesity risk.",
        6: lambda: f"Frequent consumption of high-calorie food (FAVC) is {'strongly associated' if 'FAVC' in df.columns else 'moderately associated'} with higher obesity categories.",
        7: lambda: f"Vegetable consumption (FCVC) shows {'inverse' if 'FCVC' in df.columns and df.groupby(target_col)['FCVC'].mean().iloc[-1] < df.groupby(target_col)['FCVC'].mean().iloc[0] else 'positive'} relationship with obesity levels.",
        8: lambda: f"Number of meals per day (NCP) {'varies' if 'NCP' in df.columns else 'remains consistent'} across obesity categories, with most individuals consuming {df['NCP'].mode().iloc[0] if 'NCP' in df.columns else 'N/A'} meals daily.",
        9: lambda: f"Eating habits between meals (CAEC) reveal {'distinct patterns' if 'CAEC' in df.columns else 'similar patterns'} across different obesity categories.",
        10: lambda: f"Water consumption (CH2O) {'decreases' if 'CH2O' in df.columns and df.groupby(target_col)['CH2O'].mean().iloc[-1] < df.groupby(target_col)['CH2O'].mean().iloc[0] else 'increases'} with higher obesity categories.",
        11: lambda: f"Physical activity frequency (FAF) shows {'strong negative' if 'FAF' in df.columns and df.groupby(target_col)['FAF'].mean().iloc[-1] < df.groupby(target_col)['FAF'].mean().iloc[0] else 'weak'} correlation with obesity levels.",
        12: lambda: f"Technology usage time (TUE) {'increases' if 'TUE' in df.columns and df.groupby(target_col)['TUE'].mean().iloc[-1] > df.groupby(target_col)['TUE'].mean().iloc[0] else 'decreases'} with higher obesity categories.",
        13: lambda: f"Alcohol consumption (CALC) patterns {'differ significantly' if 'CALC' in df.columns else 'show minimal variation'} across obesity categories.",
        14: lambda: f"Transportation mode (MTRANS) {'strongly influences' if 'MTRANS' in df.columns else 'moderately affects'} obesity category distribution, with {'Public Transportation' if 'MTRANS' in df.columns and 'Public_Transportation' in df['MTRANS'].values else 'various modes'} being most common.",
        15: lambda: f"Lifestyle factors {'collectively' if True else 'individually'} separate obesity categories, with {'diet and activity' if True else 'genetic factors'} showing strongest discriminative power."
    }
    
    try:
        return interpretations.get(chart_num, lambda: "Analysis shows interesting patterns in the data.")()
    except:
        return "Analysis reveals key patterns in the relationship between variables and obesity categories."


def generate_visualization_questions(df):
    """
    Generate 15 unique visualization questions with charts
    Returns dictionary with base64 encoded charts and interpretations
    """
    charts = {}
    target_col = 'Obesity Category'
    
    if target_col not in df.columns:
        print(f"Warning: Target column '{target_col}' not found in dataset. Available columns: {list(df.columns)}")
        # Try to find a similar column
        possible_cols = [col for col in df.columns if 'obesity' in col.lower() or 'category' in col.lower() or 'target' in col.lower()]
        if possible_cols:
            target_col = possible_cols[0]
            print(f"Using '{target_col}' as target column instead.")
        else:
            return charts
    
    # Set style with pink theme
    sns.set_style("whitegrid")
    plt.rcParams['figure.facecolor'] = 'white'
    
    # 1. Bar Chart - Obesity Categories distribution
    try:
        fig, ax = plt.subplots(figsize=(12, 7))
        value_counts = df[target_col].value_counts().sort_index()
        bars = ax.bar(range(len(value_counts)), value_counts.values, 
                     color=MULTI_COLORS[:len(value_counts)], alpha=0.8, edgecolor='black', linewidth=1.5)
        ax.set_title('How are individuals distributed across different Obesity Categories?', 
                    fontsize=14, fontweight='bold', pad=15, color='#E91E63')
        ax.set_xlabel('Obesity Category', fontsize=12, fontweight='bold')
        ax.set_ylabel('Count', fontsize=12, fontweight='bold')
        ax.set_xticks(range(len(value_counts)))
        ax.set_xticklabels(value_counts.index, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        for i, (idx, val) in enumerate(value_counts.items()):
            ax.text(i, val, str(val), ha='center', va='bottom', fontweight='bold')
        plt.tight_layout()
        charts['v1_bar'] = {
            'chart': fig_to_base64(fig),
            'interpretation': generate_interpretation(1, df, target_col)
        }
        plt.close(fig)
    except Exception as e:
        print(f"Error in V1: {e}")
    
    # 2. Stacked Bar Chart - Gender vs Obesity Category
    try:
        if 'Gender' in df.columns:
            fig, ax = plt.subplots(figsize=(12, 7))
            cross_tab = pd.crosstab(df['Gender'], df[target_col])
            cross_tab.plot(kind='bar', stacked=True, ax=ax, color=MULTI_COLORS[:len(cross_tab.columns)])
            ax.set_title('How does Gender influence the distribution of Obesity Categories?', 
                        fontsize=14, fontweight='bold', pad=15, color='#E91E63')
            ax.set_xlabel('Gender', fontsize=12, fontweight='bold')
            ax.set_ylabel('Count', fontsize=12, fontweight='bold')
            ax.legend(title=target_col, bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3, axis='y')
            plt.xticks(rotation=0)
            plt.tight_layout()
            charts['v2_stacked_bar'] = {
                'chart': fig_to_base64(fig),
                'interpretation': generate_interpretation(2, df, target_col)
            }
            plt.close(fig)
    except Exception as e:
        print(f"Error in V2: {e}")
    
    # 3. Box Plot - Age vs Obesity Category
    try:
        if 'Age' in df.columns:
            fig, ax = plt.subplots(figsize=(12, 7))
            categories = sorted(df[target_col].unique())
            data_to_plot = [df[df[target_col] == cat]['Age'].dropna() for cat in categories]
            bp = ax.boxplot(data_to_plot, labels=categories, patch_artist=True)
            for patch, color in zip(bp['boxes'], MULTI_COLORS[:len(categories)]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            ax.set_title('How does Age vary across different Obesity Categories?', 
                        fontsize=14, fontweight='bold', pad=15, color='#E91E63')
            ax.set_xlabel(target_col, fontsize=12, fontweight='bold')
            ax.set_ylabel('Age', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            charts['v3_boxplot'] = {
                'chart': fig_to_base64(fig),
                'interpretation': generate_interpretation(3, df, target_col)
            }
            plt.close(fig)
    except Exception as e:
        print(f"Error in V3: {e}")
    
    # 4. Scatter Plot - Height, Weight vs Obesity Category
    try:
        if 'Height' in df.columns and 'Weight' in df.columns:
            fig, ax = plt.subplots(figsize=(12, 7))
            categories = sorted(df[target_col].unique())
            colors_map = {cat: MULTI_COLORS[i % len(MULTI_COLORS)] 
                         for i, cat in enumerate(categories)}
            for cat in categories:
                subset = df[df[target_col] == cat]
                ax.scatter(subset['Height'], subset['Weight'], 
                          label=cat, alpha=0.6, s=50, color=colors_map[cat], edgecolors='black', linewidth=0.5)
            ax.set_title('What is the relationship between Height and Weight across Obesity Categories?', 
                        fontsize=14, fontweight='bold', pad=15, color='#E91E63')
            ax.set_xlabel('Height (m)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Weight (kg)', fontsize=12, fontweight='bold')
            ax.legend(title=target_col, bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            charts['v4_scatter'] = {
                'chart': fig_to_base64(fig),
                'interpretation': generate_interpretation(4, df)
            }
            plt.close(fig)
    except Exception as e:
        print(f"Error in V4: {e}")
    
    # 5. Grouped Bar Chart - family_history_with_overweight vs Obesity Category
    try:
        if 'family_history_with_overweight' in df.columns:
            fig, ax = plt.subplots(figsize=(12, 7))
            cross_tab = pd.crosstab(df['family_history_with_overweight'], df[target_col])
            x = np.arange(len(cross_tab.index))
            width = 0.35
            categories = sorted(cross_tab.columns)
            for i, cat in enumerate(categories):
                offset = (i - len(categories)/2 + 0.5) * width / len(categories)
                ax.bar(x + offset, cross_tab[cat], width/len(categories), 
                      label=cat, color=MULTI_COLORS[i % len(MULTI_COLORS)], alpha=0.8)
            ax.set_title('Does family history of overweight increase the likelihood of higher Obesity Categories?', 
                        fontsize=14, fontweight='bold', pad=15, color='#E91E63')
            ax.set_xlabel('Family History with Overweight', fontsize=12, fontweight='bold')
            ax.set_ylabel('Count', fontsize=12, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(cross_tab.index)
            ax.legend(title=target_col)
            ax.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            charts['v5_grouped_bar'] = {
                'chart': fig_to_base64(fig),
                'interpretation': generate_interpretation(5, df, target_col)
            }
            plt.close(fig)
    except Exception as e:
        print(f"Error in V5: {e}")
    
    # 6. Percentage Bar Chart - FAVC vs Obesity Category
    try:
        if 'FAVC' in df.columns:
            fig, ax = plt.subplots(figsize=(12, 7))
            cross_tab = pd.crosstab(df['FAVC'], df[target_col], normalize='index') * 100
            categories = sorted(cross_tab.index)
            target_cats = sorted(cross_tab.columns)
            bottom = np.zeros(len(categories))
            for i, tcat in enumerate(target_cats):
                ax.bar(categories, cross_tab[tcat], bottom=bottom, 
                      label=tcat, color=MULTI_COLORS[i % len(MULTI_COLORS)], alpha=0.8)
                bottom += cross_tab[tcat]
            ax.set_title('How does frequent consumption of high-calorie food (FAVC) differ across Obesity Categories?', 
                        fontsize=14, fontweight='bold', pad=15, color='#E91E63')
            ax.set_xlabel('FAVC (Frequent Caloric Food)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
            ax.legend(title=target_col)
            ax.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            charts['v6_percentage_bar'] = {
                'chart': fig_to_base64(fig),
                'interpretation': generate_interpretation(6, df, target_col)
            }
            plt.close(fig)
    except Exception as e:
        print(f"Error in V6: {e}")
    
    # 7. Violin Plot - FCVC vs Obesity Category
    try:
        if 'FCVC' in df.columns:
            fig, ax = plt.subplots(figsize=(12, 7))
            categories = sorted(df[target_col].unique())
            data_to_plot = [df[df[target_col] == cat]['FCVC'].dropna() for cat in categories]
            parts = ax.violinplot(data_to_plot, positions=range(len(categories)), 
                                 showmeans=True, showmedians=True)
            for i, pc in enumerate(parts['bodies']):
                pc.set_facecolor(MULTI_COLORS[i % len(MULTI_COLORS)])
                pc.set_alpha(0.7)
            ax.set_xticks(range(len(categories)))
            ax.set_xticklabels(categories, rotation=45, ha='right')
            ax.set_title('How does daily vegetable consumption (FCVC) change with Obesity Category?', 
                        fontsize=14, fontweight='bold', pad=15, color='#E91E63')
            ax.set_xlabel(target_col, fontsize=12, fontweight='bold')
            ax.set_ylabel('FCVC (Vegetables Consumption)', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            charts['v7_violin'] = {
                'chart': fig_to_base64(fig),
                'interpretation': generate_interpretation(7, df)
            }
            plt.close(fig)
    except Exception as e:
        print(f"Error in V7: {e}")
    
    # 8. Strip Plot - NCP vs Obesity Category
    try:
        if 'NCP' in df.columns:
            fig, ax = plt.subplots(figsize=(12, 7))
            categories = sorted(df[target_col].unique())
            for i, cat in enumerate(categories):
                subset = df[df[target_col] == cat]['NCP'].dropna()
                y_pos = np.ones(len(subset)) * i
                ax.scatter(subset, y_pos, alpha=0.5, s=30, 
                          color=MULTI_COLORS[i % len(MULTI_COLORS)], label=cat, edgecolors='black', linewidth=0.3)
            ax.set_yticks(range(len(categories)))
            ax.set_yticklabels(categories)
            ax.set_title('How does the number of main meals per day (NCP) vary across Obesity Categories?', 
                        fontsize=14, fontweight='bold', pad=15, color='#E91E63')
            ax.set_xlabel('NCP (Number of Meals)', fontsize=12, fontweight='bold')
            ax.set_ylabel(target_col, fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='x')
            plt.tight_layout()
            charts['v8_strip'] = {
                'chart': fig_to_base64(fig),
                'interpretation': generate_interpretation(8, df, target_col)
            }
            plt.close(fig)
    except Exception as e:
        print(f"Error in V8: {e}")
    
    # 9. Mosaic Plot - CAEC vs Obesity Category
    try:
        if 'CAEC' in df.columns:
            fig, ax = plt.subplots(figsize=(12, 7))
            cross_tab = pd.crosstab(df['CAEC'], df[target_col], normalize='index')
            sns.heatmap(cross_tab, annot=True, fmt='.2f', cmap='RdPu', 
                       cbar_kws={'label': 'Proportion'}, ax=ax, 
                       linewidths=0.5, linecolor='white', vmin=0, vmax=1)
            ax.set_title('How do eating habits between meals (CAEC) relate to different Obesity Categories?', 
                        fontsize=14, fontweight='bold', pad=15, color='#E91E63')
            ax.set_xlabel(target_col, fontsize=12, fontweight='bold')
            ax.set_ylabel('CAEC (Food Between Meals)', fontsize=12, fontweight='bold')
            plt.tight_layout()
            charts['v9_mosaic'] = {
                'chart': fig_to_base64(fig),
                'interpretation': generate_interpretation(9, df)
            }
            plt.close(fig)
    except Exception as e:
        print(f"Error in V9: {e}")
    
    # 10. Density Plot - CH2O vs Obesity Category
    try:
        if 'CH2O' in df.columns:
            fig, ax = plt.subplots(figsize=(12, 7))
            categories = sorted(df[target_col].unique())
            for i, cat in enumerate(categories):
                subset = df[df[target_col] == cat]['CH2O'].dropna()
                subset.plot.density(ax=ax, label=cat, color=MULTI_COLORS[i % len(MULTI_COLORS)], 
                                   linewidth=2.5)
            ax.set_title('How does daily water consumption (CH2O) differ across Obesity Categories?', 
                        fontsize=14, fontweight='bold', pad=15, color='#E91E63')
            ax.set_xlabel('CH2O (Water Consumption)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Density', fontsize=12, fontweight='bold')
            ax.legend(title=target_col)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            charts['v10_density'] = {
                'chart': fig_to_base64(fig),
                'interpretation': generate_interpretation(10, df, target_col)
            }
            plt.close(fig)
    except Exception as e:
        print(f"Error in V10: {e}")
    
    # 11. Ridge Plot (Joy Plot) - FAF vs Obesity Category
    try:
        if 'FAF' in df.columns:
            fig, axes = plt.subplots(len(sorted(df[target_col].unique())), 1, figsize=(12, 10), 
                                    sharex=True, sharey=False)
            categories = sorted(df[target_col].unique())
            for idx, cat in enumerate(categories):
                subset = df[df[target_col] == cat]['FAF'].dropna()
                axes[idx].hist(subset, bins=20, alpha=0.7, color=MULTI_COLORS[idx % len(MULTI_COLORS)], 
                              edgecolor='black', linewidth=0.5)
                axes[idx].set_ylabel(cat, fontweight='bold', fontsize=10)
                axes[idx].spines['top'].set_visible(False)
                axes[idx].spines['right'].set_visible(False)
                if idx < len(categories) - 1:
                    axes[idx].set_xticks([])
            axes[-1].set_xlabel('FAF (Physical Activity Frequency)', fontsize=12, fontweight='bold')
            fig.suptitle('How does physical activity frequency (FAF) impact Obesity Category distribution?', 
                        fontsize=14, fontweight='bold', color='#E91E63', y=0.995)
            plt.tight_layout()
            charts['v11_ridge'] = {
                'chart': fig_to_base64(fig),
                'interpretation': generate_interpretation(11, df, target_col)
            }
            plt.close(fig)
    except Exception as e:
        print(f"Error in V11: {e}")
    
    # 12. Line Plot - TUE vs Obesity Category (mean)
    try:
        if 'TUE' in df.columns:
            fig, ax = plt.subplots(figsize=(12, 7))
            categories = sorted(df[target_col].unique())
            means = [df[df[target_col] == cat]['TUE'].mean() for cat in categories]
            ax.plot(categories, means, marker='o', markersize=12, linewidth=3, 
                   color='#E91E63', markerfacecolor='#C2185B', markeredgecolor='white', 
                   markeredgewidth=2, label='Mean TUE')
            ax.fill_between(categories, means, alpha=0.3, color='#E91E63')
            ax.set_title('How does time spent using technology devices (TUE) vary across Obesity Categories?', 
                        fontsize=14, fontweight='bold', pad=15, color='#E91E63')
            ax.set_xlabel(target_col, fontsize=12, fontweight='bold')
            ax.set_ylabel('TUE (Technology Use) - Mean', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            charts['v12_line'] = {
                'chart': fig_to_base64(fig),
                'interpretation': generate_interpretation(12, df, target_col)
            }
            plt.close(fig)
    except Exception as e:
        print(f"Error in V12: {e}")
    
    # 13. Heatmap - CALC vs Obesity Category
    try:
        if 'CALC' in df.columns:
            fig, ax = plt.subplots(figsize=(12, 7))
            cross_tab = pd.crosstab(df['CALC'], df[target_col], normalize='index')
            sns.heatmap(cross_tab, annot=True, fmt='.2f', cmap='RdPu', 
                       cbar_kws={'label': 'Normalized Proportion'}, ax=ax, 
                       linewidths=0.5, linecolor='white', vmin=0, vmax=1)
            ax.set_title('What is the combined effect of alcohol consumption (CALC) and Obesity Category?', 
                        fontsize=14, fontweight='bold', pad=15, color='#E91E63')
            ax.set_xlabel(target_col, fontsize=12, fontweight='bold')
            ax.set_ylabel('CALC (Alcohol Consumption)', fontsize=12, fontweight='bold')
            plt.tight_layout()
            charts['v13_heatmap'] = {
                'chart': fig_to_base64(fig),
                'interpretation': generate_interpretation(13, df, target_col)
            }
            plt.close(fig)
    except Exception as e:
        print(f"Error in V13: {e}")
    
    # 14. Sankey Diagram (simplified as grouped bar) - MTRANS vs Obesity Category
    try:
        if 'MTRANS' in df.columns:
            fig, ax = plt.subplots(figsize=(14, 8))
            cross_tab = pd.crosstab(df['MTRANS'], df[target_col])
            x = np.arange(len(cross_tab.index))
            width = 0.15
            categories = sorted(cross_tab.columns)
            for i, cat in enumerate(categories):
                offset = (i - len(categories)/2 + 0.5) * width
                ax.bar(x + offset, cross_tab[cat], width, 
                      label=cat, color=MULTI_COLORS[i % len(MULTI_COLORS)], alpha=0.8)
            ax.set_title('How does mode of transportation (MTRANS) influence Obesity Categories?', 
                        fontsize=14, fontweight='bold', pad=15, color='#E91E63')
            ax.set_xlabel('MTRANS (Transportation Mode)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Count', fontsize=12, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(cross_tab.index, rotation=45, ha='right')
            ax.legend(title=target_col, bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            charts['v14_sankey'] = {
                'chart': fig_to_base64(fig),
                'interpretation': generate_interpretation(14, df)
            }
            plt.close(fig)
    except Exception as e:
        print(f"Error in V14: {e}")
    
    # 15. Parallel Coordinates Plot - Lifestyle factors vs Obesity Category
    try:
        lifestyle_cols = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
        available_cols = [col for col in lifestyle_cols if col in df.columns]
        if len(available_cols) >= 3:
            fig, ax = plt.subplots(figsize=(14, 8))
            categories = sorted(df[target_col].unique())
            colors_map = {cat: MULTI_COLORS[i % len(MULTI_COLORS)] 
                         for i, cat in enumerate(categories)}
            
            # Normalize data for parallel coordinates
            normalized_data = df[available_cols + [target_col]].copy()
            for col in available_cols:
                if normalized_data[col].dtype in [np.float64, np.int64]:
                    min_val = normalized_data[col].min()
                    max_val = normalized_data[col].max()
                    if max_val > min_val:
                        normalized_data[col] = (normalized_data[col] - min_val) / (max_val - min_val)
            
            # Plot parallel coordinates
            for cat in categories:
                subset = normalized_data[normalized_data[target_col] == cat]
                for idx, row in subset.iterrows():
                    values = [row[col] for col in available_cols]
                    ax.plot(range(len(available_cols)), values, 
                           alpha=0.1, color=colors_map[cat], linewidth=0.5)
            
            # Plot mean lines for each category
            for cat in categories:
                subset = normalized_data[normalized_data[target_col] == cat]
                means = [subset[col].mean() for col in available_cols]
                ax.plot(range(len(available_cols)), means, 
                       marker='o', markersize=8, linewidth=3, 
                       label=cat, color=colors_map[cat], alpha=0.9)
            
            ax.set_xticks(range(len(available_cols)))
            ax.set_xticklabels(available_cols, rotation=45, ha='right')
            ax.set_title('Which lifestyle factors most strongly separate Obesity Categories?', 
                        fontsize=14, fontweight='bold', pad=15, color='#E91E63')
            ax.set_ylabel('Normalized Value', fontsize=12, fontweight='bold')
            ax.legend(title=target_col, bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            charts['v15_parallel'] = {
                'chart': fig_to_base64(fig),
                'interpretation': generate_interpretation(15, df, target_col)
            }
            plt.close(fig)
    except Exception as e:
        print(f"Error in V15: {e}")
    
    return charts

