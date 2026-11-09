"""
EDA Triggering Questions with 12 Unique Chart Types
"""

import pandas as pd
import numpy as np
import base64
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
try:
    import squarify
    SQUARIFY_AVAILABLE = True
except ImportError:
    SQUARIFY_AVAILABLE = False

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


def generate_eda_triggering_charts(df):
    """
    Generate 12 unique chart types for EDA triggering questions
    Returns dictionary with base64 encoded charts
    """
    charts = {}
    target_col = 'Obesity Category'
    
    if target_col not in df.columns:
        return charts
    
    # Set style with pink theme
    sns.set_style("whitegrid")
    plt.rcParams['figure.facecolor'] = 'white'
    
    # 1. Stacked Bar Chart - Gender vs Obesity Category
    try:
        if 'Gender' in df.columns:
            fig, ax = plt.subplots(figsize=(12, 7))
            cross_tab = pd.crosstab(df['Gender'], df[target_col])
            cross_tab.plot(kind='bar', stacked=True, ax=ax, color=MULTI_COLORS[:len(cross_tab.columns)])
            ax.set_title('How does obesity category vary across different genders?', 
                        fontsize=14, fontweight='bold', pad=15, color='#E91E63')
            ax.set_xlabel('Gender', fontsize=12, fontweight='bold')
            ax.set_ylabel('Count', fontsize=12, fontweight='bold')
            ax.legend(title=target_col, bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3, axis='y')
            plt.xticks(rotation=0)
            plt.tight_layout()
            charts['q1_stacked_bar'] = fig_to_base64(fig)
            plt.close(fig)
    except Exception as e:
        print(f"Error in Q1: {e}")
    
    # 2. Box Plot - Age vs Obesity Category
    try:
        if 'Age' in df.columns:
            fig, ax = plt.subplots(figsize=(12, 7))
            categories = df[target_col].unique()
            data_to_plot = [df[df[target_col] == cat]['Age'].dropna() for cat in categories]
            bp = ax.boxplot(data_to_plot, labels=categories, patch_artist=True,
                          boxprops=dict(facecolor=PINK_COLORS[0], alpha=0.7),
                          medianprops=dict(color='#C2185B', linewidth=2))
            for patch, color in zip(bp['boxes'], MULTI_COLORS[:len(categories)]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            ax.set_title('What is the distribution of age across obesity categories?', 
                        fontsize=14, fontweight='bold', pad=15, color='#E91E63')
            ax.set_xlabel(target_col, fontsize=12, fontweight='bold')
            ax.set_ylabel('Age', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            charts['q2_boxplot'] = fig_to_base64(fig)
            plt.close(fig)
    except Exception as e:
        print(f"Error in Q2: {e}")
    
    # 3. Scatter Plot - Height, Weight, Obesity Category (color-coded)
    try:
        if 'Height' in df.columns and 'Weight' in df.columns:
            fig, ax = plt.subplots(figsize=(12, 7))
            categories = df[target_col].unique()
            colors_map = {cat: MULTI_COLORS[i % len(MULTI_COLORS)] 
                         for i, cat in enumerate(categories)}
            for cat in categories:
                subset = df[df[target_col] == cat]
                ax.scatter(subset['Height'], subset['Weight'], 
                          label=cat, alpha=0.6, s=50, color=colors_map[cat])
            ax.set_title('How do height and weight jointly differentiate obesity categories?', 
                        fontsize=14, fontweight='bold', pad=15, color='#E91E63')
            ax.set_xlabel('Height (m)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Weight (kg)', fontsize=12, fontweight='bold')
            ax.legend(title=target_col, bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            charts['q3_scatter'] = fig_to_base64(fig)
            plt.close(fig)
    except Exception as e:
        print(f"Error in Q3: {e}")
    
    # 4. Grouped Bar Chart - family_history_with_overweight vs Obesity Category
    try:
        if 'family_history_with_overweight' in df.columns:
            fig, ax = plt.subplots(figsize=(12, 7))
            cross_tab = pd.crosstab(df['family_history_with_overweight'], df[target_col])
            x = np.arange(len(cross_tab.index))
            width = 0.35
            categories = cross_tab.columns
            for i, cat in enumerate(categories):
                offset = (i - len(categories)/2 + 0.5) * width / len(categories)
                ax.bar(x + offset, cross_tab[cat], width/len(categories), 
                      label=cat, color=MULTI_COLORS[i % len(MULTI_COLORS)], alpha=0.8)
            ax.set_title('Does family history of overweight increase obesity risk?', 
                        fontsize=14, fontweight='bold', pad=15, color='#E91E63')
            ax.set_xlabel('Family History with Overweight', fontsize=12, fontweight='bold')
            ax.set_ylabel('Count', fontsize=12, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(cross_tab.index)
            ax.legend(title=target_col)
            ax.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            charts['q4_grouped_bar'] = fig_to_base64(fig)
            plt.close(fig)
    except Exception as e:
        print(f"Error in Q4: {e}")
    
    # 5. Mosaic Plot - FAVC vs Obesity Category
    try:
        if 'FAVC' in df.columns:
            fig, ax = plt.subplots(figsize=(12, 7))
            cross_tab = pd.crosstab(df['FAVC'], df[target_col], normalize='index') * 100
            categories = cross_tab.index
            target_cats = cross_tab.columns
            bottom = np.zeros(len(categories))
            for i, tcat in enumerate(target_cats):
                ax.bar(categories, cross_tab[tcat], bottom=bottom, 
                      label=tcat, color=MULTI_COLORS[i % len(MULTI_COLORS)], alpha=0.8)
                bottom += cross_tab[tcat]
            ax.set_title('How does frequent consumption of high-calorie food affect obesity levels?', 
                        fontsize=14, fontweight='bold', pad=15, color='#E91E63')
            ax.set_xlabel('FAVC (Frequent Caloric Food)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
            ax.legend(title=target_col)
            ax.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            charts['q5_mosaic'] = fig_to_base64(fig)
            plt.close(fig)
    except Exception as e:
        print(f"Error in Q5: {e}")
    
    # 6. Violin Plot - FCVC vs Obesity Category
    try:
        if 'FCVC' in df.columns:
            fig, ax = plt.subplots(figsize=(12, 7))
            categories = df[target_col].unique()
            data_to_plot = [df[df[target_col] == cat]['FCVC'].dropna() for cat in categories]
            parts = ax.violinplot(data_to_plot, positions=range(len(categories)), 
                                 showmeans=True, showmedians=True)
            for i, pc in enumerate(parts['bodies']):
                pc.set_facecolor(MULTI_COLORS[i % len(MULTI_COLORS)])
                pc.set_alpha(0.7)
            ax.set_xticks(range(len(categories)))
            ax.set_xticklabels(categories, rotation=45, ha='right')
            ax.set_title('Is vegetable consumption frequency associated with obesity status?', 
                        fontsize=14, fontweight='bold', pad=15, color='#E91E63')
            ax.set_xlabel(target_col, fontsize=12, fontweight='bold')
            ax.set_ylabel('FCVC (Vegetables Consumption)', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            charts['q6_violin'] = fig_to_base64(fig)
            plt.close(fig)
    except Exception as e:
        print(f"Error in Q6: {e}")
    
    # 7. Strip Plot - NCP vs Obesity Category
    try:
        if 'NCP' in df.columns:
            fig, ax = plt.subplots(figsize=(12, 7))
            categories = df[target_col].unique()
            for i, cat in enumerate(categories):
                subset = df[df[target_col] == cat]['NCP'].dropna()
                y_pos = np.ones(len(subset)) * i
                ax.scatter(subset, y_pos, alpha=0.5, s=30, 
                          color=MULTI_COLORS[i % len(MULTI_COLORS)], label=cat)
            ax.set_yticks(range(len(categories)))
            ax.set_yticklabels(categories)
            ax.set_title('Does the number of main meals per day influence obesity category?', 
                        fontsize=14, fontweight='bold', pad=15, color='#E91E63')
            ax.set_xlabel('NCP (Number of Meals)', fontsize=12, fontweight='bold')
            ax.set_ylabel(target_col, fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='x')
            plt.tight_layout()
            charts['q7_strip'] = fig_to_base64(fig)
            plt.close(fig)
    except Exception as e:
        print(f"Error in Q7: {e}")
    
    # 8. Heatmap - CAEC vs Obesity Category (normalized counts)
    try:
        if 'CAEC' in df.columns:
            fig, ax = plt.subplots(figsize=(12, 7))
            cross_tab = pd.crosstab(df['CAEC'], df[target_col], normalize='index')
            sns.heatmap(cross_tab, annot=True, fmt='.2f', cmap='RdPu', 
                       cbar_kws={'label': 'Normalized Count'}, ax=ax, 
                       linewidths=0.5, linecolor='white')
            ax.set_title('How does eating between meals relate to obesity categories?', 
                        fontsize=14, fontweight='bold', pad=15, color='#E91E63')
            ax.set_xlabel(target_col, fontsize=12, fontweight='bold')
            ax.set_ylabel('CAEC (Food Between Meals)', fontsize=12, fontweight='bold')
            plt.tight_layout()
            charts['q8_heatmap'] = fig_to_base64(fig)
            plt.close(fig)
    except Exception as e:
        print(f"Error in Q8: {e}")
    
    # 9. Line Plot - CH2O vs Obesity Category (mean trend)
    try:
        if 'CH2O' in df.columns:
            fig, ax = plt.subplots(figsize=(12, 7))
            categories = df[target_col].unique()
            for i, cat in enumerate(categories):
                subset = df[df[target_col] == cat]['CH2O'].dropna()
                mean_val = subset.mean()
                ax.plot([0, 1], [mean_val, mean_val], 
                       marker='o', markersize=10, linewidth=3, 
                       label=f'{cat} (Mean: {mean_val:.2f})',
                       color=MULTI_COLORS[i % len(MULTI_COLORS)])
            ax.set_title('Is daily water intake linked to obesity levels?', 
                        fontsize=14, fontweight='bold', pad=15, color='#E91E63')
            ax.set_xlabel('Category', fontsize=12, fontweight='bold')
            ax.set_ylabel('CH2O (Water Consumption) - Mean', fontsize=12, fontweight='bold')
            ax.set_xticks([0, 1])
            ax.set_xticklabels(['', ''])
            ax.legend(title=target_col)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            charts['q9_line'] = fig_to_base64(fig)
            plt.close(fig)
    except Exception as e:
        print(f"Error in Q9: {e}")
    
    # 10. Radar/Spider Chart - FAF vs Obesity Category
    try:
        if 'FAF' in df.columns:
            fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
            categories = df[target_col].unique()
            angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
            angles += angles[:1]  # Complete the circle
            
            for i, cat in enumerate(categories):
                subset = df[df[target_col] == cat]['FAF'].dropna()
                values = [subset.mean()] * len(angles)
                ax.plot(angles, values, 'o-', linewidth=2, label=cat,
                       color=MULTI_COLORS[i % len(MULTI_COLORS)])
                ax.fill(angles, values, alpha=0.25, color=MULTI_COLORS[i % len(MULTI_COLORS)])
            
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories)
            ax.set_title('How does physical activity frequency affect obesity classification?', 
                        fontsize=14, fontweight='bold', pad=20, color='#E91E63')
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
            ax.grid(True)
            plt.tight_layout()
            charts['q10_radar'] = fig_to_base64(fig)
            plt.close(fig)
    except Exception as e:
        print(f"Error in Q10: {e}")
    
    # 11. Density Plot (KDE) - TUE vs Obesity Category
    try:
        if 'TUE' in df.columns:
            fig, ax = plt.subplots(figsize=(12, 7))
            categories = df[target_col].unique()
            for i, cat in enumerate(categories):
                subset = df[df[target_col] == cat]['TUE'].dropna()
                subset.plot.density(ax=ax, label=cat, color=MULTI_COLORS[i % len(MULTI_COLORS)], 
                                   linewidth=2.5)
            ax.set_title('Does screen/technology usage time impact obesity risk?', 
                        fontsize=14, fontweight='bold', pad=15, color='#E91E63')
            ax.set_xlabel('TUE (Technology Use)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Density', fontsize=12, fontweight='bold')
            ax.legend(title=target_col)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            charts['q11_density'] = fig_to_base64(fig)
            plt.close(fig)
    except Exception as e:
        print(f"Error in Q11: {e}")
    
    # 12. Treemap - MTRANS vs Obesity Category
    try:
        if 'MTRANS' in df.columns:
            if SQUARIFY_AVAILABLE:
                fig, ax = plt.subplots(figsize=(14, 8))
                cross_tab = pd.crosstab(df['MTRANS'], df[target_col])
                sizes = []
                labels = []
                colors_list = []
                color_idx = 0
                for trans in cross_tab.index:
                    for cat in cross_tab.columns:
                        size = cross_tab.loc[trans, cat]
                        if size > 0:
                            sizes.append(size)
                            labels.append(f'{trans}\n{cat}\n{size}')
                            colors_list.append(MULTI_COLORS[color_idx % len(MULTI_COLORS)])
                            color_idx += 1
                
                if sizes:
                    squarify.plot(sizes=sizes, label=labels, color=colors_list, 
                                alpha=0.8, ax=ax, text_kwargs={'fontsize': 9, 'fontweight': 'bold'})
                    ax.set_title('Are transportation modes associated with different obesity categories?', 
                                fontsize=14, fontweight='bold', pad=15, color='#E91E63')
                    ax.axis('off')
                plt.tight_layout()
                charts['q12_treemap'] = fig_to_base64(fig)
                plt.close(fig)
            else:
                # Fallback to grouped bar chart if squarify not available
                fig, ax = plt.subplots(figsize=(12, 7))
                cross_tab = pd.crosstab(df['MTRANS'], df[target_col])
                x = np.arange(len(cross_tab.index))
                width = 0.35
                categories = cross_tab.columns
                for i, cat in enumerate(categories):
                    offset = (i - len(categories)/2 + 0.5) * width / len(categories)
                    ax.bar(x + offset, cross_tab[cat], width/len(categories), 
                          label=cat, color=MULTI_COLORS[i % len(MULTI_COLORS)], alpha=0.8)
                ax.set_title('Are transportation modes associated with different obesity categories?', 
                            fontsize=14, fontweight='bold', pad=15, color='#E91E63')
                ax.set_xlabel('MTRANS (Transportation)', fontsize=12, fontweight='bold')
                ax.set_ylabel('Count', fontsize=12, fontweight='bold')
                ax.set_xticks(x)
                ax.set_xticklabels(cross_tab.index, rotation=45, ha='right')
                ax.legend(title=target_col)
                ax.grid(True, alpha=0.3, axis='y')
                plt.tight_layout()
                charts['q12_treemap'] = fig_to_base64(fig)
                plt.close(fig)
    except Exception as e:
        print(f"Error in Q12: {e}")
    
    return charts

