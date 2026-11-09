"""
Plot helper utilities for generating charts and graphs
"""

import matplotlib.pyplot as plt
import seaborn as sns
import base64
import io


def setup_plot_style():
    """Configure default plotting style"""
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 10


def save_plot_to_base64(fig):
    """Save matplotlib figure to base64 string"""
    img = io.BytesIO()
    fig.savefig(img, format='png', bbox_inches='tight', dpi=100)
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()


def create_bar_chart(data, title, xlabel, ylabel):
    """Create a bar chart"""
    fig, ax = plt.subplots()
    data.plot(kind='bar', ax=ax, color='skyblue')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.xticks(rotation=45, ha='right')
    return fig

