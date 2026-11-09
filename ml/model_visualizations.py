"""
Model training visualization functions
"""

import base64
import io
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def fig_to_base64(fig):
    """Convert matplotlib figure to base64 string"""
    img = io.BytesIO()
    fig.savefig(img, format='png', bbox_inches='tight', dpi=100)
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()


def generate_training_metrics_chart(training_results):
    """
    Generate a vertical bar chart comparing training metrics (accuracy) for all models
    
    Args:
        training_results: List of dictionaries with 'model', 'accuracy', 'status' keys
    
    Returns:
        Base64 encoded image string
    """
    # Filter out models with errors
    valid_results = [r for r in training_results if r.get('status') == 'trained' and 'accuracy' in r]
    
    if not valid_results:
        return None
    
    # Extract data
    models = [r['model'] for r in valid_results]
    accuracies = [r['accuracy'] * 100 for r in valid_results]  # Convert to percentage
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Define colors for each model
    colors = sns.color_palette("husl", len(models))
    
    # Create bar chart
    bars = ax.bar(models, accuracies, color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)
    
    # Customize chart
    ax.set_xlabel('Models', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Model Training Performance Comparison', fontsize=14, fontweight='bold', pad=20)
    ax.set_ylim([0, max(accuracies) * 1.15 if accuracies else 100])
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on top of bars
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.2f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Add grid for better readability
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Adjust layout
    plt.tight_layout()
    
    # Convert to base64
    chart_image = fig_to_base64(fig)
    plt.close(fig)
    
    return chart_image


def generate_roc_auc_curves(model, X_test, y_test, model_name):
    """
    Generate ROC-AUC curves for a model (multi-class one-vs-rest)
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        model_name: Name of the algorithm
        
    Returns:
        Base64 encoded image string
    """
    import numpy as np
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize
    
    try:
        # Get class labels
        classes = np.unique(y_test)
        n_classes = len(classes)
        
        # Binarize the output
        y_test_bin = label_binarize(y_test, classes=classes)
        
        # Get predicted probabilities
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)
        elif hasattr(model, "decision_function"):
            y_score = model.decision_function(X_test)
        else:
            return None

        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot perfect ROC curve (reference)
        ax.plot([0, 0, 1], [0, 1, 1], color='green', linestyle='--', linewidth=2, label='Perfect Classifier (AUC = 1.00)')
        
        # Plot chance line
        ax.plot([0, 1], [0, 1], color='navy', linestyle=':', linewidth=2, label='Chance (AUC = 0.50)')
        
        # Plot micro-average ROC curve
        ax.plot(fpr["micro"], tpr["micro"],
                 label=f'Micro-average ROC (AUC = {roc_auc["micro"]:0.2f})',
                 color='deeppink', linestyle=':', linewidth=4)

        # Plot ROC curve for each class
        colors = sns.color_palette("husl", n_classes)
        for i, color in zip(range(n_classes), colors):
            ax.plot(fpr[i], tpr[i], color=color, lw=2,
                     label=f'ROC class {classes[i]} (AUC = {roc_auc[i]:0.2f})')

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        ax.set_title(f'ROC-AUC Curves - {model_name}', fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc="lower right", fontsize=9, frameon=True, shadow=True)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Convert to base64
        chart_image = fig_to_base64(fig)
        plt.close(fig)
        
        return chart_image
    except Exception as e:
        print(f"Error generating ROC curves for {model_name}: {str(e)}")
        return None


def find_best_model(training_results):
    """
    Find the best model based on training accuracy
    
    Args:
        training_results: List of dictionaries with 'model', 'accuracy', 'train_accuracy', 'status' keys
    
    Returns:
        Dictionary with best model info or None
    """
    valid_results = [r for r in training_results if r.get('status') == 'trained' and 'train_accuracy' in r]
    
    if not valid_results:
        return None
    
    # Find model with highest training accuracy
    best_model = max(valid_results, key=lambda x: x.get('train_accuracy', 0))
    return best_model


def generate_accuracy_curve(best_model_result, test_accuracy=None):
    """
    Generate training vs test accuracy curve for the best model
    
    Args:
        best_model_result: Dictionary with model info including 'training_history'
        test_accuracy: Final test accuracy from evaluation (optional)
    
    Returns:
        Base64 encoded image string
    """
    if not best_model_result or 'training_history' not in best_model_result:
        return None
    
    history = best_model_result['training_history']
    epochs = history.get('epochs', [])
    train_acc = history.get('train_accuracy', [])
    val_acc = history.get('test_accuracy', [])  # This is validation accuracy during training
    
    if not epochs or not train_acc:
        return None
    
    # Convert to percentages
    train_acc_pct = [acc * 100 for acc in train_acc]
    val_acc_pct = [acc * 100 for acc in val_acc] if val_acc else []
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot training accuracy curve
    ax.plot(epochs, train_acc_pct, 'o-', label='Training Accuracy (During Training)', 
            linewidth=3, markersize=10, color='#667eea', markerfacecolor='white', 
            markeredgewidth=2.5, markeredgecolor='#667eea', zorder=3)
    
    # Plot validation accuracy curve if available
    if val_acc_pct:
        ax.plot(epochs, val_acc_pct, 's-', label='Validation Accuracy (During Training)', 
                linewidth=3, markersize=10, color='#28a745', markerfacecolor='white', 
                markeredgewidth=2.5, markeredgecolor='#28a745', zorder=3)
    
    # Add final training accuracy point
    if 'train_accuracy' in best_model_result:
        final_train = best_model_result['train_accuracy'] * 100
        ax.scatter([epochs[-1]], [final_train], s=300, color='#667eea', 
                  marker='*', edgecolors='black', linewidths=2.5, zorder=5, 
                  label='Final Training Accuracy')
    
    # Add final test accuracy point if provided
    if test_accuracy is not None:
        final_test = test_accuracy * 100
        ax.scatter([epochs[-1]], [final_test], s=300, color='#dc3545', 
                  marker='D', edgecolors='black', linewidths=2.5, zorder=5, 
                  label='Final Test Accuracy')
    elif 'accuracy' in best_model_result:
        final_test = best_model_result['accuracy'] * 100
        ax.scatter([epochs[-1]], [final_test], s=300, color='#dc3545', 
                  marker='D', edgecolors='black', linewidths=2.5, zorder=5, 
                  label='Final Test Accuracy')
    
    # Customize chart
    ax.set_xlabel('Training Iterations / Samples', fontsize=14, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_title(f'{best_model_result.get("model", "Model")} - Training vs Test Accuracy Comparison', 
                fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='best', fontsize=11, framealpha=0.9, shadow=True, ncol=2)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=1)
    ax.set_axisbelow(True)
    
    # Set y-axis limits
    all_acc = train_acc_pct + val_acc_pct
    if all_acc:
        min_acc = min(all_acc)
        max_acc = max(all_acc)
        ax.set_ylim([max(0, min_acc - 5), min(100, max_acc + 5)])
    
    # Add annotations for final accuracies
    if 'train_accuracy' in best_model_result:
        final_train = best_model_result['train_accuracy'] * 100
        ax.annotate(f'Train: {final_train:.2f}%', 
                   xy=(epochs[-1], final_train), 
                   xytext=(15, 15), textcoords='offset points',
                   fontsize=11, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.8', facecolor='#667eea', 
                           alpha=0.7, edgecolor='black', linewidth=2),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3', 
                                 color='black', lw=1.5))
    
    final_test_val = None
    if test_accuracy is not None:
        final_test_val = test_accuracy * 100
    elif 'accuracy' in best_model_result:
        final_test_val = best_model_result['accuracy'] * 100
    
    if final_test_val is not None:
        ax.annotate(f'Test: {final_test_val:.2f}%', 
                   xy=(epochs[-1], final_test_val), 
                   xytext=(15, -25), textcoords='offset points',
                   fontsize=11, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.8', facecolor='#dc3545', 
                           alpha=0.7, edgecolor='black', linewidth=2),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=-0.3', 
                                 color='black', lw=1.5))
    
    # Fill area between curves if both exist
    if val_acc_pct and len(val_acc_pct) == len(train_acc_pct):
        ax.fill_between(epochs, train_acc_pct, val_acc_pct, alpha=0.2, color='gray', 
                       label='Gap (Overfitting Indicator)')
    
    # Adjust layout
    plt.tight_layout()
    
    # Convert to base64
    chart_image = fig_to_base64(fig)
    plt.close(fig)
    
    return chart_image

