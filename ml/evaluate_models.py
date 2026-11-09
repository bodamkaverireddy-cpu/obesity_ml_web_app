"""
Model evaluation utilities (accuracy, confusion matrix, etc.)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                            confusion_matrix, classification_report, roc_auc_score, 
                            log_loss)
from sklearn.preprocessing import LabelBinarizer, label_binarize, LabelEncoder
import joblib
import os
from config import Config
from preprocessing.encoder import encode_categorical_features, load_encoders
from preprocessing.scaler import scale_features, load_scaler
from preprocessing.data_cleaning import handle_missing_values, remove_outliers
from ml.model_visualizations import generate_roc_auc_curves

def load_model(model_name):
    """Load a trained model"""
    model_name_lower = model_name.lower().replace(' ', '_')
    
    if model_name_lower == 'ann':
        try:
            from tensorflow import keras
            model_path = os.path.join(Config.MODELS_FOLDER, 'ann_model.h5')
            if os.path.exists(model_path):
                return keras.models.load_model(model_path)
            else:
                raise FileNotFoundError("ANN model file not found")
        except ImportError:
            raise ImportError("TensorFlow not available")
    elif model_name_lower == 'tabnet':
        try:
            from pytorch_tabnet.tab_model import TabNetClassifier
            model = TabNetClassifier()
            model_path = os.path.join(Config.MODELS_FOLDER, 'tabnet_model.zip')
            if os.path.exists(model_path):
                model.load_model(model_path)
                return model
            else:
                raise FileNotFoundError("TabNet model file not found")
        except ImportError:
            raise ImportError("TabNet not available")
    else:
        # Map model names to file names
        model_file_map = {
            'xgboost': 'xgboost_model.pkl',
            'random_forest': 'random_forest_model.pkl',
            'svm': 'svm_model.pkl',
            'stacking': 'stacking_model.pkl'
        }
        
        file_name = model_file_map.get(model_name_lower, f'{model_name_lower}_model.pkl')
        model_path = os.path.join(Config.MODELS_FOLDER, file_name)
        
        if os.path.exists(model_path):
            return joblib.load(model_path)
        else:
            raise FileNotFoundError(f"Model file not found: {model_path}")


def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate a single model with comprehensive metrics"""
    try:
        model_name_lower = model_name.lower().replace(' ', '_')
        
        # Predict
        if model_name_lower == 'ann':
            X_test_array = np.array(X_test) if isinstance(X_test, pd.DataFrame) else X_test
            y_pred_proba = model.predict(X_test_array, verbose=0)
            y_pred = np.argmax(y_pred_proba, axis=1)
        elif model_name_lower == 'tabnet':
            X_test_array = np.array(X_test) if isinstance(X_test, pd.DataFrame) else X_test
            y_pred = model.predict(X_test_array)
            try:
                y_pred_proba = model.predict_proba(X_test_array)
            except:
                y_pred_proba = None
        else:
            y_pred = model.predict(X_test)
            try:
                y_pred_proba = model.predict_proba(X_test)
            except:
                y_pred_proba = None
        
        # Calculate basic metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Calculate confusion matrix percentages (percentage of row total)
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        cm_percentage = (cm.astype('float') / row_sums * 100).round(2)
        
        # Get class labels - sklearn's confusion_matrix uses sorted unique classes
        unique_classes = np.unique(np.concatenate([y_test, y_pred]))
        unique_classes_sorted = np.sort(unique_classes)
        
        # Calculate TP, FP, TN, FN for each class (for multi-class, this is one-vs-rest)
        cm_metrics = {}
        total_samples = len(y_test)
        
        for i, class_label in enumerate(unique_classes_sorted):
            # For multi-class: TP = diagonal, FP = column sum - TP, FN = row sum - TP, TN = total - TP - FP - FN
            if i < cm.shape[0] and i < cm.shape[1]:
                tp = int(cm[i, i])
                fp = int(cm[:, i].sum() - tp)
                fn = int(cm[i, :].sum() - tp)
                tn = int(total_samples - tp - fp - fn)
            else:
                tp = fp = fn = tn = 0
            
            # Calculate percentages
            tp_percentage = (tp / total_samples * 100) if total_samples > 0 else 0
            fp_percentage = (fp / total_samples * 100) if total_samples > 0 else 0
            fn_percentage = (fn / total_samples * 100) if total_samples > 0 else 0
            tn_percentage = (tn / total_samples * 100) if total_samples > 0 else 0
            
            cm_metrics[int(class_label)] = {
                'tp': int(tp),
                'fp': int(fp),
                'tn': int(tn),
                'fn': int(fn),
                'tp_percentage': round(float(tp_percentage), 2),
                'fp_percentage': round(float(fp_percentage), 2),
                'fn_percentage': round(float(fn_percentage), 2),
                'tn_percentage': round(float(tn_percentage), 2)
            }
        
        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        
        # ROC-AUC (one-vs-rest) - requires probability predictions
        roc_auc = None
        log_loss_score = None
        
        if y_pred_proba is not None:
            try:
                # Get unique classes and binarize
                classes = np.unique(np.concatenate([y_test, y_pred]))
                if len(classes) > 1:
                    y_test_binarized = label_binarize(y_test, classes=classes)
                    
                    # Handle binary case
                    if y_test_binarized.shape[1] == 1:
                        y_test_binarized = np.hstack([1 - y_test_binarized, y_test_binarized])
                    
                    # Ensure y_pred_proba has correct shape
                    if y_pred_proba.shape[1] != y_test_binarized.shape[1]:
                        # Adjust if needed
                        if len(classes) == 2 and y_pred_proba.shape[1] == 1:
                            y_pred_proba = np.hstack([1 - y_pred_proba, y_pred_proba])
                    
                    if y_pred_proba.shape[1] == y_test_binarized.shape[1]:
                        roc_auc = roc_auc_score(y_test_binarized, y_pred_proba, average='macro', multi_class='ovr')
            except Exception as e:
                roc_auc = None
            
            # Log Loss
            try:
                # Convert y_test to numeric if needed
                y_test_numeric = y_test
                if not isinstance(y_test_numeric[0], (int, np.integer)):
                    le = LabelEncoder()
                    y_test_numeric = le.fit_transform(y_test)
                    # Adjust probabilities if needed
                    if y_pred_proba.shape[1] != len(le.classes_):
                        # Use argmax to get class indices and create one-hot
                        class_indices = np.argmax(y_pred_proba, axis=1)
                        y_pred_proba_adjusted = label_binarize(class_indices, classes=range(len(le.classes_)))
                    else:
                        y_pred_proba_adjusted = y_pred_proba
                else:
                    y_pred_proba_adjusted = y_pred_proba
                
                log_loss_score = log_loss(y_test_numeric, y_pred_proba_adjusted)
            except Exception as e:
                log_loss_score = None
        
        # Convert numpy arrays to lists to ensure JSON serialization works
        result = {
            'accuracy': round(float(accuracy), 4),
            'precision': round(float(precision), 4),
            'recall': round(float(recall), 4),
            'f1_score': round(float(f1), 4),
            'confusion_matrix': cm.tolist() if hasattr(cm, 'tolist') else list(cm),
            'confusion_matrix_percentage': cm_percentage.tolist() if hasattr(cm_percentage, 'tolist') else [[float(x) for x in row] for row in cm_percentage],
            'confusion_matrix_metrics': cm_metrics,
            'class_labels': [int(x) for x in unique_classes_sorted.tolist()] if hasattr(unique_classes_sorted, 'tolist') else [int(x) for x in unique_classes_sorted],
            'classification_report': report
        }
        
        if roc_auc is not None:
            result['roc_auc'] = round(float(roc_auc), 4)
        else:
            result['roc_auc'] = 'N/A'
        
        if log_loss_score is not None:
            result['log_loss'] = round(float(log_loss_score), 4)
        else:
            result['log_loss'] = 'N/A'
        
        # Generate ROC curve image
        try:
            roc_curve_image = generate_roc_auc_curves(model, X_test, y_test, model_name.replace('_', ' ').title())
            result['roc_curve_image'] = roc_curve_image
        except Exception as e:
            print(f"Error generating ROC curve image for {model_name}: {str(e)}")
            result['roc_curve_image'] = None
        
        return result
    except Exception as e:
        return {'error': str(e)}


def evaluate_all_models(df, target_column='Obesity Category'):
    """Evaluate all trained models"""
    try:
        # Apply the same cleaning steps used during training
        df_processed = df.copy()
        df_processed = df_processed.drop_duplicates()
        df_processed = handle_missing_values(df_processed, strategy='median')
        df_processed = remove_outliers(df_processed, method='iqr')

        # Prepare data using one-hot encoding (same as training)
        df_encoded, encoders = encode_categorical_features(
            df_processed, target_column, encode_type='onehot'
        )
        X = df_encoded.drop(columns=[target_column])
        y = df_encoded[target_column]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Load scaler and scale features
        try:
            scaler = load_scaler()
            X_test_scaled, _ = scale_features(
                X_test, scaler_type='standard', fit=False, scaler=scaler
            )
        except FileNotFoundError:
            # If scaler not found, scale on the fly
            X_test_scaled, _ = scale_features(
                X_test, scaler_type='standard', fit=True
            )

        # Extra safety: ensure there are no NaN/inf values before passing to models
        import numpy as np
        X_test_scaled = X_test_scaled.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Evaluate each model (excluding ANN and TabNet)
        models_to_evaluate = ['xgboost', 'random_forest', 'svm', 'stacking']
        results = {}
        
        for model_name in models_to_evaluate:
            try:
                model = load_model(model_name)
                results[model_name] = evaluate_model(model, X_test_scaled, y_test, model_name)
            except FileNotFoundError:
                results[model_name] = {'error': 'Model file not found - model not trained yet'}
            except ImportError as e:
                results[model_name] = {'error': f'Library not available: {str(e)}'}
            except Exception as e:
                results[model_name] = {'error': f'Error loading/evaluating model: {str(e)}'}
        
        return results
    except Exception as e:
        raise Exception(f"Error evaluating models: {str(e)}")
