"""
Model training logic for various ML algorithms
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except (ImportError, Exception) as e:
    xgb = None
    XGBOOST_AVAILABLE = False
from imblearn.over_sampling import SMOTE
try:
    import tensorflow as tf
    from tensorflow import keras
    from keras.models import Sequential
    from keras.layers import Dense, Dropout
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
try:
    from pytorch_tabnet.tab_model import TabNetClassifier
    TABNET_AVAILABLE = True
except ImportError:
    TABNET_AVAILABLE = False
import joblib
import os
from config import Config
from preprocessing.encoder import encode_categorical_features, save_encoders
from preprocessing.scaler import scale_features, save_scaler
from preprocessing.data_cleaning import handle_missing_values, remove_outliers

os.makedirs(Config.MODELS_FOLDER, exist_ok=True)


def prepare_data(df, target_column='Obesity Category'):
    """
    Prepare data for training:
    - Clean data (missing values, outliers, duplicates)
    - Encode categorical variables
    - Train/test split
    - Balance training data using SMOTE
    - Scale features and persist encoders/scaler
    """
    # Work on a copy to avoid mutating original
    df_processed = df.copy()

    # Basic cleaning: drop exact duplicates
    df_processed = df_processed.drop_duplicates()

    # Handle missing values (median for numeric, mode for categorical)
    df_processed = handle_missing_values(df_processed, strategy='median')

    # Remove strong outliers from numeric columns using IQR
    df_processed = remove_outliers(df_processed, method='iqr')

    # Encode categorical features using one-hot encoding
    df_encoded, encoders = encode_categorical_features(
        df_processed, target_column, encode_type='onehot'
    )
    
    # Separate features and target
    X = df_encoded.drop(columns=[target_column])
    y = df_encoded[target_column]
    
    # Split data (keep a separate, untouched test set)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Apply SMOTE only on the training data to handle class imbalance
    try:
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    except Exception:
        # Fallback to original training data if SMOTE fails for any reason
        X_train_resampled, y_train_resampled = X_train, y_train
    
    # Scale features (fit scaler on resampled training data)
    X_train_scaled, scaler = scale_features(
        X_train_resampled, scaler_type='standard', fit=True
    )
    X_test_scaled, _ = scale_features(
        X_test, scaler_type='standard', fit=False, scaler=scaler
    )

    # As an extra safety step, replace any remaining NaN / inf values after scaling
    # Some preprocessing combinations can still produce NaNs that break SVM/Stacking
    X_train_scaled = X_train_scaled.replace([np.inf, -np.inf], np.nan).fillna(0)
    X_test_scaled = X_test_scaled.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Save encoders and scaler
    save_encoders(encoders)
    save_scaler(scaler)
    
    return X_train_scaled, X_test_scaled, y_train_resampled, y_test, encoders, scaler


def train_xgboost(X_train, y_train, X_test, y_test):
    """Train XGBoost model with hyperparameter tuning and training history tracking"""
    if not XGBOOST_AVAILABLE:
        return {'model': 'XGBoost', 'accuracy': 0.0, 'status': 'error: XGBoost not available (install or fix DLL/OpenMP)'}
    try:
        # Hyperparameter search space (kept compact for speed)
        param_dist = {
            "n_estimators": [100, 200, 300],
            "max_depth": [4, 6, 8],
            "learning_rate": [0.01, 0.05, 0.1],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
        }

        base_model = xgb.XGBClassifier(
            random_state=42,
            objective="multi:softprob",
            eval_metric="mlogloss",
        )

        search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_dist,
            n_iter=15,
            scoring="accuracy",
            cv=3,
            random_state=42,
            n_jobs=-1,
            verbose=0,
        )

        search.fit(X_train, y_train)
        model = search.best_estimator_
        
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        train_accuracy = accuracy_score(y_train, y_pred_train)
        test_accuracy = accuracy_score(y_test, y_pred_test)
        
        # Save model
        model_path = os.path.join(Config.MODELS_FOLDER, 'xgboost_model.pkl')
        joblib.dump(model, model_path)
        
        return {
            'model': 'XGBoost', 
            'accuracy': float(test_accuracy), 
            'train_accuracy': float(train_accuracy),
            'status': 'trained',
            'best_params': search.best_params_
        }
    except Exception as e:
        return {'model': 'XGBoost', 'accuracy': 0.0, 'status': f'error: {str(e)}'}


def train_random_forest(X_train, y_train, X_test, y_test):
    """Train Random Forest Classifier model with hyperparameter tuning"""
    try:
        param_dist = {
            "n_estimators": [100, 200, 300],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2"],
        }

        base_model = RandomForestClassifier(random_state=42, n_jobs=-1)

        search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_dist,
            n_iter=20,
            scoring="accuracy",
            cv=3,
            random_state=42,
            n_jobs=-1,
            verbose=0,
        )

        search.fit(X_train, y_train)
        model = search.best_estimator_
        
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        train_accuracy = accuracy_score(y_train, y_pred_train)
        test_accuracy = accuracy_score(y_test, y_pred_test)
        
        # Save model
        model_path = os.path.join(Config.MODELS_FOLDER, 'random_forest_model.pkl')
        joblib.dump(model, model_path)
        
        return {
            'model': 'Random Forest', 
            'accuracy': float(test_accuracy),
            'train_accuracy': float(train_accuracy),
            'status': 'trained',
            'best_params': search.best_params_
        }
    except Exception as e:
        return {'model': 'Random Forest', 'accuracy': 0.0, 'status': f'error: {str(e)}'}


def train_svm(X_train, y_train, X_test, y_test):
    """Train Support Vector Machine (SVM) model with hyperparameter tuning"""
    try:
        param_dist = {
            "C": [0.1, 1, 10, 50],
            "gamma": ["scale", 0.01, 0.001],
            "kernel": ["rbf"],
        }

        base_model = SVC(probability=True, random_state=42)

        search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_dist,
            n_iter=10,
            scoring="accuracy",
            cv=3,
            random_state=42,
            n_jobs=-1,
            verbose=0,
        )

        search.fit(X_train, y_train)
        model = search.best_estimator_
        
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        train_accuracy = accuracy_score(y_train, y_pred_train)
        test_accuracy = accuracy_score(y_test, y_pred_test)
        
        # Save model
        model_path = os.path.join(Config.MODELS_FOLDER, 'svm_model.pkl')
        joblib.dump(model, model_path)
        
        return {
            'model': 'SVM', 
            'accuracy': float(test_accuracy),
            'train_accuracy': float(train_accuracy),
            'status': 'trained',
            'best_params': search.best_params_
        }
    except Exception as e:
        return {'model': 'SVM', 'accuracy': 0.0, 'status': f'error: {str(e)}'}


def train_ann(X_train, y_train, X_test, y_test):
    """Train Artificial Neural Network (ANN) model"""
    if not TENSORFLOW_AVAILABLE:
        return {'model': 'ANN', 'accuracy': 0.0, 'status': 'error: TensorFlow not installed (not compatible with Python 3.13+)'}
    
    try:
        from tensorflow import keras
        from keras.models import Sequential
        from keras.layers import Dense, Dropout
        
        num_classes = len(np.unique(y_train))
        num_features = X_train.shape[1]
        
        model = Sequential([
            Dense(128, activation='relu', input_shape=(num_features,)),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dense(num_classes, activation='softmax')
        ])
        
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        # Convert to numpy arrays
        X_train_array = np.array(X_train) if isinstance(X_train, pd.DataFrame) else X_train
        X_test_array = np.array(X_test) if isinstance(X_test, pd.DataFrame) else X_test
        
        # Train model
        history = model.fit(X_train_array, y_train, epochs=50, batch_size=32, 
                           validation_split=0.2, verbose=0)
        
        # Evaluate
        loss, accuracy = model.evaluate(X_test_array, y_test, verbose=0)
        
        # Save model
        model_path = os.path.join(Config.MODELS_FOLDER, 'ann_model.h5')
        model.save(model_path)
        
        return {'model': 'ANN', 'accuracy': float(accuracy), 'status': 'trained'}
    except Exception as e:
        return {'model': 'ANN', 'accuracy': 0.0, 'status': f'error: {str(e)}'}


def train_tabnet(X_train, y_train, X_test, y_test):
    """Train TabNet model"""
    if not TABNET_AVAILABLE:
        return {'model': 'TabNet', 'accuracy': 0.0, 'status': 'error: TabNet not installed'}
    try:
        # Convert to numpy arrays
        X_train_array = np.array(X_train) if isinstance(X_train, pd.DataFrame) else X_train
        X_test_array = np.array(X_test) if isinstance(X_test, pd.DataFrame) else X_test
        
        # TabNet requires numpy arrays
        model = TabNetClassifier(
            n_d=64, n_a=64,
            n_steps=5,
            gamma=1.5,
            n_independent=2,
            n_shared=2,
            optimizer_params=dict(lr=2e-2),
            scheduler_params={"step_size":50, "gamma":0.9},
            verbose=0,
            seed=42
        )
        
        model.fit(X_train_array, y_train.values if isinstance(y_train, pd.Series) else y_train,
                 eval_set=[(X_test_array, y_test.values if isinstance(y_test, pd.Series) else y_test)],
                 eval_name=['test'],
                 max_epochs=50,
                 patience=10)
        
        # Predict
        y_pred = model.predict(X_test_array)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Save model
        model_path = os.path.join(Config.MODELS_FOLDER, 'tabnet_model.zip')
        model.save_model(model_path)
        
        return {'model': 'TabNet', 'accuracy': float(accuracy), 'status': 'trained'}
    except Exception as e:
        return {'model': 'TabNet', 'accuracy': 0.0, 'status': f'error: {str(e)}'}


def train_stacking(X_train, y_train, X_test, y_test):
    """Train Stacking Ensemble model with training history tracking during iterations"""
    try:
        # Base estimators: use XGBoost if available, otherwise RF + SVM only
        if XGBOOST_AVAILABLE:
            base_estimators = [
                ('xgb', xgb.XGBClassifier(random_state=42, n_estimators=50, max_depth=5)),
                ('rf', RandomForestClassifier(n_estimators=50, random_state=42, max_depth=8)),
                ('svm', SVC(kernel='rbf', probability=True, random_state=42))
            ]
        else:
            base_estimators = [
                ('rf', RandomForestClassifier(n_estimators=50, random_state=42, max_depth=8)),
                ('svm', SVC(kernel='rbf', probability=True, random_state=42))
            ]
        # Split training data for validation
        from sklearn.model_selection import train_test_split as tts
        X_train_split, X_val_split, y_train_split, y_val_split = tts(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        
        # Track accuracy during training iterations
        train_acc_history = []
        val_acc_history = []
        iterations = []
        
        # Train with incremental training sample sizes to show learning curve
        train_sizes = np.linspace(0.1, 1.0, 10)
        
        for size in train_sizes:
            n_samples = int(len(X_train_split) * size)
            X_train_subset = X_train_split[:n_samples]
            y_train_subset = y_train_split[:n_samples]
            
            temp_stacking = VotingClassifier(estimators=base_estimators, voting='soft')
            temp_stacking.fit(X_train_subset, y_train_subset)
            
            # Calculate accuracies
            y_pred_train_temp = temp_stacking.predict(X_train_subset)
            y_pred_val_temp = temp_stacking.predict(X_val_split)
            train_acc = accuracy_score(y_train_subset, y_pred_train_temp)
            val_acc = accuracy_score(y_val_split, y_pred_val_temp)
            
            train_acc_history.append(float(train_acc))
            val_acc_history.append(float(val_acc))
            iterations.append(n_samples)
        
        # Train final model with all training data
        stacking_model = VotingClassifier(estimators=base_estimators, voting='soft')
        stacking_model.fit(X_train, y_train)
        
        y_pred_train = stacking_model.predict(X_train)
        y_pred_test = stacking_model.predict(X_test)
        train_accuracy = accuracy_score(y_train, y_pred_train)
        test_accuracy = accuracy_score(y_test, y_pred_test)
        
        # Save model
        model_path = os.path.join(Config.MODELS_FOLDER, 'stacking_model.pkl')
        joblib.dump(stacking_model, model_path)
        
        return {
            'model': 'Stacking', 
            'accuracy': float(test_accuracy),
            'train_accuracy': float(train_accuracy),
            'status': 'trained',
            'training_history': {
                'epochs': iterations,
                'train_accuracy': train_acc_history,
                'test_accuracy': val_acc_history
            }
        }
    except Exception as e:
        return {'model': 'Stacking', 'accuracy': 0.0, 'status': f'error: {str(e)}'}


def get_algorithm_parameters():
    """Get hyperparameters for all algorithms"""
    return {
        'XGBoost': {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'random_state': 42,
            'objective': 'multi:softprob',
            'eval_metric': 'mlogloss'
        },
        'Random Forest': {
            'n_estimators': 100,
            'max_depth': 10,
            'n_jobs': -1,
            'random_state': 42,
            'criterion': 'gini',
            'min_samples_split': 2,
            'min_samples_leaf': 1
        },
        'SVM': {
            'kernel': 'rbf',
            'C': 1.0,
            'gamma': 'scale',
            'probability': True,
            'random_state': 42,
            'decision_function_shape': 'ovr'
        },
        'Stacking Ensemble': {
            'voting': 'soft',
            'base_estimators': {
                'XGBoost': {
                    'n_estimators': 50,
                    'max_depth': 5,
                    'random_state': 42
                },
                'Random Forest': {
                    'n_estimators': 50,
                    'max_depth': 8,
                    'random_state': 42
                },
                'SVM': {
                    'kernel': 'rbf',
                    'probability': True,
                    'random_state': 42
                }
            }
        },
        'ANN': {
            'layers': [
                {'units': 128, 'activation': 'relu', 'dropout': 0.3},
                {'units': 64, 'activation': 'relu', 'dropout': 0.3},
                {'units': 32, 'activation': 'relu', 'dropout': 0.0}
            ],
            'optimizer': 'adam',
            'loss': 'sparse_categorical_crossentropy',
            'epochs': 50,
            'batch_size': 32,
            'validation_split': 0.2
        }
    }


def train_all_models(df, target_column='Obesity Category'):
    """Train all models and return results"""
    try:
        # Prepare data
        X_train, X_test, y_train, y_test, encoders, scaler = prepare_data(df, target_column)
        
        results = []
        
        # Train all models in order (excluding ANN and TabNet)
        print("Training XGBoost...")
        results.append(train_xgboost(X_train, y_train, X_test, y_test))
        
        print("Training Random Forest...")
        results.append(train_random_forest(X_train, y_train, X_test, y_test))
        
        print("Training SVM...")
        results.append(train_svm(X_train, y_train, X_test, y_test))
        
        print("Training Stacking Ensemble...")
        results.append(train_stacking(X_train, y_train, X_test, y_test))
        
        return results
    except Exception as e:
        raise Exception(f"Error training models: {str(e)}")
