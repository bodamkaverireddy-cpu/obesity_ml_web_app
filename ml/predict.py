"""
Prediction logic for making predictions on new data
"""

import pandas as pd
import numpy as np
import joblib
import os
from config import Config
from preprocessing.encoder import load_encoders
from preprocessing.scaler import load_scaler, scale_features
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


def _get_category_insights(category_label: str):
    """
    Map prediction label to human-friendly risk interpretation and recommendations.
    """
    label = (category_label or "").strip()
    insights_map = {
        "Insufficient_Weight": {
            "risk_level": "Nutritional risk (underweight)",
            "interpretation": (
                "Current weight is below the typical healthy range for adults. "
                "This can be associated with nutrient deficiencies and lower muscle mass."
            ),
            "recommendations": [
                "Discuss potential causes of low weight with a clinician or dietitian.",
                "Focus on energy-dense, nutrient-rich foods to reach a healthier weight.",
                "Monitor unintentional weight loss, fatigue, or other symptoms."
            ],
        },
        "Normal_Weight": {
            "risk_level": "Low obesity risk",
            "interpretation": (
                "Weight is within the expected healthy range given the observed lifestyle factors."
            ),
            "recommendations": [
                "Maintain current eating patterns and regular physical activity.",
                "Continue monitoring screen time and sedentary behaviour.",
                "Schedule periodic health checks to track weight and metabolic markers."
            ],
        },
        "Overweight_Level_I": {
            "risk_level": "Mildly elevated obesity risk",
            "interpretation": (
                "Findings suggest early excess body weight. Without lifestyle changes this can "
                "progress to obesity and increase cardiometabolic risk."
            ),
            "recommendations": [
                "Increase weekly physical activity (aerobic + strength training).",
                "Gradually reduce high-calorie snacks and sugary drinks.",
                "Set a realistic short-term weight reduction goal (for example 5% of body weight)."
            ],
        },
        "Overweight_Level_II": {
            "risk_level": "Moderately elevated obesity risk",
            "interpretation": (
                "There is a moderate excess of body weight. This category is associated with an "
                "increased likelihood of developing hypertension, diabetes and joint problems."
            ),
            "recommendations": [
                "Work with a clinician or dietitian on a structured weight‑management plan.",
                "Target at least 150–300 minutes of moderate activity per week.",
                "Limit late‑night snacking and highly processed, energy-dense foods."
            ],
        },
        "Obesity_Type_I": {
            "risk_level": "High obesity risk (Class I)",
            "interpretation": (
                "Model indicates established obesity. Cardiometabolic risk is significantly "
                "higher compared to normal weight."
            ),
            "recommendations": [
                "Prioritise a medically supervised weight‑loss programme.",
                "Combine calorie reduction with increased daily activity and behaviour change support.",
                "Screen for associated conditions (blood pressure, glucose, lipids, sleep apnoea)."
            ],
        },
        "Obesity_Type_II": {
            "risk_level": "Very high obesity risk (Class II)",
            "interpretation": (
                "Body weight is well above the healthy range and strongly associated with "
                "cardiovascular, metabolic and musculoskeletal complications."
            ),
            "recommendations": [
                "Engage a multidisciplinary team (physician, dietitian, physical activity specialist).",
                "Consider intensive interventions including medical nutrition therapy and pharmacotherapy.",
                "Establish close follow‑up to monitor weight trend and comorbidities."
            ],
        },
        "Obesity_Type_III": {
            "risk_level": "Extremely high obesity risk (Class III)",
            "interpretation": (
                "Severe obesity is present with a very high probability of obesity‑related "
                "health complications."
            ),
            "recommendations": [
                "Urgent specialist referral for comprehensive obesity management is recommended.",
                "Discuss advanced treatment options such as bariatric surgery where appropriate.",
                "Implement structured support for long‑term lifestyle change and monitoring."
            ],
        },
    }

    default = {
        "risk_level": "Obesity risk (unspecified)",
        "interpretation": (
            "A non‑standard obesity category was predicted. Please review the underlying data "
            "and consider clinical context."
        ),
        "recommendations": [
            "Verify that input data was entered correctly.",
            "Use this result together with clinical judgement, not as a stand‑alone diagnosis."
        ],
    }

    return insights_map.get(label, default)

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
                raise FileNotFoundError("ANN model not found. Please train the model first.")
        except ImportError:
            raise ImportError("TensorFlow not available. ANN model requires TensorFlow.")
    elif model_name_lower == 'tabnet':
        try:
            from pytorch_tabnet.tab_model import TabNetClassifier
            model = TabNetClassifier()
            model_path = os.path.join(Config.MODELS_FOLDER, 'tabnet_model.zip')
            if os.path.exists(model_path):
                model.load_model(model_path)
                return model
            else:
                raise FileNotFoundError("TabNet model not found. Please train the model first.")
        except ImportError:
            raise ImportError("TabNet not available")
    else:
        # Map model names to file names - only the 6 specified algorithms
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
            raise FileNotFoundError(f"Model '{model_name}' not found. Please train the model first.")


def make_prediction(input_data, model_name='xgboost'):
    """
    Make prediction on new data
    
    Args:
        input_data: Dictionary of input features
        model_name: Name of the model to use
    
    Returns:
        Prediction result dictionary
    """
    try:
        # Convert input to DataFrame
        df_input = pd.DataFrame([input_data])
        
        # Load encoders and scaler
        try:
            encoders = load_encoders()
            scaler = load_scaler()
        except FileNotFoundError as e:
            raise Exception(f"Preprocessing files not found. Please train models first: {str(e)}")
        
        # Apply the same encoding used during training (one-hot for features)
        processed_df = pd.DataFrame(index=df_input.index)
        for col in df_input.columns:
            if col == 'target':
                continue
            
            if col in encoders:
                encoder = encoders[col]
                
                # One-hot encoded feature
                if isinstance(encoder, OneHotEncoder):
                    # Replace unseen categories with the first known category to avoid transform errors
                    categories = encoder.categories_[0]
                    safe_values = df_input[[col]].astype(str)
                    safe_values.loc[~safe_values[col].isin(categories), col] = categories[0]
                    
                    encoded = encoder.transform(safe_values)
                    if hasattr(encoded, "toarray"):
                        encoded = encoded.toarray()
                    encoded_df = pd.DataFrame(
                        encoded,
                        columns=[f"{col}_{i}" for i in range(encoded.shape[1])],
                        index=df_input.index
                    )
                    processed_df = pd.concat([processed_df, encoded_df], axis=1)
                
                # Label-encoded feature
                elif hasattr(encoder, 'classes_'):
                    try:
                        processed_df[col] = encoder.transform(df_input[col].astype(str))
                    except ValueError:
                        fallback = encoder.classes_[0]
                        processed_df[col] = encoder.transform([fallback])[0]
                else:
                    processed_df[col] = df_input[col]
            else:
                processed_df[col] = df_input[col]
        
        # Ensure feature order matches the scaler (StandardScaler stores the order)
        if hasattr(scaler, 'feature_names_in_'):
            for missing in scaler.feature_names_in_:
                if missing not in processed_df.columns:
                    processed_df[missing] = 0
            processed_df = processed_df[scaler.feature_names_in_]
        
        # Fill any remaining missing values with zeros for safety
        processed_df = processed_df.fillna(0)
        
        # Scale features
        X_scaled, _ = scale_features(processed_df, scaler_type='standard', fit=False, scaler=scaler)
        
        # Load and predict with model
        model = load_model(model_name)
        model_name_lower = model_name.lower().replace(' ', '_')
        
        if model_name_lower == 'ann':
            X_array = np.array(X_scaled) if isinstance(X_scaled, pd.DataFrame) else X_scaled
            prediction_proba = model.predict(X_array, verbose=0)
            prediction_idx = np.argmax(prediction_proba, axis=1)[0]
            confidence = float(prediction_proba[0][prediction_idx])
        elif model_name_lower == 'tabnet':
            X_array = np.array(X_scaled) if isinstance(X_scaled, pd.DataFrame) else X_scaled
            prediction_proba = model.predict_proba(X_array)
            prediction_idx = np.argmax(prediction_proba, axis=1)[0]
            confidence = float(prediction_proba[0][prediction_idx])
        else:
            prediction_proba = model.predict_proba(X_scaled)
            prediction_idx = model.predict(X_scaled)[0]
            confidence = float(prediction_proba[0][prediction_idx])
        
        # Decode prediction
        if 'target' in encoders:
            le_target = encoders['target']
            prediction_label = le_target.inverse_transform([prediction_idx])[0]
        else:
            prediction_label = str(prediction_idx)
        
        # Get all probabilities (filter out invalid / NaN labels and sort by probability)
        if 'target' in encoders:
            le_target = encoders['target']
            class_names = le_target.classes_
            raw_items = []
            for i in range(len(class_names)):
                name = str(class_names[i])
                prob = float(prediction_proba[0][i])
                if name.lower() in ('nan', 'none') or np.isnan(prob):
                    continue
                raw_items.append((name, prob))
        else:
            raw_items = [
                (f'Class_{i}', float(prediction_proba[0][i]))
                for i in range(len(prediction_proba[0]))
                if not np.isnan(prediction_proba[0][i])
            ]

        # Sort descending by probability
        raw_items.sort(key=lambda x: x[1], reverse=True)
        probabilities = {name: prob for name, prob in raw_items}

        # High-level interpretation for business users
        insights = _get_category_insights(str(prediction_label))

        return {
            'prediction': str(prediction_label),
            'confidence': round(confidence * 100, 2),
            'probabilities': probabilities,
            'model_used': model_name,
            'risk_level': insights['risk_level'],
            'interpretation': insights['interpretation'],
            'recommendations': insights['recommendations'],
            'top_classes': [name for name, _ in raw_items[:3]],
        }
    except FileNotFoundError as e:
        raise Exception(f"Model not found. Please train models first: {str(e)}")
    except Exception as e:
        raise Exception(f"Prediction error: {str(e)}")
