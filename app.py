"""
Main Flask application entry point for Obesity Risk Prediction Web App
"""

from flask import Flask, render_template, request, redirect, url_for, session, flash, send_file, jsonify
import os
from werkzeug.utils import secure_filename
import logging
from datetime import datetime

from config import Config
from database.db_init import init_db, get_db
from utils.auth_utils import register_user, login_user, logout_user, is_logged_in
from utils.file_utils import allowed_file, save_uploaded_file, load_csv_file
from ml.train_models import train_all_models, get_algorithm_parameters
from ml.predict import make_prediction
from ml.evaluate_models import evaluate_all_models
from ml.feature_importance import get_feature_importance
from ml.model_visualizations import generate_training_metrics_chart, find_best_model
from eda.eda_analysis import perform_eda
from eda.descriptive_stats import get_descriptive_stats
from eda.visualizations import generate_visualizations
from utils.pdf_generator import generate_pdf_report

app = Flask(__name__)
app.config.from_object(Config)

# Ensure required directories exist BEFORE configuring logging
os.makedirs('uploads', exist_ok=True)
os.makedirs('reports', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('logs', exist_ok=True)
os.makedirs('results', exist_ok=True)

# Helper functions to save/load large results to files instead of session (to avoid ERR_RESPONSE_HEADERS_TOO_BIG)
import json
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    """Custom encoder for numpy data types"""
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def save_results_to_file(data, filename):
    """Save large data to a JSON file in the results folder"""
    try:
        path = os.path.join('results', filename)
        with open(path, 'w') as f:
            json.dump(data, f, cls=NumpyEncoder)
        return True
    except Exception as e:
        logging.error(f"Error saving results to file {filename}: {str(e)}")
        print(f"Error saving results to file {filename}: {str(e)}")
        return False

def load_results_from_file(filename):
    """Load large data from a JSON file in the results folder"""
    try:
        path = os.path.join('results', filename)
        if os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)
        return None
    except Exception as e:
        logging.error(f"Error loading results from file {filename}: {str(e)}")
        return None

# Configure logging (after directories are created)
logging.basicConfig(
    filename='logs/app.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Initialize database
try:
    init_db()
except Exception as e:
    logging.error(f"Database initialization error: {str(e)}")
    print(f"Warning: Database initialization error: {str(e)}")


@app.route('/')
def index():
    """Redirect to home if logged in, otherwise to login"""
    if is_logged_in():
        return redirect(url_for('home'))
    return redirect(url_for('login'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    """User login page"""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if login_user(username, password):
            flash('Login successful!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Invalid username or password', 'error')
    
    return render_template('auth/login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    """User registration page"""
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        if password != confirm_password:
            flash('Passwords do not match', 'error')
        elif register_user(username, email, password):
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
        else:
            flash('Username or email already exists', 'error')
    
    return render_template('auth/register.html')


@app.route('/logout')
def logout():
    """User logout"""
    logout_user()
    flash('Logged out successfully', 'info')
    return redirect(url_for('login'))


@app.route('/home')
def home():
    """Dashboard home page"""
    if not is_logged_in():
        return redirect(url_for('login'))
    return render_template('dashboard/home.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    """Upload dataset page"""
    if not is_logged_in():
        return redirect(url_for('login'))
    
    table_data = None
    num_rows = 0
    num_cols = 0
    training_results = None
    evaluation_results = None
    training_chart_image = None
    best_model_info = None
    
    if request.method == 'POST':
        # Check if it's a training request
        if 'train_model' in request.form:
            dataset = session.get('current_dataset')
            if not dataset:
                flash('Please upload a dataset first', 'warning')
                return redirect(request.url)
            
            try:
                df = load_csv_file(dataset)
                target_column = request.form.get('target_column', 'Obesity Category')
                
                flash('Training started. This may take a while...', 'info')
                
                # Train all models
                results = train_all_models(df, target_column)
                
                # Find best model
                best_model = find_best_model(results)
                
                # Evaluate models (includes ROC images)
                eval_results = evaluate_all_models(df, target_column)
                
                # Generate training metrics chart
                training_chart = generate_training_metrics_chart(results)
                
                # Prepare session-safe evaluation results (remove large images)
                session_safe_eval = {}
                for m_name, m_metrics in eval_results.items():
                    safe_m = m_metrics.copy()
                    if 'roc_curve_image' in safe_m:
                        del safe_m['roc_curve_image']
                    session_safe_eval[m_name] = safe_m
                
                # Save results to local JSON files instead of session
                save_results_to_file(results, 'latest_training_results.json')
                save_results_to_file(session_safe_eval, 'latest_evaluation_results.json')
                save_results_to_file(best_model, 'latest_best_model.json')
                
                # Clear session of large legacy data
                session.pop('training_results', None)
                session.pop('evaluation_results', None)
                session.pop('training_chart', None)
                session.pop('best_model', None)
                
                # Just store a flag that results exist
                session['results_available'] = True
                
                training_results = results
                evaluation_results = eval_results
                training_chart_image = training_chart
                best_model_info = best_model
                
                flash('Training completed successfully!', 'success')
                
                # Reload dataset for preview
                df = load_csv_file(dataset)
                num_rows = len(df)
                num_cols = len(df.columns)
                df_preview = df.head(50)
                table_data = df_preview.to_html(classes='data-preview-table', table_id='previewTable', escape=False, index=False)
                
            except Exception as e:
                flash(f'Training error: {str(e)}', 'error')
                logging.error(f'Training error: {str(e)}')
                # Still show dataset if available
                dataset = session.get('current_dataset')
                if dataset:
                    try:
                        df = load_csv_file(dataset)
                        num_rows = len(df)
                        num_cols = len(df.columns)
                        df_preview = df.head(50)
                        table_data = df_preview.to_html(classes='data-preview-table', table_id='previewTable', escape=False, index=False)
                    except:
                        pass
        
        # Check if it's a file upload request
        elif 'file' in request.files:
            if 'file' not in request.files:
                flash('No file selected', 'error')
                return redirect(request.url)
            
            file = request.files['file']
            if file.filename == '':
                flash('No file selected', 'error')
                return redirect(request.url)
            
            if file and allowed_file(file.filename):
                try:
                    filename = save_uploaded_file(file)
                except Exception as e:
                    logging.error(f'Error saving uploaded file: {str(e)}')
                    flash(f'Error saving file: {str(e)}', 'error')
                    return redirect(request.url)
                session['current_dataset'] = filename
                flash(f'File {filename} uploaded successfully!', 'success')
                
                # Clear existing results files when a new dataset is uploaded
                for f in ['latest_training_results.json', 'latest_evaluation_results.json', 'latest_best_model.json']:
                    path = os.path.join('results', f)
                    if os.path.exists(path):
                        os.remove(path)
                session.pop('results_available', None)
                
                # Load and preview the dataset
                try:
                    df = load_csv_file(filename)
                    num_rows = len(df)
                    num_cols = len(df.columns)
                    # Limit preview to first 50 rows
                    df_preview = df.head(50)
                    # Convert to HTML table
                    table_data = df_preview.to_html(classes='data-preview-table', table_id='previewTable', escape=False, index=False)
                except Exception as e:
                    flash(f'Error loading file: {str(e)}', 'error')
                    logging.error(f'Error loading file: {str(e)}')
            else:
                flash('Invalid file type. Please upload a CSV, XLS, or XLSX file.', 'error')
    
    # Load existing dataset if available
    dataset = session.get('current_dataset')
    if dataset and not table_data:
        try:
            df = load_csv_file(dataset)
            num_rows = len(df)
            num_cols = len(df.columns)
            df_preview = df.head(50)
            table_data = df_preview.to_html(classes='data-preview-table', table_id='previewTable', escape=False, index=False)
        except Exception as e:
            logging.error(f'Error loading dataset: {str(e)}')
    
    # Load training results if available from local files instead of session
    if not training_results:
        training_results = load_results_from_file('latest_training_results.json')
    if not evaluation_results:
        evaluation_results = load_results_from_file('latest_evaluation_results.json')
    if not best_model_info:
        best_model_info = load_results_from_file('latest_best_model.json')
    
    # Regenerate chart image if results exist
    if training_results and not locals().get('training_chart_image'):
        training_chart_image = generate_training_metrics_chart(training_results)
    
    # Regenerate ROC images for evaluation results if missing
    if evaluation_results and dataset:
        try:
            missing_images = any('roc_curve_image' not in metrics for metrics in evaluation_results.values() if 'error' not in metrics)
            if missing_images:
                df = load_csv_file(dataset)
                target_column = 'Obesity Category'
                full_eval = evaluate_all_models(df, target_column)
                evaluation_results = full_eval
        except Exception as e:
            logging.error(f"Error regenerating images: {str(e)}")
    
    # Final safety check: remove legacy session data to prevent ERR_RESPONSE_HEADERS_TOO_BIG
    session.pop('training_results', None)
    session.pop('evaluation_results', None)
    session.pop('training_chart', None)
    session.pop('best_model', None)
    
    return render_template('dashboard/upload.html', 
                         table_data=table_data, 
                         num_rows=num_rows, 
                         num_cols=num_cols,
                         training_results=training_results,
                         evaluation_results=evaluation_results,
                         training_chart=training_chart_image,
                         best_model=best_model_info)


@app.route('/eda')
def eda():
    """Exploratory Data Analysis page"""
    if not is_logged_in():
        return redirect(url_for('login'))
    
    dataset = session.get('current_dataset')
    if not dataset:
        flash('Please upload a dataset first', 'warning')
        return redirect(url_for('upload'))
    
    try:
        df = load_csv_file(dataset)
        eda_results = perform_eda(df)
        stats = get_descriptive_stats(df)
        # Generate all EDA visualizations (histograms, box plots, correlation)
        from eda.visualizations import generate_all_eda_visualizations
        eda_charts = generate_all_eda_visualizations(df)
        # Generate 12 EDA triggering questions charts
        triggering_charts = {}
        try:
            from eda.triggering_questions import generate_eda_triggering_charts
            triggering_charts = generate_eda_triggering_charts(df)
        except Exception as e:
            logging.error(f'Error generating triggering questions charts: {str(e)}')
            import traceback
            logging.error(traceback.format_exc())
            flash(f'Warning: Some EDA charts could not be generated: {str(e)}', 'warning')
        
        return render_template('dashboard/eda.html', 
                             eda_results=eda_results, 
                             stats=stats,
                             shape=df.shape,
                             columns=list(df.columns),
                             eda_charts=eda_charts,
                             triggering_charts=triggering_charts)
    except Exception as e:
        flash(f'Error loading dataset: {str(e)}', 'error')
        return redirect(url_for('upload'))


@app.route('/visualization')
def visualization():
    """Data visualization page"""
    if not is_logged_in():
        return redirect(url_for('login'))
    
    dataset = session.get('current_dataset')
    if not dataset:
        flash('Please upload a dataset first', 'warning')
        return redirect(url_for('upload'))
    
    try:
        df = load_csv_file(dataset)
        logging.info(f'Loading dataset for visualization: {dataset}, Shape: {df.shape}, Columns: {list(df.columns)}')
        
        # Generate 15 visualization questions charts
        visualization_charts = {}
        try:
            from eda.visualization_questions import generate_visualization_questions
            visualization_charts = generate_visualization_questions(df)
            logging.info(f'Generated {len(visualization_charts)} visualization charts')
            
            if not visualization_charts or len(visualization_charts) == 0:
                flash('No visualization charts could be generated. Please check if your dataset contains the required columns (especially "Obesity Category").', 'warning')
                logging.warning(f'No charts generated. Dataset columns: {list(df.columns)}')
        except Exception as e:
            logging.error(f'Error generating visualization questions charts: {str(e)}')
            import traceback
            logging.error(traceback.format_exc())
            flash(f'Error generating visualization charts: {str(e)}. Check logs for details.', 'error')
        
        # Always render the template, even if charts are empty (so user can see the message)
        return render_template('dashboard/visualization.html', visualization_charts=visualization_charts)
    except Exception as e:
        flash(f'Error loading dataset: {str(e)}', 'error')
        logging.error(f'Visualization error: {str(e)}')
        import traceback
        logging.error(traceback.format_exc())
        return redirect(url_for('upload'))


@app.route('/train', methods=['GET', 'POST'])
def train():
    """Model training page"""
    if not is_logged_in():
        return redirect(url_for('login'))
    
    dataset = session.get('current_dataset')
    if not dataset:
        flash('Please upload a dataset first', 'warning')
        return redirect(url_for('upload'))
    
    training_results = None
    evaluation_results = None
    training_chart_image = None
    best_model_info = None
    table_data = None
    num_rows = 0
    num_cols = 0
    
    # Load dataset preview
    try:
        df = load_csv_file(dataset)
        num_rows = len(df)
        num_cols = len(df.columns)
        df_preview = df.head(50)
        table_data = df_preview.to_html(classes='data-preview-table', table_id='previewTable', escape=False, index=False)
    except Exception as e:
        logging.error(f'Error loading dataset: {str(e)}')
    
    # Always load existing training results from session first
    training_results = load_results_from_file('latest_training_results.json')
    evaluation_results = load_results_from_file('latest_evaluation_results.json')
    best_model_info = load_results_from_file('latest_best_model.json')
    
    # Regenerate images if needed
    if training_results and not locals().get('training_chart_image'):
        training_chart_image = generate_training_metrics_chart(training_results)
    
    if evaluation_results and dataset:
        try:
            # Check if any model is missing its image
            missing_images = any('roc_curve_image' not in metrics for metrics in evaluation_results.values() if 'error' not in metrics)
            if missing_images:
                df = load_csv_file(dataset)
                target_column = 'Obesity Category' # Default
                full_eval = evaluate_all_models(df, target_column)
                evaluation_results = full_eval
        except Exception as e:
            logging.error(f"Error regenerating images: {str(e)}")
    
    # Final safety check: remove legacy session data to prevent ERR_RESPONSE_HEADERS_TOO_BIG
    session.pop('training_results', None)
    session.pop('evaluation_results', None)
    session.pop('training_chart', None)
    session.pop('best_model', None)
    
    if request.method == 'POST':
        # Check if results already exist - if yes, use them instead of retraining
        if training_results and evaluation_results:
            flash('Using existing training results. To retrain, please clear session or upload a new dataset.', 'info')
        else:
            # Only train if no results exist
            try:
                df = load_csv_file(dataset)
                target_column = request.form.get('target_column', 'Obesity Category')
                
                flash('Training started. This may take a while...', 'info')
                
                # Train all models
                results = train_all_models(df, target_column)
                
                # Find best model
                best_model = find_best_model(results)
                
                # Evaluate models (includes ROC images)
                eval_results = evaluate_all_models(df, target_column)
                
                # Generate training metrics chart
                training_chart = generate_training_metrics_chart(results)
                
                # Prepare session-safe evaluation results (remove large images)
                session_safe_eval = {}
                for m_name, m_metrics in eval_results.items():
                    safe_m = m_metrics.copy()
                    if 'roc_curve_image' in safe_m:
                        del safe_m['roc_curve_image']
                    session_safe_eval[m_name] = safe_m
                
                # Save results to local JSON files instead of session
                save_results_to_file(results, 'latest_training_results.json')
                save_results_to_file(session_safe_eval, 'latest_evaluation_results.json')
                save_results_to_file(best_model, 'latest_best_model.json')
                
                # Clear session of large legacy data
                session.pop('training_results', None)
                session.pop('evaluation_results', None)
                session.pop('training_chart', None)
                session.pop('best_model', None)
                
                # Just store a flag that results exist
                session['results_available'] = True
                
                training_results = results
                evaluation_results = eval_results
                training_chart_image = training_chart
                best_model_info = best_model
                
                flash('Training completed successfully!', 'success')
            except Exception as e:
                flash(f'Training error: {str(e)}', 'error')
                logging.error(f'Training error: {str(e)}')
        
        # Reload dataset preview after training (or if using existing results)
        try:
            df = load_csv_file(dataset)
            num_rows = len(df)
            num_cols = len(df.columns)
            df_preview = df.head(50)
            table_data = df_preview.to_html(classes='data-preview-table', table_id='previewTable', escape=False, index=False)
        except Exception as e:
            logging.error(f'Error reloading dataset: {str(e)}')
    
    return render_template('dashboard/model_training.html', 
                         training_results=training_results,
                         evaluation_results=evaluation_results,
                         training_chart=training_chart_image,
                         best_model=best_model_info,
                         table_data=table_data,
                         num_rows=num_rows,
                         num_cols=num_cols)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Prediction page"""
    if not is_logged_in():
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        try:
            # Get form data
            input_data = {
                'Gender': request.form.get('Gender'),
                'Age': float(request.form.get('Age')),
                'Height': float(request.form.get('Height')),
                'Weight': float(request.form.get('Weight')),
                'family_history_with_overweight': request.form.get('family_history_with_overweight'),
                'FAVC': request.form.get('FAVC'),
                'FCVC': float(request.form.get('FCVC')),
                'NCP': float(request.form.get('NCP')),
                'CAEC': request.form.get('CAEC'),
                'SMOKE': request.form.get('SMOKE'),
                'CH2O': float(request.form.get('CH2O')),
                'SCC': request.form.get('SCC'),
                'FAF': float(request.form.get('FAF')),
                'TUE': float(request.form.get('TUE')),
                'CALC': request.form.get('CALC'),
                'MTRANS': request.form.get('MTRANS')
            }
            
            model_type = request.form.get('model_type', 'xgboost')
            prediction = make_prediction(input_data, model_type)

            # Compute BMI and simple BMI category for additional context
            try:
                height_m = input_data['Height']
                weight_kg = input_data['Weight']
                bmi = weight_kg / (height_m ** 2) if height_m > 0 else 0
            except Exception:
                bmi = 0

            if bmi < 18.5:
                bmi_category = 'Underweight'
            elif bmi < 25:
                bmi_category = 'Normal range'
            elif bmi < 30:
                bmi_category = 'Overweight'
            elif bmi < 35:
                bmi_category = 'Obesity Class I'
            elif bmi < 40:
                bmi_category = 'Obesity Class II'
            else:
                bmi_category = 'Obesity Class III'

            prediction['bmi'] = round(bmi, 1)
            prediction['bmi_category'] = bmi_category
            
            return render_template(
                'dashboard/prediction.html',
                prediction=prediction,
                input_data=input_data,
                model_type=model_type
            )
        except Exception as e:
            flash(f'Prediction error: {str(e)}', 'error')
            logging.error(f'Prediction error: {str(e)}')
    
    return render_template('dashboard/prediction.html')


@app.route('/report')
def report():
    """Generate and download PDF report"""
    if not is_logged_in():
        return redirect(url_for('login'))
    
    dataset = session.get('current_dataset')
    if not dataset:
        flash('Please upload a dataset first', 'warning')
        return redirect(url_for('upload'))
    
    try:
        df = load_csv_file(dataset)
        # Load from local files instead of session
        training_results = load_results_from_file('latest_training_results.json') or {}
        evaluation_results = load_results_from_file('latest_evaluation_results.json') or {}
        
        # If evaluation_results is missing ROC images, re-evaluate to get them for the report
        if evaluation_results:
            missing_images = any('roc_curve_image' not in metrics for metrics in evaluation_results.values() if 'error' not in metrics)
            if missing_images:
                evaluation_results = evaluate_all_models(df, 'Obesity Category')
        
        # Generate PDF report
        report_path = generate_pdf_report(df, training_results, evaluation_results)
        
        return send_file(report_path, as_attachment=True, 
                        download_name='obesity_prediction_report.pdf')
    except Exception as e:
        flash(f'Report generation error: {str(e)}', 'error')
        return redirect(url_for('home'))


@app.route('/api/feature_importance')
def api_feature_importance():
    """API endpoint for feature importance"""
    if not is_logged_in():
        return jsonify({'error': 'Not authenticated'}), 401
    
    dataset = session.get('current_dataset')
    if not dataset:
        return jsonify({'error': 'No dataset loaded'}), 400
    
    try:
        df = load_csv_file(dataset)
        importance = get_feature_importance(df, 'Obesity Category')
        return jsonify(importance)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("  Obesity Risk Prediction Web Application")
    print("=" * 60)
    print("  Starting Flask development server...")
    print("  Server will be available at: http://127.0.0.1:5000")
    print("  Press CTRL+C to stop the server")
    print("=" * 60 + "\n")
    
    try:
        app.run(debug=True, host='127.0.0.1', port=5000, use_reloader=False, threaded=True)
    except OSError as e:
        if "Address already in use" in str(e) or "address already in use" in str(e).lower():
            print("\nERROR: Port 5000 is already in use!")
            print("Please either:")
            print("  1. Stop the other application using port 5000")
            print("  2. Change the port in app.py (line 307) to a different number (e.g., 5001)")
        else:
            print(f"\nERROR starting server: {str(e)}")
        logging.error(f"Server startup error: {str(e)}")
    except Exception as e:
        print(f"\nERROR starting server: {str(e)}")
        logging.error(f"Server startup error: {str(e)}")
        import traceback
        traceback.print_exc()

