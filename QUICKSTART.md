# Quick Start Guide

## Prerequisites
- Python 3.8 or higher
- pip package manager

## Setup Instructions

1. **Navigate to the project directory:**
   ```bash
   cd obesity_ml_web_app
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   ```
   
   Activate it:
   - Windows: `venv\Scripts\activate`
   - Linux/Mac: `source venv/bin/activate`

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   For a faster install (no TensorFlow/TabNet; works on Python 3.13):
   ```bash
   pip install -r requirements-minimal.txt
   ```

   Note: Full `requirements.txt` may take longer (TensorFlow is skipped on Python 3.13; TabNet is optional).

4. **Run the application:**
   ```bash
   python app.py
   ```

5. **Access the application:**
   Open your web browser and go to: `http://localhost:5000`

## First Steps

1. **Register an account:**
   - Click on "Register" 
   - Create a username, email, and password
   - Log in with your credentials

2. **Upload a dataset:**
   - Navigate to "Upload"
   - Upload the provided `sample_data.csv` file or your own dataset
   - The dataset should follow the format described in README.md

3. **Explore the data:**
   - Go to "EDA" to see dataset statistics
   - Check "Visualization" for charts

4. **Train models:**
   - Go to "Train Models"
   - Click "Start Training"
   - Wait for all models to train (this may take several minutes)

5. **Make predictions:**
   - Use the "Predict" section
   - Fill in the form with patient data
   - Select a model and get predictions

6. **Generate reports:**
   - Download a PDF report from the "Report" section

## Troubleshooting

- **Import errors:** Make sure you're in the `obesity_ml_web_app` directory when running the app
- **Port already in use:** Change the port in `app.py` (e.g. in `app.run(...)`) from 5000 to another port
- **Model training fails:** Ensure you have enough memory and the dataset is properly formatted
- **TabNet errors:** TabNet requires PyTorch which may have platform-specific installation issues

## Notes

- The application will create necessary directories (database, models, reports, logs) automatically
- Models are saved after training for reuse
- The sample dataset is already included in the `uploads/` folder

