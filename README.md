# Obesity Risk Prediction Web Application

A comprehensive Flask-based web application for predicting obesity risk using multiple machine learning algorithms including XGBoost, Random Forest, SVM, ANN, TabNet, and Stacking ensemble methods.

## Features

- **User Authentication**: Secure login and registration system
- **Data Upload**: Upload and manage CSV datasets
- **Exploratory Data Analysis (EDA)**: Comprehensive data analysis and statistics
- **Data Visualization**: Interactive charts and graphs
- **Model Training**: Train multiple ML models simultaneously
- **Model Evaluation**: Detailed metrics and performance analysis
- **Prediction**: Make predictions on new data using trained models
- **PDF Reports**: Generate comprehensive PDF reports

## Installation

1. **Clone or navigate to the project directory:**
   ```bash
   cd obesity_ml_web_app
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Initialize the database:**
   The database will be automatically initialized on first run.

## Usage

1. **Run the application:**
   ```bash
   python app.py
   ```

2. **Access the application:**
   Open your browser and navigate to `http://localhost:5000`

3. **Register a new account:**
   - Click on "Register" or navigate to the registration page
   - Create a new account with username, email, and password

4. **Upload a dataset:**
   - After logging in, go to the "Upload" section
   - Upload a CSV file with your obesity dataset
   - The sample dataset is provided in the `uploads/sample_data.csv` file

5. **Explore your data:**
   - Navigate to "EDA" for exploratory data analysis
   - Check "Visualization" for charts and graphs

6. **Train models:**
   - Go to "Train Models" section
   - Specify the target column (default: "Obesity Category")
   - Click "Start Training" to train all models

7. **Make predictions:**
   - Use the "Predict" section to make predictions on new data
   - Fill in all required fields and select a model
   - View the prediction results with confidence scores

8. **Generate reports:**
   - Download comprehensive PDF reports from the "Report" section

## Dataset Format

The application expects a CSV file with the following columns:
- Gender
- Age
- Height
- Weight
- family_history_with_overweight
- FAVC (Frequent consumption of high caloric food)
- FCVC (Frequency of consumption of vegetables)
- NCP (Number of main meals)
- CAEC (Consumption of food between meals)
- SMOKE
- CH2O (Consumption of water daily)
- SCC (Calories consumption monitoring)
- FAF (Physical activity frequency)
- TUE (Time using technology devices)
- CALC (Consumption of alcohol)
- MTRANS (Transportation used)
- Obesity Category (target variable)

## Models Supported

1. **XGBoost**: Gradient boosting framework
2. **Random Forest**: Ensemble learning method
3. **SVM**: Support Vector Machine
4. **ANN**: Artificial Neural Network
5. **TabNet**: Deep learning model for tabular data
6. **Stacking**: Ensemble method combining multiple models

## Project Structure

```
obesity_ml_web_app/
├── app.py                       # Main Flask application
├── config.py                    # Configuration settings
├── requirements.txt             # Python dependencies
├── README.md                    # Project documentation
├── database/
│   ├── users.db                 # SQLite database
│   └── db_init.py               # Database initialization
├── uploads/                     # Uploaded datasets
├── reports/                     # Generated PDF reports
├── models/                      # Trained ML models
├── preprocessing/               # Data preprocessing modules
├── ml/                          # Machine learning modules
├── eda/                         # Exploratory data analysis
├── static/                      # Static files (CSS, JS, images)
├── templates/                   # HTML templates
└── utils/                       # Utility functions
```

## Configuration

Edit `config.py` to customize:
- Secret key for session management
- File upload limits
- Database path
- Model storage paths

## Requirements

See `requirements.txt` for a complete list of dependencies. Key packages include:
- Flask 3.0.0
- pandas 2.1.3
- scikit-learn 1.3.2
- xgboost 2.0.3
- tensorflow 2.15.0
- pytorch-tabnet 4.0
- reportlab 4.0.7

## Notes

- The application uses SQLite for user authentication
- Trained models are saved in the `models/` directory
- All file uploads are stored in the `uploads/` directory
- PDF reports are generated in the `reports/` directory
- Application logs are stored in `logs/app.log`

## License

This project is provided as-is for educational and research purposes.

## Support

For issues or questions, please refer to the project documentation or create an issue in the repository.

