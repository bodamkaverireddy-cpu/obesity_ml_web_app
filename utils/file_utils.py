"""
File handling utilities for uploads and CSV/Excel processing
"""

import os
from werkzeug.utils import secure_filename
import pandas as pd
from config import Config


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS


def save_uploaded_file(file):
    """Save uploaded file to uploads folder"""
    filename = secure_filename(file.filename)
    filepath = os.path.join(Config.UPLOAD_FOLDER, filename)
    file.save(filepath)
    return filename


def load_csv_file(filename):
    """Load CSV or Excel file into pandas DataFrame"""
    filepath = os.path.join(Config.UPLOAD_FOLDER, filename)
    file_ext = filename.rsplit('.', 1)[1].lower()
    
    if file_ext == 'csv':
        return pd.read_csv(filepath)
    elif file_ext == 'xlsx':
        return pd.read_excel(filepath, engine='openpyxl')
    elif file_ext == 'xls':
        # Try with xlrd engine (requires xlrd<2.0 for .xls support)
        try:
            return pd.read_excel(filepath, engine='xlrd')
        except Exception:
            # Fallback: try without specifying engine
            return pd.read_excel(filepath)
    else:
        raise ValueError(f"Unsupported file format: {file_ext}")

