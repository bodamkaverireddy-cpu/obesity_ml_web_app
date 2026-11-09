"""
Configuration settings for the Flask application
"""

import os
from datetime import timedelta

basedir = os.path.abspath(os.path.dirname(__file__))


class Config:
    """Application configuration class"""
    
    # Flask secret key for session management
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    
    # Database configuration
    DATABASE_PATH = os.path.join(basedir, 'database', 'users.db')
    
    # File upload configuration
    UPLOAD_FOLDER = os.path.join(basedir, 'uploads')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    ALLOWED_EXTENSIONS = {'csv', 'xls', 'xlsx'}
    
    # Reports folder
    REPORTS_FOLDER = os.path.join(basedir, 'reports')
    
    # Models folder
    MODELS_FOLDER = os.path.join(basedir, 'models')
    
    # Session configuration
    PERMANENT_SESSION_LIFETIME = timedelta(hours=2)
    SESSION_COOKIE_SECURE = False  # Set to True in production with HTTPS
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'

