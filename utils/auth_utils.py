"""
Authentication utilities for user login, registration, and session management
"""

from flask import session
import hashlib
from database.db_init import get_db


def hash_password(password):
    """Hash password using SHA256"""
    return hashlib.sha256(password.encode()).hexdigest()


def register_user(username, email, password):
    """Register a new user"""
    try:
        conn = get_db()
        cursor = conn.cursor()
        
        # Check if username or email already exists
        cursor.execute('SELECT * FROM users WHERE username = ? OR email = ?', 
                      (username, email))
        if cursor.fetchone():
            conn.close()
            return False
        
        # Insert new user
        hashed_password = hash_password(password)
        cursor.execute('''
            INSERT INTO users (username, email, password)
            VALUES (?, ?, ?)
        ''', (username, email, hashed_password))
        
        conn.commit()
        conn.close()
        return True
    except Exception:
        return False


def login_user(username, password):
    """Authenticate user and create session"""
    try:
        conn = get_db()
        cursor = conn.cursor()
        
        hashed_password = hash_password(password)
        cursor.execute('''
            SELECT * FROM users 
            WHERE username = ? AND password = ?
        ''', (username, hashed_password))
        
        user = cursor.fetchone()
        conn.close()
        
        if user:
            session['user_id'] = user['id']
            session['username'] = user['username']
            session.permanent = True
            return True
        return False
    except Exception:
        return False


def logout_user():
    """Clear user session"""
    session.clear()


def is_logged_in():
    """Check if user is logged in"""
    return 'user_id' in session

