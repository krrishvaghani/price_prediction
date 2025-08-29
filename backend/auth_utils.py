import sqlite3
import bcrypt
import logging
from typing import Tuple, Optional, Dict, Any

logger = logging.getLogger(__name__)

def init_db():
    """Initialize the SQLite database with users table"""
    try:
        conn = sqlite3.connect('auction_users.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing database: {e}")

def signup_user(name: str, email: str, password: str) -> Tuple[bool, str]:
    """Sign up a new user"""
    try:
        conn = sqlite3.connect('auction_users.db')
        cursor = conn.cursor()
        
        # Check if user already exists
        cursor.execute('SELECT id FROM users WHERE email = ?', (email,))
        if cursor.fetchone():
            conn.close()
            return False, "User with this email already exists"
        
        # Hash password
        password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        
        # Insert new user
        cursor.execute(
            'INSERT INTO users (name, email, password_hash) VALUES (?, ?, ?)',
            (name, email, password_hash.decode('utf-8'))
        )
        
        conn.commit()
        conn.close()
        return True, "User created successfully"
        
    except Exception as e:
        logger.error(f"Error in signup: {e}")
        return False, f"Error creating user: {str(e)}"

def login_user(email: str, password: str) -> Tuple[bool, Any]:
    """Login a user"""
    try:
        conn = sqlite3.connect('auction_users.db')
        cursor = conn.cursor()
        
        # Get user by email
        cursor.execute('SELECT id, name, email, password_hash FROM users WHERE email = ?', (email,))
        user = cursor.fetchone()
        
        if not user:
            conn.close()
            return False, "User not found"
        
        user_id, name, user_email, password_hash = user
        
        # Verify password
        if bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8')):
            conn.close()
            return True, {
                'id': user_id,
                'name': name,
                'email': user_email
            }
        else:
            conn.close()
            return False, "Invalid password"
            
    except Exception as e:
        logger.error(f"Error in login: {e}")
        return False, f"Error during login: {str(e)}" 