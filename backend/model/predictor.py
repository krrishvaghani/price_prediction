from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
import logging
import sqlite3
import bcrypt
import streamlit as st
from ..auth_utils import init_db, signup_user, login_user

# Configure logging
logger = logging.getLogger(__name__)

# Load the pre-trained model
try:
    model = MobileNetV2(weights="imagenet")
    logger.info("MobileNetV2 model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    model = None

# Category price ranges (min, max)
category_price_ranges = {
    "car": (5000, 2000000),
    "phone": (3000, 150000),
    "handbag": (500, 50000),
    "laptop": (10000, 300000),
    "furniture": (1000, 100000),
    "watch": (1000, 1000000),
    "tv": (5000, 500000),
    "camera": (2000, 500000),
    "book": (100, 5000),
    "shoe": (200, 20000),
    "backpack": (200, 10000),
    "microwave": (1000, 30000),
    "refrigerator": (5000, 200000),
    "sofa": (2000, 100000),
    "chair": (500, 20000),
    "table": (1000, 50000),
    "bicycle": (1000, 100000),
    "headphones": (500, 50000),
    "projector": (2000, 100000),
    # ... add more as needed ...
}

# Map MobileNetV2 labels to categories
label_to_category = {
    "sports_car": "car",
    "convertible": "car",
    "cab": "car",
    "minivan": "car",
    "jeep": "car",
    "cellular_telephone": "phone",
    "laptop": "laptop",
    "handbag": "handbag",
    "sofa": "furniture",
    "armchair": "furniture",
    "studio_couch": "furniture",
    "digital_watch": "watch",
    "tv": "tv",
    "camera": "camera",
    "book": "book",
    "shoe": "shoe",
    "backpack": "backpack",
    "microwave": "microwave",
    "refrigerator": "refrigerator",
    "bicycle": "bicycle",
    "headphones": "headphones",
    "projector": "projector",
    # ... add more as needed ...
}

def estimate_price(label, confidence):
    category = label_to_category.get(label, None)
    if category and category in category_price_ranges:
        min_price, max_price = category_price_ranges[category]
        price = int(min_price + (max_price - min_price) * float(confidence))
        # Luxury boost for high-confidence luxury items
        if category in ["car", "handbag", "watch", "laptop", "tv", "camera"] and confidence > 0.9:
            price = int(price * 1.2)
        return price, category, (min_price, max_price)
    else:
        # Fallback: generic range
        fallback_min, fallback_max = 500, 5000
        price = int(fallback_min + (fallback_max - fallback_min) * float(confidence))
        return price, "other", (fallback_min, fallback_max)

def predict_image(img, return_confidence=False):
    """
    Predict item name, price, and description from image
    
    Args:
        img: PIL Image object
        return_confidence: bool, whether to return confidence
        
    Returns:
        tuple: (name, price, description, confidence) if return_confidence is True
        tuple: (name, price, description) if return_confidence is False
    """
    if model is None:
        logger.error("Model not loaded")
        return "Unknown Item", 1000, "Unable to analyze image - model not available"
    
    try:
        # Preprocess image
        img = img.resize((224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Get predictions
        preds = model.predict(img_array, verbose=0)
        decoded = decode_predictions(preds, top=1)[0][0]

        label = decoded[1].lower().replace(" ", "_")
        confidence = float(decoded[2])

        logger.info(f"Predicted: {label} with confidence: {confidence:.2f}")

        # Estimate price
        price, category, price_range = estimate_price(label, confidence)
        name = label.replace("_", " ").title()
        desc = f"Seems like a {label.replace('_', ' ')}. Confidence: {round(confidence * 100, 2)}%"

        if return_confidence:
            return name, price, desc, confidence
        else:
            return name, price, desc
            
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        return "Unknown Item", 1000, "Error analyzing image - please try again"

def format_price(val):
    try:
        return f"{int(val):,}"
    except (ValueError, TypeError):
        return "?"

def show_login():
    st.subheader("🔐 Login")
    email = st.text_input("Email", key="login_email")
    password = st.text_input("Password", type="password", key="login_password")
    if st.button("Login"):
        ok, user_or_msg = login_user(email, password)
        if ok:
            st.session_state.user = user_or_msg
            st.success(f"Welcome, {user_or_msg['name']}!")
        else:
            st.error(user_or_msg)
    st.markdown("Don't have an account? [Sign up](#)", unsafe_allow_html=True)
    if st.button("Go to Signup"):
        st.session_state.auth_mode = "signup"

def show_signup():
    st.subheader("📝 Signup")
    name = st.text_input("Name", key="signup_name")
    email = st.text_input("Email", key="signup_email")
    password = st.text_input("Password", type="password", key="signup_password")
    if st.button("Signup"):
        ok, msg = signup_user(name, email, password)
        if ok:
            st.success("Signup successful! Please log in.")
            st.session_state.auth_mode = "login"
        else:
            st.error(msg)
    st.markdown("Already have an account? [Login](#)", unsafe_allow_html=True)
    if st.button("Go to Login"):
        st.session_state.auth_mode = "login"

init_db()  # Ensure DB is initialized

if "user" not in st.session_state:
    st.session_state.user = None
if "auth_mode" not in st.session_state:
    st.session_state.auth_mode = "login"

if st.session_state.user is None:
    if st.session_state.auth_mode == "login":
        show_login()
    else:
        show_signup()
    st.stop()
else:
    st.sidebar.write(f"👤 Logged in as: {st.session_state.user['name']}")
    if st.sidebar.button("Logout"):
        st.session_state.user = None
        st.session_state.auth_mode = "login"
        st.experimental_rerun()

st.title("Welcome to Smart Auction AI System!")
st.write(f"Hello, {st.session_state.user['name']} 👋")
# ...rest of your auction app logic here...