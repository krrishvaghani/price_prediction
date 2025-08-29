from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
import logging
import random

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

# Blacklist: Irrelevant/low-value items
blacklist_labels = {
    "bottle", "plastic_bag", "plant", "carton", "can", "envelope", "packet", "vase", "pot",
    "plate", "cup", "bowl", "spoon", "fork", "knife", "toothbrush", "toothpaste", "soap",
    "towel", "bucket", "mop", "broom", "sponge", "rag", "tissue", "napkin", "paper_towel",
    "plastic_container"
}

# Whitelist: High-priority categories
whitelist_categories = {
    "car", "phone", "handbag", "laptop", "furniture", "watch", "tv", "camera", "book",
    "shoe", "backpack", "microwave", "refrigerator", "sofa", "chair", "table", "bicycle",
    "headphones", "projector"
}

# Example: assign tiers and multipliers
category_tiers = {
    "car": ("luxury", 1.2),
    "airplane": ("luxury", 1.5),
    "phone": ("high", 1.1),
    "laptop": ("high", 1.1),
    "handbag": ("mid", 1.0),
    "book": ("low", 0.7),
    "bottle": ("low", 0.5),
    "plant": ("low", 0.5),
    # ... add more ...
}

# Add high-value categories to your price ranges
category_price_ranges.update({
    "airplane": (100000, 10000000),
    # ... add more as needed ...
})

fallback_min, fallback_max = 500, 1500  # For unknown/uncertain

def estimate_price(label, confidence):
    # Blacklist handling
    if label in blacklist_labels:
        return random.randint(10, 200), "misc", (10, 200)
    
    # Category and tier lookup
    category = label_to_category.get(label, None)
    tier, multiplier = category_tiers.get(category, ("other", 1.0))
    price_range = category_price_ranges.get(category, (fallback_min, fallback_max))
    min_price, max_price = price_range

    # Confidence buckets
    if confidence < 0.5:
        # Low confidence: fallback
        price = random.randint(fallback_min, fallback_max)
        return price, "uncertain", (fallback_min, fallback_max)
    elif confidence < 0.8:
        # Medium confidence: average price, weighted
        avg_price = int((min_price + max_price) / 2)
        price = int(avg_price * multiplier)
    else:
        # High confidence: max price, weighted
        price = int(max_price * multiplier)
        # Optional: luxury boost
        if tier == "luxury" and confidence > 0.9:
            price = int(price * 1.1)

    # Clamp price to min/max
    price = max(min_price, min(price, max_price))

    return price, category or "other", price_range

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