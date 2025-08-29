# backend/main.py
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from backend.model.clean_predictor import predict_image
from backend.utils.image_preprocess import preprocess_image
from PIL import Image
import io
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Smart Auction AI API",
    description="AI-powered image analysis and auction system",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CONDITION_MAP = {"New": 0.0, "Used": 0.25, "Heavily Used": 0.5}

@app.get("/")
def read_root():
    """Health check endpoint"""
    return {
        "message": "Smart Auction AI API is running!",
        "status": "healthy",
        "version": "1.0.0"
    }

@app.get("/health")
def health_check():
    """Detailed health check endpoint"""
    return {
        "status": "healthy",
        "service": "Smart Auction AI API",
        "version": "1.0.0"
    }

@app.post("/predict/")
async def predict(
    file: UploadFile = File(...),
    condition: str = Form("New")
):
    """
    Analyze uploaded image and return predictions
    
    Args:
        file: Image file to analyze
        condition: Condition of the item
        
    Returns:
        JSON with name, price, and description predictions
    """
    contents = await file.read()
    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")
    processed_img = preprocess_image(img)
    result = predict_image(processed_img, return_confidence=True)
    if len(result) == 4:
        name, base_price, description, confidence = result
    else:
        name, base_price, description = result[:3]
        confidence = 0.0
    try:
        base_price = int(base_price)
    except Exception:
        base_price = 1000
    if base_price is None or not isinstance(base_price, int):
        base_price = 1000
    cond_adj = CONDITION_MAP.get(condition, 0.0)
    price_after_condition = int(base_price * (1 - cond_adj))
    conf_adj = 0.0
    if confidence < 0.6:
        conf_adj = -0.10
    elif confidence > 0.9:
        conf_adj = 0.05
    final_price = int(price_after_condition * (1 + conf_adj))
    breakdown = {
        "base_price": base_price,
        "condition": condition,
        "condition_adjustment": f"-{int(cond_adj*100)}%" if cond_adj > 0 else "0%",
        "price_after_condition": price_after_condition,
        "confidence": round(float(confidence)*100, 1),
        "confidence_adjustment": f"{int(conf_adj*100)}%" if conf_adj != 0 else "0%",
        "final_price": final_price
    }
    logger.info(f"Successfully analyzed image: {name} - ₹{final_price}")
    return {
        "name": name,
        "description": description,
        "confidence": round(float(confidence)*100, 1),
        "breakdown": breakdown
    }

@app.get("/supported-formats")
def get_supported_formats():
    """Get list of supported image formats"""
    return {
        "supported_formats": ["jpg", "jpeg", "png", "webp"],
        "max_file_size": "10MB"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)