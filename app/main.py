import os
import io
import time
import logging
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from PIL import Image
from pydantic import BaseModel
from typing import List

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("classifier-api")

# Initialize FastAPI app
app = FastAPI(
    title="Cats vs Dogs Classifier API",
    description="A monitored API to classify images as Cats or Dogs",
    version="1.1.0"
)

# --- Monitoring Metrics (Simple In-Memory) ---
metrics = {
    "total_requests": 0,
    "predictions_served": 0,
    "errors": 0,
    "start_time": time.time()
}

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Middleware to log request details and track latency."""
    metrics["total_requests"] += 1
    start_time = time.time()
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    logger.info(
        f"Method: {request.method} | Path: {request.url.path} | "
        f"Status: {response.status_code} | Latency: {process_time:.4f}s"
    )
    
    if response.status_code >= 400:
        metrics["errors"] += 1
        
    return response

# Constants
MODEL_PATH = os.getenv("MODEL_PATH", "models/final_cnn_model.h5")
IMAGE_SIZE = 224

# --- Endpoints ---

@app.get("/metrics", summary="Get Monitoring Metrics")
def get_metrics():
    """Exposes basic API metrics for monitoring."""
    uptime = time.time() - metrics["start_time"]
    return {
        "uptime_seconds": round(uptime, 2),
        "total_requests": metrics["total_requests"],
        "predictions_served": metrics["predictions_served"],
        "error_count": metrics["errors"],
        "error_rate": round(metrics["errors"] / max(1, metrics["total_requests"]), 4)
    }

# Load the trained model
# Note: We use a global variable to load the model once when the server starts
model = None

@app.on_event("startup")
def load_model():
    global model
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model file not found at {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"Model loaded from {MODEL_PATH}")

class PredictionResponse(BaseModel):
    filename: str
    label: str
    probability: float

def preprocess_image(image_bytes: bytes):
    """
    Preprocess the uploaded image to match model requirements (224x224 RGB).
    """
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
        img_array = np.array(img) / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        return img_array
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image format: {e}")

@app.get("/health", summary="Health Check")
def health_check():
    """Returns the status of the API."""
    if model is not None:
        return {"status": "healthy", "model_loaded": True}
    return {"status": "degraded", "model_loaded": False}

@app.post("/predict", response_model=PredictionResponse, summary="Predict Cat or Dog")
async def predict(file: UploadFile = File(...)):
    """
    Accepts an image file and returns the classification result.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    
    # Read image
    contents = await file.read()
    
    # Preprocess
    img_array = preprocess_image(contents)
    
    # Inference
    prediction = model.predict(img_array)[0][0]
    metrics["predictions_served"] += 1
    
    # Thresholding (0.5)
    # Binary classification: 0 = Cat, 1 = Dog (assuming alphabetical order from Keras flow)
    if prediction >= 0.5:
        label = "dog"
        probability = float(prediction)
    else:
        label = "cat"
        probability = float(1 - prediction)
        
    return {
        "filename": file.filename,
        "label": label,
        "probability": round(probability, 4)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
