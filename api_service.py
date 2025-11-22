# FastAPI Model Serving Service
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import numpy as np
import pandas as pd
import pickle
import os
from typing import List, Dict, Any
from model import TabularMLP
from sklearn.preprocessing import StandardScaler
import logging
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Federated Learning Model API", version="1.0.0")

# Prometheus metrics
PREDICTION_COUNTER = Counter('predictions_total', 'Total number of predictions made')
PREDICTION_LATENCY = Histogram('prediction_duration_seconds', 'Prediction latency')
MODEL_LOAD_COUNTER = Counter('model_loads_total', 'Total number of model loads')

# Global variables for model and preprocessing
current_model = None
global_scaler = None
feature_columns = None

class PredictionRequest(BaseModel):
    features: Dict[str, float]

class PredictionResponse(BaseModel):
    prediction: float
    prediction_class: str
    confidence: float
    model_version: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str

def load_latest_model():
    """Load the latest trained model and preprocessing components"""
    global current_model, global_scaler, feature_columns
    
    try:
        # Load client datasets for preprocessing info
        if os.path.exists("client_datasets.pkl"):
            with open("client_datasets.pkl", "rb") as f:
                client_datasets = pickle.load(f)
            
            # Build global scaler from training data
            frames = []
            for cid, ds in client_datasets.items():
                df = ds.get("X_train_raw")
                if isinstance(df, pd.DataFrame):
                    frames.append(df)
            
            if frames:
                global_df = pd.concat(frames, axis=0)
                global_scaler = StandardScaler()
                global_scaler.fit(global_df.values)
                feature_columns = list(global_df.columns)
                logger.info(f"Loaded global scaler with {len(feature_columns)} features")
        
        # Load latest model
        model_files = [f for f in os.listdir("models") if f.endswith(".pt")]
        if model_files:
            latest_model = max(model_files, key=lambda x: os.path.getctime(os.path.join("models", x)))
            model_path = os.path.join("models", latest_model)
            
            # Load model state
            state_dict = torch.load(model_path, map_location="cpu")
            
            # Infer input dimension from first layer
            first_weight = None
            for v in state_dict.values():
                if isinstance(v, torch.Tensor) and v.ndim == 2:
                    first_weight = v
                    break
            
            if first_weight is not None:
                input_dim = first_weight.shape[1]
                current_model = TabularMLP(input_dim=input_dim)
                current_model.load_state_dict(state_dict)
                current_model.eval()
                
                MODEL_LOAD_COUNTER.inc()
                logger.info(f"Loaded model: {latest_model}, input_dim: {input_dim}")
                return latest_model
            
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return None
    
    return None

@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    model_version = load_latest_model()
    if model_version:
        logger.info(f"API started with model: {model_version}")
    else:
        logger.warning("API started without a valid model")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    model_version = "unknown"
    if current_model is not None:
        model_files = [f for f in os.listdir("models") if f.endswith(".pt")]
        if model_files:
            model_version = max(model_files, key=lambda x: os.path.getctime(os.path.join("models", x)))
    
    return HealthResponse(
        status="healthy",
        model_loaded=current_model is not None,
        model_version=model_version
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make prediction using the loaded model"""
    start_time = time.time()
    
    try:
        # Check if model is loaded
        if current_model is None or global_scaler is None or feature_columns is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        # Prepare input features
        input_df = pd.DataFrame([request.features])
        
        # One-hot encode if needed (for categorical features)
        input_encoded = pd.get_dummies(input_df, drop_first=False)
        
        # Align with training features
        input_aligned = input_encoded.reindex(columns=feature_columns, fill_value=0)
        
        # Normalize
        input_normalized = global_scaler.transform(input_aligned.values)
        
        # Make prediction
        with torch.no_grad():
            input_tensor = torch.from_numpy(input_normalized.astype(np.float32))
            prediction = current_model(input_tensor).cpu().numpy()[0]
        
        # Convert to classification
        prediction_class = ">50k" if prediction >= 0.5 else "<=50k"
        confidence = float(prediction if prediction >= 0.5 else 1 - prediction)
        
        # Get model version
        model_files = [f for f in os.listdir("models") if f.endswith(".pt")]
        model_version = max(model_files, key=lambda x: os.path.getctime(os.path.join("models", x))) if model_files else "unknown"
        
        # Update metrics
        PREDICTION_COUNTER.inc()
        PREDICTION_LATENCY.observe(time.time() - start_time)
        
        return PredictionResponse(
            prediction=float(prediction),
            prediction_class=prediction_class,
            confidence=confidence,
            model_version=model_version
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/reload-model")
async def reload_model():
    """Reload the latest model"""
    try:
        model_version = load_latest_model()
        if model_version:
            return {"status": "success", "model_version": model_version}
        else:
            raise HTTPException(status_code=404, detail="No valid model found")
    except Exception as e:
        logger.error(f"Model reload error: {e}")
        raise HTTPException(status_code=500, detail=f"Model reload failed: {str(e)}")

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/model-info")
async def model_info():
    """Get information about the current model"""
    if current_model is None:
        return {"model_loaded": False}
    
    model_files = [f for f in os.listdir("models") if f.endswith(".pt")]
    model_version = max(model_files, key=lambda x: os.path.getctime(os.path.join("models", x))) if model_files else "unknown"
    
    return {
        "model_loaded": True,
        "model_version": model_version,
        "input_features": len(feature_columns) if feature_columns else 0,
        "feature_names": feature_columns if feature_columns else []
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)