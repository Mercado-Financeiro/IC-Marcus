"""
FastAPI application for model serving.
"""

from typing import Dict, List, Optional, Union
from datetime import datetime
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import logging

from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import uvicorn

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="ML Trading Pipeline API",
    description="API for cryptocurrency price prediction using ML models",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security (basic bearer token)
security = HTTPBearer()

# Global model storage
models = {}


# ============= Pydantic Models =============

class PredictionRequest(BaseModel):
    """Request model for predictions."""
    
    symbol: str = Field(..., description="Trading symbol (e.g., BTCUSDT)")
    timeframe: str = Field("15m", description="Timeframe for prediction")
    features: Dict[str, float] = Field(..., description="Feature dictionary")
    
    @validator('symbol')
    def validate_symbol(cls, v):
        if not v or not v.endswith('USDT'):
            raise ValueError('Symbol must end with USDT')
        return v.upper()
    
    @validator('timeframe')
    def validate_timeframe(cls, v):
        valid_timeframes = ['1m', '5m', '15m', '30m', '1h', '4h', '1d']
        if v not in valid_timeframes:
            raise ValueError(f'Timeframe must be one of {valid_timeframes}')
        return v


class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions."""
    
    symbol: str
    timeframe: str = "15m"
    data: List[Dict[str, float]] = Field(..., description="List of feature dictionaries")
    
    @validator('data')
    def validate_data(cls, v):
        if not v or len(v) == 0:
            raise ValueError('Data must not be empty')
        if len(v) > 1000:
            raise ValueError('Batch size must not exceed 1000')
        return v


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    
    symbol: str
    timeframe: str
    timestamp: datetime
    prediction: float = Field(..., ge=0, le=1, description="Probability [0, 1]")
    signal: str = Field(..., description="Trading signal: LONG/SHORT/NEUTRAL")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score")
    threshold_long: float = Field(0.65, description="Long threshold")
    threshold_short: float = Field(0.35, description="Short threshold")


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions."""
    
    symbol: str
    timeframe: str
    timestamp: datetime
    predictions: List[PredictionResponse]
    execution_time_ms: float


class ModelInfo(BaseModel):
    """Model information."""
    
    name: str
    version: str
    type: str
    trained_at: datetime
    metrics: Dict[str, float]
    features: List[str]
    status: str


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str
    timestamp: datetime
    models_loaded: int
    uptime_seconds: float


# ============= Helper Functions =============

def load_model(model_path: str):
    """Load a trained model from disk."""
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"Model loaded from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")


def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    """Simple token verification (implement properly for production)."""
    token = credentials.credentials
    # --- SECURITY WARNING ---
    # Hardcoded tokens are a major security risk.
    # In production, use a secure method like OAuth2 and load secrets
    # from environment variables or a secret management system.
    # Example: API_TOKEN = os.getenv("API_TOKEN")
    if token != "demo-token": # Replace with your actual token logic
        raise HTTPException(status_code=403, detail="Invalid token")
    return token


def predict_single(model, features: Dict[str, float]) -> Dict:
    """Make a single prediction."""
    try:
        # Convert features to DataFrame
        X = pd.DataFrame([features])
        
        # Get prediction
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X)[0, 1]
        else:
            proba = model.predict(X)[0]
        
        # Determine signal
        if proba > 0.65:
            signal = "LONG"
        elif proba < 0.35:
            signal = "SHORT"
        else:
            signal = "NEUTRAL"
        
        # Calculate confidence
        confidence = abs(proba - 0.5) * 2
        
        return {
            "prediction": float(proba),
            "signal": signal,
            "confidence": float(confidence)
        }
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# ============= API Endpoints =============

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint."""
    return {
        "message": "ML Trading Pipeline API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    import time
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(),
        models_loaded=len(models),
        uptime_seconds=time.time() - app.state.start_time if hasattr(app.state, 'start_time') else 0
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(
    request: PredictionRequest,
    token: str = Depends(verify_token)
):
    """Make a single prediction."""
    
    # Get model
    model_key = f"{request.symbol}_{request.timeframe}"
    if model_key not in models:
        # Try to load default model
        default_model_path = f"artifacts/models/xgboost_optimized.pkl"
        if Path(default_model_path).exists():
            models[model_key] = load_model(default_model_path)
        else:
            raise HTTPException(status_code=404, detail=f"Model not found for {model_key}")
    
    # Make prediction
    result = predict_single(models[model_key], request.features)
    
    return PredictionResponse(
        symbol=request.symbol,
        timeframe=request.timeframe,
        timestamp=datetime.now(),
        **result
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(
    request: BatchPredictionRequest,
    token: str = Depends(verify_token)
):
    """Make batch predictions."""
    import time
    start_time = time.time()
    
    # Get model
    model_key = f"{request.symbol}_{request.timeframe}"
    if model_key not in models:
        default_model_path = f"artifacts/models/xgboost_optimized.pkl"
        if Path(default_model_path).exists():
            models[model_key] = load_model(default_model_path)
        else:
            raise HTTPException(status_code=404, detail=f"Model not found for {model_key}")
    
    # Make predictions
    predictions = []
    for features in request.data:
        result = predict_single(models[model_key], features)
        predictions.append(PredictionResponse(
            symbol=request.symbol,
            timeframe=request.timeframe,
            timestamp=datetime.now(),
            **result
        ))
    
    execution_time = (time.time() - start_time) * 1000
    
    return BatchPredictionResponse(
        symbol=request.symbol,
        timeframe=request.timeframe,
        timestamp=datetime.now(),
        predictions=predictions,
        execution_time_ms=execution_time
    )


@app.get("/models", response_model=List[ModelInfo], tags=["Models"])
async def list_models(token: str = Depends(verify_token)):
    """List available models."""
    model_list = []
    
    # Check artifacts directory for models
    models_dir = Path("artifacts/models")
    if models_dir.exists():
        for model_file in models_dir.glob("*.pkl"):
            model_list.append(ModelInfo(
                name=model_file.stem,
                version="1.0.0",
                type="XGBoost",
                trained_at=datetime.fromtimestamp(model_file.stat().st_mtime),
                metrics={
                    "f1_score": 0.434,
                    "pr_auc": 0.714,
                    "roc_auc": 0.500
                },
                features=["rsi_14", "sma_20", "volume_ratio", "returns"],
                status="active" if model_file.stem in [k.split('_')[0] for k in models.keys()] else "available"
            ))
    
    return model_list


@app.post("/models/load", tags=["Models"])
async def load_model_endpoint(
    model_name: str,
    symbol: str,
    timeframe: str = "15m",
    token: str = Depends(verify_token)
):
    """Load a specific model."""
    model_path = f"artifacts/models/{model_name}.pkl"
    
    if not Path(model_path).exists():
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
    
    model_key = f"{symbol}_{timeframe}"
    models[model_key] = load_model(model_path)
    
    return {
        "message": f"Model {model_name} loaded successfully",
        "model_key": model_key
    }


@app.delete("/models/unload", tags=["Models"])
async def unload_model(
    symbol: str,
    timeframe: str = "15m",
    token: str = Depends(verify_token)
):
    """Unload a model from memory."""
    model_key = f"{symbol}_{timeframe}"
    
    if model_key in models:
        del models[model_key]
        return {"message": f"Model {model_key} unloaded successfully"}
    else:
        raise HTTPException(status_code=404, detail=f"Model {model_key} not loaded")


# ============= Startup/Shutdown Events =============

@app.on_event("startup")
async def startup_event():
    """Initialize application state on startup."""
    import time
    app.state.start_time = time.time()
    logger.info("API server started")
    
    # Try to load default model
    default_model_path = "artifacts/models/xgboost_optimized.pkl"
    if Path(default_model_path).exists():
        try:
            models["BTCUSDT_15m"] = load_model(default_model_path)
            logger.info("Default model loaded")
        except Exception as e:
            logger.warning(f"Failed to load default model: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("API server shutting down")
    models.clear()


# ============= Main =============

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )