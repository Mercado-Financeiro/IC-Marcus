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

import os
from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, Field, validator, ConfigDict
import uvicorn

# Import routers
from src.api.routes import market_data

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

# Security: Configure allowed origins from environment
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:8080").split(",")

# CORS middleware - secure configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,  # Specific origins only
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],  # Explicit methods
    allow_headers=["Authorization", "Content-Type"],  # Explicit headers
)

# Security: Trusted host middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["localhost", "127.0.0.1", "*.example.com"]
)

# Security headers middleware
@app.middleware("http")
async def security_headers(request, call_next):
    """Add security headers to all responses."""
    response = await call_next(request)
    
    # Security headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
    
    return response

# Include routers
app.include_router(market_data.router, prefix="/api/v1", tags=["Market Data"])

# Security (basic bearer token)
security = HTTPBearer()

# Global model storage
models = {}


# ============= Pydantic Models =============

class PredictionRequest(BaseModel):
    """Request model for predictions."""
    model_config = ConfigDict(protected_namespaces=())
    
    symbol: str = Field(..., description="Trading symbol (e.g., BTCUSDT)")
    timeframe: str = Field("15m", description="Timeframe for prediction")
    features: Dict[str, float] = Field(..., description="Feature dictionary")
    
    @validator('symbol')
    def validate_symbol(cls, v):
        if not v or not isinstance(v, str):
            raise ValueError('Symbol must be a non-empty string')
        
        # Sanitize: remove potentially dangerous characters
        import re
        sanitized = re.sub(r'[^A-Z0-9]', '', v.upper())
        
        if not sanitized.endswith('USDT'):
            raise ValueError('Symbol must end with USDT')
        
        if len(sanitized) > 20:  # Reasonable length limit
            raise ValueError('Symbol too long')
            
        return sanitized
    
    @validator('timeframe')
    def validate_timeframe(cls, v):
        valid_timeframes = ['1m', '5m', '15m', '30m', '1h', '4h', '1d']
        if v not in valid_timeframes:
            raise ValueError(f'Timeframe must be one of {valid_timeframes}')
        return v


class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions."""
    model_config = ConfigDict(protected_namespaces=())
    
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
    model_config = ConfigDict(protected_namespaces=())
    
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
    model_config = ConfigDict(protected_namespaces=())
    
    symbol: str
    timeframe: str
    timestamp: datetime
    predictions: List[PredictionResponse]
    execution_time_ms: float


class ModelInfo(BaseModel):
    """Model information."""
    model_config = ConfigDict(protected_namespaces=())
    
    name: str
    version: str
    type: str
    trained_at: datetime
    metrics: Dict[str, float]
    features: List[str]
    status: str


class HealthResponse(BaseModel):
    """Health check response."""
    model_config = ConfigDict(protected_namespaces=())
    
    status: str
    timestamp: datetime
    models_loaded: int
    uptime_seconds: float


# ============= Helper Functions =============

def load_model(model_path: str):
    """Load a trained model from disk with path validation."""
    import os.path
    
    try:
        # Security: Validate and sanitize the model path
        # Ensure the path is within the allowed models directory
        base_models_dir = os.path.abspath("artifacts/models")
        requested_path = os.path.abspath(model_path)
        
        # Prevent path traversal attacks
        if not requested_path.startswith(base_models_dir):
            logger.error(f"Path traversal attempt detected: {model_path}")
            raise HTTPException(status_code=400, detail="Invalid model path")
        
        # Ensure file exists and has valid extension
        if not os.path.exists(requested_path):
            raise HTTPException(status_code=404, detail="Model file not found")
        
        if not requested_path.endswith(('.pkl', '.joblib')):
            raise HTTPException(status_code=400, detail="Invalid model file type")
        
        with open(requested_path, 'rb') as f:
            model = pickle.load(f)
        
        logger.info(f"Model loaded from {requested_path}")
        return model
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise HTTPException(status_code=500, detail="Failed to load model")


def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    """Secure token verification using environment variables."""
    token = credentials.credentials
    
    # Load API token from environment variable
    expected_token = os.getenv("API_TOKEN")
    
    if not expected_token:
        logger.error("API_TOKEN environment variable not set")
        raise HTTPException(status_code=500, detail="Server configuration error")
    
    # Secure token comparison (constant time)
    import secrets
    if not secrets.compare_digest(token, expected_token):
        logger.warning(f"Invalid token attempt from credentials: {credentials.scheme}")
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