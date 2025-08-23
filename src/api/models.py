"""Pydantic models for API requests and responses."""

from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Any, Union
from datetime import datetime


class PredictionRequest(BaseModel):
    """Request para predição única."""
    
    features: Dict[str, float] = Field(..., description="Features para predição")
    request_id: Optional[str] = Field(None, description="ID único da requisição")
    return_probabilities: bool = Field(False, description="Retornar probabilidades")
    return_shap: bool = Field(False, description="Retornar SHAP values")
    
    @validator('features')
    def validate_features(cls, v):
        if not v:
            raise ValueError("Features cannot be empty")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "features": {
                    "close": 50000.0,
                    "volume": 1000000.0,
                    "rsi_14": 65.0,
                    "nvt_ratio": 1.2
                },
                "request_id": "req_123",
                "return_probabilities": True
            }
        }


class BatchPredictionRequest(BaseModel):
    """Request para predição em batch."""
    
    samples: List[Dict[str, float]] = Field(..., description="Lista de amostras")
    request_id: Optional[str] = Field(None, description="ID único da requisição")
    return_probabilities: bool = Field(False, description="Retornar probabilidades")
    
    @validator('samples')
    def validate_samples(cls, v):
        from src.api.config import settings
        if not v:
            raise ValueError("Samples cannot be empty")
        if len(v) > settings.max_batch_size:
            raise ValueError(f"Batch size exceeds maximum of {settings.max_batch_size}")
        return v


class PredictionResponse(BaseModel):
    """Response de predição."""
    
    prediction: Union[float, int] = Field(..., description="Predição do modelo")
    probabilities: Optional[Dict[str, float]] = Field(None, description="Probabilidades por classe")
    shap_values: Optional[Dict[str, float]] = Field(None, description="SHAP values")
    confidence: Optional[float] = Field(None, description="Confiança da predição")
    request_id: Optional[str] = Field(None, description="ID da requisição")
    model_version: str = Field(..., description="Versão do modelo")
    timestamp: str = Field(..., description="Timestamp da predição")


class HealthResponse(BaseModel):
    """Response de health check."""
    
    status: str
    model_loaded: bool
    cache_connected: bool
    uptime_seconds: float
    last_prediction: Optional[str]
    model_info: Dict[str, Any]


class MetricsResponse(BaseModel):
    """Response de métricas."""
    
    total_predictions: int
    avg_latency_ms: float
    cache_hit_rate: float
    error_rate: float
    requests_per_minute: float
    model_drift_score: Optional[float]


class WebSocketMessage(BaseModel):
    """Mensagem WebSocket."""
    
    token: str
    features: Dict[str, float]
    request_id: Optional[str] = None


class WebSocketResponse(BaseModel):
    """Response WebSocket."""
    
    prediction: Optional[float] = None
    error: Optional[str] = None
    timestamp: str
    model_version: Optional[str] = None