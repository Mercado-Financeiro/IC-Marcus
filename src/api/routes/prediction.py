"""Prediction endpoints for API."""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import List
import pandas as pd
from datetime import datetime
import structlog

from src.api.models import (
    PredictionRequest,
    BatchPredictionRequest,
    PredictionResponse
)
from src.api.middleware.auth import verify_token
from src.api.state import app_state
from src.api.cache import CacheManager

log = structlog.get_logger()

router = APIRouter(prefix="/predict", tags=["predictions"])


@router.post("", response_model=PredictionResponse)
async def predict_single(
    request: PredictionRequest,
    background_tasks: BackgroundTasks,
    token: str = Depends(verify_token)
):
    """Single prediction endpoint."""
    start_time = datetime.now()
    
    if not app_state.model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Initialize cache manager
        cache_manager = CacheManager(
            app_state.redis_client,
            ttl=300  # 5 minutes
        )
        
        # Check cache
        cache_key = cache_manager.generate_key(request.features)
        cached_result = await cache_manager.get(cache_key)
        
        if cached_result and not request.return_shap:
            log.info("cache_hit", request_id=request.request_id)
            app_state.update_cache_metrics(hit=True)
            
            cached_result["request_id"] = request.request_id
            cached_result["timestamp"] = datetime.now().isoformat()
            return PredictionResponse(**cached_result)
        
        app_state.update_cache_metrics(hit=False)
        
        # Prepare features
        features_df = pd.DataFrame([request.features])
        
        # Reorder columns if necessary
        if app_state.feature_names:
            features_df = features_df[app_state.feature_names]
        
        # Make prediction
        prediction = app_state.model.predict(features_df)[0]
        
        # Prepare response
        response_data = {
            "prediction": float(prediction),
            "model_version": app_state.model_version,
            "timestamp": datetime.now().isoformat(),
            "request_id": request.request_id
        }
        
        # Add probabilities if requested
        if request.return_probabilities and hasattr(app_state.model, "predict_proba"):
            probas = app_state.model.predict_proba(features_df)[0]
            response_data["probabilities"] = {
                f"class_{i}": float(p) for i, p in enumerate(probas)
            }
            response_data["confidence"] = float(max(probas))
        
        # Add SHAP if requested
        if request.return_shap:
            try:
                import shap
                explainer = shap.TreeExplainer(app_state.model)
                shap_values = explainer.shap_values(features_df)
                
                if isinstance(shap_values, list):
                    shap_values = shap_values[1] if len(shap_values) == 2 else shap_values[0]
                
                response_data["shap_values"] = dict(zip(
                    features_df.columns,
                    shap_values[0].tolist()
                ))
            except Exception as e:
                log.warning(f"SHAP calculation failed: {e}")
        
        # Update metrics
        latency = (datetime.now() - start_time).total_seconds() * 1000
        app_state.update_prediction_metrics(latency)
        
        # Cache result
        if not request.return_shap:  # Don't cache with SHAP
            background_tasks.add_task(cache_manager.set, cache_key, response_data)
        
        # Log for monitoring
        background_tasks.add_task(
            log_prediction,
            request.features,
            prediction,
            latency
        )
        
        return PredictionResponse(**response_data)
        
    except Exception as e:
        app_state.increment_errors()
        log.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch", response_model=List[PredictionResponse])
async def predict_batch(
    request: BatchPredictionRequest,
    background_tasks: BackgroundTasks,
    token: str = Depends(verify_token)
):
    """Batch prediction endpoint."""
    if not app_state.model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Prepare features
        features_df = pd.DataFrame(request.samples)
        
        # Reorder columns if necessary
        if app_state.feature_names:
            features_df = features_df[app_state.feature_names]
        
        # Make predictions
        predictions = app_state.model.predict(features_df)
        
        # Prepare responses
        responses = []
        
        for i, pred in enumerate(predictions):
            response_data = {
                "prediction": float(pred),
                "model_version": app_state.model_version,
                "timestamp": datetime.now().isoformat(),
                "request_id": f"{request.request_id}_{i}" if request.request_id else None
            }
            
            if request.return_probabilities and hasattr(app_state.model, "predict_proba"):
                probas = app_state.model.predict_proba(features_df.iloc[[i]])[0]
                response_data["probabilities"] = {
                    f"class_{j}": float(p) for j, p in enumerate(probas)
                }
                response_data["confidence"] = float(max(probas))
            
            responses.append(PredictionResponse(**response_data))
        
        # Update metrics
        app_state.prediction_count += len(predictions)
        app_state.last_prediction_time = datetime.now()
        
        return responses
        
    except Exception as e:
        app_state.increment_errors()
        log.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def log_prediction(features: dict, prediction: float, latency: float):
    """Log prediction for monitoring."""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "features": features,
        "prediction": prediction,
        "latency_ms": latency,
        "model_version": app_state.model_version
    }
    
    # Send to monitoring system
    # Placeholder - implement real monitoring
    log.info("prediction_logged", **log_entry)