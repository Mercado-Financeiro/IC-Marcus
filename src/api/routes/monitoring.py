"""Monitoring endpoints for API."""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict
import structlog

from src.api.models import HealthResponse, MetricsResponse
from src.api.middleware.auth import verify_token
from src.api.state import app_state
from src.api.config import settings

log = structlog.get_logger()

router = APIRouter(tags=["monitoring"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    metrics = app_state.get_metrics()
    
    return HealthResponse(
        status="healthy" if app_state.model else "degraded",
        model_loaded=app_state.model is not None,
        cache_connected=app_state.redis_client is not None,
        uptime_seconds=metrics["uptime_seconds"],
        last_prediction=app_state.last_prediction_time.isoformat() if app_state.last_prediction_time else None,
        model_info={
            "name": settings.model_name,
            "version": app_state.model_version,
            "stage": settings.model_stage
        }
    )


@router.get("/metrics", response_model=MetricsResponse)
async def get_metrics(token: str = Depends(verify_token)):
    """Get service metrics."""
    metrics = app_state.get_metrics()
    
    # Calculate drift score if available
    drift_score = None
    if app_state.drift_monitor:
        try:
            # Placeholder - implement real drift calculation
            drift_score = 0.05
        except Exception:
            pass
    
    return MetricsResponse(
        total_predictions=metrics["total_predictions"],
        avg_latency_ms=metrics["avg_latency_ms"],
        cache_hit_rate=metrics["cache_hit_rate"],
        error_rate=metrics["error_rate"],
        requests_per_minute=metrics["requests_per_minute"],
        model_drift_score=drift_score
    )


@router.post("/reload-model")
async def reload_model(token: str = Depends(verify_token)):
    """Reload model from MLflow."""
    try:
        await app_state.load_model(settings)
        log.info("model_reloaded", version=app_state.model_version)
        return {
            "message": "Model reloaded successfully",
            "version": app_state.model_version
        }
    except Exception as e:
        log.error(f"Failed to reload model: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to reload model: {e}"
        )


@router.post("/clear-cache")
async def clear_cache(token: str = Depends(verify_token)):
    """Clear prediction cache."""
    try:
        from src.api.cache import CacheManager
        cache_manager = CacheManager(app_state.redis_client)
        await cache_manager.clear()
        
        return {"message": "Cache cleared successfully"}
    except Exception as e:
        log.error(f"Failed to clear cache: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clear cache: {e}"
        )


@router.get("/drift-report")
async def get_drift_report(token: str = Depends(verify_token)):
    """Get drift monitoring report."""
    if not app_state.drift_monitor:
        raise HTTPException(
            status_code=503,
            detail="Drift monitor not initialized"
        )
    
    try:
        # Placeholder - implement real drift report
        report = {
            "status": "stable",
            "psi_score": 0.05,
            "kl_divergence": 0.02,
            "wasserstein_distance": 0.01,
            "features_drifted": [],
            "last_check": "2024-01-01T00:00:00"
        }
        
        return report
    except Exception as e:
        log.error(f"Failed to generate drift report: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate drift report: {e}"
        )