"""WebSocket endpoints for real-time predictions."""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import pandas as pd
from datetime import datetime
import json
import structlog

from src.api.state import app_state
from src.api.config import settings

log = structlog.get_logger()

router = APIRouter()


@router.websocket("/ws/predictions")
async def websocket_predictions(websocket: WebSocket):
    """WebSocket for streaming predictions."""
    await websocket.accept()
    client_id = f"ws_{id(websocket)}"
    
    log.info("websocket_connected", client_id=client_id)
    
    try:
        while True:
            # Receive data
            data = await websocket.receive_json()
            
            # Validate token
            if data.get("token") != settings.api_key:
                await websocket.send_json({
                    "error": "Invalid token",
                    "timestamp": datetime.now().isoformat()
                })
                continue
            
            # Check model
            if not app_state.model:
                await websocket.send_json({
                    "error": "Model not loaded",
                    "timestamp": datetime.now().isoformat()
                })
                continue
            
            # Get features
            features = data.get("features", {})
            if not features:
                await websocket.send_json({
                    "error": "No features provided",
                    "timestamp": datetime.now().isoformat()
                })
                continue
            
            try:
                # Prepare features
                features_df = pd.DataFrame([features])
                
                if app_state.feature_names:
                    features_df = features_df[app_state.feature_names]
                
                # Make prediction
                prediction = app_state.model.predict(features_df)[0]
                
                # Send result
                response = {
                    "prediction": float(prediction),
                    "timestamp": datetime.now().isoformat(),
                    "model_version": app_state.model_version,
                    "request_id": data.get("request_id")
                }
                
                # Add probabilities if requested
                if data.get("return_probabilities") and hasattr(app_state.model, "predict_proba"):
                    probas = app_state.model.predict_proba(features_df)[0]
                    response["probabilities"] = {
                        f"class_{i}": float(p) for i, p in enumerate(probas)
                    }
                    response["confidence"] = float(max(probas))
                
                await websocket.send_json(response)
                
                # Update metrics
                app_state.prediction_count += 1
                app_state.last_prediction_time = datetime.now()
                
            except Exception as e:
                log.error(f"WebSocket prediction error: {e}")
                await websocket.send_json({
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })
    
    except WebSocketDisconnect:
        log.info("websocket_disconnected", client_id=client_id)
    except Exception as e:
        log.error(f"WebSocket error: {e}", client_id=client_id)
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


@router.websocket("/ws/metrics")
async def websocket_metrics(websocket: WebSocket):
    """WebSocket for streaming metrics."""
    await websocket.accept()
    client_id = f"ws_metrics_{id(websocket)}"
    
    log.info("metrics_websocket_connected", client_id=client_id)
    
    try:
        # Validate token
        data = await websocket.receive_json()
        if data.get("token") != settings.api_key:
            await websocket.send_json({"error": "Invalid token"})
            await websocket.close()
            return
        
        # Stream metrics
        import asyncio
        while True:
            metrics = app_state.get_metrics()
            
            await websocket.send_json({
                "metrics": {
                    "total_predictions": metrics["total_predictions"],
                    "avg_latency_ms": metrics["avg_latency_ms"],
                    "cache_hit_rate": metrics["cache_hit_rate"],
                    "error_rate": metrics["error_rate"],
                    "requests_per_minute": metrics["requests_per_minute"]
                },
                "timestamp": datetime.now().isoformat()
            })
            
            # Wait before next update
            await asyncio.sleep(5)  # Update every 5 seconds
    
    except WebSocketDisconnect:
        log.info("metrics_websocket_disconnected", client_id=client_id)
    except Exception as e:
        log.error(f"Metrics WebSocket error: {e}", client_id=client_id)
    finally:
        try:
            await websocket.close()
        except Exception:
            pass