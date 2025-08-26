"""API FastAPI para servir modelos de ML em produção."""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, WebSocket
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Any, Union
import pandas as pd
import numpy as np
import joblib
import mlflow
from datetime import datetime, timedelta
import asyncio
import redis
import json
import hashlib
import structlog
from pathlib import Path
import os
from contextlib import asynccontextmanager

# Configurar logging
log = structlog.get_logger()

# Configurações
class Settings(BaseModel):
    """Configurações da API."""
    
    mlflow_tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", "artifacts/mlruns")
    model_name: str = os.getenv("MODEL_NAME", "crypto_xgb")
    model_stage: str = os.getenv("MODEL_STAGE", "Production")
    redis_host: str = os.getenv("REDIS_HOST", "localhost")
    redis_port: int = int(os.getenv("REDIS_PORT", 6379))
    cache_ttl: int = int(os.getenv("CACHE_TTL", 300))  # 5 minutos
    api_key: str = os.getenv("API_KEY")
    max_batch_size: int = 100
    rate_limit_requests: int = 100
    rate_limit_window: int = 60  # segundos


settings = Settings()

if not settings.api_key:
    raise ValueError("API_KEY environment variable not set. This is required for security.")

# Modelos de request/response
class PredictionRequest(BaseModel):
    """Request para predição."""
    
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


# Estado global da aplicação
class AppState:
    """Estado compartilhado da aplicação."""
    
    def __init__(self):
        self.model = None
        self.model_version = None
        self.redis_client = None
        self.start_time = datetime.now()
        self.last_prediction_time = None
        self.prediction_count = 0
        self.total_latency = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.errors = 0
        self.drift_monitor = None
        self.feature_names = None


app_state = AppState()

# Contexto de inicialização
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gerencia ciclo de vida da aplicação."""
    # Startup
    log.info("Starting API server...")
    
    # Carregar modelo
    try:
        await load_model()
        log.info("Model loaded successfully")
    except Exception as e:
        log.error(f"Failed to load model: {e}")
    
    # Conectar Redis
    try:
        app_state.redis_client = redis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            decode_responses=True
        )
        app_state.redis_client.ping()
        log.info("Redis connected")
    except Exception as e:
        log.warning(f"Redis connection failed: {e}")
        app_state.redis_client = None
    
    # Inicializar drift monitor
    try:
        from src.monitoring.drift_monitor import DriftMonitor
        app_state.drift_monitor = DriftMonitor()
        log.info("Drift monitor initialized")
    except Exception as e:
        log.warning(f"Drift monitor initialization failed: {e}")
    
    yield
    
    # Shutdown
    log.info("Shutting down API server...")
    if app_state.redis_client:
        app_state.redis_client.close()


# Criar aplicação FastAPI
app = FastAPI(
    title="Crypto ML Prediction API",
    description="API para servir modelos de ML para trading de criptomoedas",
    version="1.0.0",
    lifespan=lifespan
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Segurança
security = HTTPBearer()


async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verifica token de autenticação."""
    token = credentials.credentials
    
    # Verificação simples - em produção usar JWT
    if token != settings.api_key:
        raise HTTPException(status_code=403, detail="Invalid authentication token")
    
    return token


async def load_model():
    """Carrega modelo do MLflow."""
    try:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        
        # Carregar modelo da produção
        model_uri = f"models:/{settings.model_name}/{settings.model_stage}"
        app_state.model = mlflow.pyfunc.load_model(model_uri)
        
        # Obter informações do modelo
        client = mlflow.tracking.MlflowClient()
        model_version = client.get_latest_versions(
            settings.model_name,
            stages=[settings.model_stage]
        )[0]
        
        app_state.model_version = model_version.version
        
        # Tentar obter nomes das features
        try:
            run = client.get_run(model_version.run_id)
            feature_names = run.data.params.get("feature_names", "").split(",")
            app_state.feature_names = feature_names if feature_names[0] else None
        except:
            app_state.feature_names = None
        
        log.info(f"Model loaded: {settings.model_name} v{app_state.model_version}")
        
    except Exception as e:
        log.error(f"Error loading model: {e}")
        raise


def get_cache_key(features: Dict[str, float]) -> str:
    """Gera chave de cache para features."""
    # Ordenar features para consistência
    sorted_features = sorted(features.items())
    features_str = json.dumps(sorted_features)
    
    # Hash para chave compacta
    hash_obj = hashlib.md5(features_str.encode())
    return f"pred:{hash_obj.hexdigest()}"


async def get_from_cache(key: str) -> Optional[Dict]:
    """Busca resultado do cache."""
    if not app_state.redis_client:
        return None
    
    try:
        cached = app_state.redis_client.get(key)
        if cached:
            app_state.cache_hits += 1
            return json.loads(cached)
        app_state.cache_misses += 1
    except Exception as e:
        log.warning(f"Cache get error: {e}")
    
    return None


async def set_cache(key: str, value: Dict):
    """Armazena resultado no cache."""
    if not app_state.redis_client:
        return
    
    try:
        app_state.redis_client.setex(
            key,
            settings.cache_ttl,
            json.dumps(value)
        )
    except Exception as e:
        log.warning(f"Cache set error: {e}")


# Endpoints
@app.get("/", response_model=Dict[str, str])
async def root():
    """Endpoint raiz."""
    return {
        "message": "Crypto ML Prediction API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check do serviço."""
    uptime = (datetime.now() - app_state.start_time).total_seconds()
    
    return HealthResponse(
        status="healthy" if app_state.model else "degraded",
        model_loaded=app_state.model is not None,
        cache_connected=app_state.redis_client is not None,
        uptime_seconds=uptime,
        last_prediction=app_state.last_prediction_time.isoformat() if app_state.last_prediction_time else None,
        model_info={
            "name": settings.model_name,
            "version": app_state.model_version,
            "stage": settings.model_stage
        }
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    background_tasks: BackgroundTasks,
    token: str = Depends(verify_token)
):
    """Endpoint de predição única."""
    start_time = datetime.now()
    
    if not app_state.model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Verificar cache
        cache_key = get_cache_key(request.features)
        cached_result = await get_from_cache(cache_key)
        
        if cached_result and not request.return_shap:
            log.info("Cache hit", request_id=request.request_id)
            cached_result["request_id"] = request.request_id
            cached_result["timestamp"] = datetime.now().isoformat()
            return PredictionResponse(**cached_result)
        
        # Preparar features
        features_df = pd.DataFrame([request.features])
        
        # Reordenar colunas se necessário
        if app_state.feature_names:
            features_df = features_df[app_state.feature_names]
        
        # Fazer predição
        prediction = app_state.model.predict(features_df)[0]
        
        # Preparar response
        response_data = {
            "prediction": float(prediction),
            "model_version": app_state.model_version,
            "timestamp": datetime.now().isoformat(),
            "request_id": request.request_id
        }
        
        # Adicionar probabilidades se solicitado
        if request.return_probabilities and hasattr(app_state.model, "predict_proba"):
            probas = app_state.model.predict_proba(features_df)[0]
            response_data["probabilities"] = {
                f"class_{i}": float(p) for i, p in enumerate(probas)
            }
            response_data["confidence"] = float(max(probas))
        
        # Adicionar SHAP se solicitado
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
        
        # Atualizar estatísticas
        app_state.prediction_count += 1
        app_state.last_prediction_time = datetime.now()
        latency = (datetime.now() - start_time).total_seconds() * 1000
        app_state.total_latency += latency
        
        # Cache resultado
        if not request.return_shap:  # Não cachear com SHAP
            background_tasks.add_task(set_cache, cache_key, response_data)
        
        # Log para monitoramento
        background_tasks.add_task(
            log_prediction,
            request.features,
            prediction,
            latency
        )
        
        return PredictionResponse(**response_data)
        
    except Exception as e:
        app_state.errors += 1
        log.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=List[PredictionResponse])
async def predict_batch(
    request: BatchPredictionRequest,
    background_tasks: BackgroundTasks,
    token: str = Depends(verify_token)
):
    """Endpoint de predição em batch."""
    if not app_state.model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Preparar features
        features_df = pd.DataFrame(request.samples)
        
        # Reordenar colunas se necessário
        if app_state.feature_names:
            features_df = features_df[app_state.feature_names]
        
        # Fazer predições
        predictions = app_state.model.predict(features_df)
        
        # Preparar responses
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
        
        # Atualizar estatísticas
        app_state.prediction_count += len(predictions)
        app_state.last_prediction_time = datetime.now()
        
        return responses
        
    except Exception as e:
        app_state.errors += 1
        log.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics(token: str = Depends(verify_token)):
    """Retorna métricas do serviço."""
    
    # Calcular métricas
    avg_latency = app_state.total_latency / app_state.prediction_count if app_state.prediction_count > 0 else 0
    
    cache_total = app_state.cache_hits + app_state.cache_misses
    cache_hit_rate = app_state.cache_hits / cache_total if cache_total > 0 else 0
    
    total_requests = app_state.prediction_count + app_state.errors
    error_rate = app_state.errors / total_requests if total_requests > 0 else 0
    
    uptime_minutes = (datetime.now() - app_state.start_time).total_seconds() / 60
    requests_per_minute = app_state.prediction_count / uptime_minutes if uptime_minutes > 0 else 0
    
    # Calcular drift score se disponível
    drift_score = None
    if app_state.drift_monitor:
        try:
            # Placeholder - implementar cálculo real
            drift_score = 0.05
        except:
            pass
    
    return MetricsResponse(
        total_predictions=app_state.prediction_count,
        avg_latency_ms=avg_latency,
        cache_hit_rate=cache_hit_rate,
        error_rate=error_rate,
        requests_per_minute=requests_per_minute,
        model_drift_score=drift_score
    )


@app.post("/reload-model")
async def reload_model(
    background_tasks: BackgroundTasks,
    token: str = Depends(verify_token)
):
    """Recarrega o modelo do MLflow."""
    try:
        await load_model()
        return {"message": "Model reloaded successfully", "version": app_state.model_version}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reload model: {e}")


@app.websocket("/ws/predictions")
async def websocket_predictions(websocket: WebSocket):
    """WebSocket para streaming de predições."""
    await websocket.accept()
    
    try:
        while True:
            # Receber dados
            data = await websocket.receive_json()
            
            # Validar token
            if data.get("token") != settings.api_key:
                await websocket.send_json({"error": "Invalid token"})
                continue
            
            # Fazer predição
            features = data.get("features", {})
            if not features:
                await websocket.send_json({"error": "No features provided"})
                continue
            
            try:
                features_df = pd.DataFrame([features])
                
                if app_state.feature_names:
                    features_df = features_df[app_state.feature_names]
                
                prediction = app_state.model.predict(features_df)[0]
                
                # Enviar resultado
                await websocket.send_json({
                    "prediction": float(prediction),
                    "timestamp": datetime.now().isoformat(),
                    "model_version": app_state.model_version
                })
                
            except Exception as e:
                await websocket.send_json({"error": str(e)})
                
    except Exception as e:
        log.error(f"WebSocket error: {e}")
    finally:
        await websocket.close()


# Funções auxiliares
async def log_prediction(features: Dict, prediction: float, latency: float):
    """Loga predição para monitoramento."""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "features": features,
        "prediction": prediction,
        "latency_ms": latency,
        "model_version": app_state.model_version
    }
    
    # Enviar para sistema de monitoramento
    # Placeholder - implementar envio real
    log.info("prediction_logged", **log_entry)


# Rate limiting simples (em produção usar Redis ou similar)
request_counts = {}


@app.middleware("http")
async def rate_limit_middleware(request, call_next):
    """Middleware para rate limiting."""
    client_ip = request.client.host
    
    # Limpar contadores antigos
    now = datetime.now()
    cutoff = now - timedelta(seconds=settings.rate_limit_window)
    request_counts[client_ip] = [
        t for t in request_counts.get(client_ip, [])
        if t > cutoff
    ]
    
    # Verificar limite
    if len(request_counts.get(client_ip, [])) >= settings.rate_limit_requests:
        return JSONResponse(
            status_code=429,
            content={"detail": "Rate limit exceeded"}
        )
    
    # Adicionar request
    if client_ip not in request_counts:
        request_counts[client_ip] = []
    request_counts[client_ip].append(now)
    
    # Processar request
    response = await call_next(request)
    return response


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )