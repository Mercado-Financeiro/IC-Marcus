"""Application state management."""

from datetime import datetime
from typing import Optional, Any, List
import mlflow
import structlog

try:
    import redis
except ImportError:
    redis = None

log = structlog.get_logger()


class AppState:
    """Global application state."""
    
    def __init__(self):
        """Initialize application state."""
        # Model state
        self.model = None
        self.model_version: Optional[str] = None
        self.feature_names: Optional[List[str]] = None
        
        # Cache state
        self.redis_client: Optional[redis.Redis] = None
        
        # Application metrics
        self.start_time = datetime.now()
        self.last_prediction_time: Optional[datetime] = None
        self.prediction_count = 0
        self.total_latency = 0.0
        self.cache_hits = 0
        self.cache_misses = 0
        self.errors = 0
        
        # Monitoring
        self.drift_monitor = None
        
        # Request tracking for rate limiting
        self.request_counts = {}
    
    async def load_model(self, settings):
        """Load model from MLflow."""
        try:
            mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
            
            # Load model from production
            model_uri = f"models:/{settings.model_name}/{settings.model_stage}"
            self.model = mlflow.pyfunc.load_model(model_uri)
            
            # Get model information
            client = mlflow.tracking.MlflowClient()
            model_version = client.get_latest_versions(
                settings.model_name,
                stages=[settings.model_stage]
            )[0]
            
            self.model_version = model_version.version
            
            # Try to get feature names
            try:
                run = client.get_run(model_version.run_id)
                feature_names = run.data.params.get("feature_names", "").split(",")
                self.feature_names = feature_names if feature_names[0] else None
            except Exception:
                self.feature_names = None
            
            log.info(
                "model_loaded",
                model_name=settings.model_name,
                version=self.model_version,
                stage=settings.model_stage
            )
            
        except Exception as e:
            log.error(f"Error loading model: {e}")
            raise
    
    async def connect_redis(self, settings):
        """Connect to Redis cache."""
        if redis is None:
            log.warning("Redis module not installed")
            self.redis_client = None
            return
            
        try:
            self.redis_client = redis.Redis(
                host=settings.redis_host,
                port=settings.redis_port,
                decode_responses=True
            )
            self.redis_client.ping()
            log.info("Redis connected")
        except Exception as e:
            log.warning(f"Redis connection failed: {e}")
            self.redis_client = None
    
    async def initialize_drift_monitor(self):
        """Initialize drift monitoring."""
        try:
            from src.monitoring.drift_monitor import DriftMonitor
            self.drift_monitor = DriftMonitor()
            log.info("Drift monitor initialized")
        except Exception as e:
            log.warning(f"Drift monitor initialization failed: {e}")
            self.drift_monitor = None
    
    def update_prediction_metrics(self, latency_ms: float):
        """Update prediction metrics."""
        self.prediction_count += 1
        self.last_prediction_time = datetime.now()
        self.total_latency += latency_ms
    
    def update_cache_metrics(self, hit: bool):
        """Update cache metrics."""
        if hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
    
    def increment_errors(self):
        """Increment error count."""
        self.errors += 1
    
    def get_metrics(self):
        """Get current metrics."""
        avg_latency = self.total_latency / self.prediction_count if self.prediction_count > 0 else 0
        
        cache_total = self.cache_hits + self.cache_misses
        cache_hit_rate = self.cache_hits / cache_total if cache_total > 0 else 0
        
        total_requests = self.prediction_count + self.errors
        error_rate = self.errors / total_requests if total_requests > 0 else 0
        
        uptime_minutes = (datetime.now() - self.start_time).total_seconds() / 60
        requests_per_minute = self.prediction_count / uptime_minutes if uptime_minutes > 0 else 0
        
        return {
            "total_predictions": self.prediction_count,
            "avg_latency_ms": avg_latency,
            "cache_hit_rate": cache_hit_rate,
            "error_rate": error_rate,
            "requests_per_minute": requests_per_minute,
            "uptime_seconds": (datetime.now() - self.start_time).total_seconds()
        }
    
    async def cleanup(self):
        """Cleanup resources on shutdown."""
        if self.redis_client:
            self.redis_client.close()
            log.info("Redis connection closed")


# Global state instance
app_state = AppState()