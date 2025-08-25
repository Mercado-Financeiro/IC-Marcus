"""Configuration settings for the API."""

from pydantic import BaseModel
import os


class Settings(BaseModel):
    """API configuration settings."""
    
    # MLflow settings
    mlflow_tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", "artifacts/mlruns")
    model_name: str = os.getenv("MODEL_NAME", "crypto_xgb")
    model_stage: str = os.getenv("MODEL_STAGE", "Production")
    
    # Redis settings
    redis_host: str = os.getenv("REDIS_HOST", "localhost")
    redis_port: int = int(os.getenv("REDIS_PORT", "6379"))
    cache_ttl: int = int(os.getenv("CACHE_TTL", "300"))  # 5 minutes
    
    # Security settings
    api_key: str = os.getenv("API_KEY", "")
    
    # Request settings
    max_batch_size: int = 100
    rate_limit_requests: int = 100
    rate_limit_window: int = 60  # seconds
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False
    log_level: str = "info"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global settings instance
settings = Settings()

# Validate required settings only in production
import os
if os.getenv("ENV", "dev") == "production" and not settings.api_key:
    raise ValueError("API_KEY environment variable not set. This is required for security in production.")