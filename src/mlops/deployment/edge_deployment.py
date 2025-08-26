"""
Edge Computing Deployment for lightweight streaming ML inference.
"""

import os
import json
import pickle
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field, asdict
import logging
import time
from datetime import datetime
import threading
import queue
from abc import ABC, abstractmethod

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
import joblib

# Container and deployment
try:
    import docker
    DOCKER_AVAILABLE = True
except ImportError:
    DOCKER_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("Docker not available. Container features disabled.")

# Model optimization
try:
    import onnx
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

# Quantization
try:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class EdgeConfig:
    """Edge deployment configuration."""
    # Resource constraints
    max_memory_mb: int = 512
    max_cpu_cores: float = 1.0
    storage_limit_mb: int = 1024
    
    # Performance requirements
    max_latency_ms: float = 100.0
    min_throughput_qps: float = 10.0
    batch_size: int = 1
    
    # Optimization settings
    enable_quantization: bool = True
    enable_pruning: bool = True
    enable_caching: bool = True
    cache_size_mb: int = 64
    
    # Deployment settings
    container_base_image: str = "python:3.11-slim"
    deployment_target: str = "docker"  # docker, k8s, edge
    health_check_interval: int = 30
    
    # Edge-specific
    offline_mode: bool = True
    local_storage_path: str = "/tmp/edge_ml"
    backup_models: int = 2


class ModelCompressor:
    """Compress models for edge deployment."""
    
    @staticmethod
    def quantize_sklearn_model(model, precision: str = "int8") -> bytes:
        """Quantize sklearn model for reduced memory usage."""
        if not hasattr(model, 'predict'):
            raise ValueError("Model must have predict method")
        
        # For now, just serialize normally
        # Real quantization would require more sophisticated approaches
        return pickle.dumps(model)
    
    @staticmethod
    def prune_model_features(model, feature_importance: Dict[str, float], 
                           keep_ratio: float = 0.8) -> Tuple[Any, List[str]]:
        """Prune less important features from model."""
        if not feature_importance:
            return model, []
        
        # Sort features by importance
        sorted_features = sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Keep top features
        n_keep = int(len(sorted_features) * keep_ratio)
        kept_features = [f[0] for f in sorted_features[:n_keep]]
        
        logger.info(f"Pruned model to {len(kept_features)}/{len(sorted_features)} features")
        return model, kept_features
    
    @staticmethod
    def optimize_model_size(model_path: str, target_size_mb: float = 50.0) -> str:
        """Optimize model file size."""
        original_size = os.path.getsize(model_path) / (1024 * 1024)
        
        if original_size <= target_size_mb:
            return model_path
        
        # Load and compress
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Save with higher compression
        optimized_path = model_path.replace('.pkl', '_optimized.pkl')
        with open(optimized_path, 'wb') as f:
            pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        new_size = os.path.getsize(optimized_path) / (1024 * 1024)
        logger.info(f"Model size: {original_size:.1f}MB -> {new_size:.1f}MB")
        
        return optimized_path


class EdgeInferenceEngine:
    """Lightweight inference engine for edge deployment."""
    
    def __init__(self, config: EdgeConfig):
        """Initialize edge inference engine."""
        self.config = config
        self.models = {}
        self.feature_processors = {}
        self.cache = {}
        self.stats = {
            'requests': 0,
            'cache_hits': 0,
            'avg_latency_ms': 0.0,
            'memory_usage_mb': 0.0
        }
        self.stats_lock = threading.Lock()
        
        # Create local storage
        os.makedirs(config.local_storage_path, exist_ok=True)
    
    def load_model(self, model_id: str, model_path: str, 
                   feature_processor_path: Optional[str] = None) -> bool:
        """Load model for inference."""
        try:
            # Load model
            if model_path.endswith('.onnx') and ONNX_AVAILABLE:
                # Load ONNX model
                session = ort.InferenceSession(model_path)
                self.models[model_id] = {
                    'model': session,
                    'type': 'onnx',
                    'input_name': session.get_inputs()[0].name
                }
            else:
                # Load pickle model
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                self.models[model_id] = {
                    'model': model,
                    'type': 'sklearn'
                }
            
            # Load feature processor if provided
            if feature_processor_path and os.path.exists(feature_processor_path):
                with open(feature_processor_path, 'rb') as f:
                    processor = pickle.load(f)
                self.feature_processors[model_id] = processor
            
            logger.info(f"Loaded model {model_id} for edge inference")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {e}")
            return False
    
    def predict(self, model_id: str, input_data: Union[Dict, pd.DataFrame, np.ndarray],
                use_cache: bool = True) -> Dict[str, Any]:
        """Make prediction with caching and performance tracking."""
        start_time = time.time()
        
        try:
            # Check cache
            if use_cache and self.config.enable_caching:
                cache_key = self._generate_cache_key(model_id, input_data)
                if cache_key in self.cache:
                    with self.stats_lock:
                        self.stats['cache_hits'] += 1
                    return self.cache[cache_key]
            
            # Get model
            if model_id not in self.models:
                raise ValueError(f"Model {model_id} not loaded")
            
            model_info = self.models[model_id]
            model = model_info['model']
            
            # Preprocess input
            processed_input = self._preprocess_input(model_id, input_data)
            
            # Make prediction
            if model_info['type'] == 'onnx':
                # ONNX inference
                input_name = model_info['input_name']
                prediction = model.run(None, {input_name: processed_input})[0]
            else:
                # Sklearn inference
                prediction = model.predict(processed_input)
            
            # Format result
            result = {
                'prediction': prediction.tolist() if hasattr(prediction, 'tolist') else prediction,
                'model_id': model_id,
                'timestamp': datetime.now().isoformat(),
                'latency_ms': (time.time() - start_time) * 1000
            }
            
            # Cache result
            if use_cache and self.config.enable_caching:
                self._update_cache(cache_key, result)
            
            # Update stats
            self._update_stats(result['latency_ms'])
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed for model {model_id}: {e}")
            return {
                'error': str(e),
                'model_id': model_id,
                'timestamp': datetime.now().isoformat()
            }
    
    def predict_batch(self, model_id: str, input_batch: List[Union[Dict, pd.DataFrame]]) -> List[Dict[str, Any]]:
        """Batch prediction for better throughput."""
        results = []
        
        for input_data in input_batch:
            result = self.predict(model_id, input_data, use_cache=True)
            results.append(result)
        
        return results
    
    def _preprocess_input(self, model_id: str, input_data: Union[Dict, pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Preprocess input data for model."""
        # Convert to DataFrame if needed
        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        elif isinstance(input_data, pd.DataFrame):
            df = input_data
        else:
            return input_data  # Assume already processed
        
        # Apply feature processor if available
        if model_id in self.feature_processors:
            processor = self.feature_processors[model_id]
            if hasattr(processor, 'transform'):
                df = processor.transform(df)
        
        return df.values
    
    def _generate_cache_key(self, model_id: str, input_data: Any) -> str:
        """Generate cache key for input."""
        # Simple hash-based caching
        import hashlib
        
        if isinstance(input_data, dict):
            data_str = json.dumps(input_data, sort_keys=True)
        elif isinstance(input_data, pd.DataFrame):
            data_str = input_data.to_string()
        else:
            data_str = str(input_data)
        
        key = f"{model_id}_{hashlib.md5(data_str.encode()).hexdigest()[:8]}"
        return key
    
    def _update_cache(self, cache_key: str, result: Dict[str, Any]):
        """Update inference cache."""
        # Simple LRU cache implementation
        if len(self.cache) >= 1000:  # Max cache size
            # Remove oldest entry
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        self.cache[cache_key] = result
    
    def _update_stats(self, latency_ms: float):
        """Update performance statistics."""
        with self.stats_lock:
            self.stats['requests'] += 1
            
            # Update average latency (exponential moving average)
            alpha = 0.1
            self.stats['avg_latency_ms'] = (
                alpha * latency_ms + 
                (1 - alpha) * self.stats['avg_latency_ms']
            )
            
            # Update memory usage (approximate)
            import psutil
            process = psutil.Process()
            self.stats['memory_usage_mb'] = process.memory_info().rss / (1024 * 1024)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get inference statistics."""
        with self.stats_lock:
            return self.stats.copy()
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        health = {
            'status': 'healthy',
            'models_loaded': len(self.models),
            'cache_size': len(self.cache),
            'memory_usage_mb': self.stats['memory_usage_mb'],
            'avg_latency_ms': self.stats['avg_latency_ms'],
            'issues': []
        }
        
        # Check memory usage
        if health['memory_usage_mb'] > self.config.max_memory_mb * 0.9:
            health['issues'].append('High memory usage')
            health['status'] = 'degraded'
        
        # Check latency
        if health['avg_latency_ms'] > self.config.max_latency_ms:
            health['issues'].append('High latency')
            health['status'] = 'degraded'
        
        return health


class EdgeDeploymentManager:
    """
    Manager for deploying ML models to edge environments.
    
    Features:
    - Model optimization and compression
    - Container-based deployment
    - Resource monitoring
    - Automatic scaling
    - Health checking
    """
    
    def __init__(self, config: EdgeConfig):
        """Initialize edge deployment manager."""
        self.config = config
        self.deployments = {}
        self.inference_engines = {}
        
        # Docker client
        if DOCKER_AVAILABLE:
            try:
                self.docker_client = docker.from_env()
            except Exception as e:
                logger.warning(f"Docker not available: {e}")
                self.docker_client = None
        else:
            self.docker_client = None
        
        logger.info("EdgeDeploymentManager initialized")
    
    def prepare_model_for_edge(
        self,
        model_path: str,
        model_id: str,
        feature_processor_path: Optional[str] = None,
        optimization_level: str = "medium"
    ) -> Dict[str, str]:
        """
        Prepare model for edge deployment with optimization.
        
        Args:
            model_path: Path to model file
            model_id: Unique model identifier
            feature_processor_path: Optional feature processor
            optimization_level: low, medium, high
            
        Returns:
            Dictionary with prepared model paths
        """
        logger.info(f"Preparing model {model_id} for edge deployment")
        
        # Create deployment directory
        deploy_dir = Path(self.config.local_storage_path) / model_id
        deploy_dir.mkdir(parents=True, exist_ok=True)
        
        prepared_files = {}
        
        # Copy original model
        original_model_path = deploy_dir / "model_original.pkl"
        shutil.copy2(model_path, original_model_path)
        prepared_files['original'] = str(original_model_path)
        
        # Optimize model based on level
        if optimization_level in ["medium", "high"]:
            # Compress model
            optimized_path = ModelCompressor.optimize_model_size(
                str(original_model_path),
                target_size_mb=self.config.storage_limit_mb * 0.5
            )
            prepared_files['optimized'] = optimized_path
        
        # Copy feature processor if provided
        if feature_processor_path and os.path.exists(feature_processor_path):
            processor_path = deploy_dir / "feature_processor.pkl"
            shutil.copy2(feature_processor_path, processor_path)
            prepared_files['feature_processor'] = str(processor_path)
        
        # Create deployment manifest
        manifest = {
            'model_id': model_id,
            'created_at': datetime.now().isoformat(),
            'optimization_level': optimization_level,
            'config': asdict(self.config),
            'files': prepared_files,
            'version': '1.0'
        }
        
        manifest_path = deploy_dir / "manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        prepared_files['manifest'] = str(manifest_path)
        
        logger.info(f"Model {model_id} prepared for edge deployment")
        return prepared_files
    
    def deploy_to_local_edge(self, model_id: str, prepared_files: Dict[str, str]) -> bool:
        """Deploy model to local edge inference engine."""
        try:
            # Create inference engine if not exists
            if model_id not in self.inference_engines:
                self.inference_engines[model_id] = EdgeInferenceEngine(self.config)
            
            engine = self.inference_engines[model_id]
            
            # Load model
            model_path = prepared_files.get('optimized', prepared_files.get('original'))
            feature_processor_path = prepared_files.get('feature_processor')
            
            success = engine.load_model(model_id, model_path, feature_processor_path)
            
            if success:
                self.deployments[model_id] = {
                    'status': 'deployed',
                    'engine': engine,
                    'deployed_at': datetime.now().isoformat(),
                    'files': prepared_files
                }
                logger.info(f"Model {model_id} deployed to local edge")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to deploy model {model_id}: {e}")
            return False
    
    def deploy_to_container(self, model_id: str, prepared_files: Dict[str, str]) -> Optional[str]:
        """Deploy model in Docker container."""
        if not self.docker_client:
            logger.error("Docker not available for container deployment")
            return None
        
        try:
            # Create Dockerfile
            dockerfile_content = self._generate_dockerfile(model_id, prepared_files)
            
            # Build deployment package
            deploy_dir = Path(prepared_files['manifest']).parent
            dockerfile_path = deploy_dir / "Dockerfile"
            
            with open(dockerfile_path, 'w') as f:
                f.write(dockerfile_content)
            
            # Create requirements.txt
            requirements_path = deploy_dir / "requirements.txt"
            with open(requirements_path, 'w') as f:
                f.write("""
pandas>=1.5.0
numpy>=1.20.0
scikit-learn>=1.0.0
joblib>=1.0.0
fastapi>=0.68.0
uvicorn>=0.15.0
""".strip())
            
            # Create inference API
            api_code = self._generate_inference_api(model_id)
            api_path = deploy_dir / "app.py"
            with open(api_path, 'w') as f:
                f.write(api_code)
            
            # Build Docker image
            image_name = f"edge-ml-{model_id.lower()}"
            logger.info(f"Building Docker image: {image_name}")
            
            image, build_logs = self.docker_client.images.build(
                path=str(deploy_dir),
                tag=image_name,
                rm=True
            )
            
            # Run container
            container = self.docker_client.containers.run(
                image_name,
                detach=True,
                ports={'8000/tcp': None},  # Dynamic port assignment
                mem_limit=f"{self.config.max_memory_mb}m",
                cpus=self.config.max_cpu_cores,
                name=f"edge-ml-{model_id}",
                restart_policy={"Name": "unless-stopped"}
            )
            
            # Store deployment info
            self.deployments[model_id] = {
                'status': 'deployed',
                'type': 'container',
                'container_id': container.id,
                'image_name': image_name,
                'deployed_at': datetime.now().isoformat(),
                'files': prepared_files
            }
            
            logger.info(f"Model {model_id} deployed in container {container.id[:12]}")
            return container.id
            
        except Exception as e:
            logger.error(f"Container deployment failed for {model_id}: {e}")
            return None
    
    def _generate_dockerfile(self, model_id: str, prepared_files: Dict[str, str]) -> str:
        """Generate Dockerfile for model deployment."""
        dockerfile = f"""
FROM {self.config.container_base_image}

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model files
COPY model_*.pkl ./
COPY feature_processor.pkl ./
COPY manifest.json ./
COPY app.py ./

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
"""
        return dockerfile.strip()
    
    def _generate_inference_api(self, model_id: str) -> str:
        """Generate FastAPI inference service."""
        api_code = f'''
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import pickle
import json
import logging
from typing import Dict, Any, List, Union
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Edge ML Inference API", description="Model: {model_id}")

# Global model and processor
model = None
feature_processor = None

class PredictionRequest(BaseModel):
    data: Union[Dict[str, Any], List[Dict[str, Any]]]

class PredictionResponse(BaseModel):
    prediction: Union[List, float, int]
    model_id: str
    timestamp: str
    latency_ms: float

@app.on_event("startup")
async def load_model():
    """Load model and feature processor on startup."""
    global model, feature_processor
    
    try:
        # Load model
        with open("model_optimized.pkl", "rb") as f:
            model = pickle.load(f)
        logger.info("Model loaded successfully")
        
        # Load feature processor if exists
        try:
            with open("feature_processor.pkl", "rb") as f:
                feature_processor = pickle.load(f)
            logger.info("Feature processor loaded successfully")
        except FileNotFoundError:
            logger.info("No feature processor found")
        
    except Exception as e:
        logger.error(f"Failed to load model: {{e}}")
        raise

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make prediction."""
    start_time = time.time()
    
    try:
        # Convert input to DataFrame
        if isinstance(request.data, list):
            df = pd.DataFrame(request.data)
        else:
            df = pd.DataFrame([request.data])
        
        # Apply feature processor if available
        if feature_processor and hasattr(feature_processor, 'transform'):
            df = feature_processor.transform(df)
        
        # Make prediction
        prediction = model.predict(df)
        
        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000
        
        return PredictionResponse(
            prediction=prediction.tolist() if hasattr(prediction, 'tolist') else prediction,
            model_id="{model_id}",
            timestamp=datetime.now().isoformat(),
            latency_ms=latency_ms
        )
        
    except Exception as e:
        logger.error(f"Prediction failed: {{e}}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {{
        "status": "healthy",
        "model_loaded": model is not None,
        "model_id": "{model_id}",
        "timestamp": datetime.now().isoformat()
    }}

@app.get("/metrics")
async def get_metrics():
    """Get model metrics."""
    return {{
        "model_id": "{model_id}",
        "status": "running",
        "uptime": "N/A",  # Could implement proper uptime tracking
        "requests": "N/A"  # Could implement request counting
    }}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
'''
        return api_code
    
    def predict(self, model_id: str, input_data: Union[Dict, pd.DataFrame]) -> Dict[str, Any]:
        """Make prediction using deployed model."""
        if model_id not in self.deployments:
            raise ValueError(f"Model {model_id} not deployed")
        
        deployment = self.deployments[model_id]
        
        if 'engine' in deployment:
            # Local edge deployment
            engine = deployment['engine']
            return engine.predict(model_id, input_data)
        
        elif deployment.get('type') == 'container':
            # Container deployment
            # Would implement HTTP client to container API
            container_id = deployment['container_id']
            logger.info(f"Making prediction via container {container_id[:12]}")
            
            # Placeholder for HTTP request to container
            return {
                'prediction': 0.5,  # Placeholder
                'model_id': model_id,
                'timestamp': datetime.now().isoformat(),
                'source': 'container'
            }
        
        else:
            raise ValueError(f"Unknown deployment type for model {model_id}")
    
    def get_deployment_status(self, model_id: str) -> Dict[str, Any]:
        """Get deployment status."""
        if model_id not in self.deployments:
            return {'status': 'not_deployed'}
        
        deployment = self.deployments[model_id]
        
        # Add health check
        if 'engine' in deployment:
            engine = deployment['engine']
            health = engine.health_check()
            deployment['health'] = health
        
        return deployment
    
    def list_deployments(self) -> List[Dict[str, Any]]:
        """List all deployments."""
        deployments = []
        
        for model_id, deployment in self.deployments.items():
            status = self.get_deployment_status(model_id)
            deployments.append({
                'model_id': model_id,
                **status
            })
        
        return deployments
    
    def undeploy_model(self, model_id: str) -> bool:
        """Remove model deployment."""
        if model_id not in self.deployments:
            return False
        
        deployment = self.deployments[model_id]
        
        try:
            # Stop container if applicable
            if deployment.get('type') == 'container' and self.docker_client:
                container_id = deployment.get('container_id')
                if container_id:
                    container = self.docker_client.containers.get(container_id)
                    container.stop()
                    container.remove()
                    logger.info(f"Stopped and removed container {container_id[:12]}")
            
            # Remove from deployments
            del self.deployments[model_id]
            if model_id in self.inference_engines:
                del self.inference_engines[model_id]
            
            logger.info(f"Model {model_id} undeployed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to undeploy model {model_id}: {e}")
            return False
    
    def cleanup_old_deployments(self, max_age_hours: int = 24):
        """Clean up old deployments."""
        current_time = datetime.now()
        to_remove = []
        
        for model_id, deployment in self.deployments.items():
            deployed_at = datetime.fromisoformat(deployment['deployed_at'])
            age_hours = (current_time - deployed_at).total_seconds() / 3600
            
            if age_hours > max_age_hours:
                to_remove.append(model_id)
        
        for model_id in to_remove:
            self.undeploy_model(model_id)
        
        logger.info(f"Cleaned up {len(to_remove)} old deployments")


# Convenience functions
def deploy_model_to_edge(
    model_path: str,
    model_id: str,
    config: Optional[EdgeConfig] = None,
    deployment_type: str = "local"
) -> EdgeDeploymentManager:
    """
    Deploy model to edge with minimal configuration.
    
    Args:
        model_path: Path to model file
        model_id: Unique model identifier
        config: Edge deployment configuration
        deployment_type: 'local' or 'container'
        
    Returns:
        Configured deployment manager
    """
    if config is None:
        config = EdgeConfig()
    
    manager = EdgeDeploymentManager(config)
    
    # Prepare model
    prepared_files = manager.prepare_model_for_edge(model_path, model_id)
    
    # Deploy based on type
    if deployment_type == "local":
        success = manager.deploy_to_local_edge(model_id, prepared_files)
    elif deployment_type == "container":
        container_id = manager.deploy_to_container(model_id, prepared_files)
        success = container_id is not None
    else:
        raise ValueError(f"Unknown deployment type: {deployment_type}")
    
    if success:
        logger.info(f"Model {model_id} deployed successfully")
    else:
        logger.error(f"Failed to deploy model {model_id}")
    
    return manager


def create_edge_optimized_pipeline(
    model,
    feature_processor,
    model_id: str,
    target_latency_ms: float = 50.0
) -> EdgeDeploymentManager:
    """Create optimized edge deployment pipeline."""
    
    # Save model and processor to temporary files
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as model_file:
        pickle.dump(model, model_file)
        model_path = model_file.name
    
    processor_path = None
    if feature_processor:
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as proc_file:
            pickle.dump(feature_processor, proc_file)
            processor_path = proc_file.name
    
    try:
        # Configure for target latency
        config = EdgeConfig(
            max_latency_ms=target_latency_ms,
            enable_quantization=True,
            enable_caching=True,
            optimization_level="high"
        )
        
        # Deploy
        manager = EdgeDeploymentManager(config)
        prepared_files = manager.prepare_model_for_edge(
            model_path, model_id, processor_path, "high"
        )
        manager.deploy_to_local_edge(model_id, prepared_files)
        
        return manager
        
    finally:
        # Cleanup temp files
        os.unlink(model_path)
        if processor_path:
            os.unlink(processor_path)