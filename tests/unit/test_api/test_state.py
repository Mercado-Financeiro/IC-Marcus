"""Unit tests for API state module."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch
from src.api.state import AppState


class TestAppState:
    """Test cases for AppState."""
    
    @pytest.fixture
    def app_state(self):
        """Create app state instance."""
        return AppState()
    
    def test_initialization(self, app_state):
        """Test app state initialization."""
        assert app_state.model is None
        assert app_state.model_version is None
        assert app_state.feature_names is None
        assert app_state.redis_client is None
        assert app_state.prediction_count == 0
        assert app_state.total_latency == 0.0
        assert app_state.cache_hits == 0
        assert app_state.cache_misses == 0
        assert app_state.errors == 0
        assert isinstance(app_state.start_time, datetime)
    
    @pytest.mark.asyncio
    async def test_load_model_success(self, app_state):
        """Test successful model loading."""
        with patch("mlflow.set_tracking_uri") as mock_set_uri, \
             patch("mlflow.pyfunc.load_model") as mock_load_model, \
             patch("mlflow.tracking.MlflowClient") as mock_client_class:
            
            # Mock settings
            settings = Mock()
            settings.mlflow_tracking_uri = "test_uri"
            settings.model_name = "test_model"
            settings.model_stage = "Production"
            
            # Mock MLflow client
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            
            # Mock model version
            mock_version = Mock()
            mock_version.version = "1.0"
            mock_version.run_id = "run123"
            mock_client.get_latest_versions.return_value = [mock_version]
            
            # Mock run
            mock_run = Mock()
            mock_run.data.params = {"feature_names": "feat1,feat2,feat3"}
            mock_client.get_run.return_value = mock_run
            
            # Mock loaded model
            mock_model = Mock()
            mock_load_model.return_value = mock_model
            
            await app_state.load_model(settings)
            
            assert app_state.model == mock_model
            assert app_state.model_version == "1.0"
            assert app_state.feature_names == ["feat1", "feat2", "feat3"]
    
    @pytest.mark.asyncio
    async def test_load_model_no_features(self, app_state):
        """Test model loading without feature names."""
        with patch("mlflow.set_tracking_uri"), \
             patch("mlflow.pyfunc.load_model") as mock_load_model, \
             patch("mlflow.tracking.MlflowClient") as mock_client_class:
            
            settings = Mock()
            settings.mlflow_tracking_uri = "test_uri"
            settings.model_name = "test_model"
            settings.model_stage = "Production"
            
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            
            mock_version = Mock()
            mock_version.version = "1.0"
            mock_version.run_id = "run123"
            mock_client.get_latest_versions.return_value = [mock_version]
            
            # Simulate exception when getting run
            mock_client.get_run.side_effect = Exception("Run not found")
            
            await app_state.load_model(settings)
            
            assert app_state.model is not None
            assert app_state.model_version == "1.0"
            assert app_state.feature_names is None
    
    @pytest.mark.asyncio
    async def test_connect_redis_success(self, app_state):
        """Test successful Redis connection."""
        settings = Mock()
        settings.redis_host = "localhost"
        settings.redis_port = 6379
        
        with patch("redis.Redis") as mock_redis_class:
            mock_redis = Mock()
            mock_redis_class.return_value = mock_redis
            
            await app_state.connect_redis(settings)
            
            assert app_state.redis_client == mock_redis
            mock_redis.ping.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_connect_redis_failure(self, app_state):
        """Test Redis connection failure."""
        settings = Mock()
        settings.redis_host = "localhost"
        settings.redis_port = 6379
        
        with patch("redis.Redis") as mock_redis_class:
            mock_redis = Mock()
            mock_redis.ping.side_effect = Exception("Connection failed")
            mock_redis_class.return_value = mock_redis
            
            await app_state.connect_redis(settings)
            
            assert app_state.redis_client is None
    
    @pytest.mark.asyncio
    async def test_initialize_drift_monitor(self, app_state):
        """Test drift monitor initialization."""
        with patch("src.monitoring.drift_monitor.DriftMonitor") as mock_drift_class:
            mock_drift = Mock()
            mock_drift_class.return_value = mock_drift
            
            await app_state.initialize_drift_monitor()
            
            assert app_state.drift_monitor == mock_drift
    
    @pytest.mark.asyncio
    async def test_initialize_drift_monitor_failure(self, app_state):
        """Test drift monitor initialization failure."""
        with patch("src.monitoring.drift_monitor.DriftMonitor") as mock_drift_class:
            mock_drift_class.side_effect = Exception("Import error")
            
            await app_state.initialize_drift_monitor()
            
            assert app_state.drift_monitor is None
    
    def test_update_prediction_metrics(self, app_state):
        """Test updating prediction metrics."""
        initial_count = app_state.prediction_count
        initial_latency = app_state.total_latency
        
        app_state.update_prediction_metrics(150.5)
        
        assert app_state.prediction_count == initial_count + 1
        assert app_state.total_latency == initial_latency + 150.5
        assert app_state.last_prediction_time is not None
    
    def test_update_cache_metrics_hit(self, app_state):
        """Test updating cache metrics for hit."""
        initial_hits = app_state.cache_hits
        
        app_state.update_cache_metrics(hit=True)
        
        assert app_state.cache_hits == initial_hits + 1
        assert app_state.cache_misses == 0
    
    def test_update_cache_metrics_miss(self, app_state):
        """Test updating cache metrics for miss."""
        initial_misses = app_state.cache_misses
        
        app_state.update_cache_metrics(hit=False)
        
        assert app_state.cache_misses == initial_misses + 1
        assert app_state.cache_hits == 0
    
    def test_increment_errors(self, app_state):
        """Test incrementing error count."""
        initial_errors = app_state.errors
        
        app_state.increment_errors()
        
        assert app_state.errors == initial_errors + 1
    
    def test_get_metrics_empty(self, app_state):
        """Test getting metrics with no activity."""
        metrics = app_state.get_metrics()
        
        assert metrics["total_predictions"] == 0
        assert metrics["avg_latency_ms"] == 0
        assert metrics["cache_hit_rate"] == 0
        assert metrics["error_rate"] == 0
        assert metrics["requests_per_minute"] == 0
        assert metrics["uptime_seconds"] >= 0
    
    def test_get_metrics_with_activity(self, app_state):
        """Test getting metrics with activity."""
        # Simulate activity
        app_state.update_prediction_metrics(100)
        app_state.update_prediction_metrics(200)
        app_state.update_cache_metrics(hit=True)
        app_state.update_cache_metrics(hit=False)
        app_state.increment_errors()
        
        metrics = app_state.get_metrics()
        
        assert metrics["total_predictions"] == 2
        assert metrics["avg_latency_ms"] == 150  # (100 + 200) / 2
        assert metrics["cache_hit_rate"] == 0.5  # 1 hit / 2 total
        assert metrics["error_rate"] == 1/3  # 1 error / 3 total requests
        assert metrics["requests_per_minute"] > 0
    
    @pytest.mark.asyncio
    async def test_cleanup_with_redis(self, app_state):
        """Test cleanup with Redis connection."""
        mock_redis = Mock()
        app_state.redis_client = mock_redis
        
        await app_state.cleanup()
        
        mock_redis.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_cleanup_without_redis(self, app_state):
        """Test cleanup without Redis connection."""
        app_state.redis_client = None
        
        # Should not raise exception
        await app_state.cleanup()
    
    def test_request_counts_tracking(self, app_state):
        """Test request counts tracking."""
        assert app_state.request_counts == {}
        
        app_state.request_counts["192.168.1.1"] = [datetime.now()]
        assert "192.168.1.1" in app_state.request_counts
        assert len(app_state.request_counts["192.168.1.1"]) == 1