"""Unit tests for API Pydantic models."""

import pytest
from pydantic import ValidationError
from src.api.models import (
    PredictionRequest,
    BatchPredictionRequest,
    PredictionResponse,
    HealthResponse,
    MetricsResponse,
    WebSocketMessage,
    WebSocketResponse
)


class TestPredictionRequest:
    """Test cases for PredictionRequest model."""
    
    def test_valid_request(self):
        """Test valid prediction request."""
        request = PredictionRequest(
            features={"feature1": 1.0, "feature2": 2.0},
            request_id="req_123",
            return_probabilities=True,
            return_shap=False
        )
        
        assert request.features == {"feature1": 1.0, "feature2": 2.0}
        assert request.request_id == "req_123"
        assert request.return_probabilities is True
        assert request.return_shap is False
    
    def test_minimal_request(self):
        """Test minimal valid request."""
        request = PredictionRequest(features={"feature1": 1.0})
        
        assert request.features == {"feature1": 1.0}
        assert request.request_id is None
        assert request.return_probabilities is False
        assert request.return_shap is False
    
    def test_empty_features_invalid(self):
        """Test that empty features are invalid."""
        with pytest.raises(ValidationError) as exc_info:
            PredictionRequest(features={})
        
        assert "Features cannot be empty" in str(exc_info.value)
    
    def test_missing_features_invalid(self):
        """Test that missing features are invalid."""
        with pytest.raises(ValidationError):
            PredictionRequest()


class TestBatchPredictionRequest:
    """Test cases for BatchPredictionRequest model."""
    
    def test_valid_batch_request(self):
        """Test valid batch prediction request."""
        request = BatchPredictionRequest(
            samples=[
                {"feature1": 1.0, "feature2": 2.0},
                {"feature1": 3.0, "feature2": 4.0}
            ],
            request_id="batch_123",
            return_probabilities=True
        )
        
        assert len(request.samples) == 2
        assert request.request_id == "batch_123"
        assert request.return_probabilities is True
    
    def test_empty_samples_invalid(self):
        """Test that empty samples are invalid."""
        with pytest.raises(ValidationError) as exc_info:
            BatchPredictionRequest(samples=[])
        
        assert "Samples cannot be empty" in str(exc_info.value)
    
    def test_batch_size_limit(self):
        """Test batch size limit validation."""
        with patch("src.api.config.settings") as mock_settings:
            mock_settings.max_batch_size = 2
            
            # Should fail with 3 samples when limit is 2
            with pytest.raises(ValidationError) as exc_info:
                BatchPredictionRequest(
                    samples=[{"f": 1.0}, {"f": 2.0}, {"f": 3.0}]
                )
            
            assert "Batch size exceeds maximum" in str(exc_info.value)


class TestPredictionResponse:
    """Test cases for PredictionResponse model."""
    
    def test_full_response(self):
        """Test full prediction response."""
        response = PredictionResponse(
            prediction=0.8,
            probabilities={"class_0": 0.2, "class_1": 0.8},
            shap_values={"feature1": 0.3, "feature2": -0.1},
            confidence=0.8,
            request_id="req_123",
            model_version="1.0",
            timestamp="2024-01-01T00:00:00"
        )
        
        assert response.prediction == 0.8
        assert response.probabilities["class_1"] == 0.8
        assert response.shap_values["feature1"] == 0.3
        assert response.confidence == 0.8
    
    def test_minimal_response(self):
        """Test minimal prediction response."""
        response = PredictionResponse(
            prediction=1,
            model_version="1.0",
            timestamp="2024-01-01T00:00:00"
        )
        
        assert response.prediction == 1
        assert response.probabilities is None
        assert response.shap_values is None
        assert response.confidence is None
        assert response.request_id is None


class TestHealthResponse:
    """Test cases for HealthResponse model."""
    
    def test_health_response(self):
        """Test health response model."""
        response = HealthResponse(
            status="healthy",
            model_loaded=True,
            cache_connected=True,
            uptime_seconds=3600.0,
            last_prediction="2024-01-01T00:00:00",
            model_info={"name": "test_model", "version": "1.0"}
        )
        
        assert response.status == "healthy"
        assert response.model_loaded is True
        assert response.cache_connected is True
        assert response.uptime_seconds == 3600.0
    
    def test_degraded_health(self):
        """Test degraded health response."""
        response = HealthResponse(
            status="degraded",
            model_loaded=True,
            cache_connected=False,
            uptime_seconds=100.0,
            last_prediction=None,
            model_info={}
        )
        
        assert response.status == "degraded"
        assert response.cache_connected is False
        assert response.last_prediction is None


class TestMetricsResponse:
    """Test cases for MetricsResponse model."""
    
    def test_metrics_response(self):
        """Test metrics response model."""
        response = MetricsResponse(
            total_predictions=1000,
            avg_latency_ms=50.5,
            cache_hit_rate=0.75,
            error_rate=0.01,
            requests_per_minute=10.5,
            model_drift_score=0.05
        )
        
        assert response.total_predictions == 1000
        assert response.avg_latency_ms == 50.5
        assert response.cache_hit_rate == 0.75
        assert response.error_rate == 0.01
        assert response.requests_per_minute == 10.5
        assert response.model_drift_score == 0.05
    
    def test_metrics_without_drift(self):
        """Test metrics response without drift score."""
        response = MetricsResponse(
            total_predictions=100,
            avg_latency_ms=25.0,
            cache_hit_rate=0.5,
            error_rate=0.0,
            requests_per_minute=5.0,
            model_drift_score=None
        )
        
        assert response.model_drift_score is None


class TestWebSocketModels:
    """Test cases for WebSocket models."""
    
    def test_websocket_message(self):
        """Test WebSocket message model."""
        message = WebSocketMessage(
            token="auth_token",
            features={"feature1": 1.0, "feature2": 2.0},
            request_id="ws_req_123"
        )
        
        assert message.token == "auth_token"
        assert message.features == {"feature1": 1.0, "feature2": 2.0}
        assert message.request_id == "ws_req_123"
    
    def test_websocket_message_minimal(self):
        """Test minimal WebSocket message."""
        message = WebSocketMessage(
            token="auth_token",
            features={"feature1": 1.0}
        )
        
        assert message.request_id is None
    
    def test_websocket_response_success(self):
        """Test successful WebSocket response."""
        response = WebSocketResponse(
            prediction=0.8,
            error=None,
            timestamp="2024-01-01T00:00:00",
            model_version="1.0"
        )
        
        assert response.prediction == 0.8
        assert response.error is None
        assert response.model_version == "1.0"
    
    def test_websocket_response_error(self):
        """Test error WebSocket response."""
        response = WebSocketResponse(
            prediction=None,
            error="Model not loaded",
            timestamp="2024-01-01T00:00:00",
            model_version=None
        )
        
        assert response.prediction is None
        assert response.error == "Model not loaded"
        assert response.model_version is None


# Import patch for the batch size test
from unittest.mock import patch