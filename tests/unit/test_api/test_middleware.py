"""Unit tests for API middleware."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

from src.api.middleware.auth import verify_token, optional_token
from src.api.middleware.rate_limit import RateLimiter


class TestAuthMiddleware:
    """Test cases for authentication middleware."""
    
    @pytest.mark.asyncio
    async def test_verify_token_valid(self):
        """Test token verification with valid token."""
        with patch("src.api.config.settings") as mock_settings:
            mock_settings.api_key = "valid_token_123"
            
            credentials = Mock(spec=HTTPAuthorizationCredentials)
            credentials.credentials = "valid_token_123"
            
            result = await verify_token(credentials)
            assert result == "valid_token_123"
    
    @pytest.mark.asyncio
    async def test_verify_token_invalid(self):
        """Test token verification with invalid token."""
        with patch("src.api.config.settings") as mock_settings:
            mock_settings.api_key = "valid_token_123"
            
            credentials = Mock(spec=HTTPAuthorizationCredentials)
            credentials.credentials = "invalid_token"
            
            with pytest.raises(HTTPException) as exc_info:
                await verify_token(credentials)
            
            assert exc_info.value.status_code == 403
            assert "Invalid authentication token" in str(exc_info.value.detail)
    
    @pytest.mark.asyncio
    async def test_verify_token_with_custom_api_key(self):
        """Test token verification with custom API key."""
        credentials = Mock(spec=HTTPAuthorizationCredentials)
        credentials.credentials = "custom_key_456"
        
        result = await verify_token(credentials, api_key="custom_key_456")
        assert result == "custom_key_456"
    
    @pytest.mark.asyncio
    async def test_optional_token_valid(self):
        """Test optional token with valid credentials."""
        with patch("src.api.middleware.auth.verify_token") as mock_verify:
            mock_verify.return_value = "valid_token"
            
            credentials = Mock(spec=HTTPAuthorizationCredentials)
            result = await optional_token(credentials)
            
            assert result == "valid_token"
    
    @pytest.mark.asyncio
    async def test_optional_token_invalid(self):
        """Test optional token with invalid credentials."""
        with patch("src.api.middleware.auth.verify_token") as mock_verify:
            mock_verify.side_effect = HTTPException(status_code=403, detail="Invalid")
            
            credentials = Mock(spec=HTTPAuthorizationCredentials)
            result = await optional_token(credentials)
            
            assert result is None


class TestRateLimiter:
    """Test cases for rate limiting middleware."""
    
    @pytest.fixture
    def rate_limiter(self):
        """Create rate limiter instance."""
        return RateLimiter(requests_limit=3, window_seconds=60)
    
    @pytest.fixture
    def mock_request(self):
        """Create mock request."""
        request = Mock()
        request.client.host = "192.168.1.1"
        return request
    
    @pytest.fixture
    def mock_call_next(self):
        """Create mock call_next function."""
        async def call_next(request):
            response = Mock()
            response.status_code = 200
            return response
        return call_next
    
    @pytest.mark.asyncio
    async def test_rate_limit_allow(self, rate_limiter, mock_request, mock_call_next):
        """Test rate limiter allowing requests."""
        response = await rate_limiter(mock_request, mock_call_next)
        assert response.status_code == 200
    
    @pytest.mark.asyncio
    async def test_rate_limit_exceeded(self, rate_limiter, mock_request, mock_call_next):
        """Test rate limiter blocking excessive requests."""
        # Make requests up to limit
        for _ in range(3):
            response = await rate_limiter(mock_request, mock_call_next)
            assert response.status_code == 200
        
        # Next request should be blocked
        response = await rate_limiter(mock_request, mock_call_next)
        assert response.status_code == 429
        assert response.content["detail"] == "Rate limit exceeded"
    
    @pytest.mark.asyncio
    async def test_rate_limit_window_expiry(self, rate_limiter, mock_request, mock_call_next):
        """Test rate limit window expiry."""
        # Set window to 1 second for testing
        rate_limiter.window_seconds = 1
        
        # Make requests up to limit
        for _ in range(3):
            await rate_limiter(mock_request, mock_call_next)
        
        # Manually expire timestamps
        import time
        time.sleep(1.1)
        
        # Clean expired timestamps and allow new request
        response = await rate_limiter(mock_request, mock_call_next)
        assert response.status_code == 200
    
    @pytest.mark.asyncio
    async def test_rate_limit_multiple_clients(self, rate_limiter, mock_call_next):
        """Test rate limiting for multiple clients."""
        # Client 1
        request1 = Mock()
        request1.client.host = "192.168.1.1"
        
        # Client 2
        request2 = Mock()
        request2.client.host = "192.168.1.2"
        
        # Each client should have independent limits
        for _ in range(3):
            response1 = await rate_limiter(request1, mock_call_next)
            response2 = await rate_limiter(request2, mock_call_next)
            assert response1.status_code == 200
            assert response2.status_code == 200
        
        # Both should be blocked after limit
        response1 = await rate_limiter(request1, mock_call_next)
        response2 = await rate_limiter(request2, mock_call_next)
        assert response1.status_code == 429
        assert response2.status_code == 429
    
    def test_reset_client(self, rate_limiter):
        """Test resetting rate limit for specific client."""
        client_ip = "192.168.1.1"
        rate_limiter.request_counts[client_ip] = [datetime.now()]
        
        rate_limiter.reset_client(client_ip)
        
        assert client_ip not in rate_limiter.request_counts
    
    def test_reset_client_not_exists(self, rate_limiter):
        """Test resetting non-existent client."""
        # Should not raise exception
        rate_limiter.reset_client("192.168.1.99")
    
    def test_get_client_status(self, rate_limiter):
        """Test getting client rate limit status."""
        client_ip = "192.168.1.1"
        now = datetime.now()
        
        # Add some requests
        rate_limiter.request_counts[client_ip] = [
            now - timedelta(seconds=30),
            now - timedelta(seconds=20),
            now - timedelta(seconds=10)
        ]
        
        status = rate_limiter.get_client_status(client_ip)
        
        assert status["requests"] == 3
        assert status["limit"] == 3
        assert status["window_seconds"] == 60
        assert "reset_at" in status
    
    def test_get_client_status_no_requests(self, rate_limiter):
        """Test getting status for client with no requests."""
        status = rate_limiter.get_client_status("192.168.1.99")
        
        assert status["requests"] == 0
        assert status["limit"] == 3
    
    def test_get_client_status_expired_requests(self, rate_limiter):
        """Test getting status with expired requests."""
        client_ip = "192.168.1.1"
        now = datetime.now()
        
        # Add old requests (outside window)
        rate_limiter.request_counts[client_ip] = [
            now - timedelta(seconds=120),
            now - timedelta(seconds=90),
            now - timedelta(seconds=30)  # Only this one is valid
        ]
        
        status = rate_limiter.get_client_status(client_ip)
        
        assert status["requests"] == 1  # Only one request in window