"""Rate limiting middleware for API."""

from datetime import datetime, timedelta
from typing import Dict, List
from fastapi import Request
from fastapi.responses import JSONResponse
import structlog

log = structlog.get_logger()


class RateLimiter:
    """Rate limiting middleware."""
    
    def __init__(self, requests_limit: int = 100, window_seconds: int = 60):
        """
        Initialize rate limiter.
        
        Args:
            requests_limit: Maximum requests per window
            window_seconds: Time window in seconds
        """
        self.requests_limit = requests_limit
        self.window_seconds = window_seconds
        self.request_counts: Dict[str, List[datetime]] = {}
    
    async def __call__(self, request: Request, call_next):
        """
        Rate limit middleware.
        
        Args:
            request: FastAPI request
            call_next: Next middleware/handler
            
        Returns:
            Response or rate limit error
        """
        client_ip = request.client.host
        
        # Clean old timestamps
        now = datetime.now()
        cutoff = now - timedelta(seconds=self.window_seconds)
        
        if client_ip in self.request_counts:
            self.request_counts[client_ip] = [
                t for t in self.request_counts[client_ip]
                if t > cutoff
            ]
        
        # Check limit
        request_count = len(self.request_counts.get(client_ip, []))
        
        if request_count >= self.requests_limit:
            log.warning(
                "rate_limit_exceeded",
                client_ip=client_ip,
                count=request_count
            )
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded"}
            )
        
        # Add request
        if client_ip not in self.request_counts:
            self.request_counts[client_ip] = []
        self.request_counts[client_ip].append(now)
        
        # Process request
        response = await call_next(request)
        return response
    
    def reset_client(self, client_ip: str):
        """Reset rate limit for specific client."""
        if client_ip in self.request_counts:
            del self.request_counts[client_ip]
    
    def get_client_status(self, client_ip: str) -> Dict:
        """Get rate limit status for client."""
        now = datetime.now()
        cutoff = now - timedelta(seconds=self.window_seconds)
        
        timestamps = self.request_counts.get(client_ip, [])
        valid_timestamps = [t for t in timestamps if t > cutoff]
        
        return {
            "requests": len(valid_timestamps),
            "limit": self.requests_limit,
            "window_seconds": self.window_seconds,
            "reset_at": (cutoff + timedelta(seconds=self.window_seconds)).isoformat()
        }