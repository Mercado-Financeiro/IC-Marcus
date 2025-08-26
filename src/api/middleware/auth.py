"""Authentication middleware for API."""

from typing import Optional
from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import structlog

log = structlog.get_logger()

# Security scheme
security = HTTPBearer()


async def verify_token(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    api_key: str = None
) -> str:
    """
    Verify authentication token.
    
    Args:
        credentials: HTTP authorization credentials
        api_key: Expected API key
        
    Returns:
        Validated token
        
    Raises:
        HTTPException: If token is invalid
    """
    from src.api.config import settings
    
    token = credentials.credentials
    expected_key = api_key or settings.api_key
    
    # Simple token verification - in production use JWT
    if token != expected_key:
        log.warning("Invalid authentication attempt", token_prefix=token[:8])
        raise HTTPException(
            status_code=403,
            detail="Invalid authentication token"
        )
    
    return token


async def optional_token(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> Optional[str]:
    """
    Optional token verification.
    
    Args:
        credentials: HTTP authorization credentials
        
    Returns:
        Token if valid, None otherwise
    """
    try:
        return await verify_token(credentials)
    except HTTPException:
        return None