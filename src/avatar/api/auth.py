"""
Simple API Authentication for Session Control

Task 21: Basic API key authentication for admin endpoints.
Minimal implementation following Linus principle: "Simple things should be simple."
"""

from fastapi import HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional

from avatar.core.config import config

security = HTTPBearer()


def verify_api_key(credentials: HTTPAuthorizationCredentials = Security(security)) -> str:
    """
    Verify API key for admin endpoints

    Args:
        credentials: Bearer token credentials

    Returns:
        API key if valid

    Raises:
        HTTPException: If invalid or missing API key
    """
    if not credentials:
        raise HTTPException(status_code=401, detail="API key required")

    api_key = credentials.credentials

    # Simple validation - in production, use proper secret management
    valid_api_key = getattr(config, 'ADMIN_API_KEY', 'avatar-admin-key-2024')

    if api_key != valid_api_key:
        raise HTTPException(status_code=403, detail="Invalid API key")

    return api_key


def optional_api_key(credentials: Optional[HTTPAuthorizationCredentials] = None) -> Optional[str]:
    """
    Optional API key verification for enhanced endpoints

    Args:
        credentials: Optional bearer token credentials

    Returns:
        API key if valid, None if not provided
    """
    if not credentials:
        return None

    try:
        return verify_api_key(credentials)
    except HTTPException:
        return None