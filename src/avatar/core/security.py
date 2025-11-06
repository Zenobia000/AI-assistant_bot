"""
Security module for AVATAR

Implements authentication, authorization, and security utilities.
Based on Linus principle: "Security should be simple and obvious, not clever."
"""

import os
import secrets
from typing import Optional

import structlog
from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

logger = structlog.get_logger()

# Security configuration
security = HTTPBearer(auto_error=False)

# API Token configuration
API_TOKEN_LENGTH = 32
DEFAULT_API_TOKEN = "dev-token-change-in-production"


def get_api_token() -> str:
    """
    Get API token from environment or return default for development

    Returns:
        API token string

    Note:
        In production, AVATAR_API_TOKEN environment variable MUST be set
    """
    token = os.getenv("AVATAR_API_TOKEN")

    if not token:
        if os.getenv("AVATAR_ENV") == "production":
            raise ValueError(
                "AVATAR_API_TOKEN environment variable is required in production"
            )

        # Development mode - use default token
        logger.warning(
            "security.default_token",
            message="Using default API token. Set AVATAR_API_TOKEN in production."
        )
        return DEFAULT_API_TOKEN

    return token


def generate_api_token() -> str:
    """
    Generate a cryptographically secure API token

    Returns:
        Random token string suitable for API authentication
    """
    return secrets.token_urlsafe(API_TOKEN_LENGTH)


async def verify_api_token(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> bool:
    """
    Verify API token from Authorization header

    Args:
        credentials: HTTP Authorization credentials

    Returns:
        True if token is valid

    Raises:
        HTTPException: If token is invalid or missing
    """
    # Check if credentials provided
    if not credentials:
        logger.warning("security.auth_missing", reason="no_credentials")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Verify token
    expected_token = get_api_token()
    provided_token = credentials.credentials

    # Use constant-time comparison to prevent timing attacks
    if not secrets.compare_digest(provided_token, expected_token):
        logger.warning(
            "security.auth_failed",
            reason="invalid_token",
            token_length=len(provided_token)
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    logger.info("security.auth_success")
    return True


async def optional_api_token(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> bool:
    """
    Optional API token verification for endpoints that support both auth and no-auth

    Args:
        credentials: HTTP Authorization credentials

    Returns:
        True if authenticated, False if no credentials provided

    Raises:
        HTTPException: If token is provided but invalid
    """
    if not credentials:
        logger.debug("security.no_auth", reason="no_credentials")
        return False

    # If credentials provided, they must be valid
    return await verify_api_token(credentials)


class SecurityConfig:
    """Security configuration and utilities"""

    # Rate limiting
    DEFAULT_RATE_LIMIT = "60/minute"
    UPLOAD_RATE_LIMIT = "10/minute"

    # File upload limits
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    MIN_FILE_SIZE = 1024  # 1KB

    # Allowed file extensions (whitelist)
    ALLOWED_AUDIO_EXTENSIONS = {'.wav', '.mp3', '.m4a', '.flac', '.ogg'}

    # Allowed MIME types (strict)
    ALLOWED_AUDIO_MIME_TYPES = {
        'audio/wav', 'audio/wave', 'audio/x-wav',
        'audio/mpeg', 'audio/mp3',
        'audio/mp4', 'audio/m4a',
        'audio/flac',
        'audio/ogg', 'audio/vorbis'
    }

    @staticmethod
    def is_production() -> bool:
        """Check if running in production environment"""
        return os.getenv("AVATAR_ENV", "development").lower() == "production"

    @staticmethod
    def require_https() -> bool:
        """Check if HTTPS should be enforced"""
        return SecurityConfig.is_production()


# Security headers middleware configuration
def get_security_headers() -> dict:
    """
    Get recommended security headers for HTTP responses

    Returns:
        Dictionary of security headers
    """
    headers = {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
        "Referrer-Policy": "strict-origin-when-cross-origin",
        "Content-Security-Policy": "default-src 'self'",
    }

    if SecurityConfig.require_https():
        headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"

    return headers


# Utility function for input validation
def validate_input_string(value: str, field_name: str, max_length: int = 255) -> str:
    """
    Validate and sanitize input string

    Args:
        value: Input string to validate
        field_name: Field name for error messages
        max_length: Maximum allowed length

    Returns:
        Validated and sanitized string

    Raises:
        HTTPException: If validation fails
    """
    if not value or not value.strip():
        raise HTTPException(
            status_code=400,
            detail=f"{field_name} cannot be empty"
        )

    value = value.strip()

    if len(value) > max_length:
        raise HTTPException(
            status_code=400,
            detail=f"{field_name} exceeds maximum length of {max_length} characters"
        )

    # Basic XSS prevention - reject HTML tags
    if '<' in value or '>' in value:
        raise HTTPException(
            status_code=400,
            detail=f"{field_name} contains invalid characters"
        )

    return value


def safe_error_response(error_message: str, include_detail: bool = False) -> str:
    """
    Create safe error response that doesn't leak sensitive information

    Args:
        error_message: Original error message
        include_detail: Whether to include detailed error (dev mode only)

    Returns:
        Sanitized error message
    """
    if SecurityConfig.is_production() and not include_detail:
        return "An error occurred while processing your request"

    # In development, include more detail but sanitize sensitive paths
    safe_message = error_message

    # Remove file system paths
    import re
    safe_message = re.sub(r'/[a-zA-Z0-9/_-]+', '/[path-hidden]', safe_message)

    # Remove potential secrets (anything that looks like a key)
    safe_message = re.sub(r'[a-zA-Z0-9]{20,}', '[key-hidden]', safe_message)

    return safe_message