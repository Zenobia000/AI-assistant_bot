"""
WebSocket Recovery API - Session state management and recovery endpoints

Task 22: REST API endpoints to support WebSocket reconnection and session recovery.
Provides session status, recovery capabilities, and connection guidance.

Key Features:
1. Session status and recovery endpoints
2. Connection health monitoring API
3. Client-side reconnection guidance
4. Session preservation management
"""

from fastapi import APIRouter, HTTPException, Query, Depends
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
import time

import structlog

from avatar.core.websocket_reconnect import get_reconnect_manager, ConnectionState
from avatar.api.auth import optional_api_key

logger = structlog.get_logger()
router = APIRouter(prefix="/api/v1/websocket", tags=["WebSocket Recovery"])


# Pydantic models for recovery API
class SessionStatusResponse(BaseModel):
    """Session status response model"""
    session_id: str
    exists: bool
    recoverable: bool
    turn_number: int
    last_activity: float
    processing_stage: str
    voice_profile_id: Optional[int] = None
    age_seconds: float
    idle_seconds: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ConnectionStatusResponse(BaseModel):
    """Connection status response model"""
    connection_id: str
    state: str
    retry_count: int
    last_disconnect_reason: Optional[str]
    active_sessions: int
    statistics: Dict[str, Any]
    next_retry_delay_seconds: Optional[float] = None
    can_reconnect: bool
    timestamp: float = Field(default_factory=time.time)


class RecoveryGuidanceResponse(BaseModel):
    """Recovery guidance response model"""
    should_reconnect: bool
    wait_seconds: Optional[float]
    max_retries_exceeded: bool
    session_recovery_available: bool
    reconnection_url: str
    error_classification: Optional[str]
    guidance_message: str
    timestamp: float = Field(default_factory=time.time)


class SessionListResponse(BaseModel):
    """List of recoverable sessions"""
    sessions: List[SessionStatusResponse]
    total_sessions: int
    recoverable_sessions: int
    expired_sessions: int
    timestamp: float = Field(default_factory=time.time)


@router.get("/session/{session_id}/status", response_model=SessionStatusResponse)
async def get_session_status(session_id: str):
    """
    Get status of a specific WebSocket session

    Returns detailed information about session state, including:
    - Whether session exists and is recoverable
    - Current processing stage and progress
    - Activity timestamps and metadata
    """
    reconnect_manager = get_reconnect_manager()
    snapshot = reconnect_manager.active_sessions.get(session_id)

    if not snapshot:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    return SessionStatusResponse(
        session_id=session_id,
        exists=True,
        recoverable=snapshot.is_recoverable(),
        turn_number=snapshot.turn_number,
        last_activity=snapshot.last_activity,
        processing_stage=snapshot.processing_stage,
        voice_profile_id=snapshot.voice_profile_id,
        age_seconds=snapshot.age_seconds,
        idle_seconds=snapshot.idle_seconds,
        metadata=snapshot.metadata
    )


@router.get("/session/{session_id}/recovery", response_model=RecoveryGuidanceResponse)
async def get_recovery_guidance(session_id: str):
    """
    Get reconnection guidance for a specific session

    Provides intelligent guidance on whether and how to reconnect:
    - Reconnection feasibility and timing
    - Session recovery availability
    - Error classification and retry strategy
    """
    reconnect_manager = get_reconnect_manager()
    snapshot = reconnect_manager.active_sessions.get(session_id)

    # Check connection manager status
    status = reconnect_manager.get_status()
    can_reconnect = status["state"] not in ["failed", "suspended"]
    is_retrying = status["state"] == "reconnecting"

    # Calculate retry delay if actively retrying
    wait_seconds = None
    if is_retrying and status["retry_count"] > 0:
        wait_seconds = reconnect_manager.calculate_backoff_delay()

    # Determine session recovery availability
    session_recovery_available = snapshot is not None and snapshot.is_recoverable()

    # Generate guidance message
    if not can_reconnect:
        guidance_message = "Reconnection not available - maximum retries exceeded"
    elif is_retrying:
        guidance_message = f"Reconnection in progress (attempt {status['retry_count']})"
    elif session_recovery_available:
        guidance_message = "Session can be recovered - reconnect with session ID"
    else:
        guidance_message = "Start new session - previous session expired"

    return RecoveryGuidanceResponse(
        should_reconnect=can_reconnect and not is_retrying,
        wait_seconds=wait_seconds,
        max_retries_exceeded=status["state"] == "failed",
        session_recovery_available=session_recovery_available,
        reconnection_url=f"/ws/chat?session_id={session_id}" if session_recovery_available else "/ws/chat",
        error_classification=status["last_disconnect_reason"],
        guidance_message=guidance_message
    )


@router.get("/connection/status", response_model=ConnectionStatusResponse)
async def get_connection_status():
    """
    Get current WebSocket connection manager status

    Returns comprehensive information about:
    - Connection state and retry attempts
    - Active session count and statistics
    - Reconnection feasibility and timing
    """
    reconnect_manager = get_reconnect_manager()
    status = reconnect_manager.get_status()

    # Calculate next retry delay if applicable
    next_retry_delay = None
    if status["state"] == "reconnecting":
        next_retry_delay = reconnect_manager.calculate_backoff_delay()

    return ConnectionStatusResponse(
        connection_id=status["connection_id"],
        state=status["state"],
        retry_count=status["retry_count"],
        last_disconnect_reason=status["last_disconnect_reason"],
        active_sessions=status["active_sessions"],
        statistics=status["statistics"],
        next_retry_delay_seconds=next_retry_delay,
        can_reconnect=status["state"] not in ["failed", "suspended"]
    )


@router.get("/sessions", response_model=SessionListResponse)
async def list_recoverable_sessions():
    """
    List all recoverable WebSocket sessions

    Returns summary of all sessions that can be recovered:
    - Session IDs and basic status
    - Recovery feasibility and timing
    - Cleanup and expiration information
    """
    reconnect_manager = get_reconnect_manager()

    sessions = []
    recoverable_count = 0
    expired_count = 0

    for session_id, snapshot in reconnect_manager.active_sessions.items():
        is_recoverable = snapshot.is_recoverable()

        if is_recoverable:
            recoverable_count += 1
        else:
            expired_count += 1

        sessions.append(SessionStatusResponse(
            session_id=session_id,
            exists=True,
            recoverable=is_recoverable,
            turn_number=snapshot.turn_number,
            last_activity=snapshot.last_activity,
            processing_stage=snapshot.processing_stage,
            voice_profile_id=snapshot.voice_profile_id,
            age_seconds=snapshot.age_seconds,
            idle_seconds=snapshot.idle_seconds,
            metadata=snapshot.metadata
        ))

    return SessionListResponse(
        sessions=sessions,
        total_sessions=len(sessions),
        recoverable_sessions=recoverable_count,
        expired_sessions=expired_count
    )


@router.post("/session/{session_id}/preserve")
async def preserve_session(session_id: str, extend_minutes: int = Query(30, ge=1, le=120)):
    """
    Extend session preservation time

    Allows extending the recovery window for a session:
    - Useful for planned maintenance or long disconnections
    - Prevents automatic session cleanup
    - Requires valid session ID
    """
    reconnect_manager = get_reconnect_manager()
    snapshot = reconnect_manager.active_sessions.get(session_id)

    if not snapshot:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    # Extend last activity time
    extend_seconds = extend_minutes * 60
    snapshot.last_activity = time.time()
    snapshot.metadata["preservation_extended"] = extend_seconds
    snapshot.metadata["preservation_reason"] = "manual_extension"

    logger.info("websocket_recovery.session_preserved",
               session_id=session_id,
               extend_minutes=extend_minutes)

    return {
        "success": True,
        "session_id": session_id,
        "extended_minutes": extend_minutes,
        "new_expiry_time": snapshot.last_activity + extend_seconds,
        "message": f"Session preserved for additional {extend_minutes} minutes"
    }


@router.delete("/session/{session_id}")
async def cleanup_session(session_id: str):
    """
    Manually cleanup a session

    Removes session from recovery cache:
    - Useful for privacy or storage management
    - Irreversible operation
    - Session cannot be recovered after cleanup
    """
    reconnect_manager = get_reconnect_manager()

    if session_id not in reconnect_manager.active_sessions:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    del reconnect_manager.active_sessions[session_id]

    logger.info("websocket_recovery.session_cleaned",
               session_id=session_id,
               reason="manual_cleanup")

    return {
        "success": True,
        "session_id": session_id,
        "message": "Session cleaned up successfully"
    }


@router.post("/connection/reset")
async def reset_connection_manager(api_key: str = Depends(optional_api_key)):
    """
    Reset connection manager state (admin only)

    Resets retry counters and connection state:
    - Useful for debugging and recovery
    - Clears retry backoff state
    - Does not affect active sessions
    """
    if not api_key:
        raise HTTPException(status_code=401, detail="API key required for admin operations")

    reconnect_manager = get_reconnect_manager()

    # Reset connection state
    reconnect_manager.retry_count = 0
    reconnect_manager.first_attempt_time = None
    reconnect_manager.last_attempt_time = None
    reconnect_manager.state = ConnectionState.DISCONNECTED
    reconnect_manager.last_disconnect_reason = None

    logger.info("websocket_recovery.connection_reset",
               connection_id=reconnect_manager.connection_id,
               admin_action=True)

    return {
        "success": True,
        "message": "Connection manager reset successfully",
        "connection_id": reconnect_manager.connection_id,
        "preserved_sessions": len(reconnect_manager.active_sessions)
    }


@router.get("/health")
async def get_websocket_health():
    """
    WebSocket system health check

    Returns health status for monitoring:
    - Connection manager state
    - Session recovery capability
    - System load and capacity
    """
    reconnect_manager = get_reconnect_manager()
    status = reconnect_manager.get_status()

    # Calculate health score
    health_score = 100
    issues = []

    if status["state"] == "failed":
        health_score -= 50
        issues.append("Connection manager in failed state")

    if status["retry_count"] > 5:
        health_score -= 20
        issues.append(f"High retry count: {status['retry_count']}")

    if status["active_sessions"] > 50:
        health_score -= 15
        issues.append(f"High session count: {status['active_sessions']}")

    return {
        "healthy": health_score >= 70,
        "health_score": health_score,
        "issues": issues,
        "connection_state": status["state"],
        "active_sessions": status["active_sessions"],
        "statistics": status["statistics"],
        "timestamp": time.time()
    }


@router.get("/metrics")
async def get_websocket_metrics():
    """
    Prometheus-compatible WebSocket metrics

    Returns metrics in plain text format:
    - Connection statistics
    - Session recovery rates
    - Error and retry metrics
    """
    reconnect_manager = get_reconnect_manager()
    status = reconnect_manager.get_status()

    metrics = []

    # Connection metrics
    metrics.append(f"avatar_websocket_active_sessions {status['active_sessions']}")
    metrics.append(f"avatar_websocket_retry_count {status['retry_count']}")

    # State metrics (1 for current state, 0 for others)
    for state in ["disconnected", "connecting", "connected", "reconnecting", "failed"]:
        value = 1 if status["state"] == state else 0
        metrics.append(f"avatar_websocket_state{{state=\"{state}\"}} {value}")

    # Statistics
    stats = status["statistics"]
    for metric_name, value in stats.items():
        metrics.append(f"avatar_websocket_{metric_name} {value}")

    return "\n".join(metrics), {"Content-Type": "text/plain"}