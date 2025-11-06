"""
WebSocket Reconnection and Recovery Manager

Task 22: Intelligent WebSocket reconnection with session state recovery.
Linus principle: "Robust systems handle failures gracefully, not perfectly."

Key Design Principles:
1. Exponential backoff with jitter - prevent thundering herd
2. Session state persistence - seamless user experience
3. Connection health monitoring - proactive failure detection
4. Graceful degradation - partial functionality over total failure
"""

import asyncio
import time
import uuid
from typing import Dict, Optional, Set, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import random

import structlog
from fastapi import WebSocket

from avatar.core.config import config

logger = structlog.get_logger()


class ConnectionState(Enum):
    """WebSocket connection states"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    FAILED = "failed"           # Max retries exceeded
    SUSPENDED = "suspended"     # Manually paused


class DisconnectReason(Enum):
    """Reasons for disconnection"""
    NETWORK_ERROR = "network_error"
    SERVER_ERROR = "server_error"
    CLIENT_CLOSE = "client_close"
    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit"
    AUTHENTICATION_FAILED = "auth_failed"
    SERVER_OVERLOAD = "server_overload"


@dataclass
class SessionSnapshot:
    """Snapshot of session state for recovery"""
    session_id: str
    turn_number: int
    voice_profile_id: Optional[int]
    last_user_text: Optional[str]
    last_ai_text: Optional[str]
    processing_stage: str  # ready, stt, llm, tts
    created_at: float
    last_activity: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def age_seconds(self) -> float:
        """Age of session in seconds"""
        return time.time() - self.created_at

    @property
    def idle_seconds(self) -> float:
        """Idle time since last activity"""
        return time.time() - self.last_activity

    def is_recoverable(self, max_age_seconds: int = 3600) -> bool:
        """Check if session is still recoverable"""
        return (self.age_seconds < max_age_seconds and
                self.idle_seconds < max_age_seconds / 2)


@dataclass
class ReconnectionConfig:
    """Configuration for reconnection behavior"""
    # Retry limits
    max_retries: int = 10
    max_retry_time_seconds: int = 300  # 5 minutes total

    # Backoff configuration
    initial_delay_ms: int = 1000      # 1 second
    max_delay_ms: int = 30000         # 30 seconds
    backoff_multiplier: float = 1.5
    jitter_ratio: float = 0.1         # 10% random jitter

    # Health check
    heartbeat_interval_seconds: int = 30
    heartbeat_timeout_seconds: int = 5

    # Session recovery
    session_recovery_timeout_seconds: int = 3600  # 1 hour

    # Error handling
    retriable_errors: Set[DisconnectReason] = field(default_factory=lambda: {
        DisconnectReason.NETWORK_ERROR,
        DisconnectReason.TIMEOUT,
        DisconnectReason.SERVER_OVERLOAD
    })


class WebSocketReconnectManager:
    """
    WebSocket reconnection and recovery manager

    Linus-style design:
    - Simple state machine: DISCONNECTED → CONNECTING → CONNECTED
    - No complex threading: everything is async
    - Fail fast: don't retry non-retriable errors
    - Clean interfaces: callbacks for state changes
    """

    def __init__(self, config: Optional[ReconnectionConfig] = None):
        """Initialize reconnection manager"""
        self.config = config or ReconnectionConfig()

        # Connection state
        self.state = ConnectionState.DISCONNECTED
        self.current_websocket: Optional[WebSocket] = None
        self.connection_id = str(uuid.uuid4())

        # Retry tracking
        self.retry_count = 0
        self.first_attempt_time: Optional[float] = None
        self.last_attempt_time: Optional[float] = None
        self.last_disconnect_reason: Optional[DisconnectReason] = None

        # Session management
        self.active_sessions: Dict[str, SessionSnapshot] = {}
        self.session_callbacks: Dict[str, Callable] = {}

        # Background tasks
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None

        # Statistics
        self.connection_stats = {
            "total_connections": 0,
            "successful_reconnections": 0,
            "failed_reconnections": 0,
            "session_recoveries": 0,
            "total_disconnects": 0
        }

        logger.info("websocket_reconnect.init",
                   connection_id=self.connection_id,
                   config=self.config.__dict__)

    def calculate_backoff_delay(self) -> float:
        """
        Calculate next retry delay with exponential backoff and jitter

        Formula: min(max_delay, initial_delay * multiplier^retry_count) + jitter
        Jitter prevents thundering herd when many clients reconnect simultaneously
        """
        if self.retry_count == 0:
            base_delay = self.config.initial_delay_ms
        else:
            base_delay = min(
                self.config.max_delay_ms,
                self.config.initial_delay_ms * (self.config.backoff_multiplier ** (self.retry_count - 1))
            )

        # Add jitter: ±10% random variation
        jitter = base_delay * self.config.jitter_ratio * (random.random() * 2 - 1)
        delay_ms = base_delay + jitter

        return max(delay_ms / 1000.0, 0.1)  # Convert to seconds, minimum 100ms

    def should_retry(self, reason: DisconnectReason) -> bool:
        """
        Determine if reconnection should be attempted

        Linus principle: "Be conservative in what you retry"
        """
        # Check retry limits
        if self.retry_count >= self.config.max_retries:
            logger.warning("websocket_reconnect.max_retries_exceeded",
                          retry_count=self.retry_count,
                          max_retries=self.config.max_retries)
            return False

        # Check time limits
        if (self.first_attempt_time and
            time.time() - self.first_attempt_time > self.config.max_retry_time_seconds):
            logger.warning("websocket_reconnect.max_time_exceeded",
                          elapsed_time=time.time() - self.first_attempt_time,
                          max_time=self.config.max_retry_time_seconds)
            return False

        # Check if error is retriable
        if reason not in self.config.retriable_errors:
            logger.info("websocket_reconnect.non_retriable_error", reason=reason.value)
            return False

        return True

    async def handle_disconnect(self, reason: DisconnectReason,
                              session_snapshot: Optional[SessionSnapshot] = None):
        """
        Handle WebSocket disconnection

        Args:
            reason: Reason for disconnection
            session_snapshot: Current session state to preserve
        """
        self.last_disconnect_reason = reason
        self.connection_stats["total_disconnects"] += 1

        # Preserve session state if provided
        if session_snapshot:
            self.active_sessions[session_snapshot.session_id] = session_snapshot
            logger.info("websocket_reconnect.session_preserved",
                       session_id=session_snapshot.session_id,
                       turn_number=session_snapshot.turn_number,
                       stage=session_snapshot.processing_stage)

        # Stop heartbeat monitoring
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            self._heartbeat_task = None

        # Update state
        if self.state == ConnectionState.CONNECTED:
            self.state = ConnectionState.DISCONNECTED

        logger.info("websocket_reconnect.disconnected",
                   reason=reason.value,
                   retry_count=self.retry_count,
                   preserved_sessions=len(self.active_sessions))

        # Attempt reconnection if appropriate
        if self.should_retry(reason):
            await self.attempt_reconnection()
        else:
            self.state = ConnectionState.FAILED
            logger.error("websocket_reconnect.reconnection_failed",
                        reason=reason.value,
                        retry_count=self.retry_count)

    async def attempt_reconnection(self):
        """
        Attempt to reconnect with exponential backoff
        """
        if self.state in [ConnectionState.CONNECTING, ConnectionState.RECONNECTING]:
            logger.debug("websocket_reconnect.already_attempting")
            return

        self.state = ConnectionState.RECONNECTING
        self.retry_count += 1

        if self.first_attempt_time is None:
            self.first_attempt_time = time.time()

        delay = self.calculate_backoff_delay()

        logger.info("websocket_reconnect.attempting",
                   retry_count=self.retry_count,
                   delay_seconds=delay,
                   preserved_sessions=len(self.active_sessions))

        # Wait with exponential backoff
        await asyncio.sleep(delay)

        try:
            # This would be called by the application to establish new connection
            # The actual reconnection logic is handled by the WebSocket client/server
            self.state = ConnectionState.CONNECTING
            self.last_attempt_time = time.time()

            # Connection establishment would happen here via callback
            # For now, we just track the attempt

        except Exception as e:
            logger.error("websocket_reconnect.attempt_failed",
                        retry_count=self.retry_count,
                        error=str(e))

            # Retry again after delay
            await self.handle_disconnect(DisconnectReason.NETWORK_ERROR)

    async def handle_successful_connection(self, websocket: WebSocket):
        """
        Handle successful WebSocket connection

        Args:
            websocket: New WebSocket connection
        """
        self.current_websocket = websocket
        self.state = ConnectionState.CONNECTED
        self.connection_stats["total_connections"] += 1

        # Track successful reconnection
        if self.retry_count > 0:
            self.connection_stats["successful_reconnections"] += 1
            logger.info("websocket_reconnect.reconnection_successful",
                       retry_count=self.retry_count,
                       total_time=time.time() - (self.first_attempt_time or time.time()))

        # Reset retry tracking
        self.retry_count = 0
        self.first_attempt_time = None
        self.last_attempt_time = None

        # Start heartbeat monitoring
        self._heartbeat_task = asyncio.create_task(self._heartbeat_monitor())

        # Start session cleanup task
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._session_cleanup())

        logger.info("websocket_reconnect.connected",
                   preserved_sessions=len(self.active_sessions))

    def try_recover_session(self, session_id: str) -> Optional[SessionSnapshot]:
        """
        Attempt to recover a session by ID

        Args:
            session_id: Session ID to recover

        Returns:
            SessionSnapshot if recoverable, None otherwise
        """
        snapshot = self.active_sessions.get(session_id)

        if not snapshot:
            return None

        if not snapshot.is_recoverable(self.config.session_recovery_timeout_seconds):
            # Session too old, remove it
            del self.active_sessions[session_id]
            logger.info("websocket_reconnect.session_expired",
                       session_id=session_id,
                       age_seconds=snapshot.age_seconds)
            return None

        # Session recovered successfully
        self.connection_stats["session_recoveries"] += 1
        logger.info("websocket_reconnect.session_recovered",
                   session_id=session_id,
                   turn_number=snapshot.turn_number,
                   idle_seconds=snapshot.idle_seconds)

        return snapshot

    async def _heartbeat_monitor(self):
        """
        Background task to monitor connection health
        """
        try:
            while self.state == ConnectionState.CONNECTED:
                await asyncio.sleep(self.config.heartbeat_interval_seconds)

                if self.current_websocket:
                    try:
                        # Send ping frame
                        await asyncio.wait_for(
                            self.current_websocket.ping(),
                            timeout=self.config.heartbeat_timeout_seconds
                        )

                    except asyncio.TimeoutError:
                        logger.warning("websocket_reconnect.heartbeat_timeout")
                        await self.handle_disconnect(DisconnectReason.TIMEOUT)
                        break
                    except Exception as e:
                        logger.warning("websocket_reconnect.heartbeat_error", error=str(e))
                        await self.handle_disconnect(DisconnectReason.NETWORK_ERROR)
                        break

        except asyncio.CancelledError:
            logger.debug("websocket_reconnect.heartbeat_cancelled")

    async def _session_cleanup(self):
        """
        Background task to clean up expired sessions
        """
        try:
            while True:
                await asyncio.sleep(300)  # Check every 5 minutes

                current_time = time.time()
                expired_sessions = []

                for session_id, snapshot in self.active_sessions.items():
                    if not snapshot.is_recoverable(self.config.session_recovery_timeout_seconds):
                        expired_sessions.append(session_id)

                for session_id in expired_sessions:
                    del self.active_sessions[session_id]

                if expired_sessions:
                    logger.info("websocket_reconnect.sessions_cleaned",
                               expired_count=len(expired_sessions),
                               remaining_count=len(self.active_sessions))

        except asyncio.CancelledError:
            logger.debug("websocket_reconnect.cleanup_cancelled")

    def get_status(self) -> Dict[str, Any]:
        """Get current reconnection manager status"""
        return {
            "connection_id": self.connection_id,
            "state": self.state.value,
            "retry_count": self.retry_count,
            "last_disconnect_reason": self.last_disconnect_reason.value if self.last_disconnect_reason else None,
            "active_sessions": len(self.active_sessions),
            "statistics": self.connection_stats.copy(),
            "config": {
                "max_retries": self.config.max_retries,
                "max_retry_time_seconds": self.config.max_retry_time_seconds,
                "heartbeat_interval_seconds": self.config.heartbeat_interval_seconds
            }
        }

    async def shutdown(self):
        """Gracefully shutdown the reconnection manager"""
        self.state = ConnectionState.SUSPENDED

        # Cancel background tasks
        for task in [self._heartbeat_task, self._cleanup_task]:
            if task and not task.done():
                task.cancel()

        # Clear sessions
        self.active_sessions.clear()

        logger.info("websocket_reconnect.shutdown",
                   connection_id=self.connection_id)


# Global reconnection manager instance
_reconnect_manager: Optional[WebSocketReconnectManager] = None


def get_reconnect_manager() -> WebSocketReconnectManager:
    """Get singleton reconnection manager instance"""
    global _reconnect_manager

    if _reconnect_manager is None:
        _reconnect_manager = WebSocketReconnectManager()

    return _reconnect_manager