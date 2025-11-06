"""
WebSocket Heartbeat and Connection Monitoring System

Task 22: Proactive connection health monitoring with heartbeat detection.
Prevents zombie connections and enables early failure detection.

Design Principles:
1. Lightweight heartbeat protocol - minimal overhead
2. Adaptive timing - adjusts based on network conditions
3. Graceful degradation - continues operation even if heartbeat fails
4. Connection quality metrics - tracks latency and reliability
"""

import asyncio
import time
import uuid
from typing import Dict, Optional, Callable, Any, List
from dataclasses import dataclass, field
from enum import Enum

import structlog
from fastapi import WebSocket

logger = structlog.get_logger()


class HeartbeatState(Enum):
    """Heartbeat monitoring states"""
    STOPPED = "stopped"
    STARTING = "starting"
    ACTIVE = "active"
    DEGRADED = "degraded"    # Some heartbeats failing
    FAILING = "failing"      # Most heartbeats failing
    FAILED = "failed"        # Heartbeat completely failed


@dataclass
class HeartbeatMetrics:
    """Metrics for connection quality assessment"""
    total_pings: int = 0
    successful_pings: int = 0
    failed_pings: int = 0
    timeout_pings: int = 0

    # Latency tracking
    min_latency_ms: float = float('inf')
    max_latency_ms: float = 0.0
    avg_latency_ms: float = 0.0
    recent_latencies: List[float] = field(default_factory=list)

    # Connection quality
    success_rate: float = 1.0
    connection_quality: str = "excellent"  # excellent, good, poor, critical

    # Timing
    first_ping_time: Optional[float] = None
    last_ping_time: Optional[float] = None
    last_success_time: Optional[float] = None

    def update_latency(self, latency_ms: float):
        """Update latency metrics with new measurement"""
        self.recent_latencies.append(latency_ms)

        # Keep only recent 50 measurements
        if len(self.recent_latencies) > 50:
            self.recent_latencies.pop(0)

        # Update min/max
        self.min_latency_ms = min(self.min_latency_ms, latency_ms)
        self.max_latency_ms = max(self.max_latency_ms, latency_ms)

        # Update average
        self.avg_latency_ms = sum(self.recent_latencies) / len(self.recent_latencies)

    def update_success_rate(self):
        """Update success rate and connection quality assessment"""
        if self.total_pings > 0:
            self.success_rate = self.successful_pings / self.total_pings
        else:
            self.success_rate = 1.0

        # Classify connection quality
        if self.success_rate >= 0.95:
            self.connection_quality = "excellent"
        elif self.success_rate >= 0.85:
            self.connection_quality = "good"
        elif self.success_rate >= 0.70:
            self.connection_quality = "poor"
        else:
            self.connection_quality = "critical"


@dataclass
class HeartbeatConfig:
    """Configuration for heartbeat monitoring"""
    # Basic timing
    interval_seconds: float = 30.0      # Time between heartbeats
    timeout_seconds: float = 5.0        # Timeout for ping response

    # Adaptive behavior
    adaptive_timing: bool = True        # Adjust interval based on network conditions
    min_interval_seconds: float = 10.0  # Minimum heartbeat interval
    max_interval_seconds: float = 60.0  # Maximum heartbeat interval

    # Failure thresholds
    consecutive_failures_threshold: int = 3   # Consecutive failures before degraded
    total_failure_rate_threshold: float = 0.3  # 30% failure rate threshold

    # Quality adjustment
    latency_threshold_ms: float = 1000.0  # High latency threshold
    quality_sample_size: int = 20         # Number of pings to assess quality


class WebSocketHeartbeatMonitor:
    """
    WebSocket heartbeat monitoring system

    Manages connection health through periodic ping/pong exchanges.
    Provides connection quality metrics and adaptive timing.
    """

    def __init__(self, websocket: WebSocket, session_id: str,
                 config: Optional[HeartbeatConfig] = None):
        """
        Initialize heartbeat monitor for a WebSocket connection

        Args:
            websocket: WebSocket connection to monitor
            session_id: Session identifier for logging
            config: Heartbeat configuration
        """
        self.websocket = websocket
        self.session_id = session_id
        self.config = config or HeartbeatConfig()

        # State management
        self.state = HeartbeatState.STOPPED
        self.monitor_id = str(uuid.uuid4())[:8]

        # Monitoring task
        self._heartbeat_task: Optional[asyncio.Task] = None

        # Metrics
        self.metrics = HeartbeatMetrics()

        # Failure tracking
        self.consecutive_failures = 0
        self.last_failure_time: Optional[float] = None

        # Callbacks
        self.on_state_change: Optional[Callable[[HeartbeatState], None]] = None
        self.on_quality_change: Optional[Callable[[str], None]] = None
        self.on_connection_lost: Optional[Callable[[], None]] = None

        logger.info("heartbeat.init",
                   session_id=session_id,
                   monitor_id=self.monitor_id,
                   interval=self.config.interval_seconds)

    async def start(self):
        """Start heartbeat monitoring"""
        if self.state != HeartbeatState.STOPPED:
            logger.warning("heartbeat.already_started",
                          session_id=self.session_id,
                          current_state=self.state.value)
            return

        self.state = HeartbeatState.STARTING
        self._notify_state_change()

        # Initialize metrics
        self.metrics.first_ping_time = time.time()

        # Start monitoring task
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

        logger.info("heartbeat.started",
                   session_id=self.session_id,
                   monitor_id=self.monitor_id)

    async def stop(self):
        """Stop heartbeat monitoring"""
        if self.state == HeartbeatState.STOPPED:
            return

        previous_state = self.state
        self.state = HeartbeatState.STOPPED
        self._notify_state_change()

        # Cancel monitoring task
        if self._heartbeat_task and not self._heartbeat_task.done():
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass

        logger.info("heartbeat.stopped",
                   session_id=self.session_id,
                   monitor_id=self.monitor_id,
                   previous_state=previous_state.value,
                   total_pings=self.metrics.total_pings,
                   success_rate=self.metrics.success_rate)

    async def _heartbeat_loop(self):
        """Main heartbeat monitoring loop"""
        try:
            self.state = HeartbeatState.ACTIVE
            self._notify_state_change()

            while self.state not in [HeartbeatState.STOPPED, HeartbeatState.FAILED]:
                # Calculate current interval (adaptive or fixed)
                current_interval = self._calculate_interval()

                # Wait for next heartbeat
                await asyncio.sleep(current_interval)

                # Send ping and measure response
                success, latency_ms = await self._send_ping()

                # Update metrics
                self._update_metrics(success, latency_ms)

                # Update state based on recent performance
                self._update_state()

                logger.debug("heartbeat.ping_completed",
                            session_id=self.session_id,
                            success=success,
                            latency_ms=latency_ms,
                            success_rate=self.metrics.success_rate,
                            state=self.state.value)

        except asyncio.CancelledError:
            logger.debug("heartbeat.loop_cancelled",
                        session_id=self.session_id)
        except Exception as e:
            logger.error("heartbeat.loop_error",
                        session_id=self.session_id,
                        error=str(e))
            self.state = HeartbeatState.FAILED
            self._notify_state_change()

    async def _send_ping(self) -> tuple[bool, Optional[float]]:
        """
        Send ping and measure response time

        Returns:
            (success, latency_ms) tuple
        """
        start_time = time.time()

        try:
            # Send WebSocket ping
            pong_waiter = await self.websocket.ping()

            # Wait for pong response with timeout
            await asyncio.wait_for(pong_waiter, timeout=self.config.timeout_seconds)

            # Calculate latency
            latency_ms = (time.time() - start_time) * 1000

            return True, latency_ms

        except asyncio.TimeoutError:
            logger.warning("heartbeat.ping_timeout",
                          session_id=self.session_id,
                          timeout=self.config.timeout_seconds)
            return False, None

        except Exception as e:
            logger.warning("heartbeat.ping_error",
                          session_id=self.session_id,
                          error=str(e))
            return False, None

    def _update_metrics(self, success: bool, latency_ms: Optional[float]):
        """Update heartbeat metrics with latest result"""
        self.metrics.total_pings += 1
        self.metrics.last_ping_time = time.time()

        if success:
            self.metrics.successful_pings += 1
            self.metrics.last_success_time = time.time()
            self.consecutive_failures = 0

            if latency_ms is not None:
                self.metrics.update_latency(latency_ms)

        else:
            self.metrics.failed_pings += 1
            self.consecutive_failures += 1
            self.last_failure_time = time.time()

            if latency_ms is None:  # Timeout
                self.metrics.timeout_pings += 1

        # Update derived metrics
        self.metrics.update_success_rate()

    def _update_state(self):
        """Update heartbeat state based on recent performance"""
        previous_state = self.state

        # Check for complete failure
        if self.consecutive_failures >= self.config.consecutive_failures_threshold * 2:
            self.state = HeartbeatState.FAILED
            if self.on_connection_lost:
                asyncio.create_task(self._safe_callback(self.on_connection_lost))

        # Check for degraded performance
        elif (self.consecutive_failures >= self.config.consecutive_failures_threshold or
              self.metrics.success_rate < self.config.total_failure_rate_threshold):
            self.state = HeartbeatState.FAILING

        elif self.consecutive_failures > 0 or self.metrics.success_rate < 0.9:
            self.state = HeartbeatState.DEGRADED

        else:
            self.state = HeartbeatState.ACTIVE

        # Notify state change
        if self.state != previous_state:
            self._notify_state_change()

        # Notify quality change
        previous_quality = getattr(self, '_last_notified_quality', None)
        current_quality = self.metrics.connection_quality
        if current_quality != previous_quality:
            self._last_notified_quality = current_quality
            if self.on_quality_change:
                asyncio.create_task(self._safe_callback(self.on_quality_change, current_quality))

    def _calculate_interval(self) -> float:
        """Calculate adaptive heartbeat interval based on connection quality"""
        if not self.config.adaptive_timing:
            return self.config.interval_seconds

        base_interval = self.config.interval_seconds

        # Adjust based on connection quality
        if self.metrics.connection_quality == "excellent":
            # Excellent connection - can use longer intervals
            multiplier = 1.2
        elif self.metrics.connection_quality == "good":
            # Good connection - normal intervals
            multiplier = 1.0
        elif self.metrics.connection_quality == "poor":
            # Poor connection - shorter intervals for better monitoring
            multiplier = 0.8
        else:  # critical
            # Critical connection - very short intervals
            multiplier = 0.6

        # Apply failure penalty
        if self.consecutive_failures > 0:
            failure_penalty = 0.9 ** self.consecutive_failures  # Exponential decrease
            multiplier *= failure_penalty

        adapted_interval = base_interval * multiplier

        # Enforce limits
        return max(
            self.config.min_interval_seconds,
            min(self.config.max_interval_seconds, adapted_interval)
        )

    def _notify_state_change(self):
        """Notify state change callback"""
        if self.on_state_change:
            asyncio.create_task(self._safe_callback(self.on_state_change, self.state))

    async def _safe_callback(self, callback: Callable, *args):
        """Safely execute callback without breaking monitoring"""
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(*args)
            else:
                callback(*args)
        except Exception as e:
            logger.error("heartbeat.callback_error",
                        session_id=self.session_id,
                        error=str(e))

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive heartbeat status"""
        return {
            "monitor_id": self.monitor_id,
            "session_id": self.session_id,
            "state": self.state.value,
            "consecutive_failures": self.consecutive_failures,
            "metrics": {
                "total_pings": self.metrics.total_pings,
                "successful_pings": self.metrics.successful_pings,
                "failed_pings": self.metrics.failed_pings,
                "timeout_pings": self.metrics.timeout_pings,
                "success_rate": round(self.metrics.success_rate, 3),
                "connection_quality": self.metrics.connection_quality,
                "avg_latency_ms": round(self.metrics.avg_latency_ms, 1),
                "min_latency_ms": round(self.metrics.min_latency_ms, 1) if self.metrics.min_latency_ms != float('inf') else None,
                "max_latency_ms": round(self.metrics.max_latency_ms, 1) if self.metrics.max_latency_ms > 0 else None,
            },
            "timing": {
                "current_interval": self._calculate_interval(),
                "last_ping_time": self.metrics.last_ping_time,
                "last_success_time": self.metrics.last_success_time,
                "uptime_seconds": time.time() - self.metrics.first_ping_time if self.metrics.first_ping_time else 0
            },
            "config": {
                "interval_seconds": self.config.interval_seconds,
                "timeout_seconds": self.config.timeout_seconds,
                "adaptive_timing": self.config.adaptive_timing
            }
        }

    def is_healthy(self) -> bool:
        """Check if connection is considered healthy"""
        return (self.state in [HeartbeatState.ACTIVE, HeartbeatState.DEGRADED] and
                self.metrics.success_rate > 0.5 and
                self.consecutive_failures < self.config.consecutive_failures_threshold)


# Global heartbeat monitors registry
_heartbeat_monitors: Dict[str, WebSocketHeartbeatMonitor] = {}


def register_heartbeat_monitor(session_id: str, monitor: WebSocketHeartbeatMonitor):
    """Register a heartbeat monitor for global tracking"""
    _heartbeat_monitors[session_id] = monitor


def unregister_heartbeat_monitor(session_id: str):
    """Unregister a heartbeat monitor"""
    _heartbeat_monitors.pop(session_id, None)


def get_heartbeat_monitor(session_id: str) -> Optional[WebSocketHeartbeatMonitor]:
    """Get heartbeat monitor for a session"""
    return _heartbeat_monitors.get(session_id)


def get_all_heartbeat_status() -> Dict[str, Dict[str, Any]]:
    """Get status of all active heartbeat monitors"""
    return {
        session_id: monitor.get_status()
        for session_id, monitor in _heartbeat_monitors.items()
    }