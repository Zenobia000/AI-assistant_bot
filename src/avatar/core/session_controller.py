"""
Session Controller - Enhanced concurrent session management

Task 21: Unified session control with intelligent queuing and VRAM monitoring.
Linus principle: "Do one thing and do it well" - orchestrate specialized components.

Architecture:
- SessionController: High-level orchestration
- SessionQueue: Intelligent queuing with priority
- VRAMMonitor: Resource availability monitoring
- Simple, clean interfaces between components
"""

import asyncio
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

import structlog

from avatar.core.config import config
from avatar.core.vram_monitor import get_vram_monitor
from avatar.core.session_queue import get_session_queue, QueuedSession, QueueState


logger = structlog.get_logger()


@dataclass
class SessionResult:
    """Result of a session request"""
    success: bool
    session_id: str
    message: str
    queue_position: Optional[int] = None
    estimated_wait_time: Optional[float] = None
    processing_started: bool = False
    error_code: Optional[str] = None


class SessionController:
    """
    High-level session controller - orchestrates queue and VRAM monitoring

    Linus-style design:
    - Single clear responsibility: session lifecycle orchestration
    - Delegate specialized tasks to specialized components
    - Simple, predictable interface
    - No duplicate functionality
    """

    def __init__(self):
        """Initialize session controller"""
        self.vram_monitor = get_vram_monitor()
        self.session_queue = get_session_queue()
        self.active_sessions: Dict[str, QueuedSession] = {}

        # Ensure VRAM monitoring is active
        if hasattr(self.vram_monitor, 'start_monitoring'):
            self.vram_monitor.start_monitoring()

        logger.info("session_controller.init",
                   max_concurrent=self.session_queue.max_concurrent,
                   queue_max_size=self.session_queue.max_queue_size)

    async def start(self):
        """Start the session controller and its components"""
        await self.session_queue.start()
        logger.info("session_controller.started")

    async def stop(self):
        """Stop the session controller"""
        await self.session_queue.stop()
        if hasattr(self.vram_monitor, 'stop_monitoring'):
            self.vram_monitor.stop_monitoring()
        logger.info("session_controller.stopped")

    async def request_session(self,
                            session_id: str,
                            service_type: str,
                            websocket_connection: Any = None,
                            timeout: Optional[float] = None,
                            priority_boost: bool = False) -> SessionResult:
        """
        Request a new session with intelligent queuing

        Args:
            session_id: Unique session identifier
            service_type: Service type (stt, llm, tts_fast, tts_hq)
            websocket_connection: WebSocket for real-time updates
            timeout: Custom timeout (uses queue default if None)
            priority_boost: Whether to boost priority (admin feature)

        Returns:
            SessionResult with outcome and details
        """
        try:
            # Check if we can handle this service type at all
            prediction = self.vram_monitor.predict_can_handle_service(service_type)

            if not prediction["can_handle"] and not prediction.get("reasoning"):
                return SessionResult(
                    success=False,
                    session_id=session_id,
                    message="Service type not supported or system overloaded",
                    error_code="SERVICE_UNAVAILABLE"
                )

            # Try immediate processing if resources available and queue empty
            if (len(self.session_queue.queue) == 0 and
                len(self.session_queue.processing) < self.session_queue.max_concurrent and
                prediction["can_handle"]):

                # Fast path: immediate processing
                queued_session = await self.session_queue.enqueue_session(
                    session_id=session_id,
                    service_type=service_type,
                    websocket_connection=websocket_connection,
                    timeout=timeout or 5.0  # Short timeout for immediate processing
                )

                return SessionResult(
                    success=True,
                    session_id=session_id,
                    message="Session started immediately",
                    processing_started=True
                )

            else:
                # Queue path: intelligent queuing
                queued_session = await self.session_queue.enqueue_session(
                    session_id=session_id,
                    service_type=service_type,
                    websocket_connection=websocket_connection,
                    timeout=timeout
                )

                # Find position in queue
                queue_position = None
                for i, session in enumerate(self.session_queue.queue):
                    if session.session_id == session_id:
                        queue_position = i
                        break

                estimated_wait = self.session_queue._estimate_wait_time(queue_position or 0)

                return SessionResult(
                    success=True,
                    session_id=session_id,
                    message=f"Session queued at position {queue_position}",
                    queue_position=queue_position,
                    estimated_wait_time=estimated_wait
                )

        except asyncio.QueueFull:
            return SessionResult(
                success=False,
                session_id=session_id,
                message=f"Server at capacity (queue size: {self.session_queue.max_queue_size})",
                error_code="QUEUE_FULL"
            )

        except Exception as e:
            logger.error("session_controller.request_error",
                        session_id=session_id,
                        service_type=service_type,
                        error=str(e))

            return SessionResult(
                success=False,
                session_id=session_id,
                message=f"Internal error: {str(e)}",
                error_code="INTERNAL_ERROR"
            )

    async def cancel_session(self, session_id: str) -> bool:
        """
        Cancel a queued or processing session

        Args:
            session_id: Session to cancel

        Returns:
            True if cancelled, False if not found or already completed
        """
        return await self.session_queue.cancel_session(session_id)

    def get_session_status(self, session_id: str) -> Optional[Dict]:
        """
        Get status of a specific session

        Args:
            session_id: Session to check

        Returns:
            Session status dict or None if not found
        """
        # Check queue
        for i, session in enumerate(self.session_queue.queue):
            if session.session_id == session_id:
                return {
                    "session_id": session_id,
                    "state": session.state.value,
                    "service_type": session.service_type,
                    "queue_position": i,
                    "wait_time": session.wait_time_seconds,
                    "estimated_remaining": self.session_queue._estimate_wait_time(i)
                }

        # Check processing
        if session_id in self.session_queue.processing:
            session = self.session_queue.processing[session_id]
            return {
                "session_id": session_id,
                "state": session.state.value,
                "service_type": session.service_type,
                "processing_time": session.processing_time_seconds,
                "gpu_allocation": session.gpu_allocation
            }

        # Check completed (recent)
        if session_id in self.session_queue.completed:
            session = self.session_queue.completed[session_id]
            return {
                "session_id": session_id,
                "state": session.state.value,
                "service_type": session.service_type,
                "total_time": session.wait_time_seconds + (session.processing_time_seconds or 0),
                "completed_at": session.completed_at
            }

        return None

    def get_system_status(self) -> Dict:
        """
        Get comprehensive system status

        Returns:
            System status including queue, VRAM, and statistics
        """
        queue_status = self.session_queue.get_queue_status()
        vram_stats = self.vram_monitor.get_monitoring_stats()

        return {
            "timestamp": time.time(),
            "queue": queue_status,
            "vram": vram_stats,
            "capacity": {
                "max_concurrent": self.session_queue.max_concurrent,
                "current_processing": len(self.session_queue.processing),
                "queue_size": len(self.session_queue.queue),
                "queue_capacity": self.session_queue.max_queue_size,
                "utilization_percent": (
                    len(self.session_queue.processing) / self.session_queue.max_concurrent * 100
                    if self.session_queue.max_concurrent > 0 else 0
                )
            },
            "performance": {
                "average_wait_time": queue_status["statistics"]["avg_wait_time"],
                "average_processing_time": queue_status["statistics"]["avg_processing_time"],
                "total_processed": queue_status["statistics"]["total_processed"],
                "total_rejected": queue_status["statistics"]["total_rejected"]
            }
        }

    def get_service_availability(self) -> Dict[str, Dict]:
        """
        Check availability for each service type

        Returns:
            Dict mapping service types to availability info
        """
        services = ["stt", "llm", "tts_fast", "tts_hq"]
        availability = {}

        for service_type in services:
            prediction = self.vram_monitor.predict_can_handle_service(service_type)
            queue_estimate = self.session_queue._estimate_wait_time(len(self.session_queue.queue))

            availability[service_type] = {
                "available": prediction["can_handle"],
                "vram_required_gb": prediction["vram_required_gb"],
                "recommended_gpu": prediction["recommended_gpu"],
                "reasoning": prediction["reasoning"],
                "estimated_wait_time": queue_estimate if not prediction["can_handle"] else 0.0,
                "priority": prediction["priority"]
            }

        return availability

    async def force_cleanup(self):
        """
        Force cleanup of resources (emergency function)
        """
        logger.warning("session_controller.force_cleanup_requested")

        # Force VRAM cleanup
        self.vram_monitor.force_cleanup()

        # Could add queue cleanup if needed
        # (Current design: let queue naturally process)

        logger.info("session_controller.force_cleanup_completed")

    def get_health_check(self) -> Dict:
        """
        Health check for monitoring systems

        Returns:
            Health status with key metrics
        """
        vram_status = self.vram_monitor.get_all_gpu_status()
        queue_status = self.session_queue.get_queue_status()

        # Simple health scoring
        health_score = 100
        issues = []

        # Check VRAM health
        for gpu_status in vram_status:
            if gpu_status.usage_percent > 90:
                health_score -= 20
                issues.append(f"GPU {gpu_status.device_id} high VRAM usage: {gpu_status.usage_percent}%")

        # Check queue health
        if len(self.session_queue.queue) > self.session_queue.max_queue_size * 0.8:
            health_score -= 15
            issues.append(f"Queue nearly full: {len(self.session_queue.queue)}/{self.session_queue.max_queue_size}")

        # Check processing health
        if len(self.session_queue.processing) == self.session_queue.max_concurrent:
            health_score -= 10
            issues.append("All processing slots occupied")

        return {
            "healthy": health_score >= 70,
            "health_score": max(0, health_score),
            "issues": issues,
            "metrics": {
                "queue_size": len(self.session_queue.queue),
                "processing_count": len(self.session_queue.processing),
                "vram_usage_max": max([s.usage_percent for s in vram_status] or [0]),
                "recent_rejections": queue_status["statistics"]["total_rejected"]
            }
        }


# Global singleton instance
_session_controller: Optional[SessionController] = None


def get_session_controller() -> SessionController:
    """Get singleton session controller instance"""
    global _session_controller

    if _session_controller is None:
        _session_controller = SessionController()

    return _session_controller


async def initialize_session_controller():
    """Initialize and start the global session controller"""
    controller = get_session_controller()
    await controller.start()
    return controller