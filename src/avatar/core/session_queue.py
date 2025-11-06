"""
Session Queue Manager for AVATAR - Concurrent Control with Intelligent Queuing

Task 21: Implement concurrent session control and queuing mechanism
Design Philosophy: Linus-style "Good Taste" - eliminate special cases

Key Principles:
1. Single queue for all services, priority-based processing
2. No complex state machines - simple queue + semaphore
3. VRAM-aware queuing with predictive acceptance
4. Fail fast when queue is full (no infinite waiting)
"""

import asyncio
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from uuid import uuid4

import structlog

from avatar.core.config import config
from avatar.core.vram_monitor import get_vram_monitor, ServicePriority

logger = structlog.get_logger()


class QueueState(Enum):
    """Session queue states"""
    WAITING = "waiting"         # In queue, waiting for resources
    PROCESSING = "processing"   # Currently being processed
    COMPLETED = "completed"     # Successfully completed
    REJECTED = "rejected"       # Rejected (queue full, timeout, etc.)
    CANCELLED = "cancelled"     # Cancelled by user


@dataclass
class QueuedSession:
    """A session in the queue"""
    session_id: str
    service_type: str
    priority: ServicePriority
    requested_at: float
    timeout_at: float
    websocket_connection: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    state: QueueState = QueueState.WAITING
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    gpu_allocation: Optional[int] = None
    error_reason: Optional[str] = None

    @property
    def wait_time_seconds(self) -> float:
        """Calculate current wait time"""
        if self.started_at:
            return self.started_at - self.requested_at
        return time.time() - self.requested_at

    @property
    def is_expired(self) -> bool:
        """Check if session has timed out"""
        return time.time() > self.timeout_at

    @property
    def processing_time_seconds(self) -> Optional[float]:
        """Calculate processing time if applicable"""
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        elif self.started_at:
            return time.time() - self.started_at
        return None


class SessionQueue:
    """
    Intelligent session queue with VRAM-aware processing

    Design: Single priority queue + semaphore-based concurrency control

    Features:
    - Priority-based queuing (CRITICAL > HIGH > MEDIUM > LOW)
    - VRAM-aware admission control
    - Automatic timeout handling
    - Real-time queue status
    - WebSocket notification support
    """

    def __init__(self,
                 max_concurrent: int = None,
                 max_queue_size: int = None,
                 default_timeout: float = 30.0):
        """
        Initialize session queue

        Args:
            max_concurrent: Maximum concurrent processing sessions
            max_queue_size: Maximum queue size (prevents memory bloat)
            default_timeout: Default timeout for queued sessions
        """
        self.max_concurrent = max_concurrent or config.MAX_CONCURRENT_SESSIONS
        self.max_queue_size = max_queue_size or (self.max_concurrent * 3)  # 3x buffer
        self.default_timeout = default_timeout

        # Core data structures
        self.queue: List[QueuedSession] = []
        self.processing: Dict[str, QueuedSession] = {}
        self.completed: Dict[str, QueuedSession] = {}  # Recent completions for stats

        # Concurrency control
        self._semaphore = asyncio.Semaphore(self.max_concurrent)
        self._queue_lock = asyncio.Lock()

        # Monitoring
        self.vram_monitor = get_vram_monitor()
        self.stats = {
            "total_queued": 0,
            "total_processed": 0,
            "total_rejected": 0,
            "total_timeouts": 0,
            "avg_wait_time": 0.0,
            "avg_processing_time": 0.0
        }

        # Background tasks
        self._queue_processor_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None

        logger.info("session_queue.init",
                   max_concurrent=self.max_concurrent,
                   max_queue_size=self.max_queue_size,
                   default_timeout=self.default_timeout)

    async def start(self):
        """Start background queue processing"""
        if self._queue_processor_task is None or self._queue_processor_task.done():
            self._queue_processor_task = asyncio.create_task(self._queue_processor())

        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._cleanup_completed())

        logger.info("session_queue.started")

    async def stop(self):
        """Stop background processing"""
        tasks = [self._queue_processor_task, self._cleanup_task]
        for task in tasks:
            if task and not task.done():
                task.cancel()

        logger.info("session_queue.stopped")

    async def enqueue_session(self,
                            session_id: str,
                            service_type: str,
                            websocket_connection: Any = None,
                            timeout: Optional[float] = None,
                            metadata: Optional[Dict] = None) -> QueuedSession:
        """
        Add a session to the queue

        Args:
            session_id: Unique session identifier
            service_type: Service type (stt, llm, tts_fast, tts_hq)
            websocket_connection: WebSocket connection for notifications
            timeout: Custom timeout (uses default if None)
            metadata: Additional session metadata

        Returns:
            QueuedSession object

        Raises:
            asyncio.QueueFull: If queue is at capacity
        """
        async with self._queue_lock:
            # Check queue capacity
            if len(self.queue) >= self.max_queue_size:
                self.stats["total_rejected"] += 1
                raise asyncio.QueueFull(f"Queue at capacity ({self.max_queue_size})")

            # Get service priority
            priority = self.vram_monitor.service_priority.get(
                service_type, ServicePriority.LOW
            )

            # Create queued session
            session_timeout = timeout or self.default_timeout
            queued_session = QueuedSession(
                session_id=session_id,
                service_type=service_type,
                priority=priority,
                requested_at=time.time(),
                timeout_at=time.time() + session_timeout,
                websocket_connection=websocket_connection,
                metadata=metadata or {}
            )

            # Insert in priority order (higher priority = lower number)
            insert_position = 0
            for i, existing in enumerate(self.queue):
                if existing.priority.value > priority.value:
                    insert_position = i
                    break
                insert_position = i + 1

            self.queue.insert(insert_position, queued_session)
            self.stats["total_queued"] += 1

            logger.info("session_queue.enqueued",
                       session_id=session_id,
                       service_type=service_type,
                       priority=priority.name,
                       queue_position=insert_position,
                       queue_size=len(self.queue))

            # Notify via WebSocket if available
            if websocket_connection:
                await self._notify_websocket(websocket_connection, "queued", {
                    "session_id": session_id,
                    "queue_position": insert_position,
                    "estimated_wait_time": self._estimate_wait_time(insert_position)
                })

            return queued_session

    async def _queue_processor(self):
        """Background queue processor - the heart of the system"""
        try:
            while True:
                await self._process_next_session()
                await asyncio.sleep(0.1)  # Small delay to prevent busy waiting

        except asyncio.CancelledError:
            logger.info("session_queue.processor_cancelled")
        except Exception as e:
            logger.error("session_queue.processor_error", error=str(e))

    async def _process_next_session(self):
        """Process the next session in queue if resources available"""
        # Get next session from queue
        session = await self._get_next_processable_session()
        if not session:
            return

        # Try to acquire processing slot
        try:
            await asyncio.wait_for(
                self._semaphore.acquire(),
                timeout=0.1  # Non-blocking check
            )
        except asyncio.TimeoutError:
            # No slots available, wait for next cycle
            return

        # Move to processing
        async with self._queue_lock:
            if session in self.queue:
                self.queue.remove(session)
                session.state = QueueState.PROCESSING
                session.started_at = time.time()
                self.processing[session.session_id] = session

        logger.info("session_queue.processing_started",
                   session_id=session.session_id,
                   service_type=session.service_type,
                   wait_time=session.wait_time_seconds)

        # Process session in background
        asyncio.create_task(self._execute_session(session))

    async def _get_next_processable_session(self) -> Optional[QueuedSession]:
        """Get next session that can be processed based on VRAM availability"""
        async with self._queue_lock:
            # Clean up expired sessions first
            expired_sessions = [s for s in self.queue if s.is_expired]
            for session in expired_sessions:
                self.queue.remove(session)
                session.state = QueueState.REJECTED
                session.error_reason = "timeout"
                self.stats["total_timeouts"] += 1

                logger.warning("session_queue.timeout",
                             session_id=session.session_id,
                             wait_time=session.wait_time_seconds)

                # Notify timeout
                if session.websocket_connection:
                    await self._notify_websocket(session.websocket_connection, "timeout", {
                        "session_id": session.session_id,
                        "reason": "Queue timeout"
                    })

            # Find next processable session
            for session in self.queue:
                # Check VRAM availability
                prediction = self.vram_monitor.predict_can_handle_service(session.service_type)
                if prediction["can_handle"]:
                    session.gpu_allocation = prediction.get("recommended_gpu")
                    return session

            return None

    async def _execute_session(self, session: QueuedSession):
        """Execute a session (placeholder for actual processing)"""
        try:
            # Notify processing started
            if session.websocket_connection:
                await self._notify_websocket(session.websocket_connection, "processing", {
                    "session_id": session.session_id,
                    "gpu_allocation": session.gpu_allocation
                })

            # Simulate processing (this would call actual AI services)
            processing_time = {
                "stt": 0.6,
                "llm": 2.0,
                "tts_fast": 1.5,
                "tts_hq": 4.0
            }.get(session.service_type, 1.0)

            await asyncio.sleep(processing_time)

            # Mark as completed
            session.state = QueueState.COMPLETED
            session.completed_at = time.time()

            # Move to completed
            async with self._queue_lock:
                if session.session_id in self.processing:
                    del self.processing[session.session_id]
                    self.completed[session.session_id] = session
                    self.stats["total_processed"] += 1

            # Update average times
            self._update_average_times(session)

            logger.info("session_queue.completed",
                       session_id=session.session_id,
                       processing_time=session.processing_time_seconds,
                       total_time=session.wait_time_seconds + session.processing_time_seconds)

            # Notify completion
            if session.websocket_connection:
                await self._notify_websocket(session.websocket_connection, "completed", {
                    "session_id": session.session_id,
                    "processing_time": session.processing_time_seconds
                })

        except Exception as e:
            session.state = QueueState.REJECTED
            session.error_reason = str(e)

            logger.error("session_queue.execution_error",
                        session_id=session.session_id,
                        error=str(e))

            # Notify error
            if session.websocket_connection:
                await self._notify_websocket(session.websocket_connection, "error", {
                    "session_id": session.session_id,
                    "error": str(e)
                })

        finally:
            # Always release semaphore
            self._semaphore.release()

    async def _cleanup_completed(self):
        """Cleanup old completed sessions"""
        try:
            while True:
                await asyncio.sleep(60)  # Cleanup every minute

                current_time = time.time()
                cutoff_time = current_time - 300  # Keep for 5 minutes

                async with self._queue_lock:
                    to_remove = []
                    for session_id, session in self.completed.items():
                        if session.completed_at and session.completed_at < cutoff_time:
                            to_remove.append(session_id)

                    for session_id in to_remove:
                        del self.completed[session_id]

                    if to_remove:
                        logger.info("session_queue.cleanup",
                                   removed_count=len(to_remove),
                                   remaining_completed=len(self.completed))

        except asyncio.CancelledError:
            logger.info("session_queue.cleanup_cancelled")
        except Exception as e:
            logger.error("session_queue.cleanup_error", error=str(e))

    def _estimate_wait_time(self, queue_position: int) -> float:
        """Estimate wait time based on queue position and historical data"""
        if self.stats["avg_processing_time"] > 0:
            # Estimate based on average processing time and current load
            concurrent_factor = max(1, self.max_concurrent)
            return (queue_position / concurrent_factor) * self.stats["avg_processing_time"]
        else:
            # Default estimate
            return queue_position * 2.0  # 2 seconds per position

    def _update_average_times(self, session: QueuedSession):
        """Update running averages for wait and processing times"""
        # Simple moving average (could be improved with exponential decay)
        weight = 0.1  # 10% weight for new sample

        if session.processing_time_seconds:
            if self.stats["avg_processing_time"] == 0:
                self.stats["avg_processing_time"] = session.processing_time_seconds
            else:
                self.stats["avg_processing_time"] = (
                    (1 - weight) * self.stats["avg_processing_time"] +
                    weight * session.processing_time_seconds
                )

        if self.stats["avg_wait_time"] == 0:
            self.stats["avg_wait_time"] = session.wait_time_seconds
        else:
            self.stats["avg_wait_time"] = (
                (1 - weight) * self.stats["avg_wait_time"] +
                weight * session.wait_time_seconds
            )

    async def _notify_websocket(self, websocket, event_type: str, data: Dict):
        """Send notification via WebSocket"""
        try:
            message = {
                "type": "queue_status",
                "event": event_type,
                "data": data,
                "timestamp": time.time()
            }
            await websocket.send_json(message)
        except Exception as e:
            logger.warning("session_queue.websocket_notify_failed", error=str(e))

    def get_queue_status(self) -> Dict:
        """Get current queue status"""
        return {
            "queue_size": len(self.queue),
            "processing_count": len(self.processing),
            "max_concurrent": self.max_concurrent,
            "max_queue_size": self.max_queue_size,
            "statistics": self.stats.copy(),
            "queue_sessions": [
                {
                    "session_id": s.session_id,
                    "service_type": s.service_type,
                    "priority": s.priority.name,
                    "wait_time": s.wait_time_seconds,
                    "estimated_remaining": self._estimate_wait_time(i)
                }
                for i, s in enumerate(self.queue)
            ],
            "processing_sessions": [
                {
                    "session_id": s.session_id,
                    "service_type": s.service_type,
                    "processing_time": s.processing_time_seconds,
                    "gpu_allocation": s.gpu_allocation
                }
                for s in self.processing.values()
            ]
        }

    async def cancel_session(self, session_id: str) -> bool:
        """Cancel a queued or processing session"""
        async with self._queue_lock:
            # Check queue
            for session in self.queue:
                if session.session_id == session_id:
                    self.queue.remove(session)
                    session.state = QueueState.CANCELLED
                    logger.info("session_queue.cancelled_from_queue", session_id=session_id)
                    return True

            # Check processing (can't cancel, but mark for early termination)
            if session_id in self.processing:
                # TODO: Implement graceful termination for processing sessions
                logger.warning("session_queue.cancel_processing_not_implemented",
                             session_id=session_id)
                return False

        return False


# Global singleton instance
_session_queue: Optional[SessionQueue] = None


def get_session_queue() -> SessionQueue:
    """Get singleton session queue instance"""
    global _session_queue

    if _session_queue is None:
        _session_queue = SessionQueue()

    return _session_queue


async def initialize_session_queue():
    """Initialize and start the global session queue"""
    queue = get_session_queue()
    await queue.start()
    return queue