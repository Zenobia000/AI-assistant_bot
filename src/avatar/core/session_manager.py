"""
Enhanced Session Manager with Queue-based Concurrent Control

Task 21: Integrated with SessionQueue for intelligent concurrent session management.
Linus-style refactor: Remove duplicate VRAM logic, delegate to specialized components.

Responsibilities:
- Simple session acquisition/release
- Integration with SessionQueue for queuing
- Basic session tracking and status
VRAM monitoring delegated to VRAMMonitor, queuing delegated to SessionQueue.
"""

import asyncio
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

import structlog
import torch

from avatar.core.config import config

logger = structlog.get_logger()


class VRAMThreshold(Enum):
    """VRAM usage threshold levels"""
    SAFE = 70      # < 70% - Normal operation
    WARNING = 85   # 70-85% - Start throttling
    CRITICAL = 95  # 85-95% - Emergency throttling
    DANGER = 98    # > 95% - Reject all new sessions


@dataclass
class VRAMStatus:
    """VRAM status for a specific GPU"""
    device_id: int
    device_name: str
    total_gb: float
    allocated_gb: float
    reserved_gb: float
    free_gb: float
    usage_percent: float
    threshold: VRAMThreshold
    can_accept_new: bool


@dataclass
class SessionLoad:
    """Current session load information"""
    active_sessions: int
    max_sessions: int
    load_percent: float
    pending_requests: int
    recent_rejections: int


class SessionManager:
    """
    Global session manager (Singleton pattern)

    Responsibilities:
    - Track active session count
    - Monitor VRAM usage
    - Enforce concurrency limits
    - Prevent OOM errors

    Design Philosophy (Linus-style):
    - Simple semaphore-based limiting
    - No complex queue management
    - Fail fast when over capacity
    """

    def __init__(self, max_sessions: int = None, vram_limit_gb: int = None):
        """
        Initialize advanced session manager with multi-GPU support

        Args:
            max_sessions: Maximum concurrent sessions (default from config)
            vram_limit_gb: VRAM limit per GPU in GB (default from config)
        """
        self.max_sessions = max_sessions or config.MAX_CONCURRENT_SESSIONS
        self.vram_limit_gb = vram_limit_gb or config.VRAM_LIMIT_GB
        self.active_sessions: Dict[str, dict] = {}  # session_id -> session_info
        self._semaphore = asyncio.Semaphore(self.max_sessions)
        self._lock = asyncio.Lock()

        # Enhanced monitoring
        self.rejection_count = 0
        self.last_vram_check = 0
        self.vram_history: List[Tuple[float, float]] = []  # (timestamp, usage_percent)
        self.gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0

        logger.info("session_manager.init",
                   max_sessions=self.max_sessions,
                   vram_limit_gb=self.vram_limit_gb,
                   gpu_count=self.gpu_count)

    async def acquire_session(self, session_id: str, timeout: float = 1.0,
                            service_type: str = "general") -> bool:
        """
        Try to acquire a session slot with enhanced VRAM monitoring

        Args:
            session_id: Unique session identifier
            timeout: Timeout in seconds (fail fast)
            service_type: Type of service (stt, llm, tts_fast, tts_hq)

        Returns:
            True if session acquired, False if server is full

        Design note:
        Enhanced with multi-GPU VRAM monitoring and intelligent throttling.
        """
        # 1. Multi-GPU VRAM availability check
        vram_status = self._get_multi_gpu_vram_status()
        can_accept = self._evaluate_session_acceptance(vram_status, service_type)

        if not can_accept:
            self.rejection_count += 1
            logger.warning("session_manager.rejected",
                          session_id=session_id,
                          service_type=service_type,
                          reason="vram_insufficient",
                          rejection_count=self.rejection_count,
                          vram_status=[{
                              "gpu": i,
                              "usage": status.usage_percent,
                              "threshold": status.threshold.name
                          } for i, status in enumerate(vram_status)])
            return False

        # 2. Try to acquire semaphore with timeout
        try:
            await asyncio.wait_for(
                self._semaphore.acquire(),
                timeout=timeout
            )

            # 3. Register session with metadata
            async with self._lock:
                self.active_sessions[session_id] = {
                    "service_type": service_type,
                    "started_at": time.time(),
                    "gpu_allocation": self._suggest_gpu_allocation(service_type),
                    "vram_snapshot": vram_status
                }

            logger.info("session_manager.acquired",
                       session_id=session_id,
                       service_type=service_type,
                       active_count=len(self.active_sessions),
                       max_sessions=self.max_sessions,
                       gpu_allocation=self.active_sessions[session_id]["gpu_allocation"])

            return True

        except asyncio.TimeoutError:
            self.rejection_count += 1
            logger.warning("session_manager.timeout",
                          session_id=session_id,
                          active_count=len(self.active_sessions),
                          max_sessions=self.max_sessions,
                          rejection_count=self.rejection_count)
            return False

    def release_session(self, session_id: str):
        """
        Release a session slot

        Args:
            session_id: Session identifier to release
        """
        # Use sync context manager since this is called from finally block
        # which may not be in async context
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
            self._semaphore.release()

            logger.info("session_manager.released",
                       session_id=session_id,
                       active_count=len(self.active_sessions))

    def _check_vram_available(self) -> bool:
        """
        Check if VRAM is available for new session

        Returns:
            True if VRAM usage < 90%, False otherwise

        Design note:
        90% threshold leaves 10% buffer for spikes.
        Aggressive but prevents OOM.
        """
        if not torch.cuda.is_available():
            return True  # CPU mode always OK

        allocated_gb = torch.cuda.memory_allocated(0) / 1024**3
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        usage_pct = (allocated_gb / total_gb) * 100

        # Threshold: 90%
        return usage_pct < 90.0

    def _get_vram_usage_gb(self) -> float:
        """
        Get current VRAM usage in GB

        Returns:
            VRAM allocated in GB, or 0.0 if CUDA not available
        """
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated(0) / 1024**3
        return 0.0

    def get_status(self) -> dict:
        """
        Get current session manager status

        Returns:
            Dictionary with status information
        """
        vram_info = {}
        if torch.cuda.is_available():
            allocated_gb = torch.cuda.memory_allocated(0) / 1024**3
            reserved_gb = torch.cuda.memory_reserved(0) / 1024**3
            total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3

            vram_info = {
                "allocated_gb": round(allocated_gb, 2),
                "reserved_gb": round(reserved_gb, 2),
                "total_gb": round(total_gb, 2),
                "usage_pct": round((allocated_gb / total_gb) * 100, 1)
            }

        return {
            "active_sessions": len(self.active_sessions),
            "max_sessions": self.max_sessions,
            "capacity_pct": round((len(self.active_sessions) / self.max_sessions) * 100, 1),
            "vram": vram_info
        }

    def get_vram_status(self) -> dict:
        """
        Get VRAM status (convenience method for monitoring)

        Returns:
            Dictionary with VRAM information:
            - total_gb: Total VRAM in GB
            - used_gb: Allocated VRAM in GB
            - free_gb: Free VRAM in GB
            - usage_percent: Usage percentage
        """
        if not torch.cuda.is_available():
            return {
                "total_gb": 0.0,
                "used_gb": 0.0,
                "free_gb": 0.0,
                "usage_percent": 0.0
            }

        allocated_gb = torch.cuda.memory_allocated(0) / 1024**3
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        free_gb = total_gb - allocated_gb
        usage_pct = (allocated_gb / total_gb) * 100

        return {
            "total_gb": round(total_gb, 2),
            "used_gb": round(allocated_gb, 2),
            "free_gb": round(free_gb, 2),
            "usage_percent": round(usage_pct, 1)
        }

    async def try_acquire_session(self, session_id: str, timeout: float = 1.0) -> bool:
        """
        Alias for acquire_session() (convenience method for tests)

        Args:
            session_id: Unique session identifier
            timeout: Timeout in seconds

        Returns:
            True if session acquired, False otherwise
        """
        return await self.acquire_session(session_id, timeout)


# Global singleton instance
# Linus would approve: simple, no factory pattern, just a global
_session_manager: SessionManager = SessionManager()


def get_session_manager() -> SessionManager:
    """
    Get global session manager instance

    Returns:
        SessionManager singleton

    Design note:
    No async, no lock, no complexity.
    SessionManager is created at module load time.
    """
    return _session_manager
