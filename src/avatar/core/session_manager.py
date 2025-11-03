"""
Session Manager for VRAM monitoring and concurrency control

Manages global session limits and VRAM usage to prevent OOM errors.
Implements Linus-style simplicity: single responsibility, no special cases.
"""

import asyncio
from typing import Dict

import structlog
import torch

from avatar.core.config import config

logger = structlog.get_logger()


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

    def __init__(self, max_sessions: int = None):
        """
        Initialize session manager

        Args:
            max_sessions: Maximum concurrent sessions (default from config)
        """
        self.max_sessions = max_sessions or config.MAX_CONCURRENT_SESSIONS
        self.active_sessions: Dict[str, bool] = {}
        self._semaphore = asyncio.Semaphore(self.max_sessions)
        self._lock = asyncio.Lock()  # Protect active_sessions dict

        logger.info("session_manager.init",
                   max_sessions=self.max_sessions,
                   vram_limit_gb=config.VRAM_LIMIT_GB)

    async def acquire_session(self, session_id: str, timeout: float = 1.0) -> bool:
        """
        Try to acquire a session slot

        Args:
            session_id: Unique session identifier
            timeout: Timeout in seconds (fail fast)

        Returns:
            True if session acquired, False if server is full

        Design note:
        Uses timeout=1.0s to fail fast. No waiting queue.
        Linus would say: "Don't make users wait. Tell them immediately."
        """
        # 1. Check VRAM availability (fast path)
        if not self._check_vram_available():
            vram_usage = self._get_vram_usage_gb()
            logger.warning("session_manager.vram_full",
                          session_id=session_id,
                          vram_allocated_gb=vram_usage,
                          vram_limit_gb=config.VRAM_LIMIT_GB)
            return False

        # 2. Try to acquire semaphore with timeout
        try:
            acquired = await asyncio.wait_for(
                self._semaphore.acquire(),
                timeout=timeout
            )

            if not acquired:
                return False

            # 3. Register session
            async with self._lock:
                self.active_sessions[session_id] = True

            logger.info("session_manager.acquired",
                       session_id=session_id,
                       active_count=len(self.active_sessions),
                       max_sessions=self.max_sessions)

            return True

        except asyncio.TimeoutError:
            logger.warning("session_manager.limit_reached",
                          session_id=session_id,
                          active_count=len(self.active_sessions),
                          max_sessions=self.max_sessions)
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
