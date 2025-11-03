"""
TDD Unit Tests for SessionManager

Following TDD Red-Green-Refactor cycle for session management functionality.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from avatar.core.session_manager import SessionManager


class TestSessionManagerInitialization:
    """Test SessionManager initialization following TDD principles"""

    def test_init_with_default_max_sessions(self):
        """Test SessionManager uses default max sessions from config"""
        manager = SessionManager()
        assert manager.max_sessions == 4  # From config
        assert isinstance(manager.active_sessions, dict)
        assert manager._semaphore._value == 4

    def test_init_with_custom_max_sessions(self):
        """Test SessionManager with custom max sessions"""
        manager = SessionManager(max_sessions=10)
        assert manager.max_sessions == 10
        assert manager._semaphore._value == 10

    def test_init_creates_empty_active_sessions(self):
        """Test initialization creates empty active sessions dict"""
        manager = SessionManager()
        assert manager.active_sessions == {}


class TestVRAMMonitoring:
    """Test VRAM monitoring functionality"""

    @patch('avatar.core.session_manager.torch')
    def test_check_vram_available_no_cuda(self, mock_torch):
        """Test VRAM check when CUDA not available returns True"""
        mock_torch.cuda.is_available.return_value = False

        manager = SessionManager()
        result = manager._check_vram_available()

        assert result is True
        mock_torch.cuda.is_available.assert_called_once()

    @patch('avatar.core.session_manager.torch')
    def test_check_vram_available_under_threshold(self, mock_torch):
        """Test VRAM check under 90% threshold returns True"""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.memory_allocated.return_value = 8 * (1024**3)  # 8GB
        mock_device_props = MagicMock()
        mock_device_props.total_memory = 24 * (1024**3)  # 24GB
        mock_torch.cuda.get_device_properties.return_value = mock_device_props

        manager = SessionManager()
        result = manager._check_vram_available()

        assert result is True  # 33% < 90%

    @patch('avatar.core.session_manager.torch')
    def test_check_vram_available_over_threshold(self, mock_torch):
        """Test VRAM check over 90% threshold returns False"""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.memory_allocated.return_value = 22 * (1024**3)  # 22GB
        mock_device_props = MagicMock()
        mock_device_props.total_memory = 24 * (1024**3)  # 24GB
        mock_torch.cuda.get_device_properties.return_value = mock_device_props

        manager = SessionManager()
        result = manager._check_vram_available()

        assert result is False  # 92% > 90%

    @patch('avatar.core.session_manager.torch')
    def test_get_vram_status_returns_dict(self, mock_torch):
        """Test get_vram_status returns properly formatted dict"""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.memory_allocated.return_value = 8 * (1024**3)  # 8GB
        mock_device_props = MagicMock()
        mock_device_props.total_memory = 24 * (1024**3)  # 24GB
        mock_torch.cuda.get_device_properties.return_value = mock_device_props

        manager = SessionManager()
        status = manager.get_vram_status()

        assert isinstance(status, dict)
        assert 'total_gb' in status
        assert 'used_gb' in status
        assert 'usage_percent' in status
        assert status['total_gb'] == 24.0
        assert status['used_gb'] == 8.0
        assert status['usage_percent'] == pytest.approx(33.3, rel=1e-2)


class TestSessionAcquisition:
    """Test session acquisition and release"""

    @pytest.fixture
    def mock_manager(self):
        """Create SessionManager with mocked VRAM check"""
        with patch.object(SessionManager, '_check_vram_available', return_value=True):
            return SessionManager(max_sessions=2)

    @pytest.mark.asyncio
    async def test_acquire_session_success(self, mock_manager):
        """Test successful session acquisition"""
        session_id = "test_session_1"

        result = await mock_manager.acquire_session(session_id, timeout=1.0)

        assert result is True
        assert session_id in mock_manager.active_sessions
        assert mock_manager.active_sessions[session_id] is True

    @pytest.mark.asyncio
    async def test_acquire_session_vram_exhausted(self):
        """Test session acquisition fails when VRAM exhausted"""
        with patch.object(SessionManager, '_check_vram_available', return_value=False):
            manager = SessionManager(max_sessions=2)

            result = await manager.acquire_session("test_session", timeout=1.0)

            assert result is False
            assert "test_session" not in manager.active_sessions

    @pytest.mark.asyncio
    async def test_acquire_session_max_sessions_reached(self, mock_manager):
        """Test session acquisition fails when max sessions reached"""
        # Acquire max sessions
        session1 = await mock_manager.acquire_session("session_1", timeout=1.0)
        session2 = await mock_manager.acquire_session("session_2", timeout=1.0)

        assert session1 is True
        assert session2 is True

        # Try to acquire one more (should timeout)
        result = await mock_manager.acquire_session("session_3", timeout=0.1)

        assert result is False
        assert "session_3" not in mock_manager.active_sessions

    @pytest.mark.asyncio
    async def test_release_session_success(self, mock_manager):
        """Test successful session release"""
        session_id = "test_session"

        # Acquire session first
        await mock_manager.acquire_session(session_id, timeout=1.0)
        assert session_id in mock_manager.active_sessions

        # Release session
        mock_manager.release_session(session_id)

        assert session_id not in mock_manager.active_sessions

    @pytest.mark.asyncio
    async def test_release_nonexistent_session(self, mock_manager):
        """Test releasing non-existent session doesn't raise error"""
        # Should not raise exception
        mock_manager.release_session("nonexistent_session")

        # Active sessions should remain empty
        assert mock_manager.active_sessions == {}

    @pytest.mark.asyncio
    async def test_session_lifecycle_allows_reacquisition(self, mock_manager):
        """Test that releasing session allows new acquisition"""
        session_id_1 = "session_1"
        session_id_2 = "session_2"
        session_id_3 = "session_3"

        # Fill up all sessions
        await mock_manager.acquire_session(session_id_1, timeout=1.0)
        await mock_manager.acquire_session(session_id_2, timeout=1.0)

        # Release one session
        mock_manager.release_session(session_id_1)

        # Should be able to acquire new session
        result = await mock_manager.acquire_session(session_id_3, timeout=1.0)

        assert result is True
        assert session_id_3 in mock_manager.active_sessions
        assert session_id_1 not in mock_manager.active_sessions


class TestSessionManagerSingleton:
    """Test singleton pattern implementation"""

    def test_get_session_manager_returns_same_instance(self):
        """Test that get_session_manager returns singleton instance"""
        from avatar.core.session_manager import get_session_manager

        manager1 = get_session_manager()
        manager2 = get_session_manager()

        assert manager1 is manager2

    def test_singleton_preserves_state(self):
        """Test that singleton preserves state across calls"""
        from avatar.core.session_manager import get_session_manager

        manager1 = get_session_manager()
        manager1.test_property = "test_value"

        manager2 = get_session_manager()

        assert hasattr(manager2, 'test_property')
        assert manager2.test_property == "test_value"


class TestSessionManagerIntegration:
    """Integration tests for SessionManager with real async operations"""

    @pytest.mark.asyncio
    async def test_concurrent_session_acquisition(self):
        """Test concurrent session acquisition works correctly"""
        with patch.object(SessionManager, '_check_vram_available', return_value=True):
            manager = SessionManager(max_sessions=3)

            # Simulate concurrent acquisition
            tasks = [
                manager.acquire_session(f"session_{i}", timeout=1.0)
                for i in range(5)  # Try to acquire 5 sessions (max is 3)
            ]

            results = await asyncio.gather(*tasks)

            # Exactly 3 should succeed, 2 should fail
            successful = sum(1 for r in results if r is True)
            failed = sum(1 for r in results if r is False)

            assert successful == 3
            assert failed == 2
            assert len(manager.active_sessions) == 3

    @pytest.mark.asyncio
    async def test_session_timeout_behavior(self):
        """Test session acquisition timeout behavior"""
        with patch.object(SessionManager, '_check_vram_available', return_value=True):
            manager = SessionManager(max_sessions=1)

            # Acquire the only session
            result1 = await manager.acquire_session("session_1", timeout=1.0)
            assert result1 is True

            # Measure timeout
            start_time = asyncio.get_event_loop().time()
            result2 = await manager.acquire_session("session_2", timeout=0.5)
            end_time = asyncio.get_event_loop().time()

            assert result2 is False
            assert (end_time - start_time) >= 0.4  # Allow some margin
            assert (end_time - start_time) <= 0.8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])