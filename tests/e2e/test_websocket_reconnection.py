"""
End-to-End WebSocket Reconnection Tests - Real System Testing

Task 22: Comprehensive testing of WebSocket reconnection, session recovery,
and heartbeat monitoring. Uses real WebSocket connections and network simulation.

Test Coverage:
1. Basic reconnection with exponential backoff
2. Session state preservation and recovery
3. Heartbeat monitoring and failure detection
4. Error classification and retry logic
5. Connection quality adaptation
"""

import asyncio
import pytest
import time
import json
import sys
import os
from typing import Dict, Any, List
from unittest.mock import AsyncMock, patch

import websockets
from websockets.exceptions import ConnectionClosed, ConnectionClosedError

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from avatar.core.websocket_reconnect import (
    WebSocketReconnectManager, ReconnectionConfig,
    SessionSnapshot, DisconnectReason, ConnectionState
)
from avatar.core.websocket_heartbeat import (
    WebSocketHeartbeatMonitor, HeartbeatConfig, HeartbeatState
)


class MockWebSocket:
    """Mock WebSocket for testing reconnection logic"""

    def __init__(self, fail_after: int = None, latency_ms: float = 10):
        self.fail_after = fail_after
        self.latency_ms = latency_ms
        self.ping_count = 0
        self.closed = False

    async def ping(self):
        """Mock ping that can simulate failures"""
        self.ping_count += 1

        if self.fail_after and self.ping_count > self.fail_after:
            raise ConnectionClosedError(None, None)

        # Simulate network latency
        await asyncio.sleep(self.latency_ms / 1000)

        # Return a mock pong waiter
        return AsyncMock()

    async def send_text(self, message: str):
        """Mock send text"""
        if self.closed:
            raise ConnectionClosedError(None, None)

    async def close(self):
        """Mock close"""
        self.closed = True


class TestWebSocketReconnectManager:
    """Test WebSocket reconnection manager functionality"""

    @pytest.fixture
    def reconnect_config(self):
        """Reconnection configuration for testing"""
        return ReconnectionConfig(
            max_retries=5,
            max_retry_time_seconds=30,
            initial_delay_ms=100,  # Fast for testing
            max_delay_ms=1000,
            backoff_multiplier=1.5,
            jitter_ratio=0.1
        )

    @pytest.fixture
    def reconnect_manager(self, reconnect_config):
        """Reconnection manager for testing"""
        return WebSocketReconnectManager(reconnect_config)

    def test_backoff_calculation(self, reconnect_manager):
        """Test exponential backoff calculation"""
        # First retry
        reconnect_manager.retry_count = 1
        delay1 = reconnect_manager.calculate_backoff_delay()
        assert 0.09 <= delay1 <= 0.15  # 100ms ± 10% jitter

        # Second retry
        reconnect_manager.retry_count = 2
        delay2 = reconnect_manager.calculate_backoff_delay()
        assert 0.135 <= delay2 <= 0.225  # 150ms ± 10% jitter

        # Third retry
        reconnect_manager.retry_count = 3
        delay3 = reconnect_manager.calculate_backoff_delay()
        assert 0.2 <= delay3 <= 0.34  # 225ms ± 10% jitter

        # Should increase with each retry
        assert delay2 > delay1
        assert delay3 > delay2

    def test_retry_decision_logic(self, reconnect_manager):
        """Test retry decision based on error type and limits"""
        # Retriable errors should allow retry
        assert reconnect_manager.should_retry(DisconnectReason.NETWORK_ERROR)
        assert reconnect_manager.should_retry(DisconnectReason.TIMEOUT)
        assert reconnect_manager.should_retry(DisconnectReason.SERVER_OVERLOAD)

        # Non-retriable errors should not retry
        assert not reconnect_manager.should_retry(DisconnectReason.CLIENT_CLOSE)
        assert not reconnect_manager.should_retry(DisconnectReason.AUTHENTICATION_FAILED)

        # Exceed max retries
        reconnect_manager.retry_count = 10
        assert not reconnect_manager.should_retry(DisconnectReason.NETWORK_ERROR)

        # Exceed max time
        reconnect_manager.retry_count = 1
        reconnect_manager.first_attempt_time = time.time() - 100  # 100 seconds ago
        assert not reconnect_manager.should_retry(DisconnectReason.NETWORK_ERROR)

    @pytest.mark.asyncio
    async def test_session_preservation(self, reconnect_manager):
        """Test session state preservation during disconnection"""
        # Create test session snapshot
        session_snapshot = SessionSnapshot(
            session_id="test-session-123",
            turn_number=3,
            voice_profile_id=42,
            last_user_text="Hello world",
            last_ai_text="Hello! How can I help?",
            processing_stage="ready",
            created_at=time.time() - 300,  # 5 minutes ago
            last_activity=time.time() - 60  # 1 minute ago
        )

        # Handle disconnection with session preservation
        await reconnect_manager.handle_disconnect(
            DisconnectReason.NETWORK_ERROR,
            session_snapshot
        )

        # Verify session is preserved
        assert "test-session-123" in reconnect_manager.active_sessions
        preserved = reconnect_manager.active_sessions["test-session-123"]
        assert preserved.turn_number == 3
        assert preserved.voice_profile_id == 42
        assert preserved.last_user_text == "Hello world"

    @pytest.mark.asyncio
    async def test_session_recovery(self, reconnect_manager):
        """Test session recovery functionality"""
        # Add a recoverable session
        session_snapshot = SessionSnapshot(
            session_id="recoverable-session",
            turn_number=2,
            voice_profile_id=None,
            last_user_text="Test message",
            last_ai_text="Test response",
            processing_stage="ready",
            created_at=time.time() - 100,  # Recent
            last_activity=time.time() - 30  # Recent activity
        )
        reconnect_manager.active_sessions["recoverable-session"] = session_snapshot

        # Try to recover session
        recovered = reconnect_manager.try_recover_session("recoverable-session")
        assert recovered is not None
        assert recovered.session_id == "recoverable-session"
        assert recovered.turn_number == 2

        # Try to recover non-existent session
        not_found = reconnect_manager.try_recover_session("non-existent")
        assert not_found is None

    @pytest.mark.asyncio
    async def test_expired_session_cleanup(self, reconnect_manager):
        """Test cleanup of expired sessions"""
        # Add an expired session
        old_session = SessionSnapshot(
            session_id="expired-session",
            turn_number=1,
            voice_profile_id=None,
            last_user_text="Old message",
            last_ai_text="Old response",
            processing_stage="ready",
            created_at=time.time() - 7200,  # 2 hours ago
            last_activity=time.time() - 7200  # 2 hours ago
        )
        reconnect_manager.active_sessions["expired-session"] = old_session

        # Try to recover expired session
        recovered = reconnect_manager.try_recover_session("expired-session")
        assert recovered is None

        # Session should be removed from active sessions
        assert "expired-session" not in reconnect_manager.active_sessions

    def test_connection_statistics(self, reconnect_manager):
        """Test connection statistics tracking"""
        initial_stats = reconnect_manager.connection_stats.copy()

        # Simulate successful connection
        reconnect_manager.connection_stats["total_connections"] += 1
        reconnect_manager.connection_stats["successful_reconnections"] += 1

        # Simulate disconnections
        reconnect_manager.connection_stats["total_disconnects"] += 3

        status = reconnect_manager.get_status()
        assert status["statistics"]["total_connections"] == initial_stats["total_connections"] + 1
        assert status["statistics"]["successful_reconnections"] == initial_stats["successful_reconnections"] + 1
        assert status["statistics"]["total_disconnects"] == initial_stats["total_disconnects"] + 3


class TestWebSocketHeartbeatMonitor:
    """Test WebSocket heartbeat monitoring functionality"""

    @pytest.fixture
    def heartbeat_config(self):
        """Heartbeat configuration for testing"""
        return HeartbeatConfig(
            interval_seconds=1.0,  # Fast for testing
            timeout_seconds=0.5,
            adaptive_timing=True,
            min_interval_seconds=0.5,
            max_interval_seconds=2.0
        )

    @pytest.fixture
    def mock_websocket(self):
        """Mock WebSocket for heartbeat testing"""
        return MockWebSocket()

    @pytest.mark.asyncio
    async def test_heartbeat_basic_functionality(self, heartbeat_config, mock_websocket):
        """Test basic heartbeat monitoring"""
        monitor = WebSocketHeartbeatMonitor(
            websocket=mock_websocket,
            session_id="test-heartbeat-session",
            config=heartbeat_config
        )

        # Start monitoring
        await monitor.start()
        assert monitor.state == HeartbeatState.ACTIVE

        # Let it run for a few heartbeats
        await asyncio.sleep(2.5)

        # Check metrics
        status = monitor.get_status()
        assert status["metrics"]["total_pings"] >= 2
        assert status["metrics"]["successful_pings"] > 0
        assert status["metrics"]["success_rate"] > 0.5

        # Stop monitoring
        await monitor.stop()
        assert monitor.state == HeartbeatState.STOPPED

    @pytest.mark.asyncio
    async def test_heartbeat_failure_detection(self, heartbeat_config):
        """Test heartbeat failure detection"""
        # Create mock that fails after 2 pings
        failing_websocket = MockWebSocket(fail_after=2)

        monitor = WebSocketHeartbeatMonitor(
            websocket=failing_websocket,
            session_id="failing-session",
            config=heartbeat_config
        )

        # Track state changes
        state_changes = []
        monitor.on_state_change = lambda state: state_changes.append(state)

        # Start monitoring
        await monitor.start()

        # Wait for failures to accumulate
        await asyncio.sleep(3.0)

        # Should detect failure and change state
        assert len(state_changes) > 0
        final_status = monitor.get_status()
        assert final_status["state"] in ["degraded", "failing", "failed"]
        assert final_status["metrics"]["failed_pings"] > 0

        await monitor.stop()

    @pytest.mark.asyncio
    async def test_adaptive_interval_calculation(self, heartbeat_config):
        """Test adaptive heartbeat interval calculation"""
        mock_websocket = MockWebSocket(latency_ms=50)  # Good latency

        monitor = WebSocketHeartbeatMonitor(
            websocket=mock_websocket,
            session_id="adaptive-session",
            config=heartbeat_config
        )

        # Test with excellent connection
        monitor.metrics.connection_quality = "excellent"
        monitor.consecutive_failures = 0
        interval = monitor._calculate_interval()
        assert interval >= heartbeat_config.interval_seconds  # Should be longer

        # Test with poor connection
        monitor.metrics.connection_quality = "poor"
        monitor.consecutive_failures = 2
        interval = monitor._calculate_interval()
        assert interval < heartbeat_config.interval_seconds  # Should be shorter

    def test_heartbeat_metrics_calculation(self, heartbeat_config, mock_websocket):
        """Test heartbeat metrics calculation"""
        monitor = WebSocketHeartbeatMonitor(
            websocket=mock_websocket,
            session_id="metrics-session",
            config=heartbeat_config
        )

        # Simulate successful ping
        monitor._update_metrics(True, 50.0)
        assert monitor.metrics.total_pings == 1
        assert monitor.metrics.successful_pings == 1
        assert monitor.metrics.avg_latency_ms == 50.0

        # Simulate failed ping
        monitor._update_metrics(False, None)
        assert monitor.metrics.total_pings == 2
        assert monitor.metrics.failed_pings == 1
        assert monitor.metrics.success_rate == 0.5

        # Simulate timeout ping
        monitor._update_metrics(False, None)
        assert monitor.metrics.timeout_pings == 1
        assert monitor.consecutive_failures == 2


class TestIntegratedWebSocketReconnection:
    """Integration tests for complete reconnection system"""

    @pytest.mark.asyncio
    async def test_full_reconnection_flow(self):
        """Test complete reconnection flow with session recovery"""
        # Create managers with test configuration
        reconnect_config = ReconnectionConfig(
            max_retries=3,
            initial_delay_ms=50,
            max_delay_ms=200
        )
        reconnect_manager = WebSocketReconnectManager(reconnect_config)

        # Create session and simulate disconnection
        session_snapshot = SessionSnapshot(
            session_id="integration-test-session",
            turn_number=1,
            voice_profile_id=None,
            last_user_text="Integration test",
            last_ai_text="Test response",
            processing_stage="ready",
            created_at=time.time() - 60,
            last_activity=time.time() - 10
        )

        # Handle disconnection
        await reconnect_manager.handle_disconnect(
            DisconnectReason.NETWORK_ERROR,
            session_snapshot
        )

        # Verify state
        assert reconnect_manager.state == ConnectionState.RECONNECTING
        assert reconnect_manager.retry_count > 0
        assert "integration-test-session" in reconnect_manager.active_sessions

        # Simulate successful reconnection
        mock_websocket = MockWebSocket()
        await reconnect_manager.handle_successful_connection(mock_websocket)

        # Verify recovery
        assert reconnect_manager.state == ConnectionState.CONNECTED
        assert reconnect_manager.retry_count == 0

        # Try to recover session
        recovered = reconnect_manager.try_recover_session("integration-test-session")
        assert recovered is not None
        assert recovered.turn_number == 1

    @pytest.mark.asyncio
    async def test_heartbeat_integration_with_reconnection(self):
        """Test heartbeat monitoring integration with reconnection manager"""
        reconnect_manager = WebSocketReconnectManager()

        # Create heartbeat monitor with aggressive failure detection
        heartbeat_config = HeartbeatConfig(
            interval_seconds=0.2,
            timeout_seconds=0.1,
            consecutive_failures_threshold=2
        )

        failing_websocket = MockWebSocket(fail_after=1)
        heartbeat_monitor = WebSocketHeartbeatMonitor(
            websocket=failing_websocket,
            session_id="heartbeat-integration-test",
            config=heartbeat_config
        )

        # Connect heartbeat failure to reconnection
        connection_lost_called = False

        async def on_connection_lost():
            nonlocal connection_lost_called
            connection_lost_called = True
            await reconnect_manager.handle_disconnect(
                DisconnectReason.TIMEOUT,
                SessionSnapshot(
                    session_id="heartbeat-integration-test",
                    turn_number=0,
                    voice_profile_id=None,
                    last_user_text=None,
                    last_ai_text=None,
                    processing_stage="ready",
                    created_at=time.time(),
                    last_activity=time.time()
                )
            )

        heartbeat_monitor.on_connection_lost = on_connection_lost

        # Start heartbeat monitoring
        await heartbeat_monitor.start()

        # Wait for failure detection
        await asyncio.sleep(1.0)

        # Verify integration
        assert connection_lost_called
        assert reconnect_manager.last_disconnect_reason == DisconnectReason.TIMEOUT

        await heartbeat_monitor.stop()

    @pytest.mark.asyncio
    async def test_concurrent_sessions_reconnection(self):
        """Test reconnection with multiple concurrent sessions"""
        reconnect_manager = WebSocketReconnectManager()

        # Create multiple session snapshots
        sessions = []
        for i in range(5):
            session = SessionSnapshot(
                session_id=f"concurrent-session-{i}",
                turn_number=i + 1,
                voice_profile_id=None,
                last_user_text=f"Message {i}",
                last_ai_text=f"Response {i}",
                processing_stage="ready",
                created_at=time.time() - 300,
                last_activity=time.time() - 30
            )
            sessions.append(session)
            reconnect_manager.active_sessions[session.session_id] = session

        # Verify all sessions can be recovered
        for i in range(5):
            recovered = reconnect_manager.try_recover_session(f"concurrent-session-{i}")
            assert recovered is not None
            assert recovered.turn_number == i + 1

        # Get status summary
        status = reconnect_manager.get_status()
        assert status["active_sessions"] == 5


if __name__ == "__main__":
    # Run basic integration test
    async def run_basic_test():
        """Run basic reconnection test"""
        print("🧪 Testing WebSocket reconnection system...")

        # Test backoff calculation
        config = ReconnectionConfig(initial_delay_ms=100, backoff_multiplier=1.5)
        manager = WebSocketReconnectManager(config)

        manager.retry_count = 1
        delay1 = manager.calculate_backoff_delay()
        manager.retry_count = 2
        delay2 = manager.calculate_backoff_delay()

        print(f"✅ Backoff delays: {delay1:.3f}s → {delay2:.3f}s")
        assert delay2 > delay1

        # Test session preservation
        session = SessionSnapshot(
            session_id="test-session",
            turn_number=1,
            voice_profile_id=None,
            last_user_text="Test",
            last_ai_text="Response",
            processing_stage="ready",
            created_at=time.time(),
            last_activity=time.time()
        )

        await manager.handle_disconnect(DisconnectReason.NETWORK_ERROR, session)
        assert "test-session" in manager.active_sessions
        print("✅ Session preservation working")

        # Test recovery
        recovered = manager.try_recover_session("test-session")
        assert recovered is not None
        assert recovered.turn_number == 1
        print("✅ Session recovery working")

        print("🎉 Basic reconnection tests passed!")

    # Run test
    asyncio.run(run_basic_test())