"""
Integration tests for concurrent session control system

Task 21: Comprehensive testing of the session queuing and VRAM monitoring integration.
Tests the complete flow from request to processing to completion.
"""

import asyncio
import pytest
import time
from unittest.mock import patch, MagicMock

from avatar.core.session_controller import get_session_controller, initialize_session_controller
from avatar.core.session_queue import get_session_queue, QueueState
from avatar.core.vram_monitor import get_vram_monitor


class TestSessionQueueBasics:
    """Test basic session queue functionality"""

    @pytest.fixture
    async def session_controller(self):
        """Initialize session controller for testing"""
        controller = get_session_controller()
        await controller.start()
        yield controller
        await controller.stop()

    @pytest.mark.asyncio
    async def test_single_session_immediate_processing(self, session_controller):
        """Test that a single session gets processed immediately when resources available"""
        # Mock VRAM monitor to always allow sessions
        with patch.object(session_controller.vram_monitor, 'predict_can_handle_service') as mock_predict:
            mock_predict.return_value = {
                "can_handle": True,
                "recommended_gpu": 0,
                "vram_required_gb": 1.0,
                "priority": "HIGH",
                "reasoning": ["Test case - GPU available"]
            }

            result = await session_controller.request_session(
                session_id="test-001",
                service_type="llm"
            )

            assert result.success
            assert result.processing_started
            assert result.queue_position is None or result.queue_position == 0

    @pytest.mark.asyncio
    async def test_queue_when_resources_unavailable(self, session_controller):
        """Test that sessions are queued when resources unavailable"""
        # Mock VRAM monitor to reject sessions
        with patch.object(session_controller.vram_monitor, 'predict_can_handle_service') as mock_predict:
            mock_predict.return_value = {
                "can_handle": False,
                "recommended_gpu": None,
                "vram_required_gb": 5.0,
                "priority": "HIGH",
                "reasoning": ["Test case - insufficient VRAM"]
            }

            result = await session_controller.request_session(
                session_id="test-002",
                service_type="llm"
            )

            assert result.success
            assert not result.processing_started
            assert result.queue_position is not None

    @pytest.mark.asyncio
    async def test_priority_ordering(self, session_controller):
        """Test that sessions are processed in priority order"""
        # Mock to queue all sessions
        with patch.object(session_controller.vram_monitor, 'predict_can_handle_service') as mock_predict:
            mock_predict.return_value = {"can_handle": False, "reasoning": ["Test queuing"]}

            # Add sessions in reverse priority order
            await session_controller.request_session("low-priority", "tts_hq")  # LOW priority
            await session_controller.request_session("high-priority", "llm")   # HIGH priority
            await session_controller.request_session("critical", "stt")        # CRITICAL priority

            queue = session_controller.session_queue.queue

            # Should be ordered by priority (critical first)
            assert queue[0].service_type == "stt"  # CRITICAL
            assert queue[1].service_type == "llm"  # HIGH
            assert queue[2].service_type == "tts_hq"  # LOW

    @pytest.mark.asyncio
    async def test_timeout_handling(self, session_controller):
        """Test that sessions timeout appropriately"""
        # Queue a session with very short timeout
        result = await session_controller.request_session(
            session_id="timeout-test",
            service_type="llm",
            timeout=0.1  # 100ms timeout
        )

        assert result.success

        # Wait for timeout
        await asyncio.sleep(0.2)

        # Session should be removed from queue due to timeout
        status = session_controller.get_session_status("timeout-test")
        # Should either be not found or have timeout status
        assert status is None or status.get("state") == "rejected"


class TestVRAMIntegration:
    """Test VRAM monitoring integration"""

    @pytest.fixture
    def mock_torch_cuda(self):
        """Mock torch.cuda for testing"""
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.device_count', return_value=2), \
             patch('torch.cuda.memory_allocated', return_value=8 * 1024**3), \
             patch('torch.cuda.memory_reserved', return_value=10 * 1024**3), \
             patch('torch.cuda.get_device_properties') as mock_props:

            # Mock GPU properties
            mock_device = MagicMock()
            mock_device.total_memory = 24 * 1024**3  # 24GB
            mock_device.name = "Test GPU"
            mock_props.return_value = mock_device

            yield mock_props

    def test_vram_status_reporting(self, mock_torch_cuda):
        """Test VRAM status is correctly reported"""
        vram_monitor = get_vram_monitor()
        status = vram_monitor.get_all_gpu_status()

        assert len(status) == 2  # Two GPUs
        for gpu_status in status:
            assert gpu_status.total_gb == 24.0
            assert gpu_status.allocated_gb == 8.0
            assert 0 <= gpu_status.usage_percent <= 100

    def test_service_prediction(self, mock_torch_cuda):
        """Test service handling prediction"""
        vram_monitor = get_vram_monitor()

        # Test different service types
        services = ["stt", "llm", "tts_fast", "tts_hq"]
        for service in services:
            prediction = vram_monitor.predict_can_handle_service(service)

            assert "can_handle" in prediction
            assert "vram_required_gb" in prediction
            assert "priority" in prediction
            assert isinstance(prediction["reasoning"], list)

            # STT should always be available (CPU only)
            if service == "stt":
                assert prediction["vram_required_gb"] == 0


class TestSessionControllerAPI:
    """Test session controller high-level functionality"""

    @pytest.fixture
    async def controller(self):
        """Session controller fixture"""
        controller = get_session_controller()
        await controller.start()
        yield controller
        await controller.stop()

    @pytest.mark.asyncio
    async def test_system_status(self, controller):
        """Test system status reporting"""
        status = controller.get_system_status()

        required_keys = ["timestamp", "queue", "vram", "capacity", "performance"]
        for key in required_keys:
            assert key in status

        assert "queue_size" in status["queue"]
        assert "processing_count" in status["queue"]
        assert "utilization_percent" in status["capacity"]

    @pytest.mark.asyncio
    async def test_service_availability(self, controller):
        """Test service availability reporting"""
        availability = controller.get_service_availability()

        expected_services = ["stt", "llm", "tts_fast", "tts_hq"]
        for service in expected_services:
            assert service in availability
            assert "available" in availability[service]
            assert "vram_required_gb" in availability[service]

    @pytest.mark.asyncio
    async def test_health_check(self, controller):
        """Test health check functionality"""
        health = controller.get_health_check()

        assert "healthy" in health
        assert "health_score" in health
        assert "issues" in health
        assert "metrics" in health

        assert isinstance(health["healthy"], bool)
        assert 0 <= health["health_score"] <= 100
        assert isinstance(health["issues"], list)


class TestConcurrentLoad:
    """Test system under concurrent load"""

    @pytest.fixture
    async def controller(self):
        """Session controller fixture"""
        controller = get_session_controller()
        await controller.start()
        yield controller
        await controller.stop()

    @pytest.mark.asyncio
    async def test_multiple_concurrent_requests(self, controller):
        """Test handling multiple concurrent session requests"""
        # Mock VRAM to handle some but not all requests
        with patch.object(controller.vram_monitor, 'predict_can_handle_service') as mock_predict:
            mock_predict.return_value = {
                "can_handle": True,
                "recommended_gpu": 0,
                "reasoning": ["Test concurrent load"]
            }

            # Create multiple concurrent requests
            tasks = []
            for i in range(10):
                task = controller.request_session(
                    session_id=f"concurrent-{i}",
                    service_type="llm"
                )
                tasks.append(task)

            # Wait for all requests to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # All requests should succeed (either immediate or queued)
            for result in results:
                assert not isinstance(result, Exception)
                assert result.success

    @pytest.mark.asyncio
    async def test_queue_full_handling(self, controller):
        """Test behavior when queue reaches capacity"""
        # Set small queue size for testing
        controller.session_queue.max_queue_size = 3

        # Mock to always queue sessions
        with patch.object(controller.vram_monitor, 'predict_can_handle_service') as mock_predict:
            mock_predict.return_value = {"can_handle": False, "reasoning": ["Test queue full"]}

            results = []

            # Fill the queue
            for i in range(5):  # More than max_queue_size
                result = await controller.request_session(
                    session_id=f"queue-full-{i}",
                    service_type="llm"
                )
                results.append(result)

            # First 3 should succeed (queued), later ones should fail
            successful = [r for r in results if r.success]
            failed = [r for r in results if not r.success]

            assert len(successful) == 3
            assert len(failed) == 2

            # Failed requests should have appropriate error codes
            for failed_result in failed:
                assert failed_result.error_code == "QUEUE_FULL"


class TestErrorRecovery:
    """Test error handling and recovery scenarios"""

    @pytest.fixture
    async def controller(self):
        """Session controller fixture"""
        controller = get_session_controller()
        await controller.start()
        yield controller
        await controller.stop()

    @pytest.mark.asyncio
    async def test_vram_monitor_failure_handling(self, controller):
        """Test behavior when VRAM monitor fails"""
        # Mock VRAM monitor to raise exception
        with patch.object(controller.vram_monitor, 'predict_can_handle_service', side_effect=Exception("VRAM monitor failed")):
            result = await controller.request_session(
                session_id="vram-fail-test",
                service_type="llm"
            )

            # Should handle gracefully
            assert not result.success
            assert result.error_code == "INTERNAL_ERROR"

    @pytest.mark.asyncio
    async def test_session_cancellation(self, controller):
        """Test session cancellation functionality"""
        # Queue a session
        with patch.object(controller.vram_monitor, 'predict_can_handle_service') as mock_predict:
            mock_predict.return_value = {"can_handle": False, "reasoning": ["Test cancellation"]}

            result = await controller.request_session(
                session_id="cancel-test",
                service_type="llm"
            )

            assert result.success

            # Cancel the session
            cancelled = await controller.cancel_session("cancel-test")
            assert cancelled

            # Should no longer be in queue
            status = controller.get_session_status("cancel-test")
            if status:
                assert status["state"] == "cancelled"


@pytest.mark.asyncio
async def test_full_integration_flow():
    """Test complete flow from request to completion"""
    controller = get_session_controller()
    await controller.start()

    try:
        # Mock successful processing
        with patch.object(controller.vram_monitor, 'predict_can_handle_service') as mock_predict:
            mock_predict.return_value = {
                "can_handle": True,
                "recommended_gpu": 0,
                "reasoning": ["Integration test"]
            }

            # Request session
            result = await controller.request_session(
                session_id="integration-test",
                service_type="llm"
            )

            assert result.success

            # Check initial status
            initial_status = controller.get_session_status("integration-test")
            assert initial_status is not None

            # Wait a moment for processing to potentially start
            await asyncio.sleep(0.1)

            # Check final system status
            system_status = controller.get_system_status()
            assert system_status["timestamp"] > 0

    finally:
        await controller.stop()


if __name__ == "__main__":
    # Run basic tests
    asyncio.run(test_full_integration_flow())
    print("✅ Basic integration test passed!")