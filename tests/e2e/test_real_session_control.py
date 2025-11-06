"""
End-to-End Real Session Control Tests - NO MOCKS

Task 21 TDD: Real testing with actual VRAM, PyTorch tensors, and system resources.
Linus principle: "Test the real thing, not a simulation."

This test suite:
- Uses actual PyTorch CUDA operations
- Creates real VRAM pressure
- Tests actual queue processing
- Validates real system behavior under load
"""

import asyncio
import pytest
import time
import os
import sys
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor

import torch
import structlog

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from avatar.core.session_controller import get_session_controller, initialize_session_controller
from avatar.core.session_queue import get_session_queue, QueueState
from avatar.core.vram_monitor import get_vram_monitor
from avatar.core.config import config

logger = structlog.get_logger()


class RealVRAMLoader:
    """
    Real VRAM loader for testing - creates actual memory pressure
    No mocks, uses real PyTorch tensors and GPU operations
    """

    def __init__(self):
        self.active_tensors: List[torch.Tensor] = []
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def allocate_vram(self, gb: float) -> torch.Tensor:
        """
        Allocate actual VRAM using PyTorch tensors

        Args:
            gb: Amount of VRAM to allocate in GB

        Returns:
            PyTorch tensor consuming the requested VRAM
        """
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for real VRAM testing")

        # Calculate tensor size for requested GB
        bytes_needed = int(gb * 1024**3)
        elements_needed = bytes_needed // 4  # float32 = 4 bytes

        try:
            tensor = torch.randn(elements_needed, device=self.device, dtype=torch.float32)
            self.active_tensors.append(tensor)

            # Force allocation
            torch.cuda.synchronize()

            logger.info("real_vram_loader.allocated",
                       requested_gb=gb,
                       actual_gb=tensor.element_size() * tensor.nelement() / 1024**3,
                       total_tensors=len(self.active_tensors))

            return tensor

        except torch.cuda.OutOfMemoryError as e:
            logger.error("real_vram_loader.oom", requested_gb=gb, error=str(e))
            raise

    def free_vram(self, gb: float = None):
        """
        Free VRAM by deleting tensors

        Args:
            gb: Amount to free (None = free all)
        """
        if not torch.cuda.is_available():
            return

        if gb is None:
            # Free all
            self.active_tensors.clear()
        else:
            # Free approximately the requested amount
            bytes_to_free = int(gb * 1024**3)
            bytes_freed = 0

            while self.active_tensors and bytes_freed < bytes_to_free:
                tensor = self.active_tensors.pop()
                bytes_freed += tensor.element_size() * tensor.nelement()
                del tensor

        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        logger.info("real_vram_loader.freed",
                   remaining_tensors=len(self.active_tensors))

    def get_current_usage_gb(self) -> float:
        """Get current VRAM usage in GB"""
        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.memory_allocated(0) / 1024**3

    def cleanup(self):
        """Complete cleanup"""
        self.free_vram()  # Free all


class TestRealSessionQueue:
    """Test session queue with real system resources"""

    @pytest.fixture(autouse=True)
    async def setup_real_system(self):
        """Setup real system components - no mocks"""
        # Initialize real session controller
        self.controller = get_session_controller()
        await self.controller.start()

        # Real VRAM loader
        self.vram_loader = RealVRAMLoader()

        # Real VRAM monitor
        self.vram_monitor = get_vram_monitor()

        # Ensure clean state
        self.vram_loader.cleanup()
        await asyncio.sleep(0.1)  # Allow monitoring to update

        yield

        # Cleanup
        self.vram_loader.cleanup()
        await self.controller.stop()

    @pytest.mark.asyncio
    async def test_real_immediate_processing_when_vram_available(self):
        """Test immediate processing when real VRAM is available"""
        # Ensure VRAM is available
        self.vram_loader.cleanup()
        await asyncio.sleep(0.5)  # Allow VRAM monitor to update

        # Request session with real system
        result = await self.controller.request_session(
            session_id="real-immediate-test",
            service_type="llm"
        )

        # Should succeed (either immediate or queued, both are valid)
        assert result.success, f"Session request failed: {result.message}"

        # Verify session is tracked in real system
        status = self.controller.get_session_status("real-immediate-test")
        assert status is not None, "Session not found in real system"

        logger.info("test.immediate_processing",
                   result=result.__dict__,
                   status=status)

    @pytest.mark.asyncio
    async def test_real_queuing_under_vram_pressure(self):
        """Test queuing behavior under real VRAM pressure"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA required for VRAM pressure testing")

        # Create real VRAM pressure
        available_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        pressure_gb = available_gb * 0.9  # Use 90% of VRAM

        logger.info("test.creating_vram_pressure",
                   available_gb=available_gb,
                   pressure_gb=pressure_gb)

        try:
            pressure_tensor = self.vram_loader.allocate_vram(pressure_gb)
            await asyncio.sleep(1.0)  # Allow VRAM monitor to detect pressure

            # Request session under pressure
            result = await self.controller.request_session(
                session_id="real-pressure-test",
                service_type="llm"
            )

            # Should either succeed with queuing or indicate resource constraints
            if result.success:
                status = self.controller.get_session_status("real-pressure-test")
                logger.info("test.pressure_session_status", status=status)

                # If queued, should have queue position
                if not result.processing_started:
                    assert result.queue_position is not None

            logger.info("test.vram_pressure_result",
                       result=result.__dict__,
                       current_vram_gb=self.vram_loader.get_current_usage_gb())

        finally:
            self.vram_loader.free_vram()

    @pytest.mark.asyncio
    async def test_real_priority_ordering(self):
        """Test that priority ordering works with real system"""
        # Create moderate VRAM pressure to force queuing
        if torch.cuda.is_available():
            available_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            pressure_gb = available_gb * 0.7  # 70% pressure to force some queuing

            try:
                self.vram_loader.allocate_vram(pressure_gb)
                await asyncio.sleep(0.5)
            except torch.cuda.OutOfMemoryError:
                pressure_gb = available_gb * 0.5  # Reduce pressure if OOM
                self.vram_loader.allocate_vram(pressure_gb)
                await asyncio.sleep(0.5)

        try:
            # Submit sessions in reverse priority order
            low_result = await self.controller.request_session(
                session_id="real-low-priority",
                service_type="tts_hq"  # LOW priority
            )

            high_result = await self.controller.request_session(
                session_id="real-high-priority",
                service_type="llm"  # HIGH priority
            )

            critical_result = await self.controller.request_session(
                session_id="real-critical-priority",
                service_type="stt"  # CRITICAL priority
            )

            # All should succeed (queued or processing)
            assert low_result.success
            assert high_result.success
            assert critical_result.success

            # Check queue state
            queue_status = self.controller.session_queue.get_queue_status()

            logger.info("test.priority_queue_status",
                       queue_sessions=queue_status["queue_sessions"],
                       processing_sessions=queue_status["processing_sessions"])

            # If there are queued sessions, they should be in priority order
            if queue_status["queue_sessions"]:
                # Critical (stt) should be first, then high (llm), then low (tts_hq)
                service_types = [s["service_type"] for s in queue_status["queue_sessions"]]
                logger.info("test.queue_order", service_types=service_types)

                # Verify ordering (allowing for some sessions to be processing)
                stt_pos = next((i for i, s in enumerate(service_types) if s == "stt"), -1)
                llm_pos = next((i for i, s in enumerate(service_types) if s == "llm"), -1)
                tts_pos = next((i for i, s in enumerate(service_types) if s == "tts_hq"), -1)

                # STT should come before LLM if both are queued
                if stt_pos >= 0 and llm_pos >= 0:
                    assert stt_pos < llm_pos, f"STT position {stt_pos} should be before LLM {llm_pos}"

                # LLM should come before TTS if both are queued
                if llm_pos >= 0 and tts_pos >= 0:
                    assert llm_pos < tts_pos, f"LLM position {llm_pos} should be before TTS {tts_pos}"

        finally:
            self.vram_loader.free_vram()

    @pytest.mark.asyncio
    async def test_real_concurrent_session_processing(self):
        """Test multiple concurrent sessions with real system"""
        # Ensure clean state
        self.vram_loader.cleanup()
        await asyncio.sleep(0.5)

        # Submit multiple concurrent requests
        session_count = 5
        tasks = []

        for i in range(session_count):
            task = self.controller.request_session(
                session_id=f"real-concurrent-{i}",
                service_type="llm"
            )
            tasks.append(task)

        # Wait for all requests
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Verify results
        successful_results = []
        failed_results = []

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                failed_results.append((i, result))
            elif result.success:
                successful_results.append((i, result))
            else:
                failed_results.append((i, result))

        logger.info("test.concurrent_results",
                   successful=len(successful_results),
                   failed=len(failed_results),
                   total=session_count)

        # At least some should succeed
        assert len(successful_results) > 0, "No sessions succeeded"

        # Check system state
        system_status = self.controller.get_system_status()
        logger.info("test.system_status_after_concurrent",
                   capacity=system_status["capacity"],
                   queue=system_status["queue"])

    @pytest.mark.asyncio
    async def test_real_session_timeout(self):
        """Test session timeout with real system timing"""
        # Request session with very short timeout
        result = await self.controller.request_session(
            session_id="real-timeout-test",
            service_type="llm",
            timeout=0.5  # 500ms timeout
        )

        assert result.success  # Initial request should succeed

        # Wait for timeout to occur
        await asyncio.sleep(1.0)

        # Check if session was cleaned up
        status = self.controller.get_session_status("real-timeout-test")

        logger.info("test.timeout_status", status=status)

        # Session should either be not found or marked as timed out/rejected
        if status:
            assert status["state"] in ["rejected", "cancelled", "completed"]


class TestRealVRAMMonitoring:
    """Test VRAM monitoring with real GPU operations"""

    @pytest.fixture(autouse=True)
    def setup_vram_monitor(self):
        """Setup real VRAM monitor"""
        self.vram_monitor = get_vram_monitor()
        self.vram_loader = RealVRAMLoader()

        # Clean state
        self.vram_loader.cleanup()

        yield

        self.vram_loader.cleanup()

    def test_real_vram_status_reporting(self):
        """Test VRAM status with real GPU state"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA required for VRAM monitoring tests")

        # Get baseline status
        baseline_status = self.vram_monitor.get_all_gpu_status()
        assert len(baseline_status) > 0, "No GPUs detected"

        baseline_usage = baseline_status[0].allocated_gb

        # Allocate real VRAM
        allocation_gb = 2.0
        try:
            tensor = self.vram_loader.allocate_vram(allocation_gb)

            # Check updated status
            updated_status = self.vram_monitor.get_all_gpu_status()
            updated_usage = updated_status[0].allocated_gb

            # Should show increased usage
            usage_increase = updated_usage - baseline_usage
            assert usage_increase >= allocation_gb * 0.8, f"Usage increase {usage_increase:.2f}GB less than expected {allocation_gb}GB"

            logger.info("test.vram_monitoring",
                       baseline_gb=baseline_usage,
                       after_allocation_gb=updated_usage,
                       increase_gb=usage_increase)

        finally:
            self.vram_loader.free_vram()

    def test_real_vram_service_prediction(self):
        """Test service prediction with real VRAM state"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA required for service prediction tests")

        # Test with clean VRAM
        self.vram_loader.cleanup()

        services = ["stt", "llm", "tts_fast", "tts_hq"]
        clean_predictions = {}

        for service in services:
            prediction = self.vram_monitor.predict_can_handle_service(service)
            clean_predictions[service] = prediction

            logger.info("test.clean_vram_prediction",
                       service=service,
                       can_handle=prediction["can_handle"],
                       vram_required=prediction["vram_required_gb"])

        # STT should always be available (CPU only)
        assert clean_predictions["stt"]["can_handle"], "STT should always be available"
        assert clean_predictions["stt"]["vram_required_gb"] == 0, "STT should require 0 VRAM"

        # Create VRAM pressure and test again
        try:
            available_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            pressure_gb = available_gb * 0.85  # 85% usage

            self.vram_loader.allocate_vram(pressure_gb)

            pressure_predictions = {}
            for service in services:
                prediction = self.vram_monitor.predict_can_handle_service(service)
                pressure_predictions[service] = prediction

                logger.info("test.pressure_vram_prediction",
                           service=service,
                           can_handle=prediction["can_handle"],
                           reasoning=prediction["reasoning"])

            # STT should still be available under VRAM pressure
            assert pressure_predictions["stt"]["can_handle"], "STT should be available under VRAM pressure"

            # Some GPU services might be restricted
            logger.info("test.prediction_comparison",
                       clean=clean_predictions,
                       pressure=pressure_predictions)

        finally:
            self.vram_loader.free_vram()


class TestRealSystemIntegration:
    """Test full system integration with real components"""

    @pytest.fixture(autouse=True)
    async def setup_full_system(self):
        """Setup complete real system"""
        self.controller = get_session_controller()
        await self.controller.start()

        self.vram_loader = RealVRAMLoader()
        self.vram_loader.cleanup()

        yield

        self.vram_loader.cleanup()
        await self.controller.stop()

    @pytest.mark.asyncio
    async def test_real_system_health_check(self):
        """Test system health with real components"""
        health = self.controller.get_health_check()

        assert "healthy" in health
        assert "health_score" in health
        assert "issues" in health
        assert "metrics" in health

        assert isinstance(health["healthy"], bool)
        assert 0 <= health["health_score"] <= 100

        logger.info("test.system_health", health=health)

    @pytest.mark.asyncio
    async def test_real_service_availability(self):
        """Test service availability with real system state"""
        availability = self.controller.get_service_availability()

        expected_services = ["stt", "llm", "tts_fast", "tts_hq"]
        for service in expected_services:
            assert service in availability
            assert "available" in availability[service]
            assert "vram_required_gb" in availability[service]

            logger.info("test.service_availability",
                       service=service,
                       details=availability[service])

    @pytest.mark.asyncio
    async def test_real_system_under_load(self):
        """Test system behavior under real load"""
        # Create background VRAM pressure
        if torch.cuda.is_available():
            available_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            background_pressure = min(available_gb * 0.6, 10.0)  # 60% or 10GB, whichever is smaller

            try:
                self.vram_loader.allocate_vram(background_pressure)
                await asyncio.sleep(0.5)  # Allow monitoring to update
            except torch.cuda.OutOfMemoryError:
                background_pressure = available_gb * 0.3  # Reduce if OOM
                self.vram_loader.allocate_vram(background_pressure)
                await asyncio.sleep(0.5)

        try:
            # Submit multiple sessions
            session_results = []
            for i in range(8):  # More sessions than max concurrent
                result = await self.controller.request_session(
                    session_id=f"load-test-{i}",
                    service_type="llm"
                )
                session_results.append(result)

                # Small delay between requests
                await asyncio.sleep(0.1)

            # Check system state under load
            system_status = self.controller.get_system_status()

            logger.info("test.system_under_load",
                       submitted_sessions=len(session_results),
                       successful_sessions=len([r for r in session_results if r.success]),
                       queue_size=system_status["queue"]["queue_size"],
                       processing_count=system_status["queue"]["processing_count"],
                       capacity_utilization=system_status["capacity"]["utilization_percent"])

            # System should handle load gracefully
            successful_count = len([r for r in session_results if r.success])
            assert successful_count > 0, "No sessions were accepted"

            # Queue should be managing overflow
            if successful_count < len(session_results):
                # Some sessions should be queued if not all can be processed immediately
                total_active = system_status["queue"]["queue_size"] + system_status["queue"]["processing_count"]
                logger.info("test.load_management",
                           total_active_sessions=total_active,
                           submitted=len(session_results))

        finally:
            self.vram_loader.free_vram()


if __name__ == "__main__":
    # Run a basic real test
    async def run_basic_test():
        """Run basic real system test"""
        print("🧪 Running real system test (no mocks)...")

        # Test VRAM monitoring
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.device_count()} GPUs")

            vram_loader = RealVRAMLoader()
            try:
                # Test allocation
                print("📊 Testing real VRAM allocation...")
                tensor = vram_loader.allocate_vram(1.0)  # 1GB
                usage = vram_loader.get_current_usage_gb()
                print(f"✅ Allocated 1GB, current usage: {usage:.2f}GB")

            finally:
                vram_loader.cleanup()
                print("🧹 VRAM cleaned up")
        else:
            print("⚠️ CUDA not available, skipping VRAM tests")

        # Test session controller
        print("🎮 Testing real session controller...")
        controller = get_session_controller()
        await controller.start()

        try:
            result = await controller.request_session(
                session_id="basic-real-test",
                service_type="llm"
            )

            print(f"✅ Session request result: {result.success}")

            health = controller.get_health_check()
            print(f"✅ System health score: {health['health_score']}")

        finally:
            await controller.stop()
            print("🛑 Session controller stopped")

        print("🎉 Basic real system test completed!")

    # Run test
    asyncio.run(run_basic_test())