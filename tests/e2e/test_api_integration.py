"""
End-to-End API Integration Tests - Real FastAPI Testing

TDD for session control API endpoints with real HTTP requests, no mocks.
Tests the complete API stack from HTTP request to system response.
"""

import asyncio
import pytest
import time
import sys
import os
from typing import Dict, Any

import httpx
import torch
from fastapi.testclient import TestClient

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from avatar.main import app
from avatar.core.session_controller import get_session_controller
from avatar.api.session_control import router


class TestRealAPIEndpoints:
    """Test API endpoints with real FastAPI server"""

    @pytest.fixture(autouse=True)
    async def setup_api_server(self):
        """Setup real FastAPI server for testing"""
        # Add session control router to main app
        app.include_router(router, prefix="/api/v1")

        self.client = TestClient(app)

        # Initialize session controller
        self.controller = get_session_controller()
        await self.controller.start()

        yield

        await self.controller.stop()

    def test_real_session_request_endpoint(self):
        """Test POST /api/v1/sessions/request with real system"""
        # Test session request
        request_data = {
            "session_id": "api-test-001",
            "service_type": "llm",
            "timeout": 30.0,
            "priority_boost": False
        }

        response = self.client.post(
            "/api/v1/sessions/request",
            json=request_data
        )

        # Should succeed
        assert response.status_code == 200, f"API request failed: {response.text}"

        data = response.json()
        assert data["success"] is True
        assert data["session_id"] == "api-test-001"
        assert "message" in data
        assert "timestamp" in data

        print(f"✅ Session request API response: {data}")

    def test_real_session_status_endpoint(self):
        """Test GET /api/v1/sessions/status/{session_id} with real system"""
        # First create a session
        request_data = {
            "session_id": "api-status-test",
            "service_type": "llm"
        }

        create_response = self.client.post(
            "/api/v1/sessions/request",
            json=request_data
        )
        assert create_response.status_code == 200

        # Now get status
        status_response = self.client.get(
            "/api/v1/sessions/status/api-status-test"
        )

        assert status_response.status_code == 200
        status_data = status_response.json()

        assert "session_status" in status_data
        assert status_data["session_status"]["session_id"] == "api-status-test"
        assert "state" in status_data["session_status"]

        print(f"✅ Session status API response: {status_data}")

    def test_real_queue_status_endpoint(self):
        """Test GET /api/v1/sessions/queue with real system"""
        response = self.client.get("/api/v1/sessions/queue")

        assert response.status_code == 200
        data = response.json()

        required_fields = [
            "queue_size", "processing_count", "max_concurrent",
            "max_queue_size", "utilization_percent", "statistics"
        ]

        for field in required_fields:
            assert field in data, f"Missing field: {field}"

        assert isinstance(data["queue_size"], int)
        assert isinstance(data["processing_count"], int)
        assert 0 <= data["utilization_percent"] <= 100

        print(f"✅ Queue status API response: {data}")

    def test_real_system_status_endpoint(self):
        """Test GET /api/v1/sessions/system with real system"""
        response = self.client.get("/api/v1/sessions/system")

        assert response.status_code == 200
        data = response.json()

        required_sections = ["timestamp", "queue", "vram", "capacity", "performance"]
        for section in required_sections:
            assert section in data, f"Missing section: {section}"

        # Validate data structure
        assert isinstance(data["timestamp"], (int, float))
        assert data["timestamp"] > 0

        assert "queue_size" in data["queue"]
        assert "processing_count" in data["queue"]

        if torch.cuda.is_available():
            assert "gpus" in data["vram"]
            assert len(data["vram"]["gpus"]) > 0

        print(f"✅ System status API response keys: {list(data.keys())}")

    def test_real_health_endpoint(self):
        """Test GET /api/v1/sessions/health with real system"""
        response = self.client.get("/api/v1/sessions/health")

        assert response.status_code == 200
        data = response.json()

        assert "healthy" in data
        assert "health_score" in data
        assert "issues" in data
        assert "metrics" in data

        assert isinstance(data["healthy"], bool)
        assert 0 <= data["health_score"] <= 100
        assert isinstance(data["issues"], list)
        assert isinstance(data["metrics"], dict)

        print(f"✅ Health check: healthy={data['healthy']}, score={data['health_score']}")

    def test_real_service_availability_endpoint(self):
        """Test GET /api/v1/sessions/availability with real system"""
        response = self.client.get("/api/v1/sessions/availability")

        assert response.status_code == 200
        data = response.json()

        assert "services" in data
        assert "overall_capacity" in data
        assert "vram_summary" in data

        # Check all expected services
        expected_services = ["stt", "llm", "tts_fast", "tts_hq"]
        for service in expected_services:
            assert service in data["services"]
            service_data = data["services"][service]

            assert "available" in service_data
            assert "vram_required_gb" in service_data
            assert isinstance(service_data["available"], bool)
            assert isinstance(service_data["vram_required_gb"], (int, float))

        print(f"✅ Service availability: {list(data['services'].keys())}")

    def test_real_metrics_endpoint(self):
        """Test GET /api/v1/sessions/metrics for Prometheus format"""
        response = self.client.get("/api/v1/sessions/metrics")

        assert response.status_code == 200

        # Should return plain text metrics
        metrics_text = response.text
        assert "avatar_queue_size" in metrics_text
        assert "avatar_processing_count" in metrics_text
        assert "avatar_utilization_percent" in metrics_text

        # Parse some metrics
        lines = metrics_text.strip().split('\n')
        metrics = {}
        for line in lines:
            if ' ' in line:
                name, value = line.split(' ', 1)
                metrics[name] = value

        assert "avatar_queue_size" in metrics
        assert "avatar_processing_count" in metrics

        print(f"✅ Metrics format: {len(lines)} metrics exported")

    def test_real_session_cancellation(self):
        """Test DELETE /api/v1/sessions/cancel/{session_id} with real system"""
        # Create a session to cancel
        request_data = {
            "session_id": "cancel-test-session",
            "service_type": "llm"
        }

        create_response = self.client.post(
            "/api/v1/sessions/request",
            json=request_data
        )
        assert create_response.status_code == 200

        # Try to cancel it
        cancel_response = self.client.delete(
            "/api/v1/sessions/cancel/cancel-test-session"
        )

        # Should succeed (200) or fail (404) depending on timing
        assert cancel_response.status_code in [200, 404]

        if cancel_response.status_code == 200:
            data = cancel_response.json()
            assert data["success"] is True
            assert "cancel-test-session" in data["message"]

        print(f"✅ Cancel response: {cancel_response.status_code}")


class TestRealAPIErrorHandling:
    """Test API error handling with real system"""

    @pytest.fixture(autouse=True)
    async def setup_api_server(self):
        """Setup API server"""
        app.include_router(router, prefix="/api/v1")
        self.client = TestClient(app)

        self.controller = get_session_controller()
        await self.controller.start()

        yield

        await self.controller.stop()

    def test_real_invalid_session_request(self):
        """Test invalid session request data"""
        # Invalid service type
        invalid_data = {
            "session_id": "invalid-test",
            "service_type": "invalid_service",
            "timeout": 30.0
        }

        response = self.client.post(
            "/api/v1/sessions/request",
            json=invalid_data
        )

        assert response.status_code == 422  # Validation error

    def test_real_missing_session_status(self):
        """Test getting status for non-existent session"""
        response = self.client.get(
            "/api/v1/sessions/status/non-existent-session"
        )

        assert response.status_code == 404

    def test_real_malformed_request_data(self):
        """Test malformed JSON request"""
        response = self.client.post(
            "/api/v1/sessions/request",
            data="invalid json"  # Not JSON
        )

        assert response.status_code == 422


class TestRealAPIWithVRAMPressure:
    """Test API behavior under real VRAM pressure"""

    @pytest.fixture(autouse=True)
    async def setup_api_server(self):
        """Setup API server"""
        app.include_router(router, prefix="/api/v1")
        self.client = TestClient(app)

        self.controller = get_session_controller()
        await self.controller.start()

        # VRAM pressure setup
        self.vram_loader = None
        if torch.cuda.is_available():
            sys.path.insert(0, os.path.dirname(__file__))
            from test_real_session_control import RealVRAMLoader
            self.vram_loader = RealVRAMLoader()

        yield

        if self.vram_loader:
            self.vram_loader.cleanup()
        await self.controller.stop()

    def test_real_api_under_vram_pressure(self):
        """Test API behavior when VRAM is under pressure"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA required for VRAM pressure testing")

        # Create VRAM pressure
        available_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        pressure_gb = available_gb * 0.8  # 80% pressure

        try:
            self.vram_loader.allocate_vram(pressure_gb)
            time.sleep(0.5)  # Allow monitoring to update

            # Make multiple session requests under pressure
            session_results = []
            for i in range(5):
                request_data = {
                    "session_id": f"pressure-test-{i}",
                    "service_type": "llm"
                }

                response = self.client.post(
                    "/api/v1/sessions/request",
                    json=request_data
                )

                session_results.append({
                    "status_code": response.status_code,
                    "response": response.json() if response.status_code == 200 else None
                })

            # Check system status under pressure
            system_response = self.client.get("/api/v1/sessions/system")
            assert system_response.status_code == 200

            system_data = system_response.json()

            print(f"✅ Under VRAM pressure:")
            print(f"   - Sessions requested: {len(session_results)}")
            print(f"   - Successful requests: {len([r for r in session_results if r['status_code'] == 200])}")
            print(f"   - Queue size: {system_data['queue']['queue_size']}")
            print(f"   - Processing count: {system_data['queue']['processing_count']}")

            # At least some requests should be handled (success or queued)
            successful_requests = [r for r in session_results if r['status_code'] == 200]
            assert len(successful_requests) > 0, "No requests succeeded under VRAM pressure"

        finally:
            if self.vram_loader:
                self.vram_loader.free_vram()

    def test_real_health_degradation_under_pressure(self):
        """Test health score degradation under VRAM pressure"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA required for VRAM pressure testing")

        # Get baseline health
        baseline_response = self.client.get("/api/v1/sessions/health")
        assert baseline_response.status_code == 200
        baseline_health = baseline_response.json()

        # Create VRAM pressure
        available_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        pressure_gb = available_gb * 0.85  # 85% pressure

        try:
            self.vram_loader.allocate_vram(pressure_gb)
            time.sleep(0.5)  # Allow monitoring to update

            # Check health under pressure
            pressure_response = self.client.get("/api/v1/sessions/health")
            assert pressure_response.status_code == 200
            pressure_health = pressure_response.json()

            print(f"✅ Health comparison:")
            print(f"   - Baseline score: {baseline_health['health_score']}")
            print(f"   - Under pressure score: {pressure_health['health_score']}")
            print(f"   - Issues under pressure: {len(pressure_health['issues'])}")

            # Health score should be lower or there should be issues
            assert (pressure_health['health_score'] < baseline_health['health_score'] or
                    len(pressure_health['issues']) > len(baseline_health['issues'])), \
                   "Health should degrade under VRAM pressure"

        finally:
            if self.vram_loader:
                self.vram_loader.free_vram()


class TestRealAPIConcurrency:
    """Test API with concurrent requests"""

    @pytest.fixture(autouse=True)
    async def setup_api_server(self):
        """Setup API server"""
        app.include_router(router, prefix="/api/v1")
        self.client = TestClient(app)

        self.controller = get_session_controller()
        await self.controller.start()

        yield

        await self.controller.stop()

    def test_real_concurrent_api_requests(self):
        """Test multiple concurrent API requests"""
        import threading
        import queue

        results_queue = queue.Queue()

        def make_request(session_id: str):
            """Make a session request"""
            try:
                request_data = {
                    "session_id": session_id,
                    "service_type": "llm"
                }

                response = self.client.post(
                    "/api/v1/sessions/request",
                    json=request_data
                )

                results_queue.put({
                    "session_id": session_id,
                    "status_code": response.status_code,
                    "success": response.status_code == 200,
                    "response": response.json() if response.status_code == 200 else None
                })

            except Exception as e:
                results_queue.put({
                    "session_id": session_id,
                    "error": str(e),
                    "success": False
                })

        # Create threads for concurrent requests
        threads = []
        for i in range(8):  # 8 concurrent requests
            thread = threading.Thread(
                target=make_request,
                args=[f"concurrent-api-{i}"]
            )
            threads.append(thread)

        # Start all threads
        for thread in threads:
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Collect results
        results = []
        while not results_queue.empty():
            results.append(results_queue.get())

        print(f"✅ Concurrent API test results:")
        print(f"   - Total requests: {len(results)}")
        print(f"   - Successful: {len([r for r in results if r['success']])}")
        print(f"   - Failed: {len([r for r in results if not r['success']])}")

        # Should handle concurrent requests gracefully
        successful_count = len([r for r in results if r['success']])
        assert successful_count > 0, "No concurrent requests succeeded"

        # Check final system state
        system_response = self.client.get("/api/v1/sessions/system")
        assert system_response.status_code == 200


if __name__ == "__main__":
    # Run basic API test
    print("🧪 Running basic API integration test...")

    app.include_router(router, prefix="/api/v1")
    client = TestClient(app)

    # Test basic endpoint
    response = client.get("/api/v1/sessions/health")
    print(f"Health endpoint: {response.status_code}")

    if response.status_code == 200:
        print("✅ Basic API integration test passed!")
    else:
        print(f"❌ Basic API test failed: {response.text}")