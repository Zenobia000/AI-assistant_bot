"""
Standalone API Integration Test - Real FastAPI without full app complexity

Direct testing of session control API endpoints in isolation.
"""

import asyncio
import pytest
import time
import sys
import os
from typing import Dict, Any

import torch
from fastapi import FastAPI
from fastapi.testclient import TestClient

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from avatar.api.session_control import router
from avatar.core.session_controller import get_session_controller


class TestStandaloneSessionAPI:
    """Test session control API in standalone mode"""

    @pytest.fixture(autouse=True)
    async def setup_standalone_api(self):
        """Setup minimal FastAPI app with session control"""
        # Create minimal FastAPI app
        self.app = FastAPI(title="AVATAR Session Control Test")
        self.app.include_router(router)

        self.client = TestClient(self.app)

        # Initialize session controller
        self.controller = get_session_controller()
        await self.controller.start()

        yield

        await self.controller.stop()

    def test_real_health_endpoint(self):
        """Test health endpoint with real system"""
        response = self.client.get("/api/v1/sessions/health")

        assert response.status_code == 200
        data = response.json()

        assert "healthy" in data
        assert "health_score" in data
        assert isinstance(data["healthy"], bool)
        assert 0 <= data["health_score"] <= 100

        print(f"✅ Health: {data['healthy']}, Score: {data['health_score']}")

    def test_real_queue_status(self):
        """Test queue status endpoint"""
        response = self.client.get("/api/v1/sessions/queue")

        assert response.status_code == 200
        data = response.json()

        assert "queue_size" in data
        assert "processing_count" in data
        assert "max_concurrent" in data

        print(f"✅ Queue: {data['queue_size']} queued, {data['processing_count']} processing")

    def test_real_session_request(self):
        """Test session request endpoint"""
        request_data = {
            "session_id": "standalone-test-001",
            "service_type": "llm",
            "timeout": 30.0
        }

        response = self.client.post("/api/v1/sessions/request", json=request_data)

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert data["session_id"] == "standalone-test-001"

        print(f"✅ Session request: {data['success']}")

    def test_real_system_status(self):
        """Test system status endpoint"""
        response = self.client.get("/api/v1/sessions/system")

        assert response.status_code == 200
        data = response.json()

        assert "timestamp" in data
        assert "queue" in data
        assert "capacity" in data

        print(f"✅ System status: {list(data.keys())}")

    def test_real_service_availability(self):
        """Test service availability endpoint"""
        response = self.client.get("/api/v1/sessions/availability")

        assert response.status_code == 200
        data = response.json()

        assert "services" in data
        services = data["services"]

        for service_type in ["stt", "llm", "tts_fast", "tts_hq"]:
            assert service_type in services
            assert "available" in services[service_type]

        print(f"✅ Services: {list(services.keys())}")

    @pytest.mark.asyncio
    async def test_real_concurrent_requests(self):
        """Test multiple concurrent requests"""
        import asyncio
        from concurrent.futures import ThreadPoolExecutor

        def make_request(session_id: str):
            request_data = {
                "session_id": session_id,
                "service_type": "llm"
            }
            response = self.client.post("/api/v1/sessions/request", json=request_data)
            return response.status_code == 200

        # Run concurrent requests
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            for i in range(8):
                future = executor.submit(make_request, f"concurrent-{i}")
                futures.append(future)

            results = [f.result() for f in futures]

        successful = sum(results)
        print(f"✅ Concurrent: {successful}/{len(results)} successful")
        assert successful > 0


class TestRealVRAMPressureAPI:
    """Test API under real VRAM pressure"""

    @pytest.fixture(autouse=True)
    async def setup_with_vram_loader(self):
        """Setup API with VRAM loading capability"""
        self.app = FastAPI(title="AVATAR VRAM Pressure Test")
        self.app.include_router(router)
        self.client = TestClient(self.app)

        self.controller = get_session_controller()
        await self.controller.start()

        # VRAM loader
        if torch.cuda.is_available():
            sys.path.insert(0, os.path.dirname(__file__))
            from test_real_session_control import RealVRAMLoader
            self.vram_loader = RealVRAMLoader()
        else:
            self.vram_loader = None

        yield

        if self.vram_loader:
            self.vram_loader.cleanup()
        await self.controller.stop()

    def test_real_api_vram_pressure_response(self):
        """Test API response under VRAM pressure"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA required")

        # Baseline test
        baseline_response = self.client.get("/api/v1/sessions/health")
        baseline_health = baseline_response.json()

        # Create VRAM pressure
        available_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        pressure_gb = min(available_gb * 0.8, 15.0)  # 80% or 15GB max

        try:
            self.vram_loader.allocate_vram(pressure_gb)
            time.sleep(1.0)  # Allow monitoring to detect

            # Test under pressure
            pressure_response = self.client.get("/api/v1/sessions/health")
            pressure_health = pressure_response.json()

            # Test session request under pressure
            request_data = {
                "session_id": "pressure-api-test",
                "service_type": "llm"
            }
            session_response = self.client.post("/api/v1/sessions/request", json=request_data)

            print(f"✅ Pressure test:")
            print(f"   Baseline health: {baseline_health['health_score']}")
            print(f"   Pressure health: {pressure_health['health_score']}")
            print(f"   Session request: {session_response.status_code}")

            # System should respond (may degrade gracefully)
            assert pressure_response.status_code == 200
            assert session_response.status_code in [200, 503]  # Success or service unavailable

        finally:
            self.vram_loader.free_vram()


def test_basic_integration():
    """Standalone basic integration test"""
    print("🧪 Running standalone API integration test...")

    # Create minimal app
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    # Test health endpoint
    response = client.get("/api/v1/sessions/health")
    print(f"Health status: {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        print(f"Health score: {data.get('health_score', 'unknown')}")
        print("✅ Standalone API test passed!")
    else:
        print(f"❌ Health check failed: {response.text}")

    # Test availability
    avail_response = client.get("/api/v1/sessions/availability")
    print(f"Availability status: {avail_response.status_code}")

    if avail_response.status_code == 200:
        avail_data = avail_response.json()
        print(f"Services available: {list(avail_data.get('services', {}).keys())}")


if __name__ == "__main__":
    test_basic_integration()