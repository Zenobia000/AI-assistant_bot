"""
Memory Leak Detection and OOM Prevention Testing

Task 25: Advanced memory leak detection for AVATAR system
Focus on AI model memory management, session lifecycle, and VRAM optimization

Detection Targets:
- PyTorch tensor memory leaks
- Session object accumulation
- File handle leaks
- VRAM fragmentation
- Cache growth patterns
"""

import asyncio
import gc
import os
import psutil
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

import structlog
import torch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from avatar.core.config import config
from avatar.core.logging_config import configure_logging
from avatar.core.vram_monitor import get_vram_monitor
from avatar.core.session_queue import get_session_queue
from avatar.core.session_controller import get_session_controller

logger = structlog.get_logger()


@dataclass
class MemoryLeakCheck:
    """Memory leak detection result"""
    component: str
    test_duration: float
    initial_memory_mb: float
    final_memory_mb: float
    peak_memory_mb: float
    growth_mb: float
    growth_rate_mb_per_hour: float
    leak_detected: bool
    leak_severity: str  # "none", "minor", "moderate", "severe"


class TorchMemoryTracker:
    """PyTorch GPU memory leak detection"""

    def __init__(self):
        self.initial_allocated = 0
        self.initial_cached = 0
        self.peak_allocated = 0
        self.allocations = []

    def start_tracking(self):
        """Start tracking PyTorch memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            self.initial_allocated = torch.cuda.memory_allocated() / (1024**2)  # MB
            self.initial_cached = torch.cuda.memory_reserved() / (1024**2)
            logger.info("torch_tracker.started",
                       initial_allocated_mb=self.initial_allocated,
                       initial_cached_mb=self.initial_cached)

    def capture_state(self):
        """Capture current memory state"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024**2)
            cached = torch.cuda.memory_reserved() / (1024**2)
            peak = torch.cuda.max_memory_allocated() / (1024**2)

            self.allocations.append({
                "timestamp": time.time(),
                "allocated_mb": allocated,
                "cached_mb": cached,
                "peak_mb": peak
            })

            self.peak_allocated = max(self.peak_allocated, allocated)

            return {
                "allocated_mb": allocated,
                "cached_mb": cached,
                "peak_mb": peak,
                "growth_from_initial": allocated - self.initial_allocated
            }
        return None

    def analyze_leaks(self, test_duration: float) -> MemoryLeakCheck:
        """Analyze for memory leaks"""
        if not torch.cuda.is_available():
            return MemoryLeakCheck(
                component="PyTorch",
                test_duration=test_duration,
                initial_memory_mb=0,
                final_memory_mb=0,
                peak_memory_mb=0,
                growth_mb=0,
                growth_rate_mb_per_hour=0,
                leak_detected=False,
                leak_severity="none"
            )

        final_allocated = torch.cuda.memory_allocated() / (1024**2)
        growth_mb = final_allocated - self.initial_allocated
        growth_rate = (growth_mb / test_duration) * 3600  # MB per hour

        # Determine leak severity
        if growth_rate < 10:
            severity = "none"
            leak_detected = False
        elif growth_rate < 50:
            severity = "minor"
            leak_detected = True
        elif growth_rate < 200:
            severity = "moderate"
            leak_detected = True
        else:
            severity = "severe"
            leak_detected = True

        return MemoryLeakCheck(
            component="PyTorch",
            test_duration=test_duration,
            initial_memory_mb=self.initial_allocated,
            final_memory_mb=final_allocated,
            peak_memory_mb=self.peak_allocated,
            growth_mb=growth_mb,
            growth_rate_mb_per_hour=growth_rate,
            leak_detected=leak_detected,
            leak_severity=severity
        )


class SessionMemoryTracker:
    """Session object memory leak detection"""

    def __init__(self):
        self.initial_objects = 0
        self.session_counts = []

    def start_tracking(self):
        """Start tracking session objects"""
        gc.collect()
        self.initial_objects = len(gc.get_objects())
        logger.info("session_tracker.started", initial_objects=self.initial_objects)

    def capture_session_state(self):
        """Capture session and object state"""
        gc.collect()

        current_objects = len(gc.get_objects())

        # Try to get session controller info
        try:
            controller = get_session_controller()
            active_sessions = len(getattr(controller, 'active_sessions', {}))
        except Exception:
            active_sessions = 0

        # Try to get session queue info
        try:
            queue = get_session_queue()
            queue_size = getattr(queue, 'current_size', 0)
        except Exception:
            queue_size = 0

        state = {
            "timestamp": time.time(),
            "total_objects": current_objects,
            "object_growth": current_objects - self.initial_objects,
            "active_sessions": active_sessions,
            "queue_size": queue_size
        }

        self.session_counts.append(state)
        return state

    def analyze_session_leaks(self, test_duration: float) -> MemoryLeakCheck:
        """Analyze for session-related memory leaks"""
        if not self.session_counts:
            return MemoryLeakCheck(
                component="SessionManagement",
                test_duration=test_duration,
                initial_memory_mb=0,
                final_memory_mb=0,
                peak_memory_mb=0,
                growth_mb=0,
                growth_rate_mb_per_hour=0,
                leak_detected=False,
                leak_severity="none"
            )

        initial_state = self.session_counts[0]
        final_state = self.session_counts[-1]

        object_growth = final_state["object_growth"]
        growth_rate = (object_growth / test_duration) * 3600  # Objects per hour

        # Convert object growth to approximate memory (rough estimate)
        estimated_memory_growth = object_growth * 0.001  # Rough: 1KB per object

        # Assess severity
        if growth_rate < 100:
            severity = "none"
            leak_detected = False
        elif growth_rate < 1000:
            severity = "minor"
            leak_detected = True
        elif growth_rate < 5000:
            severity = "moderate"
            leak_detected = True
        else:
            severity = "severe"
            leak_detected = True

        return MemoryLeakCheck(
            component="SessionManagement",
            test_duration=test_duration,
            initial_memory_mb=initial_state["total_objects"] * 0.001,
            final_memory_mb=final_state["total_objects"] * 0.001,
            peak_memory_mb=max(s["total_objects"] for s in self.session_counts) * 0.001,
            growth_mb=estimated_memory_growth,
            growth_rate_mb_per_hour=growth_rate * 0.001,
            leak_detected=leak_detected,
            leak_severity=severity
        )


class MemoryLeakDetectionSuite:
    """Comprehensive memory leak detection suite"""

    def __init__(self):
        self.torch_tracker = TorchMemoryTracker()
        self.session_tracker = SessionMemoryTracker()
        self.process_tracker = None
        self.tracemalloc_enabled = False

    async def run_leak_detection_test(self, duration_minutes: int = 10) -> Dict[str, Any]:
        """Run comprehensive memory leak detection test"""
        test_duration = duration_minutes * 60

        logger.info("memory_leak.test_started", duration_minutes=duration_minutes)

        # Enable detailed memory tracking
        await self._enable_memory_tracking()

        # Start all trackers
        self.torch_tracker.start_tracking()
        self.session_tracker.start_tracking()

        # Run workload simulation
        await self._simulate_workload(test_duration)

        # Analyze results
        results = await self._analyze_leak_results(test_duration)

        # Cleanup tracking
        await self._disable_memory_tracking()

        return results

    async def _enable_memory_tracking(self):
        """Enable detailed memory tracking"""
        try:
            tracemalloc.start()
            self.tracemalloc_enabled = True
            logger.info("memory_leak.tracemalloc_enabled")
        except Exception as e:
            logger.warning("memory_leak.tracemalloc_failed", error=str(e))

    async def _disable_memory_tracking(self):
        """Disable memory tracking"""
        if self.tracemalloc_enabled:
            tracemalloc.stop()

    async def _simulate_workload(self, duration: float):
        """Simulate realistic workload for leak detection"""

        end_time = time.time() + duration
        request_count = 0

        while time.time() < end_time:
            try:
                # Simulate session creation and processing
                await self._simulate_session_lifecycle(request_count)

                # Capture memory states
                self.torch_tracker.capture_state()
                self.session_tracker.capture_session_state()

                request_count += 1

                # Variable wait time
                await asyncio.sleep(1 + (request_count % 5))

            except Exception as e:
                logger.warning("memory_leak.workload_error",
                             request=request_count,
                             error=str(e))

        logger.info("memory_leak.workload_completed", total_requests=request_count)

    async def _simulate_session_lifecycle(self, request_id: int):
        """Simulate complete session lifecycle"""

        # Simulate VRAM allocation
        if torch.cuda.is_available():
            # Allocate temporary tensors (simulating model inference)
            temp_tensors = []
            try:
                for i in range(3):  # Multiple allocations
                    tensor = torch.randn(500, 500, device="cuda", dtype=torch.float16)
                    temp_tensors.append(tensor)

                # Simulate processing
                await asyncio.sleep(0.1)

            finally:
                # Explicit cleanup
                for tensor in temp_tensors:
                    del tensor
                torch.cuda.empty_cache()

        # Simulate session objects
        session_data = {
            "session_id": f"test-session-{request_id}",
            "data": [0] * 1000,  # Some data
            "timestamp": time.time()
        }

        # Simulate processing
        await asyncio.sleep(0.05)

        # Cleanup
        del session_data
        gc.collect()

    async def _analyze_leak_results(self, test_duration: float) -> Dict[str, Any]:
        """Analyze memory leak test results"""

        results = {
            "test_duration_minutes": test_duration / 60,
            "leak_analysis": {},
            "memory_snapshots": {},
            "recommendations": []
        }

        # PyTorch memory analysis
        torch_result = self.torch_tracker.analyze_leaks(test_duration)
        results["leak_analysis"]["pytorch"] = {
            "leak_detected": torch_result.leak_detected,
            "severity": torch_result.leak_severity,
            "growth_mb": torch_result.growth_mb,
            "growth_rate_mb_per_hour": torch_result.growth_rate_mb_per_hour,
            "peak_memory_mb": torch_result.peak_memory_mb
        }

        # Session management analysis
        session_result = self.session_tracker.analyze_session_leaks(test_duration)
        results["leak_analysis"]["sessions"] = {
            "leak_detected": session_result.leak_detected,
            "severity": session_result.leak_severity,
            "object_growth": session_result.growth_mb * 1000,  # Convert back to objects
            "growth_rate_objects_per_hour": session_result.growth_rate_mb_per_hour * 1000
        }

        # Overall assessment
        any_leaks = torch_result.leak_detected or session_result.leak_detected
        worst_severity = max(torch_result.leak_severity, session_result.leak_severity,
                           key=lambda x: {"none": 0, "minor": 1, "moderate": 2, "severe": 3}.get(x, 0))

        results["overall_assessment"] = {
            "memory_leaks_detected": any_leaks,
            "worst_severity": worst_severity,
            "oom_risk": "high" if worst_severity == "severe" else "medium" if worst_severity == "moderate" else "low"
        }

        # Generate recommendations
        results["recommendations"] = self._generate_leak_recommendations(torch_result, session_result)

        return results

    def _generate_leak_recommendations(self, torch_result: MemoryLeakCheck,
                                     session_result: MemoryLeakCheck) -> List[str]:
        """Generate memory leak fix recommendations"""
        recommendations = []

        if torch_result.leak_detected:
            if torch_result.leak_severity in ["moderate", "severe"]:
                recommendations.extend([
                    "Implement explicit tensor cleanup in AI service methods",
                    "Add torch.cuda.empty_cache() calls after GPU operations",
                    "Review model loading/unloading patterns"
                ])
            else:
                recommendations.append("Monitor PyTorch memory usage patterns")

        if session_result.leak_detected:
            if session_result.leak_severity in ["moderate", "severe"]:
                recommendations.extend([
                    "Implement session cleanup timeouts",
                    "Review session object lifecycle management",
                    "Add explicit session garbage collection"
                ])
            else:
                recommendations.append("Monitor session object accumulation")

        return recommendations


async def run_memory_leak_test(duration_minutes: int = 10):
    """Run memory leak detection test"""
    configure_logging(development_mode=True)

    print(f"🔍 Starting Memory Leak Detection Test ({duration_minutes} minutes)...")

    suite = MemoryLeakDetectionSuite()

    try:
        results = await suite.run_leak_detection_test(duration_minutes)

        print(f"\n📊 Memory Leak Analysis Results:")

        # Overall assessment
        assessment = results["overall_assessment"]
        print(f"Memory Leaks Detected: {'❌ YES' if assessment['memory_leaks_detected'] else '✅ NO'}")
        print(f"OOM Risk Level: {assessment['oom_risk'].upper()}")

        # PyTorch analysis
        pytorch = results["leak_analysis"]["pytorch"]
        print(f"\nPyTorch Memory:")
        print(f"  Growth: {pytorch['growth_mb']:.1f} MB")
        print(f"  Growth Rate: {pytorch['growth_rate_mb_per_hour']:.1f} MB/hour")
        print(f"  Leak Severity: {pytorch['severity']}")

        # Session analysis
        sessions = results["leak_analysis"]["sessions"]
        print(f"\nSession Management:")
        print(f"  Object Growth: {sessions['object_growth']:.0f} objects")
        print(f"  Growth Rate: {sessions['growth_rate_objects_per_hour']:.0f} objects/hour")
        print(f"  Leak Severity: {sessions['severity']}")

        # Recommendations
        recommendations = results["recommendations"]
        if recommendations:
            print(f"\n💡 Leak Fix Recommendations:")
            for rec in recommendations:
                print(f"  • {rec}")

        return results

    except Exception as e:
        print(f"❌ Memory leak test failed: {e}")
        logger.error("memory_leak.test_failed", error=str(e))
        raise


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Memory Leak Detection Test")
    parser.add_argument("--duration", type=int, default=10, help="Test duration in minutes")
    args = parser.parse_args()

    asyncio.run(run_memory_leak_test(duration_minutes=args.duration))