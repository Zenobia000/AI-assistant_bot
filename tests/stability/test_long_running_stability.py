"""
Long-Running Stability Test

Task 25: 2-hour 5-concurrent stress test with OOM prevention
Linus principle: "Stability problems are usually resource management problems"

Test Requirements:
- Duration: 2 hours continuous operation
- Concurrency: 5 simultaneous sessions
- Zero OOM (Out of Memory) events
- Memory usage stability over time
- VRAM management validation
"""

import asyncio
import gc
import os
import psutil
import time
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from collections import deque

import structlog
import torch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from avatar.core.config import config
from avatar.core.logging_config import configure_logging
from avatar.core.vram_monitor import get_vram_monitor
from avatar.core.error_handling import get_error_handler

logger = structlog.get_logger()


@dataclass
class ResourceSnapshot:
    """System resource snapshot at a point in time"""
    timestamp: float

    # Memory metrics
    ram_used_gb: float
    ram_available_gb: float
    ram_percent: float

    # GPU metrics
    vram_used_gb: float
    vram_total_gb: float
    vram_percent: float
    gpu_temperature: float = 0.0

    # Process metrics
    process_ram_gb: float = 0.0
    process_threads: int = 0
    process_fds: int = 0  # File descriptors

    # Session metrics
    active_sessions: int = 0
    session_queue_size: int = 0

    # Error metrics
    error_count: int = 0
    warning_count: int = 0


@dataclass
class StabilityTestConfig:
    """Configuration for stability testing"""
    # Test duration
    duration_hours: float = 2.0          # 2 hours
    concurrent_sessions: int = 5         # 5 concurrent sessions

    # Monitoring intervals
    resource_check_interval: int = 30    # seconds
    detailed_check_interval: int = 300   # 5 minutes

    # Safety thresholds
    max_ram_percent: float = 85.0        # 85% RAM usage
    max_vram_percent: float = 90.0       # 90% VRAM usage
    max_temperature_c: float = 85.0      # 85°C GPU temp

    # Request patterns
    request_interval_min: float = 5.0    # 5 seconds between requests
    request_interval_max: float = 30.0   # 30 seconds between requests

    # OOM detection
    memory_growth_threshold_mb: float = 100.0  # 100MB growth per hour
    consecutive_warnings: int = 5        # Consecutive warnings before abort


class SystemResourceMonitor:
    """Real-time system resource monitoring"""

    def __init__(self):
        self.snapshots: deque = deque(maxlen=1000)  # Keep last 1000 snapshots
        self.vram_monitor = None
        self.process = psutil.Process()

    async def initialize(self):
        """Initialize monitoring components"""
        self.vram_monitor = get_vram_monitor()
        logger.info("stability.monitor_initialized")

    def capture_snapshot(self, active_sessions: int = 0, session_queue_size: int = 0) -> ResourceSnapshot:
        """Capture current resource state"""

        # System memory
        memory = psutil.virtual_memory()

        # GPU memory
        vram_info = {"used_gb": 0, "total_gb": 0, "percent": 0, "temperature": 0}
        if torch.cuda.is_available():
            try:
                device = torch.cuda.current_device()
                vram_info["used_gb"] = torch.cuda.memory_allocated(device) / (1024**3)
                vram_info["total_gb"] = torch.cuda.get_device_properties(device).total_memory / (1024**3)
                vram_info["percent"] = (vram_info["used_gb"] / vram_info["total_gb"]) * 100

                # GPU temperature (if available)
                if hasattr(torch.cuda, 'temperature'):
                    vram_info["temperature"] = torch.cuda.temperature(device)
            except Exception:
                pass

        # Process metrics
        try:
            process_memory = self.process.memory_info()
            process_ram_gb = process_memory.rss / (1024**3)
            process_threads = self.process.num_threads()
            process_fds = self.process.num_fds() if hasattr(self.process, 'num_fds') else 0
        except Exception:
            process_ram_gb = 0
            process_threads = 0
            process_fds = 0

        # Error metrics
        error_handler = get_error_handler()
        recent_errors = len([e for e in error_handler.recent_errors
                           if time.time() - e.timestamp < 300])  # Last 5 minutes

        snapshot = ResourceSnapshot(
            timestamp=time.time(),
            ram_used_gb=memory.used / (1024**3),
            ram_available_gb=memory.available / (1024**3),
            ram_percent=memory.percent,
            vram_used_gb=vram_info["used_gb"],
            vram_total_gb=vram_info["total_gb"],
            vram_percent=vram_info["percent"],
            gpu_temperature=vram_info["temperature"],
            process_ram_gb=process_ram_gb,
            process_threads=process_threads,
            process_fds=process_fds,
            active_sessions=active_sessions,
            session_queue_size=session_queue_size,
            error_count=recent_errors
        )

        self.snapshots.append(snapshot)
        return snapshot

    def analyze_stability(self, window_hours: float = 1.0) -> Dict[str, Any]:
        """Analyze stability over time window"""

        window_seconds = window_hours * 3600
        current_time = time.time()

        # Get snapshots in time window
        window_snapshots = [
            s for s in self.snapshots
            if current_time - s.timestamp <= window_seconds
        ]

        if len(window_snapshots) < 2:
            return {"error": "Insufficient data for analysis"}

        # Calculate trends
        ram_usage = [s.ram_percent for s in window_snapshots]
        vram_usage = [s.vram_percent for s in window_snapshots]
        process_memory = [s.process_ram_gb for s in window_snapshots]

        return {
            "time_window_hours": window_hours,
            "snapshots_analyzed": len(window_snapshots),
            "ram_trend": {
                "min": min(ram_usage),
                "max": max(ram_usage),
                "avg": sum(ram_usage) / len(ram_usage),
                "growth_rate": self._calculate_growth_rate([s.timestamp for s in window_snapshots], ram_usage)
            },
            "vram_trend": {
                "min": min(vram_usage),
                "max": max(vram_usage),
                "avg": sum(vram_usage) / len(vram_usage),
                "growth_rate": self._calculate_growth_rate([s.timestamp for s in window_snapshots], vram_usage)
            },
            "process_memory_trend": {
                "min": min(process_memory),
                "max": max(process_memory),
                "avg": sum(process_memory) / len(process_memory),
                "growth_mb_per_hour": self._calculate_growth_rate(
                    [s.timestamp for s in window_snapshots],
                    [m * 1024 for m in process_memory]  # Convert to MB
                ) * 3600  # Per hour
            },
            "stability_indicators": self._assess_stability_indicators(window_snapshots)
        }

    def _calculate_growth_rate(self, timestamps: List[float], values: List[float]) -> float:
        """Calculate linear growth rate (value per second)"""
        if len(timestamps) < 2:
            return 0.0

        # Simple linear regression
        n = len(timestamps)
        sum_x = sum(timestamps)
        sum_y = sum(values)
        sum_xy = sum(t * v for t, v in zip(timestamps, values))
        sum_x2 = sum(t * t for t in timestamps)

        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        return slope

    def _assess_stability_indicators(self, snapshots: List[ResourceSnapshot]) -> Dict[str, Any]:
        """Assess various stability indicators"""

        # Memory leak detection
        memory_growth = self._calculate_growth_rate(
            [s.timestamp for s in snapshots],
            [s.process_ram_gb * 1024 for s in snapshots]  # MB
        ) * 3600  # MB per hour

        # Resource spikes
        ram_spikes = len([s for s in snapshots if s.ram_percent > 80])
        vram_spikes = len([s for s in snapshots if s.vram_percent > 85])

        # Error rate
        total_errors = sum(s.error_count for s in snapshots)
        error_rate = total_errors / len(snapshots) if snapshots else 0

        return {
            "memory_leak_risk": {
                "growth_mb_per_hour": memory_growth,
                "risk_level": "high" if memory_growth > 50 else "medium" if memory_growth > 20 else "low"
            },
            "resource_stability": {
                "ram_spike_count": ram_spikes,
                "vram_spike_count": vram_spikes,
                "stable_operation": ram_spikes == 0 and vram_spikes == 0
            },
            "error_stability": {
                "total_errors": total_errors,
                "avg_error_rate": error_rate,
                "stable_errors": error_rate < 0.1  # Less than 0.1 errors per snapshot
            }
        }


class StabilityStressTest:
    """
    Long-running stability stress test

    Simulates 2 hours of 5 concurrent user sessions with:
    - Continuous request generation
    - Resource monitoring
    - Memory leak detection
    - OOM prevention
    """

    def __init__(self, config_obj: StabilityTestConfig = None):
        self.config = config_obj or StabilityTestConfig()
        self.monitor = SystemResourceMonitor()
        self.is_running = False
        self.session_tasks: List[asyncio.Task] = []
        self.monitoring_task: Optional[asyncio.Task] = None
        self.start_time: float = 0

        # Statistics
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "oom_events": 0,
            "safety_stops": 0
        }

    async def initialize(self):
        """Initialize stress test environment"""
        logger.info("stability.initializing")

        await self.monitor.initialize()

        # Initial resource check
        initial_snapshot = self.monitor.capture_snapshot()
        logger.info("stability.baseline_captured",
                   ram_percent=initial_snapshot.ram_percent,
                   vram_percent=initial_snapshot.vram_percent)

    async def run_stability_test(self, duration_minutes: int = None) -> Dict[str, Any]:
        """Run complete stability stress test"""

        duration = duration_minutes * 60 if duration_minutes else (self.config.duration_hours * 3600)

        logger.info("stability.test_started",
                   duration_hours=duration / 3600,
                   concurrent_sessions=self.config.concurrent_sessions)

        self.is_running = True
        self.start_time = time.time()

        try:
            # Start concurrent sessions
            for i in range(self.config.concurrent_sessions):
                task = asyncio.create_task(self._run_session_loop(f"session-{i}"))
                self.session_tasks.append(task)

            # Start monitoring
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())

            # Run for specified duration
            await asyncio.sleep(duration)

            logger.info("stability.test_duration_completed", duration_hours=duration/3600)

        except Exception as e:
            logger.error("stability.test_failed", error=str(e))
            raise
        finally:
            await self._cleanup()

        # Analyze results
        results = await self._analyze_test_results()
        return results

    async def _run_session_loop(self, session_id: str):
        """Run continuous requests for a single session"""

        request_count = 0

        while self.is_running:
            try:
                # Simulate user request processing
                await self._simulate_user_request(session_id, request_count)

                request_count += 1
                self.stats["total_requests"] += 1
                self.stats["successful_requests"] += 1

                # Variable interval between requests (realistic user behavior)
                import random
                interval = random.uniform(
                    self.config.request_interval_min,
                    self.config.request_interval_max
                )
                await asyncio.sleep(interval)

            except Exception as e:
                self.stats["failed_requests"] += 1
                logger.warning("stability.session_error",
                             session_id=session_id,
                             request=request_count,
                             error=str(e))

                # Brief pause before retry
                await asyncio.sleep(5)

    async def _simulate_user_request(self, session_id: str, request_id: int):
        """Simulate a complete user request (STT → LLM → TTS)"""

        # Simulate STT processing
        await self._simulate_stt(session_id, request_id)

        # Simulate LLM processing
        await self._simulate_llm(session_id, request_id)

        # Simulate TTS processing
        await self._simulate_tts(session_id, request_id)

        # Cleanup simulation (important for stability)
        gc.collect()

    async def _simulate_stt(self, session_id: str, request_id: int):
        """Simulate STT processing with memory management"""

        # Simulate audio data processing
        audio_data = bytearray(1024 * 1024 * 2)  # 2MB audio

        # Simulate processing time
        await asyncio.sleep(0.1 + (request_id % 5) * 0.05)  # 100-350ms

        # Cleanup
        del audio_data

    async def _simulate_llm(self, session_id: str, request_id: int):
        """Simulate LLM processing with GPU memory simulation"""

        if torch.cuda.is_available():
            device = torch.cuda.current_device()

            # Simulate model inference memory pattern
            try:
                # Allocate temporary computation tensors
                temp_tensor = torch.randn(1000, 1000, device=device, dtype=torch.float16)

                # Simulate computation
                await asyncio.sleep(0.2 + (request_id % 3) * 0.1)  # 200-500ms

                # Cleanup
                del temp_tensor
                torch.cuda.empty_cache()

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    self.stats["oom_events"] += 1
                    logger.critical("stability.oom_detected",
                                  session_id=session_id,
                                  request_id=request_id,
                                  error=str(e))
                    torch.cuda.empty_cache()
                raise
        else:
            # CPU simulation
            await asyncio.sleep(0.3)

    async def _simulate_tts(self, session_id: str, request_id: int):
        """Simulate TTS processing with resource cleanup"""

        # Simulate text processing and synthesis
        text_data = "Simulated response text " * (50 + request_id % 100)

        # Simulate synthesis time
        await asyncio.sleep(0.4 + (request_id % 4) * 0.1)  # 400-800ms

        # Cleanup
        del text_data

    async def _monitoring_loop(self):
        """Continuous system monitoring loop"""

        consecutive_warnings = 0

        while self.is_running:
            try:
                # Capture resource snapshot
                snapshot = self.monitor.capture_snapshot(
                    active_sessions=len(self.session_tasks),
                    session_queue_size=0
                )

                # Check for dangerous resource levels
                warnings = []

                if snapshot.ram_percent > self.config.max_ram_percent:
                    warnings.append(f"High RAM usage: {snapshot.ram_percent:.1f}%")

                if snapshot.vram_percent > self.config.max_vram_percent:
                    warnings.append(f"High VRAM usage: {snapshot.vram_percent:.1f}%")

                if snapshot.gpu_temperature > self.config.max_temperature_c:
                    warnings.append(f"High GPU temperature: {snapshot.gpu_temperature:.1f}°C")

                if warnings:
                    consecutive_warnings += 1
                    logger.warning("stability.resource_warning",
                                 warnings=warnings,
                                 consecutive_count=consecutive_warnings)

                    # Emergency stop if too many consecutive warnings
                    if consecutive_warnings >= self.config.consecutive_warnings:
                        logger.critical("stability.emergency_stop",
                                      reason="consecutive_resource_warnings")
                        self.stats["safety_stops"] += 1
                        self.is_running = False
                        break
                else:
                    consecutive_warnings = 0

                # Detailed logging every 5 minutes
                elapsed = time.time() - self.start_time
                if int(elapsed) % self.config.detailed_check_interval == 0:
                    await self._log_detailed_status(snapshot, elapsed)

                await asyncio.sleep(self.config.resource_check_interval)

            except Exception as e:
                logger.error("stability.monitoring_error", error=str(e))
                await asyncio.sleep(self.config.resource_check_interval)

    async def _log_detailed_status(self, snapshot: ResourceSnapshot, elapsed_seconds: float):
        """Log detailed status every few minutes"""

        # Analyze recent trends
        analysis = self.monitor.analyze_stability(window_hours=0.5)

        logger.info("stability.status_report",
                   elapsed_hours=elapsed_seconds / 3600,
                   ram_percent=snapshot.ram_percent,
                   vram_percent=snapshot.vram_percent,
                   process_ram_gb=snapshot.process_ram_gb,
                   total_requests=self.stats["total_requests"],
                   success_rate=(self.stats["successful_requests"] / max(self.stats["total_requests"], 1)) * 100,
                   memory_growth=analysis.get("process_memory_trend", {}).get("growth_mb_per_hour", 0))

    async def _cleanup(self):
        """Cleanup test resources"""
        logger.info("stability.cleanup_started")

        self.is_running = False

        # Cancel all session tasks
        for task in self.session_tasks:
            if not task.done():
                task.cancel()

        # Cancel monitoring task
        if self.monitoring_task and not self.monitoring_task.done():
            self.monitoring_task.cancel()

        # Wait for tasks to complete
        await asyncio.gather(*self.session_tasks, self.monitoring_task, return_exceptions=True)

        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("stability.cleanup_completed")

    async def _analyze_test_results(self) -> Dict[str, Any]:
        """Analyze complete test results"""

        test_duration = time.time() - self.start_time

        # Overall statistics
        success_rate = (self.stats["successful_requests"] / max(self.stats["total_requests"], 1)) * 100

        # Resource stability analysis
        stability_analysis = self.monitor.analyze_stability(window_hours=test_duration / 3600)

        # Test outcome assessment
        test_passed = (
            self.stats["oom_events"] == 0 and
            self.stats["safety_stops"] == 0 and
            success_rate >= 95 and
            stability_analysis.get("stability_indicators", {}).get("memory_leak_risk", {}).get("risk_level") != "high"
        )

        return {
            "test_outcome": {
                "passed": test_passed,
                "duration_hours": test_duration / 3600,
                "oom_events": self.stats["oom_events"],
                "safety_stops": self.stats["safety_stops"]
            },
            "request_statistics": {
                "total_requests": self.stats["total_requests"],
                "successful_requests": self.stats["successful_requests"],
                "failed_requests": self.stats["failed_requests"],
                "success_rate": success_rate,
                "requests_per_hour": self.stats["total_requests"] / (test_duration / 3600)
            },
            "resource_analysis": stability_analysis,
            "stability_assessment": self._generate_stability_assessment(stability_analysis, test_passed),
            "recommendations": self._generate_stability_recommendations(stability_analysis)
        }

    def _generate_stability_assessment(self, analysis: Dict[str, Any], test_passed: bool) -> Dict[str, Any]:
        """Generate stability assessment"""

        indicators = analysis.get("stability_indicators", {})
        memory_risk = indicators.get("memory_leak_risk", {}).get("risk_level", "unknown")

        return {
            "overall_stability": "excellent" if test_passed else "needs_improvement",
            "memory_management": "stable" if memory_risk == "low" else "unstable",
            "resource_efficiency": "good" if analysis.get("vram_trend", {}).get("avg", 0) < 80 else "poor",
            "error_resilience": "robust" if indicators.get("error_stability", {}).get("stable_errors", False) else "fragile"
        }

    def _generate_stability_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate stability recommendations"""

        recommendations = []

        memory_growth = analysis.get("process_memory_trend", {}).get("growth_mb_per_hour", 0)
        if memory_growth > 50:
            recommendations.append("Investigate memory leaks in session management")

        vram_avg = analysis.get("vram_trend", {}).get("avg", 0)
        if vram_avg > 80:
            recommendations.append("Optimize VRAM usage patterns")

        error_rate = analysis.get("stability_indicators", {}).get("error_stability", {}).get("avg_error_rate", 0)
        if error_rate > 0.1:
            recommendations.append("Improve error handling and recovery mechanisms")

        return recommendations


async def run_stability_test(duration_minutes: int = 10, quick_mode: bool = False):
    """Run stability test"""
    configure_logging(development_mode=True)

    if quick_mode:
        duration_minutes = 5
        concurrent = 2
    else:
        concurrent = 5

    print(f"🧪 Starting Stability Test ({duration_minutes} minutes, {concurrent} concurrent sessions)...")

    # Create test configuration
    test_config = StabilityTestConfig(
        duration_hours=duration_minutes / 60,
        concurrent_sessions=concurrent
    )

    stress_test = StabilityStressTest(test_config)

    try:
        await stress_test.initialize()
        results = await stress_test.run_stability_test(duration_minutes)

        # Display results
        print(f"\n📊 Stability Test Results:")

        outcome = results["test_outcome"]
        print(f"Test Result: {'✅ PASSED' if outcome['passed'] else '❌ FAILED'}")
        print(f"Duration: {outcome['duration_hours']:.2f} hours")
        print(f"OOM Events: {outcome['oom_events']}")
        print(f"Safety Stops: {outcome['safety_stops']}")

        stats = results["request_statistics"]
        print(f"\nRequest Statistics:")
        print(f"Total Requests: {stats['total_requests']}")
        print(f"Success Rate: {stats['success_rate']:.1f}%")
        print(f"Requests/Hour: {stats['requests_per_hour']:.1f}")

        # Resource analysis
        resource = results["resource_analysis"]
        if "process_memory_trend" in resource:
            memory = resource["process_memory_trend"]
            print(f"\nMemory Analysis:")
            print(f"Memory Growth: {memory['growth_mb_per_hour']:.1f} MB/hour")
            print(f"Peak Memory: {memory['max']:.2f} GB")

        # Stability assessment
        assessment = results["stability_assessment"]
        print(f"\nStability Assessment:")
        print(f"Overall: {assessment['overall_stability']}")
        print(f"Memory Management: {assessment['memory_management']}")
        print(f"Resource Efficiency: {assessment['resource_efficiency']}")

        # Recommendations
        recommendations = results["recommendations"]
        if recommendations:
            print(f"\n💡 Recommendations:")
            for rec in recommendations:
                print(f"  • {rec}")

        return results

    except Exception as e:
        print(f"❌ Stability test failed: {e}")
        logger.error("stability.test_failed", error=str(e))
        raise


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AVATAR Stability Test")
    parser.add_argument("--duration", type=int, default=10, help="Test duration in minutes")
    parser.add_argument("--quick", action="store_true", help="Quick test mode (5 minutes, 2 sessions)")
    args = parser.parse_args()

    asyncio.run(run_stability_test(duration_minutes=args.duration, quick_mode=args.quick))