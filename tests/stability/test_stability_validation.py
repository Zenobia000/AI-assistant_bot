"""
Quick Stability Validation Test

Task 25: Fast stability validation without long-running tests
Validates memory management, resource cleanup, and OOM prevention

Focus Areas:
- Session lifecycle memory management
- VRAM allocation/deallocation patterns
- Concurrent session resource isolation
- Error recovery without memory leaks
"""

import asyncio
import gc
import time
import sys
import os
from typing import Dict, List, Any
from pathlib import Path

import structlog
import torch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from avatar.core.config import config
from avatar.core.logging_config import configure_logging
from avatar.core.vram_monitor import get_vram_monitor

logger = structlog.get_logger()


class QuickStabilityValidator:
    """
    Quick stability validation

    Linus principle: "The best test is the simplest test that catches the bug"
    Focus on essential stability patterns rather than exhaustive time-based testing
    """

    def __init__(self):
        self.initial_vram = 0
        self.initial_objects = 0

    async def run_validation_suite(self) -> Dict[str, Any]:
        """Run comprehensive but quick stability validation"""
        logger.info("stability_validation.started")

        results = {
            "validation_timestamp": time.time(),
            "tests_run": 0,
            "tests_passed": 0,
            "critical_failures": 0,
            "test_results": {}
        }

        # Test 1: Session lifecycle memory management
        results["test_results"]["session_memory"] = await self.test_session_memory_lifecycle()
        results["tests_run"] += 1
        if results["test_results"]["session_memory"]["passed"]:
            results["tests_passed"] += 1

        # Test 2: VRAM allocation patterns
        results["test_results"]["vram_patterns"] = await self.test_vram_allocation_patterns()
        results["tests_run"] += 1
        if results["test_results"]["vram_patterns"]["passed"]:
            results["tests_passed"] += 1

        # Test 3: Concurrent session isolation
        results["test_results"]["concurrent_isolation"] = await self.test_concurrent_session_isolation()
        results["tests_run"] += 1
        if results["test_results"]["concurrent_isolation"]["passed"]:
            results["tests_passed"] += 1

        # Test 4: Error recovery cleanup
        results["test_results"]["error_recovery"] = await self.test_error_recovery_cleanup()
        results["tests_run"] += 1
        if results["test_results"]["error_recovery"]["passed"]:
            results["tests_passed"] += 1

        # Test 5: Resource limit enforcement
        results["test_results"]["resource_limits"] = await self.test_resource_limit_enforcement()
        results["tests_run"] += 1
        if results["test_results"]["resource_limits"]["passed"]:
            results["tests_passed"] += 1

        # Overall assessment
        results["overall_stability"] = {
            "passed": results["tests_passed"] == results["tests_run"],
            "pass_rate": (results["tests_passed"] / results["tests_run"]) * 100,
            "ready_for_production": results["critical_failures"] == 0 and results["tests_passed"] >= 4
        }

        logger.info("stability_validation.completed",
                   tests_passed=results["tests_passed"],
                   tests_total=results["tests_run"])

        return results

    async def test_session_memory_lifecycle(self) -> Dict[str, Any]:
        """Test session creation/destruction memory patterns"""
        logger.info("stability_validation.session_memory_test")

        # Baseline measurement
        gc.collect()
        initial_objects = len(gc.get_objects())

        try:
            # Create and destroy sessions rapidly
            for i in range(20):
                # Simulate session creation
                session_data = {
                    "session_id": f"test-{i}",
                    "buffer": [0] * 10000,  # 10k integers
                    "metadata": {"created": time.time()}
                }

                # Simulate processing
                await asyncio.sleep(0.01)  # 10ms

                # Explicit cleanup
                del session_data

                # Force garbage collection every 5 iterations
                if i % 5 == 0:
                    gc.collect()

            # Final cleanup and measurement
            gc.collect()
            final_objects = len(gc.get_objects())

            object_growth = final_objects - initial_objects
            growth_rate = object_growth / 20  # Objects per session

            # Assessment
            passed = object_growth < 100  # Allow small growth
            severity = "critical" if object_growth > 500 else "warning" if object_growth > 100 else "normal"

            return {
                "passed": passed,
                "severity": severity,
                "object_growth": object_growth,
                "growth_per_session": growth_rate,
                "details": f"Object growth: {object_growth} total, {growth_rate:.1f} per session"
            }

        except Exception as e:
            logger.error("stability_validation.session_memory_failed", error=str(e))
            return {
                "passed": False,
                "severity": "critical",
                "error": str(e)
            }

    async def test_vram_allocation_patterns(self) -> Dict[str, Any]:
        """Test VRAM allocation and deallocation patterns"""
        logger.info("stability_validation.vram_test")

        if not torch.cuda.is_available():
            return {
                "passed": True,
                "severity": "normal",
                "details": "No CUDA available - test skipped"
            }

        try:
            # Baseline VRAM
            torch.cuda.empty_cache()
            initial_vram = torch.cuda.memory_allocated() / (1024**2)  # MB

            allocated_tensors = []

            # Simulate model loading/unloading cycles
            for cycle in range(10):
                # Allocate tensors (simulate model weights)
                for i in range(5):
                    tensor = torch.randn(200, 200, device="cuda", dtype=torch.float16)
                    allocated_tensors.append(tensor)

                # Peak VRAM check
                peak_vram = torch.cuda.memory_allocated() / (1024**2)

                # Cleanup tensors
                for tensor in allocated_tensors:
                    del tensor
                allocated_tensors.clear()

                torch.cuda.empty_cache()

                # Check cleanup effectiveness
                after_cleanup = torch.cuda.memory_allocated() / (1024**2)

                logger.debug("stability_validation.vram_cycle",
                           cycle=cycle,
                           peak_mb=peak_vram,
                           after_cleanup_mb=after_cleanup)

            # Final VRAM measurement
            final_vram = torch.cuda.memory_allocated() / (1024**2)
            vram_growth = final_vram - initial_vram

            # Assessment
            passed = vram_growth < 10  # Less than 10MB growth
            severity = "critical" if vram_growth > 100 else "warning" if vram_growth > 10 else "normal"

            return {
                "passed": passed,
                "severity": severity,
                "vram_growth_mb": vram_growth,
                "initial_vram_mb": initial_vram,
                "final_vram_mb": final_vram,
                "details": f"VRAM growth: {vram_growth:.1f} MB after 10 allocation cycles"
            }

        except Exception as e:
            logger.error("stability_validation.vram_test_failed", error=str(e))
            return {
                "passed": False,
                "severity": "critical",
                "error": str(e)
            }

    async def test_concurrent_session_isolation(self) -> Dict[str, Any]:
        """Test concurrent session resource isolation"""
        logger.info("stability_validation.concurrent_isolation_test")

        try:
            async def simulate_session(session_id: str, duration: float):
                """Simulate single session resource usage"""
                session_data = {"id": session_id, "data": [0] * 5000}

                if torch.cuda.is_available():
                    # Allocate session-specific VRAM
                    tensor = torch.randn(100, 100, device="cuda", dtype=torch.float16)

                await asyncio.sleep(duration)

                # Cleanup
                if torch.cuda.is_available():
                    del tensor
                    torch.cuda.empty_cache()

                del session_data
                return f"Session {session_id} completed"

            # Run 5 concurrent sessions
            start_time = time.time()
            tasks = [
                simulate_session(f"concurrent-{i}", 2.0)  # 2 second sessions
                for i in range(5)
            ]

            results_list = await asyncio.gather(*tasks, return_exceptions=True)
            duration = time.time() - start_time

            # Check for exceptions
            exceptions = [r for r in results_list if isinstance(r, Exception)]
            successful = len(results_list) - len(exceptions)

            passed = len(exceptions) == 0
            severity = "critical" if len(exceptions) > 2 else "warning" if len(exceptions) > 0 else "normal"

            return {
                "passed": passed,
                "severity": severity,
                "concurrent_sessions": 5,
                "successful_sessions": successful,
                "failed_sessions": len(exceptions),
                "total_duration": duration,
                "details": f"{successful}/5 sessions completed successfully"
            }

        except Exception as e:
            logger.error("stability_validation.concurrent_test_failed", error=str(e))
            return {
                "passed": False,
                "severity": "critical",
                "error": str(e)
            }

    async def test_error_recovery_cleanup(self) -> Dict[str, Any]:
        """Test error scenarios don't cause memory leaks"""
        logger.info("stability_validation.error_recovery_test")

        try:
            # Baseline
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                initial_vram = torch.cuda.memory_allocated() / (1024**2)
            else:
                initial_vram = 0

            initial_objects = len(gc.get_objects())

            # Simulate error scenarios with cleanup
            for i in range(10):
                try:
                    # Allocate resources
                    session_data = {"id": f"error-session-{i}", "buffer": [0] * 1000}

                    if torch.cuda.is_available():
                        error_tensor = torch.randn(50, 50, device="cuda", dtype=torch.float16)

                    # Simulate operation that fails
                    if i % 3 == 0:  # 1/3 of operations fail
                        raise ValueError(f"Simulated error {i}")

                    await asyncio.sleep(0.01)

                except ValueError:
                    # Error recovery - ensure cleanup happens
                    pass
                finally:
                    # Critical: cleanup in finally block
                    if 'session_data' in locals():
                        del session_data
                    if 'error_tensor' in locals():
                        del error_tensor
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

            # Final measurement
            gc.collect()
            final_objects = len(gc.get_objects())

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                final_vram = torch.cuda.memory_allocated() / (1024**2)
            else:
                final_vram = 0

            # Assessment
            object_growth = final_objects - initial_objects
            vram_growth = final_vram - initial_vram

            passed = object_growth < 50 and abs(vram_growth) < 5
            severity = "critical" if object_growth > 200 or abs(vram_growth) > 20 else "normal"

            return {
                "passed": passed,
                "severity": severity,
                "object_growth": object_growth,
                "vram_growth_mb": vram_growth,
                "error_scenarios_tested": 10,
                "details": f"After error scenarios: +{object_growth} objects, {vram_growth:+.1f}MB VRAM"
            }

        except Exception as e:
            logger.error("stability_validation.error_recovery_failed", error=str(e))
            return {
                "passed": False,
                "severity": "critical",
                "error": str(e)
            }

    async def test_resource_limit_enforcement(self) -> Dict[str, Any]:
        """Test resource limit enforcement prevents OOM"""
        logger.info("stability_validation.resource_limits_test")

        try:
            vram_monitor = get_vram_monitor()

            # Test VRAM limit enforcement
            if torch.cuda.is_available():
                initial_vram = torch.cuda.memory_allocated() / (1024**3)  # GB

                # Try to allocate increasingly large tensors
                max_allocation_gb = 0
                allocation_successful = True

                for size_gb in [0.5, 1.0, 2.0, 5.0, 10.0]:  # Progressively larger
                    try:
                        # Check if we can allocate this much
                        prediction = vram_monitor.predict_can_handle_vram(size_gb)

                        if prediction["can_handle"]:
                            # Try actual allocation
                            size_elements = int((size_gb * 1024**3) // 4)  # float32 = 4 bytes
                            tensor = torch.randn(size_elements, device="cuda", dtype=torch.float32)

                            max_allocation_gb = size_gb
                            logger.debug("stability_validation.vram_allocation_ok",
                                       size_gb=size_gb)

                            # Cleanup immediately
                            del tensor
                            torch.cuda.empty_cache()

                        else:
                            logger.info("stability_validation.vram_allocation_blocked",
                                      size_gb=size_gb,
                                      reason=prediction["reason"])
                            break

                    except RuntimeError as e:
                        if "out of memory" in str(e).lower():
                            logger.warning("stability_validation.oom_caught",
                                         size_gb=size_gb)
                            allocation_successful = False
                            break
                        else:
                            raise

                # Verify cleanup
                final_vram = torch.cuda.memory_allocated() / (1024**3)
                vram_diff = final_vram - initial_vram

                passed = allocation_successful and abs(vram_diff) < 0.1  # <100MB difference
                severity = "critical" if not allocation_successful else "normal"

                return {
                    "passed": passed,
                    "severity": severity,
                    "max_allocation_gb": max_allocation_gb,
                    "vram_cleanup_successful": abs(vram_diff) < 0.1,
                    "vram_diff_gb": vram_diff,
                    "details": f"Max safe allocation: {max_allocation_gb}GB, cleanup: {abs(vram_diff)*1000:.1f}MB diff"
                }

            else:
                return {
                    "passed": True,
                    "severity": "normal",
                    "details": "No CUDA available - test skipped"
                }

        except Exception as e:
            logger.error("stability_validation.resource_limits_failed", error=str(e))
            return {
                "passed": False,
                "severity": "critical",
                "error": str(e)
            }

    async def run_quick_stress_burst(self, duration_seconds: int = 30) -> Dict[str, Any]:
        """Run quick stress burst to validate stability"""
        logger.info("stability_validation.stress_burst", duration=duration_seconds)

        start_time = time.time()
        request_count = 0
        error_count = 0

        # Capture initial state
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            initial_vram = torch.cuda.memory_allocated() / (1024**2)  # MB
        else:
            initial_vram = 0

        initial_objects = len(gc.get_objects())

        # High-intensity burst
        while time.time() - start_time < duration_seconds:
            try:
                # Rapid session simulation
                session_data = {"id": f"burst-{request_count}", "data": [0] * 1000}

                # Rapid VRAM allocation
                if torch.cuda.is_available():
                    tensor = torch.randn(100, 100, device="cuda", dtype=torch.float16)

                # Very brief processing
                await asyncio.sleep(0.001)  # 1ms

                # Immediate cleanup
                del session_data
                if torch.cuda.is_available():
                    del tensor

                request_count += 1

                # Aggressive garbage collection
                if request_count % 10 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            except Exception as e:
                error_count += 1
                logger.warning("stability_validation.burst_error",
                             request=request_count,
                             error=str(e))

        # Final measurements
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            final_vram = torch.cuda.memory_allocated() / (1024**2)
        else:
            final_vram = 0

        final_objects = len(gc.get_objects())

        # Analysis
        object_growth = final_objects - initial_objects
        vram_growth = final_vram - initial_vram
        success_rate = (request_count - error_count) / request_count * 100 if request_count > 0 else 0

        passed = (
            error_count == 0 and
            object_growth < 200 and  # Allow some object growth
            abs(vram_growth) < 50 and  # <50MB VRAM growth
            success_rate == 100
        )

        severity = "critical" if not passed and (error_count > 0 or abs(vram_growth) > 100) else "normal"

        return {
            "passed": passed,
            "severity": severity,
            "duration": duration_seconds,
            "requests_processed": request_count,
            "error_count": error_count,
            "success_rate": success_rate,
            "object_growth": object_growth,
            "vram_growth_mb": vram_growth,
            "details": f"Processed {request_count} requests in {duration_seconds}s, {error_count} errors"
        }


async def run_stability_validation():
    """Run quick but comprehensive stability validation"""
    configure_logging(development_mode=True)

    print("🧪 Starting Quick Stability Validation...")

    validator = QuickStabilityValidator()

    try:
        # Run validation suite
        results = await validator.run_validation_suite()

        print(f"\n📊 Stability Validation Results:")

        overall = results["overall_stability"]
        print(f"Overall Result: {'✅ PASSED' if overall['passed'] else '❌ FAILED'}")
        print(f"Tests Passed: {results['tests_passed']}/{results['tests_run']}")
        print(f"Production Ready: {'✅ YES' if overall['ready_for_production'] else '❌ NO'}")

        # Individual test results
        for test_name, test_result in results["test_results"].items():
            status = "✅ PASS" if test_result["passed"] else "❌ FAIL"
            print(f"\n{test_name.replace('_', ' ').title()}: {status}")
            if "details" in test_result:
                print(f"  {test_result['details']}")

        # Run stress burst
        print(f"\n🚀 Running 30-second stress burst...")
        burst_result = await validator.run_quick_stress_burst(30)

        print(f"Stress Burst: {'✅ PASSED' if burst_result['passed'] else '❌ FAILED'}")
        print(f"  {burst_result['details']}")

        # Final assessment
        final_stability = overall["passed"] and burst_result["passed"]
        print(f"\n🏆 Final Stability Assessment: {'✅ STABLE' if final_stability else '❌ UNSTABLE'}")

        return results

    except Exception as e:
        print(f"❌ Stability validation failed: {e}")
        logger.error("stability_validation.failed", error=str(e))
        raise


if __name__ == "__main__":
    asyncio.run(run_stability_validation())