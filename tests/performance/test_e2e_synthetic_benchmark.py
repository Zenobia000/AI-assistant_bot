"""
Synthetic E2E Performance Benchmark

Task 24: Lightweight performance testing without model dependencies
Measures system infrastructure performance and provides baseline metrics

This test simulates the E2E pipeline without loading actual AI models,
focusing on framework overhead, database performance, and infrastructure latency.
"""

import asyncio
import time
import statistics
import json
import sys
import os
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass, field

import structlog
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from avatar.core.config import config
from avatar.core.logging_config import configure_logging, PerformanceLogger
from avatar.services.database import db

logger = structlog.get_logger()


@dataclass
class SyntheticBenchmarkResult:
    """Results from synthetic performance benchmark"""
    component: str
    operation: str
    latencies: List[float]
    success_rate: float
    throughput: float
    additional_metrics: Dict[str, Any] = field(default_factory=dict)

    @property
    def stats(self) -> Dict[str, float]:
        """Calculate latency statistics"""
        if not self.latencies:
            return {"min": 0, "max": 0, "avg": 0, "p50": 0, "p95": 0, "p99": 0}

        return {
            "min": min(self.latencies),
            "max": max(self.latencies),
            "avg": statistics.mean(self.latencies),
            "p50": float(np.percentile(self.latencies, 50)),
            "p95": float(np.percentile(self.latencies, 95)),
            "p99": float(np.percentile(self.latencies, 99))
        }


class SyntheticE2EBenchmark:
    """
    Synthetic E2E performance benchmark

    Tests infrastructure performance without AI model dependencies:
    - Database operations
    - File I/O operations
    - Network simulation
    - Memory allocation patterns
    - Async task overhead
    """

    def __init__(self):
        self.results: List[SyntheticBenchmarkResult] = []
        self.test_dir = Path("performance_test_data")
        self.test_dir.mkdir(exist_ok=True)

    async def run_full_benchmark(self, num_requests: int = 50) -> Dict[str, Any]:
        """Run complete synthetic benchmark suite"""
        logger.info("synthetic_benchmark.started", requests=num_requests)

        # Database operations benchmark
        db_result = await self.benchmark_database_operations(num_requests)
        self.results.append(db_result)

        # File I/O benchmark
        file_result = await self.benchmark_file_operations(num_requests)
        self.results.append(file_result)

        # Memory allocation benchmark (simulates model loading)
        memory_result = await self.benchmark_memory_operations(num_requests)
        self.results.append(memory_result)

        # Async task benchmark (simulates streaming)
        async_result = await self.benchmark_async_operations(num_requests)
        self.results.append(async_result)

        # Network simulation benchmark
        network_result = await self.benchmark_network_simulation(num_requests)
        self.results.append(network_result)

        # Analyze overall results
        analysis = self.analyze_results()

        logger.info("synthetic_benchmark.completed",
                   total_operations=len(self.results),
                   total_requests=num_requests * len(self.results))

        return analysis

    async def benchmark_database_operations(self, num_requests: int) -> SyntheticBenchmarkResult:
        """Benchmark database operations (CREATE, READ, UPDATE)"""
        logger.info("synthetic_benchmark.db_operations", requests=num_requests)

        latencies = []
        errors = 0

        for i in range(num_requests):
            start_time = time.time()

            try:
                # Simulate conversation creation
                conversation_id = await db.create_conversation(
                    user_audio_path=f"synthetic_test_{i}.wav",
                    user_text=f"Synthetic test message {i}",
                    ai_text=f"Synthetic test response {i}",
                    ai_audio_fast_path=f"synthetic_response_{i}.wav",
                    session_id=f"synthetic-session-{i}",
                    turn_number=i + 1,
                    voice_profile_id=None
                )

                # Simulate reading conversation
                conversations = await db.get_conversations_by_session(f"synthetic-session-{i}")

                latency = time.time() - start_time
                latencies.append(latency)

            except Exception as e:
                errors += 1
                logger.warning("synthetic_benchmark.db_error", request=i, error=str(e))

        success_rate = (num_requests - errors) / num_requests * 100
        throughput = num_requests / sum(latencies) if latencies else 0

        return SyntheticBenchmarkResult(
            component="Database",
            operation="CRUD Operations",
            latencies=latencies,
            success_rate=success_rate,
            throughput=throughput,
            additional_metrics={
                "avg_operations_per_request": 2,  # CREATE + READ
                "errors": errors
            }
        )

    async def benchmark_file_operations(self, num_requests: int) -> SyntheticBenchmarkResult:
        """Benchmark file I/O operations (simulates audio file handling)"""
        logger.info("synthetic_benchmark.file_operations", requests=num_requests)

        latencies = []
        errors = 0
        file_sizes = []

        for i in range(num_requests):
            start_time = time.time()

            try:
                # Simulate audio file write (1-5MB files)
                file_size = 1024 * 1024 * (1 + (i % 5))  # 1-5MB
                file_sizes.append(file_size)

                test_file = self.test_dir / f"synthetic_audio_{i}.bin"

                # Write synthetic data
                with open(test_file, 'wb') as f:
                    f.write(b'0' * file_size)

                # Read back (simulates audio processing)
                with open(test_file, 'rb') as f:
                    data = f.read()

                # Cleanup
                test_file.unlink()

                latency = time.time() - start_time
                latencies.append(latency)

            except Exception as e:
                errors += 1
                logger.warning("synthetic_benchmark.file_error", request=i, error=str(e))

        success_rate = (num_requests - errors) / num_requests * 100
        throughput = num_requests / sum(latencies) if latencies else 0

        return SyntheticBenchmarkResult(
            component="FileSystem",
            operation="Read/Write",
            latencies=latencies,
            success_rate=success_rate,
            throughput=throughput,
            additional_metrics={
                "avg_file_size_mb": statistics.mean(file_sizes) / (1024*1024) if file_sizes else 0,
                "total_data_processed_mb": sum(file_sizes) / (1024*1024) if file_sizes else 0
            }
        )

    async def benchmark_memory_operations(self, num_requests: int) -> SyntheticBenchmarkResult:
        """Benchmark memory allocation patterns (simulates model loading)"""
        logger.info("synthetic_benchmark.memory_operations", requests=num_requests)

        latencies = []
        errors = 0
        memory_sizes = []

        for i in range(num_requests):
            start_time = time.time()

            try:
                # Simulate model weight allocation (10-50MB chunks)
                memory_size = 10 * 1024 * 1024 * (1 + (i % 5))  # 10-50MB
                memory_sizes.append(memory_size)

                # Allocate memory
                data = bytearray(memory_size)

                # Simulate computation (memory access patterns)
                for j in range(0, len(data), 4096):  # 4KB chunks
                    data[j:j+10] = b'1' * 10

                # Simulate tensor operations
                import hashlib
                checksum = hashlib.md5(data[:1024]).hexdigest()

                # Cleanup
                del data

                latency = time.time() - start_time
                latencies.append(latency)

            except Exception as e:
                errors += 1
                logger.warning("synthetic_benchmark.memory_error", request=i, error=str(e))

        success_rate = (num_requests - errors) / num_requests * 100
        throughput = num_requests / sum(latencies) if latencies else 0

        return SyntheticBenchmarkResult(
            component="Memory",
            operation="Allocation/Computation",
            latencies=latencies,
            success_rate=success_rate,
            throughput=throughput,
            additional_metrics={
                "avg_allocation_mb": statistics.mean(memory_sizes) / (1024*1024) if memory_sizes else 0,
                "total_memory_processed_mb": sum(memory_sizes) / (1024*1024) if memory_sizes else 0
            }
        )

    async def benchmark_async_operations(self, num_requests: int) -> SyntheticBenchmarkResult:
        """Benchmark async task management (simulates streaming operations)"""
        logger.info("synthetic_benchmark.async_operations", requests=num_requests)

        latencies = []
        errors = 0
        task_counts = []

        for i in range(num_requests):
            start_time = time.time()

            try:
                # Simulate concurrent streaming tasks
                num_tasks = 5 + (i % 10)  # 5-15 concurrent tasks
                task_counts.append(num_tasks)

                async def simulate_stream_chunk(chunk_id: int):
                    await asyncio.sleep(0.001 + (chunk_id % 3) * 0.001)  # 1-4ms
                    return f"chunk_{chunk_id}"

                # Run concurrent tasks (simulates token streaming)
                tasks = [simulate_stream_chunk(j) for j in range(num_tasks)]
                results = await asyncio.gather(*tasks)

                latency = time.time() - start_time
                latencies.append(latency)

            except Exception as e:
                errors += 1
                logger.warning("synthetic_benchmark.async_error", request=i, error=str(e))

        success_rate = (num_requests - errors) / num_requests * 100
        throughput = num_requests / sum(latencies) if latencies else 0

        return SyntheticBenchmarkResult(
            component="AsyncFramework",
            operation="Concurrent Tasks",
            latencies=latencies,
            success_rate=success_rate,
            throughput=throughput,
            additional_metrics={
                "avg_concurrent_tasks": statistics.mean(task_counts) if task_counts else 0,
                "total_tasks_executed": sum(task_counts)
            }
        )

    async def benchmark_network_simulation(self, num_requests: int) -> SyntheticBenchmarkResult:
        """Benchmark network-like operations (simulates API calls)"""
        logger.info("synthetic_benchmark.network_simulation", requests=num_requests)

        latencies = []
        errors = 0
        payload_sizes = []

        for i in range(num_requests):
            start_time = time.time()

            try:
                # Simulate API request/response with varying payload sizes
                payload_size = 1024 * (10 + (i % 50))  # 10-60KB payloads
                payload_sizes.append(payload_size)

                # Simulate request serialization
                request_data = {"text": "x" * payload_size, "request_id": i}
                serialized = json.dumps(request_data)

                # Simulate network delay
                await asyncio.sleep(0.001 + (i % 5) * 0.002)  # 1-10ms

                # Simulate response processing
                response_data = json.loads(serialized)
                processed_text = response_data["text"].upper()

                latency = time.time() - start_time
                latencies.append(latency)

            except Exception as e:
                errors += 1
                logger.warning("synthetic_benchmark.network_error", request=i, error=str(e))

        success_rate = (num_requests - errors) / num_requests * 100
        throughput = num_requests / sum(latencies) if latencies else 0

        return SyntheticBenchmarkResult(
            component="Network",
            operation="API Simulation",
            latencies=latencies,
            success_rate=success_rate,
            throughput=throughput,
            additional_metrics={
                "avg_payload_kb": statistics.mean(payload_sizes) / 1024 if payload_sizes else 0,
                "total_data_transferred_mb": sum(payload_sizes) / (1024*1024) if payload_sizes else 0
            }
        )

    def analyze_results(self) -> Dict[str, Any]:
        """Analyze benchmark results and generate insights"""

        component_analysis = {}
        total_latencies = []

        for result in self.results:
            stats = result.stats
            component_analysis[result.component.lower()] = {
                "operation": result.operation,
                "latency_stats": stats,
                "success_rate": result.success_rate,
                "throughput": result.throughput,
                "metrics": result.additional_metrics
            }

            total_latencies.extend(result.latencies)

        # Overall system performance
        if total_latencies:
            overall_stats = {
                "min": min(total_latencies),
                "max": max(total_latencies),
                "avg": statistics.mean(total_latencies),
                "p50": float(np.percentile(total_latencies, 50)),
                "p95": float(np.percentile(total_latencies, 95)),
                "p99": float(np.percentile(total_latencies, 99))
            }
        else:
            overall_stats = {"min": 0, "max": 0, "avg": 0, "p50": 0, "p95": 0, "p99": 0}

        # Performance insights
        insights = self._generate_performance_insights(component_analysis)

        # Simulate E2E projection
        e2e_projection = self._project_e2e_performance(component_analysis)

        return {
            "component_results": component_analysis,
            "overall_infrastructure": overall_stats,
            "performance_insights": insights,
            "e2e_projection": e2e_projection,
            "benchmark_summary": {
                "components_tested": len(self.results),
                "total_operations": sum(len(r.latencies) for r in self.results),
                "overall_success_rate": statistics.mean([r.success_rate for r in self.results])
            }
        }

    def _generate_performance_insights(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate performance insights from component analysis"""

        bottlenecks = []
        strengths = []

        for component, data in analysis.items():
            p95_latency = data["latency_stats"]["p95"]
            success_rate = data["success_rate"]

            # Identify bottlenecks
            if p95_latency > 0.1:  # > 100ms
                bottlenecks.append(f"{component}: P95 latency {p95_latency*1000:.1f}ms")

            if success_rate < 95:
                bottlenecks.append(f"{component}: Success rate {success_rate:.1f}%")

            # Identify strengths
            if p95_latency < 0.05 and success_rate > 98:  # < 50ms, >98%
                strengths.append(f"{component}: Excellent performance")

        return {
            "bottlenecks": bottlenecks,
            "strengths": strengths,
            "recommendations": self._generate_recommendations(analysis)
        }

    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []

        for component, data in analysis.items():
            p95 = data["latency_stats"]["p95"]

            if component == "database" and p95 > 0.05:  # 50ms
                recommendations.append("Database: Consider connection pooling and query optimization")

            elif component == "filesystem" and p95 > 0.1:  # 100ms
                recommendations.append("FileSystem: Consider SSD storage or async I/O optimization")

            elif component == "memory" and p95 > 0.05:  # 50ms
                recommendations.append("Memory: Consider memory pool allocation or smaller batch sizes")

            elif component == "asyncframework" and p95 > 0.02:  # 20ms
                recommendations.append("Async: Optimize concurrency limits or task scheduling")

            elif component == "network" and p95 > 0.02:  # 20ms
                recommendations.append("Network: Consider request batching or connection reuse")

        return recommendations

    def _project_e2e_performance(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Project E2E performance based on infrastructure metrics"""

        # Base infrastructure overhead
        infrastructure_overhead = sum(
            data["latency_stats"]["p95"] for data in analysis.values()
        )

        # Estimated AI component latencies (based on typical values)
        estimated_ai_latencies = {
            "stt": 0.4,    # 400ms for Whisper
            "llm": 0.6,    # 600ms for LLM inference
            "tts": 1.2     # 1.2s for TTS synthesis
        }

        # Project total E2E latency
        projected_e2e_p95 = infrastructure_overhead + sum(estimated_ai_latencies.values())

        # SLA compliance projection
        target_e2e_p95 = 3.5  # seconds
        compliance_margin = target_e2e_p95 - projected_e2e_p95

        return {
            "infrastructure_overhead_p95": infrastructure_overhead,
            "estimated_ai_components": estimated_ai_latencies,
            "projected_e2e_p95": projected_e2e_p95,
            "sla_target": target_e2e_p95,
            "compliance_margin": compliance_margin,
            "projected_compliance": compliance_margin > 0,
            "infrastructure_percentage": (infrastructure_overhead / projected_e2e_p95) * 100
        }


async def run_synthetic_benchmark(quick_mode: bool = False):
    """Run synthetic performance benchmark"""
    configure_logging(development_mode=True)

    print(f"🧪 Starting Synthetic E2E Performance Benchmark ({'Quick' if quick_mode else 'Full'} mode)...")

    num_requests = 20 if quick_mode else 50

    benchmark = SyntheticE2EBenchmark()

    try:
        results = await benchmark.run_full_benchmark(num_requests)

        print("\n📊 Infrastructure Performance Results:")

        for component, data in results["component_results"].items():
            print(f"\n{component.upper()}:")
            print(f"  Operation: {data['operation']}")
            print(f"  P95 Latency: {data['latency_stats']['p95']*1000:.1f}ms")
            print(f"  Success Rate: {data['success_rate']:.1f}%")
            print(f"  Throughput: {data['throughput']:.1f} ops/s")

        # Overall infrastructure
        overall = results["overall_infrastructure"]
        print(f"\n📈 Overall Infrastructure:")
        print(f"  P95 Latency: {overall['p95']*1000:.1f}ms")
        print(f"  Average Latency: {overall['avg']*1000:.1f}ms")

        # E2E projection
        projection = results["e2e_projection"]
        print(f"\n🎯 E2E Performance Projection:")
        print(f"  Infrastructure Overhead: {projection['infrastructure_overhead_p95']*1000:.1f}ms")
        print(f"  Projected E2E P95: {projection['projected_e2e_p95']:.3f}s")
        print(f"  SLA Target: {projection['sla_target']}s")
        print(f"  Compliance Margin: {projection['compliance_margin']:.3f}s")
        print(f"  {'✅ PROJECTED PASS' if projection['projected_compliance'] else '❌ PROJECTED FAIL'}")
        print(f"  Infrastructure %: {projection['infrastructure_percentage']:.1f}%")

        # Insights
        insights = results["performance_insights"]
        if insights["bottlenecks"]:
            print(f"\n⚠️ Performance Bottlenecks:")
            for bottleneck in insights["bottlenecks"]:
                print(f"  • {bottleneck}")

        if insights["strengths"]:
            print(f"\n✅ Performance Strengths:")
            for strength in insights["strengths"]:
                print(f"  • {strength}")

        if insights["recommendations"]:
            print(f"\n💡 Optimization Recommendations:")
            for rec in insights["recommendations"]:
                print(f"  • {rec}")

        return results

    except Exception as e:
        print(f"❌ Synthetic benchmark failed: {e}")
        logger.error("synthetic_benchmark.failed", error=str(e))
        raise


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Synthetic E2E Performance Benchmark")
    parser.add_argument("--quick", action="store_true", help="Run quick test (20 requests)")
    args = parser.parse_args()

    asyncio.run(run_synthetic_benchmark(quick_mode=args.quick))