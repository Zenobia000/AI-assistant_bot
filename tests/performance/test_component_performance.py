"""
Component-Level Performance Testing

Task 24: Individual AI service performance analysis
Tests each component (STT, LLM, TTS) in isolation for detailed profiling

Components:
- STT (Whisper): CPU inference optimization
- LLM (vLLM): GPU inference with streaming
- TTS (F5-TTS): GPU synthesis optimization
- Database operations and caching
"""

import asyncio
import time
import statistics
import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import wave
import struct

import structlog
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from avatar.core.config import config
from avatar.core.logging_config import configure_logging
from avatar.services.stt import get_stt_service
from avatar.services.llm import get_llm_service
from avatar.services.tts import get_tts_service
from avatar.services.database import db

logger = structlog.get_logger()


@dataclass
class ComponentBenchmark:
    """Performance benchmark for individual component"""
    component: str
    test_count: int
    latencies: List[float]
    throughput: float
    error_count: int
    success_rate: float

    # Component-specific metrics
    additional_metrics: Dict[str, Any]


class STTPerformanceTest:
    """STT (Speech-to-Text) performance testing"""

    def __init__(self):
        self.stt_service = None
        self.test_audio_files = []

    async def initialize(self):
        """Initialize STT service and test data"""
        logger.info("stt_perf.initializing")

        self.stt_service = await get_stt_service()
        await self._prepare_test_audio()

    async def _prepare_test_audio(self):
        """Create test audio files of various lengths"""
        test_dir = config.AUDIO_RAW
        test_dir.mkdir(exist_ok=True)

        # Create test audio files
        durations = [1.0, 3.0, 5.0, 10.0]  # seconds
        for duration in durations:
            audio_path = test_dir / f"test_audio_{duration}s.wav"
            if not audio_path.exists():
                await self._create_test_audio(audio_path, duration)
            self.test_audio_files.append(str(audio_path))

    async def _create_test_audio(self, path: Path, duration: float):
        """Create synthetic test audio"""
        with wave.open(str(path), 'wb') as wav_file:
            channels = 1
            sample_width = 2
            framerate = 16000

            wav_file.setnchannels(channels)
            wav_file.setsampwidth(sample_width)
            wav_file.setframerate(framerate)

            # Generate sine wave for realistic audio
            num_frames = int(duration * framerate)
            frequency = 440  # A4 note

            for i in range(num_frames):
                value = int(32767 * 0.3 * np.sin(2 * np.pi * frequency * i / framerate))
                wav_file.writeframes(struct.pack('<h', value))

    async def run_latency_test(self, num_requests: int = 20) -> ComponentBenchmark:
        """Test STT latency across different audio lengths"""
        logger.info("stt_perf.latency_test_started", requests=num_requests)

        latencies = []
        errors = 0
        audio_durations = []
        throughput_ratios = []  # Ratio of real-time processing

        for i in range(num_requests):
            audio_file = self.test_audio_files[i % len(self.test_audio_files)]

            # Get audio duration
            with wave.open(audio_file, 'rb') as wav_file:
                frames = wav_file.getnframes()
                rate = wav_file.getframerate()
                audio_duration = frames / float(rate)
                audio_durations.append(audio_duration)

            start_time = time.time()

            try:
                result = await self.stt_service.transcribe_audio(audio_file)
                latency = time.time() - start_time
                latencies.append(latency)

                # Calculate real-time factor (lower is better)
                rtf = latency / audio_duration
                throughput_ratios.append(rtf)

                logger.debug("stt_perf.request_completed",
                           request=i+1,
                           latency=latency,
                           audio_duration=audio_duration,
                           rtf=rtf,
                           text_length=len(result.text) if result else 0)

            except Exception as e:
                errors += 1
                logger.warning("stt_perf.request_failed", request=i+1, error=str(e))

        # Calculate statistics
        success_rate = (num_requests - errors) / num_requests * 100
        avg_throughput = 1 / statistics.mean(latencies) if latencies else 0

        return ComponentBenchmark(
            component="STT",
            test_count=num_requests,
            latencies=latencies,
            throughput=avg_throughput,
            error_count=errors,
            success_rate=success_rate,
            additional_metrics={
                "avg_rtf": statistics.mean(throughput_ratios) if throughput_ratios else 0,
                "min_rtf": min(throughput_ratios) if throughput_ratios else 0,
                "max_rtf": max(throughput_ratios) if throughput_ratios else 0,
                "avg_audio_duration": statistics.mean(audio_durations)
            }
        )


class LLMPerformanceTest:
    """LLM (Language Model) performance testing"""

    def __init__(self):
        self.llm_service = None
        self.test_prompts = [
            "Hello, how are you today?",
            "What is the capital of France?",
            "Explain machine learning in simple terms.",
            "Tell me a short story about a robot.",
            "What are the benefits of renewable energy? Please provide a detailed explanation with examples.",
            "Can you help me write a Python function that sorts a list of numbers?"
        ]

    async def initialize(self):
        """Initialize LLM service"""
        logger.info("llm_perf.initializing")
        self.llm_service = await get_llm_service()

    async def run_latency_test(self, num_requests: int = 30) -> ComponentBenchmark:
        """Test LLM latency with various prompt types"""
        logger.info("llm_perf.latency_test_started", requests=num_requests)

        latencies = []
        ttft_latencies = []  # Time to First Token
        errors = 0
        token_counts = []
        throughput_scores = []  # Tokens per second

        for i in range(num_requests):
            prompt = self.test_prompts[i % len(self.test_prompts)]

            start_time = time.time()
            ttft = 0
            total_tokens = 0
            response_text = ""

            try:
                # Test streaming response for TTFT
                async for chunk in self.llm_service.generate_response_stream(prompt):
                    if ttft == 0:
                        ttft = time.time() - start_time
                        ttft_latencies.append(ttft)

                    # Count tokens (approximate)
                    if hasattr(chunk, 'content') and chunk.content:
                        response_text += chunk.content
                        total_tokens += len(chunk.content.split())
                    elif hasattr(chunk, 'text') and chunk.text:
                        response_text += chunk.text
                        total_tokens += len(chunk.text.split())

                total_latency = time.time() - start_time
                latencies.append(total_latency)
                token_counts.append(total_tokens)

                # Calculate tokens per second
                if total_latency > 0 and total_tokens > 0:
                    tps = total_tokens / total_latency
                    throughput_scores.append(tps)

                logger.debug("llm_perf.request_completed",
                           request=i+1,
                           latency=total_latency,
                           ttft=ttft,
                           tokens=total_tokens,
                           tps=throughput_scores[-1] if throughput_scores else 0)

            except Exception as e:
                errors += 1
                logger.warning("llm_perf.request_failed", request=i+1, error=str(e))

        success_rate = (num_requests - errors) / num_requests * 100
        avg_throughput = statistics.mean(throughput_scores) if throughput_scores else 0

        return ComponentBenchmark(
            component="LLM",
            test_count=num_requests,
            latencies=latencies,
            throughput=avg_throughput,
            error_count=errors,
            success_rate=success_rate,
            additional_metrics={
                "avg_ttft": statistics.mean(ttft_latencies) if ttft_latencies else 0,
                "min_ttft": min(ttft_latencies) if ttft_latencies else 0,
                "max_ttft": max(ttft_latencies) if ttft_latencies else 0,
                "avg_tokens": statistics.mean(token_counts) if token_counts else 0,
                "avg_tokens_per_second": avg_throughput
            }
        )


class TTSPerformanceTest:
    """TTS (Text-to-Speech) performance testing"""

    def __init__(self):
        self.tts_service = None
        self.test_texts = [
            "Hello world.",
            "The quick brown fox jumps over the lazy dog.",
            "This is a longer sentence to test the TTS performance with more complex text.",
            "In a hole in the ground there lived a hobbit. Not a nasty, dirty, wet hole filled with the ends of worms and an oozy smell, nor yet a dry, bare, sandy hole with nothing in it to sit down on or to eat.",
            "To be or not to be, that is the question. Whether 'tis nobler in the mind to suffer the slings and arrows of outrageous fortune, or to take arms against a sea of troubles and by opposing end them."
        ]

    async def initialize(self):
        """Initialize TTS service"""
        logger.info("tts_perf.initializing")
        self.tts_service = await get_tts_service()

    async def run_latency_test(self, num_requests: int = 25) -> ComponentBenchmark:
        """Test TTS latency across different text lengths"""
        logger.info("tts_perf.latency_test_started", requests=num_requests)

        latencies = []
        errors = 0
        text_lengths = []
        audio_durations = []
        synthesis_ratios = []  # Real-time factor for synthesis

        for i in range(num_requests):
            text = self.test_texts[i % len(self.test_texts)]
            text_length = len(text)
            text_lengths.append(text_length)

            start_time = time.time()

            try:
                result = await self.tts_service.synthesize_speech(
                    text=text,
                    voice_profile_id=None
                )

                latency = time.time() - start_time
                latencies.append(latency)

                # Estimate audio duration (approximate: 150 words per minute)
                estimated_words = len(text.split())
                estimated_audio_duration = estimated_words / 2.5  # 150 WPM = 2.5 WPS
                audio_durations.append(estimated_audio_duration)

                # Calculate synthesis ratio
                if estimated_audio_duration > 0:
                    ratio = latency / estimated_audio_duration
                    synthesis_ratios.append(ratio)

                logger.debug("tts_perf.request_completed",
                           request=i+1,
                           latency=latency,
                           text_length=text_length,
                           estimated_duration=estimated_audio_duration,
                           synthesis_ratio=ratio if estimated_audio_duration > 0 else 0)

            except Exception as e:
                errors += 1
                logger.warning("tts_perf.request_failed", request=i+1, error=str(e))

        success_rate = (num_requests - errors) / num_requests * 100
        avg_throughput = 1 / statistics.mean(latencies) if latencies else 0

        return ComponentBenchmark(
            component="TTS",
            test_count=num_requests,
            latencies=latencies,
            throughput=avg_throughput,
            error_count=errors,
            success_rate=success_rate,
            additional_metrics={
                "avg_synthesis_ratio": statistics.mean(synthesis_ratios) if synthesis_ratios else 0,
                "min_synthesis_ratio": min(synthesis_ratios) if synthesis_ratios else 0,
                "max_synthesis_ratio": max(synthesis_ratios) if synthesis_ratios else 0,
                "avg_text_length": statistics.mean(text_lengths) if text_lengths else 0,
                "avg_estimated_audio_duration": statistics.mean(audio_durations) if audio_durations else 0
            }
        )


class DatabasePerformanceTest:
    """Database operations performance testing"""

    async def initialize(self):
        """Initialize database connection"""
        logger.info("db_perf.initializing")

    async def run_latency_test(self, num_requests: int = 50) -> ComponentBenchmark:
        """Test database operation latency"""
        logger.info("db_perf.latency_test_started", requests=num_requests)

        latencies = []
        errors = 0

        for i in range(num_requests):
            start_time = time.time()

            try:
                # Test conversation creation (main database operation)
                conversation_id = await db.create_conversation(
                    user_audio_path=f"test_audio_{i}.wav",
                    user_text=f"Test message {i}",
                    ai_text=f"Test response {i}",
                    ai_audio_fast_path=f"test_response_{i}.wav",
                    session_id=f"test-session-{i}",
                    turn_number=i + 1,
                    voice_profile_id=None
                )

                latency = time.time() - start_time
                latencies.append(latency)

                logger.debug("db_perf.request_completed",
                           request=i+1,
                           latency=latency,
                           conversation_id=conversation_id)

            except Exception as e:
                errors += 1
                logger.warning("db_perf.request_failed", request=i+1, error=str(e))

        success_rate = (num_requests - errors) / num_requests * 100
        avg_throughput = 1 / statistics.mean(latencies) if latencies else 0

        return ComponentBenchmark(
            component="Database",
            test_count=num_requests,
            latencies=latencies,
            throughput=avg_throughput,
            error_count=errors,
            success_rate=success_rate,
            additional_metrics={
                "avg_query_time": statistics.mean(latencies) if latencies else 0
            }
        )


class ComponentPerformanceSuite:
    """Complete component performance testing suite"""

    def __init__(self):
        self.stt_test = STTPerformanceTest()
        self.llm_test = LLMPerformanceTest()
        self.tts_test = TTSPerformanceTest()
        self.db_test = DatabasePerformanceTest()

    async def run_full_suite(self, quick_mode: bool = False) -> Dict[str, Any]:
        """Run complete component performance suite"""
        logger.info("component_perf.suite_started", quick_mode=quick_mode)

        # Adjust test sizes for quick mode
        test_counts = {
            "stt": 10 if quick_mode else 20,
            "llm": 15 if quick_mode else 30,
            "tts": 12 if quick_mode else 25,
            "db": 25 if quick_mode else 50
        }

        results = {}

        # Initialize all services
        try:
            await self.stt_test.initialize()
            await self.llm_test.initialize()
            await self.tts_test.initialize()
            await self.db_test.initialize()
        except Exception as e:
            logger.error("component_perf.init_failed", error=str(e))
            raise

        # Run individual component tests
        try:
            logger.info("component_perf.testing_stt")
            results["stt"] = await self.stt_test.run_latency_test(test_counts["stt"])

            logger.info("component_perf.testing_llm")
            results["llm"] = await self.llm_test.run_latency_test(test_counts["llm"])

            logger.info("component_perf.testing_tts")
            results["tts"] = await self.tts_test.run_latency_test(test_counts["tts"])

            logger.info("component_perf.testing_database")
            results["db"] = await self.db_test.run_latency_test(test_counts["db"])

        except Exception as e:
            logger.error("component_perf.test_failed", error=str(e))
            raise

        # Analyze results
        analysis = self._analyze_component_results(results)

        logger.info("component_perf.suite_completed",
                   components_tested=len(results),
                   total_requests=sum(r.test_count for r in results.values()))

        return analysis

    def _analyze_component_results(self, results: Dict[str, ComponentBenchmark]) -> Dict[str, Any]:
        """Analyze component performance results"""

        component_analysis = {}

        for component_name, benchmark in results.items():
            if not benchmark.latencies:
                continue

            latencies = benchmark.latencies

            component_analysis[component_name] = {
                "latency_stats": {
                    "min": min(latencies),
                    "max": max(latencies),
                    "avg": statistics.mean(latencies),
                    "p50": float(np.percentile(latencies, 50)),
                    "p95": float(np.percentile(latencies, 95)),
                    "p99": float(np.percentile(latencies, 99))
                },
                "performance_metrics": {
                    "test_count": benchmark.test_count,
                    "success_rate": benchmark.success_rate,
                    "throughput": benchmark.throughput,
                    "error_count": benchmark.error_count
                },
                "component_specific": benchmark.additional_metrics
            }

        # Generate recommendations
        recommendations = self._generate_component_recommendations(component_analysis)

        return {
            "component_results": component_analysis,
            "recommendations": recommendations,
            "summary": self._generate_summary(component_analysis)
        }

    def _generate_component_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate optimization recommendations for each component"""
        recommendations = []

        for component, data in analysis.items():
            latency_p95 = data["latency_stats"]["p95"]

            if component == "stt":
                rtf = data["component_specific"].get("avg_rtf", 0)
                if rtf > 0.5:  # Real-time factor > 0.5 means slower than real-time
                    recommendations.append(f"STT optimization: Real-time factor {rtf:.2f} > 0.5, consider model optimization")

            elif component == "llm":
                ttft = data["component_specific"].get("avg_ttft", 0)
                if ttft > 0.8:  # Target: TTFT < 800ms
                    recommendations.append(f"LLM optimization: TTFT {ttft:.3f}s > 0.8s, check vLLM configuration")

            elif component == "tts":
                synthesis_ratio = data["component_specific"].get("avg_synthesis_ratio", 0)
                if synthesis_ratio > 1.0:
                    recommendations.append(f"TTS optimization: Synthesis ratio {synthesis_ratio:.2f} > 1.0, consider GPU optimization")

            elif component == "db" and latency_p95 > 0.05:  # 50ms
                recommendations.append(f"Database optimization: P95 latency {latency_p95*1000:.1f}ms > 50ms")

        return recommendations

    def _generate_summary(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate performance summary"""
        total_requests = sum(data["performance_metrics"]["test_count"] for data in analysis.values())
        avg_success_rate = statistics.mean([data["performance_metrics"]["success_rate"] for data in analysis.values()])

        # Find slowest component
        slowest_component = max(
            analysis.items(),
            key=lambda x: x[1]["latency_stats"]["p95"]
        )

        return {
            "total_requests": total_requests,
            "average_success_rate": avg_success_rate,
            "slowest_component": {
                "name": slowest_component[0],
                "p95_latency": slowest_component[1]["latency_stats"]["p95"]
            },
            "components_tested": len(analysis)
        }


async def run_component_performance_test(quick_mode: bool = False):
    """Run component performance testing"""
    configure_logging(development_mode=True)

    print(f"🧪 Starting Component Performance Testing ({'Quick' if quick_mode else 'Full'} mode)...")

    suite = ComponentPerformanceSuite()

    try:
        results = await suite.run_full_suite(quick_mode)

        print("\n📊 Component Performance Results:")

        for component, data in results["component_results"].items():
            print(f"\n{component.upper()}:")
            print(f"  P95 Latency: {data['latency_stats']['p95']:.3f}s")
            print(f"  Success Rate: {data['performance_metrics']['success_rate']:.1f}%")
            print(f"  Throughput: {data['performance_metrics']['throughput']:.1f} req/s")

            # Component-specific metrics
            if component == "stt":
                rtf = data["component_specific"].get("avg_rtf", 0)
                print(f"  Real-time Factor: {rtf:.3f}")
            elif component == "llm":
                ttft = data["component_specific"].get("avg_ttft", 0)
                tps = data["component_specific"].get("avg_tokens_per_second", 0)
                print(f"  Time to First Token: {ttft:.3f}s")
                print(f"  Tokens per Second: {tps:.1f}")
            elif component == "tts":
                ratio = data["component_specific"].get("avg_synthesis_ratio", 0)
                print(f"  Synthesis Ratio: {ratio:.3f}")

        # Summary
        summary = results["summary"]
        print(f"\n📈 Summary:")
        print(f"  Total Requests: {summary['total_requests']}")
        print(f"  Average Success Rate: {summary['average_success_rate']:.1f}%")
        print(f"  Slowest Component: {summary['slowest_component']['name'].upper()} ({summary['slowest_component']['p95_latency']:.3f}s)")

        # Recommendations
        if results["recommendations"]:
            print(f"\n💡 Recommendations:")
            for rec in results["recommendations"]:
                print(f"  • {rec}")

        return results

    except Exception as e:
        print(f"❌ Component testing failed: {e}")
        logger.error("component_perf.failed", error=str(e))
        raise


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Component Performance Testing")
    parser.add_argument("--quick", action="store_true", help="Run quick test mode")
    args = parser.parse_args()

    asyncio.run(run_component_performance_test(quick_mode=args.quick))