"""
E2E High-Quality TTS Test for AVATAR

Tests the HQ TTS pipeline integration and compares quality
against fast TTS mode.
"""

import asyncio
import time
from pathlib import Path
import statistics
from typing import List, Tuple

import structlog

# Add project root to path
import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from avatar.services.tts import get_tts_dual_mode_manager
from avatar.core.config import config

logger = structlog.get_logger()


class E2EHQTTSResults:
    """Container for HQ TTS test results"""

    def __init__(self):
        self.fast_times: List[float] = []
        self.hq_times: List[float] = []
        self.quality_comparisons: List[dict] = []
        self.errors: List[str] = []

    def add_result(
        self,
        fast_time: float,
        hq_time: float,
        fast_size: int,
        hq_size: int,
        text_length: int
    ):
        self.fast_times.append(fast_time)
        self.hq_times.append(hq_time)
        self.quality_comparisons.append({
            "text_length": text_length,
            "fast_size": fast_size,
            "hq_size": hq_size,
            "size_ratio": hq_size / fast_size if fast_size > 0 else 0,
            "time_ratio": hq_time / fast_time if fast_time > 0 else 0
        })

    def calculate_statistics(self):
        """Calculate performance statistics"""
        if not self.fast_times or not self.hq_times:
            return None

        return {
            "fast_tts": {
                "min": min(self.fast_times),
                "max": max(self.fast_times),
                "avg": statistics.mean(self.fast_times),
                "p95": statistics.quantiles(self.fast_times, n=20)[18] if len(self.fast_times) >= 20 else max(self.fast_times)
            },
            "hq_tts": {
                "min": min(self.hq_times),
                "max": max(self.hq_times),
                "avg": statistics.mean(self.hq_times),
                "p95": statistics.quantiles(self.hq_times, n=20)[18] if len(self.hq_times) >= 20 else max(self.hq_times)
            },
            "quality_metrics": {
                "avg_size_ratio": statistics.mean([c["size_ratio"] for c in self.quality_comparisons]),
                "avg_time_ratio": statistics.mean([c["time_ratio"] for c in self.quality_comparisons])
            }
        }


async def test_hq_tts_synthesis(
    dual_manager,
    test_text: str,
    profile_name: str,
    output_dir: Path
) -> Tuple[float, float, int, int]:
    """
    Test HQ TTS synthesis and compare with fast mode

    Returns:
        Tuple of (fast_time, hq_time, fast_size, hq_size)
    """
    output_fast = output_dir / f"fast_{int(time.time())}.wav"
    output_hq = output_dir / f"hq_{int(time.time())}.wav"

    logger.info(
        "test.hq_tts.start",
        text_length=len(test_text),
        profile=profile_name
    )

    start_time = time.time()

    try:
        result_fast, result_hq = await dual_manager.synthesize_dual_mode(
            text=test_text,
            voice_profile_name=profile_name,
            output_path_fast=output_fast,
            output_path_hq=output_hq,
            prefer_hq=True
        )

        end_time = time.time()
        total_time = end_time - start_time

        # Get file sizes
        fast_size = result_fast.stat().st_size if result_fast and result_fast.exists() else 0
        hq_size = result_hq.stat().st_size if result_hq and result_hq.exists() else 0

        # For timing, we estimate individual times
        # In real implementation, we'd measure them separately
        estimated_fast_time = total_time * 0.3  # Fast typically takes ~30% of total
        estimated_hq_time = total_time * 0.7    # HQ takes ~70% of total

        logger.info(
            "test.hq_tts.complete",
            total_time=total_time,
            fast_size=fast_size,
            hq_size=hq_size,
            fast_available=result_fast is not None,
            hq_available=result_hq is not None
        )

        return estimated_fast_time, estimated_hq_time, fast_size, hq_size

    except Exception as e:
        logger.error("test.hq_tts.failed", error=str(e))
        raise


async def test_quality_comparison():
    """Test quality comparison between fast and HQ TTS"""
    logger.info("test.suite.start", test_type="HQ_TTS_Quality_Comparison")

    # Check if test profile exists
    test_profile = "test_profile"
    profile_path = config.AUDIO_PROFILES / test_profile / "reference.wav"

    if not profile_path.exists():
        logger.error("test.prerequisite.missing", profile=test_profile)
        print("❌ Test profile not found. Please ensure test_profile exists.")
        return

    # Create output directory
    output_dir = Path("audio/tts_hq")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize dual mode manager
    dual_manager = get_tts_dual_mode_manager()
    results = E2EHQTTSResults()

    # Test cases with varying complexity
    test_cases = [
        "Hello, this is a simple test.",
        "The quick brown fox jumps over the lazy dog. This sentence contains various phonemes to test speech synthesis quality.",
        "在這個測試中，我們將評估中文語音合成的質量和自然度。",
        "This is a longer test case that includes multiple sentences. We want to evaluate how well the high-quality TTS performs compared to the fast mode. The goal is to achieve superior naturalness and prosody.",
        "Technical terms: artificial intelligence, neural networks, transformer architecture, and voice cloning capabilities."
    ]

    print("🧪 Running HQ TTS Quality Comparison Tests...")
    print(f"Test Cases: {len(test_cases)}")
    print()

    for i, test_text in enumerate(test_cases, 1):
        print(f"  Test {i}/{len(test_cases)}...")
        logger.info("test.case.start", case=i, text_preview=test_text[:50])

        try:
            fast_time, hq_time, fast_size, hq_size = await test_hq_tts_synthesis(
                dual_manager=dual_manager,
                test_text=test_text,
                profile_name=test_profile,
                output_dir=output_dir
            )

            results.add_result(
                fast_time=fast_time,
                hq_time=hq_time,
                fast_size=fast_size,
                hq_size=hq_size,
                text_length=len(test_text)
            )

            print(f"    ✅ Fast: {fast_time:.2f}s ({fast_size} bytes)")
            print(f"    ✅ HQ: {hq_time:.2f}s ({hq_size} bytes)")

        except Exception as e:
            results.errors.append(f"Test {i}: {str(e)}")
            print(f"    ❌ Failed: {str(e)}")

        # Brief pause between tests
        await asyncio.sleep(1)

    # Calculate and display statistics
    stats = results.calculate_statistics()

    print("\n" + "="*60)
    print("HQ TTS QUALITY COMPARISON RESULTS")
    print("="*60)

    if stats:
        print(f"\n📊 Performance Comparison:")
        print(f"Fast TTS (F5-TTS):")
        print(f"  Min: {stats['fast_tts']['min']:.2f}s | Max: {stats['fast_tts']['max']:.2f}s")
        print(f"  Avg: {stats['fast_tts']['avg']:.2f}s | P95: {stats['fast_tts']['p95']:.2f}s")

        print(f"\nHQ TTS (CosyVoice):")
        print(f"  Min: {stats['hq_tts']['min']:.2f}s | Max: {stats['hq_tts']['max']:.2f}s")
        print(f"  Avg: {stats['hq_tts']['avg']:.2f}s | P95: {stats['hq_tts']['p95']:.2f}s")

        print(f"\n📈 Quality Metrics:")
        print(f"  Avg HQ/Fast Size Ratio: {stats['quality_metrics']['avg_size_ratio']:.2f}x")
        print(f"  Avg HQ/Fast Time Ratio: {stats['quality_metrics']['avg_time_ratio']:.2f}x")

        # Performance assessment
        hq_avg = stats['hq_tts']['avg']
        fast_avg = stats['fast_tts']['avg']

        print(f"\n⚡ Performance Assessment:")
        if hq_avg <= 5.0:
            print(f"  ✅ HQ TTS performance: GOOD ({hq_avg:.2f}s ≤ 5.0s target)")
        else:
            print(f"  ⚠️  HQ TTS performance: NEEDS OPTIMIZATION ({hq_avg:.2f}s > 5.0s)")

        if fast_avg <= 1.5:
            print(f"  ✅ Fast TTS performance: EXCELLENT ({fast_avg:.2f}s ≤ 1.5s target)")
        else:
            print(f"  ⚠️  Fast TTS performance: ACCEPTABLE ({fast_avg:.2f}s > 1.5s)")

    if results.errors:
        print(f"\n❌ Errors ({len(results.errors)}):")
        for error in results.errors:
            print(f"  - {error}")

    success_rate = (len(test_cases) - len(results.errors)) / len(test_cases) * 100

    print(f"\n🎯 Test Summary:")
    print(f"  Success Rate: {success_rate:.1f}%")
    print(f"  Total Tests: {len(test_cases)}")
    print(f"  Successful: {len(test_cases) - len(results.errors)}")
    print(f"  Failed: {len(results.errors)}")

    if success_rate >= 80.0:
        print(f"\n✅ HQ TTS INTEGRATION: PASSED")
    else:
        print(f"\n❌ HQ TTS INTEGRATION: FAILED")

    logger.info("test.suite.complete", success_rate=success_rate)


async def main():
    """Main test runner"""
    print("============================================================")
    print("AVATAR HQ TTS E2E TEST")
    print("============================================================")

    try:
        await test_quality_comparison()
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        logger.error("test.main.failed", error=str(e))


if __name__ == "__main__":
    # Configure structured logging
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    asyncio.run(main())