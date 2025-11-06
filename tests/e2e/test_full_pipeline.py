"""
Full End-to-End Pipeline Test for AVATAR

Tests the complete voice conversation pipeline:
Audio Input → STT → LLM → TTS → Audio Output

This is the real-world test that matters - can the user have a conversation?
"""

import asyncio
import time
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import structlog
from avatar.services.stt import get_stt_service
from avatar.services.llm import get_llm_service
from avatar.services.tts import get_tts_service, get_tts_dual_mode_manager
from avatar.services.tts_hq import get_tts_hq_service
from avatar.core.config import config

logger = structlog.get_logger()


class E2EPipelineTest:
    """
    End-to-End Pipeline Test Framework

    Tests the complete AVATAR conversation flow with performance metrics.
    """

    def __init__(self):
        self.test_audio = config.AUDIO_RAW / "test_sample.wav"
        self.test_profile = "test_profile"
        self.metrics = {}

    async def test_stt_llm_tts_fast_pipeline(self) -> Dict[str, Any]:
        """Test STT → LLM → TTS Fast complete pipeline"""
        print("\n🎤 Testing STT → LLM → TTS Fast Pipeline...")

        start_total = time.time()
        pipeline_metrics = {
            "pipeline": "STT → LLM → TTS Fast",
            "success": False,
            "total_time": 0,
            "stt_time": 0,
            "llm_time": 0,
            "tts_time": 0,
            "audio_duration": 0,
            "rtf": 0,
            "error": None
        }

        try:
            # Step 1: STT (Speech-to-Text)
            print("  🎤 Step 1: Speech-to-Text...")
            stt_service = await get_stt_service()

            start_stt = time.time()
            stt_result = await stt_service.transcribe(self.test_audio)
            stt_time = time.time() - start_stt

            # Extract text from STT result
            if isinstance(stt_result, tuple):
                text, metadata = stt_result
                print(f"     STT Result: '{text}'")
                print(f"     Language: {metadata.get('language', 'unknown')}")
            else:
                text = str(stt_result)
                print(f"     STT Result: '{text}'")

            pipeline_metrics["stt_time"] = stt_time
            print(f"     STT Time: {stt_time:.2f}s")

            # Step 2: LLM (Language Model)
            print("  🧠 Step 2: Language Model...")
            llm_service = await get_llm_service()

            # Create conversation prompt
            messages = [{
                "role": "user",
                "content": f"Please respond to this in English with a single sentence: {text}"
            }]

            start_llm = time.time()
            llm_response = ""
            ttft = None  # Time to first token

            async for chunk in llm_service.chat_stream(messages, max_tokens=50, temperature=0.7):
                if ttft is None:
                    ttft = time.time() - start_llm
                llm_response += chunk

            llm_time = time.time() - start_llm
            pipeline_metrics["llm_time"] = llm_time
            pipeline_metrics["ttft"] = ttft

            print(f"     LLM Response: '{llm_response}'")
            print(f"     LLM Time: {llm_time:.2f}s (TTFT: {ttft*1000:.0f}ms)")

            # Step 3: TTS Fast (Text-to-Speech)
            print("  🔊 Step 3: Text-to-Speech (Fast)...")
            tts_service = await get_tts_service()

            output_path = config.AUDIO_TTS_FAST / f"e2e_fast_test_{int(time.time())}.wav"

            start_tts = time.time()
            result_path = await tts_service.synthesize_fast(
                text=llm_response,
                voice_profile_name=self.test_profile,
                output_path=output_path
            )
            tts_time = time.time() - start_tts

            pipeline_metrics["tts_time"] = tts_time
            print(f"     TTS Time: {tts_time:.2f}s")

            # Calculate total metrics
            total_time = time.time() - start_total
            pipeline_metrics["total_time"] = total_time
            pipeline_metrics["success"] = True

            # Calculate audio duration and RTF
            try:
                import torchaudio
                audio, sr = torchaudio.load(result_path)
                audio_duration = audio.shape[-1] / sr
                pipeline_metrics["audio_duration"] = audio_duration
                pipeline_metrics["rtf"] = total_time / audio_duration if audio_duration > 0 else float('inf')
            except Exception:
                pipeline_metrics["audio_duration"] = 0
                pipeline_metrics["rtf"] = float('inf')

            print(f"  ✅ Fast Pipeline Success!")
            print(f"     Total Time: {total_time:.2f}s")
            print(f"     Audio Duration: {pipeline_metrics['audio_duration']:.2f}s")
            print(f"     Real-Time Factor: {pipeline_metrics['rtf']:.2f}")
            print(f"     Output: {result_path}")

        except Exception as e:
            pipeline_metrics["error"] = str(e)
            pipeline_metrics["total_time"] = time.time() - start_total
            print(f"  ❌ Fast Pipeline Failed: {e}")

        return pipeline_metrics

    async def test_stt_llm_tts_hq_pipeline(self) -> Dict[str, Any]:
        """Test STT → LLM → TTS HQ complete pipeline"""
        print("\n🎤 Testing STT → LLM → TTS HQ Pipeline...")

        start_total = time.time()
        pipeline_metrics = {
            "pipeline": "STT → LLM → TTS HQ",
            "success": False,
            "total_time": 0,
            "stt_time": 0,
            "llm_time": 0,
            "tts_time": 0,
            "audio_duration": 0,
            "rtf": 0,
            "error": None
        }

        try:
            # Step 1: STT (reuse logic)
            print("  🎤 Step 1: Speech-to-Text...")
            stt_service = await get_stt_service()

            start_stt = time.time()
            stt_result = await stt_service.transcribe(self.test_audio)
            stt_time = time.time() - start_stt

            if isinstance(stt_result, tuple):
                text, metadata = stt_result
            else:
                text = str(stt_result)

            pipeline_metrics["stt_time"] = stt_time
            print(f"     STT Time: {stt_time:.2f}s")

            # Step 2: LLM (reuse logic)
            print("  🧠 Step 2: Language Model...")
            llm_service = await get_llm_service()

            messages = [{
                "role": "user",
                "content": f"Please respond to this in English with a single sentence: {text}"
            }]

            start_llm = time.time()
            llm_response = ""
            ttft = None

            async for chunk in llm_service.chat_stream(messages, max_tokens=50, temperature=0.7):
                if ttft is None:
                    ttft = time.time() - start_llm
                llm_response += chunk

            llm_time = time.time() - start_llm
            pipeline_metrics["llm_time"] = llm_time
            pipeline_metrics["ttft"] = ttft

            print(f"     LLM Time: {llm_time:.2f}s")

            # Step 3: TTS HQ (High Quality)
            print("  🔊 Step 3: Text-to-Speech (High Quality)...")
            tts_hq_service = get_tts_hq_service()

            output_path = config.AUDIO_TTS_HQ / f"e2e_hq_test_{int(time.time())}.wav"

            start_tts = time.time()
            result_path = await tts_hq_service.synthesize_hq(
                text=llm_response,
                voice_profile_name=self.test_profile,
                output_path=output_path
            )
            tts_time = time.time() - start_tts

            pipeline_metrics["tts_time"] = tts_time
            print(f"     TTS HQ Time: {tts_time:.2f}s")

            # Calculate metrics
            total_time = time.time() - start_total
            pipeline_metrics["total_time"] = total_time
            pipeline_metrics["success"] = True

            # Audio metrics
            try:
                import torchaudio
                audio, sr = torchaudio.load(result_path)
                audio_duration = audio.shape[-1] / sr
                pipeline_metrics["audio_duration"] = audio_duration
                pipeline_metrics["rtf"] = total_time / audio_duration if audio_duration > 0 else float('inf')
            except Exception:
                pipeline_metrics["audio_duration"] = 0
                pipeline_metrics["rtf"] = float('inf')

            print(f"  ✅ HQ Pipeline Success!")
            print(f"     Total Time: {total_time:.2f}s")
            print(f"     Audio Duration: {pipeline_metrics['audio_duration']:.2f}s")
            print(f"     Real-Time Factor: {pipeline_metrics['rtf']:.2f}")
            print(f"     Output: {result_path}")

        except Exception as e:
            pipeline_metrics["error"] = str(e)
            pipeline_metrics["total_time"] = time.time() - start_total
            print(f"  ❌ HQ Pipeline Failed: {e}")

        return pipeline_metrics

    async def test_dual_mode_pipeline(self) -> Dict[str, Any]:
        """Test dual-mode TTS switching"""
        print("\n🎤 Testing Dual-Mode TTS Pipeline...")

        start_total = time.time()
        pipeline_metrics = {
            "pipeline": "STT → LLM → TTS Dual-Mode",
            "success": False,
            "total_time": 0,
            "fast_path": None,
            "hq_path": None,
            "error": None
        }

        try:
            # STT + LLM (same as above)
            stt_service = await get_stt_service()
            stt_result = await stt_service.transcribe(self.test_audio)

            if isinstance(stt_result, tuple):
                text, _ = stt_result
            else:
                text = str(stt_result)

            llm_service = await get_llm_service()
            messages = [{"role": "user", "content": f"Respond briefly: {text}"}]

            llm_response = ""
            async for chunk in llm_service.chat_stream(messages, max_tokens=30, temperature=0.7):
                llm_response += chunk

            print(f"     LLM Response: '{llm_response}'")

            # Dual-mode TTS
            print("  🔊 Step 3: Dual-Mode TTS...")
            dual_manager = get_tts_dual_mode_manager()

            fast_output = config.AUDIO_TTS_FAST / f"e2e_dual_fast_{int(time.time())}.wav"
            hq_output = config.AUDIO_TTS_HQ / f"e2e_dual_hq_{int(time.time())}.wav"

            fast_path, hq_path = await dual_manager.synthesize_dual_mode(
                text=llm_response,
                voice_profile_name=self.test_profile,
                output_path_fast=fast_output,
                output_path_hq=hq_output,
                prefer_hq=True
            )

            pipeline_metrics["fast_path"] = str(fast_path) if fast_path else None
            pipeline_metrics["hq_path"] = str(hq_path) if hq_path else None
            pipeline_metrics["success"] = fast_path is not None or hq_path is not None

            total_time = time.time() - start_total
            pipeline_metrics["total_time"] = total_time

            print(f"  ✅ Dual-Mode Success!")
            print(f"     Fast Output: {fast_path}")
            print(f"     HQ Output: {hq_path}")
            print(f"     Total Time: {total_time:.2f}s")

        except Exception as e:
            pipeline_metrics["error"] = str(e)
            pipeline_metrics["total_time"] = time.time() - start_total
            print(f"  ❌ Dual-Mode Pipeline Failed: {e}")

        return pipeline_metrics

    async def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run all E2E pipeline tests"""
        print("🚀 Starting Comprehensive End-to-End Pipeline Tests")
        print("=" * 70)

        # Check prerequisites
        if not self.test_audio.exists():
            raise FileNotFoundError(f"Test audio not found: {self.test_audio}")

        # Run all pipeline tests
        results = {}

        # Test 1: Fast Pipeline
        results["fast_pipeline"] = await self.test_stt_llm_tts_fast_pipeline()

        # Test 2: HQ Pipeline
        results["hq_pipeline"] = await self.test_stt_llm_tts_hq_pipeline()

        # Test 3: Dual-Mode Pipeline
        results["dual_pipeline"] = await self.test_dual_mode_pipeline()

        return results

    def generate_performance_report(self, results: Dict[str, Any]) -> str:
        """Generate a comprehensive performance report"""
        report = []
        report.append("\n" + "="*70)
        report.append("📊 AVATAR END-TO-END PERFORMANCE REPORT")
        report.append("="*70)

        # Summary table
        report.append("\n📋 Pipeline Summary:")
        report.append("-" * 50)

        for test_name, metrics in results.items():
            status = "✅ PASS" if metrics.get("success", False) else "❌ FAIL"
            total_time = metrics.get("total_time", 0)
            pipeline_name = metrics.get("pipeline", test_name)

            report.append(f"{pipeline_name:30s} {status:8s} {total_time:6.2f}s")

        # Detailed metrics
        report.append("\n📊 Detailed Performance Metrics:")
        report.append("-" * 50)

        for test_name, metrics in results.items():
            if not metrics.get("success", False):
                continue

            report.append(f"\n{metrics.get('pipeline', test_name)}:")

            if "stt_time" in metrics:
                report.append(f"  STT Time:        {metrics['stt_time']:.2f}s")
            if "llm_time" in metrics:
                report.append(f"  LLM Time:        {metrics['llm_time']:.2f}s")
            if "ttft" in metrics:
                report.append(f"  TTFT:            {metrics['ttft']*1000:.0f}ms")
            if "tts_time" in metrics:
                report.append(f"  TTS Time:        {metrics['tts_time']:.2f}s")
            if "total_time" in metrics:
                report.append(f"  Total Time:      {metrics['total_time']:.2f}s")
            if "audio_duration" in metrics and metrics["audio_duration"] > 0:
                report.append(f"  Audio Duration:  {metrics['audio_duration']:.2f}s")
                report.append(f"  RTF:             {metrics['rtf']:.2f}")

        # Performance targets comparison
        report.append(f"\n🎯 Performance vs Targets:")
        report.append("-" * 50)

        target_e2e = config.TARGET_E2E_LATENCY_SEC
        target_ttft = config.TARGET_LLM_TTFT_MS / 1000
        target_tts = config.TARGET_FAST_TTS_SEC

        for test_name, metrics in results.items():
            if not metrics.get("success", False):
                continue

            total_time = metrics.get("total_time", 0)
            ttft = metrics.get("ttft", 0)
            tts_time = metrics.get("tts_time", 0)

            e2e_status = "✅" if total_time <= target_e2e else "⚠️"
            ttft_status = "✅" if ttft <= target_ttft else "⚠️"

            if "fast" in test_name.lower():
                tts_status = "✅" if tts_time <= target_tts else "⚠️"
            else:
                tts_status = "🎯"  # HQ TTS has different expectations

            report.append(f"{metrics.get('pipeline', test_name)}:")
            report.append(f"  E2E Latency: {e2e_status} {total_time:.2f}s (target: ≤{target_e2e}s)")
            report.append(f"  TTFT:        {ttft_status} {ttft*1000:.0f}ms (target: ≤{target_ttft*1000:.0f}ms)")
            report.append(f"  TTS:         {tts_status} {tts_time:.2f}s")

        # Error summary
        errors = [(name, metrics.get("error")) for name, metrics in results.items()
                 if metrics.get("error")]

        if errors:
            report.append(f"\n❌ Errors Encountered:")
            report.append("-" * 50)
            for test_name, error in errors:
                report.append(f"{test_name}: {error}")

        # Conclusion
        passed_tests = sum(1 for metrics in results.values() if metrics.get("success", False))
        total_tests = len(results)

        report.append(f"\n🏆 Test Results: {passed_tests}/{total_tests} pipelines passed")

        if passed_tests == total_tests:
            report.append("🎉 All pipelines working! AVATAR is ready for conversation!")
        else:
            report.append("⚠️ Some pipelines failed. Check errors above.")

        report.append("="*70)
        return "\n".join(report)


async def main():
    """Run comprehensive end-to-end pipeline tests"""
    try:
        # Initialize test framework
        e2e_test = E2EPipelineTest()

        # Run comprehensive tests
        results = await e2e_test.run_comprehensive_test()

        # Generate and display report
        report = e2e_test.generate_performance_report(results)
        print(report)

        # Return success status
        passed = sum(1 for metrics in results.values() if metrics.get("success", False))
        total = len(results)

        return passed == total

    except Exception as e:
        print(f"\n❌ E2E Test Framework Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)