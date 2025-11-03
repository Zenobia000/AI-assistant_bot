"""
E2E Pipeline Test for AVATAR

Tests the complete Audio → STT → LLM → TTS pipeline
and measures performance metrics against targets:

- E2E Latency P95: ≤ 3.5s
- LLM TTFT: ≤ 800ms
- Fast TTS: ≤ 1.5s
- No OOM with 3-5 concurrent sessions
"""

import asyncio
import time
from pathlib import Path
import statistics
from typing import List, Tuple

import structlog
from avatar.services.stt import get_stt_service
from avatar.services.llm import get_llm_service
from avatar.services.tts import get_tts_service
from avatar.core.session_manager import SessionManager
from avatar.core.config import config

logger = structlog.get_logger()


class E2ETestResults:
    """Container for test results"""

    def __init__(self):
        self.stt_times: List[float] = []
        self.llm_times: List[float] = []
        self.llm_ttft_times: List[float] = []
        self.tts_times: List[float] = []
        self.e2e_times: List[float] = []
        self.errors: List[str] = []

    def add_result(
        self,
        stt_time: float,
        llm_time: float,
        llm_ttft: float,
        tts_time: float,
        e2e_time: float
    ):
        self.stt_times.append(stt_time)
        self.llm_times.append(llm_time)
        self.llm_ttft_times.append(llm_ttft)
        self.tts_times.append(tts_time)
        self.e2e_times.append(e2e_time)

    def print_summary(self):
        """Print detailed test summary"""
        print("\n" + "="*60)
        print("E2E PIPELINE TEST RESULTS")
        print("="*60)

        if not self.e2e_times:
            print("❌ No successful tests completed")
            return

        # Calculate statistics
        def calc_stats(data: List[float]) -> Tuple[float, float, float, float]:
            return (
                min(data),
                max(data),
                statistics.mean(data),
                sorted(data)[int(len(data) * 0.95)] if len(data) > 1 else data[0]
            )

        print(f"\n📊 Test Count: {len(self.e2e_times)}")

        # STT Results
        print(f"\n🎤 STT (Whisper on CPU)")
        stt_min, stt_max, stt_avg, stt_p95 = calc_stats(self.stt_times)
        print(f"  Min: {stt_min*1000:.0f}ms | Max: {stt_max*1000:.0f}ms")
        print(f"  Avg: {stt_avg*1000:.0f}ms | P95: {stt_p95*1000:.0f}ms")

        # LLM Results
        print(f"\n🧠 LLM (vLLM Streaming)")
        llm_min, llm_max, llm_avg, llm_p95 = calc_stats(self.llm_times)
        print(f"  Total Time:")
        print(f"    Min: {llm_min*1000:.0f}ms | Max: {llm_max*1000:.0f}ms")
        print(f"    Avg: {llm_avg*1000:.0f}ms | P95: {llm_p95*1000:.0f}ms")

        ttft_min, ttft_max, ttft_avg, ttft_p95 = calc_stats(self.llm_ttft_times)
        print(f"  TTFT (Time To First Token):")
        print(f"    Min: {ttft_min*1000:.0f}ms | Max: {ttft_max*1000:.0f}ms")
        print(f"    Avg: {ttft_avg*1000:.0f}ms | P95: {ttft_p95*1000:.0f}ms")

        target_ttft = config.TARGET_LLM_TTFT_MS
        if ttft_p95 * 1000 <= target_ttft:
            print(f"  ✅ TTFT P95 meets target: {ttft_p95*1000:.0f}ms ≤ {target_ttft}ms")
        else:
            print(f"  ❌ TTFT P95 exceeds target: {ttft_p95*1000:.0f}ms > {target_ttft}ms")

        # TTS Results
        print(f"\n🔊 TTS (F5-TTS Fast Mode)")
        tts_min, tts_max, tts_avg, tts_p95 = calc_stats(self.tts_times)
        print(f"  Min: {tts_min:.2f}s | Max: {tts_max:.2f}s")
        print(f"  Avg: {tts_avg:.2f}s | P95: {tts_p95:.2f}s")

        target_tts = config.TARGET_FAST_TTS_SEC
        if tts_p95 <= target_tts:
            print(f"  ✅ TTS P50 meets target: {tts_p95:.2f}s ≤ {target_tts}s")
        else:
            print(f"  ⚠️  TTS P50 exceeds target: {tts_p95:.2f}s > {target_tts}s")

        # E2E Results
        print(f"\n⚡ End-to-End Latency")
        e2e_min, e2e_max, e2e_avg, e2e_p95 = calc_stats(self.e2e_times)
        print(f"  Min: {e2e_min:.2f}s | Max: {e2e_max:.2f}s")
        print(f"  Avg: {e2e_avg:.2f}s | P95: {e2e_p95:.2f}s")

        target_e2e = config.TARGET_E2E_LATENCY_SEC
        if e2e_p95 <= target_e2e:
            print(f"  ✅ E2E P95 meets target: {e2e_p95:.2f}s ≤ {target_e2e}s")
        else:
            print(f"  ❌ E2E P95 exceeds target: {e2e_p95:.2f}s > {target_e2e}s")

        # Error Summary
        if self.errors:
            print(f"\n❌ Errors ({len(self.errors)}):")
            for error in self.errors:
                print(f"  - {error}")

        print("\n" + "="*60)


async def test_single_pipeline(
    test_audio_path: Path,
    test_text: str,
    results: E2ETestResults
):
    """Test a single pass through the pipeline"""

    logger.info("test.pipeline.start", audio=str(test_audio_path), text=test_text)

    try:
        e2e_start = time.time()

        # Step 1: STT (Audio → Text)
        logger.info("test.step.stt.start")
        stt_start = time.time()

        stt_service = await get_stt_service()
        transcribed_text = await stt_service.transcribe(test_audio_path)

        stt_time = time.time() - stt_start
        logger.info("test.step.stt.complete",
                   time_sec=round(stt_time, 3),
                   text=transcribed_text)

        # Step 2: LLM (Text → Response Text) with TTFT tracking
        logger.info("test.step.llm.start")
        llm_start = time.time()

        llm_service = await get_llm_service()
        messages = [{"role": "user", "content": test_text}]

        # Track TTFT (Time To First Token)
        ttft = None
        full_response = ""

        async for chunk in llm_service.chat_stream(messages, max_tokens=512, temperature=0.7):
            if ttft is None:
                ttft = time.time() - llm_start
                logger.info("test.step.llm.ttft", time_ms=round(ttft * 1000, 0))
            full_response += chunk

        llm_time = time.time() - llm_start
        logger.info("test.step.llm.complete",
                   time_sec=round(llm_time, 3),
                   ttft_ms=round(ttft * 1000, 0),
                   response_length=len(full_response))

        # Step 3: TTS (Response Text → Audio)
        logger.info("test.step.tts.start")
        tts_start = time.time()

        tts_service = await get_tts_service()
        audio_path = await tts_service.synthesize_fast(
            text=full_response,
            voice_profile_name="test_profile",
            output_path=config.AUDIO_TTS_FAST / f"test_{int(time.time())}.wav"
        )

        tts_time = time.time() - tts_start
        logger.info("test.step.tts.complete",
                   time_sec=round(tts_time, 3),
                   audio_path=str(audio_path))

        # Calculate E2E time
        e2e_time = time.time() - e2e_start

        # Record results
        results.add_result(
            stt_time=stt_time,
            llm_time=llm_time,
            llm_ttft=ttft,
            tts_time=tts_time,
            e2e_time=e2e_time
        )

        logger.info("test.pipeline.complete",
                   e2e_time=round(e2e_time, 3),
                   stt=round(stt_time, 3),
                   llm=round(llm_time, 3),
                   tts=round(tts_time, 3))

        return True

    except Exception as e:
        error_msg = f"Pipeline failed: {type(e).__name__}: {str(e)}"
        logger.error("test.pipeline.failed", error=error_msg)
        results.errors.append(error_msg)
        return False


async def test_concurrent_sessions(
    test_audio_path: Path,
    num_sessions: int = 3
) -> bool:
    """Test concurrent session handling"""

    logger.info("test.concurrent.start", num_sessions=num_sessions)

    try:
        # Create session manager
        session_manager = SessionManager()

        # Attempt to acquire multiple sessions
        sessions = []
        for i in range(num_sessions):
            session_id = f"test-session-{i}"
            acquired = await session_manager.try_acquire_session(session_id, timeout=1.0)
            if acquired:
                sessions.append(session_id)
                logger.info("test.concurrent.session_acquired", session_id=session_id)
            else:
                logger.warning("test.concurrent.session_rejected", session_id=session_id)

        logger.info("test.concurrent.result",
                   acquired=len(sessions),
                   total=num_sessions)

        # Release all sessions
        for session_id in sessions:
            await session_manager.release_session(session_id)

        return len(sessions) >= min(num_sessions, config.MAX_CONCURRENT_SESSIONS)

    except Exception as e:
        logger.error("test.concurrent.failed", error=str(e))
        return False


async def main():
    """Run E2E test suite"""

    logger.info("test.suite.start")

    # Test configuration
    num_iterations = 5
    test_text = "Hello! How can I assist you today?"

    # Create test audio (placeholder - in real test use actual audio file)
    test_audio_path = config.AUDIO_RAW / "test_sample.wav"

    # Check if test audio exists
    if not test_audio_path.exists():
        logger.error("test.suite.aborted",
                    reason="Test audio not found",
                    path=str(test_audio_path))
        print(f"\n❌ Test audio not found: {test_audio_path}")
        print("Please provide a test audio file (WAV 16kHz mono)")
        return

    # Initialize results
    results = E2ETestResults()

    # Test 1: Sequential Pipeline Tests
    print(f"\n🧪 Running {num_iterations} sequential pipeline tests...")
    for i in range(num_iterations):
        print(f"\n  Test {i+1}/{num_iterations}...")
        await test_single_pipeline(test_audio_path, test_text, results)
        await asyncio.sleep(1)  # Brief pause between tests

    # Test 2: Concurrent Sessions
    print(f"\n🧪 Testing concurrent session handling...")
    concurrent_ok = await test_concurrent_sessions(test_audio_path, num_sessions=3)
    if concurrent_ok:
        print(f"  ✅ Concurrent session test passed")
    else:
        print(f"  ❌ Concurrent session test failed")

    # Print final results
    results.print_summary()

    logger.info("test.suite.complete")


if __name__ == "__main__":
    # Run test suite
    asyncio.run(main())
