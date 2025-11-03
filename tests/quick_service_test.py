"""
Quick service test - verifies individual AI services work

Tests each service independently before running full E2E pipeline.
"""

import sys
from pathlib import Path

# Add src to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import asyncio
import time
from avatar.core.config import config

print("\n" + "="*60)
print("QUICK SERVICE TEST")
print("="*60 + "\n")


async def test_stt_service():
    """Test STT (Speech-to-Text) service"""
    print("🎤 Testing STT Service (Whisper)...")

    try:
        from avatar.services.stt import get_stt_service

        test_audio = config.AUDIO_RAW / "test_sample.wav"

        if not test_audio.exists():
            print(f"  ❌ Test audio not found: {test_audio}")
            return False

        stt = await get_stt_service()

        start = time.time()
        result = await stt.transcribe(test_audio)
        elapsed = time.time() - start

        print(f"  ✅ STT Service OK")
        print(f"     Time: {elapsed:.2f}s")
        print(f"     Result: '{result}'")

        return True

    except Exception as e:
        print(f"  ❌ STT Service Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_llm_service():
    """Test LLM service"""
    print("\n🧠 Testing LLM Service (vLLM)...")

    try:
        from avatar.services.llm import get_llm_service

        llm = await get_llm_service()

        messages = [{"role": "user", "content": "Say hello in one sentence."}]

        start = time.time()
        ttft = None
        full_response = ""

        async for chunk in llm.chat_stream(messages, max_tokens=50, temperature=0.7):
            if ttft is None:
                ttft = time.time() - start
            full_response += chunk

        elapsed = time.time() - start

        print(f"  ✅ LLM Service OK")
        print(f"     TTFT: {ttft*1000:.0f}ms")
        print(f"     Total Time: {elapsed:.2f}s")
        print(f"     Response: '{full_response[:100]}...'")

        return True

    except Exception as e:
        print(f"  ❌ LLM Service Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_tts_service():
    """Test TTS service"""
    print("\n🔊 Testing TTS Service (F5-TTS)...")

    try:
        from avatar.services.tts import TTSService
        import torchaudio

        # Create TTS service instance directly (GPU mode)
        tts = TTSService()

        test_text = "Hello, this is a test."

        # Use direct synthesis (not voice profile method for this test)
        # Generate a dummy reference audio for testing
        ref_audio_path = config.AUDIO_RAW / "test_sample.wav"
        ref_text = "Test reference text"
        output_path = config.AUDIO_TTS_FAST / f"test_{int(time.time())}.wav"

        print(f"  ⚠️  Note: TTS requires reference audio for voice cloning")
        print(f"     Using test audio as reference: {ref_audio_path}")

        start = time.time()
        audio_path = await tts.synthesize(
            text=test_text,
            ref_audio_path=ref_audio_path,
            ref_text=ref_text,
            output_path=output_path
        )
        elapsed = time.time() - start

        print(f"  ✅ TTS Service OK")
        print(f"     Time: {elapsed:.2f}s")
        print(f"     Output: {audio_path}")
        print(f"     Size: {audio_path.stat().st_size / 1024:.1f} KB")

        return True

    except Exception as e:
        print(f"  ❌ TTS Service Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_vram_monitoring():
    """Test VRAM monitoring"""
    print("\n📊 Testing VRAM Monitoring...")

    try:
        from avatar.core.session_manager import SessionManager

        manager = SessionManager()
        vram_info = manager.get_vram_status()

        print(f"  ✅ VRAM Monitoring OK")
        print(f"     Total: {vram_info['total_gb']:.2f} GB")
        print(f"     Used: {vram_info['used_gb']:.2f} GB")
        print(f"     Free: {vram_info['free_gb']:.2f} GB")
        print(f"     Usage: {vram_info['usage_percent']:.1f}%")

        return True

    except Exception as e:
        print(f"  ❌ VRAM Monitoring Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all quick tests"""

    results = {}

    # Test 1: VRAM monitoring (fastest, no model loading)
    results['vram'] = await test_vram_monitoring()

    # Test 2: STT (CPU-based, should be fast)
    results['stt'] = await test_stt_service()

    # Test 3: LLM (GPU-intensive, will take time to load)
    results['llm'] = await test_llm_service()

    # Test 4: TTS (GPU-intensive)
    results['tts'] = await test_tts_service()

    # Summary
    print("\n" + "="*60)
    print("QUICK TEST SUMMARY")
    print("="*60)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for service, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {service.upper():12s}: {status}")

    print(f"\n  Total: {passed}/{total} tests passed")

    if passed == total:
        print("\n✅ All services operational! Ready for E2E testing.")
    else:
        print("\n❌ Some services failed. Please fix before E2E testing.")

    print("="*60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
