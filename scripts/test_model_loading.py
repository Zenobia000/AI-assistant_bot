#!/usr/bin/env python3
"""
Test script for AI model loading and VRAM validation

Tests each service individually and monitors VRAM usage to validate
the resource allocation strategy.

Usage:
    poetry run python scripts/test_model_loading.py
"""

import asyncio
import sys
from pathlib import Path

import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def print_vram_usage(prefix: str = ""):
    """Print current VRAM usage if CUDA is available"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3    # GB
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"\n{prefix}VRAM Usage:")
        print(f"  Allocated: {allocated:.2f} GB")
        print(f"  Reserved:  {reserved:.2f} GB")
        print(f"  Total:     {total:.2f} GB")
        print(f"  Usage:     {(allocated/total)*100:.1f}%")
    else:
        print(f"\n{prefix}CUDA not available - using CPU only")


def print_separator(title: str):
    """Print a section separator"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


async def test_stt_service():
    """Test STTService (CPU-based, no VRAM)"""
    print_separator("Test 1: STTService (Whisper on CPU)")

    try:
        from avatar.services.stt import get_stt_service

        print("\n[1/3] Getting STT service instance...")
        stt = await get_stt_service()
        print(f"✓ STT service created: {stt.model_size} on {stt.device}")

        print("\n[2/3] Loading Whisper model...")
        stt._load_model()
        print(f"✓ Whisper model loaded: {stt.model_size}")

        print("\n[3/3] Checking VRAM (should be ~0 GB for CPU)...")
        print_vram_usage("  ")

        print("\n✅ STTService test PASSED")
        return True

    except Exception as e:
        print(f"\n❌ STTService test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_llm_service():
    """Test LLMService (GPU-based, ~12 GB VRAM)"""
    print_separator("Test 2: LLMService (vLLM on GPU)")

    try:
        from avatar.services.llm import get_llm_service

        print("\n[1/3] Getting LLM service instance...")
        llm = await get_llm_service()
        print(f"✓ LLM service created: {llm.model_path}")
        print(f"  GPU Memory Utilization: {llm.gpu_memory_utilization*100:.0f}%")

        print("\n[2/3] Loading vLLM engine (this may take 30-60 seconds)...")
        print("  Note: vLLM will show detailed loading logs...")
        await llm._load_model()
        print(f"✓ vLLM engine loaded")

        print("\n[3/3] Checking VRAM (should be ~9-12 GB)...")
        print_vram_usage("  ")

        print("\n✅ LLMService test PASSED")
        return True

    except Exception as e:
        print(f"\n❌ LLMService test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_tts_service():
    """Test TTSService (GPU-based, ~4 GB VRAM)"""
    print_separator("Test 3: TTSService (F5-TTS on GPU)")

    try:
        from avatar.services.tts import get_tts_service

        print("\n[1/3] Getting TTS service instance...")
        tts = await get_tts_service()
        print(f"✓ TTS service created: {tts.model_name} on {tts.device}")

        print("\n[2/3] Loading F5-TTS model...")
        tts._load_model()
        print(f"✓ F5-TTS model loaded")

        print("\n[3/3] Checking VRAM (should be +~4 GB from LLM)...")
        print_vram_usage("  ")

        print("\n✅ TTSService test PASSED")
        return True

    except Exception as e:
        print(f"\n❌ TTSService test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_concurrent_services():
    """Test all services loaded concurrently"""
    print_separator("Test 4: All Services Concurrent")

    try:
        from avatar.services.stt import get_stt_service
        from avatar.services.llm import get_llm_service
        from avatar.services.tts import get_tts_service

        print("\n[1/2] Getting all service instances...")
        stt = await get_stt_service()
        llm = await get_llm_service()
        tts = await get_tts_service()
        print("✓ All services retrieved (already loaded)")

        print("\n[2/2] Checking total VRAM usage...")
        print("  Expected: ~16-17 GB (LLM ~12 GB + TTS ~4 GB)")
        print_vram_usage("  Actual:   ")

        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            usage_pct = (allocated / total) * 100

            if usage_pct > 90:
                print(f"\n⚠️  WARNING: VRAM usage is very high ({usage_pct:.1f}%)")
                print("  This may cause OOM errors with multiple concurrent users")
            elif usage_pct > 75:
                print(f"\n✓ VRAM usage is acceptable ({usage_pct:.1f}%)")
                print("  Should support 1-2 concurrent users")
            else:
                print(f"\n✓ VRAM usage is good ({usage_pct:.1f}%)")
                print("  Should support 3-5 concurrent users")

        print("\n✅ Concurrent services test PASSED")
        return True

    except Exception as e:
        print(f"\n❌ Concurrent services test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


async def cleanup_services():
    """Cleanup loaded models"""
    print_separator("Cleanup")

    try:
        from avatar.services.stt import _stt_service
        from avatar.services.llm import _llm_service
        from avatar.services.tts import _tts_service

        print("\nUnloading models...")

        if _stt_service is not None:
            _stt_service.unload_model()
            print("✓ STT model unloaded")

        if _llm_service is not None:
            await _llm_service.unload_model()
            print("✓ LLM model unloaded")

        if _tts_service is not None:
            _tts_service.unload_model()
            print("✓ TTS model unloaded")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("✓ CUDA cache cleared")

        print_vram_usage("\nFinal ")

    except Exception as e:
        print(f"\n⚠️  Cleanup warning: {e}")


async def main():
    """Run all tests"""
    print("="*60)
    print("  AVATAR AI Model Loading Test")
    print("  Testing: STT → LLM → TTS → Concurrent")
    print("="*60)

    # Initial VRAM state
    print_vram_usage("Initial ")

    # Run tests
    results = []

    # Test 1: STT (CPU)
    results.append(await test_stt_service())
    await asyncio.sleep(1)

    # Test 2: LLM (GPU)
    if results[-1]:  # Only continue if previous test passed
        results.append(await test_llm_service())
        await asyncio.sleep(1)
    else:
        print("\n⚠️  Skipping LLM test due to STT failure")
        results.append(False)

    # Test 3: TTS (GPU)
    if results[-1]:
        results.append(await test_tts_service())
        await asyncio.sleep(1)
    else:
        print("\n⚠️  Skipping TTS test due to LLM failure")
        results.append(False)

    # Test 4: Concurrent
    if all(results):
        results.append(await test_concurrent_services())
    else:
        print("\n⚠️  Skipping concurrent test due to previous failures")
        results.append(False)

    # Cleanup
    await cleanup_services()

    # Summary
    print_separator("Test Summary")
    test_names = ["STT Service", "LLM Service", "TTS Service", "Concurrent"]

    for i, (name, result) in enumerate(zip(test_names, results), 1):
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"  Test {i}: {name:20s} {status}")

    passed = sum(results)
    total = len(results)
    print(f"\nTotal: {passed}/{total} tests passed")

    if all(results):
        print("\n🎉 All tests PASSED! Ready for Task 13 (E2E Integration)")
    else:
        print("\n⚠️  Some tests FAILED. Please fix issues before continuing.")
        sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
