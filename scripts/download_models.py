#!/usr/bin/env python3
"""
AVATAR - AI Model Download Script
下載專案所需的 AI 模型

Usage:
    poetry run python scripts/download_models.py
    # 或激活環境後: python scripts/download_models.py
"""

import os
import sys
from pathlib import Path


def print_section(title: str):
    """Print section header"""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print('=' * 60)


def download_whisper_model():
    """Download Whisper model using faster-whisper"""
    print("\n[INFO] Downloading Whisper large-v3-turbo model...")
    print("       This will be downloaded on first use by faster-whisper")
    print("       Model size: ~1.5GB")

    try:
        from faster_whisper import WhisperModel

        # Initialize model (will download if not cached)
        print("\n[INFO] Initializing Whisper model...")
        model = WhisperModel(
            "large-v3-turbo",
            device="cpu",  # Whisper runs on CPU in AVATAR
            compute_type="int8"
        )

        print("[OK] Whisper large-v3-turbo model ready")
        print(f"     Cache location: ~/.cache/huggingface/")
        return True

    except ImportError:
        print("[FAIL] faster-whisper not installed")
        print("       Run: pip install faster-whisper")
        return False
    except Exception as e:
        print(f"[FAIL] Error downloading Whisper model: {e}")
        return False


def download_llm_model():
    """Download LLM model for vLLM"""
    print("\n[INFO] LLM Model Information")
    print("       Model: Qwen/Qwen2.5-7B-Instruct-AWQ")
    print("       Size: ~4-5GB")
    print("       Format: AWQ quantized (4-bit)")

    print("\n[NOTE] vLLM will download models on first use")
    print("       You can pre-download using:")
    print("       huggingface-cli download Qwen/Qwen2.5-7B-Instruct-AWQ")

    # Check if vLLM is installed
    try:
        import vllm
        print(f"\n[OK] vLLM {vllm.__version__} installed")
        print("     Model will be downloaded when first starting vLLM server")
        return True
    except ImportError:
        print("[FAIL] vLLM not installed")
        print("       Run: pip install vllm>=0.6.0")
        return False


def download_tts_models():
    """Information about TTS models"""
    print("\n[INFO] TTS Models Information")
    print("       Fast TTS: F5-TTS")
    print("       HQ TTS: CosyVoice3")

    print("\n[NOTE] TTS models will be downloaded in Phase 2 (Week 1-2)")
    print("       F5-TTS: https://github.com/SWivid/F5-TTS")
    print("       CosyVoice3: https://github.com/FunAudioLLM/CosyVoice")

    return True


def check_disk_space():
    """Check available disk space"""
    print("\n[INFO] Checking disk space...")

    try:
        import shutil
        stat = shutil.disk_usage(".")
        free_gb = stat.free / (1024**3)

        print(f"       Available: {free_gb:.1f} GB")

        if free_gb < 20:
            print("[WARN] Low disk space! Recommended: >20GB free")
            print("       Models require ~10GB storage")
            return False
        else:
            print("[OK] Sufficient disk space available")
            return True

    except Exception as e:
        print(f"[WARN] Could not check disk space: {e}")
        return True


def main():
    """Main download function"""
    print("[CHECK] AVATAR Model Download Utility")
    print(f"Python: {sys.version}")

    # Check disk space first
    check_disk_space()

    results = []

    # Download models
    print_section("Phase 1: Core Models")

    # Whisper STT
    results.append(("Whisper STT", download_whisper_model()))

    # LLM (vLLM)
    results.append(("LLM (vLLM)", download_llm_model()))

    # TTS (info only for now)
    print_section("Phase 2: TTS Models (Deferred)")
    results.append(("TTS Models", download_tts_models()))

    # Summary
    print_section("Download Summary")
    for name, success in results:
        status = "[OK]" if success else "[FAIL]"
        print(f"{status} {name}")

    success_count = sum(1 for _, s in results if s)
    total_count = len(results)

    print(f"\nCompleted: {success_count}/{total_count}")

    if success_count >= 2:  # At least Whisper and vLLM info
        print("\n[OK] Model download preparation complete")
        print("\n[NEXT] Run: python scripts/validate_setup.py")
        return 0
    else:
        print("\n[FAIL] Model download incomplete")
        print("       Please check error messages above")
        return 1


if __name__ == '__main__':
    sys.exit(main())
