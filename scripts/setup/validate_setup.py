#!/usr/bin/env python3
"""
AVATAR - Environment Validation Script
驗證開發環境是否正確設置

Usage:
    poetry run python scripts/validate_setup.py
    # 或激活環境後: python scripts/validate_setup.py
"""

import sys
import platform


def print_section(title: str):
    """Print section header"""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print('=' * 60)


def check_import(module_name: str, display_name: str = None) -> bool:
    """Check if a module can be imported and display version"""
    display_name = display_name or module_name
    try:
        mod = __import__(module_name)
        version = getattr(mod, '__version__', 'unknown')
        print(f"[OK]   {display_name:25s} {version}")
        return True
    except ImportError:
        print(f"[FAIL] {display_name:25s} Not installed")
        return False


def check_torch_cuda():
    """Check PyTorch CUDA availability"""
    try:
        import torch
        print(f"\n[INFO] PyTorch Configuration:")
        print(f"   Version: {torch.__version__}")
        print(f"   CUDA Available: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            print(f"   CUDA Version: {torch.version.cuda}")
            print(f"   Device Count: {torch.cuda.device_count()}")
            print(f"   Current Device: {torch.cuda.current_device()}")
            print(f"   Device Name: {torch.cuda.get_device_name(0)}")

            # Check VRAM
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"   Total VRAM: {total_memory:.2f} GB")

            if total_memory < 20:
                print(f"   [WARN] VRAM < 20GB, may not run full models")

            # Check compute capability for Flash-Attention
            capability = torch.cuda.get_device_capability(0)
            compute_cap = capability[0] + capability[1] / 10
            print(f"   Compute Capability: {compute_cap}")
            if compute_cap < 7.5:
                print(f"   [WARN] Compute Capability < 7.5, Flash-Attention not supported")
        else:
            print(f"   [FAIL] CUDA not available! Please check:")
            print(f"      1. NVIDIA driver installed correctly")
            print(f"      2. PyTorch CUDA version installed")
            print(f"      3. Run: pip install torch --index-url https://download.pytorch.org/whl/cu121")
            return False

        return True
    except ImportError:
        print(f"[FAIL] PyTorch not installed")
        return False


def main():
    """Main validation function"""
    print("[CHECK] AVATAR Environment Validation")
    print(f"Python Version: {sys.version}")
    print(f"Platform: {platform.system()} {platform.release()}")

    # Track validation results
    results = {
        'core': [],
        'ai': [],
        'optional': [],
        'dev': []
    }

    # Core Dependencies
    print_section("Core Dependencies")
    results['core'].append(check_import('fastapi', 'FastAPI'))
    results['core'].append(check_import('uvicorn', 'Uvicorn'))
    results['core'].append(check_import('websockets', 'WebSockets'))
    results['core'].append(check_import('aiosqlite', 'aiosqlite'))
    results['core'].append(check_import('pydantic', 'Pydantic'))
    results['core'].append(check_import('aiofiles', 'aiofiles'))
    results['core'].append(check_import('dotenv', 'python-dotenv'))

    # AI Dependencies
    print_section("AI Model Dependencies")
    results['ai'].append(check_torch_cuda())
    results['ai'].append(check_import('vllm', 'vLLM'))
    results['ai'].append(check_import('faster_whisper', 'faster-whisper'))

    # Optional Dependencies
    print_section("Optional Dependencies")
    results['optional'].append(check_import('flash_attn', 'Flash-Attention'))

    # Dev Dependencies
    print_section("Development Tools")
    results['dev'].append(check_import('pytest', 'pytest'))
    results['dev'].append(check_import('black', 'black'))
    results['dev'].append(check_import('ruff', 'ruff'))

    # Summary
    print_section("Validation Summary")

    core_pass = sum(results['core'])
    core_total = len(results['core'])
    ai_pass = sum(results['ai'])
    ai_total = len(results['ai'])
    opt_pass = sum(results['optional'])
    opt_total = len(results['optional'])
    dev_pass = sum(results['dev'])
    dev_total = len(results['dev'])

    print(f"Core Dependencies:     {core_pass}/{core_total} passed")
    print(f"AI Dependencies:       {ai_pass}/{ai_total} passed")
    print(f"Optional Dependencies: {opt_pass}/{opt_total} passed")
    print(f"Dev Dependencies:      {dev_pass}/{dev_total} passed")

    # Exit code
    if core_pass == core_total and ai_pass >= 1:  # At least PyTorch
        print("\n[OK] Environment validation passed! Ready to start development")
        return 0
    else:
        print("\n[FAIL] Environment validation failed, please check missing dependencies")
        print("\n[DOC] Installation guide:")
        print("   1. Read docs/planning/mvp_tech_spec.md Section 7.1")
        print("   2. Or run: cat pyproject.toml | grep -A 50 'Environment Setup Guide'")
        return 1


if __name__ == '__main__':
    sys.exit(main())
