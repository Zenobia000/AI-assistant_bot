"""
Configuration management for AVATAR application

Loads settings from environment variables with sensible defaults.
"""

import os
from pathlib import Path
from typing import Optional


class Config:
    """Application configuration"""

    # Project paths
    PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
    BASE_DIR = PROJECT_ROOT  # Alias for backward compatibility
    AUDIO_DIR = BASE_DIR / "audio"
    AUDIO_RAW = AUDIO_DIR / "raw"
    AUDIO_PROFILES = AUDIO_DIR / "profiles"
    AUDIO_TTS_FAST = AUDIO_DIR / "tts_fast"
    AUDIO_TTS_HQ = AUDIO_DIR / "tts_hq"

    # Database
    DATABASE_PATH = BASE_DIR / "app.db"

    # Server settings
    HOST: str = os.getenv("AVATAR_HOST", "0.0.0.0")
    PORT: int = int(os.getenv("AVATAR_PORT", "8000"))

    # CORS settings
    CORS_ORIGINS: list[str] = os.getenv(
        "AVATAR_CORS_ORIGINS",
        "http://localhost:3000,http://localhost:8000"
    ).split(",")

    # Resource limits (VRAM management) - Adjusted for RTX 4000 (20GB)
    MAX_CONCURRENT_SESSIONS: int = int(os.getenv("AVATAR_MAX_SESSIONS", "4"))  # Reduced for 20GB
    VRAM_LIMIT_GB: int = int(os.getenv("AVATAR_VRAM_LIMIT", "20"))  # RTX 4000 SFF Ada

    # Multi-GPU configuration
    GPU_DEVICE: Optional[int] = None if os.getenv("AVATAR_GPU_DEVICE") is None else int(os.getenv("AVATAR_GPU_DEVICE", "1"))  # Use GPU 1 (RTX 4000)
    AUTO_SELECT_GPU: bool = os.getenv("AVATAR_AUTO_SELECT_GPU", "true").lower() == "true"

    # ============================================================
    # Service Provider Configuration (地端/API 切換)
    # ============================================================

    # Provider Mode Selection
    # 支援的 STT Providers: local (Whisper), openai, azure, google
    STT_PROVIDER: str = os.getenv("AVATAR_STT_PROVIDER", "local")

    # 支援的 LLM Providers: local (vLLM), openai, anthropic, azure
    LLM_PROVIDER: str = os.getenv("AVATAR_LLM_PROVIDER", "local")

    # 支援的 TTS Providers: local (F5-TTS), elevenlabs, azure, openai
    TTS_PROVIDER: str = os.getenv("AVATAR_TTS_PROVIDER", "local")

    # -------------------- STT API Configuration --------------------
    STT_API_KEY: Optional[str] = os.getenv("AVATAR_STT_API_KEY")
    STT_API_ENDPOINT: Optional[str] = os.getenv("AVATAR_STT_API_ENDPOINT")
    STT_API_MODEL: str = os.getenv("AVATAR_STT_API_MODEL", "whisper-1")

    # -------------------- LLM API Configuration --------------------
    LLM_API_KEY: Optional[str] = os.getenv("AVATAR_LLM_API_KEY")
    LLM_API_ENDPOINT: Optional[str] = os.getenv("AVATAR_LLM_API_ENDPOINT")
    LLM_API_MODEL: str = os.getenv("AVATAR_LLM_API_MODEL", "gpt-4")
    LLM_API_BASE_URL: Optional[str] = os.getenv("AVATAR_LLM_API_BASE_URL")  # For custom endpoints

    # -------------------- TTS API Configuration --------------------
    TTS_API_KEY: Optional[str] = os.getenv("AVATAR_TTS_API_KEY")
    TTS_API_ENDPOINT: Optional[str] = os.getenv("AVATAR_TTS_API_ENDPOINT")
    TTS_API_VOICE: str = os.getenv("AVATAR_TTS_API_VOICE", "alloy")
    TTS_API_MODEL: str = os.getenv("AVATAR_TTS_API_MODEL", "tts-1")

    # ============================================================
    # Local AI Model Settings (when using local providers)
    # ============================================================

    # Whisper STT (Local)
    # ⚠️ Whisper uses CPU to avoid VRAM contention with LLM/TTS
    WHISPER_MODEL_SIZE: str = os.getenv("AVATAR_WHISPER_MODEL", "base")  # tiny, base, small, medium, large
    WHISPER_DEVICE: str = os.getenv("AVATAR_WHISPER_DEVICE", "cpu")  # Force CPU inference
    WHISPER_COMPUTE_TYPE: str = os.getenv("AVATAR_WHISPER_COMPUTE", "int8")  # int8 for CPU efficiency

    # vLLM (Local)
    VLLM_MODEL: str = os.getenv(
        "AVATAR_VLLM_MODEL",
        "Qwen/Qwen2.5-7B-Instruct-AWQ"
    )
    VLLM_GPU_MEMORY: float = float(os.getenv("AVATAR_VLLM_MEMORY", "0.75"))  # GPU 記憶體比例
    VLLM_MAX_TOKENS: int = int(os.getenv("AVATAR_VLLM_MAX_TOKENS", "2048"))

    # TTS settings
    F5_TTS_SPEED: float = float(os.getenv("AVATAR_F5_SPEED", "1.0"))
    COSYVOICE_SAMPLE_RATE: int = int(os.getenv("AVATAR_COSY_SAMPLE_RATE", "24000"))  # CosyVoice2 uses 24kHz

    # TTS Quality Mode Settings (CosyVoice2)
    TTS_ENABLE_HQ_MODE: bool = bool(os.getenv("AVATAR_TTS_ENABLE_HQ", "true").lower() == "true")  # Enable by default
    TTS_HQ_MODEL_PATH: str = os.getenv("AVATAR_TTS_HQ_MODEL", "CosyVoice/pretrained_models/CosyVoice2-0.5B")

    # Performance thresholds (KPIs)
    TARGET_E2E_LATENCY_SEC: float = 3.5  # P95 target
    TARGET_LLM_TTFT_MS: int = 800        # Time to first token
    TARGET_FAST_TTS_SEC: float = 1.5     # Fast TTS P50

    # Logging
    LOG_LEVEL: str = os.getenv("AVATAR_LOG_LEVEL", "INFO")

    @classmethod
    def get_optimal_gpu(cls) -> int:
        """
        Select GPU with largest available VRAM

        Returns:
            GPU device ID with most available VRAM
        """
        try:
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA not available")

            gpu_count = torch.cuda.device_count()
            best_gpu = 0
            max_memory = 0

            print(f"🔍 Scanning {gpu_count} GPUs for optimal selection:")

            for i in range(gpu_count):
                props = torch.cuda.get_device_properties(i)
                total_memory = props.total_memory / (1024**3)  # GB

                # Get current memory usage
                torch.cuda.set_device(i)
                allocated = torch.cuda.memory_allocated(i) / (1024**3)
                available = total_memory - allocated

                print(f"  GPU {i}: {props.name}")
                print(f"    Total: {total_memory:.1f}GB, Available: {available:.1f}GB")

                if available > max_memory:
                    max_memory = available
                    best_gpu = i

            print(f"✅ Selected GPU {best_gpu} with {max_memory:.1f}GB available VRAM")
            return best_gpu

        except ImportError:
            print("⚠️ PyTorch not installed, defaulting to GPU 1")
            return 1
        except Exception as e:
            print(f"⚠️ GPU selection failed: {e}, defaulting to GPU 1")
            return 1

    @classmethod
    def validate(cls) -> bool:
        """Validate configuration and create necessary directories"""
        try:
            # Create audio directories if not exist
            cls.AUDIO_RAW.mkdir(parents=True, exist_ok=True)
            cls.AUDIO_PROFILES.mkdir(parents=True, exist_ok=True)
            cls.AUDIO_TTS_FAST.mkdir(parents=True, exist_ok=True)
            cls.AUDIO_TTS_HQ.mkdir(parents=True, exist_ok=True)

            # Auto-select optimal GPU if enabled
            if cls.AUTO_SELECT_GPU and cls.GPU_DEVICE is None:
                cls.GPU_DEVICE = cls.get_optimal_gpu()
                print(f"🎯 Auto-selected GPU {cls.GPU_DEVICE}")
            elif cls.GPU_DEVICE is not None:
                print(f"🎯 Using configured GPU {cls.GPU_DEVICE}")

            # Check database exists
            if not cls.DATABASE_PATH.exists():
                raise FileNotFoundError(
                    f"Database not found: {cls.DATABASE_PATH}\n"
                    f"Run: poetry run python scripts/init_database.py"
                )

            return True
        except Exception as e:
            print(f"[FAIL] Configuration validation failed: {e}")
            return False


# Global config instance
config = Config()
