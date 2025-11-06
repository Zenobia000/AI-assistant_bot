"""
Test Configuration and Fixtures for AVATAR

Provides FastAPI test client, database fixtures, and mock services.
Enhanced with GPU memory management for reliable testing.
"""

import asyncio
import pytest
import tempfile
import shutil
import os
from pathlib import Path
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock

import aiosqlite
from fastapi.testclient import TestClient
from httpx import AsyncClient

# Add src to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import AVATAR components
from avatar.main import app
from avatar.core.config import Config


# GPU Memory Management for Tests
@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Setup test environment with proper GPU management"""

    # Set test environment variables
    os.environ["AVATAR_ENV"] = "test"
    os.environ["AVATAR_API_TOKEN"] = "test-integration-token"

    # Reduce VRAM allocation for tests
    os.environ["AVATAR_VLLM_MEMORY"] = "0.2"  # 20% instead of 50%
    os.environ["AVATAR_MAX_SESSIONS"] = "1"   # Only 1 concurrent session in tests

    yield

    # Cleanup GPU memory after all tests
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("🧹 GPU memory cleaned after tests")
    except Exception:
        pass


@pytest.fixture
def clear_gpu_memory():
    """Clear GPU memory before and after each test"""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    yield

    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def test_config():
    """Test configuration with temporary paths"""
    # Create temporary directory for test files
    temp_dir = Path(tempfile.mkdtemp())

    # Override config paths for testing
    original_paths = {
        'BASE_DIR': Config.BASE_DIR,
        'DATABASE_PATH': Config.DATABASE_PATH,
        'AUDIO_DIR': Config.AUDIO_DIR,
        'AUDIO_RAW': Config.AUDIO_RAW,
        'AUDIO_PROFILES': Config.AUDIO_PROFILES,
        'AUDIO_TTS_FAST': Config.AUDIO_TTS_FAST,
        'AUDIO_TTS_HQ': Config.AUDIO_TTS_HQ,
    }

    # Set test paths
    Config.BASE_DIR = temp_dir
    Config.DATABASE_PATH = temp_dir / "test.db"
    Config.AUDIO_DIR = temp_dir / "audio"
    Config.AUDIO_RAW = temp_dir / "audio" / "raw"
    Config.AUDIO_PROFILES = temp_dir / "audio" / "profiles"
    Config.AUDIO_TTS_FAST = temp_dir / "audio" / "tts_fast"
    Config.AUDIO_TTS_HQ = temp_dir / "audio" / "tts_hq"

    # Create test directories
    for audio_dir in [Config.AUDIO_RAW, Config.AUDIO_PROFILES,
                      Config.AUDIO_TTS_FAST, Config.AUDIO_TTS_HQ]:
        audio_dir.mkdir(parents=True, exist_ok=True)

    yield Config

    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)

    # Restore original paths
    for key, value in original_paths.items():
        setattr(Config, key, value)


@pytest.fixture
async def test_db(test_config):
    """Initialize test database"""
    try:
        from avatar.services.database import init_database
        await init_database()
    except ImportError:
        # Create empty database file for testing
        Config.DATABASE_PATH.touch()

    yield Config.DATABASE_PATH

    # Cleanup
    if Config.DATABASE_PATH.exists():
        Config.DATABASE_PATH.unlink()


@pytest.fixture
def sync_client(test_config):
    """Synchronous FastAPI test client"""
    with TestClient(app) as client:
        yield client


@pytest.fixture
async def async_client(test_config) -> AsyncGenerator[AsyncClient, None]:
    """Asynchronous FastAPI test client"""
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client


# Mock services for unit testing (without actual model loading)
@pytest.fixture
def mock_whisper_service():
    """Mock Whisper STT service"""
    mock = AsyncMock()
    mock.transcribe.return_value = ("Hello, this is a test transcription.", {"language": "en"})
    return mock


@pytest.fixture
def mock_vllm_service():
    """Mock vLLM inference service"""
    mock = AsyncMock()
    mock.generate.return_value = "This is a test AI response."

    async def mock_stream(messages, **kwargs):
        for word in ["This ", "is ", "a ", "test ", "AI ", "response."]:
            yield word

    mock.chat_stream = mock_stream
    return mock


@pytest.fixture
def mock_f5_tts_service():
    """Mock F5-TTS service"""
    mock = AsyncMock()
    mock.synthesize.return_value = Path("/fake/audio/output.wav")
    mock.synthesize_fast.return_value = Path("/fake/audio/output.wav")
    return mock


@pytest.fixture
def mock_cosyvoice_service():
    """Mock CosyVoice service"""
    mock = AsyncMock()
    mock.synthesize.return_value = Path("/fake/audio/hq_output.wav")
    mock.synthesize_hq.return_value = Path("/fake/audio/hq_output.wav")
    return mock


@pytest.fixture
def mock_gpu_monitor():
    """Mock GPU monitoring"""
    mock = MagicMock()
    mock.get_memory_info.return_value = {
        'total': 20480,  # 20GB in MB
        'used': 2048,    # 2GB used
        'free': 18432    # 18GB free
    }
    mock.is_memory_available.return_value = True
    return mock


@pytest.fixture
def sample_audio_data():
    """Sample audio data for testing"""
    # Generate a simple sine wave audio sample
    import numpy as np

    sample_rate = 16000
    duration = 1.0  # 1 second
    frequency = 440  # A4 note

    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = np.sin(2 * np.pi * frequency * t)

    # Convert to int16 format
    audio_int16 = (audio * 32767).astype(np.int16)

    return audio_int16.tobytes()


@pytest.fixture
def sample_wav_file(test_config, sample_audio_data):
    """Create a sample WAV file for testing"""
    import wave

    wav_path = test_config.AUDIO_RAW / "test_sample.wav"

    # Create WAV file
    with wave.open(str(wav_path), 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 2 bytes per sample (int16)
        wav_file.setframerate(16000)  # 16kHz
        wav_file.writeframes(sample_audio_data)

    yield wav_path

    # Cleanup
    wav_path.unlink(missing_ok=True)


@pytest.fixture
def sample_voice_profile():
    """Sample voice profile data"""
    return {
        'name': 'Test Voice',
        'description': 'A test voice profile',
        'language': 'zh-TW',
        'gender': 'female',
        'audio_path': 'test_voice.wav'
    }


async def async_generator_mock(items):
    """Helper to create async generator mock"""
    for item in items:
        yield item


class MockWebSocket:
    """Mock WebSocket for testing"""

    def __init__(self):
        self.messages = []
        self.closed = False

    async def send_text(self, data: str):
        if not self.closed:
            self.messages.append(('text', data))

    async def send_bytes(self, data: bytes):
        if not self.closed:
            self.messages.append(('bytes', data))

    async def receive_text(self):
        return '{"type": "audio_chunk", "session_id": "test", "data": "fake_base64_audio"}'

    async def receive_bytes(self):
        return b"fake_audio_data"

    async def close(self):
        self.closed = True


@pytest.fixture
def mock_websocket():
    """Mock WebSocket connection"""
    return MockWebSocket()


@pytest.fixture
def performance_timer():
    """Timer for performance testing"""
    import time

    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None

        def start(self):
            self.start_time = time.time()

        def stop(self):
            self.end_time = time.time()

        @property
        def elapsed(self):
            if self.start_time and self.end_time:
                return self.end_time - self.start_time
            return None

    return Timer()


# Test markers configuration
def pytest_configure(config):
    """Configure pytest markers"""
    config.addinivalue_line(
        "markers", "gpu_required: mark test as requiring GPU access"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "unit: mark test as unit test"
    )


# Skip GPU tests if CUDA not available
def pytest_collection_modifyitems(config, items):
    """Skip GPU tests if CUDA not available"""
    try:
        import torch
        cuda_available = torch.cuda.is_available()
    except ImportError:
        cuda_available = False

    if not cuda_available:
        skip_gpu = pytest.mark.skip(reason="CUDA not available")
        for item in items:
            if "gpu_required" in item.keywords:
                item.add_marker(skip_gpu)