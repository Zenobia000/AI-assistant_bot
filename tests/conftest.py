"""
Test Configuration and Fixtures for AVATAR

Provides FastAPI test client, database fixtures, and mock services.
"""

import asyncio
import pytest
import tempfile
import shutil
from pathlib import Path
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock

import aiosqlite
from fastapi.testclient import TestClient
from httpx import AsyncClient

# Add src to path for imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import AVATAR components
from avatar.main import app
from avatar.core.config import Config


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


@pytest.fixture
def mock_whisper_service():
    """Mock Whisper STT service"""
    mock = AsyncMock()
    mock.transcribe.return_value = "Hello, this is a test transcription."
    return mock


@pytest.fixture
def mock_vllm_service():
    """Mock vLLM inference service"""
    mock = AsyncMock()
    mock.generate.return_value = "This is a test AI response."
    mock.generate_stream.return_value = async_generator_mock([
        "This ", "is ", "a ", "test ", "AI ", "response."
    ])
    return mock


@pytest.fixture
def mock_f5_tts_service():
    """Mock F5-TTS service"""
    mock = AsyncMock()
    mock.synthesize.return_value = b"fake_audio_data"
    return mock


@pytest.fixture
def mock_cosyvoice_service():
    """Mock CosyVoice service"""
    mock = AsyncMock()
    mock.synthesize.return_value = b"fake_hq_audio_data"
    return mock


@pytest.fixture
def mock_gpu_monitor():
    """Mock GPU monitoring"""
    mock = MagicMock()
    mock.get_memory_info.return_value = {
        'total': 24576,  # 24GB in MB
        'used': 8192,    # 8GB used
        'free': 16384    # 16GB free
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
        # For testing, return a sample message
        return '{"type": "audio_chunk", "session_id": "test", "data": "fake_base64_audio"}'

    async def receive_bytes(self):
        return b"fake_audio_data"

    async def close(self):
        self.closed = True


@pytest.fixture
def mock_websocket():
    """Mock WebSocket connection"""
    return MockWebSocket()


# Performance testing utilities
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


# Skip marks for different environments
def pytest_configure(config):
    """Configure pytest markers"""
    config.addinivalue_line(
        "markers", "gpu_required: mark test as requiring GPU access"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )