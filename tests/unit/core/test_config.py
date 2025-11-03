"""
TDD Unit Tests for Config

Testing configuration management and validation functionality.
"""

import pytest
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from avatar.core.config import Config, config


class TestConfigInitialization:
    """Test Config class initialization and default values"""

    def test_config_has_required_paths(self):
        """Test Config has all required path attributes"""
        config = Config()

        # Test base paths exist
        assert hasattr(config, 'BASE_DIR')
        assert hasattr(config, 'AUDIO_DIR')
        assert hasattr(config, 'DATABASE_PATH')

        # Test audio subdirectories exist
        assert hasattr(config, 'AUDIO_RAW')
        assert hasattr(config, 'AUDIO_PROFILES')
        assert hasattr(config, 'AUDIO_TTS_FAST')
        assert hasattr(config, 'AUDIO_TTS_HQ')

    def test_config_paths_are_path_objects(self):
        """Test that config paths are Path objects"""
        config = Config()

        assert isinstance(config.BASE_DIR, Path)
        assert isinstance(config.AUDIO_DIR, Path)
        assert isinstance(config.DATABASE_PATH, Path)
        assert isinstance(config.AUDIO_RAW, Path)

    def test_config_default_values(self):
        """Test default configuration values"""
        config = Config()

        # Server defaults
        assert config.HOST == "0.0.0.0"
        assert config.PORT == 8000

        # Resource limits
        assert config.MAX_CONCURRENT_SESSIONS == 4
        assert config.VRAM_LIMIT_GB == 20

        # Model settings
        assert config.WHISPER_MODEL_SIZE == "base"
        assert config.WHISPER_DEVICE == "cpu"


class TestConfigEnvironmentVariables:
    """Test Config reads environment variables correctly"""

    def test_host_default_value(self):
        """Test HOST has correct default value"""
        assert config.HOST == '0.0.0.0'

    def test_port_default_value(self):
        """Test PORT has correct default value"""
        assert config.PORT == 8000

    def test_max_sessions_default_value(self):
        """Test MAX_CONCURRENT_SESSIONS has correct default value"""
        assert config.MAX_CONCURRENT_SESSIONS == 4

    def test_vram_limit_default_value(self):
        """Test VRAM_LIMIT_GB has correct default value"""
        assert config.VRAM_LIMIT_GB == 20

    def test_cors_origins_default_value(self):
        """Test CORS_ORIGINS has correct default value"""
        expected = ["http://localhost:3000", "http://localhost:8000"]
        assert config.CORS_ORIGINS == expected

    def test_gpu_device_none_when_not_set(self):
        """Test GPU_DEVICE is None when env var not set"""
        with patch.dict(os.environ, {}, clear=True):
            config = Config()
            assert config.GPU_DEVICE is None

    def test_gpu_device_default_value(self):
        """Test GPU_DEVICE default is None (set by get_optimal_gpu)"""
        assert config.GPU_DEVICE is None


class TestConfigGPUFunctional:
    """Test GPU functionality with REAL hardware (Linus approved)"""

    def test_gpu_selection_real_hardware(self):
        """Test GPU selection works with actual hardware"""
        try:
            gpu_id = config.get_optimal_gpu()
            assert isinstance(gpu_id, int)
            assert gpu_id >= 0
            print(f"✅ Real GPU selected: {gpu_id}")
        except RuntimeError as e:
            # CUDA not available is acceptable
            assert "CUDA not available" in str(e)
            print("✅ CUDA not available (acceptable in CI)")


class TestConfigValidation:
    """Test Config validation functionality"""

    def test_validate_creates_audio_directories(self):
        """Test validate() creates required audio directories"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Mock config paths to use temp directory
            with patch.object(Config, 'BASE_DIR', temp_path), \
                 patch.object(Config, 'AUDIO_RAW', temp_path / 'audio' / 'raw'), \
                 patch.object(Config, 'AUDIO_PROFILES', temp_path / 'audio' / 'profiles'), \
                 patch.object(Config, 'AUDIO_TTS_FAST', temp_path / 'audio' / 'tts_fast'), \
                 patch.object(Config, 'AUDIO_TTS_HQ', temp_path / 'audio' / 'tts_hq'), \
                 patch.object(Config, 'DATABASE_PATH', temp_path / 'app.db'):

                # Create the database file so validation doesn't fail
                (temp_path / 'app.db').touch()

                result = Config.validate()

                assert result is True
                assert (temp_path / 'audio' / 'raw').exists()
                assert (temp_path / 'audio' / 'profiles').exists()
                assert (temp_path / 'audio' / 'tts_fast').exists()
                assert (temp_path / 'audio' / 'tts_hq').exists()

    def test_validate_fails_missing_database(self):
        """Test validate() fails when database file missing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            with patch.object(Config, 'DATABASE_PATH', temp_path / 'nonexistent.db'):
                result = Config.validate()

                assert result is False

    @patch.object(Config, 'AUTO_SELECT_GPU', True)
    @patch.object(Config, 'GPU_DEVICE', None)
    @patch.object(Config, 'get_optimal_gpu')
    def test_validate_auto_selects_gpu(self, mock_get_optimal_gpu):
        """Test validate() auto-selects GPU when enabled"""
        mock_get_optimal_gpu.return_value = 1

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            (temp_path / 'app.db').touch()

            with patch.object(Config, 'DATABASE_PATH', temp_path / 'app.db'), \
                 patch.object(Config, 'AUDIO_RAW', temp_path / 'audio' / 'raw'), \
                 patch.object(Config, 'AUDIO_PROFILES', temp_path / 'audio' / 'profiles'), \
                 patch.object(Config, 'AUDIO_TTS_FAST', temp_path / 'audio' / 'tts_fast'), \
                 patch.object(Config, 'AUDIO_TTS_HQ', temp_path / 'audio' / 'tts_hq'):

                result = Config.validate()

                assert result is True
                mock_get_optimal_gpu.assert_called_once()

    def test_validate_exception_handling(self):
        """Test validate() handles exceptions gracefully"""
        with patch.object(Config, 'AUDIO_RAW') as mock_audio_raw:
            # Simulate permission error
            mock_audio_raw.mkdir.side_effect = PermissionError("Access denied")

            result = Config.validate()

            assert result is False


class TestConfigConstants:
    """Test configuration constants"""

    def test_performance_thresholds(self):
        """Test performance threshold constants"""
        config = Config()

        assert config.TARGET_E2E_LATENCY_SEC == 3.5
        assert config.TARGET_LLM_TTFT_MS == 800
        assert config.TARGET_FAST_TTS_SEC == 1.5

    def test_model_defaults(self):
        """Test AI model default settings"""
        config = Config()

        assert config.VLLM_MODEL == "Qwen/Qwen2.5-7B-Instruct-AWQ"
        assert config.VLLM_GPU_MEMORY == 0.5
        assert config.VLLM_MAX_TOKENS == 2048

        assert config.F5_TTS_SPEED == 1.0
        assert config.COSYVOICE_SAMPLE_RATE == 22050

    def test_logging_config(self):
        """Test logging configuration defaults"""
        config = Config()

        assert config.LOG_LEVEL == "INFO"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])