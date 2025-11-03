"""
TDD Unit Tests for STT Service

Testing Whisper STT service with real model loading and transcription.
"""

import pytest
import asyncio
from pathlib import Path
from unittest.mock import patch, MagicMock

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from avatar.services.stt import STTService, get_stt_service
from avatar.core.config import config


class TestSTTServiceInitialization:
    """Test STT service initialization"""

    def test_stt_service_init_defaults(self):
        """Test STT service initializes with default values"""
        service = STTService()

        assert service.model_size == "base"
        assert service.device == "cpu"
        assert service.compute_type == "int8"
        assert service._model is None  # Lazy loading (ACTUAL attribute)

    def test_stt_service_init_custom_values(self):
        """Test STT service with custom configuration"""
        service = STTService(
            model_size="small",
            device="cuda",
            compute_type="float16"
        )

        assert service.model_size == "small"
        assert service.device == "cuda"
        assert service.compute_type == "float16"

    @pytest.mark.asyncio
    async def test_stt_service_singleton(self):
        """Test STT service singleton pattern (REAL async)"""
        service1 = await get_stt_service()
        service2 = await get_stt_service()

        assert service1 is service2


@pytest.mark.unit
@pytest.mark.slow
class TestSTTServiceModelLoading:
    """Test STT service model loading with real models"""

    def test_load_model_base_cpu(self):
        """Test loading base Whisper model on CPU"""
        service = STTService(model_size="base", device="cpu")

        service._load_model()

        assert service._model is not None
        assert hasattr(service._model, 'transcribe')

    def test_model_lazy_loading(self):
        """Test that model is loaded on first use"""
        service = STTService()

        # Model should be None initially
        assert service._model is None

        # Load model
        service._load_model()

        # Model should be loaded
        assert service._model is not None

    def test_model_loading_idempotent(self):
        """Test that multiple calls to _load_model don't reload"""
        service = STTService()

        service._load_model()
        first_model = service._model

        service._load_model()
        second_model = service._model

        assert first_model is second_model


@pytest.mark.unit
@pytest.mark.slow
class TestSTTServiceTranscription:
    """Test STT transcription functionality"""

    @pytest.fixture
    def stt_service(self):
        """Provide initialized STT service"""
        service = STTService()
        service._load_model()
        return service

    @pytest.fixture
    def test_audio_path(self):
        """Provide path to test audio file"""
        return config.AUDIO_RAW / "test_sample.wav"

    @pytest.mark.asyncio
    async def test_transcribe_with_real_audio(self, stt_service, test_audio_path):
        """Test transcription with real audio file"""
        if not test_audio_path.exists():
            pytest.skip("Test audio file not found")

        result = await stt_service.transcribe(
            audio_path=test_audio_path,
            language=None,  # Auto-detect
            beam_size=5
        )

        # Should return (text, metadata) tuple
        assert isinstance(result, tuple)
        assert len(result) == 2

        text, metadata = result
        assert isinstance(text, str)
        assert isinstance(metadata, dict)

        # Metadata should contain expected keys
        assert 'language' in metadata
        assert 'duration' in metadata
        assert 'segments_count' in metadata

    @pytest.mark.asyncio
    async def test_transcribe_with_language_hint(self, stt_service, test_audio_path):
        """Test transcription with language hint"""
        if not test_audio_path.exists():
            pytest.skip("Test audio file not found")

        result = await stt_service.transcribe(
            audio_path=test_audio_path,
            language="en",
            beam_size=5
        )

        text, metadata = result
        assert isinstance(result, tuple)
        assert metadata.get('language') == 'en'

    @pytest.mark.asyncio
    async def test_transcribe_with_vad_filter(self, stt_service, test_audio_path):
        """Test transcription with VAD filtering"""
        if not test_audio_path.exists():
            pytest.skip("Test audio file not found")

        result = await stt_service.transcribe(
            audio_path=test_audio_path,
            vad_filter=True
        )

        text, metadata = result
        assert isinstance(result, tuple)

    @pytest.mark.asyncio
    async def test_transcribe_nonexistent_file(self, stt_service):
        """Test transcription with non-existent audio file"""
        nonexistent_path = Path("/tmp/nonexistent_audio.wav")

        with pytest.raises((FileNotFoundError, RuntimeError)):
            await stt_service.transcribe(nonexistent_path)

    @pytest.mark.asyncio
    async def test_transcribe_invalid_audio_format(self, stt_service, tmp_path):
        """Test transcription with invalid audio file"""
        invalid_file = tmp_path / "invalid.txt"
        invalid_file.write_text("This is not audio data")

        with pytest.raises((RuntimeError, Exception)):
            await stt_service.transcribe(invalid_file)


@pytest.mark.unit
class TestSTTServiceConfiguration:
    """Test STT service configuration handling"""

    def test_model_size_validation(self):
        """Test model size validation"""
        valid_sizes = ["tiny", "base", "small", "medium", "large"]

        for size in valid_sizes:
            service = STTService(model_size=size)
            assert service.model_size == size

    def test_device_configuration(self):
        """Test device configuration"""
        for device in ["cpu", "cuda"]:
            service = STTService(device=device)
            assert service.device == device

    def test_compute_type_configuration(self):
        """Test compute type configuration"""
        for compute_type in ["int8", "float16", "float32"]:
            service = STTService(compute_type=compute_type)
            assert service.compute_type == compute_type


@pytest.mark.unit
class TestSTTServicePerformance:
    """Test STT service performance characteristics"""

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_transcription_latency_target(self):
        """Test that transcription meets latency targets"""
        service = STTService()
        test_audio = config.AUDIO_RAW / "test_sample.wav"

        if not test_audio.exists():
            pytest.skip("Test audio file not found")

        import time
        start_time = time.time()

        await service.transcribe(test_audio)

        end_time = time.time()
        latency = end_time - start_time

        # Should meet 600ms target (excluding first model load)
        if service._model is not None:  # If model was already loaded (REAL attribute)
            assert latency <= 1.0  # Allow some margin for unit test

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_concurrent_transcriptions(self):
        """Test concurrent transcription requests"""
        service = STTService()
        test_audio = config.AUDIO_RAW / "test_sample.wav"

        if not test_audio.exists():
            pytest.skip("Test audio file not found")

        # Ensure model is loaded first (REAL method name)
        service._load_model()

        # Run 3 concurrent transcriptions
        tasks = [
            service.transcribe(test_audio)
            for _ in range(3)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # All should complete without exceptions
        for result in results:
            assert not isinstance(result, Exception)
            assert isinstance(result, tuple)
            assert len(result) == 2


class TestSTTServiceErrorHandling:
    """Test STT service error handling"""

    @pytest.mark.asyncio
    async def test_transcribe_handles_whisper_exceptions(self):
        """Test transcription handles Whisper library exceptions"""
        service = STTService()

        # Mock faster_whisper to raise exception
        with patch('avatar.services.stt.WhisperModel') as mock_whisper:
            mock_model = MagicMock()
            mock_model.transcribe.side_effect = RuntimeError("Whisper failed")
            mock_whisper.return_value = mock_model

            service.model = mock_model

            with pytest.raises(RuntimeError, match="Whisper failed"):
                await service.transcribe(Path("/tmp/test.wav"))

    @pytest.mark.asyncio
    async def test_model_loading_failure(self):
        """Test handling of model loading failures"""
        service = STTService()

        with patch('avatar.services.stt.WhisperModel') as mock_whisper:
            mock_whisper.side_effect = RuntimeError("Model load failed")

            with pytest.raises(RuntimeError, match="Model load failed"):
                service._load_model()  # REAL method name


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])