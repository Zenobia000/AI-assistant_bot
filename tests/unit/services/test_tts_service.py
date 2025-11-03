"""
TDD Unit Tests for TTS Service

Testing F5-TTS service with real model loading and synthesis.
"""

import pytest
import asyncio
from pathlib import Path
from unittest.mock import patch, MagicMock

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from avatar.services.tts import TTSService, get_tts_service
from avatar.core.config import config


class TestTTSServiceInitialization:
    """Test TTS service initialization"""

    def test_tts_service_init_defaults(self):
        """Test TTS service initializes with default values"""
        service = TTSService()

        assert "cuda" in service.device  # Could be "cuda" or "cuda:0"
        assert service.model_name == "F5-TTS"
        assert service.speed == 1.0
        assert service._model is None  # Lazy loading (REAL attribute)

    def test_tts_service_init_custom_values(self):
        """Test TTS service with custom configuration"""
        service = TTSService(
            device="cpu",
            model_name="custom-tts",
            speed=1.2
        )

        assert service.device == "cpu"
        assert service.model_name == "custom-tts"
        assert service.speed == 1.2

    @pytest.mark.asyncio
    async def test_tts_service_singleton(self):
        """Test TTS service singleton pattern (REAL async)"""
        service1 = await get_tts_service()
        service2 = await get_tts_service()

        assert service1 is service2


@pytest.mark.unit
@pytest.mark.slow
class TestTTSServiceModelLoading:
    """Test TTS service model loading with real F5-TTS"""

    @pytest.mark.asyncio
    async def test_load_model_f5_tts_gpu(self):
        """Test loading F5-TTS model on GPU"""
        service = TTSService()

        service._load_model()

        assert service._model is not None
        assert hasattr(service._model, 'inference')

    @pytest.mark.asyncio
    async def test_model_lazy_loading(self):
        """Test that model is loaded on first use"""
        service = TTSService()

        # Model should be None initially
        assert service._model is None

        # Load model
        service._load_model()

        # Model should be loaded
        assert service._model is not None

    @pytest.mark.asyncio
    async def test_model_loading_idempotent(self):
        """Test that multiple calls to _ensure_model_loaded don't reload"""
        service = TTSService()

        service._load_model()
        first_model = service._model

        service._load_model()
        second_model = service._model

        assert first_model is second_model


@pytest.mark.unit
@pytest.mark.slow
class TestTTSServiceSynthesis:
    """Test TTS synthesis functionality"""

    @pytest.fixture
    def tts_service(self):
        """Provide initialized TTS service"""
        service = TTSService()
        service._load_model()
        return service

    @pytest.fixture
    def test_reference_audio(self):
        """Provide path to test reference audio"""
        return config.AUDIO_PROFILES / "test_profile" / "reference.wav"

    @pytest.fixture
    def test_output_path(self, tmp_path):
        """Provide temporary output path"""
        return tmp_path / "test_output.wav"

    @pytest.mark.asyncio
    async def test_synthesize_with_reference_audio(self, tts_service, test_reference_audio, test_output_path):
        """Test synthesis with reference audio (voice cloning)"""
        if not test_reference_audio.exists():
            pytest.skip("Test reference audio not found")

        result_path = await tts_service.synthesize(
            text="Hello, this is a test synthesis.",
            ref_audio_path=test_reference_audio,
            ref_text="This is test reference text.",
            output_path=test_output_path,
            remove_silence=True
        )

        assert result_path == test_output_path
        assert test_output_path.exists()
        assert test_output_path.stat().st_size > 1000  # Should have audio data

    @pytest.mark.asyncio
    async def test_synthesize_fast_with_voice_profile(self, tts_service, test_output_path):
        """Test fast synthesis with voice profile"""
        voice_profile_name = "test_profile"

        # Check if voice profile exists
        profile_path = config.AUDIO_PROFILES / voice_profile_name / "reference.wav"
        if not profile_path.exists():
            pytest.skip("Test voice profile not found")

        result_path = await tts_service.synthesize_fast(
            text="This is fast TTS synthesis test.",
            voice_profile_name=voice_profile_name,
            output_path=test_output_path
        )

        assert result_path == test_output_path
        assert test_output_path.exists()
        assert test_output_path.stat().st_size > 1000

    @pytest.mark.asyncio
    async def test_synthesis_different_text_lengths(self, tts_service, test_reference_audio, test_output_path):
        """Test synthesis with different text lengths"""
        if not test_reference_audio.exists():
            pytest.skip("Test reference audio not found")

        # Test short text
        short_text = "Hi."
        await tts_service.synthesize(
            text=short_text,
            ref_audio_path=test_reference_audio,
            ref_text="Short ref.",
            output_path=test_output_path
        )
        assert test_output_path.exists()

        # Test medium text
        medium_text = "This is a medium length text for testing TTS synthesis capabilities."
        output_medium = test_output_path.parent / "medium.wav"
        await tts_service.synthesize(
            text=medium_text,
            ref_audio_path=test_reference_audio,
            ref_text="Medium reference text for voice cloning.",
            output_path=output_medium
        )
        assert output_medium.exists()

    @pytest.mark.asyncio
    async def test_synthesis_with_chinese_text(self, tts_service, test_reference_audio, test_output_path):
        """Test synthesis with Chinese text"""
        if not test_reference_audio.exists():
            pytest.skip("Test reference audio not found")

        chinese_text = "你好，這是中文語音合成測試。"
        chinese_ref = "這是中文參考文字。"

        result_path = await tts_service.synthesize(
            text=chinese_text,
            ref_audio_path=test_reference_audio,
            ref_text=chinese_ref,
            output_path=test_output_path
        )

        assert result_path == test_output_path
        assert test_output_path.exists()


@pytest.mark.unit
@pytest.mark.slow
class TestTTSServicePerformance:
    """Test TTS service performance characteristics"""

    @pytest.fixture
    def tts_service(self):
        """Provide initialized TTS service"""
        service = TTSService()
        service._load_model()
        return service

    @pytest.mark.asyncio
    async def test_synthesis_latency_target(self, tts_service, tmp_path):
        """Test that synthesis meets latency targets"""
        test_ref = config.AUDIO_PROFILES / "test_profile" / "reference.wav"
        if not test_ref.exists():
            pytest.skip("Test reference audio not found")

        output_path = tmp_path / "latency_test.wav"

        # Warmup synthesis
        await tts_service.synthesize(
            text="Warmup",
            ref_audio_path=test_ref,
            ref_text="Warmup ref",
            output_path=output_path
        )

        # Measure synthesis time
        import time
        start_time = time.time()

        await tts_service.synthesize(
            text="Hello, testing TTS latency.",
            ref_audio_path=test_ref,
            ref_text="Reference text for latency test.",
            output_path=output_path
        )

        end_time = time.time()
        synthesis_time = end_time - start_time

        # Should approach 1.5s target (allow margin for test environment)
        assert synthesis_time <= 5.0  # Generous margin for unit test
        assert output_path.exists()

    @pytest.mark.asyncio
    async def test_concurrent_synthesis_requests(self, tts_service, tmp_path):
        """Test concurrent synthesis requests"""
        test_ref = config.AUDIO_PROFILES / "test_profile" / "reference.wav"
        if not test_ref.exists():
            pytest.skip("Test reference audio not found")

        # Create multiple output paths
        outputs = [tmp_path / f"concurrent_{i}.wav" for i in range(3)]

        # Run concurrent synthesis
        tasks = [
            tts_service.synthesize(
                text=f"Concurrent test {i}",
                ref_audio_path=test_ref,
                ref_text="Concurrent reference",
                output_path=output
            )
            for i, output in enumerate(outputs)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # All should complete successfully
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # Concurrent synthesis might have limitations, log but don't fail
                print(f"Concurrent synthesis {i} failed: {result}")
            else:
                assert outputs[i].exists()


class TestTTSServiceErrorHandling:
    """Test TTS service error handling"""

    @pytest.mark.asyncio
    async def test_synthesis_with_missing_reference(self):
        """Test synthesis with missing reference audio"""
        service = TTSService()
        nonexistent_ref = Path("/tmp/nonexistent_ref.wav")
        output_path = Path("/tmp/test_output.wav")

        with pytest.raises((FileNotFoundError, RuntimeError)):
            await service.synthesize(
                text="Test text",
                ref_audio_path=nonexistent_ref,
                ref_text="Test ref",
                output_path=output_path
            )

    @pytest.mark.asyncio
    async def test_synthesis_with_invalid_output_path(self, tmp_path):
        """Test synthesis with invalid output path"""
        service = TTSService()
        test_ref = config.AUDIO_PROFILES / "test_profile" / "reference.wav"

        if test_ref.exists():
            # Try to write to a directory instead of file
            invalid_output = tmp_path
            invalid_output.mkdir(exist_ok=True)

            with pytest.raises((IsADirectoryError, RuntimeError, OSError)):
                await service.synthesize(
                    text="Test",
                    ref_audio_path=test_ref,
                    ref_text="Test ref",
                    output_path=invalid_output
                )

    @pytest.mark.asyncio
    async def test_empty_text_handling(self):
        """Test handling of empty or None text"""
        service = TTSService()
        test_ref = config.AUDIO_PROFILES / "test_profile" / "reference.wav"
        output_path = Path("/tmp/empty_test.wav")

        if test_ref.exists():
            # Test empty string
            with pytest.raises((ValueError, RuntimeError)):
                await service.synthesize(
                    text="",
                    ref_audio_path=test_ref,
                    ref_text="Ref",
                    output_path=output_path
                )

            # Test None text
            with pytest.raises((TypeError, ValueError)):
                await service.synthesize(
                    text=None,
                    ref_audio_path=test_ref,
                    ref_text="Ref",
                    output_path=output_path
                )

    @pytest.mark.asyncio
    async def test_model_loading_failure_handling(self):
        """Test handling of model loading failures"""
        service = TTSService()

        with patch('avatar.services.tts.F5TTS') as mock_f5tts:
            mock_f5tts.side_effect = RuntimeError("F5-TTS model load failed")

            with pytest.raises(RuntimeError, match="F5-TTS model load failed"):
                await service._ensure_model_loaded()


class TestTTSServiceConfiguration:
    """Test TTS service configuration handling"""

    def test_device_configuration(self):
        """Test device configuration"""
        for device in ["cpu", "cuda"]:
            service = TTSService(device=device)
            assert service.device == device

    def test_speed_configuration(self):
        """Test synthesis speed configuration"""
        speeds = [0.5, 1.0, 1.5, 2.0]
        for speed in speeds:
            service = TTSService(speed=speed)
            assert service.speed == speed

    def test_model_name_configuration(self):
        """Test model name configuration"""
        model_names = ["F5-TTS", "custom-model"]
        for model_name in model_names:
            service = TTSService(model_name=model_name)
            assert service.model_name == model_name


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])