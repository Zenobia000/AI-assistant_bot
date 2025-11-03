"""
TDD Unit Tests for HQ TTS Service (CosyVoice)

Testing high-quality TTS service with CosyVoice integration.
"""

import pytest
import asyncio
from pathlib import Path
from unittest.mock import patch, MagicMock

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from avatar.services.tts_hq import TTSHQService, get_tts_hq_service
from avatar.core.config import config


class TestTTSHQServiceInitialization:
    """Test HQ TTS service initialization"""

    def test_tts_hq_service_init_defaults(self):
        """Test HQ TTS service initializes with default values"""
        service = TTSHQService()

        assert service.device == "cuda"
        assert service.model_name == "CosyVoice-300M"
        assert service.sample_rate == config.COSYVOICE_SAMPLE_RATE
        assert service.model is None  # Lazy loading

    def test_tts_hq_service_init_custom_values(self):
        """Test HQ TTS service with custom configuration"""
        service = TTSHQService(
            model_name="CosyVoice-1B",
            device="cpu",
            sample_rate=24000
        )

        assert service.device == "cpu"
        assert service.model_name == "CosyVoice-1B"
        assert service.sample_rate == 24000

    def test_tts_hq_service_singleton(self):
        """Test HQ TTS service singleton pattern"""
        service1 = get_tts_hq_service()
        service2 = get_tts_hq_service()

        assert service1 is service2


@pytest.mark.unit
@pytest.mark.slow
class TestTTSHQServiceModelLoading:
    """Test HQ TTS service model loading"""

    @pytest.mark.asyncio
    @patch('avatar.services.tts_hq.CosyVoice')
    async def test_load_model_cosyvoice_success(self, mock_cosyvoice_class):
        """Test successful CosyVoice model loading"""
        # Mock CosyVoice instance
        mock_model = MagicMock()
        mock_cosyvoice_class.return_value = mock_model

        service = TTSHQService()

        # Mock the import to avoid actual CosyVoice dependency
        with patch.dict('sys.modules', {'cosyvoice.cli.cosyvoice': MagicMock()}):
            await service._ensure_model_loaded()

        assert service.model is not None
        mock_cosyvoice_class.assert_called_once_with("CosyVoice-300M", device="cuda")

    @pytest.mark.asyncio
    async def test_model_loading_import_error(self):
        """Test handling of CosyVoice import error"""
        service = TTSHQService()

        # CosyVoice not installed scenario
        with pytest.raises(RuntimeError, match="CosyVoice not installed"):
            await service._ensure_model_loaded()

    @pytest.mark.asyncio
    @patch('avatar.services.tts_hq.CosyVoice')
    async def test_model_loading_idempotent(self, mock_cosyvoice_class):
        """Test that multiple calls to _ensure_model_loaded don't reload"""
        mock_model = MagicMock()
        mock_cosyvoice_class.return_value = mock_model

        service = TTSHQService()

        with patch.dict('sys.modules', {'cosyvoice.cli.cosyvoice': MagicMock()}):
            await service._ensure_model_loaded()
            first_model = service.model

            await service._ensure_model_loaded()
            second_model = service.model

        assert first_model is second_model
        mock_cosyvoice_class.assert_called_once()  # Only called once


@pytest.mark.unit
class TestTTSHQServiceSynthesis:
    """Test HQ TTS synthesis functionality"""

    @pytest.fixture
    def mock_cosyvoice_service(self):
        """Provide mocked HQ TTS service"""
        service = TTSHQService()

        # Mock the model
        mock_model = MagicMock()
        mock_model.inference.return_value = MagicMock()  # Mock audio data
        service._model = mock_model

        return service

    @pytest.fixture
    def test_reference_audio(self):
        """Provide path to test reference audio"""
        return config.AUDIO_PROFILES / "test_profile" / "reference.wav"

    @pytest.fixture
    def test_output_path(self, tmp_path):
        """Provide temporary output path"""
        return tmp_path / "test_hq_output.wav"

    @pytest.mark.asyncio
    @patch('torchaudio.save')
    async def test_synthesize_with_reference_audio(
        self,
        mock_torchaudio_save,
        mock_cosyvoice_service,
        test_reference_audio,
        test_output_path
    ):
        """Test HQ synthesis with reference audio"""
        if not test_reference_audio.exists():
            pytest.skip("Test reference audio not found")

        # Mock torchaudio.save to simulate file creation
        def mock_save(*args, **kwargs):
            test_output_path.touch()  # Create the file

        mock_torchaudio_save.side_effect = mock_save

        result_path = await mock_cosyvoice_service.synthesize(
            text="Hello, this is a high-quality TTS test.",
            ref_audio_path=test_reference_audio,
            ref_text="This is test reference text.",
            output_path=test_output_path,
            speaker_mode="clone"
        )

        assert result_path == test_output_path
        assert test_output_path.exists()

        # Verify CosyVoice inference was called with correct parameters
        mock_cosyvoice_service.model.inference.assert_called_once()
        call_args = mock_cosyvoice_service.model.inference.call_args
        assert call_args[1]["text"] == "Hello, this is a high-quality TTS test."
        assert call_args[1]["mode"] == "clone"

    @pytest.mark.asyncio
    @patch('torchaudio.save')
    async def test_synthesize_hq_with_voice_profile(
        self,
        mock_torchaudio_save,
        mock_cosyvoice_service,
        test_output_path
    ):
        """Test HQ synthesis using voice profile"""
        voice_profile_name = "test_profile"

        # Check if voice profile exists
        profile_path = config.AUDIO_PROFILES / voice_profile_name / "reference.wav"
        if not profile_path.exists():
            pytest.skip("Test voice profile not found")

        # Mock torchaudio.save to simulate file creation
        def mock_save(*args, **kwargs):
            test_output_path.touch()

        mock_torchaudio_save.side_effect = mock_save

        result_path = await mock_cosyvoice_service.synthesize_hq(
            text="This is high-quality TTS synthesis test.",
            voice_profile_name=voice_profile_name,
            output_path=test_output_path
        )

        assert result_path == test_output_path
        assert test_output_path.exists()

    @pytest.mark.asyncio
    async def test_synthesis_error_handling(self, mock_cosyvoice_service, tmp_path):
        """Test synthesis error handling"""
        # Test with missing reference audio
        nonexistent_ref = tmp_path / "nonexistent.wav"
        output_path = tmp_path / "output.wav"

        with pytest.raises(FileNotFoundError):
            await mock_cosyvoice_service.synthesize(
                text="Test text",
                ref_audio_path=nonexistent_ref,
                ref_text="Test ref",
                output_path=output_path
            )

        # Test with empty text
        test_ref = config.AUDIO_PROFILES / "test_profile" / "reference.wav"
        if test_ref.exists():
            with pytest.raises(ValueError, match="Text cannot be empty"):
                await mock_cosyvoice_service.synthesize(
                    text="",
                    ref_audio_path=test_ref,
                    ref_text="Test ref",
                    output_path=output_path
                )


@pytest.mark.unit
class TestTTSDualModeManager:
    """Test dual mode TTS manager"""

    @pytest.mark.asyncio
    @patch('avatar.services.tts.get_tts_service')
    @patch('avatar.services.tts_hq.get_tts_hq_service')
    async def test_dual_mode_synthesis(self, mock_get_hq, mock_get_fast):
        """Test dual mode synthesis with both fast and HQ"""
        from avatar.services.tts import get_tts_dual_mode_manager

        # Mock services
        mock_fast_service = MagicMock()
        mock_hq_service = MagicMock()

        mock_get_fast.return_value = mock_fast_service
        mock_get_hq.return_value = mock_hq_service

        # Mock synthesis results
        fast_path = Path("/tmp/fast.wav")
        hq_path = Path("/tmp/hq.wav")

        mock_fast_service.synthesize_fast = MagicMock(return_value=fast_path)
        mock_hq_service.synthesize_hq = MagicMock(return_value=hq_path)

        manager = get_tts_dual_mode_manager()

        result_fast, result_hq = await manager.synthesize_dual_mode(
            text="Test dual mode",
            voice_profile_name="test_profile",
            output_path_fast=fast_path,
            output_path_hq=hq_path,
            prefer_hq=True
        )

        assert result_fast == fast_path
        assert result_hq == hq_path

        # Verify both services were called
        mock_fast_service.synthesize_fast.assert_called_once()
        mock_hq_service.synthesize_hq.assert_called_once()

    @pytest.mark.asyncio
    @patch('avatar.services.tts.get_tts_service')
    async def test_dual_mode_fallback_to_fast_only(self, mock_get_fast):
        """Test dual mode falls back to fast when HQ unavailable"""
        from avatar.services.tts import get_tts_dual_mode_manager

        mock_fast_service = MagicMock()
        mock_get_fast.return_value = mock_fast_service

        fast_path = Path("/tmp/fast_only.wav")
        mock_fast_service.synthesize_fast = MagicMock(return_value=fast_path)

        manager = get_tts_dual_mode_manager()

        # HQ service unavailable (ImportError)
        with patch.object(manager, 'get_hq_service', return_value=None):
            result_fast, result_hq = await manager.synthesize_dual_mode(
                text="Test fallback",
                voice_profile_name="test_profile",
                output_path_fast=fast_path,
                prefer_hq=False
            )

        assert result_fast == fast_path
        assert result_hq is None
        mock_fast_service.synthesize_fast.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])