"""
Audio Utils Tests - Linus Style (Test Real Audio Processing)

Tests actual audio conversion functionality with real files.
No mock bullsh*t - test what actually works.
"""

import pytest
import tempfile
import shutil
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from avatar.core.audio_utils import convert_to_wav, validate_audio_for_whisper
from avatar.core.config import config


class TestAudioUtilsReal:
    """Test audio utilities with REAL audio files (Linus approved)"""

    @pytest.fixture
    def test_audio_file(self):
        """Use actual test audio file"""
        test_file = config.AUDIO_RAW / "test_sample.wav"
        if test_file.exists():
            return test_file

        # Use voice profile audio as backup
        voice_file = config.AUDIO_PROFILES / "test_profile" / "reference.wav"
        if voice_file.exists():
            return voice_file

        pytest.skip("No test audio file available")

    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory"""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    def test_convert_to_wav_real_file(self, test_audio_file, temp_output_dir):
        """Test WAV conversion with real audio file"""
        output_path = temp_output_dir / "converted.wav"

        # Convert actual audio file
        result_path, metadata = convert_to_wav(
            input_path=test_audio_file,
            output_path=output_path,
            target_sample_rate=16000,
            target_channels=1
        )

        # Verify conversion worked
        assert result_path == output_path
        assert output_path.exists()
        assert output_path.stat().st_size > 1000  # Should have audio data

        # Check metadata makes sense (REAL keys from actual function)
        assert isinstance(metadata, dict)
        assert 'original_sample_rate' in metadata
        assert 'original_channels' in metadata
        assert metadata['converted_sample_rate'] == 16000
        assert metadata['converted_channels'] == 1

    def test_validate_audio_for_whisper_real_file(self, test_audio_file):
        """Test Whisper audio validation with real file"""
        # Test with actual audio file
        is_valid = validate_audio_for_whisper(test_audio_file)

        assert isinstance(is_valid, bool)
        # Should be valid if file exists and has audio data
        assert is_valid is True

    def test_validate_audio_for_whisper_nonexistent(self):
        """Test Whisper validation with non-existent file"""
        nonexistent = Path("/tmp/nonexistent_audio.wav")

        is_valid = validate_audio_for_whisper(nonexistent)
        assert is_valid is False

    def test_convert_different_sample_rates(self, test_audio_file, temp_output_dir):
        """Test conversion to different sample rates"""
        sample_rates = [8000, 16000, 22050, 44100]

        for rate in sample_rates:
            output_path = temp_output_dir / f"test_{rate}hz.wav"

            result_path, metadata = convert_to_wav(
                input_path=test_audio_file,
                output_path=output_path,
                target_sample_rate=rate,
                target_channels=1
            )

            assert output_path.exists()
            assert metadata['converted_sample_rate'] == rate

    def test_convert_to_stereo(self, test_audio_file, temp_output_dir):
        """Test conversion to stereo"""
        output_path = temp_output_dir / "stereo.wav"

        result_path, metadata = convert_to_wav(
            input_path=test_audio_file,
            output_path=output_path,
            target_sample_rate=16000,
            target_channels=2  # Stereo
        )

        assert output_path.exists()
        assert metadata['converted_channels'] == 2


class TestAudioUtilsErrorHandling:
    """Test audio utils error handling (Real scenarios)"""

    def test_convert_nonexistent_file(self):
        """Test conversion with non-existent file"""
        nonexistent = Path("/tmp/nonexistent_audio.wav")
        output = Path("/tmp/output.wav")

        with pytest.raises(FileNotFoundError):
            convert_to_wav(nonexistent, output)

    def test_convert_to_invalid_output_path(self, tmp_path):
        """Test conversion with invalid output path"""
        # Use actual test file if available
        input_file = config.AUDIO_RAW / "test_sample.wav"
        if not input_file.exists():
            pytest.skip("No test audio file")

        # Try to write to a directory instead of file
        invalid_output = tmp_path
        invalid_output.mkdir(exist_ok=True)

        with pytest.raises((IsADirectoryError, OSError)):
            convert_to_wav(input_file, invalid_output)

    def test_convert_error_handling_nonexistent_file(self):
        """Test conversion error handling with nonexistent file"""
        nonexistent = Path("/tmp/nonexistent_audio.wav")
        output = Path("/tmp/output.wav")

        # Should fail validation first
        is_valid = validate_audio_for_whisper(nonexistent)
        assert is_valid is False

        # Conversion should also fail
        with pytest.raises(FileNotFoundError):
            convert_to_wav(nonexistent, output)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])