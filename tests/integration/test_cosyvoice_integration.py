"""
CosyVoice Integration Test

Tests CosyVoice2 high-quality TTS functionality for AVATAR project.
"""

import sys
import asyncio
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

# Add CosyVoice paths
cosyvoice_path = project_root / "CosyVoice"
sys.path.append(str(cosyvoice_path))
sys.path.append(str(cosyvoice_path / "third_party" / "Matcha-TTS"))

import torchaudio
import pytest
from avatar.core.config import config


class TestCosyVoiceIntegration:
    """Test CosyVoice2 integration with AVATAR system."""

    @pytest.fixture(scope="class")
    def cosyvoice_model(self):
        """Load CosyVoice2 model once for all tests."""
        from cosyvoice.cli.cosyvoice import CosyVoice2

        model_path = cosyvoice_path / "pretrained_models" / "CosyVoice2-0.5B"
        cosyvoice = CosyVoice2(str(model_path))
        return cosyvoice

    @pytest.fixture
    def test_audio_data(self):
        """Prepare test audio and text data."""
        ref_audio_path = config.AUDIO_RAW / "test_sample.wav"
        ref_text = "老鼓,我今天很需要可以不得逃"  # From STT test
        gen_text = "Hello, this is a test of CosyVoice2 voice cloning."

        return {
            "ref_audio_path": ref_audio_path,
            "ref_text": ref_text,
            "gen_text": gen_text
        }

    def test_cosyvoice_model_loading(self, cosyvoice_model):
        """Test CosyVoice2 model can be loaded successfully."""
        assert cosyvoice_model is not None
        assert cosyvoice_model.sample_rate == 24000
        assert hasattr(cosyvoice_model, 'inference_zero_shot')

    def test_zero_shot_synthesis(self, cosyvoice_model, test_audio_data):
        """Test zero-shot voice synthesis."""
        # Load reference audio
        ref_audio, sr = torchaudio.load(test_audio_data["ref_audio_path"])
        if sr != cosyvoice_model.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, cosyvoice_model.sample_rate)
            ref_audio = resampler(ref_audio)

        # Measure synthesis time
        start_time = time.time()

        results = cosyvoice_model.inference_zero_shot(
            test_audio_data["gen_text"],
            test_audio_data["ref_text"],
            ref_audio
        )

        synthesis_time = time.time() - start_time

        # Verify results - convert generator to list
        result_list = list(results)
        assert len(result_list) > 0
        result = result_list[0]
        assert 'tts_speech' in result
        assert result['tts_speech'].shape[0] > 0  # Has audio data

        # Check performance (should be faster than F5-TTS)
        assert synthesis_time < 10.0, f"Synthesis too slow: {synthesis_time:.2f}s"

        return result, synthesis_time

    def test_multilingual_synthesis(self, cosyvoice_model, test_audio_data):
        """Test multilingual synthesis capability."""
        # Load reference audio
        ref_audio, sr = torchaudio.load(test_audio_data["ref_audio_path"])
        if sr != cosyvoice_model.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, cosyvoice_model.sample_rate)
            ref_audio = resampler(ref_audio)

        # Test Chinese synthesis
        chinese_text = "你好世界，這是中文語音合成測試。"
        results = cosyvoice_model.inference_zero_shot(
            chinese_text,
            test_audio_data["ref_text"],
            ref_audio
        )

        result_list = list(results)
        assert len(result_list) > 0
        assert result_list[0]['tts_speech'].shape[0] > 0

    def test_audio_quality(self, cosyvoice_model, test_audio_data):
        """Test audio quality metrics."""
        # Load reference audio
        ref_audio, sr = torchaudio.load(test_audio_data["ref_audio_path"])
        if sr != cosyvoice_model.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, cosyvoice_model.sample_rate)
            ref_audio = resampler(ref_audio)

        results = cosyvoice_model.inference_zero_shot(
            test_audio_data["gen_text"],
            test_audio_data["ref_text"],
            ref_audio
        )

        result_list = list(results)
        audio = result_list[0]['tts_speech']

        # Basic quality checks
        assert audio.max() <= 1.0, "Audio levels too high"
        assert audio.min() >= -1.0, "Audio levels too low"
        assert audio.std() > 0.01, "Audio too quiet or silent"

        # Check duration (should be reasonable for the text)
        duration_seconds = audio.shape[-1] / cosyvoice_model.sample_rate
        assert 5.0 <= duration_seconds <= 15.0, f"Unexpected duration: {duration_seconds:.2f}s"

    @pytest.mark.performance
    def test_synthesis_performance(self, cosyvoice_model, test_audio_data):
        """Test synthesis performance metrics."""
        # Load reference audio
        ref_audio, sr = torchaudio.load(test_audio_data["ref_audio_path"])
        if sr != cosyvoice_model.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, cosyvoice_model.sample_rate)
            ref_audio = resampler(ref_audio)

        # Run multiple synthesis to get average performance
        times = []
        for i in range(3):
            start_time = time.time()
            results = cosyvoice_model.inference_zero_shot(
                f"{test_audio_data['gen_text']} Run {i+1}.",
                test_audio_data["ref_text"],
                ref_audio
            )
            result_list = list(results)  # Convert generator to list
            times.append(time.time() - start_time)

        avg_time = sum(times) / len(times)

        # Performance assertions
        assert avg_time < 8.0, f"Average synthesis time too slow: {avg_time:.2f}s"

        # Calculate RTF (Real Time Factor)
        audio_duration = result_list[0]['tts_speech'].shape[-1] / cosyvoice_model.sample_rate
        rtf = avg_time / audio_duration

        assert rtf < 1.0, f"RTF should be < 1.0 (faster than real-time), got {rtf:.2f}"

        print(f"CosyVoice Performance Metrics:")
        print(f"  Average synthesis time: {avg_time:.2f}s")
        print(f"  Audio duration: {audio_duration:.2f}s")
        print(f"  RTF: {rtf:.2f}")


async def test_cosyvoice_integration_async():
    """Async integration test for CosyVoice."""
    print("🔊 Running CosyVoice Integration Test...")

    # This would be called from AVATAR's async TTS service
    from cosyvoice.cli.cosyvoice import CosyVoice2

    model_path = cosyvoice_path / "pretrained_models" / "CosyVoice2-0.5B"
    cosyvoice = CosyVoice2(str(model_path))

    # Test basic synthesis
    ref_audio_path = config.AUDIO_RAW / "test_sample.wav"
    ref_audio, sr = torchaudio.load(ref_audio_path)
    if sr != cosyvoice.sample_rate:
        resampler = torchaudio.transforms.Resample(sr, cosyvoice.sample_rate)
        ref_audio = resampler(ref_audio)

    results = cosyvoice.inference_zero_shot(
        "Hello from async CosyVoice integration!",
        "老鼓,我今天很需要可以不得逃",
        ref_audio
    )

    # Save test output - results is a generator, convert to list
    output_path = config.AUDIO_TTS_HQ / "cosyvoice_async_test.wav"
    result_list = list(results)
    torchaudio.save(output_path, result_list[0]['tts_speech'], cosyvoice.sample_rate)

    print(f"✅ Async CosyVoice test completed!")
    print(f"Output saved to: {output_path}")
    return True


if __name__ == "__main__":
    # Run async test
    result = asyncio.run(test_cosyvoice_integration_async())
    if result:
        print("🎉 CosyVoice integration test passed!")
    else:
        print("❌ CosyVoice integration test failed!")