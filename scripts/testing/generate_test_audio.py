"""
Generate test audio file for E2E testing

Creates a simple WAV file (16kHz mono) with synthesized speech
using gTTS (Google Text-to-Speech) or fallback to sine wave tone.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from avatar.core.config import config

def generate_test_audio_gtts(output_path: Path, text: str = "Hello, this is a test."):
    """Generate test audio using gTTS (requires internet connection)"""
    try:
        from gtts import gTTS
        import tempfile
        import torchaudio

        print(f"📝 Generating audio from text: '{text}'")

        # Generate speech using gTTS
        tts = gTTS(text=text, lang='en')

        # Save to temporary MP3 file
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            tts.save(str(tmp_path))

        print(f"🎵 Converting MP3 to WAV 16kHz mono...")

        # Load MP3 and convert to WAV
        waveform, sample_rate = torchaudio.load(str(tmp_path))

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample to 16kHz if needed
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=16000
            )
            waveform = resampler(waveform)

        # Save as WAV
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torchaudio.save(
            str(output_path),
            waveform,
            16000,
            encoding="PCM_S",
            bits_per_sample=16
        )

        # Clean up temporary file
        tmp_path.unlink()

        print(f"✅ Test audio created: {output_path}")
        print(f"   Duration: {waveform.shape[1] / 16000:.2f}s")
        print(f"   Size: {output_path.stat().st_size / 1024:.1f} KB")

        return True

    except ImportError:
        print("⚠️  gTTS not installed. Install with: poetry add gtts")
        return False
    except Exception as e:
        print(f"❌ gTTS generation failed: {e}")
        return False


def generate_test_audio_tone(output_path: Path, duration_sec: float = 1.0):
    """Generate test audio using simple sine wave (fallback method)"""
    try:
        import torch
        import torchaudio
        import math

        print(f"🔊 Generating {duration_sec}s sine wave test tone...")

        sample_rate = 16000
        num_samples = int(sample_rate * duration_sec)

        # Generate 440Hz sine wave (A4 note)
        t = torch.linspace(0, duration_sec, num_samples)
        frequency = 440.0  # A4
        waveform = (torch.sin(2 * math.pi * frequency * t) * 0.3).unsqueeze(0)

        # Save as WAV
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torchaudio.save(
            str(output_path),
            waveform,
            sample_rate,
            encoding="PCM_S",
            bits_per_sample=16
        )

        print(f"✅ Test tone created: {output_path}")
        print(f"   Frequency: {frequency}Hz")
        print(f"   Duration: {duration_sec}s")
        print(f"   Size: {output_path.stat().st_size / 1024:.1f} KB")

        return True

    except Exception as e:
        print(f"❌ Tone generation failed: {e}")
        return False


def main():
    """Generate test audio file"""

    print("\n" + "="*60)
    print("GENERATE TEST AUDIO FOR E2E TESTING")
    print("="*60 + "\n")

    output_path = config.AUDIO_RAW / "test_sample.wav"

    # Try gTTS first (better for STT testing)
    if generate_test_audio_gtts(output_path, text="Hello, how can I help you today?"):
        return

    # Fallback to sine wave tone
    print("\n⚠️  Falling back to sine wave tone (less useful for STT testing)")
    if not generate_test_audio_tone(output_path, duration_sec=2.0):
        print("\n❌ All audio generation methods failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
