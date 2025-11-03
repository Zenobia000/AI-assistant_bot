"""
Simple test audio generator - no dependencies on avatar package
"""

from pathlib import Path
import torch
import torchaudio
import math

def create_test_tone(output_path: Path, duration_sec: float = 2.0):
    """Create a simple sine wave test tone"""

    print(f"🔊 Generating {duration_sec}s test tone...")

    sample_rate = 16000
    num_samples = int(sample_rate * duration_sec)

    # Generate 440Hz sine wave (A4 note)
    t = torch.linspace(0, duration_sec, num_samples)
    frequency = 440.0
    waveform = (torch.sin(2 * math.pi * frequency * t) * 0.3).unsqueeze(0)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save as WAV 16kHz mono PCM
    torchaudio.save(
        str(output_path),
        waveform,
        sample_rate,
        encoding="PCM_S",
        bits_per_sample=16
    )

    print(f"✅ Test audio created: {output_path}")
    print(f"   Sample rate: 16kHz")
    print(f"   Channels: Mono")
    print(f"   Duration: {duration_sec}s")
    print(f"   Size: {output_path.stat().st_size / 1024:.1f} KB")

if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    output_path = project_root / "audio" / "raw" / "test_sample.wav"

    print("\n" + "="*60)
    print("CREATE TEST AUDIO")
    print("="*60 + "\n")

    create_test_tone(output_path, duration_sec=2.0)

    print("\n✅ Ready for E2E testing!")
    print(f"   Run: poetry run python tests/e2e_pipeline_test.py\n")
