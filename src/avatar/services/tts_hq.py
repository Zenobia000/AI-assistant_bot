"""
High-Quality Text-to-Speech service using CosyVoice

Provides async interface for high-quality text-to-speech synthesis.
Supports advanced voice cloning and natural prosody generation.

Design Philosophy (Linus-style):
- Simple, predictable interface
- Lazy loading for memory efficiency
- No special cases in error handling
"""

import asyncio
from pathlib import Path
from typing import Optional, Union

import structlog
import torch

from avatar.core.config import config

logger = structlog.get_logger()


class TTSHQService:
    """
    High-Quality Text-to-Speech service powered by CosyVoice

    Features:
    - Superior voice quality compared to F5-TTS
    - Advanced prosody control
    - Multi-speaker voice cloning
    - GPU-accelerated synthesis
    - Async API to avoid blocking
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        sample_rate: Optional[int] = None
    ):
        """
        Initialize HQ TTS service

        Args:
            model_path: Path to CosyVoice2 model directory (uses config default if None)
            device: Device for inference (auto-detect GPU if available)
            sample_rate: Output audio sample rate (uses config default if None)
        """
        self.model_path = Path(model_path or config.TTS_HQ_MODEL_PATH)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.sample_rate = sample_rate or config.COSYVOICE_SAMPLE_RATE
        self._model = None  # Will hold CosyVoice2 instance

        logger.info(
            "tts_hq.init",
            device=self.device,
            model_path=str(self.model_path),
            sample_rate=self.sample_rate
        )

    @property
    def model(self):
        """Get the loaded model instance"""
        return self._model

    async def _ensure_model_loaded(self):
        """
        Lazy load CosyVoice model

        This is called automatically before synthesis.
        Model loading is expensive so we defer it until needed.
        """
        if self._model is not None:
            return

        logger.info("tts_hq.loading_model", model_path=str(self.model_path))

        # Load CosyVoice2 model in executor to avoid blocking
        loop = asyncio.get_event_loop()

        def _load_model():
            try:
                import sys

                # Add CosyVoice paths to Python path
                cosyvoice_root = config.PROJECT_ROOT / "CosyVoice"
                sys.path.append(str(cosyvoice_root))
                sys.path.append(str(cosyvoice_root / "third_party" / "Matcha-TTS"))

                # Import CosyVoice2
                from cosyvoice.cli.cosyvoice import CosyVoice2

                # Initialize CosyVoice2 model
                model = CosyVoice2(str(self.model_path))
                return model

            except ImportError as e:
                raise RuntimeError(
                    f"CosyVoice2 not available. Please check CosyVoice installation: {e}"
                )
            except Exception as e:
                raise RuntimeError(f"Failed to load CosyVoice2 model: {e}")

        try:
            self._model = await loop.run_in_executor(None, _load_model)
            logger.info("tts_hq.model_loaded", device=self.device, model_path=str(self.model_path))

        except Exception as e:
            logger.error("tts_hq.model_load_failed", error=str(e))
            raise

    async def synthesize(
        self,
        text: str,
        ref_audio_path: Union[str, Path],
        ref_text: str,
        output_path: Union[str, Path],
        speaker_mode: str = "clone"
    ) -> Path:
        """
        Synthesize high-quality speech from text

        Args:
            text: Text to synthesize
            ref_audio_path: Reference audio for voice cloning
            ref_text: Reference text corresponding to ref_audio
            output_path: Where to save the synthesized audio
            speaker_mode: Voice cloning mode ("clone", "cross-lingual")

        Returns:
            Path to the synthesized audio file

        Raises:
            RuntimeError: If synthesis fails
            FileNotFoundError: If reference audio not found
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        ref_audio_path = Path(ref_audio_path)
        output_path = Path(output_path)

        if not ref_audio_path.exists():
            raise FileNotFoundError(f"Reference audio not found: {ref_audio_path}")

        # Ensure model is loaded
        await self._ensure_model_loaded()

        logger.info(
            "tts_hq.synthesize_start",
            output=str(output_path),
            ref_audio=str(ref_audio_path),
            text_length=len(text),
            speaker_mode=speaker_mode
        )

        # Create output directory if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Run synthesis in executor to avoid blocking
        loop = asyncio.get_event_loop()

        def _synthesize():
            try:
                import torchaudio

                # Load reference audio
                ref_audio, sr = torchaudio.load(str(ref_audio_path))
                if sr != self.sample_rate:
                    resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                    ref_audio = resampler(ref_audio)

                # CosyVoice2 zero-shot synthesis
                results = self._model.inference_zero_shot(
                    text,           # Text to synthesize
                    ref_text,       # Reference text
                    ref_audio       # Reference audio tensor
                )

                # Convert generator to list and get first result
                result_list = list(results)
                if not result_list:
                    raise RuntimeError("CosyVoice2 returned no results")

                audio_data = result_list[0]['tts_speech']

                # Save audio data to file
                torchaudio.save(
                    str(output_path),
                    audio_data,
                    self.sample_rate
                )

                return output_path

            except Exception as e:
                raise RuntimeError(f"CosyVoice2 synthesis failed: {e}")

        try:
            result_path = await loop.run_in_executor(None, _synthesize)

            # Verify output file was created
            if not result_path.exists():
                raise RuntimeError("Synthesis completed but output file not found")

            file_size = result_path.stat().st_size
            logger.info(
                "tts_hq.synthesize_complete",
                output=str(result_path),
                size_bytes=file_size
            )

            return result_path

        except Exception as e:
            logger.error("tts_hq.synthesize_failed", error=str(e))
            raise

    async def synthesize_hq(
        self,
        text: str,
        voice_profile_name: str,
        output_path: Union[str, Path]
    ) -> Path:
        """
        High-quality synthesis using voice profile

        Convenience method that loads voice profile and calls synthesize.

        Args:
            text: Text to synthesize
            voice_profile_name: Name of voice profile directory
            output_path: Where to save the synthesized audio

        Returns:
            Path to the synthesized audio file
        """
        from avatar.services.tts import _load_voice_profile

        # Load voice profile
        ref_audio_path, ref_text = _load_voice_profile(voice_profile_name)

        # Call main synthesis method
        return await self.synthesize(
            text=text,
            ref_audio_path=ref_audio_path,
            ref_text=ref_text,
            output_path=output_path,
            speaker_mode="clone"
        )


# Singleton instance
_tts_hq_service: Optional[TTSHQService] = None


def get_tts_hq_service() -> TTSHQService:
    """
    Get singleton TTS HQ service instance

    Returns:
        TTSHQService: The singleton service instance
    """
    global _tts_hq_service

    if _tts_hq_service is None:
        _tts_hq_service = TTSHQService()
        logger.info("tts_hq.service_created")

    return _tts_hq_service