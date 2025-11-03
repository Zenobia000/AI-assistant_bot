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
        model_name: str = "CosyVoice-300M",
        device: Optional[str] = None,
        sample_rate: int = config.COSYVOICE_SAMPLE_RATE
    ):
        """
        Initialize HQ TTS service

        Args:
            model_name: CosyVoice model variant
            device: Device for inference (auto-detect GPU if available)
            sample_rate: Output audio sample rate
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.sample_rate = sample_rate
        self._model = None  # Will hold CosyVoice instance

        logger.info(
            "tts_hq.init",
            device=self.device,
            model=self.model_name,
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

        logger.info("tts_hq.loading_model", model=self.model_name)

        # Load CosyVoice model in executor to avoid blocking
        loop = asyncio.get_event_loop()

        def _load_model():
            try:
                # Import CosyVoice (assumed to be installed separately)
                from cosyvoice.cli.cosyvoice import CosyVoice

                # Initialize model
                model = CosyVoice(self.model_name, device=self.device)
                return model

            except ImportError as e:
                raise RuntimeError(
                    f"CosyVoice not installed. Please install CosyVoice: {e}"
                )
            except Exception as e:
                raise RuntimeError(f"Failed to load CosyVoice model: {e}")

        try:
            self._model = await loop.run_in_executor(None, _load_model)
            logger.info("tts_hq.model_loaded", device=self.device, model=self.model_name)

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
                # CosyVoice synthesis call
                # This is a placeholder - actual API depends on CosyVoice implementation
                audio_data = self._model.inference(
                    text=text,
                    ref_audio=str(ref_audio_path),
                    ref_text=ref_text,
                    mode=speaker_mode
                )

                # Save audio data to file
                import torchaudio
                torchaudio.save(
                    str(output_path),
                    audio_data,
                    self.sample_rate
                )

                return output_path

            except Exception as e:
                raise RuntimeError(f"CosyVoice synthesis failed: {e}")

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