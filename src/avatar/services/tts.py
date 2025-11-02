"""
Text-to-Speech service using F5-TTS

Provides async interface for fast text-to-speech synthesis.
Supports voice cloning from reference audio.
"""

import asyncio
from pathlib import Path
from typing import Optional, Union

import structlog
import torch
import torchaudio
from cached_path import cached_path

from avatar.core.config import config

logger = structlog.get_logger()


def _load_voice_profile(profile_name: str) -> tuple[Path, str]:
    """
    Load voice profile from disk (helper function)

    Separates file system operations from TTS business logic.

    Args:
        profile_name: Name of voice profile directory

    Returns:
        Tuple of (ref_audio_path, ref_text)

    Raises:
        FileNotFoundError: Profile directory or audio files not found
    """
    profile_dir = config.AUDIO_PROFILES / profile_name

    if not profile_dir.exists():
        raise FileNotFoundError(
            f"Voice profile directory not found: {profile_dir}"
        )

    # Find reference audio (first .wav file)
    ref_audio_files = list(profile_dir.glob("*.wav"))
    if not ref_audio_files:
        raise FileNotFoundError(
            f"No .wav files found in profile: {profile_name}"
        )

    ref_audio_path = ref_audio_files[0]

    # Load reference text
    ref_text_path = profile_dir / "reference.txt"
    if not ref_text_path.exists():
        raise FileNotFoundError(
            f"reference.txt not found in profile: {profile_name}\n"
            f"Expected at: {ref_text_path}"
        )

    ref_text = ref_text_path.read_text(encoding="utf-8").strip()

    if not ref_text:
        raise ValueError(
            f"reference.txt is empty in profile: {profile_name}"
        )

    return ref_audio_path, ref_text


class TTSService:
    """
    Text-to-Speech service powered by F5-TTS

    Features:
    - GPU-accelerated synthesis
    - Voice cloning from reference audio
    - Fast synthesis mode (target: ≤1.5s)
    - Lazy model loading
    - Async API to avoid blocking
    """

    def __init__(
        self,
        model_name: str = "F5-TTS",
        device: Optional[str] = None,
        speed: float = config.F5_TTS_SPEED
    ):
        """
        Initialize TTS service

        Args:
            model_name: F5-TTS model variant (F5-TTS or E2-TTS)
            device: Device for inference (auto-detect GPU if available)
            speed: Speech speed multiplier (1.0 = normal, >1.0 = faster)
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.speed = speed
        self._model = None
        self._vocoder = None

        logger.info(
            "tts.init",
            model=model_name,
            device=self.device,
            speed=speed
        )

    def _load_model(self):
        """Lazy load F5-TTS model (first call only)"""
        if self._model is None:
            logger.info("tts.loading_model", model=self.model_name)

            try:
                # Import F5-TTS components (module-level utilities)
                from f5_tts.infer.utils_infer import load_model, load_vocoder

                # Load model
                self._model, self._model_config = load_model(
                    self.model_name,
                    self.device
                )

                # Load vocoder (for converting to waveform)
                self._vocoder = load_vocoder(vocoder_name="vocos", device=self.device)

                logger.info("tts.model_loaded", model=self.model_name, device=self.device)

            except ImportError as e:
                logger.error("tts.import_failed", error=str(e))
                raise RuntimeError(
                    f"F5-TTS not installed. Run: poetry add f5-tts\n{e}"
                ) from e
            except Exception as e:
                logger.error("tts.load_failed", error=str(e))
                raise RuntimeError(f"Failed to load F5-TTS model: {e}") from e

    async def synthesize(
        self,
        text: str,
        ref_audio_path: Union[str, Path],
        ref_text: str,
        output_path: Union[str, Path],
        remove_silence: bool = True
    ) -> Path:
        """
        Synthesize speech from text using voice cloning

        Args:
            text: Text to synthesize
            ref_audio_path: Path to reference audio for voice cloning
            ref_text: Transcript of reference audio
            output_path: Path to save synthesized audio
            remove_silence: Remove leading/trailing silence

        Returns:
            Path to generated audio file

        Raises:
            FileNotFoundError: Reference audio not found
            RuntimeError: Synthesis failed
        """
        # Validate reference audio exists
        ref_audio_path = Path(ref_audio_path)
        if not ref_audio_path.exists():
            logger.error("tts.ref_audio_not_found", path=str(ref_audio_path))
            raise FileNotFoundError(f"Reference audio not found: {ref_audio_path}")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(
            "tts.synthesize_start",
            text_length=len(text),
            ref_audio=str(ref_audio_path),
            output=str(output_path)
        )

        # Load model if not already loaded
        self._load_model()

        # Run synthesis in thread pool to avoid blocking
        loop = asyncio.get_event_loop()

        try:
            await loop.run_in_executor(
                None,
                self._synthesize_blocking,
                text,
                ref_audio_path,
                ref_text,
                output_path,
                remove_silence
            )

            # Verify output exists
            if not output_path.exists():
                raise RuntimeError("Output file not created")

            audio_size = output_path.stat().st_size

            logger.info(
                "tts.synthesize_complete",
                output=str(output_path),
                size_bytes=audio_size
            )

            return output_path

        except Exception as e:
            logger.error(
                "tts.synthesize_failed",
                error=str(e),
                text=text[:100]
            )
            raise RuntimeError(f"TTS synthesis failed: {e}") from e

    def _synthesize_blocking(
        self,
        text: str,
        ref_audio_path: Path,
        ref_text: str,
        output_path: Path,
        remove_silence: bool
    ):
        """
        Blocking synthesis function (runs in thread pool)

        Uses F5-TTS utility functions directly (not stored as instance variables).
        """
        # Import F5-TTS utilities (module-level functions, not instance state)
        from f5_tts.infer.utils_infer import (
            infer_process,
            preprocess_ref_audio_text,
            remove_silence_for_generated_wav
        )

        # Preprocess reference audio
        ref_audio, ref_text = preprocess_ref_audio_text(
            str(ref_audio_path),
            ref_text
        )

        # Generate speech
        with torch.inference_mode():
            # Inference
            generated_audio, final_sample_rate, _ = infer_process(
                ref_audio=ref_audio,
                ref_text=ref_text,
                gen_text=text,
                model_obj=self._model,
                vocoder=self._vocoder,
                device=self.device,
                speed=self.speed
            )

        # Remove silence if requested
        if remove_silence:
            generated_audio = remove_silence_for_generated_wav(generated_audio)

        # Save to file
        torchaudio.save(
            str(output_path),
            generated_audio.unsqueeze(0),
            final_sample_rate
        )

    async def synthesize_fast(
        self,
        text: str,
        voice_profile_name: str,
        output_path: Union[str, Path]
    ) -> Path:
        """
        Convenience method for synthesis using pre-stored voice profile

        This method loads the voice profile and delegates to synthesize().
        File system operations are handled by _load_voice_profile().

        Args:
            text: Text to synthesize
            voice_profile_name: Name of stored voice profile
            output_path: Path to save synthesized audio

        Returns:
            Path to generated audio file

        Raises:
            FileNotFoundError: Voice profile or required files not found
            ValueError: Profile files are invalid
            RuntimeError: Synthesis failed
        """
        logger.info(
            "tts.fast_synthesis_start",
            profile=voice_profile_name,
            text_length=len(text)
        )

        # Load voice profile (file system operations separated)
        ref_audio_path, ref_text = _load_voice_profile(voice_profile_name)

        # Delegate to core synthesis method
        return await self.synthesize(
            text=text,
            ref_audio_path=ref_audio_path,
            ref_text=ref_text,
            output_path=output_path,
            remove_silence=True
        )

    def unload_model(self):
        """Unload model to free VRAM"""
        if self._model is not None:
            logger.info("tts.unloading_model")
            del self._model
            del self._vocoder
            self._model = None
            self._vocoder = None

            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info("tts.model_unloaded")


# Global singleton instance
_tts_service: Optional[TTSService] = None
_tts_service_lock = asyncio.Lock()


async def get_tts_service() -> TTSService:
    """
    Get global TTS service instance (singleton pattern)

    Thread-safe singleton using asyncio.Lock to prevent race conditions
    in concurrent initialization.

    Returns:
        TTSService instance
    """
    global _tts_service

    # Fast path: if already initialized, return immediately
    if _tts_service is not None:
        return _tts_service

    # Slow path: acquire lock and initialize
    async with _tts_service_lock:
        # Double-check after acquiring lock (another coroutine might have initialized)
        if _tts_service is None:
            _tts_service = TTSService()
            logger.info("tts.service_created")

    return _tts_service
