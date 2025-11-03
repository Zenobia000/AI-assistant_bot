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

        # Use configured GPU device consistently
        if device:
            self.device = device
        elif torch.cuda.is_available():
            # Use the configured optimal GPU
            gpu_device = config.get_optimal_gpu() if hasattr(config, 'get_optimal_gpu') else 0
            self.device = f"cuda:{gpu_device}"
            # Set CUDA device for consistency
            torch.cuda.set_device(gpu_device)
        else:
            self.device = "cpu"

        self.speed = speed
        self._model = None  # Will hold F5TTS instance

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
                # Import F5-TTS high-level API
                from f5_tts.api import F5TTS

                # Load model (auto-downloads from HuggingFace)
                # model_name maps: "F5-TTS" -> "F5TTS_v1_Base"
                model_id = "F5TTS_v1_Base" if self.model_name == "F5-TTS" else self.model_name

                self._model = F5TTS(
                    model=model_id,
                    device=self.device
                )

                logger.info("tts.model_loaded", model=model_id, device=self.device)

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

        Uses F5TTS high-level API for inference.
        """
        # Workaround for F5-TTS bug: progress parameter expects object with .tqdm() method
        # but code has bug where it calls progress.tqdm() instead of progress()
        class NoOpProgress:
            """No-op progress bar to suppress F5-TTS progress output"""
            @staticmethod
            def tqdm(iterable):
                """Fake tqdm method that just returns the iterable"""
                return iterable

        # Use F5TTS.infer() which handles everything
        self._model.infer(
            ref_file=str(ref_audio_path),
            ref_text=ref_text,
            gen_text=text,
            file_wave=str(output_path),
            speed=self.speed,
            remove_silence=remove_silence,
            show_info=lambda x: None,  # Suppress print output
            progress=NoOpProgress  # Suppress progress bar
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
            self._model = None

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


class TTSDualModeManager:
    """
    Manages both Fast (F5-TTS) and HQ (CosyVoice) TTS modes

    Provides unified interface for dual TTS modes with intelligent
    mode selection based on requirements and system resources.

    Design Philosophy:
    - Simple unified interface
    - Resource-aware mode selection
    - Graceful fallback to fast mode
    """

    def __init__(self):
        self._fast_service = None
        self._hq_service = None

    async def get_fast_service(self):
        """Get fast TTS service (F5-TTS)"""
        if self._fast_service is None:
            self._fast_service = await get_tts_service()
        return self._fast_service

    async def get_hq_service(self):
        """Get HQ TTS service (CosyVoice)"""
        if self._hq_service is None:
            try:
                from avatar.services.tts_hq import get_tts_hq_service
                self._hq_service = get_tts_hq_service()
            except ImportError:
                logger.warning("tts.hq_unavailable", reason="CosyVoice not installed")
                return None
        return self._hq_service

    async def synthesize_dual_mode(
        self,
        text: str,
        voice_profile_name: str,
        output_path_fast: Path,
        output_path_hq: Optional[Path] = None,
        prefer_hq: bool = False
    ) -> tuple[Path, Optional[Path]]:
        """
        Synthesize using dual mode strategy

        Args:
            text: Text to synthesize
            voice_profile_name: Voice profile to use
            output_path_fast: Output path for fast TTS
            output_path_hq: Output path for HQ TTS (optional)
            prefer_hq: Whether to prioritize HQ over fast

        Returns:
            Tuple of (fast_path, hq_path) - hq_path may be None if unavailable
        """
        fast_service = await self.get_fast_service()
        hq_service = await self.get_hq_service()

        fast_result = None
        hq_result = None

        if prefer_hq and hq_service and output_path_hq:
            # Try HQ first
            try:
                hq_result = await hq_service.synthesize_hq(
                    text=text,
                    voice_profile_name=voice_profile_name,
                    output_path=output_path_hq
                )
                logger.info("tts.dual_mode.hq_success")
            except Exception as e:
                logger.warning("tts.dual_mode.hq_failed", error=str(e))

        # Always generate fast version (fallback or parallel)
        try:
            fast_result = await fast_service.synthesize_fast(
                text=text,
                voice_profile_name=voice_profile_name,
                output_path=output_path_fast
            )
            logger.info("tts.dual_mode.fast_success")
        except Exception as e:
            logger.error("tts.dual_mode.fast_failed", error=str(e))
            if hq_result is None:
                raise RuntimeError("Both fast and HQ TTS failed")

        return fast_result, hq_result


# Global dual mode manager
_dual_mode_manager: Optional[TTSDualModeManager] = None


def get_tts_dual_mode_manager() -> TTSDualModeManager:
    """Get singleton dual mode TTS manager"""
    global _dual_mode_manager

    if _dual_mode_manager is None:
        _dual_mode_manager = TTSDualModeManager()
        logger.info("tts.dual_mode_manager_created")

    return _dual_mode_manager
