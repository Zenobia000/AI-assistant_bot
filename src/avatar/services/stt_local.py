"""
Speech-to-Text service using faster-whisper (Local Provider)

Provides async interface for audio transcription using Whisper models.
Uses CPU inference to avoid VRAM contention with LLM/TTS.

Implements: STTProvider protocol (see protocols.py)
"""

import asyncio
from pathlib import Path
from typing import Optional, Tuple

import structlog
from faster_whisper import WhisperModel

from avatar.core.config import config

logger = structlog.get_logger()


class WhisperSTTProvider:
    """
    Speech-to-Text service powered by faster-whisper

    Features:
    - CPU-based inference (no VRAM usage)
    - Lazy model loading
    - Auto language detection
    - Async API to avoid blocking
    """

    def __init__(
        self,
        model_size: str = "base",
        device: str = "cpu",
        compute_type: str = "int8"
    ):
        """
        Initialize STT service

        Args:
            model_size: Whisper model size (tiny, base, small, medium, large)
            device: Device for inference (cpu recommended)
            compute_type: Quantization type (int8, float16, float32)
        """
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self._model: Optional[WhisperModel] = None

        logger.info(
            "stt.init",
            model_size=model_size,
            device=device,
            compute_type=compute_type
        )

    def _load_model(self):
        """Lazy load Whisper model (first call only)"""
        if self._model is None:
            logger.info("stt.loading_model", model_size=self.model_size)

            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type
            )

            logger.info("stt.model_loaded", model_size=self.model_size)

    async def transcribe(
        self,
        audio_path: Path,
        language: Optional[str] = None,
        beam_size: int = 5,
        vad_filter: bool = True
    ) -> Tuple[str, dict]:
        """
        Transcribe audio file to text

        Args:
            audio_path: Path to audio file
            language: Language code (e.g., 'zh', 'en'), None for auto-detection
            beam_size: Beam size for decoding (higher = slower but more accurate)
            vad_filter: Enable Voice Activity Detection to filter silence

        Returns:
            Tuple of (transcribed_text, metadata)
            metadata includes: language, duration, segments_count

        Raises:
            FileNotFoundError: Audio file not found
            RuntimeError: Transcription failed
        """
        # Validate audio file exists
        if not audio_path.exists():
            logger.error("stt.file_not_found", path=str(audio_path))
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        logger.info(
            "stt.transcribe_start",
            audio=str(audio_path),
            language=language or "auto-detect"
        )

        # Load model if not already loaded
        self._load_model()

        # Run transcription in thread pool to avoid blocking
        loop = asyncio.get_event_loop()

        try:
            segments, info = await loop.run_in_executor(
                None,
                lambda: self._model.transcribe(
                    str(audio_path),
                    language=language,
                    beam_size=beam_size,
                    vad_filter=vad_filter
                )
            )

            # Collect all segments
            segment_list = list(segments)
            full_text = " ".join(seg.text.strip() for seg in segment_list)

            metadata = {
                "language": info.language,
                "language_probability": info.language_probability,
                "duration": info.duration,
                "segments_count": len(segment_list)
            }

            logger.info(
                "stt.transcribe_complete",
                text_length=len(full_text),
                duration_sec=info.duration,
                language=info.language,
                segments=len(segment_list)
            )

            return full_text, metadata

        except Exception as e:
            logger.error(
                "stt.transcribe_failed",
                error=str(e),
                audio=str(audio_path)
            )
            raise RuntimeError(f"Transcription failed: {e}") from e

    def unload_model(self):
        """Unload model to free memory"""
        if self._model is not None:
            logger.info("stt.unloading_model")
            del self._model
            self._model = None
            logger.info("stt.model_unloaded")


# Global singleton instance
_stt_service: Optional[WhisperSTTProvider] = None
_stt_service_lock = asyncio.Lock()


async def get_stt_service() -> WhisperSTTProvider:
    """
    Get global STT service instance (singleton pattern)

    Thread-safe singleton using asyncio.Lock to prevent race conditions
    in concurrent initialization.

    Returns:
        WhisperSTTProvider instance
    """
    global _stt_service

    # Fast path: if already initialized, return immediately
    if _stt_service is not None:
        return _stt_service

    # Slow path: acquire lock and initialize
    async with _stt_service_lock:
        # Double-check after acquiring lock (another coroutine might have initialized)
        if _stt_service is None:
            model_size = config.WHISPER_MODEL_SIZE
            _stt_service = WhisperSTTProvider(
                model_size=model_size,
                device="cpu",
                compute_type="int8"
            )
            logger.info("stt.service_created", model_size=model_size)

    return _stt_service
