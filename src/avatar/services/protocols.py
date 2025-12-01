"""
Service provider protocols (interfaces)

Define contracts for STT/LLM/TTS services.
Enables switching between local models and API services.

Design Philosophy (Linus Torvalds):
- "Good programmers worry about data structures and their relationships"
- Protocol defines the data flow contract
- Implementation details are hidden behind the interface
"""
from typing import Protocol, AsyncIterator, Optional
from pathlib import Path


class STTProvider(Protocol):
    """
    Speech-to-Text provider interface

    Implementations:
    - WhisperSTTProvider: Local faster-whisper (CPU)
    - OpenAISTTProvider: OpenAI Whisper API
    - AzureSTTProvider: Azure Speech Services
    - GoogleSTTProvider: Google Cloud Speech-to-Text
    """

    async def transcribe(
        self,
        audio_path: Path,
        language: Optional[str] = None,
        **kwargs
    ) -> tuple[str, dict]:
        """
        Transcribe audio to text

        Args:
            audio_path: Path to audio file (WAV 16kHz mono recommended)
            language: ISO 639-1 language code (None for auto-detect)
            **kwargs: Provider-specific options

        Returns:
            (transcribed_text, metadata)

            metadata format:
            {
                "language": str,           # Detected or specified language
                "duration": float,         # Audio duration in seconds
                "segments_count": int,     # Number of transcription segments
                "confidence": float,       # Average confidence score (0-1)
                "provider": str,           # Provider name (e.g., "whisper_local")
            }

        Raises:
            FileNotFoundError: Audio file not found
            RuntimeError: Transcription failed
        """
        ...


class LLMProvider(Protocol):
    """
    Large Language Model provider interface

    Implementations:
    - VLLMProvider: Local vLLM with quantized models
    - OpenAILLMProvider: OpenAI GPT models
    - AnthropicLLMProvider: Anthropic Claude models
    - AzureLLMProvider: Azure OpenAI Service
    """

    async def chat_stream(
        self,
        messages: list[dict],
        max_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs
    ) -> AsyncIterator[str]:
        """
        Stream chat completion chunks (for lower TTFT)

        Args:
            messages: Chat history in OpenAI format
                      [{"role": "user", "content": "..."}]
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0-2.0)
            **kwargs: Provider-specific options

        Yields:
            Response text chunks (delta only, not cumulative)

        Example:
            async for chunk in llm.chat_stream(messages):
                print(chunk, end="", flush=True)

        Raises:
            RuntimeError: Generation failed
        """
        ...


class TTSProvider(Protocol):
    """
    Text-to-Speech provider interface

    Implementations:
    - F5TTSProvider: Local F5-TTS (fast mode)
    - CosyVoiceTTSProvider: Local CosyVoice2 (HQ mode)
    - ElevenLabsTTSProvider: ElevenLabs API
    - AzureTTSProvider: Azure Text-to-Speech
    - OpenAITTSProvider: OpenAI TTS API
    """

    async def synthesize(
        self,
        text: str,
        output_path: Path,
        ref_audio_path: Optional[Path] = None,
        ref_text: Optional[str] = None,
        **kwargs
    ) -> Path:
        """
        Synthesize speech from text (with optional voice cloning)

        Args:
            text: Text to synthesize
            output_path: Where to save synthesized audio
            ref_audio_path: Reference audio for voice cloning
            ref_text: Reference text matching ref_audio
            **kwargs: Provider-specific options

        Returns:
            Path to synthesized audio file (WAV format)

        Raises:
            RuntimeError: Synthesis failed
        """
        ...

    async def synthesize_fast(
        self,
        text: str,
        voice_profile_name: str,
        output_path: Path,
        **kwargs
    ) -> Path:
        """
        Fast synthesis using pre-registered voice profile

        Args:
            text: Text to synthesize
            voice_profile_name: Name of registered voice profile
            output_path: Where to save synthesized audio
            **kwargs: Provider-specific options

        Returns:
            Path to synthesized audio file (WAV format)

        Raises:
            FileNotFoundError: Voice profile not found
            RuntimeError: Synthesis failed
        """
        ...
