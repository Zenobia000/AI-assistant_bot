"""
STT Service Factory

Returns appropriate STT provider based on AVATAR_STT_PROVIDER configuration.

Linus Principle: "Never break userspace"
- Existing get_stt_service() calls continue to work
- Default behavior (local mode) is unchanged
- Adding new providers requires zero changes to callers
"""
import structlog
from avatar.core.config import config
from avatar.services.protocols import STTProvider

logger = structlog.get_logger()

# Singleton instance
_stt_service: STTProvider | None = None


async def get_stt_service() -> STTProvider:
    """
    Get STT service instance (singleton)

    Returns appropriate provider based on AVATAR_STT_PROVIDER env var.

    Supported providers:
    - local: Whisper (faster-whisper) on CPU
    - openai: OpenAI Whisper API (future)
    - azure: Azure Speech Services (future)
    - google: Google Cloud Speech-to-Text (future)

    Returns:
        STTProvider implementation

    Raises:
        ValueError: Unknown provider
        RuntimeError: Provider initialization failed

    Example:
        stt = await get_stt_service()
        text, metadata = await stt.transcribe(audio_path)
    """
    global _stt_service

    if _stt_service is not None:
        return _stt_service

    provider = config.STT_PROVIDER.lower()

    if provider == "local":
        from avatar.services.stt_local import WhisperSTTProvider
        logger.info("stt.factory.init",
                   provider="local",
                   model=config.WHISPER_MODEL_SIZE,
                   device=config.WHISPER_DEVICE)
        _stt_service = WhisperSTTProvider(
            model_size=config.WHISPER_MODEL_SIZE,
            device=config.WHISPER_DEVICE,
            compute_type=config.WHISPER_COMPUTE_TYPE
        )

    # 未來擴展點 (選型完成後解除註釋)
    # elif provider == "openai":
    #     from avatar.services.stt_openai import OpenAISTTProvider
    #     logger.info("stt.factory.init", provider="openai", model=config.STT_API_MODEL)
    #     _stt_service = OpenAISTTProvider(
    #         api_key=config.STT_API_KEY,
    #         model=config.STT_API_MODEL,
    #         endpoint=config.STT_API_ENDPOINT
    #     )

    # elif provider == "azure":
    #     from avatar.services.stt_azure import AzureSTTProvider
    #     logger.info("stt.factory.init", provider="azure")
    #     _stt_service = AzureSTTProvider(
    #         api_key=config.STT_API_KEY,
    #         endpoint=config.STT_API_ENDPOINT
    #     )

    # elif provider == "google":
    #     from avatar.services.stt_google import GoogleSTTProvider
    #     logger.info("stt.factory.init", provider="google")
    #     _stt_service = GoogleSTTProvider(
    #         api_key=config.STT_API_KEY
    #     )

    else:
        raise ValueError(
            f"Unknown STT provider: '{provider}'\n"
            f"Supported providers: local, openai, azure, google\n"
            f"Set AVATAR_STT_PROVIDER environment variable"
        )

    logger.info("stt.factory.ready", provider=provider)
    return _stt_service
