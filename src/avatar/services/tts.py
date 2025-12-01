"""
TTS Service Factory

Returns appropriate TTS provider based on AVATAR_TTS_PROVIDER configuration.
"""
import structlog
from avatar.core.config import config
from avatar.services.protocols import TTSProvider

logger = structlog.get_logger()

_tts_service: TTSProvider | None = None


async def get_tts_service() -> TTSProvider:
    """
    Get TTS service instance (singleton)

    Supported providers:
    - local: F5-TTS (fast mode)
    - elevenlabs: ElevenLabs API (future)
    - openai: OpenAI TTS API (future)
    - azure: Azure Text-to-Speech (future)

    Returns:
        TTSProvider implementation

    Raises:
        ValueError: Unknown provider
    """
    global _tts_service

    if _tts_service is not None:
        return _tts_service

    provider = config.TTS_PROVIDER.lower()

    if provider == "local":
        from avatar.services.tts_local import F5TTSProvider
        logger.info("tts.factory.init", provider="local")
        _tts_service = F5TTSProvider()

    # 未來擴展點
    # elif provider == "elevenlabs":
    #     from avatar.services.tts_elevenlabs import ElevenLabsTTSProvider
    #     _tts_service = ElevenLabsTTSProvider(
    #         api_key=config.TTS_API_KEY
    #     )

    # elif provider == "openai":
    #     from avatar.services.tts_openai import OpenAITTSProvider
    #     _tts_service = OpenAITTSProvider(
    #         api_key=config.TTS_API_KEY,
    #         model=config.TTS_API_MODEL,
    #         voice=config.TTS_API_VOICE
    #     )

    else:
        raise ValueError(
            f"Unknown TTS provider: '{provider}'\n"
            f"Supported providers: local, elevenlabs, openai, azure"
        )

    logger.info("tts.factory.ready", provider=provider)
    return _tts_service
