"""
LLM Service Factory

Returns appropriate LLM provider based on AVATAR_LLM_PROVIDER configuration.
"""
import structlog
from avatar.core.config import config
from avatar.services.protocols import LLMProvider

logger = structlog.get_logger()

_llm_service: LLMProvider | None = None


async def get_llm_service() -> LLMProvider:
    """
    Get LLM service instance (singleton)

    Supported providers:
    - local: vLLM with quantized models (AWQ)
    - openai: OpenAI GPT models (future)
    - anthropic: Anthropic Claude models (future)
    - azure: Azure OpenAI Service (future)

    Returns:
        LLMProvider implementation

    Raises:
        ValueError: Unknown provider
    """
    global _llm_service

    if _llm_service is not None:
        return _llm_service

    provider = config.LLM_PROVIDER.lower()

    if provider == "local":
        from avatar.services.llm_local import VLLMProvider
        logger.info("llm.factory.init",
                   provider="local",
                   model=config.VLLM_MODEL,
                   gpu_memory=config.VLLM_GPU_MEMORY)
        _llm_service = VLLMProvider(
            model_path=config.VLLM_MODEL,
            gpu_memory_utilization=config.VLLM_GPU_MEMORY,
            max_model_len=config.VLLM_MAX_TOKENS
        )

    # 未來擴展點
    # elif provider == "openai":
    #     from avatar.services.llm_openai import OpenAILLMProvider
    #     _llm_service = OpenAILLMProvider(
    #         api_key=config.LLM_API_KEY,
    #         model=config.LLM_API_MODEL
    #     )

    # elif provider == "anthropic":
    #     from avatar.services.llm_anthropic import AnthropicLLMProvider
    #     _llm_service = AnthropicLLMProvider(
    #         api_key=config.LLM_API_KEY,
    #         model=config.LLM_API_MODEL
    #     )

    else:
        raise ValueError(
            f"Unknown LLM provider: '{provider}'\n"
            f"Supported providers: local, openai, anthropic, azure"
        )

    logger.info("llm.factory.ready", provider=provider)
    return _llm_service
