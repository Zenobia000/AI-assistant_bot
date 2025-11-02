"""
LLM inference service using vLLM

Provides async interface for text generation using Qwen2.5-7B-Instruct.
Supports streaming responses for low Time-To-First-Token (TTFT).
"""

import asyncio
from typing import AsyncIterator, Optional

import structlog
from vllm import AsyncLLMEngine, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs

from avatar.core.config import config

logger = structlog.get_logger()


class LLMService:
    """
    LLM inference service powered by vLLM

    Features:
    - GPU-accelerated inference
    - Streaming response support
    - Lazy model loading
    - Optimized for low TTFT (Time To First Token)
    """

    def __init__(
        self,
        model_path: str = config.VLLM_MODEL,
        gpu_memory_utilization: float = config.VLLM_GPU_MEMORY,
        max_model_len: int = config.VLLM_MAX_TOKENS
    ):
        """
        Initialize LLM service

        Args:
            model_path: HuggingFace model path or local path
            gpu_memory_utilization: Fraction of GPU memory to use (0.0-1.0)
            max_model_len: Maximum sequence length
        """
        self.model_path = model_path
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self._engine: Optional[AsyncLLMEngine] = None

        logger.info(
            "llm.init",
            model=model_path,
            gpu_memory=f"{gpu_memory_utilization*100:.0f}%",
            max_tokens=max_model_len
        )

    async def _load_model(self):
        """Lazy load vLLM engine (first call only)"""
        if self._engine is None:
            logger.info("llm.loading_model", model=self.model_path)

            engine_args = AsyncEngineArgs(
                model=self.model_path,
                gpu_memory_utilization=self.gpu_memory_utilization,
                max_model_len=self.max_model_len,
                trust_remote_code=True,  # Required for Qwen models
                dtype="auto",
                enforce_eager=False  # Enable CUDA graph for better performance
            )

            self._engine = AsyncLLMEngine.from_engine_args(engine_args)

            logger.info("llm.model_loaded", model=self.model_path)

    def _create_sampling_params(
        self,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop: Optional[list[str]]
    ) -> SamplingParams:
        """
        Create sampling parameters (DRY helper)

        Extracted to avoid duplication between generate() and generate_stream().
        """
        return SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop or []
        )

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: Optional[list[str]] = None
    ) -> str:
        """
        Generate text completion (non-streaming)

        Args:
            prompt: Input text prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 = greedy, higher = more random)
            top_p: Nucleus sampling parameter
            stop: Stop sequences

        Returns:
            Generated text
        """
        await self._load_model()

        # Use helper to create sampling params (DRY)
        sampling_params = self._create_sampling_params(
            max_tokens, temperature, top_p, stop
        )

        logger.info(
            "llm.generate_start",
            prompt_len=len(prompt),
            max_tokens=max_tokens,
            temperature=temperature
        )

        try:
            # Generate using async engine
            request_id = f"req-{asyncio.current_task().get_name()}"
            results_generator = self._engine.generate(
                prompt,
                sampling_params,
                request_id
            )

            # Collect final result
            final_output = None
            async for request_output in results_generator:
                final_output = request_output

            if final_output is None:
                raise RuntimeError("No output generated")

            generated_text = final_output.outputs[0].text

            logger.info(
                "llm.generate_complete",
                output_len=len(generated_text),
                tokens=len(final_output.outputs[0].token_ids)
            )

            return generated_text

        except Exception as e:
            logger.error("llm.generate_failed", error=str(e))
            raise RuntimeError(f"LLM generation failed: {e}") from e

    async def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: Optional[list[str]] = None
    ) -> AsyncIterator[str]:
        """
        Generate text completion with streaming (yields tokens as they're generated)

        Args:
            prompt: Input text prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            stop: Stop sequences

        Yields:
            Generated text chunks
        """
        await self._load_model()

        # Use helper to create sampling params (DRY)
        sampling_params = self._create_sampling_params(
            max_tokens, temperature, top_p, stop
        )

        logger.info(
            "llm.stream_start",
            prompt_len=len(prompt),
            max_tokens=max_tokens
        )

        try:
            request_id = f"stream-{asyncio.current_task().get_name()}"
            results_generator = self._engine.generate(
                prompt,
                sampling_params,
                request_id
            )

            previous_text = ""
            token_count = 0

            async for request_output in results_generator:
                current_text = request_output.outputs[0].text
                new_text = current_text[len(previous_text):]

                if new_text:
                    token_count += 1
                    yield new_text
                    previous_text = current_text

            logger.info(
                "llm.stream_complete",
                total_tokens=token_count,
                total_chars=len(previous_text)
            )

        except Exception as e:
            logger.error("llm.stream_failed", error=str(e))
            raise RuntimeError(f"LLM streaming failed: {e}") from e

    async def chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7
    ) -> str:
        """
        Chat completion with message history

        Args:
            messages: List of message dicts with 'role' and 'content'
                      Example: [{'role': 'user', 'content': 'Hello'}]
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Assistant's response text
        """
        # Format messages into Qwen2.5 chat template
        # Qwen2.5-Instruct uses: <|im_start|>role\ncontent<|im_end|>
        formatted_prompt = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            formatted_prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"

        # Add assistant prefix to trigger response
        formatted_prompt += "<|im_start|>assistant\n"

        logger.info(
            "llm.chat_start",
            messages_count=len(messages),
            prompt_len=len(formatted_prompt)
        )

        # Generate response
        response = await self.generate(
            formatted_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=["<|im_end|>"]
        )

        return response.strip()

    async def unload_model(self):
        """Unload model to free VRAM"""
        if self._engine is not None:
            logger.info("llm.unloading_model")
            # vLLM doesn't have explicit unload, relies on process termination
            del self._engine
            self._engine = None
            logger.info("llm.model_unloaded")


# Global singleton instance
_llm_service: Optional[LLMService] = None
_llm_service_lock = asyncio.Lock()


async def get_llm_service() -> LLMService:
    """
    Get global LLM service instance (singleton pattern)

    Thread-safe singleton using asyncio.Lock to prevent race conditions
    in concurrent initialization.

    Returns:
        LLMService instance
    """
    global _llm_service

    # Fast path: if already initialized, return immediately
    if _llm_service is not None:
        return _llm_service

    # Slow path: acquire lock and initialize
    async with _llm_service_lock:
        # Double-check after acquiring lock (another coroutine might have initialized)
        if _llm_service is None:
            _llm_service = LLMService()
            logger.info("llm.service_created")

    return _llm_service
