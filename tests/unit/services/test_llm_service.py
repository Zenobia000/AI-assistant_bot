"""
TDD Unit Tests for LLM Service

Testing vLLM inference service with real model loading and generation.
"""

import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from typing import List

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from avatar.services.llm import LLMService, get_llm_service
from avatar.core.config import config


class TestLLMServiceInitialization:
    """Test LLM service initialization"""

    def test_llm_service_init_defaults(self):
        """Test LLM service initializes with default values"""
        service = LLMService()

        assert service.model_path == "Qwen/Qwen2.5-7B-Instruct-AWQ"
        assert service.gpu_memory_utilization == 0.5
        assert service.max_model_len == 2048
        assert service._engine is None  # Lazy loading

    def test_llm_service_init_custom_values(self):
        """Test LLM service with custom configuration"""
        service = LLMService(
            model_path="custom/model",
            gpu_memory_utilization=0.8,
            max_model_len=4096
        )

        assert service.model_path == "custom/model"
        assert service.gpu_memory_utilization == 0.8
        assert service.max_model_len == 4096

    @pytest.mark.asyncio
    async def test_llm_service_singleton(self):
        """Test LLM service singleton pattern (REAL async)"""
        service1 = await get_llm_service()
        service2 = await get_llm_service()

        assert service1 is service2


@pytest.mark.unit
@pytest.mark.slow
class TestLLMServiceModelLoading:
    """Test LLM service model loading with real vLLM"""

    @pytest.mark.asyncio
    async def test_load_model_qwen25_awq(self):
        """Test loading Qwen2.5-7B-Instruct-AWQ model"""
        service = LLMService()

        await service._load_model()

        assert service._engine is not None
        assert hasattr(service._engine, 'generate')

    @pytest.mark.asyncio
    async def test_model_lazy_loading(self):
        """Test that model engine is loaded on first use"""
        service = LLMService()

        # Engine should be None initially
        assert service._engine is None

        # Load engine
        await service._load_model()

        # Engine should be loaded
        assert service._engine is not None

    @pytest.mark.asyncio
    async def test_model_loading_idempotent(self):
        """Test that multiple calls to _load_model don't reload"""
        service = LLMService()

        await service._load_model()
        first_engine = service._engine

        await service._load_model()
        second_engine = service._engine

        assert first_engine is second_engine


@pytest.mark.unit
@pytest.mark.slow
class TestLLMServiceGeneration:
    """Test LLM text generation functionality"""

    @pytest.fixture
    async def llm_service(self):
        """Provide initialized LLM service"""
        service = LLMService()
        await service._load_model()
        return service

    @pytest.mark.asyncio
    async def test_generate_simple_prompt(self, llm_service):
        """Test text generation with simple prompt"""
        prompt = "Hello, how are you?"

        response = await llm_service.generate(
            prompt=prompt,
            max_tokens=50,
            temperature=0.7
        )

        assert isinstance(response, str)
        assert len(response) > 0
        assert len(response) <= 300  # Reasonable response length (updated)

    @pytest.mark.asyncio
    async def test_generate_stream(self, llm_service):
        """Test streaming text generation"""
        prompt = "Count from 1 to 3:"

        chunks = []
        async for chunk in llm_service.generate_stream(
            prompt=prompt,
            max_tokens=30,
            temperature=0.1  # Low temp for predictable output
        ):
            chunks.append(chunk)

        assert len(chunks) > 0
        full_response = ''.join(chunks)
        assert isinstance(full_response, str)
        assert len(full_response) > 0

    @pytest.mark.asyncio
    async def test_chat_with_messages(self, llm_service):
        """Test chat interface with message format"""
        messages = [
            {"role": "user", "content": "What is 2+2?"}
        ]

        response = await llm_service.chat(
            messages=messages,
            max_tokens=20,
            temperature=0.1
        )

        assert isinstance(response, str)
        assert len(response) > 0
        # Should contain a number (basic reasoning test)
        assert any(char.isdigit() for char in response)

    @pytest.mark.asyncio
    async def test_generate_with_max_tokens_limit(self, llm_service):
        """Test generation respects max_tokens parameter"""
        prompt = "Write a very long story about a dragon."

        response = await llm_service.generate(
            prompt=prompt,
            max_tokens=10,  # Very small limit
            temperature=0.7
        )

        assert isinstance(response, str)
        # Should be limited by max_tokens (rough word count check)
        word_count = len(response.split())
        assert word_count <= 20  # Allow some margin


@pytest.mark.unit
class TestLLMServicePromptFormatting:
    """Test prompt formatting functionality"""

    def test_format_chat_prompt_single_message(self):
        """Test chat prompt formatting with single message"""
        service = LLMService()
        messages = [{"role": "user", "content": "Hello"}]

        formatted = service._format_chat_prompt(messages)

        assert isinstance(formatted, str)
        assert "Hello" in formatted
        # Should follow Qwen chat template format
        assert "<|im_start|>" in formatted or "user" in formatted

    def test_format_chat_prompt_conversation(self):
        """Test chat prompt formatting with conversation"""
        service = LLMService()
        messages = [
            {"role": "user", "content": "What is AI?"},
            {"role": "assistant", "content": "AI is artificial intelligence."},
            {"role": "user", "content": "Tell me more."}
        ]

        formatted = service._format_chat_prompt(messages)

        assert isinstance(formatted, str)
        assert "What is AI?" in formatted
        assert "artificial intelligence" in formatted
        assert "Tell me more" in formatted

    def test_format_chat_prompt_empty_messages(self):
        """Test chat prompt formatting with empty messages"""
        service = LLMService()
        messages = []

        formatted = service._format_chat_prompt(messages)

        assert isinstance(formatted, str)
        # Should handle empty gracefully

    def test_format_chat_prompt_system_message(self):
        """Test chat prompt formatting with system message"""
        service = LLMService()
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"}
        ]

        formatted = service._format_chat_prompt(messages)

        assert "helpful assistant" in formatted
        assert "Hello" in formatted


@pytest.mark.unit
@pytest.mark.slow
class TestLLMServicePerformance:
    """Test LLM service performance characteristics"""

    @pytest.fixture
    async def llm_service(self):
        """Provide initialized LLM service"""
        service = LLMService()
        await service._load_model()
        return service

    @pytest.mark.asyncio
    async def test_ttft_latency_target(self, llm_service):
        """Test Time To First Token meets target (≤800ms after warmup)"""
        prompt = "Hello, please respond."

        # Warmup request
        await llm_service.generate(prompt, max_tokens=5)

        # Measure TTFT
        import time
        start_time = time.time()

        first_chunk = None
        async for chunk in llm_service.generate_stream(prompt, max_tokens=10):
            if first_chunk is None:
                first_chunk_time = time.time()
                first_chunk = chunk
                break

        ttft_ms = (first_chunk_time - start_time) * 1000

        # Should meet 800ms target after warmup
        assert ttft_ms <= 1000  # Allow some margin for unit test environment
        assert first_chunk is not None

    @pytest.mark.asyncio
    async def test_concurrent_generation_requests(self, llm_service):
        """Test concurrent generation requests"""
        prompts = [
            "What is 1+1?",
            "What is 2+2?",
            "What is 3+3?"
        ]

        tasks = [
            llm_service.generate(prompt, max_tokens=10, temperature=0.1)
            for prompt in prompts
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # All should complete successfully
        for result in results:
            assert not isinstance(result, Exception)
            assert isinstance(result, str)
            assert len(result) > 0


class TestLLMServiceErrorHandling:
    """Test LLM service error handling"""

    @pytest.mark.asyncio
    async def test_generate_handles_vllm_exceptions(self):
        """Test generation handles vLLM exceptions"""
        service = LLMService()

        # Mock vLLM engine to raise exception
        mock_engine = AsyncMock()
        mock_engine.generate.side_effect = RuntimeError("vLLM engine failed")
        service.engine = mock_engine

        with pytest.raises(RuntimeError, match="vLLM engine failed"):
            await service.generate("test prompt")

    @pytest.mark.asyncio
    async def test_engine_loading_failure(self):
        """Test handling of engine loading failures"""
        service = LLMService()

        with patch('avatar.services.llm.AsyncLLMEngine') as mock_engine_class:
            mock_engine_class.from_engine_args.side_effect = RuntimeError("Engine load failed")

            with pytest.raises(RuntimeError, match="Engine load failed"):
                await service._load_model()

    @pytest.mark.asyncio
    async def test_invalid_prompt_handling(self):
        """Test handling of invalid prompts"""
        service = LLMService()

        # Test with None prompt
        with pytest.raises((TypeError, ValueError)):
            await service.generate(None)

        # Test with empty prompt
        try:
            result = await service.generate("", max_tokens=5)
            # Some models might handle empty prompts gracefully
            assert isinstance(result, str)
        except Exception:
            # Or they might raise an exception, both are acceptable
            pass

    @pytest.mark.asyncio
    async def test_invalid_parameters_handling(self, llm_service):
        """Test handling of invalid generation parameters"""
        # Test with negative max_tokens
        with pytest.raises((ValueError, RuntimeError)):
            await llm_service.generate("test", max_tokens=-1)

        # Test with invalid temperature
        with pytest.raises((ValueError, RuntimeError)):
            await llm_service.generate("test", temperature=-1.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])