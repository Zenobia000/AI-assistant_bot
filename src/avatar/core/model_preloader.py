"""
Model Preloader for AVATAR - Eliminate Cold Start Latency

Loads all AI models during application startup to achieve target E2E latency.
Based on Linus's principle: "The best optimization is doing work once, not optimizing repetition."

Design Philosophy:
- Load everything at startup (simple, predictable)
- No lazy loading for production models
- Clear separation of concerns
- Robust error handling with fallbacks
"""

import asyncio
import time
from typing import Dict, Any, Optional

import structlog
import torch

from avatar.services.stt import get_stt_service
from avatar.services.llm import get_llm_service
from avatar.services.tts import get_tts_service
from avatar.services.tts_hq import get_tts_hq_service
from avatar.core.config import config

logger = structlog.get_logger()


class ModelPreloader:
    """
    Preloads all AI models during application startup

    Ensures all models are warm and ready for immediate inference,
    eliminating cold start latency that would otherwise kill user experience.
    """

    def __init__(self):
        self.preload_status = {
            "stt": {"loaded": False, "load_time": 0, "error": None},
            "llm": {"loaded": False, "load_time": 0, "error": None},
            "tts_fast": {"loaded": False, "load_time": 0, "error": None},
            "tts_hq": {"loaded": False, "load_time": 0, "error": None}
        }
        self.total_preload_time = 0

    async def preload_all_models(self, enable_hq_tts: bool = True) -> Dict[str, Any]:
        """
        Preload all AI models in optimal order

        Args:
            enable_hq_tts: Whether to preload HQ TTS (CosyVoice2)

        Returns:
            Dict with preload status and timing information
        """
        logger.info("model_preloader.start", enable_hq_tts=enable_hq_tts)
        start_total = time.time()

        # Preload in optimal order based on resource usage and dependencies
        # 1. STT first (CPU, fastest, no VRAM)
        await self._preload_stt()

        # 2. LLM (largest VRAM consumer, takes longest)
        await self._preload_llm()

        # 3. TTS Fast (medium VRAM, shared with LLM GPU)
        await self._preload_tts_fast()

        # 4. TTS HQ (if enabled, uses separate GPU ideally)
        if enable_hq_tts and config.TTS_ENABLE_HQ_MODE:
            await self._preload_tts_hq()

        self.total_preload_time = time.time() - start_total

        # Generate preload summary
        summary = self._generate_preload_summary()
        logger.info("model_preloader.complete",
                   total_time=self.total_preload_time,
                   summary=summary)

        return summary

    async def _preload_stt(self):
        """Preload Speech-to-Text model (Whisper)"""
        try:
            start = time.time()
            logger.info("model_preloader.stt_start")

            stt_service = await get_stt_service()

            # Warm up with dummy transcription to fully load model
            dummy_audio = config.AUDIO_RAW / "test_sample.wav"
            if dummy_audio.exists():
                await stt_service.transcribe(dummy_audio)

            load_time = time.time() - start
            self.preload_status["stt"].update({
                "loaded": True,
                "load_time": load_time
            })

            logger.info("model_preloader.stt_success", load_time=load_time)

        except Exception as e:
            self.preload_status["stt"]["error"] = str(e)
            logger.error("model_preloader.stt_failed", error=str(e))

    async def _preload_llm(self):
        """Preload Large Language Model (vLLM)"""
        try:
            start = time.time()
            logger.info("model_preloader.llm_start")

            llm_service = await get_llm_service()

            # Warm up with dummy inference to build CUDA graphs
            dummy_messages = [{"role": "user", "content": "Hello"}]
            dummy_response = ""
            async for chunk in llm_service.chat_stream(
                dummy_messages,
                max_tokens=5,
                temperature=0.1
            ):
                dummy_response += chunk
                break  # Just get first token

            load_time = time.time() - start
            self.preload_status["llm"].update({
                "loaded": True,
                "load_time": load_time
            })

            logger.info("model_preloader.llm_success",
                       load_time=load_time,
                       warmup_response=dummy_response[:50])

        except Exception as e:
            self.preload_status["llm"]["error"] = str(e)
            logger.error("model_preloader.llm_failed", error=str(e))

    async def _preload_tts_fast(self):
        """Preload Fast TTS model (F5-TTS)"""
        try:
            start = time.time()
            logger.info("model_preloader.tts_fast_start")

            tts_service = await get_tts_service()

            # Trigger model loading by accessing private model
            # This loads F5-TTS model without full synthesis
            tts_service._load_model()

            load_time = time.time() - start
            self.preload_status["tts_fast"].update({
                "loaded": True,
                "load_time": load_time
            })

            logger.info("model_preloader.tts_fast_success", load_time=load_time)

        except Exception as e:
            self.preload_status["tts_fast"]["error"] = str(e)
            logger.error("model_preloader.tts_fast_failed", error=str(e))

    async def _preload_tts_hq(self):
        """Preload High-Quality TTS model (CosyVoice2)"""
        try:
            start = time.time()
            logger.info("model_preloader.tts_hq_start")

            tts_hq_service = get_tts_hq_service()

            # Trigger model loading
            await tts_hq_service._ensure_model_loaded()

            load_time = time.time() - start
            self.preload_status["tts_hq"].update({
                "loaded": True,
                "load_time": load_time
            })

            logger.info("model_preloader.tts_hq_success", load_time=load_time)

        except Exception as e:
            self.preload_status["tts_hq"]["error"] = str(e)
            logger.error("model_preloader.tts_hq_failed", error=str(e))

    def _generate_preload_summary(self) -> Dict[str, Any]:
        """Generate comprehensive preload summary"""
        summary = {
            "total_preload_time": self.total_preload_time,
            "models_loaded": sum(1 for status in self.preload_status.values() if status["loaded"]),
            "total_models": len(self.preload_status),
            "success_rate": 0,
            "details": self.preload_status.copy(),
            "recommendations": []
        }

        # Calculate success rate
        loaded_count = summary["models_loaded"]
        total_count = summary["total_models"]
        summary["success_rate"] = (loaded_count / total_count) * 100 if total_count > 0 else 0

        # Generate recommendations based on results
        if summary["success_rate"] == 100:
            summary["recommendations"].append("All models loaded successfully. E2E latency should be optimal.")
        else:
            failed_models = [name for name, status in self.preload_status.items() if not status["loaded"]]
            summary["recommendations"].append(f"Failed models: {failed_models}. Check errors and retry.")

        # VRAM usage recommendations
        total_load_time = sum(status["load_time"] for status in self.preload_status.values() if status["loaded"])
        if total_load_time > 30:
            summary["recommendations"].append("Consider parallel model loading to reduce startup time.")

        return summary

    def get_preload_status(self) -> Dict[str, Any]:
        """Get current preload status"""
        return {
            "status": self.preload_status.copy(),
            "total_time": self.total_preload_time,
            "loaded_models": [name for name, status in self.preload_status.items() if status["loaded"]],
            "failed_models": [name for name, status in self.preload_status.items() if status.get("error")]
        }

    async def warm_up_models(self) -> Dict[str, float]:
        """
        Run warm-up inference on all loaded models

        Ensures CUDA graphs are built and models are at peak performance.
        Returns timing for each model's first inference.
        """
        warmup_times = {}

        # Warm up STT
        if self.preload_status["stt"]["loaded"]:
            try:
                start = time.time()
                stt_service = await get_stt_service()
                dummy_audio = config.AUDIO_RAW / "test_sample.wav"
                if dummy_audio.exists():
                    await stt_service.transcribe(dummy_audio)
                warmup_times["stt"] = time.time() - start
            except Exception as e:
                logger.warning("model_preloader.stt_warmup_failed", error=str(e))

        # Warm up LLM
        if self.preload_status["llm"]["loaded"]:
            try:
                start = time.time()
                llm_service = await get_llm_service()
                messages = [{"role": "user", "content": "Hi"}]
                async for _ in llm_service.chat_stream(messages, max_tokens=1):
                    break
                warmup_times["llm"] = time.time() - start
            except Exception as e:
                logger.warning("model_preloader.llm_warmup_failed", error=str(e))

        logger.info("model_preloader.warmup_complete", warmup_times=warmup_times)
        return warmup_times


# Global singleton
_model_preloader: Optional[ModelPreloader] = None


def get_model_preloader() -> ModelPreloader:
    """Get singleton model preloader instance"""
    global _model_preloader

    if _model_preloader is None:
        _model_preloader = ModelPreloader()

    return _model_preloader


async def preload_all_models(enable_hq_tts: bool = True) -> Dict[str, Any]:
    """
    Convenience function to preload all models

    Args:
        enable_hq_tts: Whether to preload high-quality TTS

    Returns:
        Preload summary with timing and status information
    """
    preloader = get_model_preloader()
    return await preloader.preload_all_models(enable_hq_tts=enable_hq_tts)