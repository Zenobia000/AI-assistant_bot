"""
AVATAR - FastAPI Main Application

Entry point for the AI Voice Assistant API server.
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

import time
import structlog
from typing import Optional
from fastapi import FastAPI, Request, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from avatar.core.config import config
from avatar.core.model_preloader import preload_all_models, get_model_preloader
from avatar.core.security import get_security_headers, verify_api_token
from avatar.core.vram_monitor import get_vram_monitor
from avatar.core.logging_config import configure_logging
from avatar.core.error_handling import get_error_handler
from avatar.core.monitoring import setup_monitoring
from avatar.core.logging_config import get_metrics_collector
from avatar.api.websocket import websocket_endpoint
from avatar.api.voice_profiles import router as voice_profiles_router
from avatar.api.conversations import router as conversations_router
from avatar.api.monitoring import router as monitoring_router

# Configure unified structured logging
configure_logging()
logger = structlog.get_logger()

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)

# Rate limit exceeded handler
def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    logger.warning(
        "rate_limit_exceeded",
        client_ip=get_remote_address(request),
        path=request.url.path,
        limit=str(exc.detail)
    )
    return JSONResponse(
        status_code=429,
        content={"error": "Rate limit exceeded. Please try again later."},
        headers=get_security_headers()
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan manager

    Handles startup and shutdown events:
    - Validates configuration
    - Initializes AI models (lazy loading)
    - Cleanup resources on shutdown
    """
    # Startup
    logger.info("avatar.startup", version=app.version)

    # Validate configuration
    if not config.validate():
        logger.error("avatar.startup.failed", reason="config_validation_failed")
        raise RuntimeError("Configuration validation failed")

    logger.info("avatar.config.validated",
                database=str(config.DATABASE_PATH),
                max_sessions=config.MAX_CONCURRENT_SESSIONS,
                vram_limit_gb=config.VRAM_LIMIT_GB)

    # Preload AI models for optimal performance
    # This eliminates cold start latency for better user experience
    logger.info("avatar.models.preloading",
                message="Loading all AI models for optimal E2E latency")

    try:
        preload_summary = await preload_all_models(enable_hq_tts=config.TTS_ENABLE_HQ_MODE)

        logger.info("avatar.models.preloaded",
                   total_time=preload_summary["total_preload_time"],
                   loaded_models=preload_summary["models_loaded"],
                   success_rate=preload_summary["success_rate"])

        if preload_summary["success_rate"] < 100:
            logger.warning("avatar.models.partial_failure",
                          failed_models=[name for name, status in preload_summary["details"].items()
                                       if not status["loaded"]])
    except Exception as e:
        logger.error("avatar.models.preload_failed", error=str(e))
        # Continue startup even if preload fails (graceful degradation)
        logger.info("avatar.startup.degraded",
                   message="Continuing with lazy loading due to preload failure")

    # Setup integrated monitoring system
    error_handler = get_error_handler()
    metrics_collector = get_metrics_collector()
    setup_monitoring(error_handler, metrics_collector)

    logger.info("avatar.startup.complete")

    yield  # Server is running

    # Shutdown
    logger.info("avatar.shutdown", message="Cleaning up resources")
    # TODO: Cleanup AI model resources
    logger.info("avatar.shutdown.complete")


# Create FastAPI application
app = FastAPI(
    title="AVATAR - AI Voice Assistant",
    description="Real-time AI voice conversation system with voice cloning",
    version="0.1.0",
    lifespan=lifespan,
)

# Configure rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, rate_limit_handler)

# CORS middleware - security enhanced
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],  # Explicit methods only
    allow_headers=["Authorization", "Content-Type"],  # Explicit headers only
)

# Security headers middleware
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)

    # Add security headers
    for header_name, header_value in get_security_headers().items():
        response.headers[header_name] = header_value

    return response


# Health check endpoint
@app.get("/health", tags=["System"])
@limiter.limit("30/minute")  # Reasonable limit for health checks
async def health_check(request: Request):
    """Health check endpoint"""
    return JSONResponse({
        "status": "healthy",
        "version": app.version,
        "database": str(config.DATABASE_PATH.exists()),
    })


# System info endpoint
@app.get("/api/system/info", tags=["System"])
@limiter.limit("10/minute")
async def system_info(request: Request):
    """Get system information and resource status"""
    import torch

    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info = {
            "available": True,
            "device_name": torch.cuda.get_device_name(0),
            "total_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
            "allocated_memory_gb": round(torch.cuda.memory_allocated(0) / 1024**3, 2),
            "reserved_memory_gb": round(torch.cuda.memory_reserved(0) / 1024**3, 2),
        }
    else:
        gpu_info = {"available": False}

    return JSONResponse({
        "version": app.version,
        "config": {
            "max_concurrent_sessions": config.MAX_CONCURRENT_SESSIONS,
            "vram_limit_gb": config.VRAM_LIMIT_GB,
        },
        "gpu": gpu_info,
    })


# Model management endpoints
@app.get("/api/system/models/status", tags=["System", "Models"])
@limiter.limit("5/minute")
async def models_status(request: Request):
    """Get current model loading status"""
    preloader = get_model_preloader()
    status = preloader.get_preload_status()

    return JSONResponse({
        "preload_status": status["status"],
        "total_preload_time": status["total_time"],
        "loaded_models": status["loaded_models"],
        "failed_models": status["failed_models"],
        "models_ready": len(status["loaded_models"]),
        "total_models": len(status["status"])
    })


@app.post("/api/system/models/preload", tags=["System", "Models"])
@limiter.limit("1/minute")
async def trigger_model_preload(request: Request, enable_hq_tts: bool = True):
    """Manually trigger model preloading"""
    logger.info("api.models.preload_triggered", enable_hq_tts=enable_hq_tts)

    try:
        summary = await preload_all_models(enable_hq_tts=enable_hq_tts)

        return JSONResponse({
            "success": True,
            "message": "Model preloading completed",
            "summary": summary
        })

    except Exception as e:
        logger.error("api.models.preload_failed", error=str(e))
        return JSONResponse({
            "success": False,
            "message": f"Model preloading failed: {e}",
            "error": str(e)
        }, status_code=500)


@app.post("/api/system/models/warmup", tags=["System", "Models"])
@limiter.limit("2/minute")
async def trigger_model_warmup(request: Request):
    """Trigger model warm-up inference to optimize performance"""
    logger.info("api.models.warmup_triggered")

    try:
        preloader = get_model_preloader()
        warmup_times = await preloader.warm_up_models()

        return JSONResponse({
            "success": True,
            "message": "Model warm-up completed",
            "warmup_times": warmup_times,
            "total_warmup_time": sum(warmup_times.values())
        })

    except Exception as e:
        logger.error("api.models.warmup_failed", error=str(e))
        return JSONResponse({
            "success": False,
            "message": f"Model warm-up failed: {e}",
            "error": str(e)
        }, status_code=500)


# VRAM monitoring endpoints (Task 20)
@app.get("/api/system/vram/status", tags=["System", "VRAM"])
@limiter.limit("30/minute")
async def get_vram_status(request: Request):
    """Get real-time VRAM status for all GPUs"""
    logger.info("api.vram.status_requested")

    try:
        vram_monitor = get_vram_monitor()
        monitoring_stats = vram_monitor.get_monitoring_stats()

        return JSONResponse({
            "success": True,
            "vram_status": monitoring_stats,
            "timestamp": time.time()
        })

    except Exception as e:
        logger.error("api.vram.status_failed", error=str(e))
        return JSONResponse({
            "success": False,
            "error": "Failed to get VRAM status"
        }, status_code=500)


@app.get("/api/system/vram/history", tags=["System", "VRAM"])
@limiter.limit("10/minute")
async def get_vram_history(
    request: Request,
    device_id: int = Query(0, ge=0, description="GPU device ID"),
    minutes: int = Query(10, ge=1, le=60, description="History period in minutes")
):
    """Get VRAM usage history for specified GPU"""
    logger.info("api.vram.history_requested", device_id=device_id, minutes=minutes)

    try:
        vram_monitor = get_vram_monitor()
        trend_data = vram_monitor.get_usage_trend(device_id=device_id, minutes=minutes)

        return JSONResponse({
            "success": True,
            "trend_data": trend_data,
            "timestamp": time.time()
        })

    except Exception as e:
        logger.error("api.vram.history_failed", error=str(e))
        return JSONResponse({
            "success": False,
            "error": "Failed to get VRAM history"
        }, status_code=500)


@app.post("/api/system/vram/cleanup", tags=["System", "VRAM"])
@limiter.limit("5/minute")
async def trigger_vram_cleanup(
    request: Request,
    device_id: Optional[int] = Query(None, ge=0, description="GPU device ID (None for all)"),
    authenticated: bool = Depends(verify_api_token)
):
    """Force GPU memory cleanup"""
    logger.info("api.vram.cleanup_triggered", device_id=device_id)

    try:
        vram_monitor = get_vram_monitor()
        vram_monitor.force_cleanup(device_id=device_id)

        # Get status after cleanup
        cleanup_stats = vram_monitor.get_monitoring_stats()

        return JSONResponse({
            "success": True,
            "message": f"VRAM cleanup completed for GPU {device_id if device_id is not None else 'all'}",
            "vram_status_after": cleanup_stats
        })

    except Exception as e:
        logger.error("api.vram.cleanup_failed", error=str(e))
        return JSONResponse({
            "success": False,
            "error": "VRAM cleanup failed"
        }, status_code=500)


@app.get("/api/system/vram/predict", tags=["System", "VRAM"])
@limiter.limit("20/minute")
async def predict_service_capacity(
    request: Request,
    service_type: str = Query(..., regex="^(stt|llm|tts_fast|tts_hq)$", description="Service type")
):
    """Predict if system can handle a new service request"""
    logger.info("api.vram.predict_requested", service_type=service_type)

    try:
        vram_monitor = get_vram_monitor()
        prediction = vram_monitor.predict_can_handle_service(service_type)

        return JSONResponse({
            "success": True,
            "prediction": prediction,
            "timestamp": time.time()
        })

    except Exception as e:
        logger.error("api.vram.predict_failed", error=str(e))
        return JSONResponse({
            "success": False,
            "error": "Prediction failed"
        }, status_code=500)


# Include routers
app.include_router(voice_profiles_router)
app.include_router(conversations_router)
app.include_router(monitoring_router)


# Root endpoint
@app.get("/", tags=["System"])
@limiter.limit("30/minute")
async def root(request: Request):
    """Root endpoint"""
    return JSONResponse({
        "message": "AVATAR - AI Voice Assistant API",
        "version": app.version,
        "docs": "/docs",
        "health": "/health",
        "websocket": "/ws/chat",
        "endpoints": {
            "voice_profiles": "/api/voice-profiles",
            "conversations": "/api/conversations",
            "system": "/api/system"
        }
    })


# WebSocket endpoint for voice conversation
@app.websocket("/ws/chat")
async def chat_websocket(websocket):
    """WebSocket endpoint for real-time voice conversation"""
    await websocket_endpoint(websocket)


if __name__ == "__main__":
    import uvicorn

    logger.info("avatar.dev_server.starting",
                host=config.HOST,
                port=config.PORT)

    uvicorn.run(
        "avatar.main:app",
        host=config.HOST,
        port=config.PORT,
        reload=True,
        log_level=config.LOG_LEVEL.lower(),
    )
