"""
AVATAR - FastAPI Main Application

Entry point for the AI Voice Assistant API server.
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from avatar.core.config import config
from avatar.api.websocket import websocket_endpoint
from avatar.api.voice_profiles import router as voice_profiles_router

# Initialize structured logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.dev.ConsoleRenderer()
    ]
)
logger = structlog.get_logger()


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

    # Note: AI models will be lazy-loaded on first request
    # to avoid long startup times
    logger.info("avatar.startup.complete",
                message="AI models will be loaded on demand")

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

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Health check endpoint
@app.get("/health", tags=["System"])
async def health_check():
    """Health check endpoint"""
    return JSONResponse({
        "status": "healthy",
        "version": app.version,
        "database": str(config.DATABASE_PATH.exists()),
    })


# System info endpoint
@app.get("/api/system/info", tags=["System"])
async def system_info():
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


# Include routers
app.include_router(voice_profiles_router)


# Root endpoint
@app.get("/", tags=["System"])
async def root():
    """Root endpoint"""
    return JSONResponse({
        "message": "AVATAR - AI Voice Assistant API",
        "version": app.version,
        "docs": "/docs",
        "health": "/health",
        "websocket": "/ws/chat",
        "voice_profiles": "/api/voice-profiles",
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
