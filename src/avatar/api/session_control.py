"""
Session Control API - REST endpoints for concurrent session management

Task 21: API endpoints for queue status, session control, and system monitoring.
Provides real-time visibility into session queuing and VRAM status.

Endpoints:
- Queue status and statistics
- Individual session status
- System capacity and health
- Admin controls for session management
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, WebSocket
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
import time
import asyncio

import structlog

from avatar.core.session_controller import get_session_controller, SessionResult
from avatar.core.vram_monitor import get_vram_monitor
from avatar.api.auth import verify_api_key  # Assuming auth system exists

logger = structlog.get_logger()
router = APIRouter(prefix="/api/v1/sessions", tags=["Session Control"])


# Pydantic models for API
class SessionRequest(BaseModel):
    """Request model for creating a session"""
    session_id: str = Field(..., min_length=1, max_length=128, description="Unique session identifier")
    service_type: str = Field(..., pattern="^(stt|llm|tts_fast|tts_hq)$", description="Service type")
    timeout: Optional[float] = Field(None, ge=1.0, le=300.0, description="Timeout in seconds")
    priority_boost: bool = Field(False, description="Admin-only priority boost")


class SessionResponse(BaseModel):
    """Response model for session requests"""
    success: bool
    session_id: str
    message: str
    queue_position: Optional[int] = None
    estimated_wait_time: Optional[float] = None
    processing_started: bool = False
    error_code: Optional[str] = None
    timestamp: float = Field(default_factory=time.time)


class QueueStatus(BaseModel):
    """Queue status response model"""
    queue_size: int
    processing_count: int
    max_concurrent: int
    max_queue_size: int
    utilization_percent: float
    statistics: Dict
    queue_sessions: List[Dict]
    processing_sessions: List[Dict]
    timestamp: float = Field(default_factory=time.time)


class SystemHealth(BaseModel):
    """System health response model"""
    healthy: bool
    health_score: int
    issues: List[str]
    metrics: Dict
    timestamp: float = Field(default_factory=time.time)


class ServiceAvailability(BaseModel):
    """Service availability response model"""
    services: Dict[str, Dict]
    overall_capacity: Dict
    vram_summary: Dict
    timestamp: float = Field(default_factory=time.time)


# API endpoints
@router.post("/request", response_model=SessionResponse)
async def request_session(request: SessionRequest):
    """
    Request a new session with intelligent queuing

    Creates a new session request that will be processed based on:
    - Current system capacity
    - VRAM availability
    - Service type priority
    - Queue position

    Returns immediate processing if resources available, otherwise queues the request.
    """
    try:
        controller = get_session_controller()

        result = await controller.request_session(
            session_id=request.session_id,
            service_type=request.service_type,
            timeout=request.timeout,
            priority_boost=request.priority_boost
        )

        return SessionResponse(
            success=result.success,
            session_id=result.session_id,
            message=result.message,
            queue_position=result.queue_position,
            estimated_wait_time=result.estimated_wait_time,
            processing_started=result.processing_started,
            error_code=result.error_code
        )

    except Exception as e:
        logger.error("session_api.request_error",
                    session_id=request.session_id,
                    error=str(e))
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/status/{session_id}")
async def get_session_status(session_id: str):
    """
    Get status of a specific session

    Returns detailed information about session state:
    - Queue position and wait time (if queued)
    - Processing status and GPU allocation (if processing)
    - Completion status and timing (if completed)
    """
    controller = get_session_controller()
    status = controller.get_session_status(session_id)

    if status is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    return {
        "session_status": status,
        "timestamp": time.time()
    }


@router.delete("/cancel/{session_id}")
async def cancel_session(session_id: str):
    """
    Cancel a queued or processing session

    Removes session from queue if waiting, or attempts graceful termination if processing.
    Note: Sessions already processing may not be immediately cancelled.
    """
    controller = get_session_controller()
    cancelled = await controller.cancel_session(session_id)

    if not cancelled:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found or cannot be cancelled")

    return {
        "success": True,
        "message": f"Session {session_id} cancelled",
        "timestamp": time.time()
    }


@router.get("/queue", response_model=QueueStatus)
async def get_queue_status():
    """
    Get current queue status and statistics

    Returns comprehensive information about:
    - Current queue size and processing count
    - System capacity and utilization
    - Per-session queue details
    - Historical statistics
    """
    controller = get_session_controller()
    status = controller.session_queue.get_queue_status()

    return QueueStatus(
        queue_size=status["queue_size"],
        processing_count=status["processing_count"],
        max_concurrent=status["max_concurrent"],
        max_queue_size=status["max_queue_size"],
        utilization_percent=(
            status["processing_count"] / status["max_concurrent"] * 100
            if status["max_concurrent"] > 0 else 0
        ),
        statistics=status["statistics"],
        queue_sessions=status["queue_sessions"],
        processing_sessions=status["processing_sessions"]
    )


@router.get("/system", response_model=Dict)
async def get_system_status():
    """
    Get comprehensive system status

    Returns detailed system information including:
    - Queue status and statistics
    - VRAM usage and monitoring
    - Capacity and performance metrics
    - Service availability
    """
    controller = get_session_controller()
    return controller.get_system_status()


@router.get("/health", response_model=SystemHealth)
async def get_health_status():
    """
    Get system health check

    Returns health score and issues for monitoring systems.
    Used by load balancers and monitoring tools.
    """
    controller = get_session_controller()
    health = controller.get_health_check()

    return SystemHealth(
        healthy=health["healthy"],
        health_score=health["health_score"],
        issues=health["issues"],
        metrics=health["metrics"]
    )


@router.get("/availability", response_model=ServiceAvailability)
async def get_service_availability():
    """
    Get service availability for all service types

    Returns availability and resource requirements for:
    - STT (Speech-to-Text)
    - LLM (Language Model)
    - TTS Fast (Fast Text-to-Speech)
    - TTS HQ (High-Quality Text-to-Speech)
    """
    controller = get_session_controller()
    availability = controller.get_service_availability()
    system_status = controller.get_system_status()

    return ServiceAvailability(
        services=availability,
        overall_capacity=system_status["capacity"],
        vram_summary={
            "total_gpus": len(system_status["vram"]["gpus"]),
            "gpus_available": len([g for g in system_status["vram"]["gpus"] if g["can_accept_new"]]),
            "average_usage": sum(g["usage_percent"] for g in system_status["vram"]["gpus"]) / len(system_status["vram"]["gpus"]) if system_status["vram"]["gpus"] else 0
        }
    )


# Admin endpoints (require authentication)
@router.post("/admin/cleanup")
async def force_cleanup(api_key: str = Depends(verify_api_key)):
    """
    Force system cleanup (admin only)

    Emergency function to clean up:
    - GPU memory caches
    - Stale sessions
    - System resources

    Requires admin API key.
    """
    controller = get_session_controller()
    await controller.force_cleanup()

    return {
        "success": True,
        "message": "System cleanup completed",
        "timestamp": time.time()
    }


@router.get("/admin/detailed")
async def get_detailed_status(api_key: str = Depends(verify_api_key)):
    """
    Get detailed system status (admin only)

    Returns comprehensive debugging information including:
    - Internal queue states
    - VRAM historical data
    - Session processing details
    - Performance metrics

    Requires admin API key.
    """
    controller = get_session_controller()
    vram_monitor = get_vram_monitor()

    return {
        "system_status": controller.get_system_status(),
        "vram_monitoring": vram_monitor.get_monitoring_stats(),
        "vram_alerts": vram_monitor.get_alert_summary(),
        "queue_internals": {
            "queue_objects": len(controller.session_queue.queue),
            "processing_objects": len(controller.session_queue.processing),
            "completed_objects": len(controller.session_queue.completed),
            "background_tasks_active": (
                controller.session_queue._queue_processor_task is not None and
                not controller.session_queue._queue_processor_task.done()
            )
        },
        "performance_trends": {
            gpu.device_id: vram_monitor.get_usage_trend(gpu.device_id, 10)
            for gpu in vram_monitor.get_all_gpu_status()
        },
        "timestamp": time.time()
    }


@router.post("/admin/pause")
async def pause_system(api_key: str = Depends(verify_api_key)):
    """
    Pause session processing (admin only)

    Stops accepting new sessions and pauses queue processing.
    Existing sessions continue to completion.

    Requires admin API key.
    """
    # TODO: Implement pause functionality
    return {
        "success": False,
        "message": "Pause functionality not yet implemented",
        "timestamp": time.time()
    }


@router.post("/admin/resume")
async def resume_system(api_key: str = Depends(verify_api_key)):
    """
    Resume session processing (admin only)

    Resumes accepting new sessions and queue processing.

    Requires admin API key.
    """
    # TODO: Implement resume functionality
    return {
        "success": False,
        "message": "Resume functionality not yet implemented",
        "timestamp": time.time()
    }


# WebSocket status streaming (for real-time monitoring)
@router.websocket("/stream")
async def session_status_stream(websocket):
    """
    WebSocket endpoint for real-time session status updates

    Streams live updates about:
    - Queue changes
    - Session state transitions
    - System capacity changes
    - VRAM usage updates

    Format: JSON messages with type field
    """
    await websocket.accept()

    try:
        controller = get_session_controller()

        while True:
            # Send periodic status updates
            status = {
                "type": "status_update",
                "data": controller.get_system_status(),
                "timestamp": time.time()
            }

            await websocket.send_json(status)
            await asyncio.sleep(5)  # Update every 5 seconds

    except Exception as e:
        logger.error("session_api.websocket_error", error=str(e))
    finally:
        await websocket.close()


# Metrics endpoint for Prometheus/monitoring
@router.get("/metrics")
async def get_metrics():
    """
    Prometheus-compatible metrics endpoint

    Returns metrics in plain text format for monitoring systems:
    - Queue size and processing count
    - VRAM usage per GPU
    - Session processing times
    - Error rates and rejections
    """
    controller = get_session_controller()
    system_status = controller.get_system_status()

    metrics = []

    # Queue metrics
    metrics.append(f"avatar_queue_size {system_status['queue']['queue_size']}")
    metrics.append(f"avatar_processing_count {system_status['queue']['processing_count']}")
    metrics.append(f"avatar_max_concurrent {system_status['queue']['max_concurrent']}")
    metrics.append(f"avatar_utilization_percent {system_status['capacity']['utilization_percent']}")

    # Performance metrics
    metrics.append(f"avatar_avg_wait_time {system_status['performance']['average_wait_time']}")
    metrics.append(f"avatar_avg_processing_time {system_status['performance']['average_processing_time']}")
    metrics.append(f"avatar_total_processed {system_status['performance']['total_processed']}")
    metrics.append(f"avatar_total_rejected {system_status['performance']['total_rejected']}")

    # VRAM metrics
    for gpu in system_status['vram']['gpus']:
        metrics.append(f"avatar_vram_usage_percent{{gpu=\"{gpu['device_id']}\"}} {gpu['usage_percent']}")
        metrics.append(f"avatar_vram_free_gb{{gpu=\"{gpu['device_id']}\"}} {gpu['free_gb']}")

    return "\n".join(metrics), {"Content-Type": "text/plain"}