"""
Monitoring and Error Handling API Endpoints

Task 23: RESTful API for monitoring and error management
Provides real-time system health, alerts, and error analytics
"""

from fastapi import APIRouter, HTTPException, Query, Depends
from typing import Dict, List, Optional, Any
from pydantic import BaseModel
import time

import structlog

from avatar.core.monitoring import get_monitoring_system, HealthStatus, AlertLevel
from avatar.core.error_handling import get_error_handler
from avatar.core.logging_config import get_metrics_collector
from avatar.api.auth import optional_api_key

logger = structlog.get_logger()
router = APIRouter(prefix="/api/v1/monitoring", tags=["Monitoring"])


# Response models
class HealthResponse(BaseModel):
    """System health response"""
    status: str
    error_rate_per_minute: float
    critical_errors_last_5min: int
    active_alerts: int
    uptime_seconds: float
    last_update: float
    timestamp: float


class AlertResponse(BaseModel):
    """Alert response model"""
    id: str
    level: str
    title: str
    message: str
    component: str
    count: int
    first_seen: float
    last_seen: float
    acknowledged: bool
    resolved: bool
    metadata: Dict[str, Any]


class ErrorStatsResponse(BaseModel):
    """Error statistics response"""
    total_errors: int
    error_breakdown: Dict[str, int]
    recent_error_count: int
    error_rate_per_minute: float


class MetricsSummaryResponse(BaseModel):
    """Comprehensive metrics summary"""
    health: HealthResponse
    alerts: Dict[str, Any]
    errors: ErrorStatsResponse
    performance: Dict[str, Any]
    timestamp: float


@router.get("/health", response_model=HealthResponse)
async def get_system_health():
    """
    Get current system health status

    Returns comprehensive health metrics including:
    - Overall system status (healthy/degraded/unhealthy/critical)
    - Error rates and critical error counts
    - Active alert count and system uptime
    """
    try:
        monitoring = get_monitoring_system()
        health_data = monitoring.get_health_status()

        return HealthResponse(**health_data)

    except Exception as e:
        logger.error("monitoring.api.health_failed", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get health status")


@router.get("/alerts", response_model=List[AlertResponse])
async def get_alerts(
    include_resolved: bool = Query(False, description="Include resolved alerts"),
    level: Optional[str] = Query(None, description="Filter by alert level")
):
    """
    Get current system alerts

    Returns list of active alerts with filtering options:
    - Filter by resolution status
    - Filter by alert level (low/medium/high/critical)
    """
    try:
        monitoring = get_monitoring_system()
        alerts = monitoring.get_alerts(include_resolved=include_resolved)

        # Filter by level if specified
        if level:
            level_filter = level.lower()
            alerts = [alert for alert in alerts if alert["level"] == level_filter]

        return [AlertResponse(**alert) for alert in alerts]

    except Exception as e:
        logger.error("monitoring.api.alerts_failed", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get alerts")


@router.post("/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(alert_id: str, api_key: str = Depends(optional_api_key)):
    """
    Acknowledge an alert (admin only)

    Marks alert as acknowledged to prevent further notifications
    """
    if not api_key:
        raise HTTPException(status_code=401, detail="API key required for alert management")

    try:
        monitoring = get_monitoring_system()
        success = monitoring.acknowledge_alert(alert_id)

        if not success:
            raise HTTPException(status_code=404, detail="Alert not found")

        return {"success": True, "message": "Alert acknowledged"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error("monitoring.api.acknowledge_failed", alert_id=alert_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to acknowledge alert")


@router.post("/alerts/{alert_id}/resolve")
async def resolve_alert(alert_id: str, api_key: str = Depends(optional_api_key)):
    """
    Mark alert as resolved (admin only)

    Removes alert from active list and marks as resolved
    """
    if not api_key:
        raise HTTPException(status_code=401, detail="API key required for alert management")

    try:
        monitoring = get_monitoring_system()
        success = monitoring.resolve_alert(alert_id)

        if not success:
            raise HTTPException(status_code=404, detail="Alert not found")

        return {"success": True, "message": "Alert resolved"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error("monitoring.api.resolve_failed", alert_id=alert_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to resolve alert")


@router.get("/errors", response_model=ErrorStatsResponse)
async def get_error_statistics():
    """
    Get error statistics and patterns

    Returns detailed error analytics including:
    - Total error counts and breakdown by category/severity
    - Recent error rates and trending information
    """
    try:
        error_handler = get_error_handler()
        stats = error_handler.get_stats()

        return ErrorStatsResponse(**stats)

    except Exception as e:
        logger.error("monitoring.api.error_stats_failed", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get error statistics")


@router.get("/metrics", response_model=MetricsSummaryResponse)
async def get_metrics_summary():
    """
    Get comprehensive metrics summary

    Returns unified view of system health, alerts, errors, and performance
    """
    try:
        monitoring = get_monitoring_system()
        summary = monitoring.get_metrics_summary()

        return MetricsSummaryResponse(**summary)

    except Exception as e:
        logger.error("monitoring.api.metrics_failed", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get metrics summary")


@router.get("/metrics/prometheus")
async def get_prometheus_metrics():
    """
    Get metrics in Prometheus format

    Returns metrics formatted for Prometheus scraping
    """
    try:
        monitoring = get_monitoring_system()
        metrics = monitoring.export_prometheus_metrics()

        return metrics, {"Content-Type": "text/plain"}

    except Exception as e:
        logger.error("monitoring.api.prometheus_failed", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to export Prometheus metrics")


@router.get("/performance")
async def get_performance_metrics():
    """
    Get detailed performance metrics

    Returns performance statistics for all monitored operations
    """
    try:
        metrics_collector = get_metrics_collector()
        summary = metrics_collector.get_summary()

        return {
            "performance": summary.get("performance", {}),
            "timestamp": summary.get("timestamp", time.time())
        }

    except Exception as e:
        logger.error("monitoring.api.performance_failed", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get performance metrics")


@router.get("/errors/recent")
async def get_recent_errors(
    limit: int = Query(50, ge=1, le=500, description="Number of recent errors to return"),
    severity: Optional[str] = Query(None, description="Filter by severity level")
):
    """
    Get recent error details

    Returns detailed information about recent errors for debugging
    """
    try:
        error_handler = get_error_handler()
        recent_errors = error_handler.recent_errors[-limit:]

        # Filter by severity if specified
        if severity:
            severity_filter = severity.upper()
            recent_errors = [
                error for error in recent_errors
                if error.severity.value.upper() == severity_filter
            ]

        # Convert to serializable format
        errors_data = []
        for error in recent_errors:
            errors_data.append({
                "error_id": error.error_id,
                "timestamp": error.timestamp,
                "severity": error.severity.value,
                "category": error.category.value,
                "message": error.message,
                "error_type": error.error_type,
                "operation": error.operation,
                "component": error.component,
                "session_id": error.session_id,
                "is_retriable": error.is_retriable,
                "retry_count": error.retry_count,
                "metadata": error.metadata
            })

        return {
            "errors": errors_data,
            "count": len(errors_data),
            "timestamp": time.time()
        }

    except Exception as e:
        logger.error("monitoring.api.recent_errors_failed", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get recent errors")


@router.post("/monitoring/reset")
async def reset_monitoring_stats(api_key: str = Depends(optional_api_key)):
    """
    Reset monitoring statistics (admin only)

    Clears error statistics and resolved alerts for a fresh start
    """
    if not api_key:
        raise HTTPException(status_code=401, detail="API key required for monitoring reset")

    try:
        # Reset error handler stats
        error_handler = get_error_handler()
        error_handler.error_stats.clear()
        error_handler.recent_errors.clear()

        # Reset monitoring system
        monitoring = get_monitoring_system()
        monitoring.active_alerts.clear()
        monitoring.alert_history.clear()

        logger.info("monitoring.api.stats_reset", admin_action=True)

        return {
            "success": True,
            "message": "Monitoring statistics reset successfully",
            "timestamp": time.time()
        }

    except Exception as e:
        logger.error("monitoring.api.reset_failed", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to reset monitoring statistics")


@router.get("/dashboard")
async def get_dashboard_data():
    """
    Get dashboard-ready data

    Returns formatted data for monitoring dashboards
    """
    try:
        monitoring = get_monitoring_system()
        health = monitoring.get_health_status()
        alerts = monitoring.get_alerts()

        # Count alerts by level
        alert_counts = {"low": 0, "medium": 0, "high": 0, "critical": 0}
        for alert in alerts:
            if not alert.get("resolved", False):
                alert_counts[alert["level"]] += 1

        # Get error handler stats
        error_handler = get_error_handler()
        error_stats = error_handler.get_stats()

        return {
            "status_overview": {
                "health": health["status"],
                "error_rate": health["error_rate_per_minute"],
                "uptime_hours": round(health["uptime_seconds"] / 3600, 1),
                "active_alerts": len(alerts)
            },
            "alert_breakdown": alert_counts,
            "error_summary": {
                "total": error_stats.get("total_errors", 0),
                "recent_rate": error_stats.get("error_rate_per_minute", 0)
            },
            "recent_alerts": alerts[:5],
            "timestamp": time.time()
        }

    except Exception as e:
        logger.error("monitoring.api.dashboard_failed", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get dashboard data")