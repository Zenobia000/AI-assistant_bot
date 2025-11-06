"""
Integrated Error Monitoring and Alerting System

Task 23: Real-time monitoring with intelligent alerting
Linus principle: "Monitoring should be simple and actionable"

Design:
1. Real-time error tracking with pattern detection
2. Intelligent alert throttling to prevent spam
3. Health check integration
4. Prometheus metrics export
5. Dashboard-ready statistics
"""

import asyncio
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Any, Optional, Callable

import structlog

from avatar.core.error_handling import ErrorHandler, ErrorContext, ErrorSeverity, ErrorCategory
from avatar.core.logging_config import MetricsCollector

logger = structlog.get_logger()


class AlertLevel(Enum):
    """Alert priority levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Alert:
    """Alert definition"""
    id: str
    level: AlertLevel
    title: str
    message: str
    component: str
    timestamp: float = field(default_factory=time.time)
    count: int = 1
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    acknowledged: bool = False
    resolved: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class HealthStatus(Enum):
    """System health status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


@dataclass
class HealthMetrics:
    """System health metrics"""
    status: HealthStatus = HealthStatus.HEALTHY
    error_rate: float = 0.0
    critical_errors: int = 0
    response_time_p95: float = 0.0
    active_sessions: int = 0
    vram_usage_percent: float = 0.0
    uptime_seconds: float = 0.0
    last_update: float = field(default_factory=time.time)


class MonitoringSystem:
    """
    Integrated monitoring system for AVATAR

    Linus-style design:
    - Single source of truth for system health
    - Simple interfaces for different consumers
    - Automatic pattern detection without manual rules
    """

    def __init__(self):
        # Core components
        self.error_handler: ErrorHandler = None
        self.metrics_collector: MetricsCollector = None

        # Alert management
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: deque = deque(maxlen=1000)
        self.alert_callbacks: List[Callable[[Alert], None]] = []

        # Error pattern detection
        self.error_patterns: defaultdict = defaultdict(lambda: deque(maxlen=100))
        self.pattern_thresholds = {
            "error_rate": 10,  # errors per minute
            "error_burst": 5,  # errors in 30 seconds
            "critical_errors": 3,  # critical errors in 5 minutes
        }

        # Health tracking
        self.health_metrics = HealthMetrics()
        self.startup_time = time.time()

        # Configuration
        self.alert_cooldown = 300  # 5 minutes between duplicate alerts
        self.monitoring_enabled = True

        logger.info("monitoring.initialized")

    def integrate_components(self, error_handler: ErrorHandler, metrics_collector: MetricsCollector):
        """Integrate with error handler and metrics collector"""
        self.error_handler = error_handler
        self.metrics_collector = metrics_collector

        logger.info("monitoring.components_integrated")

    def process_error(self, error_context: ErrorContext):
        """Process error for monitoring and alerting"""
        if not self.monitoring_enabled:
            return

        # Record error pattern
        pattern_key = f"{error_context.category.value}.{error_context.severity.value}"
        self.error_patterns[pattern_key].append(error_context.timestamp)

        # Check for alert conditions
        self._check_alert_conditions(error_context)

        # Update health metrics
        self._update_health_metrics()

    def _check_alert_conditions(self, error_context: ErrorContext):
        """Check if error should trigger alerts"""

        # Critical error immediate alert
        if error_context.severity == ErrorSeverity.CRITICAL:
            self._create_alert(
                level=AlertLevel.CRITICAL,
                title=f"Critical Error in {error_context.component}",
                message=f"{error_context.operation}: {error_context.message}",
                component=error_context.component,
                metadata={"error_id": error_context.error_id}
            )

        # Error rate alert
        recent_errors = self._get_recent_error_count(60)  # Last minute
        if recent_errors >= self.pattern_thresholds["error_rate"]:
            self._create_alert(
                level=AlertLevel.HIGH,
                title="High Error Rate Detected",
                message=f"{recent_errors} errors in the last minute",
                component="system",
                metadata={"error_rate": recent_errors}
            )

        # Error burst alert
        burst_errors = self._get_recent_error_count(30)  # Last 30 seconds
        if burst_errors >= self.pattern_thresholds["error_burst"]:
            self._create_alert(
                level=AlertLevel.MEDIUM,
                title="Error Burst Detected",
                message=f"{burst_errors} errors in 30 seconds",
                component="system",
                metadata={"burst_count": burst_errors}
            )

    def _get_recent_error_count(self, seconds: int) -> int:
        """Count errors in recent time window"""
        if not self.error_handler:
            return 0

        current_time = time.time()
        recent_errors = [
            error for error in self.error_handler.recent_errors
            if current_time - error.timestamp < seconds
        ]
        return len(recent_errors)

    def _create_alert(self, level: AlertLevel, title: str, message: str,
                     component: str, metadata: Dict[str, Any] = None):
        """Create new alert with deduplication"""

        # Create alert key for deduplication
        alert_key = f"{component}.{title}"

        # Check if similar alert exists and is recent
        if alert_key in self.active_alerts:
            existing = self.active_alerts[alert_key]
            if (time.time() - existing.last_seen) < self.alert_cooldown:
                # Update existing alert
                existing.count += 1
                existing.last_seen = time.time()
                return

        # Create new alert
        alert = Alert(
            id=f"alert_{int(time.time())}_{hash(alert_key) % 10000}",
            level=level,
            title=title,
            message=message,
            component=component,
            metadata=metadata or {}
        )

        self.active_alerts[alert_key] = alert
        self.alert_history.append(alert)

        # Trigger callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error("monitoring.callback_failed", error=str(e))

        logger.warning("monitoring.alert_created",
                      alert_id=alert.id,
                      level=alert.level.value,
                      title=alert.title,
                      component=alert.component)

    def _update_health_metrics(self):
        """Update system health metrics"""
        current_time = time.time()

        # Calculate error rate
        recent_errors = self._get_recent_error_count(60)
        self.health_metrics.error_rate = recent_errors

        # Count critical errors
        critical_errors = len([
            error for error in self.error_handler.recent_errors
            if (current_time - error.timestamp < 300 and  # Last 5 minutes
                error.severity == ErrorSeverity.CRITICAL)
        ]) if self.error_handler else 0

        self.health_metrics.critical_errors = critical_errors

        # Calculate uptime
        self.health_metrics.uptime_seconds = current_time - self.startup_time

        # Determine overall health status
        if critical_errors > 0:
            self.health_metrics.status = HealthStatus.CRITICAL
        elif recent_errors >= 20:  # High error rate
            self.health_metrics.status = HealthStatus.UNHEALTHY
        elif recent_errors >= 5:   # Moderate error rate
            self.health_metrics.status = HealthStatus.DEGRADED
        else:
            self.health_metrics.status = HealthStatus.HEALTHY

        self.health_metrics.last_update = current_time

    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status"""
        self._update_health_metrics()

        return {
            "status": self.health_metrics.status.value,
            "error_rate_per_minute": self.health_metrics.error_rate,
            "critical_errors_last_5min": self.health_metrics.critical_errors,
            "active_alerts": len(self.active_alerts),
            "uptime_seconds": self.health_metrics.uptime_seconds,
            "last_update": self.health_metrics.last_update,
            "timestamp": time.time()
        }

    def get_alerts(self, include_resolved: bool = False) -> List[Dict[str, Any]]:
        """Get current alerts"""
        alerts = []

        for alert in self.active_alerts.values():
            if include_resolved or not alert.resolved:
                alerts.append({
                    "id": alert.id,
                    "level": alert.level.value,
                    "title": alert.title,
                    "message": alert.message,
                    "component": alert.component,
                    "count": alert.count,
                    "first_seen": alert.first_seen,
                    "last_seen": alert.last_seen,
                    "acknowledged": alert.acknowledged,
                    "resolved": alert.resolved,
                    "metadata": alert.metadata
                })

        # Sort by severity and recency
        level_priority = {
            AlertLevel.CRITICAL: 4,
            AlertLevel.HIGH: 3,
            AlertLevel.MEDIUM: 2,
            AlertLevel.LOW: 1
        }

        alerts.sort(key=lambda x: (
            level_priority.get(AlertLevel(x["level"]), 0),
            -x["last_seen"]
        ), reverse=True)

        return alerts

    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert"""
        for alert in self.active_alerts.values():
            if alert.id == alert_id:
                alert.acknowledged = True
                logger.info("monitoring.alert_acknowledged", alert_id=alert_id)
                return True
        return False

    def resolve_alert(self, alert_id: str) -> bool:
        """Mark alert as resolved"""
        for key, alert in list(self.active_alerts.items()):
            if alert.id == alert_id:
                alert.resolved = True
                del self.active_alerts[key]
                logger.info("monitoring.alert_resolved", alert_id=alert_id)
                return True
        return False

    def add_alert_callback(self, callback: Callable[[Alert], None]):
        """Add callback for new alerts"""
        self.alert_callbacks.append(callback)

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary"""
        health = self.get_health_status()
        alerts = self.get_alerts()

        error_stats = self.error_handler.get_stats() if self.error_handler else {}
        performance_stats = self.metrics_collector.get_summary() if self.metrics_collector else {}

        return {
            "health": health,
            "alerts": {
                "active_count": len(alerts),
                "by_level": self._group_alerts_by_level(alerts),
                "recent": alerts[:5]  # Last 5 alerts
            },
            "errors": error_stats,
            "performance": performance_stats.get("performance", {}),
            "timestamp": time.time()
        }

    def _group_alerts_by_level(self, alerts: List[Dict[str, Any]]) -> Dict[str, int]:
        """Group alerts by level"""
        grouped = defaultdict(int)
        for alert in alerts:
            if not alert.get("resolved", False):
                grouped[alert["level"]] += 1
        return dict(grouped)

    def export_prometheus_metrics(self) -> str:
        """Export metrics in Prometheus format"""
        metrics = []

        # Health metrics
        health_value = {
            "healthy": 1, "degraded": 2, "unhealthy": 3, "critical": 4
        }.get(self.health_metrics.status.value, 0)

        metrics.append(f"avatar_health_status {health_value}")
        metrics.append(f"avatar_error_rate {self.health_metrics.error_rate}")
        metrics.append(f"avatar_critical_errors {self.health_metrics.critical_errors}")
        metrics.append(f"avatar_uptime_seconds {self.health_metrics.uptime_seconds}")

        # Alert metrics
        for level in AlertLevel:
            count = len([a for a in self.active_alerts.values()
                        if a.level == level and not a.resolved])
            metrics.append(f'avatar_alerts{{level="{level.value}"}} {count}')

        # Error metrics
        if self.error_handler:
            error_stats = self.error_handler.get_stats()
            metrics.append(f"avatar_total_errors {error_stats.get('total_errors', 0)}")

            for error_type, count in error_stats.get('error_breakdown', {}).items():
                category, severity = error_type.split('.')
                metrics.append(f'avatar_errors{{category="{category}",severity="{severity}"}} {count}')

        return "\n".join(metrics)


# Global monitoring system
_monitoring_system = MonitoringSystem()


def get_monitoring_system() -> MonitoringSystem:
    """Get global monitoring system"""
    return _monitoring_system


def setup_monitoring(error_handler: ErrorHandler, metrics_collector: MetricsCollector):
    """Setup integrated monitoring"""
    _monitoring_system.integrate_components(error_handler, metrics_collector)

    # Setup alert callback for critical errors
    def critical_alert_handler(alert: Alert):
        if alert.level == AlertLevel.CRITICAL:
            logger.critical("monitoring.critical_alert",
                           alert_id=alert.id,
                           title=alert.title,
                           component=alert.component)

    _monitoring_system.add_alert_callback(critical_alert_handler)

    logger.info("monitoring.setup_complete")