"""
Enhanced Structured Logging Configuration

Task 23: Unified logging with error integration
Linus principle: "Be consistent in your interfaces"

Features:
1. Structured JSON logging with consistent format
2. Performance metrics integration
3. Error context enrichment
4. Development vs Production configuration
5. Log rotation and retention
"""

import logging
import logging.handlers
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

import structlog
from structlog.typing import FilteringBoundLogger

from avatar.core.config import config


def get_log_level() -> int:
    """Get log level from environment"""
    level_str = os.getenv("AVATAR_LOG_LEVEL", "INFO").upper()
    return getattr(logging, level_str, logging.INFO)


def get_log_dir() -> Path:
    """Get logs directory"""
    log_dir = Path(os.getenv("AVATAR_LOG_DIR", config.PROJECT_ROOT / "logs"))
    log_dir.mkdir(exist_ok=True)
    return log_dir


def add_performance_context(logger: FilteringBoundLogger,
                          wrapped_method,
                          event_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Add performance context to log events"""

    # Add timestamp in multiple formats for different use cases
    event_dict["timestamp"] = time.time()
    event_dict["iso_timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    # Add process/thread info for debugging
    event_dict["process_id"] = os.getpid()

    # Add component hierarchy for filtering
    if "event" in event_dict:
        event_parts = event_dict["event"].split(".")
        if len(event_parts) >= 2:
            event_dict["component"] = event_parts[0]
            event_dict["operation"] = event_parts[1]
            if len(event_parts) >= 3:
                event_dict["sub_operation"] = ".".join(event_parts[2:])

    return event_dict


def add_request_context(logger: FilteringBoundLogger,
                       wrapped_method,
                       event_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Add request context when available"""

    # Try to get FastAPI request context
    try:
        from contextvars import copy_context
        ctx = copy_context()

        # Add request ID if available
        for var, value in ctx.items():
            if hasattr(var, 'name') and 'request' in var.name.lower():
                if hasattr(value, 'headers'):
                    request_id = value.headers.get('X-Request-ID')
                    if request_id:
                        event_dict["request_id"] = request_id

                    # Add client IP
                    if hasattr(value, 'client'):
                        event_dict["client_ip"] = value.client.host

                break
    except (ImportError, AttributeError):
        pass

    return event_dict


def add_error_enrichment(logger: FilteringBoundLogger,
                        wrapped_method,
                        event_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Enrich error events with additional context"""

    # If this is an error event, add enrichment
    if event_dict.get("level", "").lower() in ["error", "critical"]:

        # Add error fingerprint for deduplication
        error_key = f"{event_dict.get('component', 'unknown')}.{event_dict.get('operation', 'unknown')}"
        event_dict["error_fingerprint"] = error_key

        # Add severity classification
        if "critical" in str(event_dict.get("event", "")).lower():
            event_dict["alert_priority"] = "high"
        elif "error" in str(event_dict.get("event", "")).lower():
            event_dict["alert_priority"] = "medium"
        else:
            event_dict["alert_priority"] = "low"

    return event_dict


def format_for_humans(logger: FilteringBoundLogger,
                     name: str,
                     event_dict: Dict[str, Any]) -> str:
    """Human-readable format for development"""

    timestamp = event_dict.pop("timestamp", time.time())
    level = event_dict.pop("level", "info").upper()
    event = event_dict.pop("event", "")

    # Format timestamp - handle both float and string timestamps
    if isinstance(timestamp, str):
        time_str = timestamp.split('T')[1][:8] if 'T' in timestamp else timestamp[:8]
    else:
        time_str = time.strftime("%H:%M:%S", time.localtime(timestamp))

    # Color coding for terminal output
    colors = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    reset = "\033[0m"

    color = colors.get(level, "")

    # Build message
    message = f"{color}[{time_str}] {level:8s} {event}{reset}"

    # Add key context
    important_keys = ["session_id", "error_id", "component", "operation"]
    context_parts = []

    for key in important_keys:
        if key in event_dict:
            context_parts.append(f"{key}={event_dict[key]}")

    if context_parts:
        message += f" ({', '.join(context_parts)})"

    # Add remaining fields if not too many
    remaining = {k: v for k, v in event_dict.items()
                if k not in important_keys and not k.startswith('_')}

    if remaining and len(remaining) <= 3:
        extra = ", ".join(f"{k}={v}" for k, v in remaining.items())
        message += f" | {extra}"

    return message


def configure_logging(development_mode: bool = None) -> None:
    """Configure structured logging for AVATAR"""

    if development_mode is None:
        development_mode = os.getenv("AVATAR_ENV", "development") == "development"

    log_level = get_log_level()

    # Configure processors
    processors = [
        structlog.stdlib.filter_by_level,
        add_performance_context,
        add_request_context,
        add_error_enrichment,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    if development_mode:
        # Human-readable format for development
        processors.append(format_for_humans)

        # Console output
        handler = logging.StreamHandler(sys.stdout)

    else:
        # JSON format for production
        processors.extend([
            structlog.processors.dict_tracebacks,
            structlog.processors.JSONRenderer()
        ])

        # File output with rotation
        log_dir = get_log_dir()
        log_file = log_dir / "avatar.log"

        handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=50 * 1024 * 1024,  # 50MB
            backupCount=5
        )

    # Configure standard logging
    logging.basicConfig(
        format="%(message)s",
        stream=handler.stream if hasattr(handler, 'stream') else None,
        level=log_level,
        handlers=[handler] if not hasattr(handler, 'stream') else None
    )

    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Log configuration
    logger = structlog.get_logger("avatar.logging")
    logger.info("logging.configured",
               development_mode=development_mode,
               log_level=logging.getLevelName(log_level),
               log_dir=str(get_log_dir()) if not development_mode else "console")


def get_logger(name: str = "") -> FilteringBoundLogger:
    """Get a configured logger instance"""
    return structlog.get_logger(name)


# Performance logging utilities

class PerformanceLogger:
    """Helper for logging performance metrics"""

    def __init__(self, logger: FilteringBoundLogger, operation: str):
        self.logger = logger
        self.operation = operation
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        self.logger.debug("performance.start", operation=self.operation)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.time() - self.start_time

            if exc_type:
                self.logger.warning("performance.failed",
                                  operation=self.operation,
                                  duration_seconds=round(duration, 3),
                                  error_type=exc_type.__name__)
            else:
                self.logger.info("performance.completed",
                               operation=self.operation,
                               duration_seconds=round(duration, 3))


def log_performance(operation: str):
    """Decorator for automatic performance logging"""
    def decorator(func):
        logger = structlog.get_logger(func.__module__)

        if asyncio.iscoroutinefunction(func):
            async def async_wrapper(*args, **kwargs):
                with PerformanceLogger(logger, operation):
                    return await func(*args, **kwargs)
            return async_wrapper
        else:
            def sync_wrapper(*args, **kwargs):
                with PerformanceLogger(logger, operation):
                    return func(*args, **kwargs)
            return sync_wrapper
    return decorator


# Metrics collection for monitoring

class MetricsCollector:
    """Collect metrics from logs for monitoring"""

    def __init__(self):
        self.metrics: Dict[str, Dict[str, Any]] = {
            "performance": {},
            "errors": {},
            "requests": {}
        }

    def record_performance(self, operation: str, duration: float):
        """Record performance metric"""
        if operation not in self.metrics["performance"]:
            self.metrics["performance"][operation] = {
                "count": 0,
                "total_time": 0.0,
                "min_time": float('inf'),
                "max_time": 0.0
            }

        stats = self.metrics["performance"][operation]
        stats["count"] += 1
        stats["total_time"] += duration
        stats["min_time"] = min(stats["min_time"], duration)
        stats["max_time"] = max(stats["max_time"], duration)

    def record_error(self, component: str, error_type: str):
        """Record error metric"""
        key = f"{component}.{error_type}"
        self.metrics["errors"][key] = self.metrics["errors"].get(key, 0) + 1

    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary"""
        summary = {"timestamp": time.time()}

        # Performance summary
        perf_summary = {}
        for operation, stats in self.metrics["performance"].items():
            if stats["count"] > 0:
                perf_summary[operation] = {
                    "count": stats["count"],
                    "avg_time": stats["total_time"] / stats["count"],
                    "min_time": stats["min_time"],
                    "max_time": stats["max_time"]
                }
        summary["performance"] = perf_summary

        # Error summary
        summary["errors"] = self.metrics["errors"].copy()

        return summary


# Global metrics collector
_metrics_collector = MetricsCollector()


def get_metrics_collector() -> MetricsCollector:
    """Get global metrics collector"""
    return _metrics_collector


# Import asyncio here to avoid circular imports
import asyncio