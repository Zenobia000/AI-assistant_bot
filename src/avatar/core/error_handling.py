"""
Unified Error Handling and Recovery System

Task 23: Linus-style error handling - "Do one thing and do it well"
Principle: "Good error handling eliminates special cases, not creates them"

Design Philosophy:
1. Classify errors by recovery strategy, not by origin
2. Single unified interface for all error handling
3. Automatic context capture for debugging
4. Graceful degradation over complete failure
"""

import asyncio
import inspect
import time
import traceback
import uuid
from enum import Enum
from typing import Dict, Any, Optional, Callable, Type, Union
from dataclasses import dataclass, field
from contextlib import asynccontextmanager, contextmanager

import structlog

logger = structlog.get_logger()


class ErrorSeverity(Enum):
    """Error severity classification by impact"""
    DEBUG = "debug"           # Development info, no user impact
    INFO = "info"             # Normal operation info
    WARNING = "warning"       # Degraded performance, service continues
    ERROR = "error"          # Feature failure, other features work
    CRITICAL = "critical"     # Service failure, immediate action needed
    FATAL = "fatal"          # System failure, cannot continue


class ErrorCategory(Enum):
    """Error categories by recovery strategy (Linus principle)"""
    # Retriable errors - can be fixed by retrying
    NETWORK = "network"           # Network timeout, connection reset
    RESOURCE = "resource"         # Out of memory, disk full, VRAM limit
    RATE_LIMIT = "rate_limit"     # API rate limits, throttling

    # Non-retriable but recoverable - alternative approach needed
    VALIDATION = "validation"     # Input validation, schema errors
    AUTHENTICATION = "auth"       # Permission denied, invalid tokens
    NOT_FOUND = "not_found"      # Missing resources, 404 errors

    # System errors - require operator intervention
    CONFIGURATION = "config"      # Missing config, invalid settings
    DEPENDENCY = "dependency"     # External service unavailable
    HARDWARE = "hardware"        # GPU error, disk failure

    # Programming errors - require code fix
    LOGIC = "logic"              # Assertion failures, unexpected state
    TYPE = "type"                # Type errors, attribute errors
    UNKNOWN = "unknown"          # Unclassified errors


@dataclass
class ErrorContext:
    """Structured error context for debugging and recovery"""
    # Core identification
    error_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: float = field(default_factory=time.time)

    # Error classification
    severity: ErrorSeverity = ErrorSeverity.ERROR
    category: ErrorCategory = ErrorCategory.UNKNOWN

    # Error details
    message: str = ""
    original_exception: Optional[Exception] = None
    error_type: str = ""

    # Context information
    operation: str = ""           # What was being attempted
    component: str = ""           # Which component failed
    session_id: Optional[str] = None
    user_id: Optional[str] = None

    # Technical details
    stack_trace: str = ""
    function_name: str = ""
    file_name: str = ""
    line_number: int = 0

    # Recovery information
    is_retriable: bool = False
    retry_count: int = 0
    max_retries: int = 3
    backoff_seconds: float = 1.0

    # Additional context
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging"""
        return {
            "error_id": self.error_id,
            "timestamp": self.timestamp,
            "severity": self.severity.value,
            "category": self.category.value,
            "message": self.message,
            "error_type": self.error_type,
            "operation": self.operation,
            "component": self.component,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "function_name": self.function_name,
            "file_name": self.file_name,
            "line_number": self.line_number,
            "is_retriable": self.is_retriable,
            "retry_count": self.retry_count,
            "metadata": self.metadata
        }


class ErrorHandler:
    """
    Unified error handler - Linus principle: "Do one thing and do it well"

    Single responsibility: Convert any error into structured context
    and route to appropriate recovery strategy.
    """

    def __init__(self):
        self.error_stats: Dict[str, int] = {}
        self.recent_errors: list[ErrorContext] = []
        self.max_recent_errors = 100

        # Error classification rules
        self.classification_rules = self._build_classification_rules()

    def _build_classification_rules(self) -> Dict[Type[Exception], tuple[ErrorCategory, ErrorSeverity]]:
        """Build error classification rules - eliminates special cases"""
        return {
            # Network errors
            ConnectionError: (ErrorCategory.NETWORK, ErrorSeverity.WARNING),
            ConnectionResetError: (ErrorCategory.NETWORK, ErrorSeverity.WARNING),
            ConnectionRefusedError: (ErrorCategory.NETWORK, ErrorSeverity.ERROR),
            TimeoutError: (ErrorCategory.NETWORK, ErrorSeverity.WARNING),
            asyncio.TimeoutError: (ErrorCategory.NETWORK, ErrorSeverity.WARNING),

            # Resource errors
            MemoryError: (ErrorCategory.RESOURCE, ErrorSeverity.CRITICAL),
            OSError: (ErrorCategory.RESOURCE, ErrorSeverity.ERROR),

            # Validation errors
            ValueError: (ErrorCategory.VALIDATION, ErrorSeverity.WARNING),
            TypeError: (ErrorCategory.TYPE, ErrorSeverity.ERROR),
            AttributeError: (ErrorCategory.TYPE, ErrorSeverity.ERROR),
            KeyError: (ErrorCategory.VALIDATION, ErrorSeverity.WARNING),

            # File system errors
            FileNotFoundError: (ErrorCategory.NOT_FOUND, ErrorSeverity.WARNING),
            PermissionError: (ErrorCategory.AUTHENTICATION, ErrorSeverity.ERROR),

            # Programming errors
            AssertionError: (ErrorCategory.LOGIC, ErrorSeverity.CRITICAL),
            NotImplementedError: (ErrorCategory.LOGIC, ErrorSeverity.ERROR),

            # Generic fallback
            Exception: (ErrorCategory.UNKNOWN, ErrorSeverity.ERROR),
        }

    def classify_error(self, exception: Exception) -> tuple[ErrorCategory, ErrorSeverity]:
        """Classify error by type - eliminates if/else chains"""
        exception_type = type(exception)

        # Direct match
        if exception_type in self.classification_rules:
            return self.classification_rules[exception_type]

        # Check inheritance hierarchy
        for error_type, classification in self.classification_rules.items():
            if isinstance(exception, error_type):
                return classification

        # Ultimate fallback
        return ErrorCategory.UNKNOWN, ErrorSeverity.ERROR

    def create_context(self,
                      exception: Exception,
                      operation: str = "",
                      component: str = "",
                      session_id: Optional[str] = None,
                      **metadata) -> ErrorContext:
        """Create structured error context from exception"""

        # Get caller information
        frame = inspect.currentframe()
        caller_frame = frame.f_back if frame else None

        function_name = ""
        file_name = ""
        line_number = 0

        if caller_frame:
            function_name = caller_frame.f_code.co_name
            file_name = caller_frame.f_code.co_filename.split("/")[-1]
            line_number = caller_frame.f_lineno

        # Classify error
        category, severity = self.classify_error(exception)

        # Determine if retriable
        is_retriable = category in {
            ErrorCategory.NETWORK,
            ErrorCategory.RESOURCE,
            ErrorCategory.RATE_LIMIT
        }

        context = ErrorContext(
            severity=severity,
            category=category,
            message=str(exception),
            original_exception=exception,
            error_type=type(exception).__name__,
            operation=operation,
            component=component,
            session_id=session_id,
            stack_trace=traceback.format_exc(),
            function_name=function_name,
            file_name=file_name,
            line_number=line_number,
            is_retriable=is_retriable,
            metadata=metadata
        )

        return context

    def handle_error(self, context: ErrorContext) -> None:
        """Handle error with unified logging and stats"""

        # Update statistics
        key = f"{context.category.value}.{context.severity.value}"
        self.error_stats[key] = self.error_stats.get(key, 0) + 1

        # Store recent error
        self.recent_errors.append(context)
        if len(self.recent_errors) > self.max_recent_errors:
            self.recent_errors.pop(0)

        # Log with appropriate level
        log_data = context.to_dict()

        if context.severity == ErrorSeverity.DEBUG:
            logger.debug("error.handled", **log_data)
        elif context.severity == ErrorSeverity.INFO:
            logger.info("error.handled", **log_data)
        elif context.severity == ErrorSeverity.WARNING:
            logger.warning("error.handled", **log_data)
        elif context.severity == ErrorSeverity.ERROR:
            logger.error("error.handled", **log_data)
        elif context.severity == ErrorSeverity.CRITICAL:
            logger.critical("error.handled", **log_data)
        elif context.severity == ErrorSeverity.FATAL:
            logger.critical("error.fatal", **log_data)

    def get_stats(self) -> Dict[str, Any]:
        """Get error statistics for monitoring"""
        return {
            "total_errors": sum(self.error_stats.values()),
            "error_breakdown": self.error_stats.copy(),
            "recent_error_count": len(self.recent_errors),
            "error_rate_per_minute": self._calculate_error_rate()
        }

    def _calculate_error_rate(self) -> float:
        """Calculate recent error rate"""
        if not self.recent_errors:
            return 0.0

        current_time = time.time()
        recent_errors = [
            error for error in self.recent_errors
            if current_time - error.timestamp < 60  # Last minute
        ]

        return len(recent_errors)


# Global error handler instance
_error_handler = ErrorHandler()


def get_error_handler() -> ErrorHandler:
    """Get global error handler instance"""
    return _error_handler


# Convenient decorator and context managers

def handle_errors(operation: str = "", component: str = ""):
    """Decorator for automatic error handling"""
    def decorator(func):
        if asyncio.iscoroutinefunction(func):
            async def async_wrapper(*args, **kwargs):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    context = _error_handler.create_context(
                        e,
                        operation=operation or func.__name__,
                        component=component or func.__module__
                    )
                    _error_handler.handle_error(context)
                    raise
            return async_wrapper
        else:
            def sync_wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    context = _error_handler.create_context(
                        e,
                        operation=operation or func.__name__,
                        component=component or func.__module__
                    )
                    _error_handler.handle_error(context)
                    raise
            return sync_wrapper
    return decorator


@asynccontextmanager
async def error_context(operation: str, component: str = "", **metadata):
    """Async context manager for error handling"""
    try:
        yield
    except Exception as e:
        context = _error_handler.create_context(
            e,
            operation=operation,
            component=component,
            **metadata
        )
        _error_handler.handle_error(context)
        raise


@contextmanager
def sync_error_context(operation: str, component: str = "", **metadata):
    """Sync context manager for error handling"""
    try:
        yield
    except Exception as e:
        context = _error_handler.create_context(
            e,
            operation=operation,
            component=component,
            **metadata
        )
        _error_handler.handle_error(context)
        raise


# Utility functions for common patterns

def log_and_raise(exception: Exception,
                 operation: str = "",
                 component: str = "",
                 **metadata) -> None:
    """Log error and re-raise with context"""
    context = _error_handler.create_context(
        exception,
        operation=operation,
        component=component,
        **metadata
    )
    _error_handler.handle_error(context)
    raise exception


def log_and_return_error(exception: Exception,
                        default_return: Any = None,
                        operation: str = "",
                        component: str = "",
                        **metadata) -> Any:
    """Log error and return default value instead of raising"""
    context = _error_handler.create_context(
        exception,
        operation=operation,
        component=component,
        **metadata
    )
    _error_handler.handle_error(context)
    return default_return


def create_error_response(exception: Exception,
                         operation: str = "",
                         component: str = "",
                         **metadata) -> Dict[str, Any]:
    """Create standardized error response for APIs"""
    context = _error_handler.create_context(
        exception,
        operation=operation,
        component=component,
        **metadata
    )
    _error_handler.handle_error(context)

    return {
        "success": False,
        "error": {
            "id": context.error_id,
            "type": context.error_type,
            "message": context.message,
            "category": context.category.value,
            "is_retriable": context.is_retriable,
            "timestamp": context.timestamp
        }
    }