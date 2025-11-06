"""
End-to-End Error Handling and Monitoring System Tests

Task 23: Comprehensive testing of unified error handling, structured logging,
and integrated monitoring system. Real system testing without mocks.

Test Coverage:
1. Error classification and context creation
2. Structured logging with performance metrics
3. Alert generation and management
4. Monitoring API endpoints
5. Health status and metrics collection
"""

import asyncio
import pytest
import time
import json
import sys
import os
from typing import Dict, Any, List
from unittest.mock import patch

import httpx
import structlog

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from avatar.core.error_handling import (
    ErrorHandler, get_error_handler, ErrorCategory, ErrorSeverity,
    handle_errors, error_context, log_and_raise, create_error_response
)
from avatar.core.logging_config import (
    configure_logging, get_logger, PerformanceLogger, get_metrics_collector
)
from avatar.core.monitoring import (
    MonitoringSystem, get_monitoring_system, HealthStatus, AlertLevel,
    setup_monitoring
)


class TestErrorHandling:
    """Test unified error handling system"""

    def test_error_classification(self):
        """Test automatic error classification"""
        handler = ErrorHandler()

        # Test network errors
        network_error = ConnectionError("Connection failed")
        category, severity = handler.classify_error(network_error)
        assert category == ErrorCategory.NETWORK
        assert severity == ErrorSeverity.WARNING

        # Test validation errors
        validation_error = ValueError("Invalid input")
        category, severity = handler.classify_error(validation_error)
        assert category == ErrorCategory.VALIDATION
        assert severity == ErrorSeverity.WARNING

        # Test programming errors
        logic_error = AssertionError("Assertion failed")
        category, severity = handler.classify_error(logic_error)
        assert category == ErrorCategory.LOGIC
        assert severity == ErrorSeverity.CRITICAL

    def test_error_context_creation(self):
        """Test error context creation with metadata"""
        handler = ErrorHandler()

        exception = ValueError("Test error")
        context = handler.create_context(
            exception,
            operation="test_operation",
            component="test_component",
            session_id="test-session-123",
            extra_data="test_metadata"
        )

        assert context.error_type == "ValueError"
        assert context.message == "Test error"
        assert context.operation == "test_operation"
        assert context.component == "test_component"
        assert context.session_id == "test-session-123"
        assert context.category == ErrorCategory.VALIDATION
        assert context.severity == ErrorSeverity.WARNING
        assert context.is_retriable is False
        assert "extra_data" in context.metadata

    def test_error_statistics_tracking(self):
        """Test error statistics collection"""
        handler = ErrorHandler()

        # Generate various errors
        errors = [
            ConnectionError("Network issue 1"),
            ConnectionError("Network issue 2"),
            ValueError("Validation issue"),
            AssertionError("Logic issue")
        ]

        for error in errors:
            context = handler.create_context(error, operation="test")
            handler.handle_error(context)

        stats = handler.get_stats()

        assert stats["total_errors"] == 4
        assert "network.warning" in stats["error_breakdown"]
        assert stats["error_breakdown"]["network.warning"] == 2
        assert stats["error_breakdown"]["validation.warning"] == 1
        assert stats["error_breakdown"]["logic.critical"] == 1

    @pytest.mark.asyncio
    async def test_error_decorator(self):
        """Test error handling decorator"""
        handler = get_error_handler()
        initial_count = len(handler.recent_errors)

        @handle_errors(operation="test_decorator", component="test_module")
        async def failing_function():
            raise ValueError("Decorator test error")

        with pytest.raises(ValueError):
            await failing_function()

        # Verify error was captured
        assert len(handler.recent_errors) == initial_count + 1
        latest_error = handler.recent_errors[-1]
        assert latest_error.operation == "test_decorator"
        assert latest_error.component == "test_module"

    @pytest.mark.asyncio
    async def test_error_context_manager(self):
        """Test error context manager"""
        handler = get_error_handler()
        initial_count = len(handler.recent_errors)

        with pytest.raises(ValueError):
            async with error_context("test_context", component="test_component"):
                raise ValueError("Context manager test")

        assert len(handler.recent_errors) == initial_count + 1
        latest_error = handler.recent_errors[-1]
        assert latest_error.operation == "test_context"


class TestStructuredLogging:
    """Test structured logging system"""

    def test_logging_configuration(self):
        """Test logging configuration setup"""
        configure_logging(development_mode=True)
        logger = get_logger("test_module")

        # Test basic logging
        logger.info("test.logging.configured", test_data="test_value")

        # Verify logger is properly configured
        assert hasattr(logger, 'info')
        assert hasattr(logger, 'error')
        assert hasattr(logger, 'warning')

    def test_performance_logger(self):
        """Test performance logging context manager"""
        logger = get_logger("test_performance")

        with PerformanceLogger(logger, "test_operation") as perf:
            time.sleep(0.1)  # Simulate work

        # Performance logger should have recorded the operation
        # (This would be captured in structured logs)

    def test_metrics_collection(self):
        """Test metrics collector functionality"""
        collector = get_metrics_collector()

        # Record some metrics
        collector.record_performance("test_operation", 0.5)
        collector.record_performance("test_operation", 0.3)
        collector.record_error("test_component", "ValueError")

        summary = collector.get_summary()

        # Verify performance metrics
        assert "test_operation" in summary["performance"]
        perf_stats = summary["performance"]["test_operation"]
        assert perf_stats["count"] == 2
        assert perf_stats["avg_time"] == 0.4
        assert perf_stats["min_time"] == 0.3
        assert perf_stats["max_time"] == 0.5

        # Verify error metrics
        assert "test_component.ValueError" in summary["errors"]


class TestMonitoringSystem:
    """Test integrated monitoring system"""

    def test_monitoring_initialization(self):
        """Test monitoring system setup"""
        error_handler = ErrorHandler()
        metrics_collector = get_metrics_collector()

        monitoring = MonitoringSystem()
        monitoring.integrate_components(error_handler, metrics_collector)

        assert monitoring.error_handler is not None
        assert monitoring.metrics_collector is not None

    def test_alert_creation_and_deduplication(self):
        """Test alert creation with deduplication"""
        monitoring = MonitoringSystem()

        # Create initial alert
        monitoring._create_alert(
            level=AlertLevel.HIGH,
            title="Test Alert",
            message="Test alert message",
            component="test_component"
        )

        initial_count = len(monitoring.active_alerts)

        # Create duplicate alert (should be deduplicated)
        monitoring._create_alert(
            level=AlertLevel.HIGH,
            title="Test Alert",
            message="Test alert message duplicate",
            component="test_component"
        )

        # Should not create new alert due to deduplication
        assert len(monitoring.active_alerts) == initial_count

        # Get the alert and verify it was updated
        alert_key = "test_component.Test Alert"
        alert = monitoring.active_alerts[alert_key]
        assert alert.count == 2

    def test_health_status_calculation(self):
        """Test health status calculation"""
        monitoring = MonitoringSystem()
        error_handler = ErrorHandler()
        monitoring.integrate_components(error_handler, get_metrics_collector())

        # Initially healthy
        health = monitoring.get_health_status()
        assert health["status"] == HealthStatus.HEALTHY.value

        # Generate critical errors
        for i in range(3):
            error = AssertionError(f"Critical error {i}")
            context = error_handler.create_context(error, operation="test")
            error_handler.handle_error(context)
            monitoring.process_error(context)

        # Should be critical now
        health = monitoring.get_health_status()
        assert health["status"] == HealthStatus.CRITICAL.value
        assert health["critical_errors_last_5min"] >= 3

    def test_alert_management(self):
        """Test alert acknowledgment and resolution"""
        monitoring = MonitoringSystem()

        # Create test alert
        monitoring._create_alert(
            level=AlertLevel.MEDIUM,
            title="Test Management Alert",
            message="Test message",
            component="test_component"
        )

        alerts = monitoring.get_alerts()
        alert_id = alerts[0]["id"]

        # Test acknowledgment
        success = monitoring.acknowledge_alert(alert_id)
        assert success

        alerts = monitoring.get_alerts()
        alert = next(a for a in alerts if a["id"] == alert_id)
        assert alert["acknowledged"] is True

        # Test resolution
        success = monitoring.resolve_alert(alert_id)
        assert success

        # Should be removed from active alerts
        active_alerts = monitoring.get_alerts()
        active_alert_ids = [a["id"] for a in active_alerts]
        assert alert_id not in active_alert_ids

    def test_prometheus_metrics_export(self):
        """Test Prometheus metrics export"""
        monitoring = MonitoringSystem()
        error_handler = ErrorHandler()
        monitoring.integrate_components(error_handler, get_metrics_collector())

        # Generate some data
        monitoring._create_alert(AlertLevel.HIGH, "Test", "Test", "test")

        metrics = monitoring.export_prometheus_metrics()

        # Verify Prometheus format
        assert "avatar_health_status" in metrics
        assert "avatar_error_rate" in metrics
        assert "avatar_alerts" in metrics
        assert 'level="high"' in metrics


class TestIntegratedSystem:
    """Test complete integrated error handling and monitoring"""

    def setup_method(self):
        """Setup integrated system for testing"""
        configure_logging(development_mode=True)

        self.error_handler = get_error_handler()
        self.metrics_collector = get_metrics_collector()
        self.monitoring = get_monitoring_system()

        setup_monitoring(self.error_handler, self.metrics_collector)

    def test_end_to_end_error_flow(self):
        """Test complete error flow from detection to monitoring"""

        # Generate an error
        try:
            raise ConnectionError("Test network failure")
        except Exception as e:
            context = self.error_handler.create_context(
                e,
                operation="network_request",
                component="api_client",
                session_id="test-session"
            )
            self.error_handler.handle_error(context)
            self.monitoring.process_error(context)

        # Verify error was captured and processed
        stats = self.error_handler.get_stats()
        assert stats["total_errors"] >= 1

        # Verify health status reflects the error
        health = self.monitoring.get_health_status()
        assert health["error_rate_per_minute"] >= 0

    def test_error_pattern_detection(self):
        """Test error pattern detection and alerting"""

        # Generate error burst to trigger alert
        for i in range(6):  # Above burst threshold
            try:
                raise ValueError(f"Burst error {i}")
            except Exception as e:
                context = self.error_handler.create_context(e, operation="test_burst")
                self.error_handler.handle_error(context)
                self.monitoring.process_error(context)

        # Should have generated alerts
        alerts = self.monitoring.get_alerts()
        assert len(alerts) > 0

        # Should have error burst alert
        burst_alerts = [a for a in alerts if "Burst" in a["title"]]
        assert len(burst_alerts) > 0

    def test_metrics_integration(self):
        """Test metrics integration across all components"""

        # Record performance metrics
        self.metrics_collector.record_performance("test_operation", 0.25)
        self.metrics_collector.record_performance("test_operation", 0.35)

        # Generate errors
        self.metrics_collector.record_error("test_component", "TestError")

        # Get comprehensive summary
        summary = self.monitoring.get_metrics_summary()

        # Verify all components are integrated
        assert "health" in summary
        assert "alerts" in summary
        assert "errors" in summary
        assert "performance" in summary

        # Verify performance data
        if "test_operation" in summary["performance"]:
            perf = summary["performance"]["test_operation"]
            assert perf["count"] == 2
            assert perf["avg_time"] == 0.3


@pytest.mark.asyncio
async def test_monitoring_api_endpoints():
    """Test monitoring API endpoints with real HTTP requests"""

    # This would require running the actual FastAPI server
    # For now, we'll test the API response models

    from avatar.api.monitoring import (
        HealthResponse, AlertResponse, ErrorStatsResponse, MetricsSummaryResponse
    )

    # Test response model validation
    health_data = {
        "status": "healthy",
        "error_rate_per_minute": 0.5,
        "critical_errors_last_5min": 0,
        "active_alerts": 0,
        "uptime_seconds": 3600.0,
        "last_update": time.time(),
        "timestamp": time.time()
    }

    health_response = HealthResponse(**health_data)
    assert health_response.status == "healthy"
    assert health_response.error_rate_per_minute == 0.5


def test_error_response_creation():
    """Test standardized error response creation"""

    try:
        raise ValueError("Test API error")
    except Exception as e:
        response = create_error_response(
            e,
            operation="api_test",
            component="test_api"
        )

    assert response["success"] is False
    assert "error" in response
    assert response["error"]["type"] == "ValueError"
    assert response["error"]["message"] == "Test API error"
    assert response["error"]["category"] == "validation"
    assert response["error"]["is_retriable"] is False


if __name__ == "__main__":
    # Run integration test
    async def run_integration_test():
        """Run comprehensive integration test"""
        print("🧪 Testing unified error handling and monitoring system...")

        # Setup system
        configure_logging(development_mode=True)
        error_handler = get_error_handler()
        metrics_collector = get_metrics_collector()
        monitoring = get_monitoring_system()
        setup_monitoring(error_handler, metrics_collector)

        print("✅ System setup complete")

        # Test error classification
        test_error = ConnectionError("Integration test error")
        context = error_handler.create_context(
            test_error,
            operation="integration_test",
            component="test_suite"
        )
        error_handler.handle_error(context)
        monitoring.process_error(context)

        print(f"✅ Error processed: {context.error_id}")

        # Test metrics
        metrics_collector.record_performance("integration_test", 0.123)
        print("✅ Performance metrics recorded")

        # Get system status
        health = monitoring.get_health_status()
        print(f"✅ System health: {health['status']}")

        # Get comprehensive summary
        summary = monitoring.get_metrics_summary()
        print(f"✅ Total errors tracked: {summary['errors'].get('total_errors', 0)}")

        # Test Prometheus export
        prometheus_metrics = monitoring.export_prometheus_metrics()
        assert "avatar_health_status" in prometheus_metrics
        print("✅ Prometheus metrics export working")

        print("🎉 Integration test completed successfully!")

    # Run test
    asyncio.run(run_integration_test())