"""
Advanced VRAM Monitoring and Throttling System for AVATAR

Phase 4 Task 20: Enhanced GPU memory management with intelligent throttling.
Based on Linus principle: "Monitor what matters, throttle what kills the system."
"""

import asyncio
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

import structlog
import torch

from avatar.core.config import config

logger = structlog.get_logger()


class VRAMThreshold(Enum):
    """VRAM usage threshold levels for intelligent throttling"""
    SAFE = 70      # < 70% - Normal operation, accept all
    WARNING = 85   # 70-85% - Start selective throttling
    CRITICAL = 95  # 85-95% - Emergency mode, minimal acceptance
    DANGER = 98    # > 95% - Reject all new sessions


class ServicePriority(Enum):
    """Service priority levels for resource allocation"""
    CRITICAL = 1   # STT (always allow)
    HIGH = 2       # LLM (core conversation)
    MEDIUM = 3     # TTS Fast (user experience)
    LOW = 4        # TTS HQ (nice to have)


@dataclass
class VRAMStatus:
    """VRAM status for a specific GPU"""
    device_id: int
    device_name: str
    total_gb: float
    allocated_gb: float
    reserved_gb: float
    free_gb: float
    usage_percent: float
    threshold: VRAMThreshold
    can_accept_new: bool
    temperature_c: Optional[float] = None
    power_usage_w: Optional[float] = None


@dataclass
class VRAMHistory:
    """Historical VRAM usage data"""
    timestamp: float
    device_id: int
    usage_percent: float
    session_count: int


class VRAMMonitor:
    """
    Advanced VRAM monitoring and throttling system

    Features:
    - Real-time multi-GPU monitoring
    - Intelligent service prioritization
    - Historical usage tracking
    - Predictive throttling
    - Emergency protection
    """

    def __init__(self):
        """Initialize VRAM monitor"""
        self.gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        self.history: List[VRAMHistory] = []
        self.last_emergency_alert = 0
        self.monitoring_interval = 5.0  # seconds
        self._monitoring_task: Optional[asyncio.Task] = None
        self.emergency_mode = False
        self.rejection_count = 0
        self._cached_vram_status: List[VRAMStatus] = []
        self.last_vram_check = 0

        # Service type to GPU mapping
        self.service_gpu_preference = {
            "stt": None,        # CPU only
            "llm": 0,          # Primary GPU
            "tts_fast": 1,     # Secondary GPU if available
            "tts_hq": "auto"   # Auto-select best available
        }

        # Service priority mapping
        self.service_priority = {
            "stt": ServicePriority.CRITICAL,
            "llm": ServicePriority.HIGH,
            "tts_fast": ServicePriority.MEDIUM,
            "tts_hq": ServicePriority.LOW
        }

        logger.info("vram_monitor.init",
                   gpu_count=self.gpu_count,
                   monitoring_interval=self.monitoring_interval)

    def start_monitoring(self):
        """Start background VRAM monitoring"""
        if self._monitoring_task is None or self._monitoring_task.done():
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())
            logger.info("vram_monitor.started")

    def stop_monitoring(self):
        """Stop background monitoring"""
        if self._monitoring_task and not self._monitoring_task.done():
            self._monitoring_task.cancel()
            logger.info("vram_monitor.stopped")

    async def _monitoring_loop(self):
        """Background monitoring loop"""
        try:
            while True:
                await self._collect_vram_metrics()
                await asyncio.sleep(self.monitoring_interval)

        except asyncio.CancelledError:
            logger.info("vram_monitor.loop_cancelled")
        except Exception as e:
            logger.error("vram_monitor.loop_error", error=str(e))

    async def _collect_vram_metrics(self):
        """Collect VRAM metrics for all GPUs"""
        if not torch.cuda.is_available():
            return

        current_time = time.time()

        for device_id in range(self.gpu_count):
            try:
                # Get VRAM usage
                allocated = torch.cuda.memory_allocated(device_id) / 1024**3
                reserved = torch.cuda.memory_reserved(device_id) / 1024**3
                total = torch.cuda.get_device_properties(device_id).total_memory / 1024**3

                usage_percent = (allocated / total) * 100 if total > 0 else 0

                # Record history
                # Get session count from session controller if available
                session_count = 0
                try:
                    from avatar.core.session_controller import get_session_controller
                    controller = get_session_controller()
                    if hasattr(controller, 'session_queue'):
                        session_count = len([s for s in controller.session_queue.processing.values()
                                           if s.gpu_allocation == device_id])
                except (ImportError, AttributeError):
                    # Fallback: no session tracking
                    session_count = 0

                self.history.append(VRAMHistory(
                    timestamp=current_time,
                    device_id=device_id,
                    usage_percent=usage_percent,
                    session_count=session_count
                ))

                # Trim history (keep last 100 records per GPU)
                if len(self.history) > self.gpu_count * 100:
                    self.history = self.history[-self.gpu_count * 100:]

                # Check for emergency conditions
                if usage_percent > VRAMThreshold.DANGER.value:
                    await self._handle_emergency_condition(device_id, usage_percent)

            except Exception as e:
                logger.error("vram_monitor.collect_error", device_id=device_id, error=str(e))

    async def _handle_emergency_condition(self, device_id: int, usage_percent: float):
        """Handle emergency VRAM conditions"""
        current_time = time.time()

        # Rate limit emergency alerts (max 1 per minute)
        if current_time - self.last_emergency_alert < 60:
            return

        self.last_emergency_alert = current_time
        self.emergency_mode = True

        logger.critical("vram_monitor.emergency",
                       device_id=device_id,
                       usage_percent=usage_percent,
                       threshold=VRAMThreshold.DANGER.value,
                       emergency_mode=self.emergency_mode)

        # Trigger cleanup
        torch.cuda.empty_cache()
        logger.info("vram_monitor.emergency_cleanup", device_id=device_id)

        # Reset emergency mode after cleanup
        await asyncio.sleep(5)  # Give cleanup time to work
        self.emergency_mode = False

    def get_all_gpu_status(self) -> List[VRAMStatus]:
        """Get comprehensive VRAM status for all GPUs"""
        if not torch.cuda.is_available():
            return []

        status_list = []

        for device_id in range(self.gpu_count):
            try:
                props = torch.cuda.get_device_properties(device_id)
                allocated = torch.cuda.memory_allocated(device_id) / 1024**3
                reserved = torch.cuda.memory_reserved(device_id) / 1024**3
                total = props.total_memory / 1024**3
                free = total - allocated
                usage_percent = (allocated / total) * 100 if total > 0 else 0

                # Determine threshold level
                if usage_percent < VRAMThreshold.SAFE.value:
                    threshold = VRAMThreshold.SAFE
                elif usage_percent < VRAMThreshold.WARNING.value:
                    threshold = VRAMThreshold.WARNING
                elif usage_percent < VRAMThreshold.CRITICAL.value:
                    threshold = VRAMThreshold.CRITICAL
                else:
                    threshold = VRAMThreshold.DANGER

                # Determine if can accept new sessions
                can_accept = (
                    threshold in [VRAMThreshold.SAFE, VRAMThreshold.WARNING] and
                    not self.emergency_mode
                )

                status_list.append(VRAMStatus(
                    device_id=device_id,
                    device_name=props.name,
                    total_gb=round(total, 2),
                    allocated_gb=round(allocated, 2),
                    reserved_gb=round(reserved, 2),
                    free_gb=round(free, 2),
                    usage_percent=round(usage_percent, 1),
                    threshold=threshold,
                    can_accept_new=can_accept
                ))

            except Exception as e:
                logger.error("vram_monitor.status_error", device_id=device_id, error=str(e))

        return status_list

    def _get_multi_gpu_vram_status(self) -> List[VRAMStatus]:
        """Get VRAM status for all GPUs (cached for performance)"""
        current_time = time.time()

        # Cache for 1 second to avoid excessive GPU queries
        if current_time - self.last_vram_check < 1.0:
            return getattr(self, '_cached_vram_status', [])

        self.last_vram_check = current_time
        self._cached_vram_status = self.get_all_gpu_status()
        return self._cached_vram_status

    def _evaluate_session_acceptance(self, vram_status: List[VRAMStatus],
                                   service_type: str) -> bool:
        """
        Intelligent session acceptance based on service priority and VRAM status

        Args:
            vram_status: Current VRAM status for all GPUs
            service_type: Type of service requesting resources

        Returns:
            True if session should be accepted
        """
        if not vram_status or self.emergency_mode:
            return False

        # Get service priority
        priority = self.service_priority.get(service_type, ServicePriority.LOW)

        # STT always allowed (CPU only)
        if service_type == "stt":
            return True

        # Check if any GPU can accept the service
        for status in vram_status:
            # Critical services (LLM) allowed even in WARNING state
            if priority == ServicePriority.CRITICAL:
                return status.threshold != VRAMThreshold.DANGER

            # High priority services allowed in SAFE and WARNING
            elif priority == ServicePriority.HIGH:
                return status.threshold in [VRAMThreshold.SAFE, VRAMThreshold.WARNING]

            # Medium/Low priority only in SAFE state
            else:
                if status.threshold == VRAMThreshold.SAFE and status.can_accept_new:
                    return True

        return False

    def _suggest_gpu_allocation(self, service_type: str) -> Optional[int]:
        """
        Suggest optimal GPU allocation for a service

        Args:
            service_type: Type of service

        Returns:
            Recommended GPU device ID, or None for CPU
        """
        if service_type == "stt":
            return None  # CPU only

        # Check preference
        preferred_gpu = self.service_gpu_preference.get(service_type)

        if preferred_gpu == "auto":
            # Auto-select GPU with most free VRAM
            vram_status = self._get_multi_gpu_vram_status()
            if vram_status:
                best_gpu = max(vram_status, key=lambda s: s.free_gb)
                if best_gpu.can_accept_new:
                    return best_gpu.device_id

        elif isinstance(preferred_gpu, int):
            # Check if preferred GPU is available
            vram_status = self._get_multi_gpu_vram_status()
            if preferred_gpu < len(vram_status):
                if vram_status[preferred_gpu].can_accept_new:
                    return preferred_gpu

        # Fallback: auto-select best available
        vram_status = self._get_multi_gpu_vram_status()
        available_gpus = [s for s in vram_status if s.can_accept_new]
        if available_gpus:
            return max(available_gpus, key=lambda s: s.free_gb).device_id

        return 0  # Default to GPU 0

    def get_monitoring_stats(self) -> dict:
        """Get comprehensive monitoring statistics"""
        vram_status = self.get_all_gpu_status()
        recent_history = [h for h in self.history if time.time() - h.timestamp < 300]  # Last 5 minutes

        avg_usage = sum(h.usage_percent for h in recent_history) / len(recent_history) if recent_history else 0

        return {
            "gpus": [
                {
                    "device_id": status.device_id,
                    "device_name": status.device_name,
                    "total_gb": status.total_gb,
                    "allocated_gb": status.allocated_gb,
                    "free_gb": status.free_gb,
                    "usage_percent": status.usage_percent,
                    "threshold": status.threshold.name,
                    "can_accept_new": status.can_accept_new
                }
                for status in vram_status
            ],
            "monitoring": {
                "history_points": len(self.history),
                "average_usage_5min": round(avg_usage, 1),
                "emergency_mode": self.emergency_mode,
                "rejection_count": self.rejection_count,
                "monitoring_active": self._monitoring_task is not None and not self._monitoring_task.done()
            },
            "thresholds": {
                "safe": VRAMThreshold.SAFE.value,
                "warning": VRAMThreshold.WARNING.value,
                "critical": VRAMThreshold.CRITICAL.value,
                "danger": VRAMThreshold.DANGER.value
            }
        }

    def get_usage_trend(self, device_id: int = 0, minutes: int = 10) -> dict:
        """
        Get VRAM usage trend for specified time period

        Args:
            device_id: GPU device ID
            minutes: Time period in minutes

        Returns:
            Usage trend information
        """
        cutoff_time = time.time() - (minutes * 60)
        relevant_history = [
            h for h in self.history
            if h.device_id == device_id and h.timestamp >= cutoff_time
        ]

        if not relevant_history:
            return {"error": "No data available for specified period"}

        usage_values = [h.usage_percent for h in relevant_history]

        return {
            "device_id": device_id,
            "period_minutes": minutes,
            "data_points": len(relevant_history),
            "min_usage": round(min(usage_values), 1),
            "max_usage": round(max(usage_values), 1),
            "avg_usage": round(sum(usage_values) / len(usage_values), 1),
            "current_usage": round(usage_values[-1], 1),
            "trend": "increasing" if usage_values[-1] > usage_values[0] else "stable" if abs(usage_values[-1] - usage_values[0]) < 5 else "decreasing"
        }

    def predict_can_handle_service(self, service_type: str) -> dict:
        """
        Predict if system can handle a new service request

        Args:
            service_type: Type of service (stt, llm, tts_fast, tts_hq)

        Returns:
            Prediction results with reasoning
        """
        vram_status = self.get_all_gpu_status()
        priority = self.service_priority.get(service_type, ServicePriority.LOW)

        # Estimate VRAM requirements
        vram_requirements = {
            "stt": 0,        # CPU only
            "llm": 5.2,      # Based on actual measurements
            "tts_fast": 1.5, # Based on actual measurements
            "tts_hq": 3.0    # Based on actual measurements
        }

        required_gb = vram_requirements.get(service_type, 1.0)

        prediction = {
            "service_type": service_type,
            "priority": priority.name,
            "vram_required_gb": required_gb,
            "can_handle": False,
            "recommended_gpu": None,
            "reasoning": []
        }

        # If emergency mode, only allow critical services
        if self.emergency_mode and priority != ServicePriority.CRITICAL:
            prediction["reasoning"].append("Emergency mode active - only critical services allowed")
            return prediction

        # Find suitable GPU
        for status in vram_status:
            if status.free_gb >= required_gb:
                # Check threshold compatibility
                if priority == ServicePriority.CRITICAL:
                    # Critical services allowed unless DANGER
                    if status.threshold != VRAMThreshold.DANGER:
                        prediction["can_handle"] = True
                        prediction["recommended_gpu"] = status.device_id
                        prediction["reasoning"].append(f"Critical service, GPU {status.device_id} has {status.free_gb}GB free")
                        break

                elif priority == ServicePriority.HIGH:
                    # High priority allowed in SAFE and WARNING
                    if status.threshold in [VRAMThreshold.SAFE, VRAMThreshold.WARNING]:
                        prediction["can_handle"] = True
                        prediction["recommended_gpu"] = status.device_id
                        prediction["reasoning"].append(f"High priority service, GPU {status.device_id} in {status.threshold.name} state")
                        break

                else:
                    # Medium/Low priority only in SAFE state
                    if status.threshold == VRAMThreshold.SAFE:
                        prediction["can_handle"] = True
                        prediction["recommended_gpu"] = status.device_id
                        prediction["reasoning"].append(f"Normal service, GPU {status.device_id} in SAFE state")
                        break

        if not prediction["can_handle"]:
            prediction["reasoning"].append("No suitable GPU found with sufficient free VRAM")

        return prediction

    def force_cleanup(self, device_id: Optional[int] = None):
        """
        Force GPU memory cleanup

        Args:
            device_id: Specific GPU to clean, or None for all
        """
        if not torch.cuda.is_available():
            return

        if device_id is not None:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            logger.info("vram_monitor.forced_cleanup", device_id=device_id)
        else:
            for i in range(self.gpu_count):
                torch.cuda.set_device(i)
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            logger.info("vram_monitor.forced_cleanup_all", gpu_count=self.gpu_count)

    def get_alert_summary(self) -> dict:
        """Get summary of recent alerts and rejections"""
        recent_history = [h for h in self.history if time.time() - h.timestamp < 3600]  # Last hour

        alert_counts = {
            "safe": 0,
            "warning": 0,
            "critical": 0,
            "danger": 0
        }

        for history_point in recent_history:
            if history_point.usage_percent < VRAMThreshold.SAFE.value:
                alert_counts["safe"] += 1
            elif history_point.usage_percent < VRAMThreshold.WARNING.value:
                alert_counts["warning"] += 1
            elif history_point.usage_percent < VRAMThreshold.CRITICAL.value:
                alert_counts["critical"] += 1
            else:
                alert_counts["danger"] += 1

        return {
            "period_hours": 1,
            "total_measurements": len(recent_history),
            "alert_distribution": alert_counts,
            "rejection_count_hour": self.rejection_count,  # TODO: Make this time-bound
            "emergency_activations": 1 if self.emergency_mode else 0
        }


# Global VRAM monitor instance
_vram_monitor: Optional[VRAMMonitor] = None


def get_vram_monitor() -> VRAMMonitor:
    """Get singleton VRAM monitor instance"""
    global _vram_monitor

    if _vram_monitor is None:
        _vram_monitor = VRAMMonitor()
        _vram_monitor.start_monitoring()  # Auto-start monitoring

    return _vram_monitor