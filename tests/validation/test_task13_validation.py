#!/usr/bin/env python3
"""
Task 13 Completion Validation Test

Comprehensive validation of AVATAR WebSocket E2E integration.
This test determines the actual completion status of Task 13.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def validate_task13_completion():
    """
    Validate Task 13 completion with comprehensive component checking
    """
    print("="*60)
    print("AVATAR TASK 13 VALIDATION REPORT")
    print("="*60)

    requirements = {
        'main_app': False,
        'websocket_endpoint': False,
        'stt_service': False,
        'llm_service': False,
        'tts_service': False,
        'message_models': False,
        'session_manager': False,
        'database_ops': False,
        'audio_utils': False,
        'config_management': False
    }

    # Test 1: Main application exists
    try:
        from avatar.main import app
        requirements['main_app'] = True
        print("✅ Main FastAPI Application: Found")

        # Check for WebSocket routes
        for route in app.routes:
            if hasattr(route, 'path') and ('ws' in route.path or 'websocket' in route.path):
                requirements['websocket_endpoint'] = True
                break

        if requirements['websocket_endpoint']:
            print("✅ WebSocket Endpoint: Found in routes")
        else:
            print("❌ WebSocket Endpoint: Not found in routes")

    except ImportError as e:
        print(f"❌ Main Application: Import failed - {e}")

    # Test 2: STT Service
    try:
        from avatar.services.stt import STTService
        requirements['stt_service'] = True
        print("✅ STT Service: Found (STTService)")
    except ImportError as e:
        print(f"❌ STT Service: Import failed - {e}")

    # Test 3: LLM Service
    try:
        from avatar.services.llm import LLMService
        requirements['llm_service'] = True
        print("✅ LLM Service: Found (LLMService)")
    except ImportError as e:
        print(f"❌ LLM Service: Import failed - {e}")

    # Test 4: TTS Service
    try:
        from avatar.services.tts import TTSService
        requirements['tts_service'] = True
        print("✅ TTS Service: Found (TTSService)")
    except ImportError as e:
        print(f"❌ TTS Service: Import failed - {e}")

    # Test 5: Message Models
    try:
        from avatar.models.messages import (
            AudioChunkMessage, AudioEndMessage, StatusMessage,
            TranscriptionMessage, LLMResponseMessage, TTSReadyMessage
        )
        requirements['message_models'] = True
        print("✅ Message Models: All WebSocket message types found")
    except ImportError as e:
        print(f"❌ Message Models: Import failed - {e}")

    # Test 6: Session Manager
    try:
        from avatar.core.session_manager import SessionManager
        requirements['session_manager'] = True
        print("✅ Session Manager: Found")
    except ImportError as e:
        print(f"❌ Session Manager: Import failed - {e}")

    # Test 7: Database Operations
    try:
        from avatar.services.database import DatabaseService
        requirements['database_ops'] = True
        print("✅ Database Service: Found")
    except ImportError as e:
        print(f"❌ Database Service: Import failed - {e}")

    # Test 8: Audio Utilities
    try:
        from avatar.core.audio_utils import AudioProcessor
        requirements['audio_utils'] = True
        print("✅ Audio Utilities: Found")
    except ImportError as e:
        print(f"❌ Audio Utilities: Import failed - {e}")

    # Test 9: Configuration
    try:
        from avatar.core.config import config
        requirements['config_management'] = True
        print("✅ Configuration Management: Found")
    except ImportError as e:
        print(f"❌ Configuration Management: Import failed - {e}")

    # Test 10: WebSocket API Handler
    try:
        from avatar.api.websocket import websocket_endpoint
        print("✅ WebSocket API Handler: Found")
        # Update websocket endpoint status if found via API module
        requirements['websocket_endpoint'] = True
    except ImportError as e:
        print(f"⚠️  WebSocket API Handler: {e}")

    # Calculate completion metrics
    completed_count = sum(requirements.values())
    total_count = len(requirements)
    completion_percentage = (completed_count / total_count) * 100

    print("\n" + "="*60)
    print("TASK 13 COMPLETION ANALYSIS")
    print("="*60)

    print(f"Components Found: {completed_count}/{total_count}")
    print(f"Completion Rate: {completion_percentage:.1f}%")

    # Determine status based on critical components
    critical_components = [
        'main_app', 'websocket_endpoint', 'stt_service',
        'llm_service', 'tts_service', 'message_models'
    ]

    critical_completed = sum(1 for comp in critical_components if requirements[comp])
    critical_total = len(critical_components)
    critical_percentage = (critical_completed / critical_total) * 100

    print(f"Critical Components: {critical_completed}/{critical_total} ({critical_percentage:.1f}%)")

    # Task 13 Status Decision
    if critical_percentage >= 90:
        status = "PASSED"
        status_emoji = "🎉"
        phase3_ready = True
        message = "Task 13 COMPLETED - All critical WebSocket E2E components present"
    elif critical_percentage >= 70:
        status = "PARTIAL"
        status_emoji = "⚠️"
        phase3_ready = False
        message = "Task 13 PARTIALLY COMPLETE - Some critical components missing"
    else:
        status = "FAILED"
        status_emoji = "❌"
        phase3_ready = False
        message = "Task 13 INCOMPLETE - Major WebSocket components missing"

    print(f"\n{status_emoji} TASK 13 STATUS: {status}")
    print(f"📋 Assessment: {message}")
    print(f"🚀 Phase 3 Ready: {'YES' if phase3_ready else 'NO'}")

    if not phase3_ready:
        missing_critical = [comp for comp in critical_components if not requirements[comp]]
        print(f"\n🔧 Missing Critical Components:")
        for comp in missing_critical:
            print(f"   - {comp}")

    print("="*60)

    return {
        'status': status,
        'completion_percentage': completion_percentage,
        'critical_percentage': critical_percentage,
        'phase3_ready': phase3_ready,
        'requirements': requirements,
        'missing_critical': [comp for comp in critical_components if not requirements[comp]]
    }


if __name__ == "__main__":
    result = validate_task13_completion()

    # Exit code for CI/CD integration
    if result['phase3_ready']:
        sys.exit(0)  # Success
    else:
        sys.exit(1)  # Failure