"""
Enhanced WebSocket endpoint with reconnection and recovery

Task 22: Upgraded WebSocket handler with automatic reconnection, session recovery,
and intelligent error handling. Maintains backward compatibility with existing clients.

Key Enhancements:
1. Automatic reconnection with exponential backoff
2. Session state preservation and recovery
3. Heartbeat monitoring and connection health
4. Graceful error handling with retry classification
"""

import asyncio
import base64
import json
import time
import uuid
from pathlib import Path
from typing import Optional, Dict, Any

import structlog
from fastapi import WebSocket, WebSocketDisconnect, WebSocketException
from pydantic import ValidationError

from avatar.core.config import config
from avatar.core.websocket_reconnect import (
    get_reconnect_manager, SessionSnapshot, DisconnectReason, ConnectionState
)
from avatar.models.messages import (
    AudioChunkMessage,
    AudioEndMessage,
    ErrorMessage,
    StatusMessage,
)
from avatar.services.database import db

logger = structlog.get_logger()

# Buffer limits (same as original)
MAX_BUFFER_SIZE_BYTES = 10 * 1024 * 1024  # 10MB
MAX_CHUNK_COUNT = 1000
BUFFER_TIMEOUT_SECONDS = 60


class EnhancedConversationSession:
    """
    Enhanced conversation session with reconnection support

    Extends the original ConversationSession with:
    - State snapshotting for recovery
    - Reconnection-aware error handling
    - Progress tracking for resume capability
    """

    def __init__(self, session_id: str, websocket: WebSocket, recovered_state: Optional[SessionSnapshot] = None):
        self.session_id = session_id
        self.websocket = websocket
        self.audio_buffer: list[bytes] = []
        self.is_processing = False

        # Initialize from recovered state or start fresh
        if recovered_state:
            self.turn_number = recovered_state.turn_number
            self.voice_profile_id = recovered_state.voice_profile_id
            self.last_user_text = recovered_state.last_user_text
            self.last_ai_text = recovered_state.last_ai_text
            self.processing_stage = recovered_state.processing_stage
            logger.info("session.recovered",
                       session_id=session_id,
                       turn_number=self.turn_number,
                       stage=self.processing_stage)
        else:
            self.turn_number = 0
            self.voice_profile_id = None
            self.last_user_text = None
            self.last_ai_text = None
            self.processing_stage = "ready"
            logger.info("session.created", session_id=session_id)

        # Buffer tracking
        self.buffer_size_bytes = 0
        self.buffer_first_chunk_time: Optional[float] = None
        self.last_activity = time.time()

        # Reconnection manager
        self.reconnect_manager = get_reconnect_manager()

    def create_snapshot(self) -> SessionSnapshot:
        """Create a snapshot of current session state"""
        return SessionSnapshot(
            session_id=self.session_id,
            turn_number=self.turn_number,
            voice_profile_id=self.voice_profile_id,
            last_user_text=self.last_user_text,
            last_ai_text=self.last_ai_text,
            processing_stage=self.processing_stage,
            created_at=time.time() - (self.turn_number * 60),  # Estimate creation time
            last_activity=self.last_activity,
            metadata={
                "buffer_size": self.buffer_size_bytes,
                "is_processing": self.is_processing
            }
        )

    async def send_status(self, message: str, stage: str):
        """Send status update to client with reconnection tracking"""
        self.processing_stage = stage
        self.last_activity = time.time()

        status = StatusMessage(
            message=message,
            stage=stage,  # type: ignore
            session_id=self.session_id,
        )

        try:
            await self.websocket.send_text(status.model_dump_json())
        except Exception as e:
            logger.warning("session.send_status_failed",
                          session_id=self.session_id,
                          stage=stage,
                          error=str(e))
            # Don't raise - status updates are not critical

    async def send_error(self, error: str, code: str, is_recoverable: bool = True):
        """Send error message with recovery information"""
        error_msg = ErrorMessage(
            error=error,
            code=code,
            session_id=self.session_id,
        )

        # Add recovery information to error message
        error_data = error_msg.model_dump()
        error_data["recoverable"] = is_recoverable
        error_data["retry_after_seconds"] = 5 if is_recoverable else None

        try:
            await self.websocket.send_text(json.dumps(error_data))
        except Exception as e:
            logger.warning("session.send_error_failed",
                          session_id=self.session_id,
                          error=error,
                          send_error=str(e))

        logger.error("session.error",
                    session_id=self.session_id,
                    error=error,
                    code=code,
                    recoverable=is_recoverable)

    def add_audio_chunk(self, data_b64: str):
        """Add audio chunk with enhanced buffer management"""
        self.last_activity = time.time()

        try:
            audio_bytes = base64.b64decode(data_b64)
            chunk_size = len(audio_bytes)

            if self.buffer_first_chunk_time is None:
                self.buffer_first_chunk_time = time.time()

            # Enhanced buffer limit checks
            if len(self.audio_buffer) >= MAX_CHUNK_COUNT:
                raise RuntimeError(f"Too many audio chunks (max: {MAX_CHUNK_COUNT})")

            if self.buffer_size_bytes + chunk_size > MAX_BUFFER_SIZE_BYTES:
                raise RuntimeError(f"Audio buffer too large (max: {MAX_BUFFER_SIZE_BYTES / 1024 / 1024:.1f}MB)")

            if (time.time() - self.buffer_first_chunk_time) > BUFFER_TIMEOUT_SECONDS:
                raise RuntimeError(f"Audio buffering timeout (max: {BUFFER_TIMEOUT_SECONDS}s)")

            self.audio_buffer.append(audio_bytes)
            self.buffer_size_bytes += chunk_size

        except Exception as e:
            logger.error("session.add_chunk_failed",
                        session_id=self.session_id,
                        error=str(e))
            raise

    async def process_audio(self):
        """Process audio with enhanced error handling and recovery"""
        if self.is_processing:
            await self.send_error("Already processing audio", "PROCESSING_IN_PROGRESS", is_recoverable=True)
            return

        if not self.audio_buffer:
            await self.send_error("No audio data to process", "NO_AUDIO_DATA", is_recoverable=True)
            return

        self.is_processing = True
        self.turn_number += 1

        try:
            # Enhanced processing with recovery points
            await self.send_status("Processing audio...", "processing")

            # Save audio with recovery checkpoint
            audio_path = await self._save_audio()
            await self.send_status("Transcribing speech...", "stt")

            # STT processing
            transcription = await self._run_stt(audio_path)
            self.last_user_text = transcription
            await self.send_status("Understanding request...", "llm")

            # LLM processing
            llm_response = await self._run_llm(transcription)
            self.last_ai_text = llm_response
            await self.send_status("Synthesizing speech...", "tts")

            # TTS processing
            tts_url = await self._run_tts(
                text=llm_response,
                user_audio_path=audio_path,
                user_text=transcription
            )

            # Send completion notification
            from avatar.models.messages import TTSReadyMessage
            tts_msg = TTSReadyMessage(
                audio_url=tts_url,
                audio_format="wav",
                mode="fast",
                session_id=self.session_id,
            )
            await self.websocket.send_text(tts_msg.model_dump_json())

            await self.send_status("Ready", "ready")

            # Save to database
            await self._save_conversation(
                user_audio_path=str(audio_path),
                user_text=transcription,
                ai_text=llm_response,
                ai_audio_fast_path=tts_url,
            )

        except Exception as e:
            logger.exception("session.processing_failed", session_id=self.session_id)

            # Classify error for recovery guidance
            is_recoverable = not isinstance(e, (ValidationError, ValueError))
            await self.send_error(f"Processing failed: {str(e)}", "PROCESSING_ERROR", is_recoverable)

        finally:
            self.is_processing = False
            self.audio_buffer.clear()
            self.buffer_size_bytes = 0
            self.buffer_first_chunk_time = None

    async def _save_audio(self) -> Path:
        """Save audio with error recovery"""
        from avatar.core.audio_utils import convert_audio_to_wav

        raw_audio = b"".join(self.audio_buffer)
        audio_filename = f"user_audio_{self.session_id}_{self.turn_number}_{int(time.time())}.webm"
        audio_path = Path(config.AUDIO_RAW_DIR) / audio_filename

        try:
            audio_path.write_bytes(raw_audio)
            converted_path = await convert_audio_to_wav(audio_path)
            return converted_path

        except Exception as e:
            logger.error("session.save_audio_failed", session_id=self.session_id, error=str(e))
            raise RuntimeError(f"Failed to save audio: {str(e)}")

    async def _run_stt(self, audio_path: Path) -> str:
        """Run STT with recovery"""
        from avatar.services.stt import get_stt_service

        try:
            stt_service = get_stt_service()
            result = await stt_service.transcribe_audio(str(audio_path))
            return result.text

        except Exception as e:
            logger.error("session.stt_failed", session_id=self.session_id, error=str(e))
            raise RuntimeError(f"Speech recognition failed: {str(e)}")

    async def _run_llm(self, text: str) -> str:
        """Run LLM with recovery"""
        from avatar.services.llm import get_llm_service

        try:
            llm_service = get_llm_service()
            result = await llm_service.generate_response(text)
            return result.text

        except Exception as e:
            logger.error("session.llm_failed", session_id=self.session_id, error=str(e))
            raise RuntimeError(f"Language model failed: {str(e)}")

    async def _run_tts(self, text: str, user_audio_path: Path, user_text: str) -> str:
        """Run TTS with recovery"""
        from avatar.services.tts import get_tts_service

        try:
            tts_service = get_tts_service()
            result = await tts_service.synthesize_speech(
                text=text,
                voice_profile_id=self.voice_profile_id,
                reference_audio_path=str(user_audio_path)
            )
            return result.audio_url

        except Exception as e:
            logger.error("session.tts_failed", session_id=self.session_id, error=str(e))
            raise RuntimeError(f"Speech synthesis failed: {str(e)}")

    async def _save_conversation(self, user_audio_path: str, user_text: str,
                               ai_text: str, ai_audio_fast_path: str):
        """Save conversation with recovery"""
        try:
            conversation_id = await db.create_conversation(
                user_audio_path=user_audio_path,
                user_text=user_text,
                ai_text=ai_text,
                ai_audio_fast_path=ai_audio_fast_path,
                session_id=self.session_id,
                turn_number=self.turn_number,
                voice_profile_id=self.voice_profile_id
            )

            logger.info("session.conversation_saved",
                       session_id=self.session_id,
                       conversation_id=conversation_id,
                       turn_number=self.turn_number)

        except Exception as e:
            logger.error("session.save_conversation_failed",
                        session_id=self.session_id,
                        error=str(e))
            # Don't raise - database save failure shouldn't break the session


def classify_disconnect_reason(exception: Exception) -> DisconnectReason:
    """Classify disconnect reason for appropriate retry logic"""
    if isinstance(exception, WebSocketDisconnect):
        if exception.code == 1000:  # Normal closure
            return DisconnectReason.CLIENT_CLOSE
        elif exception.code in [1001, 1006]:  # Going away or abnormal
            return DisconnectReason.NETWORK_ERROR
        else:
            return DisconnectReason.CLIENT_CLOSE

    elif isinstance(exception, (asyncio.TimeoutError, TimeoutError)):
        return DisconnectReason.TIMEOUT

    elif isinstance(exception, (ConnectionResetError, ConnectionAbortedError)):
        return DisconnectReason.NETWORK_ERROR

    else:
        return DisconnectReason.SERVER_ERROR


async def enhanced_websocket_endpoint(websocket: WebSocket, session_id: Optional[str] = None):
    """
    Enhanced WebSocket endpoint with reconnection support

    Args:
        websocket: WebSocket connection
        session_id: Optional session ID for recovery (from query params)
    """
    reconnect_manager = get_reconnect_manager()

    # Handle session recovery or create new session
    if session_id:
        recovered_state = reconnect_manager.try_recover_session(session_id)
        if recovered_state:
            logger.info("websocket.session_recovery_attempted",
                       session_id=session_id,
                       turn_number=recovered_state.turn_number)
        else:
            session_id = str(uuid.uuid4())
    else:
        session_id = str(uuid.uuid4())
        recovered_state = None

    # Session slot management (integrated with existing system)
    from avatar.core.session_controller import get_session_controller

    controller = get_session_controller()
    session_result = await controller.request_session(
        session_id=session_id,
        service_type="websocket",
        websocket_connection=websocket
    )

    if not session_result.success:
        await websocket.accept()
        error_msg = ErrorMessage(
            error="Server is at capacity. Please try again later.",
            code="SERVER_FULL",
            session_id=session_id,
        )
        await websocket.send_text(error_msg.model_dump_json())
        await websocket.close()

        logger.warning("websocket.rejected",
                      session_id=session_id,
                      reason="server_full")
        return

    # Accept connection and create session
    await websocket.accept()
    await reconnect_manager.handle_successful_connection(websocket)

    session = EnhancedConversationSession(session_id, websocket, recovered_state)

    # Send session status
    if recovered_state:
        await session.send_status(f"Session recovered: {session_id}", "recovered")
    else:
        await session.send_status(f"Session created: {session_id}", "ready")

    try:
        while True:
            # Receive message with timeout for heartbeat
            try:
                raw_message = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=reconnect_manager.config.heartbeat_interval_seconds * 2
                )
            except asyncio.TimeoutError:
                # Send heartbeat check
                await session.send_status("Heartbeat", "heartbeat")
                continue

            try:
                # Parse and handle message
                message_data = json.loads(raw_message)
                message_type = message_data.get("type")

                if message_type == "audio_chunk":
                    msg = AudioChunkMessage(**message_data)
                    session.add_audio_chunk(msg.data)

                elif message_type == "audio_end":
                    msg = AudioEndMessage(**message_data)
                    session.voice_profile_id = msg.voice_profile_id
                    await session.process_audio()

                elif message_type == "ping":
                    # Handle client ping
                    await session.send_status("Pong", "pong")

                else:
                    await session.send_error(
                        f"Unknown message type: {message_type}",
                        "UNKNOWN_MESSAGE_TYPE",
                        is_recoverable=True
                    )

            except ValidationError as e:
                await session.send_error(
                    f"Invalid message format: {str(e)}",
                    "VALIDATION_ERROR",
                    is_recoverable=True
                )
            except json.JSONDecodeError:
                await session.send_error(
                    "Invalid JSON format",
                    "JSON_DECODE_ERROR",
                    is_recoverable=True
                )
            except RuntimeError as e:
                # Buffer limit exceeded
                await session.send_error(
                    str(e),
                    "BUFFER_LIMIT_EXCEEDED",
                    is_recoverable=True
                )
                # Clear buffer to allow retry
                session.audio_buffer.clear()
                session.buffer_size_bytes = 0
                session.buffer_first_chunk_time = None

    except Exception as e:
        disconnect_reason = classify_disconnect_reason(e)

        # Preserve session state for recovery
        session_snapshot = session.create_snapshot()

        # Handle disconnection through reconnection manager
        await reconnect_manager.handle_disconnect(disconnect_reason, session_snapshot)

        logger.info("websocket.disconnected",
                   session_id=session_id,
                   reason=disconnect_reason.value,
                   turns=session.turn_number)

    finally:
        # Release session slot
        await controller.cancel_session(session_id)
        logger.info("session.closed",
                   session_id=session_id,
                   turns=session.turn_number)