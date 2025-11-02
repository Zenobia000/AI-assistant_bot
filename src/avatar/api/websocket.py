"""
WebSocket endpoint for real-time voice conversation

Handles the full conversation pipeline:
1. Receive audio stream from client
2. Transcribe with Whisper (STT)
3. Generate response with vLLM
4. Synthesize speech with TTS
5. Send audio URL back to client
"""

import asyncio
import base64
import json
import time
import uuid
from pathlib import Path
from typing import Optional

import structlog
from fastapi import WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from avatar.core.config import config
from avatar.models.messages import (
    AudioChunkMessage,
    AudioEndMessage,
    ErrorMessage,
    StatusMessage,
)
from avatar.services.database import db

logger = structlog.get_logger()


# Buffer limits to prevent memory leaks and DoS attacks
# Following Linus principle: "Fail fast, no special cases"
MAX_BUFFER_SIZE_BYTES = 10 * 1024 * 1024  # 10MB (~2 minutes of audio)
MAX_CHUNK_COUNT = 1000  # Max number of chunks
BUFFER_TIMEOUT_SECONDS = 60  # 60 seconds max buffering time


class ConversationSession:
    """
    Manages a single conversation session

    Handles audio buffering, state tracking, and AI pipeline coordination.
    """

    def __init__(self, session_id: str, websocket: WebSocket):
        self.session_id = session_id
        self.websocket = websocket
        self.audio_buffer: list[bytes] = []
        self.turn_number = 0
        self.voice_profile_id: Optional[int] = None
        self.is_processing = False

        # Buffer limit tracking
        self.buffer_size_bytes = 0
        self.buffer_first_chunk_time: Optional[float] = None

        logger.info("session.created", session_id=session_id)

    async def send_status(self, message: str, stage: str):
        """Send status update to client"""
        status = StatusMessage(
            message=message,
            stage=stage,  # type: ignore
            session_id=self.session_id,
        )
        await self.websocket.send_text(status.model_dump_json())

    async def send_error(self, error: str, code: str):
        """Send error message to client"""
        error_msg = ErrorMessage(
            error=error,
            code=code,
            session_id=self.session_id,
        )
        await self.websocket.send_text(error_msg.model_dump_json())
        logger.error("session.error", session_id=self.session_id, error=error, code=code)

    def add_audio_chunk(self, data_b64: str):
        """
        Add audio chunk to buffer with limits checking

        Prevents memory leaks and DoS attacks by enforcing:
        - Maximum buffer size (10MB)
        - Maximum chunk count (1000)
        - Buffering timeout (60s)

        Raises:
            RuntimeError: Buffer limit exceeded
        """
        try:
            audio_bytes = base64.b64decode(data_b64)
            chunk_size = len(audio_bytes)

            # Track first chunk time for timeout detection
            if self.buffer_first_chunk_time is None:
                self.buffer_first_chunk_time = time.time()

            # Check 1: Chunk count limit
            if len(self.audio_buffer) >= MAX_CHUNK_COUNT:
                logger.error("session.buffer.chunk_limit_exceeded",
                            session_id=self.session_id,
                            chunk_count=len(self.audio_buffer),
                            limit=MAX_CHUNK_COUNT)
                raise RuntimeError(
                    f"Buffer chunk limit exceeded: {len(self.audio_buffer)} >= {MAX_CHUNK_COUNT}"
                )

            # Check 2: Size limit
            if self.buffer_size_bytes + chunk_size > MAX_BUFFER_SIZE_BYTES:
                logger.error("session.buffer.size_limit_exceeded",
                            session_id=self.session_id,
                            buffer_size_mb=round(self.buffer_size_bytes / 1024 / 1024, 2),
                            chunk_size_kb=round(chunk_size / 1024, 2),
                            limit_mb=round(MAX_BUFFER_SIZE_BYTES / 1024 / 1024, 2))
                raise RuntimeError(
                    f"Buffer size limit exceeded: "
                    f"{self.buffer_size_bytes + chunk_size} > {MAX_BUFFER_SIZE_BYTES}"
                )

            # Check 3: Timeout
            elapsed = time.time() - self.buffer_first_chunk_time
            if elapsed > BUFFER_TIMEOUT_SECONDS:
                logger.error("session.buffer.timeout",
                            session_id=self.session_id,
                            elapsed_sec=round(elapsed, 2),
                            timeout_sec=BUFFER_TIMEOUT_SECONDS)
                raise RuntimeError(
                    f"Buffer timeout: {elapsed:.2f}s > {BUFFER_TIMEOUT_SECONDS}s"
                )

            # All checks passed, add to buffer
            self.audio_buffer.append(audio_bytes)
            self.buffer_size_bytes += chunk_size

            logger.debug("session.audio_chunk",
                        session_id=self.session_id,
                        chunk_size=chunk_size,
                        total_chunks=len(self.audio_buffer),
                        total_size_kb=round(self.buffer_size_bytes / 1024, 2))

        except base64.binascii.Error as e:
            logger.error("session.audio_decode_failed",
                        session_id=self.session_id,
                        error=str(e))
            raise RuntimeError(f"Invalid base64 audio data: {e}") from e

    async def process_audio(self):
        """
        Process accumulated audio through AI pipeline

        Pipeline: Audio → STT → LLM → TTS → Client
        """
        if self.is_processing:
            await self.send_error("Already processing a request", "ALREADY_PROCESSING")
            return

        if not self.audio_buffer:
            await self.send_error("No audio data received", "NO_AUDIO_DATA")
            return

        self.is_processing = True
        self.turn_number += 1

        try:
            # Save audio to file
            await self.send_status("Saving audio...", "stt")
            audio_path = await self._save_audio()

            # Step 1: STT - Transcribe audio
            await self.send_status("Transcribing speech...", "stt")
            transcription = await self._run_stt(audio_path)

            # Send transcription to client
            from avatar.models.messages import TranscriptionMessage
            trans_msg = TranscriptionMessage(
                text=transcription,
                session_id=self.session_id,
            )
            await self.websocket.send_text(trans_msg.model_dump_json())

            # Step 2: LLM - Generate response
            await self.send_status("Thinking...", "llm")
            llm_response = await self._run_llm(transcription)

            # Send LLM response to client
            from avatar.models.messages import LLMResponseMessage
            llm_msg = LLMResponseMessage(
                text=llm_response,
                is_final=True,
                session_id=self.session_id,
            )
            await self.websocket.send_text(llm_msg.model_dump_json())

            # Step 3: TTS - Synthesize speech
            await self.send_status("Synthesizing speech...", "tts")
            tts_url = await self._run_tts(
                text=llm_response,
                user_audio_path=audio_path,
                user_text=transcription
            )

            # Send TTS ready notification
            from avatar.models.messages import TTSReadyMessage
            tts_msg = TTSReadyMessage(
                audio_url=tts_url,
                audio_format="wav",
                mode="fast",
                session_id=self.session_id,
            )
            await self.websocket.send_text(tts_msg.model_dump_json())

            # Final status
            await self.send_status("Ready", "ready")

            # Save conversation to database
            await self._save_conversation(
                user_audio_path=str(audio_path),
                user_text=transcription,
                ai_text=llm_response,
                ai_audio_fast_path=tts_url,
            )

        except Exception as e:
            logger.exception("session.processing_failed", session_id=self.session_id)
            await self.send_error(f"Processing failed: {str(e)}", "PROCESSING_ERROR")

        finally:
            self.is_processing = False
            self.audio_buffer.clear()
            # Reset buffer tracking variables
            self.buffer_size_bytes = 0
            self.buffer_first_chunk_time = None

    async def _save_audio(self) -> Path:
        """
        Save buffered audio and convert to WAV 16kHz mono

        Browsers typically send WebM/Opus format, but Whisper requires
        WAV PCM 16kHz mono. This method handles the conversion automatically.

        Returns:
            Path to converted WAV file

        Raises:
            RuntimeError: Audio conversion failed
        """
        from avatar.core.audio_utils import convert_to_wav_async

        audio_data = b"".join(self.audio_buffer)

        # Step 1: Save raw audio from browser (WebM/Opus/etc.)
        raw_filename = f"{self.session_id}_turn{self.turn_number}_{uuid.uuid4().hex[:8]}.webm"
        raw_path = config.AUDIO_RAW / raw_filename
        raw_path.write_bytes(audio_data)

        logger.info("session.audio.raw_saved",
                   session_id=self.session_id,
                   path=str(raw_path),
                   size_bytes=len(audio_data))

        # Step 2: Convert to WAV 16kHz mono for Whisper
        wav_filename = f"{self.session_id}_turn{self.turn_number}_{uuid.uuid4().hex[:8]}.wav"
        wav_path = config.AUDIO_RAW / wav_filename

        try:
            converted_path, metadata = await convert_to_wav_async(
                input_path=raw_path,
                output_path=wav_path,
                target_sample_rate=16000,
                target_channels=1
            )

            logger.info("session.audio.converted",
                       session_id=self.session_id,
                       raw_path=str(raw_path),
                       wav_path=str(converted_path),
                       duration_sec=metadata["converted_duration_sec"],
                       compression_ratio=metadata["compression_ratio"])

            # Clean up raw file (optional, to save disk space)
            # raw_path.unlink(missing_ok=True)

            return converted_path

        except Exception as e:
            logger.error("session.audio.conversion_failed",
                        session_id=self.session_id,
                        raw_path=str(raw_path),
                        error=str(e))
            raise RuntimeError(f"Audio conversion failed: {e}") from e

    async def _run_stt(self, audio_path: Path) -> str:
        """Run speech-to-text on audio file using Whisper"""
        from avatar.services.stt import get_stt_service

        logger.info("session.stt.start", session_id=self.session_id, audio_path=str(audio_path))

        # Get STT service (singleton)
        stt = await get_stt_service()

        # Transcribe audio (auto language detection)
        text, metadata = await stt.transcribe(
            audio_path=audio_path,
            language=None,  # Auto-detect
            beam_size=5,
            vad_filter=True
        )

        logger.info("session.stt.complete",
                   session_id=self.session_id,
                   text=text,
                   language=metadata["language"],
                   duration_sec=metadata["duration"],
                   segments=metadata["segments_count"])

        return text

    async def _run_llm(self, user_text: str) -> str:
        """
        Generate LLM response using vLLM with streaming

        Streams response chunks to client for lower TTFT (Time To First Token).
        Returns complete response text at the end.

        Args:
            user_text: User's input text

        Returns:
            Complete LLM response text
        """
        from avatar.services.llm import get_llm_service
        from avatar.models.messages import LLMResponseMessage

        logger.info("session.llm.start", session_id=self.session_id, prompt=user_text)

        # Get LLM service (singleton)
        llm = await get_llm_service()

        # Format as chat messages
        messages = [{"role": "user", "content": user_text}]

        # Stream response chunks to client
        full_response = ""
        chunk_count = 0

        async for chunk in llm.chat_stream(
            messages=messages,
            max_tokens=512,
            temperature=0.7
        ):
            full_response += chunk
            chunk_count += 1

            # Send intermediate chunk to client
            chunk_msg = LLMResponseMessage(
                text=chunk,
                is_final=False,
                session_id=self.session_id
            )
            await self.websocket.send_text(chunk_msg.model_dump_json())

        logger.info("session.llm.complete",
                   session_id=self.session_id,
                   response_length=len(full_response),
                   chunks_sent=chunk_count,
                   prompt_length=len(user_text))

        return full_response.strip()

    async def _run_tts(self, text: str, user_audio_path: Optional[Path] = None, user_text: Optional[str] = None) -> str:
        """
        Synthesize speech from text using F5-TTS

        Supports two modes:
        1. Voice profile mode: Use pre-registered voice profile
        2. Self-cloning mode: Use user's own audio as reference (fallback)

        Args:
            text: Text to synthesize
            user_audio_path: Path to user's audio (for self-cloning fallback)
            user_text: User's transcribed text (for self-cloning fallback)

        Returns:
            URL to synthesized audio file
        """
        from avatar.services.tts import get_tts_service

        logger.info("session.tts.start",
                   session_id=self.session_id,
                   text_length=len(text),
                   has_voice_profile=self.voice_profile_id is not None)

        # Get TTS service (singleton)
        tts = await get_tts_service()

        # Output file path
        filename = f"{self.session_id}_turn{self.turn_number}_tts.wav"
        output_path = config.AUDIO_TTS_FAST / filename

        try:
            if self.voice_profile_id:
                # Mode 1: Use voice profile
                # TODO: Query database for voice_profile_name
                # For now, use placeholder
                voice_profile_name = f"profile_{self.voice_profile_id}"

                await tts.synthesize_fast(
                    text=text,
                    voice_profile_name=voice_profile_name,
                    output_path=output_path
                )

                logger.info("session.tts.complete",
                           session_id=self.session_id,
                           mode="voice_profile",
                           profile_id=self.voice_profile_id)

            elif user_audio_path and user_text:
                # Mode 2: Self-cloning fallback
                logger.info("session.tts.self_cloning",
                           session_id=self.session_id,
                           ref_audio=str(user_audio_path))

                await tts.synthesize(
                    text=text,
                    ref_audio_path=user_audio_path,
                    ref_text=user_text,
                    output_path=output_path,
                    remove_silence=True
                )

                logger.info("session.tts.complete",
                           session_id=self.session_id,
                           mode="self_cloning")

            else:
                # No voice profile and no reference audio
                raise RuntimeError(
                    "TTS requires either voice_profile_id or user audio for reference"
                )

        except FileNotFoundError as e:
            logger.error("session.tts.profile_not_found",
                        session_id=self.session_id,
                        error=str(e))
            raise RuntimeError(f"Voice profile not found: {e}") from e

        # Return audio URL
        audio_url = f"/api/audio/tts/{filename}"

        logger.info("session.tts.complete",
                   session_id=self.session_id,
                   url=audio_url,
                   size_bytes=output_path.stat().st_size)

        return audio_url

    async def _save_conversation(
        self,
        user_audio_path: str,
        user_text: str,
        ai_text: str,
        ai_audio_fast_path: str,
    ):
        """Save conversation turn to database"""
        try:
            conversation_id = await db.save_conversation(
                session_id=self.session_id,
                turn_number=self.turn_number,
                user_audio_path=user_audio_path,
                user_text=user_text,
                ai_text=ai_text,
                ai_audio_fast_path=ai_audio_fast_path,
                voice_profile_id=self.voice_profile_id,
            )
            logger.info("session.db.saved",
                       conversation_id=conversation_id,
                       session_id=self.session_id,
                       turn=self.turn_number)
        except Exception as e:
            logger.error("session.db.save_failed",
                        session_id=self.session_id,
                        error=str(e))


async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint handler for /ws/chat

    Manages the full lifecycle of a conversation session.
    Includes VRAM monitoring and concurrency control.
    """
    from avatar.core.session_manager import get_session_manager

    # Generate session ID before accepting connection
    session_id = str(uuid.uuid4())

    # Try to acquire session slot (with VRAM check)
    session_manager = get_session_manager()

    if not await session_manager.acquire_session(session_id, timeout=1.0):
        # Server is full or VRAM exhausted
        await websocket.accept()  # Accept to send error message

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

    # Session acquired successfully
    await websocket.accept()
    session = ConversationSession(session_id, websocket)

    # Send session created notification
    await session.send_status(f"Session created: {session_id}", "ready")

    try:
        while True:
            # Receive message from client
            raw_message = await websocket.receive_text()

            try:
                # Parse message type
                message_data = json.loads(raw_message)
                message_type = message_data.get("type")

                if message_type == "audio_chunk":
                    msg = AudioChunkMessage(**message_data)
                    session.add_audio_chunk(msg.data)

                elif message_type == "audio_end":
                    msg = AudioEndMessage(**message_data)
                    session.voice_profile_id = msg.voice_profile_id

                    # Process the complete audio
                    await session.process_audio()

                else:
                    await session.send_error(
                        f"Unknown message type: {message_type}",
                        "UNKNOWN_MESSAGE_TYPE"
                    )

            except ValidationError as e:
                await session.send_error(
                    f"Invalid message format: {str(e)}",
                    "VALIDATION_ERROR"
                )
            except json.JSONDecodeError:
                await session.send_error(
                    "Invalid JSON format",
                    "JSON_DECODE_ERROR"
                )
            except RuntimeError as e:
                # Buffer limit exceeded (size/count/timeout)
                await session.send_error(
                    str(e),
                    "BUFFER_LIMIT_EXCEEDED"
                )
                # Clear buffer to allow retry
                session.audio_buffer.clear()
                session.buffer_size_bytes = 0
                session.buffer_first_chunk_time = None

    except WebSocketDisconnect:
        logger.info("session.disconnected", session_id=session_id)
    except Exception as e:
        logger.exception("session.fatal_error", session_id=session_id)
        await session.send_error(f"Fatal error: {str(e)}", "FATAL_ERROR")
    finally:
        # Release session slot
        session_manager.release_session(session_id)
        logger.info("session.closed",
                   session_id=session_id,
                   turns=session.turn_number)
