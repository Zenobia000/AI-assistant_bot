"""
WebSocket E2E Test for AVATAR

Tests the complete WebSocket flow:
1. Connect to WebSocket
2. Send audio chunks
3. Receive and validate responses (STT, LLM, TTS)
4. Verify database persistence
5. Test error handling

This is Task 13 P0: WebSocket End-to-End Integration Test
"""

import asyncio
import base64
import json
import time
from pathlib import Path
from typing import List, Dict, Any

import structlog
from websockets import connect, WebSocketClientProtocol
from websockets.exceptions import ConnectionClosedError

from avatar.core.config import config
from avatar.services.database import db

logger = structlog.get_logger()


class WebSocketTestClient:
    """WebSocket test client for E2E testing"""

    def __init__(self, url: str):
        self.url = url
        self.ws: WebSocketClientProtocol = None
        self.session_id: str = None
        self.messages_received: List[Dict[str, Any]] = []

    async def connect(self):
        """Connect to WebSocket server"""
        try:
            self.ws = await connect(self.url)
            logger.info("ws_test.connected", url=self.url)
            return True
        except Exception as e:
            logger.error("ws_test.connection_failed", error=str(e))
            return False

    async def send_audio_chunks(self, audio_path: Path, chunk_size: int = 4096):
        """
        Send audio file in chunks

        Simulates browser MediaRecorder API behavior
        """
        if not audio_path.exists():
            raise FileNotFoundError(f"Test audio not found: {audio_path}")

        audio_data = audio_path.read_bytes()
        total_size = len(audio_data)
        chunks_sent = 0

        logger.info("ws_test.sending_audio",
                   path=str(audio_path),
                   size_bytes=total_size,
                   chunk_size=chunk_size)

        # Send audio in chunks
        for offset in range(0, total_size, chunk_size):
            chunk = audio_data[offset:offset + chunk_size]
            chunk_b64 = base64.b64encode(chunk).decode('utf-8')

            message = {
                "type": "audio_chunk",
                "data": chunk_b64,
                "session_id": self.session_id or "test"
            }

            await self.ws.send(json.dumps(message))
            chunks_sent += 1
            await asyncio.sleep(0.01)  # Simulate streaming delay

        logger.info("ws_test.audio_sent", chunks=chunks_sent, total_bytes=total_size)
        return chunks_sent

    async def send_audio_end(self, voice_profile_id: int = None):
        """Signal end of audio recording"""
        message = {
            "type": "audio_end",
            "session_id": self.session_id or "test",
            "voice_profile_id": voice_profile_id
        }

        await self.ws.send(json.dumps(message))
        logger.info("ws_test.audio_end_sent", voice_profile_id=voice_profile_id)

    async def receive_messages(self, timeout: float = 60.0) -> List[Dict[str, Any]]:
        """
        Receive all messages from server until TTS ready or timeout

        Returns list of all received messages
        """
        start_time = time.time()
        messages = []

        try:
            while True:
                # Check timeout
                if time.time() - start_time > timeout:
                    logger.warning("ws_test.receive_timeout", elapsed=timeout)
                    break

                # Receive message with timeout
                try:
                    raw_message = await asyncio.wait_for(
                        self.ws.recv(),
                        timeout=5.0
                    )
                except asyncio.TimeoutError:
                    # No more messages
                    break

                # Parse message
                message = json.loads(raw_message)
                messages.append(message)

                logger.info("ws_test.message_received",
                           type=message.get("type"),
                           stage=message.get("stage"),
                           is_final=message.get("is_final"))

                # Extract session ID from first message
                if self.session_id is None and "session_id" in message:
                    self.session_id = message["session_id"]
                    logger.info("ws_test.session_id", session_id=self.session_id)

                # Check if pipeline is complete
                if message.get("type") == "tts_ready":
                    logger.info("ws_test.pipeline_complete", tts_url=message.get("audio_url"))
                    break

                # Check for errors
                if message.get("type") == "error":
                    logger.error("ws_test.error_received",
                                code=message.get("code"),
                                error=message.get("error"))
                    break

        except ConnectionClosedError as e:
            logger.error("ws_test.connection_closed", code=e.code, reason=e.reason)

        self.messages_received = messages
        return messages

    async def close(self):
        """Close WebSocket connection"""
        if self.ws:
            await self.ws.close()
            logger.info("ws_test.closed")


async def test_websocket_happy_path():
    """
    Test Case 1: Happy Path - Complete conversation flow

    Expected flow:
    1. Connect → receive session_id
    2. Send audio chunks → receive status updates
    3. Send audio_end → trigger pipeline
    4. Receive: transcription → llm_response → tts_ready
    5. Verify database persistence
    """
    print("\n" + "="*60)
    print("TEST 1: WebSocket Happy Path")
    print("="*60)

    # Test audio path
    test_audio = config.AUDIO_RAW / "test_sample.wav"
    if not test_audio.exists():
        print(f"❌ Test audio not found: {test_audio}")
        return False

    # Connect to WebSocket
    ws_url = f"ws://{config.HOST}:{config.PORT}/ws/chat"
    client = WebSocketTestClient(ws_url)

    if not await client.connect():
        print(f"❌ Failed to connect to {ws_url}")
        print("   Make sure FastAPI server is running:")
        print(f"   PYTHONPATH=src poetry run python -m avatar.main")
        return False

    print(f"✅ Connected to {ws_url}")

    try:
        # Start receiving messages in background
        receive_task = asyncio.create_task(client.receive_messages(timeout=90.0))

        # Wait for initial status message
        await asyncio.sleep(0.5)

        # Send audio chunks
        print(f"\n📤 Sending audio file: {test_audio.name}")
        chunks_sent = await client.send_audio_chunks(test_audio, chunk_size=8192)
        print(f"   Sent {chunks_sent} chunks")

        # Signal end of audio
        print("\n🏁 Signaling audio end")
        await client.send_audio_end(voice_profile_id=None)

        # Wait for all responses
        print("\n⏳ Waiting for AI pipeline to complete...")
        messages = await receive_task

        # Analyze results
        print(f"\n📨 Received {len(messages)} messages:")

        message_types = {}
        transcription_text = None
        llm_response_text = None
        tts_url = None

        for msg in messages:
            msg_type = msg.get("type")
            message_types[msg_type] = message_types.get(msg_type, 0) + 1

            if msg_type == "status":
                print(f"   - Status: {msg.get('stage')} - {msg.get('message')}")
            elif msg_type == "transcription":
                transcription_text = msg.get("text")
                print(f"   - Transcription: '{transcription_text[:50]}...'")
            elif msg_type == "llm_response":
                if msg.get("is_final"):
                    llm_response_text = msg.get("text")
                    print(f"   - LLM Response: '{llm_response_text[:50]}...'")
            elif msg_type == "tts_ready":
                tts_url = msg.get("audio_url")
                print(f"   - TTS Ready: {tts_url}")
            elif msg_type == "error":
                print(f"   ❌ Error: {msg.get('code')} - {msg.get('error')}")

        # Validate pipeline completion
        print("\n✅ Validation:")
        success = True

        if "status" not in message_types:
            print("   ❌ No status messages received")
            success = False
        else:
            print(f"   ✅ Status messages: {message_types['status']}")

        if transcription_text:
            print(f"   ✅ Transcription: {len(transcription_text)} chars")
        else:
            print("   ❌ No transcription received")
            success = False

        if llm_response_text:
            print(f"   ✅ LLM Response: {len(llm_response_text)} chars")
        else:
            print("   ❌ No LLM response received")
            success = False

        if tts_url:
            print(f"   ✅ TTS URL: {tts_url}")
        else:
            print("   ❌ No TTS audio URL received")
            success = False

        # Verify database persistence
        if client.session_id:
            print(f"\n🗄️  Verifying database persistence (session: {client.session_id[:8]}...)")
            history = await db.get_conversation_history(client.session_id, limit=10)
            if history:
                print(f"   ✅ Database: {len(history)} conversation(s) saved")
                conv = history[0]
                print(f"   - User text: {conv['user_text'][:50]}...")
                print(f"   - AI text: {conv['ai_text'][:50]}...")
            else:
                print("   ❌ Database: No conversations found")
                success = False

        print("\n" + "="*60)
        if success:
            print("✅ TEST PASSED: WebSocket Happy Path")
        else:
            print("❌ TEST FAILED: WebSocket Happy Path")
        print("="*60)

        return success

    except Exception as e:
        logger.exception("ws_test.happy_path.failed")
        print(f"\n❌ Test failed with exception: {e}")
        return False

    finally:
        await client.close()


async def test_websocket_buffer_limits():
    """
    Test Case 2: Buffer Limit Enforcement

    Test that WebSocket properly rejects:
    - Too many chunks (>1000)
    - Too large buffer (>10MB)
    - Timeout (>60s buffering)
    """
    print("\n" + "="*60)
    print("TEST 2: WebSocket Buffer Limits")
    print("="*60)

    ws_url = f"ws://{config.HOST}:{config.PORT}/ws/chat"
    client = WebSocketTestClient(ws_url)

    if not await client.connect():
        print("❌ Failed to connect")
        return False

    print("✅ Connected")

    try:
        # Test: Too many chunks (>1000)
        print("\n📤 Test: Sending 1001 chunks (should trigger limit)")

        receive_task = asyncio.create_task(client.receive_messages(timeout=10.0))

        for i in range(1001):
            chunk_b64 = base64.b64encode(b"X" * 100).decode('utf-8')
            message = {
                "type": "audio_chunk",
                "data": chunk_b64,
                "session_id": "test"
            }
            await client.ws.send(json.dumps(message))

            if i % 100 == 0:
                print(f"   Sent {i} chunks...")

        # Signal end (should fail due to buffer limit)
        await client.send_audio_end()

        messages = await receive_task

        # Check for error
        error_received = False
        for msg in messages:
            if msg.get("type") == "error" and "BUFFER" in msg.get("code", ""):
                error_received = True
                print(f"   ✅ Buffer limit error received: {msg.get('code')}")
                break

        if not error_received:
            print("   ❌ No buffer limit error received")
            return False

        print("\n" + "="*60)
        print("✅ TEST PASSED: Buffer Limits Enforced")
        print("="*60)

        return True

    except Exception as e:
        logger.exception("ws_test.buffer_limits.failed")
        print(f"\n❌ Test failed: {e}")
        return False

    finally:
        await client.close()


async def test_websocket_concurrency():
    """
    Test Case 3: Concurrent Session Handling

    Test that SessionManager properly:
    - Accepts up to MAX_CONCURRENT_SESSIONS
    - Rejects sessions when full
    - Releases sessions on disconnect
    """
    print("\n" + "="*60)
    print("TEST 3: Concurrent Session Handling")
    print("="*60)

    max_sessions = config.MAX_CONCURRENT_SESSIONS
    print(f"Max concurrent sessions: {max_sessions}")

    ws_url = f"ws://{config.HOST}:{config.PORT}/ws/chat"
    clients = []

    try:
        # Create max_sessions + 1 connections
        for i in range(max_sessions + 1):
            client = WebSocketTestClient(ws_url)
            if await client.connect():
                clients.append(client)
                print(f"✅ Client {i+1} connected")

                # Wait for initial status
                await asyncio.sleep(0.2)

                # Check if this is the extra client that should be rejected
                if len(clients) == max_sessions + 1:
                    # Try to receive messages
                    messages = []
                    try:
                        raw_msg = await asyncio.wait_for(client.ws.recv(), timeout=1.0)
                        messages.append(json.loads(raw_msg))
                    except asyncio.TimeoutError:
                        pass

                    # Check for SERVER_FULL error
                    for msg in messages:
                        if msg.get("type") == "error" and msg.get("code") == "SERVER_FULL":
                            print(f"   ✅ Client {i+1} rejected (SERVER_FULL)")
                            break
                    else:
                        print(f"   ❌ Client {i+1} should have been rejected")
                        return False

        print(f"\n✅ Concurrency test passed:")
        print(f"   - {max_sessions} clients accepted")
        print(f"   - 1 client rejected (SERVER_FULL)")

        print("\n" + "="*60)
        print("✅ TEST PASSED: Concurrent Session Handling")
        print("="*60)

        return True

    except Exception as e:
        logger.exception("ws_test.concurrency.failed")
        print(f"\n❌ Test failed: {e}")
        return False

    finally:
        # Cleanup all clients
        for client in clients:
            await client.close()


async def main():
    """Run all WebSocket E2E tests"""
    print("\n" + "="*60)
    print("AVATAR WEBSOCKET E2E TEST SUITE")
    print("="*60)
    print(f"Server: {config.HOST}:{config.PORT}")
    print(f"Test audio: {config.AUDIO_RAW / 'test_sample.wav'}")
    print()

    # Note: These tests require FastAPI server to be running
    print("⚠️  NOTE: These tests require the FastAPI server to be running:")
    print("   PYTHONPATH=src poetry run python -m avatar.main")
    print()

    input("Press Enter to start tests (or Ctrl+C to cancel)...")

    results = []

    # Test 1: Happy Path
    try:
        result = await test_websocket_happy_path()
        results.append(("Happy Path", result))
    except Exception as e:
        logger.exception("test.happy_path.crashed")
        results.append(("Happy Path", False))

    await asyncio.sleep(2)

    # Test 2: Buffer Limits
    try:
        result = await test_websocket_buffer_limits()
        results.append(("Buffer Limits", result))
    except Exception as e:
        logger.exception("test.buffer_limits.crashed")
        results.append(("Buffer Limits", False))

    await asyncio.sleep(2)

    # Test 3: Concurrency
    try:
        result = await test_websocket_concurrency()
        results.append(("Concurrency", result))
    except Exception as e:
        logger.exception("test.concurrency.crashed")
        results.append(("Concurrency", False))

    # Print summary
    print("\n" + "="*60)
    print("TEST SUITE SUMMARY")
    print("="*60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")

    print()
    print(f"Results: {passed}/{total} tests passed")
    print("="*60)

    return passed == total


if __name__ == "__main__":
    # Run test suite
    success = asyncio.run(main())
    exit(0 if success else 1)
