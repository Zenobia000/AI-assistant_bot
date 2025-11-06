/**
 * AVATAR WebSocket Hook
 *
 * React hook for managing WebSocket connection with AVATAR backend
 * Handles connection state, audio streaming, and message handling
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import {
  AVATARWebSocketClient,
  ConnectionState,
  StatusMessage,
  ErrorMessage,
  TTSReadyMessage,
  WebSocketEventHandlers
} from '@/lib/websocket-client';

export interface AudioRecordingState {
  isRecording: boolean;
  mediaRecorder: MediaRecorder | null;
  audioChunks: Blob[];
  recordingDuration: number;
}

export interface ConversationState {
  sessionId: string | null;
  currentStage: string;
  transcript: string;
  aiResponse: string;
  audioUrl: string | null;
  error: string | null;
}

export interface UseAvatarWebSocketReturn {
  // Connection state
  connectionState: ConnectionState;
  isConnected: boolean;
  sessionId: string | null;

  // Conversation state
  conversation: ConversationState;

  // Audio recording state
  recording: AudioRecordingState;

  // Methods
  connect: (sessionId?: string) => Promise<boolean>;
  disconnect: () => void;
  startRecording: () => Promise<boolean>;
  stopRecording: () => void;
  sendAudio: (audioBlob: Blob, voiceProfileId?: number) => Promise<boolean>;

  // Event handlers
  onStatusUpdate: (handler: (status: StatusMessage) => void) => void;
  onError: (handler: (error: ErrorMessage) => void) => void;
  onTTSReady: (handler: (tts: TTSReadyMessage) => void) => void;
}

export const useAvatarWebSocket = (): UseAvatarWebSocketReturn => {
  const [connectionState, setConnectionState] = useState<ConnectionState>(ConnectionState.DISCONNECTED);
  const [conversation, setConversation] = useState<ConversationState>({
    sessionId: null,
    currentStage: 'ready',
    transcript: '',
    aiResponse: '',
    audioUrl: null,
    error: null
  });

  const [recording, setRecording] = useState<AudioRecordingState>({
    isRecording: false,
    mediaRecorder: null,
    audioChunks: [],
    recordingDuration: 0
  });

  const wsClient = useRef<AVATARWebSocketClient | null>(null);
  const recordingTimer = useRef<NodeJS.Timeout | null>(null);
  const eventHandlers = useRef<{
    onStatus?: (status: StatusMessage) => void;
    onError?: (error: ErrorMessage) => void;
    onTTSReady?: (tts: TTSReadyMessage) => void;
  }>({});

  // Initialize WebSocket client
  useEffect(() => {
    const handlers: WebSocketEventHandlers = {
      onConnected: (sessionId: string) => {
        setConversation(prev => ({ ...prev, sessionId, error: null }));
      },
      onDisconnected: (reason: string) => {
        setConversation(prev => ({ ...prev, error: `Disconnected: ${reason}` }));
      },
      onConnectionStateChange: (state: ConnectionState) => {
        setConnectionState(state);
      },
      onStatus: (status: StatusMessage) => {
        setConversation(prev => ({
          ...prev,
          currentStage: status.stage,
          sessionId: status.session_id || prev.sessionId
        }));

        // Call external handler if set
        if (eventHandlers.current.onStatus) {
          eventHandlers.current.onStatus(status);
        }
      },
      onError: (error: ErrorMessage) => {
        setConversation(prev => ({
          ...prev,
          error: error.error,
          currentStage: 'error'
        }));

        // Call external handler if set
        if (eventHandlers.current.onError) {
          eventHandlers.current.onError(error);
        }
      },
      onTTSReady: (tts: TTSReadyMessage) => {
        setConversation(prev => ({
          ...prev,
          audioUrl: tts.audio_url,
          currentStage: 'ready'
        }));

        // Call external handler if set
        if (eventHandlers.current.onTTSReady) {
          eventHandlers.current.onTTSReady(tts);
        }
      }
    };

    wsClient.current = new AVATARWebSocketClient({}, handlers);

    return () => {
      if (wsClient.current) {
        wsClient.current.disconnect();
      }
    };
  }, []);

  // Connection methods
  const connect = useCallback(async (sessionId?: string): Promise<boolean> => {
    if (!wsClient.current) return false;

    try {
      const success = await wsClient.current.connect(sessionId);
      if (success) {
        setConversation(prev => ({ ...prev, error: null }));
      }
      return success;
    } catch (error) {
      setConversation(prev => ({
        ...prev,
        error: error instanceof Error ? error.message : 'Connection failed'
      }));
      return false;
    }
  }, []);

  const disconnect = useCallback(() => {
    if (wsClient.current) {
      wsClient.current.disconnect();
    }
    setConversation(prev => ({ ...prev, sessionId: null, currentStage: 'ready' }));
  }, []);

  // Audio recording methods
  const startRecording = useCallback(async (): Promise<boolean> => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate: API_CONFIG.AUDIO.SAMPLE_RATE,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true
        }
      });

      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: 'audio/webm;codecs=opus'
      });

      const audioChunks: Blob[] = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunks.push(event.data);
        }
      };

      mediaRecorder.start(100); // 100ms chunks

      setRecording({
        isRecording: true,
        mediaRecorder,
        audioChunks: [],
        recordingDuration: 0
      });

      // Start recording timer
      const startTime = Date.now();
      recordingTimer.current = setInterval(() => {
        const duration = (Date.now() - startTime) / 1000;
        setRecording(prev => ({ ...prev, recordingDuration: duration }));

        // Auto-stop after max duration
        if (duration >= API_CONFIG.AUDIO.MAX_DURATION) {
          stopRecording();
        }
      }, 100);

      // Store chunks as they come
      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          setRecording(prev => ({
            ...prev,
            audioChunks: [...prev.audioChunks, event.data]
          }));
        }
      };

      return true;

    } catch (error) {
      console.error('Failed to start recording:', error);
      setConversation(prev => ({
        ...prev,
        error: 'Microphone access denied or unavailable'
      }));
      return false;
    }
  }, []);

  const stopRecording = useCallback(() => {
    if (recording.mediaRecorder && recording.isRecording) {
      recording.mediaRecorder.stop();

      // Stop all media tracks
      recording.mediaRecorder.stream.getTracks().forEach(track => track.stop());
    }

    if (recordingTimer.current) {
      clearInterval(recordingTimer.current);
      recordingTimer.current = null;
    }

    setRecording(prev => ({
      ...prev,
      isRecording: false,
      mediaRecorder: null
    }));
  }, [recording.mediaRecorder, recording.isRecording]);

  // Audio sending method
  const sendAudio = useCallback(async (
    audioBlob: Blob,
    voiceProfileId?: number
  ): Promise<boolean> => {
    if (!wsClient.current || !wsClient.current.isConnected) {
      setConversation(prev => ({ ...prev, error: 'WebSocket not connected' }));
      return false;
    }

    try {
      // Convert blob to base64 chunks
      const arrayBuffer = await audioBlob.arrayBuffer();
      const uint8Array = new Uint8Array(arrayBuffer);

      // Send in chunks
      const chunkSize = API_CONFIG.AUDIO.CHUNK_SIZE;
      const totalChunks = Math.ceil(uint8Array.length / chunkSize);

      for (let i = 0; i < totalChunks; i++) {
        const start = i * chunkSize;
        const end = Math.min(start + chunkSize, uint8Array.length);
        const chunk = uint8Array.slice(start, end);

        // Convert to base64
        const base64Chunk = btoa(String.fromCharCode.apply(null, Array.from(chunk)));

        const success = wsClient.current.sendAudioChunk(base64Chunk, i);
        if (!success) {
          throw new Error(`Failed to send audio chunk ${i}`);
        }
      }

      // Send end message
      const success = wsClient.current.sendAudioEnd(totalChunks, voiceProfileId);
      if (!success) {
        throw new Error('Failed to send audio end message');
      }

      // Reset conversation state
      setConversation(prev => ({
        ...prev,
        transcript: '',
        aiResponse: '',
        audioUrl: null,
        currentStage: 'processing',
        error: null
      }));

      return true;

    } catch (error) {
      console.error('Failed to send audio:', error);
      setConversation(prev => ({
        ...prev,
        error: error instanceof Error ? error.message : 'Failed to send audio'
      }));
      return false;
    }
  }, []);

  // Event handler setters
  const onStatusUpdate = useCallback((handler: (status: StatusMessage) => void) => {
    eventHandlers.current.onStatus = handler;
  }, []);

  const onError = useCallback((handler: (error: ErrorMessage) => void) => {
    eventHandlers.current.onError = handler;
  }, []);

  const onTTSReady = useCallback((handler: (tts: TTSReadyMessage) => void) => {
    eventHandlers.current.onTTSReady = handler;
  }, []);

  return {
    // State
    connectionState,
    isConnected: connectionState === ConnectionState.CONNECTED,
    sessionId: conversation.sessionId,
    conversation,
    recording,

    // Methods
    connect,
    disconnect,
    startRecording,
    stopRecording,
    sendAudio,

    // Event handlers
    onStatusUpdate,
    onError,
    onTTSReady
  };
};