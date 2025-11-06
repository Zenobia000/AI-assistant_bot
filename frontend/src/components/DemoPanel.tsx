import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Mic, MicOff, Play, Zap, Cpu, Clock, Wifi, WifiOff } from "lucide-react";
import { useAvatarWebSocket } from "@/hooks/use-avatar-websocket";
import { useSystemHealth, useVRAMStatus } from "@/hooks/use-avatar-api";
import { ConnectionState } from "@/lib/websocket-client";

const DemoPanel = () => {
  const {
    connectionState,
    isConnected,
    sessionId,
    conversation,
    recording,
    connect,
    disconnect,
    startRecording,
    stopRecording,
    sendAudio,
    onStatusUpdate,
    onTTSReady,
    onError
  } = useAvatarWebSocket();

  const { data: systemHealth } = useSystemHealth();
  const { data: vramStatus } = useVRAMStatus();

  const [streamingText, setStreamingText] = useState("");
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [latencyMetrics, setLatencyMetrics] = useState({ stt: 0, llm: 0, tts: 0 });

  // Handle WebSocket events
  useEffect(() => {
    onStatusUpdate((status) => {
      if (status.stage === 'stt') {
        setStreamingText("🎤 Processing speech...");
      } else if (status.stage === 'llm') {
        setStreamingText("🧠 Thinking...");
      } else if (status.stage === 'tts') {
        setStreamingText("🔊 Generating voice...");
      }
    });

    onTTSReady((tts) => {
      setAudioUrl(tts.audio_url);
      setStreamingText(conversation.aiResponse || "Response ready!");
    });

    onError((error) => {
      setStreamingText(`❌ Error: ${error.error}`);
    });
  }, [onStatusUpdate, onTTSReady, onError, conversation.aiResponse]);

  // Auto-connect on component mount
  useEffect(() => {
    if (connectionState === ConnectionState.DISCONNECTED) {
      connect();
    }
  }, [connect, connectionState]);

  const toggleRecording = async () => {
    if (recording.isRecording) {
      stopRecording();

      // Send recorded audio
      if (recording.audioChunks.length > 0) {
        const audioBlob = new Blob(recording.audioChunks, { type: 'audio/webm' });
        await sendAudio(audioBlob);
      }
    } else {
      if (!isConnected) {
        await connect();
      }
      await startRecording();
    }
  };

  const playAudio = () => {
    if (audioUrl) {
      const audio = new Audio(audioUrl);
      audio.play();
    }
  };

  // Connection status badge
  const getConnectionBadge = () => {
    switch (connectionState) {
      case ConnectionState.CONNECTED:
        return (
          <Badge className="bg-green-500/20 text-green-500 border-green-500/50 px-4 py-2 text-sm">
            <Wifi className="w-4 h-4 mr-2" />
            Connected
          </Badge>
        );
      case ConnectionState.CONNECTING:
        return (
          <Badge className="bg-yellow-500/20 text-yellow-500 border-yellow-500/50 px-4 py-2 text-sm">
            Connecting...
          </Badge>
        );
      case ConnectionState.RECONNECTING:
        return (
          <Badge className="bg-orange-500/20 text-orange-500 border-orange-500/50 px-4 py-2 text-sm">
            Reconnecting...
          </Badge>
        );
      default:
        return (
          <Badge className="bg-red-500/20 text-red-500 border-red-500/50 px-4 py-2 text-sm">
            <WifiOff className="w-4 h-4 mr-2" />
            Disconnected
          </Badge>
        );
    }
  };

  return (
    <section className="py-20 px-4">
      <div className="max-w-6xl mx-auto">
        <h2 className="text-4xl font-bold text-center mb-12 text-foreground">
          Live <span className="text-accent">Demo</span> Panel
        </h2>

        <Card className="bg-glass-gradient backdrop-blur-xl border-2 border-neon-blue/30 rounded-2xl p-8 shadow-2xl">
          {/* Status Chips */}
          <div className="flex flex-wrap gap-3 mb-8">
            {getConnectionBadge()}
            <Badge className="bg-neon-blue/20 text-neon-blue border-neon-blue/50 px-4 py-2 text-sm">
              STT {systemHealth?.status === 'healthy' ? '✓' : '⚠️'}
            </Badge>
            <Badge className="bg-neon-pink/20 text-neon-pink border-neon-pink/50 px-4 py-2 text-sm">
              LLM {systemHealth?.status === 'healthy' ? '✓' : '⚠️'}
            </Badge>
            <Badge className="bg-accent/20 text-accent border-accent/50 px-4 py-2 text-sm">
              TTS {systemHealth?.status === 'healthy' ? '✓' : '⚠️'}
            </Badge>
            {sessionId && (
              <Badge className="bg-purple-500/20 text-purple-500 border-purple-500/50 px-4 py-2 text-sm">
                Session: {sessionId.slice(-8)}
              </Badge>
            )}
          </div>

          <div className="grid md:grid-cols-2 gap-8">
            {/* Left: Mic Control & Waveform */}
            <div className="space-y-6">
              {/* Mic Button */}
              <div className="flex justify-center">
                <Button
                  onClick={toggleRecording}
                  size="lg"
                  disabled={!isConnected && !recording.isRecording}
                  className={`w-32 h-32 rounded-full transition-all duration-300 ${
                    recording.isRecording
                      ? "bg-destructive hover:bg-destructive/90 animate-pulse shadow-lg shadow-destructive/50"
                      : "bg-neon-gradient hover:opacity-90 shadow-lg shadow-neon-blue/50"
                  } ${
                    !isConnected && !recording.isRecording
                      ? "opacity-50 cursor-not-allowed"
                      : ""
                  }`}
                >
                  {recording.isRecording ? (
                    <MicOff className="w-12 h-12" />
                  ) : (
                    <Mic className="w-12 h-12" />
                  )}
                </Button>
              </div>

              {/* Recording duration */}
              {recording.isRecording && (
                <div className="text-center">
                  <p className="text-sm text-muted-foreground">
                    Recording: {recording.recordingDuration.toFixed(1)}s
                  </p>
                </div>
              )}

              {/* Waveform Visualization */}
              <div className="bg-muted/30 rounded-xl p-6 border border-neon-blue/20">
                <div className="flex items-end justify-center gap-1 h-32">
                  {[...Array(20)].map((_, i) => (
                    <div
                      key={i}
                      className={`w-2 bg-neon-gradient rounded-full transition-all duration-300 ${
                        isRecording ? "animate-pulse" : ""
                      }`}
                      style={{
                        height: isRecording ? `${Math.random() * 100 + 20}%` : "20%",
                        animationDelay: `${i * 0.1}s`,
                      }}
                    />
                  ))}
                </div>
              </div>

              {/* Transcript Display */}
              <div className="bg-muted/30 rounded-xl p-6 border border-neon-pink/20 min-h-[120px]">
                <p className="text-sm text-muted-foreground mb-2">Your Speech:</p>
                <p className="text-foreground">
                  {conversation.transcript || "Your voice transcript will appear here..."}
                </p>
              </div>
            </div>

            {/* Right: Streaming Response & Controls */}
            <div className="space-y-6">
              {/* Streaming Token Visualization */}
              <div className="bg-muted/30 rounded-xl p-6 border border-accent/20 min-h-[200px]">
                <p className="text-sm text-muted-foreground mb-2">AI Response:</p>
                <p className="text-foreground font-mono typewriter">
                  {streamingText || conversation.aiResponse || "AI response will stream here..."}
                </p>
                {conversation.currentStage !== 'ready' && (
                  <p className="text-xs text-muted-foreground mt-2">
                    Stage: {conversation.currentStage}
                  </p>
                )}
                {conversation.error && (
                  <p className="text-xs text-red-500 mt-2">
                    Error: {conversation.error}
                  </p>
                )}
              </div>

              {/* TTS Controls */}
              <div className="flex gap-4">
                <Button
                  variant="outline"
                  className="flex-1 border-neon-blue/50 hover:bg-neon-blue/10"
                  onClick={playAudio}
                  disabled={!audioUrl}
                >
                  <Play className="w-4 h-4 mr-2" />
                  Play Audio
                </Button>
                <Button
                  variant="outline"
                  className="flex-1 border-accent/50 hover:bg-accent/10"
                  disabled={!audioUrl}
                  onClick={() => {
                    if (audioUrl) {
                      window.open(audioUrl, '_blank');
                    }
                  }}
                >
                  <Zap className="w-4 h-4 mr-2" />
                  Download
                </Button>
              </div>

              {/* Metrics Cards */}
              <div className="grid grid-cols-3 gap-4">
                <div className="bg-gradient-to-br from-neon-blue/10 to-transparent border border-neon-blue/30 rounded-lg p-4">
                  <Clock className="w-5 h-5 text-neon-blue mb-2" />
                  <p className="text-xs text-muted-foreground">Uptime</p>
                  <p className="text-lg font-bold text-foreground">
                    {systemHealth?.uptime_seconds
                      ? Math.floor(systemHealth.uptime_seconds / 3600) + 'h'
                      : '---'}
                  </p>
                </div>

                <div className="bg-gradient-to-br from-accent/10 to-transparent border border-accent/30 rounded-lg p-4">
                  <Cpu className="w-5 h-5 text-accent mb-2" />
                  <p className="text-xs text-muted-foreground">VRAM</p>
                  <p className="text-lg font-bold text-foreground">
                    {vramStatus?.usage_percentage
                      ? `${vramStatus.usage_percentage.toFixed(0)}%`
                      : '---'}
                  </p>
                </div>

                <div className="bg-gradient-to-br from-neon-pink/10 to-transparent border border-neon-pink/30 rounded-lg p-4">
                  <Zap className="w-5 h-5 text-neon-pink mb-2" />
                  <p className="text-xs text-muted-foreground">Errors</p>
                  <p className="text-lg font-bold text-foreground">
                    {systemHealth?.error_rate_per_minute
                      ? `${systemHealth.error_rate_per_minute.toFixed(1)}/min`
                      : '0/min'}
                  </p>
                </div>
              </div>
            </div>
          </div>
        </Card>
      </div>
    </section>
  );
};

export default DemoPanel;
