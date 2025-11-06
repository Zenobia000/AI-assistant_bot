import { useState, useRef } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Upload, Trash2, Play, Loader2 } from "lucide-react";
import { useVoiceProfiles, useUploadVoiceProfile, useDeleteVoiceProfile, useTestVoiceProfileSynthesis } from "@/hooks/use-avatar-api";
import type { VoiceProfile } from "@/lib/api-config";

const VoiceProfileManager = () => {
  const { data: profiles, isLoading, error, refetch } = useVoiceProfiles();
  const uploadMutation = useUploadVoiceProfile();
  const deleteMutation = useDeleteVoiceProfile();
  const synthesisMutation = useTestVoiceProfileSynthesis();

  const [uploadName, setUploadName] = useState("");
  const [dragActive, setDragActive] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDelete = async (profileId: number) => {
    if (confirm("Are you sure you want to delete this voice profile?")) {
      try {
        await deleteMutation.mutateAsync(profileId);
      } catch (error) {
        console.error("Failed to delete profile:", error);
      }
    }
  };

  const handleFileUpload = async (file: File) => {
    if (!uploadName.trim()) {
      alert("Please enter a name for the voice profile");
      return;
    }

    try {
      await uploadMutation.mutateAsync({
        name: uploadName.trim(),
        audioFile: file
      });
      setUploadName("");
    } catch (error) {
      console.error("Failed to upload voice profile:", error);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragActive(false);

    const files = Array.from(e.dataTransfer.files);
    const audioFile = files.find(file => file.type.startsWith('audio/'));

    if (audioFile) {
      handleFileUpload(audioFile);
    }
  };

  const handleFileInput = () => {
    fileInputRef.current?.click();
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      handleFileUpload(file);
    }
  };

  const testSynthesis = async (profileId: number, profileName: string) => {
    try {
      const result = await synthesisMutation.mutateAsync({
        profileId,
        text: `Hello, this is a test of the ${profileName} voice profile.`
      });

      // Play the generated audio
      if (result.audio_url) {
        const audio = new Audio(result.audio_url);
        audio.play();
      }
    } catch (error) {
      console.error("Failed to test synthesis:", error);
    }
  };

  return (
    <section className="py-20 px-4 bg-gradient-to-b from-transparent via-muted/5 to-transparent">
      <div className="max-w-6xl mx-auto">
        <h2 className="text-4xl font-bold text-center mb-12 text-foreground">
          Voice <span className="text-neon-pink">Profile</span> Manager
        </h2>

        <Card className="bg-glass-gradient backdrop-blur-xl border-2 border-neon-pink/30 rounded-2xl p-8 shadow-2xl">
          {/* Upload Section */}
          <div className="mb-8 space-y-4">
            <div className="flex gap-4">
              <Input
                placeholder="Enter voice profile name..."
                value={uploadName}
                onChange={(e) => setUploadName(e.target.value)}
                className="flex-1 bg-muted/30 border-accent/20 focus:border-accent"
              />
              <Button
                onClick={handleFileInput}
                disabled={!uploadName.trim() || uploadMutation.isPending}
                className="bg-gold-gradient hover:opacity-90 text-black font-semibold"
              >
                {uploadMutation.isPending ? (
                  <Loader2 className="w-4 h-4 animate-spin mr-2" />
                ) : (
                  <Upload className="w-4 h-4 mr-2" />
                )}
                Choose File
              </Button>
            </div>

            <div
              className={`border-2 border-dashed rounded-xl p-8 text-center hover:border-accent transition-colors cursor-pointer bg-accent/5 ${
                dragActive ? 'border-accent bg-accent/10' : 'border-accent/50'
              }`}
              onDrop={handleDrop}
              onDragOver={(e) => {
                e.preventDefault();
                setDragActive(true);
              }}
              onDragLeave={() => setDragActive(false)}
              onClick={handleFileInput}
            >
              <Upload className="w-8 h-8 text-accent mx-auto mb-2" />
              <p className="text-sm text-muted-foreground">
                Drag & drop audio file or click to browse
              </p>
              <p className="text-xs text-muted-foreground mt-1">
                Supports: WAV, MP3, FLAC (max 10MB)
              </p>
            </div>

            <input
              ref={fileInputRef}
              type="file"
              accept="audio/*"
              onChange={handleFileSelect}
              className="hidden"
            />

            {uploadMutation.error && (
              <p className="text-sm text-red-500">
                Upload failed: {uploadMutation.error.message}
              </p>
            )}
          </div>

          {/* Profile List */}
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-xl font-semibold text-foreground">
                Saved Profiles
              </h3>
              <Button
                onClick={() => refetch()}
                variant="outline"
                size="sm"
                disabled={isLoading}
                className="border-accent/50"
              >
                {isLoading ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  "Refresh"
                )}
              </Button>
            </div>

            {error && (
              <div className="text-center p-8">
                <p className="text-red-500 mb-4">Failed to load voice profiles</p>
                <Button onClick={() => refetch()} variant="outline">
                  Retry
                </Button>
              </div>
            )}

            {isLoading && (
              <div className="text-center p-8">
                <Loader2 className="w-8 h-8 animate-spin text-accent mx-auto mb-4" />
                <p className="text-muted-foreground">Loading voice profiles...</p>
              </div>
            )}

            {profiles && profiles.length === 0 && !isLoading && (
              <div className="text-center p-8">
                <p className="text-muted-foreground">No voice profiles found</p>
                <p className="text-sm text-muted-foreground mt-2">
                  Upload your first voice sample to get started
                </p>
              </div>
            )}

            {profiles && profiles.map((profile) => (
              <div
                key={profile.id}
                className="bg-muted/30 border border-neon-pink/20 rounded-xl p-6 hover:border-neon-pink/40 transition-all group"
              >
                <div className="flex items-center justify-between">
                  <div className="flex-1">
                    <h4 className="text-lg font-semibold text-foreground mb-2">
                      {profile.name}
                    </h4>

                    {/* Audio file info */}
                    <div className="flex items-center gap-1 mb-3">
                      <div className="flex-1 bg-neon-pink/10 rounded-full h-2">
                        <div className="bg-neon-pink/60 h-full rounded-full" style={{ width: '100%' }}></div>
                      </div>
                    </div>

                    <div className="flex gap-6 text-sm text-muted-foreground">
                      <span>Created: {new Date(profile.created_at).toLocaleDateString()}</span>
                      <span className="font-mono text-xs">ID: {profile.id}</span>
                      {profile.embedding_path && (
                        <span className="text-green-500">✓ Processed</span>
                      )}
                    </div>
                  </div>

                  <div className="flex gap-2">
                    <Button
                      variant="outline"
                      size="sm"
                      className="border-neon-blue/50 hover:bg-neon-blue/10"
                      onClick={() => testSynthesis(profile.id, profile.name)}
                      disabled={synthesisMutation.isPending}
                    >
                      {synthesisMutation.isPending ? (
                        <Loader2 className="w-4 h-4 animate-spin" />
                      ) : (
                        <Play className="w-4 h-4 mr-2" />
                      )}
                      Test
                    </Button>

                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => handleDelete(profile.id)}
                      disabled={deleteMutation.isPending}
                      className="border-destructive/50 hover:bg-destructive/10 text-destructive"
                    >
                      {deleteMutation.isPending ? (
                        <Loader2 className="w-4 h-4 animate-spin" />
                      ) : (
                        <Trash2 className="w-4 h-4" />
                      )}
                    </Button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </Card>
      </div>
    </section>
  );
};

export default VoiceProfileManager;
