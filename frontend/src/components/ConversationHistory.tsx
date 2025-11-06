import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Play, Download, Search, Calendar, Loader2 } from "lucide-react";
import { useConversations, useSearchConversations, useExportConversations } from "@/hooks/use-avatar-api";
import type { Conversation } from "@/lib/api-config";

const ConversationHistory = () => {
  const [searchQuery, setSearchQuery] = useState("");
  const [currentPage, setCurrentPage] = useState(0);
  const pageSize = 20;

  const { data: conversationData, isLoading, error, refetch } = useConversations(pageSize, currentPage * pageSize);
  const searchMutation = useSearchConversations();
  const exportMutation = useExportConversations();

  const conversations = conversationData?.conversations || [];
  const totalConversations = conversationData?.total || 0;

  const handleSearch = async () => {
    if (searchQuery.trim()) {
      try {
        await searchMutation.mutateAsync(searchQuery.trim());
      } catch (error) {
        console.error("Search failed:", error);
      }
    } else {
      refetch(); // Reload all conversations if search is empty
    }
  };

  const handleExport = async (format: 'json' | 'txt') => {
    try {
      await exportMutation.mutateAsync(format);
    } catch (error) {
      console.error("Export failed:", error);
    }
  };

  const playAudio = (audioPath: string) => {
    if (audioPath) {
      const audio = new Audio(audioPath);
      audio.play();
    }
  };

  return (
    <section className="py-20 px-4">
      <div className="max-w-6xl mx-auto">
        <h2 className="text-4xl font-bold text-center mb-12 text-foreground">
          Conversation <span className="text-neon-blue">History</span>
        </h2>

        <Card className="bg-glass-gradient backdrop-blur-xl border-2 border-neon-blue/30 rounded-2xl p-8 shadow-2xl">
          {/* Search & Filter Bar */}
          <div className="flex flex-col sm:flex-row gap-4 mb-8">
            <div className="flex-1 relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-muted-foreground" />
              <Input
                placeholder="Search conversations..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
                className="pl-10 bg-muted/30 border-neon-blue/20 focus:border-neon-blue"
              />
            </div>

            <Button
              onClick={handleSearch}
              disabled={searchMutation.isPending}
              className="border-neon-blue/50 hover:bg-neon-blue/10"
            >
              {searchMutation.isPending ? (
                <Loader2 className="w-4 h-4 animate-spin mr-2" />
              ) : (
                <Search className="w-4 h-4 mr-2" />
              )}
              Search
            </Button>

            <Button
              onClick={() => handleExport('json')}
              variant="outline"
              disabled={exportMutation.isPending}
              className="border-accent/50 hover:bg-accent/10"
            >
              <Download className="w-4 h-4 mr-2" />
              Export
            </Button>
          </div>

          {/* Loading & Error States */}
          {isLoading && (
            <div className="text-center p-8">
              <Loader2 className="w-8 h-8 animate-spin text-neon-blue mx-auto mb-4" />
              <p className="text-muted-foreground">Loading conversations...</p>
            </div>
          )}

          {error && (
            <div className="text-center p-8">
              <p className="text-red-500 mb-4">Failed to load conversations</p>
              <Button onClick={() => refetch()} variant="outline">
                Retry
              </Button>
            </div>
          )}

          {searchMutation.data && searchMutation.data.length === 0 && (
            <div className="text-center p-8">
              <p className="text-muted-foreground">No conversations found for "{searchQuery}"</p>
            </div>
          )}

          {conversations.length === 0 && !isLoading && !error && (
            <div className="text-center p-8">
              <p className="text-muted-foreground">No conversations found</p>
              <p className="text-sm text-muted-foreground mt-2">
                Start a voice conversation to see your history here
              </p>
            </div>
          )}

          {/* Timeline */}
          <div className="space-y-6">
            {(searchMutation.data || conversations).map((conv, index) => (
              <div key={conv.id} className="relative">
                {/* Timeline Line */}
                {index < conversations.length - 1 && (
                  <div className="absolute left-6 top-16 bottom-0 w-0.5 bg-gradient-to-b from-neon-blue to-transparent" />
                )}

                <div className="flex gap-6 group">
                  {/* Timeline Dot */}
                  <div className="relative flex-shrink-0">
                    <div className="w-12 h-12 rounded-full bg-neon-gradient flex items-center justify-center shadow-lg shadow-neon-blue/50 group-hover:scale-110 transition-transform">
                      <div className="w-4 h-4 rounded-full bg-background" />
                    </div>
                  </div>

                  {/* Content Card */}
                  <div className="flex-1 bg-muted/30 border border-neon-blue/20 rounded-xl p-6 hover:border-neon-blue/40 transition-all">
                    <div className="flex items-start justify-between mb-4">
                      <div className="flex-1">
                        <div className="flex items-center gap-4 mb-2">
                          <p className="text-sm text-muted-foreground">
                            {new Date(conv.created_at).toLocaleString()}
                          </p>
                          <span className="text-xs font-mono text-accent bg-accent/10 px-2 py-1 rounded">
                            Turn {conv.turn_number}
                          </span>
                          <span className="text-xs font-mono text-neon-blue bg-neon-blue/10 px-2 py-1 rounded">
                            Session: {conv.session_id.slice(-8)}
                          </span>
                        </div>

                        <div className="space-y-3">
                          <div>
                            <p className="text-xs text-muted-foreground mb-1">User said:</p>
                            <p className="text-foreground text-sm">{conv.user_text}</p>
                          </div>

                          <div>
                            <p className="text-xs text-muted-foreground mb-1">AI responded:</p>
                            <p className="text-foreground text-sm">{conv.ai_text}</p>
                          </div>
                        </div>
                      </div>
                    </div>

                    <div className="flex gap-2 flex-wrap">
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={() => playAudio(conv.user_audio_path)}
                        className="border-neon-blue/50 hover:bg-neon-blue/10"
                      >
                        <Play className="w-4 h-4 mr-2" />
                        User Audio
                      </Button>

                      <Button
                        size="sm"
                        variant="outline"
                        onClick={() => playAudio(conv.ai_audio_fast_path)}
                        className="border-neon-pink/50 hover:bg-neon-pink/10"
                      >
                        <Play className="w-4 h-4 mr-2" />
                        AI Audio
                      </Button>

                      {conv.ai_audio_hq_path && (
                        <Button
                          size="sm"
                          variant="outline"
                          onClick={() => playAudio(conv.ai_audio_hq_path!)}
                          className="border-accent/50 hover:bg-accent/10"
                        >
                          <Play className="w-4 h-4 mr-2" />
                          HQ Audio
                        </Button>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>

          {/* Pagination */}
          {conversations.length > 0 && (
            <div className="flex items-center justify-between mt-8">
              <p className="text-sm text-muted-foreground">
                Showing {currentPage * pageSize + 1} to {Math.min((currentPage + 1) * pageSize, totalConversations)} of {totalConversations} conversations
              </p>

              <div className="flex gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setCurrentPage(Math.max(0, currentPage - 1))}
                  disabled={currentPage === 0 || isLoading}
                >
                  Previous
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setCurrentPage(currentPage + 1)}
                  disabled={conversations.length < pageSize || isLoading}
                >
                  Next
                </Button>
              </div>
            </div>
          )}
          </div>
        </Card>
      </div>
    </section>
  );
};

export default ConversationHistory;
