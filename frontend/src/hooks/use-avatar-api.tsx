/**
 * AVATAR API Hooks
 *
 * React hooks for AVATAR REST API operations
 * Uses TanStack Query for caching, loading states, and error handling
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { avatarAPI } from '@/lib/api-client';
import type { VoiceProfile, Conversation, SystemHealth, VRAMStatus } from '@/lib/api-config';

// Query keys for cache management
export const QUERY_KEYS = {
  VOICE_PROFILES: ['voice-profiles'] as const,
  CONVERSATIONS: ['conversations'] as const,
  SYSTEM_HEALTH: ['system-health'] as const,
  VRAM_STATUS: ['vram-status'] as const,
  SESSION_STATUS: ['session-status'] as const
};

// Voice Profile hooks
export const useVoiceProfiles = () => {
  return useQuery({
    queryKey: QUERY_KEYS.VOICE_PROFILES,
    queryFn: () => avatarAPI.getVoiceProfiles(),
    refetchInterval: 30000, // Refresh every 30 seconds
    staleTime: 10000 // Consider stale after 10 seconds
  });
};

export const useUploadVoiceProfile = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ name, audioFile }: { name: string; audioFile: File }) =>
      avatarAPI.uploadVoiceProfile(name, audioFile),
    onSuccess: () => {
      // Invalidate voice profiles query to refresh the list
      queryClient.invalidateQueries({ queryKey: QUERY_KEYS.VOICE_PROFILES });
    },
    onError: (error) => {
      console.error('Voice profile upload failed:', error);
    }
  });
};

export const useDeleteVoiceProfile = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (profileId: number) => avatarAPI.deleteVoiceProfile(profileId),
    onSuccess: () => {
      // Invalidate voice profiles query to refresh the list
      queryClient.invalidateQueries({ queryKey: QUERY_KEYS.VOICE_PROFILES });
    },
    onError: (error) => {
      console.error('Voice profile deletion failed:', error);
    }
  });
};

export const useTestVoiceProfileSynthesis = () => {
  return useMutation({
    mutationFn: ({ profileId, text }: { profileId: number; text: string }) =>
      avatarAPI.testVoiceProfileSynthesis(profileId, text),
    onError: (error) => {
      console.error('Voice synthesis test failed:', error);
    }
  });
};

// Conversation History hooks
export const useConversations = (limit: number = 50, offset: number = 0) => {
  return useQuery({
    queryKey: [...QUERY_KEYS.CONVERSATIONS, limit, offset],
    queryFn: () => avatarAPI.getConversations(limit, offset),
    refetchInterval: 60000, // Refresh every minute
    staleTime: 30000 // Consider stale after 30 seconds
  });
};

export const useSearchConversations = () => {
  return useMutation({
    mutationFn: (query: string) => avatarAPI.searchConversations(query),
    onError: (error) => {
      console.error('Conversation search failed:', error);
    }
  });
};

export const useExportConversations = () => {
  return useMutation({
    mutationFn: (format: 'json' | 'txt' = 'json') => avatarAPI.exportConversations(format),
    onSuccess: (blob, variables) => {
      // Auto-download the exported file
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `avatar-conversations.${variables}`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(url);
    },
    onError: (error) => {
      console.error('Conversation export failed:', error);
    }
  });
};

// System monitoring hooks
export const useSystemHealth = () => {
  return useQuery({
    queryKey: QUERY_KEYS.SYSTEM_HEALTH,
    queryFn: () => avatarAPI.getSystemHealth(),
    refetchInterval: 15000, // Refresh every 15 seconds for real-time monitoring
    staleTime: 5000 // Consider stale after 5 seconds
  });
};

export const useVRAMStatus = () => {
  return useQuery({
    queryKey: QUERY_KEYS.VRAM_STATUS,
    queryFn: () => avatarAPI.getVRAMStatus(),
    refetchInterval: 10000, // Refresh every 10 seconds
    staleTime: 5000
  });
};

export const useSessionStatus = () => {
  return useQuery({
    queryKey: QUERY_KEYS.SESSION_STATUS,
    queryFn: () => avatarAPI.getSessionStatus(),
    refetchInterval: 20000, // Refresh every 20 seconds
    staleTime: 10000
  });
};

// Utility hooks for system information
export const useSystemInfo = () => {
  return useQuery({
    queryKey: ['system-info'],
    queryFn: () => avatarAPI.getSystemInfo(),
    refetchInterval: 60000, // Refresh every minute
    staleTime: 30000
  });
};

// Combined hook for dashboard data
export const useDashboardData = () => {
  const health = useSystemHealth();
  const vram = useVRAMStatus();
  const sessions = useSessionStatus();
  const profiles = useVoiceProfiles();

  return {
    health: health.data,
    vram: vram.data,
    sessions: sessions.data,
    voiceProfiles: profiles.data,
    isLoading: health.isLoading || vram.isLoading || sessions.isLoading || profiles.isLoading,
    error: health.error || vram.error || sessions.error || profiles.error,
    refetch: () => {
      health.refetch();
      vram.refetch();
      sessions.refetch();
      profiles.refetch();
    }
  };
};

// Error handling utilities
export const useAPIErrorHandler = () => {
  const handleError = (error: any) => {
    console.error('API Error:', error);

    // You can add toast notifications, error reporting, etc. here
    if (error.code === 'NETWORK_ERROR') {
      console.warn('Network error - check connection to AVATAR backend');
    } else if (error.code === 'HTTP_401') {
      console.warn('Authentication required');
    } else if (error.code === 'HTTP_429') {
      console.warn('Rate limit exceeded');
    }

    return error;
  };

  return { handleError };
};