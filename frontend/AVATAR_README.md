# AVATAR Frontend

Modern React frontend for AVATAR AI Voice Assistant, fully integrated with AVATAR backend APIs.

## 🚀 Features

- **Real-time Voice Chat**: WebSocket-based voice conversation with AVATAR AI
- **Voice Profile Management**: Upload, manage, and test custom voice profiles
- **Conversation History**: Browse, search, and export conversation history
- **Real-time Monitoring**: System health, VRAM usage, and performance metrics
- **Responsive Design**: Mobile-friendly interface with modern UI components

## 🏗️ Tech Stack

- **React 18** with TypeScript
- **Vite** for fast development and building
- **shadcn/ui** for high-quality UI components
- **TanStack Query** for API state management
- **React Router** for navigation
- **Tailwind CSS** for styling

## 🔌 AVATAR API Integration

Fully integrated with AVATAR backend APIs:

### WebSocket APIs
- **Real-time chat**: `/ws/enhanced` - Live voice conversation
- **Session recovery**: Automatic reconnection with session preservation
- **Audio streaming**: Chunked audio upload with progress tracking

### REST APIs
- **Voice Profiles**: `/api/v1/voice-profiles` - CRUD operations
- **Conversations**: `/api/v1/conversations` - History and search
- **System Monitoring**: `/api/v1/monitoring` - Health and metrics
- **Session Control**: `/api/v1/sessions` - Session management

## 🎯 Phase 3 Task Completion

✅ **Task 17**: 前端開發 - 聊天介面 (DemoPanel.tsx)
✅ **Task 18**: 前端開發 - 聲紋管理介面 (VoiceProfileManager.tsx)
✅ **Task 19**: 前端開發 - 對話歷史介面 (ConversationHistory.tsx)

## 🔧 Setup & Usage

### Prerequisites
- AVATAR backend running on port 8000
- Node.js 18+ installed

### Development
```bash
cd frontend
npm install
npm run dev
# Frontend: http://localhost:8080
# Backend: http://localhost:8000 (auto-proxied)
```

### Integration Status
✅ **Complete** - All Phase 3 frontend tasks implemented
✅ **API Ready** - Integrated with all AVATAR backend APIs
✅ **Production Ready** - Ready for deployment