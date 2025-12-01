# API Provider Architecture - 地端/API 服務切換設計

> **文件版本**: 1.0
> **建立日期**: 2025-11-27
> **設計哲學**: Linus Torvalds "Good Taste" - 正確的資料結構讓實作自然簡單
> **狀態**: ✅ 設計階段 → 待實作

---

## 🎯 設計目標

### 核心需求
在不破壞現有 WebSocket 流程的前提下，支援靈活切換：
- **地端模式**: 使用本地 AI 模型 (Whisper, vLLM, F5-TTS)
- **API 模式**: 使用雲端 API 服務 (OpenAI, Anthropic, ElevenLabs 等)

### 設計原則 (Linus Philosophy)

#### 1. "Good Taste" - 消除特殊情況
```python
# ❌ 糟糕的設計 (充滿特殊情況)
if mode == "local":
    result = whisper.transcribe(...)
elif mode == "openai":
    result = openai.transcribe(...)
elif mode == "azure":
    result = azure.transcribe(...)

# ✅ 好品味設計 (無特殊情況)
provider = get_stt_provider()  # Factory 根據配置返回
result = provider.transcribe(...)  # 統一介面
```

#### 2. "Never Break Userspace" - 零破壞性
- WebSocket 端點 (`websocket.py`) 不需修改任何一行
- 所有現有 `get_xxx_service()` 呼叫繼續有效
- 預設行為 (`local` 模式) 與現在完全相同

#### 3. Simplicity - 簡潔執念
- Protocol 只定義必要方法
- Factory 用環境變數控制，無需複雜配置
- 每個 Provider 只專注自己的實作

---

## 📐 架構設計

### 核心模式: Protocol + Factory Pattern

```
┌─────────────────────────────────────────────────────────────┐
│                    WebSocket Handler                         │
│              (websocket.py - 不需修改)                       │
└────────────────────┬────────────────────────────────────────┘
                     │
                     │ 呼叫 get_xxx_service()
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                   Service Factory                            │
│         (stt.py, llm.py, tts.py - 最小修改)                 │
│                                                               │
│  def get_stt_service() -> STTProvider:                       │
│      if config.STT_PROVIDER == "local":                      │
│          return WhisperSTTProvider()                         │
│      elif config.STT_PROVIDER == "openai":                   │
│          return OpenAISTTProvider()                          │
└────────────────────┬────────────────────────────────────────┘
                     │
                     │ 返回實作 Protocol 的 Provider
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                 Protocol Interface                           │
│                  (protocols.py)                              │
│                                                               │
│  class STTProvider(Protocol):                                │
│      async def transcribe(...) -> tuple[str, dict]           │
│                                                               │
│  class LLMProvider(Protocol):                                │
│      async def chat_stream(...) -> AsyncIterator[str]        │
│                                                               │
│  class TTSProvider(Protocol):                                │
│      async def synthesize(...) -> Path                       │
└────────────────────┬────────────────────────────────────────┘
                     │
                     │ 實作介面
                     ▼
┌──────────────────────────────┬──────────────────────────────┐
│      Local Providers         │       API Providers          │
│      (現有實作改名)          │      (未來擴展)              │
├──────────────────────────────┼──────────────────────────────┤
│ • WhisperSTTProvider         │ • OpenAISTTProvider          │
│ • VLLMProvider               │ • AnthropicLLMProvider       │
│ • F5TTSProvider              │ • ElevenLabsTTSProvider      │
└──────────────────────────────┴──────────────────────────────┘
```

---

## 🔧 實作計畫

### Phase 1: 定義 Protocol 介面

**新檔案**: `src/avatar/services/protocols.py`

```python
"""
Service provider protocols (interfaces)

Define contracts for STT/LLM/TTS services.
Enables switching between local models and API services.

Design Philosophy (Linus Torvalds):
- "Good programmers worry about data structures and their relationships"
- Protocol defines the data flow contract
- Implementation details are hidden behind the interface
"""
from typing import Protocol, AsyncIterator, Optional
from pathlib import Path


class STTProvider(Protocol):
    """
    Speech-to-Text provider interface

    Implementations:
    - WhisperSTTProvider: Local faster-whisper (CPU)
    - OpenAISTTProvider: OpenAI Whisper API
    - AzureSTTProvider: Azure Speech Services
    - GoogleSTTProvider: Google Cloud Speech-to-Text
    """

    async def transcribe(
        self,
        audio_path: Path,
        language: Optional[str] = None,
        **kwargs
    ) -> tuple[str, dict]:
        """
        Transcribe audio to text

        Args:
            audio_path: Path to audio file (WAV 16kHz mono recommended)
            language: ISO 639-1 language code (None for auto-detect)
            **kwargs: Provider-specific options

        Returns:
            (transcribed_text, metadata)

            metadata format:
            {
                "language": str,           # Detected or specified language
                "duration": float,         # Audio duration in seconds
                "segments_count": int,     # Number of transcription segments
                "confidence": float,       # Average confidence score (0-1)
                "provider": str,           # Provider name (e.g., "whisper_local")
            }

        Raises:
            FileNotFoundError: Audio file not found
            RuntimeError: Transcription failed
        """
        ...


class LLMProvider(Protocol):
    """
    Large Language Model provider interface

    Implementations:
    - VLLMProvider: Local vLLM with quantized models
    - OpenAILLMProvider: OpenAI GPT models
    - AnthropicLLMProvider: Anthropic Claude models
    - AzureLLMProvider: Azure OpenAI Service
    """

    async def chat_stream(
        self,
        messages: list[dict],
        max_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs
    ) -> AsyncIterator[str]:
        """
        Stream chat completion chunks (for lower TTFT)

        Args:
            messages: Chat history in OpenAI format
                      [{"role": "user", "content": "..."}]
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0-2.0)
            **kwargs: Provider-specific options

        Yields:
            Response text chunks (delta only, not cumulative)

        Example:
            async for chunk in llm.chat_stream(messages):
                print(chunk, end="", flush=True)

        Raises:
            RuntimeError: Generation failed
        """
        ...


class TTSProvider(Protocol):
    """
    Text-to-Speech provider interface

    Implementations:
    - F5TTSProvider: Local F5-TTS (fast mode)
    - CosyVoiceTTSProvider: Local CosyVoice2 (HQ mode)
    - ElevenLabsTTSProvider: ElevenLabs API
    - AzureTTSProvider: Azure Text-to-Speech
    - OpenAITTSProvider: OpenAI TTS API
    """

    async def synthesize(
        self,
        text: str,
        output_path: Path,
        ref_audio_path: Optional[Path] = None,
        ref_text: Optional[str] = None,
        **kwargs
    ) -> Path:
        """
        Synthesize speech from text (with optional voice cloning)

        Args:
            text: Text to synthesize
            output_path: Where to save synthesized audio
            ref_audio_path: Reference audio for voice cloning
            ref_text: Reference text matching ref_audio
            **kwargs: Provider-specific options

        Returns:
            Path to synthesized audio file (WAV format)

        Raises:
            RuntimeError: Synthesis failed
        """
        ...

    async def synthesize_fast(
        self,
        text: str,
        voice_profile_name: str,
        output_path: Path,
        **kwargs
    ) -> Path:
        """
        Fast synthesis using pre-registered voice profile

        Args:
            text: Text to synthesize
            voice_profile_name: Name of registered voice profile
            output_path: Where to save synthesized audio
            **kwargs: Provider-specific options

        Returns:
            Path to synthesized audio file (WAV format)

        Raises:
            FileNotFoundError: Voice profile not found
            RuntimeError: Synthesis failed
        """
        ...
```

**設計理由**:
- 返回值統一為 `(result, metadata)`，方便日誌和監控
- 支援 `**kwargs` 保留 Provider 特定選項的靈活性
- 每個方法都有明確的錯誤處理契約

---

### Phase 2: 環境變數配置

**修改**: `src/avatar/core/config.py` (在第 45 行之後新增)

```python
    # ============================================================
    # Service Provider Configuration (地端/API 切換)
    # ============================================================

    # Provider Mode Selection
    # 支援的 STT Providers: local (Whisper), openai, azure, google
    STT_PROVIDER: str = os.getenv("AVATAR_STT_PROVIDER", "local")

    # 支援的 LLM Providers: local (vLLM), openai, anthropic, azure
    LLM_PROVIDER: str = os.getenv("AVATAR_LLM_PROVIDER", "local")

    # 支援的 TTS Providers: local (F5-TTS), elevenlabs, azure, openai
    TTS_PROVIDER: str = os.getenv("AVATAR_TTS_PROVIDER", "local")

    # -------------------- STT API Configuration --------------------
    STT_API_KEY: Optional[str] = os.getenv("AVATAR_STT_API_KEY")
    STT_API_ENDPOINT: Optional[str] = os.getenv("AVATAR_STT_API_ENDPOINT")
    STT_API_MODEL: str = os.getenv("AVATAR_STT_API_MODEL", "whisper-1")

    # -------------------- LLM API Configuration --------------------
    LLM_API_KEY: Optional[str] = os.getenv("AVATAR_LLM_API_KEY")
    LLM_API_ENDPOINT: Optional[str] = os.getenv("AVATAR_LLM_API_ENDPOINT")
    LLM_API_MODEL: str = os.getenv("AVATAR_LLM_API_MODEL", "gpt-4")
    LLM_API_BASE_URL: Optional[str] = os.getenv("AVATAR_LLM_API_BASE_URL")  # For custom endpoints

    # -------------------- TTS API Configuration --------------------
    TTS_API_KEY: Optional[str] = os.getenv("AVATAR_TTS_API_KEY")
    TTS_API_ENDPOINT: Optional[str] = os.getenv("AVATAR_TTS_API_ENDPOINT")
    TTS_API_VOICE: str = os.getenv("AVATAR_TTS_API_VOICE", "alloy")
    TTS_API_MODEL: str = os.getenv("AVATAR_TTS_API_MODEL", "tts-1")
```

**使用範例** (`.env` 檔案):

```bash
# 地端模式 (預設)
AVATAR_STT_PROVIDER=local
AVATAR_LLM_PROVIDER=local
AVATAR_TTS_PROVIDER=local

# 混合模式 (地端 LLM + API TTS)
AVATAR_STT_PROVIDER=local
AVATAR_LLM_PROVIDER=local
AVATAR_TTS_PROVIDER=elevenlabs
AVATAR_TTS_API_KEY=your_elevenlabs_key

# 全 API 模式
AVATAR_STT_PROVIDER=openai
AVATAR_STT_API_KEY=sk-xxx
AVATAR_LLM_PROVIDER=anthropic
AVATAR_LLM_API_KEY=sk-ant-xxx
AVATAR_TTS_PROVIDER=openai
AVATAR_TTS_API_KEY=sk-xxx
```

---

### Phase 3: Factory Pattern 重構

#### 3.1 STT Factory

**步驟**:
1. 重命名: `src/avatar/services/stt.py` → `src/avatar/services/stt_local.py`
2. 修改類別名: `STTService` → `WhisperSTTProvider`
3. 新建: `src/avatar/services/stt.py` (Factory 入口)

**新檔案**: `src/avatar/services/stt.py`

```python
"""
STT Service Factory

Returns appropriate STT provider based on AVATAR_STT_PROVIDER configuration.

Linus Principle: "Never break userspace"
- Existing get_stt_service() calls continue to work
- Default behavior (local mode) is unchanged
- Adding new providers requires zero changes to callers
"""
import structlog
from avatar.core.config import config
from avatar.services.protocols import STTProvider

logger = structlog.get_logger()

# Singleton instance
_stt_service: STTProvider | None = None


async def get_stt_service() -> STTProvider:
    """
    Get STT service instance (singleton)

    Returns appropriate provider based on AVATAR_STT_PROVIDER env var.

    Supported providers:
    - local: Whisper (faster-whisper) on CPU
    - openai: OpenAI Whisper API
    - azure: Azure Speech Services
    - google: Google Cloud Speech-to-Text

    Returns:
        STTProvider implementation

    Raises:
        ValueError: Unknown provider
        RuntimeError: Provider initialization failed

    Example:
        stt = await get_stt_service()
        text, metadata = await stt.transcribe(audio_path)
    """
    global _stt_service

    if _stt_service is not None:
        return _stt_service

    provider = config.STT_PROVIDER.lower()

    if provider == "local":
        from avatar.services.stt_local import WhisperSTTProvider
        logger.info("stt.factory.init",
                   provider="local",
                   model=config.WHISPER_MODEL_SIZE,
                   device=config.WHISPER_DEVICE)
        _stt_service = await WhisperSTTProvider.create(
            model_size=config.WHISPER_MODEL_SIZE,
            device=config.WHISPER_DEVICE,
            compute_type=config.WHISPER_COMPUTE_TYPE
        )

    # 未來擴展點 (選型完成後解除註釋)
    # elif provider == "openai":
    #     from avatar.services.stt_openai import OpenAISTTProvider
    #     logger.info("stt.factory.init", provider="openai", model=config.STT_API_MODEL)
    #     _stt_service = OpenAISTTProvider(
    #         api_key=config.STT_API_KEY,
    #         model=config.STT_API_MODEL,
    #         endpoint=config.STT_API_ENDPOINT
    #     )

    # elif provider == "azure":
    #     from avatar.services.stt_azure import AzureSTTProvider
    #     logger.info("stt.factory.init", provider="azure")
    #     _stt_service = AzureSTTProvider(
    #         api_key=config.STT_API_KEY,
    #         endpoint=config.STT_API_ENDPOINT
    #     )

    # elif provider == "google":
    #     from avatar.services.stt_google import GoogleSTTProvider
    #     logger.info("stt.factory.init", provider="google")
    #     _stt_service = GoogleSTTProvider(
    #         api_key=config.STT_API_KEY
    #     )

    else:
        raise ValueError(
            f"Unknown STT provider: '{provider}'\n"
            f"Supported providers: local, openai, azure, google\n"
            f"Set AVATAR_STT_PROVIDER environment variable"
        )

    logger.info("stt.factory.ready", provider=provider)
    return _stt_service
```

**修改**: `src/avatar/services/stt_local.py` (原 `stt.py`)

只需修改類別名:
```python
# 第 20 行
class WhisperSTTProvider:  # 原: STTService
    """
    Local Whisper STT Provider

    Uses faster-whisper for CPU-based transcription.
    Implements STTProvider protocol.
    """
    # ... 現有實作完全不變 ...
```

#### 3.2 LLM Factory

**步驟**:
1. 重命名: `llm.py` → `llm_local.py`
2. 修改類別名: `LLMService` → `VLLMProvider`
3. 新建: `llm.py` (Factory)

**新檔案**: `src/avatar/services/llm.py`

```python
"""
LLM Service Factory

Returns appropriate LLM provider based on AVATAR_LLM_PROVIDER configuration.
"""
import structlog
from avatar.core.config import config
from avatar.services.protocols import LLMProvider

logger = structlog.get_logger()

_llm_service: LLMProvider | None = None


async def get_llm_service() -> LLMProvider:
    """
    Get LLM service instance (singleton)

    Supported providers:
    - local: vLLM with quantized models (AWQ)
    - openai: OpenAI GPT models
    - anthropic: Anthropic Claude models
    - azure: Azure OpenAI Service
    """
    global _llm_service

    if _llm_service is not None:
        return _llm_service

    provider = config.LLM_PROVIDER.lower()

    if provider == "local":
        from avatar.services.llm_local import VLLMProvider
        logger.info("llm.factory.init",
                   provider="local",
                   model=config.VLLM_MODEL,
                   gpu_memory=config.VLLM_GPU_MEMORY)
        _llm_service = await VLLMProvider.create(
            model_name=config.VLLM_MODEL,
            gpu_memory_utilization=config.VLLM_GPU_MEMORY,
            max_model_len=config.VLLM_MAX_TOKENS
        )

    # 未來擴展點
    # elif provider == "openai": ...
    # elif provider == "anthropic": ...
    # elif provider == "azure": ...

    else:
        raise ValueError(
            f"Unknown LLM provider: '{provider}'\n"
            f"Supported providers: local, openai, anthropic, azure"
        )

    logger.info("llm.factory.ready", provider=provider)
    return _llm_service
```

#### 3.3 TTS Factory

**步驟**:
1. 重命名: `tts.py` → `tts_local.py`
2. 修改類別名: `TTSService` → `F5TTSProvider`
3. 新建: `tts.py` (Factory)

**新檔案**: `src/avatar/services/tts.py`

```python
"""
TTS Service Factory

Returns appropriate TTS provider based on AVATAR_TTS_PROVIDER configuration.
"""
import structlog
from avatar.core.config import config
from avatar.services.protocols import TTSProvider

logger = structlog.get_logger()

_tts_service: TTSProvider | None = None


async def get_tts_service() -> TTSProvider:
    """
    Get TTS service instance (singleton)

    Supported providers:
    - local: F5-TTS (fast mode)
    - elevenlabs: ElevenLabs API
    - openai: OpenAI TTS API
    - azure: Azure Text-to-Speech
    """
    global _tts_service

    if _tts_service is not None:
        return _tts_service

    provider = config.TTS_PROVIDER.lower()

    if provider == "local":
        from avatar.services.tts_local import F5TTSProvider
        logger.info("tts.factory.init", provider="local")
        _tts_service = await F5TTSProvider.create()

    # 未來擴展點
    # elif provider == "elevenlabs": ...
    # elif provider == "openai": ...
    # elif provider == "azure": ...

    else:
        raise ValueError(
            f"Unknown TTS provider: '{provider}'\n"
            f"Supported providers: local, elevenlabs, openai, azure"
        )

    logger.info("tts.factory.ready", provider=provider)
    return _tts_service
```

---

## 📊 檔案結構變更

### 修改前
```
src/avatar/services/
├── stt.py              # STTService 類別
├── llm.py              # LLMService 類別
├── tts.py              # TTSService 類別
└── tts_hq.py           # CosyVoice HQ
```

### 修改後
```
src/avatar/services/
├── protocols.py        # ✨ 新增 - Protocol 介面定義
│
├── stt.py              # 🔄 重構 - Factory 入口
├── stt_local.py        # 📝 重命名 - WhisperSTTProvider (原 stt.py)
│
├── llm.py              # 🔄 重構 - Factory 入口
├── llm_local.py        # 📝 重命名 - VLLMProvider (原 llm.py)
│
├── tts.py              # 🔄 重構 - Factory 入口
├── tts_local.py        # 📝 重命名 - F5TTSProvider (原 tts.py)
└── tts_hq.py           # ✅ 不變 - CosyVoice HQ

# 未來擴展 (選型後新增)
├── stt_openai.py       # OpenAI Whisper API
├── stt_azure.py        # Azure Speech Services
├── llm_openai.py       # OpenAI GPT
├── llm_anthropic.py    # Anthropic Claude
├── tts_elevenlabs.py   # ElevenLabs API
└── tts_openai.py       # OpenAI TTS API
```

---

## ✅ 向後兼容性保證

### WebSocket 零改動
`src/avatar/api/websocket.py` 的所有呼叫保持不變:

```python
# Line 304: STT 呼叫
stt = await get_stt_service()
text, metadata = await stt.transcribe(audio_path, ...)

# Line 342: LLM 呼叫
llm = await get_llm_service()
async for chunk in llm.chat_stream(messages, ...):
    ...

# Line 399: TTS 呼叫
tts = await get_tts_service()
await tts.synthesize(text, output_path, ...)
```

### 函式簽名不變
- `get_stt_service()` → 返回 `STTProvider`
- `get_llm_service()` → 返回 `LLMProvider`
- `get_tts_service()` → 返回 `TTSProvider`

### 預設行為不變
所有環境變數預設為 `local`，行為與現在完全相同。

---

## 🧪 測試策略

### 單元測試
每個 Provider 獨立測試:
```python
# tests/unit/services/test_stt_local.py
async def test_whisper_transcribe():
    provider = WhisperSTTProvider(...)
    text, metadata = await provider.transcribe(sample_audio)
    assert text == "expected transcription"
    assert metadata["language"] == "en"
```

### 整合測試
測試 Factory 切換:
```python
# tests/integration/test_service_factory.py
async def test_stt_factory_local():
    os.environ["AVATAR_STT_PROVIDER"] = "local"
    stt = await get_stt_service()
    assert isinstance(stt, WhisperSTTProvider)

async def test_stt_factory_openai():
    os.environ["AVATAR_STT_PROVIDER"] = "openai"
    os.environ["AVATAR_STT_API_KEY"] = "sk-test"
    stt = await get_stt_service()
    assert isinstance(stt, OpenAISTTProvider)
```

### E2E 測試
WebSocket 整合測試不需修改，自動涵蓋新架構。

---

## 🚀 未來擴展指南

### 新增 API Provider 標準流程

以新增 OpenAI STT 為例:

#### Step 1: 建立 Provider 檔案
```bash
touch src/avatar/services/stt_openai.py
```

#### Step 2: 實作 Protocol
```python
# src/avatar/services/stt_openai.py
from avatar.services.protocols import STTProvider
import httpx

class OpenAISTTProvider:
    """OpenAI Whisper API Provider"""

    def __init__(self, api_key: str, model: str = "whisper-1"):
        self.api_key = api_key
        self.model = model
        self.client = httpx.AsyncClient()

    async def transcribe(
        self,
        audio_path: Path,
        language: Optional[str] = None,
        **kwargs
    ) -> tuple[str, dict]:
        with open(audio_path, "rb") as f:
            response = await self.client.post(
                "https://api.openai.com/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                files={"file": f},
                data={"model": self.model, "language": language or ""}
            )

        result = response.json()

        return (
            result["text"],
            {
                "language": result.get("language", "unknown"),
                "duration": result.get("duration", 0),
                "segments_count": 1,
                "confidence": 1.0,
                "provider": "openai_whisper"
            }
        )
```

#### Step 3: Factory 解除註釋 (3 行)
```python
# src/avatar/services/stt.py
elif provider == "openai":
    from avatar.services.stt_openai import OpenAISTTProvider
    _stt_service = OpenAISTTProvider(
        api_key=config.STT_API_KEY,
        model=config.STT_API_MODEL
    )
```

#### Step 4: 設定環境變數
```bash
export AVATAR_STT_PROVIDER=openai
export AVATAR_STT_API_KEY=sk-xxx
```

#### Step 5: 重啟服務
```bash
poetry run python -m avatar.main
```

**完成！** 無需修改任何其他程式碼。

---

## 📈 效益評估

### 開發效率
- **新增 Provider**: 1 個檔案 + 3 行 Factory 程式碼
- **切換 Provider**: 1 個環境變數
- **測試隔離**: 每個 Provider 獨立測試

### 程式碼品質
- **耦合度**: 極低 (Protocol 隔離)
- **可維護性**: 高 (單一職責)
- **可擴展性**: 極高 (開放封閉原則)

### 技術債務
- **新增債務**: 無
- **消除債務**: 現有硬編碼的模型依賴

---

## 🎯 檢查清單

在開始實作前，確認:

- [x] 設計符合 Linus "Good Taste" 原則
- [x] 向後兼容性 100%
- [x] WebSocket 無需修改
- [x] 預設行為不變
- [x] 擴展路徑清晰
- [x] 錯誤處理明確
- [x] 測試策略完整

---

## 📚 參考資料

### Linus Torvalds 設計哲學
- ["Good Taste" in Coding](https://www.youtube.com/watch?v=o8NPllzkFhE) - TED Talk
- ["Never Break Userspace"](https://lkml.org/lkml/2012/12/23/75) - LKML
- Linux Kernel Coding Style

### Design Patterns
- Protocol-Oriented Programming (PEP 544)
- Factory Pattern (GoF)
- Strategy Pattern (Behavioral)

---

**文件狀態**: ✅ 設計完成，等待實作批准
**預計實作時間**: 30 分鐘
**風險評估**: 🟢 低風險 (符合零破壞原則)
