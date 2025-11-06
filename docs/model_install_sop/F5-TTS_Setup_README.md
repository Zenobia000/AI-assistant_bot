# F5-TTS 安裝與設定 SOP

## 概述
F5-TTS 是一個快速且高質量的文字轉語音（TTS）系統，支持聲音克隆功能。本文檔提供完整的安裝和使用指南。

## 系統需求
- Python 3.10 或更高版本
- CUDA 支持的 GPU（推薦，CPU 也可運行但較慢）
- 至少 8GB RAM
- 至少 5GB 硬碟空間（用於模型下載）

## 安裝步驟

### 1. 基本安裝
```bash
# 安裝 F5-TTS
pip install f5-tts

# 驗證安裝
python -c "from f5_tts.api import F5TTS; print('F5-TTS installed successfully!')"
```

### 2. 可能的依賴問題解決

如果遇到版本衝突警告，可以忽略或執行以下命令更新相關套件：

```bash
# 更新可能衝突的套件
pip install --upgrade pydantic numpy
```

### 3. CUDA 環境設定（GPU 加速）

確保 CUDA 環境正確設定：

```bash
# 檢查 CUDA 是否可用
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# 如果需要設定 cuDNN 路徑（如果遇到 cuDNN 錯誤）
export LD_LIBRARY_PATH="/home/$USER/.local/lib/python3.10/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH"

# 永久設定（加入到 ~/.bashrc）
echo 'export LD_LIBRARY_PATH="/home/$USER/.local/lib/python3.10/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH"' >> ~/.bashrc
source ~/.bashrc
```

## 使用方法

### 1. 準備參考音檔
- 格式：WAV 檔案
- 長度：3-10 秒為佳
- 品質：清晰、無背景噪音
- 採樣率：建議 22050Hz 或更高

### 2. 基本程式範例

```python
import os
import time
from f5_tts.api import F5TTS

# 設定檔案路徑
REFERENCE_AUDIO = "path/to/reference.wav"
REFERENCE_TEXT = "參考音檔對應的文字內容"
GENERATE_TEXT = "要合成的文字內容"
OUTPUT_PATH = "output.wav"

# 初始化模型
print("正在載入 F5-TTS 模型...")
f5tts = F5TTS()
print("模型載入完成!")

# 生成語音
print("正在生成語音...")
start_time = time.time()

f5tts.infer(
    ref_file=REFERENCE_AUDIO,
    ref_text=REFERENCE_TEXT,
    gen_text=GENERATE_TEXT,
    file_wave=OUTPUT_PATH,
    remove_silence=True
)

end_time = time.time()
print(f"語音生成完成! 耗時: {end_time - start_time:.2f} 秒")
print(f"輸出檔案: {OUTPUT_PATH}")
```

### 3. 多語言支援

F5-TTS 支援多種語言，包括：
- 英文
- 中文（繁體/簡體）
- 其他語言（依模型而定）

```python
# 英文生成
f5tts.infer(
    ref_file="reference.wav",
    ref_text="This is a reference audio.",
    gen_text="Hello world! This is English synthesis.",
    file_wave="output_english.wav"
)

# 中文生成
f5tts.infer(
    ref_file="reference.wav",
    ref_text="這是參考音檔。",
    gen_text="你好世界！這是中文合成。",
    file_wave="output_chinese.wav"
)
```

### 4. 參數說明

`f5tts.infer()` 方法的主要參數：

- `ref_file`: 參考音檔路徑
- `ref_text`: 參考音檔對應的文字
- `gen_text`: 要合成的文字
- `file_wave`: 輸出音檔路徑
- `remove_silence`: 是否移除靜音部分（默認 False）
- `speed`: 語速調整（默認 1.0）
- `cross_fade_duration`: 交叉淡化時間（默認 0.15）

## 常見問題解決

### 1. 模型下載問題
如果首次運行時模型下載失敗：
```bash
# 手動觸發模型下載
python -c "from f5_tts.api import F5TTS; F5TTS()"
```

### 2. CUDA 記憶體不足
```python
# 在程式開頭加入以下程式碼來清理 GPU 記憶體
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
```

### 3. 音檔格式問題
確保參考音檔格式正確：
```bash
# 使用 ffmpeg 轉換格式
ffmpeg -i input.mp3 -ar 22050 -ac 1 output.wav
```

### 4. 權限問題
如果遇到寫入權限問題：
```bash
# 確保輸出目錄有寫入權限
chmod 755 output_directory/
```

## 性能優化建議

### 1. GPU 加速
- 確保使用 CUDA 版本的 PyTorch
- 設定正確的 cuDNN 路徑
- 監控 GPU 記憶體使用情況

### 2. 批次處理
```python
# 對於多個文字，可以分批處理以提高效率
texts = ["文字1", "文字2", "文字3"]
for i, text in enumerate(texts):
    f5tts.infer(
        ref_file="reference.wav",
        ref_text="參考文字",
        gen_text=text,
        file_wave=f"output_{i}.wav"
    )
```

### 3. 記憶體管理
```python
# 處理大量音檔時定期清理記憶體
import gc
import torch

# 每處理幾個檔案後執行
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
```

## 範例專案結構
```
TTS_demo/
├── demos/
│   ├── reference_audio/
│   │   ├── boy.wav
│   │   ├── girl.wav
│   │   └── clone.wav
│   ├── output/
│   │   ├── f5_tts_output_boy.wav
│   │   └── f5_tts_output_chinese_boy.wav
│   └── tts_f5_tts_demo.py
├── F5-TTS_Setup_README.md
└── requirements.txt
```

## 相關資源

- [F5-TTS GitHub 專案](https://github.com/SWivid/F5-TTS)
- [F5-TTS 論文](https://arxiv.org/abs/2410.06226)
- [PyTorch CUDA 安裝指南](https://pytorch.org/get-started/locally/)

## 版本資訊
- F5-TTS: 1.1.9
- PyTorch: 2.8.0+
- Python: 3.10+
- 最後更新: 2025-11-03