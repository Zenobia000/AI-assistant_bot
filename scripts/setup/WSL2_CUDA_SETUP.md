# WSL2 CUDA Toolkit 安裝指南

本指南用於在 WSL2 中安裝 CUDA Toolkit，以支持 Flash-Attention 編譯。

---

## 🎯 安裝目標

- ✅ 安裝 CUDA Toolkit 12.5 for WSL2
- ✅ 配置 CUDA 環境變量（CUDA_HOME, PATH, LD_LIBRARY_PATH）
- ✅ 驗證 nvcc 編譯器可用
- ✅ 確保 GCC 編譯器已安裝

---

## ⚡ 快速安裝（推薦）

### 方法 A：使用 PowerShell 一鍵安裝

1. **以管理員身份**打開 PowerShell
2. 執行自動化腳本：

```powershell
cd D:\python_workspace\python-sideproject\AI-related\AI-assistant_bot
.\scripts\setup_cuda_wsl2.ps1
```

3. 等待安裝完成（約 10-30 分鐘）

---

### 方法 B：在 WSL2 中手動執行

1. 打開 WSL2 終端
2. 導航到專案目錄：

```bash
cd /mnt/d/python_workspace/python-sideproject/AI-related/AI-assistant_bot
```

3. 執行安裝腳本：

```bash
chmod +x scripts/setup_cuda_wsl2.sh
bash scripts/setup_cuda_wsl2.sh
```

4. 重新加載環境變量：

```bash
source ~/.bashrc
```

---

## 📋 安裝步驟詳解

如果自動化腳本失敗，可以按以下步驟手動安裝：

### 步驟 1：檢查系統信息

```bash
# 檢查 Ubuntu 版本
cat /etc/os-release

# 檢查 GPU 驅動
nvidia-smi
```

**預期輸出**：
- Ubuntu 20.04/22.04
- nvidia-smi 顯示 RTX 3090 和 CUDA Version 12.7

---

### 步驟 2：安裝 CUDA Toolkit

```bash
# 下載 CUDA keyring
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb

# 安裝 keyring
sudo dpkg -i cuda-keyring_1.1-1_all.deb

# 更新套件列表
sudo apt-get update

# 安裝 CUDA Toolkit 12.5
sudo apt-get install -y cuda-toolkit-12-5

# 清理下載檔案
rm cuda-keyring_1.1-1_all.deb
```

**安裝時間**：約 10-20 分鐘（取決於網速）

---

### 步驟 3：配置環境變量

```bash
# 編輯 .bashrc
nano ~/.bashrc

# 在文件末尾添加以下內容：
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# 保存並退出（Ctrl+X, Y, Enter）

# 重新加載配置
source ~/.bashrc
```

---

### 步驟 4：驗證 CUDA 安裝

```bash
# 檢查 nvcc 版本
nvcc --version

# 預期輸出示例：
# Cuda compilation tools, release 12.5, V12.5.xx
```

---

### 步驟 5：安裝 GCC 編譯器（如需要）

```bash
# 檢查 gcc 是否已安裝
gcc --version

# 如果未安裝，執行：
sudo apt-get install -y build-essential
```

---

## 🔧 安裝 Flash-Attention

CUDA Toolkit 安裝完成後，執行以下步驟：

### 1. 打開新的 WSL2 終端（確保環境變量已加載）

### 2. 導航到專案目錄

```bash
cd /mnt/d/python_workspace/python-sideproject/AI-related/AI-assistant_bot
```

### 3. 激活 Poetry 虛擬環境

```bash
# 方法 A：使用 poetry shell（推薦）
poetry shell

# 方法 B：手動激活
source $(poetry env info --path)/bin/activate
```

### 4. 驗證環境

```bash
# 檢查 Python 版本（應該是 3.11.x）
python --version

# 檢查虛擬環境路徑
echo $VIRTUAL_ENV

# 檢查 CUDA 環境
echo $CUDA_HOME
nvcc --version
```

### 5. 編譯安裝 Flash-Attention

```bash
MAX_JOBS=4 pip install flash-attn --no-build-isolation --no-cache-dir
```

**編譯時間**：約 15-30 分鐘（取決於 CPU 和記憶體）

**說明**：
- `MAX_JOBS=4`：限制並行編譯數為 4（32GB RAM 推薦值）
- `--no-build-isolation`：使用當前環境的 CUDA
- `--no-cache-dir`：不緩存編譯產物

---

## ⚠️ 常見問題

### Q1: `nvcc not found` 錯誤

**原因**：環境變量未加載

**解決**：
```bash
source ~/.bashrc
echo $CUDA_HOME  # 應該顯示 /usr/local/cuda
```

---

### Q2: 編譯時記憶體不足

**症狀**：編譯過程中系統卡死或出現 `killed`

**解決**：降低並行數
```bash
MAX_JOBS=2 pip install flash-attn --no-build-isolation --no-cache-dir
# 或
MAX_JOBS=1 pip install flash-attn --no-build-isolation --no-cache-dir
```

---

### Q3: GCC 版本不兼容

**症狀**：`unsupported GNU version`

**解決**：
```bash
# 檢查 GCC 版本
gcc --version

# CUDA 12.5 要求 GCC 9-12
# 如果版本不對，安裝對應版本：
sudo apt-get install -y gcc-11 g++-11
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 60
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 60
```

---

### Q4: Poetry 環境在 Windows 而非 WSL2

**症狀**：`poetry env info` 顯示 Windows 路徑

**解決**：在 WSL2 中重新初始化 Poetry 環境
```bash
cd /mnt/d/python_workspace/python-sideproject/AI-related/AI-assistant_bot

# 配置 Poetry 使用項目內虛擬環境
poetry config virtualenvs.in-project true

# 重新安裝依賴（會在 WSL2 中創建 .venv）
poetry install --no-root

# 檢查新環境
poetry env info
```

---

## 📊 驗證完整環境

安裝完成後，執行驗證腳本：

```bash
cd /mnt/d/python_workspace/python-sideproject/AI-related/AI-assistant_bot
poetry run python scripts/validate_setup.py
```

**預期輸出**：
```
[CHECK] AVATAR Environment Validation
...
[OK]   PyTorch             2.x.x+cu121
[OK]   vLLM                0.x.x
[OK]   faster-whisper      1.x.x

[INFO] PyTorch Configuration:
   CUDA Available: True
   Device Name: NVIDIA GeForce RTX 3090
   Compute Capability: 8.6

[OK] Environment validation passed!
```

---

## 🔗 參考資源

- [NVIDIA CUDA WSL2 官方文檔](https://docs.nvidia.com/cuda/wsl-user-guide/)
- [Flash-Attention GitHub](https://github.com/Dao-AILab/flash-attention)
- [Poetry 官方文檔](https://python-poetry.org/)

---

**最後更新**: 2025-11-01
**CUDA 版本**: 12.5
**WSL2 內核**: 6.6.87.2-microsoft-standard-WSL2
**目標 GPU**: NVIDIA RTX 3090
