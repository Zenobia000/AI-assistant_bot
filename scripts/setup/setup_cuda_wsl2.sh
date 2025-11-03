#!/bin/bash
# AVATAR - CUDA Toolkit Installation for WSL2
# 此腳本需在 WSL2 中執行

set -e  # 遇到錯誤立即退出

echo "=========================================="
echo "  AVATAR WSL2 CUDA Setup"
echo "=========================================="
echo ""

# 步驟 1: 檢查系統信息
echo "[1/6] Checking system information..."
cat /etc/os-release | grep -E "^(NAME|VERSION_ID)=" || true
uname -a

# 步驟 2: 檢查 GPU 驅動
echo ""
echo "[2/6] Checking GPU driver..."
if nvidia-smi &> /dev/null; then
    echo "✅ GPU driver is available"
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
else
    echo "❌ GPU driver not available"
    exit 1
fi

# 步驟 3: 安裝 CUDA Toolkit
echo ""
echo "[3/6] Installing CUDA Toolkit..."

# 檢查是否已安裝
if command -v nvcc &> /dev/null; then
    echo "⚠️  nvcc already installed: $(nvcc --version | grep release)"
    read -p "Reinstall? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Skipping CUDA installation"
    else
        INSTALL_CUDA=true
    fi
else
    echo "📦 Installing CUDA Toolkit 12.5..."
    INSTALL_CUDA=true
fi

if [ "$INSTALL_CUDA" = true ]; then
    # 下載並安裝 CUDA keyring
    wget -q https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
    sudo dpkg -i cuda-keyring_1.1-1_all.deb
    rm cuda-keyring_1.1-1_all.deb

    # 更新並安裝 CUDA Toolkit
    sudo apt-get update
    sudo apt-get install -y cuda-toolkit-12-5

    echo "✅ CUDA Toolkit installed"
fi

# 步驟 4: 配置環境變量
echo ""
echo "[4/6] Configuring environment variables..."

# 檢查 .bashrc 中是否已有 CUDA 配置
if grep -q "CUDA_HOME" ~/.bashrc; then
    echo "⚠️  CUDA_HOME already configured in .bashrc"
else
    echo "📝 Adding CUDA configuration to .bashrc..."
    cat >> ~/.bashrc << 'EOF'

# CUDA Configuration for Flash-Attention
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
EOF
    echo "✅ Environment variables configured"
fi

# 臨時設置環境變量（當前 session）
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# 步驟 5: 驗證 CUDA 安裝
echo ""
echo "[5/6] Verifying CUDA installation..."
if command -v nvcc &> /dev/null; then
    echo "✅ nvcc found: $(which nvcc)"
    nvcc --version | grep "release"
else
    echo "❌ nvcc not found after installation"
    echo "   Try: source ~/.bashrc"
    exit 1
fi

# 步驟 6: 檢查 GCC 編譯器
echo ""
echo "[6/6] Checking GCC compiler..."
if command -v gcc &> /dev/null; then
    echo "✅ gcc found: $(gcc --version | head -1)"
else
    echo "⚠️  gcc not found, installing build-essential..."
    sudo apt-get install -y build-essential
fi

# 完成
echo ""
echo "=========================================="
echo "  ✅ CUDA Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Reload environment: source ~/.bashrc"
echo "  2. Verify: nvcc --version"
echo "  3. Install Flash-Attention:"
echo "     cd /mnt/d/python_workspace/python-sideproject/AI-related/AI-assistant_bot"
echo "     poetry shell"
echo "     MAX_JOBS=4 pip install flash-attn --no-build-isolation --no-cache-dir"
echo ""
