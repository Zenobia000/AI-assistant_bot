# AVATAR Scripts Directory

> **組織原則**：Linus 式簡潔設計 - 按功能分類，消除複雜性

## 🎯 快速使用

```bash
# 主控制腳本 - 一個命令管理所有功能
./scripts/avatar-scripts help

# 常用操作
./scripts/avatar-scripts setup-env      # 環境設置
./scripts/avatar-scripts cleanup        # 清理快取
./scripts/avatar-scripts test-models    # 測試模型
```

## 📁 目錄結構

```
scripts/
├── avatar-scripts           # 🎮 主控制腳本
├── README.md               # 📖 本文件
├── setup/                  # ⚙️ 環境設置
│   ├── download_models.py   # 下載 AI 模型
│   ├── validate_setup.py    # 驗證環境
│   ├── init_database.py     # 初始化資料庫
│   ├── setup_cuda_wsl2.sh   # CUDA 設置 (Linux)
│   └── setup_cuda_wsl2.ps1  # CUDA 設置 (Windows)
├── maintenance/            # 🧹 系統維護
│   ├── cleanup_cache.sh     # 智能快取清理
│   ├── quick_cleanup.sh     # 快速清理
│   └── linux_resource_cleanup.sh # 深度資源清理
├── testing/               # 🧪 測試與驗證
│   ├── test_model_loading.py      # 模型載入測試
│   ├── generate_test_audio.py     # 音檔生成測試
│   ├── create_simple_test_audio.py # 簡單音檔測試
│   └── run_tests.sh              # 測試套件
└── development/           # 🛠️ 開發工具 (預留)
    └── (未來擴展)
```

## 📋 功能分類

### ⚙️ Setup (環境設置)
> **用途**：專案初始化、環境配置、依賴安裝

| 腳本 | 功能 | Phase |
|------|------|-------|
| `download_models.py` | 下載 Whisper, vLLM 模型 | Phase 1 |
| `validate_setup.py` | 驗證 Python 環境、CUDA、依賴 | Phase 1 |
| `init_database.py` | 建立 SQLite schema | Phase 1 |
| `setup_cuda_wsl2.sh` | WSL2 CUDA 環境設置 | Phase 1 |

### 🧹 Maintenance (系統維護)
> **用途**：清理快取、釋放空間、資源管理

| 腳本 | 功能 | 清理目標 |
|------|------|----------|
| `cleanup_cache.sh` | 針對性清理 pip/poetry/HF 快取 | ~100GB |
| `quick_cleanup.sh` | 快速清理臨時檔案 | ~5GB |
| `linux_resource_cleanup.sh` | 深度清理系統資源 | 全面 |

### 🧪 Testing (測試與驗證)
> **用途**：模型測試、功能驗證、E2E 測試

| 腳本 | 功能 | 測試範圍 |
|------|------|----------|
| `test_model_loading.py` | AI 模型載入、VRAM 監控 | GPU 資源 |
| `generate_test_audio.py` | 生成測試音檔 (gTTS) | STT 輸入 |
| `create_simple_test_audio.py` | 簡單音檔生成 | 基礎功能 |
| `run_tests.sh` | 完整測試套件 | 整合測試 |

### 🛠️ Development (開發工具)
> **用途**：開發輔助、調試工具、效能分析

```
development/
└── (預留目錄，未來可擴展)
    ├── profiling/          # 效能分析工具
    ├── debugging/          # 調試腳本
    └── automation/         # 自動化工具
```

## 🎮 主控制腳本使用

### 基本語法
```bash
./scripts/avatar-scripts [command]
```

### 可用命令

#### 🔧 Setup Commands
```bash
setup-env      # 下載模型並驗證環境
setup-db       # 初始化資料庫
setup-cuda     # 設置 CUDA (WSL2)
dev-validate   # 驗證開發環境
```

#### 🧹 Maintenance Commands
```bash
cleanup        # 智能快取清理 (互動式)
cleanup-quick  # 快速清理
cleanup-deep   # 深度資源清理
```

#### 🧪 Testing Commands
```bash
test-models    # 測試 AI 模型載入
test-audio     # 生成測試音檔
test-all       # 執行完整測試套件
```

## 🚀 常見工作流程

### 初次設置
```bash
./scripts/avatar-scripts setup-cuda     # 1. 設置 CUDA
./scripts/avatar-scripts setup-env      # 2. 下載模型
./scripts/avatar-scripts setup-db       # 3. 初始化資料庫
./scripts/avatar-scripts dev-validate   # 4. 驗證環境
```

### 開發過程
```bash
./scripts/avatar-scripts test-models    # 測試模型載入
./scripts/avatar-scripts cleanup        # 定期清理快取
```

### 故障排除
```bash
./scripts/avatar-scripts dev-validate   # 檢查環境
./scripts/avatar-scripts cleanup-deep   # 深度清理
./scripts/avatar-scripts test-all       # 全面測試
```

## 📝 Linus 式設計原則

1. **簡潔性**：一個命令完成複雜操作
2. **分類清晰**：功能導向的目錄結構
3. **無特殊情況**：統一的調用介面
4. **向後兼容**：保持原腳本路徑可用

## 🔗 整合資訊

- **TaskMaster WBS**：與 CLAUDE.md 中的 32 項任務對應
- **Phase 對應**：setup/ 對應 Phase 1，testing/ 跨越多個 Phase
- **VRAM 監控**：testing/ 中的腳本包含 GPU 資源檢查
- **自動化**：可整合到 CI/CD 流程中

---

**維護**：請保持此分類結構，新腳本按功能加入對應目錄
**原則**：Linus - "好品味是消除特殊情況，讓程式碼自然而清晰"