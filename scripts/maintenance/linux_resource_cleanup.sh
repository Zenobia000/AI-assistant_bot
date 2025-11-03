#!/bin/bash
# Linux 系統資源釋放腳本
# 用途：清理暫存檔案、釋放記憶體快取、優化系統資源

set -e

echo "=========================================="
echo "Linux 系統資源釋放工具"
echo "=========================================="
echo ""

# 1. 顯示當前資源狀態
echo "【1】當前系統資源狀態："
echo "---"
free -h
echo ""
df -h / | tail -1
echo ""
echo "---"
echo ""

# 2. 清理 Python 快取（專案相關）
echo "【2】清理 Python 快取檔案..."
PYTHON_CACHE=$(find . -type d -name "__pycache__" 2>/dev/null | wc -l)
PYTHON_BYTECODE=$(find . -name "*.pyc" 2>/dev/null | wc -l)
find . -type d -name "__pycache__" -exec rm -r {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true
find . -name "*.pyo" -delete 2>/dev/null || true
echo "   ✓ 已清理 $PYTHON_CACHE 個 __pycache__ 目錄"
echo "   ✓ 已清理 $PYTHON_BYTECODE 個 .pyc 檔案"
echo ""

# 3. 清理系統暫存檔案
echo "【3】清理系統暫存檔案..."
# 清理 /tmp（保留最近 7 天的檔案）
TMP_SIZE_BEFORE=$(du -sh /tmp 2>/dev/null | cut -f1 || echo "0")
find /tmp -type f -atime +7 -delete 2>/dev/null || true
find /tmp -type d -empty -delete 2>/dev/null || true
echo "   ✓ 已清理 /tmp 目錄（保留最近 7 天）"
echo ""

# 4. 清理用戶暫存
echo "【4】清理用戶暫存目錄..."
USER_TMP_SIZE_BEFORE=$(du -sh ~/.cache 2>/dev/null | cut -f1 || echo "0")
# 清理 pip 快取
pip cache purge 2>/dev/null || true
# 清理 poetry 快取
poetry cache clear pypi --all 2>/dev/null || true
echo "   ✓ 已清理 pip 和 poetry 快取"
echo ""

# 5. 清理系統日誌（可選，需要 sudo）
if [ "$EUID" -eq 0 ]; then
    echo "【5】清理系統日誌（需要 sudo 權限）..."
    journalctl --vacuum-time=7d 2>/dev/null || true
    echo "   ✓ 已清理 7 天前的系統日誌"
else
    echo "【5】跳過系統日誌清理（需要 sudo 權限）"
fi
echo ""

# 6. 清理 Docker（如果已安裝）
if command -v docker &> /dev/null; then
    echo "【6】清理 Docker 未使用的資源..."
    docker system prune -f 2>/dev/null || true
    echo "   ✓ 已清理 Docker 未使用的容器、網路和映像"
    echo ""
fi

# 7. 釋放記憶體快取（可選，需要 sudo）
echo "【7】釋放記憶體快取..."
if [ "$EUID" -eq 0 ]; then
    # 釋放 page cache, dentries 和 inodes
    sync
    echo 3 > /proc/sys/vm/drop_caches
    echo "   ✓ 已釋放系統記憶體快取"
else
    echo "   ⚠ 需要 sudo 權限才能釋放系統快取"
    echo "   可手動執行: sudo sh -c 'sync && echo 3 > /proc/sys/vm/drop_caches'"
fi
echo ""

# 8. 顯示清理後資源狀態
echo "【8】清理後系統資源狀態："
echo "---"
free -h
echo ""
df -h / | tail -1
echo ""
echo "---"
echo ""

echo "=========================================="
echo "資源釋放完成！"
echo "=========================================="

