#!/bin/bash
# 快速清理腳本 - 無互動，安全清理 pip 和 poetry 快取
# 預期可釋放：23GB (pip) + 16GB (poetry) = 39GB

set -e

echo "=========================================="
echo "快速清理 - 釋放 pip 和 poetry 快取"
echo "=========================================="
echo ""

# 顯示清理前狀態
echo "【清理前 .cache 大小】"
du -sh ~/.cache/pip ~/.cache/pypoetry 2>/dev/null | sort -hr
echo ""

# 清理 pip 快取
echo "【清理 pip 快取】"
if command -v pip &> /dev/null; then
    source /home/os-sunnie.gd.weng/python_workstation/side-project/AI-assistant_bot/.venv/bin/activate 2>/dev/null || true
    pip cache purge 2>/dev/null || true
    echo "   ✓ pip 快取已清理"
else
    echo "   ⚠ pip 未安裝或不在 PATH"
fi
echo ""

# 清理 Poetry 快取
echo "【清理 Poetry 快取】"
if command -v poetry &> /dev/null; then
    poetry cache clear pypi --all 2>/dev/null || true
    echo "   ✓ Poetry 快取已清理"
else
    echo "   ⚠ Poetry 未安裝或不在 PATH"
fi
echo ""

# 顯示清理後狀態
echo "【清理後 .cache 大小】"
du -sh ~/.cache/pip ~/.cache/pypoetry 2>/dev/null | sort -hr
echo ""

# 總計
echo "【總計 .cache 大小】"
du -sh ~/.cache
echo ""

echo "=========================================="
echo "快速清理完成！"
echo "=========================================="
echo ""
echo "💡 如需清理 HuggingFace 快取（61GB），請執行："
echo "   ./scripts/cleanup_cache.sh"
echo ""

