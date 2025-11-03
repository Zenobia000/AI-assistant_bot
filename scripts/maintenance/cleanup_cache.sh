#!/bin/bash
# 針對性快取清理腳本 - 釋放 .cache 目錄空間
# 專注於安全清理 pip, poetry, huggingface 快取

set -e

echo "=========================================="
echo "快取清理工具 - 釋放磁碟空間"
echo "=========================================="
echo ""

# 顯示當前 .cache 佔用情況
echo "【當前 .cache 目錄大小】"
du -sh ~/.cache
echo ""
echo "【各子目錄大小】"
du -sh ~/.cache/* 2>/dev/null | sort -hr | head -10
echo ""
echo "---"
echo ""

# 選項 1: 清理 pip 快取（23GB）
echo "【1】清理 pip 快取..."
if [ -d ~/.cache/pip ]; then
    PIP_SIZE_BEFORE=$(du -sh ~/.cache/pip | cut -f1)
    pip cache purge 2>/dev/null || true
    echo "   ✓ 已清理 pip 快取（之前：$PIP_SIZE_BEFORE）"
else
    echo "   ⚠ pip 快取目錄不存在"
fi
echo ""

# 選項 2: 清理 Poetry 快取（16GB）
echo "【2】清理 Poetry 快取..."
if [ -d ~/.cache/pypoetry ]; then
    POETRY_SIZE_BEFORE=$(du -sh ~/.cache/pypoetry | cut -f1)
    poetry cache clear pypi --all 2>/dev/null || true
    # 也可以清理整個目錄（更徹底）
    # rm -rf ~/.cache/pypoetry/cache/repositories/pypi 2>/dev/null || true
    echo "   ✓ 已清理 Poetry 快取（之前：$POETRY_SIZE_BEFORE）"
else
    echo "   ⚠ Poetry 快取目錄不存在"
fi
echo ""

# 選項 3: 清理 HuggingFace 快取（61GB）- 需要互動確認
echo "【3】HuggingFace 模型快取（61GB）..."
echo "   發現以下模型："
du -sh ~/.cache/huggingface/hub/models--* 2>/dev/null | sort -hr | head -10
echo ""
echo "   選項："
echo "   a) 清理所有 HuggingFace 快取（會刪除所有已下載的模型）"
echo "   b) 清理 30 天未使用的模型快取"
echo "   c) 跳過（保留所有模型）"
echo ""
read -p "   請選擇 [a/b/c] (預設: c): " hf_choice
hf_choice=${hf_choice:-c}

case $hf_choice in
    a)
        echo "   ⚠ 即將刪除所有 HuggingFace 快取..."
        read -p "   確認刪除？輸入 'yes' 繼續: " confirm
        if [ "$confirm" = "yes" ]; then
            HF_SIZE_BEFORE=$(du -sh ~/.cache/huggingface | cut -f1)
            rm -rf ~/.cache/huggingface/*
            echo "   ✓ 已清理所有 HuggingFace 快取（之前：$HF_SIZE_BEFORE）"
        else
            echo "   ⊗ 已取消"
        fi
        ;;
    b)
        echo "   清理 30 天未使用的模型..."
        HF_SIZE_BEFORE=$(du -sh ~/.cache/huggingface | cut -f1)
        find ~/.cache/huggingface -type f -atime +30 -delete 2>/dev/null || true
        find ~/.cache/huggingface -type d -empty -delete 2>/dev/null || true
        echo "   ✓ 已清理 30 天未使用的 HuggingFace 快取（之前：$HF_SIZE_BEFORE）"
        ;;
    c)
        echo "   ⊗ 跳過 HuggingFace 快取清理"
        ;;
    *)
        echo "   ⊗ 無效選擇，跳過"
        ;;
esac
echo ""

# 選項 4: 清理其他小型快取
echo "【4】清理其他快取..."
# Playwright
if [ -d ~/.cache/ms-playwright ]; then
    rm -rf ~/.cache/ms-playwright/* 2>/dev/null || true
    echo "   ✓ 已清理 Playwright 快取"
fi

# Node.js
if [ -d ~/.cache/node-gyp ]; then
    rm -rf ~/.cache/node-gyp/* 2>/dev/null || true
    echo "   ✓ 已清理 node-gyp 快取"
fi
echo ""

# 顯示清理後結果
echo "【清理後 .cache 目錄大小】"
du -sh ~/.cache
echo ""
echo "【各子目錄大小】"
du -sh ~/.cache/* 2>/dev/null | sort -hr | head -10
echo ""

# 計算釋放空間
echo "=========================================="
echo "快取清理完成！"
echo "=========================================="

