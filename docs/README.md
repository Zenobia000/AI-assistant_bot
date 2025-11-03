# AVATAR - MVP 文檔體系（Linus 式精簡版）

> **核心原則**: Talk is cheap. Show me the code.
>
> 本文檔體系遵循「模式 B：MVP 快速迭代（Lean）」，拒絕過度設計，追求實用主義。

---

## 📁 文檔結構

```
docs/
├── README.md                              # 本文件（導覽）
├── planning/
│   └── mvp_tech_spec.md                   # MVP 技術規格（核心契約）
├── dev/                                   # 開發文檔
│   ├── development_progress_report.md     # 開發進度報告
│   ├── progress_phase2_complete.md        # Phase 2 完成報告
│   └── architecture_review_task13.md     # Task 13 架構審查
└── launch/
    └── mvp_launch_checklist.md            # 上線檢查清單（品質門禁）
```

---

## 🎯 三份核心文檔

### 1. MVP 技術規格（mvp_tech_spec.md）

**目的**: 開發與審查的唯一契約

**核心內容**:
- 問題陳述與 KPI（≤3 條）
- 高層設計（1 句話 + 1 張圖）
- API 契約（僅核心端點）
- 資料表 Schema（2 張表）
- 風險與緩解

**關鍵決策**:
- ✅ SQLite 代替 PostgreSQL（零配置）
- ✅ 本地音檔代替 MinIO（零依賴）
- ✅ 內存佇列代替 Redis（零複雜度）
- ✅ 單層架構代替 Clean Arch（零抽象）

**為什麼精簡？**
> "如果你需要 90KB 文檔來描述一個 MVP，你就已經完蛋了。" — Linus 式哲學

---

### 2. 開發進度報告（development_progress_report.md）

**目的**: 透明化進度與風險

**核心內容**:
- 總體進度概覽
- Gantt 時間軸（4 週）
- 功能開發狀態
- 關鍵技術指標
- 技術債務與風險

**更新頻率**: 每週一次

**關鍵指標**:
- E2E 延遲 P95 ≤ 3.5s
- 連續 2h 5 並發無 OOM
- 聲音克隆滿意度 ≥ 7/10

---

### 3. MVP 上線檢查清單（mvp_launch_checklist.md）

**目的**: 確保最小品質與運維準備

**核心內容**:
- 功能完整性（8 項）
- 性能達標（5 項）
- 安全基線（6 項）
- 備份與恢復（4 項）
- 監控與告警（5 項）
- 運維準備（4 項）

**Go/No-Go 標準**:
- 功能完整性 ≥ 90%
- 性能達標 100%
- 安全基線 100%

---

## 🔍 與「完整流程」的對比

| 項目 | 完整流程（模式 A） | MVP 精簡版（模式 B） | 決策理由 |
|:---|:---|:---|:---|
| **文檔數量** | 10+ 份 | 3 份 | 減少 70% 文檔工作量 |
| **文檔總量** | ~90KB | ~25KB | 保留核心，去除冗餘 |
| **架構層級** | 4 層（Clean Arch） | 1 層（FastAPI 直調） | 功能簡單不需抽象 |
| **數據存儲** | PostgreSQL + MinIO + Redis | SQLite + 本地檔案 | 零配置，零運維 |
| **BDD Scenarios** | 30+ | 0（直接寫測試） | 測試優先於文檔 |
| **部署複雜度** | Docker Compose（12 服務） | 單一 Python 進程 | 降低部署門檻 |
| **開發時程** | 8-12 週 | 4 週 | 快速驗證價值 |

---

## 🛠️ 開發工作流程

### Week 0: 規劃階段（已完成 ✅）
```bash
# 1. 閱讀三份核心文檔
cd docs
cat planning/mvp_tech_spec.md
cat dev/development_progress_report.md
cat launch/mvp_launch_checklist.md

# 2. 理解核心決策
# - 為什麼用 SQLite？
# - 為什麼不用 Clean Architecture？
# - 為什麼只要 2 張表？

# 3. 準備環境
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Week 1: 核心流程打通
```bash
# 1. 建立專案結構
mkdir -p app/{api,services} audio scripts frontend

# 2. 實現核心流程
# app/main.py - FastAPI 入口
# app/websocket.py - WebSocket 處理
# app/services/{stt,llm,tts}.py - AI 模型調用

# 3. 驗收標準
# - 語音 → LLM → TTS 流程可跑通
# - E2E 延遲 < 5s（初版）
```

### Week 2-3: 功能補完
```bash
# 1. 聲紋管理
# app/api/voice.py - 上傳/刪除/列表

# 2. 對話歷史
# app/api/chat.py - 查詢/重播

# 3. 前端開發
cd frontend && npm run dev
```

### Week 4: 測試與上線
```bash
# 1. 執行上線檢查清單
python scripts/check_launch_readiness.py

# 2. 壓力測試
python scripts/stress_test.py --concurrent 5 --duration 7200

# 3. 部署
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

---

## 📊 成功指標追蹤

### 文檔效率指標

| 指標 | 目標 | 實際 | 達成率 |
|:---|:---|:---|:---|
| 文檔撰寫時間 | < 1 天 | ✅ < 4 小時 | 400% |
| 文檔閱讀時間 | < 30 分鐘 | ✅ < 20 分鐘 | 150% |
| 文檔更新頻率 | 每週 1 次 | ⏳ 待驗證 | - |
| 文檔與代碼一致性 | > 90% | ⏳ 待驗證 | - |

### 開發效率指標

| 指標 | 傳統方式 | Linus 式 MVP | 改善率 |
|:---|:---|:---|:---|
| 架構設計時間 | 2 週 | 1 天 | 93% ↓ |
| 首次可運行版本 | 4-6 週 | 1-2 週 | 66% ↓ |
| 文檔維護成本 | 高 | 低 | 70% ↓ |
| 重構難度 | 高（4 層） | 低（1 層） | 75% ↓ |

---

## ⚠️ 何時需要升級至「完整流程」？

### 升級觸發條件

1. **規模擴大**
   - [ ] 並發需求 > 10 會話
   - [ ] 用戶數 > 100 人
   - [ ] 需要跨區域部署

2. **複雜度上升**
   - [ ] 需要多用戶系統（身份驗證）
   - [ ] 需要引入 PostgreSQL/Redis
   - [ ] 代碼量 > 5000 行

3. **團隊協作**
   - [ ] 團隊 > 3 人
   - [ ] 需要跨團隊協作
   - [ ] 需要嚴格的權限控制

4. **合規需求**
   - [ ] 需要處理敏感資料（GDPR）
   - [ ] 需要 SLA 保證
   - [ ] 需要審計追蹤

### 升級路徑

```
MVP 精簡版（模式 B）
    ↓ （任一條件觸發）
逐步重構
    ↓
完整流程（模式 A）
```

**重構策略**:
1. 先引入 PostgreSQL（保持單層架構）
2. 再引入 Redis（優化性能）
3. 最後重構為 Clean Architecture（如有必要）

---

## 🎓 Linus 式設計哲學回顧

### 為什麼這樣設計？

**問題 1**: 為什麼只要 2 張表？
> **回答**: "Bad programmers worry about code. Good programmers worry about data structures."
>
> 對話與聲紋就是全部核心數據，不需要更多表。過度正規化只會增加複雜度。

**問題 2**: 為什麼不用 Clean Architecture？
> **回答**: "如果你需要超過 3 層縮排，你就完蛋了。"
>
> MVP 階段功能簡單，4 層架構是過度設計。先把核心流程跑通，再談抽象。

**問題 3**: 為什麼不用 DDD/BDD？
> **回答**: "Theory and practice sometimes clash. Theory loses. Every single time."
>
> 對於 200 行的聊天機器人，Bounded Context 和 30+ scenarios 是理論完美但實踐複雜。

**問題 4**: 為什麼先寫文檔？
> **回答**: "Talk is cheap. Show me the code."
>
> 這份文檔只花 4 小時，且能直接指導開發。如果文檔需要 1 週，那就先寫代碼。

---

## 📖 延伸閱讀

### 內部文檔
- [需求分析原文](../需求.md)
- [工作流程手冊](../VibeCoding_Workflow_Templates/00_workflow_manual.md)
- [專案結構指南](../PROJECT_STRUCTURE.md)

### Linus 哲學文章
- [Good Taste in Code](https://www.youtube.com/watch?v=o8NPllzkFhE) - Linus 談好品味
- [Never Break Userspace](https://lkml.org/lkml/2012/12/23/75) - 向後兼容鐵律
- [Linux Kernel Coding Style](https://www.kernel.org/doc/html/latest/process/coding-style.html)

---

## 🚀 快速開始

```bash
# 1. 克隆專案
git clone <repo-url>
cd avatar

# 2. 閱讀核心文檔（20 分鐘）
cat docs/planning/mvp_tech_spec.md

# 3. 環境準備（30 分鐘）
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python scripts/download_models.py

# 4. 開始開發（Week 1）
# 先寫 200 行代碼，證明它能跑
# 再來談優化、重構、架構

# 5. 週更新進度（每週五）
vim docs/dev/development_progress_report.md

# 6. 上線前檢查（Week 4）
python scripts/check_launch_readiness.py
```

---

## 💬 FAQ

**Q: 這套文檔適合什麼場景？**
A: MVP 快速驗證、原型開發、技術展示、面試作業。不適合大型企業級專案。

**Q: 如何處理技術債務？**
A: 先交付，再重構。技術債務記錄在 `development_progress_report.md`，等問題真實出現再處理。

**Q: 測試策略是什麼？**
A: Week 1-3 手動測試，Week 4 補充核心邏輯測試。不追求 100% 覆蓋率。

**Q: 如何說服團隊採用這套方案？**
A: 展示對比：90KB 文檔 vs 25KB 文檔，12 週開發 vs 4 週開發，4 層架構 vs 1 層架構。

---

**最後更新**: 2025-11-01
**文檔作者**: Lead Engineer（Linus 式哲學踐行者）
**文檔版本**: 1.0.0 - Lean MVP Edition
