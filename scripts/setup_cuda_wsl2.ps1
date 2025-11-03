# AVATAR - Launch CUDA Setup in WSL2
# 在 PowerShell 中執行此腳本

Write-Host "=========================================="  -ForegroundColor Cyan
Write-Host "  AVATAR WSL2 CUDA Setup Launcher" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

$scriptPath = "/mnt/d/python_workspace/python-sideproject/AI-related/AI-assistant_bot/scripts/setup_cuda_wsl2.sh"

Write-Host "[INFO] Launching CUDA setup in WSL2..." -ForegroundColor Yellow
Write-Host "[INFO] This may take 10-30 minutes depending on your internet speed" -ForegroundColor Yellow
Write-Host ""

# 使腳本可執行並運行
wsl bash -c "chmod +x $scriptPath && bash $scriptPath"

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "=========================================="  -ForegroundColor Green
    Write-Host "  Setup completed successfully!" -ForegroundColor Green
    Write-Host "==========================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Next: Install Flash-Attention in WSL2" -ForegroundColor Cyan
    Write-Host "  1. Open new WSL2 terminal" -ForegroundColor White
    Write-Host "  2. Run: cd /mnt/d/python_workspace/python-sideproject/AI-related/AI-assistant_bot" -ForegroundColor White
    Write-Host "  3. Run: poetry shell" -ForegroundColor White
    Write-Host "  4. Run: MAX_JOBS=4 pip install flash-attn --no-build-isolation --no-cache-dir" -ForegroundColor White
} else {
    Write-Host ""
    Write-Host "=========================================="  -ForegroundColor Red
    Write-Host "  Setup encountered errors" -ForegroundColor Red
    Write-Host "==========================================" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please check the error messages above" -ForegroundColor Yellow
}

Write-Host ""
Read-Host "Press Enter to exit"
