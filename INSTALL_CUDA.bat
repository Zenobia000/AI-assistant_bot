@echo off
REM AVATAR - Quick CUDA Setup Launcher
REM Double-click this file to start installation

echo ==========================================
echo   AVATAR WSL2 CUDA Setup
echo ==========================================
echo.
echo This will install CUDA Toolkit in WSL2
echo Estimated time: 10-30 minutes
echo.
echo Requirements:
echo   - Administrator privileges
echo   - Internet connection
echo   - WSL2 enabled
echo.
pause

echo.
echo [INFO] Launching PowerShell script...
echo.

PowerShell -ExecutionPolicy Bypass -File "%~dp0scripts\setup_cuda_wsl2.ps1"

if %ERRORLEVEL% EQU 0 (
    echo.
    echo [SUCCESS] Installation completed!
) else (
    echo.
    echo [ERROR] Installation failed. Check errors above.
)

echo.
pause
