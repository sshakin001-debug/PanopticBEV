@echo off
REM Simple build script - Run this from "x64 Native Tools Command Prompt for VS 2022"
REM Start Menu -> Visual Studio 2022 -> x64 Native Tools Command Prompt

echo ============================================
echo PanopticBEV ROI Sampling Build
echo ============================================
echo.
echo IMPORTANT: Run this from "x64 Native Tools Command Prompt for VS 2022"
echo.

cd /d %~dp0

set MAX_JOBS=2
set DISTUTILS_USE_SDK=1

echo Building...
panoptic_bev_venv\Scripts\python.exe setup.py build_ext --inplace

if errorlevel 1 (
    echo.
    echo BUILD FAILED
    echo Make sure you are running from x64 Native Tools Command Prompt
) else (
    echo.
    echo BUILD SUCCESSFUL
    dir /b panoptic_bev\utils\roi_sampling\*.pyd 2>nul
)

pause
