@echo off
REM Build script for PanopticBEV CUDA extensions on Windows
REM This script ensures proper 64-bit compilation environment

echo ============================================
echo PanopticBEV CUDA Extension Builder
echo ============================================

REM Set CUDA path
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8
if not exist "%CUDA_PATH%" (
    echo ERROR: CUDA 12.8 not found at %CUDA_PATH%
    echo Please install CUDA Toolkit 12.8
    pause
    exit /b 1
)

REM Check if already in VS environment (check for cl.exe)
where cl.exe >nul 2>&1
if errorlevel 1 (
    echo Activating MSVC x64 build environment...
    call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
    if errorlevel 1 (
        echo ERROR: Failed to activate MSVC environment
        echo Please run this from "x64 Native Tools Command Prompt for VS 2022"
        pause
        exit /b 1
    )
) else (
    echo MSVC environment already active
)

REM Navigate to project directory
cd /d %~dp0

REM Clean stale build artifacts (optional - comment out to keep)
REM echo Cleaning old build artifacts...
REM if exist build rd /s /q build 2>nul

REM Set environment variables for build
set MAX_JOBS=2
set DISTUTILS_USE_SDK=1

echo.
echo Building CUDA extensions...
echo CUDA Path: %CUDA_PATH%
echo.

REM Build extensions
panoptic_bev_venv\Scripts\python.exe setup.py build_ext --inplace 2>&1

if errorlevel 1 (
    echo.
    echo ============================================
    echo BUILD FAILED
    echo ============================================
    echo.
    echo Common fixes:
    echo 1. Ensure CUDA 12.8 is installed
    echo 2. Run from "x64 Native Tools Command Prompt for VS 2022"
    echo 3. Check that ninja is installed: pip install ninja
    echo.
) else (
    echo.
    echo ============================================
    echo BUILD SUCCESSFUL
    echo ============================================
    echo.
    echo Verifying extensions...
    dir /s /b panoptic_bev\utils\*.pyd 2>nul
    if errorlevel 1 (
        echo WARNING: No .pyd files found after build
    )
)

pause
