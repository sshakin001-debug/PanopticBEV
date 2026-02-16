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

REM Activate Visual Studio x64 build environment
echo Activating MSVC x64 build environment...
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 (
    echo ERROR: Failed to activate MSVC environment
    pause
    exit /b 1
)

REM Verify architecture
echo Target architecture: %VSCMD_ARG_TGT_ARCH%
if not "%VSCMD_ARG_TGT_ARCH%"=="x64" (
    echo ERROR: Failed to set x64 target architecture
    pause
    exit /b 1
)

REM Navigate to project directory
cd /d %~dp0

REM Clean stale build artifacts
echo Cleaning old build artifacts...
if exist build rd /s /q build 2>nul
if exist dist rd /s /q dist 2>nul
for /d %%d in (*.egg-info) do rd /s /q "%%d" 2>nul

REM Set environment variables for build
set MAX_JOBS=4
set TORCH_CUDA_ARCH_LIST=8.6;8.9;9.0;12.0
set TORCH_NVCC_FLAGS=-Xfatbin=-compress-all
set DISTUTILS_USE_SDK=1

echo.
echo Building CUDA extensions...
echo CUDA Path: %CUDA_PATH%
echo Python: %~dp0panoptic_bev_venv\Scripts\python.exe
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
    echo 2. Ensure Visual Studio 2022 BuildTools with C++ workload is installed
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
