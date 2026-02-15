@echo off
REM Build script for PanopticBEV CUDA extensions on Windows
REM This script ensures proper 64-bit compilation environment

REM Set CUDA path
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8

REM Activate Visual Studio x64 build environment
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"

REM Verify architecture
echo Target architecture: %VSCMD_ARG_TGT_ARCH%
if not "%VSCMD_ARG_TGT_ARCH%"=="x64" (
    echo ERROR: Failed to set x64 target architecture
    exit /b 1
)

REM Clean stale build artifacts
if exist build rd /s /q build
if exist dist rd /s /q dist
if exist *.egg-info rd /s /q *.egg-info

REM Navigate to project directory and build
cd /d %~dp0
panoptic_bev_venv\Scripts\python.exe setup.py build_ext --inplace

echo Build complete!