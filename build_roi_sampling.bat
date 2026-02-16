@echo off
REM Minimal build script for ROI sampling CUDA extension
REM Run this from a Visual Studio Developer Command Prompt or x64 Native Tools Command Prompt

echo Building ROI sampling CUDA extension...

cd /d %~dp0

set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8
set MAX_JOBS=2
set DISTUTILS_USE_SDK=1

panoptic_bev_venv\Scripts\python.exe setup.py build_ext --inplace 2>&1

echo Done.
