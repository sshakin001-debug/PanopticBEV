@echo off
echo ==========================================
echo PanopticBEV Fast Training - Native Windows
echo ==========================================

set PYTHONPATH=%CD%;%PYTHONPATH%

:: Set optimal environment variables
set OMP_NUM_THREADS=4
set MKL_NUM_THREADS=4
set CUDA_VISIBLE_DEVICES=0

:: Windows optimizations
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

:: Default paths - modify these for your setup
set CONFIG_FILE=experiments\config\kitti.ini
set DATASET=kitti
set DATASET_ROOT=D:\datasets\kitti360
set SEAM_ROOT=D:\datasets\panopticbev\seam

:: Check if user provided arguments
if not "%~1"=="" (
    :: User provided arguments, use them
    python scripts\train_panoptic_bev_windows.py %*
) else (
    :: Use default arguments
    echo Using default configuration:
    echo   Config: %CONFIG_FILE%
    echo   Dataset: %DATASET%
    echo   Dataset Root: %DATASET_ROOT%
    echo   SEAM Root: %SEAM_ROOT%
    echo.
    
    python scripts\train_panoptic_bev_windows.py ^
        --config %CONFIG_FILE% ^
        --dataset %DATASET% ^
        --dataset-root-dir %DATASET_ROOT% ^
        --seam-root-dir %SEAM_ROOT%
)

echo.
echo ==========================================
echo Training complete or exited
echo ==========================================
pause
