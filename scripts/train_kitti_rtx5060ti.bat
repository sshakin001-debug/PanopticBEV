@echo off
echo ==========================================
echo PanopticBEV - RTX 5060 Ti Optimized
echo ==========================================

:: Set optimal environment for RTX 5060 Ti
set OMP_NUM_THREADS=1
set MKL_NUM_THREADS=1
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,garbage_collection_threshold:0.6

:: Dataset paths (YOUR SETUP)
set DATASET_ROOT=D:\datasets\kitti360
set SEAM_ROOT=D:\kitti360_panopticbev

echo Dataset: %DATASET_ROOT%
echo PanopticBEV: %SEAM_ROOT%

:: Verify paths exist
if not exist "%DATASET_ROOT%" (
    echo ERROR: Dataset not found at %DATASET_ROOT%
    exit /b 1
)
if not exist "%SEAM_ROOT%" (
    echo ERROR: PanopticBEV data not found at %SEAM_ROOT%
    exit /b 1
)

:: Run training with RTX 5060 Ti optimizations
python scripts\train_panoptic_bev_windows.py ^
    --config experiments\config\kitti.ini ^
    --dataset kitti ^
    --dataset_root_dir "%DATASET_ROOT%" ^
    --seam_root_dir "%SEAM_ROOT%" ^
    --run_name rtx5060ti_16gb_test ^
    --num_workers 4

pause