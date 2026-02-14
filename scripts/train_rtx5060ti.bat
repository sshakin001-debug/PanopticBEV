@echo off
echo ==========================================
echo PanopticBEV Training - RTX 5060 Ti
echo ==========================================

:: Set environment
set OMP_NUM_THREADS=1
set MKL_NUM_THREADS=1
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

:: Your dataset paths
set PROJECT_ROOT=D:\PanopticBEV
set DATASET_ROOT=D:\datasets\kitti360
set SEAM_ROOT=D:\kitti360_panopticbev

:: Verify paths
if not exist "%DATASET_ROOT%" (
    echo ERROR: Dataset not found at %DATASET_ROOT%
    exit /b 1
)
if not exist "%SEAM_ROOT%" (
    echo ERROR: PanopticBEV data not found at %SEAM_ROOT%
    exit /b 1
)

echo Dataset: %DATASET_ROOT%
echo PanopticBEV: %SEAM_ROOT%

:: Run training (single GPU for RTX 5060 Ti)
python scripts\train_panoptic_bev.py ^
    --local_rank 0 ^
    --run_name rtx5060ti_test ^
    --project_root_dir "%PROJECT_ROOT%" ^
    --seam_root_dir "%SEAM_ROOT%" ^
    --dataset_root_dir "%DATASET_ROOT%" ^
    --mode train ^
    --train_dataset Kitti360 ^
    --val_dataset Kitti360 ^
    --config kitti.ini ^
    --debug False

pause
