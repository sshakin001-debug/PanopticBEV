@echo off
setlocal enabledelayedexpansion

:: ============================================================================
:: Auto-generated Windows batch file for PanopticBEV
:: Generated from eval_panoptic_bev_kitti.sh
:: ============================================================================

:: Set repository root (directory containing this script's parent)
set "REPO_ROOT=%~dp0.."
set "PYTHONPATH=%REPO_ROOT%;%PYTHONPATH%"

:: Set dataset paths (MODIFY THESE TO MATCH YOUR SYSTEM)
set "DATASET_ROOT=D:\datasets\kitti"
set "SEAM_ROOT=D:\datasets\panopticbev\seam"
set "OUTPUT_DIR=D:\outputs\panopticbev\kitti"

:: Create output directory if it doesn't exist
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

echo ========================================
echo PanopticBEV eval - KITTI
echo ========================================
echo Dataset: %DATASET_ROOT%
echo Output:  %OUTPUT_DIR%
echo.

:: Set environment variables for the Python script
set "dataset_root_dir=%DATASET_ROOT%"
set "seam_root_dir=%SEAM_ROOT%"
set "output_dir=%OUTPUT_DIR%"
set "config_file=%REPO_ROOT%\experiments\config\kitti.ini"

:: Run the Python script
python "%REPO_ROOT%\scripts\eval_panoptic_bev.py" ^
    --config "%config_file%" ^
    %*

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: Script failed with exit code %ERRORLEVEL%
    exit /b %ERRORLEVEL%
)

echo.
echo Evaluation complete!
pause
