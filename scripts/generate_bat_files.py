#!/usr/bin/env python3
"""
Batch File Generator for PanopticBEV on Windows

This script generates .bat files as an alternative to shell scripts
and Python runners for users who prefer traditional batch files.

Usage:
    python scripts/generate_bat_files.py
    
This will generate .bat files for training and evaluation on KITTI and nuScenes.
"""
from pathlib import Path


BAT_TEMPLATE = '''@echo off
setlocal enabledelayedexpansion

:: ============================================================================
:: Auto-generated Windows batch file for PanopticBEV
:: Generated from {source_script}
:: ============================================================================

:: Set repository root (directory containing this script's parent)
set "REPO_ROOT=%~dp0.."
set "PYTHONPATH=%REPO_ROOT%;%PYTHONPATH%"

:: Set dataset paths (MODIFY THESE TO MATCH YOUR SYSTEM)
set "DATASET_ROOT=D:\\datasets\\{dataset}"
set "SEAM_ROOT=D:\\datasets\\panopticbev\\seam"
set "OUTPUT_DIR=D:\\outputs\\panopticbev\\{dataset}"

:: Create output directory if it doesn't exist
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

echo ========================================
echo PanopticBEV {operation} - {dataset_upper}
echo ========================================
echo Dataset: %DATASET_ROOT%
echo Output:  %OUTPUT_DIR%
echo.

:: Set environment variables for the Python script
set "dataset_root_dir=%DATASET_ROOT%"
set "seam_root_dir=%SEAM_ROOT%"
set "output_dir=%OUTPUT_DIR%"
set "config_file=%REPO_ROOT%\\experiments\\config\\{dataset}.ini"

:: Run the Python script
python "%REPO_ROOT%\\scripts\\{operation}_panoptic_bev.py" ^
    --config "%config_file%" ^
    %*

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: Script failed with exit code %ERRORLEVEL%
    exit /b %ERRORLEVEL%
)

echo.
echo {operation_cap} complete!
pause
'''


def generate_bat_files():
    """Generate .bat files for training and evaluation."""
    scripts_dir = Path(__file__).parent
    generated_files = []
    
    operations = [
        ('train', 'Training'),
        ('eval', 'Evaluation')
    ]
    
    datasets = [
        ('kitti', 'KITTI'),
        ('nuscenes', 'nuScenes')
    ]
    
    for operation, operation_cap in operations:
        for dataset, dataset_upper in datasets:
            content = BAT_TEMPLATE.format(
                source_script=f'{operation}_panoptic_bev_{dataset}.sh',
                operation=operation,
                operation_cap=operation_cap,
                dataset=dataset,
                dataset_upper=dataset_upper
            )
            
            filename = f"{operation}_{dataset}.bat"
            filepath = scripts_dir / filename
            
            with open(filepath, 'w') as f:
                f.write(content)
            
            generated_files.append(filename)
            print(f"Generated {filename}")
    
    # Generate a master run script
    master_content = '''@echo off
echo ========================================
echo PanopticBEV Windows Launcher
echo ========================================
echo.
echo Select an operation:
echo 1. Train on KITTI
echo 2. Train on nuScenes
echo 3. Evaluate on KITTI
echo 4. Evaluate on nuScenes
echo 5. Exit
echo.

set /p choice="Enter choice (1-5): "

if "%choice%"=="1" goto train_kitti
if "%choice%"=="2" goto train_nuscenes
if "%choice%"=="3" goto eval_kitti
if "%choice%"=="4" goto eval_nuscenes
if "%choice%"=="5" goto end

echo Invalid choice
goto end

:train_kitti
call train_kitti.bat
goto end

:train_nuscenes
call train_nuscenes.bat
goto end

:eval_kitti
call eval_kitti.bat
goto end

:eval_nuscenes
call eval_nuscenes.bat
goto end

:end
'''
    
    master_path = scripts_dir / "run_panopticbev.bat"
    with open(master_path, 'w') as f:
        f.write(master_content)
    generated_files.append("run_panopticbev.bat")
    print(f"Generated run_panopticbev.bat")
    
    print(f"\nGeneration complete: {len(generated_files)} file(s) created")
    print("\nIMPORTANT: Edit the .bat files to set your actual dataset paths!")
    print("Look for the lines with 'MODIFY THESE TO MATCH YOUR SYSTEM'")


if __name__ == "__main__":
    generate_bat_files()
