@echo off
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
