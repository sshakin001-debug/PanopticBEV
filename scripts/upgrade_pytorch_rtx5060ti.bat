@echo off
REM Upgrade PyTorch for RTX 5060 Ti (Blackwell sm_120) support
REM PyTorch 2.7.1+ with CUDA 12.8 is required for sm_120 support

echo ============================================================
echo PyTorch Upgrade for RTX 5060 Ti (Blackwell sm_120)
echo ============================================================
echo.

REM Activate virtual environment
call ..\panoptic_bev_venv\Scripts\activate.bat

echo Current PyTorch version:
python -c "import torch; print(f'  PyTorch: {torch.__version__}'); print(f'  CUDA: {torch.version.cuda}')"
echo.

echo Uninstalling old PyTorch...
pip uninstall torch torchvision torchaudio -y
echo.

echo Installing PyTorch 2.7.1+ with CUDA 12.8...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
echo.

echo ============================================================
echo Verifying installation...
echo ============================================================
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}'); print(f'Device: {torch.cuda.get_device_name(0)}'); print(f'Capability: {torch.cuda.get_device_capability()}'); x = torch.zeros(1).cuda(); print('CUDA tensor test passed!')"

echo.
echo ============================================================
echo Upgrade complete! Now run verify_setup.py
echo ============================================================
pause
