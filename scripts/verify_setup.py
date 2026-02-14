#!/usr/bin/env python3
"""Verify all prerequisites before training."""
import sys
import os
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))
print(f"Added to path: {PROJECT_ROOT}")

import torch


def check():
    print("="*60)
    print("PanopticBEV Setup Verification")
    print("="*60)
    
    # 1. Check Python
    print(f"\n1. Python: {sys.version.split()[0]}")
    
    # 2. Check PyTorch
    print(f"\n2. PyTorch: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"   Device: {props.name}")
        print(f"   Memory: {props.total_memory / 1e9:.1f} GB")
        print(f"   CUDA Capability: {props.major}.{props.minor} (sm_{props.major}{props.minor})")
        
        # Check sm_120 support
        if props.major == 12 and props.minor == 0:
            print("   [OK] sm_120 (Blackwell) detected")
            
            # Test actual tensor operation to verify kernel support
            try:
                test_tensor = torch.randn(100, 100).cuda()
                result = test_tensor @ test_tensor.T
                print(f"   [OK] Tensor operations working (test shape: {result.shape})")
            except RuntimeError as e:
                print(f"   [FAIL] Tensor operations failed: {e}")
                print("   -> PyTorch doesn't support sm_120 - upgrade required!")
                print("   -> Run: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128")
                return False
    
    # 3. Check imports
    print("\n3. Testing imports...")
    try:
        from panoptic_bev.utils.windows_dataloader import create_safe_dataloader
        print("   [OK] windows_dataloader")
    except Exception as e:
        print(f"   [FAIL] windows_dataloader: {e}")
        return False
    
    try:
        from panoptic_bev.data.dataset import BEVKitti360Dataset
        print("   [OK] BEVKitti360Dataset")
    except Exception as e:
        print(f"   [FAIL] BEVKitti360Dataset: {e}")
        return False
    
    # 4. Check dataset paths
    print("\n4. Checking dataset paths...")
    dataset = Path("D:/datasets/kitti360")
    seam = Path("D:/kitti360_panopticbev")
    
    if dataset.exists():
        print(f"   [OK] Dataset: {dataset}")
    else:
        print(f"   [FAIL] Dataset not found: {dataset}")
        return False
        
    if seam.exists():
        print(f"   [OK] PanopticBEV: {seam}")
    else:
        print(f"   [FAIL] PanopticBEV not found: {seam}")
        return False
    
    print("\n" + "="*60)
    print("[OK] All checks passed!")
    print("="*60)
    return True


if __name__ == '__main__':
    success = check()
    sys.exit(0 if success else 1)
