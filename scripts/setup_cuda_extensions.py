#!/usr/bin/env python3
"""
CUDA Extensions Setup Script for PanopticBEV on Windows

This script builds CUDA extensions with correct architecture flags for
RTX 5060 Ti (sm_120) and other modern GPUs on Windows.

Usage:
    python scripts/setup_cuda_extensions.py
    
This will build all CUDA extensions in src/ with the correct architecture flags.
"""
import torch
import subprocess
import sys
import os
from pathlib import Path


def setup_cuda_extensions():
    """Build CUDA extensions with correct architecture flags."""
    
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available. Cannot build CUDA extensions.")
        print("CPU-only mode will be used (slow).")
        return False
    
    # Check compute capability
    capability = torch.cuda.get_device_capability()
    major, minor = capability
    arch_flag = f"-gencode=arch=compute_{major}{minor},code=sm_{major}{minor}"
    
    print("=" * 60)
    print("PanopticBEV CUDA Extensions Setup")
    print("=" * 60)
    print(f"Building for architecture: sm_{major}{minor}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print("=" * 60)
    
    # Repository root
    repo_root = Path(__file__).parent.parent
    
    # Set environment variables for building
    env = {
        **os.environ,
        "TORCH_CUDA_ARCH_LIST": f"{major}.{minor}",
        "FORCE_CUDA": "1"
    }
    
    # On Windows, we need to use the setup.py build
    print("\nBuilding CUDA extensions via setup.py...")
    print("This may take several minutes...")
    print("-" * 60)
    
    try:
        result = subprocess.run(
            [sys.executable, "setup.py", "build_ext", "--inplace"],
            cwd=repo_root,
            env=env,
            check=True,
            capture_output=False
        )
        print("-" * 60)
        print("✅ CUDA extensions built successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print("-" * 60)
        print("❌ Failed to build CUDA extensions")
        print(f"Exit code: {e.returncode}")
        print("\nTroubleshooting:")
        print("1. Ensure you have Visual Studio Build Tools installed")
        print("2. Ensure CUDA toolkit matches your PyTorch CUDA version")
        print("3. Try running from a Developer Command Prompt")
        return False
    except FileNotFoundError:
        print("-" * 60)
        print("❌ setup.py not found")
        return False


def check_extensions_built():
    """Check if extensions are already built."""
    repo_root = Path(__file__).parent.parent
    
    # Check for built extensions
    ext_patterns = [
        "panoptic_bev/utils/bbx/*.pyd",
        "panoptic_bev/utils/nms/*.pyd",
        "panoptic_bev/utils/roi_sampling/*.pyd",
    ]
    
    found_extensions = []
    for pattern in ext_patterns:
        matches = list(repo_root.glob(pattern))
        found_extensions.extend(matches)
    
    if found_extensions:
        print(f"Found {len(found_extensions)} built extension(s):")
        for ext in found_extensions:
            print(f"  - {ext.name}")
        return True
    
    return False


def main():
    """Main entry point."""
    # Check if extensions are already built
    if check_extensions_built():
        print("\nExtensions already built. Rebuild? (y/N)")
        response = input().strip().lower()
        if response != 'y':
            print("Skipping build.")
            return
    
    # Build extensions
    success = setup_cuda_extensions()
    
    if success:
        print("\nNext steps:")
        print("1. Run scripts/validate_windows_env.py to verify setup")
        print("2. Run training with scripts/train_kitti.bat or scripts/train_nuscenes.bat")
    else:
        print("\nBuild failed. Check the error messages above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
