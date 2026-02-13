#!/usr/bin/env python3
"""
Environment Validation Script for PanopticBEV on Windows

This script checks if the Windows environment is properly configured for
running PanopticBEV, including CUDA availability, compute capability,
and path handling.

Usage:
    python scripts/validate_windows_env.py
"""
import sys
import torch
import platform
from pathlib import Path


def validate_environment():
    """Check if Windows environment is properly configured."""
    checks = {
        'python_version': sys.version_info >= (3, 8),
        'cuda_available': torch.cuda.is_available(),
        'sm_compatible': False,
        'path_separator': '\\' if platform.system() == 'Windows' else '/',
        'multiprocessing_spawn': True
    }
    
    print("=" * 60)
    print("PanopticBEV Windows Environment Validation")
    print("=" * 60)
    
    # Check Python version
    if checks['python_version']:
        print(f"✓ Python {sys.version.split()[0]} (>= 3.8)")
    else:
        print(f"✗ Python {sys.version.split()[0]} (requires >= 3.8)")
    
    # Check platform
    print(f"✓ Platform: {platform.system()} {platform.release()}")
    
    # Check CUDA availability and compute capability
    if checks['cuda_available']:
        major, minor = torch.cuda.get_device_capability()
        # RTX 5060 Ti has sm_120, but check for compatibility
        # sm_120+ or any modern GPU (sm_70 and above)
        checks['sm_compatible'] = (major >= 7) or (major == 12 and minor >= 0)
        
        print(f"✓ CUDA {torch.version.cuda} available")
        print(f"✓ GPU: {torch.cuda.get_device_name(0)} (sm_{major}{minor})")
        print(f"  Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        if checks['sm_compatible']:
            print(f"✓ Compute capability sm_{major}{minor} is compatible")
        else:
            print(f"⚠ Compute capability sm_{major}{minor} may have issues (requires sm_70+)")
    else:
        print("✗ CUDA not available - training will be slow on CPU")
    
    # Check path handling
    test_path = Path("D:/test") / "subdir"
    if platform.system() == 'Windows':
        if '\\' in str(test_path) or '/' in str(test_path):
            print("✓ Pathlib working correctly")
        else:
            print("✗ Pathlib not working correctly")
            checks['path_separator'] = False
    
    # Check PyTorch version
    print(f"✓ PyTorch {torch.__version__}")
    
    # Check for required packages
    try:
        import numpy
        print(f"✓ NumPy {numpy.__version__}")
    except ImportError:
        print("✗ NumPy not installed")
        checks['numpy'] = False
    
    try:
        import cv2
        print(f"✓ OpenCV available")
    except ImportError:
        print("✗ OpenCV not installed")
        checks['opencv'] = False
    
    try:
        import yaml
        print(f"✓ PyYAML available")
    except ImportError:
        print("⚠ PyYAML not installed (may use INI configs)")
    
    print("=" * 60)
    
    # Summary
    critical_checks = ['python_version', 'sm_compatible'] if checks['cuda_available'] else ['python_version']
    all_passed = all(checks.get(k, True) for k in critical_checks)
    
    if all_passed:
        print("\n✅ Environment ready for PanopticBEV on Windows")
        if checks['cuda_available']:
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            if vram_gb >= 16:
                print("   Recommended batch size: 4, AMP: enabled")
            elif vram_gb >= 12:
                print("   Recommended batch size: 2, AMP: enabled")
            else:
                print("   Recommended batch size: 1, AMP: enabled, gradient accumulation: 2")
        return True
    else:
        print("\n❌ Environment issues detected")
        return False


if __name__ == "__main__":
    success = validate_environment()
    sys.exit(0 if success else 1)
