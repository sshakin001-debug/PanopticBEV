"""
Windows Performance Utilities (ASCII-only version for cp1252 compatibility)
"""
import os
import sys
import subprocess
from pathlib import Path


def is_native_windows():
    """Check if running on native Windows (not WSL2)."""
    return sys.platform == 'win32' and 'WSL_DISTRO_NAME' not in os.environ


def is_wsl2():
    """Check if running on WSL2."""
    return 'WSL_DISTRO_NAME' in os.environ


def is_linux():
    """Check if running on native Linux."""
    return sys.platform.startswith('linux') and 'WSL_DISTRO_NAME' not in os.environ


def get_windows_version():
    """Get Windows version information."""
    if is_native_windows():
        try:
            result = subprocess.run(
                ['cmd', '/c', 'ver'],
                capture_output=True,
                text=True,
                check=False
            )
            return result.stdout.strip()
        except Exception:
            return "Unknown Windows version"
    return None


def check_cuda_windows():
    """Check CUDA availability on Windows."""
    if not is_native_windows():
        return None
    
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
            capture_output=True,
            text=True,
            check=False
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except FileNotFoundError:
        pass
    return None


def suggest_wsl2_installation():
    """Print instructions for installing WSL2 for best performance."""
    msg = """
    +====================================================================+
    |                    PERFORMANCE RECOMMENDATION                     |
    +====================================================================+
    |  For Linux-level performance on Windows, use WSL2:               |
    |                                                                   |
    |  1. Install WSL2:                                                |
    |     wsl --install -d Ubuntu                                      |
    |                                                                   |
    |  2. Install CUDA in WSL2:                                        |
    |     https://developer.nvidia.com/cuda/wsl                        |
    |                                                                   |
    |  3. Clone and run in WSL2:                                       |
    |     cd /mnt/<your-drive>/PanopticBEV                             |
    |     python scripts/train_panoptic_bev.py ...                     |
    |                                                                   |
    |  WSL2 provides:                                                  |
    |  * Native Linux kernel (no Windows translation layer)            |
    |  * Full GPU support with NCCL                                    |
    |  * Native multiprocessing (fork, not spawn)                      |
    |  * ~95-100% of native Linux performance                          |
    +====================================================================+
    """
    print(msg)


def check_performance_mode():
    """
    Check current environment and suggest optimizations.
    
    Returns:
        str: 'native_windows', 'wsl2', or 'linux'
    """
    if is_native_windows():
        print("[Perf] Running on native Windows")
        print("[Perf] For best performance, consider using WSL2")
        # Skip the WSL2 suggestion to avoid encoding issues
        return 'native_windows'
    elif is_wsl2():
        print("[Perf] Running on WSL2")
        return 'wsl2'
    elif is_linux():
        print("[Perf] Running on native Linux")
        return 'linux'
    else:
        print("[Perf] Unknown platform")
        return 'unknown'


def apply_windows_optimizations():
    """Apply Windows-specific optimizations."""
    if not is_native_windows():
        return
    
    print("[Perf] Native Windows detected - applying optimized settings")
    
    # Set multiprocessing start method
    import multiprocessing
    try:
        multiprocessing.set_start_method('spawn', force=True)
        print("[Perf] Multiprocessing start method set to 'spawn'")
    except RuntimeError:
        pass
    
    # Set environment variables for better performance
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')
    
    # Check CUDA
    gpu_name = check_cuda_windows()
    if gpu_name:
        print(f"[Perf] CUDA GPU detected: {gpu_name}")
    else:
        print("[Perf] No CUDA GPU detected or nvidia-smi not available")


def get_recommended_num_workers():
    """Get recommended number of DataLoader workers for current platform."""
    import multiprocessing
    cpu_count = multiprocessing.cpu_count()
    
    if is_native_windows():
        # Windows spawn method has more overhead
        return min(4, cpu_count)
    else:
        # Linux/WSL2 can handle more workers
        return min(8, cpu_count)


def get_platform_info():
    """Get detailed platform information."""
    info = {
        'platform': sys.platform,
        'is_native_windows': is_native_windows(),
        'is_wsl2': is_wsl2(),
        'is_linux': is_linux(),
        'python_version': sys.version,
        'cpu_count': os.cpu_count(),
    }
    
    if is_native_windows():
        info['windows_version'] = get_windows_version()
        info['cuda_gpu'] = check_cuda_windows()
    
    return info