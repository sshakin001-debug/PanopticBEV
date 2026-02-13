"""
Windows CPU Affinity and Optimization Utilities for PanopticBEV

This module provides functions to optimize CPU affinity and threading settings
for DataLoader workers on Windows. Windows handles large core counts differently
than Linux, which can cause stability issues with PyTorch DataLoader.
"""
import os
import platform


def set_cpu_affinity():
    """
    Optimize CPU affinity for DataLoader workers on Windows.
    
    This function sets environment variables to control thread usage and
    optionally limits DataLoader to physical cores (avoiding hyperthreading)
    for better stability on Windows.
    
    Should be called early in the training script, before creating DataLoaders.
    
    Example:
        >>> from panoptic_bev.utils.windows_cpu_affinity import set_cpu_affinity
        >>> set_cpu_affinity()
        >>> # Now create your DataLoaders
    """
    if platform.system() != 'Windows':
        return
    
    try:
        # Get physical core count
        import psutil
        cpu_count = psutil.cpu_count(logical=False)
        logical_count = psutil.cpu_count(logical=True)
        
        # Set OpenMP and MKL thread counts to physical cores
        # This prevents oversubscription which can hurt performance on Windows
        os.environ['OMP_NUM_THREADS'] = str(cpu_count)
        os.environ['MKL_NUM_THREADS'] = str(cpu_count)
        
        # Set Intel OpenMP threads
        os.environ['OMP_THREAD_LIMIT'] = str(cpu_count)
        
        # Disable OpenMP nesting (can cause issues on Windows)
        os.environ['OMP_NESTED'] = 'FALSE'
        
        # Set PyTorch thread count
        import torch
        torch.set_num_threads(cpu_count)
        
        print(f"CPU affinity set: Using {cpu_count} physical cores (of {logical_count} logical)")
        
    except ImportError:
        # psutil not available, set conservative defaults
        import multiprocessing
        cpu_count = multiprocessing.cpu_count() // 2  # Assume hyperthreading
        
        os.environ['OMP_NUM_THREADS'] = str(cpu_count)
        os.environ['MKL_NUM_THREADS'] = str(cpu_count)
        
        print(f"CPU affinity set (psutil not available): Using {cpu_count} estimated cores")


def set_process_cpu_affinity(process=None, cores=None):
    """
    Set CPU affinity for a specific process (Windows only).
    
    Args:
        process: psutil.Process instance or None for current process
        cores: List of core indices to use, or None for all physical cores
    
    Returns:
        bool: True if successful, False otherwise
    """
    if platform.system() != 'Windows':
        return True
    
    try:
        import psutil
        
        if process is None:
            process = psutil.Process()
        
        if cores is None:
            # Use only physical cores (even indices if hyperthreading)
            logical_cores = psutil.cpu_count(logical=True)
            physical_cores = psutil.cpu_count(logical=False)
            
            if logical_cores == physical_cores:
                # No hyperthreading, use all cores
                cores = list(range(logical_cores))
            else:
                # Use every other core (physical only)
                cores = list(range(0, logical_cores, 2))
        
        process.cpu_affinity(cores)
        print(f"Process CPU affinity set to cores: {cores}")
        return True
        
    except ImportError:
        print("Warning: psutil not available, cannot set CPU affinity")
        return False
    except Exception as e:
        print(f"Warning: Could not set CPU affinity: {e}")
        return False


def configure_windows_performance():
    """
    Configure Windows for optimal deep learning performance.
    
    This includes:
    - Setting CPU affinity
    - Configuring thread counts
    - Setting high process priority (if possible)
    
    Example:
        >>> from panoptic_bev.utils.windows_cpu_affinity import configure_windows_performance
        >>> configure_windows_performance()
    """
    if platform.system() != 'Windows':
        return
    
    print("Configuring Windows for optimal performance...")
    
    # Set CPU affinity first
    set_cpu_affinity()
    
    # Try to set high priority
    try:
        import psutil
        process = psutil.Process()
        
        # Set high priority (below realtime, above normal)
        process.nice(psutil.HIGH_PRIORITY_CLASS)
        print("Process priority set to HIGH")
        
    except ImportError:
        pass
    except Exception as e:
        print(f"Could not set process priority: {e}")
    
    # Disable Windows Error Reporting dialogs (can interrupt training)
    try:
        import ctypes
        ctypes.windll.kernel32.SetErrorMode(0x0002)  # SEM_FAILCRITICALERRORS
    except:
        pass
    
    print("Windows performance configuration complete")


def get_optimal_num_workers(safe_mode=True):
    """
    Get the optimal number of DataLoader workers for Windows.
    
    Args:
        safe_mode: If True, use conservative settings for stability
    
    Returns:
        int: Recommended number of workers
    """
    if platform.system() != 'Windows':
        import multiprocessing
        return min(8, multiprocessing.cpu_count())
    
    try:
        import psutil
        physical_cores = psutil.cpu_count(logical=False)
        
        if safe_mode:
            # Conservative: Use fewer workers to avoid Windows issues
            return max(1, physical_cores // 2)
        else:
            # Aggressive: Use all physical cores
            return physical_cores
            
    except ImportError:
        # Conservative default
        return 2


def check_numa_support():
    """
    Check and report NUMA (Non-Uniform Memory Access) configuration.
    
    NUMA awareness can improve performance on multi-socket systems.
    
    Returns:
        dict: NUMA information or None if not available
    """
    if platform.system() != 'Windows':
        return None
    
    try:
        import psutil
        
        # Check if NUMA is available
        if hasattr(psutil, 'cpu_count'):
            # psutil doesn't directly expose NUMA, but we can check core counts
            logical = psutil.cpu_count(logical=True)
            physical = psutil.cpu_count(logical=False)
            
            return {
                'logical_cores': logical,
                'physical_cores': physical,
                'hyperthreading': logical > physical,
                'numa_available': False  # Would need specialized library
            }
    except ImportError:
        pass
    
    return None


# Auto-configure on import if running on Windows
if platform.system() == 'Windows':
    # Set basic environment variables immediately
    try:
        import psutil
        cpu_count = psutil.cpu_count(logical=False)
        os.environ.setdefault('OMP_NUM_THREADS', str(cpu_count))
        os.environ.setdefault('MKL_NUM_THREADS', str(cpu_count))
    except ImportError:
        pass
