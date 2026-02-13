"""
Windows-Safe DataLoader for PanopticBEV

This module provides a DataLoader wrapper that handles Windows-specific
multiprocessing issues and provides platform-appropriate defaults.

Windows requires 'spawn' method for multiprocessing, while Linux uses 'fork'.
This wrapper ensures proper initialization on both platforms.
"""
import torch
import platform
from torch.utils.data import DataLoader


def create_safe_dataloader(dataset, batch_size, num_workers=None, **kwargs):
    """
    Creates a DataLoader with Windows-safe defaults.
    
    This function automatically handles the differences between Windows and Linux
    multiprocessing, sets appropriate defaults for each platform, and ensures
    CUDA tensors are handled correctly.
    
    Args:
        dataset: PyTorch Dataset instance
        batch_size: Number of samples per batch
        num_workers: Number of subprocesses for data loading. 
                     Default: 0 on Windows, 4 on Linux
        **kwargs: Additional arguments passed to DataLoader
    
    Returns:
        DataLoader instance with platform-appropriate settings
    
    Example:
        >>> from panoptic_bev.utils.windows_dataloader import create_safe_dataloader
        >>> dataloader = create_safe_dataloader(
        ...     dataset=my_dataset,
        ...     batch_size=4,
        ...     shuffle=True,
        ...     num_workers=2  # Optional: override default
        ... )
    """
    
    # Set default num_workers based on platform
    if num_workers is None:
        # Windows: Start with 0 (no multiprocessing) to avoid issues
        # Linux: Use 4 workers for better performance
        num_workers = 0 if platform.system() == 'Windows' else 4
    
    # Windows-specific: Always use spawn method for multiprocessing
    if platform.system() == 'Windows' and num_workers > 0:
        import multiprocessing as mp
        try:
            if mp.get_start_method(allow_none=True) != 'spawn':
                mp.set_start_method('spawn', force=True)
        except RuntimeError:
            # Method may already be set
            pass
    
    # Disable persistent_workers on Windows if num_workers=0
    if platform.system() == 'Windows':
        if num_workers == 0:
            kwargs.pop('persistent_workers', None)
        # Ensure drop_last is set to avoid issues with uneven batches
        if 'drop_last' not in kwargs:
            kwargs['drop_last'] = True
    
    # Pin memory for faster GPU transfer if CUDA is available
    pin_memory = kwargs.pop('pin_memory', torch.cuda.is_available())
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        **kwargs
    )


def get_optimal_worker_count():
    """
    Get the optimal number of DataLoader workers for the current platform.
    
    Returns:
        int: Recommended number of workers
    """
    import multiprocessing as mp
    
    cpu_count = mp.cpu_count()
    
    if platform.system() == 'Windows':
        # Windows: Use fewer workers to avoid issues
        # Use physical cores if possible, avoiding hyperthreading
        try:
            import psutil
            physical_cores = psutil.cpu_count(logical=False)
            return min(physical_cores - 1, 4) if physical_cores else 2
        except ImportError:
            return min(cpu_count // 2, 4)
    else:
        # Linux: Can use more workers
        return min(cpu_count, 8)


def get_optimal_batch_size():
    """
    Get the optimal batch size based on available GPU memory.
    
    Automatically adjusts batch size for different VRAM capacities,
    particularly important for RTX 5060 Ti with 8GB or 16GB variants.
    
    Returns:
        dict: Configuration dict with 'batch_size', 'amp', 'num_workers', 
              and optionally 'gradient_accumulation'
    """
    if not torch.cuda.is_available():
        return {
            'batch_size': 2,
            'amp': False,
            'num_workers': 0,
            'gradient_accumulation': 1
        }
    
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9  # GB
    
    if total_memory >= 24:
        return {
            'batch_size': 8,
            'amp': True,
            'num_workers': 4,
            'gradient_accumulation': 1
        }
    elif total_memory >= 16:
        return {
            'batch_size': 4,
            'amp': True,
            'num_workers': 2,
            'gradient_accumulation': 1
        }
    elif total_memory >= 12:
        return {
            'batch_size': 2,
            'amp': True,
            'num_workers': 2,
            'gradient_accumulation': 2
        }
    else:  # 8GB or less
        return {
            'batch_size': 1,
            'amp': True,
            'num_workers': 0,
            'gradient_accumulation': 4
        }


class SafeDataLoader:
    """
    A wrapper around DataLoader that provides additional Windows safety features.
    
    This class ensures proper cleanup and handles common Windows-specific issues
    with DataLoader workers.
    
    Example:
        >>> from panoptic_bev.utils.windows_dataloader import SafeDataLoader
        >>> loader = SafeDataLoader(dataset, batch_size=4)
        >>> for batch in loader:
        ...     # Training code
        >>> loader.cleanup()  # Clean shutdown
    """
    
    def __init__(self, dataset, batch_size, num_workers=None, **kwargs):
        self.dataset = dataset
        self.batch_size = batch_size
        self.kwargs = kwargs
        self._loader = None
        self._num_workers = num_workers
    
    def __iter__(self):
        if self._loader is None:
            self._loader = create_safe_dataloader(
                self.dataset,
                self.batch_size,
                self._num_workers,
                **self.kwargs
            )
        return iter(self._loader)
    
    def __len__(self):
        if self._loader is None:
            self._loader = create_safe_dataloader(
                self.dataset,
                self.batch_size,
                self._num_workers,
                **self.kwargs
            )
        return len(self._loader)
    
    def cleanup(self):
        """Clean shutdown of the DataLoader."""
        if self._loader is not None:
            # Force cleanup of worker processes on Windows
            if platform.system() == 'Windows' and hasattr(self._loader, '_iterator'):
                if self._loader._iterator is not None:
                    self._loader._iterator._shutdown_workers()
            self._loader = None


# Convenience function for quick setup
def auto_configure_dataloader(dataset, prefer_memory_efficiency=True):
    """
    Automatically configure a DataLoader with optimal settings.
    
    Args:
        dataset: PyTorch Dataset instance
        prefer_memory_efficiency: If True, prioritize memory efficiency over speed
    
    Returns:
        Configured DataLoader instance
    """
    config = get_optimal_batch_size()
    
    if prefer_memory_efficiency and torch.cuda.is_available():
        # Reduce batch size further for memory safety
        config['batch_size'] = max(1, config['batch_size'] // 2)
    
    num_workers = get_optimal_worker_count() if not prefer_memory_efficiency else 0
    
    return create_safe_dataloader(
        dataset,
        batch_size=config['batch_size'],
        num_workers=num_workers,
        shuffle=True,
        drop_last=True
    )
