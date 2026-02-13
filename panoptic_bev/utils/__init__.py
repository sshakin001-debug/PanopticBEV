"""
PanopticBEV Utilities Package

This package provides utility functions for the PanopticBEV project,
including Windows compatibility utilities.
"""

# Import Windows compatibility utilities for easy access
try:
    from .config_resolver import WindowsPathResolver, load_config
except ImportError:
    pass

try:
    from .windows_dataloader import (
        create_safe_dataloader,
        get_optimal_batch_size,
        get_optimal_worker_count,
        SafeDataLoader,
        auto_configure_dataloader
    )
except ImportError:
    pass

try:
    from .windows_cpu_affinity import (
        set_cpu_affinity,
        configure_windows_performance,
        get_optimal_num_workers
    )
except ImportError:
    pass

try:
    from .windows_symlink import create_symlink, remove_link
except ImportError:
    pass

__all__ = [
    # Config resolver
    'WindowsPathResolver',
    'load_config',
    # Windows dataloader
    'create_safe_dataloader',
    'get_optimal_batch_size',
    'get_optimal_worker_count',
    'SafeDataLoader',
    'auto_configure_dataloader',
    # Windows CPU affinity
    'set_cpu_affinity',
    'configure_windows_performance',
    'get_optimal_num_workers',
    # Windows symlink
    'create_symlink',
    'remove_link',
]
