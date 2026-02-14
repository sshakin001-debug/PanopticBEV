"""
Bounding box utilities with fallback for missing CUDA extensions.
"""

import torch

# Check if CUDA backend is available by trying to import and use it
HAS_CUDA_BACKEND = False
try:
    from . import _backend as _cuda_backend
    # Test if the functions actually work (not just stubs)
    # The stub file uses "..." which returns NotImplemented
    test_mask = torch.zeros(1, 10, 10, dtype=torch.float32)
    try:
        result = _cuda_backend.extract_boxes(test_mask, 1)
        # If we get here without NotImplementedError, CUDA backend works
        HAS_CUDA_BACKEND = True
    except (NotImplementedError, RuntimeError):
        # Functions are stubs or CUDA not available
        pass
except ImportError:
    pass

if not HAS_CUDA_BACKEND:
    import warnings
    warnings.warn("BBX CUDA extension not found. Using PyTorch fallback (slower).")

from .bbx import *

__all__ = [
    "extract_boxes",
    "shift_boxes",
    "calculate_shift",
    "corners_to_center_scale",
    "center_scale_to_corners",
    "invert_roi_bbx",
    "ious",
    "mask_overlap",
    "bbx_overlap",
    "HAS_CUDA_BACKEND"
]
