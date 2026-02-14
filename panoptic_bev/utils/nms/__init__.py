"""
NMS (Non-Maximum Suppression) with fallback for missing CUDA extensions.
"""

try:
    from . import _backend
    HAS_CUDA_BACKEND = True
except ImportError:
    HAS_CUDA_BACKEND = False
    import warnings
    warnings.warn("NMS CUDA extension not found. Using PyTorch fallback (slower).")

from .nms import nms

__all__ = ["nms", "HAS_CUDA_BACKEND"]
