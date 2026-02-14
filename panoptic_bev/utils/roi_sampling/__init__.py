"""
ROI Sampling with fallback for missing CUDA extensions.
"""

try:
    from . import _backend
    HAS_CUDA_BACKEND = True
except ImportError:
    HAS_CUDA_BACKEND = False
    import warnings
    warnings.warn("ROI sampling CUDA extension not found. Using PyTorch fallback (slower).")

from .functions import roi_sampling

__all__ = ["roi_sampling", "HAS_CUDA_BACKEND"]
