"""
PanopticBEV Models Package

This package provides the neural network models for PanopticBEV,
including the main PanopticBevNet and the distance estimation head.
"""

try:
    from .panoptic_bev import PanopticBevNet, NETWORK_INPUTS
except ImportError:
    pass

try:
    from .distance_head import (
        DistanceHead,
        PanopticBEVWithDistance,
        create_model_with_distance
    )
except ImportError:
    pass

__all__ = [
    # Main model
    'PanopticBevNet',
    'NETWORK_INPUTS',
    # Distance head
    'DistanceHead',
    'PanopticBEVWithDistance',
    'create_model_with_distance',
]