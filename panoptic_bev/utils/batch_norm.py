"""
Batch Normalization compatibility layer.
Uses inplace_abn if available, falls back to standard PyTorch BatchNorm2d.
"""
import torch.nn as nn

try:
    from inplace_abn import ABN as InplaceABN
    from inplace_abn import InPlaceABN, InPlaceABNSync, active_group, set_active_group
    HAS_INPLACE_ABN = True
    print("[OK] Using inplace_abn for faster training")
except ImportError:
    HAS_INPLACE_ABN = False
    print("[WARN] inplace_abn not available, using standard BatchNorm2d")


if HAS_INPLACE_ABN:
    # Use the original inplace_abn classes directly
    ABN = InplaceABN
else:
    # Create a fallback ABN that inherits from BatchNorm2d for proper attribute access
    class ABN(nn.BatchNorm2d):
        """
        Activated Batch Normalization.
        Falls back to BatchNorm2d + activation if inplace_abn is not available.
        Inherits from BatchNorm2d to provide all expected attributes (running_mean, running_var, weight, bias, etc.)
        """
        def __init__(self, num_features, activation="leaky_relu", activation_param=0.01, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
            super().__init__(num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats)
            self.activation = activation
            self.activation_param = activation_param

        def forward(self, x):
            # Apply batch norm
            x = super().forward(x)
            
            # Apply activation
            if self.activation == "leaky_relu":
                import torch.nn.functional as F
                x = F.leaky_relu(x, negative_slope=self.activation_param)
            elif self.activation == "elu":
                import torch.nn.functional as F
                x = F.elu(x, alpha=self.activation_param)
            elif self.activation == "relu":
                import torch.nn.functional as F
                x = F.relu(x)
            
            return x


# Wrapper classes for compatibility
class InPlaceABNWrapper(ABN):
    """Wrapper for compatibility with code expecting InPlaceABN."""
    pass


class InPlaceABNSyncWrapper(ABN):
    """Wrapper for compatibility with code expecting InPlaceABNSync."""
    pass
