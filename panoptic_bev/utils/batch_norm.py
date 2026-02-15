"""
Batch Normalization compatibility layer.
Uses inplace_abn if available, falls back to standard PyTorch BatchNorm.
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


# Global variable to track active group for synchronized batch normalization
_active_group = None


def active_group():
    """Get the currently active group for synchronized batch normalization."""
    global _active_group
    return _active_group


def set_active_group(group):
    """Set the active group for synchronized batch normalization."""
    global _active_group
    _active_group = group


if HAS_INPLACE_ABN:
    # Use the original inplace_abn classes directly
    ABN = InplaceABN
else:
    # Create a fallback ABN that works with both 2D and 3D inputs
    class ABN(nn.Module):
        """
        Activated Batch Normalization.
        Falls back to BatchNorm2d/3d + activation if inplace_abn is not available.
        Automatically detects input dimensionality and uses appropriate BatchNorm.
        """
        def __init__(self, num_features, activation="leaky_relu", activation_param=0.01, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
            super().__init__()
            self.num_features = num_features
            self.activation = activation
            self.activation_param = activation_param
            self.eps = eps
            self.momentum = momentum
            self.affine = affine
            self.track_running_stats = track_running_stats
            
            # Create both 2D and 3D batch norms, will use appropriate one based on input
            self.bn2d = nn.BatchNorm2d(num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats)
            self.bn3d = nn.BatchNorm3d(num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats)
        
        @property
        def running_mean(self):
            return self.bn2d.running_mean
        
        @property
        def running_var(self):
            return self.bn2d.running_var
        
        @property
        def weight(self):
            return self.bn2d.weight
        
        @property
        def bias(self):
            return self.bn2d.bias
        
        def reset_parameters(self):
            """Reset parameters of both batch norm layers."""
            self.bn2d.reset_parameters()
            self.bn3d.reset_parameters()

        def forward(self, x):
            # Choose appropriate batch norm based on input dimensions
            if x.dim() == 4:  # 2D input (N, C, H, W)
                x = self.bn2d(x)
            elif x.dim() == 5:  # 3D input (N, C, D, H, W)
                x = self.bn3d(x)
            else:
                raise ValueError(f"ABN expected 4D or 5D input, got {x.dim()}D")
            
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
