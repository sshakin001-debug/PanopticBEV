"""
Distance prediction head for PanopticBEV.
Adds lateral and longitudinal distance estimation to the model.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
from collections import OrderedDict


class DistanceHead(nn.Module):
    """
    Predicts lateral and longitudinal distance maps from BEV features.
    """
    
    def __init__(self, in_channels: int = 256, hidden_channels: int = 128):
        super().__init__()
        
        # Reduce channels
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Distance regression
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_channels)
        
        # Output: 2 channels (lateral distance, longitudinal distance)
        self.distance_conv = nn.Conv2d(hidden_channels, 2, 1)
        
        # Uncertainty estimation (optional)
        self.uncertainty_conv = nn.Conv2d(hidden_channels, 2, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: BEV features (B, C, H, W)
        
        Returns:
            Dictionary with:
                - distances: (B, 2, H, W) tensor of (lateral, longitudinal) in meters
                - uncertainty: (B, 2, H, W) predicted uncertainty
        """
        # Feature processing
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        # Predict distances
        distances = self.distance_conv(x)
        
        # Predict uncertainty (aleatoric)
        uncertainty = torch.exp(self.uncertainty_conv(x))
        
        return {
            'distances': distances,
            'uncertainty': uncertainty
        }


class PanopticBEVWithDistance(nn.Module):
    """
    Extended PanopticBEV with distance estimation.
    Wraps the base PanopticBevNet model and adds distance prediction.
    
    This wrapper captures BEV features during the forward pass to avoid
    redundant computation.
    """
    
    def __init__(self, base_model, distance_head_channels: int = 256):
        super().__init__()
        self.base_model = base_model
        self.distance_head = DistanceHead(distance_head_channels)
        
        # Whether to use distance prediction during inference
        self.predict_distances = True
        
        # Get BEV feature channels from base model
        # The transformer outputs ms_bev which has 256 channels by default
        self.bev_feature_channels = distance_head_channels
        
        # Cache for BEV features
        self._cached_bev_features = None
        
        # Register forward hook to capture BEV features
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks to capture BEV features during forward pass."""
        def hook_fn(module, input, output):
            # output is (ms_bev, vf_logits_list, v_region_logits_list, f_region_logits_list)
            if isinstance(output, tuple) and len(output) >= 1:
                ms_bev = output[0]
                # ms_bev is a list of feature maps at different scales
                # Use the highest resolution (first) for distance prediction
                if isinstance(ms_bev, (list, tuple)) and len(ms_bev) > 0:
                    self._cached_bev_features = ms_bev[0]
                else:
                    self._cached_bev_features = ms_bev
        
        # Register hook on the transformer module
        if hasattr(self.base_model, 'transformer'):
            self.base_model.transformer.register_forward_hook(hook_fn)
    
    def forward(self, img, bev_msk=None, front_msk=None, weights_msk=None, 
                cat=None, iscrowd=None, bbx=None, calib=None,
                do_loss=False, do_prediction=False):
        """
        Forward pass with distance estimation.
        
        Args:
            All arguments from base PanopticBevNet
            
        Returns:
            loss, result, stats (same as base model) with distance outputs added
        """
        # Clear cached features
        self._cached_bev_features = None
        
        # Get base model outputs
        loss, result, stats = self.base_model(
            img, bev_msk, front_msk, weights_msk, cat, iscrowd, bbx, calib,
            do_loss, do_prediction
        )
        
        if (self.predict_distances or self.training) and self._cached_bev_features is not None:
            # Predict distances using cached BEV features
            distance_outputs = self.distance_head(self._cached_bev_features)
            result['distances'] = distance_outputs['distances']
            result['distance_uncertainty'] = distance_outputs['uncertainty']
        
        return loss, result, stats
    
    def enable_distance_prediction(self, enabled: bool = True):
        """Enable or disable distance prediction."""
        self.predict_distances = enabled


def create_model_with_distance(config, base_model_class, **kwargs):
    """
    Factory function to create PanopticBEV model with distance head.
    
    Args:
        config: Model configuration
        base_model_class: The base PanopticBevNet class
        **kwargs: Additional arguments for base model
        
    Returns:
        PanopticBEVWithDistance model
    """
    # Create base model
    base_model = base_model_class(**kwargs)
    
    # Wrap with distance head
    model = PanopticBEVWithDistance(base_model)
    
    return model