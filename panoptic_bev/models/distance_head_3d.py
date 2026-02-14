"""
Distance prediction head supervised by 3D ground truth.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class DistanceHead3D(nn.Module):
    """
    Predict 3D distances from BEV features.
    Supervised by accurate 3D bounding box ground truth.
    """
    
    def __init__(self, in_channels: int = 256, num_classes: int = 19):
        super().__init__()
        
        # Shared features
        self.shared_conv = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        # Distance regression (lateral, longitudinal)
        self.distance_conv = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 2, 1)  # [lateral, longitudinal]
        )
        
        # Uncertainty estimation (aleatoric uncertainty)
        self.uncertainty_conv = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 2, 1)  # [sigma_lat, sigma_long]
        )
        
        # Objectness (is there an object at this pixel?)
        self.objectness_conv = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 1),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, bev_features):
        """
        Args:
            bev_features: (B, C, H, W) BEV feature maps
        
        Returns:
            Dictionary with distance predictions
        """
        shared = self.shared_conv(bev_features)
        
        # Distance predictions (in meters)
        distances = self.distance_conv(shared)
        lateral = distances[:, 0]      # X: left/right
        longitudinal = distances[:, 1]  # Z: forward
        
        # Uncertainty (standard deviation in meters)
        log_variance = self.uncertainty_conv(shared)
        uncertainty_lat = torch.exp(log_variance[:, 0])
        uncertainty_long = torch.exp(log_variance[:, 1])
        
        # Objectness probability
        objectness = torch.sigmoid(self.objectness_conv(shared)[:, 0])
        
        return {
            'lateral': lateral,                    # (B, H, W)
            'longitudinal': longitudinal,          # (B, H, W)
            'uncertainty_lat': uncertainty_lat,    # (B, H, W)
            'uncertainty_long': uncertainty_long,  # (B, H, W)
            'objectness': objectness,              # (B, H, W)
        }


class DistanceLoss3D(nn.Module):
    """
    Loss function for 3D distance prediction.
    Uses accurate 3D ground truth from KITTI-360 annotations.
    """
    
    def __init__(self, 
                 use_uncertainty: bool = True,
                 min_distance: float = 0.0,
                 max_distance: float = 100.0):
        super().__init__()
        self.use_uncertainty = use_uncertainty
        self.min_dist = min_distance
        self.max_dist = max_distance
    
    def forward(self, 
                predictions: Dict[str, torch.Tensor],
                ground_truth_3d: List[Dict],  # List of 3D objects per sample
                bev_mask: Optional[torch.Tensor] = None):
        """
        Args:
            predictions: Output from DistanceHead3D
            ground_truth_3d: List of 3D object annotations per batch sample
            bev_mask: Valid BEV region mask
        
        Returns:
            Total loss
        """
        batch_size = predictions['lateral'].size(0)
        device = predictions['lateral'].device
        
        total_loss = torch.tensor(0.0, device=device)
        num_valid = 0
        
        for b in range(batch_size):
            # Get 3D ground truth for this sample
            objects_3d = ground_truth_3d[b] if b < len(ground_truth_3d) else []
            
            if not objects_3d:
                continue
            
            # Project 3D objects to BEV grid
            target_lat, target_long, valid_mask = self._project_3d_to_bev(
                objects_3d,
                predictions['lateral'].shape[1:],
                device
            )
            
            if not valid_mask.any():
                continue
            
            # Compute L1 loss for valid pixels
            pred_lat = predictions['lateral'][b]
            pred_long = predictions['longitudinal'][b]
            
            if self.use_uncertainty:
                # Aleatoric uncertainty weighted loss
                sigma_lat = predictions['uncertainty_lat'][b] + 1e-6
                sigma_long = predictions['uncertainty_long'][b] + 1e-6
                
                loss_lat = torch.abs(pred_lat - target_lat) / sigma_lat + torch.log(sigma_lat)
                loss_long = torch.abs(pred_long - target_long) / sigma_long + torch.log(sigma_long)
            else:
                loss_lat = torch.abs(pred_lat - target_lat)
                loss_long = torch.abs(pred_long - target_long)
            
            # Mask to valid regions
            loss = (loss_lat + loss_long) * valid_mask
            total_loss += loss.sum() / (valid_mask.sum() + 1e-6)
            num_valid += 1
        
        return total_loss / max(num_valid, 1)
    
    def _project_3d_to_bev(self, 
                          objects_3d: List,
                          bev_size: Tuple[int, int],
                          device: torch.device):
        """
        Project 3D object centers to BEV grid for supervision.
        """
        H, W = bev_size
        target_lat = torch.zeros(H, W, device=device)
        target_long = torch.zeros(H, W, device=device)
        valid_mask = torch.zeros(H, W, dtype=torch.bool, device=device)
        
        # BEV parameters (should match your BEV generation)
        bev_resolution = 0.05  # 5cm per pixel
        ego_x = W // 2
        ego_y = H - 10
        
        for obj in objects_3d:
            # Convert 3D to BEV pixels
            bev_x = int(ego_x + obj.lateral_distance / bev_resolution)
            bev_y = int(ego_y - obj.longitudinal_distance / bev_resolution)
            
            # Check bounds
            if 0 <= bev_x < W and 0 <= bev_y < H:
                # Mark region around object center
                radius = 2
                y_min = max(0, bev_y - radius)
                y_max = min(H, bev_y + radius + 1)
                x_min = max(0, bev_x - radius)
                x_max = min(W, bev_x + radius + 1)
                
                target_lat[y_min:y_max, x_min:x_max] = obj.lateral_distance
                target_long[y_min:y_max, x_min:x_max] = obj.longitudinal_distance
                valid_mask[y_min:y_max, x_min:x_max] = True
        
        return target_lat, target_long, valid_mask


# Integration with PanopticBEV
class PanopticBEVWith3DDistance(nn.Module):
    """
    Extended PanopticBEV with 3D distance prediction head.
    """
    
    def __init__(self, base_model, enable_distance: bool = True):
        super().__init__()
        self.base_model = base_model
        self.enable_distance = enable_distance
        
        if enable_distance:
            # Add distance head (assumes base_model outputs BEV features)
            self.distance_head = DistanceHead3D(in_channels=256)
    
    def forward(self, images, targets=None, return_3d_distances: bool = False):
        # Get base model outputs
        outputs = self.base_model(images, targets)
        
        if self.enable_distance and return_3d_distances:
            # Extract BEV features from base model
            # This depends on your base architecture
            bev_features = outputs.get('bev_features')
            
            if bev_features is not None:
                distance_outputs = self.distance_head(bev_features)
                outputs['3d_distances'] = distance_outputs
        
        return outputs


def test_distance_head():
    """Test the distance head module."""
    print("Testing DistanceHead3D")
    print("="*60)
    
    # Create model
    model = DistanceHead3D(in_channels=256)
    
    # Create dummy input
    batch_size = 2
    bev_features = torch.randn(batch_size, 256, 128, 128)
    
    # Forward pass
    outputs = model(bev_features)
    
    print(f"Input shape: {bev_features.shape}")
    print(f"Output shapes:")
    for key, value in outputs.items():
        print(f"  {key}: {value.shape}")
    
    # Test loss
    loss_fn = DistanceLoss3D(use_uncertainty=True)
    
    # Create dummy ground truth (empty for now)
    ground_truth = [[] for _ in range(batch_size)]
    loss = loss_fn(outputs, ground_truth)
    print(f"\nLoss (empty GT): {loss.item():.4f}")
    
    print("\n" + "="*60)
    print("Test passed!")


if __name__ == '__main__':
    test_distance_head()