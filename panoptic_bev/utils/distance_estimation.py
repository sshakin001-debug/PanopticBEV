"""
Distance estimation from BEV panoptic segmentation.
Extracts lateral (left/right) and longitudinal (forward) distances from ego vehicle.
"""
import torch
import numpy as np
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass

from panoptic_bev.utils.kitti360_calibration import (
    KITTI360Calibration, 
    load_calibration_from_dataset
)


@dataclass
class DistanceMeasurement:
    """Distance measurement for an object."""
    object_id: int
    class_name: str
    lateral_distance: float  # meters, positive=right, negative=left
    longitudinal_distance: float  # meters, positive=forward
    centroid_x: float  # BEV coordinates
    centroid_y: float  # BEV coordinates
    bbox: Tuple[float, float, float, float]  # min_x, min_y, max_x, max_y in meters


class BEVDistanceEstimator:
    """
    Estimate distances from BEV panoptic segmentation using REAL calibration.
    
    Uses calibration parameters to convert pixel coordinates to real-world meters.
    """
    
    # Class IDs from KITTI-360
    CLASS_NAMES = {
        0: 'road',
        1: 'sidewalk', 
        2: 'building',
        3: 'wall',
        4: 'fence',
        5: 'pole',
        6: 'traffic_sign',
        7: 'vegetation',
        8: 'terrain',
        9: 'sky',
        10: 'person',
        11: 'rider',
        12: 'car',
        13: 'truck',
        14: 'bus',
        15: 'train',
        16: 'motorcycle',
        17: 'bicycle',
        18: 'garage',
        19: 'gate',
        20: 'stop',
        21: 'smallpole',
        22: 'lamp',
        23: 'trash_bin',
        24: 'vending_machine',
        25: 'box',
        26: 'unknown_construction',
        27: 'unknown_vehicle',
        28: 'unknown_object',
        29: 'license_plate',
    }
    
    # Dynamic objects we care about for distance
    DYNAMIC_CLASSES = [10, 11, 12, 13, 14, 15, 16, 17]  # person, rider, car, truck, etc.
    
    def __init__(self, 
                 calibration: Optional[KITTI360Calibration] = None,
                 dataset_root: Optional[str] = None,
                 bev_resolution: Optional[float] = None,
                 camera_height: Optional[float] = None,
                 bev_size: Tuple[int, int] = (512, 512),
                 ego_position: Tuple[int, int] = None):
        """
        Args:
            calibration: KITTI360Calibration object (preferred)
            dataset_root: Path to dataset root (will load calibration automatically)
            bev_resolution: Override BEV resolution (meters per pixel)
            camera_height: Override camera height (meters)
            bev_size: (height, width) of BEV image
            ego_position: (x, y) pixel coordinates of ego vehicle in BEV
                         If None, assumes center-bottom of image
        """
        # Load calibration if provided or from dataset root
        if calibration is not None:
            self.calibration = calibration
        elif dataset_root is not None:
            self.calibration = load_calibration_from_dataset(dataset_root)
        else:
            # Use default values if no calibration provided
            self.calibration = None
            self.bev_resolution = bev_resolution or 0.05
            self.camera_height = camera_height or 1.65
            self.focal_length = 721.5377
            print("Warning: No calibration provided, using default values")
            print(f"  Focal length: {self.focal_length:.2f}px")
            print(f"  Camera height: {self.camera_height:.2f}m")
            print(f"  BEV resolution: {self.bev_resolution*100:.1f}cm/px")
            self.bev_size = bev_size
            if ego_position is None:
                self.ego_x = bev_size[1] // 2
                self.ego_y = bev_size[0] - 10
            else:
                self.ego_x, self.ego_y = ego_position
            return
        
        # Get parameters from calibration
        self.bev_resolution = bev_resolution or self.calibration.get_bev_resolution()
        self.camera_height = camera_height or self.calibration.get_camera_height()
        self.focal_length = self.calibration.get_focal_length()
        
        self.bev_size = bev_size
        
        # Ego position
        if ego_position is None:
            self.ego_x = bev_size[1] // 2
            self.ego_y = bev_size[0] - 10
        else:
            self.ego_x, self.ego_y = ego_position
        
        # Print calibration info
        print(f"Distance estimator initialized:")
        print(f"  Focal length: {self.focal_length:.2f}px")
        print(f"  Camera height: {self.camera_height:.2f}m")
        print(f"  BEV resolution: {self.bev_resolution*100:.1f}cm/px")
    
    def panoptic_to_distances(self, 
                             panoptic_bev: np.ndarray,
                             return_visualization: bool = False) -> Dict:
        """
        Convert panoptic BEV segmentation to distance measurements.
        
        Args:
            panoptic_bev: HxW array with instance IDs (semantic_id * 1000 + instance_id)
            return_visualization: If True, return visualization images
            
        Returns:
            Dictionary with distance measurements and optional visualizations
        """
        # Extract semantic and instance masks
        semantic_mask = panoptic_bev // 1000
        instance_mask = panoptic_bev % 1000
        
        measurements = []
        distance_map = np.zeros((*panoptic_bev.shape, 2), dtype=np.float32)  # (lateral, longitudinal)
        
        # Process each instance
        unique_instances = np.unique(panoptic_bev)
        
        for instance_id in unique_instances:
            if instance_id == 0:  # background
                continue
            
            semantic_id = instance_id // 1000
            instance_num = instance_id % 1000
            
            # Get mask for this instance
            mask = (panoptic_bev == instance_id)
            
            if not np.any(mask):
                continue
            
            # Calculate centroid in pixel coordinates
            y_coords, x_coords = np.where(mask)
            centroid_x = np.mean(x_coords)
            centroid_y = np.mean(y_coords)
            
            # Convert to meters relative to ego
            lateral_meters = (centroid_x - self.ego_x) * self.bev_resolution
            longitudinal_meters = (self.ego_y - centroid_y) * self.bev_resolution
            
            # Get bounding box in meters
            min_x = np.min(x_coords) * self.bev_resolution
            max_x = np.max(x_coords) * self.bev_resolution
            min_y = np.min(y_coords) * self.bev_resolution
            max_y = np.max(y_coords) * self.bev_resolution
            
            # Only keep objects in front and within reasonable distance
            if longitudinal_meters > 0 and longitudinal_meters < 100:  # 0-100m ahead
                
                measurement = DistanceMeasurement(
                    object_id=int(instance_id),
                    class_name=self.CLASS_NAMES.get(int(semantic_id), 'unknown'),
                    lateral_distance=float(lateral_meters),
                    longitudinal_distance=float(longitudinal_meters),
                    centroid_x=float(centroid_x),
                    centroid_y=float(centroid_y),
                    bbox=(float(min_x), float(min_y), float(max_x), float(max_y))
                )
                measurements.append(measurement)
                
                # Fill distance map
                distance_map[mask, 0] = lateral_meters
                distance_map[mask, 1] = longitudinal_meters
        
        # Sort by longitudinal distance (closest first)
        measurements.sort(key=lambda x: x.longitudinal_distance)
        
        result = {
            'measurements': measurements,
            'ego_position': (self.ego_x, self.ego_y),
            'distance_map': distance_map,
            'num_objects': len(measurements)
        }
        
        if return_visualization:
            result['visualization'] = self._create_visualization(
                panoptic_bev, measurements, distance_map
            )
        
        return result
    
    def _create_visualization(self, 
                             panoptic_bev: np.ndarray,
                             measurements: List[DistanceMeasurement],
                             distance_map: np.ndarray) -> np.ndarray:
        """Create visualization of distance measurements."""
        import cv2
        
        # Create RGB visualization
        vis = np.zeros((*panoptic_bev.shape, 3), dtype=np.uint8)
        
        # Color by longitudinal distance (red=close, green=far)
        longitudinal_norm = np.clip(distance_map[:, :, 1] / 50.0, 0, 1)  # normalize to 50m
        
        vis[:, :, 0] = (255 * (1 - longitudinal_norm)).astype(np.uint8)  # Red channel
        vis[:, :, 1] = (255 * longitudinal_norm).astype(np.uint8)        # Green channel
        
        # Draw ego position
        cv2.circle(vis, (self.ego_x, self.ego_y), 10, (255, 255, 255), -1)
        cv2.circle(vis, (self.ego_x, self.ego_y), 10, (0, 0, 0), 2)
        
        # Draw measurement lines and labels
        for meas in measurements[:20]:  # Top 20 closest objects
            cx, cy = int(meas.centroid_x), int(meas.centroid_y)
            
            # Draw line from ego to object
            cv2.line(vis, (self.ego_x, self.ego_y), (cx, cy), (255, 255, 0), 2)
            
            # Draw label with distance
            label = f"{meas.class_name[:3]}: {meas.longitudinal_distance:.1f}m, {meas.lateral_distance:+.1f}m"
            cv2.putText(vis, label, (cx + 10, cy), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return vis
    
    def get_closest_objects(self, 
                           measurements: List[DistanceMeasurement],
                           n: int = 5,
                           class_filter: List[str] = None) -> List[DistanceMeasurement]:
        """Get n closest objects, optionally filtered by class."""
        filtered = measurements
        
        if class_filter:
            filtered = [m for m in measurements if m.class_name in class_filter]
        
        return filtered[:n]
    
    def get_lateral_clearance(self, 
                             measurements: List[DistanceMeasurement],
                             longitudinal_range: Tuple[float, float] = (0, 30),
                             vehicle_width: float = 2.0) -> Dict:
        """
        Calculate lateral clearance (free space) in each direction.
        
        Returns:
            Dictionary with left_clearance, right_clearance, is_safe
        """
        # Filter objects in front longitudinal range
        front_objects = [
            m for m in measurements 
            if longitudinal_range[0] <= m.longitudinal_distance <= longitudinal_range[1]
        ]
        
        # Find closest objects on left and right
        left_objects = [m for m in front_objects if m.lateral_distance < -vehicle_width/2]
        right_objects = [m for m in front_objects if m.lateral_distance > vehicle_width/2]
        
        left_clearance = abs(max([m.lateral_distance for m in left_objects], default=-10))
        right_clearance = abs(min([m.lateral_distance for m in right_objects], default=10))
        
        return {
            'left_clearance_meters': left_clearance,
            'right_clearance_meters': right_clearance,
            'min_clearance': min(left_clearance, right_clearance),
            'is_safe': min(left_clearance, right_clearance) > 0.5,  # 50cm safety margin
            'closest_left': left_objects[0] if left_objects else None,
            'closest_right': right_objects[0] if right_objects else None,
        }


class DistanceLoss(torch.nn.Module):
    """
    Training loss for distance estimation.
    Can be added to PanopticBEV training to predict distances.
    """
    
    def __init__(self, bev_resolution: float = 0.05):
        super().__init__()
        self.bev_resolution = bev_resolution
        self.l1_loss = torch.nn.L1Loss()
        self.mse_loss = torch.nn.MSELoss()
    
    def forward(self, 
                pred_distances: torch.Tensor,
                target_panoptic: torch.Tensor,
                target_depth: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            pred_distances: Bx2xHxW tensor (lateral, longitudinal)
            target_panoptic: BxHxW panoptic segmentation
            target_depth: Optional BxHxW depth ground truth
        """
        # Create target distance maps from panoptic
        batch_size = pred_distances.size(0)
        device = pred_distances.device
        
        target_distances = torch.zeros_like(pred_distances)
        
        for b in range(batch_size):
            # Convert panoptic to distance map (simplified)
            # In practice, you'd use the actual 3D positions
            panoptic = target_panoptic[b].cpu().numpy()
            
            # Estimate distances from instance sizes and positions
            # This is a placeholder - real implementation needs 3D ground truth
            estimator = BEVDistanceEstimator(bev_resolution=self.bev_resolution)
            result = estimator.panoptic_to_distances(panoptic)
            
            target_distances[b, 0] = torch.from_numpy(result['distance_map'][:, :, 0]).to(device)
            target_distances[b, 1] = torch.from_numpy(result['distance_map'][:, :, 1]).to(device)
        
        # Compute loss
        l1 = self.l1_loss(pred_distances, target_distances)
        mse = self.mse_loss(pred_distances, target_distances)
        
        return l1 + 0.5 * mse


def create_estimator_from_dataset(dataset_root: str, **kwargs) -> BEVDistanceEstimator:
    """
    Factory function to create estimator from dataset root.
    
    Example:
        >>> estimator = create_estimator_from_dataset('D:/datasets/kitti360')
    """
    calibration = load_calibration_from_dataset(dataset_root)
    return BEVDistanceEstimator(calibration=calibration, **kwargs)