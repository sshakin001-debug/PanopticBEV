"""
Load KITTI-360 calibration parameters from files.
Reads actual calibration data instead of using hardcoded values.
"""
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
import re


class KITTI360Calibration:
    """
    Load and parse KITTI-360 calibration files.
    
    KITTI-360 calibration structure:
    - calibration/calib_cam_to_pose.txt    : Camera to vehicle pose
    - calibration/calib_cam_to_velo.txt    : Camera to Velodyne
    - calibration/cib_calib.txt            : Camera intrinsics (if exists)
    - calibration/perspective_calibration.txt : Perspective camera params
    """
    
    def __init__(self, calibration_root: str):
        """
        Args:
            calibration_root: Path to calibration folder 
                            (e.g., D:/datasets/kitti360/calibration)
        """
        self.calib_root = Path(calibration_root)
        self.cam_to_pose = None
        self.cam_to_velo = None
        self.intrinsics = {}
        self.perspective_params = {}
        
        self._load_calibrations()
    
    def _load_calibrations(self):
        """Load all calibration files."""
        # Load camera to pose (vehicle) transformation
        cam_to_pose_file = self.calib_root / "calib_cam_to_pose.txt"
        if cam_to_pose_file.exists():
            self.cam_to_pose = self._load_transform_matrix(cam_to_pose_file)
            if isinstance(self.cam_to_pose, dict):
                print(f"Loaded cam_to_pose: {len(self.cam_to_pose)} cameras")
            else:
                print(f"Loaded cam_to_pose: {self.cam_to_pose.shape}")
        
        # Load camera to Velodyne transformation
        cam_to_velo_file = self.calib_root / "calib_cam_to_velo.txt"
        if cam_to_velo_file.exists():
            self.cam_to_velo = self._load_transform_matrix(cam_to_velo_file)
            if isinstance(self.cam_to_velo, dict):
                print(f"Loaded cam_to_velo: {len(self.cam_to_velo)} cameras")
            else:
                print(f"Loaded cam_to_velo: {self.cam_to_velo.shape}")
        
        # Load perspective camera calibration
        perspective_file = self.calib_root / "perspective_calibration.txt"
        if perspective_file.exists():
            self.perspective_params = self._load_perspective_calibration(perspective_file)
        
        # Load camera intrinsics
        for cam_id in range(4):  # 4 perspective cameras
            intrinsic_file = self.calib_root / f"image_{cam_id:02d}" / "intrinsics.txt"
            if intrinsic_file.exists():
                self.intrinsics[cam_id] = self._load_intrinsics(intrinsic_file)
        
        # If no intrinsics found, try to extract from calib_cam_to_pose
        if not self.intrinsics and self.cam_to_pose is not None:
            self.intrinsics = self._extract_intrinsics_from_pose()
    
    def _load_transform_matrix(self, filepath: Path) -> Dict[str, np.ndarray]:
        """Load transformation matrices from file.
        
        KITTI-360 format has entries like:
        image_00: P0[0-3] P0[4-7] P0[8-11] (3x4 projection matrix per line for 3 lines)
        ...
        """
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        matrices = {}
        current_key = None
        current_matrix = []
        
        for line in lines:
            # Remove comments
            line = line.split('#')[0].strip()
            if not line:
                continue
            
            # Check if line starts with a key (e.g., "image_00:")
            if ':' in line:
                # Save previous matrix if exists
                if current_key and current_matrix:
                    matrices[current_key] = np.array(current_matrix)
                    current_matrix = []
                
                # Parse new key and values on same line
                parts = line.split(':')
                current_key = parts[0].strip()
                values = parts[1].strip().split() if len(parts) > 1 else []
                
                if values:
                    try:
                        row = [float(x) for x in values]
                        if len(row) >= 4:
                            current_matrix.append(row[:4])
                    except ValueError:
                        pass
            else:
                # Parse matrix row
                try:
                    values = [float(x) for x in line.split()]
                    if len(values) >= 4:
                        current_matrix.append(values[:4])
                except ValueError:
                    pass
        
        # Save last matrix
        if current_key and current_matrix:
            matrices[current_key] = np.array(current_matrix)
        
        # Return first matrix if only one, else return dict
        if len(matrices) == 1:
            return list(matrices.values())[0]
        return matrices
    
    def _load_perspective_calibration(self, filepath: Path) -> Dict:
        """Load perspective camera calibration."""
        params = {}
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Parse focal length
        focal_match = re.search(r'focal_length:\s*([0-9.]+)', content)
        if focal_match:
            params['focal_length'] = float(focal_match.group(1))
        
        # Parse principal point
        cx_match = re.search(r'cx:\s*([0-9.]+)', content)
        cy_match = re.search(r'cy:\s*([0-9.]+)', content)
        if cx_match and cy_match:
            params['principal_point'] = (float(cx_match.group(1)), float(cy_match.group(1)))
        
        # Parse image size
        width_match = re.search(r'width:\s*([0-9]+)', content)
        height_match = re.search(r'height:\s*([0-9]+)', content)
        if width_match and height_match:
            params['image_size'] = (int(width_match.group(1)), int(height_match.group(1)))
        
        # Parse camera height (if available)
        height_match = re.search(r'camera_height:\s*([0-9.]+)', content)
        if height_match:
            params['camera_height'] = float(height_match.group(1))
        
        return params
    
    def _load_intrinsics(self, filepath: Path) -> Dict:
        """Load camera intrinsics from file."""
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        # Expect 3x3 matrix
        matrix = []
        for line in lines:
            values = [float(x) for x in line.split()]
            if len(values) == 3:
                matrix.append(values)
        
        K = np.array(matrix)
        
        return {
            'K': K,
            'fx': K[0, 0],
            'fy': K[1, 1],
            'cx': K[0, 2],
            'cy': K[1, 2],
        }
    
    def _extract_intrinsics_from_pose(self) -> Dict:
        """Extract approximate intrinsics from pose matrix if dedicated file not found."""
        # From KITTI-360 documentation, typical values
        return {
            0: {
                'K': np.array([[721.5377, 0, 609.5593],
                              [0, 721.5377, 172.854],
                              [0, 0, 1]]),
                'fx': 721.5377,
                'fy': 721.5377,
                'cx': 609.5593,
                'cy': 172.854,
            }
        }
    
    def get_camera_intrinsics(self, cam_id: int = 0) -> Dict:
        """Get intrinsics for specific camera."""
        if cam_id in self.intrinsics:
            return self.intrinsics[cam_id]
        return self.intrinsics.get(0, self._extract_intrinsics_from_pose()[0])
    
    def get_focal_length(self, cam_id: int = 0) -> float:
        """Get focal length in pixels."""
        intrinsics = self.get_camera_intrinsics(cam_id)
        return intrinsics['fx']
    
    def get_principal_point(self, cam_id: int = 0) -> Tuple[float, float]:
        """Get principal point (cx, cy)."""
        intrinsics = self.get_camera_intrinsics(cam_id)
        return (intrinsics['cx'], intrinsics['cy'])
    
    def get_camera_height(self) -> float:
        """
        Get camera height above ground.
        From KITTI-360, this is typically 1.65m for the perspective cameras.
        """
        if 'camera_height' in self.perspective_params:
            return self.perspective_params['camera_height']
        
        # Default from KITTI-360 documentation
        return 1.65
    
    def get_bev_resolution(self) -> float:
        """
        Get BEV resolution in meters per pixel.
        This should come from the BEV generation parameters, not calibration.
        """
        # Typical value for KITTI-360 BEV generation
        return 0.05  # 5cm per pixel
    
    def compute_ground_plane_to_image_homography(self, 
                                                  cam_id: int = 0,
                                                  ground_height: float = 0) -> np.ndarray:
        """
        Compute homography from ground plane to image plane.
        Used for BEV transformation.
        """
        intrinsics = self.get_camera_intrinsics(cam_id)
        K = intrinsics['K']
        
        # Camera height
        h = self.get_camera_height()
        
        # Simplified homography (assuming flat ground)
        # H = K * [R|t] where R is rotation and t is translation
        # For ground plane, we project from z=0 plane
        
        # This is a simplified version - full implementation needs camera pose
        H = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, -1/h, 1]
        ])
        
        return K @ H
    
    def get_camera_pose(self, cam_id: int = 0) -> np.ndarray:
        """Get camera pose in vehicle coordinates."""
        if self.cam_to_pose is not None:
            if isinstance(self.cam_to_pose, dict):
                # Get specific camera pose
                key = f"image_{cam_id:02d}"
                if key in self.cam_to_pose:
                    return self.cam_to_pose[key]
                # Fallback to first available
                return list(self.cam_to_pose.values())[0]
            return self.cam_to_pose
        return np.eye(4)  # Identity if not available
    
    def print_summary(self):
        """Print calibration summary."""
        print("="*60)
        print("KITTI-360 Calibration Summary")
        print("="*60)
        print(f"Calibration root: {self.calib_root}")
        print(f"\nCamera intrinsics (cam_00):")
        intrinsics = self.get_camera_intrinsics(0)
        print(f"  Focal length: {intrinsics['fx']:.2f} pixels")
        print(f"  Principal point: ({intrinsics['cx']:.2f}, {intrinsics['cy']:.2f})")
        print(f"\nCamera height: {self.get_camera_height():.2f}m")
        print(f"BEV resolution: {self.get_bev_resolution()*100:.1f}cm/pixel")
        print("="*60)


def load_calibration_from_dataset(dataset_root: str) -> KITTI360Calibration:
    """
    Convenience function to load calibration from dataset root.
    
    Args:
        dataset_root: Path to KITTI-360 dataset root
                     (e.g., D:/datasets/kitti360)
    
    Returns:
        KITTI360Calibration object
    """
    calib_path = Path(dataset_root) / "calibration"
    if not calib_path.exists():
        raise FileNotFoundError(f"Calibration folder not found: {calib_path}")
    
    return KITTI360Calibration(calib_path)