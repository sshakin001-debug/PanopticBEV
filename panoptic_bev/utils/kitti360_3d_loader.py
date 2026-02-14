"""
Load KITTI-360 3D bounding boxes from XML format.
Actual format is space-separated text (KITTI format), not HTML.
"""
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class Object3D:
    """3D object with accurate ground truth position."""
    object_id: int
    class_name: str
    class_id: int
    truncated: float
    occluded: int
    alpha: float
    bbox2d: np.ndarray  # [left, top, right, bottom]
    dimensions: np.ndarray  # [height, width, length]
    center: np.ndarray  # [x, y, z] in camera coordinates
    rotation_y: float
    score: float
    
    @property
    def lateral_distance(self) -> float:
        """X axis: positive = right, negative = left."""
        return float(self.center[0])
    
    @property
    def longitudinal_distance(self) -> float:
        """Z axis: positive = forward."""
        return float(self.center[2])
    
    @property
    def height_above_ground(self) -> float:
        """Y axis: negative = up."""
        return float(-self.center[1])
    
    @property
    def euclidean_distance(self) -> float:
        """Straight-line distance from camera."""
        return float(np.linalg.norm(self.center))
    
    @property
    def size(self) -> np.ndarray:
        """Get size as [length, width, height] for compatibility."""
        # dimensions is [height, width, length], we want [length, width, height]
        return np.array([self.dimensions[2], self.dimensions[1], self.dimensions[0]])
    
    @property
    def rotation(self) -> np.ndarray:
        """Get rotation as [yaw, pitch, roll] for compatibility."""
        return np.array([0, 0, self.rotation_y])
    
    @property
    def corners_3d(self) -> np.ndarray:
        """Get 8 corners of 3D bounding box."""
        h, w, l = self.dimensions  # height, width, length
        # Corners in object coordinates
        corners = np.array([
            [l/2, h/2, w/2], [l/2, h/2, -w/2], [l/2, -h/2, w/2], [l/2, -h/2, -w/2],
            [-l/2, h/2, w/2], [-l/2, h/2, -w/2], [-l/2, -h/2, w/2], [-l/2, -h/2, -w/2]
        ])
        
        # Rotate around Y axis
        yaw = self.rotation_y
        R = np.array([
            [np.cos(yaw), 0, np.sin(yaw)],
            [0, 1, 0],
            [-np.sin(yaw), 0, np.cos(yaw)]
        ])
        corners = corners @ R.T
        
        # Translate
        corners += self.center
        return corners


class KITTI3603DLoader:
    """
    Load 3D bounding boxes from KITTI-360 XML annotation files.
    Format: KITTI tracking format with 3D bounding boxes.
    """
    
    # KITTI-360 class mapping
    CLASS_NAMES = {
        'road': 0, 'sidewalk': 1, 'building': 2, 'wall': 3, 'fence': 4,
        'pole': 5, 'traffic_sign': 6, 'vegetation': 7, 'terrain': 8, 'sky': 9,
        'person': 10, 'rider': 11, 'car': 12, 'truck': 13, 'bus': 14,
        'train': 15, 'motorcycle': 16, 'bicycle': 17, 'garage': 18, 'gate': 19,
        'stop': 20, 'smallpole': 21, 'lamp': 22, 'trash_bin': 23, 'vending_machine': 24,
        'box': 25, 'unknown_construction': 26, 'unknown_vehicle': 27, 'unknown_object': 28,
        'license_plate': 29
    }
    
    DYNAMIC_CLASSES = ['person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']
    
    def __init__(self, 
                 bboxes_root: str = r"D:\datasets\kitti360\data_3d_bboxes",
                 use_train_full: bool = True,
                 use_cache: bool = True):
        """
        Args:
            bboxes_root: Path to data_3d_bboxes
            use_train_full: Use train_full vs train
            use_cache: Cache loaded annotations for faster access
        """
        self.bboxes_root = Path(bboxes_root)
        self.split_dir = "train_full" if use_train_full else "train"
        self.annotations_dir = self.bboxes_root / self.split_dir
        self.use_cache = use_cache
        self.cache = {}
        
        if not self.annotations_dir.exists():
            raise FileNotFoundError(f"Annotations not found: {self.annotations_dir}")
        
        # Index available sequences
        self.sequences = self._index_sequences()
        print(f"Found {len(self.sequences)} sequences in {self.split_dir}")
    
    def _index_sequences(self) -> Dict[str, Path]:
        """Index all available sequence annotation files."""
        sequences = {}
        
        # Look for .xml files (which are actually text files in KITTI format)
        for seq_file in self.annotations_dir.glob("*.xml"):
            seq_name = seq_file.stem  # Remove .xml extension
            sequences[seq_name] = seq_file
        
        # Also check for .txt files
        for seq_file in self.annotations_dir.glob("*.txt"):
            seq_name = seq_file.stem
            if seq_name not in sequences:
                sequences[seq_name] = seq_file
        
        return sequences
    
    def _parse_line(self, line: str) -> Optional[Dict]:
        """
        Parse one line of KITTI format annotation.
        
        Format (16+ fields):
        frame track_id type truncated occluded alpha 
        bbox_left bbox_top bbox_right bbox_bottom 
        height width length 
        x y z 
        rotation_y score
        """
        parts = line.strip().split()
        
        if len(parts) < 16:
            return None
        
        try:
            return {
                'frame': int(float(parts[0])),
                'track_id': int(float(parts[1])),
                'type': parts[2],
                'truncated': float(parts[3]),
                'occluded': int(float(parts[4])),
                'alpha': float(parts[5]),
                'bbox_left': float(parts[6]),
                'bbox_top': float(parts[7]),
                'bbox_right': float(parts[8]),
                'bbox_bottom': float(parts[9]),
                'height': float(parts[10]),
                'width': float(parts[11]),
                'length': float(parts[12]),
                'x': float(parts[13]),
                'y': float(parts[14]),
                'z': float(parts[15]),
                'rotation_y': float(parts[16]) if len(parts) > 16 else 0.0,
                'score': float(parts[17]) if len(parts) > 17 else 1.0,
            }
        except (ValueError, IndexError) as e:
            print(f"Error parsing line: {line[:50]}... - {e}")
            return None
    
    def load_sequence_annotations(self, sequence: str) -> Dict[int, List[Dict]]:
        """
        Load all annotations for a sequence, organized by frame.
        
        Returns:
            Dictionary mapping frame_number -> list of object annotations
        """
        cache_key = f"seq_{sequence}"
        if self.use_cache and cache_key in self.cache:
            return self.cache[cache_key]
        
        if sequence not in self.sequences:
            return {}
        
        seq_file = self.sequences[sequence]
        
        # Read and parse all lines
        annotations_by_frame = {}
        
        with open(seq_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parsed = self._parse_line(line)
                if parsed is None:
                    continue
                
                frame = parsed['frame']
                
                if frame not in annotations_by_frame:
                    annotations_by_frame[frame] = []
                
                annotations_by_frame[frame].append(parsed)
        
        if self.use_cache:
            self.cache[cache_key] = annotations_by_frame
        
        return annotations_by_frame
    
    def load_3d_bboxes(self, sequence: str, frame: int) -> List[Object3D]:
        """
        Load 3D bounding boxes for a specific frame.
        
        Args:
            sequence: Sequence name (e.g., '2013_05_28_drive_0000_sync')
            frame: Frame number
        
        Returns:
            List of Object3D instances
        """
        # Load all annotations for this sequence
        all_annotations = self.load_sequence_annotations(sequence)
        
        # Get annotations for specific frame
        frame_annotations = all_annotations.get(frame, [])
        
        # Convert to Object3D
        objects_3d = []
        for ann in frame_annotations:
            class_name = ann['type']
            class_id = self.CLASS_NAMES.get(class_name, -1)
            
            obj = Object3D(
                object_id=ann['track_id'],
                class_name=class_name,
                class_id=class_id,
                truncated=ann['truncated'],
                occluded=ann['occluded'],
                alpha=ann['alpha'],
                bbox2d=np.array([ann['bbox_left'], ann['bbox_top'], 
                                ann['bbox_right'], ann['bbox_bottom']]),
                dimensions=np.array([ann['height'], ann['width'], ann['length']]),
                center=np.array([ann['x'], ann['y'], ann['z']]),
                rotation_y=ann['rotation_y'],
                score=ann['score']
            )
            objects_3d.append(obj)
        
        return objects_3d
    
    def get_objects_in_fov(self, 
                          objects: List[Object3D],
                          max_distance: float = 100.0,
                          min_distance: float = 0.0,
                          lateral_range: float = 50.0) -> List[Object3D]:
        """
        Filter to front field of view.
        
        Args:
            objects: List of all objects
            max_distance: Maximum longitudinal distance
            min_distance: Minimum longitudinal distance
            lateral_range: Maximum lateral distance (absolute)
        
        Returns:
            Filtered list of objects
        """
        filtered = []
        for obj in objects:
            # Check longitudinal distance (Z axis, forward)
            if not (min_distance <= obj.longitudinal_distance <= max_distance):
                continue
            
            # Check lateral distance (X axis, left/right)
            if abs(obj.lateral_distance) > lateral_range:
                continue
            
            # Check if object is in front (Z > 0)
            if obj.longitudinal_distance <= 0:
                continue
            
            filtered.append(obj)
        
        # Sort by longitudinal distance
        filtered.sort(key=lambda x: x.longitudinal_distance)
        return filtered
    
    def get_dynamic_objects(self, objects: List[Object3D]) -> List[Object3D]:
        """Filter to dynamic objects (vehicles, pedestrians)."""
        return [o for o in objects if o.class_name in self.DYNAMIC_CLASSES]
    
    def get_closest_objects(self,
                           objects: List[Object3D],
                           n: int = 5,
                           class_filter: Optional[List[str]] = None) -> List[Object3D]:
        """Get n closest objects, optionally filtered by class."""
        if class_filter:
            objects = [o for o in objects if o.class_name in class_filter]
        
        # Already sorted by distance
        return objects[:n]
    
    def compute_lateral_clearance(self,
                                   objects: List[Object3D],
                                   longitudinal_range: Tuple[float, float] = (0, 50),
                                   vehicle_width: float = 2.0,
                                   safety_margin: float = 0.5) -> Dict:
        """
        Compute lateral clearance using accurate 3D positions.
        
        Args:
            objects: List of 3D objects
            longitudinal_range: Range ahead to check (min, max) in meters
            vehicle_width: Width of ego vehicle in meters
            safety_margin: Additional safety margin in meters
        
        Returns:
            Dictionary with clearance information
        """
        # Filter to relevant range
        front_objects = [
            o for o in objects
            if longitudinal_range[0] <= o.longitudinal_distance <= longitudinal_range[1]
        ]
        
        half_width = vehicle_width / 2
        
        # Objects on left (X < -half_width)
        left_objects = [o for o in front_objects if o.lateral_distance < -half_width]
        # Objects on right (X > half_width)  
        right_objects = [o for o in front_objects if o.lateral_distance > half_width]
        
        # Calculate clearance
        if left_objects:
            closest_left = max(left_objects, key=lambda o: o.lateral_distance)
            left_clearance = abs(closest_left.lateral_distance) - half_width
        else:
            closest_left = None
            left_clearance = float('inf')
        
        if right_objects:
            closest_right = min(right_objects, key=lambda o: o.lateral_distance)
            right_clearance = closest_right.lateral_distance - half_width
        else:
            closest_right = None
            right_clearance = float('inf')
        
        min_clearance = min(left_clearance, right_clearance)
        
        return {
            'left_clearance_meters': left_clearance,
            'right_clearance_meters': right_clearance,
            'min_clearance': min_clearance,
            'is_safe': min_clearance > safety_margin,
            'closest_left': closest_left,
            'closest_right': closest_right,
            'num_objects_in_range': len(front_objects),
            'vehicle_width': vehicle_width,
            'safety_margin': safety_margin
        }
    
    def project_to_bev(self, 
                      objects: List[Object3D],
                      bev_resolution: float = 0.05,
                      bev_size: Tuple[int, int] = (512, 512)) -> np.ndarray:
        """
        Project 3D objects to BEV image for visualization.
        
        Returns:
            BEV image with object centers marked
        """
        try:
            import cv2
        except ImportError:
            print("OpenCV not available, returning empty array")
            return np.zeros((*bev_size, 3), dtype=np.uint8)
        
        bev = np.zeros((*bev_size, 3), dtype=np.uint8)
        
        # Ego position at bottom center
        ego_x = bev_size[1] // 2
        ego_y = bev_size[0] - 10
        
        # Draw ego vehicle
        cv2.circle(bev, (ego_x, ego_y), 10, (0, 255, 0), -1)
        
        # Draw objects
        for obj in objects:
            # Convert 3D to BEV pixels
            bev_x = int(ego_x + obj.lateral_distance / bev_resolution)
            bev_y = int(ego_y - obj.longitudinal_distance / bev_resolution)
            
            # Check bounds
            if 0 <= bev_x < bev_size[1] and 0 <= bev_y < bev_size[0]:
                # Color by distance (red=close, green=far)
                dist_norm = min(obj.longitudinal_distance / 50.0, 1.0)
                color = (
                    int(255 * (1 - dist_norm)),  # R
                    int(255 * dist_norm),         # G
                    0                             # B
                )
                
                cv2.circle(bev, (bev_x, bev_y), 5, color, -1)
                
                # Draw label
                label = f"{obj.class_name}: {obj.longitudinal_distance:.1f}m"
                cv2.putText(bev, label, (bev_x + 10, bev_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        return bev


def test_loader():
    """Test the 3D loader."""
    print("Testing KITTI-360 3D Loader (XML Format)")
    print("="*60)
    
    loader = KITTI3603DLoader()
    
    # Test first sequence
    sequence = list(loader.sequences.keys())[0]
    print(f"\nSequence: {sequence}")
    print(f"File: {loader.sequences[sequence]}")
    
    # Load all frames
    all_frames = loader.load_sequence_annotations(sequence)
    print(f"Total frames with annotations: {len(all_frames)}")
    
    # Test specific frame
    test_frame = list(all_frames.keys())[0]
    objects = loader.load_3d_bboxes(sequence, frame=test_frame)
    print(f"\nFrame {test_frame}: {len(objects)} objects")
    
    for obj in objects[:5]:
        print(f"  {obj.class_name:12s} "
              f"track={obj.object_id:3d}  "
              f"lat={obj.lateral_distance:+8.2f}m  "
              f"long={obj.longitudinal_distance:8.2f}m  "
              f"size=[{obj.dimensions[0]:.2f}, {obj.dimensions[1]:.2f}, {obj.dimensions[2]:.2f}]")
    
    # Test clearance
    front_objects = loader.get_objects_in_fov(objects, max_distance=50)
    clearance = loader.compute_lateral_clearance(front_objects)
    print(f"\nLateral Clearance:")
    print(f"  Left:  {clearance['left_clearance_meters']:.2f}m")
    print(f"  Right: {clearance['right_clearance_meters']:.2f}m")
    print(f"  Safe:  {'Yes' if clearance['is_safe'] else 'No'}")
    
    print("\n" + "="*60)
    print("Test passed!")


if __name__ == '__main__':
    test_loader()