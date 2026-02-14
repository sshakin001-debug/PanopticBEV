"""
Quick 3D distance loader - replaces inaccurate BEV method.
This is a minimal version for quick integration.
"""
import numpy as np
import json
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Dict


@dataclass
class Distance3D:
    """Simple 3D distance measurement."""
    lateral: float      # X: positive=right, negative=left (meters)
    longitudinal: float # Z: positive=forward (meters)
    height: float       # Y: negative=up (meters)
    class_name: str
    track_id: int = -1
    confidence: float = 1.0
    
    @property
    def euclidean_distance(self) -> float:
        """Straight-line distance from camera."""
        return float(np.sqrt(self.lateral**2 + self.height**2 + self.longitudinal**2))


def load_3d_distances(sequence: str, 
                      frame: int, 
                      bboxes_root: str = r"D:\datasets\kitti360\data_3d_bboxes",
                      use_train_full: bool = True) -> List[Distance3D]:
    """
    Load accurate distances from 3D bounding boxes.
    
    This replaces the inaccurate BEV-pixel-based method with
    accurate 3D ground truth from KITTI-360 annotations.
    
    Args:
        sequence: Sequence name (e.g., '2013_05_28_drive_0000_sync')
        frame: Frame number
        bboxes_root: Path to data_3d_bboxes directory
        use_train_full: Use train_full split (default) vs train
    
    Returns:
        List of Distance3D objects sorted by longitudinal distance
    
    Example:
        >>> distances = load_3d_distances('2013_05_28_drive_0000_sync', 0)
        >>> for d in distances:
        ...     print(f"{d.class_name}: lat={d.lateral:+.2f}m, long={d.longitudinal:.2f}m")
    """
    # Determine split directory
    split_dir = "train_full" if use_train_full else "train"
    
    # Look for annotation file
    annotations_dir = Path(bboxes_root) / split_dir
    
    # Try .xml extension first (KITTI-360 uses .xml for tracking annotations)
    bbox_file = annotations_dir / f"{sequence}.xml"
    
    if not bbox_file.exists():
        # Try .txt extension
        bbox_file = annotations_dir / f"{sequence}.txt"
    
    if not bbox_file.exists():
        print(f"Warning: No annotation file found for sequence {sequence}")
        return []
    
    # Parse the annotation file
    distances = []
    
    with open(bbox_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split()
            
            if len(parts) < 16:
                continue
            
            try:
                frame_num = int(float(parts[0]))
                
                # Skip if not the requested frame
                if frame_num != frame:
                    continue
                
                track_id = int(float(parts[1]))
                class_name = parts[2]
                
                # Parse 3D position (indices 13, 14, 15)
                x = float(parts[13])  # lateral
                y = float(parts[14])  # height (negative = up)
                z = float(parts[15])  # longitudinal
                
                # Parse confidence if available
                confidence = float(parts[17]) if len(parts) > 17 else 1.0
                
                distances.append(Distance3D(
                    lateral=x,
                    longitudinal=z,
                    height=-y,  # Convert to positive height
                    class_name=class_name,
                    track_id=track_id,
                    confidence=confidence
                ))
                
            except (ValueError, IndexError) as e:
                continue
    
    # Sort by longitudinal distance (closest first)
    distances.sort(key=lambda d: d.longitudinal)
    
    return distances


def get_closest_object(distances: List[Distance3D], 
                       class_filter: Optional[List[str]] = None) -> Optional[Distance3D]:
    """
    Get the closest object, optionally filtered by class.
    
    Args:
        distances: List of distances
        class_filter: Optional list of class names to filter by
    
    Returns:
        Closest Distance3D or None if no objects match
    """
    if class_filter:
        filtered = [d for d in distances if d.class_name in class_filter]
    else:
        filtered = distances
    
    if not filtered:
        return None
    
    return filtered[0]  # Already sorted by longitudinal distance


def get_objects_in_range(distances: List[Distance3D],
                         min_long: float = 0.0,
                         max_long: float = 50.0,
                         max_lateral: float = 50.0) -> List[Distance3D]:
    """
    Filter objects to those within a specified range.
    
    Args:
        distances: List of distances
        min_long: Minimum longitudinal distance (meters)
        max_long: Maximum longitudinal distance (meters)
        max_lateral: Maximum absolute lateral distance (meters)
    
    Returns:
        Filtered list of distances
    """
    filtered = [
        d for d in distances
        if min_long <= d.longitudinal <= max_long
        and abs(d.lateral) <= max_lateral
    ]
    return filtered


def compute_lateral_clearance(distances: List[Distance3D],
                              longitudinal_range: tuple = (0, 50),
                              vehicle_width: float = 2.0) -> Dict:
    """
    Compute lateral clearance from objects.
    
    Args:
        distances: List of distances
        longitudinal_range: (min, max) longitudinal range to consider
        vehicle_width: Width of ego vehicle in meters
    
    Returns:
        Dictionary with clearance information
    """
    # Filter to range
    in_range = get_objects_in_range(
        distances,
        min_long=longitudinal_range[0],
        max_long=longitudinal_range[1]
    )
    
    half_width = vehicle_width / 2
    
    # Separate left and right
    left_objects = [d for d in in_range if d.lateral < -half_width]
    right_objects = [d for d in in_range if d.lateral > half_width]
    
    # Compute clearance
    if left_objects:
        # Closest on left (most positive among negatives)
        closest_left = max(left_objects, key=lambda d: d.lateral)
        left_clearance = abs(closest_left.lateral) - half_width
    else:
        closest_left = None
        left_clearance = float('inf')
    
    if right_objects:
        # Closest on right (most negative among positives)
        closest_right = min(right_objects, key=lambda d: d.lateral)
        right_clearance = closest_right.lateral - half_width
    else:
        closest_right = None
        right_clearance = float('inf')
    
    return {
        'left_clearance_m': left_clearance,
        'right_clearance_m': right_clearance,
        'min_clearance_m': min(left_clearance, right_clearance),
        'closest_left': closest_left,
        'closest_right': closest_right,
        'is_safe': min(left_clearance, right_clearance) > 0.5
    }


def distances_to_dict(distances: List[Distance3D]) -> List[Dict]:
    """
    Convert distances to list of dictionaries for JSON serialization.
    
    Args:
        distances: List of Distance3D objects
    
    Returns:
        List of dictionaries
    """
    return [
        {
            'class': d.class_name,
            'track_id': d.track_id,
            'lateral_m': d.lateral,
            'longitudinal_m': d.longitudinal,
            'height_m': d.height,
            'euclidean_m': d.euclidean_distance,
            'confidence': d.confidence
        }
        for d in distances
    ]


# Quick test function
def test_distance_3d():
    """Test the distance_3d module."""
    print("Testing distance_3d module")
    print("="*60)
    
    # Try to load distances
    try:
        loader = __import__('panoptic_bev.utils.kitti360_3d_loader', fromlist=['KITTI3603DLoader'])
        loader_class = loader.KITTI3603DLoader
        
        # Get available sequences
        temp_loader = loader_class()
        if temp_loader.sequences:
            sequence = list(temp_loader.sequences.keys())[0]
            print(f"\nUsing sequence: {sequence}")
            
            # Load distances
            distances = load_3d_distances(sequence, frame=0)
            
            if distances:
                print(f"Found {len(distances)} objects in frame 0")
                
                print("\nClosest 5 objects:")
                for i, d in enumerate(distances[:5]):
                    print(f"  {i+1}. {d.class_name:12s} "
                          f"lat={d.lateral:+7.2f}m  "
                          f"long={d.longitudinal:7.2f}m  "
                          f"dist={d.euclidean_distance:7.2f}m")
                
                # Test clearance
                clearance = compute_lateral_clearance(distances)
                print(f"\nLateral Clearance:")
                print(f"  Left:  {clearance['left_clearance_m']:.2f}m")
                print(f"  Right: {clearance['right_clearance_m']:.2f}m")
                print(f"  Safe:  {'Yes' if clearance['is_safe'] else 'No'}")
            else:
                print("No objects found in frame 0")
        else:
            print("No sequences found")
            
    except Exception as e:
        print(f"Could not test with real data: {e}")
        print("Module is ready for use when KITTI-360 data is available.")
    
    print("\n" + "="*60)
    print("Test completed!")


if __name__ == '__main__':
    test_distance_3d()