#!/usr/bin/env python3
"""
Test calibration loading from actual KITTI-360 files.
Verifies that calibration data is loaded correctly.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import numpy as np


def test_calibration_loading(dataset_root: str):
    """Test loading calibration from KITTI-360 dataset."""
    from panoptic_bev.utils.kitti360_calibration import (
        load_calibration_from_dataset,
        KITTI360Calibration
    )
    
    print("="*60)
    print("KITTI-360 CALIBRATION TEST")
    print("="*60)
    
    # Check if path exists
    dataset_path = Path(dataset_root)
    if not dataset_path.exists():
        print(f"\nERROR: Dataset root not found: {dataset_root}")
        print("\nPlease provide a valid path to KITTI-360 dataset.")
        print("Expected structure:")
        print("  dataset_root/")
        print("    calibration/")
        print("      calib_cam_to_pose.txt")
        print("      calib_cam_to_velo.txt")
        print("      perspective_calibration.txt")
        return False
    
    calib_path = dataset_path / "calibration"
    if not calib_path.exists():
        print(f"\nERROR: Calibration folder not found: {calib_path}")
        return False
    
    print(f"\nDataset root: {dataset_root}")
    print(f"Calibration path: {calib_path}")
    
    # List available calibration files
    print("\nAvailable calibration files:")
    for f in calib_path.iterdir():
        if f.is_file():
            print(f"  - {f.name}")
    
    # Load calibration
    print("\n" + "-"*60)
    print("Loading calibration...")
    print("-"*60)
    
    try:
        calib = load_calibration_from_dataset(dataset_root)
    except Exception as e:
        print(f"\nERROR loading calibration: {e}")
        return False
    
    # Print summary
    calib.print_summary()
    
    # Test specific values
    print("\n" + "-"*60)
    print("Testing specific values:")
    print("-"*60)
    
    try:
        fx = calib.get_focal_length()
        cx, cy = calib.get_principal_point()
        height = calib.get_camera_height()
        resolution = calib.get_bev_resolution()
        
        print(f"  Focal length (fx): {fx:.4f} pixels")
        print(f"  Principal point (cx, cy): ({cx:.2f}, {cy:.2f})")
        print(f"  Camera height: {height:.2f}m")
        print(f"  BEV resolution: {resolution*100:.1f} cm/pixel")
        
        # Validate values
        if fx <= 0:
            print("\nWARNING: Focal length should be positive!")
        if height <= 0:
            print("\nWARNING: Camera height should be positive!")
        if resolution <= 0:
            print("\nWARNING: BEV resolution should be positive!")
        
        print("\n" + "="*60)
        print("CALIBRATION TEST PASSED!")
        print("="*60)
        return True
        
    except Exception as e:
        print(f"\nERROR accessing calibration values: {e}")
        return False


def test_distance_estimator(dataset_root: str = None):
    """Test distance estimator with calibration."""
    from panoptic_bev.utils.distance_estimation import (
        BEVDistanceEstimator,
        create_estimator_from_dataset
    )
    
    print("\n" + "="*60)
    print("DISTANCE ESTIMATOR TEST")
    print("="*60)
    
    # Create estimator
    if dataset_root and Path(dataset_root).exists():
        print(f"\nCreating estimator from dataset: {dataset_root}")
        try:
            estimator = create_estimator_from_dataset(dataset_root)
        except Exception as e:
            print(f"ERROR: {e}")
            print("\nFalling back to default values...")
            estimator = BEVDistanceEstimator(
                dataset_root=None,
                bev_resolution=0.05,
                bev_size=(512, 512)
            )
    else:
        print("\nUsing default calibration values")
        estimator = BEVDistanceEstimator(
            dataset_root=None,
            bev_resolution=0.05,
            bev_size=(512, 512)
        )
    
    # Create test panoptic BEV
    print("\nCreating test panoptic BEV...")
    test_panoptic = np.zeros((512, 512), dtype=np.int32)
    
    # Add a car at known position (20m forward, 3m right)
    # Ego is at (256, 502) in pixel coords
    # 20m forward = 20/0.05 = 400 pixels up from ego
    # 3m right = 3/0.05 = 60 pixels right from center
    car_y = 502 - 400  # = 102
    car_x = 256 + 60   # = 316
    
    # Draw car (class 12, instance 0 -> ID = 12000)
    test_panoptic[car_y-10:car_y+10, car_x-10:car_x+10] = 12000
    
    print(f"  Test car placed at pixel ({car_x}, {car_y})")
    print(f"  Expected: ~20m forward, ~3m right")
    
    # Extract distances
    print("\nExtracting distances...")
    results = estimator.panoptic_to_distances(test_panoptic)
    
    print(f"\nFound {results['num_objects']} object(s)")
    
    for meas in results['measurements']:
        print(f"\n  Object: {meas.class_name}")
        print(f"    Longitudinal: {meas.longitudinal_distance:.2f}m (expected: ~20m)")
        print(f"    Lateral: {meas.lateral_distance:.2f}m (expected: ~3m)")
        
        # Check accuracy
        long_error = abs(meas.longitudinal_distance - 20)
        lat_error = abs(meas.lateral_distance - 3)
        
        if long_error < 1 and lat_error < 1:
            print("    ACCURACY: GOOD")
        else:
            print(f"    ACCURACY: Check errors (long={long_error:.2f}m, lat={lat_error:.2f}m)")
    
    print("\n" + "="*60)
    print("DISTANCE ESTIMATOR TEST COMPLETE!")
    print("="*60)
    
    return True


def test_homography(dataset_root: str):
    """Test homography computation."""
    from panoptic_bev.utils.kitti360_calibration import load_calibration_from_dataset
    
    print("\n" + "="*60)
    print("HOMOGRAPHY TEST")
    print("="*60)
    
    if not Path(dataset_root).exists():
        print("Dataset not found, skipping homography test")
        return False
    
    calib = load_calibration_from_dataset(dataset_root)
    
    # Compute homography
    H = calib.compute_ground_plane_to_image_homography()
    
    print(f"\nGround plane to image homography:")
    print(H)
    
    # Test point projection
    # Point on ground 10m ahead, 2m right
    ground_point = np.array([2, 10, 1])  # x, y, 1 (in meters)
    image_point = H @ ground_point
    image_point = image_point / image_point[2]  # Normalize
    
    print(f"\nTest projection:")
    print(f"  Ground point: (2m right, 10m ahead)")
    print(f"  Image point: ({image_point[0]:.1f}, {image_point[1]:.1f}) pixels")
    
    print("\n" + "="*60)
    print("HOMOGRAPHY TEST COMPLETE!")
    print("="*60)
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Test KITTI-360 calibration loading')
    parser.add_argument('--dataset_root', type=str, default='D:/datasets/kitti360',
                       help='Path to KITTI-360 dataset root')
    parser.add_argument('--test_estimator', action='store_true',
                       help='Also test distance estimator')
    parser.add_argument('--test_homography', action='store_true',
                       help='Also test homography computation')
    args = parser.parse_args()
    
    # Run calibration test
    success = test_calibration_loading(args.dataset_root)
    
    # Run additional tests if requested
    if args.test_estimator or args.test_homography:
        if args.test_estimator:
            test_distance_estimator(args.dataset_root)
        if args.test_homography:
            test_homography(args.dataset_root)
    
    # If no dataset found, still test estimator with defaults
    if not Path(args.dataset_root).exists():
        print("\n" + "="*60)
        print("Dataset not found, testing with default values...")
        print("="*60)
        test_distance_estimator(None)
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())