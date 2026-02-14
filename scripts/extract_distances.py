#!/usr/bin/env python3
"""
Extract lateral and longitudinal distances from trained model.
Inference script for distance estimation from BEV panoptic segmentation.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import cv2
import argparse
import json
from datetime import datetime
from typing import Dict, Optional

from panoptic_bev.utils.kitti360_calibration import load_calibration_from_dataset
from panoptic_bev.utils.distance_estimation import (
    BEVDistanceEstimator, 
    DistanceMeasurement,
    create_estimator_from_dataset
)


def load_model(model_path: str, config_path: str = None):
    """
    Load trained PanopticBEV model.
    
    Args:
        model_path: Path to model checkpoint
        config_path: Path to config file (optional)
        
    Returns:
        Loaded model
    """
    from panoptic_bev.config import load_config
    from panoptic_bev.models.panoptic_bev import PanopticBevNet
    from panoptic_bev.models.distance_head import PanopticBEVWithDistance
    from panoptic_bev.models.backbone_edet.efficientdet import EfficientDet
    from panoptic_bev.modules.ms_transformer import MultiScaleTransformerVF
    from panoptic_bev.modules.heads import RPNHead, FPNMaskHead, FPNSemanticHeadDPC
    from panoptic_bev.utils.misc import norm_act_from_config
    
    # Load config if provided
    if config_path:
        config = load_config(config_path)
    else:
        # Use default config
        config = {
            'backbone': {'name': 'efficientnet-b4', 'pretrained': False},
            'transformer': {
                'num_layers': 6, 'd_model': 256, 'dff': 1024, 'heads': 8,
                'dropout': 0.1, 'pe_only': False, 'num_bev_layers': 3,
                'use_semantic_guidance': True
            }
        }
    
    norm_act = norm_act_from_config(config)
    
    # Create model components
    body = EfficientDet(config['backbone'], norm_act=norm_act)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Check if checkpoint has distance head
    has_distance_head = any('distance_head' in k for k in checkpoint.keys())
    
    if has_distance_head:
        print("Loading model with distance head...")
        # Create base model and wrap with distance head
        # ... model creation code ...
    else:
        print("Loading base model without distance head...")
        # Create base model only
        # ... model creation code ...
    
    # For now, return None and use post-hoc distance estimation
    return None


def preprocess_image(image_path: str, target_size: tuple = (768, 1408)):
    """
    Preprocess image for model input.
    
    Args:
        image_path: Path to input image
        target_size: Target size (height, width)
        
    Returns:
        Preprocessed image tensor
    """
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize
    image = cv2.resize(image, (target_size[1], target_size[0]))
    
    # Normalize
    image = image.astype(np.float32) / 255.0
    image = (image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    
    # Convert to tensor
    image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    
    return image_tensor, image


def extract_distances_from_panoptic(
    panoptic_bev: np.ndarray,
    estimator: BEVDistanceEstimator,
    return_visualization: bool = True
) -> Dict:
    """
    Extract distance measurements from panoptic BEV segmentation.
    
    Args:
        panoptic_bev: HxW panoptic segmentation array
        estimator: BEVDistanceEstimator instance
        return_visualization: Whether to create visualization
        
    Returns:
        Dictionary with distance measurements
    """
    # Convert panoptic to distances
    results = estimator.panoptic_to_distances(panoptic_bev, return_visualization)
    
    return results


def format_measurement(measurement: DistanceMeasurement) -> str:
    """Format a distance measurement for display."""
    direction = "right" if measurement.lateral_distance >= 0 else "left"
    return (f"{measurement.class_name:<15} "
            f"Long: {measurement.longitudinal_distance:>6.2f}m  "
            f"Lat: {abs(measurement.lateral_distance):>5.2f}m {direction}")


def save_distance_report(results: Dict, output_path: str):
    """Save distance results to JSON file."""
    
    def measurement_to_dict(m):
        """Convert DistanceMeasurement to dict."""
        if m is None:
            return None
        return {
            'object_id': m.object_id,
            'class_name': m.class_name,
            'lateral_distance_m': m.lateral_distance,
            'longitudinal_distance_m': m.longitudinal_distance,
            'centroid_x': m.centroid_x,
            'centroid_y': m.centroid_y,
            'bbox_m': list(m.bbox)
        }
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'num_objects': results['num_objects'],
        'ego_position': results['ego_position'],
        'measurements': [measurement_to_dict(m) for m in results['measurements']]
    }
    
    # Add clearance info if available
    if 'clearance' in results:
        clearance = results['clearance'].copy()
        # Convert DistanceMeasurement objects to dicts
        if 'closest_left' in clearance:
            clearance['closest_left'] = measurement_to_dict(clearance['closest_left'])
        if 'closest_right' in clearance:
            clearance['closest_right'] = measurement_to_dict(clearance['closest_right'])
        report['clearance'] = clearance
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Report saved to: {output_path}")


def create_demo_panoptic_bev(size: tuple = (512, 512)) -> np.ndarray:
    """
    Create a demo panoptic BEV segmentation for testing.
    This creates synthetic objects at known positions.
    
    Args:
        size: (height, width) of BEV image
        
    Returns:
        Synthetic panoptic BEV array
    """
    panoptic = np.zeros(size, dtype=np.int32)
    
    # Road (class 0) - background
    panoptic[:] = 0
    
    # Add some synthetic objects
    # Car (class 12) at 20m forward, 3m right
    car_y, car_x = size[0] - 10 - int(20 / 0.05), size[1] // 2 + int(3 / 0.05)
    cv2.rectangle(panoptic, (car_x - 10, car_y - 10), (car_x + 10, car_y + 10), 12000, -1)
    
    # Person (class 10) at 10m forward, 2m left
    person_y, person_x = size[0] - 10 - int(10 / 0.05), size[1] // 2 - int(2 / 0.05)
    cv2.circle(panoptic, (person_x, person_y), 5, 10000, -1)
    
    # Truck (class 13) at 30m forward, 5m right
    truck_y, truck_x = size[0] - 10 - int(30 / 0.05), size[1] // 2 + int(5 / 0.05)
    cv2.rectangle(panoptic, (truck_x - 15, truck_y - 15), (truck_x + 15, truck_y + 15), 13000, -1)
    
    # Bicycle (class 17) at 15m forward, 4m left
    bike_y, bike_x = size[0] - 10 - int(15 / 0.05), size[1] // 2 - int(4 / 0.05)
    cv2.circle(panoptic, (bike_x, bike_y), 3, 17000, -1)
    
    return panoptic


def main():
    parser = argparse.ArgumentParser(description='Extract distances from BEV panoptic segmentation')
    parser.add_argument('--model', type=str, help='Path to trained model checkpoint')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--image', type=str, help='Input image path')
    parser.add_argument('--panoptic', type=str, help='Pre-computed panoptic BEV path (numpy file)')
    parser.add_argument('--dataset_root', type=str, default='D:/datasets/kitti360',
                       help='Dataset root for calibration')
    parser.add_argument('--output', type=str, default='distance_result.png',
                       help='Output visualization path')
    parser.add_argument('--report', type=str, default=None,
                       help='Output JSON report path')
    parser.add_argument('--demo', action='store_true',
                       help='Run with demo/synthetic data')
    parser.add_argument('--bev_resolution', type=float, default=0.05,
                       help='BEV resolution in meters per pixel')
    parser.add_argument('--bev_size', type=int, nargs=2, default=[512, 512],
                       help='BEV size (height width)')
    args = parser.parse_args()
    
    # Create distance estimator
    try:
        if Path(args.dataset_root).exists():
            print(f"Loading calibration from: {args.dataset_root}")
            estimator = create_estimator_from_dataset(
                args.dataset_root,
                bev_size=tuple(args.bev_size)
            )
        else:
            print(f"Dataset root not found: {args.dataset_root}")
            print("Using default calibration values")
            estimator = BEVDistanceEstimator(
                dataset_root=None,
                bev_resolution=args.bev_resolution,
                bev_size=tuple(args.bev_size)
            )
    except FileNotFoundError as e:
        print(f"Calibration not found: {e}")
        print("Using default calibration values")
        estimator = BEVDistanceEstimator(
            dataset_root=None,
            bev_resolution=args.bev_resolution,
            bev_size=tuple(args.bev_size)
        )
    
    # Get panoptic BEV
    if args.demo:
        print("\n=== Running in DEMO mode with synthetic data ===")
        panoptic_bev = create_demo_panoptic_bev(tuple(args.bev_size))
    elif args.panoptic:
        print(f"\nLoading panoptic BEV from: {args.panoptic}")
        panoptic_bev = np.load(args.panoptic)
    elif args.image and args.model:
        print(f"\nProcessing image: {args.image}")
        # Load model and run inference
        model = load_model(args.model, args.config)
        if model is None:
            print("Model loading not implemented. Using demo data.")
            panoptic_bev = create_demo_panoptic_bev(tuple(args.bev_size))
        else:
            # Run inference
            # ... inference code ...
            pass
    else:
        print("No input specified. Using demo data.")
        print("Use --image, --panoptic, or --demo to specify input.")
        panoptic_bev = create_demo_panoptic_bev(tuple(args.bev_size))
    
    # Extract distances
    print("\n" + "="*60)
    print("DISTANCE MEASUREMENTS")
    print("="*60)
    
    results = extract_distances_from_panoptic(panoptic_bev, estimator)
    
    # Print results
    print(f"\nFound {results['num_objects']} objects:")
    print(f"{'Object':<15} {'Distance':<20} {'Lateral':<15}")
    print("-" * 60)
    
    for meas in results['measurements'][:15]:  # Top 15 closest
        print(format_measurement(meas))
    
    # Get lateral clearance
    clearance = estimator.get_lateral_clearance(results['measurements'])
    results['clearance'] = clearance
    
    print(f"\n{'='*60}")
    print("LATERAL CLEARANCE")
    print(f"{'='*60}")
    print(f"  Left:  {clearance['left_clearance_meters']:.2f}m")
    print(f"  Right: {clearance['right_clearance_meters']:.2f}m")
    print(f"  Min:   {clearance['min_clearance']:.2f}m")
    print(f"  Safe:  {'YES' if clearance['is_safe'] else 'NO'}")
    
    # Save visualization
    if 'visualization' in results:
        vis = results['visualization']
        cv2.imwrite(str(args.output), vis)
        print(f"\nVisualization saved to: {args.output}")
    
    # Save report
    report_path = args.report or args.output.replace('.png', '.json')
    save_distance_report(results, report_path)
    
    print("\nDone!")


if __name__ == '__main__':
    main()