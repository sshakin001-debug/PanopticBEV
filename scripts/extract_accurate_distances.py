#!/usr/bin/env python3
"""
Extract accurate distances using 3D ground truth or trained model.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import numpy as np

from panoptic_bev.utils.accurate_distance_estimator import AccurateDistanceEstimator
from panoptic_bev.utils.kitti360_3d_loader import KITTI3603DLoader


def main():
    parser = argparse.ArgumentParser(description='Extract accurate 3D distances')
    parser.add_argument('--sequence', required=True, help='Sequence name')
    parser.add_argument('--frame', type=int, required=True, help='Frame number')
    parser.add_argument('--dataset_root', default=r'D:\datasets\kitti360',
                       help='Path to KITTI-360 dataset')
    parser.add_argument('--output', default='distance_output.png',
                       help='Output visualization path')
    parser.add_argument('--max_distance', type=float, default=100.0,
                       help='Maximum distance to consider')
    parser.add_argument('--save_json', action='store_true',
                       help='Save results as JSON')
    args = parser.parse_args()
    
    print("="*60)
    print("Accurate 3D Distance Extraction")
    print("="*60)
    
    # Initialize estimator
    estimator = AccurateDistanceEstimator(dataset_root=args.dataset_root)
    
    # Get accurate distances
    result = estimator.estimate(
        sequence=args.sequence,
        frame=args.frame,
        max_distance=args.max_distance
    )
    
    # Print results
    print(f"\nSequence: {args.sequence}, Frame: {args.frame}")
    print(f"Total objects: {result['num_objects']}")
    
    print("\nAll objects (sorted by distance):")
    print(f"{'#':<4} {'Class':<12} {'Lateral':<10} {'Longitudinal':<12} {'Euclidean':<10}")
    print("-" * 60)
    
    for i, obj in enumerate(result['objects']):
        est = obj['estimate']
        print(f"{i+1:<4} {obj['class']:<12} "
              f"{est.lateral:+9.2f}m "
              f"{est.longitudinal:9.2f}m "
              f"{obj['euclidean_distance']:9.2f}m")
    
    # Clearance analysis
    if result['clearance']:
        c = result['clearance']
        print(f"\n{'='*60}")
        print("LATERAL CLEARANCE ANALYSIS")
        print(f"{'='*60}")
        
        left_obj_name = c['closest_left'].class_name if c['closest_left'] else 'none'
        right_obj_name = c['closest_right'].class_name if c['closest_right'] else 'none'
        
        print(f"Left clearance:  {c['left_clearance_meters']:6.2f}m ({left_obj_name})")
        print(f"Right clearance: {c['right_clearance_meters']:6.2f}m ({right_obj_name})")
        print(f"Minimum:         {c['min_clearance']:6.2f}m")
        print(f"Safe to proceed: {'YES' if c['is_safe'] else 'NO'}")
        
        if not c['is_safe']:
            print("\nWARNING: Insufficient lateral clearance!")
    
    # Lane analysis
    lanes = estimator.get_objects_in_lane(result['objects'])
    print(f"\n{'='*60}")
    print("LANE ANALYSIS")
    print(f"{'='*60}")
    print(f"Ego lane:    {len(lanes['ego_lane'])} objects")
    print(f"Left lane:   {len(lanes['left_lane'])} objects")
    print(f"Right lane:  {len(lanes['right_lane'])} objects")
    
    if lanes['closest_in_ego_lane']:
        closest = lanes['closest_in_ego_lane']
        print(f"\nClosest in ego lane: {closest['class']} "
              f"at {closest['estimate'].longitudinal:.2f}m")
    
    # Visualize
    print(f"\nGenerating visualization...")
    loader = estimator.loader
    bev_vis = loader.project_to_bev(
        [o['object'] for o in result['objects']],
        bev_resolution=0.05,
        bev_size=(512, 512)
    )
    
    # Save visualization
    try:
        import cv2
        cv2.imwrite(args.output, bev_vis)
        print(f"Saved visualization to: {args.output}")
    except ImportError:
        print("OpenCV not available, skipping visualization")
    
    # Save detailed report as JSON
    if args.save_json:
        report_file = args.output.replace('.png', '.json')
        
        # Prepare clearance data
        clearance_data = None
        if result['clearance']:
            c = result['clearance']
            clearance_data = {
                'left': c['left_clearance_meters'],
                'right': c['right_clearance_meters'],
                'min': c['min_clearance'],
                'is_safe': c['is_safe'],
                'num_objects_in_range': c['num_objects_in_range']
            }
        
        report = {
            'sequence': args.sequence,
            'frame': args.frame,
            'num_objects': result['num_objects'],
            'objects': [
                {
                    'class': o['class'],
                    'lateral': float(o['estimate'].lateral),
                    'longitudinal': float(o['estimate'].longitudinal),
                    'euclidean': float(o['euclidean_distance']),
                    'uncertainty': float(o['estimate'].uncertainty),
                    'source': o['estimate'].source
                }
                for o in result['objects']
            ],
            'clearance': clearance_data,
            'lane_analysis': {
                'ego_lane_count': len(lanes['ego_lane']),
                'left_lane_count': len(lanes['left_lane']),
                'right_lane_count': len(lanes['right_lane'])
            }
        }
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Report saved to: {report_file}")
    
    print("\n" + "="*60)
    print("Done!")
    print("="*60)


def batch_extract():
    """
    Batch extraction of distances for multiple frames.
    """
    parser = argparse.ArgumentParser(description='Batch extract 3D distances')
    parser.add_argument('--dataset_root', default=r'D:\datasets\kitti360',
                       help='Path to KITTI-360 dataset')
    parser.add_argument('--output_dir', default='output/distances',
                       help='Output directory')
    parser.add_argument('--sequences', default=None,
                       help='Comma-separated list of sequences (None = all)')
    parser.add_argument('--max_distance', type=float, default=100.0,
                       help='Maximum distance to consider')
    parser.add_argument('--frame_step', type=int, default=1,
                       help='Process every Nth frame')
    args = parser.parse_args()
    
    print("="*60)
    print("Batch 3D Distance Extraction")
    print("="*60)
    
    # Initialize
    estimator = AccurateDistanceEstimator(dataset_root=args.dataset_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get sequences to process
    if args.sequences:
        sequences = args.sequences.split(',')
    else:
        sequences = list(estimator.loader.sequences.keys())
    
    print(f"\nProcessing {len(sequences)} sequences...")
    
    all_results = {}
    
    for seq_name in sequences:
        print(f"\n{'='*60}")
        print(f"Sequence: {seq_name}")
        print(f"{'='*60}")
        
        # Get all frames for this sequence
        all_frames = estimator.loader.load_sequence_annotations(seq_name)
        frame_nums = sorted(all_frames.keys())[::args.frame_step]
        
        seq_results = []
        
        for frame_num in frame_nums:
            result = estimator.estimate(
                sequence=seq_name,
                frame=frame_num,
                max_distance=args.max_distance
            )
            
            # Store summary
            seq_results.append({
                'frame': frame_num,
                'num_objects': result['num_objects'],
                'min_clearance': result['clearance']['min_clearance'] if result['clearance'] else None,
            })
            
            print(f"  Frame {frame_num}: {result['num_objects']} objects")
        
        all_results[seq_name] = seq_results
        
        # Save sequence summary
        seq_file = output_dir / f"{seq_name}_summary.json"
        with open(seq_file, 'w') as f:
            json.dump(seq_results, f, indent=2)
        print(f"  Saved: {seq_file}")
    
    # Save overall summary
    summary_file = output_dir / "all_sequences_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nOverall summary saved to: {summary_file}")
    
    print("\n" + "="*60)
    print("Batch extraction completed!")
    print("="*60)


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--batch':
        # Batch mode
        sys.argv.pop(1)  # Remove --batch flag
        batch_extract()
    else:
        # Single frame mode
        main()