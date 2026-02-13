#!/usr/bin/env python3
"""
PanopticBEV Training - Windows Optimized for RTX 5060 Ti
"""
import sys
import os
from pathlib import Path

# CRITICAL: Setup multiprocessing FIRST
from panoptic_bev.utils.windows_dataloader import setup_windows_multiprocessing
setup_windows_multiprocessing()

# Now import torch
import torch
import argparse
from panoptic_bev.utils.windows_dataloader import create_fast_windows_dataloader, get_memory_optimized_config


def validate_dataset_paths(dataset_root: str, seam_root: str) -> bool:
    """Verify dataset structure matches expected layout."""
    dataset_path = Path(dataset_root)
    seam_path = Path(seam_root)
    
    print(f"\n{'='*60}")
    print("VALIDATING DATASET PATHS")
    print(f"{'='*60}")
    
    # Check original KITTI-360
    required_kitti = ['calibration', 'data_2d_raw', 'data_poses']
    for folder in required_kitti:
        if not (dataset_path / folder).exists():
            print(f"Missing: {dataset_path / folder}")
            return False
        print(f"Found: {folder}")
    
    # Check PanopticBEV processed data
    required_seam = ['bev_msk', 'front_msk_seam', 'img', 'split']
    for folder in required_seam:
        if not (seam_path / folder).exists():
            print(f"Missing: {seam_path / folder}")
            return False
        print(f"Found: {folder}")
    
    print(f"{'='*60}")
    print("All paths validated successfully!")
    print(f"{'='*60}\n")
    return True


def main():
    parser = argparse.ArgumentParser(description='PanopticBEV Windows Training')
    parser.add_argument('--config', required=True, help='Config file path')
    parser.add_argument('--dataset', choices=['kitti', 'nuscenes'], default='kitti')
    parser.add_argument('--dataset_root_dir', required=True, help='e.g., D:\\datasets\\kitti360')
    parser.add_argument('--seam_root_dir', required=True, help='e.g., D:\\kitti360_panopticbev')
    parser.add_argument('--run_name', default='windows_test')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=None)
    args = parser.parse_args()
    
    # Convert Windows paths to Path objects (handles backslashes correctly)
    dataset_root = Path(args.dataset_root_dir)
    seam_root = Path(args.seam_root_dir)
    
    # Validate paths
    if not validate_dataset_paths(str(dataset_root), str(seam_root)):
        print("ERROR: Dataset validation failed!")
        print(f"Make sure these paths exist:")
        print(f"  Dataset: {dataset_root}")
        print(f"  PanopticBEV: {seam_root}")
        sys.exit(1)
    
    # Get RTX 5060 Ti optimized config
    config = get_memory_optimized_config()
    if args.batch_size:
        config['batch_size'] = args.batch_size
    
    print(f"\nConfiguration:")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Workers: {args.num_workers}")
    print(f"  AMP: {config['amp']}")
    print(f"  Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    
    # TODO: Add your actual training loop here
    # For now, just verify DataLoader works
    print("\nTesting DataLoader...")
    from panoptic_bev.data.dataset import BEVKitti360Dataset, BEVTransform
    
    transform = BEVTransform(
        shortest_size=512,
        longest_max_size=1024,
        rgb_mean=[0.485, 0.456, 0.406],
        rgb_std=[0.229, 0.224, 0.225],
        front_resize=[512, 1024],
        bev_crop=[512, 512]
    )
    
    dataset = BEVKitti360Dataset(
        seam_root_dir=str(seam_root),
        dataset_root_dir=str(dataset_root),
        split_name='train',
        transform=transform
    )
    
    print(f"Dataset loaded: {len(dataset)} samples")
    
    # Create optimized DataLoader
    loader = create_fast_windows_dataloader(
        dataset,
        batch_size=config['batch_size'],
        num_workers=args.num_workers,
        shuffle=True,
        drop_last=True
    )
    
    print(f"DataLoader created: {len(loader)} batches")
    print("Testing first batch...")
    
    for i, batch in enumerate(loader):
        print(f"Batch {i+1} loaded successfully")
        if i >= 2:  # Test first 3 batches
            break
    
    print("\nAll tests passed! Ready for training.")


if __name__ == '__main__':
    main()
