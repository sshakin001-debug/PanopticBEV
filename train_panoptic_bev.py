#!/usr/bin/env python3
"""
Main training script for PanopticBEV
Simplified for Windows single-GPU training
"""
import sys
import os
from pathlib import Path

# Add repo to path
repo_root = Path(__file__).parent
sys.path.insert(0, str(repo_root))

import torch
import torch.multiprocessing as mp

# Windows: must use spawn
if os.name == 'nt':
    mp.set_start_method('spawn', force=True)

from panoptic_bev.config.config import load_config
from panoptic_bev.data.dataset import BEVKitti360Dataset
from panoptic_bev.models.panoptic_bev import PanopticBEV
from torch.utils.data import DataLoader


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', required=True, help='Path to config file')
    parser.add_argument('opts', nargs='*', help='Override config options')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.cfg, args.opts)
    
    # Get paths from config or args
    dataset_root = config.get('dataset_root_dir', r'D:\datasets\kitti360')
    seam_root = config.get('seam_root_dir', r'D:\kitti360_panopticbev')
    
    print(f"Dataset: {dataset_root}")
    print(f"Seam: {seam_root}")
    
    # Create dataset
    # TODO: Add proper transform
    train_dataset = BEVKitti360Dataset(
        seam_root_dir=seam_root,
        dataset_root_dir=dataset_root,
        split_name='train',
        transform=None  # Add transform
    )
    
    # Windows: num_workers=0, pin_memory=False
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['dataloader']['train_batch_size'],
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )
    
    print(f"Dataset loaded: {len(train_dataset)} samples")
    
    # Create model
    model = PanopticBEV(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    print(f"Model on {device}")
    print("Ready to train!")


if __name__ == '__main__':
    main()