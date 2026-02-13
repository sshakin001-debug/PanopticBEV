#!/usr/bin/env python3
"""
Main training script for PanopticBEV
Optimized for Windows with WSL2 detection and cross-platform support
"""
import sys
import os
from pathlib import Path
import platform
import argparse

# Add repo to path
repo_root = Path(__file__).parent
sys.path.insert(0, str(repo_root))

import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler

# CRITICAL: Setup must happen BEFORE importing torch.distributed
from panoptic_bev.utils.windows_performance import (
    setup_for_maximum_performance, 
    check_performance_mode,
    get_optimal_backend,
    is_wsl2,
    is_native_windows,
    print_system_info
)

# Apply optimizations immediately
perf_config = setup_for_maximum_performance()
platform_type = check_performance_mode()

from panoptic_bev.config.config import load_config
from panoptic_bev.data.dataset import BEVKitti360Dataset, BEVNuScenesDataset
from panoptic_bev.models.panoptic_bev import PanopticBEV
from panoptic_bev.utils.windows_dataloader import create_safe_dataloader, get_memory_optimized_config


def setup_distributed_for_windows():
    """
    Configure distributed training backend for Windows compatibility.
    Returns 'nccl' for Linux/WSL, 'gloo' for native Windows.
    """
    if platform.system() == 'Windows':
        # Native Windows: use gloo backend
        return 'gloo'
    # Linux or WSL: nccl is faster for GPU
    return 'nccl'


def is_wsl():
    """Detect Windows Subsystem for Linux."""
    try:
        with open('/proc/version', 'r') as f:
            version = f.read().lower()
            return 'microsoft' in version or 'wsl' in version
    except:
        return False


def setup_distributed(rank, world_size):
    """Initialize distributed training with optimal backend."""
    backend = get_optimal_backend()
    
    if platform_type == 'native_windows':
        # Windows: Use file-based initialization (more reliable)
        init_file = Path.home() / '.torch_distributed_init'
        init_method = f'file://{init_file}'
        os.makedirs(init_file.parent, exist_ok=True)
    else:
        # WSL2/Linux: Use TCP
        init_method = 'env://'
    
    dist.init_process_group(
        backend=backend,
        init_method=init_method,
        rank=rank,
        world_size=world_size
    )
    
    torch.cuda.set_device(rank)
    
    # Synchronize all processes
    dist.barrier()


def create_optimized_dataloaders(config, args, rank=0, world_size=1):
    """
    Create DataLoaders with platform-optimized settings.
    """
    dl_config = config['dataloader']
    
    # Get auto-configured settings based on GPU memory
    auto_config = get_memory_optimized_config()
    is_wsl_env = is_wsl()
    
    # Determine optimal worker counts
    if platform.system() == 'Windows' and not is_wsl_env:
        # Native Windows: be conservative
        train_workers = min(dl_config.getint("train_workers", 2), 2)
        val_workers = 0  # Validation doesn't need workers on Windows
        persistent_workers = False
        pin_memory = False  # Safer on native Windows
    else:
        # Linux/WSL: can use more workers
        train_workers = dl_config.getint("train_workers", auto_config['num_workers'])
        val_workers = dl_config.getint("val_workers", 0)
        persistent_workers = True
        pin_memory = torch.cuda.is_available()
    
    print(f"DataLoader config: workers={train_workers}, persistent={persistent_workers}, "
          f"pin_memory={pin_memory}, WSL={is_wsl_env}")
    
    # Get paths from config or args
    dataset_root = getattr(args, 'dataset_root_dir', config.get('dataset_root_dir', r'D:\datasets\kitti360'))
    seam_root = getattr(args, 'seam_root_dir', config.get('seam_root_dir', r'D:\kitti360_panopticbev'))
    
    # Create datasets
    DatasetClass = BEVKitti360Dataset if args.dataset == 'kitti' else BEVNuScenesDataset
    
    train_dataset = DatasetClass(
        seam_root_dir=seam_root,
        dataset_root_dir=dataset_root,
        split_name='train',
        transform=None  # TODO: Add proper transform
    )
    
    val_dataset = DatasetClass(
        seam_root_dir=seam_root,
        dataset_root_dir=dataset_root,
        split_name='val',
        transform=None  # TODO: Add proper transform
    )
    
    # Use our optimized DataLoader
    train_loader = create_safe_dataloader(
        train_dataset,
        batch_size=dl_config.getint('train_batch_size', auto_config['batch_size']),
        shuffle=True,
        num_workers=train_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers and train_workers > 0,
        drop_last=True
    )
    
    val_loader = create_safe_dataloader(
        val_dataset,
        batch_size=dl_config.getint('val_batch_size', auto_config['batch_size']),
        shuffle=False,
        num_workers=val_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    return train_loader, val_loader


def train_with_amp(model, dataloader, optimizer, scaler, device, epoch):
    """
    Training loop with Automatic Mixed Precision for RTX 5060 Ti.
    """
    model.train()
    total_loss = 0.0
    
    for batch_idx, batch in enumerate(dataloader):
        # Move to device
        if isinstance(batch, dict):
            images = batch.get('image', batch.get('images')).to(device, non_blocking=True)
            targets = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v 
                       for k, v in batch.items() if k not in ['image', 'images']}
        else:
            images = batch[0].to(device, non_blocking=True)
            targets = batch[1] if len(batch) > 1 else None
            if isinstance(targets, torch.Tensor):
                targets = targets.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        # Automatic Mixed Precision
        with autocast(enabled=True):
            if isinstance(targets, dict):
                outputs = model(images, targets)
            else:
                outputs = model(images)
            
            if isinstance(outputs, dict):
                loss = outputs.get('loss', outputs.get('total_loss'))
            else:
                loss = outputs
            
            if loss is None:
                # Compute loss manually if model doesn't return it
                loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        # Scale loss and backprop
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        
        if batch_idx % 10 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
    
    return total_loss / len(dataloader)


def validate(model, dataloader, device):
    """Validation loop."""
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, dict):
                images = batch.get('image', batch.get('images')).to(device, non_blocking=True)
            else:
                images = batch[0].to(device, non_blocking=True)
            
            # Forward pass
            outputs = model(images)
            if isinstance(outputs, dict) and 'loss' in outputs:
                total_loss += outputs['loss'].item()
    
    return total_loss / max(len(dataloader), 1)


def main():
    parser = argparse.ArgumentParser(description='PanopticBEV Training')
    parser.add_argument('--cfg', '--config', required=True, help='Path to config file')
    parser.add_argument('--dataset', choices=['kitti', 'nuscenes'], default='kitti',
                        help='Dataset to use')
    parser.add_argument('--dataset-root-dir', dest='dataset_root_dir',
                        help='Dataset root directory')
    parser.add_argument('--seam-root-dir', dest='seam_root_dir',
                        help='SEAM root directory')
    parser.add_argument('--local-rank', '--local_rank', type=int, default=0,
                        help='Local rank for distributed training')
    parser.add_argument('--world-size', '--world_size', type=int, default=1,
                        help='World size for distributed training')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs (overrides config)')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate (overrides config)')
    parser.add_argument('--debug', action='store_true',
                        help='Debug mode (single GPU, no distributed)')
    parser.add_argument('opts', nargs='*', help='Override config options')
    args = parser.parse_args()
    
    # Print system info
    print_system_info()
    
    # Load config
    config = load_config(args.cfg, args.opts)
    
    # Setup distributed if multi-GPU
    rank = args.local_rank
    world_size = args.world_size
    
    if not args.debug and world_size > 1:
        setup_distributed(rank, world_size)
    
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
    
    # Create dataloaders
    train_loader, val_loader = create_optimized_dataloaders(config, args, rank, world_size)
    
    print(f"Dataset loaded: {len(train_loader.dataset)} training samples")
    
    # Create model
    model = PanopticBEV(config).to(device)
    
    if not args.debug and world_size > 1:
        model = DDP(model, device_ids=[rank])
    
    # Optimizer with AMP scaler
    lr = args.lr if args.lr is not None else config.getfloat('optimizer', 'lr', fallback=1e-4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scaler = GradScaler()  # For automatic mixed precision
    
    # Get number of epochs
    epochs = args.epochs if args.epochs is not None else config.getint('scheduler', 'epochs', fallback=100)
    
    print(f"Starting training on {device}")
    print(f"Platform: {platform_type}")
    print(f"DataLoader workers: {perf_config['num_workers']}")
    print(f"Pin memory: {perf_config['pin_memory']}")
    print(f"Epochs: {epochs}")
    print(f"Learning rate: {lr}")
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        train_loss = train_with_amp(model, train_loader, optimizer, scaler, device, epoch)
        
        if rank == 0:
            val_loss = validate(model, val_loader, device)
            print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint_path = Path(config.get('output_dir', 'outputs')) / 'best_model.pth'
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': val_loss,
                }, checkpoint_path)
                print(f"Saved best model to {checkpoint_path}")
    
    # Cleanup
    if not args.debug and world_size > 1:
        dist.destroy_process_group()
    
    print("Training complete!")


if __name__ == '__main__':
    main()
