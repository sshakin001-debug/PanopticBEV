#!/usr/bin/env python3
"""
Training script with 3D ground truth supervision for accurate distances.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import argparse
from torch.utils.data import DataLoader

from panoptic_bev.models.distance_head_3d import PanopticBEVWith3DDistance, DistanceLoss3D
from panoptic_bev.utils.kitti360_3d_loader import KITTI3603DLoader


def load_base_model(config_path: str):
    """
    Load the base PanopticBEV model.
    
    Args:
        config_path: Path to config file
    
    Returns:
        Base model
    """
    # Import here to avoid circular imports
    from panoptic_bev.models.panoptic_bev import PanopticBEV
    from panoptic_bev.config.config import load_config
    
    config = load_config(config_path)
    model = PanopticBEV(config)
    
    return model


class PanopticLoss(torch.nn.Module):
    """
    Combined loss for panoptic segmentation.
    Includes semantic segmentation and instance segmentation losses.
    """
    
    def __init__(self):
        super().__init__()
        self.semantic_loss = torch.nn.CrossEntropyLoss(ignore_index=255)
        self.instance_loss = torch.nn.MSELoss()
    
    def forward(self, outputs, targets):
        # Semantic loss
        if 'semantic' in outputs and 'semantic_target' in targets:
            semantic_loss = self.semantic_loss(
                outputs['semantic'], 
                targets['semantic_target']
            )
        else:
            semantic_loss = torch.tensor(0.0, device=outputs.get('device', 'cpu'))
        
        # Instance loss (if available)
        if 'instance' in outputs and 'instance_target' in targets:
            instance_loss = self.instance_loss(
                outputs['instance'],
                targets['instance_target']
            )
        else:
            instance_loss = torch.tensor(0.0, device=outputs.get('device', 'cpu'))
        
        return semantic_loss + 0.5 * instance_loss


class KITTI360Dataset3D(torch.utils.data.Dataset):
    """
    Dataset that loads images and 3D ground truth for training.
    """
    
    def __init__(self, 
                 dataset_root: str,
                 sequences: list = None,
                 transform=None):
        """
        Args:
            dataset_root: Path to KITTI-360 dataset
            sequences: List of sequence names to use (None = all)
            transform: Optional transforms to apply
        """
        self.dataset_root = Path(dataset_root)
        self.transform = transform
        
        # Initialize 3D loader
        self.loader_3d = KITTI3603DLoader(
            bboxes_root=str(self.dataset_root / "data_3d_bboxes"),
            use_train_full=True,
            use_cache=True
        )
        
        # Get sequences
        if sequences:
            self.sequences = {k: v for k, v in self.loader_3d.sequences.items() 
                            if k in sequences}
        else:
            self.sequences = self.loader_3d.sequences
        
        # Build frame index
        self.frames = []
        for seq_name in self.sequences:
            all_frames = self.loader_3d.load_sequence_annotations(seq_name)
            for frame_num in all_frames.keys():
                self.frames.append((seq_name, frame_num))
        
        print(f"Dataset initialized with {len(self.frames)} frames from {len(self.sequences)} sequences")
    
    def __len__(self):
        return len(self.frames)
    
    def __getitem__(self, idx):
        seq_name, frame_num = self.frames[idx]
        
        # Load 3D ground truth
        objects_3d = self.loader_3d.load_3d_bboxes(seq_name, frame_num)
        
        # Load image (placeholder - implement based on your data structure)
        # image = self._load_image(seq_name, frame_num)
        image = torch.zeros(3, 512, 512)  # Placeholder
        
        # Load BEV target (placeholder - implement based on your data structure)
        # bev_target = self._load_bev_target(seq_name, frame_num)
        bev_target = {
            'semantic_target': torch.zeros(1, 128, 128, dtype=torch.long),
            'instance_target': torch.zeros(1, 128, 128)
        }
        
        return {
            'image': image,
            'bev': bev_target,
            'sequence': seq_name,
            'frame': torch.tensor(frame_num),
            'objects_3d': objects_3d
        }


def train_with_3d_supervision(args):
    """
    Train with 3D ground truth supervision.
    """
    print("="*60)
    print("Training with 3D Supervision")
    print("="*60)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load base model
    print(f"\nLoading base model from: {args.config}")
    base_model = load_base_model(args.config)
    base_model = base_model.to(device)
    
    # Wrap with 3D distance head
    model = PanopticBEVWith3DDistance(base_model, enable_distance=True)
    model = model.to(device)
    
    # Initialize 3D data loader for ground truth
    loader_3d = KITTI3603DLoader(
        bboxes_root=f"{args.dataset_root}/data_3d_bboxes",
        use_train_full=True,
        use_cache=True
    )
    
    # Create dataset and dataloader
    dataset = KITTI360Dataset3D(
        dataset_root=args.dataset_root,
        sequences=args.sequences.split(',') if args.sequences else None
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=custom_collate
    )
    
    # Loss functions
    panoptic_loss = PanopticLoss()
    distance_loss = DistanceLoss3D(use_uncertainty=True)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    
    for epoch in range(args.epochs):
        model.train()
        total_loss_epoch = 0.0
        panoptic_loss_epoch = 0.0
        distance_loss_epoch = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(dataloader):
            images = batch['image'].to(device)
            targets = batch['bev']
            objects_3d_batch = batch['objects_3d']
            
            # Move targets to device
            for key in targets:
                if isinstance(targets[key], torch.Tensor):
                    targets[key] = targets[key].to(device)
            
            # Forward pass
            outputs = model(images, targets, return_3d_distances=True)
            
            # Compute panoptic loss
            loss_panoptic = panoptic_loss(outputs, targets)
            
            # Compute distance loss
            loss_distance = distance_loss(
                outputs.get('3d_distances', {}),
                objects_3d_batch,
                bev_mask=batch.get('bev_mask')
            )
            
            # Total loss
            total_loss = loss_panoptic + args.distance_weight * loss_distance
            
            # Backward
            optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            
            optimizer.step()
            
            # Accumulate losses
            total_loss_epoch += total_loss.item()
            panoptic_loss_epoch += loss_panoptic.item()
            distance_loss_epoch += loss_distance.item()
            num_batches += 1
            
            # Print progress
            if (batch_idx + 1) % args.log_interval == 0:
                print(f"  Epoch {epoch+1}/{args.epochs}, Batch {batch_idx+1}/{len(dataloader)}: "
                      f"Loss={total_loss.item():.4f} "
                      f"(Panoptic: {loss_panoptic.item():.4f}, "
                      f"Distance: {loss_distance.item():.4f})")
        
        # Update scheduler
        scheduler.step()
        
        # Print epoch summary
        avg_total = total_loss_epoch / max(num_batches, 1)
        avg_panoptic = panoptic_loss_epoch / max(num_batches, 1)
        avg_distance = distance_loss_epoch / max(num_batches, 1)
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Average Loss: {avg_total:.4f}")
        print(f"  Panoptic Loss: {avg_panoptic:.4f}")
        print(f"  Distance Loss: {avg_distance:.4f}")
        print(f"  Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0:
            checkpoint_path = Path(args.output_dir) / f"checkpoint_epoch_{epoch+1}.pth"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_total,
            }, checkpoint_path)
            
            print(f"  Saved checkpoint: {checkpoint_path}")
    
    # Save final model
    final_path = Path(args.output_dir) / "model_final.pth"
    torch.save(model.state_dict(), final_path)
    print(f"\nFinal model saved to: {final_path}")
    
    print("\n" + "="*60)
    print("Training completed!")
    print("="*60)


def custom_collate(batch):
    """
    Custom collate function to handle variable-length 3D object lists.
    """
    images = torch.stack([item['image'] for item in batch])
    
    # Collate BEV targets
    bev_keys = batch[0]['bev'].keys()
    bev = {}
    for key in bev_keys:
        if isinstance(batch[0]['bev'][key], torch.Tensor):
            bev[key] = torch.stack([item['bev'][key] for item in batch])
        else:
            bev[key] = [item['bev'][key] for item in batch]
    
    # Keep objects_3d as list of lists
    objects_3d = [item['objects_3d'] for item in batch]
    
    return {
        'image': images,
        'bev': bev,
        'sequence': [item['sequence'] for item in batch],
        'frame': torch.stack([item['frame'] for item in batch]),
        'objects_3d': objects_3d
    }


def main():
    parser = argparse.ArgumentParser(description='Train PanopticBEV with 3D supervision')
    
    # Data arguments
    parser.add_argument('--dataset_root', default=r'D:\datasets\kitti360',
                       help='Path to KITTI-360 dataset')
    parser.add_argument('--sequences', default=None,
                       help='Comma-separated list of sequences to use')
    
    # Model arguments
    parser.add_argument('--config', default='experiments/config/kitti.ini',
                       help='Path to config file')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=2,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                       help='Gradient clipping (0 to disable)')
    parser.add_argument('--distance_weight', type=float, default=0.5,
                       help='Weight for distance loss')
    
    # System arguments
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--output_dir', default='output/3d_supervision',
                       help='Output directory for checkpoints')
    
    # Logging arguments
    parser.add_argument('--log_interval', type=int, default=10,
                       help='Log every N batches')
    parser.add_argument('--save_interval', type=int, default=5,
                       help='Save checkpoint every N epochs')
    
    args = parser.parse_args()
    
    train_with_3d_supervision(args)


if __name__ == '__main__':
    main()