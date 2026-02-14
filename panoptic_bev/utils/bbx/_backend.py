"""
Fallback implementations for BBX CUDA operations.
These are pure PyTorch implementations used when CUDA extensions are not available.
"""
import torch


def extract_boxes(mask: torch.Tensor, n_instances: int) -> torch.Tensor:
    """
    Pure PyTorch implementation of bounding box extraction from instance mask.
    
    Parameters
    ----------
    mask : torch.Tensor
        A tensor with shape 1 x H x W containing an instance segmentation mask
    n_instances : int
        The number of instances to look for
    
    Returns
    -------
    bbx : torch.Tensor
        A tensor with shape `n_instances` x 4 containing the coordinates of 
        the bounding boxes in "corners" form (y0, x0, y1, x1)
    """
    device = mask.device
    dtype = mask.dtype if torch.is_floating_point(mask) else torch.float32
    
    # Work with 2D mask
    if mask.dim() == 3:
        mask = mask[0]
    
    # Initialize output boxes
    bbx = torch.zeros(n_instances, 4, dtype=dtype, device=device)
    
    # For each instance, find the bounding box
    for i in range(n_instances):
        instance_mask = (mask == i + 1)  # Instance IDs start from 1
        
        if instance_mask.any():
            # Find non-zero coordinates
            rows = torch.any(instance_mask, dim=1)
            cols = torch.any(instance_mask, dim=0)
            
            if rows.any() and cols.any():
                y_indices = torch.where(rows)[0]
                x_indices = torch.where(cols)[0]
                
                bbx[i, 0] = y_indices[0].float()  # y0
                bbx[i, 1] = x_indices[0].float()  # x0
                bbx[i, 2] = y_indices[-1].float() + 1  # y1
                bbx[i, 3] = x_indices[-1].float() + 1  # x1
    
    return bbx


def mask_count(bbx: torch.Tensor, int_mask: torch.Tensor) -> torch.Tensor:
    """
    Pure PyTorch implementation of mask counting using integral image.
    
    Parameters
    ----------
    bbx : torch.Tensor
        A tensor of bounding boxes in "corners" form with shape N x 4
    int_mask : torch.Tensor
        An integral image with shape (H+1) x (W+1)
    
    Returns
    -------
    count : torch.Tensor
        A tensor with shape N containing the count of non-zero pixels in each box
    """
    num_boxes = bbx.size(0)
    counts = torch.zeros(num_boxes, dtype=int_mask.dtype, device=int_mask.device)
    
    for i in range(num_boxes):
        y0, x0, y1, x1 = bbx[i].long().tolist()
        
        # Clamp to valid range
        h, w = int_mask.size(0) - 1, int_mask.size(1) - 1
        y0 = max(0, min(y0, h))
        y1 = max(0, min(y1, h))
        x0 = max(0, min(x0, w))
        x1 = max(0, min(x1, w))
        
        # Sum using integral image: S = I[y1,x1] - I[y0,x1] - I[y1,x0] + I[y0,x0]
        counts[i] = (int_mask[y1, x1] - int_mask[y0, x1] - 
                     int_mask[y1, x0] + int_mask[y0, x0])
    
    return counts
