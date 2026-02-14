import torch

try:
    from . import _backend
    HAS_BACKEND = True
except ImportError:
    HAS_BACKEND = False
    _backend = None
    import warnings
    warnings.warn("NMS CUDA backend not available. Using PyTorch fallback implementation.")


def _compute_iou(box, boxes):
    """
    Compute IoU between a single box and a set of boxes.
    
    Parameters
    ----------
    box : torch.Tensor
        A tensor with shape 4 (y0, x0, y1, x1)
    boxes : torch.Tensor
        A tensor with shape N x 4
    
    Returns
    -------
    iou : torch.Tensor
        A tensor with shape N containing IoU values
    """
    # Intersection coordinates
    y0 = torch.max(box[0], boxes[:, 0])
    x0 = torch.max(box[1], boxes[:, 1])
    y1 = torch.min(box[2], boxes[:, 2])
    x1 = torch.min(box[3], boxes[:, 3])
    
    # Intersection area
    intersection = torch.clamp(y1 - y0, min=0) * torch.clamp(x1 - x0, min=0)
    
    # Union area
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = box_area + boxes_area - intersection
    
    return intersection / (union + 1e-6)


def _nms_pytorch(bbx, scores, threshold=0.5, n_max=-1):
    """
    Pure PyTorch implementation of Non-Maximum Suppression.
    
    Parameters
    ----------
    bbx : torch.Tensor
        A tensor of bounding boxes with shape N x 4
    scores : torch.Tensor
        A tensor of bounding box scores with shape N
    threshold : float
        The minimum IoU value for a pair of bounding boxes to be considered a match
    n_max : int
        Maximum number of bounding boxes to select. If n_max <= 0, keep all surviving boxes
    
    Returns
    -------
    selection : torch.Tensor
        A tensor with the indices of the selected boxes
    """
    if bbx.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=bbx.device)
    
    # Sort by scores in descending order
    order = torch.argsort(scores, descending=True)
    
    # Keep track of suppressed boxes
    suppressed = torch.zeros(len(bbx), dtype=torch.bool, device=bbx.device)
    
    selected = []
    
    for i, idx in enumerate(order):
        if suppressed[idx]:
            continue
        
        selected.append(idx.item())
        
        if n_max > 0 and len(selected) >= n_max:
            break
        
        # Suppress overlapping boxes
        current_box = bbx[idx]
        remaining_indices = order[i+1:]
        
        for j in remaining_indices:
            if suppressed[j]:
                continue
            
            iou = _compute_iou(current_box, bbx[j].unsqueeze(0))[0]
            if iou > threshold:
                suppressed[j] = True
    
    return torch.tensor(selected, dtype=torch.long, device=bbx.device)


def nms(bbx, scores, threshold=0.5, n_max=-1):
    """Perform non-maxima suppression

    Select up to n_max bounding boxes from bbx, giving priorities to bounding boxes with greater scores. Each selected
    bounding box suppresses all other not yet selected boxes that intersect it by more than the given threshold.

    Parameters
    ----------
    bbx : torch.Tensor
        A tensor of bounding boxes with shape N x 4
    scores : torch.Tensor
        A tensor of bounding box scores with shape N
    threshold : float
        The minimum iou value for a pair of bounding boxes to be considered a match
    n_max : int
        Maximum number of bounding boxes to select. If n_max <= 0, keep all surviving boxes

    Returns
    -------
    selection : torch.Tensor
        A tensor with the indices of the selected boxes

    """
    if HAS_BACKEND:
        selection = _backend.nms(bbx, scores, threshold, n_max)
    else:
        selection = _nms_pytorch(bbx, scores, threshold, n_max)
    
    return selection.to(device=bbx.device)


__all__ = ["nms"]
