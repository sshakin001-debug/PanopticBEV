from math import log

import torch

try:
    from . import _backend
    HAS_BACKEND = True
except ImportError:
    HAS_BACKEND = False
    _backend = None
    import warnings
    warnings.warn("BBX CUDA backend not available. Using PyTorch fallback implementation.")


__all__ = [
    "extract_boxes",
    "shift_boxes",
    "calculate_shift",
    "corners_to_center_scale",
    "center_scale_to_corners",
    "invert_roi_bbx",
    "ious",
    "mask_overlap",
    "bbx_overlap"
]


def _extract_boxes_pytorch(mask, num_instances):
    """
    Pure PyTorch implementation of bounding box extraction from instance mask.
    
    Parameters
    ----------
    mask : torch.Tensor
        A tensor with shape H x W containing an instance segmentation mask
    num_instances : int
        The number of instances to look for
    
    Returns
    -------
    bbx : torch.Tensor
        A tensor with shape `num_instances` x 4 containing the coordinates of 
        the bounding boxes in "corners" form (y0, x0, y1, x1)
    """
    device = mask.device
    dtype = mask.dtype if torch.is_floating_point(mask) else torch.float32
    
    # Initialize output boxes
    bbx = torch.zeros(num_instances, 4, dtype=dtype, device=device)
    
    # For each instance, find the bounding box
    for i in range(num_instances):
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


def _mask_count_pytorch(bbx, int_mask):
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


def extract_boxes(mask, num_instances):
    """Calculate bounding boxes from instance segmentation mask

    Parameters
    ----------
    mask : torch.Tensor
        A tensor with shape H x W containing an instance segmentation mask
    num_instances : int
        The number of instances to look for

    Returns
    -------
    bbx : torch.Tensor
        A tensor with shape `num_instances` x 4 containing the coordinates of the bounding boxes in "corners" form

    """
    if mask.ndimension() == 2:
        mask = mask.unsqueeze(0)
    
    if HAS_BACKEND:
        return _backend.extract_boxes(mask, num_instances)
    else:
        return _extract_boxes_pytorch(mask[0], num_instances)


def shift_boxes(bbx, shift, dim=-1, scale_clip=log(1000. / 16.)):
    """Shift bounding boxes using the faster r-CNN formulas

    Each 4-vector of `bbx` and `shift` contain, respectively, bounding box coordiantes in "corners" form and shifts
    in the form `(dy, dx, dh, dw)`. The output is calculated according to the Faster r-CNN formulas:

        y_out = y_in + h_in * dy
        x_out = x_in + w_in * dx
        h_out = h_in * exp(dh)
        w_out = w_in * exp(dw)

    Parameters
    ----------
    bbx : torch.Tensor
        A tensor of bounding boxes with shape N_0 x ... x N_i = 4 x ... x N_n
    shift : torch.Tensor
        A tensor of shifts with shape N_0 x ... x N_i = 4 x ... x N_n
    dim : int
        The dimension i of the input tensors which contains the bounding box coordinates and the shifts
    scale_clip : float
        Maximum scale shift value to avoid exp overflow

    Returns
    -------
    bbx_out : torch.Tensor
        A tensor of shifted bounding boxes with shape N_0 x ... x N_i = 4 x ... x N_n

    """
    yx_in, hw_in = corners_to_center_scale(*bbx.split(2, dim=dim))
    dyx, dhw = shift.split(2, dim=dim)

    yx_out = yx_in + hw_in * dyx
    hw_out = hw_in * dhw.clamp(max=scale_clip).exp()

    return torch.cat(center_scale_to_corners(yx_out, hw_out), dim=dim)


def calculate_shift(bbx0, bbx1, dim=-1, eps=1e-5):
    """Calculate shift parameters between bounding boxes using the faster r-CNN formulas

    Each 4-vector of `bbx0` and `bbx1` contains bounding box coordiantes in "corners" form. The output is calculated
    according to the Faster r-CNN formulas:

        dy = (y1 - y0) / h0
        dx = (x1 - x0) / w0
        dh = log(h1 / h0)
        dw = log(w1 / w0)

    Parameters
    ----------
    bbx0 : torch.Tensor
        A tensor of source bounding boxes with shape N_0 x ... x N_i = 4 x ... x N_n
    bbx1 : torch.Tensor
        A tensor of target bounding boxes with shape N_0 x ... x N_i = 4 x ... x N_n
    dim : int
        The dimension `i` of the input tensors which contains the bounding box coordinates
    eps : float
        Small number used to avoid overflow

    Returns
    -------
    shift : torch.Tensor
        A tensor of calculated shifts from `bbx0` to `bbx1` with shape N_0 x ... x N_i = 4 x ... x N_n

    """
    yx0, hw0 = corners_to_center_scale(*bbx0.split(2, dim=dim))
    yx1, hw1 = corners_to_center_scale(*bbx1.split(2, dim=dim))

    hw0 = hw0.clamp(min=eps)

    dyx = (yx1 - yx0) / hw0
    dhw = (hw1 / hw0).log()

    return torch.cat([dyx, dhw], dim=dim)


def corners_to_center_scale(p0, p1):
    """Convert bounding boxes from "corners" form to "center+scale" form"""
    yx = 0.5 * (p0 + p1)
    hw = p1 - p0
    return yx, hw


def center_scale_to_corners(yx, hw):
    """Convert bounding boxes from "center+scale" form to "corners" form"""
    hw_half = 0.5 * hw
    p0 = yx - hw_half
    p1 = yx + hw_half
    return p0, p1


def invert_roi_bbx(bbx, roi_size, img_size):
    """Compute bbx coordinates to perform inverse roi sampling"""
    bbx_size = bbx[:, 2:] - bbx[:, :2]
    return torch.cat([
        -bbx.new(roi_size) * bbx[:, :2] / bbx_size,
        bbx.new(roi_size) * (bbx.new(img_size) - bbx[:, :2]) / bbx_size
    ], dim=1)


def ious(bbx0, bbx1):
    """Calculate intersection over union between sets of bounding boxes

    Parameters
    ----------
    bbx0 : torch.Tensor
        A tensor of bounding boxes in "corners" form with shape N x 4
    bbx1 : torch.Tensor
        A tensor of bounding boxes in "corners" form with shape M x 4

    Returns
    -------
    iou : torch.Tensor
        A tensor with shape N x M containing the IoUs between all pairs of bounding boxes in bbx0 and bbx1
    """
    bbx0_tl, bbx0_br = bbx0.unsqueeze(dim=1).split(2, -1)
    bbx1_tl, bbx1_br = bbx1.unsqueeze(dim=0).split(2, -1)

    # Intersection coordinates
    int_tl = torch.max(bbx0_tl, bbx1_tl)
    int_br = torch.min(bbx0_br, bbx1_br)

    intersection = (int_br - int_tl).clamp(min=0).prod(dim=-1)
    bbx0_area = (bbx0_br - bbx0_tl).prod(dim=-1)
    bbx1_area = (bbx1_br - bbx1_tl).prod(dim=-1)
    return intersection / (bbx0_area + bbx1_area - intersection)


def mask_overlap(bbx, mask):
    """Calculate overlap between a set of bounding boxes and a mask

    Parameters
    ----------
    bbx : torch.Tensor
        A tensor of bounding boxes in "corners" form with shape N x 4
    mask : torch.Tensor
        A binary tensor with shape H x W

    Returns
    -------
    overlap : torch.Tensor
        A tensor with shape N containing the proportion of non-zero pixels in each box
    """
    # Compute integral image of the mask
    int_mask = bbx.new_zeros((mask.size(0) + 1, mask.size(1) + 1))
    int_mask[1:, 1:] = mask > 0
    int_mask = int_mask.cumsum(0).cumsum(1)

    if HAS_BACKEND:
        count = _backend.mask_count(bbx, int_mask)
    else:
        count = _mask_count_pytorch(bbx, int_mask)
    
    area = (bbx[:, 2:] - bbx[:, :2]).prod(dim=1)

    return count / area


def bbx_overlap(bbx0, bbx1):
    """Calculate intersection over area between two sets of bounding boxes

    Intersection over area is defined as:
        area(inter(bbx0, bbx1)) / area(bbx0)

    Parameters
    ----------
    bbx0 : torch.Tensor
        A tensor of bounding boxes in "corners" form with shape N x 4
    bbx1 : torch.Tensor
        A tensor of bounding boxes in "corners" form with shape M x 4

    Returns
    -------
    ratios : torch.Tensor
        A tensor with shape N x M containing the intersection over areas between all pairs of bounding boxes
    """
    bbx0_tl, bbx0_br = bbx0.unsqueeze(dim=1).split(2, -1)
    bbx1_tl, bbx1_br = bbx1.unsqueeze(dim=0).split(2, -1)

    # Intersection coordinates
    int_tl = torch.max(bbx0_tl, bbx1_tl)
    int_br = torch.min(bbx0_br, bbx1_br)

    intersection = (int_br - int_tl).clamp(min=0).prod(dim=-1)
    bbx0_area = (bbx0_br - bbx0_tl).prod(dim=-1)

    return intersection / bbx0_area
