import torch
import torch.autograd as autograd
import torch.nn.functional as F
from torch.autograd.function import once_differentiable

try:
    from . import _backend
    HAS_BACKEND = True
except ImportError:
    HAS_BACKEND = False
    _backend = None
    import warnings
    warnings.warn("ROI sampling CUDA backend not available. Using PyTorch fallback implementation.")


# Fallback enums for when CUDA backend is not available
class InterpolationFallback:
    Bilinear = 0
    Nearest = 1


class PaddingModeFallback:
    Zero = 0
    Border = 1


if HAS_BACKEND:
    _INTERPOLATION = {"bilinear": _backend.Interpolation.Bilinear, "nearest": _backend.Interpolation.Nearest}
    _PADDING = {"zero": _backend.PaddingMode.Zero, "border": _backend.PaddingMode.Border}
else:
    _INTERPOLATION = {"bilinear": InterpolationFallback.Bilinear, "nearest": InterpolationFallback.Nearest}
    _PADDING = {"zero": PaddingModeFallback.Zero, "border": PaddingModeFallback.Border}


def _roi_sampling_forward_pytorch(x, bbx, idx, roi_size, interpolation, padding, valid_mask):
    """
    Pure PyTorch fallback implementation of ROI sampling forward pass.
    
    Uses grid_sample for bilinear interpolation.
    """
    # Store original dtype and convert to float if needed
    # F.grid_sample on CUDA does not support Long (int64) tensors
    original_dtype = x.dtype
    needs_conversion = not torch.is_floating_point(x)
    if needs_conversion:
        x = x.float()
    
    batch_size, num_channels, height, width = x.shape
    num_rois = bbx.size(0)
    roi_h, roi_w = roi_size
    
    # Initialize output (always use current x dtype, which is float if conversion was needed)
    output = torch.zeros(num_rois, num_channels, roi_h, roi_w, 
                        dtype=x.dtype, device=x.device)
    
    if valid_mask:
        mask_out = torch.zeros(num_rois, roi_h, roi_w, 
                              dtype=torch.bool, device=x.device)
    else:
        mask_out = None
    
    # Create sampling grid for all ROIs at once
    # bbx format: [y0, x0, y1, x1]
    for roi_idx in range(num_rois):
        batch_idx = idx[roi_idx].item()
        y0, x0, y1, x1 = bbx[roi_idx].tolist()
        
        # ROI dimensions
        roi_height = max(y1 - y0, 1.0)
        roi_width = max(x1 - x0, 1.0)
        
        # Create normalized grid coordinates
        # Grid coordinates are in [-1, 1] range for grid_sample
        grid_y = torch.linspace(0, roi_h - 1, roi_h, device=x.device, dtype=x.dtype)
        grid_x = torch.linspace(0, roi_w - 1, roi_w, device=x.device, dtype=x.dtype)
        
        # Map ROI coordinates to image coordinates
        # y_img = y0 + (y_roi + 0.5) / roi_h * (y1 - y0)
        # x_img = x0 + (x_roi + 0.5) / roi_w * (x1 - x0)
        y_coords = y0 + (grid_y + 0.5) * roi_height / roi_h
        x_coords = x0 + (grid_x + 0.5) * roi_width / roi_w
        
        # Normalize to [-1, 1] for grid_sample
        y_norm = 2.0 * y_coords / height - 1.0
        x_norm = 2.0 * x_coords / width - 1.0
        
        # Create meshgrid
        grid_yy, grid_xx = torch.meshgrid(y_norm, x_norm, indexing='ij')
        grid = torch.stack([grid_xx, grid_yy], dim=-1).unsqueeze(0)  # 1 x H x W x 2
        
        # Sample using grid_sample
        padding_mode = 'border' if padding == PaddingModeFallback.Border else 'zeros'
        mode = 'bilinear' if interpolation == InterpolationFallback.Bilinear else 'nearest'
        
        sampled = F.grid_sample(
            x[batch_idx:batch_idx+1].float(),  # Make sure input is float
            grid.float(),  # Make sure grid is float
            mode=mode,
            padding_mode=padding_mode,
            align_corners=False
        )
        
        output[roi_idx] = sampled[0]
        
        # Compute valid mask if requested
        if valid_mask:
            # Check which sample points are within the image bounds
            valid_y = (y_coords >= 0) & (y_coords < height)
            valid_x = (x_coords >= 0) & (x_coords < width)
            valid_grid = valid_y.unsqueeze(1) & valid_x.unsqueeze(0)
            mask_out[roi_idx] = valid_grid
    
    if valid_mask:
        return output, mask_out
    return output, None


def _roi_sampling_backward_pytorch(dy, bbx, idx, input_shape, interpolation, padding):
    """
    Pure PyTorch fallback implementation of ROI sampling backward pass.
    
    Uses grid_sample's gradient computation.
    """
    batch_size, num_channels, height, width = input_shape
    num_rois = dy.size(0)
    roi_h, roi_w = dy.size(2), dy.size(3)
    
    # Initialize gradient output
    dx = torch.zeros(batch_size, num_channels, height, width, 
                    dtype=dy.dtype, device=dy.device)
    
    for roi_idx in range(num_rois):
        batch_idx = idx[roi_idx].item()
        y0, x0, y1, x1 = bbx[roi_idx].tolist()
        
        # ROI dimensions
        roi_height = max(y1 - y0, 1.0)
        roi_width = max(x1 - x0, 1.0)
        
        # Create normalized grid coordinates
        grid_y = torch.linspace(0, roi_h - 1, roi_h, device=dy.device, dtype=dy.dtype)
        grid_x = torch.linspace(0, roi_w - 1, roi_w, device=dy.device, dtype=dy.dtype)
        
        y_coords = y0 + (grid_y + 0.5) * roi_height / roi_h
        x_coords = x0 + (grid_x + 0.5) * roi_width / roi_w
        
        y_norm = 2.0 * y_coords / height - 1.0
        x_norm = 2.0 * x_coords / width - 1.0
        
        grid_yy, grid_xx = torch.meshgrid(y_norm, x_norm, indexing='ij')
        grid = torch.stack([grid_xx, grid_yy], dim=-1).unsqueeze(0)
        
        # For backward pass, we need to use autograd
        grid.requires_grad_(True)
        
        # Create a zero input for gradient computation
        dummy_input = torch.zeros(1, num_channels, height, width, 
                                 dtype=dy.dtype, device=dy.device, requires_grad=True)
        
        padding_mode = 'border' if padding == PaddingModeFallback.Border else 'zeros'
        mode = 'bilinear' if interpolation == InterpolationFallback.Bilinear else 'nearest'
        
        sampled = F.grid_sample(
            dummy_input, 
            grid,
            mode=mode,
            padding_mode=padding_mode,
            align_corners=False
        )
        
        # Backprop to get gradient w.r.t. input
        sampled.backward(dy[roi_idx:roi_idx+1])
        
        # Accumulate gradient
        dx[batch_idx] += dummy_input.grad[0]
    
    return dx


class ROISampling(autograd.Function):
    @staticmethod
    def forward(ctx, x, bbx, idx, roi_size, interpolation, padding, valid_mask):
        ctx.save_for_backward(bbx, idx)
        ctx.input_shape = (x.size(0), x.size(2), x.size(3))
        ctx.valid_mask = valid_mask

        try:
            ctx.interpolation = _INTERPOLATION[interpolation]
        except KeyError:
            raise ValueError("Unknown interpolation {}".format(interpolation))
        try:
            ctx.padding = _PADDING[padding]
        except KeyError:
            raise ValueError("Unknown padding {}".format(padding))

        if HAS_BACKEND:
            y, mask = _backend.roi_sampling_forward(x, bbx, idx, roi_size, ctx.interpolation, ctx.padding, valid_mask)
        else:
            y, mask = _roi_sampling_forward_pytorch(x, bbx, idx, roi_size, ctx.interpolation, ctx.padding, valid_mask)

        if not torch.is_floating_point(x):
            ctx.mark_non_differentiable(y)
        if valid_mask:
            ctx.mark_non_differentiable(mask)
            return y, mask
        else:
            return y

    @staticmethod
    @once_differentiable
    def backward(ctx, *args):
        if ctx.valid_mask:
            dy, _ = args
        else:
            dy = args[0]

        assert torch.is_floating_point(dy), "ROISampling.backward is only defined for floating point types"
        bbx, idx = ctx.saved_tensors

        if HAS_BACKEND:
            dx = _backend.roi_sampling_backward(dy, bbx, idx, ctx.input_shape, ctx.interpolation, ctx.padding)
        else:
            dx = _roi_sampling_backward_pytorch(dy, bbx, idx, ctx.input_shape, ctx.interpolation, ctx.padding)
        
        return dx, None, None, None, None, None, None


def roi_sampling(x, bbx, idx, roi_size, interpolation="bilinear", padding="border", valid_mask=False):
    """Sample ROIs from a batch of images using bi-linear interpolation

    ROIs are sampled from the input by bi-linear interpolation, using the following equations to transform from
    ROI coordinates to image coordinates:

        y_img = y0 + y_roi / h_roi * (y1 - y0),     for y_roi in range(0, h_roi)
        x_img = x0 + x_roi / w_roi * (x1 - x0),     for x_roi in range(0, w_roi)

    where `(h_roi, w_roi)` is the shape of the ROI and `(y0, x0, y1, x1)` are its bounding box coordinates on the image

    Parameters
    ----------
    x : torch.Tensor
        A tensor with shape N x C x H x W containing a batch of images to sample from
    bbx : torch.Tensor
        A tensor with shape K x 4 containing the bounding box coordinates of the ROIs in "corners" format
    idx : torch.Tensor
        A tensor with shape K containing the batch indices of the image each ROI should be sampled from
    roi_size : tuple of int
        The size `(h_roi, w_roi)` of the output ROIs
    interpolation : str
        Sampling mode, one of "bilinear" or "nearest"
    padding : str
        Padding mode, one of "border" or "zero"
    valid_mask : bool
        If `True` also return a mask tensor that indicates which points of the outputs where sampled from within the
        valid region of the input

    Returns
    -------
    y : torch.Tensor
        A tensor with shape K x C x h_roi x w_roi containing the sampled ROIs
    mask : torch.Tensor
        Optional output returned only when valid_mask is `True`: a mask tensor with shape K x h_roi x w_roi, whose
        entries are `!= 0` where the corresponding location in `y` was sampled from within the limits of the input image
    """
    return ROISampling.apply(x, bbx, idx, roi_size, interpolation, padding, valid_mask)


__all__ = ["roi_sampling"]
