# Windows Setup Guide for PanopticBEV

This guide provides instructions for setting up and running PanopticBEV on Windows, including support for RTX 5060 Ti (sm_120) and other modern GPUs.

## Quick Start

1. **Validate your environment:**
   ```cmd
   python scripts/validate_windows_env.py
   ```

2. **Build CUDA extensions:**
   ```cmd
   python scripts/setup_cuda_extensions.py
   ```

3. **Generate Windows scripts:**
   ```cmd
   python scripts/generate_bat_files.py
   ```

4. **Edit the batch files** to set your dataset paths, then run:
   ```cmd
   scripts\train_kitti.bat
   ```

## Detailed Setup

### Prerequisites

- Windows 10/11
- Python 3.8 or higher
- CUDA 11.8 or higher (matching your PyTorch CUDA version)
- Visual Studio Build Tools 2019 or 2022
- NVIDIA GPU with compute capability 7.0 or higher

### Installation

1. **Create a virtual environment:**
   ```cmd
   python -m venv panoptic_bev_venv
   panoptic_bev_venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```cmd
   pip install -r requirements.txt
   ```

3. **Build CUDA extensions:**
   ```cmd
   python scripts/setup_cuda_extensions.py
   ```

### Running Training/Evaluation

#### Option 1: Using Batch Files (Recommended)

Generate and use batch files:
```cmd
python scripts/generate_bat_files.py
```

Edit `scripts\train_kitti.bat` or `scripts\train_nuscenes.bat` to set your dataset paths, then run:
```cmd
scripts\train_kitti.bat
```

#### Option 2: Using Python Scripts

Convert shell scripts to Python:
```cmd
python scripts/convert_sh_to_py.py
```

Then run:
```cmd
python scripts/train_panoptic_bev_kitti.py
```

#### Option 3: Direct Python Execution

```cmd
set dataset_root_dir=D:\datasets\kitti
set seam_root_dir=D:\datasets\panopticbev\seam
set output_dir=D:\outputs\panopticbev

python scripts/train_panoptic_bev.py --config experiments\config\kitti.ini
```

## Windows Compatibility Features

### Path Handling

The codebase now uses `pathlib` for cross-platform path handling:

```python
from panoptic_bev.utils.config_resolver import WindowsPathResolver

config = WindowsPathResolver("experiments/config/kitti.ini").get_config()
```

### DataLoader

Use the Windows-safe DataLoader wrapper:

```python
from panoptic_bev.utils.windows_dataloader import create_safe_dataloader

dataloader = create_safe_dataloader(
    dataset=my_dataset,
    batch_size=4,
    num_workers=0  # Use 0 on Windows for stability
)
```

### CPU Affinity

Optimize CPU settings for Windows:

```python
from panoptic_bev.utils.windows_cpu_affinity import set_cpu_affinity

set_cpu_affinity()
```

### Symbolic Links

Create cross-platform symlinks:

```python
from panoptic_bev.utils.windows_symlink import create_symlink

create_symlink(
    src="D:/datasets/kitti",
    dst="./data/kitti_link"
)
```

## Memory Optimization for RTX 5060 Ti

The code automatically detects VRAM and adjusts settings:

| VRAM | Batch Size | AMP | Workers | Gradient Accumulation |
|------|------------|-----|---------|----------------------|
| 16GB | 4 | Yes | 2 | 1 |
| 8GB  | 1 | Yes | 0 | 4 |

## Troubleshooting

### Issue: "CUDA out of memory"

**Solution:** Reduce batch size or enable gradient accumulation:
```python
from panoptic_bev.utils.windows_dataloader import get_optimal_batch_size

config = get_optimal_batch_size()
# config = {'batch_size': 1, 'amp': True, 'gradient_accumulation': 4}
```

### Issue: "RuntimeError: DataLoader worker exited unexpectedly"

**Solution:** Set num_workers=0 on Windows:
```python
dataloader = create_safe_dataloader(dataset, batch_size=4, num_workers=0)
```

### Issue: "CUDA extension build failed"

**Solution:** 
1. Ensure Visual Studio Build Tools are installed
2. Run from "Developer Command Prompt for VS 2019/2022"
3. Check CUDA version matches PyTorch CUDA version

### Issue: "Permission denied creating symlinks"

**Solution:** Enable Windows Developer Mode or run as Administrator. Alternatively, the code will automatically use directory junctions instead.

### Issue: "Module not found"

**Solution:** Ensure PYTHONPATH includes the repository root:
```cmd
set PYTHONPATH=%CD%;%PYTHONPATH%
```

## File Reference

### Scripts

| Script | Purpose |
|--------|---------|
| `scripts/validate_windows_env.py` | Check environment setup |
| `scripts/setup_cuda_extensions.py` | Build CUDA extensions |
| `scripts/generate_bat_files.py` | Generate .bat files |
| `scripts/convert_sh_to_py.py` | Convert .sh to .py scripts |

### Utilities

| Module | Purpose |
|--------|---------|
| `panoptic_bev/utils/config_resolver.py` | Cross-platform config loading |
| `panoptic_bev/utils/windows_dataloader.py` | Windows-safe DataLoader |
| `panoptic_bev/utils/windows_cpu_affinity.py` | CPU optimization |
| `panoptic_bev/utils/windows_symlink.py` | Cross-platform symlinks |

### Tools

| Tool | Purpose |
|------|---------|
| `tools/refactor_to_pathlib.py` | Convert os.path to pathlib |

## Additional Notes

- **Windows Defender:** Consider excluding your dataset and code directories from real-time scanning for better performance.
- **Indexing:** Disable Windows Search indexing on dataset folders.
- **Power Settings:** Set power plan to "High Performance" for consistent GPU clocks.

## Support

For issues specific to Windows compatibility, please check:
1. Run `scripts/validate_windows_env.py` and verify all checks pass
2. Ensure CUDA extensions are built successfully
3. Try running with `num_workers=0` if you encounter DataLoader issues
