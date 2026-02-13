"""Quick test to verify dataset paths work correctly."""
from pathlib import Path
import sys


def test_paths():
    # Your paths
    DATASET_ROOT = Path("D:/datasets/kitti360")
    SEAM_ROOT = Path("D:/kitti360_panopticbev")
    
    print("Testing dataset paths...")
    print(f"Dataset: {DATASET_ROOT}")
    print(f"PanopticBEV: {SEAM_ROOT}")
    
    # Test 1: Paths exist
    assert DATASET_ROOT.exists(), f"Dataset not found: {DATASET_ROOT}"
    assert SEAM_ROOT.exists(), f"PanopticBEV data not found: {SEAM_ROOT}"
    print("Both paths exist")
    
    # Test 2: Key folders present
    assert (DATASET_ROOT / "calibration").exists(), "Missing calibration"
    assert (DATASET_ROOT / "data_2d_raw").exists(), "Missing data_2d_raw"
    print("KITTI-360 structure valid")
    
    assert (SEAM_ROOT / "bev_msk").exists(), "Missing bev_msk"
    assert (SEAM_ROOT / "img").exists(), "Missing img"
    assert (SEAM_ROOT / "split").exists(), "Missing split"
    print("PanopticBEV structure valid")
    
    # Test 3: Path manipulation (Windows/Unix compatibility)
    test_file = SEAM_ROOT / "split" / "train.txt"
    print(f"  Example path: {test_file}")
    assert test_file.parent.exists(), f"Parent doesn't exist: {test_file.parent}"
    print("Path manipulation works")
    
    print("\nAll path tests passed!")
    print("You can now run training with:")
    print(f'  --dataset_root_dir "{DATASET_ROOT}"')
    print(f'  --seam_root_dir "{SEAM_ROOT}"')


if __name__ == '__main__':
    test_paths()