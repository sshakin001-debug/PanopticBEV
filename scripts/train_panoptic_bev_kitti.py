#!/usr/bin/env python3
import subprocess
import sys
import os
from pathlib import Path

repo_root = Path(__file__).parent.parent.absolute()
env = os.environ.copy()
env["PYTHONPATH"] = str(repo_root)

# Simple command - no distributed launch for Windows single GPU
cmd = [
    sys.executable,
    str(repo_root / "train_panoptic_bev.py"),
    "--cfg", str(repo_root / "experiments" / "config" / "kitti.ini"),
    "dataset_root_dir", r"D:\datasets\kitti360",
    "seam_root_dir", r"D:\kitti360_panopticbev"
]

print(f"Running: {' '.join(cmd)}")
subprocess.run(cmd, env=env, cwd=str(repo_root), check=True)
