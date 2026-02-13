#!/usr/bin/env python3
"""
Auto-converted from eval_panoptic_bev_nuscenes.sh
Windows-compatible runner script for PanopticBEV
"""
import subprocess
import sys
import os
from pathlib import Path

# Original environment variables from shell script
extra_env = {}

# Merge with current environment
env = {**os.environ, **extra_env}

# Ensure PYTHONPATH includes repo root
repo_root = Path(__file__).parent.parent.absolute()
env["PYTHONPATH"] = str(repo_root) + os.pathsep + env.get("PYTHONPATH", "")

# Windows-specific: Convert paths if needed
path_keys = ["dataset_root_dir", "seam_root_dir", "output_dir", "config_file", "pretrained_model"]
for key in path_keys:
    if key in env:
        env[key] = str(Path(env[key]))

# Build the command
cmd = [sys.executable] + ['-m', 'torch.distributed.launch', '--nproc_per_node=1', '--master_addr={IP', 'ADDR}', '--master_port={PORT', 'NUM}', 'eval_panoptic_bev.py', '\\']
print(f"Running: {' '.join(cmd)}")
print(f"Working directory: {repo_root}")

# Run the command
try:
    subprocess.run(cmd, env=env, check=True, cwd=repo_root)
except subprocess.CalledProcessError as e:
    print(f"Command failed with exit code {e.returncode}")
    sys.exit(e.returncode)
