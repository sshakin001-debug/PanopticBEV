#!/usr/bin/env python3
"""
Generate correct split files for KITTI-360 dataset by scanning actual files.
"""
import os
from pathlib import Path

# Directories
seam_root_dir = r"D:\kitti360_panopticbev"
split_dir = os.path.join(seam_root_dir, "split")
front_msk_dir = os.path.join(seam_root_dir, "front_msk_trainid", "front")

os.makedirs(split_dir, exist_ok=True)

# Scan actual front mask files
print("Scanning front mask files...")
front_msk_files = os.listdir(front_msk_dir)
front_msk_frames = [f.replace('.png', '') for f in front_msk_files if f.endswith('.png')]
print(f"Found {len(front_msk_frames)} front mask files")

# Group by scene
scenes = {}
for frame_id in front_msk_frames:
    parts = frame_id.split(';')
    if len(parts) == 2:
        scene_name = parts[0]
        frame_num = int(parts[1])
        if scene_name not in scenes:
            scenes[scene_name] = []
        scenes[scene_name].append(frame_num)

# Sort and report
for scene_name in sorted(scenes.keys()):
    frames = sorted(scenes[scene_name])
    print(f"  {scene_name}: {len(frames)} frames, range {min(frames)} to {max(frames)}")

# Generate train.txt - use all available frames
train_lines = sorted(front_msk_frames)

train_path = os.path.join(split_dir, "train.txt")
with open(train_path, 'w') as f:
    f.write('\n'.join(train_lines))

print(f"\nCreated {train_path}")
print(f"Total training frames: {len(train_lines)}")

# Generate val.txt - use first 100 frames of first scene
first_scene = sorted(scenes.keys())[0]
first_scene_frames = sorted(scenes[first_scene])[:100]
val_lines = [f"{first_scene};{f:010d}" for f in first_scene_frames]

val_path = os.path.join(split_dir, "val.txt")
with open(val_path, 'w') as f:
    f.write('\n'.join(val_lines))

print(f"\nCreated {val_path}")
print(f"Total validation frames: {len(val_lines)}")

# Show first few lines
print("\nFirst 5 lines of train.txt:")
for line in train_lines[:5]:
    print(f"  {line}")

print("\nLast 5 lines of train.txt:")
for line in train_lines[-5:]:
    print(f"  {line}")
