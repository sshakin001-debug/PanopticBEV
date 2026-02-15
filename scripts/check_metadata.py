#!/usr/bin/env python3
"""Check metadata and debug dataset loading"""
import umsgpack
from pathlib import Path

seam_root = Path(r"D:\kitti360_panopticbev")
metadata_file = seam_root / "metadata_ortho.bin"

with open(metadata_file, "rb") as fid:
    metadata = umsgpack.unpack(fid, encoding="utf-8")

print(f"Total images in metadata: {len(metadata['images'])}")
print(f"\nFirst 5 image IDs:")
for img in metadata['images'][:5]:
    print(f"  {img['id']}")

# Check if frame 3810 is in metadata
frame_id = "2013_05_28_drive_0000_sync;0000003810"
found = any(img['id'] == frame_id for img in metadata['images'])
print(f"\nFrame {frame_id} in metadata: {found}")

# Check what frames are in front_msk_trainid
front_msk_dir = seam_root / "front_msk_trainid" / "front"
front_frames = [f.stem for f in front_msk_dir.glob("*.png")]
print(f"\nTotal frames in front_msk_trainid: {len(front_frames)}")
print(f"First 5: {front_frames[:5]}")

# Check if frame 3810 is in front_msk
frame_in_front = frame_id in front_frames
print(f"\nFrame {frame_id} in front_msk: {frame_in_front}")

# Check the img_map
import json
img_dir = seam_root / "img"
with open(img_dir / "front.json", 'r') as f:
    front_data = json.load(f)

img_map = {k: v for d in front_data for k, v in d.items()}
print(f"\nTotal entries in front.json: {len(img_map)}")

# Check if frame 3810 is in img_map
frame_key = f"{frame_id}.png"
if frame_key in img_map:
    print(f"Frame {frame_key} path: {img_map[frame_key]}")
else:
    print(f"Frame {frame_key} NOT in img_map")
