#!/usr/bin/env python3
"""
Generate split files from the actual available frames in front.json
"""
import json
from pathlib import Path

def main():
    # Paths
    seam_root = Path(r"D:\kitti360_panopticbev")
    front_json_path = seam_root / "img" / "front.json"
    split_dir = seam_root / "split"
    
    # Load front.json
    with open(front_json_path, 'r') as f:
        front_data = json.load(f)
    
    # Extract frame IDs
    frame_ids = []
    for item in front_data:
        for key in item.keys():
            # Key format: "2013_05_28_drive_0000_sync;0000000009.png"
            # We need: "2013_05_28_drive_0000_sync;0000000009"
            frame_id = key.replace('.png', '')
            frame_ids.append(frame_id)
    
    print(f"Total frames found in front.json: {len(frame_ids)}")
    
    # Group by scene
    scenes = {}
    for frame_id in frame_ids:
        scene, frame = frame_id.split(';')
        if scene not in scenes:
            scenes[scene] = []
        scenes[scene].append(int(frame))
    
    for scene, frames in scenes.items():
        print(f"Scene {scene}: {len(frames)} frames (min={min(frames)}, max={max(frames)})")
    
    # Create split files
    # Use 90% for training, 10% for validation
    train_frames = []
    val_frames = []
    
    for scene, frames in scenes.items():
        frames_sorted = sorted(frames)
        split_idx = int(len(frames_sorted) * 0.9)
        
        for frame in frames_sorted[:split_idx]:
            train_frames.append(f"{scene};{frame:010d}")
        
        for frame in frames_sorted[split_idx:]:
            val_frames.append(f"{scene};{frame:010d}")
    
    # Write train.txt
    train_path = split_dir / "train.txt"
    with open(train_path, 'w') as f:
        for frame_id in train_frames:
            f.write(frame_id + '\n')
    print(f"Wrote {len(train_frames)} frames to {train_path}")
    
    # Write val.txt
    val_path = split_dir / "val.txt"
    with open(val_path, 'w') as f:
        for frame_id in val_frames:
            f.write(frame_id + '\n')
    print(f"Wrote {len(val_frames)} frames to {val_path}")
    
    # Show first few lines
    print("\nFirst 5 lines of train.txt:")
    for line in train_frames[:5]:
        print(f"  {line}")

if __name__ == "__main__":
    main()
