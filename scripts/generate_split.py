#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate correct split file for KITTI-360 training.

This script generates the train.txt split file with correct frame indices:
- Scene 0000: frames 0 to 11517 (11,518 frames total)
- Scene 0002: frames 4391 to 18997 (14,607 frames total)

The issue was that the original split file assumed both scenes start from
frame 0, but scene 0002 actually starts from frame 4391.
"""

import os
import shutil
from datetime import datetime
from pathlib import Path


def backup_existing_file(filepath: str) -> str | None:
    """
    Backup existing file with timestamp.
    
    Args:
        filepath: Path to the file to backup
        
    Returns:
        Path to the backup file, or None if no file existed
    """
    if os.path.exists(filepath):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"{filepath}.backup_{timestamp}"
        shutil.copy2(filepath, backup_path)
        return backup_path
    return None


def generate_split_file(output_path: str) -> dict:
    """
    Generate the correct split file for KITTI-360 training.
    
    Args:
        output_path: Path to write the split file
        
    Returns:
        Dictionary with statistics about generated frames
    """
    # Scene configurations based on actual frame data
    scenes = [
        {
            "name": "2013_05_28_drive_0000_sync",
            "start_frame": 0,
            "end_frame": 11517,  # inclusive
        },
        {
            "name": "2013_05_28_drive_0002_sync",
            "start_frame": 4391,
            "end_frame": 18997,  # inclusive
        },
    ]
    
    stats = {
        "total_frames": 0,
        "scenes": {},
    }
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Generate split file
    with open(output_path, 'w') as f:
        for scene in scenes:
            scene_name = scene["name"]
            start = scene["start_frame"]
            end = scene["end_frame"]
            
            frame_count = end - start + 1
            stats["scenes"][scene_name] = {
                "start_frame": start,
                "end_frame": end,
                "frame_count": frame_count,
            }
            stats["total_frames"] += frame_count
            
            # Write frames in 10-digit zero-padded format
            for frame_idx in range(start, end + 1):
                frame_str = f"{frame_idx:010d}"
                f.write(f"{scene_name} {frame_str}\n")
    
    return stats


def main():
    """Main entry point."""
    # Output path for the split file
    output_path = r"D:\kitti360_panopticbev\split\train.txt"
    
    print("=" * 60)
    print("KITTI-360 Split File Generator")
    print("=" * 60)
    print()
    
    # Backup existing file
    print(f"Output path: {output_path}")
    backup_path = backup_existing_file(output_path)
    if backup_path:
        print(f"Backed up existing file to: {backup_path}")
    else:
        print("No existing file to backup.")
    print()
    
    # Generate new split file
    print("Generating split file...")
    stats = generate_split_file(output_path)
    
    # Print statistics
    print()
    print("Generation complete!")
    print("-" * 60)
    print("Scene statistics:")
    for scene_name, scene_stats in stats["scenes"].items():
        print(f"  {scene_name}:")
        print(f"    Start frame: {scene_stats['start_frame']:010d}")
        print(f"    End frame:   {scene_stats['end_frame']:010d}")
        print(f"    Frame count: {scene_stats['frame_count']:,}")
    print("-" * 60)
    print(f"Total frames: {stats['total_frames']:,}")
    print(f"Output file: {output_path}")
    print("=" * 60)
    
    # Verify file was created
    if os.path.exists(output_path):
        file_size = os.path.getsize(output_path)
        print(f"File size: {file_size:,} bytes")
        
        # Show first and last few lines
        with open(output_path, 'r') as f:
            lines = f.readlines()
        
        print()
        print("First 3 lines:")
        for line in lines[:3]:
            print(f"  {line.rstrip()}")
        
        print()
        print("Last 3 lines:")
        for line in lines[-3:]:
            print(f"  {line.rstrip()}")
    else:
        print("ERROR: File was not created!")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
