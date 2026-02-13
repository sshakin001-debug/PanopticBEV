"""
Cross-Platform Symbolic Link Utilities for PanopticBEV

This module provides cross-platform symlink creation that works on both
Windows and Unix systems. Windows requires special permissions for symlinks,
so this module provides fallbacks using junction points for directories.

Windows behavior:
- Windows 10/11: Requires Developer Mode or Administrator privileges for symlinks
- Fallback: Uses directory junctions for directories (no special permissions needed)
- Files: Requires symlinks or falls back to copying

Unix behavior:
- Uses standard os.symlink
"""
import os
import platform
import shutil
from pathlib import Path


def create_symlink(src, dst, target_is_directory=None, force=False):
    """
    Create a cross-platform symbolic link.
    
    On Windows, attempts to create a true symlink first, then falls back to
    a directory junction if that fails. On Unix, uses standard symlinks.
    
    Args:
        src: Source path (the target of the link)
        dst: Destination path (where the link will be created)
        target_is_directory: Whether the target is a directory. If None, auto-detect.
        force: If True, remove existing file/directory at dst before creating link
    
    Returns:
        bool: True if successful, False otherwise
    
    Example:
        >>> from panoptic_bev.utils.windows_symlink import create_symlink
        >>> create_symlink("/data/dataset", "/project/dataset_link")
    """
    src = Path(src)
    dst = Path(dst)
    
    # Auto-detect if target is directory
    if target_is_directory is None:
        target_is_directory = src.is_dir()
    
    # Remove existing if force=True
    if force and dst.exists():
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    
    # Check if already exists
    if dst.exists():
        print(f"Warning: {dst} already exists, skipping")
        return False
    
    # Ensure parent directory exists
    dst.parent.mkdir(parents=True, exist_ok=True)
    
    if platform.system() == 'Windows':
        return _create_windows_link(src, dst, target_is_directory)
    else:
        # Unix - use standard symlink
        try:
            os.symlink(src, dst, target_is_directory=target_is_directory)
            return True
        except OSError as e:
            print(f"Error creating symlink: {e}")
            return False


def _create_windows_link(src, dst, target_is_directory):
    """
    Create a link on Windows, trying multiple methods.
    
    Priority:
    1. True symlink (requires Developer Mode or Admin)
    2. Directory junction (directories only, no special permissions)
    3. File hardlink (files only, same volume)
    4. Copy as fallback
    """
    src = Path(src).resolve()
    dst = Path(dst)
    
    # Method 1: Try true symlink first
    try:
        # Use os.symlink with dir_symlink for Windows
        if target_is_directory:
            import _winapi
            _winapi.CreateJunction(str(src), str(dst))
            return True
        else:
            os.symlink(src, dst, target_is_directory=False)
            return True
    except (OSError, ImportError):
        pass
    
    # Method 2: Try junction for directories
    if target_is_directory:
        try:
            # Use mklink /J command for junction
            import subprocess
            result = subprocess.run(
                ['cmd', '/c', 'mklink', '/J', str(dst), str(src)],
                capture_output=True,
                shell=False
            )
            if result.returncode == 0:
                return True
        except Exception:
            pass
    
    # Method 3: Try hardlink for files
    if not target_is_directory:
        try:
            import _winapi
            _winapi.CreateHardLink(str(dst), str(src))
            return True
        except (OSError, ImportError, AttributeError):
            pass
    
    # Method 4: Copy as last resort
    try:
        if target_is_directory:
            shutil.copytree(src, dst, symlinks=True)
        else:
            shutil.copy2(src, dst)
        print(f"Warning: Copied instead of linking (insufficient permissions)")
        return True
    except Exception as e:
        print(f"Error copying: {e}")
        return False


def remove_link(path):
    """
    Safely remove a symlink/junction/copy without affecting the target.
    
    Args:
        path: Path to the link to remove
    
    Returns:
        bool: True if successful
    """
    path = Path(path)
    
    if not path.exists():
        return True
    
    try:
        if path.is_symlink() or path.is_junction():
            if path.is_dir():
                path.rmdir()  # Remove junction/symlink to dir
            else:
                path.unlink()  # Remove symlink to file
        elif path.is_dir():
            shutil.rmtree(path)  # Remove copied directory
        else:
            path.unlink()  # Remove copied file
        return True
    except Exception as e:
        print(f"Error removing link: {e}")
        return False


def is_junction(path):
    """
    Check if a path is a Windows junction point.
    
    Args:
        path: Path to check
    
    Returns:
        bool: True if path is a junction
    """
    if platform.system() != 'Windows':
        return False
    
    path = Path(path)
    if not path.exists():
        return False
    
    try:
        # Check if it's a reparse point (junction)
        import stat
        st = path.lstat()
        return st.st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT != 0
    except (AttributeError, OSError):
        return False


def create_dataset_links(dataset_root, link_root, dataset_names=None):
    """
    Create symbolic links for dataset directories.
    
    This is useful for organizing datasets without duplicating data.
    
    Args:
        dataset_root: Root directory containing datasets
        link_root: Directory where links will be created
        dataset_names: List of dataset names to link, or None for all
    
    Returns:
        dict: Mapping of dataset names to link paths
    """
    dataset_root = Path(dataset_root)
    link_root = Path(link_root)
    link_root.mkdir(parents=True, exist_ok=True)
    
    if dataset_names is None:
        # Auto-detect datasets
        dataset_names = [d.name for d in dataset_root.iterdir() if d.is_dir()]
    
    links = {}
    for name in dataset_names:
        src = dataset_root / name
        dst = link_root / name
        
        if create_symlink(src, dst, target_is_directory=True):
            links[name] = dst
        else:
            print(f"Failed to create link for {name}")
    
    return links


def setup_windows_developer_mode():
    """
    Check and report Windows Developer Mode status.
    
    Developer Mode allows creating true symlinks without Administrator privileges.
    
    Returns:
        bool: True if Developer Mode is likely enabled
    """
    if platform.system() != 'Windows':
        return True
    
    try:
        import winreg
        # Check registry for Developer Mode
        key = winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"SOFTWARE\Microsoft\Windows\CurrentVersion\AppModelUnlock"
        )
        value, _ = winreg.QueryValueEx(key, "AllowDevelopmentWithoutDevLicense")
        winreg.CloseKey(key)
        
        if value == 1:
            print("✓ Windows Developer Mode is enabled")
            return True
        else:
            print("⚠ Windows Developer Mode is NOT enabled")
            print("  Symlinks may require Administrator privileges")
            print("  Junction points will be used as fallback for directories")
            return False
            
    except (ImportError, OSError, WindowsError):
        print("Could not check Developer Mode status")
        return False


# Patch Path class to support is_junction method
if platform.system() == 'Windows':
    Path.is_junction = is_junction
else:
    Path.is_junction = lambda self: False
