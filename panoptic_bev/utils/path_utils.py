"""
Cross-platform path utilities for PanopticBEV

Provides consistent path handling across Windows, Linux, and WSL environments.
Handles path resolution, directory creation, and filename sanitization.
"""
import os
import platform
from pathlib import Path
from typing import Union, List, Optional


def resolve_path(path_str: Union[str, Path, None]) -> Optional[Path]:
    """
    Resolve path for current platform.
    
    - Expands user home (~)
    - Converts to absolute path
    - Handles Windows/Unix differences
    
    Args:
        path_str: Input path as string or Path object
        
    Returns:
        Resolved Path object, or None if input was None
    """
    if path_str is None:
        return None
    
    path_obj = Path(path_str).expanduser()
    
    # Convert relative paths to absolute
    if not path_obj.is_absolute():
        path_obj = Path.cwd() / path_obj
    
    return path_obj.resolve()


def ensure_dir(path_str: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if necessary.
    
    Args:
        path_str: Directory path to ensure exists
        
    Returns:
        Path object for the directory
    """
    path_obj = resolve_path(path_str)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def get_cache_dir() -> Path:
    """
    Get platform-appropriate cache directory.
    
    Returns:
        Path to PanopticBEV cache directory
    """
    if platform.system() == 'Windows':
        cache_dir = Path(os.environ.get('LOCALAPPDATA', '~/.cache'))
    else:
        cache_dir = Path('~/.cache').expanduser()
    
    bev_cache = cache_dir / 'panoptic_bev'
    return ensure_dir(bev_cache)


def sanitize_filename(filename: str) -> str:
    """
    Remove characters illegal in Windows filenames.
    
    Args:
        filename: Input filename
        
    Returns:
        Sanitized filename safe for all platforms
    """
    illegal = '<>:"/\\|?*'
    for char in illegal:
        filename = filename.replace(char, '_')
    return filename


def get_temp_dir() -> Path:
    """
    Get platform-appropriate temporary directory.
    
    Returns:
        Path to temp directory for PanopticBEV
    """
    import tempfile
    
    if platform.system() == 'Windows':
        # Use Windows temp directory
        temp_base = Path(os.environ.get('TEMP', tempfile.gettempdir()))
    else:
        temp_base = Path(tempfile.gettempdir())
    
    bev_temp = temp_base / 'panoptic_bev'
    return ensure_dir(bev_temp)


def normalize_path(path: Union[str, Path]) -> str:
    """
    Normalize path for the current platform.
    
    Converts forward slashes to backslashes on Windows and vice versa on Unix.
    
    Args:
        path: Input path
        
    Returns:
        Normalized path string
    """
    return str(Path(path))


def join_paths(*args) -> Path:
    """
    Join path components in a cross-platform way.
    
    Args:
        *args: Path components to join
        
    Returns:
        Joined Path object
    """
    result = Path(args[0])
    for part in args[1:]:
        result = result / part
    return result


def get_relative_path(path: Union[str, Path], base: Union[str, Path]) -> Path:
    """
    Get relative path from base directory.
    
    Args:
        path: Target path
        base: Base directory
        
    Returns:
        Relative path from base to target
    """
    path_obj = resolve_path(path)
    base_obj = resolve_path(base)
    return path_obj.relative_to(base_obj)


def split_path(path: Union[str, Path]) -> tuple:
    """
    Split path into directory and filename.
    
    Cross-platform replacement for os.path.split()
    
    Args:
        path: Input path
        
    Returns:
        Tuple of (parent_path, filename)
    """
    path_obj = Path(path)
    return (path_obj.parent, path_obj.name)


def split_ext(path: Union[str, Path]) -> tuple:
    """
    Split path into stem and extension.
    
    Cross-platform replacement for os.path.splitext()
    
    Args:
        path: Input path
        
    Returns:
        Tuple of (stem, suffix)
    """
    path_obj = Path(path)
    return (path_obj.stem, path_obj.suffix)


def is_valid_path(path: Union[str, Path]) -> bool:
    """
    Check if path is valid for the current platform.
    
    Args:
        path: Path to validate
        
    Returns:
        True if path is valid, False otherwise
    """
    try:
        path_obj = Path(path)
        # Check for invalid characters on Windows
        if platform.system() == 'Windows':
            invalid_chars = '<>"|?*'
            for char in invalid_chars:
                if char in str(path_obj):
                    return False
        return True
    except Exception:
        return False


def find_file(
    filename: str,
    search_paths: List[Union[str, Path]],
    recursive: bool = False
) -> Optional[Path]:
    """
    Find a file in a list of search paths.
    
    Args:
        filename: Filename to search for
        search_paths: List of directories to search
        recursive: Whether to search recursively
        
    Returns:
        Path to found file, or None if not found
    """
    for search_path in search_paths:
        search_path = resolve_path(search_path)
        if not search_path.exists():
            continue
            
        if recursive:
            for root, dirs, files in os.walk(search_path):
                if filename in files:
                    return Path(root) / filename
        else:
            candidate = search_path / filename
            if candidate.exists():
                return candidate
    
    return None


def get_home_dir() -> Path:
    """
    Get user's home directory in a cross-platform way.
    
    Returns:
        Path to user's home directory
    """
    return Path.home()


def get_project_root() -> Path:
    """
    Get the PanopticBEV project root directory.
    
    Returns:
        Path to project root
    """
    # This file is in panoptic_bev/utils/
    return Path(__file__).parent.parent.parent.resolve()


def convert_to_forward_slashes(path: Union[str, Path]) -> str:
    """
    Convert path to use forward slashes (for config files, etc.).
    
    Args:
        path: Input path
        
    Returns:
        Path string with forward slashes
    """
    return str(Path(path)).replace('\\', '/')


def safe_path_join(base: Union[str, Path], *parts) -> Path:
    """
    Safely join paths, preventing directory traversal attacks.
    
    Args:
        base: Base directory
        *parts: Path components to join
        
    Returns:
        Joined Path object
        
    Raises:
        ValueError: If resulting path would be outside base directory
    """
    base_path = resolve_path(base)
    result_path = base_path
    
    for part in parts:
        result_path = result_path / part
    
    # Resolve and check it's still under base
    result_path = result_path.resolve()
    
    try:
        result_path.relative_to(base_path)
    except ValueError:
        raise ValueError(f"Path traversal detected: {result_path} is outside {base_path}")
    
    return result_path


class PathManager:
    """
    Centralized path management for PanopticBEV.
    
    Provides a single point for managing all project paths,
    with automatic platform detection and normalization.
    """
    
    def __init__(self, project_root: Optional[Union[str, Path]] = None):
        """
        Initialize PathManager.
        
        Args:
            project_root: Optional custom project root. If None, auto-detected.
        """
        self._project_root = resolve_path(project_root) if project_root else get_project_root()
        self._cache_dir = None
        self._temp_dir = None
    
    @property
    def project_root(self) -> Path:
        """Get project root directory."""
        return self._project_root
    
    @property
    def cache_dir(self) -> Path:
        """Get cache directory."""
        if self._cache_dir is None:
            self._cache_dir = get_cache_dir()
        return self._cache_dir
    
    @property
    def temp_dir(self) -> Path:
        """Get temporary directory."""
        if self._temp_dir is None:
            self._temp_dir = get_temp_dir()
        return self._temp_dir
    
    @property
    def data_dir(self) -> Path:
        """Get data directory."""
        return self._project_root / 'data'
    
    @property
    def config_dir(self) -> Path:
        """Get config directory."""
        return self._project_root / 'experiments' / 'config'
    
    @property
    def output_dir(self) -> Path:
        """Get output directory."""
        return self._project_root / 'outputs'
    
    @property
    def log_dir(self) -> Path:
        """Get log directory."""
        return self._project_root / 'logs'
    
    def resolve_dataset_path(self, dataset_name: str, base_dir: Optional[Union[str, Path]] = None) -> Path:
        """
        Resolve path to a dataset.
        
        Args:
            dataset_name: Name of the dataset
            base_dir: Optional base directory override
            
        Returns:
            Path to dataset directory
        """
        if base_dir:
            return resolve_path(base_dir) / dataset_name
        return self.data_dir / dataset_name
    
    def ensure_output_dir(self, experiment_name: str) -> Path:
        """
        Ensure output directory exists for an experiment.
        
        Args:
            experiment_name: Name of the experiment
            
        Returns:
            Path to experiment output directory
        """
        output_path = self.output_dir / sanitize_filename(experiment_name)
        return ensure_dir(output_path)
    
    def get_checkpoint_path(self, experiment_name: str, checkpoint_name: str = 'latest.pth') -> Path:
        """
        Get path to a checkpoint file.
        
        Args:
            experiment_name: Name of the experiment
            checkpoint_name: Name of the checkpoint file
            
        Returns:
            Path to checkpoint file
        """
        return self.ensure_output_dir(experiment_name) / 'checkpoints' / checkpoint_name


# Global path manager instance
_path_manager = None


def get_path_manager() -> PathManager:
    """
    Get the global PathManager instance.
    
    Returns:
        PathManager singleton instance
    """
    global _path_manager
    if _path_manager is None:
        _path_manager = PathManager()
    return _path_manager
