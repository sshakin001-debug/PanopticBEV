"""
Configuration Path Resolver for PanopticBEV

This module provides platform-aware configuration loading that handles
path resolution dynamically for both Windows and Unix systems.

The project uses INI configuration files rather than YAML.
"""
import configparser
from pathlib import Path
import platform
import os


class WindowsPathResolver:
    """
    Configuration preprocessor that handles path resolution dynamically.
    
    This resolver handles both Windows and Unix paths in INI configuration files,
    converting them to the appropriate format for the current platform.
    
    Args:
        config_path: Path to the INI configuration file
    
    Example:
        >>> resolver = WindowsPathResolver("experiments/config/kitti.ini")
        >>> config = resolver.get_config()
        >>> print(config['dataset']['root'])
    """
    
    def __init__(self, config_path):
        self.config_path = Path(config_path)
        self.config = self._load_and_resolve()
    
    def _load_and_resolve(self):
        """Load INI config and resolve paths."""
        parser = configparser.ConfigParser()
        parser.read(self.config_path)
        
        # Convert to dict for easier manipulation
        config = {section: dict(parser.items(section)) 
                  for section in parser.sections()}
        
        # Resolve dataset paths
        if 'dataset' in config:
            config['dataset'] = self._resolve_dataset_paths(config['dataset'])
        
        # Resolve output paths
        if 'output' in config:
            config['output'] = self._resolve_output_paths(config['output'])
        
        # Resolve logging paths
        if 'logging' in config:
            config['logging'] = self._resolve_logging_paths(config['logging'])
        
        return config
    
    def _resolve_dataset_paths(self, dataset_cfg):
        """Handle both Windows and Unix paths in config."""
        path_keys = ['root', 'seam_root', 'label_root', 'image_root', 
                     'calib_root', 'lidar_root']
        
        for key in path_keys:
            if key in dataset_cfg:
                path = Path(dataset_cfg[key])
                # Handle Windows drive letters and UNC paths
                if platform.system() == 'Windows':
                    if not path.is_absolute():
                        # Relative paths are resolved relative to repo root
                        repo_root = self.config_path.parent.parent.parent
                        path = repo_root / path
                dataset_cfg[key] = str(path.resolve())
        
        return dataset_cfg
    
    def _resolve_output_paths(self, output_cfg):
        """Resolve output directory paths."""
        if 'save_dir' in output_cfg:
            path = Path(output_cfg['save_dir'])
            if platform.system() == 'Windows':
                if not path.is_absolute():
                    repo_root = self.config_path.parent.parent.parent
                    path = repo_root / path
            output_cfg['save_dir'] = str(path.resolve())
        
        return output_cfg
    
    def _resolve_logging_paths(self, logging_cfg):
        """Resolve logging directory paths."""
        if 'log_dir' in logging_cfg:
            path = Path(logging_cfg['log_dir'])
            if platform.system() == 'Windows':
                if not path.is_absolute():
                    repo_root = self.config_path.parent.parent.parent
                    path = repo_root / path
            logging_cfg['log_dir'] = str(path.resolve())
        
        return logging_cfg
    
    def get_config(self):
        """Return the resolved configuration dictionary."""
        return self.config
    
    def get(self, section, key, default=None):
        """Get a configuration value with optional default."""
        return self.config.get(section, {}).get(key, default)
    
    def __getitem__(self, section):
        """Allow dict-style access to sections."""
        return self.config[section]


def resolve_path_env_variables(path_str):
    """
    Resolve environment variables and user home in path strings.
    
    Handles both Windows (%VAR%) and Unix ($VAR) style variables.
    
    Args:
        path_str: Path string that may contain environment variables
    
    Returns:
        Resolved Path object
    """
    # Expand environment variables
    expanded = os.path.expandvars(path_str)
    # Expand user home (~)
    expanded = os.path.expanduser(expanded)
    return Path(expanded)


def ensure_dir(path):
    """
    Ensure a directory exists, creating it if necessary.
    
    Cross-platform directory creation that works on both Windows and Unix.
    
    Args:
        path: Directory path (string or Path)
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


# For backward compatibility, also provide a simple load_config function
def load_config(config_path):
    """
    Load and resolve a configuration file.
    
    Args:
        config_path: Path to the INI configuration file
    
    Returns:
        Resolved configuration dictionary
    """
    resolver = WindowsPathResolver(config_path)
    return resolver.get_config()
