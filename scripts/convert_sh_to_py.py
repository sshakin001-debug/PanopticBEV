#!/usr/bin/env python3
"""
Smart Shell-to-Python Converter Script for PanopticBEV

This script parses existing shell scripts and generates Windows-compatible
Python runner scripts automatically.

Usage:
    python scripts/convert_sh_to_py.py

This will convert all .sh files in the scripts directory to .py equivalents.
"""
import re
import sys
from pathlib import Path


def parse_shell_script(sh_path):
    """Parse bash script and extract Python command with env vars."""
    content = sh_path.read_text()
    
    # Extract export statements
    env_vars = {}
    for match in re.finditer(r'export\s+(\w+)=(.+)', content):
        value = match.group(2).strip('"\'')
        # Remove $PWD and other bash variables that won't work on Windows
        value = re.sub(r'\$\{?PWD\}?', str(Path.cwd()), value)
        env_vars[match.group(1)] = value
    
    # Extract python command
    python_match = re.search(r'python[3]?\s+(.+)', content)
    if not python_match:
        return None, env_vars
    
    args = python_match.group(1).split()
    return args, env_vars


def create_windows_runner(sh_name, args, env_vars):
    """Generate Windows-compatible Python runner."""
    py_content = f'''#!/usr/bin/env python3
"""
Auto-converted from {sh_name}
Windows-compatible runner script for PanopticBEV
"""
import subprocess
import sys
import os
from pathlib import Path

# Original environment variables from shell script
extra_env = {env_vars}

# Merge with current environment
env = {{**os.environ, **extra_env}}

# Ensure PYTHONPATH includes repo root
repo_root = Path(__file__).parent.parent.absolute()
env["PYTHONPATH"] = str(repo_root) + os.pathsep + env.get("PYTHONPATH", "")

# Windows-specific: Convert paths if needed
path_keys = ["dataset_root_dir", "seam_root_dir", "output_dir", "config_file", "pretrained_model"]
for key in path_keys:
    if key in env:
        env[key] = str(Path(env[key]))

# Build the command
cmd = [sys.executable] + {args}
print(f"Running: {{' '.join(cmd)}}")
print(f"Working directory: {{repo_root}}")

# Run the command
try:
    subprocess.run(cmd, env=env, check=True, cwd=repo_root)
except subprocess.CalledProcessError as e:
    print(f"Command failed with exit code {{e.returncode}}")
    sys.exit(e.returncode)
'''
    return py_content


def main():
    """Main entry point."""
    scripts_dir = Path(__file__).parent
    converted_count = 0
    
    for sh_file in scripts_dir.glob("*.sh"):
        args, env_vars = parse_shell_script(sh_file)
        if args:
            py_code = create_windows_runner(sh_file.name, args, env_vars)
            py_file = sh_file.with_suffix(".py")
            
            # Write the new Python file
            py_file.write_text(py_code)
            print(f"Converted {sh_file.name} -> {py_file.name}")
            converted_count += 1
        else:
            print(f"Warning: Could not parse Python command from {sh_file.name}")
    
    print(f"\nConversion complete: {converted_count} file(s) converted")
    print("You can now run the .py scripts directly on Windows")


if __name__ == "__main__":
    main()
