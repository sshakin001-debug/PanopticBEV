#!/usr/bin/env python3
"""
Smart Dataset Pathlib Converter for PanopticBEV

This automated refactoring tool converts os.path operations to pathlib in Python files.
It provides a safer, more modern approach to path handling that works consistently
across Windows and Unix systems.

Usage:
    python tools/refactor_to_pathlib.py [file_or_directory]
    
    If no argument is provided, it will scan the entire panoptic_bev package.
    
Examples:
    # Convert entire package
    python tools/refactor_to_pathlib.py
    
    # Convert specific file
    python tools/refactor_to_pathlib.py panoptic_bev/data/dataset.py
    
    # Convert specific directory
    python tools/refactor_to_pathlib.py panoptic_bev/data/

WARNING: Always backup your files before running this tool!
Use --dry-run to preview changes without applying them.
"""
import re
import sys
import argparse
from pathlib import Path


def add_pathlib_import(content):
    """Add pathlib import if missing."""
    if 'from pathlib import Path' in content:
        return content
    if 'import pathlib' in content:
        return content
    
    # Find a good place to add the import
    lines = content.split('\n')
    import_idx = 0
    
    for i, line in enumerate(lines):
        if line.startswith('import ') or line.startswith('from '):
            import_idx = i + 1
    
    lines.insert(import_idx, 'from pathlib import Path')
    return '\n'.join(lines)


def convert_os_path_join(content):
    """Convert os.path.join() to Path() / syntax."""
    
    def replace_join(match):
        args = [arg.strip() for arg in match.group(1).split(',')]
        if len(args) == 0:
            return match.group(0)
        
        # Convert first argument to Path()
        result = f'Path({args[0]})'
        
        # Append remaining arguments with /
        for arg in args[1:]:
            # Handle string literals that need quotes
            if arg.startswith('"') or arg.startswith("'"):
                result += f' / {arg}'
            else:
                result += f' / {arg}'
        
        return result
    
    # Match os.path.join(a, b, c) patterns
    pattern = r'os\.path\.join\(([^)]+)\)'
    content = re.sub(pattern, replace_join, content)
    
    return content


def convert_os_path_split(content):
    """
    Convert os.path.split() to Path attributes.
    
    Pattern: head, tail = os.path.split(path)
    Becomes: head = Path(path).parent; tail = Path(path).name
    """
    # Pattern: head, tail = os.path.split(path)
    pattern = r'(\w+),\s*(\w+)\s*=\s*os\.path\.split\(([^)]+)\)'
    
    def replace_split(match):
        head_var, tail_var, path = match.groups()
        return f'{head_var} = Path({path}).parent\n{tail_var} = Path({path}).name'
    
    content = re.sub(pattern, replace_split, content)
    
    # Also handle os.path.split() used in other contexts
    # Return tuple: (parent, name)
    pattern2 = r'os\.path\.split\(([^)]+)\)'
    
    def replace_split2(match):
        path = match.group(1)
        return f'(Path({path}).parent, Path({path}).name)'
    
    content = re.sub(pattern2, replace_split2, content)
    
    return content


def convert_os_path_splitext(content):
    """
    Convert os.path.splitext() to Path attributes.
    
    Pattern: stem, ext = os.path.splitext(path)
    Becomes: stem = Path(path).stem; ext = Path(path).suffix
    """
    # Pattern: stem, ext = os.path.splitext(path)
    pattern = r'(\w+),\s*(\w+)\s*=\s*os\.path\.splitext\(([^)]+)\)'
    
    def replace_splitext(match):
        stem_var, ext_var, path = match.groups()
        return f'{stem_var} = Path({path}).stem\n{ext_var} = Path({path}).suffix'
    
    content = re.sub(pattern, replace_splitext, content)
    
    # Also handle os.path.splitext() used in other contexts
    # Return tuple: (stem, suffix)
    pattern2 = r'os\.path\.splitext\(([^)]+)\)'
    
    def replace_splitext2(match):
        path = match.group(1)
        return f'(Path({path}).stem, Path({path}).suffix)'
    
    content = re.sub(pattern2, replace_splitext2, content)
    
    return content


def convert_string_concat_paths(content):
    """
    Convert path string concatenation to Path operations.
    
    Pattern: "path/" + filename or 'path/' + filename
    Becomes: Path("path") / filename
    """
    # Pattern: "path/" + filename or 'path/' + filename
    pattern = r'(["\'][^"\']*[\\/]["\'])\s*\+\s*(\w+)'
    
    def replace_concat(match):
        prefix = match.group(1).strip('"\'')
        var = match.group(2)
        # Remove trailing slash from prefix
        prefix = prefix.rstrip('/\\')
        return f'Path("{prefix}") / {var}'
    
    content = re.sub(pattern, replace_concat, content)
    
    # Also handle: var + "/" + filename
    pattern2 = r'(\w+)\s*\+\s*["\'][\\/]["\']\s*\+\s*(\w+)'
    
    def replace_concat2(match):
        var1 = match.group(1)
        var2 = match.group(2)
        return f'Path({var1}) / {var2}'
    
    content = re.sub(pattern2, replace_concat2, content)
    
    return content


def convert_fstring_paths(content):
    """
    Convert f-string paths to Path operations.
    
    Pattern: f"{var1}/{var2}"
    Becomes: Path(var1) / var2
    """
    # Simple case: f"{var1}/{var2}"
    pattern = r'f["\']\{([^}]+)\}/([^"\']+)["\']'
    
    def replace_fstring(match):
        var1 = match.group(1)
        rest = match.group(2)
        
        # Split the rest by /
        parts = rest.split('/')
        
        result = f'Path({var1})'
        for part in parts:
            if part.startswith('{') and part.endswith('}'):
                # Variable
                result += f' / {part[1:-1]}'
            elif part:
                # String literal
                result += f' / "{part}"'
        
        return result
    
    content = re.sub(pattern, replace_fstring, content)
    
    return content


def flag_hardcoded_paths(content, filepath):
    """
    Flag hardcoded Unix paths that need Windows attention.
    
    Returns a list of warnings for hardcoded paths.
    """
    warnings = []
    
    # Find hardcoded Unix paths
    unix_paths = re.findall(r'["\']/home/[^"\']+["\']', content)
    if unix_paths:
        warnings.append(f"WARNING: {filepath} contains hardcoded Unix paths:")
        for p in unix_paths:
            warnings.append(f"  {p}")
        warnings.append("  These should be converted to relative paths or config variables")
    
    # Find hardcoded /tmp paths
    tmp_paths = re.findall(r'["\']/tmp/[^"\']*["\']', content)
    if tmp_paths:
        warnings.append(f"WARNING: {filepath} contains hardcoded /tmp paths:")
        for p in tmp_paths:
            warnings.append(f"  {p}")
        warnings.append("  These should use a cross-platform temp directory")
    
    return warnings


def convert_os_path_functions(content):
    """Convert various os.path functions to Path methods."""
    
    conversions = [
        # os.path.exists(p) -> Path(p).exists()
        (r'os\.path\.exists\(([^)]+)\)', r'Path(\1).exists()'),
        
        # os.path.isfile(p) -> Path(p).is_file()
        (r'os\.path\.isfile\(([^)]+)\)', r'Path(\1).is_file()'),
        
        # os.path.isdir(p) -> Path(p).is_dir()
        (r'os\.path\.isdir\(([^)]+)\)', r'Path(\1).is_dir()'),
        
        # os.path.basename(p) -> Path(p).name
        (r'os\.path\.basename\(([^)]+)\)', r'Path(\1).name'),
        
        # os.path.dirname(p) -> Path(p).parent
        (r'os\.path\.dirname\(([^)]+)\)', r'Path(\1).parent'),
        
        # os.path.abspath(p) -> Path(p).resolve()
        (r'os\.path\.abspath\(([^)]+)\)', r'Path(\1).resolve()'),
        
        # os.path.getsize(p) -> Path(p).stat().st_size
        (r'os\.path\.getsize\(([^)]+)\)', r'Path(\1).stat().st_size'),
        
        # os.path.expanduser(p) -> Path(p).expanduser()
        (r'os\.path\.expanduser\(([^)]+)\)', r'Path(\1).expanduser()'),
        
        # os.path.realpath(p) -> Path(p).resolve()
        (r'os\.path\.realpath\(([^)]+)\)', r'Path(\1).resolve()'),
        
        # os.path.isabs(p) -> Path(p).is_absolute()
        (r'os\.path\.isabs\(([^)]+)\)', r'Path(\1).is_absolute()'),
        
        # os.path.getmtime(p) -> Path(p).stat().st_mtime
        (r'os\.path\.getmtime\(([^)]+)\)', r'Path(\1).stat().st_mtime'),
        
        # os.path.getctime(p) -> Path(p).stat().st_ctime
        (r'os\.path\.getctime\(([^)]+)\)', r'Path(\1).stat().st_ctime'),
        
        # os.makedirs(p) -> Path(p).mkdir(parents=True, exist_ok=True)
        (r'os\.makedirs\(([^)]+)\)', r'Path(\1).mkdir(parents=True, exist_ok=True)'),
        
        # os.mkdir(p) -> Path(p).mkdir()
        (r'os\.mkdir\(([^)]+)\)', r'Path(\1).mkdir()'),
    ]
    
    for pattern, replacement in conversions:
        content = re.sub(pattern, replacement, content)
    
    return content


def convert_open_with_path_join(content):
    """Convert open(os.path.join(...)) patterns."""
    
    def replace_open_path(match):
        # Extract the os.path.join part and convert it
        join_content = match.group(1)
        # Remove 'os.path.join(' and trailing ')'
        if join_content.startswith('os.path.join('):
            join_content = join_content[13:-1]  # Remove 'os.path.join(' and trailing ')'
        
        args = [arg.strip() for arg in join_content.split(',')]
        if len(args) == 0:
            return match.group(0)
        
        result = f'Path({args[0]})'
        for arg in args[1:]:
            result += f' / {arg}'
        
        return result
    
    # This is a complex pattern - handle with care
    # Match open(os.path.join(...), 'r') patterns
    pattern = r'os\.path\.join\(([^)]+)\)'
    
    # Find all open calls with os.path.join inside
    open_pattern = r'open\((os\.path\.join\([^)]+\))\s*,?'
    
    for match in re.finditer(open_pattern, content):
        old_join = match.group(1)
        # Convert just the join part
        args = old_join[13:-1]  # Remove 'os.path.join(' and ')'
        parts = [p.strip() for p in args.split(',')]
        new_path = f'Path({parts[0]})' + ''.join(f' / {p}' for p in parts[1:])
        content = content.replace(old_join, str(new_path))
    
    return content


def convert_file_to_pathlib(filepath, dry_run=False):
    """
    Convert os.path operations to pathlib in a Python file.
    
    Args:
        filepath: Path to the Python file
        dry_run: If True, only print changes without applying them
    
    Returns:
        tuple: (bool: True if changes were made, list: warnings)
    """
    filepath = Path(filepath)
    warnings = []
    
    if not filepath.suffix == '.py':
        return False, warnings
    
    original_content = filepath.read_text()
    content = original_content
    
    # Check if file uses os.path
    if 'os.path' not in content and 'os.makedirs' not in content and 'os.mkdir' not in content:
        return False, warnings
    
    # Flag hardcoded paths before conversion
    warnings = flag_hardcoded_paths(content, filepath)
    
    # Apply conversions
    content = add_pathlib_import(content)
    content = convert_os_path_join(content)
    content = convert_os_path_split(content)
    content = convert_os_path_splitext(content)
    content = convert_string_concat_paths(content)
    content = convert_fstring_paths(content)
    content = convert_os_path_functions(content)
    content = convert_open_with_path_join(content)
    
    # Check if anything changed
    if content == original_content:
        return False, warnings
    
    if dry_run:
        print(f"Would modify: {filepath}")
        # Show a diff or summary
        original_lines = original_content.split('\n')
        new_lines = content.split('\n')
        for i, (old, new) in enumerate(zip(original_lines, new_lines)):
            if old != new:
                print(f"  Line {i+1}: {old[:60]}... -> {new[:60]}...")
        return True, warnings
    
    # Write the changes
    filepath.write_text(content)
    print(f"Refactored: {filepath}")
    return True, warnings


def find_python_files(path):
    """Find all Python files in a directory or return single file."""
    path = Path(path)
    
    if path.is_file():
        if path.suffix == '.py':
            return [path]
        return []
    
    if path.is_dir():
        return list(path.rglob('*.py'))
    
    return []


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Convert os.path operations to pathlib'
    )
    parser.add_argument(
        'path',
        nargs='?',
        default='panoptic_bev',
        help='File or directory to refactor (default: panoptic_bev/)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show changes without applying them'
    )
    parser.add_argument(
        '--exclude',
        action='append',
        default=[],
        help='Patterns to exclude (can be used multiple times)'
    )
    parser.add_argument(
        '--check-hardcoded',
        action='store_true',
        help='Check for hardcoded Unix paths that need attention'
    )
    
    args = parser.parse_args()
    
    # Find repository root
    repo_root = Path(__file__).parent.parent
    target_path = repo_root / args.path
    
    if not target_path.exists():
        print(f"Error: Path not found: {target_path}")
        sys.exit(1)
    
    # Find Python files
    py_files = find_python_files(target_path)
    
    # Apply exclusions
    for exclude_pattern in args.exclude:
        py_files = [f for f in py_files if exclude_pattern not in str(f)]
    
    print(f"Found {len(py_files)} Python file(s) to process")
    
    if args.dry_run:
        print("\nDRY RUN - No changes will be applied\n")
    
    # Process files
    modified_count = 0
    all_warnings = []
    
    for py_file in py_files:
        modified, warnings = convert_file_to_pathlib(py_file, dry_run=args.dry_run)
        if modified:
            modified_count += 1
        if warnings:
            all_warnings.extend(warnings)
    
    print(f"\n{'='*60}")
    if args.dry_run:
        print(f"Dry run complete: {modified_count} file(s) would be modified")
        print("Run without --dry-run to apply changes")
    else:
        print(f"Refactoring complete: {modified_count} file(s) modified")
    print(f"{'='*60}")
    
    # Print warnings
    if all_warnings:
        print("\n" + "="*60)
        print("WARNINGS - Manual review required:")
        print("="*60)
        for warning in all_warnings:
            print(warning)
    
    if modified_count > 0:
        print("\nNOTE: Please review the changes carefully!")
        print("Some complex patterns may need manual adjustment.")
        print("Test your code after refactoring.")
    
    # Print usage tips
    print("\n" + "-"*60)
    print("TIPS:")
    print("-"*60)
    print("• Use --check-hardcoded to find Unix-specific paths")
    print("• Use --exclude to skip files (e.g., --exclude test_)")
    print("• Always test after refactoring!")
    print("• Complex patterns like os.path.split() are converted")
    print("  but may need manual review.")


if __name__ == "__main__":
    main()
