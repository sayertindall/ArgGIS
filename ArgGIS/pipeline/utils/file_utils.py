import os
import shutil
import logging
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Set, Callable
from datetime import datetime, timezone

# Set up logging
logger = logging.getLogger(__name__)

def list_files(directory: Union[str, Path], pattern: str = "*") -> List[Path]:
    """
    List all files in a directory matching a pattern.
    
    Args:
        directory: Directory to search
        pattern: Glob pattern to match (default: "*")
        
    Returns:
        List[Path]: List of file paths
    """
    directory = Path(directory)
    files = list(directory.glob(pattern))
    return sorted(f for f in files if f.is_file())

def list_directories(directory: Union[str, Path]) -> List[Path]:
    """
    List all subdirectories in a directory.
    
    Args:
        directory: Directory to search
        
    Returns:
        List[Path]: List of directory paths
    """
    directory = Path(directory)
    return sorted(d for d in directory.iterdir() if d.is_dir())

def ensure_directory(directory: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        directory: Directory path
        
    Returns:
        Path: Path to the directory
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    return directory

def clean_directory(directory: Union[str, Path], pattern: str = "*") -> int:
    """
    Remove all files in a directory matching a pattern.
    
    Args:
        directory: Directory to clean
        pattern: Glob pattern to match (default: "*")
        
    Returns:
        int: Number of files removed
    """
    directory = Path(directory)
    count = 0
    for file_path in directory.glob(pattern):
        if file_path.is_file():
            file_path.unlink()
            count += 1
    logger.info(f"Removed {count} files from {directory}")
    return count

def backup_directory(source: Union[str, Path], 
                    destination: Union[str, Path], 
                    timestamp: bool = True) -> Path:
    """
    Backup a directory by copying its contents to a backup location.
    
    Args:
        source: Source directory
        destination: Destination directory
        timestamp: Whether to add a timestamp to the backup folder name
        
    Returns:
        Path: Path to the backup directory
    """
    source = Path(source)
    destination = Path(destination)
    
    if timestamp:
        timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        backup_dir = destination / f"{source.name}_backup_{timestamp_str}"
    else:
        backup_dir = destination / f"{source.name}_backup"
    
    # Create the backup directory
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy all files
    for item in source.glob("*"):
        if item.is_file():
            shutil.copy2(item, backup_dir)
        elif item.is_dir():
            shutil.copytree(item, backup_dir / item.name)
    
    logger.info(f"Backed up {source} to {backup_dir}")
    return backup_dir

def find_orphaned_files(directory: Union[str, Path], 
                       reference_directory: Union[str, Path],
                       file_extensions: Tuple[str, str] = (".csv", ".json")) -> Dict[str, List[Path]]:
    """
    Find files in a directory that don't have corresponding files in another directory.
    
    Args:
        directory: Directory to search for orphaned files
        reference_directory: Directory containing reference files
        file_extensions: Tuple of extensions to compare (e.g., (".csv", ".json"))
        
    Returns:
        Dict[str, List[Path]]: Dictionary of orphaned files by extension
    """
    directory = Path(directory)
    reference_directory = Path(reference_directory)
    
    # Get all base names (without extensions) from the reference directory
    reference_files = set()
    for ext in file_extensions:
        for file_path in reference_directory.glob(f"*{ext}"):
            reference_files.add(file_path.stem)
    
    # Find orphaned files in the directory
    orphaned = {ext: [] for ext in file_extensions}
    
    for ext in file_extensions:
        for file_path in directory.glob(f"*{ext}"):
            if file_path.stem not in reference_files:
                orphaned[ext].append(file_path)
    
    return orphaned

def find_mismatched_pairs(directory1: Union[str, Path], 
                         directory2: Union[str, Path],
                         ext1: str = ".csv", 
                         ext2: str = ".json") -> Tuple[Set[str], Set[str]]:
    """
    Find files that don't have matching pairs across two directories.
    
    Args:
        directory1: First directory
        directory2: Second directory
        ext1: File extension in first directory
        ext2: File extension in second directory
        
    Returns:
        Tuple[Set[str], Set[str]]: Sets of stems that exist in only one directory
    """
    directory1 = Path(directory1)
    directory2 = Path(directory2)
    
    # Get all base names from both directories
    stems1 = {f.stem for f in directory1.glob(f"*{ext1}")}
    stems2 = {f.stem for f in directory2.glob(f"*{ext2}")}
    
    # Find stems that exist in only one directory
    only_in_dir1 = stems1 - stems2
    only_in_dir2 = stems2 - stems1
    
    return only_in_dir1, only_in_dir2

def rename_files(directory: Union[str, Path], 
                rename_map: Dict[str, str],
                dry_run: bool = False) -> List[Tuple[Path, Path]]:
    """
    Rename files in a directory according to a mapping.
    
    Args:
        directory: Directory containing files to rename
        rename_map: Dictionary mapping old filenames to new filenames
        dry_run: If True, don't actually rename files, just log what would happen
        
    Returns:
        List[Tuple[Path, Path]]: List of (old_path, new_path) pairs that were renamed
    """
    directory = Path(directory)
    renamed = []
    
    for old_name, new_name in rename_map.items():
        old_path = directory / old_name
        new_path = directory / new_name
        
        if old_path.exists():
            if dry_run:
                logger.info(f"Would rename {old_path} to {new_path}")
            else:
                if new_path.exists():
                    logger.warning(f"Cannot rename {old_path} to {new_path}: destination already exists")
                    continue
                    
                old_path.rename(new_path)
                logger.info(f"Renamed {old_path} to {new_path}")
                renamed.append((old_path, new_path))
        else:
            logger.warning(f"Cannot rename {old_path}: file not found")
    
    return renamed

def rename_by_pattern(directory: Union[str, Path],
                     pattern: str,
                     replacement: str,
                     file_ext: Optional[str] = None,
                     dry_run: bool = False) -> List[Tuple[Path, Path]]:
    """
    Rename files in a directory by replacing a pattern in the filename.
    
    Args:
        directory: Directory containing files to rename
        pattern: Pattern to search for
        replacement: Replacement string
        file_ext: Optional file extension to filter by
        dry_run: If True, don't actually rename files, just log what would happen
        
    Returns:
        List[Tuple[Path, Path]]: List of (old_path, new_path) pairs that were renamed
    """
    directory = Path(directory)
    renamed = []
    
    # Get all files, optionally filtered by extension
    if file_ext:
        files = directory.glob(f"*{file_ext}")
    else:
        files = directory.glob("*")
    
    for file_path in files:
        if not file_path.is_file():
            continue
            
        # Create new filename
        old_name = file_path.name
        if pattern in old_name:
            new_name = old_name.replace(pattern, replacement)
            new_path = file_path.with_name(new_name)
            
            if dry_run:
                logger.info(f"Would rename {file_path} to {new_path}")
            else:
                if new_path.exists():
                    logger.warning(f"Cannot rename {file_path} to {new_path}: destination already exists")
                    continue
                    
                file_path.rename(new_path)
                logger.info(f"Renamed {file_path} to {new_path}")
                renamed.append((file_path, new_path))
    
    return renamed

def sync_file_names(source_dir: Union[str, Path],
                   target_dir: Union[str, Path],
                   source_ext: str,
                   target_ext: str,
                   force: bool = False) -> Dict[str, List[Tuple[Path, Path]]]:
    """
    Synchronize filenames between two directories, renaming files in the target directory
    to match the base names of files in the source directory.
    
    Args:
        source_dir: Source directory with reference filenames
        target_dir: Target directory with files to be renamed
        source_ext: File extension in source directory
        target_ext: File extension in target directory
        force: If True, overwrite existing files during renaming
        
    Returns:
        Dict[str, List[Tuple[Path, Path]]]: Results of the synchronization operation
    """
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    
    result = {
        "renamed": [],
        "skipped": [],
        "not_found": []
    }
    
    # Get all source files with their base names
    source_files = {f.stem: f for f in source_dir.glob(f"*{source_ext}")}
    
    # Get all target files grouped by potential matching criteria
    target_files = {}
    for target_file in target_dir.glob(f"*{target_ext}"):
        # Store by stem
        target_files[target_file.stem] = target_file
        
        # Also store by parts of the name to aid in fuzzy matching
        parts = target_file.stem.split('_')
        for i in range(len(parts) - 1):
            key = '_'.join(parts[:i+1])
            if key not in target_files:
                target_files[key] = target_file
    
    # For each source file, find and rename corresponding target file
    for source_stem, source_file in source_files.items():
        # Try exact match first
        if source_stem in target_files:
            target_file = target_files[source_stem]
            
            # Skip if names already match
            if target_file.stem == source_stem:
                result["skipped"].append((target_file, target_file))
                continue
            
            # Rename target file to match source base name
            new_target_path = target_file.with_name(f"{source_stem}{target_ext}")
            
            if new_target_path.exists() and not force:
                logger.warning(f"Cannot rename {target_file} to {new_target_path}: destination already exists")
                result["skipped"].append((target_file, new_target_path))
            else:
                if new_target_path.exists() and force:
                    new_target_path.unlink()
                    
                target_file.rename(new_target_path)
                logger.info(f"Renamed {target_file} to {new_target_path}")
                result["renamed"].append((target_file, new_target_path))
        else:
            # Try to find a partial match
            found = False
            for key, target_file in target_files.items():
                if source_stem.startswith(key) or key in source_stem:
                    new_target_path = target_file.with_name(f"{source_stem}{target_ext}")
                    
                    if new_target_path.exists() and not force:
                        logger.warning(f"Cannot rename {target_file} to {new_target_path}: destination already exists")
                        result["skipped"].append((target_file, new_target_path))
                    else:
                        if new_target_path.exists() and force:
                            new_target_path.unlink()
                            
                        target_file.rename(new_target_path)
                        logger.info(f"Renamed {target_file} to {new_target_path} (partial match)")
                        result["renamed"].append((target_file, new_target_path))
                    
                    found = True
                    break
            
            if not found:
                logger.warning(f"No matching target file found for {source_file}")
                result["not_found"].append(source_file)
    
    return result

def apply_to_all_files(directory: Union[str, Path],
                      function: Callable,
                      pattern: str = "*",
                      recursive: bool = False,
                      **kwargs) -> List[Tuple[Path, object]]:
    """
    Apply a function to all files in a directory.
    
    Args:
        directory: Directory containing files
        function: Function to apply to each file (takes file path as first argument)
        pattern: Glob pattern to match files
        recursive: Whether to search subdirectories
        **kwargs: Additional keyword arguments to pass to the function
        
    Returns:
        List[Tuple[Path, object]]: List of (file_path, function_result) pairs
    """
    directory = Path(directory)
    results = []
    
    if recursive:
        glob_pattern = f"**/{pattern}"
    else:
        glob_pattern = pattern
    
    for file_path in directory.glob(glob_pattern):
        if file_path.is_file():
            try:
                result = function(file_path, **kwargs)
                results.append((file_path, result))
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                results.append((file_path, e))
    
    return results

def delete_orphaned_metadata(data_dir: Union[str, Path],
                            metadata_dir: Union[str, Path],
                            data_ext: str = ".csv",
                            meta_ext: str = ".json",
                            dry_run: bool = True) -> List[Path]:
    """
    Delete metadata files that don't have corresponding data files.
    
    Args:
        data_dir: Directory containing data files
        metadata_dir: Directory containing metadata files
        data_ext: Extension of data files
        meta_ext: Extension of metadata files
        dry_run: If True, don't actually delete files, just log what would happen
        
    Returns:
        List[Path]: List of metadata files that were deleted (or would be deleted in dry run)
    """
    data_dir = Path(data_dir)
    metadata_dir = Path(metadata_dir)
    
    # Get all data file stems
    data_stems = {f.stem for f in data_dir.glob(f"*{data_ext}")}
    
    # Find orphaned metadata files
    orphaned = []
    for meta_file in metadata_dir.glob(f"*{meta_ext}"):
        if meta_file.stem not in data_stems:
            orphaned.append(meta_file)
    
    # Delete orphaned files
    deleted = []
    for meta_file in orphaned:
        if dry_run:
            logger.info(f"Would delete orphaned metadata file: {meta_file}")
            deleted.append(meta_file)
        else:
            try:
                meta_file.unlink()
                logger.info(f"Deleted orphaned metadata file: {meta_file}")
                deleted.append(meta_file)
            except Exception as e:
                logger.error(f"Error deleting {meta_file}: {e}")
    
    return deleted

def create_missing_metadata(data_dir: Union[str, Path],
                           metadata_dir: Union[str, Path],
                           data_ext: str = ".csv",
                           meta_ext: str = ".json",
                           template_function: Optional[Callable] = None) -> List[Path]:
    """
    Create metadata files for data files that don't have corresponding metadata.
    
    Args:
        data_dir: Directory containing data files
        metadata_dir: Directory containing metadata files
        data_ext: Extension of data files
        meta_ext: Extension of metadata files
        template_function: Function to generate metadata content (takes data file path as argument)
        
    Returns:
        List[Path]: List of metadata files that were created
    """
    data_dir = Path(data_dir)
    metadata_dir = Path(metadata_dir)
    
    # Ensure metadata directory exists
    metadata_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all metadata file stems
    meta_stems = {f.stem for f in metadata_dir.glob(f"*{meta_ext}")}
    
    # Find data files without metadata
    missing_meta = []
    for data_file in data_dir.glob(f"*{data_ext}"):
        if data_file.stem not in meta_stems:
            missing_meta.append(data_file)
    
    # Create missing metadata files
    created = []
    for data_file in missing_meta:
        meta_path = metadata_dir / f"{data_file.stem}{meta_ext}"
        
        try:
            if template_function:
                # Use template function to generate metadata
                meta_content = template_function(data_file)
            else:
                # Default simple metadata
                meta_content = {
                    "filename": data_file.name,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "file_size": data_file.stat().st_size
                }
            
            # Write metadata file
            with open(meta_path, 'w') as f:
                if isinstance(meta_content, dict):
                    json.dump(meta_content, f, indent=2)
                else:
                    f.write(str(meta_content))
            
            logger.info(f"Created metadata file: {meta_path}")
            created.append(meta_path)
        except Exception as e:
            logger.error(f"Error creating metadata for {data_file}: {e}")
    
    return created

def standardize_naming(directory: Union[str, Path],
                      pattern: str = "*",
                      lowercase: bool = True,
                      replace_spaces: bool = True,
                      space_replacement: str = "_",
                      dry_run: bool = False) -> List[Tuple[Path, Path]]:
    """
    Standardize filenames in a directory.
    
    Args:
        directory: Directory containing files
        pattern: Glob pattern to match files
        lowercase: Whether to convert filenames to lowercase
        replace_spaces: Whether to replace spaces
        space_replacement: Character to replace spaces with
        dry_run: If True, don't actually rename files, just log what would happen
        
    Returns:
        List[Tuple[Path, Path]]: List of (old_path, new_path) pairs that were renamed
    """
    directory = Path(directory)
    renamed = []
    
    for file_path in directory.glob(pattern):
        if not file_path.is_file():
            continue
            
        # Get stem and suffix
        stem = file_path.stem
        suffix = file_path.suffix
        
        # Apply standardization
        new_stem = stem
        if lowercase:
            new_stem = new_stem.lower()
        if replace_spaces:
            new_stem = new_stem.replace(" ", space_replacement)
        
        # Only rename if the name has changed
        if new_stem != stem:
            new_path = file_path.with_name(f"{new_stem}{suffix}")
            
            if dry_run:
                logger.info(f"Would rename {file_path} to {new_path}")
            else:
                if new_path.exists():
                    logger.warning(f"Cannot rename {file_path} to {new_path}: destination already exists")
                    continue
                    
                file_path.rename(new_path)
                logger.info(f"Renamed {file_path} to {new_path}")
                renamed.append((file_path, new_path))
    
    return renamed


# Example usage functions

def ensure_consistent_filenames(data_dir: Union[str, Path], 
                              metadata_dir: Union[str, Path],
                              backup: bool = True,
                              backup_dir: Optional[Union[str, Path]] = None) -> Dict:
    """
    Ensure data and metadata files have consistent naming.
    
    Args:
        data_dir: Directory containing data files
        metadata_dir: Directory containing metadata files
        backup: Whether to backup directories before making changes
        backup_dir: Directory to store backups (if None, use parent of data_dir)
        
    Returns:
        Dict: Summary of operations performed
    """
    data_dir = Path(data_dir)
    metadata_dir = Path(metadata_dir)
    
    result = {
        "backup_paths": {},
        "standardized": {},
        "synced": {},
        "orphaned_deleted": [],
        "missing_created": []
    }
    
    # Backup directories if requested
    if backup:
        if backup_dir is None:
            backup_dir = data_dir.parent / "backups"
        else:
            backup_dir = Path(backup_dir)
            
        result["backup_paths"]["data"] = str(backup_directory(data_dir, backup_dir))
        result["backup_paths"]["metadata"] = str(backup_directory(metadata_dir, backup_dir))
    
    # Step 1: Standardize filenames in both directories
    result["standardized"]["data"] = standardize_naming(data_dir, pattern="*.csv", dry_run=False)
    result["standardized"]["metadata"] = standardize_naming(metadata_dir, pattern="*.json", dry_run=False)
    
    # Step 2: Sync metadata filenames to match data filenames
    result["synced"] = sync_file_names(data_dir, metadata_dir, ".csv", ".json", force=False)
    
    # Step 3: Delete orphaned metadata files
    result["orphaned_deleted"] = delete_orphaned_metadata(data_dir, metadata_dir, dry_run=False)
    
    # Step 4: Create missing metadata files
    result["missing_created"] = create_missing_metadata(data_dir, metadata_dir)
    
    return result