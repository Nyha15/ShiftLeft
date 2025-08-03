"""
File Utilities Module

This module provides utility functions for file operations.
"""

import logging
import os
from pathlib import Path
from typing import List, Set, Dict, Any, Optional

logger = logging.getLogger(__name__)

def is_binary_file(file_path: Path) -> bool:
    """
    Check if a file is binary.
    
    Args:
        file_path: Path to the file
        
    Returns:
        True if the file is binary, False otherwise
    """
    try:
        with open(file_path, 'rb') as f:
            chunk = f.read(1024)
            return b'\0' in chunk
    except Exception as e:
        logger.warning(f"Error checking if file is binary: {e}")
        return False

def get_file_size(file_path: Path) -> int:
    """
    Get the size of a file in bytes.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Size of the file in bytes
    """
    try:
        return os.path.getsize(file_path)
    except Exception as e:
        logger.warning(f"Error getting file size: {e}")
        return 0

def is_excluded_file(file_path: Path, exclude_patterns: List[str]) -> bool:
    """
    Check if a file should be excluded based on patterns.
    
    Args:
        file_path: Path to the file
        exclude_patterns: List of patterns to exclude
        
    Returns:
        True if the file should be excluded, False otherwise
    """
    file_str = str(file_path)
    return any(pattern in file_str for pattern in exclude_patterns)

def find_files_by_extension(directory: Path, extensions: List[str]) -> List[Path]:
    """
    Find files with specific extensions in a directory.
    
    Args:
        directory: Directory to search
        extensions: List of file extensions to find
        
    Returns:
        List of file paths
    """
    files = []
    for ext in extensions:
        files.extend(directory.glob(f"**/*{ext}"))
    return files

def read_file_content(file_path: Path) -> Optional[str]:
    """
    Read the content of a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Content of the file, or None if an error occurs
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        logger.warning(f"Error reading file {file_path}: {e}")
        return None