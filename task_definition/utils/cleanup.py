"""
Cleanup Utility

This module provides functionality to clean up temporary files and directories.
"""

import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

def cleanup_temp_dir(temp_dir: Optional[Path] = None) -> None:
    """
    Clean up a temporary directory.
    
    Args:
        temp_dir: Path to the temporary directory to clean up.
                 If None, clean up all temporary directories created by the application.
    """
    if temp_dir is None:
        # Clean up all temporary directories created by the application
        temp_base = Path(tempfile.gettempdir())
        for item in temp_base.glob('robotics_repo_analyzer_*'):
            if item.is_dir():
                try:
                    shutil.rmtree(item)
                    logger.info(f"Cleaned up temporary directory: {item}")
                except Exception as e:
                    logger.warning(f"Error cleaning up temporary directory {item}: {e}")
    else:
        # Clean up the specified temporary directory
        try:
            if temp_dir.exists() and temp_dir.is_dir():
                shutil.rmtree(temp_dir)
                logger.info(f"Cleaned up temporary directory: {temp_dir}")
        except Exception as e:
            logger.warning(f"Error cleaning up temporary directory {temp_dir}: {e}")

def cleanup_output_files(output_dir: Path, pattern: str = '*.yaml') -> None:
    """
    Clean up output files matching a pattern.
    
    Args:
        output_dir: Path to the directory containing output files
        pattern: Glob pattern to match output files
    """
    try:
        for item in output_dir.glob(pattern):
            if item.is_file():
                try:
                    os.remove(item)
                    logger.info(f"Cleaned up output file: {item}")
                except Exception as e:
                    logger.warning(f"Error cleaning up output file {item}: {e}")
    except Exception as e:
        logger.warning(f"Error cleaning up output files in {output_dir}: {e}")

def cleanup_all() -> None:
    """
    Clean up all temporary files and directories.
    """
    # Clean up temporary directories
    cleanup_temp_dir()
    
    # Clean up output files in the current directory
    cleanup_output_files(Path.cwd())
    
    logger.info("Cleanup complete")