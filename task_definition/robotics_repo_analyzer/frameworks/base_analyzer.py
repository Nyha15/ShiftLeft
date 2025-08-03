"""
Base class for framework-specific analyzers.
"""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class FrameworkAnalyzer:
    """
    Base class for framework-specific analyzers.
    
    This class defines the interface that all framework-specific analyzers must implement.
    """
    
    def detect(self, code: str) -> bool:
        """
        Detect if this framework is used in the code.
        
        Args:
            code: The code to analyze
            
        Returns:
            True if the framework is detected, False otherwise
        """
        raise NotImplementedError("Subclasses must implement detect()")
    
    def detect_in_files(self, files: List[Path]) -> bool:
        """
        Detect if this framework is used in any of the files.
        
        Args:
            files: List of file paths to analyze
            
        Returns:
            True if the framework is detected in any file, False otherwise
        """
        for file_path in files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    code = f.read()
                if self.detect(code):
                    logger.info(f"Detected framework in {file_path}")
                    return True
            except Exception as e:
                logger.warning(f"Error reading file {file_path}: {e}")
        return False
    
    def extract_robot_config(self, code: str, file_path: Path) -> Dict[str, Any]:
        """
        Extract robot configuration from code.
        
        Args:
            code: The code to analyze
            file_path: Path to the file being analyzed
            
        Returns:
            Dictionary containing robot configuration
        """
        raise NotImplementedError("Subclasses must implement extract_robot_config()")
    
    def extract_action_sequences(self, code: str, file_path: Path) -> Dict[str, Any]:
        """
        Extract action sequences from code.
        
        Args:
            code: The code to analyze
            file_path: Path to the file being analyzed
            
        Returns:
            Dictionary containing action sequences
        """
        raise NotImplementedError("Subclasses must implement extract_action_sequences()")
    
    def extract_parameters(self, code: str, file_path: Path) -> Dict[str, Any]:
        """
        Extract parameters from code.
        
        Args:
            code: The code to analyze
            file_path: Path to the file being analyzed
            
        Returns:
            Dictionary containing parameters
        """
        raise NotImplementedError("Subclasses must implement extract_parameters()")