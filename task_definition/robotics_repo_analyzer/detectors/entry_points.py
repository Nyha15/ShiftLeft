"""
Entry Point Detector

This module detects potential entry points in a robotics repository.
"""

import ast
import logging
import re
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple
import nbformat

from task_definition.robotics_repo_analyzer.analyzers.code_analyzer import analyze_python_file
from task_definition.robotics_repo_analyzer.utils.confidence import calculate_confidence

logger = logging.getLogger(__name__)

# Patterns for identifying entry points
ENTRY_POINT_PATTERNS = [
    r'if\s+__name__\s*==\s*[\'"]__main__[\'"]',
    r'def\s+main\s*\(',
    r'class\s+\w+\(.*(?:Environment|Env|Robot|Controller)',
]

# Robotics-related imports
ROBOTICS_IMPORTS = {
    'mujoco': 0.9,
    'dm_control': 0.9,
    'pybullet': 0.9,
    'gym': 0.8,
    'rospy': 0.9,
    'rclpy': 0.9,
    'ros2': 0.9,
    'moveit': 0.9,
    'tf': 0.8,
    'tf2_ros': 0.8,
    'sensor_msgs': 0.7,
    'geometry_msgs': 0.7,
    'controller_manager': 0.8,
    'gazebo': 0.8,
    'isaac': 0.8,
    'nvidia': 0.7,
    'pytorch': 0.6,
    'tensorflow': 0.6,
    'numpy': 0.5,
    'scipy': 0.5,
    'matplotlib': 0.4,
}

class EntryPointDetector:
    """
    Detector for entry points in a robotics repository.
    """
    
    def __init__(self):
        """Initialize the entry point detector."""
        pass
        
    def detect(self, files_by_type: Dict[str, List[Path]]) -> Dict[str, Any]:
        """
        Detect entry points in the repository.
        
        Args:
            files_by_type: Dictionary mapping file types to lists of file paths
            
        Returns:
            Dictionary containing detected entry points
        """
        logger.info("Detecting entry points...")
        
        # Analyze Python files
        python_entry_points = self._analyze_python_files(files_by_type['python'])
        
        # Analyze Jupyter notebooks
        notebook_entry_points = self._analyze_notebooks(files_by_type['notebook'])
        
        # Combine all entry points
        all_entry_points = python_entry_points + notebook_entry_points
        
        # Sort by score
        sorted_entry_points = sorted(
            all_entry_points,
            key=lambda ep: ep['score'],
            reverse=True
        )
        
        logger.info(f"Entry point detection complete. "
                   f"Found {len(sorted_entry_points)} potential entry points.")
        
        return {
            'entry_points': sorted_entry_points,
            'confidence': calculate_confidence({'entry_points': sorted_entry_points}, ['entry_points'])
        }
    
    def _analyze_python_files(self, python_files: List[Path]) -> List[Dict[str, Any]]:
        """
        Analyze Python files for potential entry points.
        
        Args:
            python_files: List of Python file paths
            
        Returns:
            List of dictionaries containing entry point information
        """
        entry_points = []
        
        for file_path in python_files:
            try:
                # Skip files that are likely not entry points based on name
                if self._is_likely_utility(file_path.name):
                    continue
                
                # Analyze the file
                analysis_result = analyze_python_file(file_path)
                
                # Calculate entry point score
                score = self._calculate_entry_point_score(analysis_result, file_path)
                
                if score > 0.3:  # Only include files with a reasonable score
                    entry_point = {
                        'file': str(file_path),
                        'score': score,
                        'patterns': analysis_result.get('patterns', []),
                        'imports': analysis_result.get('imports', []),
                        'has_main': analysis_result.get('has_main', False),
                        'has_main_function': analysis_result.get('has_main_function', False),
                        'type': 'python'
                    }
                    entry_points.append(entry_point)
            except Exception as e:
                logger.warning(f"Error analyzing {file_path}: {e}")
        
        return entry_points
    
    def _analyze_notebooks(self, notebook_files: List[Path]) -> List[Dict[str, Any]]:
        """
        Analyze Jupyter notebooks for potential entry points.
        
        Args:
            notebook_files: List of notebook file paths
            
        Returns:
            List of dictionaries containing entry point information
        """
        entry_points = []
        
        for file_path in notebook_files:
            try:
                # Read the notebook
                with open(file_path, 'r', encoding='utf-8') as f:
                    nb = nbformat.read(f, as_version=4)
                
                # Extract code cells
                code_cells = [cell for cell in nb.cells if cell.cell_type == 'code']
                
                # Extract imports and robotics-related code
                imports = []
                robotics_patterns = []
                
                for cell in code_cells:
                    try:
                        # Parse the cell
                        tree = ast.parse(cell.source)
                        
                        # Extract imports
                        for node in ast.walk(tree):
                            if isinstance(node, ast.Import):
                                for name in node.names:
                                    imports.append(name.name)
                            elif isinstance(node, ast.ImportFrom):
                                if node.module:
                                    imports.append(node.module)
                        
                        # Check for robotics-related patterns
                        for pattern in ENTRY_POINT_PATTERNS:
                            if re.search(pattern, cell.source):
                                robotics_patterns.append(pattern)
                    except SyntaxError:
                        # Skip cells with syntax errors
                        continue
                
                # Calculate score
                robotics_imports = [imp for imp in imports if any(
                    ri in imp for ri in ROBOTICS_IMPORTS.keys()
                )]
                
                score = 0.0
                if robotics_imports:
                    # Base score on imports
                    for imp in robotics_imports:
                        for ri, ri_score in ROBOTICS_IMPORTS.items():
                            if ri in imp:
                                score += ri_score
                                break
                    
                    # Normalize score
                    score = min(score, 0.9)
                    
                    # Boost score for notebooks with robotics patterns
                    if robotics_patterns:
                        score += 0.1 * len(robotics_patterns)
                    
                    score = min(score, 1.0)
                    
                    entry_point = {
                        'file': str(file_path),
                        'score': score,
                        'patterns': robotics_patterns,
                        'imports': robotics_imports,
                        'type': 'notebook'
                    }
                    entry_points.append(entry_point)
            except Exception as e:
                logger.warning(f"Error analyzing notebook {file_path}: {e}")
        
        return entry_points
    
    def _calculate_entry_point_score(self, analysis_result: Dict[str, Any], file_path: Path) -> float:
        """
        Calculate a score for a potential entry point.
        
        Args:
            analysis_result: Result of analyzing a Python file
            file_path: Path to the Python file
            
        Returns:
            Score between 0.0 and 1.0
        """
        score = 0.0
        
        # Check for main block or main function
        if analysis_result.get('has_main', False):
            score += 0.3
        if analysis_result.get('has_main_function', False):
            score += 0.2
        
        # Check for robotics-related imports
        imports = analysis_result.get('imports', [])
        for imp in imports:
            for ri, ri_score in ROBOTICS_IMPORTS.items():
                if ri in imp:
                    score += ri_score * 0.1
                    break
        
        # Check for robotics-related patterns
        patterns = analysis_result.get('patterns', [])
        score += 0.05 * len(patterns)
        
        # Check filename for hints
        filename = file_path.name.lower()
        if any(name in filename for name in ['demo', 'example', 'test', 'run', 'main']):
            score += 0.1
        if any(name in filename for name in ['robot', 'control', 'env', 'sim']):
            score += 0.1
        
        # Normalize score
        return min(score, 1.0)
    
    def _is_likely_utility(self, filename: str) -> bool:
        """
        Check if a file is likely a utility file rather than an entry point.
        
        Args:
            filename: Name of the file
            
        Returns:
            True if the file is likely a utility, False otherwise
        """
        utility_patterns = [
            'utils', 'util', 'helpers', 'common', 'tools', 'setup',
            'config', 'constants', 'settings', 'types', 'exceptions',
            'test_', 'tests_', '__init__'
        ]
        
        return any(pattern in filename.lower() for pattern in utility_patterns)