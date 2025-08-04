"""
Parameter Detector

This module detects parameters and constants in a robotics repository.
"""

import ast
import logging
import re
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple, Optional
import numpy as np

from task_definition.robotics_repo_analyzer.analyzers.code_analyzer import extract_parameters_from_python
from task_definition.robotics_repo_analyzer.analyzers.config_analyzer import extract_parameters_from_config
from task_definition.robotics_repo_analyzer.utils.confidence import calculate_confidence

logger = logging.getLogger(__name__)

class ParameterDetector:
    """
    Detector for parameters and constants in a robotics repository.
    """
    
    def __init__(self):
        """Initialize the parameter detector."""
        pass
        
    def detect(self, files_by_type: Dict[str, List[Path]]) -> Dict[str, Any]:
        """
        Detect parameters and constants in the repository.
        
        Args:
            files_by_type: Dictionary mapping file types to lists of file paths
            
        Returns:
            Dictionary containing detected parameters
        """
        logger.info("Detecting parameters...")
        
        # Extract parameters from Python files
        python_params = self._extract_from_python(files_by_type['python'])
        
        # Extract parameters from config files
        config_params = self._extract_from_config(files_by_type['config'])
        
        # Merge parameters
        all_params = self._merge_parameters(python_params, config_params)
        
        # Group parameters by type
        grouped_params = self._group_parameters(all_params)
        
        # Calculate overall confidence
        overall_confidence = calculate_confidence(
            {'parameters': all_params},
            ['parameters']
        )
        
        logger.info(f"Parameter detection complete. "
                   f"Found {len(all_params)} parameters. "
                   f"Overall confidence: {overall_confidence:.2f}")
        
        return {
            'parameters': all_params,
            'grouped_parameters': grouped_params,
            'confidence': overall_confidence
        }
    
    def _extract_from_python(self, python_files: List[Path]) -> List[Dict[str, Any]]:
        """
        Extract parameters from Python files.
        
        Args:
            python_files: List of Python file paths
            
        Returns:
            List of dictionaries containing parameters
        """
        all_params = []
        
        for file_path in python_files:
            try:
                # Skip files that don't exist or can't be read
                if not file_path.exists() or not file_path.is_file():
                    continue
                    
                params = extract_parameters_from_python(file_path)
                if params:
                    for param in params:
                        param['source'] = str(file_path)
                        param['source_type'] = 'python'
                        all_params.append(param)
            except Exception as e:
                logger.warning(f"Error extracting parameters from {file_path}: {e}")
        
        return all_params
    
    def _extract_from_config(self, config_files: List[Path]) -> List[Dict[str, Any]]:
        """
        Extract parameters from config files.
        
        Args:
            config_files: List of config file paths
            
        Returns:
            List of dictionaries containing parameters
        """
        all_params = []
        
        for file_path in config_files:
            try:
                # Skip files that don't exist or can't be read
                if not file_path.exists() or not file_path.is_file():
                    continue
                    
                params = extract_parameters_from_config(file_path)
                if params:
                    for param in params:
                        param['source'] = str(file_path)
                        param['source_type'] = 'config'
                        all_params.append(param)
            except Exception as e:
                logger.warning(f"Error extracting parameters from {file_path}: {e}")
        
        return all_params
    
    def _merge_parameters(self, *param_lists: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Merge parameters from multiple sources.
        
        Args:
            *param_lists: Lists of parameter dictionaries
            
        Returns:
            Merged list of parameters
        """
        all_params = []
        for param_list in param_lists:
            all_params.extend(param_list)
        
        # Deduplicate parameters
        deduplicated = []
        seen_names = set()
        
        for param in sorted(all_params, key=lambda p: p.get('confidence', 0), reverse=True):
            name = param.get('name')
            if not name:
                continue
                
            # Check if we've seen this parameter before
            if name in seen_names:
                # Update existing parameter if this one has higher confidence
                existing = next((p for p in deduplicated if p.get('name') == name), None)
                if existing and param.get('confidence', 0) > existing.get('confidence', 0):
                    existing.update(param)
            else:
                # Add new parameter
                deduplicated.append(param)
                seen_names.add(name)
        
        return deduplicated
    
    def _group_parameters(self, parameters: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Group parameters by type.
        
        Args:
            parameters: List of parameter dictionaries
            
        Returns:
            Dictionary mapping parameter types to lists of parameters
        """
        grouped = {
            'positions': [],
            'joint_positions': [],
            'velocities': [],
            'forces': [],
            'gains': [],
            'limits': [],
            'dimensions': [],
            'times': [],
            'other': []
        }
        
        # Keywords for each category
        keywords = {
            'positions': ['pos', 'position', 'location', 'coordinate', 'xyz', 'point'],
            'joint_positions': ['joint', 'angle', 'configuration', 'q', 'qpos'],
            'velocities': ['vel', 'velocity', 'speed', 'qvel'],
            'forces': ['force', 'torque', 'effort', 'strength', 'power'],
            'gains': ['gain', 'kp', 'kd', 'ki', 'p_gain', 'd_gain', 'i_gain', 'pid'],
            'limits': ['limit', 'bound', 'min', 'max', 'range', 'threshold'],
            'dimensions': ['dim', 'dimension', 'size', 'length', 'width', 'height', 'radius'],
            'times': ['time', 'duration', 'period', 'frequency', 'rate', 'dt']
        }
        
        for param in parameters:
            name = param.get('name', '').lower()
            value = param.get('value')
            
            # Skip parameters without values
            if value is None:
                continue
                
            # Determine parameter type based on name and value
            param_type = 'other'
            
            # Check name against keywords
            for type_name, type_keywords in keywords.items():
                if any(keyword in name for keyword in type_keywords):
                    param_type = type_name
                    break
            
            # If still 'other', try to infer from value
            if param_type == 'other':
                # Check if it's a position (3D vector)
                if isinstance(value, (list, tuple)) and len(value) == 3:
                    param_type = 'positions'
                # Check if it's a joint position (vector of length > 3)
                elif isinstance(value, (list, tuple)) and len(value) > 3:
                    param_type = 'joint_positions'
            
            # Add to the appropriate group
            grouped[param_type].append(param)
        
        return grouped