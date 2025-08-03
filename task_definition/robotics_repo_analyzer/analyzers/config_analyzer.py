"""
Config Analyzer Module

This module provides functionality to analyze configuration files (JSON, YAML, etc.)
for robotics-related information.
"""

import json
import logging
import re
import yaml
import toml
import configparser
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set, Union

logger = logging.getLogger(__name__)

# Keywords for robot configuration
ROBOT_CONFIG_KEYWORDS = [
    'robot', 'arm', 'manipulator', 'joints', 'dof', 'degrees_of_freedom',
    'joint_names', 'joint_limits', 'joint_types', 'links', 'actuators',
    'effector', 'gripper', 'controller', 'kinematics', 'dynamics'
]

# Keywords for parameters
PARAMETER_KEYWORDS = [
    'position', 'orientation', 'pose', 'location', 'target', 'goal',
    'velocity', 'speed', 'acceleration', 'force', 'torque', 'mass',
    'inertia', 'damping', 'stiffness', 'friction', 'threshold',
    'gain', 'kp', 'kd', 'ki', 'p_gain', 'd_gain', 'i_gain', 'pid',
    'limit', 'bound', 'min', 'max', 'range', 'dimension', 'size',
    'length', 'width', 'height', 'radius', 'time', 'duration',
    'period', 'frequency', 'rate', 'dt', 'home', 'config', 'param',
    'setting', 'constant'
]

def extract_robot_specs_from_config(file_path: Path) -> Dict[str, Any]:
    """
    Extract robot specifications from a configuration file.
    
    Args:
        file_path: Path to the configuration file
        
    Returns:
        Dictionary containing robot specifications
    """
    try:
        # Load the configuration file
        config_data = _load_config_file(file_path)
        
        if not config_data:
            return None
        
        # Initialize robot specs
        robot_specs = {
            'name': None,
            'dof': None,
            'joint_names': [],
            'joint_limits': [],
            'joint_types': [],
            'confidence': 0.0
        }
        
        # Extract robot specifications
        _extract_robot_specs_recursive(config_data, robot_specs)
        
        # Calculate confidence
        confidence = 0.0
        if robot_specs['name']:
            confidence += 0.1
        if robot_specs['dof'] is not None:
            confidence += 0.3
        if robot_specs['joint_names']:
            confidence += 0.3
        if robot_specs['joint_limits']:
            confidence += 0.3
        
        robot_specs['confidence'] = confidence
        
        # Only return if we found something useful
        if confidence > 0:
            return robot_specs
        
        return None
    except Exception as e:
        logger.error(f"Error extracting robot specs from config file {file_path}: {e}")
        return None


def extract_parameters_from_config(file_path: Path) -> List[Dict[str, Any]]:
    """
    Extract parameters from a configuration file.
    
    Args:
        file_path: Path to the configuration file
        
    Returns:
        List of dictionaries containing parameters
    """
    try:
        # Load the configuration file
        config_data = _load_config_file(file_path)
        
        if not config_data:
            return []
        
        # Extract parameters
        parameters = []
        _extract_parameters_recursive(config_data, parameters, [])
        
        return parameters
    except Exception as e:
        logger.error(f"Error extracting parameters from config file {file_path}: {e}")
        return []


def _load_config_file(file_path: Path) -> Any:
    """
    Load a configuration file based on its extension.
    
    Args:
        file_path: Path to the configuration file
        
    Returns:
        Loaded configuration data
    """
    suffix = file_path.suffix.lower()
    
    try:
        if suffix in ['.json']:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        elif suffix in ['.yaml', '.yml']:
            with open(file_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        elif suffix in ['.toml']:
            with open(file_path, 'r', encoding='utf-8') as f:
                return toml.load(f)
        elif suffix in ['.ini', '.cfg']:
            config = configparser.ConfigParser()
            config.read(file_path)
            # Convert ConfigParser to dict
            return {section: dict(config[section]) for section in config.sections()}
        else:
            logger.warning(f"Unsupported config file type: {suffix}")
            return None
    except Exception as e:
        logger.warning(f"Error loading config file {file_path}: {e}")
        return None


def _extract_robot_specs_recursive(data: Any, robot_specs: Dict[str, Any], path: List[str] = None) -> None:
    """
    Recursively extract robot specifications from configuration data.
    
    Args:
        data: Configuration data
        robot_specs: Dictionary to store robot specifications
        path: Current path in the configuration hierarchy
    """
    if path is None:
        path = []
    
    if isinstance(data, dict):
        for key, value in data.items():
            current_path = path + [key]
            
            # Check if this is a robot configuration section
            try:
                is_robot_section = any(keyword in str(key).lower() for keyword in ROBOT_CONFIG_KEYWORDS)
            except Exception:
                is_robot_section = False
            
            if is_robot_section and isinstance(value, dict):
                # Extract robot name
                if 'name' in value and isinstance(value['name'], str) and not robot_specs['name']:
                    robot_specs['name'] = value['name']
                
                # Extract DOF
                for dof_key in ['dof', 'degrees_of_freedom', 'n_joints', 'num_joints']:
                    if dof_key in value and isinstance(value[dof_key], (int, float)) and not robot_specs['dof']:
                        robot_specs['dof'] = int(value[dof_key])
                
                # Extract joint names
                for names_key in ['joint_names', 'joints', 'actuators']:
                    if names_key in value and isinstance(value[names_key], list) and not robot_specs['joint_names']:
                        robot_specs['joint_names'] = value[names_key]
                
                # Extract joint limits
                for limits_key in ['joint_limits', 'limits', 'joint_ranges', 'ranges']:
                    if limits_key in value and isinstance(value[limits_key], list) and not robot_specs['joint_limits']:
                        # Check if it's a list of lists/tuples
                        if all(isinstance(limit, (list, tuple)) and len(limit) == 2 for limit in value[limits_key]):
                            robot_specs['joint_limits'] = value[limits_key]
                
                # Extract joint types
                for types_key in ['joint_types', 'types']:
                    if types_key in value and isinstance(value[types_key], list) and not robot_specs['joint_types']:
                        robot_specs['joint_types'] = value[types_key]
            
            # Continue recursion
            if isinstance(value, (dict, list)):
                _extract_robot_specs_recursive(value, robot_specs, current_path)
    
    elif isinstance(data, list):
        for i, item in enumerate(data):
            current_path = path + [str(i)]
            if isinstance(item, (dict, list)):
                _extract_robot_specs_recursive(item, robot_specs, current_path)


def _extract_parameters_recursive(data: Any, parameters: List[Dict[str, Any]], path: List[str], parent_key: str = None) -> None:
    """
    Recursively extract parameters from configuration data.
    
    Args:
        data: Configuration data
        parameters: List to store extracted parameters
        path: Current path in the configuration hierarchy
        parent_key: Key of the parent node
    """
    if isinstance(data, dict):
        for key, value in data.items():
            current_path = path + [key]
            
            # Check if this key is a parameter
            try:
                is_parameter = any(keyword in str(key).lower() for keyword in PARAMETER_KEYWORDS)
            except Exception:
                is_parameter = False
            
            if is_parameter and _is_valid_parameter_value(value):
                parameter = {
                    'name': '.'.join(current_path),
                    'value': value,
                    'path': '.'.join(current_path),
                    'confidence': 0.7  # High confidence for named parameters in config files
                }
                parameters.append(parameter)
            
            # Continue recursion
            if isinstance(value, (dict, list)):
                _extract_parameters_recursive(value, parameters, current_path, key)
    
    elif isinstance(data, list):
        # Check if this is a parameter array (e.g., positions, joint angles)
        if parent_key:
            try:
                is_parameter_array = any(keyword in str(parent_key).lower() for keyword in PARAMETER_KEYWORDS)
            except Exception:
                is_parameter_array = False
                
            if is_parameter_array and _is_valid_parameter_array(data):
                parameter = {
                    'name': '.'.join(path),
                    'value': data,
                    'path': '.'.join(path),
                    'confidence': 0.7  # High confidence for named parameters in config files
                }
                parameters.append(parameter)
        
        # Continue recursion for each item
        for i, item in enumerate(data):
            current_path = path + [str(i)]
            if isinstance(item, (dict, list)):
                _extract_parameters_recursive(item, parameters, current_path, parent_key)


def _is_valid_parameter_value(value: Any) -> bool:
    """
    Check if a value is a valid parameter value.
    
    Args:
        value: Value to check
        
    Returns:
        True if the value is a valid parameter value, False otherwise
    """
    if isinstance(value, (int, float)):
        # Skip very large or very small numbers
        if abs(value) > 1e10 or abs(value) < 1e-10:
            return False
        return True
    
    if isinstance(value, (list, tuple)):
        # Check if it's a numeric array
        if all(isinstance(v, (int, float)) for v in value):
            # Skip very long arrays
            if len(value) > 100:
                return False
            return True
    
    return False


def _is_valid_parameter_array(array: List[Any]) -> bool:
    """
    Check if an array is a valid parameter array.
    
    Args:
        array: Array to check
        
    Returns:
        True if the array is a valid parameter array, False otherwise
    """
    # Skip empty arrays
    if not array:
        return False
    
    # Skip very long arrays
    if len(array) > 100:
        return False
    
    # Check if it's a numeric array
    if all(isinstance(v, (int, float)) for v in array):
        return True
    
    # Check if it's an array of arrays (e.g., joint limits)
    if all(isinstance(v, (list, tuple)) for v in array):
        # Check if all inner arrays are numeric and have the same length
        if all(all(isinstance(x, (int, float)) for x in v) for v in array):
            return True
    
    return False