"""
PyBullet-specific analyzer for robotics repositories.
"""

import logging
import re
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import xml.etree.ElementTree as ET

from task_definition.robotics_repo_analyzer.frameworks.base_analyzer import FrameworkAnalyzer

logger = logging.getLogger(__name__)

class PyBulletAnalyzer(FrameworkAnalyzer):
    """
    PyBullet-specific analyzer for robotics repositories.
    """
    
    def detect(self, code: str) -> bool:
        """
        Detect if PyBullet is used in the code.
        
        Args:
            code: The code to analyze
            
        Returns:
            True if PyBullet is detected, False otherwise
        """
        patterns = [
            r'import\s+pybullet',
            r'from\s+pybullet\s+import',
            r'import\s+pybullet_data',
            r'p\.connect\s*\(',
            r'p\.loadURDF\s*\(',
            r'pybullet\.connect\s*\(',
            r'pybullet\.loadURDF\s*\('
        ]
        return any(re.search(pattern, code, re.MULTILINE) for pattern in patterns)
    
    def extract_robot_config(self, code: str, file_path: Path) -> Dict[str, Any]:
        """
        Extract robot configuration from PyBullet code.
        
        Args:
            code: The code to analyze
            file_path: Path to the file being analyzed
            
        Returns:
            Dictionary containing robot configuration
        """
        config = {
            'name': None,
            'dof': None,
            'joint_names': [],
            'joint_limits': [],
            'joint_types': [],
            'source': str(file_path),
            'confidence': 0.0
        }
        
        # Look for URDF loading
        urdf_patterns = [
            r'(?:p|pybullet)\.loadURDF\s*\(\s*[\'"](.+?)[\'"]\s*',
            r'loadURDF\s*\(\s*[\'"](.+?)[\'"]\s*'
        ]
        for pattern in urdf_patterns:
            match = re.search(pattern, code)
            if match:
                urdf_path = match.group(1)
                config['urdf_path'] = urdf_path
                config['confidence'] += 0.3
                
                # Try to extract robot name from URDF path
                robot_name = os.path.basename(urdf_path).split('.')[0]
                if robot_name:
                    config['name'] = robot_name
                    config['confidence'] += 0.1
                
                # Try to find and parse the URDF file
                urdf_file = self._find_urdf_file(file_path.parent, urdf_path)
                if urdf_file:
                    urdf_config = self._extract_from_urdf(urdf_file)
                    if urdf_config:
                        # Merge URDF config with higher confidence
                        for key, value in urdf_config.items():
                            if key == 'confidence':
                                config['confidence'] += value
                            elif not config.get(key) and value:
                                config[key] = value
        
        # Look for joint information
        joint_info_pattern = r'(?:p|pybullet)\.getJointInfo\s*\(\s*(\w+)\s*,\s*(\w+|\d+)\s*\)'
        for match in re.finditer(joint_info_pattern, code):
            # PyBullet's getJointInfo returns joint information
            # This indicates the code is accessing joint properties
            config['confidence'] += 0.1
        
        # Look for number of joints
        num_joints_pattern = r'(?:p|pybullet)\.getNumJoints\s*\(\s*(\w+)\s*\)'
        if re.search(num_joints_pattern, code):
            # Code is querying the number of joints
            config['confidence'] += 0.1
        
        # Look for joint control
        joint_control_pattern = r'(?:p|pybullet)\.setJointMotorControl2\s*\(\s*(\w+)\s*,\s*(.+?)\s*,'
        for match in re.finditer(joint_control_pattern, code):
            # Code is controlling joints
            config['confidence'] += 0.1
        
        return config
    
    def extract_action_sequences(self, code: str, file_path: Path) -> Dict[str, Any]:
        """
        Extract action sequences from PyBullet code.
        
        Args:
            code: The code to analyze
            file_path: Path to the file being analyzed
            
        Returns:
            Dictionary containing action sequences
        """
        sequences = []
        confidence = 0.0
        
        # Look for common PyBullet action patterns
        action_patterns = {
            'load_robot': r'(?:p|pybullet)\.loadURDF\s*\(\s*([^,]+)(?:,\s*([^,\)]+))?',
            'set_joint_position': r'(?:p|pybullet)\.setJointMotorControl2\s*\(\s*([^,]+)(?:,\s*([^,]+))?',
            'step_simulation': r'(?:p|pybullet)\.stepSimulation\s*\(\s*\)',
            'get_joint_state': r'(?:p|pybullet)\.getJointState\s*\(\s*([^,]+)(?:,\s*([^,\)]+))?',
            'reset_simulation': r'(?:p|pybullet)\.resetSimulation\s*\(\s*\)'
        }
        
        # Extract function definitions
        function_pattern = r'def\s+(\w+)\s*\([^)]*\):\s*((?:.|\n)*?)(?=\n\S|\Z)'
        for match in re.finditer(function_pattern, code):
            func_name = match.group(1)
            func_body = match.group(2)
            
            # Skip if function body is empty
            if not func_body.strip():
                continue
            
            # Check if function contains robotics-related actions
            actions = []
            for action_type, pattern in action_patterns.items():
                for action_match in re.finditer(pattern, func_body):
                    # Extract parameters
                    params = {}
                    if action_match.group(1):
                        params['target'] = action_match.group(1).strip()
                    if len(action_match.groups()) > 1 and action_match.group(2):
                        params['value'] = action_match.group(2).strip()
                    
                    # Add action to list
                    actions.append({
                        'action': action_type,
                        'parameters': params,
                        'line': self._get_line_number(func_body, action_match.start()) + match.start(2)
                    })
            
            # Look for for-loops with simulation steps (common pattern in PyBullet)
            step_loop_pattern = r'for\s+\w+\s+in\s+range\s*\((.+?)\):\s*((?:.|\n)*?)(?=\n\S|\Z)'
            for loop_match in re.finditer(step_loop_pattern, func_body):
                loop_body = loop_match.group(2)
                if re.search(r'(?:p|pybullet)\.stepSimulation\s*\(\s*\)', loop_body):
                    # This is a simulation loop
                    loop_actions = []
                    
                    # Extract actions in the loop
                    for action_type, pattern in action_patterns.items():
                        if action_type == 'step_simulation':
                            continue  # Skip step_simulation as it's part of the loop
                        
                        for action_match in re.finditer(pattern, loop_body):
                            # Extract parameters
                            params = {}
                            if action_match.group(1):
                                params['target'] = action_match.group(1).strip()
                            if len(action_match.groups()) > 1 and action_match.group(2):
                                params['value'] = action_match.group(2).strip()
                            
                            # Add action to list
                            loop_actions.append({
                                'action': action_type,
                                'parameters': params,
                                'line': self._get_line_number(loop_body, action_match.start()) + 
                                       match.start(2) + loop_match.start(2)
                            })
                    
                    if loop_actions:
                        # Add simulation loop as a single action
                        actions.append({
                            'action': 'simulation_loop',
                            'parameters': {
                                'iterations': loop_match.group(1).strip(),
                                'actions': [a['action'] for a in loop_actions]
                            },
                            'line': self._get_line_number(func_body, loop_match.start()) + match.start(2)
                        })
            
            # If function contains actions, add it as a sequence
            if actions:
                # Sort actions by line number
                actions.sort(key=lambda a: a['line'])
                
                # Add step numbers
                for i, action in enumerate(actions):
                    action['step'] = i + 1
                
                sequences.append({
                    'name': func_name,
                    'source': str(file_path),
                    'confidence': 0.7 if len(actions) > 3 else 0.5,
                    'steps': actions
                })
                confidence = max(confidence, 0.7 if len(actions) > 3 else 0.5)
        
        return {
            'sequences': sequences,
            'confidence': confidence
        }
    
    def extract_parameters(self, code: str, file_path: Path) -> Dict[str, Any]:
        """
        Extract parameters from PyBullet code.
        
        Args:
            code: The code to analyze
            file_path: Path to the file being analyzed
            
        Returns:
            Dictionary containing parameters
        """
        parameters = []
        confidence = 0.0
        
        # Look for constant definitions
        const_pattern = r'([A-Z][A-Z0-9_]*)\s*=\s*(.+?)(?:\s*#|\s*$)'
        for match in re.finditer(const_pattern, code, re.MULTILINE):
            name = match.group(1)
            value_str = match.group(2).strip()
            
            # Try to evaluate the value
            try:
                # Handle arrays
                if value_str.startswith('[') and value_str.endswith(']'):
                    # Simple array parsing
                    items = value_str[1:-1].split(',')
                    value = [float(item.strip()) for item in items if item.strip()]
                else:
                    # Try to evaluate as a number
                    value = float(value_str)
                
                parameters.append({
                    'name': name,
                    'value': value,
                    'source': str(file_path),
                    'line': self._get_line_number(code, match.start()),
                    'confidence': 0.8
                })
                confidence = max(confidence, 0.8)
            except (ValueError, SyntaxError):
                # Skip if we can't parse the value
                pass
        
        # Look for position parameters (common in PyBullet)
        pos_patterns = [
            r'(?:startPos|startPosition|basePosition)\s*=\s*\[([\d\.\-\+\s,]+)\]',
            r'(?:p|pybullet)\.resetBasePositionAndOrientation\s*\(\s*\w+\s*,\s*\[([\d\.\-\+\s,]+)\]'
        ]
        for pattern in pos_patterns:
            for match in re.finditer(pattern, code):
                value_str = match.group(1)
                try:
                    # Parse array values
                    items = value_str.split(',')
                    value = [float(item.strip()) for item in items if item.strip()]
                    
                    parameters.append({
                        'name': 'position',
                        'value': value,
                        'source': str(file_path),
                        'line': self._get_line_number(code, match.start()),
                        'type': 'position',
                        'confidence': 0.7
                    })
                    confidence = max(confidence, 0.7)
                except ValueError:
                    # Skip if we can't parse the value
                    pass
        
        # Look for joint control parameters
        joint_control_pattern = r'(?:p|pybullet)\.setJointMotorControl2\s*\(\s*\w+\s*,\s*(.+?)\s*,\s*(?:p|pybullet)\.(\w+)\s*,\s*(.+?)\s*,\s*(.+?)\s*\)'
        for match in re.finditer(joint_control_pattern, code):
            try:
                joint_index = match.group(1).strip()
                control_mode = match.group(2).strip()
                target_value = match.group(3).strip()
                force_value = match.group(4).strip()
                
                # Try to parse target value
                target = None
                try:
                    target = float(target_value)
                except ValueError:
                    pass
                
                # Try to parse force value
                force = None
                try:
                    force = float(force_value)
                except ValueError:
                    pass
                
                parameters.append({
                    'name': f'joint_control_{control_mode.lower()}',
                    'value': {
                        'joint': joint_index,
                        'mode': control_mode,
                        'target': target,
                        'force': force
                    },
                    'source': str(file_path),
                    'line': self._get_line_number(code, match.start()),
                    'type': 'joint_control',
                    'confidence': 0.7
                })
                confidence = max(confidence, 0.7)
            except (IndexError, ValueError):
                # Skip if we can't parse the values
                pass
        
        # Group parameters by type
        grouped_parameters = {}
        for param in parameters:
            param_type = param.get('type', 'unknown')
            if param_type not in grouped_parameters:
                grouped_parameters[param_type] = []
            grouped_parameters[param_type].append(param)
        
        return {
            'parameters': parameters,
            'grouped_parameters': grouped_parameters,
            'confidence': confidence
        }
    
    def _find_urdf_file(self, base_dir: Path, urdf_path: str) -> Optional[Path]:
        """
        Find a URDF file based on a path from code.
        
        Args:
            base_dir: Base directory to search from
            urdf_path: Path to the URDF file from code
            
        Returns:
            Path to the URDF file if found, None otherwise
        """
        # Try direct path
        direct_path = base_dir / urdf_path
        if direct_path.exists():
            return direct_path
        
        # Try with common directories
        common_dirs = ['assets', 'models', 'urdf', 'xml', 'config', 'resources']
        for common_dir in common_dirs:
            common_path = base_dir / common_dir / os.path.basename(urdf_path)
            if common_path.exists():
                return common_path
        
        # Try recursive search
        for root, _, files in os.walk(base_dir):
            if os.path.basename(urdf_path) in files:
                return Path(root) / os.path.basename(urdf_path)
        
        return None
    
    def _extract_from_urdf(self, urdf_file: Path) -> Dict[str, Any]:
        """
        Extract robot configuration from a URDF file.
        
        Args:
            urdf_file: Path to the URDF file
            
        Returns:
            Dictionary containing robot configuration
        """
        try:
            tree = ET.parse(urdf_file)
            root = tree.getroot()
            
            config = {
                'name': None,
                'dof': None,
                'joint_names': [],
                'joint_limits': [],
                'joint_types': [],
                'source': str(urdf_file),
                'confidence': 0.0
            }
            
            # Extract robot name
            if 'name' in root.attrib:
                config['name'] = root.attrib['name']
                config['confidence'] += 0.2
            
            # Extract joints
            for joint in root.findall('.//joint'):
                if 'name' in joint.attrib:
                    joint_name = joint.attrib['name']
                    config['joint_names'].append(joint_name)
                    
                    # Extract joint type
                    joint_type = joint.attrib.get('type', 'revolute')
                    config['joint_types'].append(joint_type)
                    
                    # Extract joint limits
                    lower = None
                    upper = None
                    
                    # Check for limit element
                    limit = joint.find('limit')
                    if limit is not None:
                        if 'lower' in limit.attrib:
                            try:
                                lower = float(limit.attrib['lower'])
                            except ValueError:
                                pass
                        if 'upper' in limit.attrib:
                            try:
                                upper = float(limit.attrib['upper'])
                            except ValueError:
                                pass
                    
                    config['joint_limits'].append([lower, upper])
                    config['confidence'] += 0.1
            
            # Set DOF based on number of joints
            if config['joint_names']:
                config['dof'] = len(config['joint_names'])
                config['confidence'] += 0.2
            
            return config
        except Exception as e:
            logger.warning(f"Error parsing URDF file {urdf_file}: {e}")
            return None
    
    def _get_line_number(self, code: str, pos: int) -> int:
        """
        Get the line number for a position in code.
        
        Args:
            code: The code
            pos: Position in the code
            
        Returns:
            Line number (1-based)
        """
        return code[:pos].count('\n') + 1