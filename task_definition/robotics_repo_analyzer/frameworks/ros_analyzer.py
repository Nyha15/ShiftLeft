"""
ROS-specific analyzer for robotics repositories.
"""

import logging
import re
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import xml.etree.ElementTree as ET

from robotics_repo_analyzer.frameworks.base_analyzer import FrameworkAnalyzer

logger = logging.getLogger(__name__)

class ROSAnalyzer(FrameworkAnalyzer):
    """
    ROS-specific analyzer for robotics repositories.
    """
    
    def detect(self, code: str) -> bool:
        """
        Detect if ROS is used in the code.
        
        Args:
            code: The code to analyze
            
        Returns:
            True if ROS is detected, False otherwise
        """
        patterns = [
            r'import\s+rospy',
            r'from\s+rospy\s+import',
            r'import\s+rclpy',
            r'from\s+rclpy\s+import',
            r'import\s+moveit_commander',
            r'from\s+moveit_commander\s+import',
            r'import\s+tf',
            r'from\s+tf\s+import',
            r'import\s+geometry_msgs',
            r'from\s+geometry_msgs\s+import',
            r'import\s+sensor_msgs',
            r'from\s+sensor_msgs\s+import'
        ]
        return any(re.search(pattern, code, re.MULTILINE) for pattern in patterns)
    
    def extract_robot_config(self, code: str, file_path: Path) -> Dict[str, Any]:
        """
        Extract robot configuration from ROS code.
        
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
        
        # Look for MoveIt robot commander initialization
        moveit_patterns = [
            r'(?:robot|arm)\s*=\s*moveit_commander\.RobotCommander\s*\(\s*\)',
            r'(?:robot|arm)\s*=\s*moveit_commander\.MoveGroupCommander\s*\(\s*[\'"](.+?)[\'"]\s*\)'
        ]
        for pattern in moveit_patterns:
            match = re.search(pattern, code)
            if match:
                config['confidence'] += 0.3
                if len(match.groups()) > 0 and match.group(1):
                    # Extract group name
                    group_name = match.group(1)
                    config['group_name'] = group_name
                    config['confidence'] += 0.1
        
        # Look for joint names
        joint_names_pattern = r'joint_names\s*=\s*\[(.*?)\]'
        match = re.search(joint_names_pattern, code, re.DOTALL)
        if match:
            joint_names_str = match.group(1)
            # Extract joint names from string
            joint_names = []
            for name_match in re.finditer(r'[\'"](.+?)[\'"]', joint_names_str):
                joint_names.append(name_match.group(1))
            
            if joint_names:
                config['joint_names'] = joint_names
                config['confidence'] += 0.2
                
                # Set DOF based on number of joints
                config['dof'] = len(joint_names)
                config['confidence'] += 0.1
        
        # Look for URDF loading
        urdf_patterns = [
            r'urdf\s*=\s*rospy\.get_param\s*\(\s*[\'"]robot_description[\'"]\s*\)',
            r'robot_description\s*=\s*rospy\.get_param\s*\(\s*[\'"]robot_description[\'"]\s*\)'
        ]
        for pattern in urdf_patterns:
            if re.search(pattern, code):
                config['confidence'] += 0.2
        
        # Look for joint state subscriber
        joint_state_pattern = r'rospy\.Subscriber\s*\(\s*[\'"]joint_states[\'"]\s*,\s*JointState'
        if re.search(joint_state_pattern, code):
            config['confidence'] += 0.1
        
        return config
    
    def extract_action_sequences(self, code: str, file_path: Path) -> Dict[str, Any]:
        """
        Extract action sequences from ROS code.
        
        Args:
            code: The code to analyze
            file_path: Path to the file being analyzed
            
        Returns:
            Dictionary containing action sequences
        """
        sequences = []
        confidence = 0.0
        
        # Look for common ROS action patterns
        action_patterns = {
            'move_to_joint_target': r'(?:move_group|arm|robot)\.set_joint_value_target\s*\(\s*(.+?)\s*\)',
            'move_to_pose_target': r'(?:move_group|arm|robot)\.set_pose_target\s*\(\s*(.+?)\s*\)',
            'plan': r'(?:move_group|arm|robot)\.plan\s*\(\s*\)',
            'execute': r'(?:move_group|arm|robot)\.execute\s*\(\s*(.+?)\s*\)',
            'go': r'(?:move_group|arm|robot)\.go\s*\(\s*(.+?)\s*\)',
            'publish': r'(?:pub|publisher)\.publish\s*\(\s*(.+?)\s*\)',
            'wait_for_service': r'rospy\.wait_for_service\s*\(\s*[\'"](.+?)[\'"]\s*\)',
            'call_service': r'rospy\.ServiceProxy\s*\(\s*[\'"](.+?)[\'"]\s*,\s*(.+?)\s*\)'
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
                    if len(action_match.groups()) > 0 and action_match.group(1):
                        params['target'] = action_match.group(1).strip()
                    if len(action_match.groups()) > 1 and action_match.group(2):
                        params['type'] = action_match.group(2).strip()
                    
                    # Add action to list
                    actions.append({
                        'action': action_type,
                        'parameters': params,
                        'line': self._get_line_number(func_body, action_match.start()) + match.start(2)
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
        
        # Look for ROS node main function
        main_pattern = r'if\s+__name__\s*==\s*[\'"]__main__[\'"]\s*:\s*((?:.|\n)*?)(?=\n\S|\Z)'
        match = re.search(main_pattern, code)
        if match:
            main_body = match.group(1)
            
            # Check if main contains robotics-related actions
            actions = []
            for action_type, pattern in action_patterns.items():
                for action_match in re.finditer(pattern, main_body):
                    # Extract parameters
                    params = {}
                    if len(action_match.groups()) > 0 and action_match.group(1):
                        params['target'] = action_match.group(1).strip()
                    if len(action_match.groups()) > 1 and action_match.group(2):
                        params['type'] = action_match.group(2).strip()
                    
                    # Add action to list
                    actions.append({
                        'action': action_type,
                        'parameters': params,
                        'line': self._get_line_number(main_body, action_match.start()) + match.start(1)
                    })
            
            # If main contains actions, add it as a sequence
            if actions:
                # Sort actions by line number
                actions.sort(key=lambda a: a['line'])
                
                # Add step numbers
                for i, action in enumerate(actions):
                    action['step'] = i + 1
                
                sequences.append({
                    'name': 'main',
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
        Extract parameters from ROS code.
        
        Args:
            code: The code to analyze
            file_path: Path to the file being analyzed
            
        Returns:
            Dictionary containing parameters
        """
        parameters = []
        confidence = 0.0
        
        # Look for ROS parameter server access
        param_patterns = [
            r'rospy\.get_param\s*\(\s*[\'"](.+?)[\'"]\s*(?:,\s*(.+?))?\s*\)',
            r'rospy\.set_param\s*\(\s*[\'"](.+?)[\'"]\s*,\s*(.+?)\s*\)'
        ]
        for pattern in param_patterns:
            for match in re.finditer(pattern, code):
                param_name = match.group(1)
                param_value = None
                if len(match.groups()) > 1 and match.group(2):
                    param_value = match.group(2).strip()
                
                # Try to evaluate the value
                value = None
                if param_value:
                    try:
                        # Handle arrays
                        if param_value.startswith('[') and param_value.endswith(']'):
                            # Simple array parsing
                            items = param_value[1:-1].split(',')
                            value = [float(item.strip()) for item in items if item.strip()]
                        elif param_value.startswith('"') or param_value.startswith("'"):
                            # String value
                            value = param_value.strip('"\'')
                        else:
                            # Try to evaluate as a number
                            value = float(param_value)
                    except (ValueError, SyntaxError):
                        # If we can't parse, store as string
                        value = param_value
                
                parameters.append({
                    'name': param_name,
                    'value': value,
                    'source': str(file_path),
                    'line': self._get_line_number(code, match.start()),
                    'type': 'ros_param',
                    'confidence': 0.8
                })
                confidence = max(confidence, 0.8)
        
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
                elif value_str.startswith('"') or value_str.startswith("'"):
                    # String value
                    value = value_str.strip('"\'')
                else:
                    # Try to evaluate as a number
                    value = float(value_str)
                
                parameters.append({
                    'name': name,
                    'value': value,
                    'source': str(file_path),
                    'line': self._get_line_number(code, match.start()),
                    'confidence': 0.7
                })
                confidence = max(confidence, 0.7)
            except (ValueError, SyntaxError):
                # Skip if we can't parse the value
                pass
        
        # Look for pose/position definitions
        pose_patterns = [
            r'pose\s*=\s*Pose\s*\(\s*\)\s*(?:.|\n)*?pose\.position\.x\s*=\s*(.+?)\s*(?:.|\n)*?pose\.position\.y\s*=\s*(.+?)\s*(?:.|\n)*?pose\.position\.z\s*=\s*(.+?)\s*',
            r'position\s*=\s*\[\s*(.+?)\s*,\s*(.+?)\s*,\s*(.+?)\s*\]'
        ]
        for pattern in pose_patterns:
            for match in re.finditer(pattern, code, re.DOTALL):
                try:
                    x = float(match.group(1))
                    y = float(match.group(2))
                    z = float(match.group(3))
                    
                    parameters.append({
                        'name': 'position',
                        'value': [x, y, z],
                        'source': str(file_path),
                        'line': self._get_line_number(code, match.start()),
                        'type': 'position',
                        'confidence': 0.7
                    })
                    confidence = max(confidence, 0.7)
                except (ValueError, IndexError):
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