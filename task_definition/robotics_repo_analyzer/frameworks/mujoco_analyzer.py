"""
MuJoCo-specific analyzer for robotics repositories.
"""

import logging
import re
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import xml.etree.ElementTree as ET

from task_definition.robotics_repo_analyzer.frameworks.base_analyzer import FrameworkAnalyzer

logger = logging.getLogger(__name__)

class MujocoAnalyzer(FrameworkAnalyzer):
    """
    MuJoCo-specific analyzer for robotics repositories.
    """
    
    def detect(self, code: str) -> bool:
        """
        Detect if MuJoCo is used in the code.
        
        Args:
            code: The code to analyze
            
        Returns:
            True if MuJoCo is detected, False otherwise
        """
        patterns = [
            r'import\s+mujoco',
            r'from\s+mujoco\s+import',
            r'MjModel\.from_xml',
            r'mujoco\.MjModel',
            r'mj_step',
            r'mjcf\.Physics'
        ]
        return any(re.search(pattern, code, re.MULTILINE) for pattern in patterns)
    
    def extract_robot_config(self, code: str, file_path: Path) -> Dict[str, Any]:
        """
        Extract robot configuration from MuJoCo code.
        
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
        
        # Look for MuJoCo-specific patterns
        # e.g., model = mujoco.MjModel.from_xml_path("robot.xml")
        xml_path_match = re.search(r'MjModel\.from_xml(?:_path)?\s*\(\s*[\'"](.+?)[\'"]\s*\)', code)
        if xml_path_match:
            xml_path = xml_path_match.group(1)
            config['xml_path'] = xml_path
            config['confidence'] += 0.3
            
            # Try to find and parse the XML file
            xml_file = self._find_xml_file(file_path.parent, xml_path)
            if xml_file:
                xml_config = self._extract_from_xml(xml_file)
                if xml_config:
                    # Merge XML config with higher confidence
                    for key, value in xml_config.items():
                        if key == 'confidence':
                            config['confidence'] += value
                        elif not config.get(key) and value:
                            config[key] = value
        
        # Look for joint-related code
        joint_patterns = [
            r'joint\s*=\s*mujoco\.mj_name2id\s*\(\s*model\s*,\s*mujoco\.mjtObj\.mjOBJ_JOINT\s*,\s*[\'"](.+?)[\'"]\s*\)',
            r'model\.joint\([\'"](.+?)[\'"]\)',
            r'data\.joint\([\'"](.+?)[\'"]\)'
        ]
        for pattern in joint_patterns:
            for match in re.finditer(pattern, code):
                joint_name = match.group(1)
                if joint_name not in config['joint_names']:
                    config['joint_names'].append(joint_name)
                    config['confidence'] += 0.05
        
        # Look for DOF information
        dof_patterns = [
            r'(?:dof|n_joints|num_joints|joint_num)\s*=\s*(\d+)',
            r'range\s*\(\s*(\d+)\s*\)\s*#\s*(?:joints|dof)'
        ]
        for pattern in dof_patterns:
            match = re.search(pattern, code)
            if match:
                try:
                    dof = int(match.group(1))
                    if dof > 0:
                        config['dof'] = dof
                        config['confidence'] += 0.2
                        break
                except ValueError:
                    pass
        
        return config
    
    def extract_action_sequences(self, code: str, file_path: Path) -> Dict[str, Any]:
        """
        Extract action sequences from MuJoCo code.
        
        Args:
            code: The code to analyze
            file_path: Path to the file being analyzed
            
        Returns:
            Dictionary containing action sequences
        """
        sequences = []
        confidence = 0.0
        
        # Look for common MuJoCo action patterns
        action_patterns = {
            'move_to': r'(?:set_pose|set_position|move_to)\s*\(\s*([^,]+)(?:,\s*([^,\)]+))?',
            'gripper': r'(?:set_gripper|gripper|grasp|release)\s*\(\s*([^,]+)(?:,\s*([^,\)]+))?',
            'step': r'mj_step\s*\(\s*([^,]+)(?:,\s*([^,\)]+))?',
            'reset': r'(?:reset|mj_resetData)\s*\(\s*([^,\)]+)?',
            # Add patterns for specific task functions
            'pick_and_place': r'(?:pick_and_place|pick|place)\s*\(\s*([^,\)]+)?',
            'push': r'push\s*\(\s*([^,\)]+)?',
            'slide': r'slide\s*\(\s*([^,\)]+)?',
            # Add patterns for setting twist/velocity
            'set_velocity': r'(?:self\.twist_ee|twist_ee)\s*=\s*np\.array\(\s*\[([^\]]+)\]',
            'follow_path': r'(?:follow_path|follow_trajectory|execute_path)\s*\(\s*([^,]+)(?:,\s*([^,\)]+))?'
        }
        
        # Task-related keywords for categorization
        task_categories = {
            'pick_and_place': ['pick', 'place', 'grasp', 'release', 'pick_and_place'],
            'pushing': ['push', 'pushing'],
            'sliding': ['slide', 'sliding'],
            'navigation': ['navigate', 'move_to', 'goto', 'path', 'trajectory'],
            'manipulation': ['manipulate', 'push', 'pull', 'slide', 'rotate']
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
                        # Clean up the target string to make it more readable
                        target = action_match.group(1).strip()
                        # Remove newlines and excessive whitespace
                        target = re.sub(r'\s+', ' ', target)
                        # Truncate if too long
                        if len(target) > 50:
                            target = target[:47] + "..."
                        params['target'] = target
                    
                    if len(action_match.groups()) > 1 and action_match.group(2):
                        # Clean up the value string
                        value = action_match.group(2).strip()
                        # Remove newlines and excessive whitespace
                        value = re.sub(r'\s+', ' ', value)
                        # Truncate if too long
                        if len(value) > 50:
                            value = value[:47] + "..."
                        params['value'] = value
                    
                    # Add action to list
                    actions.append({
                        'action': action_type,
                        'parameters': params,
                        'line': self._get_line_number(func_body, action_match.start()) + match.start(2),
                        'source': f"{file_path}:{self._get_line_number(func_body, action_match.start()) + match.start(2)}"
                    })
            
            # If function contains actions, organize them into task-specific sequences
            if actions:
                # Sort actions by line number
                actions.sort(key=lambda a: a['line'])
                
                # Group actions by task type
                task_groups = {}
                
                # First, check if the function name itself indicates a specific task
                func_name_lower = func_name.lower()
                assigned_to_named_task = False
                
                for task_name, keywords in task_categories.items():
                    if any(keyword in func_name_lower for keyword in keywords):
                        # This function is specifically for this task
                        task_groups[task_name] = {
                            'name': task_name,
                            'source': str(file_path),
                            'confidence': 0.8,  # Higher confidence for named tasks
                            'steps': []
                        }
                        assigned_to_named_task = True
                        break
                
                # If not assigned to a named task, create a task with the function name
                if not assigned_to_named_task:
                    task_groups[func_name] = {
                        'name': func_name,
                        'source': str(file_path),
                        'confidence': 0.7,
                        'steps': []
                    }
                
                # Now group actions by their type
                action_groups = {}
                for action in actions:
                    action_type = action['action']
                    
                    # Determine which task this action belongs to
                    assigned_task = None
                    
                    # First check if we have a named task that matches this action
                    for task_name, keywords in task_categories.items():
                        if any(keyword in action_type.lower() for keyword in keywords):
                            assigned_task = task_name
                            break
                    
                    # If not assigned to a specific task, use the function name
                    if not assigned_task and assigned_to_named_task:
                        # Use the first named task we found
                        assigned_task = next(iter(task_groups.keys()))
                    elif not assigned_task:
                        assigned_task = func_name
                    
                    # Make sure the task exists in our groups
                    if assigned_task not in task_groups:
                        task_groups[assigned_task] = {
                            'name': assigned_task,
                            'source': str(file_path),
                            'confidence': 0.7,
                            'steps': []
                        }
                    
                    # Add the action to the appropriate task
                    task_groups[assigned_task]['steps'].append(action)
                
                # Add step numbers for each task
                for task_name, task_data in task_groups.items():
                    for i, step in enumerate(task_data['steps']):
                        step['step'] = i + 1
                    
                    # Only add tasks that have steps
                    if task_data['steps']:
                        sequences.append(task_data)
                        confidence = max(confidence, task_data['confidence'])
        
        return {
            'sequences': sequences,
            'confidence': confidence
        }
    
    def extract_parameters(self, code: str, file_path: Path) -> Dict[str, Any]:
        """
        Extract parameters from MuJoCo code.
        
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
        
        # Look for position/orientation parameters
        pos_patterns = [
            r'(?:position|pos)\s*=\s*(?:np\.array\()?\s*\[([\d\.\-\+\s,]+)\]',
            r'(?:orientation|quat)\s*=\s*(?:np\.array\()?\s*\[([\d\.\-\+\s,]+)\]'
        ]
        for pattern in pos_patterns:
            for match in re.finditer(pattern, code):
                value_str = match.group(1)
                try:
                    # Parse array values
                    items = value_str.split(',')
                    value = [float(item.strip()) for item in items if item.strip()]
                    
                    # Determine parameter type
                    param_type = 'position' if 'position' in match.group(0) or 'pos' in match.group(0) else 'orientation'
                    
                    parameters.append({
                        'name': f"{param_type}_{len(parameters)}",
                        'value': value,
                        'source': str(file_path),
                        'line': self._get_line_number(code, match.start()),
                        'type': param_type,
                        'confidence': 0.7
                    })
                    confidence = max(confidence, 0.7)
                except ValueError:
                    # Skip if we can't parse the value
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
    
    def _find_xml_file(self, base_dir: Path, xml_path: str) -> Optional[Path]:
        """
        Find an XML file based on a path from code.
        
        Args:
            base_dir: Base directory to search from
            xml_path: Path to the XML file from code
            
        Returns:
            Path to the XML file if found, None otherwise
        """
        # Try direct path
        direct_path = base_dir / xml_path
        if direct_path.exists():
            return direct_path
        
        # Try with common directories
        common_dirs = ['assets', 'models', 'urdf', 'xml', 'config', 'resources']
        for common_dir in common_dirs:
            common_path = base_dir / common_dir / os.path.basename(xml_path)
            if common_path.exists():
                return common_path
        
        # Try recursive search
        for root, _, files in os.walk(base_dir):
            if os.path.basename(xml_path) in files:
                return Path(root) / os.path.basename(xml_path)
        
        return None
    
    def _extract_from_xml(self, xml_file: Path) -> Dict[str, Any]:
        """
        Extract robot configuration from an XML file.
        
        Args:
            xml_file: Path to the XML file
            
        Returns:
            Dictionary containing robot configuration
        """
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            config = {
                'name': None,
                'dof': None,
                'joint_names': [],
                'joint_limits': [],
                'joint_types': [],
                'source': str(xml_file),
                'confidence': 0.0
            }
            
            # Extract robot name
            if 'model' in root.attrib:
                config['name'] = root.attrib['model']
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
                    
                    # Check for range attribute
                    if 'range' in joint.attrib:
                        range_str = joint.attrib['range']
                        try:
                            range_parts = range_str.split()
                            if len(range_parts) >= 2:
                                lower = float(range_parts[0])
                                upper = float(range_parts[1])
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
            logger.warning(f"Error parsing XML file {xml_file}: {e}")
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