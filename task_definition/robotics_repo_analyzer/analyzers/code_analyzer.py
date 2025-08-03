"""
Code Analyzer Module

This module provides functionality to analyze Python code for robotics-related information.
"""

import ast
import logging
import re
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple, Optional, Union
import nbformat
import numpy as np

logger = logging.getLogger(__name__)

# Robotics-related function names
ROBOT_CONTROL_FUNCTIONS = {
    # Movement functions
    'move_to', 'move', 'set_position', 'set_joint_position', 'set_joint_positions',
    'set_joint_angles', 'set_pose', 'set_target', 'go_to', 'goto', 'go_home',
    'move_joints', 'move_joint', 'move_arm', 'move_gripper', 'move_base',
    'plan_motion', 'execute_motion', 'execute_trajectory', 'follow_trajectory',
    'follow_path', 'navigate_to',
    
    # Gripper functions
    'open_gripper', 'close_gripper', 'set_gripper', 'grasp', 'release',
    'pick', 'place', 'pick_and_place', 'grip', 'ungrip',
    
    # Simulation functions
    'step', 'reset', 'simulate', 'update', 'render', 'forward',
    
    # Sensor functions
    'get_state', 'get_position', 'get_joint_position', 'get_joint_positions',
    'get_joint_angles', 'get_pose', 'get_observation', 'observe',
    'read_sensors', 'get_sensor_data',
    
    # Control functions
    'set_control', 'set_velocity', 'set_torque', 'set_force',
    'apply_action', 'apply_control', 'apply_torque', 'apply_force',
    'control', 'pid_control', 'inverse_kinematics', 'forward_kinematics',
    'compute_jacobian', 'solve_ik'
}

# Robot initialization patterns
ROBOT_INIT_PATTERNS = [
    r'(?:Robot|Arm|Manipulator|Controller|Environment|Env)\s*\(',
    r'(?:load|load_robot|load_urdf|load_model|from_urdf|from_xml)\s*\(',
    r'(?:MjModel|mjcf\.Physics|Physics)\.from_(?:xml|path)',
    r'gym\.make\s*\(\s*[\'"](?:\w+)?(?:Robot|Env|Manipulator)',
    r'p\.loadURDF\s*\(',
    r'loadModel\s*\(',
    r'(?:init|initialize|setup)_robot\s*\('
]

class RoboticsASTVisitor(ast.NodeVisitor):
    """AST visitor for extracting robotics-related information from Python code."""
    
    def __init__(self):
        self.constants = []
        self.functions = []
        self.classes = []
        self.imports = []
        self.function_calls = []
        self.assignments = []
        self.has_main = False
        self.has_main_function = False
        self.robot_init_calls = []
        self.current_function = None
        self.current_class = None
        self.control_flow = []
        
    def visit_Import(self, node):
        """Visit an Import node."""
        for name in node.names:
            self.imports.append(name.name)
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node):
        """Visit an ImportFrom node."""
        if node.module:
            self.imports.append(node.module)
        self.generic_visit(node)
    
    def visit_ClassDef(self, node):
        """Visit a ClassDef node."""
        class_info = {
            'name': node.name,
            'bases': [base.id if isinstance(base, ast.Name) else '' for base in node.bases],
            'methods': [],
            'lineno': node.lineno
        }
        
        old_class = self.current_class
        self.current_class = class_info
        self.classes.append(class_info)
        
        self.generic_visit(node)
        self.current_class = old_class
    
    def visit_FunctionDef(self, node):
        """Visit a FunctionDef node."""
        function_info = {
            'name': node.name,
            'args': [arg.arg for arg in node.args.args],
            'calls': [],
            'assignments': [],
            'lineno': node.lineno,
            'class': self.current_class['name'] if self.current_class else None
        }
        
        if node.name == 'main':
            self.has_main_function = True
        
        old_function = self.current_function
        self.current_function = function_info
        
        if self.current_class:
            self.current_class['methods'].append(function_info)
        else:
            self.functions.append(function_info)
        
        self.generic_visit(node)
        self.current_function = old_function
    
    def visit_Assign(self, node):
        """Visit an Assign node."""
        for target in node.targets:
            if isinstance(target, ast.Name):
                # Check if it's a constant (uppercase name)
                if target.id.isupper():
                    value = self._extract_value(node.value)
                    if value is not None:
                        constant_info = {
                            'name': target.id,
                            'value': value,
                            'lineno': node.lineno
                        }
                        self.constants.append(constant_info)
                
                # Record all assignments
                assignment_info = {
                    'target': target.id,
                    'value': self._extract_value(node.value),
                    'lineno': node.lineno,
                    'function': self.current_function['name'] if self.current_function else None,
                    'class': self.current_class['name'] if self.current_class else None
                }
                self.assignments.append(assignment_info)
                
                if self.current_function:
                    self.current_function['assignments'].append(assignment_info)
        
        self.generic_visit(node)
    
    def visit_Call(self, node):
        """Visit a Call node."""
        func_name = self._get_call_name(node.func)
        args = [self._extract_value(arg) for arg in node.args]
        keywords = {kw.arg: self._extract_value(kw.value) for kw in node.keywords if kw.arg}
        
        call_info = {
            'name': func_name,
            'args': args,
            'keywords': keywords,
            'lineno': node.lineno,
            'function': self.current_function['name'] if self.current_function else None,
            'class': self.current_class['name'] if self.current_class else None
        }
        
        self.function_calls.append(call_info)
        
        if self.current_function:
            self.current_function['calls'].append(call_info)
        
        # Check for robot initialization
        source = self._get_source_segment(node)
        if source:
            for pattern in ROBOT_INIT_PATTERNS:
                if re.search(pattern, source):
                    self.robot_init_calls.append(call_info)
                    break
        
        self.generic_visit(node)
    
    def visit_If(self, node):
        """Visit an If node."""
        # Check for if __name__ == "__main__"
        if (isinstance(node.test, ast.Compare) and
            isinstance(node.test.left, ast.Name) and
            node.test.left.id == '__name__' and
            len(node.test.ops) == 1 and
            isinstance(node.test.ops[0], ast.Eq) and
            len(node.test.comparators) == 1 and
            isinstance(node.test.comparators[0], ast.Constant) and
            node.test.comparators[0].value == '__main__'):
            self.has_main = True
            
            # Record control flow in main block
            for stmt in node.body:
                if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                    call_name = self._get_call_name(stmt.value.func)
                    self.control_flow.append({
                        'type': 'call',
                        'name': call_name,
                        'lineno': stmt.lineno,
                        'context': 'main'
                    })
        
        self.generic_visit(node)
    
    def visit_For(self, node):
        """Visit a For node."""
        # Record loop information
        loop_info = {
            'type': 'for',
            'target': self._get_name(node.target),
            'lineno': node.lineno,
            'function': self.current_function['name'] if self.current_function else None,
            'class': self.current_class['name'] if self.current_class else None,
            'body_calls': []
        }
        
        # Extract calls in the loop body
        for stmt in node.body:
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                call_name = self._get_call_name(stmt.value.func)
                loop_info['body_calls'].append(call_name)
        
        self.control_flow.append(loop_info)
        self.generic_visit(node)
    
    def visit_While(self, node):
        """Visit a While node."""
        # Record loop information
        loop_info = {
            'type': 'while',
            'lineno': node.lineno,
            'function': self.current_function['name'] if self.current_function else None,
            'class': self.current_class['name'] if self.current_class else None,
            'body_calls': []
        }
        
        # Extract calls in the loop body
        for stmt in node.body:
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                call_name = self._get_call_name(stmt.value.func)
                loop_info['body_calls'].append(call_name)
        
        self.control_flow.append(loop_info)
        self.generic_visit(node)
    
    def _get_call_name(self, node):
        """Get the name of a function call."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_call_name(node.value)}.{node.attr}"
        return "unknown"
    
    def _get_name(self, node):
        """Get the name of a node."""
        if isinstance(node, ast.Name):
            return node.id
        return "unknown"
    
    def _extract_value(self, node):
        """Extract a Python value from an AST node."""
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.List):
            return [self._extract_value(elt) for elt in node.elts]
        elif isinstance(node, ast.Tuple):
            return tuple(self._extract_value(elt) for elt in node.elts)
        elif isinstance(node, ast.Dict):
            return {self._extract_value(k): self._extract_value(v) 
                   for k, v in zip(node.keys, node.values)}
        elif isinstance(node, ast.Name):
            return node.id  # Return variable name as string
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            # Handle negative numbers
            value = self._extract_value(node.operand)
            if isinstance(value, (int, float)):
                return -value
        elif isinstance(node, ast.BinOp):
            # Handle simple binary operations
            left = self._extract_value(node.left)
            right = self._extract_value(node.right)
            if isinstance(left, (int, float)) and isinstance(right, (int, float)):
                if isinstance(node.op, ast.Add):
                    return left + right
                elif isinstance(node.op, ast.Sub):
                    return left - right
                elif isinstance(node.op, ast.Mult):
                    return left * right
                elif isinstance(node.op, ast.Div):
                    return left / right
        return None
    
    def _get_source_segment(self, node):
        """Get the source code segment for a node."""
        # This is a placeholder - in a real implementation, you would
        # need to have access to the source code to extract segments
        return None


def analyze_python_file(file_path: Path) -> Dict[str, Any]:
    """
    Analyze a Python file for robotics-related information.
    
    Args:
        file_path: Path to the Python file
        
    Returns:
        Dictionary containing analysis results
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()
        
        tree = ast.parse(source)
        visitor = RoboticsASTVisitor()
        visitor.visit(tree)
        
        # Extract robotics-related patterns
        patterns = []
        for pattern in ROBOT_INIT_PATTERNS:
            if re.search(pattern, source):
                patterns.append(pattern)
        
        # Check for robotics-related function calls
        robotics_calls = []
        for call in visitor.function_calls:
            call_name = call['name'].split('.')[-1]  # Get the method name
            if call_name in ROBOT_CONTROL_FUNCTIONS:
                robotics_calls.append(call)
        
        return {
            'constants': visitor.constants,
            'functions': visitor.functions,
            'classes': visitor.classes,
            'imports': visitor.imports,
            'function_calls': visitor.function_calls,
            'robotics_calls': robotics_calls,
            'assignments': visitor.assignments,
            'has_main': visitor.has_main,
            'has_main_function': visitor.has_main_function,
            'robot_init_calls': visitor.robot_init_calls,
            'control_flow': visitor.control_flow,
            'patterns': patterns,
            'file_path': str(file_path)
        }
    except Exception as e:
        logger.error(f"Error analyzing Python file {file_path}: {e}")
        return {
            'error': str(e),
            'file_path': str(file_path)
        }


def extract_robot_specs_from_python(file_path: Path) -> Dict[str, Any]:
    """
    Extract robot specifications from a Python file.
    
    Args:
        file_path: Path to the Python file
        
    Returns:
        Dictionary containing robot specifications
    """
    analysis = analyze_python_file(file_path)
    
    # Initialize robot specs
    robot_specs = {
        'name': None,
        'dof': None,
        'joint_names': [],
        'joint_limits': [],
        'joint_types': [],
        'confidence': 0.0
    }
    
    # Extract information from constants
    for constant in analysis.get('constants', []):
        name = constant.get('name', '').lower()
        value = constant.get('value')
        
        if value is None:
            continue
            
        # DOF/number of joints
        if any(keyword in name for keyword in ['dof', 'n_joints', 'num_joints', 'joint_num']):
            if isinstance(value, int) and value > 0:
                robot_specs['dof'] = value
                robot_specs['confidence'] += 0.2
        
        # Joint names
        elif any(keyword in name for keyword in ['joint_names', 'joint_name', 'joints']):
            if isinstance(value, (list, tuple)) and all(isinstance(v, str) for v in value):
                robot_specs['joint_names'] = value
                robot_specs['confidence'] += 0.2
        
        # Joint limits
        elif any(keyword in name for keyword in ['joint_limits', 'limits', 'joint_range']):
            if isinstance(value, (list, tuple)):
                # Check if it's a list of tuples/lists (lower, upper)
                if all(isinstance(v, (list, tuple)) and len(v) == 2 for v in value):
                    robot_specs['joint_limits'] = value
                    robot_specs['confidence'] += 0.2
    
    # Extract information from robot initialization calls
    for call in analysis.get('robot_init_calls', []):
        # Check for robot name
        if not robot_specs['name']:
            for kw, value in call.get('keywords', {}).items():
                if kw in ['name', 'robot_name'] and isinstance(value, str):
                    robot_specs['name'] = value
                    robot_specs['confidence'] += 0.1
        
        # Check for DOF
        if not robot_specs['dof']:
            for kw, value in call.get('keywords', {}).items():
                if kw in ['dof', 'n_joints', 'num_joints'] and isinstance(value, int) and value > 0:
                    robot_specs['dof'] = value
                    robot_specs['confidence'] += 0.2
    
    # If we found some information, return it
    if robot_specs['confidence'] > 0:
        return robot_specs
    
    return None


def extract_action_sequences_from_python(file_path: Path) -> List[Dict[str, Any]]:
    """
    Extract action sequences from a Python file.
    
    Args:
        file_path: Path to the Python file
        
    Returns:
        List of dictionaries containing action sequences
    """
    analysis = analyze_python_file(file_path)
    sequences = []
    
    # Look for sequences in functions
    for function in analysis.get('functions', []):
        # Skip utility functions
        if function['name'].startswith('_'):
            continue
            
        # Check if the function contains robotics-related calls
        robotics_calls = [call for call in function.get('calls', [])
                         if call['name'].split('.')[-1] in ROBOT_CONTROL_FUNCTIONS]
        
        if robotics_calls:
            sequence = {
                'name': function['name'],
                'steps': [],
                'confidence': 0.0,
                'source_line': function.get('lineno')
            }
            
            # Add steps from calls
            for i, call in enumerate(robotics_calls):
                step = {
                    'step': i + 1,
                    'action': call['name'].split('.')[-1],
                    'parameters': {
                        'args': call.get('args', []),
                        'keywords': call.get('keywords', {})
                    },
                    'line': call.get('lineno')
                }
                sequence['steps'].append(step)
            
            # Calculate confidence based on number of steps and function name
            num_steps = len(sequence['steps'])
            if num_steps > 0:
                # Base confidence on number of steps
                sequence['confidence'] = min(0.3 + 0.1 * num_steps, 0.8)
                
                # Boost confidence for functions with descriptive names
                name = function['name'].lower()
                if any(keyword in name for keyword in 
                       ['task', 'action', 'sequence', 'routine', 'procedure', 'demo']):
                    sequence['confidence'] += 0.1
                
                # Boost confidence for functions with movement-related names
                if any(keyword in name for keyword in 
                       ['move', 'pick', 'place', 'grasp', 'navigate', 'go', 'run']):
                    sequence['confidence'] += 0.1
                
                sequences.append(sequence)
    
    # Look for sequences in classes
    for class_info in analysis.get('classes', []):
        # Check if it's a robot controller class
        class_name = class_info['name'].lower()
        is_controller = any(keyword in class_name for keyword in 
                           ['robot', 'controller', 'control', 'arm', 'manipulator', 'env'])
        
        if is_controller:
            # Look for sequences in methods
            for method in class_info.get('methods', []):
                # Skip utility methods
                if method['name'].startswith('_'):
                    continue
                    
                # Check if the method contains robotics-related calls
                robotics_calls = [call for call in method.get('calls', [])
                                 if call['name'].split('.')[-1] in ROBOT_CONTROL_FUNCTIONS]
                
                if robotics_calls:
                    sequence = {
                        'name': f"{class_info['name']}.{method['name']}",
                        'steps': [],
                        'confidence': 0.0,
                        'source_line': method.get('lineno')
                    }
                    
                    # Add steps from calls
                    for i, call in enumerate(robotics_calls):
                        step = {
                            'step': i + 1,
                            'action': call['name'].split('.')[-1],
                            'parameters': {
                                'args': call.get('args', []),
                                'keywords': call.get('keywords', {})
                            },
                            'line': call.get('lineno')
                        }
                        sequence['steps'].append(step)
                    
                    # Calculate confidence based on number of steps and method name
                    num_steps = len(sequence['steps'])
                    if num_steps > 0:
                        # Base confidence on number of steps
                        sequence['confidence'] = min(0.4 + 0.1 * num_steps, 0.9)  # Higher base for methods
                        
                        # Boost confidence for methods with descriptive names
                        name = method['name'].lower()
                        if any(keyword in name for keyword in 
                               ['task', 'action', 'sequence', 'routine', 'procedure', 'demo']):
                            sequence['confidence'] += 0.1
                        
                        # Boost confidence for methods with movement-related names
                        if any(keyword in name for keyword in 
                               ['move', 'pick', 'place', 'grasp', 'navigate', 'go', 'run']):
                            sequence['confidence'] += 0.1
                        
                        sequences.append(sequence)
    
    # Look for sequences in control flow (loops, main block)
    for flow in analysis.get('control_flow', []):
        if flow.get('type') in ['for', 'while'] and flow.get('body_calls'):
            # Check if the loop contains robotics-related calls
            robotics_calls = [call for call in flow.get('body_calls', [])
                             if call in ROBOT_CONTROL_FUNCTIONS]
            
            if robotics_calls:
                sequence = {
                    'name': f"Loop at line {flow.get('lineno')}",
                    'steps': [],
                    'confidence': 0.0,
                    'source_line': flow.get('lineno')
                }
                
                # Add steps from calls
                for i, call in enumerate(robotics_calls):
                    step = {
                        'step': i + 1,
                        'action': call,
                        'parameters': {},  # Can't extract parameters from control_flow
                        'line': flow.get('lineno')
                    }
                    sequence['steps'].append(step)
                
                # Calculate confidence based on number of steps
                num_steps = len(sequence['steps'])
                if num_steps > 0:
                    # Lower confidence for loops (often not sequential tasks)
                    sequence['confidence'] = min(0.2 + 0.05 * num_steps, 0.5)
                    sequences.append(sequence)
    
    return sequences


def extract_action_sequences_from_notebook(file_path: Path) -> List[Dict[str, Any]]:
    """
    Extract action sequences from a Jupyter notebook.
    
    Args:
        file_path: Path to the notebook file
        
    Returns:
        List of dictionaries containing action sequences
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
        
        # Extract code cells
        code_cells = [cell for cell in nb.cells if cell.cell_type == 'code']
        
        # Look for sequences in consecutive cells
        sequences = []
        current_sequence = {
            'name': f"Notebook sequence in {file_path.name}",
            'steps': [],
            'confidence': 0.0,
            'source': str(file_path)
        }
        
        for i, cell in enumerate(code_cells):
            try:
                # Parse the cell
                tree = ast.parse(cell.source)
                visitor = RoboticsASTVisitor()
                visitor.visit(tree)
                
                # Check for robotics-related calls
                for call in visitor.function_calls:
                    call_name = call['name'].split('.')[-1]  # Get the method name
                    if call_name in ROBOT_CONTROL_FUNCTIONS:
                        step = {
                            'step': len(current_sequence['steps']) + 1,
                            'action': call_name,
                            'parameters': {
                                'args': call.get('args', []),
                                'keywords': call.get('keywords', {})
                            },
                            'cell': i
                        }
                        current_sequence['steps'].append(step)
            except SyntaxError:
                # Skip cells with syntax errors
                continue
        
        # If we found steps, add the sequence
        if current_sequence['steps']:
            # Calculate confidence based on number of steps
            num_steps = len(current_sequence['steps'])
            current_sequence['confidence'] = min(0.3 + 0.05 * num_steps, 0.7)
            sequences.append(current_sequence)
        
        return sequences
    except Exception as e:
        logger.error(f"Error extracting action sequences from notebook {file_path}: {e}")
        return []


def extract_parameters_from_python(file_path: Path) -> List[Dict[str, Any]]:
    """
    Extract parameters from a Python file.
    
    Args:
        file_path: Path to the Python file
        
    Returns:
        List of dictionaries containing parameters
    """
    analysis = analyze_python_file(file_path)
    parameters = []
    
    # Extract parameters from constants
    for constant in analysis.get('constants', []):
        name = constant.get('name')
        value = constant.get('value')
        
        if name and value is not None:
            # Skip non-numeric and non-array values
            if not isinstance(value, (int, float, list, tuple, dict)):
                continue
                
            # Skip very small or very large values (likely not physical parameters)
            if isinstance(value, (int, float)) and (abs(value) < 1e-10 or abs(value) > 1e10):
                continue
                
            # For arrays, check if they contain numeric values or variable names
            if isinstance(value, (list, tuple)):
                if not all(isinstance(v, (int, float, str)) for v in value):
                    continue
                    
                # Skip very long arrays (likely not parameters)
                if len(value) > 100:
                    continue
            
            parameter = {
                'name': name,
                'value': value,
                'line': constant.get('lineno'),
                'confidence': 0.7  # High confidence for constants
            }
            parameters.append(parameter)
    
    # Extract parameters from assignments in functions
    for assignment in analysis.get('assignments', []):
        # Skip assignments in utility functions
        function_name = assignment.get('function')
        if function_name and function_name.startswith('_'):
            continue
            
        name = assignment.get('target')
        value = assignment.get('value')
        
        if name and value is not None:
            # Skip non-numeric and non-array values
            if not isinstance(value, (int, float, list, tuple, dict)):
                continue
                
            # Skip very small or very large values (likely not physical parameters)
            if isinstance(value, (int, float)) and (abs(value) < 1e-10 or abs(value) > 1e10):
                continue
                
            # For arrays, check if they contain numeric values or variable names
            if isinstance(value, (list, tuple)):
                if not all(isinstance(v, (int, float, str)) for v in value):
                    continue
                    
                # Skip very long arrays (likely not parameters)
                if len(value) > 100:
                    continue
            
            # Check if the name suggests it's a parameter
            is_parameter = any(keyword in name.lower() for keyword in 
                              ['pos', 'position', 'angle', 'joint', 'limit', 'gain',
                               'vel', 'velocity', 'accel', 'acceleration', 'force',
                               'torque', 'mass', 'inertia', 'damping', 'stiffness',
                               'friction', 'threshold', 'target', 'goal', 'home',
                               'config', 'param', 'setting', 'constant'])
            
            if is_parameter:
                parameter = {
                    'name': name,
                    'value': value,
                    'line': assignment.get('lineno'),
                    'function': assignment.get('function'),
                    'class': assignment.get('class'),
                    'confidence': 0.5  # Medium confidence for assignments
                }
                parameters.append(parameter)
    
    # Extract parameters from function calls
    for call in analysis.get('function_calls', []):
        call_name = call['name'].split('.')[-1]  # Get the method name
        
        # Check if it's a robotics-related function
        if call_name in ROBOT_CONTROL_FUNCTIONS:
            # Extract parameters from arguments
            args = call.get('args', [])
            keywords = call.get('keywords', {})
            
            # Check positional arguments
            for i, arg in enumerate(args):
                if isinstance(arg, (list, tuple)) and all(isinstance(v, (int, float, str)) for v in arg):
                    # Infer parameter name from function name and position
                    param_name = None
                    if call_name in ['move_to', 'set_position', 'go_to'] and i == 0:
                        param_name = f"{call_name}_position"
                    elif call_name in ['set_joint_position', 'set_joint_positions'] and i == 0:
                        param_name = 'joint_position'
                    
                    if param_name:
                        parameter = {
                            'name': param_name,
                            'value': arg,
                            'line': call.get('lineno'),
                            'function': call.get('function'),
                            'class': call.get('class'),
                            'confidence': 0.4  # Lower confidence for inferred parameters
                        }
                        parameters.append(parameter)
            
            # Check keyword arguments
            for kw, value in keywords.items():
                if isinstance(value, (int, float, list, tuple)):
                    if isinstance(value, (list, tuple)) and not all(isinstance(v, (int, float, str)) for v in value):
                        continue
                    
                    parameter = {
                        'name': kw,
                        'value': value,
                        'line': call.get('lineno'),
                        'function': call.get('function'),
                        'class': call.get('class'),
                        'confidence': 0.6  # Medium-high confidence for named parameters
                    }
                    parameters.append(parameter)
    
    return parameters