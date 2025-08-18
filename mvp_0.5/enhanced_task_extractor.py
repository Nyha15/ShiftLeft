#!/usr/bin/env python3
"""
Enhanced Task Extractor
=======================

Produces production-ready YAMLs with semantic descriptions, parameters (with Sobol ranges),
KPIs, and function names for robotics tasks.
"""

import re
import ast
import logging
import yaml
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, asdict
import inspect
import importlib.util

logger = logging.getLogger(__name__)

@dataclass
class TaskParameter:
    """Task parameter with Sobol analysis range"""
    variable_name: str
    current_value: float
    range_min: float
    range_max: float
    unit: str
    description: str
    source: str  # 'code', 'config', 'urdf', 'user'

@dataclass
class TaskKPI:
    """Key Performance Indicator for task monitoring"""
    name: str
    data_type: str  # 'bool', 'float', 'int', 'string'
    description: str
    success_criteria: Optional[str] = None
    monitoring_frequency: str = "continuous"

@dataclass
class EnhancedTaskInfo:
    """Enhanced task information for production-ready YAMLs"""
    task_name: str
    semantic_description: str
    parameters: List[TaskParameter]
    kpis: List[TaskKPI]
    function_names: List[str]
    file_path: str
    confidence: float

class EnhancedTaskExtractor:
    """Enhanced task extractor for production-ready YAML generation"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.EnhancedTaskExtractor")
        
        # Robotics task patterns for better identification
        self.task_patterns = {
            'manipulation': [
                r'pick.*up', r'place.*down', r'grasp', r'release', r'grip',
                r'manipulat', r'move.*object', r'transfer', r'handle',
                r'insert', r'remove', r'rotate', r'push', r'pull'
            ],
            'navigation': [
                r'navigate', r'move.*to', r'go.*to', r'path.*plan', r'avoid.*obstacle',
                r'locali[sz]', r'slam', r'waypoint', r'trajectory'
            ],
            'grasping': [
                r'grasp', r'grip', r'hold', r'clamp', r'pinch', r'squeeze',
                r'finger.*control', r'end.*effector'
            ],
            'perception': [
                r'detect', r'recogni[sz]e', r'identify', r'classify', r'segment',
                r'vision', r'camera', r'sensor.*data', r'point.*cloud'
            ],
            'control': [
                r'control', r'servo', r'pid', r'feedback', r'command',
                r'joint.*control', r'motor.*control', r'actuator'
            ],
            'motion_planning': [
                r'plan', r'path.*plan', r'trajectory.*plan', r'motion.*plan',
                r'rrt', r'prm', r'apf', r'potential.*field', r'collision.*avoid'
            ],
            'kinematics': [
                r'kinematics', r'jacobian', r'inverse.*kin', r'forward.*kin',
                r'joint.*space', r'cartesian.*space', r'end.*effector'
            ],
            'dynamics': [
                r'dynamics', r'torque', r'force', r'inertia', r'mass',
                r'gravity', r'friction', r'damping', r'stiffness'
            ]
        }
        
        # Common robotics parameters with typical ranges
        self.parameter_ranges = {
            'friction': (0.05, 1.0),
            'mass': (0.8, 1.2),
            'inertia': (0.8, 1.2),
            'damping': (0.5, 2.0),
            'stiffness': (0.5, 2.0),
            'gains': (0.75, 1.25),
            'noise_std': (1.0, 3.0),
            'joint_limits': (0.9, 1.1),
            'velocity_limits': (0.8, 1.2),
            'acceleration_limits': (0.7, 1.3),
            'position_tolerance': (0.001, 0.01),
            'orientation_tolerance': (0.01, 0.1),
            'force_threshold': (0.5, 2.0),
            'torque_threshold': (0.5, 2.0),
            'alpha': (0.5, 2.0),  # APF attractive force gain
            'beta': (0.5, 2.0),   # APF repulsive force gain
            'rho': (0.8, 1.5),    # APF influence radius
            'k': (0.5, 2.0),      # Proportional gain
            'd': (0.5, 2.0),      # Derivative gain
            'max_iterations': (0.8, 1.2),  # Iteration limits
            'resolution': (0.5, 2.0),      # Solution resolution
            'epsilon': (0.001, 0.01)       # Convergence tolerance
        }
    
    def extract_enhanced_tasks(self, repo_path: Path) -> List[EnhancedTaskInfo]:
        """Extract enhanced task information from repository"""
        tasks = []
        python_files = list(repo_path.rglob("*.py"))
        
        self.logger.info(f"Analyzing {len(python_files)} Python files for enhanced tasks")
        
        for file_path in python_files:
            try:
                file_tasks = self._analyze_file_enhanced(file_path)
                tasks.extend(file_tasks)
            except Exception as e:
                self.logger.warning(f"Failed to analyze {file_path}: {e}")
        
        # Deduplicate and sort by confidence
        deduplicated_tasks = self._deduplicate_enhanced_tasks(tasks)
        self.logger.info(f"Found {len(deduplicated_tasks)} unique enhanced tasks")
        return deduplicated_tasks
    
    def _analyze_file_enhanced(self, file_path: Path) -> List[EnhancedTaskInfo]:
        """Analyze a single Python file for enhanced task information"""
        try:
            content = file_path.read_text(encoding='utf-8', errors='ignore')
            tree = ast.parse(content)
            
            tasks = []
            
            # Extract global parameters first
            global_params = self._extract_global_parameters(tree, content)
            
            # Analyze functions and classes
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    task = self._analyze_function_enhanced(node, content, file_path, global_params)
                    if task:
                        tasks.append(task)
                elif isinstance(node, ast.ClassDef):
                    task = self._analyze_class_enhanced(node, content, file_path, global_params)
                    if task:
                        tasks.append(task)
            
            return tasks
            
        except Exception as e:
            self.logger.warning(f"Failed to parse {file_path}: {e}")
            return []
    
    def _analyze_function_enhanced(self, func_node: ast.FunctionDef, 
                                 content: str, file_path: Path, global_params: Dict[str, float] = None) -> Optional[EnhancedTaskInfo]:
        """Analyze function for enhanced task information"""
        # Check if function is robotics-related
        if not self._is_robotics_function(func_node, content):
            return None
        
        # Extract semantic description
        semantic_description = self._extract_semantic_description(func_node, content)
        
        # Extract parameters
        parameters = self._extract_function_parameters(func_node, content, global_params)
        
        # Extract KPIs
        kpis = self._extract_function_kpis(func_node, content)
        
        # Extract function names
        function_names = [func_node.name]
        
        # Calculate confidence
        confidence = self._calculate_function_confidence(func_node, content)
        
        if confidence < 0.3:  # Only include if reasonably confident
            return None
        
        return EnhancedTaskInfo(
            task_name=func_node.name,
            semantic_description=semantic_description,
            parameters=parameters,
            kpis=kpis,
            function_names=function_names,
            file_path=str(file_path),
            confidence=confidence
        )
    
    def _analyze_class_enhanced(self, class_node: ast.ClassDef, 
                               content: str, file_path: Path, global_params: Dict[str, float] = None) -> Optional[EnhancedTaskInfo]:
        """Analyze class for enhanced task information"""
        # Check if class is robotics-related
        if not self._is_robotics_class(class_node, content):
            return None
        
        # Extract semantic description
        semantic_description = self._extract_class_description(class_node, content)
        
        # Extract parameters from class attributes and methods
        parameters = self._extract_class_parameters(class_node, content, global_params)
        
        # Extract KPIs from class methods
        kpis = self._extract_class_kpis(class_node, content)
        
        # Extract function names (methods)
        function_names = [method.name for method in class_node.body 
                         if isinstance(method, ast.FunctionDef)]
        
        # Calculate confidence
        confidence = self._calculate_class_confidence(class_node, content)
        
        if confidence < 0.3:
            return None
        
        return EnhancedTaskInfo(
            task_name=class_node.name,
            semantic_description=semantic_description,
            parameters=parameters,
            kpis=kpis,
            function_names=function_names,
            file_path=str(file_path),
            confidence=confidence
        )
    
    def _is_robotics_function(self, func_node: ast.FunctionDef, content: str) -> bool:
        """Check if function is robotics-related"""
        # Check function name
        func_name = func_node.name.lower()
        for pattern_list in self.task_patterns.values():
            for pattern in pattern_list:
                if re.search(pattern, func_name, re.IGNORECASE):
                    return True
        
        # Check docstring and comments
        func_content = self._get_function_content(func_node, content)
        robotics_keywords = ['robot', 'arm', 'gripper', 'joint', 'link', 'actuator', 
                           'kinematics', 'dynamics', 'trajectory', 'mujoco', 'pybullet']
        
        for keyword in robotics_keywords:
            if keyword in func_content.lower():
                return True
        
        return False
    
    def _is_robotics_class(self, class_node: ast.ClassDef, content: str) -> bool:
        """Check if class is robotics-related"""
        # Check class name
        class_name = class_node.name.lower()
        if any(keyword in class_name for keyword in ['robot', 'arm', 'gripper', 'task', 'controller', 'potential', 'field', 'apf']):
            return True
        
        # Check docstring and comments
        class_content = self._get_class_content(class_node, content)
        robotics_keywords = ['robot', 'arm', 'gripper', 'joint', 'link', 'actuator', 
                           'kinematics', 'dynamics', 'trajectory', 'mujoco', 'pybullet',
                           'potential', 'field', 'plan', 'path', 'obstacle', 'attractive', 'repulsive']
        
        for keyword in robotics_keywords:
            if keyword in class_content.lower():
                return True
        
        return False
    
    def _extract_semantic_description(self, func_node: ast.FunctionDef, content: str) -> str:
        """Extract semantic description from function"""
        # Try to get docstring
        if ast.get_docstring(func_node):
            return ast.get_docstring(func_node).strip()
        
        # Try to get comments above function
        func_start = func_node.lineno - 1
        lines = content.split('\n')
        
        description_lines = []
        for i in range(func_start - 1, max(0, func_start - 5), -1):
            line = lines[i].strip()
            if line.startswith('#'):
                description_lines.insert(0, line[1:].strip())
            elif line:
                break
        
        if description_lines:
            return ' '.join(description_lines)
        
        # Generate description from function name
        return f"Performs {func_node.name.replace('_', ' ')} operation"
    
    def _extract_class_description(self, class_node: ast.ClassDef, content: str) -> str:
        """Extract semantic description from class"""
        # Try to get docstring
        if ast.get_docstring(class_node):
            return ast.get_docstring(class_node).strip()
        
        # Try to get comments above class
        class_start = class_node.lineno - 1
        lines = content.split('\n')
        
        description_lines = []
        for i in range(class_start - 1, max(0, class_start - 5), -1):
            line = lines[i].strip()
            if line.startswith('#'):
                description_lines.insert(0, line[1:].strip())
            elif line:
                break
        
        if description_lines:
            return ' '.join(description_lines)
        
        # Generate description from class name
        return f"Manages {class_node.name.replace('_', ' ')} operations"
    
    def _extract_function_parameters(self, func_node: ast.FunctionDef, content: str, global_params: Dict[str, float] = None) -> List[TaskParameter]:
        """Extract parameters from function with Sobol ranges"""
        parameters = []
        
        # Extract default arguments
        for arg, default in zip(func_node.args.args, func_node.args.defaults):
            if arg.arg != 'self':
                param_name = arg.arg
                current_value = self._extract_default_value(default)
                
                # Determine parameter type and range
                param_type = self._infer_parameter_type(param_name, current_value)
                range_min, range_max = self._get_parameter_range(param_name, current_value, param_type)
                
                # Determine unit
                unit = self._infer_parameter_unit(param_name, param_type)
                
                # Determine source
                source = 'code'
                
                # Create parameter
                param = TaskParameter(
                    variable_name=param_name,
                    current_value=current_value,
                    range_min=range_min,
                    range_max=range_max,
                    unit=unit,
                    description=f"Parameter {param_name} for {func_node.name}",
                    source=source
                )
                parameters.append(param)
        
        # Add relevant global parameters
        if global_params:
            for param_name, value in global_params.items():
                # Check if this global parameter is relevant to this function
                if self._is_parameter_relevant_to_function(param_name, func_node, content):
                    param_type = self._infer_parameter_type(param_name, value)
                    range_min, range_max = self._get_parameter_range(param_name, value, param_type)
                    unit = self._infer_parameter_unit(param_name, param_type)
                    
                    param = TaskParameter(
                        variable_name=param_name,
                        current_value=value,
                        range_min=range_min,
                        range_max=range_max,
                        unit=unit,
                        description=f"Global parameter {param_name} used by {func_node.name}",
                        source='global'
                    )
                    parameters.append(param)
        
        return parameters
    
    def _extract_class_parameters(self, class_node: ast.ClassDef, content: str, global_params: Dict[str, float] = None) -> List[TaskParameter]:
        """Extract parameters from class attributes and constructor"""
        parameters = []
        
        # Look for class attributes and constructor parameters
        for item in class_node.body:
            if isinstance(item, ast.Assign):
                for target in item.targets:
                    if isinstance(target, ast.Name):
                        param_name = target.id
                        current_value = self._extract_assignment_value(item.value)
                        
                        if current_value is not None:
                            param_type = self._infer_parameter_type(param_name, current_value)
                            range_min, range_max = self._get_parameter_range(param_name, current_value, param_type)
                            unit = self._infer_parameter_unit(param_name, param_type)
                            
                            param = TaskParameter(
                                variable_name=param_name,
                                current_value=current_value,
                                range_min=range_min,
                                range_max=range_max,
                                unit=unit,
                                description=f"Class parameter {param_name}",
                                source='code'
                            )
                            parameters.append(param)
        
        # Extract constructor parameters
        for item in class_node.body:
            if isinstance(item, ast.FunctionDef) and item.name == '__init__':
                # Extract default arguments from constructor
                defaults = item.args.defaults
                args = item.args.args[1:]  # Skip 'self'
                
                # Handle case where there are more args than defaults
                if len(args) >= len(defaults):
                    for i, arg in enumerate(args):
                        param_name = arg.arg
                        if i < len(defaults):
                            current_value = self._extract_default_value(defaults[i])
                        else:
                            # No default value, use a reasonable default
                            current_value = 0.0
                        
                        param_type = self._infer_parameter_type(param_name, current_value)
                        range_min, range_max = self._get_parameter_range(param_name, current_value, param_type)
                        unit = self._infer_parameter_unit(param_name, param_type)
                        
                        param = TaskParameter(
                            variable_name=param_name,
                            current_value=current_value,
                            range_min=range_min,
                            range_max=range_max,
                            unit=unit,
                            description=f"Constructor parameter {param_name}",
                            source='constructor'
                        )
                        parameters.append(param)
        
        return parameters
    
    def _extract_function_kpis(self, func_node: ast.FunctionDef, content: str) -> List[TaskKPI]:
        """Extract KPIs from function"""
        kpis = []
        
        # Look for return statements and success indicators
        func_content = self._get_function_content(func_node, content)
        
        # Common KPI patterns
        kpi_patterns = [
            (r'success', 'bool', 'Task completion status'),
            (r'error', 'string', 'Error message if task fails'),
            (r'status', 'string', 'Current task status'),
            (r'progress', 'float', 'Task completion progress (0-1)'),
            (r'position', 'float', 'Current position'),
            (r'velocity', 'float', 'Current velocity'),
            (r'force', 'float', 'Applied force'),
            (r'torque', 'float', 'Applied torque'),
            (r'time', 'float', 'Execution time'),
            (r'accuracy', 'float', 'Task accuracy'),
            (r'precision', 'float', 'Task precision')
        ]
        
        for pattern, data_type, description in kpi_patterns:
            if re.search(pattern, func_content, re.IGNORECASE):
                kpi = TaskKPI(
                    name=pattern,
                    data_type=data_type,
                    description=description,
                    success_criteria=f"{pattern} indicates successful completion",
                    monitoring_frequency="continuous"
                )
                kpis.append(kpi)
        
        return kpis
    
    def _extract_class_kpis(self, class_node: ast.ClassDef, content: str) -> List[TaskKPI]:
        """Extract KPIs from class methods"""
        kpis = []
        
        # Look for methods that return status or metrics
        for item in class_node.body:
            if isinstance(item, ast.FunctionDef):
                method_content = self._get_function_content(item, content)
                
                # Check if method returns KPIs
                if self._method_returns_kpi(item):
                    kpi_name = f"{item.name}_result"
                    data_type = self._infer_return_type(item)
                    
                    kpi = TaskKPI(
                        name=kpi_name,
                        data_type=data_type,
                        description=f"Result from {item.name} method",
                        success_criteria=f"Method {item.name} completes successfully",
                        monitoring_frequency="on_call"
                    )
                    kpis.append(kpi)
        
        return kpis
    
    def _extract_default_value(self, default_node) -> float:
        """Extract default value from AST node"""
        try:
            if isinstance(default_node, ast.Constant):
                if isinstance(default_node.value, (int, float)):
                    return float(default_node.value)
                elif isinstance(default_node.value, str):
                    # Try to convert string numbers
                    try:
                        return float(default_node.value)
                    except ValueError:
                        return 0.0
            elif isinstance(default_node, ast.Num):  # Python < 3.8
                return float(default_node.n)
            elif isinstance(default_node, ast.Name):
                # Handle cases like 'None', 'True', 'False'
                if default_node.id in ['None', 'True', 'False']:
                    return 0.0
            elif isinstance(default_node, ast.List):
                # Handle list defaults - return length
                return float(len(default_node.elts))
            elif isinstance(default_node, ast.Tuple):
                # Handle tuple defaults - return length
                return float(len(default_node.elts))
            elif isinstance(default_node, ast.Dict):
                # Handle dict defaults - return length
                return float(len(default_node.keys))
        except Exception as e:
            self.logger.debug(f"Failed to extract default value: {e}")
        
        return 0.0
    
    def _extract_assignment_value(self, value_node) -> Optional[float]:
        """Extract value from assignment"""
        try:
            if isinstance(value_node, ast.Constant):
                if isinstance(value_node.value, (int, float)):
                    return float(value_node.value)
                elif isinstance(value_node.value, str):
                    # Try to convert string numbers
                    try:
                        return float(value_node.value)
                    except ValueError:
                        return None
            elif isinstance(value_node, ast.Num):  # Python < 3.8
                return float(value_node.n)
            elif isinstance(value_node, ast.Name):
                # Handle cases like 'None', 'True', 'False'
                if value_node.id in ['None', 'True', 'False']:
                    return None
            elif isinstance(value_node, ast.List):
                # Handle list assignments - return length
                return float(len(value_node.elts))
            elif isinstance(value_node, ast.Tuple):
                # Handle tuple assignments - return length
                return float(len(value_node.elts))
            elif isinstance(value_node, ast.Dict):
                # Handle dict assignments - return length
                return float(len(value_node.keys))
            elif isinstance(value_node, ast.Call):
                # Handle function calls - try to extract meaningful value
                if hasattr(value_node.func, 'id'):
                    func_name = value_node.func.id.lower()
                    if 'pi' in func_name:
                        return 3.14159
                    elif 'inf' in func_name:
                        return float('inf')
                    elif 'zeros' in func_name or 'ones' in func_name:
                        # Handle numpy arrays
                        if value_node.args:
                            try:
                                return float(value_node.args[0].value)
                            except:
                                return 1.0
        except Exception as e:
            self.logger.debug(f"Failed to extract assignment value: {e}")
        
        return None
    
    def _infer_parameter_type(self, param_name: str, value: float) -> str:
        """Infer parameter type from name and value"""
        param_lower = param_name.lower()
        
        if any(keyword in param_lower for keyword in ['mass', 'weight']):
            return 'mass'
        elif any(keyword in param_lower for keyword in ['friction', 'mu']):
            return 'friction'
        elif any(keyword in param_name for keyword in ['damping', 'damp']):
            return 'damping'
        elif any(keyword in param_name for keyword in ['stiffness', 'k']):
            return 'stiffness'
        elif any(keyword in param_name for keyword in ['gain', 'kp', 'ki', 'kd']):
            return 'gains'
        elif any(keyword in param_name for keyword in ['limit', 'max', 'min']):
            return 'limits'
        elif any(keyword in param_name for keyword in ['alpha', 'attractive']):
            return 'alpha'
        elif any(keyword in param_name for keyword in ['beta', 'repulsive']):
            return 'beta'
        elif any(keyword in param_name for keyword in ['rho', 'radius', 'influence']):
            return 'rho'
        elif any(keyword in param_name for keyword in ['epsilon', 'tolerance', 'threshold']):
            return 'epsilon'
        elif any(keyword in param_name for keyword in ['resolution', 'step', 'delta']):
            return 'resolution'
        elif any(keyword in param_name for keyword in ['iterations', 'max_iter', 'max_iters']):
            return 'max_iterations'
        else:
            return 'generic'
    
    def _get_parameter_range(self, param_name: str, current_value: float, param_type: str) -> Tuple[float, float]:
        """Get parameter range for Sobol analysis"""
        if param_type in self.parameter_ranges:
            range_min, range_max = self.parameter_ranges[param_type]
            return current_value * range_min, current_value * range_max
        
        # Default ranges based on parameter type
        if param_type == 'mass':
            return current_value * 0.8, current_value * 1.2
        elif param_type == 'friction':
            return max(0.05, current_value * 0.5), min(1.0, current_value * 2.0)
        elif param_type == 'gains':
            return current_value * 0.75, current_value * 1.25
        else:
            return current_value * 0.5, current_value * 2.0
    
    def _infer_parameter_unit(self, param_name: str, param_type: str) -> str:
        """Infer parameter unit"""
        param_lower = param_name.lower()
        
        if param_type == 'mass':
            return 'kg'
        elif param_type == 'friction':
            return 'unitless'
        elif param_type == 'damping':
            return 'N⋅s/m'
        elif param_type == 'stiffness':
            return 'N/m'
        elif param_type == 'gains':
            return 'unitless'
        elif param_type == 'limits':
            return 'rad' if 'joint' in param_lower else 'm'
        elif param_type == 'alpha':
            return 'unitless'
        elif param_type == 'beta':
            return 'unitless'
        elif param_type == 'rho':
            return 'm'
        elif param_type == 'epsilon':
            return 'm' if 'position' in param_lower else 'rad'
        elif param_type == 'resolution':
            return 'm' if 'position' in param_lower else 'rad'
        elif param_type == 'max_iterations':
            return 'count'
        else:
            return 'unitless'
    
    def _method_returns_kpi(self, method_node: ast.FunctionDef) -> bool:
        """Check if method returns a KPI"""
        # Check if method has return statement
        for node in ast.walk(method_node):
            if isinstance(node, ast.Return):
                return True
        
        # Check method name for KPI indicators
        method_name = method_node.name.lower()
        kpi_indicators = ['get', 'status', 'result', 'metric', 'measure', 'check']
        return any(indicator in method_name for indicator in kpi_indicators)
    
    def _infer_return_type(self, method_node: ast.FunctionDef) -> str:
        """Infer return type of method"""
        method_name = method_node.name.lower()
        
        if 'status' in method_name:
            return 'string'
        elif 'success' in method_name or 'check' in method_name:
            return 'bool'
        elif 'get' in method_name:
            return 'float'
        else:
            return 'string'
    
    def _get_function_content(self, func_node: ast.FunctionDef, content: str) -> str:
        """Get function content as string"""
        lines = content.split('\n')
        start_line = func_node.lineno - 1
        end_line = func_node.end_lineno
        
        if start_line < len(lines) and end_line <= len(lines):
            return '\n'.join(lines[start_line:end_line])
        return ""
    
    def _get_class_content(self, class_node: ast.ClassDef, content: str) -> str:
        """Get class content as string"""
        lines = content.split('\n')
        start_line = class_node.lineno - 1
        end_line = class_node.end_lineno
        
        if start_line < len(lines) and end_line <= len(lines):
            return '\n'.join(lines[start_line:end_line])
        return ""
    
    def _calculate_function_confidence(self, func_node: ast.FunctionDef, content: str) -> float:
        """Calculate confidence that function is a robotics task"""
        confidence = 0.0
        
        # Function name relevance
        func_name = func_node.name.lower()
        for pattern_list in self.task_patterns.values():
            for pattern in pattern_list:
                if re.search(pattern, func_name, re.IGNORECASE):
                    confidence += 0.3
                    break
        
        # Docstring presence
        if ast.get_docstring(func_node):
            confidence += 0.2
        
        # Parameter count (more parameters = more likely to be robotics task)
        param_count = len(func_node.args.args)
        if param_count > 2:
            confidence += 0.2
        elif param_count > 0:
            confidence += 0.1
        
        # Content relevance
        func_content = self._get_function_content(func_node, content)
        robotics_keywords = ['robot', 'arm', 'gripper', 'joint', 'link', 'actuator', 
                           'kinematics', 'dynamics', 'trajectory', 'mujoco', 'pybullet']
        
        keyword_count = sum(1 for keyword in robotics_keywords if keyword in func_content.lower())
        confidence += min(0.3, keyword_count * 0.1)
        
        return min(1.0, confidence)
    
    def _calculate_class_confidence(self, class_node: ast.ClassDef, content: str) -> float:
        """Calculate confidence that class is robotics-related"""
        confidence = 0.0
        
        # Class name relevance
        class_name = class_node.name.lower()
        if any(keyword in class_name for keyword in ['robot', 'arm', 'gripper', 'task', 'controller', 'potential', 'field', 'apf']):
            confidence += 0.3
        
        # Method count
        method_count = sum(1 for item in class_node.body if isinstance(item, ast.FunctionDef))
        if method_count > 2:
            confidence += 0.2
        elif method_count > 0:
            confidence += 0.1
        
        # Docstring presence
        if ast.get_docstring(class_node):
            confidence += 0.2
        
        # Content relevance
        class_content = self._get_class_content(class_node, content)
        robotics_keywords = ['robot', 'arm', 'gripper', 'joint', 'link', 'actuator', 
                           'kinematics', 'dynamics', 'trajectory', 'mujoco', 'pybullet',
                           'potential', 'field', 'plan', 'path', 'obstacle', 'attractive', 'repulsive']
        
        keyword_count = sum(1 for keyword in robotics_keywords if keyword in class_content.lower())
        confidence += min(0.3, keyword_count * 0.1)
        
        return min(1.0, confidence)
    
    def _deduplicate_enhanced_tasks(self, tasks: List[EnhancedTaskInfo]) -> List[EnhancedTaskInfo]:
        """Deduplicate tasks and sort by confidence"""
        # Group by task name
        task_groups = {}
        for task in tasks:
            if task.task_name not in task_groups:
                task_groups[task.task_name] = []
            task_groups[task.task_name].append(task)
        
        # Keep the highest confidence version of each task
        deduplicated = []
        for task_name, task_list in task_groups.items():
            best_task = max(task_list, key=lambda t: t.confidence)
            deduplicated.append(best_task)
        
        # Sort by confidence (highest first)
        deduplicated.sort(key=lambda t: t.confidence, reverse=True)
        
        return deduplicated
    
    def generate_task_yamls(self, tasks: List[EnhancedTaskInfo], output_dir: Path) -> List[str]:
        """Generate YAML files for each task"""
        output_dir.mkdir(parents=True, exist_ok=True)
        generated_files = []
        
        for task in tasks:
            try:
                # Convert task to YAML-serializable format
                task_dict = asdict(task)
                
                # Convert numpy types to Python types if needed
                task_dict = self._convert_numpy_types(task_dict)
                
                # Generate filename
                filename = f"{task.task_name.lower().replace(' ', '_')}_task.yaml"
                filepath = output_dir / filename
                
                # Write YAML
                with open(filepath, 'w') as f:
                    yaml.dump(task_dict, f, default_flow_style=False, 
                             sort_keys=False, indent=2, width=80)
                
                generated_files.append(str(filepath))
                self.logger.info(f"Generated task YAML: {filepath}")
                
            except Exception as e:
                self.logger.error(f"Failed to generate YAML for {task.task_name}: {e}")
                continue
        
        return generated_files
    
    def _extract_global_parameters(self, tree: ast.AST, content: str) -> Dict[str, float]:
        """Extract global parameters from module level"""
        global_params = {}
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        param_name = target.id
                        
                        # Skip common non-parameter names
                        if param_name in ['__all__', '__version__', '__author__']:
                            continue
                        
                        # Extract value
                        value = self._extract_assignment_value(node.value)
                        if value is not None:
                            global_params[param_name] = value
        
        return global_params
    
    def _is_parameter_relevant_to_function(self, param_name: str, func_node: ast.FunctionDef, content: str) -> bool:
        """Check if a global parameter is relevant to a specific function"""
        # Check if parameter name appears in function content
        func_content = self._get_function_content(func_node, content)
        
        # Check for direct usage
        if param_name in func_content:
            return True
        
        # Check for related terms (e.g., 'damping' might be related to 'control' functions)
        param_lower = param_name.lower()
        func_name = func_node.name.lower()
        
        # Robotics-specific relevance patterns
        if any(keyword in param_lower for keyword in ['gain', 'k', 'damping', 'stiffness']):
            if any(keyword in func_name for keyword in ['control', 'plan', 'move', 'execute']):
                return True
        
        if any(keyword in param_lower for keyword in ['dt', 'time', 'step']):
            if any(keyword in func_name for keyword in ['simulate', 'run', 'step', 'integrate']):
                return True
        
        if any(keyword in param_name for keyword in ['alpha', 'beta', 'rho']):
            if any(keyword in func_name for keyword in ['plan', 'potential', 'field', 'apf']):
                return True
        
        return False
    
    def _convert_numpy_types(self, obj):
        """Convert numpy types to Python types for YAML serialization"""
        if hasattr(obj, 'item'):  # numpy scalar
            return obj.item()
        elif isinstance(obj, dict):
            return {k: self._convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        else:
            return obj 