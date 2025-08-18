#!/usr/bin/env python3
"""
Task Extractor
==============

Advanced AST-based task extraction from Python code with robotics relevance filtering.
"""

import re
import ast
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

from data_models import TaskInfo

logger = logging.getLogger(__name__)

class TaskExtractor:
    """Advanced AST-based task extraction"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.TaskExtractor")
        self.task_patterns = {
            'manipulation': [
                r'pick.*up', r'place.*down', r'grasp', r'release', r'grip',
                r'manipulat', r'move.*object', r'transfer', r'handle'
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
            ]
        }
        
        self.robotics_keywords = [
            'robot', 'arm', 'gripper', 'joint', 'link', 'actuator', 'motor', 'servo',
            'kinematics', 'dynamics', 'jacobian', 'dof', 'trajectory', 'path',
            'mujoco', 'pybullet', 'ros', 'moveit', 'gazebo', 'isaac', 'gym',
            'manipulation', 'grasping', 'navigation', 'perception', 'planning'
        ]
    
    def extract_tasks(self, repo_path: Path) -> List[TaskInfo]:
        """Extract tasks from Python files using AST analysis"""
        tasks = []
        python_files = list(repo_path.rglob("*.py"))
        
        self.logger.info(f"Analyzing {len(python_files)} Python files for tasks")
        
        for file_path in python_files:
            try:
                file_tasks = self._analyze_file(file_path)
                tasks.extend(file_tasks)
            except Exception as e:
                self.logger.warning(f"Failed to analyze {file_path}: {e}")
        
        # Deduplicate tasks by name and confidence
        deduplicated_tasks = self._deduplicate_tasks(tasks)
        self.logger.info(f"Found {len(deduplicated_tasks)} unique tasks")
        return deduplicated_tasks
    
    def _analyze_file(self, file_path: Path) -> List[TaskInfo]:
        """Analyze a single Python file for tasks"""
        try:
            content = file_path.read_text(encoding='utf-8', errors='ignore')
            tree = ast.parse(content)
            
            tasks = []
            
            # Analyze both functions and classes for tasks
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    task = self._analyze_function(node, content, file_path)
                    if task:
                        tasks.append(task)
                elif isinstance(node, ast.ClassDef):
                    # Also analyze classes for main task classes
                    task = self._analyze_class(node, content, file_path)
                    if task:
                        tasks.append(task)
            
            return tasks
            
        except Exception as e:
            self.logger.warning(f"Failed to parse {file_path}: {e}")
            return []
    
    def _analyze_function(self, func_node: ast.FunctionDef, content: str, file_path: Path) -> Optional[TaskInfo]:
        """Analyze function to determine if it's a robotics task"""
        func_name = func_node.name
        
        # Skip private/internal functions
        if func_name.startswith('_'):
            return None
        
        # Extract function content
        try:
            func_lines = content.split('\n')[func_node.lineno-1:func_node.end_lineno]
            func_content = '\n'.join(func_lines)
        except:
            return None
        
        # Check robotics relevance
        relevance_score = self._calculate_robotics_relevance(func_content)
        if relevance_score < 0.3:
            return None
        
        # Determine task type
        task_type = self._determine_task_type(func_content)
        if not task_type:
            return None
        
        # Extract rich metadata
        description = self._extract_description(func_node, func_content)
        required_actions = self._extract_actions(func_content)
        parameters = self._extract_parameters(func_node)
        dependencies = self._extract_dependencies(func_content, content)
        duration = self._estimate_duration(func_content)
        complexity = self._estimate_complexity(func_content)
        
        return TaskInfo(
            name=func_name,
            description=description,
            task_type=task_type,
            required_actions=required_actions,
            parameters=parameters,
            dependencies=dependencies,
            estimated_duration=duration,
            complexity=complexity,
            confidence=min(relevance_score, 0.95),
            file_path=str(file_path)
        )
    
    def _analyze_class(self, class_node: ast.ClassDef, content: str, file_path: Path) -> Optional[TaskInfo]:
        """Analyze class to determine if it's a robotics task class"""
        class_name = class_node.name
        
        # Skip private/internal classes
        if class_name.startswith('_'):
            return None
        
        # Check if this looks like a main task class (not just a utility class)
        if not self._is_main_task_class(class_name, class_node):
            return None
        
        # Extract class content
        try:
            class_lines = content.split('\n')[class_node.lineno-1:class_node.end_lineno]
            class_content = '\n'.join(class_lines)
        except:
            return None
        
        # Check robotics relevance
        relevance_score = self._calculate_robotics_relevance(class_content)
        if relevance_score < 0.2:  # Lower threshold for classes
            return None
        
        # Determine task type
        task_type = self._determine_task_type(class_content)
        if not task_type:
            return None
        
        # Extract rich metadata
        description = self._extract_class_description(class_node, class_content)
        required_actions = self._extract_actions(class_content)
        parameters = self._extract_class_parameters(class_node)
        dependencies = self._extract_dependencies(class_content, content)
        duration = self._estimate_duration(class_content)
        complexity = self._estimate_complexity(class_content)
        
        return TaskInfo(
            name=class_name,
            description=description,
            task_type=task_type,
            required_actions=required_actions,
            parameters=parameters,
            dependencies=dependencies,
            estimated_duration=duration,
            complexity=complexity,
            confidence=min(relevance_score + 0.1, 0.95),  # Boost confidence for classes
            file_path=str(file_path)
        )
    
    def _is_main_task_class(self, class_name: str, class_node: ast.ClassDef) -> bool:
        """Check if this class is a main task class, not just a utility class"""
        # Look for common task class naming patterns
        task_class_patterns = [
            'task', 'robot', 'controller', 'planner', 'perception', 'sequence',
            'base', 'main', 'core', 'primary'
        ]
        
        class_name_lower = class_name.lower()
        
        # Check if class name contains task-related keywords
        if any(pattern in class_name_lower for pattern in task_class_patterns):
            return True
        
        # Check if class has methods that suggest it's a main task class
        method_names = [method.name for method in class_node.body if isinstance(method, ast.FunctionDef)]
        method_names_lower = [name.lower() for name in method_names]
        
        # Look for methods that suggest this is a main task class
        task_methods = ['run', 'execute', 'start', 'stop', 'go_to', 'pick', 'place', 'insert', 'reach']
        if any(method in method_names_lower for method in task_methods):
            return True
        
        return False
    
    def _extract_description(self, func_node: ast.FunctionDef, func_content: str) -> str:
        """Extract function description from docstring or generate one"""
        # Try to get docstring
        if (func_node.body and 
            isinstance(func_node.body[0], ast.Expr) and 
            isinstance(func_node.body[0].value, ast.Constant) and 
            isinstance(func_node.body[0].value.value, str)):
            return func_node.body[0].value.value.strip()
        
        # Generate from function name
        name_words = re.findall(r'[A-Z][a-z]*|[a-z]+', func_node.name)
        return ' '.join(name_words).lower().capitalize()
    
    def _extract_class_description(self, class_node: ast.ClassDef, content: str) -> str:
        """Extract description from class docstring or infer from class name"""
        # Try to get docstring
        if ast.get_docstring(class_node):
            return ast.get_docstring(class_node)
        
        # Infer from class name and content
        class_name = class_node.name
        
        # Look for common task patterns in the class name
        if 'task' in class_name.lower():
            if 'pick' in class_name.lower():
                return f"Main task class for picking operations"
            elif 'place' in class_name.lower():
                return f"Main task class for placing operations"
            elif 'insert' in class_name.lower():
                return f"Main task class for insertion operations"
            elif 'reach' in class_name.lower():
                return f"Main task class for reaching operations"
            elif 'base' in class_name.lower():
                return f"Base class for all robotics tasks"
            else:
                return f"Main task class: {class_name}"
        
        return f"Robotics task class: {class_name}"
    
    def _extract_actions(self, content: str) -> List[str]:
        """Extract required actions from function content"""
        actions = []
        content_lower = content.lower()
        
        action_patterns = [
            (r'move.*to|goto|navigate', 'move'),
            (r'pick.*up|grasp|grab', 'pick'),
            (r'place.*down|put.*down|drop', 'place'),
            (r'open|close', 'actuate'),
            (r'rotate|turn', 'rotate'),
            (r'wait|sleep|delay', 'wait'),
            (r'check|verify|validate', 'verify'),
            (r'calculate|compute', 'compute')
        ]
        
        for pattern, action in action_patterns:
            if re.search(pattern, content_lower):
                actions.append(action)
        
        return list(set(actions))
    
    def _extract_parameters(self, func_node: ast.FunctionDef) -> Dict[str, Any]:
        """Extract function parameters"""
        parameters = {}
        
        for arg in func_node.args.args:
            param_info = {'name': arg.arg, 'type': 'unknown'}
            
            if arg.annotation:
                if isinstance(arg.annotation, ast.Name):
                    param_info['type'] = arg.annotation.id
                elif isinstance(arg.annotation, ast.Constant):
                    param_info['type'] = str(arg.annotation.value)
            
            parameters[arg.arg] = param_info
        
        return parameters
    
    def _calculate_robotics_relevance(self, content: str) -> float:
        """Calculate robotics relevance score"""
        content_lower = content.lower()
        matches = sum(1 for keyword in self.robotics_keywords if keyword in content_lower)
        return min(matches / 8.0, 1.0)  # Normalize to 0-1
    
    def _determine_task_type(self, content: str) -> Optional[str]:
        """Determine task type from content"""
        content_lower = content.lower()
        
        for task_type, patterns in self.task_patterns.items():
            for pattern in patterns:
                if re.search(pattern, content_lower):
                    return task_type
        
        return 'control'  # Default for robotics functions
    
    def _extract_class_parameters(self, class_node: ast.ClassDef) -> Dict[str, Any]:
        """Extract parameters from class constructor or attributes"""
        params = {}
        
        # Look for __init__ method
        for item in class_node.body:
            if isinstance(item, ast.FunctionDef) and item.name == '__init__':
                # Extract parameters from constructor
                for arg in item.args.args:
                    if arg.arg != 'self':
                        params[arg.arg] = {
                            'name': arg.arg,
                            'type': 'unknown',
                            'description': f'Parameter for {arg.arg}'
                        }
                break
        
        # Look for class attributes
        for item in class_node.body:
            if isinstance(item, ast.Assign):
                for target in item.targets:
                    if isinstance(target, ast.Name):
                        params[target.id] = {
                            'name': target.id,
                            'type': 'unknown',
                            'description': f'Class attribute: {target.id}'
                        }
        
        return params
    
    def _extract_dependencies(self, func_content: str, file_content: str = "") -> Dict[str, Any]:
        """Extract comprehensive function dependencies and imports."""
        dependencies = {
            'imports': [],
            'function_calls': [],
            'external_libraries': [],
            'robotics_libraries': [],
            'file_dependencies': [],
            'constants': [],
            'variables': []
        }
        
        # Extract imports from the entire file
        import_patterns = [
            r'import\s+([a-zA-Z_][a-zA-Z0-9_\.]*)',
            r'from\s+([a-zA-Z_][a-zA-Z0-9_\.]+)\s+import',
        ]
        
        full_content = file_content if file_content else func_content
        for pattern in import_patterns:
            for match in re.finditer(pattern, full_content):
                import_name = match.group(1)
                dependencies['imports'].append(import_name)
        
        # Extract function calls within the function
        func_call_pattern = r'([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)\s*\('
        for match in re.finditer(func_call_pattern, func_content):
            func_name = match.group(1)
            dependencies['function_calls'].append(func_name)
            if '.' in func_name:
                module_name = func_name.split('.')[0]
                if module_name not in dependencies['external_libraries']:
                    dependencies['external_libraries'].append(module_name)
        
        # Extract robotics-specific libraries
        robotics_libs = {
            'numpy': ['np.', 'numpy.'],
            'scipy': ['scipy.', 'sp.'],
            'rospy': ['rospy.'],
            'tf': ['tf.', 'tf2.'],
            'moveit': ['moveit.', 'moveit_commander'],
            'gazebo': ['gazebo.'],
            'rviz': ['rviz.'],
            'urdf': ['urdf.', 'urdfpy'],
            'mujoco': ['mujoco.', 'mjx.', 'mj.'],
            'pybullet': ['pybullet.', 'p.'],
            'opencv': ['cv2.', 'opencv'],
            'pytorch': ['torch.', 'pytorch'],
            'tensorflow': ['tf.', 'tensorflow']
        }
        
        for lib, patterns in robotics_libs.items():
            for pattern in patterns:
                if pattern in func_content or pattern in full_content:
                    if lib not in dependencies['robotics_libraries']:
                        dependencies['robotics_libraries'].append(lib)
        
        # Extract constants (ALL_CAPS variables)
        const_pattern = r'\b([A-Z][A-Z0-9_]+)\b'
        for match in re.finditer(const_pattern, func_content):
            const_name = match.group(1)
            if len(const_name) > 2:  # Avoid single letters
                dependencies['constants'].append(const_name)
        
        # Extract variable assignments
        var_pattern = r'([a-zA-Z_][a-zA-Z0-9_]*)\s*='
        for match in re.finditer(var_pattern, func_content):
            var_name = match.group(1)
            if not var_name.isupper():  # Skip constants
                dependencies['variables'].append(var_name)
        
        # Remove duplicates
        for key in dependencies:
            if isinstance(dependencies[key], list):
                dependencies[key] = list(set(dependencies[key]))
        
        return dependencies
    
    def _estimate_duration(self, content: str) -> float:
        """Estimate task duration in seconds"""
        lines = len(content.split('\n'))
        loops = len(re.findall(r'\bfor\b|\bwhile\b', content))
        waits = len(re.findall(r'sleep|wait|delay', content.lower()))
        
        return lines * 0.1 + loops * 2.0 + waits * 1.0
    
    def _estimate_complexity(self, content: str) -> str:
        """Estimate task complexity"""
        lines = len(content.split('\n'))
        conditionals = len(re.findall(r'\bif\b|\belse\b|\belif\b', content))
        loops = len(re.findall(r'\bfor\b|\bwhile\b', content))
        
        complexity_score = lines * 0.1 + conditionals * 0.5 + loops * 0.8
        
        if complexity_score < 5:
            return 'simple'
        elif complexity_score < 15:
            return 'medium'
        else:
            return 'complex'
    
    def _deduplicate_tasks(self, tasks: List[TaskInfo]) -> List[TaskInfo]:
        """Remove duplicate tasks, keeping highest confidence"""
        task_dict = {}
        
        for task in tasks:
            if task.name not in task_dict or task.confidence > task_dict[task.name].confidence:
                task_dict[task.name] = task
        
        return list(task_dict.values())
