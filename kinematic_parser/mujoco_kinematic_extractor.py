#!/usr/bin/env python3
"""
MuJoCo Kinematic Task Extractor
================================

Specialized tool for extracting kinematic information and tasks from MuJoCo-based
robot repositories for simulation testing.

Author: AI Assistant
Date: 2024
"""

import os
import sys
import json
import yaml
import argparse
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import re
import logging
from datetime import datetime
import xml.etree.ElementTree as ET

# Progress tracking
from tqdm import tqdm

@dataclass
class JointInfo:
    """Structured joint information"""
    name: str
    joint_type: str  # 'revolute', 'prismatic', 'continuous', 'fixed'
    axis: List[float]
    origin: List[float]
    limits: Optional[Dict[str, float]]
    parent_link: str
    child_link: str
    
@dataclass
class LinkInfo:
    """Structured link information"""
    name: str
    visual_geometry: Optional[Dict]
    collision_geometry: Optional[Dict]
    
@dataclass
class RobotKinematics:
    """Complete robot kinematics information"""
    name: str
    joints: List[JointInfo]
    links: List[LinkInfo]
    base_link: str
    end_effector: Optional[str]
    dof: int
    model_path: str
    
@dataclass
class ActionInfo:
    """Individual action information"""
    name: str
    action_type: str  # 'move', 'grasp', 'release', 'rotate', 'push', 'wait'
    parameters: Dict[str, Any]
    duration: float
    priority: int
    
@dataclass
class TaskInfo:
    """Task information extracted from code"""
    name: str
    description: str
    task_type: str  # 'manipulation', 'navigation', 'grasping', etc.
    actions: List[ActionInfo]
    parameters: Dict[str, Any]
    dependencies: List[str]
    estimated_duration: float
    complexity: str  # 'simple', 'medium', 'complex'
    success_criteria: Dict[str, Any]
    
@dataclass
class SimulationConfig:
    """Simulation configuration"""
    robot_kinematics: Optional[RobotKinematics]
    tasks: List[TaskInfo]
    environment: Dict[str, Any]
    physics_engine: str
    timestep: float
    max_steps: int
    
@dataclass
class CodebaseAnalysis:
    """Complete analysis of a codebase"""
    repository_url: str
    local_path: str
    simulation_config: SimulationConfig
    configuration_files: List[str]
    dependencies: List[str]
    test_files: List[str]

class MuJoCoKinematicParser:
    """Parses MuJoCo XML files and extracts kinematic information"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def find_model_files(self, codebase_path: str) -> List[str]:
        """Find all MuJoCo model files in the codebase"""
        model_files = []
        for root, dirs, files in os.walk(codebase_path):
            for file in files:
                if file.endswith(('.xml', '.mjcf')):
                    model_files.append(os.path.join(root, file))
        return model_files
    
    def parse_mujoco_xml(self, xml_path: str) -> RobotKinematics:
        """Parse a MuJoCo XML file and extract kinematic information"""
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # Extract robot name
            robot_name = root.get('model', 'robot')
            
            joints = []
            links = []
            
            # Parse joints
            for joint_elem in root.findall('.//joint'):
                joint_info = JointInfo(
                    name=joint_elem.get('name', ''),
                    joint_type=joint_elem.get('type', 'free'),
                    axis=self._parse_axis(joint_elem.get('axis', '0 0 1')),
                    origin=self._parse_origin(joint_elem.get('pos', '0 0 0')),
                    limits=self._parse_limits(joint_elem),
                    parent_link=joint_elem.get('parent', ''),
                    child_link=joint_elem.get('child', '')
                )
                joints.append(joint_info)
            
            # Parse links/geoms
            for geom_elem in root.findall('.//geom'):
                link_info = LinkInfo(
                    name=geom_elem.get('name', ''),
                    visual_geometry=self._extract_geometry(geom_elem),
                    collision_geometry=self._extract_geometry(geom_elem)
                )
                links.append(link_info)
            
            # Determine DOF
            actuated_joints = [j for j in joints if j.joint_type in ['hinge', 'slide']]
            
            # Try to identify end effector
            end_effector = self._identify_end_effector(joints, links)
            
            return RobotKinematics(
                name=robot_name,
                joints=joints,
                links=links,
                base_link='base_link',  # MuJoCo doesn't have explicit base links
                end_effector=end_effector,
                dof=len(actuated_joints),
                model_path=xml_path
            )
            
        except Exception as e:
            self.logger.error(f"Error parsing MuJoCo XML file {xml_path}: {e}")
            raise
    
    def _parse_axis(self, axis_str: str) -> List[float]:
        """Parse axis string to list of floats"""
        try:
            return [float(x) for x in axis_str.split()]
        except:
            return [0.0, 0.0, 1.0]
    
    def _parse_origin(self, pos_str: str) -> List[float]:
        """Parse position string to list of floats"""
        try:
            return [float(x) for x in pos_str.split()]
        except:
            return [0.0, 0.0, 0.0]
    
    def _parse_limits(self, joint_elem) -> Optional[Dict[str, float]]:
        """Parse joint limits"""
        limits = {}
        if joint_elem.get('range'):
            range_str = joint_elem.get('range')
            try:
                range_values = [float(x) for x in range_str.split()]
                if len(range_values) >= 2:
                    limits['lower'] = range_values[0]
                    limits['upper'] = range_values[1]
            except:
                pass
        return limits if limits else None
    
    def _extract_geometry(self, geom_elem) -> Optional[Dict]:
        """Extract geometry information"""
        geom_info = {
            'type': geom_elem.get('type', 'sphere'),
            'size': geom_elem.get('size', '1'),
            'pos': geom_elem.get('pos', '0 0 0'),
            'quat': geom_elem.get('quat', '1 0 0 0')
        }
        return geom_info
    
    def _identify_end_effector(self, joints: List[JointInfo], links: List[LinkInfo]) -> Optional[str]:
        """Try to identify the end effector link"""
        # Common end effector names in MuJoCo
        ee_names = ['gripper', 'hand', 'end_effector', 'tool', 'finger', 'claw', 'robotiq']
        
        for link in links:
            link_name_lower = link.name.lower()
            for ee_name in ee_names:
                if ee_name in link_name_lower:
                    return link.name
        
        return None

class MuJoCoTaskExtractor:
    """Extracts task information from MuJoCo-based code"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.task_patterns = {
            'manipulation': [
                r'grasp', r'pick', r'place', r'move', r'manipulate',
                r'open', r'close', r'rotate', r'turn'
            ],
            'navigation': [
                r'navigate', r'move_to', r'go_to', r'path_planning',
                r'localization', r'mapping'
            ],
            'grasping': [
                r'grasp', r'grip', r'hold', r'pick', r'grasp_planning'
            ],
            'pushing': [
                r'push', r'slide', r'move_object', r'displace'
            ]
        }
        
        self.action_patterns = {
            'move': r'move|navigate|go_to|position',
            'grasp': r'grasp|grip|pick|hold',
            'release': r'release|drop|place|let_go',
            'rotate': r'rotate|turn|spin|orient',
            'push': r'push|slide|displace|move_object',
            'wait': r'wait|sleep|delay|pause'
        }
    
    def extract_tasks_from_code(self, codebase_path: str) -> List[TaskInfo]:
        """Extract task information from code files"""
        tasks = []
        
        # Find Python files
        python_files = []
        for root, dirs, files in os.walk(codebase_path):
            for file in files:
                if file.endswith('.py'):
                    python_files.append(os.path.join(root, file))
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Extract function definitions
                functions = self._extract_functions(content)
                
                for func_name, func_content in functions:
                    task_info = self._analyze_function(func_name, func_content, file_path)
                    if task_info:
                        tasks.append(task_info)
                        
            except Exception as e:
                self.logger.warning(f"Error reading file {file_path}: {e}")
        
        # Deduplicate and merge similar tasks
        tasks = self._deduplicate_tasks(tasks)
        
        return tasks
    
    def _deduplicate_tasks(self, tasks: List[TaskInfo]) -> List[TaskInfo]:
        """Deduplicate similar tasks and merge them"""
        if not tasks:
            return tasks
        
        # Group tasks by type and action patterns
        task_groups = {}
        
        for task in tasks:
            # Create a key based on task type and action types
            action_types = tuple(sorted([action.action_type for action in task.actions]))
            key = (task.task_type, action_types)
            
            if key not in task_groups:
                task_groups[key] = []
            task_groups[key].append(task)
        
        # Merge similar tasks
        merged_tasks = []
        
        for key, task_list in task_groups.items():
            if len(task_list) == 1:
                # Single task, keep as is
                merged_tasks.append(task_list[0])
            else:
                # Multiple similar tasks, merge them
                merged_task = self._merge_similar_tasks(task_list)
                merged_tasks.append(merged_task)
        
        return merged_tasks
    
    def _merge_similar_tasks(self, tasks: List[TaskInfo]) -> TaskInfo:
        """Merge similar tasks into one representative task"""
        if not tasks:
            return None
        
        # Use the first task as base
        base_task = tasks[0]
        
        # Merge names
        names = [task.name for task in tasks]
        merged_name = f"{base_task.task_type}_group_{len(tasks)}_tasks"
        
        # Merge descriptions
        descriptions = [task.description for task in tasks if task.description]
        merged_description = f"Group of {len(tasks)} similar {base_task.task_type} tasks: {', '.join(names[:3])}"
        if len(names) > 3:
            merged_description += f" and {len(names) - 3} more"
        
        # Merge parameters (union of all parameters)
        all_parameters = {}
        for task in tasks:
            all_parameters.update(task.parameters)
        
        # Merge dependencies (union of all dependencies)
        all_dependencies = set()
        for task in tasks:
            all_dependencies.update(task.dependencies)
        
        # Calculate average duration
        avg_duration = sum(task.estimated_duration for task in tasks) / len(tasks)
        
        # Determine overall complexity
        complexities = [task.complexity for task in tasks]
        if 'complex' in complexities:
            overall_complexity = 'complex'
        elif 'medium' in complexities:
            overall_complexity = 'medium'
        else:
            overall_complexity = 'simple'
        
        # Merge success criteria (use the most comprehensive one)
        best_success_criteria = max(tasks, key=lambda t: len(t.success_criteria)).success_criteria
        
        return TaskInfo(
            name=merged_name,
            description=merged_description,
            task_type=base_task.task_type,
            actions=base_task.actions,  # Use actions from first task
            parameters=all_parameters,
            dependencies=list(all_dependencies),
            estimated_duration=avg_duration,
            complexity=overall_complexity,
            success_criteria=best_success_criteria
        )
    
    def _extract_functions(self, content: str) -> List[Tuple[str, str]]:
        """Extract function definitions from Python code"""
        functions = []
        
        # Simple regex to find function definitions
        pattern = r'def\s+(\w+)\s*\([^)]*\)\s*:'
        matches = re.finditer(pattern, content)
        
        for match in matches:
            func_name = match.group(1)
            start_pos = match.start()
            
            # Find the function body (simplified)
            lines = content[start_pos:].split('\n')
            func_lines = []
            indent_level = None
            
            for i, line in enumerate(lines):
                if i == 0:  # First line is function definition
                    func_lines.append(line)
                    # Determine indent level
                    indent_level = len(line) - len(line.lstrip())
                    continue
                
                if line.strip() == '':
                    func_lines.append(line)
                    continue
                
                current_indent = len(line) - len(line.lstrip())
                if current_indent > indent_level:
                    func_lines.append(line)
                else:
                    break
            
            func_content = '\n'.join(func_lines)
            functions.append((func_name, func_content))
        
        return functions
    
    def _analyze_function(self, func_name: str, func_content: str, file_path: str) -> Optional[TaskInfo]:
        """Analyze a function to determine if it's a task"""
        func_content_lower = func_content.lower()
        
        # Skip certain function types that are not tasks
        skip_patterns = [
            r'__init__',
            r'__str__',
            r'__repr__',
            r'get_',
            r'set_',
            r'is_',
            r'has_',
            r'calc_',
            r'compute_',
            r'update_',
            r'init_',
            r'cleanup_',
            r'reset_',
            r'print_',
            r'log_',
            r'debug_',
            r'error_',
            r'warn_'
        ]
        
        # Skip if function matches skip patterns
        for pattern in skip_patterns:
            if re.search(pattern, func_name.lower()):
                return None
        
        # Determine task type
        task_type = None
        for task_category, patterns in self.task_patterns.items():
            for pattern in patterns:
                if re.search(pattern, func_name.lower()) or re.search(pattern, func_content_lower):
                    task_type = task_category
                    break
            if task_type:
                break
        
        if not task_type:
            return None
        
        # Extract actions
        actions = self._extract_actions(func_content)
        
        # Only create task if meaningful actions are found
        if not actions:
            return None
        
        # Extract parameters
        parameters = self._extract_parameters(func_content)
        
        # Extract dependencies
        dependencies = self._extract_dependencies(func_content)
        
        # Estimate complexity
        complexity = self._estimate_complexity(func_content)
        
        # Estimate duration
        estimated_duration = self._estimate_duration(func_content)
        
        # Generate success criteria
        success_criteria = self._generate_success_criteria(func_name, task_type)
        
        return TaskInfo(
            name=func_name,
            description=self._generate_description(func_name, func_content, task_type),
            task_type=task_type,
            actions=actions,
            parameters=parameters,
            dependencies=dependencies,
            estimated_duration=estimated_duration,
            complexity=complexity,
            success_criteria=success_criteria
        )
    
    def _extract_actions(self, func_content: str) -> List[ActionInfo]:
        """Extract detailed actions from function content"""
        actions = []
        func_content_lower = func_content.lower()
        
        # More specific action patterns with context
        action_definitions = {
            'move': {
                'pattern': r'(?:def|class).*?(?:move|navigate|go_to|position).*?[:\n]',
                'parameters': {'target_position': 'required', 'velocity': 'optional'},
                'duration': 2.0,
                'priority': 1
            },
            'grasp': {
                'pattern': r'(?:def|class).*?(?:grasp|grip|pick|hold).*?[:\n]',
                'parameters': {'object_id': 'required', 'grasp_force': 'optional'},
                'duration': 1.5,
                'priority': 2
            },
            'release': {
                'pattern': r'(?:def|class).*?(?:release|drop|place|let_go).*?[:\n]',
                'parameters': {'object_id': 'required'},
                'duration': 1.0,
                'priority': 3
            },
            'rotate': {
                'pattern': r'(?:def|class).*?(?:rotate|turn|spin|orient).*?[:\n]',
                'parameters': {'target_orientation': 'required', 'angular_velocity': 'optional'},
                'duration': 2.5,
                'priority': 1
            },
            'push': {
                'pattern': r'(?:def|class).*?(?:push|slide|displace|move_object).*?[:\n]',
                'parameters': {'target_position': 'required', 'force': 'optional'},
                'duration': 3.0,
                'priority': 2
            },
            'wait': {
                'pattern': r'(?:def|class).*?(?:wait|sleep|delay|pause).*?[:\n]',
                'parameters': {'duration': 'required'},
                'duration': 1.0,
                'priority': 4
            }
        }
        
        # Only add actions if they are specifically mentioned in function definition or key parts
        found_actions = set()
        
        for action_name, action_def in action_definitions.items():
            # Check if action is mentioned in function definition or key parts
            if re.search(action_def['pattern'], func_content_lower):
                found_actions.add(action_name)
        
        # If no specific actions found, try to infer from function content
        if not found_actions:
            # Look for specific keywords in the function body
            if 'joint' in func_content_lower and 'angle' in func_content_lower:
                found_actions.add('move')
            if 'grasp' in func_content_lower or 'grip' in func_content_lower:
                found_actions.add('grasp')
            if 'push' in func_content_lower or 'slide' in func_content_lower:
                found_actions.add('push')
            if 'rotate' in func_content_lower or 'turn' in func_content_lower:
                found_actions.add('rotate')
        
        # Create action objects only for found actions
        for action_name in found_actions:
            action_def = action_definitions[action_name]
            action = ActionInfo(
                name=f"{action_name}_action",
                action_type=action_name,
                parameters=action_def['parameters'],
                duration=action_def['duration'],
                priority=action_def['priority']
            )
            actions.append(action)
        
        return actions
    
    def _extract_parameters(self, func_content: str) -> Dict[str, Any]:
        """Extract function parameters"""
        params = {}
        
        # Look for common parameter patterns
        param_patterns = {
            'position': r'position|pos|xyz|coordinate',
            'orientation': r'orientation|quat|rotation|rpy',
            'joint_angles': r'joint|angle|q_|theta',
            'velocity': r'velocity|vel|speed',
            'force': r'force|torque|effort',
            'timeout': r'timeout|time_limit|duration',
            'object_id': r'object|target|goal',
            'grasp_force': r'grasp_force|grip_force|force'
        }
        
        for param_name, pattern in param_patterns.items():
            if re.search(pattern, func_content, re.IGNORECASE):
                params[param_name] = 'required'
        
        return params
    
    def _extract_dependencies(self, func_content: str) -> List[str]:
        """Extract function dependencies"""
        dependencies = []
        
        # Look for import statements
        import_pattern = r'import\s+(\w+)'
        from_pattern = r'from\s+(\w+)'
        
        imports = re.findall(import_pattern, func_content)
        froms = re.findall(from_pattern, func_content)
        
        dependencies.extend(imports)
        dependencies.extend(froms)
        
        return list(set(dependencies))
    
    def _estimate_complexity(self, func_content: str) -> str:
        """Estimate function complexity"""
        lines = func_content.split('\n')
        non_empty_lines = [line for line in lines if line.strip()]
        
        if len(non_empty_lines) < 10:
            return 'simple'
        elif len(non_empty_lines) < 50:
            return 'medium'
        else:
            return 'complex'
    
    def _estimate_duration(self, func_content: str) -> float:
        """Estimate task duration in seconds"""
        # Very simplified estimation
        lines = len(func_content.split('\n'))
        return max(1.0, lines * 0.1)  # At least 1 second
    
    def _generate_description(self, func_name: str, func_content: str, task_type: str) -> str:
        """Generate a description for the task"""
        # Try to find docstring
        docstring_match = re.search(r'"""(.*?)"""', func_content, re.DOTALL)
        if docstring_match:
            return docstring_match.group(1).strip()
        
        # Generate based on function name and type
        return f"{task_type.title()} task: {func_name.replace('_', ' ').title()}"
    
    def _generate_success_criteria(self, func_name: str, task_type: str) -> Dict[str, Any]:
        """Generate success criteria for the task"""
        criteria = {
            'completion_timeout': 30.0,  # seconds
            'position_tolerance': 0.01,   # meters
            'orientation_tolerance': 0.1,  # radians
            'force_threshold': 10.0,      # Newtons
        }
        
        if task_type == 'grasping':
            criteria['grasp_success'] = True
            criteria['object_held'] = True
        elif task_type == 'pushing':
            criteria['object_displaced'] = True
            criteria['displacement_threshold'] = 0.05  # meters
        
        return criteria

class MuJoCoCodebaseAnalyzer:
    """Main analyzer class for MuJoCo-based repositories"""
    
    def __init__(self):
        self.kinematic_parser = MuJoCoKinematicParser()
        self.task_extractor = MuJoCoTaskExtractor()
        self.logger = logging.getLogger(__name__)
    
    def analyze_codebase(self, codebase_path: str, repository_url: str = None) -> CodebaseAnalysis:
        """Analyze a MuJoCo-based codebase"""
        self.logger.info(f"Starting analysis of MuJoCo codebase: {codebase_path}")
        
        # Find MuJoCo model files and parse kinematics
        model_files = self.kinematic_parser.find_model_files(codebase_path)
        robot_kinematics = None
        
        if model_files:
            self.logger.info(f"Found {len(model_files)} MuJoCo model files")
            # Use the first model file found
            robot_kinematics = self.kinematic_parser.parse_mujoco_xml(model_files[0])
        
        # Extract tasks
        tasks = self.task_extractor.extract_tasks_from_code(codebase_path)
        
        # Find configuration files
        config_files = self._find_config_files(codebase_path)
        
        # Find dependencies
        dependencies = self._find_dependencies(codebase_path)
        
        # Find test files
        test_files = self._find_test_files(codebase_path)
        
        # Create simulation configuration
        simulation_config = SimulationConfig(
            robot_kinematics=robot_kinematics,
            tasks=tasks,
            environment=self._extract_environment_info(codebase_path),
            physics_engine='mujoco',
            timestep=0.002,  # 500Hz
            max_steps=10000
        )
        
        return CodebaseAnalysis(
            repository_url=repository_url or '',
            local_path=codebase_path,
            simulation_config=simulation_config,
            configuration_files=config_files,
            dependencies=dependencies,
            test_files=test_files
        )
    
    def _find_config_files(self, codebase_path: str) -> List[str]:
        """Find configuration files"""
        config_files = []
        config_extensions = ['.yaml', '.yml', '.json', '.xml', '.cfg', '.ini']
        
        for root, dirs, files in os.walk(codebase_path):
            for file in files:
                if any(file.endswith(ext) for ext in config_extensions):
                    config_files.append(os.path.join(root, file))
        
        return config_files
    
    def _find_dependencies(self, codebase_path: str) -> List[str]:
        """Find project dependencies"""
        dependencies = []
        
        # Look for requirements.txt, setup.py, etc.
        dependency_files = ['requirements.txt', 'setup.py', 'package.json', 'Cargo.toml']
        
        for file in dependency_files:
            file_path = os.path.join(codebase_path, file)
            if os.path.exists(file_path):
                dependencies.append(file)
        
        return dependencies
    
    def _find_test_files(self, codebase_path: str) -> List[str]:
        """Find test files"""
        test_files = []
        
        for root, dirs, files in os.walk(codebase_path):
            for file in files:
                if 'test' in file.lower() or file.endswith('_test.py'):
                    test_files.append(os.path.join(root, file))
        
        return test_files
    
    def _extract_environment_info(self, codebase_path: str) -> Dict[str, Any]:
        """Extract environment information"""
        env_info = {
            'gravity': [0.0, 0.0, -9.81],
            'timestep': 0.002,
            'max_contacts': 100,
            'solver_iterations': 50,
            'solver_tolerance': 1e-10
        }
        
        # Look for environment configuration files
        env_files = ['environment.xml', 'world.xml', 'scene.xml']
        for env_file in env_files:
            env_path = os.path.join(codebase_path, env_file)
            if os.path.exists(env_path):
                env_info['environment_file'] = env_path
                break
        
        return env_info

class SimulationYAMLGenerator:
    """Generates YAML files optimized for simulation testing"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def generate_simulation_yaml(self, analysis: CodebaseAnalysis, output_path: str):
        """Generate YAML file optimized for simulation testing"""
        yaml_data = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'repository_url': analysis.repository_url,
                'local_path': analysis.local_path,
                'total_tasks': len(analysis.simulation_config.tasks)
            },
            'simulation_config': {
                'physics_engine': analysis.simulation_config.physics_engine,
                'timestep': analysis.simulation_config.timestep,
                'max_steps': analysis.simulation_config.max_steps,
                'environment': analysis.simulation_config.environment
            },
            'robot': self._serialize_robot(analysis.simulation_config.robot_kinematics),
            'tasks': [self._serialize_task_for_simulation(task) for task in analysis.simulation_config.tasks],
            'configuration_files': analysis.configuration_files,
            'dependencies': analysis.dependencies
        }
        
        with open(output_path, 'w') as f:
            yaml.dump(yaml_data, f, default_flow_style=False, indent=2)
        
        self.logger.info(f"Generated simulation YAML file: {output_path}")
    
    def _serialize_robot(self, kinematics: Optional[RobotKinematics]) -> Optional[Dict]:
        """Serialize robot information for simulation"""
        if kinematics is None:
            return None
        
        return {
            'name': kinematics.name,
            'dof': kinematics.dof,
            'base_link': kinematics.base_link,
            'end_effector': kinematics.end_effector,
            'model_path': kinematics.model_path,
            'joints': [self._serialize_joint(joint) for joint in kinematics.joints],
            'links': [self._serialize_link(link) for link in kinematics.links]
        }
    
    def _serialize_joint(self, joint: JointInfo) -> Dict:
        """Serialize joint information"""
        joint_data = {
            'name': joint.name,
            'type': joint.joint_type,
            'axis': joint.axis,
            'origin': joint.origin,
            'parent_link': joint.parent_link,
            'child_link': joint.child_link
        }
        
        if joint.limits:
            joint_data['limits'] = joint.limits
        
        return joint_data
    
    def _serialize_link(self, link: LinkInfo) -> Dict:
        """Serialize link information"""
        link_data = {
            'name': link.name
        }
        
        if link.visual_geometry:
            link_data['visual_geometry'] = link.visual_geometry
        
        if link.collision_geometry:
            link_data['collision_geometry'] = link.collision_geometry
        
        return link_data
    
    def _serialize_task_for_simulation(self, task: TaskInfo) -> Dict:
        """Serialize task information optimized for simulation"""
        return {
            'name': task.name,
            'description': task.description,
            'type': task.task_type,
            'complexity': task.complexity,
            'estimated_duration': task.estimated_duration,
            'parameters': task.parameters,
            'dependencies': task.dependencies,
            'success_criteria': task.success_criteria,
            'actions': [self._serialize_action(action) for action in task.actions]
        }
    
    def _serialize_action(self, action: ActionInfo) -> Dict:
        """Serialize action information"""
        return {
            'name': action.name,
            'type': action.action_type,
            'parameters': action.parameters,
            'duration': action.duration,
            'priority': action.priority
        }

def clone_repository(repo_url: str, local_path: str) -> bool:
    """Clone a repository from URL"""
    try:
        subprocess.run(['git', 'clone', repo_url, local_path], check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error cloning repository: {e}")
        return False

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="MuJoCo Kinematic Task Extractor for Simulation")
    parser.add_argument("--repository", "-r", help="Repository URL to clone")
    parser.add_argument("--codebase", "-c", help="Path to codebase directory")
    parser.add_argument("--output", "-o", help="Output directory for generated files")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    if not args.codebase and not args.repository:
        print("Error: Please provide either --codebase or --repository")
        return
    
    # Determine codebase path
    codebase_path = args.codebase
    repo_url = args.repository
    
    if repo_url:
        # Clone repository
        if not codebase_path:
            # Create temporary directory
            temp_dir = tempfile.mkdtemp()
            codebase_path = temp_dir
        
        print(f"Cloning repository: {repo_url}")
        if not clone_repository(repo_url, codebase_path):
            return
    
    # Analyze codebase
    analyzer = MuJoCoCodebaseAnalyzer()
    yaml_generator = SimulationYAMLGenerator()
    
    print(f"Analyzing MuJoCo codebase: {codebase_path}")
    analysis = analyzer.analyze_codebase(codebase_path, repo_url)
    
    # Generate output files
    output_dir = args.output or "simulation_analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    yaml_path = os.path.join(output_dir, "simulation_config.yaml")
    
    print("Generating simulation YAML file...")
    yaml_generator.generate_simulation_yaml(analysis, yaml_path)
    
    print(f"Analysis complete!")
    print(f"Simulation YAML file: {yaml_path}")
    print(f"Found {len(analysis.simulation_config.tasks)} tasks")
    if analysis.simulation_config.robot_kinematics:
        print(f"Robot: {analysis.simulation_config.robot_kinematics.name} ({analysis.simulation_config.robot_kinematics.dof} DOF)")

if __name__ == "__main__":
    main() 