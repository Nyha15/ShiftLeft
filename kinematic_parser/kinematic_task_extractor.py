#!/usr/bin/env python3
"""
Kinematic Task Extractor
========================

A comprehensive tool that parses any codebase to extract robot kinematics information,
understand tasks being performed, and generate YAML task sequences with documentation.

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
from urllib.parse import urlparse
import re
import logging
from datetime import datetime
import time

# UI Components
try:
    import tkinter as tk
    from tkinter import ttk, scrolledtext, messagebox, filedialog
    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False

# Kinematic parsing utilities
import numpy as np
from urdfpy import URDF
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
    safety_limits: Optional[Dict[str, float]]
    
@dataclass
class LinkInfo:
    """Structured link information"""
    name: str
    visual_geometry: Optional[Dict]
    collision_geometry: Optional[Dict]
    inertial: Optional[Dict]
    
@dataclass
class RobotKinematics:
    """Complete robot kinematics information"""
    name: str
    joints: List[JointInfo]
    links: List[LinkInfo]
    base_link: str
    end_effector: Optional[str]
    dof: int
    urdf_path: str
    
@dataclass
class TaskInfo:
    """Task information extracted from code"""
    name: str
    description: str
    task_type: str  # 'manipulation', 'navigation', 'grasping', etc.
    required_actions: List[str]
    parameters: Dict[str, Any]
    dependencies: List[str]
    estimated_duration: float
    complexity: str  # 'simple', 'medium', 'complex'
    
@dataclass
class CodebaseAnalysis:
    """Complete analysis of a codebase"""
    repository_url: str
    local_path: str
    robot_kinematics: Optional[RobotKinematics]
    tasks: List[TaskInfo]
    configuration_files: List[str]
    dependencies: List[str]
    test_files: List[str]
    documentation_files: List[str]

class KinematicParser:
    """Parses URDF files and extracts kinematic information"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def find_urdf_files(self, codebase_path: str) -> List[str]:
        """Find all URDF files in the codebase"""
        urdf_files = []
        for root, dirs, files in os.walk(codebase_path):
            for file in files:
                if file.endswith(('.urdf', '.xacro')):
                    urdf_files.append(os.path.join(root, file))
        return urdf_files
    
    def parse_urdf_file(self, urdf_path: str) -> RobotKinematics:
        """Parse a URDF file and extract kinematic information"""
        try:
            # Load URDF using urdfpy
            robot = URDF.load(urdf_path)
            
            joints = []
            for joint in robot.joints:
                joint_info = JointInfo(
                    name=joint.name,
                    joint_type=joint.joint_type,
                    axis=joint.axis.tolist() if joint.axis is not None else [0, 0, 0],
                    origin=joint.origin.reshape(-1).tolist() if joint.origin is not None else [0, 0, 0, 0, 0, 0, 1],
                    limits={
                        'lower': joint.limit.lower,
                        'upper': joint.limit.upper,
                        'effort': joint.limit.effort,
                        'velocity': joint.limit.velocity
                    } if joint.limit is not None else None,
                    parent_link=joint.parent,
                    child_link=joint.child,
                    safety_limits=None  # Will be filled if available
                )
                joints.append(joint_info)
            
            links = []
            for link in robot.links:
                link_info = LinkInfo(
                    name=link.name,
                    visual_geometry=self._extract_geometry(link.visual) if link.visual else None,
                    collision_geometry=self._extract_geometry(link.collision) if link.collision else None,
                    inertial=self._extract_inertial(link.inertial) if link.inertial else None
                )
                links.append(link_info)
            
            # Determine DOF
            actuated_joints = [j for j in joints if j.joint_type in ['revolute', 'prismatic', 'continuous']]
            
            # Try to identify end effector
            end_effector = self._identify_end_effector(joints, links)
            
            return RobotKinematics(
                name=robot.name if hasattr(robot, 'name') else 'robot',
                joints=joints,
                links=links,
                base_link=robot.base_link.name if robot.base_link else links[0].name if links else 'base_link',
                end_effector=end_effector,
                dof=len(actuated_joints),
                urdf_path=urdf_path
            )
            
        except Exception as e:
            self.logger.error(f"Error parsing URDF file {urdf_path}: {e}")
            raise
    
    def _extract_geometry(self, geometry) -> Optional[Dict]:
        """Extract geometry information"""
        if geometry is None:
            return None
        
        geom_info = {
            'type': type(geometry).__name__,
        }
        
        if hasattr(geometry, 'size'):
            geom_info['size'] = geometry.size.tolist()
        if hasattr(geometry, 'radius'):
            geom_info['radius'] = geometry.radius
        if hasattr(geometry, 'length'):
            geom_info['length'] = geometry.length
        if hasattr(geometry, 'filename'):
            geom_info['filename'] = geometry.filename
            
        return geom_info
    
    def _extract_inertial(self, inertial) -> Optional[Dict]:
        """Extract inertial information"""
        if inertial is None:
            return None
        
        return {
            'mass': inertial.mass,
            'origin': inertial.origin.reshape(-1).tolist() if inertial.origin is not None else None,
            'inertia': inertial.inertia.tolist() if inertial.inertia is not None else None
        }
    
    def _identify_end_effector(self, joints: List[JointInfo], links: List[LinkInfo]) -> Optional[str]:
        """Try to identify the end effector link"""
        # Common end effector names
        ee_names = ['gripper', 'hand', 'end_effector', 'tool', 'finger', 'claw']
        
        for link in links:
            link_name_lower = link.name.lower()
            for ee_name in ee_names:
                if ee_name in link_name_lower:
                    return link.name
        
        # If no obvious end effector, look for links with no children
        child_links = {joint.child_link for joint in joints}
        for link in links:
            if link.name not in child_links:
                return link.name
        
        return None

class TaskExtractor:
    """Extracts task information from codebase"""
    
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
            'assembly': [
                r'assemble', r'insert', r'connect', r'join', r'fit'
            ]
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
        
        return tasks
    
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
        
        # Extract parameters
        parameters = self._extract_parameters(func_content)
        
        # Extract dependencies
        dependencies = self._extract_dependencies(func_content)
        
        # Estimate complexity
        complexity = self._estimate_complexity(func_content)
        
        # Estimate duration (simplified)
        estimated_duration = self._estimate_duration(func_content)
        
        return TaskInfo(
            name=func_name,
            description=self._generate_description(func_name, func_content, task_type),
            task_type=task_type,
            required_actions=self._extract_actions(func_content),
            parameters=parameters,
            dependencies=dependencies,
            estimated_duration=estimated_duration,
            complexity=complexity
        )
    
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
            'timeout': r'timeout|time_limit|duration'
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
    
    def _extract_actions(self, func_content: str) -> List[str]:
        """Extract required actions from function content"""
        actions = []
        
        action_patterns = {
            'move': r'move|navigate|go_to',
            'grasp': r'grasp|grip|pick',
            'release': r'release|drop|place',
            'rotate': r'rotate|turn|spin',
            'wait': r'wait|sleep|delay',
            'plan': r'plan|planning|compute_path'
        }
        
        for action, pattern in action_patterns.items():
            if re.search(pattern, func_content, re.IGNORECASE):
                actions.append(action)
        
        return actions
    
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

class CodebaseAnalyzer:
    """Main analyzer class that coordinates the extraction process"""
    
    def __init__(self):
        self.kinematic_parser = KinematicParser()
        self.task_extractor = TaskExtractor()
        self.logger = logging.getLogger(__name__)
    
    def analyze_codebase(self, codebase_path: str, repository_url: str = None) -> CodebaseAnalysis:
        """Analyze a complete codebase"""
        self.logger.info(f"Starting analysis of codebase: {codebase_path}")
        
        # Find URDF files and parse kinematics
        urdf_files = self.kinematic_parser.find_urdf_files(codebase_path)
        robot_kinematics = None
        
        if urdf_files:
            self.logger.info(f"Found {len(urdf_files)} URDF files")
            # Use the first URDF file found
            robot_kinematics = self.kinematic_parser.parse_urdf_file(urdf_files[0])
        
        # Extract tasks
        tasks = self.task_extractor.extract_tasks_from_code(codebase_path)
        
        # Find configuration files
        config_files = self._find_config_files(codebase_path)
        
        # Find dependencies
        dependencies = self._find_dependencies(codebase_path)
        
        # Find test files
        test_files = self._find_test_files(codebase_path)
        
        # Find documentation files
        doc_files = self._find_documentation_files(codebase_path)
        
        return CodebaseAnalysis(
            repository_url=repository_url or '',
            local_path=codebase_path,
            robot_kinematics=robot_kinematics,
            tasks=tasks,
            configuration_files=config_files,
            dependencies=dependencies,
            test_files=test_files,
            documentation_files=doc_files
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
    
    def _find_documentation_files(self, codebase_path: str) -> List[str]:
        """Find documentation files"""
        doc_files = []
        doc_extensions = ['.md', '.rst', '.txt', '.pdf']
        
        for root, dirs, files in os.walk(codebase_path):
            for file in files:
                if any(file.endswith(ext) for ext in doc_extensions):
                    doc_files.append(os.path.join(root, file))
        
        return doc_files

class YAMLGenerator:
    """Generates YAML task sequences"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def generate_task_yaml(self, analysis: CodebaseAnalysis, output_path: str):
        """Generate YAML file with task sequences"""
        yaml_data = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'repository_url': analysis.repository_url,
                'local_path': analysis.local_path,
                'total_tasks': len(analysis.tasks)
            },
            'robot_kinematics': self._serialize_kinematics(analysis.robot_kinematics),
            'tasks': [self._serialize_task(task) for task in analysis.tasks],
            'configuration_files': analysis.configuration_files,
            'dependencies': analysis.dependencies,
            'test_files': analysis.test_files,
            'documentation_files': analysis.documentation_files
        }
        
        with open(output_path, 'w') as f:
            yaml.dump(yaml_data, f, default_flow_style=False, indent=2)
        
        self.logger.info(f"Generated YAML file: {output_path}")
    
    def _serialize_kinematics(self, kinematics: Optional[RobotKinematics]) -> Optional[Dict]:
        """Serialize kinematics information"""
        if kinematics is None:
            return None
        
        return {
            'name': kinematics.name,
            'dof': kinematics.dof,
            'base_link': kinematics.base_link,
            'end_effector': kinematics.end_effector,
            'urdf_path': kinematics.urdf_path,
            'joints': [asdict(joint) for joint in kinematics.joints],
            'links': [asdict(link) for link in kinematics.links]
        }
    
    def _serialize_task(self, task: TaskInfo) -> Dict:
        """Serialize task information"""
        return asdict(task)

class READMEGenerator:
    """Generates README documentation"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def generate_readme(self, analysis: CodebaseAnalysis, output_path: str):
        """Generate README file with task documentation"""
        readme_content = self._generate_readme_content(analysis)
        
        with open(output_path, 'w') as f:
            f.write(readme_content)
        
        self.logger.info(f"Generated README file: {output_path}")
    
    def _generate_readme_content(self, analysis: CodebaseAnalysis) -> str:
        """Generate README content"""
        content = f"""# Robot Task Analysis Report

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overview

This report contains the analysis of robot kinematics and task sequences extracted from the codebase.

## Repository Information

- **Repository URL**: {analysis.repository_url or 'N/A'}
- **Local Path**: {analysis.local_path}
- **Total Tasks Found**: {len(analysis.tasks)}

## Robot Kinematics

"""
        
        if analysis.robot_kinematics:
            kin = analysis.robot_kinematics
            content += f"""
### Robot: {kin.name}

- **Degrees of Freedom**: {kin.dof}
- **Base Link**: {kin.base_link}
- **End Effector**: {kin.end_effector or 'Not identified'}
- **URDF File**: {kin.urdf_path}

#### Joints ({len(kin.joints)} total)

"""
            for joint in kin.joints:
                content += f"- **{joint.name}**: {joint.joint_type} joint"
                if joint.limits:
                    content += f" (limits: {joint.limits['lower']:.2f} to {joint.limits['upper']:.2f})"
                content += f" - Axis: {joint.axis}\n"
        else:
            content += "No robot kinematics information found.\n"
        
        content += f"""
## Tasks ({len(analysis.tasks)} found)

"""
        
        for i, task in enumerate(analysis.tasks, 1):
            content += f"""
### Task {i}: {task.name}

- **Type**: {task.task_type}
- **Complexity**: {task.complexity}
- **Estimated Duration**: {task.estimated_duration:.1f} seconds
- **Description**: {task.description}

#### Required Actions
"""
            for action in task.required_actions:
                content += f"- {action}\n"
            
            if task.parameters:
                content += "\n#### Parameters\n"
                for param, value in task.parameters.items():
                    content += f"- {param}: {value}\n"
            
            if task.dependencies:
                content += "\n#### Dependencies\n"
                for dep in task.dependencies:
                    content += f"- {dep}\n"
            
            content += "\n---\n"
        
        content += f"""
## Project Structure

### Configuration Files ({len(analysis.configuration_files)})
"""
        for config_file in analysis.configuration_files:
            content += f"- {config_file}\n"
        
        content += f"""
### Dependencies ({len(analysis.dependencies)})
"""
        for dep in analysis.dependencies:
            content += f"- {dep}\n"
        
        content += f"""
### Test Files ({len(analysis.test_files)})
"""
        for test_file in analysis.test_files:
            content += f"- {test_file}\n"
        
        content += f"""
### Documentation Files ({len(analysis.documentation_files)})
"""
        for doc_file in analysis.documentation_files:
            content += f"- {doc_file}\n"
        
        content += """
## Usage

This analysis can be used for:
1. **Unit Testing**: Use the task sequences for automated testing
2. **Simulation Testing**: Implement tasks in simulation environments
3. **Documentation**: Reference for robot capabilities and tasks
4. **Development**: Guide for implementing new features

## Notes

- This analysis was generated automatically
- Task complexity and duration are estimates
- Some tasks may require additional parameters not detected
- Always validate task sequences before deployment
"""
        
        return content

class GUI:
    """Graphical user interface for the kinematic task extractor"""
    
    def __init__(self):
        if not GUI_AVAILABLE:
            raise ImportError("tkinter not available")
        
        self.root = tk.Tk()
        self.root.title("Kinematic Task Extractor")
        self.root.geometry("800x600")
        
        self.analyzer = CodebaseAnalyzer()
        self.yaml_generator = YAMLGenerator()
        self.readme_generator = READMEGenerator()
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the user interface"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(3, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="Kinematic Task Extractor", font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Input section
        input_frame = ttk.LabelFrame(main_frame, text="Input", padding="10")
        input_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        input_frame.columnconfigure(1, weight=1)
        
        # Codebase path
        ttk.Label(input_frame, text="Codebase Path:").grid(row=0, column=0, sticky=tk.W)
        self.path_var = tk.StringVar()
        path_entry = ttk.Entry(input_frame, textvariable=self.path_var, width=50)
        path_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(10, 5))
        ttk.Button(input_frame, text="Browse", command=self.browse_path).grid(row=0, column=2)
        
        # Repository URL
        ttk.Label(input_frame, text="Repository URL:").grid(row=1, column=0, sticky=tk.W, pady=(10, 0))
        self.url_var = tk.StringVar()
        url_entry = ttk.Entry(input_frame, textvariable=self.url_var, width=50)
        url_entry.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=(10, 5), pady=(10, 0))
        
        # Output directory
        ttk.Label(input_frame, text="Output Directory:").grid(row=2, column=0, sticky=tk.W, pady=(10, 0))
        self.output_var = tk.StringVar()
        output_entry = ttk.Entry(input_frame, textvariable=self.output_var, width=50)
        output_entry.grid(row=2, column=1, sticky=(tk.W, tk.E), padx=(10, 5), pady=(10, 0))
        ttk.Button(input_frame, text="Browse", command=self.browse_output).grid(row=2, column=2, pady=(10, 0))
        
        # Control buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=2, column=0, columnspan=2, pady=(10, 0))
        
        self.analyze_button = ttk.Button(button_frame, text="Analyze Codebase", command=self.analyze_codebase)
        self.analyze_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.generate_button = ttk.Button(button_frame, text="Generate Files", command=self.generate_files, state=tk.DISABLED)
        self.generate_button.pack(side=tk.LEFT)
        
        # Progress and log
        log_frame = ttk.LabelFrame(main_frame, text="Progress & Log", padding="10")
        log_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10, 0))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(log_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Log text
        self.log_text = scrolledtext.ScrolledText(log_frame, height=15, width=80)
        self.log_text.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # Store analysis result
        self.analysis_result = None
    
    def browse_path(self):
        """Browse for codebase path"""
        path = filedialog.askdirectory(title="Select Codebase Directory")
        if path:
            self.path_var.set(path)
    
    def browse_output(self):
        """Browse for output directory"""
        path = filedialog.askdirectory(title="Select Output Directory")
        if path:
            self.output_var.set(path)
    
    def log_message(self, message: str):
        """Add message to log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
        self.root.update()
    
    def analyze_codebase(self):
        """Analyze the codebase"""
        path = self.path_var.get().strip()
        url = self.url_var.get().strip()
        
        if not path:
            messagebox.showerror("Error", "Please select a codebase path")
            return
        
        if not os.path.exists(path):
            messagebox.showerror("Error", "Selected path does not exist")
            return
        
        # Disable analyze button
        self.analyze_button.config(state=tk.DISABLED)
        self.status_var.set("Analyzing...")
        self.progress_var.set(0)
        
        try:
            self.log_message("Starting codebase analysis...")
            self.progress_var.set(10)
            
            # Analyze codebase
            self.analysis_result = self.analyzer.analyze_codebase(path, url)
            
            self.progress_var.set(100)
            self.log_message(f"Analysis complete!")
            self.log_message(f"Found {len(self.analysis_result.tasks)} tasks")
            if self.analysis_result.robot_kinematics:
                self.log_message(f"Robot: {self.analysis_result.robot_kinematics.name} ({self.analysis_result.robot_kinematics.dof} DOF)")
            
            # Enable generate button
            self.generate_button.config(state=tk.NORMAL)
            self.status_var.set("Analysis complete - Ready to generate files")
            
        except Exception as e:
            self.log_message(f"Error during analysis: {e}")
            messagebox.showerror("Error", f"Analysis failed: {e}")
            self.status_var.set("Analysis failed")
        finally:
            self.analyze_button.config(state=tk.NORMAL)
    
    def generate_files(self):
        """Generate YAML and README files"""
        if not self.analysis_result:
            messagebox.showerror("Error", "No analysis result available")
            return
        
        output_dir = self.output_var.get().strip()
        if not output_dir:
            messagebox.showerror("Error", "Please select output directory")
            return
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        try:
            self.log_message("Generating files...")
            self.progress_var.set(0)
            
            # Generate YAML file
            yaml_path = os.path.join(output_dir, "task_analysis.yaml")
            self.yaml_generator.generate_task_yaml(self.analysis_result, yaml_path)
            self.progress_var.set(50)
            
            # Generate README file
            readme_path = os.path.join(output_dir, "README.md")
            self.readme_generator.generate_readme(self.analysis_result, readme_path)
            self.progress_var.set(100)
            
            self.log_message(f"Files generated successfully!")
            self.log_message(f"YAML: {yaml_path}")
            self.log_message(f"README: {readme_path}")
            
            self.status_var.set("Files generated successfully")
            
            # Open output directory
            if messagebox.askyesno("Success", "Files generated successfully. Open output directory?"):
                if sys.platform == "win32":
                    os.startfile(output_dir)
                elif sys.platform == "darwin":
                    subprocess.run(["open", output_dir])
                else:
                    subprocess.run(["xdg-open", output_dir])
            
        except Exception as e:
            self.log_message(f"Error generating files: {e}")
            messagebox.showerror("Error", f"File generation failed: {e}")
            self.status_var.set("File generation failed")
    
    def run(self):
        """Run the GUI"""
        self.root.mainloop()

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
    parser = argparse.ArgumentParser(description="Kinematic Task Extractor")
    parser.add_argument("--codebase", "-c", help="Path to codebase directory")
    parser.add_argument("--repository", "-r", help="Repository URL to clone")
    parser.add_argument("--output", "-o", help="Output directory for generated files")
    parser.add_argument("--gui", "-g", action="store_true", help="Launch GUI interface")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    if args.gui:
        if not GUI_AVAILABLE:
            print("Error: GUI not available (tkinter not installed)")
            return
        
        app = GUI()
        app.run()
        return
    
    # Command line mode
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
    analyzer = CodebaseAnalyzer()
    yaml_generator = YAMLGenerator()
    readme_generator = READMEGenerator()
    
    print(f"Analyzing codebase: {codebase_path}")
    analysis = analyzer.analyze_codebase(codebase_path, repo_url)
    
    # Generate output files
    output_dir = args.output or "output"
    os.makedirs(output_dir, exist_ok=True)
    
    yaml_path = os.path.join(output_dir, "task_analysis.yaml")
    readme_path = os.path.join(output_dir, "README.md")
    
    print("Generating YAML file...")
    yaml_generator.generate_task_yaml(analysis, yaml_path)
    
    print("Generating README file...")
    readme_generator.generate_readme(analysis, readme_path)
    
    print(f"Analysis complete!")
    print(f"YAML file: {yaml_path}")
    print(f"README file: {readme_path}")
    print(f"Found {len(analysis.tasks)} tasks")
    if analysis.robot_kinematics:
        print(f"Robot: {analysis.robot_kinematics.name} ({analysis.robot_kinematics.dof} DOF)")

if __name__ == "__main__":
    main() 