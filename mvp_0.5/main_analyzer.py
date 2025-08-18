#!/usr/bin/env python3
"""
Main Robotics Analyzer
======================

Orchestrates kinematic and task analysis to provide comprehensive robotics repository insights.
"""

import re
import logging
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import asdict

from data_models import AnalysisResult, RobotKinematics, TaskInfo
from kinematic_analyzer import KinematicAnalyzer
from task_extractor import TaskExtractor

logger = logging.getLogger(__name__)

class RoboticsAnalyzer:
    """Main analyzer that orchestrates kinematic and task analysis"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.RoboticsAnalyzer")
        self.kinematic_analyzer = KinematicAnalyzer()
        self.task_extractor = TaskExtractor()
    
    def analyze_repository(self, repo_path: str) -> AnalysisResult:
        """Analyze a robotics repository comprehensively"""
        start_time = datetime.now()
        self.logger.info(f"Starting analysis of {repo_path}")
        
        repo_path = Path(repo_path).resolve()
        
        # Analyze robot kinematics
        self.logger.info("Analyzing robot kinematics...")
        config_files = self.kinematic_analyzer.discover_config_files(repo_path)
        
        robots = []
        for config_file in config_files:
            robot_info = self.kinematic_analyzer.parse_urdf_file(config_file)
            if robot_info:
                robots.append(robot_info)
                self.logger.info(f"Parsed robot: {robot_info.name} ({robot_info.dof} DOF)")
        
        # If no robots found in URDF files, try to extract from README
        if not robots:
            self.logger.info("No robots found in URDF files, checking README for robot information...")
            readme_robots = self._extract_robots_from_readme(repo_path)
            if readme_robots:
                robots.extend(readme_robots)
                self.logger.info(f"Found {len(readme_robots)} robots from README")
        
        # Extract tasks
        self.logger.info("Extracting tasks...")
        tasks = self.task_extractor.extract_tasks(repo_path)
        self.logger.info(f"Found {len(tasks)} tasks")
        
        # Extract README information
        self.logger.info("Extracting README information...")
        readme_info = self._extract_readme_info(repo_path)
        
        # Find configuration files
        config_files = self._find_config_files(repo_path)
        
        # Generate summary
        summary = self._generate_summary(robots, tasks, config_files, readme_info)
        
        # Calculate overall confidence
        confidence = self._calculate_overall_confidence(robots, tasks)
        
        analysis_time = (datetime.now() - start_time).total_seconds()
        
        result = AnalysisResult(
            repository_path=str(repo_path),
            robots=robots,
            tasks=tasks,
            config_files=config_files,
            summary=summary,
            confidence=confidence,
            analysis_time=datetime.now().isoformat()
        )
        
        self.logger.info(f"Analysis complete in {analysis_time:.2f}s")
        return result
    
    def _find_config_files(self, repo_path: Path) -> List[str]:
        """Find robotics-related configuration files"""
        config_files = []
        patterns = ['*.yaml', '*.yml', '*.json', '*.xml', '*.cfg', '*.ini']
        
        robotics_keywords = [
            'robot', 'joint', 'link', 'control', 'config', 'param',
            'mujoco', 'pybullet', 'ros', 'gazebo'
        ]
        
        for pattern in patterns:
            for file_path in repo_path.rglob(pattern):
                try:
                    content = file_path.read_text(encoding='utf-8', errors='ignore').lower()
                    if any(keyword in content for keyword in robotics_keywords):
                        config_files.append(str(file_path))
                except:
                    continue
        
        return config_files
    
    def _extract_readme_info(self, repo_path: Path) -> Dict[str, Any]:
        """Extract information from README files"""
        readme_info = {}
        
        # Look for README files
        readme_patterns = ['README*', 'readme*', '*.md']
        readme_files = []
        
        for pattern in readme_patterns:
            readme_files.extend(repo_path.glob(pattern))
        
        if not readme_files:
            return readme_info
        
        # Read the main README
        main_readme = None
        for readme in readme_files:
            if 'README' in readme.name.upper():
                main_readme = readme
                break
        
        if not main_readme:
            main_readme = readme_files[0]  # Use first markdown file
        
        try:
            content = main_readme.read_text(encoding='utf-8', errors='ignore')
            readme_info = self._parse_readme_content(content)
            readme_info['source_file'] = str(main_readme)
        except Exception as e:
            self.logger.warning(f"Failed to read README {main_readme}: {e}")
        
        return readme_info
    
    def _parse_readme_content(self, content: str) -> Dict[str, Any]:
        """Parse README content to extract relevant information"""
        info = {}
        content_lower = content.lower()
        
        # Extract hardware information
        hardware_info = {}
        if 'franka' in content_lower or 'panda' in content_lower:
            hardware_info['robot'] = 'Franka Panda'
        if 'realsense' in content_lower or 'intel' in content_lower:
            hardware_info['camera'] = 'Intel RealSense'
        if 'april' in content_lower or 'tag' in content_lower:
            hardware_info['markers'] = 'April Tags'
        
        if hardware_info:
            info['hardware'] = hardware_info
        
        # Extract software information
        software_info = {}
        if 'frankapy' in content_lower:
            software_info['frankapy'] = True
        if 'isaac' in content_lower or 'gym' in content_lower:
            software_info['isaac_gym'] = True
        if 'ros' in content_lower:
            software_info['ros'] = True
        
        if software_info:
            info['software'] = software_info
        
        # Extract task information
        task_info = {}
        if 'pick' in content_lower:
            task_info['pick_tasks'] = True
        if 'place' in content_lower:
            task_info['place_tasks'] = True
        if 'insert' in content_lower:
            task_info['insert_tasks'] = True
        if 'reach' in content_lower:
            task_info['reach_tasks'] = True
        
        if task_info:
            info['tasks'] = task_info
        
        # Extract sequence information
        if 'sequence' in content_lower:
            info['sequences'] = True
        
        # Extract perception information
        if 'perception' in content_lower or 'detection' in content_lower:
            info['perception'] = True
        
        return info
    
    def _extract_robots_from_readme(self, repo_path: Path) -> List[RobotKinematics]:
        """Extract robot information from README when URDF files are not available"""
        robots = []
        
        # Look for README files
        readme_patterns = ['README*', 'readme*', '*.md']
        readme_files = []
        
        for pattern in readme_patterns:
            readme_files.extend(repo_path.glob(pattern))
        
        if not readme_files:
            return robots
        
        # Read the main README
        main_readme = None
        for readme in readme_files:
            if 'README' in readme.name.upper():
                main_readme = readme
                break
        
        if not main_readme:
            main_readme = readme_files[0]
        
        try:
            content = main_readme.read_text(encoding='utf-8', errors='ignore')
            content_lower = content.lower()
            
            # Check for Franka robot
            if 'franka' in content_lower or 'panda' in content_lower:
                # Franka Panda is typically 7 DOF
                robot = RobotKinematics(
                    name='Franka Panda',
                    dof=7,
                    joint_names=['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'joint7'],
                    joint_types=['revolute'] * 7,
                    joint_limits=[(-2.8973, 2.8973)] * 7,  # Approximate limits
                    mass=0.0,  # Unknown
                    source_file=str(main_readme),
                    confidence=0.7
                )
                robots.append(robot)
            
            # Check for other common robots
            if 'ur5' in content_lower:
                robot = RobotKinematics(
                    name='UR5',
                    dof=6,
                    joint_names=['shoulder_pan', 'shoulder_lift', 'elbow', 'wrist_1', 'wrist_2', 'wrist_3'],
                    joint_types=['revolute'] * 6,
                    joint_limits=[(-2*3.14159, 2*3.14159)] * 6,
                    mass=0.0,
                    source_file=str(main_readme),
                    confidence=0.7
                )
                robots.append(robot)
            
            # Add more robot types as needed
            
        except Exception as e:
            self.logger.warning(f"Failed to extract robots from README {main_readme}: {e}")
        
        return robots
    
    def _generate_summary(self, robots: List[RobotKinematics], tasks: List[TaskInfo], config_files: List[str], readme_info: Dict[str, Any]) -> Dict[str, Any]:
        """Generate analysis summary"""
        task_types = {}
        for task in tasks:
            task_types[task.task_type] = task_types.get(task.task_type, 0) + 1
        
        return {
            'total_robots': len(robots),
            'total_tasks': len(tasks),
            'total_config_files': len(config_files),
            'task_types': task_types,
            'robot_names': [robot.name for robot in robots],
            'total_dof': sum(robot.dof for robot in robots),
            'avg_task_complexity': self._calculate_avg_complexity(tasks),
            'readme_info': readme_info
        }
    
    def _calculate_avg_complexity(self, tasks: List[TaskInfo]) -> str:
        """Calculate average task complexity"""
        if not tasks:
            return 'unknown'
        
        complexity_scores = {'simple': 1, 'medium': 2, 'complex': 3}
        avg_score = sum(complexity_scores.get(task.complexity, 1) for task in tasks) / len(tasks)
        
        if avg_score < 1.5:
            return 'simple'
        elif avg_score < 2.5:
            return 'medium'
        else:
            return 'complex'
    
    def _calculate_overall_confidence(self, robots: List[RobotKinematics], tasks: List[TaskInfo]) -> float:
        """Calculate overall analysis confidence"""
        if not robots and not tasks:
            return 0.0
        
        robot_confidence = 0.9 if robots else 0.0
        task_confidence = sum(task.confidence for task in tasks) / len(tasks) if tasks else 0.0
        
        # Weight robots higher as they're more definitive
        return (robot_confidence * 0.6 + task_confidence * 0.4)
    
    def save_results(self, result: AnalysisResult, output_dir: Path):
        """Save analysis results to industry-ready format with individual task files and summary"""
        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save individual task YAML files
        tasks_dir = output_dir / "tasks"
        tasks_dir.mkdir(exist_ok=True)
        
        for task in result.tasks:
            self._save_task_yaml(task, tasks_dir)
        
        # Save robot specifications
        if result.robots:
            robots_dir = output_dir / "robots"
            robots_dir.mkdir(exist_ok=True)
            
            for robot in result.robots:
                robot_file = robots_dir / f"{robot.name}.yaml"
                self._save_robot_yaml(robot, robot_file)
        
        # Save comprehensive summary document
        summary_file = output_dir / "ANALYSIS_SUMMARY.md"
        self._save_summary_document(result, summary_file)
        
        # Save machine-readable summary
        summary_yaml = output_dir / "summary.yaml"
        self._save_summary_yaml(result, summary_yaml)
        
        self.logger.info(f"Industry-ready results saved to {output_dir}")
        self.logger.info(f"  • {len(result.tasks)} task files in tasks/")
        self.logger.info(f"  • {len(result.robots)} robot files in robots/")
        self.logger.info(f"  • Summary document: ANALYSIS_SUMMARY.md")
        self.logger.info(f"  • Machine summary: summary.yaml")
    
    def _save_task_yaml(self, task: TaskInfo, output_dir: Path) -> None:
        """Save individual task as comprehensive YAML file suitable for simulation/test generation."""
        task_data = {
            'task_info': {
                'name': task.name,
                'type': task.task_type,
                'description': task.description,
                'complexity': task.complexity,
                'confidence': task.confidence,
                'source_file': task.file_path
            },
            'execution': {
                'estimated_duration': task.estimated_duration,
                'parameters': task.parameters,
                'dependencies': task.dependencies,
                'required_actions': task.required_actions
            },
            'code_analysis': {
                'function_calls': task.dependencies.get('function_calls', []) if isinstance(task.dependencies, dict) else [],
                'imports': task.dependencies.get('imports', []) if isinstance(task.dependencies, dict) else [],
                'external_libraries': task.dependencies.get('external_libraries', []) if isinstance(task.dependencies, dict) else [],
                'robotics_libraries': task.dependencies.get('robotics_libraries', []) if isinstance(task.dependencies, dict) else [],
                'constants': task.dependencies.get('constants', []) if isinstance(task.dependencies, dict) else [],
                'variables': task.dependencies.get('variables', []) if isinstance(task.dependencies, dict) else []
            },
            'simulation_requirements': {
                'physics_engine': self._detect_physics_engine(task),
                'robot_models': self._extract_robot_models(task),
                'environment_files': self._extract_environment_files(task),
                'sensor_requirements': self._extract_sensor_requirements(task)
            },
            'assets': {
                'source_code': task.file_path,
                'related_configs': []  # TODO: Link to related config files
            },
            'code_definitions': self._extract_task_code_definitions(task, Path(task.file_path).parent),  # Full code definitions
            'metadata': {
                'analysis_time': datetime.now().isoformat(),
                'analyzer_version': '1.0.0',
                'repository_path': 'unknown'
            }
        }
        
        task_file = output_dir / f'{task.name}.yaml'
        task_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(task_file, 'w') as f:
            yaml.dump(task_data, f, default_flow_style=False, sort_keys=False)
    
    def _detect_physics_engine(self, task: TaskInfo) -> str:
        """Detect physics engine used in the task."""
        content = task.description.lower() + str(task.dependencies).lower()
        
        if 'mujoco' in content or 'mjx' in content:
            return 'MuJoCo'
        elif 'pybullet' in content or 'bullet' in content:
            return 'PyBullet'
        elif 'gazebo' in content:
            return 'Gazebo'
        elif 'isaac' in content:
            return 'Isaac Gym/Sim'
        elif 'drake' in content:
            return 'Drake'
        else:
            return 'unknown'
    
    def _extract_robot_models(self, task: TaskInfo) -> List[str]:
        """Extract robot models mentioned in the task."""
        models = []
        content = task.description.lower() + str(task.dependencies).lower()
        
        robot_patterns = [
            r'panda', r'franka', r'ur5', r'ur10', r'kuka', r'abb', r'baxter',
            r'pr2', r'turtlebot', r'husky', r'jackal', r'fetch', r'spot'
        ]
        
        for pattern in robot_patterns:
            if pattern in content:
                models.append(pattern.upper())
        
        return models
    
    def _extract_environment_files(self, task: TaskInfo) -> List[str]:
        """Extract environment/scene files mentioned in the task."""
        files = []
        content = str(task.dependencies)
        
        # Look for common environment file patterns
        env_patterns = [
            r'scene\.xml', r'world\.xml', r'\.urdf', r'\.sdf',
            r'environment', r'scene', r'world'
        ]
        
        for pattern in env_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                files.append(pattern)
        
        return files
    
    def _extract_sensor_requirements(self, task: TaskInfo) -> List[str]:
        """Extract sensor requirements from the task."""
        sensors = []
        content = task.description.lower() + str(task.dependencies).lower()
        
        sensor_patterns = {
            'camera': ['camera', 'vision', 'image', 'rgb'],
            'lidar': ['lidar', 'laser', 'scan'],
            'imu': ['imu', 'accelerometer', 'gyroscope'],
            'force_torque': ['force', 'torque', 'ft_sensor'],
            'tactile': ['tactile', 'touch', 'contact'],
            'depth': ['depth', 'rgbd', 'pointcloud']
        }
        
        for sensor_type, keywords in sensor_patterns.items():
            if any(keyword in content for keyword in keywords):
                sensors.append(sensor_type)
        
        return sensors
    
    def _extract_task_code_definitions(self, task: TaskInfo, repo_path: Path) -> Dict[str, str]:
        """Extract full code definitions for functions and classes relevant to the task"""
        code_definitions = {}
        
        try:
            # Find the actual repository directory
            repo_root = repo_path
            while repo_root.name != 'industreallib' and repo_root.parent != repo_root:
                repo_root = repo_root.parent
            
            if repo_root.name != 'industreallib':
                self.logger.warning("Could not find industreallib root directory")
                return code_definitions
            
            # Extract relevant code based on task type
            if 'task' in task.name.lower():
                # Extract task class code
                class_code = self._extract_class_code(repo_root, task.name)
                if class_code:
                    code_definitions[task.name] = class_code
                    self.logger.info(f"Extracted {task.name} class code")
                
                # Also extract base class if it exists
                if 'base' not in task.name.lower():
                    base_class_code = self._extract_class_code(repo_root, 'IndustRealTaskBase')
                    if base_class_code:
                        code_definitions['IndustRealTaskBase'] = base_class_code
                        self.logger.info("Extracted IndustRealTaskBase class code")
            
            # Extract dependent functions that the task needs
            dependent_functions = self._extract_dependent_functions(task, repo_root)
            if dependent_functions:
                code_definitions.update(dependent_functions)
                self.logger.info(f"Extracted {len(dependent_functions)} dependent functions")
            
            # Add a README appendix with parameter summary
            readme_appendix = self._generate_readme_appendix(task, code_definitions)
            if readme_appendix:
                code_definitions['README_APPENDIX'] = readme_appendix
                self.logger.info("Generated README appendix")
            
            self.logger.info(f"Extracted {len(code_definitions)} code definitions for {task.name}")
            
        except Exception as e:
            self.logger.warning(f"Failed to extract code definitions for {task.name}: {e}")
        
        return code_definitions
    
    def _extract_dependent_functions(self, task: TaskInfo, repo_root: Path) -> Dict[str, str]:
        """Extract the actual function implementations that the task depends on"""
        dependent_functions = {}
        
        try:
            # Get function calls from task dependencies
            if isinstance(task.dependencies, dict) and 'function_calls' in task.dependencies:
                function_calls = task.dependencies['function_calls']
                
                for func_name in function_calls:
                    # Skip built-in functions and methods
                    if func_name in ['__init__', 'copy', 'tolist', 'detach', 'reset', 'sleep', 'publish']:
                        continue
                    
                    # Extract the actual function implementation
                    func_code = self._extract_function_code(repo_root, func_name)
                    if func_code:
                        dependent_functions[f'function_{func_name}'] = func_code
                        self.logger.info(f"Extracted function: {func_name}")
            
            # Also extract utility functions that are commonly used
            utility_functions = [
                'go_upward', 'go_home', 'get_pose', 'set_sigint_response',
                'compose_ros_msg', 'print_pose_error', 'perturb_yaw'
            ]
            
            for util_func in utility_functions:
                util_code = self._extract_function_code(repo_root, util_func)
                if util_code:
                    dependent_functions[f'utility_{util_func}'] = util_code
                    self.logger.info(f"Extracted utility function: {util_func}")
            
        except Exception as e:
            self.logger.warning(f"Failed to extract dependent functions: {e}")
        
        return dependent_functions
    
    def _extract_function_code(self, repo_path: Path, function_name: str) -> Optional[str]:
        """Extract the full code definition of a function from the repository"""
        try:
            # Search for Python files containing the function
            python_files = list(repo_path.rglob("*.py"))
            
            for file_path in python_files:
                try:
                    content = file_path.read_text(encoding='utf-8', errors='ignore')
                    
                    # Look for function definition
                    if f"def {function_name}" in content:
                        # Parse the file to extract the function
                        import ast
                        tree = ast.parse(content)
                        
                        for node in ast.walk(tree):
                            if isinstance(node, ast.FunctionDef) and node.name == function_name:
                                # Get the function source code
                                start_line = node.lineno
                                end_line = node.end_lineno
                                
                                lines = content.split('\n')
                                function_lines = lines[start_line-1:end_line]
                                
                                # Clean and format the code
                                cleaned_code = self._clean_extracted_code(function_lines)
                                
                                # Add file header
                                header = f"# Function extracted from: {file_path.relative_to(repo_path)}\n"
                                return header + cleaned_code
                                
                except Exception as e:
                    self.logger.debug(f"Failed to parse {file_path}: {e}")
                    continue
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Failed to extract function code for {function_name}: {e}")
            return None
    
    def _extract_class_code(self, repo_path: Path, class_name: str) -> Optional[str]:
        """Extract the full code definition of a class from the repository"""
        try:
            # Search for Python files containing the class
            python_files = list(repo_path.rglob("*.py"))
            
            for file_path in python_files:
                try:
                    content = file_path.read_text(encoding='utf-8', errors='ignore')
                    
                    # Look for class definition
                    if f"class {class_name}" in content:
                        # Parse the file to extract the class
                        import ast
                        tree = ast.parse(content)
                        
                        for node in ast.walk(tree):
                            if isinstance(node, ast.ClassDef) and node.name == class_name:
                                # Get the class source code
                                start_line = node.lineno
                                end_line = node.end_lineno
                                
                                lines = content.split('\n')
                                class_lines = lines[start_line-1:end_line]
                                
                                # Clean and format the code
                                cleaned_code = self._clean_extracted_code(class_lines)
                                
                                # Add file header
                                header = f"# Class extracted from: {file_path.relative_to(repo_path)}\n"
                                return header + cleaned_code
                                
                except Exception as e:
                    self.logger.debug(f"Failed to parse {file_path}: {e}")
                    continue
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Failed to extract class code for {class_name}: {e}")
            return None
    
    def _clean_extracted_code(self, code_lines: List[str]) -> str:
        """Clean and format extracted code for better readability"""
        try:
            # Join lines and clean up
            code = '\n'.join(code_lines)
            
            # Remove excessive whitespace and normalize indentation
            lines = code.split('\n')
            cleaned_lines = []
            
            for line in lines:
                # Remove trailing whitespace
                cleaned_line = line.rstrip()
                if cleaned_line:  # Keep non-empty lines
                    cleaned_lines.append(cleaned_line)
            
            # Rejoin with proper spacing
            return '\n'.join(cleaned_lines)
            
        except Exception as e:
            self.logger.debug(f"Failed to clean extracted code: {e}")
            # Return original if cleaning fails
            return '\n'.join(code_lines)
    
    def _generate_readme_appendix(self, task: TaskInfo, code_definitions: Dict[str, str]) -> str:
        """Generate a README appendix with parameter summary, functions, and dependencies"""
        try:
            readme = f"""# {task.name} Task - README Appendix

## Task Overview
- **Type**: {task.task_type}
- **Complexity**: {task.complexity}
- **Confidence**: {task.confidence}
- **Estimated Duration**: {task.estimated_duration} seconds
- **Source File**: {task.file_path}

## Description
{task.description}

## Required Actions
{chr(10).join(f"- {action}" for action in task.required_actions)}

## Dependencies
{chr(10).join(f"- {dep}" for dep in task.dependencies.get('imports', [])) if isinstance(task.dependencies, dict) and task.dependencies.get('imports') else "- None"}

## Key Parameters Summary
"""
            
            # Add parameters if available
            if task.parameters:
                for param_name, param_data in task.parameters.items():
                    readme += f"- **{param_name}**: {param_data.get('description', 'No description')}\n"
            
            # Add code definitions summary
            readme += f"\n## Code Definitions\n"
            for code_name, code_content in code_definitions.items():
                if code_name != 'README_APPENDIX':
                    readme += f"- **{code_name}**: {len(code_content.split(chr(10)))} lines of code\n"
            
            # Add simulation requirements
            readme += f"\n## Simulation Requirements\n"
            readme += f"- **Physics Engine**: {self._detect_physics_engine(task)}\n"
            readme += f"- **Robot Models**: {', '.join(self._extract_robot_models(task))}\n"
            
            # Add usage instructions
            readme += f"\n## Usage Instructions\n"
            readme += f"1. Ensure all dependencies are installed\n"
            readme += f"2. Configure parameters according to your environment\n"
            readme += f"3. Run the task using the provided execution framework\n"
            readme += f"4. Monitor execution and verify results\n"
            
            return readme
            
        except Exception as e:
            self.logger.warning(f"Failed to generate README appendix: {e}")
            return None
    
    def _save_robot_yaml(self, robot: RobotKinematics, robot_file: Path):
        """Save individual robot to YAML file with full specifications"""
        robot_data = {
            'robot_info': {
                'name': robot.name,
                'dof': robot.dof,
                'base_link': robot.base_link,
                'end_effector': robot.end_effector,
                'urdf_path': robot.urdf_path
            },
            'kinematics': {
                'joints': [asdict(joint) for joint in robot.joints],
                'links': [asdict(link) for link in robot.links]
            },
            'capabilities': self._infer_robot_capabilities(robot)
        }
        
        with open(robot_file, 'w') as f:
            yaml.dump(robot_data, f, default_flow_style=False, indent=2)
    
    def _save_summary_document(self, result: AnalysisResult, summary_file: Path):
        """Save comprehensive markdown summary document"""
        content = f"""# Robotics Repository Analysis Summary

**Repository:** `{result.repository_path}`  
**Analysis Date:** {result.analysis_time}  
**Overall Confidence:** {result.confidence:.2f}  
**Analyzer Version:** 1.0.0

## 📊 Executive Summary

This repository contains **{result.summary['total_tasks']} robotics tasks** and **{result.summary['total_robots']} robot specifications** with an average task complexity of **{result.summary['avg_task_complexity']}**.

### Key Findings
- **Total Robots:** {result.summary['total_robots']}
- **Total Tasks:** {result.summary['total_tasks']}
- **Configuration Files:** {result.summary['total_config_files']}
- **Total DOF:** {result.summary['total_dof']}
- **Average Task Complexity:** {result.summary['avg_task_complexity']}

## 🤖 Robot Specifications

"""
        
        if result.robots:
            for robot in result.robots:
                content += f"""### {robot.name}
- **DOF:** {robot.dof}
- **Base Link:** {robot.base_link}
- **End Effector:** {robot.end_effector or 'Not detected'}
- **URDF Path:** `{robot.urdf_path}`
- **Joints:** {len(robot.joints)}
- **Links:** {len(robot.links)}

"""
        else:
            content += "No robot specifications found in URDF files.\n\n"
        
        # Add README information if available
        if result.summary.get('readme_info'):
            content += "## 📖 Repository Information (from README)\n\n"
            readme_info = result.summary['readme_info']
            
            if readme_info.get('hardware'):
                content += "### Hardware Requirements\n"
                for hw_type, hw_info in readme_info['hardware'].items():
                    content += f"- **{hw_type.title()}:** {hw_info}\n"
                content += "\n"
            
            if readme_info.get('software'):
                content += "### Software Dependencies\n"
                for sw_type, sw_info in readme_info['software'].items():
                    if sw_info:
                        content += f"- **{sw_type.replace('_', ' ').title()}:** Required\n"
                content += "\n"
            
            if readme_info.get('tasks'):
                content += "### Supported Task Types\n"
                for task_type, task_info in readme_info['tasks'].items():
                    if task_info:
                        content += f"- **{task_type.replace('_', ' ').title()}:** Supported\n"
                content += "\n"
            
            if readme_info.get('sequences'):
                content += "### Sequence Support\n"
                content += "- **Sequences:** Supported\n\n"
            
            if readme_info.get('perception'):
                content += "### Perception Capabilities\n"
                content += "- **Object Detection:** Supported\n"
                content += "- **Camera Calibration:** Supported\n\n"
        
        content += "## 🎯 Task Analysis\n\n"
        
        if result.tasks:
            # Task breakdown by type
            content += "### Task Breakdown by Type\n\n"
            for task_type, count in result.summary['task_types'].items():
                content += f"- **{task_type.title()}:** {count} tasks\n"
            
            content += "\n### Individual Tasks\n\n"
            for task in sorted(result.tasks, key=lambda x: x.confidence, reverse=True):
                content += f"""#### {task.name}
- **Type:** {task.task_type}
- **Complexity:** {task.complexity}
- **Confidence:** {task.confidence:.2f}
- **Duration:** {task.estimated_duration:.1f}s
- **Actions:** {', '.join(task.required_actions)}
- **Source:** `{task.file_path}`
- **Description:** {task.description}

"""
        else:
            content += "No tasks detected in the repository.\n\n"
        
        content += f"""## 📁 Configuration Files

{len(result.config_files)} configuration files detected:

"""
        
        for config_file in result.config_files:
            content += f"- `{config_file}`\n"
        
        content += f"""\n## 🔍 Analysis Details

### Methodology
This analysis used advanced AST parsing for Python code and URDF/XML parsing for robot specifications. Tasks were filtered for robotics relevance and scored for confidence.

### Confidence Scoring
- **Robot Detection:** Based on URDF parsing success and completeness
- **Task Detection:** Based on robotics keyword matching and code complexity
- **Overall Confidence:** Weighted average favoring robot specifications

### Limitations
- Analysis limited to Python files and URDF/XML configurations
- Task detection relies on naming patterns and keyword matching
- Complex task relationships may not be fully captured

---

*Generated by Robotics Repository Analyzer MVP v1.0.0*
"""
        
        with open(summary_file, 'w') as f:
            f.write(content)
    
    def _save_summary_yaml(self, result: AnalysisResult, summary_file: Path):
        """Save machine-readable summary YAML"""
        summary_data = {
            'metadata': {
                'repository_path': result.repository_path,
                'analysis_time': result.analysis_time,
                'confidence': result.confidence,
                'analyzer_version': '1.0.0'
            },
            'summary': result.summary,
            'robot_capabilities': [],
            'task_breakdown': {
                'by_type': result.summary['task_types'],
                'by_complexity': self._get_complexity_breakdown(result.tasks),
                'by_confidence': self._get_confidence_breakdown(result.tasks)
            },
            'file_analysis': {
                'total_python_files': len([t for t in result.tasks]),
                'total_config_files': len(result.config_files),
                'total_urdf_files': len([r for r in result.robots])
            }
        }
        
        # Add robot capabilities
        for robot in result.robots:
            capabilities = self._infer_robot_capabilities(robot)
            summary_data['robot_capabilities'].extend(capabilities)
        
        with open(summary_file, 'w') as f:
            yaml.dump(summary_data, f, default_flow_style=False, indent=2)
    
    def _infer_robot_capabilities(self, robot: RobotKinematics) -> List[str]:
        """Infer robot capabilities from kinematic structure"""
        capabilities = []
        
        # DOF-based capabilities
        if robot.dof >= 6:
            capabilities.append(f"{robot.dof}-DOF manipulation")
        if robot.dof >= 7:
            capabilities.append("Redundant kinematics")
        
        # End effector capabilities
        if robot.end_effector:
            if 'gripper' in robot.end_effector.lower():
                capabilities.append("Parallel jaw gripper")
            elif 'hand' in robot.end_effector.lower():
                capabilities.append("Multi-finger manipulation")
            else:
                capabilities.append("End effector manipulation")
        
        # Joint type capabilities
        joint_types = [joint.joint_type for joint in robot.joints]
        if 'revolute' in joint_types:
            capabilities.append("Rotational joints")
        if 'prismatic' in joint_types:
            capabilities.append("Linear joints")
        
        return capabilities
    
    def _get_complexity_breakdown(self, tasks: List[TaskInfo]) -> Dict[str, int]:
        """Get task breakdown by complexity"""
        breakdown = {'simple': 0, 'medium': 0, 'complex': 0}
        for task in tasks:
            breakdown[task.complexity] += 1
        return breakdown
    
    def _get_confidence_breakdown(self, tasks: List[TaskInfo]) -> Dict[str, int]:
        """Get task breakdown by confidence ranges"""
        breakdown = {'high': 0, 'medium': 0, 'low': 0}
        for task in tasks:
            if task.confidence >= 0.8:
                breakdown['high'] += 1
            elif task.confidence >= 0.5:
                breakdown['medium'] += 1
            else:
                breakdown['low'] += 1
        return breakdown
