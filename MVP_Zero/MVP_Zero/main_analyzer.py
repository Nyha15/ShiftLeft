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
from typing import Dict, List, Any
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
        
        # Extract tasks
        self.logger.info("Extracting tasks...")
        tasks = self.task_extractor.extract_tasks(repo_path)
        self.logger.info(f"Found {len(tasks)} tasks")
        
        # Find configuration files
        config_files = self._find_config_files(repo_path)
        
        # Generate summary
        summary = self._generate_summary(robots, tasks, config_files)
        
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
    
    def _generate_summary(self, robots: List[RobotKinematics], tasks: List[TaskInfo], config_files: List[str]) -> Dict[str, Any]:
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
            'avg_task_complexity': self._calculate_avg_complexity(tasks)
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
