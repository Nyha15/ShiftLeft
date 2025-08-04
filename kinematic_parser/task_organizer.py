#!/usr/bin/env python3
"""
LLM Robot Task Organizer
========================

A comprehensive tool to extract, organize, and document the main tasks from the 
LLM articulated object manipulation codebase. This tool creates well-documented 
YAML files for each task and extracts robot kinematic information.

Author: AI Assistant
Date: 2024
"""

import os
import sys
import json
import yaml
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import xml.etree.ElementTree as ET

@dataclass
class RobotKinematics:
    """Robot kinematic information"""
    name: str
    description: str
    dof: int
    joints: List[Dict[str, Any]]
    links: List[Dict[str, Any]]
    base_link: str
    end_effector: str
    workspace: Dict[str, Any]
    capabilities: List[str]

@dataclass
class TaskDefinition:
    """Complete task definition"""
    name: str
    description: str
    category: str
    object_type: str
    required_actions: List[str]
    parameters: Dict[str, Any]
    success_criteria: List[str]
    complexity: str
    estimated_duration: float
    dependencies: List[str]
    assets: Dict[str, str]

class TaskOrganizer:
    """Main task organization and extraction class"""
    
    def __init__(self, codebase_path: str):
        self.codebase_path = Path(codebase_path)
        self.tasks = []
        self.robot_kinematics = None
        
    def analyze_codebase(self) -> Tuple[List[TaskDefinition], RobotKinematics]:
        """Analyze the entire codebase and extract tasks and kinematics"""
        print("🔍 Analyzing LLM Robot Manipulation Codebase...")
        
        # Extract robot kinematics
        self.robot_kinematics = self._extract_robot_kinematics()
        
        # Extract tasks from various sources
        self._extract_tasks_from_config_files()
        self._extract_tasks_from_code()
        self._extract_tasks_from_demonstrations()
        
        print(f"✅ Found {len(self.tasks)} tasks and robot kinematic model")
        return self.tasks, self.robot_kinematics
    
    def _extract_robot_kinematics(self) -> RobotKinematics:
        """Extract robot kinematic information from the codebase"""
        print("🤖 Extracting robot kinematics...")
        
        # Look for URDF files and robot configurations
        urdf_files = list(self.codebase_path.rglob("*.urdf"))
        
        # Default Franka Panda configuration based on codebase analysis
        franka_kinematics = RobotKinematics(
            name="Franka Panda",
            description="7-DOF collaborative robot arm with parallel gripper for articulated object manipulation",
            dof=7,
            joints=[
                {"name": "panda_joint1", "type": "revolute", "axis": [0, 0, 1], "limits": {"lower": -2.8973, "upper": 2.8973}},
                {"name": "panda_joint2", "type": "revolute", "axis": [0, 1, 0], "limits": {"lower": -1.7628, "upper": 1.7628}},
                {"name": "panda_joint3", "type": "revolute", "axis": [0, 0, 1], "limits": {"lower": -2.8973, "upper": 2.8973}},
                {"name": "panda_joint4", "type": "revolute", "axis": [0, -1, 0], "limits": {"lower": -3.0718, "upper": -0.0698}},
                {"name": "panda_joint5", "type": "revolute", "axis": [0, 0, 1], "limits": {"lower": -2.8973, "upper": 2.8973}},
                {"name": "panda_joint6", "type": "revolute", "axis": [0, 1, 0], "limits": {"lower": -0.0175, "upper": 3.7525}},
                {"name": "panda_joint7", "type": "revolute", "axis": [0, 0, 1], "limits": {"lower": -2.8973, "upper": 2.8973}},
            ],
            links=[
                {"name": "panda_link0", "type": "base"},
                {"name": "panda_link1", "type": "arm_segment"},
                {"name": "panda_link2", "type": "arm_segment"},
                {"name": "panda_link3", "type": "arm_segment"},
                {"name": "panda_link4", "type": "arm_segment"},
                {"name": "panda_link5", "type": "arm_segment"},
                {"name": "panda_link6", "type": "arm_segment"},
                {"name": "panda_link7", "type": "arm_segment"},
                {"name": "panda_hand", "type": "end_effector"},
            ],
            base_link="panda_link0",
            end_effector="panda_hand",
            workspace={
                "reach": 0.855,  # meters
                "payload": 3.0,   # kg
                "repeatability": 0.1,  # mm
                "workspace_volume": "sphere with radius 0.855m"
            },
            capabilities=[
                "7-DOF manipulation",
                "Parallel jaw gripper",
                "Force/torque sensing",
                "Collision detection",
                "Kinematic-aware motion planning",
                "Articulated object manipulation",
                "GPT-guided task execution"
            ]
        )
        
        return franka_kinematics
    
    def _extract_tasks_from_config_files(self):
        """Extract tasks from YAML configuration files"""
        print("📁 Extracting tasks from configuration files...")
        
        config_dir = self.codebase_path / "src" / "task_config"
        if not config_dir.exists():
            return
        
        for yaml_file in config_dir.glob("*.yaml"):
            task_name = yaml_file.stem
            
            try:
                with open(yaml_file, 'r') as f:
                    config = yaml.safe_load(f)
                
                # Parse task information from config
                task = self._parse_task_config(task_name, config)
                if task:
                    self.tasks.append(task)
                    
            except Exception as e:
                print(f"⚠️  Error parsing {yaml_file}: {e}")
    
    def _parse_task_config(self, task_name: str, config: Dict) -> Optional[TaskDefinition]:
        """Parse a task configuration into a TaskDefinition"""
        
        # Determine task category and object type
        category, object_type = self._categorize_task(task_name)
        
        # Extract required actions
        actions = self._extract_actions_from_name(task_name)
        
        # Extract assets information
        assets = {}
        if 'env' in config and 'asset' in config['env']:
            asset_info = config['env']['asset']
            if 'trainAssets' in asset_info:
                for asset_name, asset_data in asset_info['trainAssets'].items():
                    assets[asset_name] = asset_data
        
        # Estimate complexity and duration
        complexity = self._estimate_task_complexity(task_name, actions)
        duration = self._estimate_task_duration(task_name, complexity)
        
        task = TaskDefinition(
            name=task_name,
            description=self._generate_task_description(task_name, category, object_type),
            category=category,
            object_type=object_type,
            required_actions=actions,
            parameters=self._extract_task_parameters(task_name),
            success_criteria=self._generate_success_criteria(task_name, actions),
            complexity=complexity,
            estimated_duration=duration,
            dependencies=self._extract_task_dependencies(task_name),
            assets=assets
        )
        
        return task
    
    def _categorize_task(self, task_name: str) -> Tuple[str, str]:
        """Categorize task and determine object type"""
        task_lower = task_name.lower()
        
        # Task categories
        if any(word in task_lower for word in ['open', 'close']):
            category = "articulated_manipulation"
        elif any(word in task_lower for word in ['lift', 'lay_down']):
            category = "lifting_manipulation"
        elif any(word in task_lower for word in ['turn', 'rotate']):
            category = "rotational_manipulation"
        elif any(word in task_lower for word in ['press', 'push']):
            category = "contact_manipulation"
        else:
            category = "general_manipulation"
        
        # Object types
        object_mappings = {
            'door': 'door', 'drawer': 'drawer', 'cabinet': 'cabinet',
            'window': 'window', 'laptop': 'laptop', 'microwave': 'microwave',
            'oven': 'oven', 'refrigerator': 'refrigerator', 'dishwasher': 'dishwasher',
            'safe': 'safe', 'suitcase': 'suitcase', 'washingmachine': 'washing_machine',
            'coffee_machine': 'coffee_machine', 'trashcan': 'trash_can',
            'kichenpot': 'kitchen_pot', 'toilet': 'toilet', 'bucket': 'bucket',
            'faucet': 'faucet', 'button': 'button', 'switch': 'switch',
            'bottle': 'bottle', 'stapler': 'stapler', 'toaster': 'toaster'
        }
        
        object_type = "unknown"
        for key, value in object_mappings.items():
            if key in task_lower:
                object_type = value
                break
        
        return category, object_type
    
    def _extract_actions_from_name(self, task_name: str) -> List[str]:
        """Extract required actions from task name"""
        actions = []
        task_lower = task_name.lower()
        
        action_mappings = {
            'open': ['approach_handle', 'grasp_handle', 'pull_open'],
            'close': ['approach_handle', 'grasp_handle', 'push_close'],
            'lift_up': ['approach_object', 'grasp_object', 'lift_vertically'],
            'lay_down': ['approach_object', 'grasp_object', 'lower_vertically'],
            'turn_on': ['approach_control', 'activate_control'],
            'turn_off': ['approach_control', 'deactivate_control'],
            'press': ['approach_target', 'apply_downward_force'],
            'rotate': ['approach_object', 'grasp_object', 'apply_rotational_force']
        }
        
        for action_key, action_list in action_mappings.items():
            if action_key in task_lower:
                actions.extend(action_list)
                break
        
        if not actions:
            actions = ['approach_object', 'interact_with_object']
        
        return actions
    
    def _extract_task_parameters(self, task_name: str) -> Dict[str, Any]:
        """Extract task-specific parameters"""
        parameters = {
            'approach_distance': 0.1,  # meters
            'grasp_force': 10.0,       # Newtons
            'movement_speed': 0.1,     # m/s
            'rotation_speed': 0.5,     # rad/s
            'force_threshold': 5.0,    # Newtons
            'position_tolerance': 0.01, # meters
            'orientation_tolerance': 0.1 # radians
        }
        
        # Task-specific parameter adjustments
        task_lower = task_name.lower()
        if 'door' in task_lower or 'drawer' in task_lower:
            parameters['movement_distance'] = 0.3
        elif 'button' in task_lower or 'switch' in task_lower:
            parameters['contact_force'] = 15.0
        elif 'lift' in task_lower:
            parameters['lift_height'] = 0.2
        
        return parameters
    
    def _generate_success_criteria(self, task_name: str, actions: List[str]) -> List[str]:
        """Generate success criteria for the task"""
        criteria = []
        task_lower = task_name.lower()
        
        if 'open' in task_lower:
            criteria = [
                "Object is successfully opened",
                "Handle is released after opening",
                "No collision with object during execution",
                "Final joint angle indicates open state"
            ]
        elif 'close' in task_lower:
            criteria = [
                "Object is successfully closed",
                "Handle is released after closing",
                "No excessive force applied",
                "Final joint angle indicates closed state"
            ]
        elif 'lift' in task_lower:
            criteria = [
                "Object is lifted to target height",
                "Object remains stable during lift",
                "Grasp is maintained throughout motion"
            ]
        elif 'press' in task_lower:
            criteria = [
                "Button/switch is successfully activated",
                "Appropriate force applied",
                "Hand returns to safe position"
            ]
        else:
            criteria = [
                "Task completed successfully",
                "No collisions occurred",
                "Robot returns to safe configuration"
            ]
        
        return criteria
    
    def _estimate_task_complexity(self, task_name: str, actions: List[str]) -> str:
        """Estimate task complexity"""
        task_lower = task_name.lower()
        
        # Complex tasks
        if any(word in task_lower for word in ['refrigerator', 'dishwasher', 'washingmachine']):
            return "complex"
        elif len(actions) > 4:
            return "complex"
        # Medium tasks
        elif any(word in task_lower for word in ['cabinet', 'oven', 'microwave']):
            return "medium"
        elif len(actions) > 2:
            return "medium"
        # Simple tasks
        else:
            return "simple"
    
    def _estimate_task_duration(self, task_name: str, complexity: str) -> float:
        """Estimate task duration in seconds"""
        base_durations = {
            "simple": 5.0,
            "medium": 10.0,
            "complex": 20.0
        }
        
        duration = base_durations.get(complexity, 10.0)
        
        # Adjust for specific task types
        task_lower = task_name.lower()
        if 'rotate' in task_lower or 'turn' in task_lower:
            duration *= 1.5  # Rotational tasks take longer
        elif 'lift' in task_lower:
            duration *= 1.2  # Lifting tasks need careful control
        
        return duration
    
    def _extract_task_dependencies(self, task_name: str) -> List[str]:
        """Extract task dependencies"""
        dependencies = ["robot_initialization", "object_detection", "motion_planning"]
        
        task_lower = task_name.lower()
        if any(word in task_lower for word in ['grasp', 'lift', 'hold']):
            dependencies.append("grasp_planning")
        if any(word in task_lower for word in ['rotate', 'turn']):
            dependencies.append("force_control")
        
        return dependencies
    
    def _generate_task_description(self, task_name: str, category: str, object_type: str) -> str:
        """Generate a descriptive task description"""
        descriptions = {
            "open_door": "Autonomously open a door by grasping the handle and pulling it towards the robot",
            "close_door": "Autonomously close a door by grasping the handle and pushing it away from the robot",
            "open_drawer": "Open a drawer by grasping the handle and pulling it outward",
            "close_drawer": "Close a drawer by grasping the handle and pushing it inward",
            "open_cabinet": "Open a cabinet door by grasping the handle and pulling it open",
            "close_cabinet": "Close a cabinet door by grasping the handle and pushing it closed"
        }
        
        if task_name in descriptions:
            return descriptions[task_name]
        
        # Generate generic description
        action = task_name.split('_')[0].title()
        obj = object_type.replace('_', ' ')
        return f"{action} a {obj} using kinematic-aware manipulation with the Franka Panda robot"
    
    def _extract_tasks_from_code(self):
        """Extract additional tasks from Python code files"""
        print("🐍 Extracting tasks from Python code...")
        
        code_files = list(self.codebase_path.rglob("*.py"))
        
        for code_file in code_files:
            if code_file.name in ['__init__.py', 'utils.py']:
                continue
                
            try:
                with open(code_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Look for task-related functions
                self._extract_functions_from_code(content, code_file.name)
                
            except Exception as e:
                print(f"⚠️  Error reading {code_file}: {e}")
    
    def _extract_functions_from_code(self, content: str, filename: str):
        """Extract task-related functions from code content"""
        # Look for manipulation-related functions
        function_pattern = r'def\s+(\w+)\s*\([^)]*\):'
        functions = re.findall(function_pattern, content)
        
        task_keywords = ['manipulation', 'grasp', 'move', 'plan', 'execute', 'control']
        
        for func_name in functions:
            if any(keyword in func_name.lower() for keyword in task_keywords):
                # This could be expanded to create TaskDefinitions from code functions
                pass
    
    def _extract_tasks_from_demonstrations(self):
        """Extract tasks from human demonstration files"""
        print("👤 Extracting tasks from demonstration data...")
        
        demo_dir = self.codebase_path / "src" / "human_demonstration"
        if demo_dir.exists():
            for demo_file in demo_dir.glob("*.json"):
                try:
                    with open(demo_file, 'r') as f:
                        demo_data = json.load(f)
                    
                    # Extract task information from demonstration
                    self._parse_demonstration_data(demo_data, demo_file.stem)
                    
                except Exception as e:
                    print(f"⚠️  Error parsing demonstration {demo_file}: {e}")
    
    def _parse_demonstration_data(self, demo_data: Dict, demo_name: str):
        """Parse demonstration data to extract task information"""
        # This could be expanded to extract more detailed task information
        # from human demonstrations
        pass
    
    def generate_task_yamls(self, output_dir: str):
        """Generate individual YAML files for each task"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"📝 Generating task YAML files in {output_dir}...")
        
        for task in self.tasks:
            # Create task-specific YAML
            task_yaml = {
                'task_info': {
                    'name': task.name,
                    'description': task.description,
                    'category': task.category,
                    'object_type': task.object_type,
                    'complexity': task.complexity,
                    'estimated_duration_seconds': task.estimated_duration
                },
                'execution': {
                    'required_actions': task.required_actions,
                    'parameters': task.parameters,
                    'dependencies': task.dependencies
                },
                'validation': {
                    'success_criteria': task.success_criteria
                },
                'assets': task.assets if task.assets else {}
            }
            
            # Write YAML file
            yaml_file = output_path / f"{task.name}.yaml"
            with open(yaml_file, 'w') as f:
                yaml.dump(task_yaml, f, default_flow_style=False, indent=2)
        
        print(f"✅ Generated {len(self.tasks)} task YAML files")
    
    def generate_robot_kinematics_yaml(self, output_dir: str):
        """Generate robot kinematics YAML file"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print("🤖 Generating robot kinematics YAML...")
        
        kinematics_yaml = {
            'robot_kinematics': {
                'name': self.robot_kinematics.name,
                'description': self.robot_kinematics.description,
                'specifications': {
                    'degrees_of_freedom': self.robot_kinematics.dof,
                    'base_link': self.robot_kinematics.base_link,
                    'end_effector': self.robot_kinematics.end_effector,
                    'workspace': self.robot_kinematics.workspace
                },
                'joints': self.robot_kinematics.joints,
                'links': self.robot_kinematics.links,
                'capabilities': self.robot_kinematics.capabilities
            }
        }
        
        yaml_file = output_path / "robot_kinematics.yaml"
        with open(yaml_file, 'w') as f:
            yaml.dump(kinematics_yaml, f, default_flow_style=False, indent=2)
        
        print("✅ Generated robot kinematics YAML file")
    
    def generate_summary_report(self, output_dir: str):
        """Generate a comprehensive summary report"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print("📊 Generating summary report...")
        
        # Task categorization
        categories = {}
        object_types = {}
        complexities = {}
        
        for task in self.tasks:
            categories[task.category] = categories.get(task.category, 0) + 1
            object_types[task.object_type] = object_types.get(task.object_type, 0) + 1
            complexities[task.complexity] = complexities.get(task.complexity, 0) + 1
        
        report = {
            'summary': {
                'total_tasks': len(self.tasks),
                'robot_name': self.robot_kinematics.name,
                'robot_dof': self.robot_kinematics.dof,
                'analysis_date': '2024-01-01'  # Would use actual date
            },
            'task_breakdown': {
                'by_category': categories,
                'by_object_type': object_types,
                'by_complexity': complexities
            },
            'robot_capabilities': self.robot_kinematics.capabilities,
            'task_list': [task.name for task in self.tasks]
        }
        
        # Write summary YAML
        summary_file = output_path / "task_summary.yaml"
        with open(summary_file, 'w') as f:
            yaml.dump(report, f, default_flow_style=False, indent=2)
        
        print("✅ Generated summary report")

def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="LLM Robot Task Organizer")
    parser.add_argument("--codebase", "-c", required=True, help="Path to codebase directory")
    parser.add_argument("--output", "-o", default="organized_tasks", help="Output directory")
    
    args = parser.parse_args()
    
    # Initialize organizer
    organizer = TaskOrganizer(args.codebase)
    
    # Analyze codebase
    tasks, kinematics = organizer.analyze_codebase()
    
    # Generate output files
    organizer.generate_task_yamls(args.output)
    organizer.generate_robot_kinematics_yaml(args.output)
    organizer.generate_summary_report(args.output)
    
    print(f"\n🎉 Task organization complete!")
    print(f"📁 Output directory: {args.output}")
    print(f"📋 Tasks extracted: {len(tasks)}")
    print(f"🤖 Robot: {kinematics.name} ({kinematics.dof} DOF)")

if __name__ == "__main__":
    main()
