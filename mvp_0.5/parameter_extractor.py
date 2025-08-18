#!/usr/bin/env python3
"""
Parameter Extractor
==================

Extracts physical and control parameters from robotics repositories.
"""

import re
import logging
import yaml
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import ast
import json

from data_models import TaskParameters, ParameterSweep, SweepMethod, ParameterPriority

logger = logging.getLogger(__name__)

class ParameterExtractor:
    """Extracts parameters from various robotics repository sources"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.ParameterExtractor")
        
        # Known parameter ranges from robotics literature
        self.parameter_ranges = {
            'friction': (0.05, 1.0),
            'mass': (0.8, 1.2),  # ±20% of nominal
            'inertia': (0.8, 1.2),
            'damping': (0.5, 2.0),
            'stiffness': (0.5, 2.0),
            'gains': (0.75, 1.25),  # ±25% of nominal
            'noise_std': (1.0, 3.0),  # 1x to 3x nominal
            'joint_limits': (0.9, 1.1),  # ±10% of nominal
            'velocity_limits': (0.8, 1.2),  # ±20% of nominal
            'acceleration_limits': (0.7, 1.3),  # ±30% of nominal
        }
    
    def extract_from_robot_models(self, repo_path: Path) -> Dict[str, TaskParameters]:
        """Extract parameters from URDF, SDF, MJCF, and USD files"""
        params = {}
        
        # Find robot model files
        model_files = []
        model_files.extend(repo_path.rglob("*.urdf"))
        model_files.extend(repo_path.rglob("*.sdf"))
        model_files.extend(repo_path.rglob("*.xml"))
        model_files.extend(repo_path.rglob("*.mjcf"))
        model_files.extend(repo_path.rglob("*.usd"))
        
        for model_file in model_files:
            try:
                if model_file.suffix == '.urdf':
                    file_params = self._parse_urdf_file(model_file)
                elif model_file.suffix == '.sdf':
                    file_params = self._parse_sdf_file(model_file)
                elif model_file.suffix == '.xml':
                    file_params = self._parse_xml_file(model_file)
                elif model_file.suffix == '.mjcf':
                    file_params = self._parse_mjcf_file(model_file)
                else:
                    continue
                
                # Add source file info
                for param_name, param in file_params.items():
                    param.source = 'robot_model'
                    param.sweep.description = f"Extracted from {model_file.name}"
                
                params.update(file_params)
                
            except Exception as e:
                self.logger.warning(f"Failed to parse {model_file}: {e}")
                continue
        
        self.logger.info(f"Extracted {len(params)} parameters from robot models")
        return params
    
    def extract_from_ros_configs(self, repo_path: Path) -> Dict[str, TaskParameters]:
        """Extract parameters from ROS 2 configuration files"""
        params = {}
        
        # Find ROS config files
        ros_files = []
        ros_files.extend(repo_path.rglob("*.yaml"))
        ros_files.extend(repo_path.rglob("*.yml"))
        ros_files.extend(repo_path.rglob("*.launch.py"))
        ros_files.extend(repo_path.rglob("*.launch.xml"))
        
        for config_file in ros_files:
            try:
                if self._is_ros_config(config_file):
                    file_params = self._parse_ros_config(config_file)
                    
                    # Add source file info
                    for param_name, param in file_params.items():
                        param.source = 'ros_config'
                        param.sweep.description = f"Extracted from {config_file.name}"
                    
                    params.update(file_params)
                    
            except Exception as e:
                self.logger.warning(f"Failed to parse {config_file}: {e}")
                continue
        
        self.logger.info(f"Extracted {len(params)} parameters from ROS configs")
        return params
    
    def extract_from_source_code(self, repo_path: Path) -> Dict[str, TaskParameters]:
        """Extract parameters from Python/C++ source code"""
        params = {}
        
        # Find source code files
        source_files = []
        source_files.extend(repo_path.rglob("*.py"))
        source_files.extend(repo_path.rglob("*.cpp"))
        source_files.extend(repo_path.rglob("*.hpp"))
        source_files.extend(repo_path.rglob("*.h"))
        
        for source_file in source_files:
            try:
                if source_file.suffix == '.py':
                    file_params = self._parse_python_file(source_file)
                else:
                    file_params = self._parse_cpp_file(source_file)
                
                # Add source file info
                for param_name, param in file_params.items():
                    param.source = 'source_code'
                    param.sweep.description = f"Extracted from {source_file.name}"
                
                params.update(file_params)
                
            except Exception as e:
                self.logger.warning(f"Failed to parse {source_file}: {e}")
                continue
        
        self.logger.info(f"Extracted {len(params)} parameters from source code")
        return params
    
    def extract_from_task_yamls(self, repo_path: Path) -> Dict[str, TaskParameters]:
        """Extract parameters from existing task YAML files"""
        params = {}
        
        # Find task YAML files
        yaml_files = []
        yaml_files.extend(repo_path.rglob("*.yaml"))
        yaml_files.extend(repo_path.rglob("*.yml"))
        
        for yaml_file in yaml_files:
            try:
                if self._is_task_yaml(yaml_file):
                    file_params = self._parse_task_yaml(yaml_file)
                    
                    # Add source file info
                    for param_name, param in file_params.items():
                        param.source = 'task_yaml'
                        param.sweep.description = f"Extracted from {yaml_file.name}"
                    
                    params.update(file_params)
                    
            except Exception as e:
                self.logger.warning(f"Failed to parse {yaml_file}: {e}")
                continue
        
        self.logger.info(f"Extracted {len(params)} parameters from task YAMLs")
        return params
    
    def _parse_urdf_file(self, urdf_file: Path) -> Dict[str, TaskParameters]:
        """Parse URDF file for physical parameters"""
        params = {}
        
        try:
            tree = ET.parse(urdf_file)
            root = tree.getroot()
            
            # Extract joint parameters
            for joint in root.findall('.//joint'):
                joint_name = joint.get('name', 'unknown')
                
                # Joint limits
                limits = joint.find('limit')
                if limits is not None:
                    # Velocity limit
                    if 'velocity' in limits.attrib:
                        vel = float(limits.get('velocity'))
                        params[f"{joint_name}_velocity_limit"] = TaskParameters(
                            name=f"{joint_name}_velocity_limit",
                            nominal_value=vel,
                            unit="rad/s",
                            sweep=self._create_sweep(vel, 'velocity_limits', 'high'),
                            source='urdf',
                            confidence=0.9
                        )
                    
                    # Effort limit
                    if 'effort' in limits.attrib:
                        effort = float(limits.get('effort'))
                        params[f"{joint_name}_effort_limit"] = TaskParameters(
                            name=f"{joint_name}_effort_limit",
                            nominal_value=effort,
                            unit="Nm",
                            sweep=self._create_sweep(effort, 'gains', 'high'),
                            source='urdf',
                            confidence=0.9
                        )
            
            # Extract link parameters
            for link in root.findall('.//link'):
                link_name = link.get('name', 'unknown')
                
                # Mass
                inertial = link.find('inertial')
                if inertial is not None and inertial.find('mass') is not None:
                    mass = float(inertial.find('mass').get('value', 0))
                    if mass > 0:
                        params[f"{link_name}_mass"] = TaskParameters(
                            name=f"{link_name}_mass",
                            nominal_value=mass,
                            unit="kg",
                            sweep=self._create_sweep(mass, 'mass', 'medium'),
                            source='urdf',
                            confidence=0.9
                        )
                
                # Friction coefficients
                collision = link.find('collision')
                if collision is not None:
                    surface = collision.find('.//surface')
                    if surface is not None:
                        friction = surface.find('friction')
                        if friction is not None:
                            for child in friction:
                                if child.tag == 'ode':
                                    mu = float(child.get('mu', 0.5))
                                    params[f"{link_name}_friction"] = TaskParameters(
                                        name=f"{link_name}_friction",
                                        nominal_value=mu,
                                        unit="dimensionless",
                                        sweep=self._create_sweep(mu, 'friction', 'high'),
                                        source='urdf',
                                        confidence=0.8
                                    )
        
        except Exception as e:
            self.logger.warning(f"Failed to parse URDF {urdf_file}: {e}")
        
        return params
    
    def _parse_sdf_file(self, sdf_file: Path) -> Dict[str, TaskParameters]:
        """Parse SDF file for physical parameters"""
        params = {}
        
        try:
            tree = ET.parse(sdf_file)
            root = tree.getroot()
            
            # Extract model parameters
            for model in root.findall('.//model'):
                model_name = model.get('name', 'unknown')
                
                # Link parameters
                for link in model.findall('.//link'):
                    link_name = link.get('name', 'unknown')
                    
                    # Mass
                    inertial = link.find('inertial')
                    if inertial is not None:
                        mass_elem = inertial.find('mass')
                        if mass_elem is not None:
                            mass = float(mass_elem.text)
                            params[f"{model_name}_{link_name}_mass"] = TaskParameters(
                                name=f"{model_name}_{link_name}_mass",
                                nominal_value=mass,
                                unit="kg",
                                sweep=self._create_sweep(mass, 'mass', 'medium'),
                                source='sdf',
                                confidence=0.9
                            )
                
                # Joint parameters
                for joint in model.findall('.//joint'):
                    joint_name = joint.get('name', 'unknown')
                    
                    # Joint limits
                    axis = joint.find('axis')
                    if axis is not None:
                        limit = axis.find('limit')
                        if limit is not None:
                            # Velocity limit
                            if 'velocity' in limit.attrib:
                                vel = float(limit.get('velocity'))
                                params[f"{model_name}_{joint_name}_velocity_limit"] = TaskParameters(
                                    name=f"{model_name}_{joint_name}_velocity_limit",
                                    nominal_value=vel,
                                    unit="rad/s",
                                    sweep=self._create_sweep(vel, 'velocity_limits', 'high'),
                                    source='sdf',
                                    confidence=0.9
                                )
        
        except Exception as e:
            self.logger.warning(f"Failed to parse SDF {sdf_file}: {e}")
        
        return params
    
    def _parse_xml_file(self, xml_file: Path) -> Dict[str, TaskParameters]:
        """Parse generic XML file for parameters"""
        params = {}
        
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            # Look for common parameter patterns
            for elem in root.iter():
                if 'param' in elem.tag.lower() or 'parameter' in elem.tag.lower():
                    name = elem.get('name') or elem.get('key')
                    value = elem.get('value') or elem.text
                    
                    if name and value:
                        try:
                            num_value = float(value)
                            params[name] = TaskParameters(
                                name=name,
                                nominal_value=num_value,
                                unit="unknown",
                                sweep=self._create_sweep(num_value, 'gains', 'medium'),
                                source='xml',
                                confidence=0.6
                            )
                        except ValueError:
                            continue
        
        except Exception as e:
            self.logger.warning(f"Failed to parse XML {xml_file}: {e}")
        
        return params
    
    def _parse_mjcf_file(self, mjcf_file: Path) -> Dict[str, TaskParameters]:
        """Parse MJCF file for MuJoCo parameters"""
        params = {}
        
        try:
            tree = ET.parse(mjcf_file)
            root = tree.getroot()
            
            # Extract MuJoCo-specific parameters
            for elem in root.iter():
                if 'geom' in elem.tag:
                    # Friction
                    friction = elem.get('friction')
                    if friction:
                        try:
                            fric_val = float(friction)
                            geom_name = elem.get('name', 'unknown')
                            params[f"{geom_name}_friction"] = TaskParameters(
                                name=f"{geom_name}_friction",
                                nominal_value=fric_val,
                                unit="dimensionless",
                                sweep=self._create_sweep(fric_val, 'friction', 'high'),
                                source='mjcf',
                                confidence=0.8
                            )
                        except ValueError:
                            continue
                
                elif 'joint' in elem.tag:
                    # Joint parameters
                    joint_name = elem.get('name', 'unknown')
                    
                    # Damping
                    damping = elem.get('damping')
                    if damping:
                        try:
                            damp_val = float(damping)
                            params[f"{joint_name}_damping"] = TaskParameters(
                                name=f"{joint_name}_damping",
                                nominal_value=damp_val,
                                unit="N⋅m⋅s/rad",
                                sweep=self._create_sweep(damp_val, 'damping', 'medium'),
                                source='mjcf',
                                confidence=0.8
                            )
                        except ValueError:
                            continue
        
        except Exception as e:
            self.logger.warning(f"Failed to parse MJCF {mjcf_file}: {e}")
        
        return params
    
    def _is_ros_config(self, file_path: Path) -> bool:
        """Check if file is a ROS configuration file"""
        try:
            with open(file_path, 'r') as f:
                content = f.read().lower()
            
            # Look for ROS-specific keywords
            ros_keywords = ['ros', 'node', 'topic', 'service', 'action', 'parameter', 'launch']
            return any(keyword in content for keyword in ros_keywords)
        except:
            return False
    
    def _parse_ros_config(self, config_file: Path) -> Dict[str, TaskParameters]:
        """Parse ROS configuration file for parameters"""
        params = {}
        
        try:
            if config_file.suffix in ['.yaml', '.yml']:
                with open(config_file, 'r') as f:
                    content = yaml.safe_load(f)
                
                if isinstance(content, dict):
                    params.update(self._extract_from_dict(content, config_file.name))
            
            elif config_file.suffix == '.launch.py':
                params.update(self._parse_python_launch(config_file))
            
            elif config_file.suffix == '.launch.xml':
                params.update(self._parse_xml_launch(config_file))
        
        except Exception as e:
            self.logger.warning(f"Failed to parse ROS config {config_file}: {e}")
        
        return params
    
    def _extract_from_dict(self, data: Dict, source_name: str) -> Dict[str, TaskParameters]:
        """Recursively extract parameters from dictionary"""
        params = {}
        
        for key, value in data.items():
            if isinstance(value, (int, float)):
                # Direct parameter value
                params[key] = TaskParameters(
                    name=key,
                    nominal_value=float(value),
                    unit="unknown",
                    sweep=self._create_sweep(float(value), 'gains', 'medium'),
                    source='ros_config',
                    confidence=0.7
                )
            
            elif isinstance(value, dict):
                # Nested dictionary
                nested_params = self._extract_from_dict(value, source_name)
                for nested_key, nested_param in nested_params.items():
                    params[f"{key}_{nested_key}"] = nested_param
            
            elif isinstance(value, list):
                # List of values
                for i, item in enumerate(value):
                    if isinstance(item, (int, float)):
                        params[f"{key}_{i}"] = TaskParameters(
                            name=f"{key}_{i}",
                            nominal_value=float(item),
                            unit="unknown",
                            sweep=self._create_sweep(float(item), 'gains', 'medium'),
                            source='ros_config',
                            confidence=0.6
                        )
        
        return params
    
    def _parse_python_launch(self, launch_file: Path) -> Dict[str, TaskParameters]:
        """Parse Python launch file for parameters"""
        params = {}
        
        try:
            with open(launch_file, 'r') as f:
                content = f.read()
            
            # Look for parameter declarations
            param_patterns = [
                r'DeclareLaunchArgument\([\'"]([^\'"]+)[\'"][^)]*default_value=[\'"]([^\'"]+)[\'"]',
                r'Node\([^)]*parameters=\{[^}]*[\'"]([^\'"]+)[\'"]:\s*([^,}]+)',
                r'([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*([0-9.]+)'
            ]
            
            for pattern in param_patterns:
                matches = re.findall(pattern, content)
                for match in matches:
                    if len(match) == 2:
                        name, value = match
                        try:
                            num_value = float(value)
                            params[name] = TaskParameters(
                                name=name,
                                nominal_value=num_value,
                                unit="unknown",
                                sweep=self._create_sweep(num_value, 'gains', 'medium'),
                                source='ros_launch',
                                confidence=0.7
                            )
                        except ValueError:
                            continue
        
        except Exception as e:
            self.logger.warning(f"Failed to parse Python launch {launch_file}: {e}")
        
        return params
    
    def _parse_xml_launch(self, launch_file: Path) -> Dict[str, TaskParameters]:
        """Parse XML launch file for parameters"""
        params = {}
        
        try:
            tree = ET.parse(launch_file)
            root = tree.getroot()
            
            # Extract parameters
            for param in root.findall('.//param'):
                name = param.get('name')
                value = param.get('value')
                
                if name and value:
                    try:
                        num_value = float(value)
                        params[name] = TaskParameters(
                            name=name,
                            nominal_value=num_value,
                            unit="unknown",
                            sweep=self._create_sweep(num_value, 'gains', 'medium'),
                            source='ros_launch',
                            confidence=0.7
                        )
                    except ValueError:
                        continue
        
        except Exception as e:
            self.logger.warning(f"Failed to parse XML launch {launch_file}: {e}")
        
        return params
    
    def _parse_python_file(self, python_file: Path) -> Dict[str, TaskParameters]:
        """Parse Python file for constants and parameters"""
        params = {}
        
        try:
            with open(python_file, 'r') as f:
                content = f.read()
            
            # Parse AST to find constants
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            name = target.id
                            
                            # Check if it's a constant (uppercase)
                            if name.isupper() and isinstance(node.value, ast.Num):
                                value = node.value.n
                                
                                # Check if it looks like a robotics parameter
                                if self._is_robotics_parameter(name, value):
                                    params[name] = TaskParameters(
                                        name=name,
                                        nominal_value=value,
                                        unit="unknown",
                                        sweep=self._create_sweep(value, 'gains', 'medium'),
                                        source='python',
                                        confidence=0.6
                                    )
            
            # Also look for common parameter patterns in strings
            param_patterns = [
                r'([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*([0-9.]+)',
                r'[\'"]([a-zA-Z_][a-zA-Z0-9_]*)[\'"]\s*:\s*([0-9.]+)'
            ]
            
            for pattern in param_patterns:
                matches = re.findall(pattern, content)
                for match in matches:
                    if len(match) == 2:
                        name, value = match
                        try:
                            num_value = float(value)
                            if self._is_robotics_parameter(name, num_value):
                                params[name] = TaskParameters(
                                    name=name,
                                    nominal_value=num_value,
                                    unit="unknown",
                                    sweep=self._create_sweep(num_value, 'gains', 'medium'),
                                    source='python',
                                    confidence=0.5
                                )
                        except ValueError:
                            continue
        
        except Exception as e:
            self.logger.warning(f"Failed to parse Python {python_file}: {e}")
        
        return params
    
    def _parse_cpp_file(self, cpp_file: Path) -> Dict[str, TaskParameters]:
        """Parse C++ file for constants and parameters"""
        params = {}
        
        try:
            with open(cpp_file, 'r') as f:
                content = f.read()
            
            # Look for constant definitions
            const_patterns = [
                r'const\s+(?:double|float|int)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*([0-9.]+)',
                r'#define\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+([0-9.]+)',
                r'([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*([0-9.]+)\s*;'
            ]
            
            for pattern in const_patterns:
                matches = re.findall(pattern, content)
                for match in matches:
                    if len(match) == 2:
                        name, value = match
                        try:
                            num_value = float(value)
                            if self._is_robotics_parameter(name, num_value):
                                params[name] = TaskParameters(
                                    name=name,
                                    nominal_value=num_value,
                                    unit="unknown",
                                    sweep=self._create_sweep(num_value, 'gains', 'medium'),
                                    source='cpp',
                                    confidence=0.6
                                )
                        except ValueError:
                            continue
        
        except Exception as e:
            self.logger.warning(f"Failed to parse C++ {cpp_file}: {e}")
        
        return params
    
    def _is_task_yaml(self, yaml_file: Path) -> bool:
        """Check if YAML file contains task definitions"""
        try:
            with open(yaml_file, 'r') as f:
                content = yaml.safe_load(f)
            
            if not content:
                return False
            
            # Check for task-related keys
            task_keys = ['task', 'parameters', 'execution', 'robot', 'simulation']
            content_str = str(content).lower()
            return any(key in content_str for key in task_keys)
        
        except:
            return False
    
    def _parse_task_yaml(self, yaml_file: Path) -> Dict[str, TaskParameters]:
        """Parse task YAML file for existing parameters"""
        params = {}
        
        try:
            with open(yaml_file, 'r') as f:
                content = yaml.safe_load(f)
            
            if not content:
                return params
            
            # Extract parameters from various locations
            param_locations = ['parameters', 'execution.parameters', 'robot.parameters']
            
            for location in param_locations:
                keys = location.split('.')
                current = content
                
                for key in keys:
                    if isinstance(current, dict) and key in current:
                        current = current[key]
                    else:
                        current = None
                        break
                
                if current and isinstance(current, dict):
                    for param_name, param_value in current.items():
                        if isinstance(param_value, (int, float)):
                            params[param_name] = TaskParameters(
                                name=param_name,
                                nominal_value=float(param_value),
                                unit="unknown",
                                sweep=self._create_sweep(float(param_value), 'gains', 'medium'),
                                source='task_yaml',
                                confidence=0.8
                            )
                        elif isinstance(param_value, dict) and 'value' in param_value:
                            try:
                                value = float(param_value['value'])
                                params[param_name] = TaskParameters(
                                    name=param_name,
                                    nominal_value=value,
                                    unit=param_value.get('unit', 'unknown'),
                                    sweep=self._create_sweep(value, 'gains', 'medium'),
                                    source='task_yaml',
                                    confidence=0.8
                                )
                            except (ValueError, TypeError):
                                continue
        
        except Exception as e:
            self.logger.warning(f"Failed to parse task YAML {yaml_file}: {e}")
        
        return params
    
    def _is_robotics_parameter(self, name: str, value: float) -> bool:
        """Check if a parameter looks like a robotics parameter"""
        # Check name patterns
        name_lower = name.lower()
        robotics_keywords = [
            'mass', 'inertia', 'damping', 'stiffness', 'friction', 'gain',
            'limit', 'velocity', 'acceleration', 'torque', 'force', 'noise',
            'threshold', 'timeout', 'frequency', 'rate', 'scale', 'offset'
        ]
        
        if any(keyword in name_lower for keyword in robotics_keywords):
            return True
        
        # Check value ranges (reasonable robotics values)
        if 0.001 <= abs(value) <= 10000:
            return True
        
        return False
    
    def _create_sweep(self, nominal_value: float, param_type: str, priority: str) -> ParameterSweep:
        """Create parameter sweep configuration"""
        # Get range based on parameter type
        if param_type in self.parameter_ranges:
            min_ratio, max_ratio = self.parameter_ranges[param_type]
            sweep_range = [nominal_value * min_ratio, nominal_value * max_ratio]
        else:
            # Default range: ±20%
            sweep_range = [nominal_value * 0.8, nominal_value * 1.2]
        
        # Determine priority
        if priority == 'high':
            priority_enum = ParameterPriority.HIGH
        elif priority == 'medium':
            priority_enum = ParameterPriority.MEDIUM
        else:
            priority_enum = ParameterPriority.LOW
        
        # Determine sweep method based on priority
        if priority_enum == ParameterPriority.HIGH:
            method = SweepMethod.SOBOL
            samples = 100
        elif priority_enum == ParameterPriority.MEDIUM:
            method = SweepMethod.OAT
            samples = 50
        else:
            method = SweepMethod.RANDOM
            samples = 25
        
        return ParameterSweep(
            method=method,
            range=sweep_range,
            priority=priority_enum,
            samples=samples,
            nominal_value=nominal_value
        ) 