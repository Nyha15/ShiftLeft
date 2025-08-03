"""
Output Generator Module

This module provides functionality to generate the output YAML file.
"""

import logging
import yaml
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class OutputGenerator:
    """
    Generates the output YAML file.
    """
    
    def __init__(self, fused_data: Dict[str, Any]):
        """
        Initialize the output generator.
        
        Args:
            fused_data: Fused data from the information fusion module
        """
        self.fused_data = fused_data
        
    def generate(self, output_path: Path) -> None:
        """
        Generate the output YAML file.
        
        Args:
            output_path: Path to the output file
        """
        logger.info(f"Generating output file: {output_path}")
        
        # Create the output data structure
        output_data = self._create_output_data()
        
        # Write the output file
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(output_data, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Output file generated: {output_path}")
    
    def _create_output_data(self) -> Dict[str, Any]:
        """
        Create the output data structure.
        
        Returns:
            Dictionary containing the output data
        """
        # Extract data from fused data
        robot_config = self.fused_data.get('robot_config', {})
        entry_points = self.fused_data.get('entry_points', {}).get('entry_points', [])
        action_sequences = self.fused_data.get('action_sequences', {}).get('sequences', [])
        named_positions = self.fused_data.get('named_positions', {})
        overall_confidence = self.fused_data.get('overall_confidence', 0.0)
        gaps = self.fused_data.get('gaps', [])
        mvp1_ready = self.fused_data.get('mvp1_ready', False)
        file_stats = self.fused_data.get('file_stats', {})
        
        # Create robot_config section
        robot_config_output = {
            'name': robot_config.get('name', 'discovered_robot'),
            'dof': robot_config.get('dof'),
            'confidence': robot_config.get('confidence', 0.0),
            'sources': robot_config.get('sources', []),
            'joints': self._create_joints_output(robot_config)
        }
        
        # Create discovered_entry_points section
        entry_points_output = []
        for ep in entry_points:
            entry_points_output.append({
                'file': ep.get('file', ''),
                'score': ep.get('score', 0.0),
                'patterns': ep.get('patterns', []),
                'type': ep.get('type', 'unknown')
            })
        
        # Create named_positions section
        named_positions_output = {}
        for name, pos in named_positions.items():
            named_positions_output[name] = {
                'values': pos.get('values', []),
                'source': pos.get('source', ''),
                'confidence': pos.get('confidence', 0.0)
            }
        
        # Create tasks section
        tasks_output = {}
        
        # First check if we have organized tasks
        organized_tasks = self.fused_data.get('tasks', {}).get('tasks', {})
        
        if organized_tasks:
            # Use organized tasks
            for task_name, task_data in organized_tasks.items():
                tasks_output[task_name] = {
                    'name': task_data.get('name', task_name),
                    'source': task_data.get('source', ''),
                    'confidence': task_data.get('confidence', 0.0),
                    'sequence': self._create_sequence_output(task_data)
                }
        else:
            # Fall back to action sequences
            for i, sequence in enumerate(action_sequences):
                sequence_name = sequence.get('name', f'discovered_sequence_{i+1}')
                tasks_output[sequence_name] = {
                    'name': sequence.get('name', f'Sequence {i+1}'),
                    'source': sequence.get('source', ''),
                    'confidence': sequence.get('confidence', 0.0),
                    'sequence': self._create_sequence_output(sequence)
                }
        
        # Check if LLM was used
        llm_used = self.fused_data.get('llm_used', False)
        llm_stats = self.fused_data.get('llm_stats', {})
        frameworks_detected = self.fused_data.get('frameworks_detected', [])
        
        # Create metadata section
        metadata_output = {
            'analysis_method': 'hybrid_ast_llm' if llm_used else 'multi_source_discovery',
            'files_analyzed': sum(file_stats.values()),
            'entry_points_found': len(entry_points),
            'frameworks_detected': frameworks_detected or self._detect_frameworks(),
            'confidence_overall': overall_confidence,
            'gaps_identified': gaps,
            'mvp1_ready': mvp1_ready,
            'timestamp': datetime.now().isoformat()
        }
        
        # Add LLM information if used
        if llm_used:
            metadata_output['llm_used'] = True
            metadata_output['llm_stats'] = llm_stats
        
        # Create the final output
        output_data = {
            'robot_config': robot_config_output,
            'discovered_entry_points': entry_points_output,
            'named_positions': named_positions_output,
            'tasks': tasks_output,
            'metadata': metadata_output
        }
        
        return output_data
    
    def _create_joints_output(self, robot_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Create the joints output section.
        
        Args:
            robot_config: Robot configuration data
            
        Returns:
            List of dictionaries containing joint information
        """
        joints_output = []
        
        joint_names = robot_config.get('joint_names', [])
        joint_types = robot_config.get('joint_types', [])
        joint_limits = robot_config.get('joint_limits', [])
        
        # Ensure all lists have the same length
        num_joints = max(len(joint_names), len(joint_types), len(joint_limits))
        
        # Pad lists if necessary
        if len(joint_names) < num_joints:
            joint_names.extend([f'joint_{i+1}' for i in range(len(joint_names), num_joints)])
        if len(joint_types) < num_joints:
            joint_types.extend(['revolute' for _ in range(len(joint_types), num_joints)])
        if len(joint_limits) < num_joints:
            joint_limits.extend([[None, None] for _ in range(len(joint_limits), num_joints)])
        
        # Create joint entries
        for i in range(num_joints):
            joint = {
                'name': joint_names[i],
                'type': joint_types[i] if i < len(joint_types) else 'revolute'
            }
            
            # Add limits if available
            if i < len(joint_limits) and joint_limits[i][0] is not None and joint_limits[i][1] is not None:
                joint['limits'] = {
                    'lower': joint_limits[i][0],
                    'upper': joint_limits[i][1]
                }
            
            # Add source and confidence
            joint['source'] = robot_config.get('sources', ['unknown'])[0] if robot_config.get('sources') else 'unknown'
            joint['confidence'] = robot_config.get('confidence', 0.0)
            
            joints_output.append(joint)
        
        return joints_output
    
    def _create_sequence_output(self, sequence: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Create the sequence output section.
        
        Args:
            sequence: Action sequence data
            
        Returns:
            List of dictionaries containing sequence steps
        """
        sequence_output = []
        
        steps = sequence.get('steps', [])
        
        for step in steps:
            # Clean up parameters for better readability
            parameters = step.get('parameters', {})
            clean_params = {}
            
            # Handle different parameter formats
            if isinstance(parameters, dict):
                # If it's a dictionary with args and keywords
                if 'args' in parameters and 'keywords' in parameters:
                    clean_params = parameters
                else:
                    # For other dictionaries, clean up each value
                    for key, value in parameters.items():
                        if isinstance(value, str) and len(value) > 100:
                            # Truncate long strings
                            clean_params[key] = value[:97] + "..."
                        else:
                            clean_params[key] = value
            else:
                clean_params = parameters
            
            step_output = {
                'step': step.get('step', 0),
                'action': step.get('action', 'unknown'),
                'parameters': clean_params,
                'source': f"{sequence.get('source', '')}:{step.get('line', '')}"
            }
            sequence_output.append(step_output)
        
        return sequence_output
    
    def _detect_frameworks(self) -> List[str]:
        """
        Detect robotics frameworks used in the repository.
        
        Returns:
            List of detected frameworks
        """
        frameworks = set()
        
        # Check entry points for imports
        entry_points = self.fused_data.get('entry_points', {}).get('entry_points', [])
        for ep in entry_points:
            imports = ep.get('imports', [])
            
            # Check for common robotics frameworks
            if any('mujoco' in imp for imp in imports):
                frameworks.add('mujoco')
            if any('dm_control' in imp for imp in imports):
                frameworks.add('dm_control')
            if any('pybullet' in imp for imp in imports):
                frameworks.add('pybullet')
            if any('gym' in imp for imp in imports):
                frameworks.add('gym')
            if any(imp in ['rospy', 'rclpy', 'ros2'] for imp in imports):
                frameworks.add('ros')
            if any('moveit' in imp for imp in imports):
                frameworks.add('moveit')
            if any('gazebo' in imp for imp in imports):
                frameworks.add('gazebo')
            if any('isaac' in imp for imp in imports):
                frameworks.add('isaac')
        
        return list(frameworks)