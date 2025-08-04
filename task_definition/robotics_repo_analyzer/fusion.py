"""
Information Fusion Module

This module provides functionality to fuse information from multiple sources.
"""

import logging
from typing import Dict, List, Any, Set, Tuple, Optional
import numpy as np

from task_definition.robotics_repo_analyzer.utils.confidence import calculate_confidence

logger = logging.getLogger(__name__)

class InformationFusion:
    """
    Fuses information from multiple sources.
    """

    def __init__(self, scan_results: Dict[str, Any]):
        """
        Initialize the information fusion module.

        Args:
            scan_results: Results from the repository scanner
        """
        self.scan_results = scan_results

    def fuse(self) -> Dict[str, Any]:
        """
        Fuse information from multiple sources.

        Returns:
            Dictionary containing fused information
        """
        logger.info("Fusing information from multiple sources...")

        # Fuse robot configuration
        robot_config = self._fuse_robot_config()

        # Fuse entry points
        entry_points = self._fuse_entry_points()

        # Fuse action sequences
        action_sequences = self._fuse_action_sequences()

        # Fuse parameters
        parameters = self._fuse_parameters()

        # Create named positions from parameters
        named_positions = self._extract_named_positions(parameters)

        # Calculate overall confidence
        overall_confidence = self._calculate_overall_confidence(
            robot_config, entry_points, action_sequences, parameters
        )

        # Identify gaps
        gaps = self._identify_gaps(
            robot_config, entry_points, action_sequences, parameters
        )

        # Determine if the result is MVP1-ready
        mvp1_ready = self._is_mvp1_ready(
            robot_config, entry_points, action_sequences, parameters, gaps
        )

        logger.info(f"Information fusion complete. "
                   f"Overall confidence: {overall_confidence:.2f}. "
                   f"MVP1-ready: {mvp1_ready}")

        # Include tasks if available
        tasks = self.scan_results.get('tasks', {})

        # Check if LLM was used
        llm_used = 'llm_client' in self.scan_results

        # Create LLM stats if LLM was used
        llm_stats = {}
        if llm_used and tasks and 'tasks' in tasks:
            # Count LLM-verified tasks
            llm_verified_tasks = sum(1 for task in tasks['tasks'].values()
                                    if task.get('llm_verified', False))
            llm_stats = {
                'total_tasks': len(tasks['tasks']),
                'llm_verified_tasks': llm_verified_tasks,
                'verification_rate': llm_verified_tasks / len(tasks['tasks']) if tasks['tasks'] else 0
            }

        return {
            'robot_config': robot_config,
            'entry_points': entry_points,
            'action_sequences': action_sequences,
            'parameters': parameters,
            'named_positions': named_positions,
            'tasks': tasks,  # Include tasks in the fused data
            'overall_confidence': overall_confidence,
            'gaps': gaps,
            'mvp1_ready': mvp1_ready,
            'file_stats': self.scan_results.get('file_stats', {}),
            'llm_used': llm_used,
            'llm_stats': llm_stats,
            'frameworks_detected': self.scan_results.get('frameworks_detected', [])
        }

    def _fuse_robot_config(self) -> Dict[str, Any]:
        """
        Fuse robot configuration information.

        Returns:
            Dictionary containing fused robot configuration
        """
        robot_config = self.scan_results.get('robot_config', {})

        # If no robot configuration was found, create a default one
        if not robot_config:
            return {
                'name': 'unknown_robot',
                'dof': None,
                'joint_names': [],
                'joint_limits': [],
                'joint_types': [],
                'sources': [],
                'confidence': 0.0
            }

        return robot_config

    def _fuse_entry_points(self) -> Dict[str, Any]:
        """
        Fuse entry point information.

        Returns:
            Dictionary containing fused entry point information
        """
        entry_points = self.scan_results.get('entry_points', {})

        # If no entry points were found, create a default one
        if not entry_points:
            return {
                'entry_points': [],
                'confidence': 0.0
            }

        return entry_points

    def _fuse_action_sequences(self) -> Dict[str, Any]:
        """
        Fuse action sequence information.

        Returns:
            Dictionary containing fused action sequence information
        """
        action_sequences = self.scan_results.get('action_sequences', {})

        # If no action sequences were found, create a default one
        if not action_sequences:
            return {
                'sequences': [],
                'confidence': 0.0
            }

        # Process sequences to standardize action names
        sequences = action_sequences.get('sequences', [])
        for sequence in sequences:
            steps = sequence.get('steps', [])
            for step in steps:
                # Standardize action names
                action = step.get('action', '')
                step['action'] = self._standardize_action_name(action)

        return action_sequences

    def _standardize_action_name(self, action: str) -> str:
        """
        Standardize action names to a common format.

        Args:
            action: Original action name

        Returns:
            Standardized action name
        """
        # Map of common action names to standardized names
        action_map = {
            # Movement actions
            'move_to': 'move_to_position',
            'move': 'move_to_position',
            'set_position': 'move_to_position',
            'set_joint_position': 'move_to_joint_position',
            'set_joint_positions': 'move_to_joint_position',
            'set_joint_angles': 'move_to_joint_position',
            'set_pose': 'move_to_pose',
            'set_target': 'move_to_position',
            'go_to': 'move_to_position',
            'goto': 'move_to_position',
            'go_home': 'move_to_home',
            'move_joints': 'move_to_joint_position',
            'move_joint': 'move_to_joint_position',
            'move_arm': 'move_to_position',
            'move_gripper': 'set_gripper',
            'move_base': 'move_to_position',

            # Gripper actions
            'open_gripper': 'open_gripper',
            'close_gripper': 'close_gripper',
            'set_gripper': 'set_gripper',
            'grasp': 'close_gripper',
            'release': 'open_gripper',
            'pick': 'pick_object',
            'place': 'place_object',
            'pick_and_place': 'pick_and_place',
            'grip': 'close_gripper',
            'ungrip': 'open_gripper',

            # Simulation actions
            'step': 'simulation_step',
            'reset': 'reset_environment',
            'simulate': 'simulation_step',
            'update': 'simulation_step',
            'render': 'render',
            'forward': 'simulation_step',

            # Sensor actions
            'get_state': 'get_state',
            'get_position': 'get_position',
            'get_joint_position': 'get_joint_position',
            'get_joint_positions': 'get_joint_position',
            'get_joint_angles': 'get_joint_position',
            'get_pose': 'get_pose',
            'get_observation': 'get_observation',
            'observe': 'get_observation',
            'read_sensors': 'get_sensor_data',
            'get_sensor_data': 'get_sensor_data',

            # Control actions
            'set_control': 'apply_control',
            'set_velocity': 'set_velocity',
            'set_torque': 'apply_torque',
            'set_force': 'apply_force',
            'apply_action': 'apply_action',
            'apply_control': 'apply_control',
            'apply_torque': 'apply_torque',
            'apply_force': 'apply_force',
            'control': 'apply_control',
            'pid_control': 'apply_control',
            'inverse_kinematics': 'compute_inverse_kinematics',
            'forward_kinematics': 'compute_forward_kinematics',
            'compute_jacobian': 'compute_jacobian',
            'solve_ik': 'compute_inverse_kinematics'
        }

        # Get the standardized name, or use the original if not found
        return action_map.get(action.lower(), action)

    def _fuse_parameters(self) -> Dict[str, Any]:
        """
        Fuse parameter information.

        Returns:
            Dictionary containing fused parameter information
        """
        parameters = self.scan_results.get('parameters', {})

        # If no parameters were found, create a default one
        if not parameters:
            return {
                'parameters': [],
                'grouped_parameters': {},
                'confidence': 0.0
            }

        return parameters

    def _extract_named_positions(self, parameters: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        Extract named positions from parameters.

        Args:
            parameters: Parameter information

        Returns:
            Dictionary mapping position names to position information
        """
        named_positions = {}

        # Get all parameters
        all_params = parameters.get('parameters', [])

        # Get position-related parameters
        position_params = parameters.get('grouped_parameters', {}).get('positions', [])
        position_params.extend(parameters.get('grouped_parameters', {}).get('joint_positions', []))

        # Extract named positions
        for param in position_params:
            name = param.get('name', '')
            value = param.get('value')

            if name and value is not None:
                # Check if it's a position array
                if isinstance(value, (list, tuple)) and all(isinstance(v, (int, float)) for v in value):
                    # Skip very long arrays (likely not positions)
                    if len(value) > 10:
                        continue

                    # Create a descriptive name
                    descriptive_name = self._create_descriptive_name(name)

                    named_positions[descriptive_name] = {
                        'values': value,
                        'source': param.get('source', ''),
                        'confidence': param.get('confidence', 0.5)
                    }

        return named_positions

    def _create_descriptive_name(self, name: str) -> str:
        """
        Create a descriptive name from a parameter name.

        Args:
            name: Parameter name

        Returns:
            Descriptive name
        """
        # Remove path components
        name = name.split('.')[-1]

        # Replace underscores with spaces
        name = name.replace('_', ' ')

        # Add position if not present
        if 'position' not in name.lower() and 'pos' not in name.lower():
            name += ' position'

        return name.lower()

    def _calculate_overall_confidence(self, robot_config: Dict[str, Any],
                                     entry_points: Dict[str, Any],
                                     action_sequences: Dict[str, Any],
                                     parameters: Dict[str, Any]) -> float:
        """
        Calculate overall confidence.

        Args:
            robot_config: Robot configuration information
            entry_points: Entry point information
            action_sequences: Action sequence information
            parameters: Parameter information

        Returns:
            Overall confidence score
        """
        # Get individual confidences
        robot_confidence = robot_config.get('confidence', 0.0)
        entry_confidence = entry_points.get('confidence', 0.0)
        action_confidence = action_sequences.get('confidence', 0.0)
        param_confidence = parameters.get('confidence', 0.0)

        # Calculate weighted average
        weights = [0.3, 0.2, 0.3, 0.2]  # Robot, entry, action, param
        confidences = [robot_confidence, entry_confidence, action_confidence, param_confidence]

        return sum(w * c for w, c in zip(weights, confidences)) / sum(weights)

    def _identify_gaps(self, robot_config: Dict[str, Any],
                      entry_points: Dict[str, Any],
                      action_sequences: Dict[str, Any],
                      parameters: Dict[str, Any]) -> List[str]:
        """
        Identify gaps in the extracted information.

        Args:
            robot_config: Robot configuration information
            entry_points: Entry point information
            action_sequences: Action sequence information
            parameters: Parameter information

        Returns:
            List of identified gaps
        """
        gaps = []

        # Check robot configuration
        if not robot_config.get('name'):
            gaps.append('robot name')
        if not robot_config.get('dof'):
            gaps.append('degrees of freedom')
        if not robot_config.get('joint_names'):
            gaps.append('joint names')
        if not robot_config.get('joint_limits'):
            gaps.append('joint limits')

        # Check entry points
        if not entry_points.get('entry_points'):
            gaps.append('entry points')

        # Check action sequences
        if not action_sequences.get('sequences'):
            gaps.append('action sequences')

        # Check parameters
        if not parameters.get('parameters'):
            gaps.append('parameters')

        # Check for specific parameter types
        grouped_params = parameters.get('grouped_parameters', {})
        if not grouped_params.get('positions'):
            gaps.append('position parameters')
        if not grouped_params.get('joint_positions'):
            gaps.append('joint position parameters')

        return gaps

    def _is_mvp1_ready(self, robot_config: Dict[str, Any],
                      entry_points: Dict[str, Any],
                      action_sequences: Dict[str, Any],
                      parameters: Dict[str, Any],
                      gaps: List[str]) -> bool:
        """
        Determine if the result is MVP1-ready.

        Args:
            robot_config: Robot configuration information
            entry_points: Entry point information
            action_sequences: Action sequence information
            parameters: Parameter information
            gaps: Identified gaps

        Returns:
            True if the result is MVP1-ready, False otherwise
        """
        # Check if we have the minimum required information
        has_robot_info = (robot_config.get('dof') is not None or
                         len(robot_config.get('joint_names', [])) > 0)
        has_entry_point = len(entry_points.get('entry_points', [])) > 0
        has_action_sequence = len(action_sequences.get('sequences', [])) > 0

        # Check overall confidence
        overall_confidence = self._calculate_overall_confidence(
            robot_config, entry_points, action_sequences, parameters
        )

        # MVP1-ready if we have basic robot info, at least one entry point,
        # at least one action sequence, and overall confidence > 0.5
        return (has_robot_info and has_entry_point and
                has_action_sequence and overall_confidence > 0.5)