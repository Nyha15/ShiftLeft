# Original code from: https://github.com/volunt4s/Simple-MuJoCo-PickNPlace
# File: main.py, robot.py, environment.py

import mujoco
import numpy as np
import cv2
from typing import Tuple, Dict, Any, List

class SimpleRobot:
    """Simple robot class for pick and place tasks"""
    
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model = model
        self.data = data
        
        # Robot parameters
        self.num_joints = 6
        self.end_effector_name = "end_effector"
        self.gripper_joints = [6, 7]  # Assuming 2 gripper joints
        
        # Control parameters
        self.position_gain = 100.0
        self.velocity_gain = 20.0
        self.max_velocity = 2.0
        
    def get_end_effector_position(self) -> np.ndarray:
        """Get current end effector position"""
        return self.data.xpos[self.model.body(self.end_effector_name).id].copy()
    
    def get_end_effector_orientation(self) -> np.ndarray:
        """Get current end effector orientation (quaternion)"""
        return self.data.xquat[self.model.body(self.end_effector_name).id].copy()
    
    def get_joint_positions(self) -> np.ndarray:
        """Get current joint positions"""
        return self.data.qpos[:self.num_joints].copy()
    
    def get_joint_velocities(self) -> np.ndarray:
        """Get current joint velocities"""
        return self.data.qvel[:self.num_joints].copy()
    
    def set_joint_positions(self, positions: np.ndarray):
        """Set joint positions"""
        self.data.qpos[:self.num_joints] = positions
        mujoco.mj_forward(self.model, self.data)
    
    def set_joint_velocities(self, velocities: np.ndarray):
        """Set joint velocities"""
        self.data.qvel[:self.num_joints] = velocities
    
    def move_to_position(self, target_position: np.ndarray, max_iterations: int = 200):
        """Move end effector to target position using inverse kinematics"""
        
        step_size = 0.01
        
        for iteration in range(max_iterations):
            current_position = self.get_end_effector_position()
            position_error = target_position - current_position
            
            # Check if target reached
            if np.linalg.norm(position_error) < 0.01:
                break
            
            # Calculate Jacobian
            jacobian = np.zeros((3, self.num_joints))
            mujoco.mj_jac(self.model, self.data, jacobian, None, 
                         np.zeros(3), self.model.body(self.end_effector_name).id)
            
            # Pseudo-inverse for joint velocity
            joint_velocity = np.linalg.pinv(jacobian) @ position_error
            
            # Limit velocity
            velocity_magnitude = np.linalg.norm(joint_velocity)
            if velocity_magnitude > self.max_velocity:
                joint_velocity = joint_velocity * (self.max_velocity / velocity_magnitude)
            
            # Update joint positions
            current_joint_pos = self.get_joint_positions()
            new_joint_pos = current_joint_pos + joint_velocity * step_size
            
            # Apply joint limits
            for i in range(self.num_joints):
                lower_limit = self.model.jnt_range[i, 0]
                upper_limit = self.model.jnt_range[i, 1]
                new_joint_pos[i] = np.clip(new_joint_pos[i], lower_limit, upper_limit)
            
            self.set_joint_positions(new_joint_pos)
    
    def control_gripper(self, action: float):
        """Control gripper (0.0 = open, 1.0 = close)"""
        
        if action > 0.5:  # Close gripper
            gripper_position = 0.0
        else:  # Open gripper
            gripper_position = 0.04
        
        for joint_id in self.gripper_joints:
            if joint_id < len(self.data.qpos):
                self.data.qpos[joint_id] = gripper_position

class SimpleEnvironment:
    """Simple pick and place environment"""
    
    def __init__(self, xml_path: str):
        self.xml_path = xml_path
        
        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        # Initialize robot
        self.robot = SimpleRobot(self.model, self.data)
        
        # Environment parameters
        self.max_steps = 1000
        self.step_count = 0
        
        # Task parameters
        self.pick_positions = [
            np.array([0.3, 0.0, 0.1]),
            np.array([0.5, 0.1, 0.1]),
            np.array([0.7, -0.05, 0.1])
        ]
        self.place_positions = [
            np.array([0.8, 0.0, 0.1]),
            np.array([0.9, 0.1, 0.1]),
            np.array([0.6, -0.05, 0.1])
        ]
        self.current_task_idx = 0
        
        # State tracking
        self.objects_grasped = [False, False, False]
        self.objects_placed = [False, False, False]
        
    def reset(self) -> Dict[str, np.ndarray]:
        """Reset environment to initial state"""
        self.step_count = 0
        self.current_task_idx = 0
        self.objects_grasped = [False, False, False]
        self.objects_placed = [False, False, False]
        
        # Reset robot to home position
        home_config = np.array([0, -np.pi/2, 0, -np.pi/2, 0, 0])
        self.robot.set_joint_positions(home_config)
        
        # Reset gripper to open position
        self.robot.control_gripper(0.0)
        
        # Get initial observation
        observation = self._get_observation()
        
        return observation
    
    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, Any]]:
        """Execute action and return next state"""
        
        reward = 0.0
        done = False
        
        # Parse action
        gripper_action = action[0]  # 0.0 = open, 1.0 = close
        target_position = action[1:4]  # [x, y, z]
        
        # Execute action
        self._execute_action(target_position, gripper_action)
        
        # Update step count
        self.step_count += 1
        
        # Check task completion
        if self._is_task_completed():
            reward += 10.0
            done = True
        elif self.step_count >= self.max_steps:
            done = True
        
        # Calculate reward
        if not done:
            reward = self._calculate_reward()
        
        # Get observation
        observation = self._get_observation()
        
        # Update info
        info = {
            "step_count": self.step_count,
            "task_completed": self._is_task_completed(),
            "current_task_idx": self.current_task_idx,
            "objects_grasped": self.objects_grasped.copy(),
            "objects_placed": self.objects_placed.copy(),
            "end_effector_pos": self.robot.get_end_effector_position()
        }
        
        return observation, reward, done, info
    
    def _execute_action(self, target_position: np.ndarray, gripper_action: float):
        """Execute the action"""
        
        # Move robot to target position
        self.robot.move_to_position(target_position)
        
        # Control gripper
        self.robot.control_gripper(gripper_action)
        
        # Update task state
        self._update_task_state()
    
    def _update_task_state(self):
        """Update the state of pick and place tasks"""
        
        end_effector_pos = self.robot.get_end_effector_position()
        
        # Check if object is grasped
        if not self.objects_grasped[self.current_task_idx]:
            pick_pos = self.pick_positions[self.current_task_idx]
            distance_to_pick = np.linalg.norm(end_effector_pos - pick_pos)
            
            if distance_to_pick < 0.02:
                self.objects_grasped[self.current_task_idx] = True
        
        # Check if object is placed
        elif not self.objects_placed[self.current_task_idx]:
            place_pos = self.place_positions[self.current_task_idx]
            distance_to_place = np.linalg.norm(end_effector_pos - place_pos)
            
            if distance_to_place < 0.02:
                self.objects_placed[self.current_task_idx] = True
                self.current_task_idx += 1
    
    def _is_task_completed(self) -> bool:
        """Check if all tasks are completed"""
        return all(self.objects_placed)
    
    def _calculate_reward(self) -> float:
        """Calculate reward based on current state"""
        
        reward = 0.0
        
        # Reward for grasping objects
        for i, grasped in enumerate(self.objects_grasped):
            if grasped:
                reward += 2.0
        
        # Reward for placing objects
        for i, placed in enumerate(self.objects_placed):
            if placed:
                reward += 3.0
        
        # Reward for being close to current task target
        if self.current_task_idx < len(self.pick_positions):
            if not self.objects_grasped[self.current_task_idx]:
                # Reward for being close to pick position
                target_pos = self.pick_positions[self.current_task_idx]
            else:
                # Reward for being close to place position
                target_pos = self.place_positions[self.current_task_idx]
            
            current_pos = self.robot.get_end_effector_position()
            distance = np.linalg.norm(current_pos - target_pos)
            
            if distance < 0.05:
                reward += 1.0
            elif distance < 0.1:
                reward += 0.5
        
        # Penalty for large joint velocities
        joint_velocities = self.robot.get_joint_velocities()
        velocity_penalty = -0.1 * np.sum(joint_velocities ** 2)
        reward += velocity_penalty
        
        return reward
    
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Get current observation"""
        
        # Robot state
        joint_positions = self.robot.get_joint_positions()
        joint_velocities = self.robot.get_joint_velocities()
        end_effector_position = self.robot.get_end_effector_position()
        end_effector_orientation = self.robot.get_end_effector_orientation()
        
        # Task state
        task_state = np.array([
            float(self.current_task_idx),
            float(sum(self.objects_grasped)),
            float(sum(self.objects_placed))
        ])
        
        # Target positions
        if self.current_task_idx < len(self.pick_positions):
            if not self.objects_grasped[self.current_task_idx]:
                target_position = self.pick_positions[self.current_task_idx]
            else:
                target_position = self.place_positions[self.current_task_idx]
        else:
            target_position = np.array([0.0, 0.0, 0.0])
        
        observation = {
            "joint_positions": joint_positions,
            "joint_velocities": joint_velocities,
            "end_effector_position": end_effector_position,
            "end_effector_orientation": end_effector_orientation,
            "task_state": task_state,
            "target_position": target_position,
            "objects_grasped": np.array(self.objects_grasped, dtype=float),
            "objects_placed": np.array(self.objects_placed, dtype=float)
        }
        
        return observation
    
    def get_action_space(self) -> Dict[str, np.ndarray]:
        """Get action space specification"""
        return {
            "shape": (4,),
            "low": np.array([0.0, -1.0, -1.0, -1.0]),
            "high": np.array([1.0, 1.0, 1.0, 1.0])
        }
    
    def get_observation_space(self) -> Dict[str, np.ndarray]:
        """Get observation space specification"""
        return {
            "joint_positions": {"shape": (self.robot.num_joints,)},
            "joint_velocities": {"shape": (self.robot.num_joints,)},
            "end_effector_position": {"shape": (3,)},
            "end_effector_orientation": {"shape": (4,)},
            "task_state": {"shape": (3,)},
            "target_position": {"shape": (3,)},
            "objects_grasped": {"shape": (len(self.pick_positions),)},
            "objects_placed": {"shape": (len(self.place_positions),)}
        }

def run_pick_and_place_episode(env: SimpleEnvironment, max_steps: int = 100) -> Dict[str, Any]:
    """Run a complete pick and place episode"""
    
    observation = env.reset()
    total_reward = 0.0
    step_count = 0
    
    while step_count < max_steps:
        # Simple policy: move to target and grasp/place
        target_position = observation["target_position"]
        current_position = observation["end_effector_position"]
        
        # Determine gripper action based on task state
        if not env.objects_grasped[env.current_task_idx]:
            # Moving to pick position
            gripper_action = 0.0  # Open gripper
        elif not env.objects_placed[env.current_task_idx]:
            # Moving to place position
            gripper_action = 1.0  # Close gripper
        else:
            # Task completed, keep gripper closed
            gripper_action = 1.0
        
        # Create action
        action = np.array([gripper_action, *target_position])
        
        # Execute step
        observation, reward, done, info = env.step(action)
        total_reward += reward
        step_count += 1
        
        if done:
            break
    
    return {
        "total_reward": total_reward,
        "step_count": step_count,
        "task_completed": env._is_task_completed(),
        "objects_grasped": env.objects_grasped.copy(),
        "objects_placed": env.objects_placed.copy()
    }

if __name__ == "__main__":
    # Example usage
    env = SimpleEnvironment("simple_picknplace.xml")
    
    # Reset environment
    obs = env.reset()
    print("Environment reset. Initial observation keys:", obs.keys())
    
    # Run episode
    episode_result = run_pick_and_place_episode(env, max_steps=50)
    
    print(f"Episode completed:")
    print(f"  Total reward: {episode_result['total_reward']}")
    print(f"  Steps taken: {episode_result['step_count']}")
    print(f"  Task completed: {episode_result['task_completed']}")
    print(f"  Objects grasped: {episode_result['objects_grasped']}")
    print(f"  Objects placed: {episode_result['objects_placed']}") 