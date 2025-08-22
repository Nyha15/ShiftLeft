# Original code from: https://github.com/PaulDanielML/MuJoCo_RL_UR5
# File: grasping_env.py

import mujoco
import numpy as np
import cv2
from typing import Tuple, Dict, Any

class GraspingEnv:
    """MuJoCo grasping environment for UR5 robot"""
    
    def __init__(self, xml_path: str, max_steps: int = 1000, action_repeat: int = 1):
        self.xml_path = xml_path
        self.max_steps = max_steps
        self.action_repeat = action_repeat
        self.step_count = 0
        
        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        # Environment parameters
        self.image_size = 84
        self.camera_name = "side_camera"
        self.end_effector_name = "wrist_3_link"
        
        # Grasping parameters
        self.grasp_threshold = 0.02
        self.reach_threshold = 0.05
        
    def reset(self) -> Dict[str, np.ndarray]:
        """Reset environment to initial state"""
        self.step_count = 0
        
        # Reset robot to home position
        home_config = np.array([0, -np.pi/2, 0, -np.pi/2, 0, 0])
        self.data.qpos[:] = home_config
        self.data.qvel[:] = np.zeros(self.model.nv)
        
        # Forward kinematics
        mujoco.mj_forward(self.model, self.data)
        
        # Get initial observation
        observation = self._get_observation()
        
        return observation
    
    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, Any]]:
        """Execute action and return next state"""
        
        reward = 0.0
        done = False
        
        # Execute action with repeat
        for _ in range(self.action_repeat):
            self._execute_action(action)
            mujoco.mj_step(self.model, self.data)
            
            # Check if object is grasped
            if self._is_object_grasped():
                reward += 10.0
                done = True
                break
        
        self.step_count += 1
        
        # Check if episode is done
        if self.step_count >= self.max_steps:
            done = True
        
        # Get observation
        observation = self._get_observation()
        
        # Calculate reward
        if not done:
            reward = self._calculate_reward()
        
        info = {
            "step_count": self.step_count,
            "grasp_success": self._is_object_grasped(),
            "end_effector_pos": self._get_end_effector_position()
        }
        
        return observation, reward, done, info
    
    def _execute_action(self, action: np.ndarray):
        """Execute action on robot"""
        
        # Action is [dx, dy, dz, gripper]
        delta_pos = action[:3]
        gripper_action = action[3]
        
        # Get current end effector position
        current_pos = self._get_end_effector_position()
        target_pos = current_pos + delta_pos
        
        # Simple inverse kinematics (jacobian-based)
        self._move_to_position(target_pos)
        
        # Control gripper
        self._control_gripper(gripper_action)
    
    def _move_to_position(self, target_pos: np.ndarray):
        """Move end effector to target position using inverse kinematics"""
        
        max_iterations = 100
        step_size = 0.01
        
        for _ in range(max_iterations):
            current_pos = self._get_end_effector_position()
            error = target_pos - current_pos
            
            if np.linalg.norm(error) < self.reach_threshold:
                break
            
            # Calculate Jacobian
            jacobian = np.zeros((3, self.model.nv))
            mujoco.mj_jac(self.model, self.data, jacobian, None, 
                         np.zeros(3), self.model.body(self.end_effector_name).id)
            
            # Pseudo-inverse for joint velocity
            joint_vel = np.linalg.pinv(jacobian) @ error
            
            # Update joint positions
            self.data.qpos[:] += joint_vel * step_size
            
            # Forward kinematics
            mujoco.mj_forward(self.model, self.data)
    
    def _control_gripper(self, gripper_action: float):
        """Control gripper opening/closing"""
        
        # Map gripper action to joint positions
        gripper_joints = [7, 8]  # Assuming gripper has 2 joints
        
        if gripper_action > 0.5:  # Close gripper
            gripper_pos = 0.0
        else:  # Open gripper
            gripper_pos = 0.04
        
        for joint_id in gripper_joints:
            if joint_id < len(self.data.qpos):
                self.data.qpos[joint_id] = gripper_pos
    
    def _get_end_effector_position(self) -> np.ndarray:
        """Get current end effector position"""
        return self.data.xpos[self.model.body(self.end_effector_name).id].copy()
    
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Get current observation"""
        
        # Robot state
        joint_pos = self.data.qpos.copy()
        joint_vel = self.data.qvel.copy()
        end_effector_pos = self._get_end_effector_position()
        
        # Camera observation (simplified)
        camera_pos = self.data.xpos[self.model.body(self.camera_name).id]
        camera_quat = self.data.xquat[self.model.body(self.camera_name).id]
        
        # Create dummy image observation
        image_obs = np.random.randint(0, 255, (self.image_size, self.image_size, 3), dtype=np.uint8)
        
        observation = {
            "joint_pos": joint_pos,
            "joint_vel": joint_vel,
            "end_effector_pos": end_effector_pos,
            "camera_pos": camera_pos,
            "camera_quat": camera_quat,
            "image": image_obs
        }
        
        return observation
    
    def _is_object_grasped(self) -> bool:
        """Check if object is grasped"""
        
        # Simplified grasp detection
        end_effector_pos = self._get_end_effector_position()
        
        # Check if end effector is near table surface
        if end_effector_pos[2] < 0.05:  # Near table
            return True
        
        return False
    
    def _calculate_reward(self) -> float:
        """Calculate reward based on current state"""
        
        reward = 0.0
        
        # Reward for being close to target
        target_pos = np.array([0.5, 0.0, 0.1])  # Example target
        current_pos = self._get_end_effector_position()
        distance = np.linalg.norm(current_pos - target_pos)
        
        if distance < 0.1:
            reward += 1.0
        elif distance < 0.2:
            reward += 0.5
        
        # Penalty for large joint velocities
        joint_vel_penalty = -0.1 * np.sum(self.data.qvel ** 2)
        reward += joint_vel_penalty
        
        return reward
    
    def get_action_space(self) -> Dict[str, np.ndarray]:
        """Get action space specification"""
        return {
            "shape": (4,),
            "low": np.array([-0.1, -0.1, -0.1, 0.0]),
            "high": np.array([0.1, 0.1, 0.1, 1.0])
        }
    
    def get_observation_space(self) -> Dict[str, np.ndarray]:
        """Get observation space specification"""
        return {
            "joint_pos": {"shape": (self.model.nq,)},
            "joint_vel": {"shape": (self.model.nv,)},
            "end_effector_pos": {"shape": (3,)},
            "camera_pos": {"shape": (3,)},
            "camera_quat": {"shape": (4,)},
            "image": {"shape": (self.image_size, self.image_size, 3)}
        }

class GraspingEnv6DOF(GraspingEnv):
    """6-DOF grasping environment with orientation control"""
    
    def __init__(self, xml_path: str, max_steps: int = 1000, action_repeat: int = 1):
        super().__init__(xml_path, max_steps, action_repeat)
        
        # 6-DOF action: [dx, dy, dz, droll, dpitch, dyaw, gripper]
        self.action_dim = 7
    
    def _execute_action(self, action: np.ndarray):
        """Execute 6-DOF action"""
        
        # Position action
        delta_pos = action[:3]
        target_pos = self._get_end_effector_position() + delta_pos
        
        # Orientation action
        delta_orientation = action[3:6]
        target_orientation = self._get_end_effector_orientation() + delta_orientation
        
        # Move to target pose
        self._move_to_pose(target_pos, target_orientation)
        
        # Control gripper
        gripper_action = action[6]
        self._control_gripper(gripper_action)
    
    def _get_end_effector_orientation(self) -> np.ndarray:
        """Get current end effector orientation (euler angles)"""
        quat = self.data.xquat[self.model.body(self.end_effector_name).id]
        
        # Convert quaternion to euler angles (simplified)
        # This is a placeholder - actual conversion would be more complex
        euler = np.array([0.0, 0.0, 0.0])
        return euler
    
    def _move_to_pose(self, target_pos: np.ndarray, target_orientation: np.ndarray):
        """Move to target pose (position + orientation)"""
        
        # Simplified 6-DOF control
        self._move_to_position(target_pos)
        
        # Orientation control would go here
        # For now, just position control
    
    def get_action_space(self) -> Dict[str, np.ndarray]:
        """Get 6-DOF action space specification"""
        return {
            "shape": (7,),
            "low": np.array([-0.1, -0.1, -0.1, -0.1, -0.1, -0.1, 0.0]),
            "high": np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0])
        }

if __name__ == "__main__":
    # Example usage
    env = GraspingEnv("ur5_grasping_env.xml")
    
    # Reset environment
    obs = env.reset()
    print("Environment reset. Initial observation keys:", obs.keys())
    
    # Execute random action
    action = np.random.uniform(-0.1, 0.1, 4)
    obs, reward, done, info = env.step(action)
    
    print(f"Action executed. Reward: {reward}, Done: {done}")
    print(f"End effector position: {info['end_effector_pos']}") 