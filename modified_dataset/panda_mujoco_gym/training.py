# Original code from: https://github.com/zichunxx/panda_mujoco_gym
# File: panda_env.py, panda_env_base.py, panda_env_reach.py, panda_env_push.py

import mujoco
import numpy as np
import cv2
from typing import Tuple, Dict, Any, Optional

class PandaEnvBase:
    """Base Panda environment class"""
    
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
        self.camera_name = "camera0"
        self.end_effector_name = "panda_hand"
        
        # Robot parameters
        self.num_joints = 7
        self.num_gripper_joints = 2
        
        # Control parameters
        self.position_threshold = 0.02
        self.velocity_threshold = 0.1
        
    def reset(self) -> Dict[str, np.ndarray]:
        """Reset environment to initial state"""
        self.step_count = 0
        
        # Reset robot to home position
        home_config = np.array([0, 0, 0, -np.pi/2, 0, np.pi/2, 0])
        self.data.qpos[:self.num_joints] = home_config
        self.data.qpos[self.num_joints:self.num_joints + self.num_gripper_joints] = [0.04, 0.04]  # Open gripper
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
            
            # Check task completion
            if self._is_task_completed():
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
            "task_completed": self._is_task_completed(),
            "end_effector_pos": self._get_end_effector_position()
        }
        
        return observation, reward, done, info
    
    def _execute_action(self, action: np.ndarray):
        """Execute action on robot - to be implemented by subclasses"""
        raise NotImplementedError
    
    def _is_task_completed(self) -> bool:
        """Check if task is completed - to be implemented by subclasses"""
        raise NotImplementedError
    
    def _calculate_reward(self) -> float:
        """Calculate reward based on current state - to be implemented by subclasses"""
        raise NotImplementedError
    
    def _get_end_effector_position(self) -> np.ndarray:
        """Get current end effector position"""
        return self.data.xpos[self.model.body(self.end_effector_name).id].copy()
    
    def _get_end_effector_orientation(self) -> np.ndarray:
        """Get current end effector orientation (quaternion)"""
        return self.data.xquat[self.model.body(self.end_effector_name).id].copy()
    
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Get current observation"""
        
        # Robot state
        joint_pos = self.data.qpos[:self.num_joints].copy()
        joint_vel = self.data.qvel[:self.num_joints].copy()
        gripper_pos = self.data.qpos[self.num_joints:self.num_joints + self.num_gripper_joints].copy()
        end_effector_pos = self._get_end_effector_position()
        end_effector_quat = self._get_end_effector_orientation()
        
        # Camera observation (simplified)
        camera_pos = self.data.xpos[self.model.body(self.camera_name).id]
        camera_quat = self.data.xquat[self.model.body(self.camera_name).id]
        
        # Create dummy image observation
        image_obs = np.random.randint(0, 255, (self.image_size, self.image_size, 3), dtype=np.uint8)
        
        observation = {
            "joint_pos": joint_pos,
            "joint_vel": joint_vel,
            "gripper_pos": gripper_pos,
            "end_effector_pos": end_effector_pos,
            "end_effector_quat": end_effector_quat,
            "camera_pos": camera_pos,
            "camera_quat": camera_quat,
            "image": image_obs
        }
        
        return observation
    
    def _control_gripper(self, gripper_action: float):
        """Control gripper opening/closing"""
        
        gripper_joints = [self.num_joints, self.num_joints + 1]
        
        if gripper_action > 0.5:  # Close gripper
            gripper_pos = 0.0
        else:  # Open gripper
            gripper_pos = 0.04
        
        for joint_id in gripper_joints:
            if joint_id < len(self.data.qpos):
                self.data.qpos[joint_id] = gripper_pos
    
    def _move_to_position(self, target_pos: np.ndarray, max_iterations: int = 100):
        """Move end effector to target position using inverse kinematics"""
        
        step_size = 0.01
        
        for _ in range(max_iterations):
            current_pos = self._get_end_effector_position()
            error = target_pos - current_pos
            
            if np.linalg.norm(error) < self.position_threshold:
                break
            
            # Calculate Jacobian
            jacobian = np.zeros((3, self.num_joints))
            mujoco.mj_jac(self.model, self.data, jacobian, None, 
                         np.zeros(3), self.model.body(self.end_effector_name).id)
            
            # Pseudo-inverse for joint velocity
            joint_vel = np.linalg.pinv(jacobian) @ error
            
            # Update joint positions
            self.data.qpos[:self.num_joints] += joint_vel * step_size
            
            # Forward kinematics
            mujoco.mj_forward(self.model, self.data)
    
    def get_action_space(self) -> Dict[str, np.ndarray]:
        """Get action space specification - to be implemented by subclasses"""
        raise NotImplementedError
    
    def get_observation_space(self) -> Dict[str, np.ndarray]:
        """Get observation space specification"""
        return {
            "joint_pos": {"shape": (self.num_joints,)},
            "joint_vel": {"shape": (self.num_joints,)},
            "gripper_pos": {"shape": (self.num_gripper_joints,)},
            "end_effector_pos": {"shape": (3,)},
            "end_effector_quat": {"shape": (4,)},
            "camera_pos": {"shape": (3,)},
            "camera_quat": {"shape": (4,)},
            "image": {"shape": (self.image_size, self.image_size, 3)}
        }

class PandaEnvReach(PandaEnvBase):
    """Panda environment for reaching tasks"""
    
    def __init__(self, xml_path: str, max_steps: int = 1000, action_repeat: int = 1):
        super().__init__(xml_path, max_steps, action_repeat)
        
        # Task-specific parameters
        self.target_positions = [
            np.array([0.5, 0.0, 0.3]),
            np.array([0.3, 0.2, 0.4]),
            np.array([0.7, -0.1, 0.2])
        ]
        self.current_target_idx = 0
        
    def _execute_action(self, action: np.ndarray):
        """Execute reaching action"""
        
        # Action is [dx, dy, dz, gripper]
        delta_pos = action[:3]
        gripper_action = action[3]
        
        # Get current end effector position
        current_pos = self._get_end_effector_position()
        target_pos = current_pos + delta_pos
        
        # Move to target position
        self._move_to_position(target_pos)
        
        # Control gripper
        self._control_gripper(gripper_action)
    
    def _is_task_completed(self) -> bool:
        """Check if reaching task is completed"""
        
        current_pos = self._get_end_effector_position()
        target_pos = self.target_positions[self.current_target_idx]
        
        distance = np.linalg.norm(current_pos - target_pos)
        return distance < self.position_threshold
    
    def _calculate_reward(self) -> float:
        """Calculate reward for reaching task"""
        
        reward = 0.0
        
        # Reward for being close to target
        current_pos = self._get_end_effector_position()
        target_pos = self.target_positions[self.current_target_idx]
        distance = np.linalg.norm(current_pos - target_pos)
        
        if distance < 0.05:
            reward += 2.0
        elif distance < 0.1:
            reward += 1.0
        elif distance < 0.2:
            reward += 0.5
        
        # Penalty for large joint velocities
        joint_vel_penalty = -0.1 * np.sum(self.data.qvel[:self.num_joints] ** 2)
        reward += joint_vel_penalty
        
        return reward
    
    def get_action_space(self) -> Dict[str, np.ndarray]:
        """Get action space specification for reaching task"""
        return {
            "shape": (4,),
            "low": np.array([-0.1, -0.1, -0.1, 0.0]),
            "high": np.array([0.1, 0.1, 0.1, 1.0])
        }

class PandaEnvPush(PandaEnvBase):
    """Panda environment for pushing tasks"""
    
    def __init__(self, xml_path: str, max_steps: int = 1000, action_repeat: int = 1):
        super().__init__(xml_path, max_steps, action_repeat)
        
        # Task-specific parameters
        self.object_positions = [
            np.array([0.4, 0.0, 0.1]),
            np.array([0.5, 0.1, 0.1]),
            np.array([0.6, -0.05, 0.1])
        ]
        self.target_positions = [
            np.array([0.8, 0.0, 0.1]),
            np.array([0.9, 0.1, 0.1]),
            np.array([0.7, -0.05, 0.1])
        ]
        self.current_object_idx = 0
        
    def _execute_action(self, action: np.ndarray):
        """Execute pushing action"""
        
        # Action is [dx, dy, dz, gripper]
        delta_pos = action[:3]
        gripper_action = action[3]
        
        # Get current end effector position
        current_pos = self._get_end_effector_position()
        target_pos = current_pos + delta_pos
        
        # Move to target position
        self._move_to_position(target_pos)
        
        # Control gripper
        self._control_gripper(gripper_action)
    
    def _is_task_completed(self) -> bool:
        """Check if pushing task is completed"""
        
        # Simplified: check if end effector is near object
        current_pos = self._get_end_effector_position()
        object_pos = self.object_positions[self.current_object_idx]
        
        distance = np.linalg.norm(current_pos - object_pos)
        return distance < self.position_threshold
    
    def _calculate_reward(self) -> float:
        """Calculate reward for pushing task"""
        
        reward = 0.0
        
        # Reward for being close to object
        current_pos = self._get_end_effector_position()
        object_pos = self.object_positions[self.current_object_idx]
        distance_to_object = np.linalg.norm(current_pos - object_pos)
        
        if distance_to_object < 0.05:
            reward += 2.0
        elif distance_to_object < 0.1:
            reward += 1.0
        elif distance_to_object < 0.2:
            reward += 0.5
        
        # Penalty for large joint velocities
        joint_vel_penalty = -0.1 * np.sum(self.data.qvel[:self.num_joints] ** 2)
        reward += joint_vel_penalty
        
        return reward
    
    def get_action_space(self) -> Dict[str, np.ndarray]:
        """Get action space specification for pushing task"""
        return {
            "shape": (4,),
            "low": np.array([-0.1, -0.1, -0.1, 0.0]),
            "high": np.array([0.1, 0.1, 0.1, 1.0])
        }

if __name__ == "__main__":
    # Example usage
    env = PandaEnvReach("panda_env.xml")
    
    # Reset environment
    obs = env.reset()
    print("Environment reset. Initial observation keys:", obs.keys())
    
    # Execute random action
    action = np.random.uniform(-0.1, 0.1, 4)
    obs, reward, done, info = env.step(action)
    
    print(f"Action executed. Reward: {reward}, Done: {done}")
    print(f"End effector position: {info['end_effector_pos']}")
    
    # Test pushing environment
    push_env = PandaEnvPush("panda_env.xml")
    push_obs = push_env.reset()
    print("Push environment reset. Observation keys:", push_obs.keys()) 