# Unit tests for: training.py (UR5 Grasping Environment)
# Tests the grasping_env.py functionality

import pytest
import numpy as np
from training import GraspingEnv, GraspingEnv6DOF

class TestGraspingEnv:
    """Test cases for GraspingEnv class"""
    
    def test_initialization(self):
        """Test environment initialization"""
        env = GraspingEnv("ur5_grasping_env.xml")
        
        assert env.max_steps == 1000
        assert env.action_repeat == 1
        assert env.image_size == 84
        assert env.camera_name == "side_camera"
        assert env.end_effector_name == "wrist_3_link"
        assert env.grasp_threshold == 0.02
        assert env.reach_threshold == 0.05
        assert env.step_count == 0
    
    def test_reset(self):
        """Test environment reset"""
        env = GraspingEnv("ur5_grasping_env.xml")
        obs = env.reset()
        
        assert env.step_count == 0
        assert isinstance(obs, dict)
        assert "joint_pos" in obs
        assert "joint_vel" in obs
        assert "end_effector_pos" in obs
        assert "camera_pos" in obs
        assert "camera_quat" in obs
        assert "image" in obs
        
        # Check observation shapes
        assert obs["joint_pos"].shape == (env.model.nq,)
        assert obs["joint_vel"].shape == (env.model.nv,)
        assert obs["end_effector_pos"].shape == (3,)
        assert obs["camera_pos"].shape == (3,)
        assert obs["camera_quat"].shape == (4,)
        assert obs["image"].shape == (env.image_size, env.image_size, 3)
    
    def test_action_space(self):
        """Test action space specification"""
        env = GraspingEnv("ur5_grasping_env.xml")
        action_space = env.get_action_space()
        
        assert action_space["shape"] == (4,)
        assert action_space["low"].shape == (4,)
        assert action_space["high"].shape == (4,)
        assert np.all(action_space["low"] == np.array([-0.1, -0.1, -0.1, 0.0]))
        assert np.all(action_space["high"] == np.array([0.1, 0.1, 0.1, 1.0]))
    
    def test_observation_space(self):
        """Test observation space specification"""
        env = GraspingEnv("ur5_grasping_env.xml")
        obs_space = env.get_observation_space()
        
        assert "joint_pos" in obs_space
        assert "joint_vel" in obs_space
        assert "end_effector_pos" in obs_space
        assert "camera_pos" in obs_space
        assert "camera_quat" in obs_space
        assert "image" in obs_space
        
        assert obs_space["joint_pos"]["shape"] == (env.model.nq,)
        assert obs_space["joint_vel"]["shape"] == (env.model.nv,)
        assert obs_space["end_effector_pos"]["shape"] == (3,)
        assert obs_space["camera_pos"]["shape"] == (3,)
        assert obs_space["camera_quat"]["shape"] == (4,)
        assert obs_space["image"]["shape"] == (env.image_size, env.image_size, 3)
    
    def test_gripper_control(self):
        """Test gripper control functionality"""
        env = GraspingEnv("ur5_grasping_env.xml")
        env.reset()
        
        # Test gripper opening
        env._control_gripper(0.0)  # Open gripper
        if 7 < len(env.data.qpos) and 8 < len(env.data.qpos):
            assert env.data.qpos[7] == 0.04
            assert env.data.qpos[8] == 0.04
        
        # Test gripper closing
        env._control_gripper(1.0)  # Close gripper
        if 7 < len(env.data.qpos) and 8 < len(env.data.qpos):
            assert env.data.qpos[7] == 0.0
            assert env.data.qpos[8] == 0.0
    
    def test_end_effector_position(self):
        """Test end effector position retrieval"""
        env = GraspingEnv("ur5_grasping_env.xml")
        env.reset()
        
        pos = env._get_end_effector_position()
        assert pos.shape == (3,)
        assert not np.any(np.isnan(pos))
        assert np.linalg.norm(pos) < 10.0  # Reasonable range
    
    def test_grasp_detection(self):
        """Test object grasp detection"""
        env = GraspingEnv("ur5_grasping_env.xml")
        env.reset()
        
        # Test normal position (not grasped)
        env.data.qpos[:] = np.array([0, -np.pi/2, 0, -np.pi/2, 0, 0])
        mujoco.mj_forward(env.model, env.data)
        assert not env._is_object_grasped()
        
        # Test near table position (grasped)
        env.data.qpos[:] = np.array([0, -np.pi/2, 0, -np.pi/2, 0, 0])
        env.data.qpos[2] = 0.0  # Move down
        mujoco.mj_forward(env.model, env.data)
        # This would depend on the actual model configuration
    
    def test_reward_calculation(self):
        """Test reward calculation"""
        env = GraspingEnv("ur5_grasping_env.xml")
        env.reset()
        
        reward = env._calculate_reward()
        assert isinstance(reward, float)
        assert reward <= 1.5  # Max reward from distance + velocity penalty
    
    def test_step_execution(self):
        """Test step execution"""
        env = GraspingEnv("ur5_grasping_env.xml")
        env.reset()
        
        # Execute random action
        action = np.random.uniform(-0.1, 0.1, 4)
        obs, reward, done, info = env.step(action)
        
        assert env.step_count == 1
        assert isinstance(obs, dict)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)
        assert "step_count" in info
        assert "grasp_success" in info
        assert "end_effector_pos" in info

class TestGraspingEnv6DOF:
    """Test cases for GraspingEnv6DOF class"""
    
    def test_initialization(self):
        """Test 6-DOF environment initialization"""
        env = GraspingEnv6DOF("ur5_grasping_env.xml")
        
        assert env.action_dim == 7
        assert isinstance(env, GraspingEnv)  # Should inherit from base class
    
    def test_action_space_6dof(self):
        """Test 6-DOF action space specification"""
        env = GraspingEnv6DOF("ur5_grasping_env.xml")
        action_space = env.get_action_space()
        
        assert action_space["shape"] == (7,)
        assert action_space["low"].shape == (7,)
        assert action_space["high"].shape == (7,)
        assert np.all(action_space["low"] == np.array([-0.1, -0.1, -0.1, -0.1, -0.1, -0.1, 0.0]))
        assert np.all(action_space["high"] == np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0]))
    
    def test_end_effector_orientation(self):
        """Test end effector orientation retrieval"""
        env = GraspingEnv6DOF("ur5_grasping_env.xml")
        env.reset()
        
        orientation = env._get_end_effector_orientation()
        assert orientation.shape == (3,)
        assert not np.any(np.isnan(orientation))
    
    def test_6dof_action_execution(self):
        """Test 6-DOF action execution"""
        env = GraspingEnv6DOF("ur5_grasping_env.xml")
        env.reset()
        
        # Execute 6-DOF action
        action = np.random.uniform(-0.1, 0.1, 7)
        env._execute_action(action)
        
        # Should not raise any errors
        assert True

class TestEnvironmentIntegration:
    """Integration tests for the grasping environment"""
    
    def test_full_episode(self):
        """Test a complete episode"""
        env = GraspingEnv("ur5_grasping_env.xml")
        obs = env.reset()
        
        episode_reward = 0.0
        step_count = 0
        
        while step_count < 10:  # Short episode for testing
            action = np.random.uniform(-0.1, 0.1, 4)
            obs, reward, done, info = env.step(action)
            
            episode_reward += reward
            step_count += 1
            
            if done:
                break
        
        assert step_count > 0
        assert isinstance(episode_reward, float)
    
    def test_environment_consistency(self):
        """Test environment state consistency"""
        env = GraspingEnv("ur5_grasping_env.xml")
        
        # Multiple resets should work
        for _ in range(3):
            obs = env.reset()
            assert env.step_count == 0
            assert "joint_pos" in obs
        
        # Step count should increment properly
        env.reset()
        action = np.random.uniform(-0.1, 0.1, 4)
        env.step(action)
        assert env.step_count == 1
    
    def test_action_validation(self):
        """Test action validation and bounds"""
        env = GraspingEnv("ur5_grasping_env.xml")
        env.reset()
        
        # Test actions within bounds
        valid_action = np.array([0.05, 0.05, 0.05, 0.5])
        obs, reward, done, info = env.step(valid_action)
        assert isinstance(obs, dict)
        
        # Test actions at bounds
        bound_action = np.array([0.1, 0.1, 0.1, 1.0])
        obs, reward, done, info = env.step(bound_action)
        assert isinstance(obs, dict)
    
    def test_observation_consistency(self):
        """Test observation consistency across steps"""
        env = GraspingEnv("ur5_grasping_env.xml")
        obs1 = env.reset()
        
        action = np.random.uniform(-0.1, 0.1, 4)
        obs2, reward, done, info = env.step(action)
        
        # Observations should have same structure
        assert obs1.keys() == obs2.keys()
        
        # All observations should have expected shapes
        for key in obs1.keys():
            assert obs1[key].shape == obs2[key].shape 