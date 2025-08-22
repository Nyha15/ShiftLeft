# Unit tests for: training.py (Panda MuJoCo Gym Environment)
# Tests the panda_env.py, panda_env_reach.py, panda_env_push.py functionality

import pytest
import numpy as np
from training import PandaEnvBase, PandaEnvReach, PandaEnvPush

class TestPandaEnvBase:
    """Test cases for PandaEnvBase class"""
    
    def test_initialization(self):
        """Test base environment initialization"""
        env = PandaEnvBase("panda_env.xml")
        
        assert env.max_steps == 1000
        assert env.action_repeat == 1
        assert env.image_size == 84
        assert env.camera_name == "camera0"
        assert env.end_effector_name == "panda_hand"
        assert env.num_joints == 7
        assert env.num_gripper_joints == 2
        assert env.position_threshold == 0.02
        assert env.velocity_threshold == 0.1
        assert env.step_count == 0
    
    def test_reset(self):
        """Test environment reset"""
        env = PandaEnvBase("panda_env.xml")
        obs = env.reset()
        
        assert env.step_count == 0
        assert isinstance(obs, dict)
        assert "joint_pos" in obs
        assert "joint_vel" in obs
        assert "gripper_pos" in obs
        assert "end_effector_pos" in obs
        assert "end_effector_quat" in obs
        assert "camera_pos" in obs
        assert "camera_quat" in obs
        assert "image" in obs
        
        # Check observation shapes
        assert obs["joint_pos"].shape == (env.num_joints,)
        assert obs["joint_vel"].shape == (env.num_joints,)
        assert obs["gripper_pos"].shape == (env.num_gripper_joints,)
        assert obs["end_effector_pos"].shape == (3,)
        assert obs["end_effector_quat"].shape == (4,)
        assert obs["camera_pos"].shape == (3,)
        assert obs["camera_quat"].shape == (4,)
        assert obs["image"].shape == (env.image_size, env.image_size, 3)
    
    def test_gripper_control(self):
        """Test gripper control functionality"""
        env = PandaEnvBase("panda_env.xml")
        env.reset()
        
        # Test gripper opening
        env._control_gripper(0.0)  # Open gripper
        gripper_joints = [env.num_joints, env.num_joints + 1]
        for joint_id in gripper_joints:
            if joint_id < len(env.data.qpos):
                assert env.data.qpos[joint_id] == 0.04
        
        # Test gripper closing
        env._control_gripper(1.0)  # Close gripper
        for joint_id in gripper_joints:
            if joint_id < len(env.data.qpos):
                assert env.data.qpos[joint_id] == 0.0
    
    def test_end_effector_position(self):
        """Test end effector position retrieval"""
        env = PandaEnvBase("panda_env.xml")
        env.reset()
        
        pos = env._get_end_effector_position()
        assert pos.shape == (3,)
        assert not np.any(np.isnan(pos))
        assert np.linalg.norm(pos) < 10.0  # Reasonable range
    
    def test_end_effector_orientation(self):
        """Test end effector orientation retrieval"""
        env = PandaEnvBase("panda_env.xml")
        env.reset()
        
        quat = env._get_end_effector_orientation()
        assert quat.shape == (4,)
        assert not np.any(np.isnan(quat))
        
        # Check quaternion normalization
        quat_magnitude = np.linalg.norm(quat)
        assert abs(quat_magnitude - 1.0) < 1e-6
    
    def test_observation_space(self):
        """Test observation space specification"""
        env = PandaEnvBase("panda_env.xml")
        obs_space = env.get_observation_space()
        
        assert "joint_pos" in obs_space
        assert "joint_vel" in obs_space
        assert "gripper_pos" in obs_space
        assert "end_effector_pos" in obs_space
        assert "end_effector_quat" in obs_space
        assert "camera_pos" in obs_space
        assert "camera_quat" in obs_space
        assert "image" in obs_space
        
        assert obs_space["joint_pos"]["shape"] == (env.num_joints,)
        assert obs_space["joint_vel"]["shape"] == (env.num_joints,)
        assert obs_space["gripper_pos"]["shape"] == (env.num_gripper_joints,)
        assert obs_space["end_effector_pos"]["shape"] == (3,)
        assert obs_space["end_effector_quat"]["shape"] == (4,)
        assert obs_space["camera_pos"]["shape"] == (3,)
        assert obs_space["camera_quat"]["shape"] == (4,)
        assert obs_space["image"]["shape"] == (env.image_size, env.image_size, 3)

class TestPandaEnvReach:
    """Test cases for PandaEnvReach class"""
    
    def test_initialization(self):
        """Test reaching environment initialization"""
        env = PandaEnvReach("panda_env.xml")
        
        assert isinstance(env, PandaEnvBase)
        assert len(env.target_positions) == 3
        assert env.current_target_idx == 0
        
        # Check target positions
        expected_targets = [
            np.array([0.5, 0.0, 0.3]),
            np.array([0.3, 0.2, 0.4]),
            np.array([0.7, -0.1, 0.2])
        ]
        
        for i, expected_target in enumerate(expected_targets):
            assert np.allclose(env.target_positions[i], expected_target)
    
    def test_action_space(self):
        """Test action space specification for reaching task"""
        env = PandaEnvReach("panda_env.xml")
        action_space = env.get_action_space()
        
        assert action_space["shape"] == (4,)
        assert action_space["low"].shape == (4,)
        assert action_space["high"].shape == (4,)
        assert np.all(action_space["low"] == np.array([-0.1, -0.1, -0.1, 0.0]))
        assert np.all(action_space["high"] == np.array([0.1, 0.1, 0.1, 1.0]))
    
    def test_task_completion(self):
        """Test reaching task completion detection"""
        env = PandaEnvReach("panda_env.xml")
        env.reset()
        
        # Test not completed
        assert not env._is_task_completed()
        
        # Test completed (would need proper model setup)
        # This is a placeholder test
    
    def test_reward_calculation(self):
        """Test reward calculation for reaching task"""
        env = PandaEnvReach("panda_env.xml")
        env.reset()
        
        reward = env._calculate_reward()
        assert isinstance(reward, float)
        assert reward <= 2.0  # Max reward from distance + velocity penalty
    
    def test_action_execution(self):
        """Test action execution in reaching environment"""
        env = PandaEnvReach("panda_env.xml")
        env.reset()
        
        # Execute random action
        action = np.random.uniform(-0.1, 0.1, 4)
        env._execute_action(action)
        
        # Should not raise any errors
        assert True

class TestPandaEnvPush:
    """Test cases for PandaEnvPush class"""
    
    def test_initialization(self):
        """Test pushing environment initialization"""
        env = PandaEnvPush("panda_env.xml")
        
        assert isinstance(env, PandaEnvBase)
        assert len(env.object_positions) == 3
        assert len(env.target_positions) == 3
        assert env.current_object_idx == 0
        
        # Check object positions
        expected_objects = [
            np.array([0.4, 0.0, 0.1]),
            np.array([0.5, 0.1, 0.1]),
            np.array([0.6, -0.05, 0.1])
        ]
        
        for i, expected_object in enumerate(expected_objects):
            assert np.allclose(env.object_positions[i], expected_object)
    
    def test_action_space(self):
        """Test action space specification for pushing task"""
        env = PandaEnvPush("panda_env.xml")
        action_space = env.get_action_space()
        
        assert action_space["shape"] == (4,)
        assert action_space["low"].shape == (4,)
        assert action_space["high"].shape == (4,)
        assert np.all(action_space["low"] == np.array([-0.1, -0.1, -0.1, 0.0]))
        assert np.all(action_space["high"] == np.array([0.1, 0.1, 0.1, 1.0]))
    
    def test_task_completion(self):
        """Test pushing task completion detection"""
        env = PandaEnvPush("panda_env.xml")
        env.reset()
        
        # Test not completed
        assert not env._is_task_completed()
    
    def test_reward_calculation(self):
        """Test reward calculation for pushing task"""
        env = PandaEnvPush("panda_env.xml")
        env.reset()
        
        reward = env._calculate_reward()
        assert isinstance(reward, float)
        assert reward <= 2.0  # Max reward from distance + velocity penalty
    
    def test_action_execution(self):
        """Test action execution in pushing environment"""
        env = PandaEnvPush("panda_env.xml")
        env.reset()
        
        # Execute random action
        action = np.random.uniform(-0.1, 0.1, 4)
        env._execute_action(action)
        
        # Should not raise any errors
        assert True

class TestEnvironmentIntegration:
    """Integration tests for the Panda environments"""
    
    def test_reach_environment_episode(self):
        """Test a complete reaching episode"""
        env = PandaEnvReach("panda_env.xml")
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
    
    def test_push_environment_episode(self):
        """Test a complete pushing episode"""
        env = PandaEnvPush("panda_env.xml")
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
        env = PandaEnvReach("panda_env.xml")
        
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
    
    def test_observation_consistency(self):
        """Test observation consistency across steps"""
        env = PandaEnvReach("panda_env.xml")
        obs1 = env.reset()
        
        action = np.random.uniform(-0.1, 0.1, 4)
        obs2, reward, done, info = env.step(action)
        
        # Observations should have same structure
        assert obs1.keys() == obs2.keys()
        
        # All observations should have expected shapes
        for key in obs1.keys():
            assert obs1[key].shape == obs2[key].shape
    
    def test_joint_limits(self):
        """Test joint limit validation"""
        env = PandaEnvReach("panda_env.xml")
        env.reset()
        
        # Test that joint positions are within reasonable bounds
        joint_pos = env.data.qpos[:env.num_joints]
        assert np.all(joint_pos >= -np.pi)
        assert np.all(joint_pos <= np.pi)
        
        # Test that gripper positions are within bounds
        gripper_pos = env.data.qpos[env.num_joints:env.num_joints + env.num_gripper_joints]
        assert np.all(gripper_pos >= 0.0)
        assert np.all(gripper_pos <= 0.04) 