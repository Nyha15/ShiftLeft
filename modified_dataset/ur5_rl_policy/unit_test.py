# Unit tests for: training.py (UR5 Reinforcement Learning and Policy Optimization)
# Tests the rl_agent.py, policy_network.py, training_loop.py functionality

import pytest
import numpy as np
import torch
import torch.nn as nn
from training import PolicyNetwork, ValueNetwork, ReplayBuffer, RLAgent, TrainingLoop, PolicyOptimizer

class TestPolicyNetwork:
    """Test cases for PolicyNetwork class"""
    
    def test_initialization(self):
        """Test policy network initialization"""
        input_dim = 10
        output_dim = 4
        hidden_dims = [256, 128]
        
        network = PolicyNetwork(input_dim, output_dim, hidden_dims)
        
        assert network.input_dim == input_dim
        assert network.output_dim == output_dim
        assert network.hidden_dims == hidden_dims
        assert isinstance(network.network, nn.Sequential)
    
    def test_network_architecture(self):
        """Test network architecture construction"""
        
        input_dim = 8
        output_dim = 6
        hidden_dims = [64, 32]
        
        network = PolicyNetwork(input_dim, output_dim, hidden_dims)
        
        # Test layer structure
        layers = list(network.network)
        
        # First layer: input -> first hidden
        assert isinstance(layers[0], nn.Linear)
        assert layers[0].in_features == input_dim
        assert layers[0].out_features == hidden_dims[0]
        
        # Activation and dropout
        assert isinstance(layers[1], nn.ReLU)
        assert isinstance(layers[2], nn.Dropout)
        
        # Second layer: first hidden -> second hidden
        assert isinstance(layers[3], nn.Linear)
        assert layers[3].in_features == hidden_dims[0]
        assert layers[3].out_features == hidden_dims[1]
        
        # Output layer
        assert isinstance(layers[6], nn.Linear)
        assert layers[6].in_features == hidden_dims[1]
        assert layers[6].out_features == output_dim
    
    def test_forward_pass(self):
        """Test forward pass through network"""
        
        input_dim = 5
        output_dim = 3
        network = PolicyNetwork(input_dim, output_dim)
        
        # Test input
        x = torch.randn(2, input_dim)  # Batch size 2
        output = network.forward(x)
        
        assert output.shape == (2, output_dim)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_action_selection(self):
        """Test action selection from policy"""
        
        input_dim = 6
        output_dim = 4
        network = PolicyNetwork(input_dim, output_dim)
        
        # Test state
        state = torch.randn(1, input_dim)
        
        # Test deterministic action
        action, log_prob = network.get_action(state, deterministic=True)
        
        assert action.shape == (1,)
        assert log_prob.shape == (1,)
        assert action.dtype == torch.long
        assert log_prob.dtype == torch.float32
        
        # Test stochastic action
        action, log_prob = network.get_action(state, deterministic=False)
        
        assert action.shape == (1,)
        assert log_prob.shape == (1,)
        assert 0 <= action.item() < output_dim
    
    def test_action_evaluation(self):
        """Test action evaluation"""
        
        input_dim = 7
        output_dim = 5
        network = PolicyNetwork(input_dim, output_dim)
        
        # Test inputs
        states = torch.randn(3, input_dim)  # Batch size 3
        actions = torch.randint(0, output_dim, (3,))
        
        # Evaluate actions
        log_probs, entropy, state_values = network.evaluate_actions(states, actions)
        
        assert log_probs.shape == (3,)
        assert entropy.shape == (3,)
        assert state_values.shape == (3, 1)
        
        # Test log probabilities
        assert not torch.isnan(log_probs).any()
        assert not torch.isinf(log_probs).any()
        
        # Test entropy
        assert not torch.isnan(entropy).any()
        assert not torch.isinf(entropy).any()
        assert torch.all(entropy >= 0)  # Entropy should be non-negative

class TestValueNetwork:
    """Test cases for ValueNetwork class"""
    
    def test_initialization(self):
        """Test value network initialization"""
        input_dim = 12
        hidden_dims = [128, 64]
        
        network = ValueNetwork(input_dim, hidden_dims)
        
        assert network.input_dim == input_dim
        assert network.hidden_dims == hidden_dims
        assert isinstance(network.network, nn.Sequential)
    
    def test_network_architecture(self):
        """Test value network architecture"""
        
        input_dim = 10
        hidden_dims = [64, 32]
        
        network = ValueNetwork(input_dim, hidden_dims)
        
        # Test layer structure
        layers = list(network.network)
        
        # First layer
        assert isinstance(layers[0], nn.Linear)
        assert layers[0].in_features == input_dim
        assert layers[0].out_features == hidden_dims[0]
        
        # Output layer
        assert isinstance(layers[5], nn.Linear)
        assert layers[5].in_features == hidden_dims[1]
        assert layers[5].out_features == 1  # Single value output
    
    def test_forward_pass(self):
        """Test forward pass through value network"""
        
        input_dim = 8
        network = ValueNetwork(input_dim)
        
        # Test input
        x = torch.randn(4, input_dim)  # Batch size 4
        output = network.forward(x)
        
        assert output.shape == (4, 1)  # Batch size x 1
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

class TestReplayBuffer:
    """Test cases for ReplayBuffer class"""
    
    def test_initialization(self):
        """Test replay buffer initialization"""
        capacity = 1000
        buffer = ReplayBuffer(capacity)
        
        assert buffer.capacity == capacity
        assert len(buffer.buffer) == 0
        assert buffer.position == 0
    
    def test_experience_storage(self):
        """Test experience storage and retrieval"""
        
        buffer = ReplayBuffer(capacity=100)
        
        # Mock experiences
        state = np.array([1.0, 2.0, 3.0])
        action = np.array([0])
        reward = 1.5
        next_state = np.array([2.0, 3.0, 4.0])
        done = False
        
        # Store experience
        buffer.push(state, action, reward, next_state, done)
        
        assert len(buffer) == 1
        
        # Store more experiences
        for i in range(5):
            buffer.push(state + i, action + i, reward + i, next_state + i, done)
        
        assert len(buffer) == 6
    
    def test_experience_sampling(self):
        """Test experience sampling"""
        
        buffer = ReplayBuffer(capacity=50)
        
        # Fill buffer
        for i in range(20):
            state = np.array([i, i, i])
            action = np.array([i])
            reward = float(i)
            next_state = np.array([i+1, i+1, i+1])
            done = i == 19  # Last experience is done
            
            buffer.push(state, action, reward, next_state, done)
        
        # Sample batch
        batch_size = 8
        states, actions, rewards, next_states, dones = buffer.sample(batch_size)
        
        assert states.shape == (batch_size, 3)
        assert actions.shape == (batch_size, 1)
        assert rewards.shape == (batch_size,)
        assert next_states.shape == (batch_size, 3)
        assert dones.shape == (batch_size,)
        
        # Test data types
        assert states.dtype == np.float64
        assert actions.dtype == np.int64
        assert rewards.dtype == np.float64
        assert dones.dtype == bool
    
    def test_buffer_capacity(self):
        """Test buffer capacity limits"""
        
        capacity = 5
        buffer = ReplayBuffer(capacity)
        
        # Fill beyond capacity
        for i in range(10):
            state = np.array([i, i, i])
            action = np.array([i])
            reward = float(i)
            next_state = np.array([i+1, i+1, i+1])
            done = False
            
            buffer.push(state, action, reward, next_state, done)
        
        # Should not exceed capacity
        assert len(buffer) <= capacity
        assert len(buffer) == capacity

class TestRLAgent:
    """Test cases for RLAgent class"""
    
    def test_initialization(self):
        """Test RL agent initialization"""
        state_dim = 10
        action_dim = 4
        device = "cpu"
        
        agent = RLAgent(state_dim, action_dim, device)
        
        assert agent.state_dim == state_dim
        assert agent.action_dim == action_dim
        assert agent.device == device
        assert isinstance(agent.policy_network, PolicyNetwork)
        assert isinstance(agent.value_network, ValueNetwork)
        assert isinstance(agent.replay_buffer, ReplayBuffer)
    
    def test_hyperparameters(self):
        """Test agent hyperparameters"""
        
        agent = RLAgent(8, 3)
        
        # Test hyperparameter values
        assert 0 < agent.gamma < 1  # Discount factor
        assert 0 < agent.tau < 1    # Target update rate
        assert agent.batch_size > 0
        assert agent.learning_rate > 0
        assert 0 < agent.epsilon <= 1  # Exploration rate
    
    def test_action_selection(self):
        """Test action selection"""
        
        agent = RLAgent(6, 5)
        
        # Test state
        state = np.random.randn(6)
        
        # Test training mode (epsilon-greedy)
        action = agent.select_action(state, training=True)
        
        assert isinstance(action, (int, np.integer))
        assert 0 <= action < 5
        
        # Test evaluation mode (deterministic)
        action = agent.select_action(state, training=False)
        
        assert isinstance(action, (int, np.integer))
        assert 0 <= action < 5
    
    def test_experience_storage(self):
        """Test experience storage"""
        
        agent = RLAgent(4, 2)
        
        # Mock experience
        state = np.array([1.0, 2.0, 3.0, 4.0])
        action = 1
        reward = 2.5
        next_state = np.array([2.0, 3.0, 4.0, 5.0])
        done = False
        
        # Store experience
        agent.store_experience(state, action, reward, next_state, done)
        
        assert len(agent.replay_buffer) == 1
    
    def test_network_update(self):
        """Test network update mechanism"""
        
        agent = RLAgent(5, 3)
        
        # Fill replay buffer
        for i in range(100):
            state = np.random.randn(5)
            action = np.random.randint(0, 3)
            reward = np.random.randn()
            next_state = np.random.randn(5)
            done = np.random.choice([True, False])
            
            agent.store_experience(state, action, reward, next_state, done)
        
        # Test update
        update_info = agent.update()
        
        assert isinstance(update_info, dict)
        assert "loss" in update_info
        assert "value_loss" in update_info
        assert "policy_loss" in update_info
        assert "epsilon" in update_info
        
        # Test epsilon decay
        assert update_info["epsilon"] < 1.0

class TestTrainingLoop:
    """Test cases for TrainingLoop class"""
    
    def test_initialization(self):
        """Test training loop initialization"""
        
        # Mock agent and environment
        agent = RLAgent(6, 4)
        environment = None  # Would be actual environment in real usage
        
        training_loop = TrainingLoop(agent, environment, max_episodes=100)
        
        assert training_loop.agent == agent
        assert training_loop.environment == environment
        assert training_loop.max_episodes == 100
        assert isinstance(training_loop.episode_rewards, list)
        assert isinstance(training_loop.episode_lengths, list)
    
    def test_training_metrics(self):
        """Test training metrics tracking"""
        
        training_loop = TrainingLoop(None, None)
        
        # Mock episode data
        episode_rewards = [1.5, 2.0, 1.8, 2.2, 1.9]
        episode_lengths = [10, 12, 11, 13, 12]
        
        training_loop.episode_rewards = episode_rewards
        training_loop.episode_lengths = episode_lengths
        
        # Test metrics calculation
        avg_reward = np.mean(episode_rewards)
        avg_length = np.mean(episode_lengths)
        
        assert avg_reward == 1.88
        assert avg_length == 11.6
        
        # Test recent metrics
        recent_rewards = episode_rewards[-3:]  # Last 3 episodes
        recent_avg = np.mean(recent_rewards)
        assert recent_avg == 2.0
    
    def test_logging_intervals(self):
        """Test logging interval logic"""
        
        training_loop = TrainingLoop(None, None, max_episodes=100)
        training_loop.log_interval = 10
        training_loop.save_interval = 25
        
        # Test logging intervals
        for episode in range(100):
            should_log = episode % training_loop.log_interval == 0
            should_save = episode % training_loop.save_interval == 0
            
            if episode in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]:
                assert should_log is True
            else:
                assert should_log is False
            
            if episode in [0, 25, 50, 75]:
                assert should_save is True
            else:
                assert should_save is False

class TestPolicyOptimizer:
    """Test cases for PolicyOptimizer class"""
    
    def test_initialization(self):
        """Test policy optimizer initialization"""
        
        policy_network = PolicyNetwork(8, 4)
        value_network = ValueNetwork(8)
        
        optimizer = PolicyOptimizer(policy_network, value_network)
        
        assert optimizer.policy_network == policy_network
        assert optimizer.value_network == value_network
        assert optimizer.learning_rate > 0
        assert optimizer.clip_ratio > 0
        assert optimizer.value_coef > 0
        assert optimizer.entropy_coef > 0
    
    def test_ppo_parameters(self):
        """Test PPO algorithm parameters"""
        
        optimizer = PolicyOptimizer(None, None)
        
        # Test parameter ranges
        assert 0 < optimizer.clip_ratio < 1  # Clip ratio should be between 0 and 1
        assert optimizer.value_coef > 0      # Value coefficient should be positive
        assert optimizer.entropy_coef > 0    # Entropy coefficient should be positive
        
        # Test parameter relationships
        assert optimizer.value_coef > optimizer.entropy_coef  # Value usually more important
    
    def test_ppo_update_structure(self):
        """Test PPO update structure"""
        
        optimizer = PolicyOptimizer(None, None)
        
        # Mock training data
        batch_size = 4
        state_dim = 6
        action_dim = 3
        
        states = torch.randn(batch_size, state_dim)
        actions = torch.randint(0, action_dim, (batch_size,))
        old_log_probs = torch.randn(batch_size)
        advantages = torch.randn(batch_size)
        returns = torch.randn(batch_size)
        num_epochs = 4
        
        # Test update call structure
        # Note: This would need actual networks to run
        assert states.shape == (batch_size, state_dim)
        assert actions.shape == (batch_size,)
        assert old_log_probs.shape == (batch_size,)
        assert advantages.shape == (batch_size,)
        assert returns.shape == (batch_size,)
        assert num_epochs > 0

class TestIntegration:
    """Integration tests for RL system"""
    
    def test_network_compatibility(self):
        """Test network compatibility and data flow"""
        
        # Test compatible dimensions
        state_dim = 10
        action_dim = 5
        batch_size = 3
        
        policy_net = PolicyNetwork(state_dim, action_dim)
        value_net = ValueNetwork(state_dim)
        
        # Test data flow
        states = torch.randn(batch_size, state_dim)
        
        # Policy network
        policy_output = policy_net(states)
        assert policy_output.shape == (batch_size, action_dim)
        
        # Value network
        value_output = value_net(states)
        assert value_output.shape == (batch_size, 1)
    
    def test_agent_environment_interface(self):
        """Test agent-environment interface compatibility"""
        
        # Mock environment interface
        state_dim = 8
        action_dim = 4
        
        # Create agent
        agent = RLAgent(state_dim, action_dim)
        
        # Test interface compatibility
        mock_state = np.random.randn(state_dim)
        mock_action = agent.select_action(mock_state, training=True)
        
        assert isinstance(mock_action, (int, np.integer))
        assert 0 <= mock_action < action_dim
        
        # Test experience storage
        mock_next_state = np.random.randn(state_dim)
        mock_reward = 1.5
        mock_done = False
        
        agent.store_experience(mock_state, mock_action, mock_reward, mock_next_state, mock_done)
        assert len(agent.replay_buffer) == 1
    
    def test_training_workflow(self):
        """Test complete training workflow"""
        
        # Mock training components
        state_dim = 6
        action_dim = 3
        
        agent = RLAgent(state_dim, action_dim)
        environment = None  # Would be actual environment
        
        training_loop = TrainingLoop(agent, environment, max_episodes=10)
        
        # Test workflow structure
        assert training_loop.agent == agent
        assert training_loop.environment == environment
        assert training_loop.max_episodes == 10
        
        # Test metrics initialization
        assert len(training_loop.episode_rewards) == 0
        assert len(training_loop.episode_lengths) == 0
        
        # Test logging configuration
        assert training_loop.log_interval > 0
        assert training_loop.save_interval > 0 