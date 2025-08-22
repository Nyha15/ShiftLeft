# Original code from: https://github.com/PaulDanielML/MuJoCo_RL_UR5
# File: rl_agent.py, policy_network.py, training_loop.py

import mujoco
import numpy as np
import cv2
from typing import Tuple, Dict, Any, List, Optional
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal, Categorical
import random
from collections import deque

class PolicyNetwork(nn.Module):
    """Neural network policy for UR5 robot"""
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int] = None):
        super(PolicyNetwork, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [256, 256, 128]
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        
        # Build network layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights"""
        
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through network"""
        
        return self.network(x)
    
    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get action from policy network"""
        
        # Get action logits
        action_logits = self.forward(state)
        
        # Create action distribution
        action_dist = Categorical(logits=action_logits)
        
        if deterministic:
            action = action_dist.probs.argmax(dim=-1)
        else:
            action = action_dist.sample()
        
        # Get log probability
        log_prob = action_dist.log_prob(action)
        
        return action, log_prob
    
    def evaluate_actions(self, states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions and get log probs, entropy, and state values"""
        
        action_logits = self.forward(states)
        action_dist = Categorical(logits=action_logits)
        
        log_probs = action_dist.log_prob(actions)
        entropy = action_dist.entropy()
        
        # State values (simplified - would normally have separate value network)
        state_values = torch.zeros(states.size(0), 1)
        
        return log_probs, entropy, state_values

class ValueNetwork(nn.Module):
    """Value network for critic in actor-critic methods"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = None):
        super(ValueNetwork, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [256, 256, 128]
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        
        # Build network layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        # Output layer (single value)
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights"""
        
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through value network"""
        
        return self.network(x)

class ReplayBuffer:
    """Experience replay buffer for off-policy learning"""
    
    def __init__(self, capacity: int = 100000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.position = 0
    
    def push(self, state: np.ndarray, action: np.ndarray, reward: float, 
             next_state: np.ndarray, done: bool):
        """Add experience to buffer"""
        
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sample batch of experiences"""
        
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (np.array(states), np.array(actions), np.array(rewards), 
                np.array(next_states), np.array(dones))
    
    def __len__(self) -> int:
        return len(self.buffer)

class RLAgent:
    """Reinforcement learning agent for UR5 robot"""
    
    def __init__(self, state_dim: int, action_dim: int, device: str = "cpu"):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        
        # Networks
        self.policy_network = PolicyNetwork(state_dim, action_dim).to(device)
        self.value_network = ValueNetwork(state_dim).to(device)
        
        # Target networks for stable learning
        self.target_policy_network = PolicyNetwork(state_dim, action_dim).to(device)
        self.target_value_network = ValueNetwork(state_dim).to(device)
        
        # Copy weights to target networks
        self._update_target_networks()
        
        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy_network.parameters(), lr=3e-4)
        self.value_optimizer = optim.Adam(self.value_network.parameters(), lr=1e-3)
        
        # Hyperparameters
        self.gamma = 0.99  # Discount factor
        self.tau = 0.005   # Target network update rate
        self.batch_size = 64
        self.learning_rate = 3e-4
        
        # Experience replay
        self.replay_buffer = ReplayBuffer()
        
        # Training state
        self.training_step = 0
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
    
    def _update_target_networks(self):
        """Update target networks with current network weights"""
        
        for target_param, param in zip(self.target_policy_network.parameters(), 
                                     self.policy_network.parameters()):
            target_param.data.copy_(param.data)
        
        for target_param, param in zip(self.target_value_network.parameters(), 
                                     self.value_network.parameters()):
            target_param.data.copy_(param.data)
    
    def _soft_update_target_networks(self):
        """Soft update target networks"""
        
        for target_param, param in zip(self.target_policy_network.parameters(), 
                                     self.policy_network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for target_param, param in zip(self.target_value_network.parameters(), 
                                     self.value_network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def select_action(self, state: np.ndarray, training: bool = True) -> np.ndarray:
        """Select action using epsilon-greedy policy"""
        
        if training and random.random() < self.epsilon:
            # Random action
            action = np.random.randint(0, self.action_dim)
        else:
            # Policy action
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                action, _ = self.policy_network.get_action(state_tensor, deterministic=True)
            action = action.cpu().numpy()[0]
        
        return action
    
    def store_experience(self, state: np.ndarray, action: np.ndarray, reward: float,
                        next_state: np.ndarray, done: bool):
        """Store experience in replay buffer"""
        
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def update(self) -> Dict[str, float]:
        """Update agent using stored experiences"""
        
        if len(self.replay_buffer) < self.batch_size:
            return {"loss": 0.0, "value_loss": 0.0, "policy_loss": 0.0}
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        # Update value network
        current_values = self.value_network(states).squeeze()
        next_values = self.target_value_network(next_states).squeeze()
        target_values = rewards + (self.gamma * next_values * ~dones)
        
        value_loss = nn.MSELoss()(current_values, target_values.detach())
        
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()
        
        # Update policy network
        log_probs, entropy, _ = self.policy_network.evaluate_actions(states, actions)
        
        # Advantage estimation (simplified)
        advantages = (target_values - current_values).detach()
        
        policy_loss = -(log_probs * advantages).mean()
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        # Soft update target networks
        self._soft_update_target_networks()
        
        # Update training state
        self.training_step += 1
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return {
            "loss": value_loss.item() + policy_loss.item(),
            "value_loss": value_loss.item(),
            "policy_loss": policy_loss.item(),
            "epsilon": self.epsilon
        }
    
    def save_model(self, filepath: str):
        """Save trained model"""
        
        torch.save({
            "policy_network_state_dict": self.policy_network.state_dict(),
            "value_network_state_dict": self.value_network.state_dict(),
            "policy_optimizer_state_dict": self.policy_optimizer.state_dict(),
            "value_optimizer_state_dict": self.value_optimizer.state_dict(),
            "training_step": self.training_step,
            "epsilon": self.epsilon
        }, filepath)
    
    def load_model(self, filepath: str):
        """Load trained model"""
        
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.policy_network.load_state_dict(checkpoint["policy_network_state_dict"])
        self.value_network.load_state_dict(checkpoint["value_network_state_dict"])
        self.policy_optimizer.load_state_dict(checkpoint["policy_optimizer_state_dict"])
        self.value_optimizer.load_state_dict(checkpoint["value_optimizer_state_dict"])
        self.training_step = checkpoint["training_step"]
        self.epsilon = checkpoint["epsilon"]

class TrainingLoop:
    """Training loop for RL agent"""
    
    def __init__(self, agent: RLAgent, environment, max_episodes: int = 1000):
        self.agent = agent
        self.environment = environment
        self.max_episodes = max_episodes
        
        # Training metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.training_losses = []
        
        # Logging
        self.log_interval = 10
        self.save_interval = 100
    
    def train(self) -> Dict[str, List[float]]:
        """Main training loop"""
        
        print("Starting training...")
        
        for episode in range(self.max_episodes):
            # Reset environment
            state = self.environment.reset()
            
            episode_reward = 0.0
            episode_length = 0
            
            # Episode loop
            while True:
                # Select action
                action = self.agent.select_action(state, training=True)
                
                # Execute action
                next_state, reward, done, info = self.environment.step(action)
                
                # Store experience
                self.agent.store_experience(state, action, reward, next_state, done)
                
                # Update agent
                update_info = self.agent.update()
                
                # Update metrics
                episode_reward += reward
                episode_length += 1
                state = next_state
                
                if done:
                    break
            
            # Store episode metrics
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            
            # Logging
            if episode % self.log_interval == 0:
                avg_reward = np.mean(self.episode_rewards[-self.log_interval:])
                avg_length = np.mean(self.episode_lengths[-self.log_interval:])
                print(f"Episode {episode}: Avg Reward = {avg_reward:.2f}, Avg Length = {avg_length:.2f}")
            
            # Save model
            if episode % self.save_interval == 0:
                self.agent.save_model(f"ur5_rl_model_episode_{episode}.pth")
        
        # Final save
        self.agent.save_model("ur5_rl_model_final.pth")
        
        print("Training completed!")
        
        return {
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths,
            "training_losses": self.training_losses
        }
    
    def evaluate(self, num_episodes: int = 10) -> Dict[str, float]:
        """Evaluate trained agent"""
        
        print("Evaluating agent...")
        
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(num_episodes):
            state = self.environment.reset()
            episode_reward = 0.0
            episode_length = 0
            
            while True:
                # Select action (deterministic)
                action = self.agent.select_action(state, training=False)
                
                # Execute action
                next_state, reward, done, info = self.environment.step(action)
                
                episode_reward += reward
                episode_length += 1
                state = next_state
                
                if done:
                    break
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
        
        # Calculate metrics
        avg_reward = np.mean(episode_rewards)
        avg_length = np.mean(episode_lengths)
        std_reward = np.std(episode_rewards)
        
        print(f"Evaluation Results:")
        print(f"  Average Reward: {avg_reward:.2f} ± {std_reward:.2f}")
        print(f"  Average Length: {avg_length:.2f}")
        
        return {
            "average_reward": avg_reward,
            "average_length": avg_length,
            "std_reward": std_reward
        }

class PolicyOptimizer:
    """Policy optimization algorithms"""
    
    def __init__(self, policy_network: PolicyNetwork, value_network: ValueNetwork):
        self.policy_network = policy_network
        self.value_network = value_network
        
        # Optimization parameters
        self.learning_rate = 3e-4
        self.clip_ratio = 0.2
        self.value_coef = 0.5
        self.entropy_coef = 0.01
        
        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy_network.parameters(), lr=self.learning_rate)
        self.value_optimizer = optim.Adam(self.value_network.parameters(), lr=self.learning_rate)
    
    def ppo_update(self, states: torch.Tensor, actions: torch.Tensor, 
                   old_log_probs: torch.Tensor, advantages: torch.Tensor,
                   returns: torch.Tensor, num_epochs: int = 4) -> Dict[str, float]:
        """PPO policy update"""
        
        for epoch in range(num_epochs):
            # Get current policy outputs
            log_probs, entropy, values = self.policy_network.evaluate_actions(states, actions)
            
            # Calculate ratio
            ratio = torch.exp(log_probs - old_log_probs)
            
            # Calculate surrogate losses
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
            
            # Policy loss
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = nn.MSELoss()(values.squeeze(), returns)
            
            # Total loss
            total_loss = (policy_loss + 
                         self.value_coef * value_loss - 
                         self.entropy_coef * entropy.mean())
            
            # Update networks
            self.policy_optimizer.zero_grad()
            self.value_optimizer.zero_grad()
            
            total_loss.backward()
            
            self.policy_optimizer.step()
            self.value_optimizer.step()
        
        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.mean().item(),
            "total_loss": total_loss.item()
        }

if __name__ == "__main__":
    # Example usage
    print("Reinforcement Learning and Policy Optimization for UR5 Robot")
    print("This module provides:")
    print("- Policy networks (actor-critic)")
    print("- Value networks")
    print("- Experience replay buffers")
    print("- RL agents with various algorithms")
    print("- Training loops and evaluation")
    print("- Policy optimization (PPO)")
    
    # Note: This code requires PyTorch and a proper MuJoCo environment
    # to run the actual functionality 