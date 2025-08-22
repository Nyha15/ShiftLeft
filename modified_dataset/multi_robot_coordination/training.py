# Original code from: https://github.com/volunt4s/Simple-MuJoCo-PickNPlace
# File: multi_robot_env.py, coordination_controller.py, task_allocator.py

import mujoco
import numpy as np
import cv2
from typing import Tuple, Dict, Any, List, Optional
import math
from collections import deque
import threading
import time

class MultiRobotEnvironment:
    """Multi-robot coordination environment"""
    
    def __init__(self, xml_path: str, num_robots: int = 2):
        self.xml_path = xml_path
        self.num_robots = num_robots
        
        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        # Robot configurations
        self.robots = []
        self.robot_names = [f"robot_{i}" for i in range(num_robots)]
        
        # Environment parameters
        self.max_steps = 2000
        self.step_count = 0
        self.task_completion_threshold = 0.02
        
        # Task management
        self.tasks = []
        self.completed_tasks = []
        self.task_queue = deque()
        
        # Coordination parameters
        self.communication_range = 2.0
        self.collision_safety_margin = 0.1
        
        # Initialize robots
        self._initialize_robots()
        
    def _initialize_robots(self):
        """Initialize robot instances"""
        
        for i in range(self.num_robots):
            robot = MultiRobot(
                robot_id=i,
                robot_name=self.robot_names[i],
                model=self.model,
                data=self.data
            )
            self.robots.append(robot)
    
    def reset(self) -> Dict[str, np.ndarray]:
        """Reset environment to initial state"""
        self.step_count = 0
        self.completed_tasks = []
        self.task_queue.clear()
        
        # Reset robots
        for robot in self.robots:
            robot.reset()
        
        # Generate new tasks
        self._generate_tasks()
        
        # Get initial observation
        observation = self._get_observation()
        
        return observation
    
    def step(self, actions: List[np.ndarray]) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, Any]]:
        """Execute actions for all robots and return next state"""
        
        assert len(actions) == self.num_robots, f"Expected {self.num_robots} actions, got {len(actions)}"
        
        reward = 0.0
        done = False
        
        # Execute actions for each robot
        for i, (robot, action) in enumerate(zip(self.robots, actions)):
            robot.execute_action(action)
        
        # Update environment
        mujoco.mj_step(self.model, self.data)
        
        # Update step count
        self.step_count += 1
        
        # Check task completion
        self._update_task_completion()
        
        # Check if episode is done
        if self.step_count >= self.max_steps or len(self.completed_tasks) >= len(self.tasks):
            done = True
        
        # Get observation
        observation = self._get_observation()
        
        # Calculate reward
        if not done:
            reward = self._calculate_reward()
        
        # Update info
        info = {
            "step_count": self.step_count,
            "tasks_completed": len(self.completed_tasks),
            "total_tasks": len(self.tasks),
            "robot_positions": [robot.get_position() for robot in self.robots],
            "collision_detected": self._check_collisions()
        }
        
        return observation, reward, done, info
    
    def _generate_tasks(self):
        """Generate tasks for robots to complete"""
        
        self.tasks = []
        
        # Generate pick and place tasks
        num_tasks = self.num_robots * 2  # 2 tasks per robot
        
        for i in range(num_tasks):
            task = {
                "id": i,
                "type": "pick_and_place",
                "pick_position": np.random.uniform([0.2, -0.5, 0.1], [0.8, 0.5, 0.1]),
                "place_position": np.random.uniform([0.2, -0.5, 0.1], [0.8, 0.5, 0.1]),
                "assigned_robot": None,
                "completed": False,
                "priority": np.random.randint(1, 6)  # 1-5 priority
            }
            self.tasks.append(task)
        
        # Add to task queue
        self.task_queue.extend(self.tasks)
    
    def _update_task_completion(self):
        """Update task completion status"""
        
        for task in self.tasks:
            if task["completed"]:
                continue
            
            # Check if any robot completed this task
            for robot in self.robots:
                if robot.current_task_id == task["id"]:
                    robot_pos = robot.get_position()
                    
                    # Check if robot is at place position
                    if np.linalg.norm(robot_pos - task["place_position"]) < self.task_completion_threshold:
                        task["completed"] = True
                        self.completed_tasks.append(task)
                        
                        # Remove from task queue
                        if task in self.task_queue:
                            self.task_queue.remove(task)
                        
                        break
    
    def _calculate_reward(self) -> float:
        """Calculate reward based on current state"""
        
        reward = 0.0
        
        # Reward for completed tasks
        reward += len(self.completed_tasks) * 10.0
        
        # Reward for robots being close to task targets
        for robot in self.robots:
            if robot.current_task_id is not None:
                task = self.tasks[robot.current_task_id]
                robot_pos = robot.get_position()
                
                if not task["completed"]:
                    # Reward for being close to pick or place position
                    if robot.task_phase == "pick":
                        target_pos = task["pick_position"]
                    else:
                        target_pos = task["place_position"]
                    
                    distance = np.linalg.norm(robot_pos - target_pos)
                    if distance < 0.1:
                        reward += 2.0
                    elif distance < 0.2:
                        reward += 1.0
        
        # Penalty for collisions
        if self._check_collisions():
            reward -= 5.0
        
        # Penalty for large joint velocities
        for robot in self.robots:
            joint_velocities = robot.get_joint_velocities()
            velocity_penalty = -0.1 * np.sum(joint_velocities ** 2)
            reward += velocity_penalty
        
        return reward
    
    def _check_collisions(self) -> bool:
        """Check for collisions between robots"""
        
        for i in range(self.num_robots):
            for j in range(i + 1, self.num_robots):
                robot1_pos = self.robots[i].get_position()
                robot2_pos = self.robots[j].get_position()
                
                distance = np.linalg.norm(robot1_pos - robot2_pos)
                if distance < self.collision_safety_margin:
                    return True
        
        return False
    
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Get current observation for all robots"""
        
        observation = {}
        
        # Robot observations
        for i, robot in enumerate(self.robots):
            robot_obs = robot.get_observation()
            observation[f"robot_{i}"] = robot_obs
        
        # Environment observation
        observation["environment"] = {
            "tasks": self.tasks,
            "completed_tasks": self.completed_tasks,
            "task_queue_length": len(self.task_queue),
            "step_count": self.step_count
        }
        
        return observation
    
    def get_action_space(self) -> Dict[str, np.ndarray]:
        """Get action space specification for all robots"""
        action_spaces = {}
        
        for i, robot in enumerate(self.robots):
            action_spaces[f"robot_{i}"] = robot.get_action_space()
        
        return action_spaces
    
    def get_observation_space(self) -> Dict[str, np.ndarray]:
        """Get observation space specification for all robots"""
        observation_spaces = {}
        
        for i, robot in enumerate(self.robots):
            observation_spaces[f"robot_{i}"] = robot.get_observation_space()
        
        return observation_spaces

class MultiRobot:
    """Individual robot in multi-robot system"""
    
    def __init__(self, robot_id: int, robot_name: str, model: mujoco.MjModel, data: mujoco.MjData):
        self.robot_id = robot_id
        self.robot_name = robot_name
        self.model = model
        self.data = data
        
        # Robot state
        self.current_task_id = None
        self.task_phase = "idle"  # idle, pick, place
        self.is_carrying_object = False
        
        # Control parameters
        self.position_gain = 50.0
        self.velocity_gain = 20.0
        self.max_velocity = 2.0
        
        # Robot parameters
        self.num_joints = 6
        self.end_effector_name = f"{robot_name}_end_effector"
        
        # Task tracking
        self.task_start_time = None
        self.task_timeout = 30.0  # seconds
        
    def reset(self):
        """Reset robot to initial state"""
        self.current_task_id = None
        self.task_phase = "idle"
        self.is_carrying_object = False
        self.task_start_time = None
        
        # Reset to home position
        home_config = np.array([0, -np.pi/2, 0, -np.pi/2, 0, 0])
        self._set_joint_positions(home_config)
    
    def execute_action(self, action: np.ndarray):
        """Execute action for robot"""
        
        # Parse action
        target_position = action[:3]
        gripper_action = action[3]
        
        # Move to target position
        self._move_to_position(target_position)
        
        # Control gripper
        self._control_gripper(gripper_action)
        
        # Update task state
        self._update_task_state()
    
    def _move_to_position(self, target_position: np.ndarray):
        """Move robot to target position"""
        
        current_position = self.get_position()
        position_error = target_position - current_position
        
        # Simple proportional control
        velocity = self.position_gain * position_error
        
        # Limit velocity
        velocity_magnitude = np.linalg.norm(velocity)
        if velocity_magnitude > self.max_velocity:
            velocity = velocity * (self.max_velocity / velocity_magnitude)
        
        # Update joint velocities
        self._set_joint_velocities(velocity)
    
    def _control_gripper(self, gripper_action: float):
        """Control robot gripper"""
        
        if gripper_action > 0.5:  # Close gripper
            gripper_pos = 0.0
        else:  # Open gripper
            gripper_pos = 0.04
        
        # Update gripper position
        gripper_joint_id = self.num_joints + self.robot_id  # Assuming gripper joints come after arm joints
        if gripper_joint_id < len(self.data.qpos):
            self.data.qpos[gripper_joint_id] = gripper_pos
    
    def _update_task_state(self):
        """Update robot task state"""
        
        if self.current_task_id is None:
            return
        
        # Check task timeout
        if self.task_start_time is not None:
            if time.time() - self.task_start_time > self.task_timeout:
                # Task timeout, reset
                self.current_task_id = None
                self.task_phase = "idle"
                self.is_carrying_object = False
                return
        
        # Update task phase based on current state
        if self.task_phase == "pick":
            if not self.is_carrying_object:
                # Check if at pick position
                current_pos = self.get_position()
                # This would need access to task information
                pass
        elif self.task_phase == "place":
            if self.is_carrying_object:
                # Check if at place position
                current_pos = self.get_position()
                # This would need access to task information
                pass
    
    def get_position(self) -> np.ndarray:
        """Get current robot position"""
        try:
            body_id = self.model.body(self.end_effector_name).id
            return self.data.xpos[body_id].copy()
        except:
            # Fallback to base position
            return np.array([0.0, 0.0, 0.0])
    
    def get_joint_positions(self) -> np.ndarray:
        """Get current joint positions"""
        start_idx = self.robot_id * self.num_joints
        end_idx = start_idx + self.num_joints
        return self.data.qpos[start_idx:end_idx].copy()
    
    def get_joint_velocities(self) -> np.ndarray:
        """Get current joint velocities"""
        start_idx = self.robot_id * self.num_joints
        end_idx = start_idx + self.num_joints
        return self.data.qvel[start_idx:end_idx].copy()
    
    def _set_joint_positions(self, positions: np.ndarray):
        """Set joint positions"""
        start_idx = self.robot_id * self.num_joints
        end_idx = start_idx + self.num_joints
        self.data.qpos[start_idx:end_idx] = positions
    
    def _set_joint_velocities(self, velocities: np.ndarray):
        """Set joint velocities"""
        start_idx = self.robot_id * self.num_joints
        end_idx = start_idx + self.num_joints
        self.data.qvel[start_idx:end_idx] = velocities
    
    def get_observation(self) -> Dict[str, np.ndarray]:
        """Get robot observation"""
        
        observation = {
            "position": self.get_position(),
            "joint_positions": self.get_joint_positions(),
            "joint_velocities": self.get_joint_velocities(),
            "current_task_id": self.current_task_id,
            "task_phase": self.task_phase,
            "is_carrying_object": self.is_carrying_object
        }
        
        return observation
    
    def get_action_space(self) -> Dict[str, np.ndarray]:
        """Get action space specification"""
        return {
            "shape": (4,),
            "low": np.array([-1.0, -1.0, -1.0, 0.0]),
            "high": np.array([1.0, 1.0, 1.0, 1.0])
        }
    
    def get_observation_space(self) -> Dict[str, np.ndarray]:
        """Get observation space specification"""
        return {
            "position": {"shape": (3,)},
            "joint_positions": {"shape": (self.num_joints,)},
            "joint_velocities": {"shape": (self.num_joints,)},
            "current_task_id": {"shape": ()},
            "task_phase": {"shape": ()},
            "is_carrying_object": {"shape": ()}
        }

class CoordinationController:
    """Controller for coordinating multiple robots"""
    
    def __init__(self, environment: MultiRobotEnvironment):
        self.environment = environment
        self.robots = environment.robots
        
        # Coordination parameters
        self.task_allocation_strategy = "greedy"  # greedy, auction, centralized
        self.collision_avoidance_enabled = True
        self.communication_enabled = True
        
        # Task allocation history
        self.task_allocation_history = []
        
    def allocate_tasks(self) -> Dict[int, int]:
        """Allocate tasks to robots"""
        
        if self.task_allocation_strategy == "greedy":
            return self._greedy_task_allocation()
        elif self.task_allocation_strategy == "auction":
            return self._auction_task_allocation()
        elif self.task_allocation_strategy == "centralized":
            return self._centralized_task_allocation()
        else:
            return self._greedy_task_allocation()
    
    def _greedy_task_allocation(self) -> Dict[int, int]:
        """Greedy task allocation strategy"""
        
        allocation = {}
        available_tasks = [task for task in self.environment.tasks if not task["completed"]]
        
        for robot in self.robots:
            if robot.current_task_id is None and available_tasks:
                # Find closest task
                robot_pos = robot.get_position()
                closest_task = None
                min_distance = float('inf')
                
                for task in available_tasks:
                    distance = np.linalg.norm(robot_pos - task["pick_position"])
                    if distance < min_distance:
                        min_distance = distance
                        closest_task = task
                
                if closest_task:
                    allocation[robot.robot_id] = closest_task["id"]
                    robot.current_task_id = closest_task["id"]
                    robot.task_phase = "pick"
                    robot.task_start_time = time.time()
                    available_tasks.remove(closest_task)
        
        return allocation
    
    def _auction_task_allocation(self) -> Dict[int, int]:
        """Auction-based task allocation strategy"""
        
        allocation = {}
        available_tasks = [task for task in self.environment.tasks if not task["completed"]]
        
        for task in available_tasks:
            # Each robot bids on the task
            bids = {}
            for robot in self.robots:
                if robot.current_task_id is None:
                    # Calculate bid based on distance and priority
                    robot_pos = robot.get_position()
                    distance = np.linalg.norm(robot_pos - task["pick_position"])
                    priority_bonus = task["priority"] * 0.1
                    bid = 1.0 / (distance + 0.1) + priority_bonus
                    bids[robot.robot_id] = bid
            
            # Assign task to highest bidder
            if bids:
                winner_robot_id = max(bids, key=bids.get)
                allocation[winner_robot_id] = task["id"]
                
                # Update robot state
                winner_robot = next(r for r in self.robots if r.robot_id == winner_robot_id)
                winner_robot.current_task_id = task["id"]
                winner_robot.task_phase = "pick"
                winner_robot.task_start_time = time.time()
        
        return allocation
    
    def _centralized_task_allocation(self) -> Dict[int, int]:
        """Centralized task allocation strategy"""
        
        allocation = {}
        available_tasks = [task for task in self.environment.tasks if not task["completed"]]
        available_robots = [r for r in self.robots if r.current_task_id is None]
        
        if not available_tasks or not available_robots:
            return allocation
        
        # Create cost matrix
        cost_matrix = np.zeros((len(available_robots), len(available_tasks)))
        
        for i, robot in enumerate(available_robots):
            for j, task in enumerate(available_tasks):
                robot_pos = robot.get_position()
                distance = np.linalg.norm(robot_pos - task["pick_position"])
                priority_cost = 1.0 / task["priority"]  # Higher priority = lower cost
                cost_matrix[i, j] = distance + priority_cost
        
        # Simple assignment: assign each robot to closest available task
        assigned_tasks = set()
        
        for i, robot in enumerate(available_robots):
            if len(assigned_tasks) >= len(available_tasks):
                break
            
            # Find best available task
            best_task_idx = None
            min_cost = float('inf')
            
            for j, task in enumerate(available_tasks):
                if j not in assigned_tasks and cost_matrix[i, j] < min_cost:
                    min_cost = cost_matrix[i, j]
                    best_task_idx = j
            
            if best_task_idx is not None:
                task = available_tasks[best_task_idx]
                allocation[robot.robot_id] = task["id"]
                robot.current_task_id = task["id"]
                robot.task_phase = "pick"
                robot.task_start_time = time.time()
                assigned_tasks.add(best_task_idx)
        
        return allocation
    
    def avoid_collisions(self):
        """Implement collision avoidance between robots"""
        
        if not self.collision_avoidance_enabled:
            return
        
        for i in range(len(self.robots)):
            for j in range(i + 1, len(self.robots)):
                robot1 = self.robots[i]
                robot2 = self.robots[j]
                
                robot1_pos = robot1.get_position()
                robot2_pos = robot2.get_position()
                
                distance = np.linalg.norm(robot1_pos - robot2_pos)
                
                if distance < self.environment.collision_safety_margin:
                    # Calculate avoidance force
                    avoidance_direction = robot1_pos - robot2_pos
                    if np.linalg.norm(avoidance_direction) > 0:
                        avoidance_direction = avoidance_direction / np.linalg.norm(avoidance_direction)
                    
                    # Apply avoidance force
                    avoidance_force = 0.1 * avoidance_direction
                    
                    # Move robots apart
                    robot1._move_to_position(robot1_pos + avoidance_force)
                    robot2._move_to_position(robot2_pos - avoidance_force)
    
    def get_coordination_status(self) -> Dict[str, Any]:
        """Get current coordination status"""
        
        status = {
            "num_robots": len(self.robots),
            "num_tasks": len(self.environment.tasks),
            "completed_tasks": len(self.environment.completed_tasks),
            "robot_status": {},
            "task_allocation": {},
            "collision_detected": self.environment._check_collisions()
        }
        
        # Robot status
        for robot in self.robots:
            status["robot_status"][robot.robot_id] = {
                "task_id": robot.current_task_id,
                "phase": robot.task_phase,
                "carrying_object": robot.is_carrying_object,
                "position": robot.get_position().tolist()
            }
        
        # Task allocation
        for robot in self.robots:
            if robot.current_task_id is not None:
                status["task_allocation"][robot.robot_id] = robot.current_task_id
        
        return status

if __name__ == "__main__":
    # Example usage
    print("Multi-Robot Coordination Environment")
    print("This module provides:")
    print("- Multi-robot task coordination")
    print("- Task allocation strategies")
    print("- Collision avoidance")
    print("- Communication protocols")
    print("- Performance optimization")
    
    # Note: This code requires a proper MuJoCo model with multiple robots
    # to run the actual functionality 