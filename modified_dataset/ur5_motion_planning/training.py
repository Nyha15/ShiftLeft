# Original code from: https://github.com/PaulDanielML/MuJoCo_RL_UR5
# File: motion_planner.py, trajectory_optimizer.py, path_finder.py

import mujoco
import numpy as np
import cv2
from typing import Tuple, Dict, Any, List, Optional
import math
from scipy.spatial import KDTree
from scipy.optimize import minimize

class MotionPlanner:
    """Motion planning module for UR5 robot"""
    
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model = model
        self.data = data
        
        # Planning parameters
        self.max_iterations = 1000
        self.planning_timeout = 5.0  # seconds
        self.resolution = 0.01  # meters
        self.max_velocity = 2.0  # rad/s
        self.max_acceleration = 5.0  # rad/s²
        
        # Robot parameters
        self.num_joints = 6
        self.joint_limits = self._get_joint_limits()
        
        # Obstacle avoidance parameters
        self.safety_margin = 0.05  # meters
        self.collision_threshold = 0.02
        
    def _get_joint_limits(self) -> Dict[str, np.ndarray]:
        """Get joint limits from model"""
        
        joint_limits = {}
        
        for i in range(self.num_joints):
            lower_limit = self.model.jnt_range[i, 0]
            upper_limit = self.model.jnt_range[i, 1]
            joint_limits[f"joint_{i}"] = {
                "lower": lower_limit,
                "upper": upper_limit,
                "range": upper_limit - lower_limit
            }
        
        return joint_limits
    
    def plan_joint_trajectory(self, start_config: np.ndarray, goal_config: np.ndarray,
                             obstacles: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Plan trajectory between joint configurations"""
        
        # Validate inputs
        assert len(start_config) == self.num_joints
        assert len(goal_config) == self.num_joints
        
        # Check joint limits
        if not self._is_configuration_valid(start_config) or not self._is_configuration_valid(goal_config):
            raise ValueError("Invalid joint configuration")
        
        # Initialize planning
        current_config = start_config.copy()
        trajectory = [current_config.copy()]
        velocities = [np.zeros(self.num_joints)]
        accelerations = [np.zeros(self.num_joints)]
        
        # RRT-style planning
        for iteration in range(self.max_iterations):
            # Generate random configuration
            if np.random.random() < 0.1:  # 10% chance to go directly to goal
                target_config = goal_config
            else:
                target_config = self._generate_random_configuration()
            
            # Find nearest neighbor
            nearest_config = self._find_nearest_configuration(current_config, trajectory)
            
            # Extend towards target
            new_config = self._extend_configuration(nearest_config, target_config)
            
            # Check collision
            if not self._check_collision(new_config, obstacles):
                # Add to trajectory
                trajectory.append(new_config)
                
                # Calculate velocity and acceleration
                if len(trajectory) > 1:
                    vel = (new_config - trajectory[-2]) / self.resolution
                    velocities.append(vel)
                    
                    if len(velocities) > 1:
                        acc = (vel - velocities[-2]) / self.resolution
                        accelerations.append(acc)
                    else:
                        accelerations.append(np.zeros(self.num_joints))
                else:
                    velocities.append(np.zeros(self.num_joints))
                    accelerations.append(np.zeros(self.num_joints))
                
                current_config = new_config
                
                # Check if goal reached
                if np.linalg.norm(current_config - goal_config) < 0.01:
                    break
        
        # Optimize trajectory
        optimized_trajectory = self._optimize_trajectory(trajectory, velocities, accelerations)
        
        return {
            "trajectory": optimized_trajectory["trajectory"],
            "velocities": optimized_trajectory["velocities"],
            "accelerations": optimized_trajectory["accelerations"],
            "success": len(trajectory) > 1,
            "iterations": iteration + 1
        }
    
    def _is_configuration_valid(self, config: np.ndarray) -> bool:
        """Check if joint configuration is within limits"""
        
        for i in range(self.num_joints):
            joint_name = f"joint_{i}"
            lower = self.joint_limits[joint_name]["lower"]
            upper = self.joint_limits[joint_name]["upper"]
            
            if config[i] < lower or config[i] > upper:
                return False
        
        return True
    
    def _generate_random_configuration(self) -> np.ndarray:
        """Generate random joint configuration within limits"""
        
        config = np.zeros(self.num_joints)
        
        for i in range(self.num_joints):
            joint_name = f"joint_{i}"
            lower = self.joint_limits[joint_name]["lower"]
            upper = self.joint_limits[joint_name]["upper"]
            
            config[i] = np.random.uniform(lower, upper)
        
        return config
    
    def _find_nearest_configuration(self, target: np.ndarray, 
                                  configurations: List[np.ndarray]) -> np.ndarray:
        """Find nearest configuration in list"""
        
        if not configurations:
            return target
        
        distances = [np.linalg.norm(target - config) for config in configurations]
        nearest_idx = np.argmin(distances)
        
        return configurations[nearest_idx]
    
    def _extend_configuration(self, from_config: np.ndarray, 
                            to_config: np.ndarray) -> np.ndarray:
        """Extend configuration towards target"""
        
        direction = to_config - from_config
        distance = np.linalg.norm(direction)
        
        if distance < self.resolution:
            return to_config
        
        # Normalize and scale
        direction = direction / distance * self.resolution
        new_config = from_config + direction
        
        return new_config
    
    def _check_collision(self, config: np.ndarray, 
                        obstacles: List[Dict[str, Any]] = None) -> bool:
        """Check if configuration causes collision"""
        
        if obstacles is None:
            return False
        
        # Set robot configuration
        self.data.qpos[:self.num_joints] = config
        mujoco.mj_forward(self.model, self.data)
        
        # Check collision with obstacles
        for obstacle in obstacles:
            if self._check_obstacle_collision(obstacle):
                return True
        
        return False
    
    def _check_obstacle_collision(self, obstacle: Dict[str, Any]) -> bool:
        """Check collision with specific obstacle"""
        
        # Get robot body positions
        robot_bodies = ["shoulder_link", "upper_arm_link", "forearm_link", 
                       "wrist_1_link", "wrist_2_link", "wrist_3_link"]
        
        for body_name in robot_bodies:
            try:
                body_id = self.model.body(body_name).id
                body_pos = self.data.xpos[body_id]
                
                # Check distance to obstacle
                obstacle_pos = obstacle["position"]
                distance = np.linalg.norm(body_pos - obstacle_pos)
                
                if distance < (obstacle["radius"] + self.safety_margin):
                    return True
            except:
                continue
        
        return False
    
    def _optimize_trajectory(self, trajectory: List[np.ndarray], 
                           velocities: List[np.ndarray], 
                           accelerations: List[np.ndarray]) -> Dict[str, Any]:
        """Optimize trajectory for smoothness and efficiency"""
        
        if len(trajectory) < 3:
            return {
                "trajectory": trajectory,
                "velocities": velocities,
                "accelerations": accelerations
            }
        
        # Convert to numpy arrays
        traj_array = np.array(trajectory)
        vel_array = np.array(velocities)
        acc_array = np.array(accelerations)
        
        # Apply velocity and acceleration limits
        vel_array = np.clip(vel_array, -self.max_velocity, self.max_velocity)
        acc_array = np.clip(acc_array, -self.max_acceleration, self.max_acceleration)
        
        # Smooth trajectory using moving average
        window_size = 3
        smoothed_trajectory = []
        
        for i in range(len(traj_array)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(traj_array), i + window_size // 2 + 1)
            
            window = traj_array[start_idx:end_idx]
            smoothed_point = np.mean(window, axis=0)
            smoothed_trajectory.append(smoothed_point)
        
        # Recalculate velocities and accelerations
        smoothed_velocities = []
        smoothed_accelerations = []
        
        for i in range(len(smoothed_trajectory)):
            if i == 0:
                vel = np.zeros(self.num_joints)
                acc = np.zeros(self.num_joints)
            elif i == 1:
                vel = (smoothed_trajectory[i] - smoothed_trajectory[i-1]) / self.resolution
                acc = np.zeros(self.num_joints)
            else:
                vel = (smoothed_trajectory[i] - smoothed_trajectory[i-1]) / self.resolution
                prev_vel = (smoothed_trajectory[i-1] - smoothed_trajectory[i-2]) / self.resolution
                acc = (vel - prev_vel) / self.resolution
            
            smoothed_velocities.append(vel)
            smoothed_accelerations.append(acc)
        
        return {
            "trajectory": smoothed_trajectory,
            "velocities": smoothed_velocities,
            "accelerations": smoothed_accelerations
        }

class TrajectoryOptimizer:
    """Trajectory optimization for smooth and efficient motion"""
    
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model = model
        self.data = data
        
        # Optimization parameters
        self.smoothness_weight = 1.0
        self.efficiency_weight = 0.5
        self.collision_weight = 10.0
        
    def optimize_trajectory(self, trajectory: List[np.ndarray], 
                          constraints: Dict[str, Any] = None) -> Dict[str, Any]:
        """Optimize trajectory using numerical optimization"""
        
        if len(trajectory) < 3:
            return {"trajectory": trajectory, "success": False}
        
        # Convert trajectory to flat array for optimization
        traj_array = np.array(trajectory)
        initial_params = traj_array.flatten()
        
        # Define objective function
        def objective(params):
            return self._trajectory_cost(params, constraints)
        
        # Define constraints
        optimization_constraints = []
        
        if constraints and "joint_limits" in constraints:
            # Add joint limit constraints
            for i in range(len(trajectory)):
                for j in range(traj_array.shape[1]):
                    param_idx = i * traj_array.shape[1] + j
                    lower_bound = constraints["joint_limits"][j]["lower"]
                    upper_bound = constraints["joint_limits"][j]["upper"]
                    
                    # Lower bound constraint
                    optimization_constraints.append({
                        'type': 'ineq',
                        'fun': lambda x, idx=param_idx, lb=lower_bound: x[idx] - lb
                    })
                    
                    # Upper bound constraint
                    optimization_constraints.append({
                        'type': 'ineq',
                        'fun': lambda x, idx=param_idx, ub=upper_bound: ub - x[idx]
                    })
        
        # Run optimization
        try:
            result = minimize(
                objective,
                initial_params,
                method='SLSQP',
                constraints=optimization_constraints,
                options={'maxiter': 1000}
            )
            
            if result.success:
                # Reshape optimized parameters back to trajectory
                optimized_trajectory = result.x.reshape(traj_array.shape)
                
                return {
                    "trajectory": optimized_trajectory.tolist(),
                    "success": True,
                    "cost": result.fun,
                    "iterations": result.nit
                }
            else:
                return {
                    "trajectory": trajectory,
                    "success": False,
                    "error": "Optimization failed to converge"
                }
                
        except Exception as e:
            return {
                "trajectory": trajectory,
                "success": False,
                "error": str(e)
            }
    
    def _trajectory_cost(self, params: np.ndarray, 
                        constraints: Dict[str, Any] = None) -> float:
        """Calculate trajectory cost"""
        
        # Reshape parameters to trajectory
        num_points = len(params) // 6  # Assuming 6 joints
        trajectory = params.reshape(num_points, 6)
        
        # Smoothness cost (minimize second derivative)
        smoothness_cost = 0.0
        for i in range(1, len(trajectory) - 1):
            second_deriv = trajectory[i+1] - 2*trajectory[i] + trajectory[i-1]
            smoothness_cost += np.sum(second_deriv**2)
        
        # Efficiency cost (minimize path length)
        efficiency_cost = 0.0
        for i in range(1, len(trajectory)):
            path_segment = trajectory[i] - trajectory[i-1]
            efficiency_cost += np.linalg.norm(path_segment)
        
        # Collision cost
        collision_cost = 0.0
        if constraints and "obstacles" in constraints:
            for i, config in enumerate(trajectory):
                for obstacle in constraints["obstacles"]:
                    # Simplified collision check
                    distance = np.linalg.norm(config[:3] - obstacle["position"])
                    if distance < obstacle["radius"]:
                        collision_cost += 100.0  # High penalty for collision
        
        # Total cost
        total_cost = (self.smoothness_weight * smoothness_cost + 
                     self.efficiency_weight * efficiency_cost + 
                     self.collision_weight * collision_cost)
        
        return total_cost

class PathFinder:
    """Path finding algorithms for robot navigation"""
    
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model = model
        self.data = data
        
        # Path finding parameters
        self.grid_resolution = 0.05  # meters
        self.max_iterations = 5000
        
    def find_path_a_star(self, start_pos: np.ndarray, goal_pos: np.ndarray,
                         obstacles: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Find path using A* algorithm"""
        
        # Create occupancy grid
        grid, grid_origin = self._create_occupancy_grid(obstacles)
        
        # Convert world coordinates to grid coordinates
        start_grid = self._world_to_grid(start_pos, grid_origin)
        goal_grid = self._world_to_grid(goal_pos, grid_origin)
        
        # A* search
        path = self._a_star_search(grid, start_grid, goal_grid)
        
        if path:
            # Convert grid coordinates back to world coordinates
            world_path = [self._grid_to_world(grid_pos, grid_origin) for grid_pos in path]
            
            return {
                "path": world_path,
                "success": True,
                "path_length": self._calculate_path_length(world_path)
            }
        else:
            return {
                "path": [],
                "success": False,
                "path_length": 0.0
            }
    
    def _create_occupancy_grid(self, obstacles: List[Dict[str, Any]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Create occupancy grid from obstacles"""
        
        # Define grid bounds
        x_min, x_max = -1.0, 1.0
        y_min, y_max = -1.0, 1.0
        z_min, z_max = 0.0, 1.0
        
        # Calculate grid dimensions
        nx = int((x_max - x_min) / self.grid_resolution)
        ny = int((y_max - y_min) / self.grid_resolution)
        nz = int((z_max - z_min) / self.grid_resolution)
        
        # Create empty grid
        grid = np.zeros((nx, ny, nz), dtype=bool)
        grid_origin = np.array([x_min, y_min, z_min])
        
        # Add obstacles to grid
        if obstacles:
            for obstacle in obstacles:
                self._add_obstacle_to_grid(grid, obstacle, grid_origin)
        
        return grid, grid_origin
    
    def _add_obstacle_to_grid(self, grid: np.ndarray, obstacle: Dict[str, Any], 
                             grid_origin: np.ndarray):
        """Add obstacle to occupancy grid"""
        
        obstacle_pos = obstacle["position"]
        obstacle_radius = obstacle["radius"]
        
        # Convert to grid coordinates
        grid_pos = self._world_to_grid(obstacle_pos, grid_origin)
        
        # Mark occupied cells within radius
        radius_cells = int(obstacle_radius / self.grid_resolution)
        
        for dx in range(-radius_cells, radius_cells + 1):
            for dy in range(-radius_cells, radius_cells + 1):
                for dz in range(-radius_cells, radius_cells + 1):
                    cell_x = grid_pos[0] + dx
                    cell_y = grid_pos[1] + dy
                    cell_z = grid_pos[2] + dz
                    
                    # Check bounds
                    if (0 <= cell_x < grid.shape[0] and 
                        0 <= cell_y < grid.shape[1] and 
                        0 <= cell_z < grid.shape[2]):
                        
                        # Check if within radius
                        dist_sq = dx*dx + dy*dy + dz*dz
                        if dist_sq <= radius_cells*radius_cells:
                            grid[cell_x, cell_y, cell_z] = True
    
    def _world_to_grid(self, world_pos: np.ndarray, grid_origin: np.ndarray) -> Tuple[int, int, int]:
        """Convert world coordinates to grid coordinates"""
        
        grid_pos = (world_pos - grid_origin) / self.grid_resolution
        return (int(grid_pos[0]), int(grid_pos[1]), int(grid_pos[2]))
    
    def _grid_to_world(self, grid_pos: Tuple[int, int, int], grid_origin: np.ndarray) -> np.ndarray:
        """Convert grid coordinates to world coordinates"""
        
        return grid_origin + np.array(grid_pos) * self.grid_resolution
    
    def _a_star_search(self, grid: np.ndarray, start: Tuple[int, int, int], 
                       goal: Tuple[int, int, int]) -> List[Tuple[int, int, int]]:
        """A* search algorithm implementation"""
        
        # Priority queue for open set
        open_set = [(0, start)]
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self._heuristic(start, goal)}
        
        while open_set:
            # Get node with lowest f_score
            current_f, current = open_set.pop(0)
            
            if current == goal:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                return path
            
            # Check neighbors
            for neighbor in self._get_neighbors(current, grid):
                tentative_g = g_score[current] + 1
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self._heuristic(neighbor, goal)
                    
                    # Add to open set
                    open_set.append((f_score[neighbor], neighbor))
                    open_set.sort()  # Sort by f_score
        
        return None  # No path found
    
    def _heuristic(self, pos1: Tuple[int, int, int], pos2: Tuple[int, int, int]) -> float:
        """Heuristic function for A* (Manhattan distance)"""
        
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1]) + abs(pos1[2] - pos2[2])
    
    def _get_neighbors(self, pos: Tuple[int, int, int], grid: np.ndarray) -> List[Tuple[int, int, int]]:
        """Get valid neighbors of a grid position"""
        
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    
                    neighbor = (pos[0] + dx, pos[1] + dy, pos[2] + dz)
                    
                    # Check bounds
                    if (0 <= neighbor[0] < grid.shape[0] and 
                        0 <= neighbor[1] < grid.shape[1] and 
                        0 <= neighbor[2] < grid.shape[2]):
                        
                        # Check if not occupied
                        if not grid[neighbor]:
                            neighbors.append(neighbor)
        
        return neighbors
    
    def _calculate_path_length(self, path: List[np.ndarray]) -> float:
        """Calculate total path length"""
        
        if len(path) < 2:
            return 0.0
        
        total_length = 0.0
        for i in range(1, len(path)):
            segment_length = np.linalg.norm(path[i] - path[i-1])
            total_length += segment_length
        
        return total_length

if __name__ == "__main__":
    # Example usage
    print("Motion Planning and Trajectory Optimization for UR5 Robot")
    print("This module provides:")
    print("- RRT-style motion planning")
    print("- Trajectory optimization")
    print("- A* path finding")
    print("- Collision avoidance")
    print("- Joint limit enforcement")
    
    # Note: This code requires a proper MuJoCo model to run the actual functionality 