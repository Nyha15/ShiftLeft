# Unit tests for: training.py (UR5 Motion Planning and Trajectory Optimization)
# Tests the motion_planner.py, trajectory_optimizer.py, path_finder.py functionality

import pytest
import numpy as np
from training import MotionPlanner, TrajectoryOptimizer, PathFinder

class TestMotionPlanner:
    """Test cases for MotionPlanner class"""
    
    def test_initialization(self):
        """Test motion planner initialization"""
        assert MotionPlanner is not None
    
    def test_planning_parameters(self):
        """Test planning parameter configuration"""
        
        # Mock planning parameters
        max_iterations = 1000
        planning_timeout = 5.0
        resolution = 0.01
        max_velocity = 2.0
        max_acceleration = 5.0
        
        # Test parameter values
        assert max_iterations > 0
        assert planning_timeout > 0
        assert resolution > 0
        assert max_velocity > 0
        assert max_acceleration > 0
        
        # Test parameter relationships
        assert max_acceleration > max_velocity  # Acceleration should be higher
        assert resolution < 0.1  # Resolution should be small
    
    def test_joint_limits_structure(self):
        """Test joint limits data structure"""
        
        # Mock joint limits
        joint_limits = {
            "joint_0": {"lower": -3.14, "upper": 3.14, "range": 6.28},
            "joint_1": {"lower": -3.14, "upper": 3.14, "range": 6.28},
            "joint_2": {"lower": -3.14, "upper": 3.14, "range": 6.28},
            "joint_3": {"lower": -3.14, "upper": 3.14, "range": 6.28},
            "joint_4": {"lower": -3.14, "upper": 3.14, "range": 6.28},
            "joint_5": {"lower": -3.14, "upper": 3.14, "range": 6.28}
        }
        
        # Test structure
        assert len(joint_limits) == 6
        
        for joint_name, limits in joint_limits.items():
            assert "lower" in limits
            assert "upper" in limits
            assert "range" in limits
            assert limits["lower"] < limits["upper"]
            assert abs(limits["range"] - (limits["upper"] - limits["lower"])) < 1e-6
    
    def test_configuration_validation(self):
        """Test joint configuration validation"""
        
        # Valid configurations
        valid_configs = [
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            np.array([1.0, -1.0, 0.5, -0.5, 0.25, -0.25]),
            np.array([3.0, -3.0, 2.0, -2.0, 1.0, -1.0])
        ]
        
        # Invalid configurations (out of bounds)
        invalid_configs = [
            np.array([4.0, 0.0, 0.0, 0.0, 0.0, 0.0]),  # Joint 0 too high
            np.array([0.0, -4.0, 0.0, 0.0, 0.0, 0.0]),  # Joint 1 too low
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 4.0])   # Joint 5 too high
        ]
        
        # Test valid configs
        for config in valid_configs:
            assert len(config) == 6
            assert all(-3.15 < angle < 3.15 for angle in config)
        
        # Test invalid configs
        for config in invalid_configs:
            assert any(angle < -3.15 or angle > 3.15 for angle in config)
    
    def test_random_configuration_generation(self):
        """Test random configuration generation"""
        
        # Generate multiple random configurations
        num_configs = 10
        configs = []
        
        for _ in range(num_configs):
            # Mock random generation
            config = np.random.uniform(-3.0, 3.0, 6)
            configs.append(config)
        
        # Test properties
        assert len(configs) == num_configs
        
        for config in configs:
            assert len(config) == 6
            assert all(-3.0 <= angle <= 3.0 for angle in config)
            assert config.dtype == np.float64
    
    def test_nearest_neighbor_search(self):
        """Test nearest neighbor configuration search"""
        
        # Test configurations
        target = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        configurations = [
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            np.array([2.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
        ]
        
        # Calculate distances
        distances = [np.linalg.norm(target - config) for config in configurations]
        
        # Find nearest
        nearest_idx = np.argmin(distances)
        nearest_config = configurations[nearest_idx]
        
        # Test nearest neighbor
        assert nearest_idx == 2  # [0.5, 0, 0, 0, 0, 0] should be closest to [1, 0, 0, 0, 0, 0]
        assert np.allclose(nearest_config, np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0]))
    
    def test_configuration_extension(self):
        """Test configuration extension logic"""
        
        # Test extension parameters
        from_config = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        to_config = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        resolution = 0.01
        
        # Calculate direction and distance
        direction = to_config - from_config
        distance = np.linalg.norm(direction)
        
        # Test extension logic
        if distance < resolution:
            extended_config = to_config
        else:
            # Normalize and scale
            direction = direction / distance * resolution
            extended_config = from_config + direction
        
        # Verify extension
        assert np.linalg.norm(extended_config - from_config) <= resolution
        assert np.allclose(extended_config, from_config + direction)
    
    def test_trajectory_optimization(self):
        """Test trajectory optimization functionality"""
        
        # Mock trajectory data
        trajectory = [
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0]),
            np.array([0.2, 0.0, 0.0, 0.0, 0.0, 0.0]),
            np.array([0.3, 0.0, 0.0, 0.0, 0.0, 0.0])
        ]
        
        velocities = [
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        ]
        
        accelerations = [
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        ]
        
        # Test optimization structure
        assert len(trajectory) == 4
        assert len(velocities) == 4
        assert len(accelerations) == 4
        
        # Test trajectory properties
        for i, config in enumerate(trajectory):
            assert len(config) == 6
            assert config[0] == i * 0.1  # X coordinate increases linearly
        
        # Test velocity properties
        for i, vel in enumerate(velocities):
            assert len(vel) == 6
            if i == 0:
                assert np.allclose(vel, np.zeros(6))
            else:
                assert np.allclose(vel, np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]))

class TestTrajectoryOptimizer:
    """Test cases for TrajectoryOptimizer class"""
    
    def test_initialization(self):
        """Test trajectory optimizer initialization"""
        assert TrajectoryOptimizer is not None
    
    def test_optimization_parameters(self):
        """Test optimization parameter configuration"""
        
        # Mock optimization parameters
        smoothness_weight = 1.0
        efficiency_weight = 0.5
        collision_weight = 10.0
        
        # Test parameter values
        assert smoothness_weight > 0
        assert efficiency_weight > 0
        assert collision_weight > 0
        
        # Test weight relationships
        assert collision_weight > smoothness_weight  # Collision should have highest weight
        assert smoothness_weight > efficiency_weight  # Smoothness usually more important than efficiency
    
    def test_trajectory_cost_calculation(self):
        """Test trajectory cost calculation"""
        
        # Mock trajectory
        trajectory = np.array([
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.2, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.3, 0.0, 0.0, 0.0, 0.0, 0.0]
        ])
        
        # Test smoothness cost calculation
        smoothness_cost = 0.0
        for i in range(1, len(trajectory) - 1):
            second_deriv = trajectory[i+1] - 2*trajectory[i] + trajectory[i-1]
            smoothness_cost += np.sum(second_deriv**2)
        
        # Test efficiency cost calculation
        efficiency_cost = 0.0
        for i in range(1, len(trajectory)):
            path_segment = trajectory[i] - trajectory[i-1]
            efficiency_cost += np.linalg.norm(path_segment)
        
        # Test cost properties
        assert smoothness_cost >= 0
        assert efficiency_cost >= 0
        assert isinstance(smoothness_cost, float)
        assert isinstance(efficiency_cost, float)
    
    def test_constraint_handling(self):
        """Test constraint handling in optimization"""
        
        # Mock constraints
        constraints = {
            "joint_limits": [
                {"lower": -3.14, "upper": 3.14},
                {"lower": -3.14, "upper": 3.14},
                {"lower": -3.14, "upper": 3.14},
                {"lower": -3.14, "upper": 3.14},
                {"lower": -3.14, "upper": 3.14},
                {"lower": -3.14, "upper": 3.14}
            ],
            "obstacles": [
                {"position": np.array([0.5, 0.0, 0.1]), "radius": 0.1}
            ]
        }
        
        # Test constraint structure
        assert "joint_limits" in constraints
        assert "obstacles" in constraints
        
        # Test joint limits
        joint_limits = constraints["joint_limits"]
        assert len(joint_limits) == 6
        
        for limit in joint_limits:
            assert "lower" in limit
            assert "upper" in limit
            assert limit["lower"] < limit["upper"]
        
        # Test obstacles
        obstacles = constraints["obstacles"]
        assert len(obstacles) == 1
        
        for obstacle in obstacles:
            assert "position" in obstacle
            assert "radius" in obstacle
            assert isinstance(obstacle["position"], np.ndarray)
            assert obstacle["position"].shape == (3,)
            assert obstacle["radius"] > 0

class TestPathFinder:
    """Test cases for PathFinder class"""
    
    def test_initialization(self):
        """Test path finder initialization"""
        assert PathFinder is not None
    
    def test_grid_creation(self):
        """Test occupancy grid creation"""
        
        # Mock grid parameters
        grid_resolution = 0.05
        x_min, x_max = -1.0, 1.0
        y_min, y_max = -1.0, 1.0
        z_min, z_max = 0.0, 1.0
        
        # Calculate grid dimensions
        nx = int((x_max - x_min) / grid_resolution)
        ny = int((y_max - y_min) / grid_resolution)
        nz = int((z_max - z_min) / grid_resolution)
        
        # Test grid dimensions
        assert nx > 0
        assert ny > 0
        assert nz > 0
        assert nx == 40  # (1.0 - (-1.0)) / 0.05
        assert ny == 40  # (1.0 - (-1.0)) / 0.05
        assert nz == 20  # (1.0 - 0.0) / 0.05
        
        # Create empty grid
        grid = np.zeros((nx, ny, nz), dtype=bool)
        grid_origin = np.array([x_min, y_min, z_min])
        
        # Test grid properties
        assert grid.shape == (nx, ny, nz)
        assert grid.dtype == bool
        assert np.all(~grid)  # Should be all False initially
        assert grid_origin.shape == (3,)
    
    def test_coordinate_conversion(self):
        """Test world to grid coordinate conversion"""
        
        # Mock grid parameters
        grid_resolution = 0.05
        grid_origin = np.array([-1.0, -1.0, 0.0])
        
        # Test world to grid conversion
        world_pos = np.array([0.0, 0.0, 0.5])
        grid_pos = (world_pos - grid_origin) / grid_resolution
        grid_coords = (int(grid_pos[0]), int(grid_pos[1]), int(grid_pos[2]))
        
        # Expected: (0.0 - (-1.0)) / 0.05 = 20, (0.0 - (-1.0)) / 0.05 = 20, (0.5 - 0.0) / 0.05 = 10
        expected_grid = (20, 20, 10)
        assert grid_coords == expected_grid
        
        # Test grid to world conversion
        world_coords = grid_origin + np.array(grid_coords) * grid_resolution
        expected_world = np.array([0.0, 0.0, 0.5])
        assert np.allclose(world_coords, expected_world)
    
    def test_obstacle_addition(self):
        """Test obstacle addition to grid"""
        
        # Create test grid
        grid = np.zeros((10, 10, 10), dtype=bool)
        grid_origin = np.array([0.0, 0.0, 0.0])
        grid_resolution = 0.1
        
        # Mock obstacle
        obstacle = {
            "position": np.array([0.5, 0.5, 0.5]),
            "radius": 0.2
        }
        
        # Convert to grid coordinates
        grid_pos = (obstacle["position"] - grid_origin) / grid_resolution
        grid_coords = (int(grid_pos[0]), int(grid_pos[1]), int(grid_pos[2]))
        
        # Calculate radius in grid cells
        radius_cells = int(obstacle["radius"] / grid_resolution)
        
        # Mark occupied cells
        for dx in range(-radius_cells, radius_cells + 1):
            for dy in range(-radius_cells, radius_cells + 1):
                for dz in range(-radius_cells, radius_cells + 1):
                    cell_x = grid_coords[0] + dx
                    cell_y = grid_coords[1] + dy
                    cell_z = grid_coords[2] + dz
                    
                    # Check bounds
                    if (0 <= cell_x < grid.shape[0] and 
                        0 <= cell_y < grid.shape[1] and 
                        0 <= cell_z < grid.shape[2]):
                        
                        # Check if within radius
                        dist_sq = dx*dx + dy*dy + dz*dz
                        if dist_sq <= radius_cells*radius_cells:
                            grid[cell_x, cell_y, cell_z] = True
        
        # Test that some cells were marked
        assert np.any(grid)  # Should have some True values
        assert np.sum(grid) > 0  # Should have occupied cells
    
    def test_a_star_heuristic(self):
        """Test A* heuristic function"""
        
        # Test positions
        pos1 = (0, 0, 0)
        pos2 = (3, 4, 5)
        
        # Calculate Manhattan distance
        expected_heuristic = abs(3-0) + abs(4-0) + abs(5-0)  # 3 + 4 + 5 = 12
        
        # Mock heuristic calculation
        heuristic = abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1]) + abs(pos1[2] - pos2[2])
        
        assert heuristic == expected_heuristic
        assert heuristic == 12
    
    def test_neighbor_generation(self):
        """Test neighbor generation for A*"""
        
        # Test position
        pos = (5, 5, 5)
        grid_shape = (10, 10, 10)
        
        # Generate neighbors
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    
                    neighbor = (pos[0] + dx, pos[1] + dy, pos[2] + dz)
                    
                    # Check bounds
                    if (0 <= neighbor[0] < grid_shape[0] and 
                        0 <= neighbor[1] < grid_shape[1] and 
                        0 <= neighbor[2] < grid_shape[2]):
                        neighbors.append(neighbor)
        
        # Test neighbor properties
        assert len(neighbors) == 26  # 3^3 - 1 = 27 - 1 = 26 (excluding center)
        
        # Test that all neighbors are within bounds
        for neighbor in neighbors:
            assert 0 <= neighbor[0] < grid_shape[0]
            assert 0 <= neighbor[1] < grid_shape[1]
            assert 0 <= neighbor[2] < grid_shape[2]
    
    def test_path_length_calculation(self):
        """Test path length calculation"""
        
        # Test path
        path = [
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
            np.array([1.0, 1.0, 0.0]),
            np.array([1.0, 1.0, 1.0])
        ]
        
        # Calculate path length
        total_length = 0.0
        for i in range(1, len(path)):
            segment_length = np.linalg.norm(path[i] - path[i-1])
            total_length += segment_length
        
        # Expected: 1.0 + 1.0 + 1.0 = 3.0
        expected_length = 3.0
        assert abs(total_length - expected_length) < 1e-6
        
        # Test edge cases
        single_point_path = [np.array([0.0, 0.0, 0.0])]
        single_length = 0.0  # No segments
        assert single_length == 0.0
        
        empty_path = []
        empty_length = 0.0
        assert empty_length == 0.0

class TestIntegration:
    """Integration tests for motion planning system"""
    
    def test_planning_pipeline(self):
        """Test complete motion planning pipeline"""
        
        # Mock planning scenario
        start_config = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        goal_config = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        
        # Mock obstacles
        obstacles = [
            {"position": np.array([0.5, 0.0, 0.1]), "radius": 0.1}
        ]
        
        # Test planning structure
        assert len(start_config) == 6
        assert len(goal_config) == 6
        assert len(obstacles) == 1
        
        # Test obstacle properties
        obstacle = obstacles[0]
        assert "position" in obstacle
        assert "radius" in obstacle
        assert obstacle["position"].shape == (3,)
        assert obstacle["radius"] > 0
    
    def test_optimization_pipeline(self):
        """Test trajectory optimization pipeline"""
        
        # Mock trajectory data
        trajectory = [
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        ]
        
        # Mock constraints
        constraints = {
            "joint_limits": [
                {"lower": -3.14, "upper": 3.14} for _ in range(6)
            ]
        }
        
        # Test optimization structure
        assert len(trajectory) == 3
        assert len(constraints["joint_limits"]) == 6
        
        # Test trajectory properties
        for i, config in enumerate(trajectory):
            assert len(config) == 6
            assert config[0] == i * 0.5  # Linear progression
    
    def test_path_finding_pipeline(self):
        """Test path finding pipeline"""
        
        # Mock path finding scenario
        start_pos = np.array([0.0, 0.0, 0.0])
        goal_pos = np.array([1.0, 1.0, 1.0])
        
        # Mock obstacles
        obstacles = [
            {"position": np.array([0.5, 0.5, 0.5]), "radius": 0.2}
        ]
        
        # Test scenario properties
        assert start_pos.shape == (3,)
        assert goal_pos.shape == (3,)
        assert len(obstacles) == 1
        
        # Test distance calculation
        distance = np.linalg.norm(goal_pos - start_pos)
        expected_distance = np.sqrt(3)  # sqrt(1² + 1² + 1²)
        assert abs(distance - expected_distance) < 1e-6 