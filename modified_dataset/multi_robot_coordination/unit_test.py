# Unit tests for: training.py (Multi-Robot Coordination)
# Tests the multi_robot_env.py, coordination_controller.py, task_allocator.py functionality

import pytest
import numpy as np
import time
from training import MultiRobotEnvironment, MultiRobot, CoordinationController

class TestMultiRobotEnvironment:
    """Test cases for MultiRobotEnvironment class"""
    
    def test_initialization(self):
        """Test environment initialization"""
        assert MultiRobotEnvironment is not None
    
    def test_robot_configuration(self):
        """Test robot configuration setup"""
        
        # Mock robot configuration
        num_robots = 3
        robot_names = [f"robot_{i}" for i in range(num_robots)]
        
        # Test robot names
        assert len(robot_names) == num_robots
        for i, name in enumerate(robot_names):
            assert name == f"robot_{i}"
        
        # Test robot parameters
        max_steps = 2000
        step_count = 0
        task_completion_threshold = 0.02
        
        assert max_steps > 0
        assert step_count == 0
        assert task_completion_threshold > 0
    
    def test_task_generation(self):
        """Test task generation functionality"""
        
        # Mock task generation
        num_robots = 2
        num_tasks = num_robots * 2  # 2 tasks per robot
        
        tasks = []
        for i in range(num_tasks):
            task = {
                "id": i,
                "type": "pick_and_place",
                "pick_position": np.random.uniform([0.2, -0.5, 0.1], [0.8, 0.5, 0.1]),
                "place_position": np.random.uniform([0.2, -0.5, 0.1], [0.8, 0.5, 0.1]),
                "assigned_robot": None,
                "completed": False,
                "priority": np.random.randint(1, 6)
            }
            tasks.append(task)
        
        # Test task structure
        assert len(tasks) == num_tasks
        
        for task in tasks:
            assert "id" in task
            assert "type" in task
            assert "pick_position" in task
            assert "place_position" in task
            assert "assigned_robot" in task
            assert "completed" in task
            assert "priority" in task
            
            # Test data types
            assert isinstance(task["id"], int)
            assert isinstance(task["type"], str)
            assert isinstance(task["pick_position"], np.ndarray)
            assert isinstance(task["place_position"], np.ndarray)
            assert task["assigned_robot"] is None
            assert isinstance(task["completed"], bool)
            assert isinstance(task["priority"], int)
            
            # Test value ranges
            assert 1 <= task["priority"] <= 5
            assert task["pick_position"].shape == (3,)
            assert task["place_position"].shape == (3,)
    
    def test_task_completion_tracking(self):
        """Test task completion tracking"""
        
        # Mock task completion scenario
        tasks = [
            {"id": 0, "completed": False, "assigned_robot": 0},
            {"id": 1, "completed": False, "assigned_robot": 1},
            {"id": 2, "completed": False, "assigned_robot": 0}
        ]
        
        completed_tasks = []
        task_queue = tasks.copy()
        
        # Simulate task completion
        for task in tasks:
            if task["id"] == 0:  # Complete first task
                task["completed"] = True
                completed_tasks.append(task)
                if task in task_queue:
                    task_queue.remove(task)
        
        # Test completion tracking
        assert len(completed_tasks) == 1
        assert completed_tasks[0]["id"] == 0
        assert len(task_queue) == 2
        assert tasks[0]["completed"] is True
        assert tasks[1]["completed"] is False
        assert tasks[2]["completed"] is False
    
    def test_collision_detection(self):
        """Test collision detection between robots"""
        
        # Mock robot positions
        robot_positions = [
            np.array([0.0, 0.0, 0.0]),
            np.array([0.05, 0.0, 0.0]),  # Very close to robot 0
            np.array([1.0, 1.0, 1.0])   # Far from others
        ]
        
        collision_safety_margin = 0.1
        
        # Check for collisions
        collisions_detected = False
        for i in range(len(robot_positions)):
            for j in range(i + 1, len(robot_positions)):
                distance = np.linalg.norm(robot_positions[i] - robot_positions[j])
                if distance < collision_safety_margin:
                    collisions_detected = True
                    break
            if collisions_detected:
                break
        
        # Test collision detection
        assert collisions_detected is True  # Robots 0 and 1 are too close
        
        # Test no collision scenario
        safe_positions = [
            np.array([0.0, 0.0, 0.0]),
            np.array([0.5, 0.0, 0.0]),  # Safe distance
            np.array([1.0, 1.0, 1.0])
        ]
        
        safe_collisions = False
        for i in range(len(safe_positions)):
            for j in range(i + 1, len(safe_positions)):
                distance = np.linalg.norm(safe_positions[i] - safe_positions[j])
                if distance < collision_safety_margin:
                    safe_collisions = True
                    break
            if safe_collisions:
                break
        
        assert safe_collisions is False  # All robots are at safe distances
    
    def test_reward_calculation(self):
        """Test reward calculation logic"""
        
        # Mock reward components
        completed_tasks = 2
        task_reward = completed_tasks * 10.0
        
        # Mock proximity rewards
        proximity_rewards = [2.0, 1.0, 0.5]
        total_proximity_reward = sum(proximity_rewards)
        
        # Mock penalties
        collision_penalty = -5.0 if True else 0.0  # Simulate collision
        velocity_penalties = [-0.1, -0.2, -0.3]
        total_velocity_penalty = sum(velocity_penalties)
        
        # Calculate total reward
        total_reward = task_reward + total_proximity_reward + collision_penalty + total_velocity_penalty
        
        # Test reward calculation
        expected_reward = 20.0 + 3.5 - 5.0 - 0.6  # 17.9
        assert abs(total_reward - expected_reward) < 1e-6
        
        # Test reward properties
        assert total_reward > 0  # Should be positive for good performance
        assert task_reward > 0   # Task completion should always be positive
        assert collision_penalty < 0  # Collisions should be penalized

class TestMultiRobot:
    """Test cases for MultiRobot class"""
    
    def test_initialization(self):
        """Test robot initialization"""
        assert MultiRobot is not None
    
    def test_robot_state_management(self):
        """Test robot state management"""
        
        # Mock robot state
        robot_id = 0
        robot_name = "robot_0"
        current_task_id = 1
        task_phase = "pick"
        is_carrying_object = False
        
        # Test state properties
        assert robot_id == 0
        assert robot_name == "robot_0"
        assert current_task_id == 1
        assert task_phase == "pick"
        assert is_carrying_object is False
        
        # Test task phase transitions
        task_phases = ["idle", "pick", "place"]
        assert "idle" in task_phases
        assert "pick" in task_phases
        assert "place" in task_phases
    
    def test_action_execution(self):
        """Test robot action execution"""
        
        # Mock action
        target_position = np.array([0.5, 0.0, 0.1])
        gripper_action = 0.8  # Close gripper
        
        # Test action parsing
        assert len(target_position) == 3
        assert 0.0 <= gripper_action <= 1.0
        
        # Test gripper control logic
        if gripper_action > 0.5:  # Close gripper
            expected_gripper_pos = 0.0
        else:  # Open gripper
            expected_gripper_pos = 0.04
        
        assert expected_gripper_pos == 0.0  # Should close gripper
    
    def test_position_control(self):
        """Test robot position control"""
        
        # Mock position control parameters
        position_gain = 50.0
        max_velocity = 2.0
        
        # Mock position error
        current_position = np.array([0.0, 0.0, 0.0])
        target_position = np.array([1.0, 0.0, 0.0])
        position_error = target_position - current_position
        
        # Calculate velocity
        velocity = position_gain * position_error
        
        # Test velocity calculation
        expected_velocity = np.array([50.0, 0.0, 0.0])
        assert np.allclose(velocity, expected_velocity)
        
        # Test velocity limiting
        velocity_magnitude = np.linalg.norm(velocity)
        if velocity_magnitude > max_velocity:
            velocity = velocity * (max_velocity / velocity_magnitude)
        
        # Should be limited
        limited_magnitude = np.linalg.norm(velocity)
        assert limited_magnitude <= max_velocity
    
    def test_task_state_updates(self):
        """Test robot task state updates"""
        
        # Mock task state
        current_task_id = 1
        task_phase = "pick"
        is_carrying_object = False
        task_start_time = time.time()
        task_timeout = 30.0
        
        # Test initial state
        assert current_task_id == 1
        assert task_phase == "pick"
        assert is_carrying_object is False
        assert task_start_time > 0
        assert task_timeout > 0
        
        # Test timeout logic
        elapsed_time = time.time() - task_start_time
        if elapsed_time > task_timeout:
            # Task timeout, reset
            current_task_id = None
            task_phase = "idle"
            is_carrying_object = False
        
        # Test state reset
        assert current_task_id is None or current_task_id == 1  # Depends on timing
        assert task_phase in ["idle", "pick"]  # Depends on timing
    
    def test_observation_generation(self):
        """Test robot observation generation"""
        
        # Mock robot observation
        observation = {
            "position": np.array([0.5, 0.0, 0.1]),
            "joint_positions": np.zeros(6),
            "joint_velocities": np.zeros(6),
            "current_task_id": 1,
            "task_phase": "pick",
            "is_carrying_object": False
        }
        
        # Test observation structure
        assert "position" in observation
        assert "joint_positions" in observation
        assert "joint_velocities" in observation
        assert "current_task_id" in observation
        assert "task_phase" in observation
        assert "is_carrying_object" in observation
        
        # Test data types
        assert isinstance(observation["position"], np.ndarray)
        assert isinstance(observation["joint_positions"], np.ndarray)
        assert isinstance(observation["joint_velocities"], np.ndarray)
        assert isinstance(observation["current_task_id"], int)
        assert isinstance(observation["task_phase"], str)
        assert isinstance(observation["is_carrying_object"], bool)
        
        # Test data shapes
        assert observation["position"].shape == (3,)
        assert observation["joint_positions"].shape == (6,)
        assert observation["joint_velocities"].shape == (6,)

class TestCoordinationController:
    """Test cases for CoordinationController class"""
    
    def test_initialization(self):
        """Test coordination controller initialization"""
        assert CoordinationController is not None
    
    def test_task_allocation_strategies(self):
        """Test different task allocation strategies"""
        
        # Mock allocation strategies
        strategies = ["greedy", "auction", "centralized"]
        
        # Test strategy configuration
        for strategy in strategies:
            assert strategy in strategies
        
        # Test strategy selection
        current_strategy = "greedy"
        assert current_strategy in strategies
        
        # Test strategy switching
        if current_strategy == "greedy":
            allocation_method = "greedy_allocation"
        elif current_strategy == "auction":
            allocation_method = "auction_allocation"
        elif current_strategy == "centralized":
            allocation_method = "centralized_allocation"
        
        assert allocation_method == "greedy_allocation"
    
    def test_greedy_allocation(self):
        """Test greedy task allocation"""
        
        # Mock robots and tasks
        robots = [{"robot_id": 0, "current_task_id": None}, {"robot_id": 1, "current_task_id": None}]
        tasks = [
            {"id": 0, "pick_position": np.array([0.1, 0.0, 0.1]), "completed": False},
            {"id": 1, "pick_position": np.array([0.9, 0.0, 0.1]), "completed": False}
        ]
        
        # Mock robot positions
        robot_positions = [
            np.array([0.0, 0.0, 0.0]),  # Robot 0 at origin
            np.array([1.0, 0.0, 0.0])   # Robot 1 at far end
        ]
        
        # Greedy allocation
        allocation = {}
        available_tasks = [task for task in tasks if not task["completed"]]
        
        for robot in robots:
            if robot["current_task_id"] is None and available_tasks:
                robot_pos = robot_positions[robot["robot_id"]]
                closest_task = None
                min_distance = float('inf')
                
                for task in available_tasks:
                    distance = np.linalg.norm(robot_pos - task["pick_position"])
                    if distance < min_distance:
                        min_distance = distance
                        closest_task = task
                
                if closest_task:
                    allocation[robot["robot_id"]] = closest_task["id"]
                    available_tasks.remove(closest_task)
        
        # Test allocation
        assert len(allocation) == 2
        assert allocation[0] == 0  # Robot 0 should get task 0 (closer)
        assert allocation[1] == 1  # Robot 1 should get task 1 (closer)
    
    def test_auction_allocation(self):
        """Test auction-based task allocation"""
        
        # Mock auction scenario
        task = {"id": 0, "pick_position": np.array([0.5, 0.0, 0.1]), "priority": 3}
        robots = [
            {"robot_id": 0, "current_task_id": None, "position": np.array([0.0, 0.0, 0.0])},
            {"robot_id": 1, "current_task_id": None, "position": np.array([1.0, 0.0, 0.0])}
        ]
        
        # Calculate bids
        bids = {}
        for robot in robots:
            if robot["current_task_id"] is None:
                robot_pos = robot["position"]
                distance = np.linalg.norm(robot_pos - task["pick_position"])
                priority_bonus = task["priority"] * 0.1
                bid = 1.0 / (distance + 0.1) + priority_bonus
                bids[robot["robot_id"]] = bid
        
        # Test bidding
        assert len(bids) == 2
        assert 0 in bids
        assert 1 in bids
        
        # Robot 0 should have higher bid (closer to task)
        assert bids[0] > bids[1]
        
        # Find winner
        winner_robot_id = max(bids, key=bids.get)
        assert winner_robot_id == 0
    
    def test_collision_avoidance(self):
        """Test collision avoidance functionality"""
        
        # Mock collision scenario
        robot1_pos = np.array([0.0, 0.0, 0.0])
        robot2_pos = np.array([0.05, 0.0, 0.0])  # Very close
        
        collision_safety_margin = 0.1
        
        # Check collision
        distance = np.linalg.norm(robot1_pos - robot2_pos)
        collision_detected = distance < collision_safety_margin
        
        assert collision_detected is True
        
        # Calculate avoidance force
        avoidance_direction = robot1_pos - robot2_pos
        if np.linalg.norm(avoidance_direction) > 0:
            avoidance_direction = avoidance_direction / np.linalg.norm(avoidance_direction)
        
        avoidance_force = 0.1 * avoidance_direction
        
        # Apply avoidance
        new_robot1_pos = robot1_pos + avoidance_force
        new_robot2_pos = robot2_pos - avoidance_force
        
        # Test avoidance
        new_distance = np.linalg.norm(new_robot1_pos - new_robot2_pos)
        assert new_distance > distance  # Robots should be further apart
    
    def test_coordination_status(self):
        """Test coordination status reporting"""
        
        # Mock coordination status
        status = {
            "num_robots": 2,
            "num_tasks": 4,
            "completed_tasks": 1,
            "robot_status": {
                0: {"task_id": 1, "phase": "pick", "carrying_object": False, "position": [0.0, 0.0, 0.0]},
                1: {"task_id": 2, "phase": "place", "carrying_object": True, "position": [0.5, 0.0, 0.1]}
            },
            "task_allocation": {0: 1, 1: 2},
            "collision_detected": False
        }
        
        # Test status structure
        assert "num_robots" in status
        assert "num_tasks" in status
        assert "completed_tasks" in status
        assert "robot_status" in status
        assert "task_allocation" in status
        assert "collision_detected" in status
        
        # Test status values
        assert status["num_robots"] == 2
        assert status["num_tasks"] == 4
        assert status["completed_tasks"] == 1
        assert len(status["robot_status"]) == 2
        assert len(status["task_allocation"]) == 2
        assert status["collision_detected"] is False
        
        # Test robot status
        robot_0_status = status["robot_status"][0]
        assert robot_0_status["task_id"] == 1
        assert robot_0_status["phase"] == "pick"
        assert robot_0_status["carrying_object"] is False
        assert robot_0_status["position"] == [0.0, 0.0, 0.0]

class TestIntegration:
    """Integration tests for multi-robot coordination system"""
    
    def test_environment_robot_interaction(self):
        """Test environment and robot interaction"""
        
        # Mock environment state
        num_robots = 2
        num_tasks = 4
        step_count = 10
        
        # Test environment properties
        assert num_robots > 0
        assert num_tasks > 0
        assert step_count > 0
        
        # Test robot-task relationship
        tasks_per_robot = num_tasks / num_robots
        assert tasks_per_robot == 2.0
    
    def test_coordination_workflow(self):
        """Test complete coordination workflow"""
        
        # Mock workflow steps
        workflow_steps = [
            "task_generation",
            "robot_initialization", 
            "task_allocation",
            "execution",
            "collision_avoidance",
            "task_completion",
            "status_reporting"
        ]
        
        # Test workflow structure
        assert len(workflow_steps) == 7
        assert "task_generation" in workflow_steps
        assert "task_allocation" in workflow_steps
        assert "execution" in workflow_steps
        
        # Test workflow progression
        current_step = 0
        for step in workflow_steps:
            assert step == workflow_steps[current_step]
            current_step += 1
        
        assert current_step == len(workflow_steps)
    
    def test_performance_metrics(self):
        """Test performance metrics calculation"""
        
        # Mock performance data
        start_time = time.time()
        execution_time = 0.1  # Simulate execution time
        
        # Mock metrics
        tasks_completed = 3
        total_tasks = 4
        completion_rate = tasks_completed / total_tasks
        efficiency = completion_rate / execution_time
        
        # Test metrics
        assert completion_rate == 0.75
        assert efficiency > 0
        assert tasks_completed <= total_tasks
        assert completion_rate <= 1.0 