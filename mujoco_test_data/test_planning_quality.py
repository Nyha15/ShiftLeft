import mujoco
import numpy as np
import pytest

def test_path_planning_optimality():
    model = mujoco.MjModel.from_xml_path("robot_arm.xml")
    data = mujoco.MjData(model)
    
    start_config = np.array([0, 0, 0, 0, 0, 0])
    goal_config = np.array([np.pi/2, np.pi/3, np.pi/4, 0, 0, 0])
    
    data.qpos[:] = start_config
    mujoco.mj_forward(model, data)
    start_pos = data.xpos[model.body("end_effector").id]
    
    data.qpos[:] = goal_config
    mujoco.mj_forward(model, data)
    goal_pos = data.xpos[model.body("end_effector").id]
    
    direct_distance = np.linalg.norm(goal_pos - start_pos)
    
    planned_path = []
    current_config = start_config.copy()
    
    for step in range(100):
        data.qpos[:] = current_config
        mujoco.mj_forward(model, data)
        current_pos = data.xpos[model.body("end_effector").id]
        planned_path.append(current_pos)
        
        if np.linalg.norm(current_pos - goal_pos) < 0.01:
            break
        
        direction = (goal_pos - current_pos)
        direction = direction / np.linalg.norm(direction)
        
        jacobian = np.zeros((3, model.nv))
        mujoco.mj_jac(model, data, jacobian, None, np.zeros(3), model.body("end_effector").id)
        
        delta_q = np.linalg.pinv(jacobian) @ (direction * 0.05)
        current_config += delta_q
    
    planned_path = np.array(planned_path)
    path_length = np.sum(np.linalg.norm(np.diff(planned_path, axis=0), axis=1))
    
    optimality_ratio = direct_distance / path_length
    assert optimality_ratio > 0.8

def test_motion_planning_efficiency():
    model = mujoco.MjModel.from_xml_path("robot_arm.xml")
    data = mujoco.MjData(model)
    
    test_scenarios = [
        (np.array([0, 0, 0, 0, 0, 0]), np.array([np.pi/4, 0, 0, 0, 0, 0])),
        (np.array([np.pi/4, 0, 0, 0, 0, 0]), np.array([np.pi/2, np.pi/6, 0, 0, 0, 0])),
        (np.array([np.pi/2, np.pi/6, 0, 0, 0, 0]), np.array([0, 0, 0, 0, 0, 0]))
    ]
    
    total_planning_time = 0
    total_path_length = 0
    
    for start, goal in test_scenarios:
        data.qpos[:] = start
        mujoco.mj_forward(model, data)
        start_pos = data.xpos[model.body("end_effector").id]
        
        data.qpos[:] = goal
        mujoco.mj_forward(model, data)
        goal_pos = data.xpos[model.body("end_effector").id]
        
        planning_steps = 0
        current_config = start.copy()
        
        while planning_steps < 200:
            data.qpos[:] = current_config
            mujoco.mj_forward(model, data)
            current_pos = data.xpos[model.body("end_effector").id]
            
            if np.linalg.norm(current_pos - goal_pos) < 0.01:
                break
            
            jacobian = np.zeros((3, model.nv))
            mujoco.mj_jac(model, data, jacobian, None, np.zeros(3), model.body("end_effector").id)
            
            error = goal_pos - current_pos
            delta_q = np.linalg.pinv(jacobian) @ error
            current_config += delta_q * 0.1
            
            planning_steps += 1
        
        total_planning_time += planning_steps
        total_path_length += np.linalg.norm(goal_pos - start_pos)
    
    avg_planning_steps = total_planning_time / len(test_scenarios)
    assert avg_planning_steps < 100

def test_collision_free_planning():
    model = mujoco.MjModel.from_xml_path("robot_arm.xml")
    data = mujoco.MjData(model)
    
    start_config = np.array([0, 0, 0, 0, 0, 0])
    goal_config = np.array([np.pi/2, np.pi/3, np.pi/4, 0, 0, 0])
    
    obstacle_positions = [
        np.array([0.3, 0.0, 0.3]),
        np.array([0.5, 0.1, 0.25])
    ]
    
    data.qpos[:] = start_config
    mujoco.mj_forward(model, data)
    start_pos = data.xpos[model.body("end_effector").id]
    
    data.qpos[:] = goal_config
    mujoco.mj_forward(model, data)
    goal_pos = data.xpos[model.body("end_effector").id]
    
    current_config = start_config.copy()
    collision_detected = False
    
    for step in range(300):
        data.qpos[:] = current_config
        mujoco.mj_forward(model, data)
        current_pos = data.xpos[model.body("end_effector").id]
        
        for obstacle in obstacle_positions:
            if np.linalg.norm(current_pos - obstacle) < 0.1:
                collision_detected = True
                break
        
        if collision_detected:
            break
        
        if np.linalg.norm(current_pos - goal_pos) < 0.01:
            break
        
        direction = goal_pos - current_pos
        direction = direction / np.linalg.norm(direction)
        
        for obstacle in obstacle_positions:
            distance_to_obstacle = np.linalg.norm(current_pos - obstacle)
            if distance_to_obstacle < 0.3:
                repulsive_direction = (current_pos - obstacle) / distance_to_obstacle
                direction += 0.5 * repulsive_direction
        
        direction = direction / np.linalg.norm(direction)
        
        jacobian = np.zeros((3, model.nv))
        mujoco.mj_jac(model, data, jacobian, None, np.zeros(3), model.body("end_effector").id)
        
        delta_q = np.linalg.pinv(jacobian) @ (direction * 0.03)
        current_config += delta_q
    
    assert not collision_detected

def test_trajectory_optimization():
    model = mujoco.MjModel.from_xml_path("robot_arm.xml")
    data = mujoco.MjData(model)
    
    waypoints = [
        np.array([0, 0, 0, 0, 0, 0]),
        np.array([np.pi/6, np.pi/12, np.pi/8, 0, 0, 0]),
        np.array([np.pi/3, np.pi/6, np.pi/4, 0, 0, 0]),
        np.array([np.pi/2, np.pi/3, np.pi/3, 0, 0, 0])
    ]
    
    trajectory_costs = []
    
    for _ in range(5):
        current_cost = 0
        current_config = waypoints[0].copy()
        
        for i in range(len(waypoints) - 1):
            target_config = waypoints[i + 1]
            
            for step in range(50):
                data.qpos[:] = current_config
                mujoco.mj_forward(model, data)
                
                config_error = target_config - current_config
                velocity_cost = np.sum(data.qvel ** 2)
                position_cost = np.sum(config_error ** 2)
                
                current_cost += velocity_cost + position_cost
                
                if np.linalg.norm(config_error) < 0.01:
                    break
                
                control_torque = 50.0 * config_error + 10.0 * (-data.qvel)
                data.ctrl[:] = control_torque
                mujoco.mj_step(model, data)
                
                current_config = data.qpos.copy()
        
        trajectory_costs.append(current_cost)
    
    cost_variance = np.var(trajectory_costs)
    assert cost_variance < 1000.0

def test_adaptive_planning():
    model = mujoco.MjModel.from_xml_path("robot_arm.xml")
    data = mujoco.MjData(model)
    
    start_config = np.array([0, 0, 0, 0, 0, 0])
    goal_config = np.array([np.pi/2, np.pi/3, np.pi/4, 0, 0, 0])
    
    data.qpos[:] = start_config
    mujoco.mj_forward(model, data)
    start_pos = data.xpos[model.body("end_effector").id]
    
    data.qpos[:] = goal_config
    mujoco.mj_forward(model, data)
    goal_pos = data.xpos[model.body("end_effector").id]
    
    current_config = start_config.copy()
    planning_adaptations = 0
    
    for step in range(200):
        data.qpos[:] = current_config
        mujoco.mj_forward(model, data)
        current_pos = data.xpos[model.body("end_effector").id]
        
        if np.linalg.norm(current_pos - goal_pos) < 0.01:
            break
        
        error = goal_pos - current_pos
        error_magnitude = np.linalg.norm(error)
        
        if error_magnitude > 0.1:
            step_size = 0.05
        elif error_magnitude > 0.05:
            step_size = 0.02
        else:
            step_size = 0.01
        
        jacobian = np.zeros((3, model.nv))
        mujoco.mj_jac(model, data, jacobian, None, np.zeros(3), model.body("end_effector").id)
        
        if np.linalg.cond(jacobian) > 1000:
            step_size *= 0.5
            planning_adaptations += 1
        
        delta_q = np.linalg.pinv(jacobian) @ (error * step_size)
        current_config += delta_q
    
    assert planning_adaptations > 0 