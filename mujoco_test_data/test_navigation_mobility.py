import mujoco
import numpy as np
import pytest

def test_workspace_coverage():
    model = mujoco.MjModel.from_xml_path("robot_arm.xml")
    data = mujoco.MjData(model)
    
    workspace_points = []
    coverage_radius = 0.1
    
    for i in range(20):
        for j in range(20):
            for k in range(20):
                test_config = np.array([
                    np.random.uniform(-np.pi, np.pi),
                    np.random.uniform(-np.pi/2, np.pi/2),
                    np.random.uniform(-np.pi/2, np.pi/2),
                    0, 0, 0
                ])
                
                data.qpos[:] = test_config
                mujoco.mj_forward(model, data)
                
                end_effector_pos = data.xpos[model.body("end_effector").id]
                workspace_points.append(end_effector_pos)
    
    workspace_points = np.array(workspace_points)
    
    coverage_map = np.zeros((20, 20, 20))
    for point in workspace_points:
        x_idx = int((point[0] + 1.0) * 10)
        y_idx = int((point[1] + 1.0) * 10)
        z_idx = int((point[2] + 0.5) * 10)
        
        if 0 <= x_idx < 20 and 0 <= y_idx < 20 and 0 <= z_idx < 20:
            coverage_map[x_idx, y_idx, z_idx] = 1
    
    coverage_percentage = np.sum(coverage_map) / (20 * 20 * 20)
    assert coverage_percentage > 0.3

def test_path_optimization():
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
    
    optimized_path = []
    current_config = start_config.copy()
    
    for step in range(150):
        data.qpos[:] = current_config
        mujoco.mj_forward(model, data)
        current_pos = data.xpos[model.body("end_effector").id]
        optimized_path.append(current_pos)
        
        if np.linalg.norm(current_pos - goal_pos) < 0.01:
            break
        
        direction = goal_pos - current_pos
        direction = direction / np.linalg.norm(direction)
        
        jacobian = np.zeros((3, model.nv))
        mujoco.mj_jac(model, data, jacobian, None, np.zeros(3), model.body("end_effector").id)
        
        if np.linalg.cond(jacobian) > 500:
            step_size = 0.02
        else:
            step_size = 0.05
        
        delta_q = np.linalg.pinv(jacobian) @ (direction * step_size)
        current_config += delta_q
    
    optimized_path = np.array(optimized_path)
    path_length = np.sum(np.linalg.norm(np.diff(optimized_path, axis=0), axis=1))
    
    optimization_ratio = direct_distance / path_length
    assert optimization_ratio > 0.85

def test_obstacle_navigation():
    model = mujoco.MjModel.from_xml_path("robot_arm.xml")
    data = mujoco.MjData(model)
    
    start_config = np.array([0, 0, 0, 0, 0, 0])
    goal_config = np.array([np.pi/2, np.pi/3, np.pi/4, 0, 0, 0])
    
    obstacles = [
        np.array([0.3, 0.0, 0.3]),
        np.array([0.5, 0.1, 0.25]),
        np.array([0.7, -0.1, 0.35])
    ]
    
    data.qpos[:] = start_config
    mujoco.mj_forward(model, data)
    start_pos = data.xpos[model.body("end_effector").id]
    
    data.qpos[:] = goal_config
    mujoco.mj_forward(model, data)
    goal_pos = data.xpos[model.body("end_effector").id]
    
    current_config = start_config.copy()
    collision_count = 0
    navigation_success = False
    
    for step in range(400):
        data.qpos[:] = current_config
        mujoco.mj_forward(model, data)
        current_pos = data.xpos[model.body("end_effector").id]
        
        for obstacle in obstacles:
            if np.linalg.norm(current_pos - obstacle) < 0.08:
                collision_count += 1
                break
        
        if np.linalg.norm(current_pos - goal_pos) < 0.01:
            navigation_success = True
            break
        
        attractive_force = goal_pos - current_pos
        attractive_force = attractive_force / np.linalg.norm(attractive_force)
        
        repulsive_forces = np.zeros(3)
        for obstacle in obstacles:
            distance_to_obstacle = np.linalg.norm(current_pos - obstacle)
            if distance_to_obstacle < 0.3:
                repulsive_direction = (current_pos - obstacle) / distance_to_obstacle
                repulsive_magnitude = 0.2 / (distance_to_obstacle ** 2)
                repulsive_forces += repulsive_magnitude * repulsive_direction
        
        total_force = attractive_force + repulsive_forces
        total_force = total_force / np.linalg.norm(total_force)
        
        jacobian = np.zeros((3, model.nv))
        mujoco.mj_jac(model, data, jacobian, None, np.zeros(3), model.body("end_effector").id)
        
        delta_q = np.linalg.pinv(jacobian) @ (total_force * 0.03)
        current_config += delta_q
    
    assert collision_count < 5
    assert navigation_success

def test_dynamic_obstacle_avoidance():
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
    dynamic_obstacle_pos = np.array([0.4, 0.0, 0.3])
    avoidance_success = True
    
    for step in range(300):
        data.qpos[:] = current_config
        mujoco.mj_forward(model, data)
        current_pos = data.xpos[model.body("end_effector").id]
        
        dynamic_obstacle_pos[0] += 0.001 * np.sin(step * 0.1)
        dynamic_obstacle_pos[1] += 0.001 * np.cos(step * 0.1)
        
        distance_to_obstacle = np.linalg.norm(current_pos - dynamic_obstacle_pos)
        if distance_to_obstacle < 0.05:
            avoidance_success = False
            break
        
        if np.linalg.norm(current_pos - goal_pos) < 0.01:
            break
        
        attractive_force = goal_pos - current_pos
        attractive_force = attractive_force / np.linalg.norm(attractive_force)
        
        repulsive_force = np.zeros(3)
        if distance_to_obstacle < 0.25:
            repulsive_direction = (current_pos - dynamic_obstacle_pos) / distance_to_obstacle
            repulsive_magnitude = 0.3 / (distance_to_obstacle ** 2)
            repulsive_force = repulsive_magnitude * repulsive_direction
        
        total_force = attractive_force + repulsive_force
        total_force = total_force / np.linalg.norm(total_force)
        
        jacobian = np.zeros((3, model.nv))
        mujoco.mj_jac(model, data, jacobian, None, np.zeros(3), model.body("end_effector").id)
        
        delta_q = np.linalg.pinv(jacobian) @ (total_force * 0.02)
        current_config += delta_q
    
    assert avoidance_success

def test_energy_efficient_navigation():
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
    energy_consumption = 0.0
    
    for step in range(200):
        data.qpos[:] = current_config
        mujoco.mj_forward(model, data)
        current_pos = data.xpos[model.body("end_effector").id]
        
        if np.linalg.norm(current_pos - goal_pos) < 0.01:
            break
        
        error = goal_pos - current_pos
        error_magnitude = np.linalg.norm(error)
        
        if error_magnitude > 0.1:
            control_gain = 50.0
        else:
            control_gain = 20.0
        
        control_torque = control_gain * error
        energy_consumption += np.sum(control_torque ** 2) * 0.01
        
        data.ctrl[:] = control_torque
        mujoco.mj_step(model, data)
        
        current_config = data.qpos.copy()
    
    assert energy_consumption < 1000.0

def test_multi_goal_navigation():
    model = mujoco.MjModel.from_xml_path("robot_arm.xml")
    data = mujoco.MjData(model)
    
    waypoints = [
        np.array([0, 0, 0, 0, 0, 0]),
        np.array([np.pi/4, np.pi/6, np.pi/8, 0, 0, 0]),
        np.array([np.pi/2, np.pi/3, np.pi/4, 0, 0, 0]),
        np.array([np.pi/3, np.pi/4, np.pi/6, 0, 0, 0]),
        np.array([0, 0, 0, 0, 0, 0])
    ]
    
    navigation_success = 0
    total_waypoints = len(waypoints) - 1
    
    for i in range(len(waypoints) - 1):
        start_config = waypoints[i]
        goal_config = waypoints[i + 1]
        
        data.qpos[:] = start_config
        mujoco.mj_forward(model, data)
        start_pos = data.xpos[model.body("end_effector").id]
        
        data.qpos[:] = goal_config
        mujoco.mj_forward(model, data)
        goal_pos = data.xpos[model.body("end_effector").id]
        
        current_config = start_config.copy()
        waypoint_reached = False
        
        for step in range(150):
            data.qpos[:] = current_config
            mujoco.mj_forward(model, data)
            current_pos = data.xpos[model.body("end_effector").id]
            
            if np.linalg.norm(current_pos - goal_pos) < 0.01:
                waypoint_reached = True
                break
            
            direction = goal_pos - current_pos
            direction = direction / np.linalg.norm(direction)
            
            jacobian = np.zeros((3, model.nv))
            mujoco.mj_jac(model, data, jacobian, None, np.zeros(3), model.body("end_effector").id)
            
            delta_q = np.linalg.pinv(jacobian) @ (direction * 0.05)
            current_config += delta_q
        
        if waypoint_reached:
            navigation_success += 1
    
    success_rate = navigation_success / total_waypoints
    assert success_rate >= 0.8 