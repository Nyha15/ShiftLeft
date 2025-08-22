import mujoco
import numpy as np
import pytest

def test_pick_and_place_accuracy():
    model = mujoco.MjModel.from_xml_path("robot_arm.xml")
    data = mujoco.MjData(model)
    
    target_positions = [
        np.array([0.5, 0.0, 0.3]),
        np.array([0.3, 0.2, 0.4]),
        np.array([0.7, -0.1, 0.2])
    ]
    
    success_count = 0
    total_attempts = len(target_positions)
    
    for target in target_positions:
        data.qpos[:] = np.zeros(model.nq)
        data.qvel[:] = np.zeros(model.nv)
        
        for _ in range(200):
            current_pos = data.xpos[model.body("end_effector").id]
            error = target - current_pos
            
            if np.linalg.norm(error) < 0.01:
                success_count += 1
                break
                
            jacobian = np.zeros((3, model.nv))
            mujoco.mj_jac(model, data, jacobian, None, np.zeros(3), model.body("end_effector").id)
            
            delta_q = np.linalg.pinv(jacobian) @ error
            data.qpos[:] += delta_q * 0.1
            
            mujoco.mj_forward(model, data)
    
    success_rate = success_count / total_attempts
    assert success_rate >= 0.8

def test_trajectory_smoothness():
    model = mujoco.MjModel.from_xml_path("robot_arm.xml")
    data = mujoco.MjData(model)
    
    waypoints = [
        np.array([0, 0, 0, 0, 0, 0]),
        np.array([np.pi/4, np.pi/6, np.pi/3, 0, 0, 0]),
        np.array([np.pi/2, np.pi/3, np.pi/4, 0, 0, 0]),
        np.array([0, 0, 0, 0, 0, 0])
    ]
    
    trajectory_data = []
    
    for i in range(len(waypoints) - 1):
        start_pos = waypoints[i]
        end_pos = waypoints[i + 1]
        
        data.qpos[:] = start_pos
        mujoco.mj_forward(model, data)
        
        for t in range(100):
            alpha = t / 100.0
            target_pos = (1 - alpha) * start_pos + alpha * end_pos
            
            current_pos = data.qpos.copy()
            error = target_pos - current_pos
            
            control_torque = 100.0 * error + 20.0 * (-data.qvel)
            data.ctrl[:] = control_torque
            
            mujoco.mj_step(model, data)
            trajectory_data.append(data.qpos.copy())
    
    trajectory_array = np.array(trajectory_data)
    velocity_changes = np.diff(trajectory_array, axis=0)
    acceleration_changes = np.diff(velocity_changes, axis=0)
    
    max_acceleration = np.max(np.linalg.norm(acceleration_changes, axis=1))
    assert max_acceleration < 10.0

def test_obstacle_avoidance():
    model = mujoco.MjModel.from_xml_path("robot_arm.xml")
    data = mujoco.MjData(model)
    
    obstacle_pos = np.array([0.4, 0.0, 0.3])
    target_pos = np.array([0.6, 0.0, 0.3])
    
    data.qpos[:] = np.zeros(model.nq)
    data.qvel[:] = np.zeros(model.nv)
    
    collision_occurred = False
    target_reached = False
    
    for _ in range(500):
        current_pos = data.xpos[model.body("end_effector").id]
        distance_to_obstacle = np.linalg.norm(current_pos - obstacle_pos)
        distance_to_target = np.linalg.norm(current_pos - target_pos)
        
        if distance_to_obstacle < 0.05:
            collision_occurred = True
            break
            
        if distance_to_target < 0.02:
            target_reached = True
            break
        
        repulsive_force = np.zeros(3)
        if distance_to_obstacle < 0.2:
            repulsive_force = 0.1 * (current_pos - obstacle_pos) / (distance_to_obstacle ** 3)
        
        attractive_force = 0.5 * (target_pos - current_pos)
        total_force = attractive_force + repulsive_force
        
        jacobian = np.zeros((3, model.nv))
        mujoco.mj_jac(model, data, jacobian, None, np.zeros(3), model.body("end_effector").id)
        
        delta_q = np.linalg.pinv(jacobian) @ total_force
        data.qpos[:] += delta_q * 0.01
        
        mujoco.mj_forward(model, data)
    
    assert not collision_occurred
    assert target_reached

def test_force_control_precision():
    model = mujoco.MjModel.from_xml_path("robot_arm.xml")
    data = mujoco.MjData(model)
    
    target_force = np.array([0, 0, 5.0])
    force_tolerance = 0.5
    
    data.qpos[:] = [np.pi/4, np.pi/6, np.pi/3, 0, 0, 0]
    data.qvel[:] = np.zeros(model.nv)
    
    force_errors = []
    
    for _ in range(200):
        mujoco.mj_forward(model, data)
        
        current_force = data.qfrc_applied
        force_error = target_force - current_force
        force_errors.append(np.linalg.norm(force_error))
        
        control_torque = 50.0 * force_error
        
        data.ctrl[:] = control_torque
        mujoco.mj_step(model, data)
    
    mean_force_error = np.mean(force_errors[-50:])
    max_force_error = np.max(force_errors[-50:])
    
    assert mean_force_error < force_tolerance
    assert max_force_error < force_tolerance * 2

def test_multi_object_manipulation():
    model = mujoco.MjModel.from_xml_path("robot_arm.xml")
    data = mujoco.MjData(model)
    
    object_positions = [
        np.array([0.3, 0.1, 0.2]),
        np.array([0.5, -0.1, 0.3]),
        np.array([0.7, 0.2, 0.25])
    ]
    
    target_positions = [
        np.array([0.8, 0.3, 0.4]),
        np.array([0.9, 0.1, 0.5]),
        np.array([0.6, 0.4, 0.35])
    ]
    
    manipulation_success = 0
    
    for obj_idx in range(len(object_positions)):
        data.qpos[:] = np.zeros(model.nq)
        data.qvel[:] = np.zeros(model.nv)
        
        start_pos = object_positions[obj_idx]
        end_pos = target_positions[obj_idx]
        
        for step in range(300):
            current_pos = data.xpos[model.body("end_effector").id]
            
            if step < 150:
                target = start_pos
            else:
                target = end_pos
            
            error = target - current_pos
            
            if np.linalg.norm(error) < 0.02:
                if step >= 150:
                    manipulation_success += 1
                break
            
            jacobian = np.zeros((3, model.nv))
            mujoco.mj_jac(model, data, jacobian, None, np.zeros(3), model.body("end_effector").id)
            
            delta_q = np.linalg.pinv(jacobian) @ error
            data.qpos[:] += delta_q * 0.05
            
            mujoco.mj_forward(model, data)
    
    success_rate = manipulation_success / len(object_positions)
    assert success_rate >= 0.7 