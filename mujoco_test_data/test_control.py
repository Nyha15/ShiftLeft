import mujoco
import numpy as np
import pytest

def test_pid_controller():
    model = mujoco.MjModel.from_xml_path("robot_arm.xml")
    data = mujoco.MjData(model)
    
    kp = np.ones(model.nv) * 100.0
    ki = np.ones(model.nv) * 10.0
    kd = np.ones(model.nv) * 20.0
    
    target_pos = np.array([np.pi/4, np.pi/6, np.pi/3, 0, 0, 0])
    integral_error = np.zeros(model.nv)
    
    for _ in range(100):
        current_pos = data.qpos.copy()
        current_vel = data.qvel.copy()
        
        position_error = target_pos - current_pos
        integral_error += position_error * 0.01
        velocity_error = -current_vel
        
        control_torque = kp * position_error + ki * integral_error + kd * velocity_error
        
        data.ctrl[:] = control_torque
        mujoco.mj_step(model, data)
        
        if np.linalg.norm(position_error) < 1e-3:
            break
    
    final_error = np.linalg.norm(target_pos - data.qpos)
    assert final_error < 1e-2

def test_computed_torque_control():
    model = mujoco.MjModel.from_xml_path("robot_arm.xml")
    data = mujoco.MjData(model)
    
    kp = np.ones(model.nv) * 100.0
    kd = np.ones(model.nv) * 20.0
    
    target_pos = np.array([np.pi/3, np.pi/4, np.pi/6, 0, 0, 0])
    target_vel = np.zeros(model.nv)
    target_acc = np.zeros(model.nv)
    
    for _ in range(100):
        current_pos = data.qpos.copy()
        current_vel = data.qvel.copy()
        
        position_error = target_pos - current_pos
        velocity_error = target_vel - current_vel
        
        desired_acc = target_acc + kp * position_error + kd * velocity_error
        
        mujoco.mj_forward(model, data)
        mujoco.mj_inverse(model, data)
        
        feedforward_torque = data.qM @ desired_acc + data.qfrc_bias
        control_torque = feedforward_torque
        
        data.ctrl[:] = control_torque
        mujoco.mj_step(model, data)
        
        if np.linalg.norm(position_error) < 1e-3:
            break
    
    final_error = np.linalg.norm(target_pos - data.qpos)
    assert final_error < 1e-2

def test_impedance_control():
    model = mujoco.MjModel.from_xml_path("robot_arm.xml")
    data = mujoco.MjData(model)
    
    mass_desired = np.eye(model.nv) * 2.0
    damping_desired = np.eye(model.nv) * 10.0
    stiffness_desired = np.eye(model.nv) * 100.0
    
    target_pos = np.array([np.pi/4, np.pi/3, np.pi/6, 0, 0, 0])
    external_force = np.array([0, 0, 10, 0, 0, 0])
    
    for _ in range(100):
        current_pos = data.qpos.copy()
        current_vel = data.qvel.copy()
        
        position_error = target_pos - current_pos
        velocity_error = -current_vel
        
        desired_force = stiffness_desired @ position_error + damping_desired @ velocity_error
        
        mujoco.mj_forward(model, data)
        mujoco.mj_inverse(model, data)
        
        control_torque = desired_force + data.qfrc_bias
        
        data.ctrl[:] = control_torque
        mujoco.mj_step(model, data)
        
        if np.linalg.norm(position_error) < 1e-3:
            break
    
    final_error = np.linalg.norm(target_pos - data.qpos)
    assert final_error < 1e-2

def test_trajectory_tracking():
    model = mujoco.MjModel.from_xml_path("robot_arm.xml")
    data = mujoco.MjData(model)
    
    t_start = 0.0
    t_end = 2.0
    dt = 0.01
    
    waypoints = [
        np.array([0, 0, 0, 0, 0, 0]),
        np.array([np.pi/4, np.pi/6, np.pi/3, 0, 0, 0]),
        np.array([np.pi/2, np.pi/3, np.pi/4, 0, 0, 0]),
        np.array([0, 0, 0, 0, 0, 0])
    ]
    
    kp = np.ones(model.nv) * 200.0
    kd = np.ones(model.nv) * 30.0
    
    t = t_start
    waypoint_idx = 0
    
    while t < t_end and waypoint_idx < len(waypoints) - 1:
        alpha = (t - t_start) / (t_end - t_start)
        target_pos = (1 - alpha) * waypoints[waypoint_idx] + alpha * waypoints[waypoint_idx + 1]
        
        current_pos = data.qpos.copy()
        current_vel = data.qvel.copy()
        
        position_error = target_pos - current_pos
        velocity_error = -current_vel
        
        control_torque = kp * position_error + kd * velocity_error
        
        data.ctrl[:] = control_torque
        mujoco.mj_step(model, data)
        
        t += dt
        
        if np.linalg.norm(position_error) < 1e-2:
            waypoint_idx += 1
    
    final_error = np.linalg.norm(waypoints[-1] - data.qpos)
    assert final_error < 1e-2

def test_force_control():
    model = mujoco.MjModel.from_xml_path("robot_arm.xml")
    data = mujoco.MjData(model)
    
    target_force = np.array([0, 0, 5.0, 0, 0, 0])
    kf = np.ones(model.nv) * 50.0
    
    for _ in range(100):
        mujoco.mj_forward(model, data)
        
        current_force = data.qfrc_applied
        force_error = target_force - current_force
        
        control_torque = kf * force_error
        
        data.ctrl[:] = control_torque
        mujoco.mj_step(model, data)
        
        if np.linalg.norm(force_error) < 1e-2:
            break
    
    final_force_error = np.linalg.norm(target_force - data.qfrc_applied)
    assert final_force_error < 1e-1 