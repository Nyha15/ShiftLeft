import mujoco
import numpy as np
import pytest

def test_grasping_precision():
    model = mujoco.MjModel.from_xml_path("robot_arm.xml")
    data = mujoco.MjData(model)
    
    target_objects = [
        np.array([0.4, 0.0, 0.2]),
        np.array([0.5, 0.1, 0.25]),
        np.array([0.6, -0.05, 0.3])
    ]
    
    grasp_success_rate = 0.0
    total_attempts = len(target_objects)
    
    for obj_pos in target_objects:
        data.qpos[:] = np.zeros(model.nq)
        data.qvel[:] = np.zeros(model.nv)
        
        for step in range(200):
            current_pos = data.xpos[model.body("end_effector").id]
            distance_to_object = np.linalg.norm(current_pos - obj_pos)
            
            if distance_to_object < 0.02:
                grasp_success_rate += 1.0
                break
            
            direction = obj_pos - current_pos
            direction = direction / np.linalg.norm(direction)
            
            jacobian = np.zeros((3, model.nv))
            mujoco.mj_jac(model, data, jacobian, None, np.zeros(3), model.body("end_effector").id)
            
            delta_q = np.linalg.pinv(jacobian) @ (direction * 0.03)
            data.qpos[:] += delta_q
            
            mujoco.mj_forward(model, data)
    
    grasp_success_rate /= total_attempts
    assert grasp_success_rate >= 0.8

def test_fine_manipulation():
    model = mujoco.MjModel.from_xml_path("robot_arm.xml")
    data = mujoco.MjData(model)
    
    start_pos = np.array([0.5, 0.0, 0.3])
    target_pos = np.array([0.52, 0.02, 0.31])
    
    data.qpos[:] = np.array([np.pi/4, np.pi/6, np.pi/3, 0, 0, 0])
    data.qvel[:] = np.zeros(model.nv)
    
    fine_control_errors = []
    
    for step in range(300):
        current_pos = data.xpos[model.body("end_effector").id]
        error = target_pos - current_pos
        fine_control_errors.append(np.linalg.norm(error))
        
        if np.linalg.norm(error) < 0.005:
            break
        
        jacobian = np.zeros((3, model.nv))
        mujoco.mj_jac(model, data, jacobian, None, np.zeros(3), model.body("end_effector").id)
        
        control_gain = 0.02
        delta_q = np.linalg.pinv(jacobian) @ (error * control_gain)
        data.qpos[:] += delta_q
        
        mujoco.mj_forward(model, data)
    
    final_precision = fine_control_errors[-1]
    assert final_precision < 0.01

def test_assembly_tasks():
    model = mujoco.MjModel.from_xml_path("robot_arm.xml")
    data = mujoco.MjData(model)
    
    assembly_positions = [
        np.array([0.4, 0.0, 0.2]),
        np.array([0.4, 0.0, 0.25]),
        np.array([0.4, 0.0, 0.3])
    ]
    
    assembly_success = 0
    
    for i, target_pos in enumerate(assembly_positions):
        data.qpos[:] = np.zeros(model.nq)
        data.qvel[:] = np.zeros(model.nv)
        
        for step in range(250):
            current_pos = data.xpos[model.body("end_effector").id]
            error = target_pos - current_pos
            
            if np.linalg.norm(error) < 0.015:
                assembly_success += 1
                break
            
            jacobian = np.zeros((3, model.nv))
            mujoco.mj_jac(model, data, jacobian, None, np.zeros(3), model.body("end_effector").id)
            
            delta_q = np.linalg.pinv(jacobian) @ (error * 0.04)
            data.qpos[:] += delta_q
            
            mujoco.mj_forward(model, data)
    
    assembly_success_rate = assembly_success / len(assembly_positions)
    assert assembly_success_rate >= 0.7

def test_tool_manipulation():
    model = mujoco.MjModel.from_xml_path("robot_arm.xml")
    data = mujoco.MjData(model)
    
    tool_positions = [
        np.array([0.3, 0.1, 0.2]),
        np.array([0.5, -0.1, 0.3]),
        np.array([0.7, 0.2, 0.25])
    ]
    
    tool_usage_success = 0
    
    for tool_pos in tool_positions:
        data.qpos[:] = np.zeros(model.nq)
        data.qvel[:] = np.zeros(model.nv)
        
        for step in range(200):
            current_pos = data.xpos[model.body("end_effector").id]
            distance_to_tool = np.linalg.norm(current_pos - tool_pos)
            
            if distance_to_tool < 0.02:
                tool_usage_success += 1
                break
            
            direction = tool_pos - current_pos
            direction = direction / np.linalg.norm(direction)
            
            jacobian = np.zeros((3, model.nv))
            mujoco.mj_jac(model, data, jacobian, None, np.zeros(3), model.body("end_effector").id)
            
            delta_q = np.linalg.pinv(jacobian) @ (direction * 0.03)
            data.qpos[:] += delta_q
            
            mujoco.mj_forward(model, data)
    
    tool_success_rate = tool_usage_success / len(tool_positions)
    assert tool_success_rate >= 0.8

def test_force_controlled_manipulation():
    model = mujoco.MjModel.from_xml_path("robot_arm.xml")
    data = mujoco.MjData(model)
    
    target_force = np.array([0, 0, 3.0])
    force_tolerance = 0.3
    
    data.qpos[:] = [np.pi/4, np.pi/6, np.pi/3, 0, 0, 0]
    data.qvel[:] = np.zeros(model.nv)
    
    force_control_errors = []
    
    for step in range(250):
        mujoco.mj_forward(model, data)
        
        current_force = data.qfrc_applied
        force_error = target_force - current_force
        force_control_errors.append(np.linalg.norm(force_error))
        
        if np.linalg.norm(force_error) < force_tolerance:
            break
        
        control_torque = 40.0 * force_error
        
        data.ctrl[:] = control_torque
        mujoco.mj_step(model, data)
    
    mean_force_error = np.mean(force_control_errors[-20:])
    assert mean_force_error < force_tolerance

def test_coordinated_motion():
    model = mujoco.MjModel.from_xml_path("robot_arm.xml")
    data = mujoco.MjData(model)
    
    waypoints = [
        np.array([0.4, 0.0, 0.2]),
        np.array([0.5, 0.1, 0.25]),
        np.array([0.6, 0.0, 0.3]),
        np.array([0.5, -0.1, 0.25]),
        np.array([0.4, 0.0, 0.2])
    ]
    
    motion_smoothness = []
    current_config = np.zeros(model.nq)
    
    for i in range(len(waypoints) - 1):
        start_pos = waypoints[i]
        end_pos = waypoints[i + 1]
        
        for step in range(100):
            data.qpos[:] = current_config
            mujoco.mj_forward(model, data)
            current_pos = data.xpos[model.body("end_effector").id]
            
            alpha = step / 100.0
            target_pos = (1 - alpha) * start_pos + alpha * end_pos
            
            error = target_pos - current_pos
            
            if np.linalg.norm(error) < 0.01:
                break
            
            jacobian = np.zeros((3, model.nv))
            mujoco.mj_jac(model, data, jacobian, None, np.zeros(3), model.body("end_effector").id)
            
            delta_q = np.linalg.pinv(jacobian) @ (error * 0.05)
            current_config += delta_q
            
            if step > 0:
                motion_smoothness.append(np.linalg.norm(delta_q))
    
    mean_motion_smoothness = np.mean(motion_smoothness)
    assert mean_motion_smoothness < 0.1

def test_adaptive_grasping():
    model = mujoco.MjModel.from_xml_path("robot_arm.xml")
    data = mujoco.MjData(model)
    
    object_sizes = [0.02, 0.04, 0.06, 0.08]
    grasping_success = 0
    
    for obj_size in object_sizes:
        data.qpos[:] = np.zeros(model.nq)
        data.qvel[:] = np.zeros(model.nv)
        
        target_pos = np.array([0.5, 0.0, 0.3])
        grasp_threshold = obj_size * 1.5
        
        for step in range(200):
            current_pos = data.xpos[model.body("end_effector").id]
            distance_to_object = np.linalg.norm(current_pos - target_pos)
            
            if distance_to_object < grasp_threshold:
                grasping_success += 1
                break
            
            direction = target_pos - current_pos
            direction = direction / np.linalg.norm(direction)
            
            jacobian = np.zeros((3, model.nv))
            mujoco.mj_jac(model, data, jacobian, None, np.zeros(3), model.body("end_effector").id)
            
            adaptive_gain = 0.03 * (1.0 + obj_size * 10)
            delta_q = np.linalg.pinv(jacobian) @ (direction * adaptive_gain)
            data.qpos[:] += delta_q
            
            mujoco.mj_forward(model, data)
    
    adaptive_grasp_success_rate = grasping_success / len(object_sizes)
    assert adaptive_grasp_success_rate >= 0.75 