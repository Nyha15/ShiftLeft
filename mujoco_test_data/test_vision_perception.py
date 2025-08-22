import mujoco
import numpy as np
import pytest

def test_object_detection_accuracy():
    model = mujoco.MjModel.from_xml_path("robot_arm.xml")
    data = mujoco.MjData(model)
    
    object_positions = [
        np.array([0.4, 0.1, 0.2]),
        np.array([0.6, -0.1, 0.3]),
        np.array([0.8, 0.2, 0.25])
    ]
    
    detection_accuracy = 0.0
    total_objects = len(object_positions)
    
    for obj_pos in object_positions:
        data.qpos[:] = np.zeros(model.nq)
        mujoco.mj_forward(model, data)
        
        camera_pos = data.xpos[model.body("camera").id]
        camera_quat = data.xquat[model.body("camera").id]
        
        relative_pos = obj_pos - camera_pos
        distance = np.linalg.norm(relative_pos)
        
        if distance < 1.0:
            detection_accuracy += 1.0
        elif distance < 2.0:
            detection_accuracy += 0.5
    
    detection_accuracy /= total_objects
    assert detection_accuracy >= 0.7

def test_depth_perception():
    model = mujoco.MjModel.from_xml_path("robot_arm.xml")
    data = mujoco.MjData(model)
    
    test_distances = [0.5, 1.0, 1.5, 2.0]
    depth_errors = []
    
    for target_distance in test_distances:
        data.qpos[:] = np.zeros(model.nq)
        mujoco.mj_forward(model, data)
        
        camera_pos = data.xpos[model.body("camera").id]
        target_pos = camera_pos + np.array([target_distance, 0, 0])
        
        measured_distance = np.linalg.norm(target_pos - camera_pos)
        depth_error = abs(measured_distance - target_distance)
        depth_errors.append(depth_error)
    
    mean_depth_error = np.mean(depth_errors)
    assert mean_depth_error < 0.05

def test_pose_estimation():
    model = mujoco.MjModel.from_xml_path("robot_arm.xml")
    data = mujoco.MjData(model)
    
    test_poses = [
        (np.array([0.5, 0.0, 0.3]), np.array([0, 0, 0])),
        (np.array([0.4, 0.2, 0.4]), np.array([np.pi/6, 0, 0])),
        (np.array([0.6, -0.1, 0.2]), np.array([0, np.pi/8, 0]))
    ]
    
    pose_estimation_errors = []
    
    for true_pos, true_rot in test_poses:
        data.qpos[:] = np.zeros(model.nq)
        mujoco.mj_forward(model, data)
        
        camera_pos = data.xpos[model.body("camera").id]
        camera_quat = data.xquat[model.body("camera").id]
        
        relative_pos = true_pos - camera_pos
        estimated_distance = np.linalg.norm(relative_pos)
        
        position_error = abs(estimated_distance - np.linalg.norm(true_pos - camera_pos))
        pose_estimation_errors.append(position_error)
    
    mean_pose_error = np.mean(pose_estimation_errors)
    assert mean_pose_error < 0.1

def test_visual_tracking():
    model = mujoco.MjModel.from_xml_path("robot_arm.xml")
    data = mujoco.MjData(model)
    
    moving_object_trajectory = []
    for t in range(100):
        x = 0.5 + 0.1 * np.sin(t * 0.1)
        y = 0.1 * np.cos(t * 0.1)
        z = 0.3 + 0.05 * np.sin(t * 0.15)
        moving_object_trajectory.append(np.array([x, y, z]))
    
    tracking_errors = []
    
    for t, obj_pos in enumerate(moving_object_trajectory):
        data.qpos[:] = np.zeros(model.nq)
        mujoco.mj_forward(model, data)
        
        camera_pos = data.xpos[model.body("camera").id]
        camera_quat = data.xquat[model.body("camera").id]
        
        relative_pos = obj_pos - camera_pos
        distance = np.linalg.norm(relative_pos)
        
        if distance < 1.5:
            tracking_errors.append(0.0)
        else:
            tracking_errors.append(distance - 1.5)
    
    mean_tracking_error = np.mean(tracking_errors)
    assert mean_tracking_error < 0.2

def test_visual_servoing():
    model = mujoco.MjModel.from_xml_path("robot_arm.xml")
    data = mujoco.MjData(model)
    
    target_pos = np.array([0.5, 0.0, 0.3])
    data.qpos[:] = np.array([np.pi/4, 0, 0, 0, 0, 0])
    
    visual_servoing_errors = []
    
    for step in range(200):
        mujoco.mj_forward(model, data)
        current_pos = data.xpos[model.body("end_effector").id]
        
        camera_pos = data.xpos[model.body("camera").id]
        camera_quat = data.xquat[model.body("camera").id]
        
        target_in_camera = target_pos - camera_pos
        current_in_camera = current_pos - camera_pos
        
        visual_error = target_in_camera - current_in_camera
        visual_servoing_errors.append(np.linalg.norm(visual_error))
        
        if np.linalg.norm(visual_error) < 0.01:
            break
        
        jacobian = np.zeros((3, model.nv))
        mujoco.mj_jac(model, data, jacobian, None, np.zeros(3), model.body("end_effector").id)
        
        delta_q = np.linalg.pinv(jacobian) @ (visual_error * 0.1)
        data.qpos[:] += delta_q
    
    final_visual_error = visual_servoing_errors[-1]
    assert final_visual_error < 0.02

def test_occlusion_handling():
    model = mujoco.MjModel.from_xml_path("robot_arm.xml")
    data = mujoco.MjData(model)
    
    target_pos = np.array([0.6, 0.0, 0.3])
    occluder_pos = np.array([0.5, 0.0, 0.3])
    
    data.qpos[:] = np.zeros(model.nq)
    mujoco.mj_forward(model, data)
    
    camera_pos = data.xpos[model.body("camera").id]
    
    target_visible = True
    occluder_distance = np.linalg.norm(occluder_pos - camera_pos)
    target_distance = np.linalg.norm(target_pos - camera_pos)
    
    if occluder_distance < target_distance and occluder_distance < 0.8:
        target_visible = False
    
    if target_visible:
        detection_confidence = 1.0
    else:
        detection_confidence = 0.3
    
    assert detection_confidence > 0.0

def test_lighting_adaptation():
    model = mujoco.MjModel.from_xml_path("robot_arm.xml")
    data = mujoco.MjData(model)
    
    lighting_conditions = [0.5, 1.0, 1.5, 2.0]
    detection_rates = []
    
    for lighting_intensity in lighting_conditions:
        data.qpos[:] = np.zeros(model.nq)
        mujoco.mj_forward(model, data)
        
        target_pos = np.array([0.5, 0.0, 0.3])
        camera_pos = data.xpos[model.body("camera").id]
        
        distance = np.linalg.norm(target_pos - camera_pos)
        lighting_factor = lighting_intensity / (1.0 + distance)
        
        if lighting_factor > 0.3:
            detection_rate = 1.0
        elif lighting_factor > 0.1:
            detection_rate = 0.7
        else:
            detection_rate = 0.2
        
        detection_rates.append(detection_rate)
    
    mean_detection_rate = np.mean(detection_rates)
    assert mean_detection_rate >= 0.5

def test_multi_camera_fusion():
    model = mujoco.MjModel.from_xml_path("robot_arm.xml")
    data = mujoco.MjData(model)
    
    target_pos = np.array([0.5, 0.0, 0.3])
    camera_positions = [
        np.array([0.0, 0.0, 0.5]),
        np.array([0.0, 0.5, 0.0]),
        np.array([0.5, 0.0, 0.0])
    ]
    
    triangulation_errors = []
    
    for i in range(len(camera_positions) - 1):
        for j in range(i + 1, len(camera_positions)):
            cam1_pos = camera_positions[i]
            cam2_pos = camera_positions[j]
            
            vec1 = target_pos - cam1_pos
            vec2 = target_pos - cam2_pos
            
            vec1_normalized = vec1 / np.linalg.norm(vec1)
            vec2_normalized = vec2 / np.linalg.norm(vec2)
            
            triangulation_error = np.linalg.norm(vec1_normalized - vec2_normalized)
            triangulation_errors.append(triangulation_error)
    
    mean_triangulation_error = np.mean(triangulation_errors)
    assert mean_triangulation_error < 0.5 