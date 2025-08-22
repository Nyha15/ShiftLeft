import mujoco
import numpy as np
import pytest

def test_forward_kinematics_single_joint():
    model = mujoco.MjModel.from_xml_path("robot_arm.xml")
    data = mujoco.MjData(model)
    
    test_angles = [np.pi/4, 0, 0, 0, 0, 0]
    data.qpos[:] = test_angles
    mujoco.mj_forward(model, data)
    
    end_effector_pos = data.xpos[model.body("end_effector").id]
    expected_x = model.body("base").pos[0] + np.cos(test_angles[0]) * 0.5
    expected_y = model.body("base").pos[1] + np.sin(test_angles[0]) * 0.5
    
    assert np.allclose(end_effector_pos[0], expected_x, atol=1e-3)
    assert np.allclose(end_effector_pos[1], expected_y, atol=1e-3)

def test_jacobian_calculation():
    model = mujoco.MjModel.from_xml_path("robot_arm.xml")
    data = mujoco.MjData(model)
    
    mujoco.mj_forward(model, data)
    mujoco.mj_jac(model, data, jacp, jacr, np.array([0, 0, 0]), model.body("end_effector").id)
    
    assert jacp.shape == (3, model.nv)
    assert jacr.shape == (3, model.nv)
    assert not np.any(np.isnan(jacp))
    assert not np.any(np.isnan(jacr))

def test_inverse_kinematics_simple():
    model = mujoco.MjModel.from_xml_path("robot_arm.xml")
    data = mujoco.MjData(model)
    
    target_pos = np.array([0.5, 0.0, 0.3])
    
    for _ in range(100):
        mujoco.mj_forward(model, data)
        current_pos = data.xpos[model.body("end_effector").id]
        error = target_pos - current_pos
        
        if np.linalg.norm(error) < 1e-3:
            break
            
        mujoco.mj_jac(model, data, jacp, jacr, np.array([0, 0, 0]), model.body("end_effector").id)
        jacobian = jacp[:3, :]
        
        delta_q = np.linalg.pinv(jacobian) @ error
        data.qpos[:] += delta_q * 0.1
    
    final_pos = data.xpos[model.body("end_effector").id]
    assert np.linalg.norm(final_pos - target_pos) < 1e-2

def test_singularity_detection():
    model = mujoco.MjModel.from_xml_path("robot_arm.xml")
    data = mujoco.MjData(model)
    
    data.qpos[:] = [0, np.pi/2, 0, 0, 0, 0]
    mujoco.mj_forward(model, data)
    mujoco.mj_jac(model, data, jacp, jacr, np.array([0, 0, 0]), model.body("end_effector").id)
    
    jacobian = jacp[:3, :]
    condition_number = np.linalg.cond(jacobian)
    
    assert condition_number > 1e-6
    assert condition_number < 1e6

def test_workspace_boundaries():
    model = mujoco.MjModel.from_xml_path("robot_arm.xml")
    data = mujoco.MjData(model)
    
    workspace_points = []
    
    for i in range(10):
        for j in range(10):
            for k in range(10):
                data.qpos[:] = np.random.uniform(-np.pi, np.pi, model.nq)
                mujoco.mj_forward(model, data)
                end_pos = data.xpos[model.body("end_effector").id]
                workspace_points.append(end_pos)
    
    workspace_points = np.array(workspace_points)
    max_reach = np.max(np.linalg.norm(workspace_points, axis=1))
    min_reach = np.min(np.linalg.norm(workspace_points, axis=1))
    
    assert max_reach < 2.0
    assert min_reach > 0.1 